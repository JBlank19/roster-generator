import math
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import airportsdata
import numpy as np
import pandas as pd
import pytz

# Shared constants
BIN_SIZE_MINS = 5
END_OF_DAY_MINS = 1440
P_NEXT_BIN_SIZE_MINS = 60
MAX_INTRADAY_RESAMPLE_ATTEMPTS = 256


class DataManager:
    """Hold and manage lookup tables required by schedule generation."""

    def __init__(
        self,
        rng: random.Random,
        routes_path: Path,
        airports_path: Path,
        markov_path: Path,
        turnaround_intraday_params_path: Path,
        turnaround_temporal_profile_path: Path,
        window_length_mins: int = END_OF_DAY_MINS,
    ):
        self.rng = rng
        self.routes_path = routes_path
        self.airports_path = airports_path
        self.markov_path = markov_path
        self.turnaround_intraday_params_path = turnaround_intraday_params_path
        self.turnaround_temporal_profile_path = turnaround_temporal_profile_path
        self.window_length_mins = int(window_length_mins)
        self.hour_bins = max(1, self.window_length_mins // P_NEXT_BIN_SIZE_MINS)

        self.turnaround_lookup_stats: Dict[str, int] = defaultdict(int)
        self._load_all()

    def _load_all(self) -> None:
        """Orchestrate loading of all datasets."""
        print("[Schedule] Loading data...")
        print(f"[Schedule]   Routes: {self.routes_path}")
        print(f"[Schedule]   Airports: {self.airports_path}")
        print(f"[Schedule]   Window length mins: {self.window_length_mins}")

        self._load_markov_data()
        self._load_turnaround_data()
        self._load_route_data()
        self._load_airport_data()

        print(f"[Schedule]   Markov hourly states: {len(self.markov_hourly)}")
        print(f"[Schedule]   Turnaround intraday keys: {len(self.turnaround_intraday_params)}")
        print(f"[Schedule]   Turnaround temporal profile keys: {len(self.turnaround_temporal_profiles)}")
        print(f"[Schedule]   Turnaround p_next (airline,wake) keys: {len(self.turnaround_temporal_next_prob)}")
        print(f"[Schedule]   Routes: {len(self.routes)}")

    @staticmethod
    def _norm(value: object) -> str:
        """Normalize values used as key components."""
        return str(value).strip()

    # ---------------------------------------------------------
    # Data Loading Methods
    # ---------------------------------------------------------

    def _read_markov_df(self) -> pd.DataFrame:
        """Load Markov transitions CSV."""
        return pd.read_csv(self.markov_path)

    def _drop_end_transitions(self, markov_df: pd.DataFrame) -> pd.DataFrame:
        """Remove END transitions (chain should remain continuous during generation)."""
        if "ARR_ICAO" not in markov_df.columns:
            return markov_df

        arr_codes = markov_df["ARR_ICAO"].astype(str).str.upper().str.strip()
        end_mask = arr_codes == "END"
        end_rows = int(end_mask.sum())
        if end_rows:
            print(f"[Schedule]   Ignored {end_rows} Markov END transitions")
            return markov_df[~end_mask].copy()
        return markov_df

    def _build_markov_tables(self, markov_df: pd.DataFrame) -> None:
        """Build primary and fallback Markov hourly dictionaries."""
        self.markov_hourly = {}
        self.markov_fallback_hourly = {}

        operators = markov_df["AC_OPER"].to_numpy(dtype=object)
        wakes = markov_df["AC_WAKE"].to_numpy(dtype=object)
        prev_origins = markov_df["PREV_ICAO"].to_numpy(dtype=object)
        origins = markov_df["DEP_ICAO"].to_numpy(dtype=object)
        destinations = markov_df["ARR_ICAO"].to_numpy(dtype=object)
        counts = markov_df["COUNT"].astype(int).to_numpy()
        dep_hour_col = "DEP_HOUR_REFTZ" if "DEP_HOUR_REFTZ" in markov_df.columns else "DEP_HOUR_UTC"
        dep_hours = pd.to_numeric(markov_df[dep_hour_col], errors="coerce").fillna(12).astype(int).to_numpy()

        for index in range(len(operators)):
            primary_key = (
                operators[index],
                wakes[index],
                prev_origins[index],
                origins[index],
            )
            fallback_key = (operators[index], wakes[index], origins[index])
            dep_hour = int(dep_hours[index])
            destination = destinations[index]
            count = int(counts[index])

            primary_hourly = self.markov_hourly.setdefault(primary_key, {})
            primary_hourly.setdefault(dep_hour, {})
            primary_hourly[dep_hour][destination] = count

            fallback_hourly = self.markov_fallback_hourly.setdefault(fallback_key, {})
            fallback_hourly.setdefault(dep_hour, {})
            existing = fallback_hourly[dep_hour].get(destination, 0)
            fallback_hourly[dep_hour][destination] = existing + count

    def _load_markov_data(self) -> None:
        """Load and prepare Markov transition lookup tables."""
        markov_df = self._read_markov_df()
        markov_df = self._drop_end_transitions(markov_df)
        self._build_markov_tables(markov_df)

    @staticmethod
    def _validate_columns(df: pd.DataFrame, expected: set[str], label: str) -> None:
        """Validate required columns for a CSV-backed table."""
        if expected.issubset(df.columns):
            return
        raise ValueError(
            f"Missing columns in {label}: expected {sorted(expected)} got {list(df.columns)}"
        )

    def _load_intraday_params(self, intraday_df: pd.DataFrame) -> Dict[Tuple[str, str], Tuple[float, float]]:
        """Parse and validate intraday turnaround lognormal parameters."""
        params: Dict[Tuple[str, str], Tuple[float, float]] = {}

        for row in intraday_df.itertuples(index=False):
            key = (self._norm(row.airline), self._norm(row.wake))
            if key in params:
                raise ValueError(f"Duplicate intraday turnaround key: {key}")

            location = float(row.location)
            shape = float(row.shape)
            if not np.isfinite(location) or not np.isfinite(shape) or shape <= 0:
                raise ValueError(
                    f"Invalid intraday params for key {key}: loc={location}, shape={shape}"
                )

            params[key] = (location, shape)

        return params

    def _load_temporal_profiles(
        self,
        temporal_df: pd.DataFrame,
    ) -> tuple[
        Dict[tuple, Tuple[Dict[int, float], Dict[int, float]]],
        Dict[tuple, Tuple[float, float]],
        Dict[Tuple[str, str], Dict[int, float]],
        Dict[Tuple[str, str], Dict[int, float]],
    ]:
        """Parse temporal turnaround profiles and aggregate by (airline, wake)."""
        temporal_profiles: Dict[tuple, Tuple[Dict[int, float], Dict[int, float]]] = {}
        temporal_totals: Dict[tuple, Tuple[float, float]] = {}

        by_airline_wake_intraday: Dict[Tuple[str, str], Dict[int, float]] = {}
        by_airline_wake_next_day: Dict[Tuple[str, str], Dict[int, float]] = {}

        for row in temporal_df.itertuples(index=False):
            key = (
                self._norm(row.airline),
                self._norm(row.previous_origin),
                self._norm(row.origin),
                self._norm(row.wake),
            )
            if key in temporal_profiles:
                raise ValueError(f"Duplicate temporal key: {key}")

            intraday_sparse = self._decode_sparse_counts(row.intraday_sparse)
            next_day_sparse = self._decode_sparse_counts(row.next_day_sparse)

            temporal_profiles[key] = (intraday_sparse, next_day_sparse)
            temporal_totals[key] = (float(row.total_intraday), float(row.total_next_day))

            airline_wake_key = (self._norm(row.airline), self._norm(row.wake))
            if airline_wake_key not in by_airline_wake_intraday:
                by_airline_wake_intraday[airline_wake_key] = {}
                by_airline_wake_next_day[airline_wake_key] = {}

            for bin_idx, count in intraday_sparse.items():
                prev = by_airline_wake_intraday[airline_wake_key].get(bin_idx, 0.0)
                by_airline_wake_intraday[airline_wake_key][bin_idx] = prev + count

            for bin_idx, count in next_day_sparse.items():
                prev = by_airline_wake_next_day[airline_wake_key].get(bin_idx, 0.0)
                by_airline_wake_next_day[airline_wake_key][bin_idx] = prev + count

        return (
            temporal_profiles,
            temporal_totals,
            by_airline_wake_intraday,
            by_airline_wake_next_day,
        )

    def _build_turnaround_next_probabilities(
        self,
        by_airline_wake_intraday: Dict[Tuple[str, str], Dict[int, float]],
        by_airline_wake_next_day: Dict[Tuple[str, str], Dict[int, float]],
    ) -> Dict[Tuple[str, str], np.ndarray]:
        """Build hourly p(next_day) vectors by (airline, wake)."""
        next_probabilities: Dict[Tuple[str, str], np.ndarray] = {}

        all_keys = set(by_airline_wake_intraday.keys()) | set(by_airline_wake_next_day.keys())
        for airline_wake_key in all_keys:
            intra_counts = by_airline_wake_intraday.get(airline_wake_key, {})
            next_counts = by_airline_wake_next_day.get(airline_wake_key, {})
            next_probabilities[airline_wake_key] = self._build_temporal_next_probability_vector(
                intra_counts,
                next_counts,
            )

        return next_probabilities

    def _load_turnaround_data(self) -> None:
        """Load turnaround parameters and temporal profiles."""
        self.turnaround_intraday_params: Dict[Tuple[str, str], Tuple[float, float]] = {}
        self.turnaround_temporal_profiles: Dict[tuple, Tuple[Dict[int, float], Dict[int, float]]] = {}
        self.turnaround_temporal_totals: Dict[tuple, Tuple[float, float]] = {}
        self.turnaround_temporal_next_prob: Dict[Tuple[str, str], np.ndarray] = {}

        intraday_df = pd.read_csv(self.turnaround_intraday_params_path)
        temporal_df = pd.read_csv(self.turnaround_temporal_profile_path)

        self._validate_columns(
            intraday_df,
            {"airline", "wake", "location", "shape"},
            "intraday params",
        )
        self._validate_columns(
            temporal_df,
            {
                "airline",
                "previous_origin",
                "origin",
                "wake",
                "intraday_sparse",
                "next_day_sparse",
                "total_intraday",
                "total_next_day",
            },
            "temporal profile",
        )

        self.turnaround_intraday_params = self._load_intraday_params(intraday_df)
        (
            self.turnaround_temporal_profiles,
            self.turnaround_temporal_totals,
            by_airline_wake_intraday,
            by_airline_wake_next_day,
        ) = self._load_temporal_profiles(temporal_df)

        self.turnaround_temporal_next_prob = self._build_turnaround_next_probabilities(
            by_airline_wake_intraday,
            by_airline_wake_next_day,
        )

    def _load_route_data(self) -> None:
        """Load route durations."""
        routes_df = pd.read_csv(self.routes_path)
        self.routes: Dict[tuple, int] = {}
        for row in routes_df.itertuples():
            key = (row.orig_id, row.dest_id, row.airline_id, row.wake_type)
            self.routes[key] = int(row.scheduled_time)

    def _load_airport_data(self) -> None:
        """Load airport capacity limits and timezone offsets."""
        airports_df = pd.read_csv(self.airports_path)
        self.rolling_capacity: Dict[str, float] = {}
        self.burst_capacity: Dict[str, float] = {}

        for row in airports_df.itertuples():
            self.rolling_capacity[row.airport_id] = float(row.rolling_capacity)
            self.burst_capacity[row.airport_id] = float(row.burst_capacity)

        self._build_tz_offsets()

    # ---------------------------------------------------------
    # Helper Utilities
    # ---------------------------------------------------------

    def _decode_sparse_counts(self, payload: str) -> Dict[int, float]:
        """Decode sparse temporal payload like 'minute:count;minute:count'."""
        out: Dict[int, float] = {}
        if payload is None:
            return out
        if isinstance(payload, float) and np.isnan(payload):
            return out

        text = str(payload).strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return out

        for token in text.split(";"):
            token = token.strip()
            if not token:
                continue
            if ":" not in token:
                raise ValueError(f"Invalid sparse temporal token '{token}'")

            minute_s, count_s = token.split(":", 1)
            minute = int(minute_s)
            if minute < 0 or minute >= END_OF_DAY_MINS or (minute % BIN_SIZE_MINS) != 0:
                raise ValueError(f"Invalid sparse temporal minute '{minute}'")

            count = float(count_s)
            if count > 0:
                out[minute // BIN_SIZE_MINS] = count

        return out

    def _build_temporal_next_probability_vector(
        self,
        intraday_counts: Dict[int, float],
        next_day_counts: Dict[int, float],
    ) -> np.ndarray:
        """Build empirical hourly p(next_day) from aggregated 5-minute counts."""
        n_bins = self.hour_bins
        intra_hourly = np.zeros(n_bins, dtype=float)
        next_hourly = np.zeros(n_bins, dtype=float)

        for bin_idx, count in intraday_counts.items():
            minute = int(bin_idx) * BIN_SIZE_MINS
            hour_idx = minute // P_NEXT_BIN_SIZE_MINS
            if 0 <= hour_idx < n_bins and count > 0:
                intra_hourly[hour_idx] += float(count)

        for bin_idx, count in next_day_counts.items():
            minute = int(bin_idx) * BIN_SIZE_MINS
            hour_idx = minute // P_NEXT_BIN_SIZE_MINS
            if 0 <= hour_idx < n_bins and count > 0:
                next_hourly[hour_idx] += float(count)

        totals = intra_hourly + next_hourly
        n_total = float(np.sum(totals))
        if n_total <= 0:
            return np.zeros(n_bins, dtype=float)

        probs = np.zeros(n_bins, dtype=float)
        nonzero = totals > 0
        probs[nonzero] = next_hourly[nonzero] / totals[nonzero]
        return probs

    def _build_tz_offsets(self) -> None:
        """Build timezone offset cache for tracked airports."""
        airports_db = airportsdata.load("ICAO")
        ref_date = datetime(2023, 9, 15)
        self.tz_offset: Dict[str, int] = {}

        for airport in set(self.rolling_capacity.keys()):
            try:
                tz_str = airports_db.get(airport, {}).get("tz", "UTC")
                tz = pytz.timezone(tz_str)
                self.tz_offset[airport] = int(tz.utcoffset(ref_date).total_seconds() // 3600)
            except Exception:
                self.tz_offset[airport] = 0

    def get_reftz_hour(self, reftz_mins: int) -> int:
        """Convert minute-of-window to REFTZ hour bin."""
        return (int(reftz_mins) // 60) % self.hour_bins

    def get_utc_hour(self, utc_mins: int) -> int:
        """Backward-compatible alias kept for tests/callers."""
        return self.get_reftz_hour(utc_mins)

    # ---------------------------------------------------------
    # Markov Lookup
    # ---------------------------------------------------------

    def _find_hourly_data_with_radius(
        self,
        data_source: dict,
        key: tuple,
        center_hour: int,
        max_radius: int = 2,
    ) -> Tuple[Dict[str, int], int]:
        """Find transition counts at center hour or nearby hours."""
        hourly_data = data_source.get(key, {})
        if not hourly_data:
            return {}, -1

        candidates = [center_hour]
        for radius in range(1, max_radius + 1):
            if self.rng.random() < 0.5:
                candidates.append((center_hour - radius) % self.hour_bins)
                candidates.append((center_hour + radius) % self.hour_bins)
            else:
                candidates.append((center_hour + radius) % self.hour_bins)
                candidates.append((center_hour - radius) % self.hour_bins)

        for hour in candidates:
            if hour in hourly_data:
                return hourly_data[hour], hour

        available_hours = sorted(hourly_data.keys())
        future_hours = [hour for hour in available_hours if hour >= center_hour]
        if future_hours:
            next_hour = future_hours[0]
            return hourly_data[next_hour], next_hour

        return {}, -1

    def get_destinations(
        self,
        op: str,
        wake: str,
        prev_origin: str,
        origin: str,
        dep_utc_mins: int,
        arr_utc_mins: int,
    ) -> Tuple[List[Tuple[str, float]], str]:
        """Get destination probabilities from primary/fallback Markov tables."""
        del arr_utc_mins

        primary_key = (op, wake, prev_origin, origin)
        fallback_key = (op, wake, origin)
        dep_hour = self.get_reftz_hour(dep_utc_mins)

        dest_counts, found_hour = self._find_hourly_data_with_radius(
            self.markov_hourly,
            primary_key,
            dep_hour,
        )
        source = "none"

        if dest_counts:
            source = "primary_exact" if found_hour == dep_hour else "primary_expanded"
        else:
            dest_counts, found_hour = self._find_hourly_data_with_radius(
                self.markov_fallback_hourly,
                fallback_key,
                dep_hour,
            )
            if dest_counts:
                source = "fallback_expanded"

        if source == "none":
            if prev_origin and prev_origin != origin:
                return [(prev_origin, 1.0)], "return_to_origin"
            return [], "none"

        total_count = sum(dest_counts.values())
        results: List[Tuple[str, float]] = []
        if total_count > 0:
            for destination, count in dest_counts.items():
                results.append((destination, count / total_count))

        results.sort(key=lambda pair: -pair[1])
        return results, source

    # ---------------------------------------------------------
    # Turnaround Lookup
    # ---------------------------------------------------------

    def _resolve_turnaround_key(
        self,
        params_map: Dict[Tuple[str, str], Tuple[float, float]],
        op: str,
        wake: str,
        count_stats: bool = True,
    ) -> Tuple[Optional[Tuple[str, str]], str]:
        """Resolve turnaround parameter key and update diagnostics."""
        key = (self._norm(op), self._norm(wake))
        if key in params_map:
            if count_stats:
                self.turnaround_lookup_stats["param_exact"] += 1
            return key, "exact"

        if count_stats:
            self.turnaround_lookup_stats["param_missing"] += 1
        return None, "none"

    def _resolve_temporal_next_probability(
        self,
        op: str,
        prev_origin: str,
        origin: str,
        wake: str,
    ) -> Tuple[Optional[np.ndarray], str]:
        """Resolve p(next_day) vector for (airline, wake)."""
        del prev_origin, origin

        key = (self._norm(op), self._norm(wake))
        if key in self.turnaround_temporal_next_prob:
            self.turnaround_lookup_stats["temporal_exact"] += 1
            return self.turnaround_temporal_next_prob[key], "exact"

        self.turnaround_lookup_stats["temporal_missing_exact"] += 1
        return None, "none"

    def _sample_lognormal_minutes(self, location: float, shape: float) -> int:
        """Sample turnaround in minutes and clamp to BIN_SIZE_MINS."""
        draw = self.rng.lognormvariate(float(location), float(max(shape, 1e-6)))
        ta = int(round(max(0.0, draw) / BIN_SIZE_MINS) * BIN_SIZE_MINS)
        return max(BIN_SIZE_MINS, ta)

    def _build_next_day_turnaround(self, arr_utc_mins: int) -> int:
        """Compute turnaround that delays departure to next day."""
        min_needed = max(BIN_SIZE_MINS, self.window_length_mins - int(arr_utc_mins) + BIN_SIZE_MINS)
        return int(math.ceil(min_needed / BIN_SIZE_MINS) * BIN_SIZE_MINS)

    def get_turnaround_category(
        self,
        op: str,
        prev_origin: str,
        origin: str,
        wake: str,
        arr_utc_mins: int,
    ) -> str:
        """Decide whether a turnaround is intraday or next-day."""
        p_next_by_bin, _ = self._resolve_temporal_next_probability(op, prev_origin, origin, wake)
        if p_next_by_bin is None:
            return "missing"

        hour_idx = (int(arr_utc_mins) % self.window_length_mins) // P_NEXT_BIN_SIZE_MINS
        p_next = float(p_next_by_bin[hour_idx])
        return "next_day" if self.rng.random() < p_next else "intraday"

    def sample_turnaround_for_prev_origin(
        self,
        op: str,
        prev_origin: str,
        origin: str,
        wake: str,
        arr_utc_mins: int,
    ) -> Tuple[int, str]:
        """Sample turnaround using temporal category and intraday parameters."""
        category = self.get_turnaround_category(op, prev_origin, origin, wake, arr_utc_mins)
        if category == "missing":
            return -1, "missing"

        if category == "next_day":
            return self._build_next_day_turnaround(arr_utc_mins), "next_day"

        key, _ = self._resolve_turnaround_key(self.turnaround_intraday_params, op, wake)
        if key is None:
            return -1, "missing"

        location, shape = self.turnaround_intraday_params[key]

        max_intraday = self.window_length_mins - int(arr_utc_mins) - BIN_SIZE_MINS
        max_intraday = int(math.floor(max_intraday / BIN_SIZE_MINS) * BIN_SIZE_MINS)
        if max_intraday < BIN_SIZE_MINS:
            return self._build_next_day_turnaround(arr_utc_mins), "next_day"

        for _ in range(MAX_INTRADAY_RESAMPLE_ATTEMPTS):
            sampled = self._sample_lognormal_minutes(location, shape)
            if sampled <= max_intraday:
                return sampled, "intraday"

        self.turnaround_lookup_stats["intraday_resample_guard"] += 1
        return max_intraday, "intraday"

    def get_turnaround_options(
        self,
        op: str,
        prev_origin: str,
        origin: str,
        wake: str,
        arr_utc_mins: int,
    ) -> List[Tuple[int, float]]:
        """Intentional API stub: no fallback option list is currently provided."""
        del op, prev_origin, origin, wake, arr_utc_mins
        return []

    # ---------------------------------------------------------
    # Route Lookup
    # ---------------------------------------------------------

    def get_flight_time(
        self,
        origin: str,
        dest: str,
        op: str,
        wake: str,
        dep_utc_mins: int = None,
    ) -> int:
        """Lookup route duration using operator-specific then ALL fallback."""
        del dep_utc_mins
        return self._get_flight_time_median(origin, dest, op, wake)

    def _get_flight_time_median(self, origin: str, dest: str, op: str, wake: str) -> int:
        """Internal route-duration lookup with operator fallback."""
        duration = self.routes.get((origin, dest, op, wake), 0)
        if duration == 0:
            duration = self.routes.get((origin, dest, "ALL", wake), 0)

        if duration > 0:
            duration = int(duration // 5 * 5)
        return duration
