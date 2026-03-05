"""
Schedule Generator - Greedy Forward Construction

Single-phase approach: Build valid chains greedily from fixed initial conditions.
First flight is real data from initial_conditions.csv - never modified.

Turnaround Selection Logic:
1. Use (airline, wake) temporal probability for p(next_day) -- 60-minute bins
2. If next-day is selected, terminate the chain at day boundary
3. Otherwise sample intraday turnaround from exact-key (airline, wake) lognormal fit
4. Search feasible departure times in 5-minute bins

Uses 2nd-order Markov: P(Dest | Operator, Wake, PrevAirport, CurrentAirport, TimeBin).

Output:
  - schedule{suffix}.csv
"""

from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import airportsdata
import numpy as np
import pandas as pd
import pytz
from datetime import datetime

from .config import PipelineConfig

# --- Constants ---

BIN_SIZE_MINS = 5
END_OF_DAY = 1440
P_NEXT_BIN_SIZE_MINS = 60
MAX_INTRADAY_RESAMPLE_ATTEMPTS = 256

# --- Data Structures ---

@dataclass
class Flight:
    orig: str
    dest: str
    std: int   # Scheduled Time of Departure (minutes from midnight UTC)
    sta: int   # Scheduled Time of Arrival (minutes from midnight UTC)
    turnaround_to_next_category: str = ""
    turnaround_to_next_minutes: int = -1


@dataclass
class Aircraft:
    reg: str
    operator: str
    wake: str
    initial_flight: Optional[Flight] = None
    prior_flight: Optional[Flight] = None
    chain: List[Flight] = field(default_factory=list)
    is_single_flight: bool = False
    is_prior_only: bool = False


@dataclass
class GenerationStats:
    """Track statistics during schedule generation."""
    total_aircraft: int = 0
    successful_chains: int = 0
    total_flights: int = 0

    ta_scheduled_primary: int = 0
    ta_scheduled_secondary: int = 0
    ta_interval_search: int = 0
    ta_extended: int = 0

    end_of_day: int = 0
    no_destinations: int = 0
    capacity_exhausted: int = 0

    single_flight_passthrough: int = 0
    single_flight_with_prior: int = 0
    prior_flight_pasted: int = 0
    prior_only_pasted: int = 0

    dest_found_primary_exact: int = 0
    dest_found_primary_expanded: int = 0
    dest_found_fallback_expanded: int = 0
    dest_found_return_to_origin: int = 0

    examples_no_destinations: List[str] = field(default_factory=list)
    examples_capacity_exhausted: List[str] = field(default_factory=list)

    single_flight_total: int = 0
    single_flight_end_of_day: int = 0
    single_flight_no_destinations: int = 0
    single_flight_capacity_exhausted: int = 0
    single_flight_termination_hours: Dict[int, int] = field(default_factory=lambda: defaultdict(int))

    def add_example_no_destinations(self, msg: str):
        if len(self.examples_no_destinations) < 2:
            self.examples_no_destinations.append(msg)

    def add_example_capacity_exhausted(self, msg: str):
        if len(self.examples_capacity_exhausted) < 2:
            self.examples_capacity_exhausted.append(msg)

    def summary(self) -> str:
        lines = [
            "=== Generation Statistics ===",
            f"Aircraft: {self.successful_chains}/{self.total_aircraft} successful",
            f"Total flights: {self.total_flights}",
            "",
            "--- Turnaround Selection ---",
            f"  Scheduled (Primary): {self.ta_scheduled_primary}",
            f"  Scheduled (Secondary): {self.ta_scheduled_secondary}",
            f"  Interval search: {self.ta_interval_search}",
            f"  Extended (+5m): {self.ta_extended}",
            "",
            "--- Chain Termination ---",
            f"  Single-flight passthrough (real data): {self.single_flight_passthrough}",
            f"    With overnight arrival: {self.single_flight_with_prior}",
            f"  Prior flights pasted (all aircraft): {self.prior_flight_pasted}",
            f"  Prior-only aircraft pasted: {self.prior_only_pasted}",
            f"  End of day: {self.end_of_day}",
            f"  No Markov data: {self.no_destinations}",
            f"  Capacity exhausted: {self.capacity_exhausted}",
            "",
            "--- Fallback Usage ---",
            f"  Primary Exact: {self.dest_found_primary_exact}",
            f"  Primary Expanded: {self.dest_found_primary_expanded}",
            f"  Fallback Expanded: {self.dest_found_fallback_expanded}",
            f"  Return to Origin: {self.dest_found_return_to_origin}",
            "",
            "--- Single Flight Chains Analysis ---",
            f"  Total Single Flights: {self.single_flight_total}",
            f"  Reasons:",
            f"    End of Day (Turnaround > Midnight): {self.single_flight_end_of_day}",
            f"    No Markov Data: {self.single_flight_no_destinations}",
            f"    Capacity Exhausted: {self.single_flight_capacity_exhausted}",
            "",
            "  Termination Hour Distribution (End of Day):",
            *[f"    {h:02d}h: {c}" for h, c in sorted(self.single_flight_termination_hours.items()) if c > 0],
        ]

        if self.examples_no_destinations:
            lines.append("")
            lines.append("--- Examples: No Destinations ---")
            for ex in self.examples_no_destinations:
                lines.append(ex)

        if self.examples_capacity_exhausted:
            lines.append("")
            lines.append("--- Examples: Capacity Exhausted ---")
            for ex in self.examples_capacity_exhausted:
                lines.append(ex)

        return "\n".join(lines)


# --- Data Loaders ---

class DataManager:
    """Holds all lookup tables for 2nd-order Markov, Turnaround, Routes."""

    def __init__(
        self,
        rng: random.Random,
        routes_path: Path,
        airports_path: Path,
        markov_path: Path,
        turnaround_intraday_params_path: Path,
        turnaround_temporal_profile_path: Path,
    ):
        self.rng = rng
        self.routes_path = routes_path
        self.airports_path = airports_path
        self.markov_path = markov_path
        self.turnaround_intraday_params_path = turnaround_intraday_params_path
        self.turnaround_temporal_profile_path = turnaround_temporal_profile_path
        self.turnaround_lookup_stats: Dict[str, int] = defaultdict(int)
        self._load_all()

    def _load_all(self):
        print(f"[Schedule] Loading data...")
        print(f"[Schedule]   Routes: {self.routes_path}")
        print(f"[Schedule]   Airports: {self.airports_path}")

        # 2nd-Order Markov
        markov_df = pd.read_csv(self.markov_path)
        self.markov_hourly: Dict[tuple, Dict[int, Dict[str, int]]] = {}
        self.markov_fallback_hourly: Dict[tuple, Dict[int, Dict[str, int]]] = {}

        if "ARR_ICAO" in markov_df.columns:
            arr_codes = markov_df["ARR_ICAO"].astype(str).str.upper().str.strip()
            end_mask = arr_codes == "END"
            end_rows = int(end_mask.sum())
            if end_rows:
                markov_df = markov_df[~end_mask].copy()
                print(f"[Schedule]   Ignored {end_rows} Markov END transitions")

        _m_ops = markov_df["AC_OPERATOR"].to_numpy(dtype=object)
        _m_wakes = markov_df["AC_WAKE"].to_numpy(dtype=object)
        _m_prevs = markov_df["PREV_ICAO"].to_numpy(dtype=object)
        _m_deps = markov_df["DEP_ICAO"].to_numpy(dtype=object)
        _m_arrs = markov_df["ARR_ICAO"].to_numpy(dtype=object)
        _m_counts = markov_df["COUNT"].astype(int).to_numpy()
        _m_hours = pd.to_numeric(markov_df["DEP_HOUR_UTC"], errors="coerce").fillna(12).astype(int).to_numpy()

        for i in range(len(_m_ops)):
            key = (_m_ops[i], _m_wakes[i], _m_prevs[i], _m_deps[i])
            dep_hour = int(_m_hours[i])

            if key not in self.markov_hourly:
                self.markov_hourly[key] = {}
            if dep_hour not in self.markov_hourly[key]:
                self.markov_hourly[key][dep_hour] = {}
            self.markov_hourly[key][dep_hour][_m_arrs[i]] = int(_m_counts[i])

            fb_key = (_m_ops[i], _m_wakes[i], _m_deps[i])
            if fb_key not in self.markov_fallback_hourly:
                self.markov_fallback_hourly[fb_key] = {}
            if dep_hour not in self.markov_fallback_hourly[fb_key]:
                self.markov_fallback_hourly[fb_key][dep_hour] = {}
            self.markov_fallback_hourly[fb_key][dep_hour][_m_arrs[i]] = (
                self.markov_fallback_hourly[fb_key][dep_hour].get(_m_arrs[i], 0) + int(_m_counts[i])
            )

        # Turnaround parameter files
        self.turnaround_intraday_params: Dict[Tuple[str, str], Tuple[float, float]] = {}
        self.turnaround_temporal_profiles: Dict[tuple, Tuple[Dict[int, float], Dict[int, float]]] = {}
        self.turnaround_temporal_totals: Dict[tuple, Tuple[float, float]] = {}
        self.turnaround_temporal_next_prob: Dict[Tuple[str, str], np.ndarray] = {}

        intraday_df = pd.read_csv(self.turnaround_intraday_params_path)
        temporal_df = pd.read_csv(self.turnaround_temporal_profile_path)

        expected_intraday_cols = {"airline", "wake", "location", "shape"}
        if not expected_intraday_cols.issubset(intraday_df.columns):
            raise ValueError(
                f"Missing columns in intraday params: expected {sorted(expected_intraday_cols)} "
                f"got {list(intraday_df.columns)}"
            )

        expected_temporal_cols = {
            "airline",
            "previous_origin",
            "origin",
            "wake",
            "intraday_sparse",
            "next_day_sparse",
            "total_intraday",
            "total_next_day",
        }
        if not expected_temporal_cols.issubset(temporal_df.columns):
            raise ValueError(
                f"Missing columns in temporal profile: expected {sorted(expected_temporal_cols)} "
                f"got {list(temporal_df.columns)}"
            )

        for row in intraday_df.itertuples(index=False):
            key = (str(row.airline).strip(), str(row.wake).strip())
            if key in self.turnaround_intraday_params:
                raise ValueError(f"Duplicate intraday turnaround key: {key}")
            loc = float(row.location)
            shape = float(row.shape)
            if not np.isfinite(loc) or not np.isfinite(shape) or shape <= 0:
                raise ValueError(f"Invalid intraday params for key {key}: loc={loc}, shape={shape}")
            self.turnaround_intraday_params[key] = (loc, shape)

        # Intermediate accumulators for p_next aggregation
        _aw_intra: Dict[Tuple[str, str], Dict[int, float]] = {}
        _aw_next: Dict[Tuple[str, str], Dict[int, float]] = {}

        for row in temporal_df.itertuples(index=False):
            key = (
                str(row.airline).strip(),
                str(row.previous_origin).strip(),
                str(row.origin).strip(),
                str(row.wake).strip(),
            )
            if key in self.turnaround_temporal_profiles:
                raise ValueError(f"Duplicate temporal key: {key}")

            intra_sparse = self._decode_sparse_counts(row.intraday_sparse)
            next_sparse = self._decode_sparse_counts(row.next_day_sparse)

            self.turnaround_temporal_profiles[key] = (intra_sparse, next_sparse)
            self.turnaround_temporal_totals[key] = (float(row.total_intraday), float(row.total_next_day))

            aw_key = (str(row.airline).strip(), str(row.wake).strip())
            if aw_key not in _aw_intra:
                _aw_intra[aw_key] = {}
                _aw_next[aw_key] = {}
            for bin_idx, count in intra_sparse.items():
                _aw_intra[aw_key][bin_idx] = _aw_intra[aw_key].get(bin_idx, 0.0) + count
            for bin_idx, count in next_sparse.items():
                _aw_next[aw_key][bin_idx] = _aw_next[aw_key].get(bin_idx, 0.0) + count

        for aw_key in set(_aw_intra.keys()) | set(_aw_next.keys()):
            self.turnaround_temporal_next_prob[aw_key] = self._build_temporal_next_probability_vector(
                _aw_intra.get(aw_key, {}),
                _aw_next.get(aw_key, {}),
            )

        # Routes
        routes_df = pd.read_csv(self.routes_path)
        self.routes: Dict[tuple, int] = {}
        for r in routes_df.itertuples():
            self.routes[(r.orig_id, r.dest_id, r.airline_id, r.wake_type)] = int(r.scheduled_time)

        # Airports: Capacity
        airports_df = pd.read_csv(self.airports_path)
        self.rolling_capacity: Dict[str, float] = {}
        self.burst_capacity: Dict[str, float] = {}

        for r in airports_df.itertuples():
            self.rolling_capacity[r.airport_id] = float(r.rolling_capacity)
            self.burst_capacity[r.airport_id] = float(r.burst_capacity)

        self._build_tz_offsets()

        print(f"[Schedule]   Markov hourly states: {len(self.markov_hourly)}")
        print(f"[Schedule]   Turnaround intraday keys: {len(self.turnaround_intraday_params)}")
        print(f"[Schedule]   Turnaround temporal profile keys: {len(self.turnaround_temporal_profiles)}")
        print(f"[Schedule]   Turnaround p_next (airline,wake) keys: {len(self.turnaround_temporal_next_prob)}")
        print(f"[Schedule]   Routes: {len(self.routes)}")

    def _decode_sparse_counts(self, payload: str) -> Dict[int, float]:
        out: Dict[int, float] = {}
        if payload is None:
            return out
        if isinstance(payload, float) and np.isnan(payload):
            return out
        text = str(payload).strip()
        if not text:
            return out
        if text.lower() in {"nan", "none", "null"}:
            return out
        for token in text.split(";"):
            token = token.strip()
            if not token:
                continue
            if ":" not in token:
                raise ValueError(f"Invalid sparse temporal token '{token}'")
            minute_s, count_s = token.split(":", 1)
            minute = int(minute_s)
            if minute < 0 or minute >= END_OF_DAY or (minute % BIN_SIZE_MINS) != 0:
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
        """Build empirical hourly p(next_day) from real counts."""
        n_bins = END_OF_DAY // P_NEXT_BIN_SIZE_MINS
        intra_hourly = np.zeros(n_bins, dtype=float)
        next_hourly = np.zeros(n_bins, dtype=float)

        for idx, count in intraday_counts.items():
            minute = int(idx) * BIN_SIZE_MINS
            hour_idx = minute // P_NEXT_BIN_SIZE_MINS
            if 0 <= hour_idx < n_bins and count > 0:
                intra_hourly[hour_idx] += float(count)

        for idx, count in next_day_counts.items():
            minute = int(idx) * BIN_SIZE_MINS
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

    def _build_tz_offsets(self):
        airports_db = airportsdata.load("ICAO")
        ref_date = datetime(2023, 9, 15)
        self.tz_offset: Dict[str, int] = {}
        for icao in set(self.rolling_capacity.keys()):
            try:
                tz_str = airports_db.get(icao, {}).get("tz", "UTC")
                tz = pytz.timezone(tz_str)
                self.tz_offset[icao] = int(tz.utcoffset(ref_date).total_seconds() // 3600)
            except Exception:
                self.tz_offset[icao] = 0

    def get_utc_hour(self, utc_mins: int) -> int:
        return (utc_mins // 60) % 24

    def _find_hourly_data_with_radius(
        self, data_source: dict, key: tuple, center_hour: int, max_radius: int = 2,
    ) -> Tuple[Dict[str, int], int]:
        """Find hourly data using expanding radius search, then next available hour fallback."""
        hourly_data = data_source.get(key, {})
        if not hourly_data:
            return {}, -1

        candidates = [center_hour]
        for radius in range(1, max_radius + 1):
            if self.rng.random() < 0.5:
                candidates.append((center_hour - radius) % 24)
                candidates.append((center_hour + radius) % 24)
            else:
                candidates.append((center_hour + radius) % 24)
                candidates.append((center_hour - radius) % 24)

        for h in candidates:
            if h in hourly_data:
                return hourly_data[h], h

        available_hours = sorted(hourly_data.keys())
        future_hours = [h for h in available_hours if h >= center_hour]

        if future_hours:
            next_hour = future_hours[0]
            return hourly_data[next_hour], next_hour

        return {}, -1

    def get_destinations(
        self, op: str, wake: str, prev_origin: str, origin: str,
        dep_utc_mins: int, arr_utc_mins: int,
    ) -> Tuple[List[Tuple[str, float]], str]:
        """Get destination probabilities with fallback logic.

        Returns (destinations, source_type) where source_type is one of:
        'primary_exact', 'primary_expanded', 'fallback_expanded', 'return_to_origin', 'none'.
        """
        primary_key = (op, wake, prev_origin, origin)
        fallback_key = (op, wake, origin)
        dep_hour = self.get_utc_hour(dep_utc_mins)

        non_end, found_hour = self._find_hourly_data_with_radius(self.markov_hourly, primary_key, dep_hour)
        source = "none"

        if non_end:
            source = "primary_exact" if found_hour == dep_hour else "primary_expanded"
        else:
            non_end, found_hour = self._find_hourly_data_with_radius(
                self.markov_fallback_hourly, fallback_key, dep_hour,
            )
            if non_end:
                source = "fallback_expanded"

        if source == "none":
            if prev_origin and prev_origin != origin:
                return [(prev_origin, 1.0)], "return_to_origin"
            return [], "none"

        non_end_total = sum(non_end.values())
        results = []
        if non_end_total > 0:
            for dest, count in non_end.items():
                results.append((dest, count / non_end_total))

        results.sort(key=lambda x: -x[1])
        return results, source

    def _resolve_turnaround_key(
        self,
        params_map: Dict[Tuple[str, str], Tuple[float, float]],
        op: str,
        wake: str,
        count_stats: bool = True,
    ) -> Tuple[Optional[Tuple[str, str]], str]:
        key = (str(op).strip(), str(wake).strip())
        if key in params_map:
            if count_stats:
                self.turnaround_lookup_stats["param_exact"] += 1
            return key, "exact"
        if count_stats:
            self.turnaround_lookup_stats["param_missing"] += 1
        return None, "none"

    def _resolve_temporal_next_probability(
        self, op: str, prev_origin: str, origin: str, wake: str,
    ) -> Tuple[Optional[np.ndarray], str]:
        op = str(op).strip()
        wake = str(wake).strip()

        key = (op, wake)
        if key in self.turnaround_temporal_next_prob:
            self.turnaround_lookup_stats["temporal_exact"] += 1
            return self.turnaround_temporal_next_prob[key], "exact"

        self.turnaround_lookup_stats["temporal_missing_exact"] += 1
        return None, "none"

    def _sample_lognormal_minutes(self, location: float, shape: float) -> int:
        draw = self.rng.lognormvariate(float(location), float(max(shape, 1e-6)))
        ta = int(round(max(0.0, draw) / BIN_SIZE_MINS) * BIN_SIZE_MINS)
        return max(BIN_SIZE_MINS, ta)

    def _build_next_day_turnaround(self, arr_utc_mins: int) -> int:
        min_needed = max(BIN_SIZE_MINS, END_OF_DAY - int(arr_utc_mins) + BIN_SIZE_MINS)
        return int(math.ceil(min_needed / BIN_SIZE_MINS) * BIN_SIZE_MINS)

    def get_turnaround_category(
        self, op: str, prev_origin: str, origin: str, wake: str, arr_utc_mins: int,
    ) -> str:
        p_next_by_bin, _ = self._resolve_temporal_next_probability(op, prev_origin, origin, wake)
        if p_next_by_bin is None:
            return "missing"
        idx = (int(arr_utc_mins) % END_OF_DAY) // P_NEXT_BIN_SIZE_MINS
        p_next = float(p_next_by_bin[idx])
        return "next_day" if self.rng.random() < p_next else "intraday"

    def sample_turnaround_for_prev_origin(
        self, op: str, prev_origin: str, origin: str, wake: str, arr_utc_mins: int,
    ) -> Tuple[int, str]:
        """Sample turnaround at *origin* conditioned on having come from *prev_origin*."""
        category = self.get_turnaround_category(op, prev_origin, origin, wake, arr_utc_mins)
        if category == "missing":
            return -1, "missing"

        if category == "next_day":
            return self._build_next_day_turnaround(arr_utc_mins), "next_day"

        key, _ = self._resolve_turnaround_key(self.turnaround_intraday_params, op, wake)
        if key is None:
            return -1, "missing"

        location, shape = self.turnaround_intraday_params[key]

        max_intraday = END_OF_DAY - int(arr_utc_mins) - BIN_SIZE_MINS
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
        self, op: str, prev_origin: str, origin: str, wake: str, arr_utc_mins: int,
    ) -> List[Tuple[int, float]]:
        """No pre-sampled fallback turnarounds."""
        del op, prev_origin, origin, wake, arr_utc_mins
        return []

    def get_flight_time(self, origin: str, dest: str, op: str, wake: str, dep_utc_mins: int = None) -> int:
        """Get median scheduled flight time from routes.csv."""
        del dep_utc_mins
        return self._get_flight_time_median(origin, dest, op, wake)

    def _get_flight_time_median(self, origin: str, dest: str, op: str, wake: str) -> int:
        t = self.routes.get((origin, dest, op, wake), 0)
        if t == 0:
            t = self.routes.get((origin, dest, "ALL", wake), 0)
        if t > 0:
            t = int(t // 5 * 5)
        return t


# --- Capacity Management ---

class CapacityTracker:
    def __init__(self, rolling_capacities: Dict[str, float], burst_capacities: Dict[str, float]):
        self.rolling_cap = rolling_capacities
        self.burst_cap = burst_capacities
        self.num_bins = END_OF_DAY // BIN_SIZE_MINS
        self.dep_slots: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.arr_slots: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.movements_rolling: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

    def _bin(self, t: int) -> int:
        return max(0, min(t // BIN_SIZE_MINS, self.num_bins - 1))

    def check_availability(self, orig: str, dest: str, std: int, sta: int) -> bool:
        if not (0 <= std < END_OF_DAY and 0 <= sta < END_OF_DAY):
            return True

        dep_bin = self._bin(std)
        if self.dep_slots[orig][dep_bin] >= self.burst_cap.get(orig, 999):
            return False

        limit_orig = self.rolling_cap.get(orig, 999)
        start_w = dep_bin
        end_w = min(self.num_bins - 1, dep_bin + 11)

        for w in range(start_w, end_w + 1):
            if self.movements_rolling[orig][w] >= limit_orig:
                return False

        arr_bin = self._bin(sta)
        if self.arr_slots[dest][arr_bin] >= self.burst_cap.get(dest, 999):
            return False

        limit_dest = self.rolling_cap.get(dest, 999)
        start_w = arr_bin
        end_w = min(self.num_bins - 1, arr_bin + 11)

        for w in range(start_w, end_w + 1):
            if self.movements_rolling[dest][w] >= limit_dest:
                return False
        return True

    def add_flight(self, f: Flight):
        if 0 <= f.std < END_OF_DAY:
            dep_bin = self._bin(f.std)
            self.dep_slots[f.orig][dep_bin] += 1

            start_w = dep_bin
            end_w = min(self.num_bins - 1, dep_bin + 11)
            for w in range(start_w, end_w + 1):
                self.movements_rolling[f.orig][w] += 1

        if 0 <= f.sta < END_OF_DAY:
            arr_bin = self._bin(f.sta)
            self.arr_slots[f.dest][arr_bin] += 1

            start_w = arr_bin
            end_w = min(self.num_bins - 1, arr_bin + 11)
            for w in range(start_w, end_w + 1):
                self.movements_rolling[f.dest][w] += 1

    def compute_violations(self) -> Tuple[int, int]:
        """Return (burst_violations, rolling_violations)."""
        burst = 0
        rolling = 0

        for airport, bins in self.dep_slots.items():
            limit = self.burst_cap.get(airport, 999)
            for c in bins.values():
                burst += max(0, c - int(limit))
        for airport, bins in self.arr_slots.items():
            limit = self.burst_cap.get(airport, 999)
            for c in bins.values():
                burst += max(0, c - int(limit))

        for airport, windows in self.movements_rolling.items():
            limit = self.rolling_cap.get(airport, 999)
            for c in windows.values():
                rolling += max(0, c - int(limit))

        return burst, rolling


# --- Schedule Generator ---

class ScheduleGenerator:
    """Greedy forward construction with exact-key turnaround logic."""

    def __init__(self, data: DataManager, tracker: CapacityTracker, stats: GenerationStats, rng: random.Random):
        self.data = data
        self.tracker = tracker
        self.stats = stats
        self.rng = rng

    def _get_turnaround_candidates(
        self, ac: Aircraft, prev_origin: str, current_airport: str, arrival_time: int,
    ) -> List[Tuple[int, str, float]]:
        """Get all turnaround candidates to try, including scheduled, interval, and extended."""
        ta_options = self.data.get_turnaround_options(
            ac.operator, prev_origin, current_airport, ac.wake, arrival_time,
        )

        if ta_options:
            scheduled_times = [t for t, _ in ta_options]
            min_ta = min(scheduled_times)
            max_ta = max(scheduled_times)
        else:
            min_ta = BIN_SIZE_MINS
            max_ta = END_OF_DAY - int(max(0, arrival_time)) + BIN_SIZE_MINS
            max_ta = max(BIN_SIZE_MINS, min(max_ta, END_OF_DAY - BIN_SIZE_MINS))
            ta_options = []

        candidates = []
        seen: set[int] = set()

        weighted_items = []
        for ta_time, prob in ta_options:
            r = self.rng.random()
            if r == 0:
                r = 1e-10
            weight = prob if prob > 0 else 1e-10
            score = pow(r, 1.0 / weight)
            weighted_items.append((ta_time, prob, score))

        weighted_items.sort(key=lambda x: -x[2])

        for ta_time, prob, _ in weighted_items:
            if ta_time not in seen:
                candidates.append((ta_time, "scheduled", prob))
                seen.add(ta_time)

        for ta_time in range(min_ta, max_ta + 1, 5):
            if ta_time not in seen:
                candidates.append((ta_time, "interval", 0.0))
                seen.add(ta_time)

        for ta_time in range(max_ta + 5, END_OF_DAY, 5):
            if ta_time not in seen:
                candidates.append((ta_time, "extended", 0.0))
                seen.add(ta_time)

        return candidates

    def generate_chain(self, ac: Aircraft) -> bool:
        """Generate flight chain starting from fixed initial condition."""
        ac.chain.clear()

        if ac.initial_flight is None:
            if ac.prior_flight:
                prior = Flight(
                    orig=ac.prior_flight.orig,
                    dest=ac.prior_flight.dest,
                    std=ac.prior_flight.std,
                    sta=ac.prior_flight.sta,
                )
                ac.chain.append(prior)
                self.tracker.add_flight(prior)
                self.stats.prior_flight_pasted += 1
                self.stats.prior_only_pasted += 1
                return True
            return False

        if ac.prior_flight:
            prior = Flight(
                orig=ac.prior_flight.orig,
                dest=ac.prior_flight.dest,
                std=ac.prior_flight.std,
                sta=ac.prior_flight.sta,
            )
            ac.chain.append(prior)
            self.tracker.add_flight(prior)
            self.stats.prior_flight_pasted += 1

        first_flight = Flight(
            orig=ac.initial_flight.orig,
            dest=ac.initial_flight.dest,
            std=ac.initial_flight.std,
            sta=ac.initial_flight.sta,
        )
        ac.chain.append(first_flight)
        self.tracker.add_flight(first_flight)

        if ac.is_single_flight:
            self.stats.single_flight_passthrough += 1
            if ac.prior_flight:
                self.stats.single_flight_with_prior += 1
            return True

        current_airport = first_flight.dest
        prev_origin = first_flight.orig
        arrival_time = first_flight.sta

        while arrival_time < END_OF_DAY:
            current_anchor_flight = ac.chain[-1] if ac.chain else None

            ta_effective, ta_category = self.data.sample_turnaround_for_prev_origin(
                op=ac.operator,
                prev_origin=prev_origin,
                origin=current_airport,
                wake=ac.wake,
                arr_utc_mins=arrival_time,
            )

            if ta_effective < 0:
                self.stats.no_destinations += 1
                if len(ac.chain) == 1:
                    self.stats.single_flight_total += 1
                    self.stats.single_flight_no_destinations += 1
                markov_key = (ac.operator, ac.wake, prev_origin, current_airport)
                hourly_data = self.data.markov_hourly.get(markov_key, {})
                available_hours = sorted(set(hourly_data.keys()))
                self.stats.add_example_no_destinations(
                    f"  Aircraft {ac.reg} ({ac.operator}, {ac.wake}):\n"
                    f"    At {current_airport} from {prev_origin}, arrival={arrival_time} mins\n"
                    f"    No turnaround params -- key: {(ac.operator, prev_origin, current_airport, ac.wake)}\n"
                    f"    Markov available hours: {available_hours}"
                )
                break

            std = int(math.ceil((arrival_time + ta_effective) / 5.0) * 5)

            if ta_category == "next_day" or std >= END_OF_DAY:
                if current_anchor_flight is not None:
                    current_anchor_flight.turnaround_to_next_category = "next_day"
                    current_anchor_flight.turnaround_to_next_minutes = int(ta_effective)
                self.stats.end_of_day += 1
                if len(ac.chain) == 1:
                    self.stats.single_flight_end_of_day += 1
                    self.stats.single_flight_total += 1
                    term_hour = (std // 60) % 24
                    self.stats.single_flight_termination_hours[term_hour] += 1
                break

            destinations, source_type = self.data.get_destinations(
                ac.operator, ac.wake, prev_origin, current_airport,
                dep_utc_mins=std, arr_utc_mins=arrival_time,
            )

            flight_added = False
            termination_reason = None

            if not destinations:
                termination_reason = "no_dest"
            else:
                reordered = []
                remaining = list(destinations)
                while remaining:
                    weights = [prob for _, prob in remaining]
                    idx = self.rng.choices(range(len(remaining)), weights=weights, k=1)[0]
                    reordered.append(remaining.pop(idx))

                for dest, _ in reordered:
                    flight_time = self.data.get_flight_time(
                        current_airport, dest, ac.operator, ac.wake, dep_utc_mins=std,
                    )
                    if flight_time <= 0:
                        continue

                    sta = std + flight_time

                    if sta < END_OF_DAY:
                        if not self.tracker.check_availability(current_airport, dest, std, sta):
                            termination_reason = "capacity"
                            continue

                    if current_anchor_flight is not None:
                        current_anchor_flight.turnaround_to_next_category = "intraday"
                        current_anchor_flight.turnaround_to_next_minutes = int(ta_effective)

                    flight = Flight(orig=current_airport, dest=dest, std=std, sta=sta)
                    ac.chain.append(flight)
                    if sta < END_OF_DAY:
                        self.tracker.add_flight(flight)

                    if source_type == "primary_exact":
                        self.stats.dest_found_primary_exact += 1
                    elif source_type == "primary_expanded":
                        self.stats.dest_found_primary_expanded += 1
                    elif source_type == "fallback_expanded":
                        self.stats.dest_found_fallback_expanded += 1
                    elif source_type == "return_to_origin":
                        self.stats.dest_found_return_to_origin += 1

                    prev_origin = current_airport
                    current_airport = dest
                    arrival_time = sta

                    if sta >= END_OF_DAY:
                        self.stats.end_of_day += 1

                    flight_added = True
                    break

            if not flight_added:
                if termination_reason == "capacity":
                    self.stats.capacity_exhausted += 1
                    if len(ac.chain) == 1:
                        self.stats.single_flight_total += 1
                        self.stats.single_flight_capacity_exhausted += 1
                else:
                    self.stats.no_destinations += 1
                    if len(ac.chain) == 1:
                        self.stats.single_flight_total += 1
                        self.stats.single_flight_no_destinations += 1

                    markov_key = (ac.operator, ac.wake, prev_origin, current_airport)
                    hourly_data = self.data.markov_hourly.get(markov_key, {})
                    available_hours = sorted(set(hourly_data.keys()))
                    self.stats.add_example_no_destinations(
                        f"  Aircraft {ac.reg} ({ac.operator}, {ac.wake}):\n"
                        f"    At {current_airport} from {prev_origin}, arrival={arrival_time} mins\n"
                        f"    Markov key: {markov_key}\n"
                        f"    Available departure hours: {available_hours}"
                    )
                break

        return len(ac.chain) > 0


# --- Public API ---

def generate_schedule(config: PipelineConfig) -> None:
    """Generate synthetic flight schedule via greedy forward construction.

    Reads initial conditions and analysis artefacts (Markov, turnaround,
    routes, airports) and builds a full day's schedule for each aircraft.

    Parameters
    ----------
    config : PipelineConfig
        Paths and parameters for the pipeline.

    Raises
    ------
    FileNotFoundError
        If required input files do not exist.
    """
    seed = config.seed
    suffix = config.suffix

    rng = random.Random(seed)
    np.random.seed(seed)

    output_path = config.output_path("schedule")
    routes_path = config.output_path("routes")
    airports_path = config.output_path("airports")
    initial_conditions_path = config.analysis_path("initial_conditions")
    markov_path = config.analysis_path("markov")
    turnaround_intraday_params_path = config.analysis_path("scheduled_turnaround_intraday_params")
    turnaround_temporal_profile_path = config.analysis_path("scheduled_turnaround_temporal_profile")

    print("[Schedule] --- SCHEDULE GENERATOR (Greedy Construction) ---")
    print(f"[Schedule] Seed={seed}, Suffix='{suffix}'")
    print(f"[Schedule] Output: {output_path}")
    print(
        f"[Schedule] Inputs: IC={initial_conditions_path}, Routes={routes_path}, "
        f"Airports={airports_path}"
    )
    print(
        f"[Schedule] Turnaround params: {turnaround_intraday_params_path}, "
        f"{turnaround_temporal_profile_path}"
    )

    for path in [initial_conditions_path, routes_path, airports_path, markov_path,
                 turnaround_intraday_params_path, turnaround_temporal_profile_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required input file not found: {path}")

    # Initialize Data
    data = DataManager(
        rng,
        routes_path,
        airports_path,
        markov_path,
        turnaround_intraday_params_path,
        turnaround_temporal_profile_path,
    )

    print("[Schedule] Loading initial conditions...")
    ic_df = pd.read_csv(initial_conditions_path)
    aircraft_list: List[Aircraft] = []

    for r in ic_df.itertuples():
        is_prior_only = bool(getattr(r, "PRIOR_ONLY", 0))
        initial_flight = None
        if not is_prior_only:
            std_utc_mins = getattr(r, "STD_UTC_MINS", None)
            sta_utc_mins = getattr(r, "STA_UTC_MINS", None)
            if pd.isna(std_utc_mins) or pd.isna(sta_utc_mins):
                raise ValueError(
                    f"Invalid initial_conditions row for AC_REG={r.AC_REG}: "
                    "missing STD_UTC_MINS/STA_UTC_MINS without PRIOR_ONLY=1"
                )
            initial_flight = Flight(
                orig=r.ORIGIN,
                dest=r.DEST,
                std=int(std_utc_mins),
                sta=int(sta_utc_mins),
            )
        is_single = bool(getattr(r, "SINGLE_FLIGHT", 0))

        prior_flight = None
        if hasattr(r, "PRIOR_ORIGIN") and pd.notna(getattr(r, "PRIOR_ORIGIN", None)):
            prior_flight = Flight(
                orig=r.PRIOR_ORIGIN,
                dest=r.PRIOR_DEST,
                std=int(r.PRIOR_STD_UTC_MINS),
                sta=int(r.PRIOR_STA_UTC_MINS),
            )

        ac = Aircraft(
            reg=r.AC_REG,
            operator=r.AC_OPERATOR,
            wake=r.AC_WAKE,
            initial_flight=initial_flight,
            prior_flight=prior_flight,
            is_single_flight=is_single,
            is_prior_only=is_prior_only,
        )
        aircraft_list.append(ac)

    single_flight_count = sum(1 for ac in aircraft_list if ac.is_single_flight)
    single_flight_with_prior = sum(1 for ac in aircraft_list if ac.is_single_flight and ac.prior_flight)
    prior_only_count = sum(1 for ac in aircraft_list if ac.is_prior_only)
    print(f"[Schedule]   Loaded {len(aircraft_list)} aircraft with initial conditions")
    print(f"[Schedule]   Single-flight aircraft (will passthrough): {single_flight_count}")
    print(f"[Schedule]     With overnight arrival: {single_flight_with_prior}")
    print(f"[Schedule]   Prior-only aircraft (arrival passthrough): {prior_only_count}")

    # Shuffle for variety
    rng.shuffle(aircraft_list)

    tracker = CapacityTracker(data.rolling_capacity, data.burst_capacity)
    stats = GenerationStats()
    stats.total_aircraft = len(aircraft_list)

    generator = ScheduleGenerator(data, tracker, stats, rng)

    print("[Schedule] Generating schedules...")
    for i, ac in enumerate(aircraft_list):
        success = generator.generate_chain(ac)
        if success:
            stats.successful_chains += 1
            stats.total_flights += len(ac.chain)

        if (i + 1) % 1000 == 0:
            burst, rolling = tracker.compute_violations()
            print(
                f"[Schedule]   {i + 1}/{len(aircraft_list)} | Flights: {stats.total_flights} | "
                f"Violations: burst={burst}, rolling={rolling}"
            )

    print("[Schedule] Saving schedule...")
    all_flights = []
    for ac in aircraft_list:
        initial_index = 1 if ac.prior_flight and ac.initial_flight is not None else 0
        for i, f in enumerate(ac.chain):
            is_prior_flight = 1 if (ac.prior_flight is not None and i == 0) else 0
            is_initial_departure = 1 if (ac.initial_flight is not None and i == initial_index) else 0
            all_flights.append({
                "airline_id": ac.operator,
                "aircraft_id": ac.reg,
                "orig_id": f.orig,
                "dest_id": f.dest,
                "STD_UTC": f.std,
                "STA_UTC": f.sta,
                "first_flight": 1 if i == 0 else 0,
                "is_prior_flight": is_prior_flight,
                "is_initial_departure": is_initial_departure,
                "single_flight_real": 1 if ac.is_single_flight else 0,
                "turnaround_to_next_category": f.turnaround_to_next_category,
                "turnaround_to_next_minutes": (
                    int(f.turnaround_to_next_minutes) if f.turnaround_to_next_minutes >= 0 else np.nan
                ),
            })

    df = pd.DataFrame(all_flights)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if not df.empty:
        df.to_csv(output_path, index=False)

    burst, rolling = tracker.compute_violations()
    print(f"[Schedule] Final violations: burst={burst}, rolling={rolling}")
    print("")
    print(stats.summary())
    if data.turnaround_lookup_stats:
        ordered_keys = sorted(data.turnaround_lookup_stats.keys())
        parts = [f"{k}={int(data.turnaround_lookup_stats[k])}" for k in ordered_keys]
        print("[Schedule] Turnaround lookup diagnostics: " + ", ".join(parts))
    print(f"[Schedule] Saved: {output_path} ({len(df)} flights)")
    print("[Schedule] --- SUCCESS ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic flight schedule")
    parser.add_argument("--schedule", type=str, required=True, help="Path to schedule CSV")
    parser.add_argument("--analysis-dir", type=str, required=True, help="Analysis output directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Final output directory")
    parser.add_argument("--suffix", type=str, default="", help="Suffix for output filenames")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    cfg = PipelineConfig(
        schedule_file=Path(args.schedule),
        analysis_dir=Path(args.analysis_dir),
        output_dir=Path(args.output_dir),
        seed=args.seed,
        suffix=f"_{args.suffix}" if args.suffix else "",
    )
    generate_schedule(cfg)
