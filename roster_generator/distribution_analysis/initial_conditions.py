"""
Synthetic Initial Conditions Model

Builds empirical distribution tables from historical flight data and uses them
to sample a synthetic day-0 fleet state: aircraft count per airline/wake,
parking origins, overnight priors, first departures, and single-flight flags.

The model is purely generative -- it does not modify the Markov transition
tables, but consumes them to resolve first-flight destinations and to infer
backward (arrival -> previous origin) distributions.

Outputs:
  - initial_conditions{suffix}.csv
  - phys_ta{suffix}.csv
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# --- Column aliases ---

AC_REG_COL = "AC_REG"
AIRLINE_COL = "AC_OPERATOR"
AC_WAKE_COL = "AC_WAKE"
DEP_COL = "DEP_ICAO"
ARR_COL = "ARR_ICAO"
STD_COL = "GATE_STD_UTC"
STA_COL = "RWY_STA_UTC"

# --- Discretisation & sampling constants ---

END_OF_DAY = 24 * 60          # minutes
BIN_SIZE = 5                  # minutes - temporal discretisation grain
P_NEXT_BIN_SIZE = 60          # minutes - bin width for P(next-day) profile
MAX_STD_RESAMPLE_ATTEMPTS = 4096


# --- Sampling helpers ---

def _normalize_airline_prefix(value):
    """Strip non-alphanumeric chars; used to build synthetic registration codes."""
    out = "".join(ch for ch in str(value).upper().strip() if ch.isalnum())
    if not out:
        raise ValueError("Empty airline prefix after normalization.")
    return out


def _weighted_choice_from_counts(counts_by_value, rng):
    """Sample one key from {value: count} using counts as weights."""
    values = list(counts_by_value.keys())
    weights = [int(counts_by_value[v]) for v in values]
    if not values or sum(weights) <= 0:
        raise ValueError("Invalid weighted choice map: empty or non-positive weights.")
    return rng.choices(values, weights=weights, k=1)[0]


def _weighted_choice_from_pairs(items, rng):
    """Sample from [(value, probability), ...] pairs."""
    if not items:
        raise ValueError("Cannot sample from empty weighted pair list.")
    vals = [v for v, _ in items]
    w = [float(p) for _, p in items]
    if sum(w) <= 0:
        raise ValueError("Cannot sample from non-positive weighted pair list.")
    return rng.choices(vals, weights=w, k=1)[0]


def _round5(v):
    """Round to nearest BIN_SIZE-minute boundary."""
    return int(round(float(v) / BIN_SIZE) * BIN_SIZE)


# --- Data structures ---

@dataclass
class SyntheticAircraft:
    """Mutable record for one synthetic aircraft being assembled during sampling.

    Fields are filled incrementally: fleet -> origin -> prior -> first flight.
    """
    reg: str
    airline: str
    wake: str
    origin: str                          # first-flight departure airport
    has_prior: bool                      # overnight arrival from previous day?
    prior_only: bool = False             # arrives but never departs on day-0
    prior_origin: Optional[str] = None   # where prior flight departed
    prior_dest: Optional[str] = None     # == origin when has_prior
    prior_std_mins: Optional[int] = None # negative (previous day)
    prior_sta_mins: Optional[int] = None # early morning of day-0
    first_std_mins: Optional[int] = None
    first_dest: Optional[str] = None
    first_sta_mins: Optional[int] = None
    single_flight: int = 0               # 1 = no further flights after first


# --- Core model ---

class InitialConditionModel:
    """Build empirical distribution tables and sample synthetic day-0 initial conditions.

    Workflow:
      1. Instantiate with normalised flight data.
      2. Call ``build_all()`` to fit every empirical table.
      3. Call ``set_markov_tables(...)`` to inject Markov chain outputs.
      4. Call ``sample_initial_conditions()`` to draw one synthetic fleet.
    """

    def __init__(self, flights, seed):
        self.flights = flights.copy()
        self.rng = random.Random(seed)

        # Empirical tables -- populated by build_all()
        self.daily_fleet_stats = {}    # (airline, wake) -> (mu, sigma)
        self.origin_counts = {}        # (airline, wake) -> {airport: count}
        self.first_std_samples = {}    # (airline, wake, origin) -> [min5, ...]
        self.p_prior = {}              # (airline, wake) -> float
        self.p_prior_only = {}         # (airline, wake) -> float
        self.backward_prev_counts = {} # (airline, wake, dep) -> {prev: count}
        self.prior_sta_samples = {}    # (airline, wake, prev, origin) -> [min5, ...]
        self.flight_time_median = {}   # (airline, wake, dep, arr) -> minutes
        self.p_next_hourly = {}        # (airline|"ALL", wake) -> {hour: p}
        self.phys_ta_min = {}          # (airline, wake) -> minutes (5th pctl)

        # Injected after Markov build
        self._markov_hourly = {}
        self._markov_fallback_hourly = {}

    # --- Table construction ---

    def build_all(self):
        """Fit all empirical tables from the flight data (order matters)."""
        self._build_first_departure_and_prev_tables()
        self._build_daily_fleet_gaussian_stats()
        self._build_origin_distribution()
        self._build_first_std_distribution()
        self._build_prior_probabilities()
        self._build_prior_only_probabilities()
        self._build_prior_sta_distribution()
        self._build_scheduled_flight_time_lookup()
        self._build_turnaround_temporal_probabilities()
        self._build_phys_ta_min()

    def set_markov_tables(self, markov_hourly, markov_fallback_hourly):
        """Inject Markov transition tables and derive backward prev-origin counts."""
        self._markov_hourly = markov_hourly
        self._markov_fallback_hourly = markov_fallback_hourly
        self._build_backward_prev_from_markov()

    def _build_first_departure_and_prev_tables(self):
        """Link each flight to its predecessor and extract the first departure per
        aircraft-day pair, including prior-flight metadata."""
        df = self.flights.sort_values([AC_REG_COL, "STD"]).copy()

        # Lag within each aircraft's sorted timeline
        grp = df.groupby(AC_REG_COL, sort=False)
        df["PREV_STD"] = grp["STD"].shift(1)
        df["PREV_STA"] = grp["STA"].shift(1)
        df["PREV_DEP"] = grp[DEP_COL].shift(1)
        df["PREV_ARR"] = grp[ARR_COL].shift(1)

        # Earliest departure per aircraft per calendar day
        df["DATE_STD"] = df["STD"].dt.normalize()
        first_dep = (
            df.sort_values([AC_REG_COL, "DATE_STD", "STD"])
            .groupby([AC_REG_COL, "DATE_STD"], as_index=False)
            .first()
        )

        # Discretise first-STD to BIN_SIZE-minute bins
        first_dep["FIRST_STD_MIN_5"] = (
            np.round(
                (
                    first_dep["STD"].dt.hour * 60
                    + first_dep["STD"].dt.minute
                    + first_dep["STD"].dt.second / 60.0
                )
                / BIN_SIZE
            )
            * BIN_SIZE
        ).astype(int)

        # A "prior" exists when the previous flight departed yesterday
        # and arrived today at the same airport where this flight departs
        prev_std_day = first_dep["PREV_STD"].dt.normalize()
        prev_sta_day = first_dep["PREV_STA"].dt.normalize()
        day0 = first_dep["DATE_STD"]

        first_dep["HAS_PRIOR"] = (
            first_dep["PREV_STD"].notna()
            & (prev_std_day == (day0 - pd.Timedelta(days=1)))
            & (prev_sta_day == day0)
            & (first_dep["PREV_ARR"] == first_dep[DEP_COL])
        )

        prior_sta_bin = (
            np.round(
                (
                    first_dep["PREV_STA"].dt.hour * 60
                    + first_dep["PREV_STA"].dt.minute
                    + first_dep["PREV_STA"].dt.second / 60.0
                )
                / BIN_SIZE
            )
            * BIN_SIZE
        )
        first_dep["PRIOR_STA_MIN_5"] = np.where(first_dep["PREV_STA"].notna(), prior_sta_bin, np.nan)

        self._df_with_prev = df
        self.first_dep = first_dep

    def _build_daily_fleet_gaussian_stats(self):
        """Fit N(mu, sigma) for daily aircraft count per (airline, wake)."""
        daily = (
            self.first_dep.groupby(["DATE_STD", AIRLINE_COL, AC_WAKE_COL])[AC_REG_COL]
            .nunique()
            .reset_index(name="N")
        )

        # Reindex to include zero-count days
        all_days = pd.date_range(
            daily["DATE_STD"].min(), daily["DATE_STD"].max(), freq="D"
        )

        for (airline, wake), g in daily.groupby([AIRLINE_COL, AC_WAKE_COL], sort=False):
            counts_by_day = g.set_index("DATE_STD")["N"]
            counts_by_day = counts_by_day.reindex(all_days, fill_value=0)
            vals = counts_by_day.astype(float).to_numpy()
            mu = float(np.mean(vals))
            sigma = float(np.std(vals, ddof=0))
            self.daily_fleet_stats[(str(airline), str(wake))] = (mu, sigma)

    def _build_origin_distribution(self):
        """Count first-departure airports per (airline, wake) for weighted sampling."""
        grp = (
            self.first_dep.groupby([AIRLINE_COL, AC_WAKE_COL, DEP_COL])
            .size()
            .reset_index(name="COUNT")
        )
        for row in grp.itertuples(index=False):
            key = (str(row.AC_OPERATOR), str(row.AC_WAKE))
            if key not in self.origin_counts:
                self.origin_counts[key] = {}
            self.origin_counts[key][str(row.DEP_ICAO)] = int(row.COUNT)

    def _build_first_std_distribution(self):
        """Collect empirical first-STD samples per (airline, wake, origin)."""
        grp = self.first_dep.groupby([AIRLINE_COL, AC_WAKE_COL, DEP_COL], sort=False)
        for (airline, wake, origin), g in grp:
            key = (str(airline), str(wake), str(origin))
            self.first_std_samples[key] = g["FIRST_STD_MIN_5"].astype(int).tolist()

    def _build_prior_probabilities(self):
        """Fraction of first-departures that have an overnight prior."""
        grp = self.first_dep.groupby([AIRLINE_COL, AC_WAKE_COL], sort=False)
        for (airline, wake), g in grp:
            p = float(g["HAS_PRIOR"].mean()) if len(g) else 0.0
            self.p_prior[(str(airline), str(wake))] = p

    def _build_prior_only_probabilities(self):
        """Fraction of priors where the aircraft arrives but never departs that day.

        "Prior-only" aircraft are those that land overnight but sit idle for the
        entire following day -- they contribute to airport occupancy without
        generating departures.
        """
        dep_presence = (
            self.flights.groupby([AC_REG_COL, self.flights["STD"].dt.normalize()])
            .size()
            .reset_index(name="N")
            .rename(columns={"STD": "DATE"})
        )
        dep_presence["HAS_DEP"] = True

        # Overnight arrivals: departed yesterday, arrived today
        ov = self.flights.copy()
        ov["ARR_DAY"] = ov["STA"].dt.normalize()
        ov["STD_DAY"] = ov["STD"].dt.normalize()
        ov = ov[ov["STD_DAY"] == (ov["ARR_DAY"] - pd.Timedelta(days=1))].copy()

        ov = ov.merge(
            dep_presence[[AC_REG_COL, "DATE", "HAS_DEP"]],
            left_on=[AC_REG_COL, "ARR_DAY"],
            right_on=[AC_REG_COL, "DATE"],
            how="left",
        )
        prior_only = ov[ov["HAS_DEP"] != True].copy()  # noqa: E712

        # Keep last arrival if multiple overnight flights to the same day
        if not prior_only.empty:
            prior_only = (
                prior_only.sort_values([AC_REG_COL, "ARR_DAY", "STA"])
                .groupby([AC_REG_COL, "ARR_DAY"], as_index=False)
                .last()
            )

        prior_with_dep = (
            self.first_dep[self.first_dep["HAS_PRIOR"]]
            .groupby([AIRLINE_COL, AC_WAKE_COL])
            .size()
            .reset_index(name="N_WITH_DEP")
        )
        prior_only_counts = (
            prior_only.groupby([AIRLINE_COL, AC_WAKE_COL])
            .size()
            .reset_index(name="N_ONLY")
            if not prior_only.empty
            else pd.DataFrame(columns=[AIRLINE_COL, AC_WAKE_COL, "N_ONLY"])
        )

        merged = prior_with_dep.merge(
            prior_only_counts,
            on=[AIRLINE_COL, AC_WAKE_COL],
            how="outer",
        ).fillna(0)

        self._prior_only_events = prior_only

        for row in merged.itertuples(index=False):
            a = str(row.AC_OPERATOR)
            w = str(row.AC_WAKE)
            n_with_dep = float(row.N_WITH_DEP)
            n_only = float(row.N_ONLY)
            den = n_with_dep + n_only
            if den <= 0:
                self.p_prior_only[(a, w)] = 0.0
            else:
                self.p_prior_only[(a, w)] = float(n_only / den)

    def _build_prior_sta_distribution(self):
        """Collect prior-arrival times for (airline, wake, prev_origin, origin) tuples."""
        with_prior = self.first_dep[self.first_dep["HAS_PRIOR"]].copy()
        grp = with_prior.groupby([AIRLINE_COL, AC_WAKE_COL, "PREV_DEP", DEP_COL], sort=False)

        for (airline, wake, prev_origin, origin), g in grp:
            key = (str(airline), str(wake), str(prev_origin), str(origin))
            self.prior_sta_samples[key] = g["PRIOR_STA_MIN_5"].astype(int).tolist()

    def _build_scheduled_flight_time_lookup(self):
        """Median scheduled block time per (airline, wake, dep, arr) route."""
        df = self.flights.copy()
        df["SCHED_MIN"] = (df["STA"] - df["STD"]).dt.total_seconds() / 60.0
        df = df[df["SCHED_MIN"] > 0].copy()

        med = (
            df.groupby([AIRLINE_COL, AC_WAKE_COL, DEP_COL, ARR_COL])["SCHED_MIN"]
            .median()
            .reset_index(name="MED")
        )

        for row in med.itertuples(index=False):
            key = (str(row.AC_OPERATOR), str(row.AC_WAKE), str(row.DEP_ICAO), str(row.ARR_ICAO))
            self.flight_time_median[key] = int(round(float(row.MED)))

    def _build_turnaround_temporal_probabilities(self):
        """P(next flight is next-day | arrival hour) per (airline, wake).

        Distinguishes intraday vs next-day connections to capture the time-of-day
        dependency: late-evening arrivals are far more likely to connect next-day.
        Also builds an aggregated ("ALL", wake) fallback.
        """
        df = self._df_with_prev.copy()

        linked = df[
            df["PREV_STA"].notna()
            & (df["PREV_ARR"] == df[DEP_COL])
        ].copy()
        if linked.empty:
            return

        day_gap = (linked["STD"].dt.normalize() - linked["PREV_STA"].dt.normalize()).dt.days
        linked = linked[day_gap.isin([0, 1])].copy()
        if linked.empty:
            return

        linked["CATEGORY"] = np.where(day_gap.loc[linked.index] == 0, "intraday", "next_day")
        linked["ARR_MIN"] = (
            linked["PREV_STA"].dt.hour * 60 + linked["PREV_STA"].dt.minute
        ).astype(int)
        linked["HOUR_BIN"] = (linked["ARR_MIN"] // P_NEXT_BIN_SIZE).astype(int)

        counts = (
            linked.groupby([AIRLINE_COL, AC_WAKE_COL, "HOUR_BIN", "CATEGORY"])
            .size()
            .reset_index(name="N")
        )

        # Per-airline profile
        for (airline, wake), g in counts.groupby([AIRLINE_COL, AC_WAKE_COL], sort=False):
            hmap = {}
            for hour in range(24):
                gh = g[g["HOUR_BIN"] == hour]
                n_intra = int(gh[gh["CATEGORY"] == "intraday"]["N"].sum())
                n_next = int(gh[gh["CATEGORY"] == "next_day"]["N"].sum())
                den = n_intra + n_next
                if den > 0:
                    hmap[hour] = float(n_next / den)
            self.p_next_hourly[(str(airline), str(wake))] = hmap

        # Aggregate fallback by wake category
        wake_counts = (
            linked.groupby([AC_WAKE_COL, "HOUR_BIN", "CATEGORY"])
            .size()
            .reset_index(name="N")
        )

        for wake, g in wake_counts.groupby(AC_WAKE_COL, sort=False):
            wake = str(wake)
            hmap = {}
            for hour in range(24):
                gh = g[g["HOUR_BIN"] == hour]
                n_intra = int(gh[gh["CATEGORY"] == "intraday"]["N"].sum())
                n_next = int(gh[gh["CATEGORY"] == "next_day"]["N"].sum())
                den = n_intra + n_next
                if den > 0:
                    hmap[hour] = float(n_next / den)
            self.p_next_hourly[("ALL", wake)] = hmap

    def _build_phys_ta_min(self):
        """5th-percentile intraday turnaround as physical minimum (floored to BIN_SIZE)."""
        df = self._df_with_prev.copy()

        linked = df[
            df["PREV_STA"].notna()
            & (df["PREV_ARR"] == df[DEP_COL])
        ].copy()
        if linked.empty:
            return

        # Only same-day connections count toward physical turnaround
        day_gap = (linked["STD"].dt.normalize() - linked["PREV_STA"].dt.normalize()).dt.days
        linked = linked[day_gap == 0].copy()
        if linked.empty:
            return

        linked["TA_MIN"] = (linked["STD"] - linked["PREV_STA"]).dt.total_seconds() / 60.0
        linked = linked[np.isfinite(linked["TA_MIN"]) & (linked["TA_MIN"] > 0)].copy()
        if linked.empty:
            return

        for (airline, wake), g in linked.groupby([AIRLINE_COL, AC_WAKE_COL], sort=False):
            p5 = float(np.percentile(g["TA_MIN"].to_numpy(dtype=float), 5))
            ta = int(np.floor(p5 / BIN_SIZE) * BIN_SIZE)
            key = (str(airline), str(wake))
            self.phys_ta_min[key] = ta

        # Global medium-wake fallback
        medium = linked[linked[AC_WAKE_COL].astype(str).str.upper() == "M"]
        if not medium.empty:
            p5 = float(np.percentile(medium["TA_MIN"].to_numpy(dtype=float), 5))
            ta = int(np.floor(p5 / BIN_SIZE) * BIN_SIZE)
            self.phys_ta_min[("ALL", "M")] = ta

        out_rows = [
            (airline, wake, int(ta))
            for (airline, wake), ta in self.phys_ta_min.items()
        ]
        self._phys_ta_df = pd.DataFrame(out_rows, columns=["airline_id", "aircraft_wake", "turnaround_time"])

    def _get_phys_ta_min(self, airline, wake):
        """Lookup minimum turnaround, falling back to global medium-wake value."""
        key = (str(airline), str(wake))
        if key in self.phys_ta_min:
            return int(self.phys_ta_min[key])

        fallback_key = ("ALL", "M")
        if fallback_key in self.phys_ta_min:
            return int(self.phys_ta_min[fallback_key])

        raise ValueError(f"Missing phys_ta for {key} and fallback {fallback_key}")

    def _build_backward_prev_from_markov(self):
        """Invert Markov transitions: for each destination, count which origins fed it.

        Used to sample plausible prior-flight origins when constructing overnight
        arrivals -- we need to know "where could this aircraft have come from?"
        """
        back = {}
        for (op, wake, prev, dep), hourly in self._markov_hourly.items():
            key = (str(op), str(wake), str(dep))
            if key not in back:
                back[key] = {}
            total = 0
            for hm in hourly.values():
                total += int(sum(hm.values()))
            if total > 0:
                back[key][str(prev)] = back[key].get(str(prev), 0) + total
        self.backward_prev_counts = back

    # --- Fleet & prior sampling ---

    def _sample_fleet(self):
        """Draw fleet size from Gaussian, assign origins and prior flags."""
        aircraft = []
        reg_counters = {}

        keys = sorted(self.daily_fleet_stats.keys())
        for airline, wake in keys:
            mu, sigma = self.daily_fleet_stats[(airline, wake)]
            sampled = float(np.random.normal(mu, sigma)) if sigma > 0 else mu
            n = max(0, int(round(sampled)))

            if n == 0:
                continue

            if (airline, wake) not in self.origin_counts:
                raise ValueError(f"Missing origin distribution for {(airline, wake)}")

            for _ in range(n):
                prefix = _normalize_airline_prefix(airline)
                reg_counters[prefix] = reg_counters.get(prefix, 0) + 1
                reg = f"{prefix}{reg_counters[prefix]:03d}"
                origin = _weighted_choice_from_counts(self.origin_counts[(airline, wake)], self.rng)
                p_prior = self.p_prior.get((airline, wake), 0.0)
                has_prior = self.rng.random() < p_prior
                aircraft.append(
                    SyntheticAircraft(
                        reg=reg,
                        airline=airline,
                        wake=wake,
                        origin=origin,
                        has_prior=has_prior,
                    )
                )

        return aircraft

    @staticmethod
    def _find_hourly_data_with_radius(hourly_data, center_hour, rng, max_radius=2):
        """Find hourly data at center_hour or nearby hours (randomised search direction).

        Falls back to the nearest future hour if nothing is found within radius.
        """
        if not hourly_data:
            return {}, -1

        candidates = [center_hour]
        for radius in range(1, max_radius + 1):
            # Randomise whether we look earlier or later first
            if rng.random() < 0.5:
                candidates.append((center_hour - radius) % 24)
                candidates.append((center_hour + radius) % 24)
            else:
                candidates.append((center_hour + radius) % 24)
                candidates.append((center_hour - radius) % 24)

        for h in candidates:
            if h in hourly_data:
                return hourly_data[h], h

        # Last resort: nearest future hour with data
        available_hours = sorted(hourly_data.keys())
        future_hours = [h for h in available_hours if h >= center_hour]
        if future_hours:
            h = future_hours[0]
            return hourly_data[h], h

        return {}, -1

    def _get_markov_destinations(self, op, wake, prev_origin, origin, dep_utc_mins):
        """Query Markov tables for destination probabilities at a given departure hour.

        Tries the primary (prev_origin-aware) table first, then the fallback
        (origin-only) table. Returns [(dest, prob), ...] sorted by probability.
        """
        dep_hour = (int(dep_utc_mins) // 60) % 24

        fallback_key = (op, wake, origin)

        counts = {}
        if prev_origin:
            primary_key = (op, wake, prev_origin, origin)
            hourly_data = self._markov_hourly.get(primary_key, {})
            counts, _ = self._find_hourly_data_with_radius(hourly_data, dep_hour, self.rng)

        if not counts:
            fb_hourly_data = self._markov_fallback_hourly.get(fallback_key, {})
            counts, _ = self._find_hourly_data_with_radius(fb_hourly_data, dep_hour, self.rng)

        if not counts:
            return []

        total = int(sum(counts.values()))
        if total <= 0:
            return []

        out = [(d, float(c / total)) for d, c in counts.items()]
        out.sort(key=lambda x: -x[1])
        return out

    def _sample_prior_for_aircraft(self, ac):
        """Assign overnight prior-flight details, trying the current origin first.

        If the current origin has no feasible prior (no route with STD < 0),
        tries alternative origins weighted by empirical frequency. Clears the
        prior flag entirely if nothing works.
        """
        if not ac.has_prior:
            return

        if self._try_sample_prior_for_origin(ac, ac.origin):
            return

        # Fallback: try other origins that have feasible prior routes
        key = (ac.airline, ac.wake)
        origins = self.origin_counts.get(key, {})
        if not origins:
            raise ValueError(f"Missing origin distribution for {key}")

        candidate_origins = [
            o for o in origins.keys()
            if o != ac.origin and self._origin_has_feasible_prior(ac.airline, ac.wake, o)
        ]
        weights = [int(origins[o]) for o in candidate_origins]

        while candidate_origins:
            sampled_origin = self.rng.choices(candidate_origins, weights=weights, k=1)[0]
            idx = candidate_origins.index(sampled_origin)
            candidate_origins.pop(idx)
            weights.pop(idx)

            ac.origin = sampled_origin
            if self._try_sample_prior_for_origin(ac, ac.origin):
                return

        # No feasible prior found -- revert to no-prior state
        ac.has_prior = False
        ac.prior_origin = None
        ac.prior_dest = None
        ac.prior_sta_mins = None
        ac.prior_std_mins = None

    def _origin_has_feasible_prior(self, airline, wake, origin):
        """Quick check: does any prior route into this origin yield STD < 0?"""
        prev_counts = self.backward_prev_counts.get((airline, wake, origin))
        if not prev_counts:
            return False

        for prev_origin in prev_counts.keys():
            key_sta = (airline, wake, prev_origin, origin)
            sta_samples = self.prior_sta_samples.get(key_sta)
            if not sta_samples:
                continue
            flight_time = self.flight_time_median.get(key_sta)
            if flight_time is None or int(flight_time) <= 0:
                continue
            # STD < 0 means departure was on the previous day
            if any(int(sta) - int(flight_time) < 0 for sta in sta_samples):
                return True

        return False

    def _try_sample_prior_for_origin(self, ac, origin):
        """Attempt to sample a valid prior flight arriving at `origin`.

        Iterates over candidate previous-origins weighted by backward Markov
        counts. For each, shuffles STA samples and checks that
        STD = STA - flight_time < 0 (i.e. departure was previous day).
        """
        key_prev = (ac.airline, ac.wake, origin)
        prev_counts = self.backward_prev_counts.get(key_prev)
        if not prev_counts:
            return False

        prev_candidates = list(prev_counts.keys())
        prev_weights = [int(prev_counts[p]) for p in prev_candidates]
        attempted_prev = set()

        while len(attempted_prev) < len(prev_candidates):
            remaining_prev = [p for p in prev_candidates if p not in attempted_prev]
            remaining_weights = [prev_weights[prev_candidates.index(p)] for p in remaining_prev]
            prev_origin = self.rng.choices(remaining_prev, weights=remaining_weights, k=1)[0]
            attempted_prev.add(prev_origin)

            key_sta = (ac.airline, ac.wake, prev_origin, origin)
            sta_samples = self.prior_sta_samples.get(key_sta)
            if not sta_samples:
                continue

            shuffled = sta_samples.copy()
            self.rng.shuffle(shuffled)

            flight_time = self.flight_time_median.get(key_sta)
            if flight_time is None or int(flight_time) <= 0:
                continue

            for sta in shuffled:
                std = int(sta) - int(flight_time)
                if std < 0:  # departed previous day
                    ac.prior_origin = prev_origin
                    ac.prior_dest = origin
                    ac.prior_sta_mins = int(sta)
                    ac.prior_std_mins = int(std)
                    return True

        return False

    def _sample_prior_only(self, ac):
        """Decide whether a prior-equipped aircraft sits idle all day."""
        if not ac.has_prior:
            ac.prior_only = False
            return
        p = self.p_prior_only.get((ac.airline, ac.wake), 0.0)
        ac.prior_only = self.rng.random() < p

    # --- First-flight sampling ---

    def _sample_first_std(self, ac):
        """Sample first-departure time, respecting minimum turnaround after prior arrival."""
        key = (ac.airline, ac.wake, ac.origin)
        samples = self.first_std_samples.get(key)
        if not samples:
            raise ValueError(f"Missing first-STD distribution for {key}")

        min_ta = 0
        if ac.has_prior:
            min_ta = self._get_phys_ta_min(ac.airline, ac.wake)
            if ac.prior_sta_mins is None:
                raise ValueError("Aircraft has prior flag but prior STA is not assigned.")

        shuffled = samples.copy()
        self.rng.shuffle(shuffled)

        if not ac.has_prior:
            return int(shuffled[0])

        # Must satisfy turnaround constraint: first_STD - prior_STA >= min_ta
        for std in shuffled:
            if int(std) - int(ac.prior_sta_mins) >= min_ta:
                return int(std)

        # No feasible STD - drop the prior
        ac.has_prior = False
        ac.prior_origin = None
        ac.prior_dest = None
        ac.prior_sta_mins = None
        ac.prior_std_mins = None
        ac.prior_only = False
        return int(shuffled[0])

    def _sample_first_destination(self, ac, first_std_mins):
        """Draw first-flight destination from Markov transition probabilities."""
        prev_origin = ac.prior_origin if ac.has_prior else None

        dest_probs = self._get_markov_destinations(
            op=ac.airline,
            wake=ac.wake,
            prev_origin=prev_origin,
            origin=ac.origin,
            dep_utc_mins=first_std_mins,
        )
        if not dest_probs:
            raise ValueError(
                f"No first destination available from Markov for airline={ac.airline}, wake={ac.wake}, "
                f"prev_origin={prev_origin}, origin={ac.origin}, dep_hour={(first_std_mins // 60) % 24}"
            )

        return _weighted_choice_from_pairs(dest_probs, self.rng)

    def _compute_first_sta(self, ac, first_std_mins, dest):
        """STA = STD + median scheduled block time for this route."""
        ft_key = (ac.airline, ac.wake, ac.origin, dest)
        flight_time = self.flight_time_median.get(ft_key)
        if flight_time is None or flight_time <= 0:
            raise ValueError(f"Missing scheduled flight time for first flight key {ft_key}")
        return int(first_std_mins + int(flight_time))

    def _sample_single_flight_flag(self, ac, first_sta_mins):
        """Bernoulli draw: does this aircraft stop after one flight (next-day pattern)?"""
        hour = (int(first_sta_mins) % END_OF_DAY) // P_NEXT_BIN_SIZE
        wake = str(ac.wake)
        key = (ac.airline, wake)
        fallback_key = ("ALL", wake)

        hmap = self.p_next_hourly.get(key)
        if hmap is None or hour not in hmap:
            hmap = self.p_next_hourly.get(fallback_key)

        if hmap is None or hour not in hmap:
            raise ValueError(
                f"Missing p_next profile for {(ac.airline, ac.wake)} with fallback {fallback_key} "
                f"at hour bin={hour}"
            )

        p_next = float(hmap[hour])
        return 1 if self.rng.random() < p_next else 0

    # --- Main entry point ---

    def sample_initial_conditions(self):
        """Generate a complete synthetic fleet for day-0.

        Returns a DataFrame with one row per aircraft containing origin, destination,
        departure/arrival times, prior-flight info, and single-flight flags.
        """
        aircraft = self._sample_fleet()

        for ac in aircraft:
            if ac.has_prior:
                self._sample_prior_for_aircraft(ac)
            self._sample_prior_only(ac)

            if ac.prior_only:
                continue

            ac.first_std_mins = self._sample_first_std(ac)
            ac.first_dest = self._sample_first_destination(ac, ac.first_std_mins)
            ac.first_sta_mins = self._compute_first_sta(ac, ac.first_std_mins, ac.first_dest)
            ac.single_flight = self._sample_single_flight_flag(ac, ac.first_sta_mins)

        rows = []
        for ac in aircraft:
            rows.append({
                "AC_REG": ac.reg,
                "AC_OPERATOR": ac.airline,
                "AC_WAKE": ac.wake,
                "PRIOR_ONLY": 1 if ac.prior_only else 0,
                "ORIGIN": "" if ac.prior_only else ac.origin,
                "DEST": "" if ac.prior_only else ac.first_dest,
                "STD_UTC_MINS": None if ac.prior_only else int(ac.first_std_mins),
                "STA_UTC_MINS": None if ac.prior_only else int(ac.first_sta_mins),
                "SINGLE_FLIGHT": 0 if ac.prior_only else int(ac.single_flight),
                "PRIOR_ORIGIN": ac.prior_origin,
                "PRIOR_DEST": ac.prior_dest,
                "PRIOR_STD_UTC_MINS": ac.prior_std_mins,
                "PRIOR_STA_UTC_MINS": ac.prior_sta_mins,
            })

        ic_df = pd.DataFrame(rows)
        self._validate_initial_conditions(ic_df)
        return ic_df

    # --- Validation ---

    def _validate_initial_conditions(self, ic_df):
        """Sanity checks on the generated initial conditions DataFrame."""
        if ic_df["AC_REG"].duplicated().any():
            dup = ic_df[ic_df["AC_REG"].duplicated()]["AC_REG"].head(5).tolist()
            raise ValueError(f"Duplicate AC_REG found: {dup}")

        # Prior STD must be negative (previous-day departure)
        has_prior = ic_df["PRIOR_STD_UTC_MINS"].notna()
        if (ic_df.loc[has_prior, "PRIOR_STD_UTC_MINS"].astype(float) >= 0).any():
            raise ValueError("Validation failed: prior rows with PRIOR_STD_UTC_MINS >= 0")

        # Non-prior-only rows must have complete first-flight data
        non_prior_only = ic_df["PRIOR_ONLY"].astype(int) != 1
        required_cols = ["ORIGIN", "DEST", "STD_UTC_MINS", "STA_UTC_MINS"]
        for c in required_cols:
            if ic_df.loc[non_prior_only, c].isna().any():
                raise ValueError(f"Validation failed: missing {c} in non-prior-only rows")
        if (ic_df.loc[non_prior_only, "ORIGIN"].astype(str).str.strip() == "").any():
            raise ValueError("Validation failed: blank ORIGIN in non-prior-only rows")
        if (ic_df.loc[non_prior_only, "DEST"].astype(str).str.strip() == "").any():
            raise ValueError("Validation failed: blank DEST in non-prior-only rows")

        # Turnaround constraint: first_STD - prior_STA >= phys_ta_min
        with_prior_and_first = (
            (ic_df["PRIOR_STD_UTC_MINS"].notna())
            & (ic_df["PRIOR_ONLY"].astype(int) != 1)
        )

        for row in ic_df[with_prior_and_first].itertuples(index=False):
            key = (str(row.AC_OPERATOR), str(row.AC_WAKE))
            min_ta = self._get_phys_ta_min(str(row.AC_OPERATOR), str(row.AC_WAKE))
            if int(row.STD_UTC_MINS) - int(row.PRIOR_STA_UTC_MINS) < min_ta:
                raise ValueError(
                    "Validation failed: first_STD - prior_STA below phys_ta for "
                    f"AC_REG={row.AC_REG}, key={key}, min_ta={min_ta}"
                )
