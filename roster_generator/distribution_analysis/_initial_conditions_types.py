"""Core constants and internal state for initial-condition generation.

This module intentionally stays small and boring: shared names, typed keys,
and data containers only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# Schedule input column names used across builders and samplers.
AC_REG_COL = "AC_REG"
AIRLINE_COL = "AC_OPER"
AC_WAKE_COL = "AC_WAKE"
DEP_COL = "DEP_ICAO"
ARR_COL = "ARR_ICAO"
STD_COL = "STD_REFTZ"
STA_COL = "STA_REFTZ"

# Time discretization knobs.
END_OF_DAY = 24 * 60  # Legacy default full-day window.
BIN_SIZE = 5
P_NEXT_BIN_SIZE = 60

# Retained for compatibility with legacy module-level exports.
MAX_STD_RESAMPLE_ATTEMPTS = 4096


# Key conventions:
# - airline+wake: (airline, wake)
# - airline+wake+origin: (airline, wake, origin)
# - airline+wake+route: (airline, wake, dep, arr)
type AirlineWakeKey = tuple[str, str]
type AirlineWakeOriginKey = tuple[str, str, str]
type AirlineWakeRouteKey = tuple[str, str, str, str]
type AirlineWakeDestinationKey = tuple[str, str, str]

type HourlyDestinationCounts = dict[int, dict[str, int]]
type HourlyProbabilities = dict[int, float]

# Markov keys:
# - primary: (airline, wake, previous_origin, origin)
# - fallback: (airline, wake, origin)
type PrimaryMarkovTable = dict[tuple[str, str, str, str], HourlyDestinationCounts]
type FallbackMarkovTable = dict[tuple[str, str, str], HourlyDestinationCounts]


def _empty_phys_ta_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["airline_id", "aircraft_wake", "turnaround_time"])


@dataclass(slots=True)
class SyntheticAircraft:
    """Mutable record for one synthetic aircraft while sampling."""

    reg: str
    airline: str
    wake: str
    origin: str
    has_prior: bool
    prior_only: bool = False
    prior_origin: str | None = None
    prior_dest: str | None = None
    prior_std_mins: int | None = None
    prior_sta_mins: int | None = None
    first_std_mins: int | None = None
    first_dest: str | None = None
    first_sta_mins: int | None = None
    single_flight: int = 0


@dataclass(slots=True)
class InitialConditionState:
    """Fitted empirical distributions and injected Markov transition tables."""
    window_length_mins: int = END_OF_DAY
    hour_bins: int = 24

    # Learned distributions and lookup tables.
    daily_fleet_stats: dict[AirlineWakeKey, tuple[float, float]] = field(default_factory=dict)
    origin_counts: dict[AirlineWakeKey, dict[str, int]] = field(default_factory=dict)
    first_std_samples: dict[AirlineWakeOriginKey, list[int]] = field(default_factory=dict)
    p_prior: dict[AirlineWakeKey, float] = field(default_factory=dict)
    p_prior_only: dict[AirlineWakeKey, float] = field(default_factory=dict)
    backward_prev_counts: dict[AirlineWakeDestinationKey, dict[str, int]] = field(default_factory=dict)
    prior_sta_samples: dict[AirlineWakeRouteKey, list[int]] = field(default_factory=dict)
    flight_time_median: dict[AirlineWakeRouteKey, int] = field(default_factory=dict)
    p_next_hourly: dict[AirlineWakeKey, HourlyProbabilities] = field(default_factory=dict)
    phys_ta_min: dict[AirlineWakeKey, int] = field(default_factory=dict)

    # Injected Markov transition tables.
    markov_hourly: PrimaryMarkovTable = field(default_factory=dict)
    markov_fallback_hourly: FallbackMarkovTable = field(default_factory=dict)

    # Intermediate materialized frames used during fitting.
    df_with_prev: pd.DataFrame = field(default_factory=pd.DataFrame)
    first_dep: pd.DataFrame = field(default_factory=pd.DataFrame)
    prior_only_events: pd.DataFrame = field(default_factory=pd.DataFrame)
    phys_ta_df: pd.DataFrame = field(default_factory=_empty_phys_ta_df)
