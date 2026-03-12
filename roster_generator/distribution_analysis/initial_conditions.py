"""Public facade for synthetic day-0 initial-condition generation.

The heavy lifting lives in private helper modules, but this class keeps the
historical API stable for ``markov.generate_markov`` and existing users.
"""

from __future__ import annotations

import random

import pandas as pd

from ._initial_conditions_builders import (
    build_all_tables,
    get_phys_ta_min,
    set_markov_tables as set_markov_state,
)
from ._initial_conditions_sampling import sample_initial_conditions
from ._initial_conditions_types import BIN_SIZE, InitialConditionState
from ._initial_conditions_validation import validate_initial_conditions

# Backward-compatible module-level names.
from . import _initial_conditions_sampling as _sampling
from . import _initial_conditions_types as _types

AC_REG_COL = _types.AC_REG_COL
AIRLINE_COL = _types.AIRLINE_COL
AC_WAKE_COL = _types.AC_WAKE_COL
DEP_COL = _types.DEP_COL
ARR_COL = _types.ARR_COL
STD_COL = _types.STD_COL
STA_COL = _types.STA_COL
END_OF_DAY = _types.END_OF_DAY
P_NEXT_BIN_SIZE = _types.P_NEXT_BIN_SIZE
MAX_STD_RESAMPLE_ATTEMPTS = _types.MAX_STD_RESAMPLE_ATTEMPTS
SyntheticAircraft = _types.SyntheticAircraft
_normalize_airline_prefix = _sampling._normalize_airline_prefix
_weighted_choice_from_counts = _sampling._weighted_choice_from_counts
_weighted_choice_from_pairs = _sampling._weighted_choice_from_pairs


def _round5(v):
    """Round to nearest BIN_SIZE-minute boundary."""
    return int(round(float(v) / BIN_SIZE) * BIN_SIZE)


class InitialConditionModel:
    """Build empirical tables and sample synthetic day-0 initial conditions.

    Workflow:
      1. Instantiate with normalized flight data.
      2. Call ``build_all()`` to fit empirical tables.
      3. Call ``set_markov_tables(...)`` to inject Markov transitions.
      4. Call ``sample_initial_conditions()`` to draw one synthetic fleet.
    """

    def __init__(self, flights: pd.DataFrame, seed: int, window_length_mins: int = END_OF_DAY):
        self.flights = flights.copy()
        self.rng = random.Random(seed)
        self._state = InitialConditionState()
        self._state.window_length_mins = int(window_length_mins)
        self._state.hour_bins = max(1, int(window_length_mins) // P_NEXT_BIN_SIZE)
        self._sync_legacy_attrs()

    def _sync_legacy_attrs(self) -> None:
        """Expose historical attributes for backward compatibility."""
        self.daily_fleet_stats = self._state.daily_fleet_stats
        self.origin_counts = self._state.origin_counts
        self.first_std_samples = self._state.first_std_samples
        self.p_prior = self._state.p_prior
        self.p_prior_only = self._state.p_prior_only
        self.backward_prev_counts = self._state.backward_prev_counts
        self.prior_sta_samples = self._state.prior_sta_samples
        self.flight_time_median = self._state.flight_time_median
        self.p_next_hourly = self._state.p_next_hourly
        self.phys_ta_min = self._state.phys_ta_min

        self._markov_hourly = self._state.markov_hourly
        self._markov_fallback_hourly = self._state.markov_fallback_hourly

        self._df_with_prev = self._state.df_with_prev
        self.first_dep = self._state.first_dep
        self._prior_only_events = self._state.prior_only_events
        self._phys_ta_df = self._state.phys_ta_df

    @property
    def phys_ta_df(self) -> pd.DataFrame:
        """Public alias for physical turnaround output DataFrame."""
        return self._state.phys_ta_df

    def build_all(self) -> None:
        """Fit all empirical tables from flight data."""
        build_all_tables(self.flights, self._state)
        self._sync_legacy_attrs()

    def set_markov_tables(self, markov_hourly, markov_fallback_hourly) -> None:
        """Inject Markov tables and derive backward previous-origin counts."""
        set_markov_state(self._state, markov_hourly, markov_fallback_hourly)
        self._sync_legacy_attrs()

    def _get_phys_ta_min(self, airline: str, wake: str) -> int:
        """Compatibility wrapper around physical-turnaround lookup."""
        return get_phys_ta_min(self._state, airline, wake)

    def sample_initial_conditions(self) -> pd.DataFrame:
        """Generate one synthetic day-0 fleet and validate invariants."""
        ic_df = sample_initial_conditions(self._state, self.rng)
        self._validate_initial_conditions(ic_df)
        return ic_df

    def _validate_initial_conditions(self, ic_df: pd.DataFrame) -> None:
        """Sanity checks on the generated initial-conditions DataFrame."""
        validate_initial_conditions(ic_df, self._get_phys_ta_min)
