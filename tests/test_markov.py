"""Tests for roster_generator.markov module.

Covers
------
- _to_minute_bin_preserve_day(): positive rounding, negative flooring, zero, float input.
- _prepare_base_flights(): ZZZ remapping, uppercasing, stripping, datetime parsing,
  same-airport removal, airline filter, ValueError on empty result.
- _build_markov_tables(): return structure, expected columns, primary/fallback separation,
  probability normalisation, positive counts, hour stratification, memory property,
  fallback memorylessness, count conservation, no self-loops.
- generate_markov(): FileNotFoundError, output file creation, suffix in filenames.
"""

import numpy as np
import pandas as pd
import pytest

from roster_generator.distribution_analysis.markov import (
    _to_minute_bin_preserve_day,
    _prepare_base_flights,
    _build_markov_tables,
    generate_markov,
    AC_REG_COL,
    AIRLINE_COL,
    AC_WAKE_COL,
    DEP_COL,
    ARR_COL,
    STD_COL,
    STA_COL,
)
from roster_generator.config import PipelineConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_flight(reg, airline, wake, dep, arr, std, sta):
    """Build a single flight row dict with the expected clean-data columns."""
    return {
        AC_REG_COL: reg,
        AIRLINE_COL: airline,
        AC_WAKE_COL: wake,
        DEP_COL: dep,
        ARR_COL: arr,
        STD_COL: std,
        STA_COL: sta,
    }


def _make_base_df(flights):
    """Create a DataFrame from a list of flight dicts."""
    return pd.DataFrame(flights)


# A minimal two-aircraft, multi-leg schedule spanning one day.
# Aircraft 1 (AC001): LEMD->EGLL 08:00, EGLL->LFPG 12:00, LFPG->LEMD 16:00
# Aircraft 2 (AC002): EGLL->LFPG 09:00, LFPG->LEMD 14:00
SCHEDULE_FLIGHTS = [
    _make_flight("AC001", "IBE", "M", "LEMD", "EGLL", "2023-09-01 08:00", "2023-09-01 10:00"),
    _make_flight("AC001", "IBE", "M", "EGLL", "LFPG", "2023-09-01 12:00", "2023-09-01 13:30"),
    _make_flight("AC001", "IBE", "M", "LFPG", "LEMD", "2023-09-01 16:00", "2023-09-01 18:00"),
    _make_flight("AC002", "IBE", "M", "EGLL", "LFPG", "2023-09-01 09:00", "2023-09-01 10:30"),
    _make_flight("AC002", "IBE", "M", "LFPG", "LEMD", "2023-09-01 14:00", "2023-09-01 16:00"),
]


@pytest.fixture
def base_df():
    """Prepared base DataFrame through _prepare_base_flights."""
    df = _make_base_df(SCHEDULE_FLIGHTS)
    return _prepare_base_flights(df)


# ---------------------------------------------------------------------------
# _to_minute_bin_preserve_day
# ---------------------------------------------------------------------------

class TestToMinuteBinPreserveDay:

    def test_positive_rounds_down(self):
        """Positive value below .5 rounds to nearest int."""
        assert _to_minute_bin_preserve_day(120.4) == 120

    def test_positive_rounds_up(self):
        """Positive value above .5 rounds up."""
        assert _to_minute_bin_preserve_day(120.6) == 121

    def test_zero(self):
        """Zero returns 0."""
        assert _to_minute_bin_preserve_day(0) == 0

    def test_negative_floors(self):
        """Negative non-integer value floors (toward -inf)."""
        assert _to_minute_bin_preserve_day(-30.3) == -31

    def test_negative_exact_integer(self):
        """Negative exact integer value returns itself."""
        assert _to_minute_bin_preserve_day(-30.0) == -30

    def test_accepts_string_like_float(self):
        """Should handle values that float() can convert."""
        assert _to_minute_bin_preserve_day("45.2") == 45


# ---------------------------------------------------------------------------
# _prepare_base_flights
# ---------------------------------------------------------------------------

class TestPrepareBaseFlights:

    def test_zzz_airline_remapped_to_ac_reg(self):
        """Flights with AC_OPERATOR='ZZZ' get the AC_REG value instead."""
        flights = [
            _make_flight("EC-ABC", "ZZZ", "M", "LEMD", "EGLL",
                         "2023-09-01 08:00", "2023-09-01 10:00"),
        ]
        df = _prepare_base_flights(_make_base_df(flights))
        assert df.iloc[0][AIRLINE_COL] == "EC-ABC"

    def test_airline_uppercased(self):
        """Airline codes should be uppercased."""
        flights = [
            _make_flight("AC001", "ibe", "M", "LEMD", "EGLL",
                         "2023-09-01 08:00", "2023-09-01 10:00"),
        ]
        df = _prepare_base_flights(_make_base_df(flights))
        assert df.iloc[0][AIRLINE_COL] == "IBE"

    def test_string_columns_stripped(self):
        """Leading/trailing whitespace stripped from string columns."""
        flights = [
            _make_flight("  AC001  ", " IBE ", " M ", "  LEMD  ", "  EGLL  ",
                         "2023-09-01 08:00", "2023-09-01 10:00"),
        ]
        df = _prepare_base_flights(_make_base_df(flights))
        assert df.iloc[0][AC_REG_COL] == "AC001"
        assert df.iloc[0][DEP_COL] == "LEMD"
        assert df.iloc[0][ARR_COL] == "EGLL"

    def test_unparseable_dates_dropped(self):
        """Rows with invalid datetime strings are removed."""
        flights = [
            _make_flight("AC001", "IBE", "M", "LEMD", "EGLL",
                         "2023-09-01 08:00", "2023-09-01 10:00"),
            _make_flight("AC002", "IBE", "M", "EGLL", "LFPG",
                         "not-a-date", "2023-09-01 10:00"),
        ]
        df = _prepare_base_flights(_make_base_df(flights))
        assert len(df) == 1

    def test_same_airport_flights_removed(self):
        """Flights where DEP == ARR are discarded."""
        flights = [
            _make_flight("AC001", "IBE", "M", "LEMD", "LEMD",
                         "2023-09-01 08:00", "2023-09-01 09:00"),
            _make_flight("AC002", "IBE", "M", "LEMD", "EGLL",
                         "2023-09-01 08:00", "2023-09-01 10:00"),
        ]
        df = _prepare_base_flights(_make_base_df(flights))
        assert len(df) == 1
        assert df.iloc[0][ARR_COL] == "EGLL"

    def test_airline_filter(self):
        """Only matching airline kept when filter is applied."""
        flights = [
            _make_flight("AC001", "IBE", "M", "LEMD", "EGLL",
                         "2023-09-01 08:00", "2023-09-01 10:00"),
            _make_flight("AC002", "BAW", "M", "EGLL", "LFPG",
                         "2023-09-01 09:00", "2023-09-01 10:30"),
        ]
        df = _prepare_base_flights(_make_base_df(flights), airline_filter="IBE")
        assert len(df) == 1
        assert df.iloc[0][AIRLINE_COL] == "IBE"

    def test_raises_when_no_usable_flights(self):
        """ValueError raised when all flights are filtered out."""
        flights = [
            _make_flight("AC001", "IBE", "M", "LEMD", "LEMD",
                         "2023-09-01 08:00", "2023-09-01 09:00"),
        ]
        with pytest.raises(ValueError, match="No usable flights"):
            _prepare_base_flights(_make_base_df(flights))

    # --- Physical tests ---

    def test_no_self_loops_in_output(self, base_df):
        """Output should never contain DEP == ARR rows."""
        assert (base_df[DEP_COL] != base_df[ARR_COL]).all()

    def test_all_datetimes_valid(self, base_df):
        """All STD and STA values should be valid datetimes."""
        assert base_df["STD"].notna().all()
        assert base_df["STA"].notna().all()


# ---------------------------------------------------------------------------
# _build_markov_tables — software tests
# ---------------------------------------------------------------------------

class TestBuildMarkovTablesSoftware:

    def test_returns_three_objects(self, base_df):
        """Should return (DataFrame, primary_dict, fallback_dict)."""
        result = _build_markov_tables(base_df)
        assert len(result) == 3
        final_markov, markov_hourly, markov_fallback_hourly = result
        assert isinstance(final_markov, pd.DataFrame)
        assert isinstance(markov_hourly, dict)
        assert isinstance(markov_fallback_hourly, dict)

    def test_primary_table_has_expected_columns(self, base_df):
        """Primary DataFrame should contain the canonical Markov columns."""
        final_markov, _, _ = _build_markov_tables(base_df)
        expected = {AIRLINE_COL, AC_WAKE_COL, "PREV_ICAO", DEP_COL, ARR_COL,
                    "DEP_HOUR_UTC", "PROB", "COUNT"}
        assert expected.issubset(set(final_markov.columns))

    def test_first_flights_excluded_from_primary(self, base_df):
        """The first flight per aircraft has no predecessor — should not appear in primary."""
        final_markov, _, _ = _build_markov_tables(base_df)
        # AC001 first flight is LEMD->EGLL, AC002 first is EGLL->LFPG.
        # These should not appear with PREV_ICAO in the primary table
        # (unless another aircraft provides the same transition with a predecessor).
        # Specifically, no row should have PREV_ICAO as NaN.
        assert final_markov["PREV_ICAO"].notna().all()

    def test_fallback_includes_first_flights(self, base_df):
        """Fallback table should include first-of-chain flights (no prev_origin conditioning)."""
        _, _, fallback = _build_markov_tables(base_df)
        # AC001 first leg: LEMD->EGLL at hour 8
        key = ("IBE", "M", "LEMD")
        assert key in fallback
        assert 8 in fallback[key]
        assert "EGLL" in fallback[key][8]


# ---------------------------------------------------------------------------
# _build_markov_tables — physical / domain tests
# ---------------------------------------------------------------------------

class TestBuildMarkovTablesPhysical:

    def test_probabilities_sum_to_one(self, base_df):
        """For each (airline, wake, prev, dep, hour), probabilities must sum to 1."""
        final_markov, _, _ = _build_markov_tables(base_df)
        group_cols = [AIRLINE_COL, AC_WAKE_COL, "PREV_ICAO", DEP_COL, "DEP_HOUR_UTC"]
        prob_sums = final_markov.groupby(group_cols)["PROB"].sum()
        np.testing.assert_allclose(prob_sums.values, 1.0, atol=1e-9)

    def test_counts_are_positive(self, base_df):
        """All transition counts must be strictly positive."""
        final_markov, _, _ = _build_markov_tables(base_df)
        assert (final_markov["COUNT"] > 0).all()

    def test_hour_stratification(self):
        """Flights at different hours should produce separate hourly entries."""
        flights = [
            _make_flight("AC001", "IBE", "M", "LEMD", "EGLL",
                         "2023-09-01 06:00", "2023-09-01 08:00"),
            _make_flight("AC001", "IBE", "M", "EGLL", "LFPG",
                         "2023-09-01 10:00", "2023-09-01 11:30"),
            # Second day, same aircraft, different departure hours
            _make_flight("AC001", "IBE", "M", "LFPG", "EGLL",
                         "2023-09-01 14:00", "2023-09-01 15:30"),
        ]
        df = _prepare_base_flights(_make_base_df(flights))
        final_markov, hourly, _ = _build_markov_tables(df)
        # EGLL departures happen at hour 10, LFPG departures at hour 14
        assert len(final_markov["DEP_HOUR_UTC"].unique()) >= 2

    def test_memory_property(self):
        """Different prev_origins should yield different entries for the same (dep, hour)."""
        flights = [
            # Path 1: LEMD -> EGLL -> LFPG
            _make_flight("AC001", "IBE", "M", "LEMD", "EGLL",
                         "2023-09-01 06:00", "2023-09-01 08:00"),
            _make_flight("AC001", "IBE", "M", "EGLL", "LFPG",
                         "2023-09-01 10:00", "2023-09-01 11:30"),
            # Path 2: LIRF -> EGLL -> LEMD (same dep EGLL, same hour, different prev)
            _make_flight("AC002", "IBE", "M", "LIRF", "EGLL",
                         "2023-09-01 07:00", "2023-09-01 09:00"),
            _make_flight("AC002", "IBE", "M", "EGLL", "LEMD",
                         "2023-09-01 10:00", "2023-09-01 12:00"),
        ]
        df = _prepare_base_flights(_make_base_df(flights))
        _, hourly, _ = _build_markov_tables(df)
        # Primary table should have two different keys for EGLL departures at hour 10
        key_lemd = ("IBE", "M", "LEMD", "EGLL")
        key_lirf = ("IBE", "M", "LIRF", "EGLL")
        assert key_lemd in hourly
        assert key_lirf in hourly
        # Destinations differ: one goes to LFPG, the other to LEMD
        assert "LFPG" in hourly[key_lemd][10]
        assert "LEMD" in hourly[key_lirf][10]

    def test_fallback_keys_are_three_tuples(self, base_df):
        """Fallback dict keys should be (op, wake, dep) — no prev_origin."""
        _, _, fallback = _build_markov_tables(base_df)
        for key in fallback:
            assert len(key) == 3, f"Expected 3-tuple key, got {key}"

    def test_primary_count_conservation(self, base_df):
        """Total primary counts ≤ total flights minus one per aircraft."""
        final_markov, _, _ = _build_markov_tables(base_df)
        total_primary_counts = final_markov["COUNT"].sum()
        n_aircraft = base_df[AC_REG_COL].nunique()
        n_flights = len(base_df)
        # Each aircraft's first flight has no predecessor
        assert total_primary_counts <= n_flights - n_aircraft

    def test_no_self_loops(self, base_df):
        """No transition row should have DEP == ARR."""
        final_markov, _, _ = _build_markov_tables(base_df)
        assert (final_markov[DEP_COL] != final_markov[ARR_COL]).all()


# ---------------------------------------------------------------------------
# generate_markov — integration tests
# ---------------------------------------------------------------------------

class TestGenerateMarkov:

    def _write_schedule_csv(self, tmp_path, flights):
        """Write a minimal schedule CSV for integration tests."""
        df = _make_base_df(flights)
        path = tmp_path / "schedule.csv"
        df.to_csv(path, index=False)
        return path

    def _make_two_day_schedule(self):
        """Build a schedule spanning two days to exercise prior-flight logic."""
        return [
            # Day 1
            _make_flight("AC001", "IBE", "M", "LEMD", "EGLL",
                         "2023-09-01 08:00", "2023-09-01 10:00"),
            _make_flight("AC001", "IBE", "M", "EGLL", "LFPG",
                         "2023-09-01 12:00", "2023-09-01 13:30"),
            _make_flight("AC001", "IBE", "M", "LFPG", "LEMD",
                         "2023-09-01 16:00", "2023-09-01 18:00"),
            _make_flight("AC002", "IBE", "M", "EGLL", "LFPG",
                         "2023-09-01 09:00", "2023-09-01 10:30"),
            _make_flight("AC002", "IBE", "M", "LFPG", "LEMD",
                         "2023-09-01 14:00", "2023-09-01 16:00"),
            # Day 2 — same patterns
            _make_flight("AC001", "IBE", "M", "LEMD", "EGLL",
                         "2023-09-02 08:00", "2023-09-02 10:00"),
            _make_flight("AC001", "IBE", "M", "EGLL", "LFPG",
                         "2023-09-02 12:00", "2023-09-02 13:30"),
            _make_flight("AC001", "IBE", "M", "LFPG", "LEMD",
                         "2023-09-02 16:00", "2023-09-02 18:00"),
            _make_flight("AC002", "IBE", "M", "EGLL", "LFPG",
                         "2023-09-02 09:00", "2023-09-02 10:30"),
            _make_flight("AC002", "IBE", "M", "LFPG", "LEMD",
                         "2023-09-02 14:00", "2023-09-02 16:00"),
        ]

    def test_raises_file_not_found(self, tmp_path):
        """FileNotFoundError when schedule file doesn't exist."""
        cfg = PipelineConfig(
            schedule_file=tmp_path / "nonexistent.csv",
            analysis_dir=tmp_path / "analysis",
            output_dir=tmp_path / "output",
        )
        with pytest.raises(FileNotFoundError):
            generate_markov(cfg)

    def test_creates_output_files(self, tmp_path):
        """All three output files should be created."""
        schedule_path = self._write_schedule_csv(tmp_path, self._make_two_day_schedule())
        cfg = PipelineConfig(
            schedule_file=schedule_path,
            analysis_dir=tmp_path / "analysis",
            output_dir=tmp_path / "output",
        )
        generate_markov(cfg)
        assert (tmp_path / "analysis" / "markov.csv").exists()
        assert (tmp_path / "analysis" / "initial_conditions.csv").exists()
        assert (tmp_path / "output" / "phys_ta.csv").exists()

    def test_suffix_in_filenames(self, tmp_path):
        """When suffix is set, it should appear in all output filenames."""
        schedule_path = self._write_schedule_csv(tmp_path, self._make_two_day_schedule())
        cfg = PipelineConfig(
            schedule_file=schedule_path,
            analysis_dir=tmp_path / "analysis",
            output_dir=tmp_path / "output",
            suffix="_test",
        )
        generate_markov(cfg)
        assert (tmp_path / "analysis" / "markov_test.csv").exists()
        assert (tmp_path / "analysis" / "initial_conditions_test.csv").exists()
        assert (tmp_path / "output" / "phys_ta_test.csv").exists()
