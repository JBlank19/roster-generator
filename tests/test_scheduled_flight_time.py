"""Tests for roster_generator.scheduled_flight_time module.

Covers
------
- _prepare_flights(): ZZZ remapping, missing wake default, datetime parsing,
  negative/zero duration filtering, departure hour extraction, bin alignment.
- _build_hourly_distributions(): output columns, MIN_SAMPLES filtering,
  probability normalisation, group key structure.
- _build_operator_agnostic_distributions(): airline_id='ALL' sentinel,
  probability normalisation, group key structure.
- analyze_flight_time_distribution(): FileNotFoundError, ValueError on empty,
  output file creation, suffix support, CSV column check.

Physical tests
--------------
- Flight durations are strictly positive.
- Duration bins are non-negative multiples of BIN_SIZE.
- Departure hours are in [0, 23].
- Probabilities are in (0, 1] and sum to 1 per group.
- Operator-agnostic rows always have airline_id='ALL'.
- No impossible routes (origin == dest).
"""

import numpy as np
import pandas as pd
import pytest

from roster_generator.distribution_analysis.scheduled_flight_time import (
    _prepare_flights,
    _build_hourly_distributions,
    _build_operator_agnostic_distributions,
    analyze_flight_time_distribution,
    AC_REG_COL,
    AIRLINE_COL,
    DEP_STATION_COL,
    ARR_STATION_COL,
    STD_COL,
    STA_COL,
    BIN_SIZE,
    MIN_SAMPLES,
)
from roster_generator.config import PipelineConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_flight(reg, airline, wake, dep, arr, std, sta):
    """Build a single flight row dict."""
    return {
        AC_REG_COL: reg,
        AIRLINE_COL: airline,
        "AC_WAKE": wake,
        DEP_STATION_COL: dep,
        ARR_STATION_COL: arr,
        STD_COL: std,
        STA_COL: sta,
    }


def _make_df(flights):
    """Create a DataFrame from a list of flight dicts."""
    return pd.DataFrame(flights)


# A schedule with enough samples (>=3) per group to pass MIN_SAMPLES.
# Route LEMD->EGLL at hour 8 with airline IBE, wake M — 4 flights across 4 days.
# Route EGLL->LFPG at hour 12 with airline IBE, wake M — 4 flights.
SCHEDULE_FLIGHTS = [
    # Day 1
    _make_flight("AC001", "IBE", "M", "LEMD", "EGLL", "2023-09-01 08:00", "2023-09-01 10:00"),
    _make_flight("AC001", "IBE", "M", "EGLL", "LFPG", "2023-09-01 12:00", "2023-09-01 13:30"),
    # Day 2
    _make_flight("AC001", "IBE", "M", "LEMD", "EGLL", "2023-09-02 08:00", "2023-09-02 10:10"),
    _make_flight("AC001", "IBE", "M", "EGLL", "LFPG", "2023-09-02 12:00", "2023-09-02 13:25"),
    # Day 3
    _make_flight("AC001", "IBE", "M", "LEMD", "EGLL", "2023-09-03 08:00", "2023-09-03 09:55"),
    _make_flight("AC001", "IBE", "M", "EGLL", "LFPG", "2023-09-03 12:00", "2023-09-03 13:35"),
    # Day 4
    _make_flight("AC001", "IBE", "M", "LEMD", "EGLL", "2023-09-04 08:00", "2023-09-04 10:05"),
    _make_flight("AC001", "IBE", "M", "EGLL", "LFPG", "2023-09-04 12:00", "2023-09-04 13:30"),
]


@pytest.fixture
def prepared_df():
    """Prepared DataFrame through _prepare_flights."""
    return _prepare_flights(_make_df(SCHEDULE_FLIGHTS))


@pytest.fixture
def hourly_dist(prepared_df):
    """Per-operator hourly distribution DataFrame."""
    return _build_hourly_distributions(prepared_df)


@pytest.fixture
def agnostic_dist(prepared_df):
    """Operator-agnostic hourly distribution DataFrame."""
    return _build_operator_agnostic_distributions(prepared_df)


# ---------------------------------------------------------------------------
# _prepare_flights — software tests
# ---------------------------------------------------------------------------

class TestPrepareFlightsSoftware:

    def test_returns_expected_columns(self, prepared_df):
        """Output should contain the derived columns."""
        for col in ["FLIGHT_MINUTES", "DEP_HOUR_UTC", "FLIGHT_BIN"]:
            assert col in prepared_df.columns

    def test_zzz_airline_remapped(self):
        """Flights with AC_OPER='ZZZ' get AC_REG as airline."""
        flights = [
            _make_flight("EC-ABC", "ZZZ", "M", "LEMD", "EGLL",
                         "2023-09-01 08:00", "2023-09-01 10:00"),
        ]
        df = _prepare_flights(_make_df(flights))
        assert (df[AIRLINE_COL] == "EC-ABC").all()

    def test_missing_wake_defaults_to_m(self):
        """When AC_WAKE column is absent, all wakes default to 'M'."""
        flights = [
            {AC_REG_COL: "AC001", AIRLINE_COL: "IBE",
             DEP_STATION_COL: "LEMD", ARR_STATION_COL: "EGLL",
             STD_COL: "2023-09-01 08:00", STA_COL: "2023-09-01 10:00"},
        ]
        df = _prepare_flights(pd.DataFrame(flights))
        assert (df["AC_WAKE"] == "M").all()

    def test_unparseable_dates_dropped(self):
        """Rows with invalid datetime strings are removed."""
        flights = [
            _make_flight("AC001", "IBE", "M", "LEMD", "EGLL",
                         "2023-09-01 08:00", "2023-09-01 10:00"),
            _make_flight("AC002", "IBE", "M", "EGLL", "LFPG",
                         "not-a-date", "2023-09-01 13:30"),
        ]
        df = _prepare_flights(_make_df(flights))
        assert len(df) == 1

    def test_negative_duration_dropped(self):
        """Flights where STA < STD (negative duration) are removed."""
        flights = [
            _make_flight("AC001", "IBE", "M", "LEMD", "EGLL",
                         "2023-09-01 10:00", "2023-09-01 08:00"),  # backwards
            _make_flight("AC002", "IBE", "M", "EGLL", "LFPG",
                         "2023-09-01 12:00", "2023-09-01 13:30"),
        ]
        df = _prepare_flights(_make_df(flights))
        assert len(df) == 1

    def test_zero_duration_dropped(self):
        """Flights where STD == STA (zero duration) are removed."""
        flights = [
            _make_flight("AC001", "IBE", "M", "LEMD", "EGLL",
                         "2023-09-01 08:00", "2023-09-01 08:00"),  # zero
            _make_flight("AC002", "IBE", "M", "EGLL", "LFPG",
                         "2023-09-01 12:00", "2023-09-01 13:30"),
        ]
        df = _prepare_flights(_make_df(flights))
        assert len(df) == 1

    def test_wake_uppercased_and_stripped(self):
        """Wake category is uppercased and stripped."""
        flights = [
            _make_flight("AC001", "IBE", " h ", "LEMD", "EGLL",
                         "2023-09-01 08:00", "2023-09-01 10:00"),
        ]
        df = _prepare_flights(_make_df(flights))
        assert df.iloc[0]["AC_WAKE"] == "H"


# ---------------------------------------------------------------------------
# _prepare_flights — physical tests
# ---------------------------------------------------------------------------

class TestPrepareFlightsPhysical:

    def test_all_durations_positive(self, prepared_df):
        """Flight duration must be strictly positive (aircraft can't arrive before departing)."""
        assert (prepared_df["FLIGHT_MINUTES"] > 0).all()

    def test_departure_hours_valid(self, prepared_df):
        """Departure hour must be in [0, 23]."""
        assert (prepared_df["DEP_HOUR_UTC"] >= 0).all()
        assert (prepared_df["DEP_HOUR_UTC"] <= 23).all()

    def test_bins_are_non_negative(self, prepared_df):
        """Flight time bins must be non-negative."""
        assert (prepared_df["FLIGHT_BIN"] >= 0).all()

    def test_bins_are_multiples_of_bin_size(self, prepared_df):
        """Flight time bins must be multiples of BIN_SIZE."""
        assert (prepared_df["FLIGHT_BIN"] % BIN_SIZE == 0).all()

    def test_bin_approximates_duration(self, prepared_df):
        """Binned value should be within BIN_SIZE/2 of the actual duration."""
        diff = (prepared_df["FLIGHT_BIN"] - prepared_df["FLIGHT_MINUTES"]).abs()
        assert (diff <= BIN_SIZE / 2 + 0.01).all()  # +epsilon for float rounding


# ---------------------------------------------------------------------------
# _build_hourly_distributions — software tests
# ---------------------------------------------------------------------------

class TestBuildHourlyDistributionsSoftware:

    def test_output_columns(self, hourly_dist):
        """Output should have the canonical columns."""
        expected = {"origin_id", "dest_id", "airline_id", "aircraft_wake",
                    "dep_hour_utc", "flight_time", "probability"}
        assert expected == set(hourly_dist.columns)

    def test_min_samples_filter(self):
        """Groups with fewer than MIN_SAMPLES observations are excluded."""
        # Only 2 flights for this route — below MIN_SAMPLES=3
        flights = [
            _make_flight("AC001", "IBE", "M", "LEMD", "EGLL",
                         "2023-09-01 08:00", "2023-09-01 10:00"),
            _make_flight("AC002", "IBE", "M", "LEMD", "EGLL",
                         "2023-09-02 08:00", "2023-09-02 10:05"),
        ]
        df = _prepare_flights(_make_df(flights))
        result = _build_hourly_distributions(df)
        assert len(result) == 0

    def test_sufficient_samples_included(self, hourly_dist):
        """Groups with >= MIN_SAMPLES observations produce output rows."""
        assert len(hourly_dist) > 0

    def test_no_count_or_total_columns_leaked(self, hourly_dist):
        """Internal COUNT/TOTAL columns should be dropped."""
        assert "COUNT" not in hourly_dist.columns
        assert "TOTAL" not in hourly_dist.columns


# ---------------------------------------------------------------------------
# _build_hourly_distributions — physical tests
# ---------------------------------------------------------------------------

class TestBuildHourlyDistributionsPhysical:

    def test_probabilities_in_valid_range(self, hourly_dist):
        """All probabilities must be in (0, 1]."""
        assert (hourly_dist["probability"] > 0).all()
        assert (hourly_dist["probability"] <= 1).all()

    def test_probabilities_sum_to_one_per_group(self, hourly_dist):
        """Probabilities must sum to 1 within each (origin, dest, airline, wake, hour) group."""
        group_cols = ["origin_id", "dest_id", "airline_id", "aircraft_wake", "dep_hour_utc"]
        sums = hourly_dist.groupby(group_cols)["probability"].sum()
        np.testing.assert_allclose(sums.values, 1.0, atol=1e-9)

    def test_flight_times_are_positive_multiples_of_bin(self, hourly_dist):
        """Output flight times must be positive multiples of BIN_SIZE."""
        assert (hourly_dist["flight_time"] > 0).all()
        assert (hourly_dist["flight_time"] % BIN_SIZE == 0).all()

    def test_departure_hours_valid(self, hourly_dist):
        """Departure hours in output must be in [0, 23]."""
        assert (hourly_dist["dep_hour_utc"] >= 0).all()
        assert (hourly_dist["dep_hour_utc"] <= 23).all()

    def test_no_self_routes(self, hourly_dist):
        """No row should have origin_id == dest_id (impossible route)."""
        if len(hourly_dist) > 0:
            assert (hourly_dist["origin_id"] != hourly_dist["dest_id"]).all()


# ---------------------------------------------------------------------------
# _build_operator_agnostic_distributions — software tests
# ---------------------------------------------------------------------------

class TestBuildOperatorAgnosticSoftware:

    def test_airline_id_is_all(self, agnostic_dist):
        """All rows must have airline_id='ALL'."""
        if len(agnostic_dist) > 0:
            assert (agnostic_dist["airline_id"] == "ALL").all()

    def test_output_columns(self, agnostic_dist):
        """Output should have the same canonical columns as per-operator."""
        expected = {"origin_id", "dest_id", "airline_id", "aircraft_wake",
                    "dep_hour_utc", "flight_time", "probability"}
        assert expected == set(agnostic_dist.columns)

    def test_no_count_or_total_columns_leaked(self, agnostic_dist):
        """Internal COUNT/TOTAL columns should be dropped."""
        assert "COUNT" not in agnostic_dist.columns
        assert "TOTAL" not in agnostic_dist.columns


# ---------------------------------------------------------------------------
# _build_operator_agnostic_distributions — physical tests
# ---------------------------------------------------------------------------

class TestBuildOperatorAgnosticPhysical:

    def test_probabilities_sum_to_one_per_group(self, agnostic_dist):
        """Probabilities must sum to 1 within each (origin, dest, wake, hour) group."""
        if len(agnostic_dist) == 0:
            return
        group_cols = ["origin_id", "dest_id", "aircraft_wake", "dep_hour_utc"]
        sums = agnostic_dist.groupby(group_cols)["probability"].sum()
        np.testing.assert_allclose(sums.values, 1.0, atol=1e-9)

    def test_probabilities_in_valid_range(self, agnostic_dist):
        """All probabilities must be in (0, 1]."""
        if len(agnostic_dist) == 0:
            return
        assert (agnostic_dist["probability"] > 0).all()
        assert (agnostic_dist["probability"] <= 1).all()

    def test_pools_across_operators(self):
        """Agnostic tier should combine flights from different operators on the same route."""
        flights = [
            # 3 flights on LEMD->EGLL hour 8 from two different airlines
            _make_flight("AC001", "IBE", "M", "LEMD", "EGLL",
                         "2023-09-01 08:00", "2023-09-01 10:00"),
            _make_flight("AC002", "BAW", "M", "LEMD", "EGLL",
                         "2023-09-02 08:00", "2023-09-02 10:05"),
            _make_flight("AC003", "IBE", "M", "LEMD", "EGLL",
                         "2023-09-03 08:00", "2023-09-03 09:55"),
        ]
        df = _prepare_flights(_make_df(flights))
        # Per-operator: IBE has 2 flights (< MIN_SAMPLES), BAW has 1 → both empty
        per_op = _build_hourly_distributions(df)
        assert len(per_op) == 0
        # Agnostic: pooled 3 flights >= MIN_SAMPLES → should produce rows
        agnostic = _build_operator_agnostic_distributions(df)
        assert len(agnostic) > 0


# ---------------------------------------------------------------------------
# analyze_flight_time_distribution — integration tests
# ---------------------------------------------------------------------------

class TestAnalyzeFlightTimeDistribution:

    def _write_schedule_csv(self, tmp_path, flights):
        """Write a minimal schedule CSV for integration tests."""
        df = _make_df(flights)
        path = tmp_path / "schedule.csv"
        df.to_csv(path, index=False)
        return path

    def test_raises_file_not_found(self, tmp_path):
        """FileNotFoundError when schedule file doesn't exist."""
        cfg = PipelineConfig(
            schedule_file=tmp_path / "nonexistent.csv",
            analysis_dir=tmp_path / "analysis",
            output_dir=tmp_path / "output",
        )
        with pytest.raises(FileNotFoundError):
            analyze_flight_time_distribution(cfg)

    def test_raises_on_empty_flights(self, tmp_path):
        """ValueError when no valid flights remain after normalisation."""
        # All flights have zero duration
        flights = [
            _make_flight("AC001", "IBE", "M", "LEMD", "EGLL",
                         "2023-09-01 08:00", "2023-09-01 08:00"),
        ]
        schedule_path = self._write_schedule_csv(tmp_path, flights)
        cfg = PipelineConfig(
            schedule_file=schedule_path,
            analysis_dir=tmp_path / "analysis",
            output_dir=tmp_path / "output",
        )
        with pytest.raises(ValueError, match="No valid flights"):
            analyze_flight_time_distribution(cfg)

    def test_creates_output_file(self, tmp_path):
        """Output CSV should be created."""
        schedule_path = self._write_schedule_csv(tmp_path, SCHEDULE_FLIGHTS)
        cfg = PipelineConfig(
            schedule_file=schedule_path,
            analysis_dir=tmp_path / "analysis",
            output_dir=tmp_path / "output",
        )
        analyze_flight_time_distribution(cfg)
        assert (tmp_path / "analysis" / "scheduled_flight_time.csv").exists()

    def test_suffix_in_filename(self, tmp_path):
        """When suffix is set, it should appear in the output filename."""
        schedule_path = self._write_schedule_csv(tmp_path, SCHEDULE_FLIGHTS)
        cfg = PipelineConfig(
            schedule_file=schedule_path,
            analysis_dir=tmp_path / "analysis",
            output_dir=tmp_path / "output",
            suffix="_test",
        )
        analyze_flight_time_distribution(cfg)
        assert (tmp_path / "analysis" / "scheduled_flight_time_test.csv").exists()

    def test_output_csv_has_correct_columns(self, tmp_path):
        """Output CSV should have the canonical columns."""
        schedule_path = self._write_schedule_csv(tmp_path, SCHEDULE_FLIGHTS)
        cfg = PipelineConfig(
            schedule_file=schedule_path,
            analysis_dir=tmp_path / "analysis",
            output_dir=tmp_path / "output",
        )
        analyze_flight_time_distribution(cfg)
        df = pd.read_csv(tmp_path / "analysis" / "scheduled_flight_time.csv")
        expected = ["origin_id", "dest_id", "airline_id", "aircraft_wake",
                    "dep_hour_utc", "flight_time", "probability"]
        assert list(df.columns) == expected

    def test_output_contains_both_tiers(self, tmp_path):
        """Output should contain both per-operator and operator-agnostic rows."""
        schedule_path = self._write_schedule_csv(tmp_path, SCHEDULE_FLIGHTS)
        cfg = PipelineConfig(
            schedule_file=schedule_path,
            analysis_dir=tmp_path / "analysis",
            output_dir=tmp_path / "output",
        )
        analyze_flight_time_distribution(cfg)
        df = pd.read_csv(tmp_path / "analysis" / "scheduled_flight_time.csv")
        airlines = df["airline_id"].unique()
        assert "ALL" in airlines
        # Should also have at least one non-ALL airline
        assert len(airlines) >= 2

    def test_output_probabilities_valid(self, tmp_path):
        """All probabilities in the output CSV must be in (0, 1] and sum to 1 per group."""
        schedule_path = self._write_schedule_csv(tmp_path, SCHEDULE_FLIGHTS)
        cfg = PipelineConfig(
            schedule_file=schedule_path,
            analysis_dir=tmp_path / "analysis",
            output_dir=tmp_path / "output",
        )
        analyze_flight_time_distribution(cfg)
        df = pd.read_csv(tmp_path / "analysis" / "scheduled_flight_time.csv")
        assert (df["probability"] > 0).all()
        assert (df["probability"] <= 1).all()
        group_cols = ["origin_id", "dest_id", "airline_id", "aircraft_wake", "dep_hour_utc"]
        sums = df.groupby(group_cols)["probability"].sum()
        np.testing.assert_allclose(sums.values, 1.0, atol=1e-9)

    def test_no_aggregate_file_created(self, tmp_path):
        """The aggregate file (levels 4-5) should NOT be created."""
        schedule_path = self._write_schedule_csv(tmp_path, SCHEDULE_FLIGHTS)
        cfg = PipelineConfig(
            schedule_file=schedule_path,
            analysis_dir=tmp_path / "analysis",
            output_dir=tmp_path / "output",
        )
        analyze_flight_time_distribution(cfg)
        assert not (tmp_path / "analysis" / "scheduled_flight_time_aggregate.csv").exists()
