"""Tests for roster_generator.routes module.

Covers
------
- _prepare_flights(): ZZZ remapping, missing wake default, datetime parsing,
  negative/zero duration filtering, both scheduled and actual time computation.
- _build_per_operator_stats(): output structure, median correctness,
  group key composition.
- _build_operator_agnostic_stats(): airline_id='ALL' sentinel, pooling across
  operators.
- _rename_and_round(): column renaming, integer rounding.
- generate_routes(): FileNotFoundError for schedule and markov files,
  ValueError on empty match, output file creation, suffix support,
  CSV column check.

Physical tests
--------------
- Flight durations are strictly positive.
- Median times are strictly positive integers.
- No impossible routes (origin == dest).
- Operator-agnostic rows always have airline_id='ALL'.
- Actual time >= scheduled time is plausible (not enforced, but checked
  when data allows).
- Output contains both per-operator and 'ALL' fallback rows.
"""

import numpy as np
import pandas as pd
import pytest

from roster_generator.auxiliary.routes import (
    _prepare_flights,
    _build_per_operator_stats,
    _build_operator_agnostic_stats,
    _rename_and_round,
    generate_routes,
    AC_REG_COL,
    AIRLINE_COL,
    AC_WAKE_COL,
    DEP_COL,
    ARR_COL,
    STD_COL,
    STA_COL,
    ATD_COL,
    ATA_COL,
)
from roster_generator.config import PipelineConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_flight(reg, airline, wake, dep, arr, std, sta, atd, ata):
    """Build a single flight row dict with scheduled and actual times."""
    return {
        AC_REG_COL: reg,
        AIRLINE_COL: airline,
        AC_WAKE_COL: wake,
        DEP_COL: dep,
        ARR_COL: arr,
        STD_COL: std,
        STA_COL: sta,
        ATD_COL: atd,
        ATA_COL: ata,
    }


def _make_df(flights):
    """Create a DataFrame from a list of flight dicts."""
    return pd.DataFrame(flights)


# A schedule with 4 flights per route to give meaningful medians.
# Route LEMD->EGLL: ~120 min scheduled, ~125 min actual
# Route EGLL->LFPG: ~90 min scheduled, ~88 min actual
SCHEDULE_FLIGHTS = [
    # Day 1
    _make_flight("AC001", "IBE", "M", "LEMD", "EGLL",
                 "2023-09-01 08:00", "2023-09-01 10:00",
                 "2023-09-01 08:05", "2023-09-01 10:10"),
    _make_flight("AC001", "IBE", "M", "EGLL", "LFPG",
                 "2023-09-01 12:00", "2023-09-01 13:30",
                 "2023-09-01 12:05", "2023-09-01 13:28"),
    # Day 2
    _make_flight("AC001", "IBE", "M", "LEMD", "EGLL",
                 "2023-09-02 08:00", "2023-09-02 10:00",
                 "2023-09-02 08:10", "2023-09-02 10:15"),
    _make_flight("AC001", "IBE", "M", "EGLL", "LFPG",
                 "2023-09-02 12:00", "2023-09-02 13:30",
                 "2023-09-02 12:03", "2023-09-02 13:33"),
    # Day 3
    _make_flight("AC001", "IBE", "M", "LEMD", "EGLL",
                 "2023-09-03 08:00", "2023-09-03 09:55",
                 "2023-09-03 08:08", "2023-09-03 10:12"),
    _make_flight("AC001", "IBE", "M", "EGLL", "LFPG",
                 "2023-09-03 12:00", "2023-09-03 13:35",
                 "2023-09-03 12:02", "2023-09-03 13:30"),
    # Day 4
    _make_flight("AC001", "IBE", "M", "LEMD", "EGLL",
                 "2023-09-04 08:00", "2023-09-04 10:05",
                 "2023-09-04 08:03", "2023-09-04 10:08"),
    _make_flight("AC001", "IBE", "M", "EGLL", "LFPG",
                 "2023-09-04 12:00", "2023-09-04 13:30",
                 "2023-09-04 12:01", "2023-09-04 13:31"),
]


@pytest.fixture
def prepared_df():
    """Prepared DataFrame through _prepare_flights."""
    return _prepare_flights(_make_df(SCHEDULE_FLIGHTS))


@pytest.fixture
def per_op_stats(prepared_df):
    """Per-operator statistics DataFrame."""
    return _build_per_operator_stats(prepared_df)


@pytest.fixture
def agnostic_stats(prepared_df):
    """Operator-agnostic statistics DataFrame."""
    return _build_operator_agnostic_stats(prepared_df)


# ---------------------------------------------------------------------------
# _prepare_flights — software tests
# ---------------------------------------------------------------------------

class TestPrepareFlightsSoftware:

    def test_returns_expected_columns(self, prepared_df):
        """Output should contain the derived duration columns."""
        for col in ["SCHEDULED_FLIGHT_TIME", "ACTUAL_FLIGHT_TIME"]:
            assert col in prepared_df.columns

    def test_zzz_airline_remapped(self):
        """Flights with AC_OPERATOR='ZZZ' get AC_REG as airline."""
        flights = [
            _make_flight("EC-ABC", "ZZZ", "M", "LEMD", "EGLL",
                         "2023-09-01 08:00", "2023-09-01 10:00",
                         "2023-09-01 08:05", "2023-09-01 10:10"),
        ]
        df = _prepare_flights(_make_df(flights))
        assert (df[AIRLINE_COL] == "EC-ABC").all()

    def test_missing_wake_defaults_to_m(self):
        """When AC_WAKE column is absent, all wakes default to 'M'."""
        flights = [
            {AC_REG_COL: "AC001", AIRLINE_COL: "IBE",
             DEP_COL: "LEMD", ARR_COL: "EGLL",
             STD_COL: "2023-09-01 08:00", STA_COL: "2023-09-01 10:00",
             ATD_COL: "2023-09-01 08:05", ATA_COL: "2023-09-01 10:10"},
        ]
        df = _prepare_flights(pd.DataFrame(flights))
        assert (df[AC_WAKE_COL] == "M").all()

    def test_unparseable_dates_dropped(self):
        """Rows with invalid datetime strings are removed."""
        flights = [
            _make_flight("AC001", "IBE", "M", "LEMD", "EGLL",
                         "2023-09-01 08:00", "2023-09-01 10:00",
                         "2023-09-01 08:05", "2023-09-01 10:10"),
            _make_flight("AC002", "IBE", "M", "EGLL", "LFPG",
                         "not-a-date", "2023-09-01 13:30",
                         "2023-09-01 12:05", "2023-09-01 13:28"),
        ]
        df = _prepare_flights(_make_df(flights))
        assert len(df) == 1

    def test_missing_actual_times_dropped(self):
        """Rows with missing actual departure/arrival times are removed."""
        flights = [
            _make_flight("AC001", "IBE", "M", "LEMD", "EGLL",
                         "2023-09-01 08:00", "2023-09-01 10:00",
                         "2023-09-01 08:05", "2023-09-01 10:10"),
            _make_flight("AC002", "IBE", "M", "EGLL", "LFPG",
                         "2023-09-01 12:00", "2023-09-01 13:30",
                         "not-a-date", "2023-09-01 13:28"),
        ]
        df = _prepare_flights(_make_df(flights))
        assert len(df) == 1

    def test_negative_scheduled_duration_dropped(self):
        """Flights where STA < STD (negative scheduled duration) are removed."""
        flights = [
            _make_flight("AC001", "IBE", "M", "LEMD", "EGLL",
                         "2023-09-01 10:00", "2023-09-01 08:00",  # backwards
                         "2023-09-01 08:05", "2023-09-01 10:10"),
            _make_flight("AC002", "IBE", "M", "EGLL", "LFPG",
                         "2023-09-01 12:00", "2023-09-01 13:30",
                         "2023-09-01 12:05", "2023-09-01 13:28"),
        ]
        df = _prepare_flights(_make_df(flights))
        assert len(df) == 1

    def test_negative_actual_duration_dropped(self):
        """Flights where ATA < ATD (negative actual duration) are removed."""
        flights = [
            _make_flight("AC001", "IBE", "M", "LEMD", "EGLL",
                         "2023-09-01 08:00", "2023-09-01 10:00",
                         "2023-09-01 10:10", "2023-09-01 08:05"),  # backwards
            _make_flight("AC002", "IBE", "M", "EGLL", "LFPG",
                         "2023-09-01 12:00", "2023-09-01 13:30",
                         "2023-09-01 12:05", "2023-09-01 13:28"),
        ]
        df = _prepare_flights(_make_df(flights))
        assert len(df) == 1

    def test_zero_duration_dropped(self):
        """Flights where STD == STA (zero duration) are removed."""
        flights = [
            _make_flight("AC001", "IBE", "M", "LEMD", "EGLL",
                         "2023-09-01 08:00", "2023-09-01 08:00",
                         "2023-09-01 08:00", "2023-09-01 08:00"),
            _make_flight("AC002", "IBE", "M", "EGLL", "LFPG",
                         "2023-09-01 12:00", "2023-09-01 13:30",
                         "2023-09-01 12:05", "2023-09-01 13:28"),
        ]
        df = _prepare_flights(_make_df(flights))
        assert len(df) == 1

    def test_wake_uppercased_and_stripped(self):
        """Wake category is uppercased and stripped."""
        flights = [
            _make_flight("AC001", "IBE", " h ", "LEMD", "EGLL",
                         "2023-09-01 08:00", "2023-09-01 10:00",
                         "2023-09-01 08:05", "2023-09-01 10:10"),
        ]
        df = _prepare_flights(_make_df(flights))
        assert df.iloc[0][AC_WAKE_COL] == "H"


# ---------------------------------------------------------------------------
# _prepare_flights — physical tests
# ---------------------------------------------------------------------------

class TestPrepareFlightsPhysical:

    def test_all_scheduled_durations_positive(self, prepared_df):
        """Scheduled flight time must be strictly positive."""
        assert (prepared_df["SCHEDULED_FLIGHT_TIME"] > 0).all()

    def test_all_actual_durations_positive(self, prepared_df):
        """Actual flight time must be strictly positive."""
        assert (prepared_df["ACTUAL_FLIGHT_TIME"] > 0).all()

    def test_durations_are_plausible(self, prepared_df):
        """Flight durations should be between 1 minute and 24 hours."""
        for col in ["SCHEDULED_FLIGHT_TIME", "ACTUAL_FLIGHT_TIME"]:
            assert (prepared_df[col] >= 1).all()
            assert (prepared_df[col] <= 1440).all()


# ---------------------------------------------------------------------------
# _build_per_operator_stats — software tests
# ---------------------------------------------------------------------------

class TestBuildPerOperatorStatsSoftware:

    def test_output_columns(self, per_op_stats):
        """Output should have the group columns plus median columns."""
        expected = {DEP_COL, ARR_COL, AIRLINE_COL, AC_WAKE_COL,
                    "SCHEDULED_FLIGHT_TIME", "ACTUAL_FLIGHT_TIME"}
        assert expected == set(per_op_stats.columns)

    def test_one_row_per_group(self, per_op_stats):
        """Each (dep, arr, airline, wake) combination should have exactly one row."""
        group_cols = [DEP_COL, ARR_COL, AIRLINE_COL, AC_WAKE_COL]
        counts = per_op_stats.groupby(group_cols).size()
        assert (counts == 1).all()

    def test_non_empty_with_data(self, per_op_stats):
        """Should produce rows when given valid input."""
        assert len(per_op_stats) > 0

    def test_median_value_correctness(self):
        """Median of 4 known values should be the average of the 2 middle ones."""
        flights = [
            _make_flight("AC001", "IBE", "M", "LEMD", "EGLL",
                         "2023-09-01 08:00", "2023-09-01 10:00",  # 120 min sched
                         "2023-09-01 08:00", "2023-09-01 10:00"),  # 120 min actual
            _make_flight("AC001", "IBE", "M", "LEMD", "EGLL",
                         "2023-09-02 08:00", "2023-09-02 10:20",  # 140 min sched
                         "2023-09-02 08:00", "2023-09-02 10:20"),  # 140 min actual
            _make_flight("AC001", "IBE", "M", "LEMD", "EGLL",
                         "2023-09-03 08:00", "2023-09-03 10:10",  # 130 min sched
                         "2023-09-03 08:00", "2023-09-03 10:10"),  # 130 min actual
            _make_flight("AC001", "IBE", "M", "LEMD", "EGLL",
                         "2023-09-04 08:00", "2023-09-04 10:30",  # 150 min sched
                         "2023-09-04 08:00", "2023-09-04 10:30"),  # 150 min actual
        ]
        df = _prepare_flights(_make_df(flights))
        stats = _build_per_operator_stats(df)
        row = stats.iloc[0]
        # Median of [120, 130, 140, 150] = (130 + 140) / 2 = 135
        assert row["SCHEDULED_FLIGHT_TIME"] == pytest.approx(135.0)
        assert row["ACTUAL_FLIGHT_TIME"] == pytest.approx(135.0)


# ---------------------------------------------------------------------------
# _build_per_operator_stats — physical tests
# ---------------------------------------------------------------------------

class TestBuildPerOperatorStatsPhysical:

    def test_median_times_positive(self, per_op_stats):
        """Median flight times must be strictly positive."""
        assert (per_op_stats["SCHEDULED_FLIGHT_TIME"] > 0).all()
        assert (per_op_stats["ACTUAL_FLIGHT_TIME"] > 0).all()

    def test_no_self_routes(self, per_op_stats):
        """No row should have origin == destination."""
        assert (per_op_stats[DEP_COL] != per_op_stats[ARR_COL]).all()


# ---------------------------------------------------------------------------
# _build_operator_agnostic_stats — software tests
# ---------------------------------------------------------------------------

class TestBuildOperatorAgnosticStatsSoftware:

    def test_airline_id_is_all(self, agnostic_stats):
        """All rows must have airline_id='ALL'."""
        assert (agnostic_stats[AIRLINE_COL] == "ALL").all()

    def test_output_columns(self, agnostic_stats):
        """Output should have the expected columns."""
        expected = {DEP_COL, ARR_COL, AIRLINE_COL, AC_WAKE_COL,
                    "SCHEDULED_FLIGHT_TIME", "ACTUAL_FLIGHT_TIME"}
        assert expected == set(agnostic_stats.columns)

    def test_pools_across_operators(self):
        """Agnostic tier should combine flights from different operators."""
        flights = [
            _make_flight("AC001", "IBE", "M", "LEMD", "EGLL",
                         "2023-09-01 08:00", "2023-09-01 10:00",
                         "2023-09-01 08:05", "2023-09-01 10:10"),
            _make_flight("AC002", "BAW", "M", "LEMD", "EGLL",
                         "2023-09-02 08:00", "2023-09-02 10:05",
                         "2023-09-02 08:03", "2023-09-02 10:08"),
            _make_flight("AC003", "AFR", "M", "LEMD", "EGLL",
                         "2023-09-03 08:00", "2023-09-03 09:55",
                         "2023-09-03 08:02", "2023-09-03 09:58"),
        ]
        df = _prepare_flights(_make_df(flights))
        # Per-operator: each has 1 flight → separate rows
        per_op = _build_per_operator_stats(df)
        assert len(per_op) == 3  # one per airline
        # Agnostic: pooled → single row
        agnostic = _build_operator_agnostic_stats(df)
        assert len(agnostic) == 1
        assert agnostic.iloc[0][AIRLINE_COL] == "ALL"


# ---------------------------------------------------------------------------
# _build_operator_agnostic_stats — physical tests
# ---------------------------------------------------------------------------

class TestBuildOperatorAgnosticStatsPhysical:

    def test_median_times_positive(self, agnostic_stats):
        """Median flight times must be strictly positive."""
        assert (agnostic_stats["SCHEDULED_FLIGHT_TIME"] > 0).all()
        assert (agnostic_stats["ACTUAL_FLIGHT_TIME"] > 0).all()

    def test_no_self_routes(self, agnostic_stats):
        """No row should have origin == destination."""
        assert (agnostic_stats[DEP_COL] != agnostic_stats[ARR_COL]).all()


# ---------------------------------------------------------------------------
# _rename_and_round — software tests
# ---------------------------------------------------------------------------

class TestRenameAndRound:

    def test_output_column_names(self, per_op_stats):
        """Columns should be renamed to the output schema."""
        result = _rename_and_round(per_op_stats)
        expected = {"orig_id", "dest_id", "airline_id", "wake_type",
                    "scheduled_time", "time"}
        assert expected == set(result.columns)

    def test_times_are_integers(self, per_op_stats):
        """Times should be rounded to integer minutes."""
        result = _rename_and_round(per_op_stats)
        assert result["scheduled_time"].dtype == int
        assert result["time"].dtype == int

    def test_rounding_correctness(self):
        """Values should round to nearest integer."""
        df = pd.DataFrame({
            DEP_COL: ["LEMD"],
            ARR_COL: ["EGLL"],
            AIRLINE_COL: ["IBE"],
            AC_WAKE_COL: ["M"],
            "SCHEDULED_FLIGHT_TIME": [120.4],
            "ACTUAL_FLIGHT_TIME": [125.6],
        })
        result = _rename_and_round(df)
        assert result.iloc[0]["scheduled_time"] == 120
        assert result.iloc[0]["time"] == 126


# ---------------------------------------------------------------------------
# generate_routes — integration tests
# ---------------------------------------------------------------------------

class TestGenerateRoutes:

    def _write_csv(self, path, df):
        """Write a DataFrame to CSV."""
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return path

    def _write_schedule_csv(self, tmp_path, flights):
        """Write a minimal schedule CSV."""
        df = _make_df(flights)
        path = tmp_path / "schedule.csv"
        df.to_csv(path, index=False)
        return path

    def _write_markov_csv(self, tmp_path):
        """Write a minimal markov CSV with routes matching SCHEDULE_FLIGHTS."""
        markov_df = pd.DataFrame({
            DEP_COL: ["LEMD", "EGLL"],
            ARR_COL: ["EGLL", "LFPG"],
            AC_WAKE_COL: ["M", "M"],
            AIRLINE_COL: ["IBE", "IBE"],
            "PREV_ICAO": ["LFPG", "LEMD"],
            "DEP_HOUR_UTC": [8, 12],
            "PROB": [1.0, 1.0],
            "COUNT": [4, 4],
        })
        path = tmp_path / "analysis" / "markov.csv"
        return self._write_csv(path, markov_df)

    def test_raises_schedule_not_found(self, tmp_path):
        """FileNotFoundError when schedule file doesn't exist."""
        self._write_markov_csv(tmp_path)
        cfg = PipelineConfig(
            schedule_file=tmp_path / "nonexistent.csv",
            analysis_dir=tmp_path / "analysis",
            output_dir=tmp_path / "output",
        )
        with pytest.raises(FileNotFoundError, match="Schedule file"):
            generate_routes(cfg)

    def test_raises_markov_not_found(self, tmp_path):
        """FileNotFoundError when markov file doesn't exist."""
        schedule_path = self._write_schedule_csv(tmp_path, SCHEDULE_FLIGHTS)
        cfg = PipelineConfig(
            schedule_file=schedule_path,
            analysis_dir=tmp_path / "analysis",
            output_dir=tmp_path / "output",
        )
        with pytest.raises(FileNotFoundError, match="Markov file"):
            generate_routes(cfg)

    def test_raises_on_no_matching_routes(self, tmp_path):
        """ValueError when no schedule flights match the markov routes."""
        schedule_path = self._write_schedule_csv(tmp_path, SCHEDULE_FLIGHTS)
        # Markov file with routes that don't exist in the schedule
        markov_df = pd.DataFrame({
            DEP_COL: ["KJFK"],
            ARR_COL: ["KLAX"],
            AC_WAKE_COL: ["H"],
            AIRLINE_COL: ["AAL"],
            "PREV_ICAO": ["KORD"],
            "DEP_HOUR_UTC": [10],
            "PROB": [1.0],
            "COUNT": [1],
        })
        self._write_csv(tmp_path / "analysis" / "markov.csv", markov_df)
        cfg = PipelineConfig(
            schedule_file=schedule_path,
            analysis_dir=tmp_path / "analysis",
            output_dir=tmp_path / "output",
        )
        with pytest.raises(ValueError, match="No flights match"):
            generate_routes(cfg)

    def test_creates_output_file(self, tmp_path):
        """Output CSV should be created in output_dir."""
        schedule_path = self._write_schedule_csv(tmp_path, SCHEDULE_FLIGHTS)
        self._write_markov_csv(tmp_path)
        cfg = PipelineConfig(
            schedule_file=schedule_path,
            analysis_dir=tmp_path / "analysis",
            output_dir=tmp_path / "output",
        )
        generate_routes(cfg)
        assert (tmp_path / "output" / "routes.csv").exists()

    def test_suffix_in_filename(self, tmp_path):
        """When suffix is set, it should appear in the output filename."""
        schedule_path = self._write_schedule_csv(tmp_path, SCHEDULE_FLIGHTS)
        # Write markov with suffix
        markov_df = pd.DataFrame({
            DEP_COL: ["LEMD", "EGLL"],
            ARR_COL: ["EGLL", "LFPG"],
            AC_WAKE_COL: ["M", "M"],
            AIRLINE_COL: ["IBE", "IBE"],
            "PREV_ICAO": ["LFPG", "LEMD"],
            "DEP_HOUR_UTC": [8, 12],
            "PROB": [1.0, 1.0],
            "COUNT": [4, 4],
        })
        self._write_csv(tmp_path / "analysis" / "markov_test.csv", markov_df)
        cfg = PipelineConfig(
            schedule_file=schedule_path,
            analysis_dir=tmp_path / "analysis",
            output_dir=tmp_path / "output",
            suffix="_test",
        )
        generate_routes(cfg)
        assert (tmp_path / "output" / "routes_test.csv").exists()

    def test_output_csv_has_correct_columns(self, tmp_path):
        """Output CSV should have the canonical columns."""
        schedule_path = self._write_schedule_csv(tmp_path, SCHEDULE_FLIGHTS)
        self._write_markov_csv(tmp_path)
        cfg = PipelineConfig(
            schedule_file=schedule_path,
            analysis_dir=tmp_path / "analysis",
            output_dir=tmp_path / "output",
        )
        generate_routes(cfg)
        df = pd.read_csv(tmp_path / "output" / "routes.csv")
        expected = ["orig_id", "dest_id", "airline_id", "wake_type",
                    "scheduled_time", "time"]
        assert list(df.columns) == expected

    def test_output_contains_both_tiers(self, tmp_path):
        """Output should contain both per-operator and operator-agnostic rows."""
        schedule_path = self._write_schedule_csv(tmp_path, SCHEDULE_FLIGHTS)
        self._write_markov_csv(tmp_path)
        cfg = PipelineConfig(
            schedule_file=schedule_path,
            analysis_dir=tmp_path / "analysis",
            output_dir=tmp_path / "output",
        )
        generate_routes(cfg)
        df = pd.read_csv(tmp_path / "output" / "routes.csv")
        airlines = df["airline_id"].unique()
        assert "ALL" in airlines
        assert len(airlines) >= 2

    def test_output_times_are_positive_integers(self, tmp_path):
        """All times in the output CSV must be positive integers."""
        schedule_path = self._write_schedule_csv(tmp_path, SCHEDULE_FLIGHTS)
        self._write_markov_csv(tmp_path)
        cfg = PipelineConfig(
            schedule_file=schedule_path,
            analysis_dir=tmp_path / "analysis",
            output_dir=tmp_path / "output",
        )
        generate_routes(cfg)
        df = pd.read_csv(tmp_path / "output" / "routes.csv")
        assert (df["scheduled_time"] > 0).all()
        assert (df["time"] > 0).all()
        # Check they are integers (no decimals in CSV)
        assert (df["scheduled_time"] == df["scheduled_time"].astype(int)).all()
        assert (df["time"] == df["time"].astype(int)).all()

    def test_output_no_self_routes(self, tmp_path):
        """No row should have orig_id == dest_id."""
        schedule_path = self._write_schedule_csv(tmp_path, SCHEDULE_FLIGHTS)
        self._write_markov_csv(tmp_path)
        cfg = PipelineConfig(
            schedule_file=schedule_path,
            analysis_dir=tmp_path / "analysis",
            output_dir=tmp_path / "output",
        )
        generate_routes(cfg)
        df = pd.read_csv(tmp_path / "output" / "routes.csv")
        assert (df["orig_id"] != df["dest_id"]).all()

    def test_output_file_in_output_dir_not_analysis(self, tmp_path):
        """Routes output must land in output_dir, not analysis_dir."""
        schedule_path = self._write_schedule_csv(tmp_path, SCHEDULE_FLIGHTS)
        self._write_markov_csv(tmp_path)
        cfg = PipelineConfig(
            schedule_file=schedule_path,
            analysis_dir=tmp_path / "analysis",
            output_dir=tmp_path / "output",
        )
        generate_routes(cfg)
        assert (tmp_path / "output" / "routes.csv").exists()
        assert not (tmp_path / "analysis" / "routes.csv").exists()
