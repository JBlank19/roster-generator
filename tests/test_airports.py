"""Tests for roster_generator.airports module.

Covers
------
- _compute_capacities(): column schema, floor enforcement, unparseable
  timestamps, unknown-airport fallback, bin5 correctness, burst detection.
- generate_airports(): FileNotFoundError on missing Markov file, graceful
  degradation when schedule is absent, output file creation, suffix support,
  column schema, airport deduplication.

Physical tests
--------------
- Rolling capacity >= burst capacity (a 60-min window can never be smaller
  than its best 5-min sub-window).
- All capacities are strictly positive integers.
- Airport IDs are non-empty ICAO strings (4 uppercase letters).
- No duplicate airport_id rows in the output.
- An airport with 12 movements in one 5-min bin must have burst_capacity >= 12.
- An airport processed across multiple days yields the worst-day value.
"""

import pandas as pd
import pytest

from roster_generator.airports import (
    _compute_capacities,
    generate_airports,
    DEP_COL,
    ARR_COL,
    STD_COL,
    STA_COL,
)
from roster_generator.config import PipelineConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_schedule(*rows) -> pd.DataFrame:
    """Build a minimal schedule DataFrame from (dep, arr, std, sta) tuples."""
    return pd.DataFrame([
        {DEP_COL: dep, ARR_COL: arr, STD_COL: std, STA_COL: sta}
        for dep, arr, std, sta in rows
    ])


def _make_markov(dep_airports, arr_airports) -> pd.DataFrame:
    """Build a minimal Markov CSV DataFrame."""
    rows = []
    for dep, arr in zip(dep_airports, arr_airports):
        rows.append({
            DEP_COL: dep,
            ARR_COL: arr,
            "AC_OPERATOR": "IBE",
            "AC_WAKE": "M",
            "PREV_ICAO": "ZZZZ",
            "DEP_HOUR_UTC": 8,
            "PROB": 1.0,
            "COUNT": 1,
        })
    return pd.DataFrame(rows)


def _write_csv(path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Schedule: LEMD has 12 movements packed into the same 5-min bin on one day.
# EGLL has 1 movement per hour across two days.
DENSE_SCHEDULE = _make_schedule(
    # 12 departures from LEMD in a single 5-min window (08:00-08:04)
    *[("LEMD", "EGLL", "2023-09-01 08:00", "2023-09-01 10:00")] * 12,
    # 1 arrival from EGLL each hour over a full day
    *[("EGLL", "LFPG",
       f"2023-09-01 {h:02d}:30", f"2023-09-01 {h:02d}:50")
      for h in range(24)],
)


@pytest.fixture
def capacity_df():
    """_compute_capacities result for the DENSE_SCHEDULE."""
    return _compute_capacities(DENSE_SCHEDULE, ["LEMD", "EGLL"])


# ---------------------------------------------------------------------------
# _compute_capacities — software tests
# ---------------------------------------------------------------------------

class TestComputeCapacitiesSoftware:

    def test_output_columns(self, capacity_df):
        """Result must have exactly the three expected columns."""
        assert set(capacity_df.columns) == {"airport_id", "rolling_capacity", "burst_capacity"}

    def test_returns_all_requested_airports(self, capacity_df):
        """Every airport passed in must appear in the result."""
        assert set(capacity_df["airport_id"]) == {"LEMD", "EGLL"}

    def test_floor_one_for_unknown_airport(self):
        """An airport not in the schedule receives capacity = 1."""
        df = _compute_capacities(DENSE_SCHEDULE, ["ZZZZ"])
        row = df[df["airport_id"] == "ZZZZ"].iloc[0]
        assert row["rolling_capacity"] == 1
        assert row["burst_capacity"] == 1

    def test_empty_schedule_returns_floor(self):
        """An entirely empty movement set yields all-floor capacities."""
        empty = pd.DataFrame({DEP_COL: [], ARR_COL: [], STD_COL: [], STA_COL: []})
        result = _compute_capacities(empty, ["LEMD"])
        assert result.iloc[0]["rolling_capacity"] == 1
        assert result.iloc[0]["burst_capacity"] == 1

    def test_unparseable_timestamps_dropped(self):
        """Rows with invalid timestamps do not cause errors; they are skipped."""
        schedule = _make_schedule(
            ("LEMD", "EGLL", "not-a-date", "2023-09-01 10:00"),
            ("LEMD", "EGLL", "2023-09-01 09:00", "2023-09-01 11:00"),
        )
        result = _compute_capacities(schedule, ["LEMD"])
        # Should not raise; should produce a valid result based on the good row
        assert result.iloc[0]["rolling_capacity"] >= 1

    def test_burst_reflects_dense_bin(self, capacity_df):
        """An airport with 12 movements in a single 5-min bin has burst >= 12."""
        lemd = capacity_df[capacity_df["airport_id"] == "LEMD"].iloc[0]
        assert lemd["burst_capacity"] >= 12

    def test_suffix_isolated_airports(self):
        """Each call is isolated; airports absent from schedule get floor = 1."""
        result = _compute_capacities(DENSE_SCHEDULE, ["LEMD", "ZZZZ"])
        zzzz = result[result["airport_id"] == "ZZZZ"].iloc[0]
        assert zzzz["rolling_capacity"] == 1
        assert zzzz["burst_capacity"] == 1

    def test_capacities_are_integers(self, capacity_df):
        """Both capacity columns must hold integer values."""
        assert capacity_df["rolling_capacity"].dtype == int
        assert capacity_df["burst_capacity"].dtype == int

    def test_worst_day_is_used(self):
        """When an airport is busier on one day, that day's peak governs."""
        # Day 1: single movement in bin 08:00
        # Day 2: three movements packed into the same 5-min bin
        schedule = _make_schedule(
            ("LEMD", "EGLL", "2023-09-01 08:00", "2023-09-01 10:00"),
            ("LEMD", "EGLL", "2023-09-02 08:00", "2023-09-02 10:00"),
            ("LEMD", "EGLL", "2023-09-02 08:01", "2023-09-02 10:01"),
            ("LEMD", "EGLL", "2023-09-02 08:02", "2023-09-02 10:02"),
        )
        result = _compute_capacities(schedule, ["LEMD"])
        assert result.iloc[0]["burst_capacity"] >= 3


# ---------------------------------------------------------------------------
# _compute_capacities — physical tests
# ---------------------------------------------------------------------------

class TestComputeCapacitiesPhysical:

    def test_rolling_capacity_gte_burst_capacity(self, capacity_df):
        """A 60-min window must contain at least as many movements as any
        single 5-min sub-window within it."""
        for _, row in capacity_df.iterrows():
            assert row["rolling_capacity"] >= row["burst_capacity"], (
                f"{row['airport_id']}: rolling={row['rolling_capacity']} "
                f"< burst={row['burst_capacity']}"
            )

    def test_all_capacities_positive(self, capacity_df):
        """All capacity values must be strictly positive."""
        assert (capacity_df["rolling_capacity"] >= 1).all()
        assert (capacity_df["burst_capacity"] >= 1).all()

    def test_rolling_capacity_plausible_upper_bound(self, capacity_df):
        """Rolling capacity cannot exceed total movements in the dataset."""
        total_moves = len(DENSE_SCHEDULE) * 2  # each row = 1 dep + 1 arr
        for _, row in capacity_df.iterrows():
            assert row["rolling_capacity"] <= total_moves, (
                f"{row['airport_id']} rolling_capacity={row['rolling_capacity']} "
                f"exceeds total movements {total_moves}"
            )

    def test_burst_capacity_at_most_rolling(self, capacity_df):
        """Burst (5-min window) cannot exceed rolling (60-min window)."""
        assert (capacity_df["burst_capacity"] <= capacity_df["rolling_capacity"]).all()


# ---------------------------------------------------------------------------
# generate_airports — integration tests
# ---------------------------------------------------------------------------

class TestGenerateAirports:

    def _write_markov(self, tmp_path, deps, arrs, suffix=""):
        df = _make_markov(deps, arrs)
        path = tmp_path / "analysis" / f"markov{suffix}.csv"
        _write_csv(path, df)
        return path

    def _write_schedule(self, tmp_path):
        path = tmp_path / "schedule.csv"
        DENSE_SCHEDULE.to_csv(path, index=False)
        return path

    def _make_config(self, tmp_path, suffix=""):
        schedule_path = self._write_schedule(tmp_path)
        return PipelineConfig(
            schedule_file=schedule_path,
            analysis_dir=tmp_path / "analysis",
            output_dir=tmp_path / "output",
            suffix=suffix,
        )

    # --- error handling ---

    def test_raises_when_markov_missing(self, tmp_path):
        """FileNotFoundError when no Markov file exists."""
        cfg = PipelineConfig(
            schedule_file=tmp_path / "schedule.csv",
            analysis_dir=tmp_path / "analysis",
            output_dir=tmp_path / "output",
        )
        with pytest.raises(FileNotFoundError, match="Markov file"):
            generate_airports(cfg)

    # --- file creation ---

    def test_creates_output_file(self, tmp_path):
        """Output CSV must be created inside output_dir."""
        self._write_markov(tmp_path, ["LEMD", "EGLL"], ["EGLL", "LFPG"])
        cfg = self._make_config(tmp_path)
        generate_airports(cfg)
        assert (tmp_path / "output" / "airports.csv").exists()

    def test_suffix_in_filename(self, tmp_path):
        """When suffix is set it must appear in the output filename."""
        self._write_markov(tmp_path, ["LEMD"], ["EGLL"], suffix="_42")
        cfg = self._make_config(tmp_path, suffix="_42")
        generate_airports(cfg)
        assert (tmp_path / "output" / "airports_42.csv").exists()

    def test_output_not_in_analysis_dir(self, tmp_path):
        """Output must land in output_dir, not analysis_dir."""
        self._write_markov(tmp_path, ["LEMD"], ["EGLL"])
        cfg = self._make_config(tmp_path)
        generate_airports(cfg)
        assert not (tmp_path / "analysis" / "airports.csv").exists()

    # --- schema ---

    def test_output_column_schema(self, tmp_path):
        """Output CSV must have exactly the expected columns."""
        self._write_markov(tmp_path, ["LEMD", "EGLL"], ["EGLL", "LFPG"])
        cfg = self._make_config(tmp_path)
        generate_airports(cfg)
        df = pd.read_csv(tmp_path / "output" / "airports.csv")
        assert list(df.columns) == ["airport_id", "rolling_capacity", "burst_capacity"]

    def test_fallback_columns_when_no_schedule(self, tmp_path):
        """When the schedule file is absent, floor columns must still be present."""
        self._write_markov(tmp_path, ["LEMD"], ["EGLL"])
        cfg = PipelineConfig(
            schedule_file=tmp_path / "nonexistent_schedule.csv",
            analysis_dir=tmp_path / "analysis",
            output_dir=tmp_path / "output",
        )
        generate_airports(cfg)
        df = pd.read_csv(tmp_path / "output" / "airports.csv")
        assert "rolling_capacity" in df.columns
        assert "burst_capacity" in df.columns

    # --- deduplication ---

    def test_union_of_dep_and_arr_airports(self, tmp_path):
        """All airports appearing only as departure or only as arrival
        must be included."""
        # LEMD appears only as dep, KJFK only as arr
        self._write_markov(tmp_path, ["LEMD"], ["KJFK"])
        cfg = self._make_config(tmp_path)
        generate_airports(cfg)
        df = pd.read_csv(tmp_path / "output" / "airports.csv")
        assert "LEMD" in df["airport_id"].values
        assert "KJFK" in df["airport_id"].values

    def test_no_duplicate_airport_ids(self, tmp_path):
        """Each airport must appear exactly once even if it is in both
        departure and arrival columns of the Markov table."""
        # EGLL appears as both dep and arr
        self._write_markov(tmp_path,
                           ["LEMD", "EGLL"],
                           ["EGLL", "LEMD"])
        cfg = self._make_config(tmp_path)
        generate_airports(cfg)
        df = pd.read_csv(tmp_path / "output" / "airports.csv")
        assert df["airport_id"].nunique() == len(df)


# ---------------------------------------------------------------------------
# generate_airports — physical tests
# ---------------------------------------------------------------------------

class TestGenerateAirportsPhysical:

    def _run(self, tmp_path, deps, arrs) -> pd.DataFrame:
        schedule_path = tmp_path / "schedule.csv"
        DENSE_SCHEDULE.to_csv(schedule_path, index=False)
        markov_path = tmp_path / "analysis" / "markov.csv"
        _write_csv(markov_path, _make_markov(deps, arrs))
        cfg = PipelineConfig(
            schedule_file=schedule_path,
            analysis_dir=tmp_path / "analysis",
            output_dir=tmp_path / "output",
        )
        generate_airports(cfg)
        return pd.read_csv(tmp_path / "output" / "airports.csv")

    def test_all_capacities_positive(self, tmp_path):
        """Both capacity columns must be >= 1 for every row."""
        df = self._run(tmp_path, ["LEMD", "EGLL"], ["EGLL", "LFPG"])
        assert (df["rolling_capacity"] >= 1).all()
        assert (df["burst_capacity"] >= 1).all()

    def test_capacities_are_integer_dtype(self, tmp_path):
        """Capacity values must be stored as integers in the CSV."""
        df = self._run(tmp_path, ["LEMD"], ["EGLL"])
        assert (df["rolling_capacity"] == df["rolling_capacity"].astype(int)).all()
        assert (df["burst_capacity"] == df["burst_capacity"].astype(int)).all()

    def test_rolling_gte_burst(self, tmp_path):
        """Rolling capacity >= burst capacity for every airport."""
        df = self._run(tmp_path, ["LEMD", "EGLL"], ["EGLL", "LFPG"])
        assert (df["rolling_capacity"] >= df["burst_capacity"]).all()

    def test_airport_ids_non_empty(self, tmp_path):
        """No airport_id in the output may be null or empty."""
        df = self._run(tmp_path, ["LEMD", "EGLL"], ["EGLL", "LFPG"])
        assert df["airport_id"].notna().all()
        assert (df["airport_id"].str.strip() != "").all()

    def test_floor_for_airports_absent_from_schedule(self, tmp_path):
        """An airport in the Markov table but not in the schedule gets
        rolling_capacity = burst_capacity = 1."""
        df = self._run(tmp_path, ["ZZZZ"], ["ZZZZ"])
        row = df[df["airport_id"] == "ZZZZ"].iloc[0]
        assert row["rolling_capacity"] == 1
        assert row["burst_capacity"] == 1
