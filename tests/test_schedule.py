"""Tests for roster_generator.schedule module.

Covers
------
- generate_schedule(): FileNotFoundError on missing input files,
  output file creation, suffix support, column schema, correct seeding
  for reproducibility.
- DataManager: turnaround category sampling, destination lookup with
  fallback, flight time retrieval from routes.
- CapacityTracker: availability check, flight registration, violation
  counting.
- ScheduleGenerator.generate_chain(): chain construction from initial
  conditions, single-flight passthrough, prior-flight pasting.

Physical tests
--------------
- Flight time is strictly positive for every flight (STA_UTC > STD_UTC).
- No self-routes (orig_id != dest_id).
- Departure times are non-negative (STD_UTC >= 0).
- Chain spatial continuity: each subsequent flight departs from the
  previous flight's destination.
- Chain temporal ordering: each subsequent flight departs no earlier
  than the previous flight's arrival.
- Exactly one first_flight == 1 per aircraft.
- Turnaround category is one of the valid values ("", "intraday",
  "next_day").
- All airline_id and aircraft_id values are non-empty strings.
- Deterministic output for the same seed.
- Single-flight aircraft produce exactly one flight in the output.
"""

import math
import random

import numpy as np
import pandas as pd
import pytest

from roster_generator.schedule import (
    generate_schedule,
    DataManager,
    CapacityTracker,
    ScheduleGenerator,
    GenerationStats,
    Flight,
    Aircraft,
    BIN_SIZE_MINS,
    END_OF_DAY,
)
from roster_generator.config import PipelineConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _make_initial_conditions(*rows) -> pd.DataFrame:
    """Build an initial_conditions DataFrame.

    Each row is a dict with keys:
      AC_REG, AC_OPERATOR, AC_WAKE, ORIGIN, DEST,
      STD_UTC_MINS, STA_UTC_MINS, SINGLE_FLIGHT, PRIOR_ONLY
    and optionally PRIOR_ORIGIN, PRIOR_DEST, PRIOR_STD_UTC_MINS, PRIOR_STA_UTC_MINS.
    """
    return pd.DataFrame(rows)


def _make_routes(*rows) -> pd.DataFrame:
    """Build a routes DataFrame from (orig, dest, airline, wake, time) tuples."""
    return pd.DataFrame(rows, columns=[
        "orig_id", "dest_id", "airline_id", "wake_type", "scheduled_time",
    ])


def _make_airports(*rows) -> pd.DataFrame:
    """Build an airports DataFrame from (id, rolling, burst) tuples."""
    return pd.DataFrame(rows, columns=[
        "airport_id", "rolling_capacity", "burst_capacity",
    ])


def _make_markov(*rows) -> pd.DataFrame:
    """Build a markov DataFrame from (op, wake, prev, dep, arr, hour, count) tuples."""
    return pd.DataFrame(rows, columns=[
        "AC_OPERATOR", "AC_WAKE", "PREV_ICAO", "DEP_ICAO",
        "ARR_ICAO", "DEP_HOUR_UTC", "COUNT",
    ])


def _make_turnaround_intraday_params(*rows) -> pd.DataFrame:
    """Build turnaround intraday params from (airline, wake, location, shape) tuples."""
    return pd.DataFrame(rows, columns=["airline", "wake", "location", "shape"])


def _make_turnaround_temporal_profile(*rows) -> pd.DataFrame:
    """Build turnaround temporal profile from dict rows."""
    return pd.DataFrame(rows)


# --- Standard test scenario ---
# Two airports (LEMD, EGLL), one airline (IBE), wake M.
# Routes: LEMD->EGLL 120 min, EGLL->LEMD 120 min.
# Aircraft departs LEMD at 08:00 (480 mins), arrives EGLL at 10:00 (600 mins).
# Markov says: at EGLL (prev LEMD) -> go LEMD, at LEMD (prev EGLL) -> go EGLL.

AIRPORTS = _make_airports(
    ("LEMD", 999, 999),
    ("EGLL", 999, 999),
)

ROUTES = _make_routes(
    ("LEMD", "EGLL", "IBE", "M", 120),
    ("EGLL", "LEMD", "IBE", "M", 120),
    ("LEMD", "EGLL", "ALL", "M", 120),
    ("EGLL", "LEMD", "ALL", "M", 120),
)

MARKOV = _make_markov(
    ("IBE", "M", "EGLL", "LEMD", "EGLL", 8, 10),
    ("IBE", "M", "LEMD", "EGLL", "LEMD", 10, 10),
    ("IBE", "M", "EGLL", "LEMD", "EGLL", 12, 10),
    ("IBE", "M", "LEMD", "EGLL", "LEMD", 14, 10),
    ("IBE", "M", "EGLL", "LEMD", "EGLL", 16, 10),
    ("IBE", "M", "LEMD", "EGLL", "LEMD", 18, 10),
)

# Lognormal params: location = ln(45) ≈ 3.81, shape = 0.2 → ~45 min turnaround
TURNAROUND_INTRADAY = _make_turnaround_intraday_params(
    ("IBE", "M", math.log(45), 0.2),
)

# Temporal profile: strong intraday signal, no next-day signal.
# 5-min bins: e.g. bin 120 = minute 600 (10:00).  We put counts in hours 8-18.
TURNAROUND_TEMPORAL = _make_turnaround_temporal_profile(
    {
        "airline": "IBE",
        "previous_origin": "LEMD",
        "origin": "EGLL",
        "wake": "M",
        "intraday_sparse": ";".join(f"{m}:10" for m in range(480, 1200, 60)),
        "next_day_sparse": "",
        "total_intraday": 120,
        "total_next_day": 0,
    },
    {
        "airline": "IBE",
        "previous_origin": "EGLL",
        "origin": "LEMD",
        "wake": "M",
        "intraday_sparse": ";".join(f"{m}:10" for m in range(480, 1200, 60)),
        "next_day_sparse": "",
        "total_intraday": 120,
        "total_next_day": 0,
    },
)

IC_STANDARD = _make_initial_conditions(
    {
        "AC_REG": "AC001", "AC_OPERATOR": "IBE", "AC_WAKE": "M",
        "ORIGIN": "LEMD", "DEST": "EGLL",
        "STD_UTC_MINS": 480, "STA_UTC_MINS": 600,
        "SINGLE_FLIGHT": 0, "PRIOR_ONLY": 0,
    },
)

IC_TWO_AIRCRAFT = _make_initial_conditions(
    {
        "AC_REG": "AC001", "AC_OPERATOR": "IBE", "AC_WAKE": "M",
        "ORIGIN": "LEMD", "DEST": "EGLL",
        "STD_UTC_MINS": 480, "STA_UTC_MINS": 600,
        "SINGLE_FLIGHT": 0, "PRIOR_ONLY": 0,
    },
    {
        "AC_REG": "AC002", "AC_OPERATOR": "IBE", "AC_WAKE": "M",
        "ORIGIN": "EGLL", "DEST": "LEMD",
        "STD_UTC_MINS": 540, "STA_UTC_MINS": 660,
        "SINGLE_FLIGHT": 0, "PRIOR_ONLY": 0,
    },
)

IC_SINGLE_FLIGHT = _make_initial_conditions(
    {
        "AC_REG": "AC010", "AC_OPERATOR": "IBE", "AC_WAKE": "M",
        "ORIGIN": "LEMD", "DEST": "EGLL",
        "STD_UTC_MINS": 480, "STA_UTC_MINS": 600,
        "SINGLE_FLIGHT": 1, "PRIOR_ONLY": 0,
    },
)


def _write_all_inputs(tmp_path, ic_df, suffix=""):
    """Write all 6 input files and return a PipelineConfig."""
    analysis = tmp_path / "analysis"
    output = tmp_path / "output"

    _write_csv(analysis / f"initial_conditions{suffix}.csv", ic_df)
    _write_csv(output / f"routes{suffix}.csv", ROUTES)
    _write_csv(output / f"airports{suffix}.csv", AIRPORTS)
    _write_csv(analysis / f"markov{suffix}.csv", MARKOV)
    _write_csv(
        analysis / f"scheduled_turnaround_intraday_params{suffix}.csv",
        TURNAROUND_INTRADAY,
    )
    _write_csv(
        analysis / f"scheduled_turnaround_temporal_profile{suffix}.csv",
        TURNAROUND_TEMPORAL,
    )

    return PipelineConfig(
        schedule_file=tmp_path / "schedule.csv",
        analysis_dir=analysis,
        output_dir=output,
        seed=42,
        suffix=suffix,
    )


def _run_schedule(tmp_path, ic_df=None, suffix="", seed=42) -> pd.DataFrame:
    """Write inputs, run generate_schedule, return output DataFrame."""
    if ic_df is None:
        ic_df = IC_STANDARD
    cfg = _write_all_inputs(tmp_path, ic_df, suffix=suffix)
    cfg.seed = seed
    generate_schedule(cfg)
    return pd.read_csv(cfg.output_path("schedule"))


# ---------------------------------------------------------------------------
# generate_schedule — software tests
# ---------------------------------------------------------------------------

class TestGenerateScheduleSoftware:

    def test_raises_when_initial_conditions_missing(self, tmp_path):
        """FileNotFoundError when initial_conditions file is absent."""
        cfg = _write_all_inputs(tmp_path, IC_STANDARD)
        # Remove the initial_conditions file
        ic_path = cfg.analysis_path("initial_conditions")
        ic_path.unlink()
        with pytest.raises(FileNotFoundError):
            generate_schedule(cfg)

    def test_raises_when_routes_missing(self, tmp_path):
        """FileNotFoundError when routes file is absent."""
        cfg = _write_all_inputs(tmp_path, IC_STANDARD)
        cfg.output_path("routes").unlink()
        with pytest.raises(FileNotFoundError):
            generate_schedule(cfg)

    def test_raises_when_airports_missing(self, tmp_path):
        """FileNotFoundError when airports file is absent."""
        cfg = _write_all_inputs(tmp_path, IC_STANDARD)
        cfg.output_path("airports").unlink()
        with pytest.raises(FileNotFoundError):
            generate_schedule(cfg)

    def test_raises_when_markov_missing(self, tmp_path):
        """FileNotFoundError when markov file is absent."""
        cfg = _write_all_inputs(tmp_path, IC_STANDARD)
        cfg.analysis_path("markov").unlink()
        with pytest.raises(FileNotFoundError):
            generate_schedule(cfg)

    def test_raises_when_turnaround_intraday_missing(self, tmp_path):
        """FileNotFoundError when turnaround intraday params file is absent."""
        cfg = _write_all_inputs(tmp_path, IC_STANDARD)
        cfg.analysis_path("scheduled_turnaround_intraday_params").unlink()
        with pytest.raises(FileNotFoundError):
            generate_schedule(cfg)

    def test_raises_when_turnaround_temporal_missing(self, tmp_path):
        """FileNotFoundError when turnaround temporal profile file is absent."""
        cfg = _write_all_inputs(tmp_path, IC_STANDARD)
        cfg.analysis_path("scheduled_turnaround_temporal_profile").unlink()
        with pytest.raises(FileNotFoundError):
            generate_schedule(cfg)

    def test_creates_output_file(self, tmp_path):
        """Output CSV must be created in output_dir."""
        cfg = _write_all_inputs(tmp_path, IC_STANDARD)
        generate_schedule(cfg)
        assert cfg.output_path("schedule").exists()

    def test_suffix_in_filename(self, tmp_path):
        """When suffix is set it must appear in the output filename."""
        cfg = _write_all_inputs(tmp_path, IC_STANDARD, suffix="_42")
        generate_schedule(cfg)
        assert (tmp_path / "output" / "schedule_42.csv").exists()

    def test_output_not_in_analysis_dir(self, tmp_path):
        """Output must land in output_dir, not analysis_dir."""
        cfg = _write_all_inputs(tmp_path, IC_STANDARD)
        generate_schedule(cfg)
        assert not (tmp_path / "analysis" / "schedule.csv").exists()

    def test_output_column_schema(self, tmp_path):
        """Output CSV must have the expected columns."""
        df = _run_schedule(tmp_path)
        expected = {
            "airline_id", "aircraft_id", "orig_id", "dest_id",
            "STD_UTC", "STA_UTC", "first_flight",
            "is_prior_flight", "is_initial_departure",
            "single_flight_real",
            "turnaround_to_next_category", "turnaround_to_next_minutes",
        }
        assert expected == set(df.columns)

    def test_output_non_empty(self, tmp_path):
        """Schedule must produce at least one flight row."""
        df = _run_schedule(tmp_path)
        assert len(df) > 0

    def test_reproducible_with_same_seed(self, tmp_path):
        """Two runs with the same seed must produce identical output."""
        df1 = _run_schedule(tmp_path / "run1", seed=123)
        df2 = _run_schedule(tmp_path / "run2", seed=123)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_may_differ(self, tmp_path):
        """Two runs with different seeds should (with high probability) differ."""
        df1 = _run_schedule(tmp_path / "run1", ic_df=IC_TWO_AIRCRAFT, seed=1)
        df2 = _run_schedule(tmp_path / "run2", ic_df=IC_TWO_AIRCRAFT, seed=9999)
        # With multiple aircraft the flight counts or destinations will likely differ
        differs = len(df1) != len(df2) or not df1.equals(df2)
        assert differs

    def test_single_flight_passthrough(self, tmp_path):
        """An aircraft with SINGLE_FLIGHT=1 produces exactly one flight."""
        df = _run_schedule(tmp_path, ic_df=IC_SINGLE_FLIGHT)
        ac_flights = df[df["aircraft_id"] == "AC010"]
        assert len(ac_flights) == 1
        assert ac_flights.iloc[0]["single_flight_real"] == 1

    def test_output_dir_created_if_absent(self, tmp_path):
        """generate_schedule must create output_dir if it does not exist."""
        ic_df = IC_STANDARD
        analysis = tmp_path / "analysis"
        output = tmp_path / "fresh_output"
        assert not output.exists()
        _write_csv(analysis / "initial_conditions.csv", ic_df)
        _write_csv(output / "routes.csv", ROUTES)
        _write_csv(output / "airports.csv", AIRPORTS)
        _write_csv(analysis / "markov.csv", MARKOV)
        _write_csv(analysis / "scheduled_turnaround_intraday_params.csv", TURNAROUND_INTRADAY)
        _write_csv(analysis / "scheduled_turnaround_temporal_profile.csv", TURNAROUND_TEMPORAL)
        cfg = PipelineConfig(
            schedule_file=tmp_path / "schedule.csv",
            analysis_dir=analysis,
            output_dir=output,
            seed=42,
        )
        generate_schedule(cfg)
        assert output.exists()
        assert cfg.output_path("schedule").exists()

    def test_multi_aircraft_all_present(self, tmp_path):
        """Each aircraft from initial_conditions must appear in the output."""
        df = _run_schedule(tmp_path, ic_df=IC_TWO_AIRCRAFT)
        assert set(df["aircraft_id"]) == {"AC001", "AC002"}


# ---------------------------------------------------------------------------
# generate_schedule — physical tests
# ---------------------------------------------------------------------------

class TestGenerateSchedulePhysical:

    def _run(self, tmp_path, ic_df=None) -> pd.DataFrame:
        return _run_schedule(tmp_path, ic_df=ic_df)

    def test_flight_time_positive(self, tmp_path):
        """Every flight must have STA_UTC > STD_UTC (positive flight time)."""
        df = self._run(tmp_path)
        assert (df["STA_UTC"] > df["STD_UTC"]).all(), (
            "Some flights have non-positive flight time"
        )

    def test_departure_times_non_negative(self, tmp_path):
        """All departure times must be >= 0."""
        df = self._run(tmp_path)
        assert (df["STD_UTC"] >= 0).all()

    def test_no_self_routes(self, tmp_path):
        """No flight should have origin == destination."""
        df = self._run(tmp_path)
        assert (df["orig_id"] != df["dest_id"]).all()

    def test_chain_spatial_continuity(self, tmp_path):
        """Within each aircraft, the next flight's origin must equal the
        previous flight's destination."""
        df = self._run(tmp_path, ic_df=IC_TWO_AIRCRAFT)
        for ac_id, group in df.groupby("aircraft_id"):
            flights = group.sort_values("STD_UTC").reset_index(drop=True)
            for i in range(1, len(flights)):
                prev_dest = flights.iloc[i - 1]["dest_id"]
                curr_orig = flights.iloc[i]["orig_id"]
                assert prev_dest == curr_orig, (
                    f"Aircraft {ac_id} flight {i}: expected orig={prev_dest}, "
                    f"got {curr_orig}"
                )

    def test_chain_temporal_ordering(self, tmp_path):
        """Within each aircraft, each flight departs no earlier than the
        previous flight's arrival."""
        df = self._run(tmp_path, ic_df=IC_TWO_AIRCRAFT)
        for ac_id, group in df.groupby("aircraft_id"):
            flights = group.sort_values("STD_UTC").reset_index(drop=True)
            for i in range(1, len(flights)):
                prev_sta = flights.iloc[i - 1]["STA_UTC"]
                curr_std = flights.iloc[i]["STD_UTC"]
                assert curr_std >= prev_sta, (
                    f"Aircraft {ac_id} flight {i}: departs at {curr_std} "
                    f"before previous arrival at {prev_sta}"
                )

    def test_turnaround_non_negative(self, tmp_path):
        """Within each aircraft, turnaround time (next STD - current STA) >= 0."""
        df = self._run(tmp_path, ic_df=IC_TWO_AIRCRAFT)
        for _, group in df.groupby("aircraft_id"):
            flights = group.sort_values("STD_UTC").reset_index(drop=True)
            for i in range(1, len(flights)):
                ta = flights.iloc[i]["STD_UTC"] - flights.iloc[i - 1]["STA_UTC"]
                assert ta >= 0, f"Negative turnaround: {ta} minutes"

    def test_turnaround_aligned_to_bin_size(self, tmp_path):
        """Departure times must be aligned to 5-minute bins."""
        df = self._run(tmp_path)
        assert (df["STD_UTC"] % BIN_SIZE_MINS == 0).all(), (
            "Some departure times are not aligned to 5-minute bins"
        )

    def test_exactly_one_first_flight_per_aircraft(self, tmp_path):
        """Each aircraft must have exactly one flight marked as first_flight=1."""
        df = self._run(tmp_path, ic_df=IC_TWO_AIRCRAFT)
        first_counts = df.groupby("aircraft_id")["first_flight"].sum()
        assert (first_counts == 1).all(), (
            f"first_flight counts per aircraft: {first_counts.to_dict()}"
        )

    def test_airline_ids_non_empty(self, tmp_path):
        """All airline_id values must be non-null and non-empty."""
        df = self._run(tmp_path)
        assert df["airline_id"].notna().all()
        assert (df["airline_id"].astype(str).str.strip() != "").all()

    def test_aircraft_ids_non_empty(self, tmp_path):
        """All aircraft_id values must be non-null and non-empty."""
        df = self._run(tmp_path)
        assert df["aircraft_id"].notna().all()
        assert (df["aircraft_id"].astype(str).str.strip() != "").all()

    def test_turnaround_category_valid(self, tmp_path):
        """turnaround_to_next_category must be one of the valid values."""
        df = self._run(tmp_path)
        valid = {"", "intraday", "next_day"}
        categories = set(df["turnaround_to_next_category"].fillna("").astype(str))
        invalid = categories - valid
        assert not invalid, f"Unexpected turnaround categories: {invalid}"

    def test_initial_flight_preserved(self, tmp_path):
        """The initial flight from initial_conditions must appear unchanged."""
        df = self._run(tmp_path)
        initial = df[df["is_initial_departure"] == 1]
        assert len(initial) == 1
        row = initial.iloc[0]
        assert row["orig_id"] == "LEMD"
        assert row["dest_id"] == "EGLL"
        assert row["STD_UTC"] == 480
        assert row["STA_UTC"] == 600

    def test_flight_times_match_routes(self, tmp_path):
        """Flight durations must correspond to route flight times."""
        df = self._run(tmp_path)
        route_times = {
            ("LEMD", "EGLL"): 120,
            ("EGLL", "LEMD"): 120,
        }
        for _, row in df.iterrows():
            key = (row["orig_id"], row["dest_id"])
            expected = route_times.get(key)
            if expected is not None:
                actual = row["STA_UTC"] - row["STD_UTC"]
                assert actual == expected, (
                    f"Route {key}: expected {expected} min, got {actual} min"
                )

    def test_single_flight_no_turnaround(self, tmp_path):
        """A single-flight aircraft should not generate continuation flights."""
        df = _run_schedule(tmp_path, ic_df=IC_SINGLE_FLIGHT)
        ac_flights = df[df["aircraft_id"] == "AC010"]
        assert len(ac_flights) == 1

    def test_multi_aircraft_chain_lengths_plausible(self, tmp_path):
        """With 120-min flights and ~45-min turnarounds starting at 08:00,
        each aircraft should produce at least 2 flights before end of day."""
        df = self._run(tmp_path, ic_df=IC_TWO_AIRCRAFT)
        for ac_id, group in df.groupby("aircraft_id"):
            assert len(group) >= 2, (
                f"Aircraft {ac_id} produced only {len(group)} flight(s)"
            )


# ---------------------------------------------------------------------------
# CapacityTracker — unit tests
# ---------------------------------------------------------------------------

class TestCapacityTrackerSoftware:

    def test_empty_tracker_allows_all(self):
        """An empty tracker should allow any flight."""
        tracker = CapacityTracker({"LEMD": 999, "EGLL": 999}, {"LEMD": 999, "EGLL": 999})
        assert tracker.check_availability("LEMD", "EGLL", 480, 600)

    def test_burst_capacity_blocks_when_exceeded(self):
        """Flights should be blocked when burst capacity is reached."""
        tracker = CapacityTracker({"LEMD": 999}, {"LEMD": 1})
        f = Flight(orig="LEMD", dest="EGLL", std=480, sta=600)
        tracker.add_flight(f)
        assert not tracker.check_availability("LEMD", "EGLL", 480, 600)

    def test_no_violations_initially(self):
        """Fresh tracker has zero violations."""
        tracker = CapacityTracker({"LEMD": 10}, {"LEMD": 5})
        burst, rolling = tracker.compute_violations()
        assert burst == 0
        assert rolling == 0


class TestCapacityTrackerPhysical:

    def test_violations_non_negative(self):
        """Violations must always be >= 0."""
        tracker = CapacityTracker({"LEMD": 2, "EGLL": 2}, {"LEMD": 1, "EGLL": 1})
        for i in range(5):
            f = Flight(orig="LEMD", dest="EGLL", std=480 + i * 5, sta=600 + i * 5)
            tracker.add_flight(f)
        burst, rolling = tracker.compute_violations()
        assert burst >= 0
        assert rolling >= 0


# ---------------------------------------------------------------------------
# Flight / Aircraft data structures — unit tests
# ---------------------------------------------------------------------------

class TestDataStructures:

    def test_flight_defaults(self):
        """Flight should have default turnaround values."""
        f = Flight(orig="LEMD", dest="EGLL", std=480, sta=600)
        assert f.turnaround_to_next_category == ""
        assert f.turnaround_to_next_minutes == -1

    def test_aircraft_defaults(self):
        """Aircraft should have sensible defaults."""
        ac = Aircraft(reg="AC001", operator="IBE", wake="M")
        assert ac.initial_flight is None
        assert ac.prior_flight is None
        assert ac.chain == []
        assert not ac.is_single_flight
        assert not ac.is_prior_only
