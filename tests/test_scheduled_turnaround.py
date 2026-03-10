"""Tests for roster_generator.scheduled_turnaround module.

Covers
------
- _safe_shape(): finite clamping, NaN/Inf handling, minimum enforcement.
- _fit_lognormal_params(): single/two/multi-sample fitting, zero/negative filtering,
  empty input, deterministic fallbacks.
- _encode_sparse_hist(): zero-suppression, correct minute labels, round-trip format.
- _prepare_turnaround_events(): ZZZ remapping, missing wake default, datetime parsing,
  linked-flight construction, day_gap filtering, category assignment, arrival binning.
- _build_param_and_temporal_rows(): return structure (2 lists), intraday keying by (airline, wake),
  temporal keying by route, histogram consistency.
- _validate_outputs(): column checks, duplicate detection, NaN/non-finite/non-positive shape.
- analyze_turnaround_distribution(): FileNotFoundError, output file creation (2 files), suffix support.

Physical tests
--------------
- Turnaround times are non-negative.
- Lognormal shape parameters are strictly positive and finite.
- Location parameters are finite.
- Histogram counts are non-negative and within valid minute range.
- Category is always intraday or next_day.
- Arrival minute bins are in [0, 1440).
"""

import numpy as np
import pandas as pd
import pytest

from roster_generator.distribution_analysis.scheduled_turnaround import (
    _safe_shape,
    _fit_lognormal_params,
    _encode_sparse_hist,
    _prepare_turnaround_events,
    _build_param_and_temporal_rows,
    _validate_outputs,
    analyze_turnaround_distribution,
    AC_REG_COL,
    AIRLINE_COL,
    DEP_STATION_COL,
    ARR_STATION_COL,
    STD_COL,
    STA_COL,
    BIN_SIZE,
    MINUTES_PER_DAY,
    INTRADAY_CATEGORY,
    NEXT_DAY_CATEGORY,
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


# A minimal two-aircraft, multi-leg schedule for intraday turnarounds.
# Aircraft 1 (AC001): LEMD->EGLL 08:00-10:00, EGLL->LFPG 12:00-13:30, LFPG->LEMD 16:00-18:00
# Aircraft 2 (AC002): EGLL->LFPG 09:00-10:30, LFPG->LEMD 14:00-16:00
INTRADAY_FLIGHTS = [
    _make_flight("AC001", "IBE", "M", "LEMD", "EGLL", "2023-09-01 08:00", "2023-09-01 10:00"),
    _make_flight("AC001", "IBE", "M", "EGLL", "LFPG", "2023-09-01 12:00", "2023-09-01 13:30"),
    _make_flight("AC001", "IBE", "M", "LFPG", "LEMD", "2023-09-01 16:00", "2023-09-01 18:00"),
    _make_flight("AC002", "IBE", "M", "EGLL", "LFPG", "2023-09-01 09:00", "2023-09-01 10:30"),
    _make_flight("AC002", "IBE", "M", "LFPG", "LEMD", "2023-09-01 14:00", "2023-09-01 16:00"),
]

# Schedule with a next-day turnaround.
# AC003 arrives EGLL at 23:00 day 1, departs EGLL at 07:00 day 2.
NEXT_DAY_FLIGHTS = [
    _make_flight("AC003", "BAW", "H", "LFPG", "EGLL", "2023-09-01 21:00", "2023-09-01 23:00"),
    _make_flight("AC003", "BAW", "H", "EGLL", "LEMD", "2023-09-02 07:00", "2023-09-02 09:30"),
]


@pytest.fixture
def intraday_events():
    """Prepared turnaround events from intraday schedule."""
    return _prepare_turnaround_events(_make_df(INTRADAY_FLIGHTS))


@pytest.fixture
def mixed_events():
    """Prepared turnaround events from combined intraday + next-day schedule."""
    return _prepare_turnaround_events(_make_df(INTRADAY_FLIGHTS + NEXT_DAY_FLIGHTS))


# ---------------------------------------------------------------------------
# _safe_shape
# ---------------------------------------------------------------------------

class TestSafeShape:

    def test_finite_value_above_minimum(self):
        """Value above minimum is returned as-is."""
        assert _safe_shape(0.5) == 0.5

    def test_value_below_minimum_clamped(self):
        """Value below default minimum (0.05) is clamped."""
        assert _safe_shape(0.01) == 0.05

    def test_nan_returns_minimum(self):
        """NaN is replaced by the minimum."""
        assert _safe_shape(float("nan")) == 0.05

    def test_inf_returns_minimum(self):
        """Positive infinity is replaced by the minimum."""
        assert _safe_shape(float("inf")) == 0.05

    def test_neg_inf_returns_minimum(self):
        """Negative infinity is replaced by the minimum."""
        assert _safe_shape(float("-inf")) == 0.05

    def test_custom_minimum(self):
        """Custom minimum is respected."""
        assert _safe_shape(0.01, minimum=0.02) == 0.02

    def test_exact_minimum_passes(self):
        """Value equal to minimum passes through."""
        assert _safe_shape(0.05) == 0.05

    def test_zero_clamped(self):
        """Zero is below default minimum and gets clamped."""
        assert _safe_shape(0.0) == 0.05


# ---------------------------------------------------------------------------
# _fit_lognormal_params
# ---------------------------------------------------------------------------

class TestFitLognormalParams:

    def test_single_sample_returns_default_shape(self):
        """With one sample, shape defaults to 0.12."""
        loc, shape = _fit_lognormal_params(np.array([30.0]))
        assert shape == pytest.approx(0.12)
        assert np.isfinite(loc)

    def test_two_samples(self):
        """With two samples, uses ddof=0 for std."""
        loc, shape = _fit_lognormal_params(np.array([30.0, 60.0]))
        assert np.isfinite(loc)
        assert shape >= 0.05

    def test_multiple_samples(self):
        """With many samples, uses ddof=1 for std."""
        values = np.array([25.0, 30.0, 35.0, 40.0, 45.0])
        loc, shape = _fit_lognormal_params(values)
        assert np.isfinite(loc)
        assert shape > 0

    def test_empty_raises(self):
        """Empty array raises ValueError."""
        with pytest.raises(ValueError, match="zero samples"):
            _fit_lognormal_params(np.array([]))

    def test_all_nan_raises(self):
        """Array of all NaNs raises ValueError."""
        with pytest.raises(ValueError, match="zero samples"):
            _fit_lognormal_params(np.array([np.nan, np.nan]))

    def test_all_negative_raises(self):
        """Array of all negative values raises ValueError."""
        with pytest.raises(ValueError, match="zero samples"):
            _fit_lognormal_params(np.array([-10.0, -5.0]))

    def test_mixed_nan_and_valid(self):
        """NaN values are filtered; fit uses remaining valid values."""
        loc, shape = _fit_lognormal_params(np.array([np.nan, 30.0, np.nan]))
        assert np.isfinite(loc)
        assert shape == pytest.approx(0.12)  # single valid sample

    def test_zero_values_filtered(self):
        """Zero values are excluded (log(0) is undefined)."""
        loc, shape = _fit_lognormal_params(np.array([0.0, 30.0]))
        assert np.isfinite(loc)

    # --- Physical tests ---

    def test_location_is_log_mean(self):
        """Location parameter should equal mean of log-values for identical inputs."""
        val = 45.0
        loc, _ = _fit_lognormal_params(np.array([val, val, val]))
        assert loc == pytest.approx(np.log(val), abs=1e-9)

    def test_shape_always_positive(self):
        """Shape must always be strictly positive (physical constraint)."""
        for vals in [[30.0], [30.0, 30.0], [25.0, 30.0, 35.0]]:
            _, shape = _fit_lognormal_params(np.array(vals))
            assert shape > 0

    def test_shape_always_finite(self):
        """Shape must always be finite."""
        _, shape = _fit_lognormal_params(np.array([30.0]))
        assert np.isfinite(shape)


# ---------------------------------------------------------------------------
# _encode_sparse_hist
# ---------------------------------------------------------------------------

class TestEncodeSparseHist:

    def test_empty_histogram(self):
        """All-zero histogram produces empty string."""
        hist = np.zeros(MINUTES_PER_DAY // BIN_SIZE, dtype=int)
        assert _encode_sparse_hist(hist) == ""

    def test_single_nonzero_bin(self):
        """Single non-zero bin encodes as 'minute:count'."""
        hist = np.zeros(MINUTES_PER_DAY // BIN_SIZE, dtype=int)
        hist[2] = 5  # bin index 2 → minute 10
        assert _encode_sparse_hist(hist) == "10:5"

    def test_multiple_nonzero_bins(self):
        """Multiple non-zero bins separated by semicolons."""
        hist = np.zeros(MINUTES_PER_DAY // BIN_SIZE, dtype=int)
        hist[0] = 3   # minute 0
        hist[24] = 7  # minute 120
        result = _encode_sparse_hist(hist)
        assert result == "0:3;120:7"

    def test_zero_bins_suppressed(self):
        """Bins with count 0 are not included in the output."""
        hist = np.zeros(10, dtype=int)
        hist[1] = 1
        hist[3] = 2
        result = _encode_sparse_hist(hist)
        parts = result.split(";")
        assert len(parts) == 2

    def test_minute_labels_are_multiples_of_bin_size(self):
        """All minute labels in output must be multiples of BIN_SIZE."""
        hist = np.zeros(MINUTES_PER_DAY // BIN_SIZE, dtype=int)
        hist[5] = 1
        hist[10] = 2
        hist[48] = 3
        result = _encode_sparse_hist(hist)
        for part in result.split(";"):
            minute = int(part.split(":")[0])
            assert minute % BIN_SIZE == 0


# ---------------------------------------------------------------------------
# _prepare_turnaround_events — software tests
# ---------------------------------------------------------------------------

class TestPrepareTurnaroundEventsSoftware:

    def test_returns_expected_columns(self, intraday_events):
        """Output should have the canonical event columns."""
        expected = {"airline", "previous_origin", "origin", "wake", "ta_minutes",
                    "category", "arr_minute_bin"}
        assert expected == set(intraday_events.columns)

    def test_zzz_airline_remapped(self):
        """Flights with AC_OPERATOR='ZZZ' get AC_REG as airline."""
        flights = [
            _make_flight("EC-ABC", "ZZZ", "M", "LEMD", "EGLL",
                         "2023-09-01 08:00", "2023-09-01 10:00"),
            _make_flight("EC-ABC", "ZZZ", "M", "EGLL", "LFPG",
                         "2023-09-01 12:00", "2023-09-01 13:30"),
        ]
        events = _prepare_turnaround_events(_make_df(flights))
        assert (events["airline"] == "EC-ABC").all()

    def test_missing_wake_defaults_to_m(self):
        """When AC_WAKE column is absent, all wakes default to 'M'."""
        flights = [
            {AC_REG_COL: "AC001", AIRLINE_COL: "IBE",
             DEP_STATION_COL: "LEMD", ARR_STATION_COL: "EGLL",
             STD_COL: "2023-09-01 08:00", STA_COL: "2023-09-01 10:00"},
            {AC_REG_COL: "AC001", AIRLINE_COL: "IBE",
             DEP_STATION_COL: "EGLL", ARR_STATION_COL: "LFPG",
             STD_COL: "2023-09-01 12:00", STA_COL: "2023-09-01 13:30"},
        ]
        events = _prepare_turnaround_events(pd.DataFrame(flights))
        assert (events["wake"] == "M").all()

    def test_unparseable_dates_dropped(self):
        """Rows with invalid datetime strings are removed before linking."""
        flights = [
            _make_flight("AC001", "IBE", "M", "LEMD", "EGLL",
                         "2023-09-01 08:00", "2023-09-01 10:00"),
            _make_flight("AC001", "IBE", "M", "EGLL", "LFPG",
                         "not-a-date", "2023-09-01 13:30"),
            _make_flight("AC001", "IBE", "M", "LFPG", "LEMD",
                         "2023-09-01 16:00", "2023-09-01 18:00"),
        ]
        events = _prepare_turnaround_events(_make_df(flights))
        # After dropping invalid row, only LEMD->EGLL and LFPG->LEMD remain.
        # But linking requires consecutive flights at same airport, so result may be empty
        # or contain only LFPG->LEMD if the gap is bridged.
        # The key check: no crash and result contains only valid data.
        assert events["ta_minutes"].notna().all()

    def test_first_flights_excluded(self, intraday_events):
        """First flight per aircraft has no predecessor and should not appear."""
        # AC001 first is LEMD->EGLL, AC002 first is EGLL->LFPG
        # These should not generate turnaround events
        assert len(intraday_events) < len(INTRADAY_FLIGHTS)

    def test_non_connecting_flights_excluded(self):
        """Flights where previous arrival != current departure are excluded."""
        flights = [
            _make_flight("AC001", "IBE", "M", "LEMD", "EGLL",
                         "2023-09-01 08:00", "2023-09-01 10:00"),
            # Departs LFPG but arrived at EGLL — non-connecting
            _make_flight("AC001", "IBE", "M", "LFPG", "LEMD",
                         "2023-09-01 14:00", "2023-09-01 16:00"),
        ]
        events = _prepare_turnaround_events(_make_df(flights))
        assert len(events) == 0

    def test_multi_day_layovers_excluded(self):
        """Turnarounds spanning more than one day (DAY_GAP > 1) are excluded."""
        flights = [
            _make_flight("AC001", "IBE", "M", "LEMD", "EGLL",
                         "2023-09-01 08:00", "2023-09-01 10:00"),
            # 3 days later
            _make_flight("AC001", "IBE", "M", "EGLL", "LFPG",
                         "2023-09-04 08:00", "2023-09-04 09:30"),
        ]
        events = _prepare_turnaround_events(_make_df(flights))
        assert len(events) == 0

    def test_intraday_category_assigned(self, intraday_events):
        """All same-day turnarounds should have category='intraday'."""
        assert (intraday_events["category"] == INTRADAY_CATEGORY).all()

    def test_next_day_category_assigned(self):
        """Overnight turnarounds should have category='next_day'."""
        events = _prepare_turnaround_events(_make_df(NEXT_DAY_FLIGHTS))
        assert (events["category"] == NEXT_DAY_CATEGORY).all()


# ---------------------------------------------------------------------------
# _prepare_turnaround_events — physical / domain tests
# ---------------------------------------------------------------------------

class TestPrepareTurnaroundEventsPhysical:

    def test_turnaround_times_non_negative(self, intraday_events):
        """Turnaround time must be >= 0 (aircraft can't depart before arriving)."""
        assert (intraday_events["ta_minutes"] >= 0).all()

    def test_arrival_bins_within_day(self, intraday_events):
        """Arrival minute bins must be in [0, 1440)."""
        assert (intraday_events["arr_minute_bin"] >= 0).all()
        assert (intraday_events["arr_minute_bin"] < MINUTES_PER_DAY).all()

    def test_arrival_bins_are_multiples_of_bin_size(self, intraday_events):
        """Arrival bins must be aligned to BIN_SIZE-minute boundaries."""
        assert (intraday_events["arr_minute_bin"] % BIN_SIZE == 0).all()

    def test_category_is_valid(self, mixed_events):
        """Category must be either 'intraday' or 'next_day'."""
        valid_categories = {INTRADAY_CATEGORY, NEXT_DAY_CATEGORY}
        assert set(mixed_events["category"].unique()).issubset(valid_categories)

    def test_origin_matches_previous_arrival(self):
        """The turnaround origin must be the airport where the aircraft previously arrived."""
        # AC001: LEMD->EGLL, then EGLL->LFPG. Turnaround at EGLL.
        events = _prepare_turnaround_events(_make_df(INTRADAY_FLIGHTS))
        # All origins should be airports where a previous flight arrived
        for _, row in events.iterrows():
            assert row["origin"] in ["EGLL", "LFPG"]

    def test_turnaround_time_matches_schedule(self):
        """Turnaround time should equal STD_next - STA_prev in minutes."""
        # AC001: arrives EGLL at 10:00, departs EGLL at 12:00 → 120 min
        events = _prepare_turnaround_events(_make_df(INTRADAY_FLIGHTS))
        egll_ta = events[events["origin"] == "EGLL"]["ta_minutes"]
        assert 120.0 in egll_ta.values


# ---------------------------------------------------------------------------
# _build_param_and_temporal_rows — software tests
# ---------------------------------------------------------------------------

class TestBuildParamAndTemporalRowsSoftware:

    def test_returns_three_lists(self, intraday_events):
        """Should return (intraday_rows, temporal_rows)."""
        result = _build_param_and_temporal_rows(intraday_events)
        assert len(result) == 2
        intraday_rows, temporal_rows = result
        assert isinstance(intraday_rows, list)
        assert isinstance(temporal_rows, list)

    def test_intraday_rows_have_expected_keys(self, intraday_events):
        """Each intraday row should have airline, wake, location, shape."""
        intraday_rows, temporal_rows = _build_param_and_temporal_rows(intraday_events)
        assert len(intraday_rows) > 0
        for row in intraday_rows:
            assert set(row.keys()) == {"airline", "wake", "location", "shape"}

    def test_temporal_rows_have_expected_keys(self, intraday_events):
        """Each temporal row should have the full route key + histogram fields."""
        _, temporal_rows = _build_param_and_temporal_rows(intraday_events)
        expected_keys = {"airline", "previous_origin", "origin", "wake",
                         "intraday_sparse", "next_day_sparse",
                         "total_intraday", "total_next_day"}
        for row in temporal_rows:
            assert set(row.keys()) == expected_keys

    def test_intraday_keyed_by_airline_wake(self, intraday_events):
        """Intraday params should be aggregated by (airline, wake), not per route."""
        intraday_rows, temporal_rows = _build_param_and_temporal_rows(intraday_events)
        keys = {(r["airline"], r["wake"]) for r in intraday_rows}
        # All flights are IBE/M, so there should be exactly one entry
        assert keys == {("IBE", "M")}

    def test_temporal_rows_keyed_by_route(self, intraday_events):
        """Temporal rows should be keyed by (airline, previous_origin, origin, wake)."""
        _, temporal_rows = _build_param_and_temporal_rows(intraday_events)
        keys = {(r["airline"], r["previous_origin"], r["origin"], r["wake"])
                for r in temporal_rows}
        # Should have distinct route-level entries
        assert len(keys) == len(temporal_rows)

    def test_no_duplicate_temporal_keys(self, mixed_events):
        """No duplicate (airline, previous_origin, origin, wake) in temporal rows."""
        _, temporal_rows = _build_param_and_temporal_rows(mixed_events)
        keys = [(r["airline"], r["previous_origin"], r["origin"], r["wake"])
                for r in temporal_rows]
        assert len(keys) == len(set(keys))


# ---------------------------------------------------------------------------
# _build_param_and_temporal_rows — physical / domain tests
# ---------------------------------------------------------------------------

class TestBuildParamAndTemporalRowsPhysical:

    def test_shape_is_positive(self, intraday_events):
        """All fitted shape parameters must be strictly positive."""
        intraday_rows, temporal_rows = _build_param_and_temporal_rows(intraday_events)
        for row in intraday_rows:
            assert row["shape"] > 0

    def test_shape_is_finite(self, intraday_events):
        """All fitted shape parameters must be finite."""
        intraday_rows, temporal_rows = _build_param_and_temporal_rows(intraday_events)
        for row in intraday_rows:
            assert np.isfinite(row["shape"])

    def test_location_is_finite(self, intraday_events):
        """All fitted location parameters must be finite."""
        intraday_rows, temporal_rows = _build_param_and_temporal_rows(intraday_events)
        for row in intraday_rows:
            assert np.isfinite(row["location"])

    def test_histogram_counts_non_negative(self, mixed_events):
        """total_intraday and total_next_day must be non-negative."""
        _, temporal_rows = _build_param_and_temporal_rows(mixed_events)
        for row in temporal_rows:
            assert row["total_intraday"] >= 0
            assert row["total_next_day"] >= 0

    def test_histogram_total_matches_event_count(self, intraday_events):
        """Sum of all temporal histogram totals should equal number of events."""
        _, temporal_rows = _build_param_and_temporal_rows(intraday_events)
        total = sum(r["total_intraday"] + r["total_next_day"] for r in temporal_rows)
        assert total == len(intraday_events)

    def test_sparse_hist_minutes_within_day(self, mixed_events):
        """All minute labels in sparse histograms must be in [0, 1440)."""
        _, temporal_rows = _build_param_and_temporal_rows(mixed_events)
        for row in temporal_rows:
            for field in ("intraday_sparse", "next_day_sparse"):
                if row[field]:
                    for part in row[field].split(";"):
                        minute = int(part.split(":")[0])
                        assert 0 <= minute < MINUTES_PER_DAY


# ---------------------------------------------------------------------------
# _validate_outputs
# ---------------------------------------------------------------------------

class TestValidateOutputs:

    def _make_valid_frames(self):
        """Build minimal valid DataFrames for validation."""
        intraday_df = pd.DataFrame({
            "airline": ["IBE"],
            "wake": ["M"],
            "location": [3.5],
            "shape": [0.3],
        })
        temporal_df = pd.DataFrame({
            "airline": ["IBE"],
            "previous_origin": ["LEMD"],
            "origin": ["EGLL"],
            "wake": ["M"],
            "intraday_sparse": ["600:3"],
            "next_day_sparse": [""],
            "total_intraday": [3],
            "total_next_day": [0],
        })
        return intraday_df, temporal_df

    def test_valid_frames_pass(self):
        """Well-formed DataFrames should pass without error."""
        _validate_outputs(*self._make_valid_frames())

    def test_wrong_intraday_columns_raises(self):
        """ValueError if intraday columns are wrong."""
        intraday_df, temporal_df = self._make_valid_frames()
        intraday_df = intraday_df.rename(columns={"shape": "sigma"})
        with pytest.raises(ValueError, match="intraday columns mismatch"):
            _validate_outputs(intraday_df, temporal_df)

    def test_duplicate_intraday_keys_raises(self):
        """ValueError if intraday has duplicate (airline, wake) pairs."""
        intraday_df, temporal_df = self._make_valid_frames()
        intraday_df = pd.concat([intraday_df, intraday_df], ignore_index=True)
        with pytest.raises(ValueError, match="duplicate keys"):
            _validate_outputs(intraday_df, temporal_df)

    def test_nan_params_raises(self):
        """ValueError if intraday params contain NaN."""
        intraday_df, temporal_df = self._make_valid_frames()
        intraday_df.loc[0, "location"] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            _validate_outputs(intraday_df, temporal_df)

    def test_non_finite_params_raises(self):
        """ValueError if intraday params contain Inf."""
        intraday_df, temporal_df = self._make_valid_frames()
        intraday_df.loc[0, "shape"] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            _validate_outputs(intraday_df, temporal_df)

    def test_non_positive_shape_raises(self):
        """ValueError if shape is zero or negative."""
        intraday_df, temporal_df = self._make_valid_frames()
        intraday_df.loc[0, "shape"] = 0.0
        with pytest.raises(ValueError, match="non-positive shape"):
            _validate_outputs(intraday_df, temporal_df)

    def test_wrong_temporal_columns_raises(self):
        """ValueError if temporal columns are wrong."""
        intraday_df, temporal_df = self._make_valid_frames()
        temporal_df = temporal_df.drop(columns=["total_next_day"])
        with pytest.raises(ValueError, match="temporal columns mismatch"):
            _validate_outputs(intraday_df, temporal_df)

    def test_duplicate_temporal_keys_raises(self):
        """ValueError if temporal has duplicate route keys."""
        intraday_df, temporal_df = self._make_valid_frames()
        temporal_df = pd.concat([temporal_df, temporal_df], ignore_index=True)
        with pytest.raises(ValueError, match="temporal output contains duplicate"):
            _validate_outputs(intraday_df, temporal_df)


# ---------------------------------------------------------------------------
# analyze_turnaround_distribution — integration tests
# ---------------------------------------------------------------------------

class TestAnalyzeTurnaroundDistribution:

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
            analyze_turnaround_distribution(cfg)

    def test_creates_output_files(self, tmp_path):
        """Both output CSVs should be created."""
        schedule_path = self._write_schedule_csv(tmp_path, INTRADAY_FLIGHTS)
        cfg = PipelineConfig(
            schedule_file=schedule_path,
            analysis_dir=tmp_path / "analysis",
            output_dir=tmp_path / "output",
        )
        analyze_turnaround_distribution(cfg)
        assert (tmp_path / "analysis" / "scheduled_turnaround_intraday_params.csv").exists()
        assert (tmp_path / "analysis" / "scheduled_turnaround_temporal_profile.csv").exists()

    def test_suffix_in_filenames(self, tmp_path):
        """When suffix is set, it should appear in output filenames."""
        schedule_path = self._write_schedule_csv(tmp_path, INTRADAY_FLIGHTS)
        cfg = PipelineConfig(
            schedule_file=schedule_path,
            analysis_dir=tmp_path / "analysis",
            output_dir=tmp_path / "output",
            suffix="_test",
        )
        analyze_turnaround_distribution(cfg)
        assert (tmp_path / "analysis" / "scheduled_turnaround_intraday_params_test.csv").exists()
        assert (tmp_path / "analysis" / "scheduled_turnaround_temporal_profile_test.csv").exists()

    def test_raises_on_empty_events(self, tmp_path):
        """ValueError when no valid turnaround events can be extracted."""
        # Single flight — no turnaround possible
        flights = [
            _make_flight("AC001", "IBE", "M", "LEMD", "EGLL",
                         "2023-09-01 08:00", "2023-09-01 10:00"),
        ]
        schedule_path = self._write_schedule_csv(tmp_path, flights)
        cfg = PipelineConfig(
            schedule_file=schedule_path,
            analysis_dir=tmp_path / "analysis",
            output_dir=tmp_path / "output",
        )
        with pytest.raises(ValueError, match="No valid turnaround"):
            analyze_turnaround_distribution(cfg)

    def test_output_intraday_csv_has_correct_columns(self, tmp_path):
        """Intraday output CSV should have the canonical columns."""
        schedule_path = self._write_schedule_csv(tmp_path, INTRADAY_FLIGHTS)
        cfg = PipelineConfig(
            schedule_file=schedule_path,
            analysis_dir=tmp_path / "analysis",
            output_dir=tmp_path / "output",
        )
        analyze_turnaround_distribution(cfg)
        df = pd.read_csv(tmp_path / "analysis" / "scheduled_turnaround_intraday_params.csv")
        assert list(df.columns) == ["airline", "wake", "location", "shape"]

    def test_output_shape_positive(self, tmp_path):
        """All shape values in the output CSV must be positive (physical constraint)."""
        schedule_path = self._write_schedule_csv(tmp_path, INTRADAY_FLIGHTS)
        cfg = PipelineConfig(
            schedule_file=schedule_path,
            analysis_dir=tmp_path / "analysis",
            output_dir=tmp_path / "output",
        )
        analyze_turnaround_distribution(cfg)
        df = pd.read_csv(tmp_path / "analysis" / "scheduled_turnaround_intraday_params.csv")
        assert (df["shape"] > 0).all()

    def test_mixed_schedule_produces_both_categories(self, tmp_path):
        """A schedule with intraday and next-day turnarounds should produce both in temporal."""
        schedule_path = self._write_schedule_csv(
            tmp_path, INTRADAY_FLIGHTS + NEXT_DAY_FLIGHTS
        )
        cfg = PipelineConfig(
            schedule_file=schedule_path,
            analysis_dir=tmp_path / "analysis",
            output_dir=tmp_path / "output",
        )
        analyze_turnaround_distribution(cfg)
        temporal = pd.read_csv(
            tmp_path / "analysis" / "scheduled_turnaround_temporal_profile.csv"
        )
        assert temporal["total_intraday"].sum() > 0
        assert temporal["total_next_day"].sum() > 0
