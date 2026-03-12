"""Targeted tests for InitialConditionModel refactor invariants."""

import numpy as np
import pandas as pd
import pytest

from roster_generator.distribution_analysis.initial_conditions import InitialConditionModel
from roster_generator.distribution_analysis.markov import (
    AC_REG_COL,
    AC_WAKE_COL,
    AIRLINE_COL,
    ARR_COL,
    DEP_COL,
    STA_COL,
    STD_COL,
    _build_markov_tables,
    _prepare_base_flights,
)


def _make_flight(reg, airline, wake, dep, arr, std, sta):
    return {
        AC_REG_COL: reg,
        AIRLINE_COL: airline,
        AC_WAKE_COL: wake,
        DEP_COL: dep,
        ARR_COL: arr,
        STD_COL: std,
        STA_COL: sta,
    }


def _make_two_day_schedule():
    return [
        _make_flight("AC001", "IBE", "M", "LEMD", "EGLL", "2023-09-01 08:00", "2023-09-01 10:00"),
        _make_flight("AC001", "IBE", "M", "EGLL", "LFPG", "2023-09-01 12:00", "2023-09-01 13:30"),
        _make_flight("AC001", "IBE", "M", "LFPG", "LEMD", "2023-09-01 16:00", "2023-09-01 18:00"),
        _make_flight("AC002", "IBE", "M", "EGLL", "LFPG", "2023-09-01 09:00", "2023-09-01 10:30"),
        _make_flight("AC002", "IBE", "M", "LFPG", "LEMD", "2023-09-01 14:00", "2023-09-01 16:00"),
        _make_flight("AC001", "IBE", "M", "LEMD", "EGLL", "2023-09-02 08:00", "2023-09-02 10:00"),
        _make_flight("AC001", "IBE", "M", "EGLL", "LFPG", "2023-09-02 12:00", "2023-09-02 13:30"),
        _make_flight("AC001", "IBE", "M", "LFPG", "LEMD", "2023-09-02 16:00", "2023-09-02 18:00"),
        _make_flight("AC002", "IBE", "M", "EGLL", "LFPG", "2023-09-02 09:00", "2023-09-02 10:30"),
        _make_flight("AC002", "IBE", "M", "LFPG", "LEMD", "2023-09-02 14:00", "2023-09-02 16:00"),
    ]


@pytest.fixture
def prepared_df():
    return _prepare_base_flights(pd.DataFrame(_make_two_day_schedule()))


@pytest.fixture
def markov_tables(prepared_df):
    _, markov_hourly, markov_fallback_hourly = _build_markov_tables(prepared_df)
    return markov_hourly, markov_fallback_hourly


def _build_model(prepared_df, markov_hourly, markov_fallback_hourly, seed=42):
    model = InitialConditionModel(prepared_df, seed=seed)
    model.build_all()
    model.set_markov_tables(markov_hourly, markov_fallback_hourly)
    return model


def test_build_all_populates_required_tables(prepared_df):
    model = InitialConditionModel(prepared_df, seed=42)
    model.build_all()

    assert not model.first_dep.empty
    assert model.daily_fleet_stats
    assert model.origin_counts
    assert model.first_std_samples
    assert model.p_prior
    assert model.p_next_hourly
    assert model.phys_ta_min
    assert list(model._phys_ta_df.columns) == ["airline_id", "aircraft_wake", "turnaround_time"]


def test_set_markov_tables_populates_backward_prev_counts(prepared_df, markov_tables):
    markov_hourly, markov_fallback_hourly = markov_tables
    model = InitialConditionModel(prepared_df, seed=42)
    model.build_all()
    model.set_markov_tables(markov_hourly, markov_fallback_hourly)

    assert model.backward_prev_counts
    assert any(model.backward_prev_counts.values())


def test_sample_initial_conditions_schema_and_unique_regs(prepared_df, markov_tables):
    markov_hourly, markov_fallback_hourly = markov_tables
    model = _build_model(prepared_df, markov_hourly, markov_fallback_hourly, seed=42)

    np.random.seed(42)
    ic_df = model.sample_initial_conditions()

    expected_cols = [
        "AC_REG",
        "AC_OPER",
        "AC_WAKE",
        "PRIOR_ONLY",
        "ORIGIN",
        "DEST",
        "STD_REFTZ_MINS",
        "STA_REFTZ_MINS",
        "SINGLE_FLIGHT",
        "PRIOR_ORIGIN",
        "PRIOR_DEST",
        "PRIOR_STD_REFTZ_MINS",
        "PRIOR_STA_REFTZ_MINS",
    ]
    assert list(ic_df.columns) == expected_cols
    assert not ic_df["AC_REG"].duplicated().any()


def test_prior_std_negative_when_present(prepared_df, markov_tables):
    markov_hourly, markov_fallback_hourly = markov_tables
    model = _build_model(prepared_df, markov_hourly, markov_fallback_hourly, seed=11)

    np.random.seed(11)
    ic_df = model.sample_initial_conditions()

    prior_std = ic_df["PRIOR_STD_REFTZ_MINS"].dropna().astype(float)
    if not prior_std.empty:
        assert (prior_std < 0).all()


def test_prior_only_rows_have_blank_or_null_first_flight(prepared_df, markov_tables):
    markov_hourly, markov_fallback_hourly = markov_tables
    model = _build_model(prepared_df, markov_hourly, markov_fallback_hourly, seed=19)

    np.random.seed(19)
    ic_df = model.sample_initial_conditions()

    prior_only = ic_df["PRIOR_ONLY"].astype(int) == 1
    if prior_only.any():
        assert (ic_df.loc[prior_only, "ORIGIN"].astype(str) == "").all()
        assert (ic_df.loc[prior_only, "DEST"].astype(str) == "").all()
        assert ic_df.loc[prior_only, "STD_REFTZ_MINS"].isna().all()
        assert ic_df.loc[prior_only, "STA_REFTZ_MINS"].isna().all()
        assert (ic_df.loc[prior_only, "SINGLE_FLIGHT"].astype(int) == 0).all()


def test_turnaround_constraint_holds(prepared_df, markov_tables):
    markov_hourly, markov_fallback_hourly = markov_tables
    model = _build_model(prepared_df, markov_hourly, markov_fallback_hourly, seed=23)

    np.random.seed(23)
    ic_df = model.sample_initial_conditions()

    rows = ic_df[
        (ic_df["PRIOR_STD_REFTZ_MINS"].notna())
        & (ic_df["PRIOR_ONLY"].astype(int) != 1)
    ]
    for row in rows.itertuples(index=False):
        min_ta = model._get_phys_ta_min(str(row.AC_OPER), str(row.AC_WAKE))
        assert int(row.STD_REFTZ_MINS) - int(row.PRIOR_STA_REFTZ_MINS) >= min_ta


def test_markov_fallback_path_works_with_empty_primary(prepared_df, markov_tables):
    _, markov_fallback_hourly = markov_tables
    model = InitialConditionModel(prepared_df, seed=31)
    model.build_all()
    model.set_markov_tables({}, markov_fallback_hourly)

    np.random.seed(31)
    ic_df = model.sample_initial_conditions()

    non_prior_only = ic_df["PRIOR_ONLY"].astype(int) != 1
    if non_prior_only.any():
        assert (ic_df.loc[non_prior_only, "DEST"].astype(str).str.strip() != "").all()


def test_seeded_determinism_smoke(prepared_df, markov_tables):
    markov_hourly, markov_fallback_hourly = markov_tables

    model_a = _build_model(prepared_df, markov_hourly, markov_fallback_hourly, seed=101)
    np.random.seed(101)
    ic_a = model_a.sample_initial_conditions()

    model_b = _build_model(prepared_df, markov_hourly, markov_fallback_hourly, seed=101)
    np.random.seed(101)
    ic_b = model_b.sample_initial_conditions()

    pd.testing.assert_frame_equal(ic_a, ic_b)
