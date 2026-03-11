"""Empirical distribution builders for synthetic initial conditions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._initial_conditions_types import (
    AC_REG_COL,
    AIRLINE_COL,
    AC_WAKE_COL,
    ARR_COL,
    BIN_SIZE,
    DEP_COL,
    FallbackMarkovTable,
    PrimaryMarkovTable,
    P_NEXT_BIN_SIZE,
    InitialConditionState,
)


def build_all_tables(flights: pd.DataFrame, state: InitialConditionState) -> None:
    """Fit all empirical tables from flight data."""
    _build_first_departure_and_prev_tables(flights, state)
    _build_daily_fleet_gaussian_stats(state)
    _build_origin_distribution(state)
    _build_first_std_distribution(state)
    _build_prior_probabilities(state)
    _build_prior_only_probabilities(flights, state)
    _build_prior_sta_distribution(state)
    _build_scheduled_flight_time_lookup(flights, state)
    _build_turnaround_temporal_probabilities(state)
    _build_phys_ta_min(state)


def set_markov_tables(
    state: InitialConditionState,
    markov_hourly: PrimaryMarkovTable,
    markov_fallback_hourly: FallbackMarkovTable,
) -> None:
    """Inject Markov transition tables and derive backward previous-origin counts."""
    state.markov_hourly = markov_hourly
    state.markov_fallback_hourly = markov_fallback_hourly
    _build_backward_prev_from_markov(state)


def get_phys_ta_min(state: InitialConditionState, airline: str, wake: str) -> int:
    """Lookup minimum turnaround, with global medium-wake fallback."""
    key = (str(airline), str(wake))
    if key in state.phys_ta_min:
        return int(state.phys_ta_min[key])

    fallback_key = ("ALL", "M")
    if fallback_key in state.phys_ta_min:
        return int(state.phys_ta_min[fallback_key])

    raise ValueError(f"Missing phys_ta for {key} and fallback {fallback_key}")


def _build_first_departure_and_prev_tables(flights: pd.DataFrame, state: InitialConditionState) -> None:
    """Link flights to predecessors and extract first departure per aircraft/day."""
    df = flights.sort_values([AC_REG_COL, "STD"]).copy()

    grouped = df.groupby(AC_REG_COL, sort=False)
    df["PREV_STD"] = grouped["STD"].shift(1)
    df["PREV_STA"] = grouped["STA"].shift(1)
    df["PREV_DEP"] = grouped[DEP_COL].shift(1)
    df["PREV_ARR"] = grouped[ARR_COL].shift(1)

    df["DATE_STD"] = df["STD"].dt.normalize()
    first_dep = (
        df.sort_values([AC_REG_COL, "DATE_STD", "STD"])
        .groupby([AC_REG_COL, "DATE_STD"], as_index=False)
        .first()
    )

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

    state.df_with_prev = df
    state.first_dep = first_dep


def _build_daily_fleet_gaussian_stats(state: InitialConditionState) -> None:
    """Fit N(mu, sigma) for daily aircraft count per (airline, wake)."""
    daily = (
        state.first_dep.groupby(["DATE_STD", AIRLINE_COL, AC_WAKE_COL])[AC_REG_COL]
        .nunique()
        .reset_index(name="N")
    )

    all_days = pd.date_range(
        daily["DATE_STD"].min(), daily["DATE_STD"].max(), freq="D"
    )

    for (airline, wake), g in daily.groupby([AIRLINE_COL, AC_WAKE_COL], sort=False):
        counts_by_day = g.set_index("DATE_STD")["N"]
        counts_by_day = counts_by_day.reindex(all_days, fill_value=0)
        vals = counts_by_day.astype(float).to_numpy()
        mu = float(np.mean(vals))
        sigma = float(np.std(vals, ddof=0))
        state.daily_fleet_stats[(str(airline), str(wake))] = (mu, sigma)


def _build_origin_distribution(state: InitialConditionState) -> None:
    """Count first-departure airports per (airline, wake) for weighted sampling."""
    grouped = (
        state.first_dep.groupby([AIRLINE_COL, AC_WAKE_COL, DEP_COL])
        .size()
        .reset_index(name="COUNT")
    )
    for row in grouped.itertuples(index=False):
        key = (str(row.AC_OPER), str(row.AC_WAKE))
        if key not in state.origin_counts:
            state.origin_counts[key] = {}
        state.origin_counts[key][str(row.DEP_ICAO)] = int(row.COUNT)


def _build_first_std_distribution(state: InitialConditionState) -> None:
    """Collect empirical first-STD samples per (airline, wake, origin)."""
    grouped = state.first_dep.groupby([AIRLINE_COL, AC_WAKE_COL, DEP_COL], sort=False)
    for (airline, wake, origin), frame in grouped:
        key = (str(airline), str(wake), str(origin))
        state.first_std_samples[key] = frame["FIRST_STD_MIN_5"].astype(int).tolist()


def _build_prior_probabilities(state: InitialConditionState) -> None:
    """Fraction of first-departures that have an overnight prior."""
    grouped = state.first_dep.groupby([AIRLINE_COL, AC_WAKE_COL], sort=False)
    for (airline, wake), frame in grouped:
        p = float(frame["HAS_PRIOR"].mean()) if len(frame) else 0.0
        state.p_prior[(str(airline), str(wake))] = p


def _build_prior_only_probabilities(flights: pd.DataFrame, state: InitialConditionState) -> None:
    """Fraction of priors where aircraft arrives overnight but never departs day-0."""
    dep_presence = (
        flights.groupby([AC_REG_COL, flights["STD"].dt.normalize()])
        .size()
        .reset_index(name="N")
        .rename(columns={"STD": "DATE"})
    )
    dep_presence["HAS_DEP"] = True

    overnight = flights.copy()
    overnight["ARR_DAY"] = overnight["STA"].dt.normalize()
    overnight["STD_DAY"] = overnight["STD"].dt.normalize()
    overnight = overnight[overnight["STD_DAY"] == (overnight["ARR_DAY"] - pd.Timedelta(days=1))].copy()

    overnight = overnight.merge(
        dep_presence[[AC_REG_COL, "DATE", "HAS_DEP"]],
        left_on=[AC_REG_COL, "ARR_DAY"],
        right_on=[AC_REG_COL, "DATE"],
        how="left",
    )
    prior_only = overnight[overnight["HAS_DEP"] != True].copy()  # noqa: E712

    if not prior_only.empty:
        prior_only = (
            prior_only.sort_values([AC_REG_COL, "ARR_DAY", "STA"])
            .groupby([AC_REG_COL, "ARR_DAY"], as_index=False)
            .last()
        )

    prior_with_dep = (
        state.first_dep[state.first_dep["HAS_PRIOR"]]
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

    combined = prior_with_dep.merge(
        prior_only_counts,
        on=[AIRLINE_COL, AC_WAKE_COL],
        how="outer",
    ).fillna(0)

    state.prior_only_events = prior_only

    for row in combined.itertuples(index=False):
        a = str(row.AC_OPER)
        w = str(row.AC_WAKE)
        n_with_dep = float(row.N_WITH_DEP)
        n_only = float(row.N_ONLY)
        den = n_with_dep + n_only
        if den <= 0:
            state.p_prior_only[(a, w)] = 0.0
        else:
            state.p_prior_only[(a, w)] = float(n_only / den)


def _build_prior_sta_distribution(state: InitialConditionState) -> None:
    """Collect prior-arrival samples for (airline, wake, prev_origin, origin)."""
    with_prior = state.first_dep[state.first_dep["HAS_PRIOR"]].copy()
    grouped = with_prior.groupby([AIRLINE_COL, AC_WAKE_COL, "PREV_DEP", DEP_COL], sort=False)

    for (airline, wake, prev_origin, origin), frame in grouped:
        key = (str(airline), str(wake), str(prev_origin), str(origin))
        state.prior_sta_samples[key] = frame["PRIOR_STA_MIN_5"].astype(int).tolist()


def _build_scheduled_flight_time_lookup(flights: pd.DataFrame, state: InitialConditionState) -> None:
    """Median scheduled block time per (airline, wake, dep, arr) route."""
    df = flights.copy()
    df["SCHED_MIN"] = (df["STA"] - df["STD"]).dt.total_seconds() / 60.0
    df = df[df["SCHED_MIN"] > 0].copy()

    med = (
        df.groupby([AIRLINE_COL, AC_WAKE_COL, DEP_COL, ARR_COL])["SCHED_MIN"]
        .median()
        .reset_index(name="MED")
    )

    for row in med.itertuples(index=False):
        key = (str(row.AC_OPER), str(row.AC_WAKE), str(row.DEP_ICAO), str(row.ARR_ICAO))
        state.flight_time_median[key] = int(round(float(row.MED)))


def _build_turnaround_temporal_probabilities(state: InitialConditionState) -> None:
    """P(next flight is next-day | arrival hour) per (airline, wake)."""
    df = state.df_with_prev.copy()

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

    for (airline, wake), frame in counts.groupby([AIRLINE_COL, AC_WAKE_COL], sort=False):
        state.p_next_hourly[(str(airline), str(wake))] = _hourly_next_day_profile(frame)

    wake_counts = (
        linked.groupby([AC_WAKE_COL, "HOUR_BIN", "CATEGORY"])
        .size()
        .reset_index(name="N")
    )

    for wake, frame in wake_counts.groupby(AC_WAKE_COL, sort=False):
        state.p_next_hourly[("ALL", str(wake))] = _hourly_next_day_profile(frame)


def _build_phys_ta_min(state: InitialConditionState) -> None:
    """5th-percentile intraday turnaround as physical minimum (floored to BIN_SIZE)."""
    df = state.df_with_prev.copy()

    linked = df[
        df["PREV_STA"].notna()
        & (df["PREV_ARR"] == df[DEP_COL])
    ].copy()
    if linked.empty:
        return

    day_gap = (linked["STD"].dt.normalize() - linked["PREV_STA"].dt.normalize()).dt.days
    linked = linked[day_gap == 0].copy()
    if linked.empty:
        return

    linked["TA_MIN"] = (linked["STD"] - linked["PREV_STA"]).dt.total_seconds() / 60.0
    linked = linked[np.isfinite(linked["TA_MIN"]) & (linked["TA_MIN"] > 0)].copy()
    if linked.empty:
        return

    for (airline, wake), frame in linked.groupby([AIRLINE_COL, AC_WAKE_COL], sort=False):
        p5 = float(np.percentile(frame["TA_MIN"].to_numpy(dtype=float), 5))
        ta = int(np.floor(p5 / BIN_SIZE) * BIN_SIZE)
        key = (str(airline), str(wake))
        state.phys_ta_min[key] = ta

    medium = linked[linked[AC_WAKE_COL].astype(str).str.upper() == "M"]
    if not medium.empty:
        p5 = float(np.percentile(medium["TA_MIN"].to_numpy(dtype=float), 5))
        ta = int(np.floor(p5 / BIN_SIZE) * BIN_SIZE)
        state.phys_ta_min[("ALL", "M")] = ta

    out_rows = [
        (airline, wake, int(ta))
        for (airline, wake), ta in state.phys_ta_min.items()
    ]
    state.phys_ta_df = pd.DataFrame(out_rows, columns=["airline_id", "aircraft_wake", "turnaround_time"])


def _build_backward_prev_from_markov(state: InitialConditionState) -> None:
    """Invert Markov transitions to count candidate previous origins by destination."""
    backward_counts = {}
    for (op, wake, prev, dep), hourly in state.markov_hourly.items():
        key = (str(op), str(wake), str(dep))
        if key not in backward_counts:
            backward_counts[key] = {}
        total = 0
        for hour_counts in hourly.values():
            total += int(sum(hour_counts.values()))
        if total > 0:
            backward_counts[key][str(prev)] = backward_counts[key].get(str(prev), 0) + total
    state.backward_prev_counts = backward_counts


def _hourly_next_day_profile(frame: pd.DataFrame) -> dict[int, float]:
    """Build hour -> P(next_day) from grouped intraday/next_day counts."""
    profile: dict[int, float] = {}
    for hour in range(24):
        hour_slice = frame[frame["HOUR_BIN"] == hour]
        n_intra = int(hour_slice[hour_slice["CATEGORY"] == "intraday"]["N"].sum())
        n_next = int(hour_slice[hour_slice["CATEGORY"] == "next_day"]["N"].sum())
        denominator = n_intra + n_next
        if denominator > 0:
            profile[hour] = float(n_next / denominator)
    return profile
