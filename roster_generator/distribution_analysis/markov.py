"""
Markov Chain Transition Tables

Builds hourly destination-probability tables from historical flight sequences.
Each aircraft's flight chain is grouped by (airline, wake, prev_origin, origin)
and stratified by departure hour to capture time-of-day route patterns.

Two table tiers are produced:
  - Primary:  P(dest | prev_origin, origin, hour)  - memory-aware
  - Fallback: P(dest | origin, hour)               - memoryless

The public entry point ``generate_markov`` orchestrates both the Markov build
and the initial condition sampling (delegated to ``InitialConditionModel``).

Outputs:
  - markov{suffix}.csv (DEP_HOUR_REFTZ)
  - initial_conditions{suffix}.csv
  - phys_ta{suffix}.csv
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd

from roster_generator.config import PipelineConfig
from roster_generator.time_window import (
    DEFAULT_REFTZ,
    DEFAULT_WINDOW_LENGTH_HOURS,
    hour_of_shifted_day,
    minute_of_shifted_day,
    parse_datetime_series_to_reftz,
    shift_series_by_window_start,
)
from .initial_conditions import InitialConditionModel

# --- Column aliases ---

AC_REG_COL = "AC_REG"
AIRLINE_COL = "AC_OPER"
AC_WAKE_COL = "AC_WAKE"
DEP_COL = "DEP_ICAO"
ARR_COL = "ARR_ICAO"
STD_COL = "STD_REFTZ"
STA_COL = "STA_REFTZ"


# --- Helpers ---

def _to_minute_bin_preserve_day(value_mins):
    """Convert to integer minute bin, preserving sign for previous-day values."""
    value = float(value_mins)
    if value < 0:
        return int(np.floor(value))
    return int(round(value))


def _require_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    """Raise a clear error when required columns are missing."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {label}: {missing}")


# --- Data preparation ---

def _prepare_base_flights(
    df,
    airline_filter=None,
    *,
    reftz: str = DEFAULT_REFTZ,
    window_start_mins: int = 0,
    window_length_mins: int = DEFAULT_WINDOW_LENGTH_HOURS * 60,
):
    """Normalise raw schedule columns, apply filters, and drop unusable rows.

    Handles the "ZZZ" sentinel airline (remaps to AC_REG), defaults missing
    wake categories to "M", and removes same-airport flights.
    """
    _require_columns(
        df,
        [AC_REG_COL, AIRLINE_COL, DEP_COL, ARR_COL, STD_COL, STA_COL],
        "schedule",
    )

    # Remap placeholder airline "ZZZ" -> actual registration
    if AIRLINE_COL in df.columns and AC_REG_COL in df.columns:
        zzz_mask = df[AIRLINE_COL].astype(str).str.upper().str.strip() == "ZZZ"
        zzz_count = int(zzz_mask.sum())
        if zzz_count:
            df.loc[zzz_mask, AIRLINE_COL] = df.loc[zzz_mask, AC_REG_COL].astype(str).str.strip()
        print(f"  Remapped {zzz_count} flights: AC_OPER='ZZZ' -> AC_REG")

    for c in [AC_REG_COL, AIRLINE_COL, AC_WAKE_COL, DEP_COL, ARR_COL]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str).str.strip()

    if AIRLINE_COL in df.columns:
        df[AIRLINE_COL] = df[AIRLINE_COL].str.upper()
    if AC_WAKE_COL in df.columns:
        df[AC_WAKE_COL] = df[AC_WAKE_COL].str.upper()

    df["STD"] = parse_datetime_series_to_reftz(df[STD_COL], reftz)
    df["STA"] = parse_datetime_series_to_reftz(df[STA_COL], reftz)
    df = df.dropna(subset=["STD", "STA"])
    df["STD"] = shift_series_by_window_start(df["STD"], window_start_mins)
    df["STA"] = shift_series_by_window_start(df["STA"], window_start_mins)

    if int(window_length_mins) < 24 * 60:
        std_mins = minute_of_shifted_day(df["STD"])
        df = df[(std_mins >= 0) & (std_mins < int(window_length_mins))].copy()
        if df.empty:
            raise ValueError("No usable flights inside configured REFTZ window.")

    df["DEP_HOUR_REFTZ"] = hour_of_shifted_day(df["STD"]).astype(int)

    if airline_filter:
        af = str(airline_filter).strip().upper()
        before = len(df)
        df = df[df[AIRLINE_COL] == af].copy()
        print(f"  Airline filter {af}: {len(df)} rows from {before}")

    df = df[df[DEP_COL] != df[ARR_COL]].copy()
    if df.empty:
        raise ValueError("No usable flights after normalization/filtering.")

    return df


# --- Markov table construction ---

def _build_markov_tables(base_df):
    """Build primary and fallback hourly transition tables.

    Returns:
        final_markov: DataFrame with per-row transition probabilities (for CSV export).
        markov_hourly: nested dict  (op, wake, prev, dep) -> hour -> {arr: count}
        markov_fallback_hourly: nested dict  (op, wake, dep) -> hour -> {arr: count}
    """
    df = base_df.sort_values(by=[AC_REG_COL, "STD"]).reset_index(drop=True).copy()

    # Previous departure airport within each aircraft's chain
    df["PREV_ICAO"] = df.groupby(AC_REG_COL, sort=False)[DEP_COL].shift(1)
    if "DEP_HOUR_REFTZ" not in df.columns:
        df["DEP_HOUR_REFTZ"] = df["STD"].dt.hour

    # --- Primary table: conditioned on previous origin ---
    primary_df = df[df["PREV_ICAO"].notna()].copy()
    markov_grp = (
        primary_df.groupby([AIRLINE_COL, AC_WAKE_COL, "PREV_ICAO", DEP_COL, ARR_COL, "DEP_HOUR_REFTZ"], sort=False)
        .size()
        .reset_index(name="COUNT")
    )

    totals = markov_grp.groupby([AIRLINE_COL, AC_WAKE_COL, "PREV_ICAO", DEP_COL, "DEP_HOUR_REFTZ"])["COUNT"].transform("sum")
    markov_grp["PROB"] = markov_grp["COUNT"] / totals

    final_markov = markov_grp[[
        AIRLINE_COL,
        AC_WAKE_COL,
        "PREV_ICAO",
        DEP_COL,
        ARR_COL,
        "DEP_HOUR_REFTZ",
        "PROB",
        "COUNT",
    ]].copy()

    # Pack into nested dicts for fast runtime lookup
    markov_hourly = {}
    markov_fallback_hourly = {}

    for row in final_markov.itertuples(index=False):
        op = str(row.AC_OPER)
        wake = str(row.AC_WAKE)
        prev = str(row.PREV_ICAO)
        dep = str(row.DEP_ICAO)
        arr = str(row.ARR_ICAO)
        hour = int(row.DEP_HOUR_REFTZ)
        cnt = int(row.COUNT)

        pkey = (op, wake, prev, dep)
        if pkey not in markov_hourly:
            markov_hourly[pkey] = {}
        if hour not in markov_hourly[pkey]:
            markov_hourly[pkey][hour] = {}
        markov_hourly[pkey][hour][arr] = markov_hourly[pkey][hour].get(arr, 0) + cnt

    # --- Fallback table: origin-only (no previous-origin conditioning) ---
    fallback_grp = (
        df.groupby([AIRLINE_COL, AC_WAKE_COL, DEP_COL, ARR_COL, "DEP_HOUR_REFTZ"], sort=False)
        .size()
        .reset_index(name="COUNT")
    )

    for row in fallback_grp.itertuples(index=False):
        op = str(row.AC_OPER)
        wake = str(row.AC_WAKE)
        dep = str(row.DEP_ICAO)
        arr = str(row.ARR_ICAO)
        hour = int(row.DEP_HOUR_REFTZ)
        cnt = int(row.COUNT)

        fkey = (op, wake, dep)
        if fkey not in markov_fallback_hourly:
            markov_fallback_hourly[fkey] = {}
        if hour not in markov_fallback_hourly[fkey]:
            markov_fallback_hourly[fkey][hour] = {}
        markov_fallback_hourly[fkey][hour][arr] = markov_fallback_hourly[fkey][hour].get(arr, 0) + cnt

    return final_markov, markov_hourly, markov_fallback_hourly


# --- Public API ---

def generate_markov(config: PipelineConfig, airline_filter: str | None = None) -> None:
    """Run Markov chain analysis and generate synthetic initial conditions.

    Pipeline:
      1. Load and normalise the schedule.
      2. Build Markov transition tables.
      3. Build empirical initial-condition distributions.
      4. Inject Markov tables into the IC model (needed for first-flight destinations).
      5. Sample one synthetic fleet and write all outputs.

    Parameters
    ----------
    config : PipelineConfig
        Paths and parameters for the pipeline.
    airline_filter : str or None, optional
        Restrict analysis to a single ICAO airline code.

    Raises
    ------
    FileNotFoundError
        If ``config.schedule_file`` does not exist.
    ValueError
        If no usable flights remain after normalization and filtering.
    """
    seed = config.seed
    suffix = config.suffix

    print("[Markov] --- Synthetic Initial Conditions + Continuation ---")
    print(f"[Markov] Using SEED={seed}")
    if airline_filter:
        print(f"[Markov] Filtering for airline: {airline_filter}")

    np.random.seed(seed)
    random.seed(seed)

    if not config.schedule_file.exists():
        raise FileNotFoundError(f"Schedule file not found: {config.schedule_file}")

    df = pd.read_csv(config.schedule_file)
    print(f"[Markov] Schedule: {config.schedule_file}")

    base_df = _prepare_base_flights(
        df,
        airline_filter=airline_filter,
        reftz=config.reftz,
        window_start_mins=config.window_start_mins,
        window_length_mins=config.window_length_mins,
    )
    print(f"[Markov]   {base_df[AC_REG_COL].nunique()} unique aircraft in normalized schedule")

    # Step 1: Markov transitions
    final_markov, markov_hourly, markov_fallback_hourly = _build_markov_tables(base_df)

    # Step 2: Initial conditions (needs Markov tables for destination sampling)
    model = InitialConditionModel(
        base_df,
        seed=seed,
        window_length_mins=config.window_length_mins,
    )
    model.build_all()
    model.set_markov_tables(markov_hourly, markov_fallback_hourly)

    ic_df = model.sample_initial_conditions()

    # Step 3: Persist outputs
    config.analysis_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    initial_conditions_path = config.analysis_path("initial_conditions")
    ic_df.to_csv(initial_conditions_path, index=False)

    markov_path = config.analysis_path("markov")
    final_markov.to_csv(markov_path, index=False)

    phys_ta_path = config.output_path("phys_ta")
    phys_ta_df = model.phys_ta_df.sort_values(["airline_id", "aircraft_wake"]).reset_index(drop=True)
    phys_ta_df.to_csv(phys_ta_path, index=False)

    prior_rows = int(ic_df["PRIOR_STD_REFTZ_MINS"].notna().sum())
    prior_only_rows = int((ic_df["PRIOR_ONLY"].astype(int) == 1).sum())
    single_flights = int(ic_df["SINGLE_FLIGHT"].fillna(0).astype(int).sum())

    print(f"[Markov] Fleet size (synthetic): {len(ic_df)}")
    print(f"[Markov]   Prior rows: {prior_rows}")
    print(f"[Markov]   Prior-only rows: {prior_only_rows}")
    print(f"[Markov]   Single-flight rows: {single_flights}")
    print(f"[Markov] Saved: {initial_conditions_path}")
    print(f"[Markov] Saved: {markov_path} ({len(final_markov)} transitions)")
    print(f"[Markov] Saved: {phys_ta_path} ({len(phys_ta_df)} rows)")
