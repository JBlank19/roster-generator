"""
Scheduled Flight Time Distributions (Hourly, REFTZ)

Builds hourly flight-duration probability tables from historical schedule data.
Each route is stratified by departure hour (REFTZ) to capture time-of-day patterns.

Two tiers are produced in a single output file:
  - Per-operator:      P(duration | origin, dest, airline, wake, hour)
  - Operator-agnostic: P(duration | origin, dest, ALL, wake, hour)

Output:
  - scheduled_flight_time{suffix}.csv
      columns: origin_id, dest_id, airline_id, aircraft_wake, dep_hour_reftz,
               flight_time, probability
"""

from __future__ import annotations

import argparse
from pathlib import Path

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

# --- Column aliases & constants ---

AC_REG_COL = "AC_REG"
AIRLINE_COL = "AC_OPER"
DEP_STATION_COL = "DEP_ICAO"
ARR_STATION_COL = "ARR_ICAO"
STD_COL = "STD_REFTZ"
STA_COL = "STA_REFTZ"

BIN_SIZE = 5        # Flight time bin size (minutes)
MIN_SAMPLES = 3     # Minimum samples to create a distribution


def _require_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    """Raise a clear error when required columns are missing."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {label}: {missing}")


# --- Data preparation ---

def _prepare_flights(
    df: pd.DataFrame,
    *,
    reftz: str = DEFAULT_REFTZ,
    window_start_mins: int = 0,
    window_length_mins: int = DEFAULT_WINDOW_LENGTH_HOURS * 60,
) -> pd.DataFrame:
    """Normalise columns, remap ZZZ airlines, compute flight duration and bins."""
    _require_columns(
        df,
        [AC_REG_COL, AIRLINE_COL, DEP_STATION_COL, ARR_STATION_COL, STD_COL, STA_COL],
        "schedule",
    )

    # Remap placeholder airline "ZZZ" -> actual registration
    if AIRLINE_COL in df.columns and AC_REG_COL in df.columns:
        zzz_mask = df[AIRLINE_COL].astype(str).str.upper().str.strip() == "ZZZ"
        zzz_count = int(zzz_mask.sum())
        if zzz_count:
            df.loc[zzz_mask, AIRLINE_COL] = df.loc[zzz_mask, AC_REG_COL].astype(str).str.strip()
        print(f"  Remapped {zzz_count} flights: AC_OPER='ZZZ' -> AC_REG")

    # Default missing wake to M
    if "AC_WAKE" not in df.columns:
        df["AC_WAKE"] = "M"
    df["AC_WAKE"] = df["AC_WAKE"].fillna("M").astype(str).str.upper().str.strip()

    # Parse times in REFTZ and shift so window start maps to minute 0.
    df[STD_COL] = parse_datetime_series_to_reftz(df[STD_COL], reftz)
    df[STA_COL] = parse_datetime_series_to_reftz(df[STA_COL], reftz)
    df = df.dropna(subset=[STD_COL, STA_COL])
    df["STD_SHIFTED"] = shift_series_by_window_start(df[STD_COL], window_start_mins)
    df["STA_SHIFTED"] = shift_series_by_window_start(df[STA_COL], window_start_mins)

    if int(window_length_mins) < 24 * 60:
        dep_mins = minute_of_shifted_day(df["STD_SHIFTED"])
        df = df[(dep_mins >= 0) & (dep_mins < int(window_length_mins))].copy()
        if df.empty:
            return df

    # Flight duration
    df["FLIGHT_MINUTES"] = (df["STA_SHIFTED"] - df["STD_SHIFTED"]).dt.total_seconds() / 60.0
    df = df[df["FLIGHT_MINUTES"] > 0]

    # Departure hour and duration bin
    df["DEP_HOUR_REFTZ"] = hour_of_shifted_day(df["STD_SHIFTED"]).astype(int)
    df["FLIGHT_BIN"] = (np.round(df["FLIGHT_MINUTES"] / BIN_SIZE) * BIN_SIZE).astype(int)

    return df


# --- Distribution builders ---

def _build_hourly_distributions(df: pd.DataFrame) -> pd.DataFrame:
    """Per-operator: P(duration | origin, dest, operator, wake, hour)."""
    group_cols = [DEP_STATION_COL, ARR_STATION_COL, AIRLINE_COL, "AC_WAKE", "DEP_HOUR_REFTZ"]

    counts = df.groupby(group_cols + ["FLIGHT_BIN"]).size().reset_index(name="COUNT")
    counts["TOTAL"] = counts.groupby(group_cols)["COUNT"].transform("sum")
    counts = counts[counts["TOTAL"] >= MIN_SAMPLES].copy()
    counts["probability"] = counts["COUNT"] / counts["TOTAL"]

    return counts.rename(columns={
        DEP_STATION_COL: "origin_id",
        ARR_STATION_COL: "dest_id",
        AIRLINE_COL: "airline_id",
        "AC_WAKE": "aircraft_wake",
        "DEP_HOUR_REFTZ": "dep_hour_reftz",
        "FLIGHT_BIN": "flight_time",
    }).drop(columns=["COUNT", "TOTAL"])


def _build_operator_agnostic_distributions(df: pd.DataFrame) -> pd.DataFrame:
    """Operator-agnostic: P(duration | origin, dest, ALL, wake, hour)."""
    group_cols = [DEP_STATION_COL, ARR_STATION_COL, "AC_WAKE", "DEP_HOUR_REFTZ"]

    counts = df.groupby(group_cols + ["FLIGHT_BIN"]).size().reset_index(name="COUNT")
    counts["TOTAL"] = counts.groupby(group_cols)["COUNT"].transform("sum")
    counts = counts[counts["TOTAL"] >= MIN_SAMPLES].copy()
    counts["probability"] = counts["COUNT"] / counts["TOTAL"]
    counts[AIRLINE_COL] = "ALL"

    return counts.rename(columns={
        DEP_STATION_COL: "origin_id",
        ARR_STATION_COL: "dest_id",
        AIRLINE_COL: "airline_id",
        "AC_WAKE": "aircraft_wake",
        "DEP_HOUR_REFTZ": "dep_hour_reftz",
        "FLIGHT_BIN": "flight_time",
    }).drop(columns=["COUNT", "TOTAL"])


# --- Public API ---

def analyze_flight_time_distribution(config: PipelineConfig) -> None:
    """Run hourly flight-time distribution analysis.

    Parameters
    ----------
    config : PipelineConfig
        Paths and parameters for the pipeline.

    Raises
    ------
    FileNotFoundError
        If ``config.schedule_file`` does not exist.
    ValueError
        If no valid flights remain after normalisation.
    """
    print("[FlightTime] --- HOURLY FLIGHT TIME ANALYSIS ---")

    output_path = config.analysis_path("scheduled_flight_time")

    if not config.schedule_file.exists():
        raise FileNotFoundError(f"Schedule file not found: {config.schedule_file}")

    print(f"[FlightTime] Schedule: {config.schedule_file}")
    df = pd.read_csv(config.schedule_file)
    df = _prepare_flights(
        df,
        reftz=config.reftz,
        window_start_mins=config.window_start_mins,
        window_length_mins=config.window_length_mins,
    )

    if df.empty:
        raise ValueError("No valid flights after normalisation.")

    print(f"[FlightTime]   {len(df)} valid flights")

    # Per-operator + operator-agnostic hourly distributions
    print("[FlightTime] Computing hourly distributions...")
    hourly_df = _build_hourly_distributions(df)
    all_operator_df = _build_operator_agnostic_distributions(df)
    combined = pd.concat([hourly_df, all_operator_df], ignore_index=True)

    # Persist
    config.analysis_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    # Summary
    n_routes = combined.groupby(["origin_id", "dest_id"]).ngroups
    total_routes = df.groupby([DEP_STATION_COL, ARR_STATION_COL]).ngroups

    print(f"[FlightTime] Saved: {output_path} ({len(combined)} rows, {n_routes} routes)")
    print(f"[FlightTime]   Coverage: {n_routes} / {total_routes} total routes")
    print("[FlightTime] --- SUCCESS ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hourly flight-time distribution builder")
    parser.add_argument("--schedule", type=str, required=True, help="Path to schedule CSV")
    parser.add_argument("--analysis-dir", type=str, required=True, help="Analysis output directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Final output directory (defaults to analysis-dir)")
    parser.add_argument("--suffix", type=str, default="", help="Suffix for output filenames")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    cfg = PipelineConfig(
        schedule_file=Path(args.schedule),
        analysis_dir=Path(args.analysis_dir),
        output_dir=Path(args.output_dir or args.analysis_dir),
        seed=args.seed,
        suffix=f"_{args.suffix}" if args.suffix else "",
    )
    analyze_flight_time_distribution(cfg)
