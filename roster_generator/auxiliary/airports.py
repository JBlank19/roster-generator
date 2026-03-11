"""
Airport Catalogue

Extracts the unique set of airports that appear in the Markov transition
table (as departure or arrival) and enriches each entry with:

  - Rolling capacity (rolling_capacity): maximum observed 60-minute
    movement window across the full schedule period.

  - Burst capacity (burst_capacity): maximum observed 5-minute movement
    window across the full schedule period.

Output:
  - airports{suffix}.csv
      columns: airport_id, rolling_capacity, burst_capacity
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from roster_generator.config import PipelineConfig

# --- Column aliases ---

DEP_COL = "DEP_ICAO"
ARR_COL = "ARR_ICAO"
STD_COL = "STD_REFTZ"
STA_COL = "STA_REFTZ"
AIRLINE_COL = "AC_OPER"
AC_REG_COL = "AC_REG"



# --- Capacity computation ---

def _compute_capacities(schedule_df: pd.DataFrame, airports: list[str]) -> pd.DataFrame:
    """Compute maximum rolling 60-minute and 5-minute movement capacities.

    Both departures and arrivals are combined into a single movement flow.
    Capacities are derived from the worst observed day across the entire
    schedule period.

    Parameters
    ----------
    schedule_df : pandas.DataFrame
        Full cleaned schedule with ``STD_REFTZ`` and ``STA_REFTZ`` columns.
    airports : list of str
        ICAO codes for which capacities should be computed.

    Returns
    -------
    pandas.DataFrame
        Columns: ``airport_id``, ``rolling_capacity``, ``burst_capacity``.
        Airports with no observed traffic receive a floor value of 1 for
        both columns.
    """
    print("[Airports] Computing capacities (max rolling 60 min & 5 min)...")

    # Merge departures and arrivals into one movement stream
    deps = schedule_df[[DEP_COL, STD_COL]].dropna().copy()
    deps.columns = ["airport", "ts"]
    arrs = schedule_df[[ARR_COL, STA_COL]].dropna().copy()
    arrs.columns = ["airport", "ts"]

    all_moves = pd.concat([deps, arrs], ignore_index=True)
    all_moves["ts"] = pd.to_datetime(all_moves["ts"], format="mixed", errors="coerce")
    all_moves.dropna(subset=["ts"], inplace=True)

    # Restrict to airports of interest
    all_moves = all_moves[all_moves["airport"].isin(set(airports))]

    if all_moves.empty:
        return pd.DataFrame({
            "airport_id": airports,
            "rolling_capacity": 1,
            "burst_capacity": 1,
        })

    # Bin into 5-minute slots within each day (288 slots per day)
    all_moves["date"] = all_moves["ts"].dt.date
    all_moves["bin5"] = (
        all_moves["ts"].dt.hour * 12 + all_moves["ts"].dt.minute // 5
    )

    # Count movements per (airport, date, bin5)
    counts = all_moves.groupby(["airport", "date", "bin5"], sort=False).size()

    # Wide matrix: index = (airport, date), columns = bin5 [0..287]
    daily_matrix = counts.unstack("bin5", fill_value=0)
    daily_matrix = daily_matrix.reindex(columns=range(288), fill_value=0)

    # Burst capacity: max single 5-min bin per airport
    burst_caps = daily_matrix.max(axis=1).groupby("airport").max()

    # Rolling capacity: max 60-min rolling sum (window = 12 x 5-min bins)
    rolling_max = (
        daily_matrix.T
        .rolling(window=12, min_periods=12)
        .sum()
        .T
        .fillna(0)
        .max(axis=1)
        .groupby("airport")
        .max()
    )

    capacity_df = pd.DataFrame({
        "rolling_capacity": rolling_max,
        "burst_capacity": burst_caps,
    }).reset_index().rename(columns={"airport": "airport_id"})

    # Left-join against the full airport list; fill missing with floor = 1
    result = pd.DataFrame({"airport_id": airports}).merge(
        capacity_df, on="airport_id", how="left"
    )
    result["rolling_capacity"] = result["rolling_capacity"].fillna(1).astype(int).clip(lower=1)
    result["burst_capacity"] = result["burst_capacity"].fillna(1).astype(int).clip(lower=1)

    top5 = result.nlargest(5, "rolling_capacity")
    print(
        f"[Airports]   Computed capacities for {len(result)} airports. "
        f"Top 5 by rolling cap: "
        f"{top5[['airport_id', 'rolling_capacity', 'burst_capacity']].values.tolist()}"
    )

    return result


def _require_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    """Raise a clear error when required columns are missing."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {label}: {missing}")


# --- Public API ---

def generate_airports(config: PipelineConfig) -> None:
    """Extract the airport catalogue from the Markov table and schedule.

    Steps:

    1. Load the Markov CSV and collect all unique departure/arrival airport
       codes.
    2. If the schedule file is available, compute curfew hours and movement
       capacities for each airport.
    3. Save the result to ``output_dir/airports{suffix}.csv``.

    Parameters
    ----------
    config : PipelineConfig
        Paths and parameters for the pipeline.

    Raises
    ------
    FileNotFoundError
        If the Markov analysis file does not exist.
    """
    print("[Airports] --- AIRPORT CATALOGUE ---")

    markov_path = config.analysis_path("markov")
    output_path = config.output_path("airports")

    if not markov_path.exists():
        raise FileNotFoundError(f"[Airports] Markov file not found: {markov_path}")

    # 1. Extract unique airports from the Markov table
    print(f"[Airports] Markov file: {markov_path}")
    markov_df = pd.read_csv(markov_path)

    dep_airports = set(markov_df[DEP_COL].dropna().unique())
    arr_airports = set(markov_df[ARR_COL].dropna().unique())
    all_airports = sorted(dep_airports | arr_airports)

    print(f"[Airports]   {len(all_airports)} unique airports found in Markov table")

    airports_df = pd.DataFrame({"airport_id": all_airports})

    # 2. Enrich with curfews and capacities from the schedule
    if config.schedule_file.exists():
        print(f"[Airports] Schedule: {config.schedule_file}")
        schedule_df = pd.read_csv(config.schedule_file)
        _require_columns(schedule_df, [DEP_COL, ARR_COL, STD_COL, STA_COL], "schedule")

        # Capacities
        capacity_df = _compute_capacities(schedule_df, all_airports)
        airports_df = airports_df.merge(capacity_df, on="airport_id", how="left")
        airports_df["rolling_capacity"] = (
            airports_df["rolling_capacity"].fillna(1).astype(int).clip(lower=1)
        )
        airports_df["burst_capacity"] = (
            airports_df["burst_capacity"].fillna(1).astype(int).clip(lower=1)
        )
    else:
        print(
            f"[Airports] Warning: schedule file not found ({config.schedule_file}). "
            "Skipping capacity computation."
        )
        airports_df["rolling_capacity"] = 1
        airports_df["burst_capacity"] = 1

    # 3. Persist
    config.output_dir.mkdir(parents=True, exist_ok=True)
    airports_df.to_csv(output_path, index=False)

    print(f"[Airports] Saved: {output_path} ({len(airports_df)} airports)")
    print("[Airports] --- SUCCESS ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Airport catalogue builder")
    parser.add_argument("--schedule", type=str, required=True, help="Path to schedule CSV")
    parser.add_argument("--analysis-dir", type=str, required=True, help="Analysis output directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Final output directory")
    parser.add_argument("--suffix", type=str, default="", help="Suffix for output filenames")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    cfg = PipelineConfig(
        schedule_file=Path(args.schedule),
        analysis_dir=Path(args.analysis_dir),
        output_dir=Path(args.output_dir),
        seed=args.seed,
        suffix=f"_{args.suffix}" if args.suffix else "",
    )
    generate_airports(cfg)
