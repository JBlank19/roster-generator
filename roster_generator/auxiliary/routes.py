"""
Route Flight-Time Catalogue

Computes median scheduled and actual flight times for every route present in
the Markov transition tables.  Results are grouped by
(origin, destination, airline, wake) with an operator-agnostic "ALL" fallback.

Output:
  - routes{suffix}.csv
      columns: orig_id, dest_id, airline_id, wake_type, scheduled_time, time
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from roster_generator.config import PipelineConfig
from roster_generator.time_window import DEFAULT_REFTZ, parse_datetime_series_to_reftz

# --- Column aliases ---

AC_REG_COL = "AC_REG"
AIRLINE_COL = "AC_OPER"
AC_WAKE_COL = "AC_WAKE"
DEP_COL = "DEP_ICAO"
ARR_COL = "ARR_ICAO"
STD_COL = "STD_REFTZ"
STA_COL = "STA_REFTZ"
ATD_COL = "ATD_REFTZ"
ATA_COL = "ATA_REFTZ"


# --- Data preparation ---


def _require_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    """Raise a clear error when required columns are missing."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {label}: {missing}")


def _prepare_flights(df: pd.DataFrame, reftz: str = DEFAULT_REFTZ) -> pd.DataFrame:
    """Normalise columns, remap ZZZ airlines, compute flight durations."""
    _require_columns(
        df,
        [AC_REG_COL, AIRLINE_COL, DEP_COL, ARR_COL, STD_COL, STA_COL, ATD_COL, ATA_COL],
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
    if AC_WAKE_COL not in df.columns:
        df[AC_WAKE_COL] = "M"
    df[AC_WAKE_COL] = df[AC_WAKE_COL].fillna("M").astype(str).str.upper().str.strip()

    # Parse times (scheduled + actual)
    for col in [STD_COL, STA_COL, ATD_COL, ATA_COL]:
        df[col] = parse_datetime_series_to_reftz(df[col], reftz)

    df = df.dropna(subset=[STD_COL, STA_COL, ATD_COL, ATA_COL])

    # Compute durations (minutes)
    df["SCHEDULED_FLIGHT_TIME"] = (df[STA_COL] - df[STD_COL]).dt.total_seconds() / 60.0
    df["ACTUAL_FLIGHT_TIME"] = (df[ATA_COL] - df[ATD_COL]).dt.total_seconds() / 60.0

    # Keep only positive durations
    df = df[(df["SCHEDULED_FLIGHT_TIME"] > 0) & (df["ACTUAL_FLIGHT_TIME"] > 0)].copy()

    return df


# --- Route statistics ---

def _build_per_operator_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Median scheduled and actual flight times per (origin, dest, airline, wake)."""
    group_cols = [DEP_COL, ARR_COL, AIRLINE_COL, AC_WAKE_COL]
    stats = df.groupby(group_cols).agg(
        SCHEDULED_FLIGHT_TIME=("SCHEDULED_FLIGHT_TIME", "median"),
        ACTUAL_FLIGHT_TIME=("ACTUAL_FLIGHT_TIME", "median"),
    ).reset_index()
    return stats


def _build_operator_agnostic_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Median flight times with airline_id = 'ALL' (fallback)."""
    group_cols = [DEP_COL, ARR_COL, AC_WAKE_COL]
    stats = df.groupby(group_cols).agg(
        SCHEDULED_FLIGHT_TIME=("SCHEDULED_FLIGHT_TIME", "median"),
        ACTUAL_FLIGHT_TIME=("ACTUAL_FLIGHT_TIME", "median"),
    ).reset_index()
    stats[AIRLINE_COL] = "ALL"
    return stats


def _rename_and_round(df: pd.DataFrame) -> pd.DataFrame:
    """Rename to output schema and round times to integer minutes."""
    df = df.rename(columns={
        DEP_COL: "orig_id",
        ARR_COL: "dest_id",
        AIRLINE_COL: "airline_id",
        AC_WAKE_COL: "wake_type",
        "SCHEDULED_FLIGHT_TIME": "scheduled_time",
        "ACTUAL_FLIGHT_TIME": "time",
    })
    df["scheduled_time"] = df["scheduled_time"].round().astype(int)
    df["time"] = df["time"].round().astype(int)
    return df


# --- Public API ---

def generate_routes(config: PipelineConfig) -> None:
    """Compute median route flight times from the schedule, filtered to Markov routes.

    Only routes present in the Markov transition table are kept.  Both
    per-operator and operator-agnostic (``ALL``) statistics are produced.

    Parameters
    ----------
    config : PipelineConfig
        Paths and parameters for the pipeline.

    Raises
    ------
    FileNotFoundError
        If ``config.schedule_file`` or the markov analysis file do not exist.
    ValueError
        If no flights match the Markov routes.
    """
    print("[Routes] --- ROUTE FLIGHT-TIME CATALOGUE ---")

    markov_path = config.analysis_path("markov")
    output_path = config.output_path("routes")

    if not config.schedule_file.exists():
        raise FileNotFoundError(f"Schedule file not found: {config.schedule_file}")
    if not markov_path.exists():
        raise FileNotFoundError(f"Markov file not found: {markov_path}")

    # 1. Load markov routes
    print(f"[Routes] Markov file: {markov_path}")
    markov_df = pd.read_csv(markov_path)
    routes = markov_df[[DEP_COL, ARR_COL, AC_WAKE_COL]].drop_duplicates()
    print(f"[Routes]   {len(routes)} unique route/wake combinations")

    # 2. Load and prepare schedule
    print(f"[Routes] Schedule: {config.schedule_file}")
    df = pd.read_csv(config.schedule_file)
    df = _prepare_flights(df, reftz=config.reftz)

    if df.empty:
        raise ValueError("No valid flights after normalisation.")

    print(f"[Routes]   {len(df)} valid flights")

    # 3. Filter schedule to markov routes
    merged = df.merge(routes, on=[DEP_COL, ARR_COL, AC_WAKE_COL], how="inner")

    if merged.empty:
        raise ValueError("No flights match the Markov routes.")

    print(f"[Routes]   {len(merged)} flights matching Markov routes")

    # 4. Build statistics
    print("[Routes] Computing median flight times...")
    per_op = _build_per_operator_stats(merged)
    agnostic = _build_operator_agnostic_stats(merged)
    combined = pd.concat([per_op, agnostic], ignore_index=True)
    combined = _rename_and_round(combined)

    # 5. Persist
    config.output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    # Summary
    print(f"[Routes] Saved: {output_path} ({len(combined)} routes)")
    print(f"[Routes]   Wake types: {sorted(combined['wake_type'].unique())}")
    print("[Routes] --- SUCCESS ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Route flight-time catalogue builder")
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
    generate_routes(cfg)
