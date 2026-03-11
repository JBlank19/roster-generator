"""
Route-aware scheduled turnaround parameter fitter.

Builds route/wake turnaround parameters from real linked turnaround events and writes:
  - scheduled_turnaround_intraday_params{suffix}.csv
      columns: airline,wake,location,shape
      (Lognormal parameters in log-space)

  - scheduled_turnaround_temporal_profile{suffix}.csv
      columns: airline,previous_origin,origin,wake,intraday_sparse,next_day_sparse,total_intraday,total_next_day
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from roster_generator.config import PipelineConfig

# --- Column aliases & constants ---

AC_REG_COL = "AC_REG"
AIRLINE_COL = "AC_OPER"
DEP_STATION_COL = "DEP_ICAO"
ARR_STATION_COL = "ARR_ICAO"
STD_COL = "STD_REFTZ"
STA_COL = "STA_REFTZ"

BIN_SIZE = 5
MINUTES_PER_DAY = 24 * 60
BIN_EDGES = np.arange(0, MINUTES_PER_DAY + BIN_SIZE, BIN_SIZE)

INTRADAY_CATEGORY = "intraday"
NEXT_DAY_CATEGORY = "next_day"


def _require_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    """Raise a clear error when required columns are missing."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {label}: {missing}")


# --- Helpers ---

def _safe_shape(value: float, minimum: float = 0.05) -> float:
    """Clamp shape parameter to a finite lower bound."""
    if not np.isfinite(value):
        return float(minimum)
    return float(max(value, minimum))


def _fit_lognormal_params(values: np.ndarray) -> Tuple[float, float]:
    """Fit lognormal parameters in log-space with deterministic tiny-sample fallbacks."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x > 0]
    if x.size == 0:
        raise ValueError("Cannot fit lognormal with zero samples.")

    lx = np.log(np.maximum(x, 1e-6))
    mu_ln = float(np.mean(lx))
    if lx.size == 1:
        sigma_ln = 0.12
    elif lx.size == 2:
        sigma_ln = float(np.std(lx, ddof=0))
    else:
        sigma_ln = float(np.std(lx, ddof=1))

    sigma_ln = _safe_shape(sigma_ln, minimum=0.05)
    return mu_ln, sigma_ln


def _encode_sparse_hist(hist: np.ndarray) -> str:
    """Serialize a 5-min histogram as 'minute:count;minute:count;...'."""
    parts = []
    for idx, count in enumerate(hist.tolist()):
        c = int(count)
        if c > 0:
            minute = idx * BIN_SIZE
            parts.append(f"{minute}:{c}")
    return ";".join(parts)



# --- Data preparation ---

def _prepare_turnaround_events(df: pd.DataFrame) -> pd.DataFrame:
    """Build linked turnaround events and keep day_gap >= 0."""
    _require_columns(
        df,
        [AC_REG_COL, AIRLINE_COL, DEP_STATION_COL, ARR_STATION_COL, STD_COL, STA_COL],
        "schedule",
    )

    if AIRLINE_COL in df.columns and AC_REG_COL in df.columns:
        zzz_mask = df[AIRLINE_COL].astype(str).str.upper().str.strip() == "ZZZ"
        zzz_count = int(zzz_mask.sum())
        if zzz_count:
            df.loc[zzz_mask, AIRLINE_COL] = df.loc[zzz_mask, AC_REG_COL].astype(str).str.strip()
        print(f"  Remapped {zzz_count} flights: AC_OPER='ZZZ' -> AC_REG")

    if "AC_WAKE" not in df.columns:
        df["AC_WAKE"] = "M"
    df["AC_WAKE"] = df["AC_WAKE"].fillna("M").astype(str).str.upper().str.strip()

    df[STD_COL] = pd.to_datetime(df[STD_COL], errors="coerce")
    df[STA_COL] = pd.to_datetime(df[STA_COL], errors="coerce")
    df = df.dropna(subset=[STD_COL, STA_COL])

    df = df.sort_values(by=[AC_REG_COL, STD_COL])
    grouped = df.groupby(AC_REG_COL, sort=False)
    df["PREV_STA"] = grouped[STA_COL].shift(1)
    df["PREV_ARR_STATION"] = grouped[ARR_STATION_COL].shift(1)
    # Previous departure airport = where the aircraft came from before the turnaround.
    df["PREV_DEP_STATION"] = grouped[DEP_STATION_COL].shift(1)

    df = df.dropna(subset=["PREV_STA", "PREV_ARR_STATION", "PREV_DEP_STATION"])
    df = df[df["PREV_ARR_STATION"] == df[DEP_STATION_COL]]

    df["TA_MINUTES"] = (df[STD_COL] - df["PREV_STA"]).dt.total_seconds() / 60.0
    df = df[np.isfinite(df["TA_MINUTES"])]
    df = df[df["TA_MINUTES"] >= 0]

    df["DAY_GAP"] = (df[STD_COL].dt.normalize() - df["PREV_STA"].dt.normalize()).dt.days
    df = df[np.isfinite(df["DAY_GAP"])]
    df = df[df["DAY_GAP"].isin([0, 1])]  # Exclude multi-day layovers (maintenance, AOG, etc.)

    df["CATEGORY"] = np.where(df["DAY_GAP"] == 0, INTRADAY_CATEGORY, NEXT_DAY_CATEGORY)

    # Temporal selector uses arrival time (PREV_STA) in 5-minute bins.
    df["ARR_MINUTE_BIN"] = (
        ((df["PREV_STA"].dt.hour * 60 + df["PREV_STA"].dt.minute) // BIN_SIZE) * BIN_SIZE
    ).astype(int)
    df["ARR_MINUTE_BIN"] = df["ARR_MINUTE_BIN"] % MINUTES_PER_DAY

    out = pd.DataFrame(
        {
            "airline": df[AIRLINE_COL].astype(str),
            "previous_origin": df["PREV_DEP_STATION"].astype(str),
            "origin": df[DEP_STATION_COL].astype(str),
            "wake": df["AC_WAKE"].astype(str),
            "ta_minutes": df["TA_MINUTES"].astype(float),
            "category": df["CATEGORY"].astype(str),
            "arr_minute_bin": df["ARR_MINUTE_BIN"].astype(int),
        }
    )
    return out


# --- Parameter fitting ---

def _iter_group_frames(events: pd.DataFrame) -> Iterable[Tuple[Tuple[str, str, str, str], pd.DataFrame]]:
    """Yield (key, frame) for exact (airline, previous_origin, origin, wake)."""
    for key, g in events.groupby(["airline", "previous_origin", "origin", "wake"], sort=False):
        yield (str(key[0]), str(key[1]), str(key[2]), str(key[3])), g


def _build_param_and_temporal_rows(
    events: pd.DataFrame,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Fit lognormal parameters per (airline, wake) and build temporal histograms per route."""
    intraday_rows: List[Dict[str, object]] = []
    temporal_rows: List[Dict[str, object]] = []

    seen_temporal = set()

    for key, g in _iter_group_frames(events):
        airline, previous_origin, origin, wake = key

        intra_bins = g.loc[g["category"] == INTRADAY_CATEGORY, "arr_minute_bin"].to_numpy(dtype=float)
        next_bins = g.loc[g["category"] == NEXT_DAY_CATEGORY, "arr_minute_bin"].to_numpy(dtype=float)

        # Temporal profile rows (always emitted for every key-level group)
        key_temporal = (airline, previous_origin, origin, wake)
        if key_temporal in seen_temporal:
            continue
        seen_temporal.add(key_temporal)

        hist_intra, _ = np.histogram(intra_bins, bins=BIN_EDGES)
        hist_next, _ = np.histogram(next_bins, bins=BIN_EDGES)
        temporal_rows.append(
            {
                "airline": airline,
                "previous_origin": previous_origin,
                "origin": origin,
                "wake": wake,
                "intraday_sparse": _encode_sparse_hist(hist_intra),
                "next_day_sparse": _encode_sparse_hist(hist_next),
                "total_intraday": int(np.sum(hist_intra)),
                "total_next_day": int(np.sum(hist_next)),
            }
        )

    # Intraday params: keyed by (airline, wake) — aggregate all routes.
    for (airline, wake), g in events.groupby(["airline", "wake"], sort=False):
        intra_vals = g.loc[g["category"] == INTRADAY_CATEGORY, "ta_minutes"].to_numpy(dtype=float)
        if intra_vals.size > 0:
            loc_i, shape_i = _fit_lognormal_params(intra_vals)
            intraday_rows.append(
                {
                    "airline": airline,
                    "wake": wake,
                    "location": float(loc_i),
                    "shape": float(shape_i),
                }
            )

    return intraday_rows, temporal_rows


# --- Validation ---

def _validate_outputs(
    intraday_df: pd.DataFrame,
    temporal_df: pd.DataFrame,
) -> None:
    """Sanity checks on output DataFrames: column names, duplicates, and finite params."""
    if list(intraday_df.columns) != ["airline", "wake", "location", "shape"]:
        raise ValueError(f"intraday columns mismatch: {list(intraday_df.columns)}")
    if intraday_df.duplicated(subset=["airline", "wake"]).any():
        raise ValueError("intraday output contains duplicate keys.")
    if intraday_df[["location", "shape"]].isna().any().any():
        raise ValueError("intraday output contains NaN params.")
    if (~np.isfinite(intraday_df[["location", "shape"]].to_numpy(dtype=float))).any():
        raise ValueError("intraday output contains non-finite params.")
    if (intraday_df["shape"].astype(float) <= 0).any():
        raise ValueError("intraday output contains non-positive shape.")

    expected_temporal_cols = [
        "airline",
        "previous_origin",
        "origin",
        "wake",
        "intraday_sparse",
        "next_day_sparse",
        "total_intraday",
        "total_next_day",
    ]
    if list(temporal_df.columns) != expected_temporal_cols:
        raise ValueError(f"temporal columns mismatch: {list(temporal_df.columns)}")
    if temporal_df.duplicated(subset=["airline", "previous_origin", "origin", "wake"]).any():
        raise ValueError("temporal output contains duplicate keys.")


# --- Public API ---

def analyze_turnaround_distribution(config: PipelineConfig) -> None:
    """Run route-aware parametric turnaround analysis.

    Parameters
    ----------
    config : PipelineConfig
        Paths and parameters for the pipeline.

    Raises
    ------
    FileNotFoundError
        If ``config.schedule_file`` does not exist.
    ValueError
        If no valid turnaround events remain after filtering.
    """
    print("[Turnaround] --- ROUTE-AWARE PARAMETRIC TURNAROUND ANALYSIS ---")

    intraday_path = config.analysis_path("scheduled_turnaround_intraday_params")
    temporal_path = config.analysis_path("scheduled_turnaround_temporal_profile")

    if not config.schedule_file.exists():
        raise FileNotFoundError(f"Schedule file not found: {config.schedule_file}")

    print(f"[Turnaround] Schedule: {config.schedule_file}")
    df = pd.read_csv(config.schedule_file)
    events = _prepare_turnaround_events(df)

    print(f"[Turnaround]   Valid turnaround events: {len(events)}")
    cat_counts = events["category"].value_counts().to_dict()
    print(
        "[Turnaround]   Categories: "
        f"intraday={cat_counts.get(INTRADAY_CATEGORY, 0)}, "
        f"next_day={cat_counts.get(NEXT_DAY_CATEGORY, 0)}"
    )
    if events.empty:
        raise ValueError("No valid turnaround events found.")

    intraday_rows, temporal_rows = _build_param_and_temporal_rows(events)

    intraday_df = pd.DataFrame(
        intraday_rows,
        columns=["airline", "wake", "location", "shape"],
    ).sort_values(["airline", "wake"]).reset_index(drop=True)

    temporal_df = pd.DataFrame(
        temporal_rows,
        columns=[
            "airline",
            "previous_origin",
            "origin",
            "wake",
            "intraday_sparse",
            "next_day_sparse",
            "total_intraday",
            "total_next_day",
        ],
    ).sort_values(["airline", "previous_origin", "origin", "wake"]).reset_index(drop=True)

    _validate_outputs(intraday_df, temporal_df)

    config.analysis_dir.mkdir(parents=True, exist_ok=True)
    intraday_df.to_csv(intraday_path, index=False)
    temporal_df.to_csv(temporal_path, index=False)

    print(f"[Turnaround] Saved intraday params: {intraday_path} ({len(intraday_df)} rows)")
    print(f"[Turnaround] Saved temporal profile: {temporal_path} ({len(temporal_df)} rows)")
    print("[Turnaround] --- SUCCESS ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Route-aware scheduled turnaround fitter")
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
    analyze_turnaround_distribution(cfg)
