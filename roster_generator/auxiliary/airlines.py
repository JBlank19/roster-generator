"""
Airline Catalogue

Extracts the unique set of airline ICAO codes (``AC_OPER``) that appear
in the sampled initial conditions and writes them as a flat list.

Input:
  - initial_conditions{suffix}.csv  (analysis_dir)

Output:
  - airlines{suffix}.csv
      columns: airline_id
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from roster_generator.config import PipelineConfig

# --- Column aliases ---

AIRLINE_COL = "AC_OPER"


# --- Public API ---

def generate_airlines(config: PipelineConfig) -> None:
    """Extract the airline catalogue from the initial-conditions file.

    Steps:

    1. Load ``initial_conditions{suffix}.csv`` from ``analysis_dir``.
    2. Extract unique ``AC_OPER`` values, sort alphabetically.
    3. Save the result to ``output_dir/airlines{suffix}.csv``.

    Parameters
    ----------
    config : PipelineConfig
        Paths and parameters for the pipeline.

    Raises
    ------
    FileNotFoundError
        If the initial-conditions file does not exist.
    """
    print("[Airlines] --- AIRLINE CATALOGUE ---")

    input_path = config.analysis_path("initial_conditions")
    output_path = config.output_path("airlines")

    if not input_path.exists():
        print(f"[Airlines] Error: initial conditions file not found ({input_path}).")
        sys.exit(1)

    # 1. Load initial conditions
    print(f"[Airlines] Initial conditions: {input_path}")
    df = pd.read_csv(input_path)

    # 2. Extract unique AC_OPER values
    airlines = (
        df[[AIRLINE_COL]]
        .drop_duplicates()
        .sort_values(AIRLINE_COL)
        .reset_index(drop=True)
        .rename(columns={AIRLINE_COL: "airline_id"})
    )

    # 3. Persist
    config.output_dir.mkdir(parents=True, exist_ok=True)
    airlines.to_csv(output_path, index=False)

    print(f"[Airlines] Saved: {output_path} ({len(airlines)} unique airlines)")
    print(f"[Airlines]   {airlines['airline_id'].tolist()}")
    print("[Airlines] --- SUCCESS ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Airline catalogue builder")
    parser.add_argument("--analysis-dir", type=str, required=True, help="Analysis output directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Final output directory")
    parser.add_argument("--suffix", type=str, default="", help="Suffix for output filenames")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    # schedule_file is not used here but PipelineConfig requires it
    parser.add_argument("--schedule", type=str, default="", help="Path to schedule CSV (unused)")
    args = parser.parse_args()

    cfg = PipelineConfig(
        schedule_file=Path(args.schedule) if args.schedule else Path("."),
        analysis_dir=Path(args.analysis_dir),
        output_dir=Path(args.output_dir),
        seed=args.seed,
        suffix=f"_{args.suffix}" if args.suffix else "",
    )
    generate_airlines(cfg)
