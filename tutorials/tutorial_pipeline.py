"""
Roster Generator - Master Pipeline Script

Runs all pre-processing stages in order to produce a complete set of
simulation input files:

  Stage 0: Data cleaning
  Stage 1: Markov chain construction + initial conditions
  Stage 2: Turnaround and flight-time analysis
  Stage 3: Output file generation (airlines, airports, fleet, routes)
  Stage 4: Schedule generation (greedy forward construction)

Usage:
    python main.py [--seed SEED] [--suffix SUFFIX]

Example:
    python main.py --seed 42 --suffix 0
"""

import argparse
import os
import sys

import roster_generator


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Roster generator pipeline"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for stochastic stages (default: 42)"
    )
    parser.add_argument(
        "--suffix", type=str, default="",
        help="Output file suffix, e.g. '0' -> schedule_0.csv (default: none)"
    )
    args = parser.parse_args()

    suffix = f"_{args.suffix}" if args.suffix else ""

    # ------------------------------------------------------------------
    # Stage 0: Data Cleaning
    # ------------------------------------------------------------------
    input_file = "input/september2023.csv"
    if not os.path.exists(input_file):
        print(f"[Main] {input_file} not found. Cleaning data...")
        roster_generator.clean_data(
            dirty_file="ECTL/Flights_20230901_20230930.csv",
            clean_file=input_file,
        )
        print("[Main] Cleaning data done.")
    else:
        print(f"[Main] {input_file} already exists. Skipping cleaning stage.")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    print("[Main] Setting up config...")
    config = roster_generator.PipelineConfig(
        schedule_file=input_file,
        analysis_dir="computed",
        output_dir="output",
        seed=args.seed,
        suffix=suffix,
    )
    print("[Main] Setting up config done.")

    # ------------------------------------------------------------------
    # Stage 1: Initial Conditions + Markov Chains
    # ------------------------------------------------------------------
    print("[Main] Generating Markov chain...")
    roster_generator.generate_markov(config)
    print("[Main] Generating Markov chain done.")

    # ------------------------------------------------------------------
    # Stage 2: Turnaround + Flight Time Analysis
    # ------------------------------------------------------------------
    print("[Main] Analyzing scheduled turnaround...")
    roster_generator.analyze_turnaround_distribution(config)
    print("[Main] Analyzing scheduled turnaround done.")

    print("[Main] Analyzing scheduled flight time...")
    roster_generator.analyze_flight_time_distribution(config)
    print("[Main] Analyzing scheduled flight time done.")

    # ------------------------------------------------------------------
    # Stage 3: Output File Generation
    # ------------------------------------------------------------------
    print("[Main] Generating airlines...")
    roster_generator.generate_airlines(config)
    print("[Main] Generating airlines done.")

    print("[Main] Generating airports...")
    roster_generator.generate_airports(config)
    print("[Main] Generating airports done.")

    print("[Main] Generating fleet...")
    roster_generator.generate_fleet(config)
    print("[Main] Generating fleet done.")

    print("[Main] Generating routes...")
    roster_generator.generate_routes(config)
    print("[Main] Generating routes done.")

    # ------------------------------------------------------------------
    # Stage 4: Schedule Generation
    # ------------------------------------------------------------------
    print("[Main] Generating schedule...")
    roster_generator.generate_schedule(config)
    print("[Main] Generating schedule done.")

    print("[Main] Pipeline completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
