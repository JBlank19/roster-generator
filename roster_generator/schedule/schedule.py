import argparse
import random
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from ..config import PipelineConfig
from ._schedule_capacity import CapacityTracker
from ._schedule_data_manager import DataManager
from ._schedule_generator import ScheduleGenerator
from ._schedule_stats import GenerationStats
from ._schedule_structures import Aircraft, Flight


def _build_runtime(seed: int) -> random.Random:
    """Initialize RNG state for Python and NumPy."""
    rng = random.Random(seed)
    np.random.seed(seed)
    return rng


def _resolve_schedule_paths(config: PipelineConfig) -> dict[str, Path]:
    """Resolve all required input/output paths for schedule generation."""
    return {
        "output": config.output_path("schedule"),
        "routes": config.output_path("routes"),
        "airports": config.output_path("airports"),
        "initial_conditions": config.analysis_path("initial_conditions"),
        "markov": config.analysis_path("markov"),
        "turnaround_intraday": config.analysis_path("scheduled_turnaround_intraday_params"),
        "turnaround_temporal": config.analysis_path("scheduled_turnaround_temporal_profile"),
    }


def _validate_required_inputs(paths: dict[str, Path]) -> None:
    """Raise on the first missing required input file."""
    required_keys = [
        "initial_conditions",
        "routes",
        "airports",
        "markov",
        "turnaround_intraday",
        "turnaround_temporal",
    ]
    for key in required_keys:
        path = paths[key]
        if not path.exists():
            raise FileNotFoundError(f"Required input file not found: {path}")


def _build_initial_flight(row, is_prior_only: bool) -> Flight | None:
    """Build the first in-day flight for an aircraft row, if applicable."""
    if is_prior_only:
        return None

    std_utc_mins = getattr(row, "STD_UTC_MINS", None)
    sta_utc_mins = getattr(row, "STA_UTC_MINS", None)
    if pd.isna(std_utc_mins) or pd.isna(sta_utc_mins):
        raise ValueError(
            f"Invalid initial_conditions row for AC_REG={row.AC_REG}: "
            "missing STD_UTC_MINS/STA_UTC_MINS without PRIOR_ONLY=1"
        )

    return Flight(
        orig=row.ORIGIN,
        dest=row.DEST,
        std=int(std_utc_mins),
        sta=int(sta_utc_mins),
    )


def _build_prior_flight(row) -> Flight | None:
    """Build prior-day arrival flight for an aircraft row, when present."""
    if not hasattr(row, "PRIOR_ORIGIN"):
        return None
    prior_origin = getattr(row, "PRIOR_ORIGIN", None)
    if pd.isna(prior_origin):
        return None

    return Flight(
        orig=row.PRIOR_ORIGIN,
        dest=row.PRIOR_DEST,
        std=int(row.PRIOR_STD_UTC_MINS),
        sta=int(row.PRIOR_STA_UTC_MINS),
    )


def _summarize_initial_conditions(aircraft_list: List[Aircraft]) -> None:
    """Print a compact summary of loaded initial conditions."""
    single_flight_count = sum(1 for ac in aircraft_list if ac.is_single_flight)
    single_flight_with_prior = sum(
        1 for ac in aircraft_list if ac.is_single_flight and ac.prior_flight
    )
    prior_only_count = sum(1 for ac in aircraft_list if ac.is_prior_only)

    print(f"[Schedule]   Loaded {len(aircraft_list)} aircraft with initial conditions")
    print(f"[Schedule]   Single-flight aircraft (will passthrough): {single_flight_count}")
    print(f"[Schedule]     With overnight arrival: {single_flight_with_prior}")
    print(f"[Schedule]   Prior-only aircraft (arrival passthrough): {prior_only_count}")


def _seed_and_collect_needs_greedy(
    aircraft_list: List[Aircraft],
    generator: ScheduleGenerator,
) -> List[Aircraft]:
    """Seed initial/prior flights and return aircraft requiring greedy extension."""
    needs_greedy: List[Aircraft] = []
    for aircraft in aircraft_list:
        if generator.seed_initial_flights(aircraft):
            needs_greedy.append(aircraft)
    return needs_greedy


def _format_chain_row(ac: Aircraft, flight: Flight, chain_index: int, initial_index: int) -> dict[str, object]:
    """Build one output row for a scheduled flight."""
    is_prior_flight = 1 if (ac.prior_flight is not None and chain_index == 0) else 0
    is_initial_departure = 1 if (ac.initial_flight is not None and chain_index == initial_index) else 0

    return {
        "airline_id": ac.operator,
        "aircraft_id": ac.reg,
        "orig_id": flight.orig,
        "dest_id": flight.dest,
        "STD_UTC": flight.std,
        "STA_UTC": flight.sta,
        "first_flight": 1 if chain_index == 0 else 0,
        "is_prior_flight": is_prior_flight,
        "is_initial_departure": is_initial_departure,
        "single_flight_real": 1 if ac.is_single_flight else 0,
        "turnaround_to_next_category": flight.turnaround_to_next_category,
        "turnaround_to_next_minutes": (
            int(flight.turnaround_to_next_minutes)
            if flight.turnaround_to_next_minutes >= 0
            else np.nan
        ),
    }


def _save_schedule(df: pd.DataFrame, output_path: Path, output_dir: Path) -> None:
    """Persist output schedule when there is at least one flight."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if not df.empty:
        df.to_csv(output_path, index=False)


def load_initial_conditions(initial_conditions_path: Path) -> List[Aircraft]:
    """Load initial conditions and create aircraft objects."""
    print("[Schedule] Loading initial conditions...")
    ic_df = pd.read_csv(initial_conditions_path)
    aircraft_list: List[Aircraft] = []

    for row in ic_df.itertuples():
        is_prior_only = bool(getattr(row, "PRIOR_ONLY", 0))
        aircraft = Aircraft(
            reg=row.AC_REG,
            operator=row.AC_OPER,
            wake=row.AC_WAKE,
            initial_flight=_build_initial_flight(row, is_prior_only=is_prior_only),
            prior_flight=_build_prior_flight(row),
            is_single_flight=bool(getattr(row, "SINGLE_FLIGHT", 0)),
            is_prior_only=is_prior_only,
        )
        aircraft_list.append(aircraft)

    _summarize_initial_conditions(aircraft_list)
    return aircraft_list


def run_generation(
    aircraft_list: List[Aircraft],
    generator: ScheduleGenerator,
    tracker: CapacityTracker,
    stats: GenerationStats,
) -> None:
    """Run schedule generation for all aircraft."""
    print("[Schedule] Seeding initial flights...")
    needs_greedy = _seed_and_collect_needs_greedy(aircraft_list, generator)

    initial_burst, initial_rolling = tracker.compute_violations()
    initial_flights = sum(len(ac.chain) for ac in aircraft_list)
    print(
        f"[Schedule]   Seeded {initial_flights} initial flights | "
        f"Initial violations: burst={initial_burst}, rolling={initial_rolling}"
    )

    print(f"[Schedule] Generating greedy chains for {len(needs_greedy)} aircraft...")
    for index, aircraft in enumerate(needs_greedy, start=1):
        generator.generate_greedy_chain(aircraft)
        stats.successful_chains += 1
        stats.total_flights = sum(len(a.chain) for a in aircraft_list)

        if index % 1000 == 0:
            burst, rolling = tracker.compute_violations()
            print(
                f"[Schedule]   {index}/{len(needs_greedy)} | Flights: {stats.total_flights} | "
                f"Violations: burst={burst}, rolling={rolling}"
            )

    greedy_ids = {id(ac) for ac in needs_greedy}
    for aircraft in aircraft_list:
        if id(aircraft) not in greedy_ids and aircraft.chain:
            stats.successful_chains += 1
    stats.total_flights = sum(len(ac.chain) for ac in aircraft_list)


def format_results(aircraft_list: List[Aircraft]) -> pd.DataFrame:
    """Format generated chains into a schedule DataFrame."""
    rows: list[dict[str, object]] = []
    for ac in aircraft_list:
        initial_index = 1 if ac.prior_flight and ac.initial_flight is not None else 0
        for chain_index, flight in enumerate(ac.chain):
            rows.append(_format_chain_row(ac, flight, chain_index, initial_index))

    return pd.DataFrame(rows)


def generate_schedule(config: PipelineConfig) -> None:
    """Generate synthetic flight schedule via greedy forward construction."""
    seed = config.seed
    suffix = config.suffix
    paths = _resolve_schedule_paths(config)
    rng = _build_runtime(seed)

    print("[Schedule] --- SCHEDULE GENERATOR (Greedy Construction) ---")
    print(f"[Schedule] Seed={seed}, Suffix='{suffix}'")
    print(f"[Schedule] Output: {paths['output']}")
    print(
        f"[Schedule] Inputs: IC={paths['initial_conditions']}, Routes={paths['routes']}, "
        f"Airports={paths['airports']}"
    )
    print(
        f"[Schedule] Turnaround params: {paths['turnaround_intraday']}, "
        f"{paths['turnaround_temporal']}"
    )

    _validate_required_inputs(paths)

    data = DataManager(
        rng,
        paths["routes"],
        paths["airports"],
        paths["markov"],
        paths["turnaround_intraday"],
        paths["turnaround_temporal"],
    )

    aircraft_list = load_initial_conditions(paths["initial_conditions"])
    rng.shuffle(aircraft_list)

    tracker = CapacityTracker(data.rolling_capacity, data.burst_capacity)
    stats = GenerationStats(total_aircraft=len(aircraft_list))
    generator = ScheduleGenerator(data, tracker, stats, rng)

    run_generation(aircraft_list, generator, tracker, stats)

    print("[Schedule] Saving schedule...")
    df = format_results(aircraft_list)
    _save_schedule(df, paths["output"], config.output_dir)

    print(stats.summary())
    if data.turnaround_lookup_stats:
        ordered_keys = sorted(data.turnaround_lookup_stats.keys())
        parts = [f"{k}={int(data.turnaround_lookup_stats[k])}" for k in ordered_keys]
        print("[Schedule] Turnaround lookup diagnostics: " + ", ".join(parts))
    print(f"[Schedule] Saved: {paths['output']} ({len(df)} flights)")
    print("[Schedule] --- SUCCESS ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic flight schedule")
    parser.add_argument("--schedule", type=str, required=True, help="Path to schedule CSV")
    parser.add_argument("--analysis-dir", type=str, required=True, help="Analysis output directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Final output directory")
    parser.add_argument("--suffix", type=str, default="", help="Suffix for output filenames")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    cfg = PipelineConfig(
        schedule_file=Path(args.schedule),
        analysis_dir=Path(args.analysis_dir),
        output_dir=Path(args.output_dir),
        seed=args.seed,
        suffix=f"_{args.suffix}" if args.suffix else "",
    )
    generate_schedule(cfg)
