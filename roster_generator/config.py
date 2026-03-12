"""Central configuration for the pipeline."""

from dataclasses import dataclass, field
from pathlib import Path

from .time_window import (
    DEFAULT_REFTZ,
    DEFAULT_WINDOW_LENGTH_HOURS,
    DEFAULT_WINDOW_START,
    validate_reftz,
    validate_window_length_hours,
    validate_window_start,
    window_start_to_minutes,
)


@dataclass
class PipelineConfig:
    """All paths and parameters every pipeline step needs.

    Parameters
    ----------
    schedule_file : Path
        Cleaned CSV (e.g. ``september2023.csv``).
    analysis_dir : Path
        Intermediate analysis outputs (markov, turnaround params, etc.).
    output_dir : Path
        Final outputs consumed by the simulation (fleet, airports, schedule, etc.).
    seed : int
        Master RNG seed.  Passed to numpy / random.
    suffix : str
        Optional file-name suffix.
    reftz : str
        Reference timezone for time-of-day/day-boundary logic.
    window_start : str
        Window start in HH:MM in reference timezone.
    window_length_hours : int
        Window length in hours (1..24).
    """

    schedule_file: Path
    analysis_dir: Path
    output_dir: Path
    seed: int = 42
    suffix: str = ""
    reftz: str = DEFAULT_REFTZ
    window_start: str = DEFAULT_WINDOW_START
    window_length_hours: int = DEFAULT_WINDOW_LENGTH_HOURS
    window_start_mins: int = field(init=False)
    window_length_mins: int = field(init=False)

    def __post_init__(self) -> None:
        # Accept strings
        self.schedule_file = Path(self.schedule_file)
        self.analysis_dir = Path(self.analysis_dir)
        self.output_dir = Path(self.output_dir)
        self.reftz = validate_reftz(self.reftz)
        self.window_start = validate_window_start(self.window_start)
        self.window_length_hours = validate_window_length_hours(self.window_length_hours)
        self.window_start_mins = window_start_to_minutes(self.window_start)
        self.window_length_mins = int(self.window_length_hours) * 60

    # helpers

    def analysis_path(self, name: str) -> Path:
        """Return ``analysis_dir / <name><suffix>.csv``."""
        return self.analysis_dir / f"{name}{self.suffix}.csv"

    def output_path(self, name: str) -> Path:
        """Return ``output_dir / <name><suffix>.csv``."""
        return self.output_dir / f"{name}{self.suffix}.csv"
