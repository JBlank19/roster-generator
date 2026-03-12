from collections import defaultdict
from typing import Dict, Tuple

from ._schedule_structures import Flight

# Configuration constants
BIN_SIZE_MINS = 5
END_OF_DAY_MINS = 1440
ROLLING_WINDOW_SIZE_BINS = 12  # 60 minutes / 5-minute bins
DEFAULT_CAPACITY = 999


class CapacityTracker:
    """Track airport burst and rolling-window capacity during schedule generation."""

    def __init__(
        self,
        rolling_capacities: Dict[str, float],
        burst_capacities: Dict[str, float],
        window_length_mins: int = END_OF_DAY_MINS,
    ):
        self.rolling_cap = rolling_capacities
        self.burst_cap = burst_capacities
        self.window_length_mins = int(window_length_mins)
        self.num_bins = max(1, self.window_length_mins // BIN_SIZE_MINS)

        self.dep_slots: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.arr_slots: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.movements_rolling: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

    def _get_bin_index(self, time_mins: int) -> int:
        """Map a minute value to the corresponding [0, num_bins-1] bin index."""
        return max(0, min(time_mins // BIN_SIZE_MINS, self.num_bins - 1))

    def _rolling_window_bounds(self, time_bin: int) -> tuple[int, int]:
        """Return inclusive rolling-window bins impacted by a movement at `time_bin`."""
        start_bin = time_bin
        end_bin = min(self.num_bins - 1, time_bin + ROLLING_WINDOW_SIZE_BINS - 1)
        return start_bin, end_bin

    def _check_airport_availability(self, airport_code: str, time_mins: int, is_departure: bool) -> bool:
        """Check burst and rolling constraints for one airport/time movement."""
        time_bin = self._get_bin_index(time_mins)
        slots_tracker = self.dep_slots if is_departure else self.arr_slots

        burst_limit = self.burst_cap.get(airport_code, DEFAULT_CAPACITY)
        if slots_tracker[airport_code][time_bin] >= burst_limit:
            return False

        rolling_limit = self.rolling_cap.get(airport_code, DEFAULT_CAPACITY)
        start_bin, end_bin = self._rolling_window_bounds(time_bin)
        for bin_idx in range(start_bin, end_bin + 1):
            if self.movements_rolling[airport_code][bin_idx] >= rolling_limit:
                return False

        return True

    def check_availability(self, origin: str, destination: str, std_mins: int, sta_mins: int) -> bool:
        """Check whether origin departure and destination arrival can be accommodated."""
        if not (
            0 <= std_mins < self.window_length_mins
            and 0 <= sta_mins < self.window_length_mins
        ):
            return True

        if not self._check_airport_availability(origin, std_mins, is_departure=True):
            return False
        if not self._check_airport_availability(destination, sta_mins, is_departure=False):
            return False
        return True

    def _update_movement_counts(self, airport_code: str, time_mins: int, is_departure: bool) -> None:
        """Increment slot and rolling counters for one movement."""
        time_bin = self._get_bin_index(time_mins)
        slots_tracker = self.dep_slots if is_departure else self.arr_slots
        slots_tracker[airport_code][time_bin] += 1

        start_bin, end_bin = self._rolling_window_bounds(time_bin)
        for bin_idx in range(start_bin, end_bin + 1):
            self.movements_rolling[airport_code][bin_idx] += 1

    def add_flight(self, flight: Flight) -> None:
        """Register a flight in capacity counters."""
        if 0 <= flight.std < self.window_length_mins:
            self._update_movement_counts(flight.orig, flight.std, is_departure=True)
        if 0 <= flight.sta < self.window_length_mins:
            self._update_movement_counts(flight.dest, flight.sta, is_departure=False)

    def compute_violations(self) -> Tuple[int, int]:
        """Compute (burst_violations, rolling_violations) across all airports."""
        total_burst_violations = 0
        total_rolling_violations = 0

        for airport, bins in self.dep_slots.items():
            limit = self.burst_cap.get(airport, DEFAULT_CAPACITY)
            for count in bins.values():
                total_burst_violations += max(0, count - int(limit))

        for airport, bins in self.arr_slots.items():
            limit = self.burst_cap.get(airport, DEFAULT_CAPACITY)
            for count in bins.values():
                total_burst_violations += max(0, count - int(limit))

        for airport, windows in self.movements_rolling.items():
            limit = self.rolling_cap.get(airport, DEFAULT_CAPACITY)
            for count in windows.values():
                total_rolling_violations += max(0, count - int(limit))

        return total_burst_violations, total_rolling_violations
