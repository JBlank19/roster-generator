import math
import random
from typing import List, Tuple

from ._schedule_capacity import CapacityTracker
from ._schedule_data_manager import BIN_SIZE_MINS, DataManager
from ._schedule_stats import GenerationStats
from ._schedule_structures import Aircraft, Flight


class ScheduleGenerator:
    """Greedy forward schedule construction with exact-key turnaround logic."""

    def __init__(
        self,
        data: DataManager,
        tracker: CapacityTracker,
        stats: GenerationStats,
        rng: random.Random,
    ):
        self.data = data
        self.tracker = tracker
        self.stats = stats
        self.rng = rng
        self.window_length_mins = int(data.window_length_mins)

    def _clone_flight(self, flight: Flight) -> Flight:
        """Create a detached copy of a flight object."""
        return Flight(
            orig=flight.orig,
            dest=flight.dest,
            std=flight.std,
            sta=flight.sta,
        )

    def _get_turnaround_candidates(
        self,
        aircraft: Aircraft,
        prev_origin: str,
        current_airport: str,
        arrival_time: int,
    ) -> List[Tuple[int, str, float]]:
        """Legacy helper kept for API parity with older turnaround strategies."""
        ta_options = self.data.get_turnaround_options(
            aircraft.operator,
            prev_origin,
            current_airport,
            aircraft.wake,
            arrival_time,
        )

        if ta_options:
            scheduled_times = [t for t, _ in ta_options]
            min_ta = min(scheduled_times)
            max_ta = max(scheduled_times)
        else:
            min_ta = BIN_SIZE_MINS
            max_ta = self.window_length_mins - int(max(0, arrival_time)) + BIN_SIZE_MINS
            max_ta = max(BIN_SIZE_MINS, min(max_ta, self.window_length_mins - BIN_SIZE_MINS))
            ta_options = []

        candidates: List[Tuple[int, str, float]] = []
        seen: set[int] = set()

        weighted_items = []
        for ta_time, prob in ta_options:
            rand = self.rng.random()
            if rand == 0:
                rand = 1e-10
            weight = prob if prob > 0 else 1e-10
            score = pow(rand, 1.0 / weight)
            weighted_items.append((ta_time, prob, score))

        weighted_items.sort(key=lambda x: -x[2])

        for ta_time, prob, _ in weighted_items:
            if ta_time not in seen:
                candidates.append((ta_time, "scheduled", prob))
                seen.add(ta_time)

        for ta_time in range(min_ta, max_ta + 1, 5):
            if ta_time not in seen:
                candidates.append((ta_time, "interval", 0.0))
                seen.add(ta_time)

        for ta_time in range(max_ta + 5, self.window_length_mins, 5):
            if ta_time not in seen:
                candidates.append((ta_time, "extended", 0.0))
                seen.add(ta_time)

        return candidates

    def _sample_turnaround(
        self,
        aircraft: Aircraft,
        prev_origin: str,
        current_airport: str,
        arrival_time: int,
    ) -> tuple[int, str]:
        """Sample turnaround minutes and category for the current airport state."""
        return self.data.sample_turnaround_for_prev_origin(
            op=aircraft.operator,
            prev_origin=prev_origin,
            origin=current_airport,
            wake=aircraft.wake,
            arr_utc_mins=arrival_time,
        )

    @staticmethod
    def _compute_next_std(arrival_time: int, turnaround_time: int) -> int:
        """Round departure up to the next 5-minute boundary."""
        return int(math.ceil((arrival_time + turnaround_time) / 5.0) * 5)

    def _choose_destination_order(
        self,
        destinations: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        """Weighted random sampling without replacement over destination options."""
        remaining = list(destinations)
        ordered: List[Tuple[str, float]] = []

        while remaining:
            weights = [prob for _, prob in remaining]
            picked_idx = self.rng.choices(range(len(remaining)), weights=weights, k=1)[0]
            ordered.append(remaining.pop(picked_idx))

        return ordered

    def _record_missing_turnaround(
        self,
        aircraft: Aircraft,
        prev_origin: str,
        current_airport: str,
        arrival_time: int,
    ) -> None:
        """Track stats and debug sample when turnaround parameters are missing."""
        self.stats.no_destinations += 1
        if len(aircraft.chain) == 1:
            self.stats.single_flight_total += 1
            self.stats.single_flight_no_destinations += 1

        markov_key = (aircraft.operator, aircraft.wake, prev_origin, current_airport)
        hourly_data = self.data.markov_hourly.get(markov_key, {})
        available_hours = sorted(set(hourly_data.keys()))
        self.stats.add_example_no_destinations(
            f"  Aircraft {aircraft.reg} ({aircraft.operator}, {aircraft.wake}):\n"
            f"    At {current_airport} from {prev_origin}, arrival={arrival_time} mins\n"
            f"    No turnaround params -- key: {(aircraft.operator, prev_origin, current_airport, aircraft.wake)}\n"
            f"    Markov available hours: {available_hours}"
        )

    def _record_end_of_day(
        self,
        aircraft: Aircraft,
        turnaround_time: int,
        scheduled_departure_time: int,
        anchor_flight: Flight | None,
    ) -> None:
        """Track end-of-day termination and annotate anchor flight turnaround."""
        if anchor_flight is not None:
            anchor_flight.turnaround_to_next_category = "next_day"
            anchor_flight.turnaround_to_next_minutes = int(turnaround_time)

        self.stats.end_of_day += 1
        if len(aircraft.chain) == 1:
            self.stats.single_flight_end_of_day += 1
            self.stats.single_flight_total += 1
            term_hour = (scheduled_departure_time // 60) % max(1, self.window_length_mins // 60)
            self.stats.single_flight_termination_hours[term_hour] += 1

    def _record_no_destination(
        self,
        aircraft: Aircraft,
        prev_origin: str,
        current_airport: str,
        arrival_time: int,
    ) -> None:
        """Track stats and debug sample when no destination can be selected."""
        self.stats.no_destinations += 1
        if len(aircraft.chain) == 1:
            self.stats.single_flight_total += 1
            self.stats.single_flight_no_destinations += 1

        markov_key = (aircraft.operator, aircraft.wake, prev_origin, current_airport)
        hourly_data = self.data.markov_hourly.get(markov_key, {})
        available_hours = sorted(set(hourly_data.keys()))
        self.stats.add_example_no_destinations(
            f"  Aircraft {aircraft.reg} ({aircraft.operator}, {aircraft.wake}):\n"
            f"    At {current_airport} from {prev_origin}, arrival={arrival_time} mins\n"
            f"    Markov key: {markov_key}\n"
            f"    Available departure hours: {available_hours}"
        )

    def _record_capacity_exhausted(self, aircraft: Aircraft) -> None:
        """Track stats for capacity-driven termination."""
        self.stats.capacity_exhausted += 1
        if len(aircraft.chain) == 1:
            self.stats.single_flight_total += 1
            self.stats.single_flight_capacity_exhausted += 1

    def _record_destination_source(self, source_type: str) -> None:
        """Track which Markov destination source resolved the next leg."""
        if source_type == "primary_exact":
            self.stats.dest_found_primary_exact += 1
        elif source_type == "primary_expanded":
            self.stats.dest_found_primary_expanded += 1
        elif source_type == "fallback_expanded":
            self.stats.dest_found_fallback_expanded += 1
        elif source_type == "return_to_origin":
            self.stats.dest_found_return_to_origin += 1

    def _try_append_flight(
        self,
        aircraft: Aircraft,
        anchor_flight: Flight | None,
        current_airport: str,
        scheduled_departure_time: int,
        turnaround_time: int,
        ordered_destinations: List[Tuple[str, float]],
    ) -> tuple[bool, str | None, str, int]:
        """Try destination candidates until one valid leg is appended."""
        termination_reason = None

        for destination, _ in ordered_destinations:
            flight_time = self.data.get_flight_time(
                current_airport,
                destination,
                aircraft.operator,
                aircraft.wake,
                dep_utc_mins=scheduled_departure_time,
            )
            if flight_time <= 0:
                continue

            scheduled_arrival_time = scheduled_departure_time + flight_time
            if scheduled_arrival_time < self.window_length_mins:
                available = self.tracker.check_availability(
                    current_airport,
                    destination,
                    scheduled_departure_time,
                    scheduled_arrival_time,
                )
                if not available:
                    termination_reason = "capacity"
                    continue

            if anchor_flight is not None:
                anchor_flight.turnaround_to_next_category = "intraday"
                anchor_flight.turnaround_to_next_minutes = int(turnaround_time)

            new_flight = Flight(
                orig=current_airport,
                dest=destination,
                std=scheduled_departure_time,
                sta=scheduled_arrival_time,
            )
            aircraft.chain.append(new_flight)

            if scheduled_arrival_time < self.window_length_mins:
                self.tracker.add_flight(new_flight)

            if scheduled_arrival_time >= self.window_length_mins:
                self.stats.end_of_day += 1

            return True, None, destination, scheduled_arrival_time

        return False, termination_reason, current_airport, -1

    def seed_initial_flights(self, aircraft: Aircraft) -> bool:
        """Seed chain with prior/initial flights and return whether greedy extension is needed."""
        aircraft.chain.clear()

        if aircraft.initial_flight is None:
            if aircraft.prior_flight:
                prior = self._clone_flight(aircraft.prior_flight)
                aircraft.chain.append(prior)
                self.tracker.add_flight(prior)
                self.stats.prior_flight_pasted += 1
                self.stats.prior_only_pasted += 1
            return False

        if aircraft.prior_flight:
            prior = self._clone_flight(aircraft.prior_flight)
            aircraft.chain.append(prior)
            self.tracker.add_flight(prior)
            self.stats.prior_flight_pasted += 1

        first_flight = self._clone_flight(aircraft.initial_flight)
        aircraft.chain.append(first_flight)
        self.tracker.add_flight(first_flight)

        if aircraft.is_single_flight:
            self.stats.single_flight_passthrough += 1
            if aircraft.prior_flight:
                self.stats.single_flight_with_prior += 1
            return False

        return True

    def generate_greedy_chain(self, aircraft: Aircraft) -> bool:
        """Extend a seeded chain greedily until day end or no valid continuation exists."""
        if not aircraft.chain:
            return False

        last_flight = aircraft.chain[-1]
        current_airport = last_flight.dest
        prev_origin = last_flight.orig
        arrival_time = last_flight.sta

        while arrival_time < self.window_length_mins:
            anchor_flight = aircraft.chain[-1] if aircraft.chain else None

            turnaround_time, ta_category = self._sample_turnaround(
                aircraft,
                prev_origin,
                current_airport,
                arrival_time,
            )
            if turnaround_time < 0:
                self._record_missing_turnaround(
                    aircraft,
                    prev_origin,
                    current_airport,
                    arrival_time,
                )
                break

            scheduled_departure_time = self._compute_next_std(arrival_time, turnaround_time)
            if ta_category == "next_day" or scheduled_departure_time >= self.window_length_mins:
                self._record_end_of_day(
                    aircraft,
                    turnaround_time,
                    scheduled_departure_time,
                    anchor_flight,
                )
                break

            destinations, source_type = self.data.get_destinations(
                aircraft.operator,
                aircraft.wake,
                prev_origin,
                current_airport,
                dep_utc_mins=scheduled_departure_time,
                arr_utc_mins=arrival_time,
            )

            if not destinations:
                self._record_no_destination(
                    aircraft,
                    prev_origin,
                    current_airport,
                    arrival_time,
                )
                break

            ordered = self._choose_destination_order(destinations)
            appended, termination_reason, next_airport, next_arrival = self._try_append_flight(
                aircraft,
                anchor_flight,
                current_airport,
                scheduled_departure_time,
                turnaround_time,
                ordered,
            )

            if appended:
                self._record_destination_source(source_type)
                prev_origin = current_airport
                current_airport = next_airport
                arrival_time = next_arrival
                continue

            if termination_reason == "capacity":
                self._record_capacity_exhausted(aircraft)
            else:
                self._record_no_destination(
                    aircraft,
                    prev_origin,
                    current_airport,
                    arrival_time,
                )
            break

        return len(aircraft.chain) > 0
