"""Sampling routines for synthetic day-0 initial conditions."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd

from ._initial_conditions_builders import get_phys_ta_min
from ._initial_conditions_types import (
    P_NEXT_BIN_SIZE,
    InitialConditionState,
    SyntheticAircraft,
)


def _normalize_airline_prefix(value: str) -> str:
    """Strip non-alphanumeric chars; used to build synthetic registration codes."""
    out = "".join(ch for ch in str(value).upper().strip() if ch.isalnum())
    if not out:
        raise ValueError("Empty airline prefix after normalization.")
    return out


def _weighted_choice_from_counts(counts_by_value: dict[str, int], rng: random.Random) -> str:
    """Sample one key from {value: count} using counts as weights."""
    values = list(counts_by_value.keys())
    weights = [int(counts_by_value[v]) for v in values]
    if not values or sum(weights) <= 0:
        raise ValueError("Invalid weighted choice map: empty or non-positive weights.")
    return rng.choices(values, weights=weights, k=1)[0]


def _weighted_choice_from_pairs(items: list[tuple[str, float]], rng: random.Random) -> str:
    """Sample from [(value, probability), ...] pairs."""
    if not items:
        raise ValueError("Cannot sample from empty weighted pair list.")
    vals = [v for v, _ in items]
    w = [float(p) for _, p in items]
    if sum(w) <= 0:
        raise ValueError("Cannot sample from non-positive weighted pair list.")
    return rng.choices(vals, weights=w, k=1)[0]


def _clear_prior_details(ac: SyntheticAircraft) -> None:
    """Reset all prior-flight fields on an aircraft object."""
    ac.has_prior = False
    ac.prior_only = False
    ac.prior_origin = None
    ac.prior_dest = None
    ac.prior_sta_mins = None
    ac.prior_std_mins = None


def _sample_fleet(state: InitialConditionState, rng: random.Random) -> list[SyntheticAircraft]:
    """Draw fleet size from Gaussian, assign origins and prior flags."""
    aircraft: list[SyntheticAircraft] = []
    reg_counters: dict[str, int] = {}

    keys = sorted(state.daily_fleet_stats.keys())
    for airline, wake in keys:
        mu, sigma = state.daily_fleet_stats[(airline, wake)]
        sampled = float(np.random.normal(mu, sigma)) if sigma > 0 else mu
        n = max(0, int(round(sampled)))

        if n == 0:
            continue

        if (airline, wake) not in state.origin_counts:
            raise ValueError(f"Missing origin distribution for {(airline, wake)}")

        for _ in range(n):
            prefix = _normalize_airline_prefix(airline)
            reg_counters[prefix] = reg_counters.get(prefix, 0) + 1
            reg = f"{prefix}{reg_counters[prefix]:03d}"
            origin = _weighted_choice_from_counts(state.origin_counts[(airline, wake)], rng)
            p_prior = state.p_prior.get((airline, wake), 0.0)
            has_prior = rng.random() < p_prior
            aircraft.append(
                SyntheticAircraft(
                    reg=reg,
                    airline=airline,
                    wake=wake,
                    origin=origin,
                    has_prior=has_prior,
                )
            )

    return aircraft


def _find_hourly_data_with_radius(
    hourly_data: dict[int, dict[str, int]],
    center_hour: int,
    rng: random.Random,
    hour_bins: int,
    max_radius: int = 2,
) -> tuple[dict[str, int], int]:
    """Find hourly data at center hour or nearby hours (randomized direction)."""
    if not hourly_data:
        return {}, -1

    candidates = [center_hour]
    for radius in range(1, max_radius + 1):
        if rng.random() < 0.5:
            candidates.append((center_hour - radius) % hour_bins)
            candidates.append((center_hour + radius) % hour_bins)
        else:
            candidates.append((center_hour + radius) % hour_bins)
            candidates.append((center_hour - radius) % hour_bins)

    for hour in candidates:
        if hour in hourly_data:
            return hourly_data[hour], hour

    available_hours = sorted(hourly_data.keys())
    future_hours = [hour for hour in available_hours if hour >= center_hour]
    if future_hours:
        hour = future_hours[0]
        return hourly_data[hour], hour

    return {}, -1


def _get_markov_destinations(
    state: InitialConditionState,
    rng: random.Random,
    op: str,
    wake: str,
    prev_origin: str | None,
    origin: str,
    dep_utc_mins: int,
) -> list[tuple[str, float]]:
    """Query Markov tables for destination probabilities at a departure hour."""
    dep_hour = (int(dep_utc_mins) // 60) % int(state.hour_bins)

    fallback_key = (op, wake, origin)

    counts: dict[str, int] = {}
    if prev_origin:
        primary_key = (op, wake, prev_origin, origin)
        hourly_data = state.markov_hourly.get(primary_key, {})
        counts, _ = _find_hourly_data_with_radius(
            hourly_data,
            dep_hour,
            rng,
            hour_bins=state.hour_bins,
        )

    if not counts:
        fb_hourly_data = state.markov_fallback_hourly.get(fallback_key, {})
        counts, _ = _find_hourly_data_with_radius(
            fb_hourly_data,
            dep_hour,
            rng,
            hour_bins=state.hour_bins,
        )

    if not counts:
        return []

    total = int(sum(counts.values()))
    if total <= 0:
        return []

    out = [(dest, float(count / total)) for dest, count in counts.items()]
    out.sort(key=lambda pair: -pair[1])
    return out


def _origin_has_feasible_prior(
    state: InitialConditionState,
    airline: str,
    wake: str,
    origin: str,
) -> bool:
    """Check whether any prior route into origin yields STD < 0."""
    prev_counts = state.backward_prev_counts.get((airline, wake, origin))
    if not prev_counts:
        return False

    for prev_origin in prev_counts:
        key_sta = (airline, wake, prev_origin, origin)
        sta_samples = state.prior_sta_samples.get(key_sta)
        if not sta_samples:
            continue
        flight_time = state.flight_time_median.get(key_sta)
        if flight_time is None or int(flight_time) <= 0:
            continue
        if any(int(sta) - int(flight_time) < 0 for sta in sta_samples):
            return True

    return False


def _try_sample_prior_for_origin(
    state: InitialConditionState,
    rng: random.Random,
    ac: SyntheticAircraft,
    origin: str,
) -> bool:
    """Attempt to sample a valid prior flight arriving at `origin`."""
    key_prev = (ac.airline, ac.wake, origin)
    prev_counts = state.backward_prev_counts.get(key_prev)
    if not prev_counts:
        return False

    weighted_prev_candidates = [
        (prev_origin, int(weight))
        for prev_origin, weight in prev_counts.items()
    ]
    attempted_prev: set[str] = set()

    while len(attempted_prev) < len(weighted_prev_candidates):
        remaining_prev = [
            (prev_origin, weight)
            for prev_origin, weight in weighted_prev_candidates
            if prev_origin not in attempted_prev
        ]
        remaining_weights = [weight for _, weight in remaining_prev]
        sampled_pair = rng.choices(remaining_prev, weights=remaining_weights, k=1)[0]
        prev_origin = sampled_pair[0]
        attempted_prev.add(prev_origin)

        key_sta = (ac.airline, ac.wake, prev_origin, origin)
        sta_samples = state.prior_sta_samples.get(key_sta)
        if not sta_samples:
            continue

        shuffled = sta_samples.copy()
        rng.shuffle(shuffled)

        flight_time = state.flight_time_median.get(key_sta)
        if flight_time is None or int(flight_time) <= 0:
            continue

        for sta in shuffled:
            std = int(sta) - int(flight_time)
            if std < 0:
                ac.prior_origin = prev_origin
                ac.prior_dest = origin
                ac.prior_sta_mins = int(sta)
                ac.prior_std_mins = int(std)
                return True

    return False


def _sample_prior_for_aircraft(state: InitialConditionState, rng: random.Random, ac: SyntheticAircraft) -> None:
    """Assign overnight prior-flight details, trying current origin first."""
    if not ac.has_prior:
        return

    if _try_sample_prior_for_origin(state, rng, ac, ac.origin):
        return

    key = (ac.airline, ac.wake)
    origins = state.origin_counts.get(key, {})
    if not origins:
        raise ValueError(f"Missing origin distribution for {key}")

    candidate_origins = [
        origin
        for origin in origins.keys()
        if origin != ac.origin and _origin_has_feasible_prior(state, ac.airline, ac.wake, origin)
    ]
    weights = [int(origins[origin]) for origin in candidate_origins]

    while candidate_origins:
        sampled_origin = rng.choices(candidate_origins, weights=weights, k=1)[0]
        idx = candidate_origins.index(sampled_origin)
        candidate_origins.pop(idx)
        weights.pop(idx)

        ac.origin = sampled_origin
        if _try_sample_prior_for_origin(state, rng, ac, ac.origin):
            return

    _clear_prior_details(ac)


def _sample_prior_only(state: InitialConditionState, rng: random.Random, ac: SyntheticAircraft) -> None:
    """Decide whether a prior-equipped aircraft sits idle all day."""
    if not ac.has_prior:
        ac.prior_only = False
        return
    p = state.p_prior_only.get((ac.airline, ac.wake), 0.0)
    ac.prior_only = rng.random() < p


def _sample_first_std(state: InitialConditionState, rng: random.Random, ac: SyntheticAircraft) -> int:
    """Sample first departure time, respecting minimum turnaround after prior arrival."""
    key = (ac.airline, ac.wake, ac.origin)
    samples = state.first_std_samples.get(key)
    if not samples:
        raise ValueError(f"Missing first-STD distribution for {key}")

    min_ta = 0
    if ac.has_prior:
        min_ta = get_phys_ta_min(state, ac.airline, ac.wake)
        if ac.prior_sta_mins is None:
            raise ValueError("Aircraft has prior flag but prior STA is not assigned.")

    shuffled = samples.copy()
    rng.shuffle(shuffled)

    if not ac.has_prior:
        return int(shuffled[0])

    for std in shuffled:
        if int(std) - int(ac.prior_sta_mins) >= min_ta:
            return int(std)

    _clear_prior_details(ac)
    return int(shuffled[0])


def _sample_first_destination(
    state: InitialConditionState,
    rng: random.Random,
    ac: SyntheticAircraft,
    first_std_mins: int,
) -> str:
    """Draw first-flight destination from Markov transition probabilities."""
    prev_origin = ac.prior_origin if ac.has_prior else None

    dest_probs = _get_markov_destinations(
        state=state,
        rng=rng,
        op=ac.airline,
        wake=ac.wake,
        prev_origin=prev_origin,
        origin=ac.origin,
        dep_utc_mins=first_std_mins,
    )
    if not dest_probs:
        raise ValueError(
            f"No first destination available from Markov for airline={ac.airline}, wake={ac.wake}, "
            f"prev_origin={prev_origin}, origin={ac.origin}, "
            f"dep_hour={(first_std_mins // 60) % int(state.hour_bins)}"
        )

    return _weighted_choice_from_pairs(dest_probs, rng)


def _compute_first_sta(state: InitialConditionState, ac: SyntheticAircraft, first_std_mins: int, dest: str) -> int:
    """Compute STA = STD + median scheduled block time for this route."""
    ft_key = (ac.airline, ac.wake, ac.origin, dest)
    flight_time = state.flight_time_median.get(ft_key)
    if flight_time is None or flight_time <= 0:
        raise ValueError(f"Missing scheduled flight time for first flight key {ft_key}")
    return int(first_std_mins + int(flight_time))


def _sample_single_flight_flag(
    state: InitialConditionState,
    rng: random.Random,
    ac: SyntheticAircraft,
    first_sta_mins: int,
) -> int:
    """Bernoulli draw: does this aircraft stop after one flight (next-day pattern)?"""
    if int(first_sta_mins) >= int(state.window_length_mins):
        return 1

    hour = (int(first_sta_mins) % int(state.window_length_mins)) // P_NEXT_BIN_SIZE
    wake = str(ac.wake)
    key = (ac.airline, wake)
    fallback_key = ("ALL", wake)

    hmap = state.p_next_hourly.get(key)
    if hmap is None or hour not in hmap:
        hmap = state.p_next_hourly.get(fallback_key)

    if hmap is None or hour not in hmap:
        raise ValueError(
            f"Missing p_next profile for {(ac.airline, ac.wake)} with fallback {fallback_key} "
            f"at hour bin={hour}"
        )

    p_next = float(hmap[hour])
    return 1 if rng.random() < p_next else 0


def sample_initial_conditions(state: InitialConditionState, rng: random.Random) -> pd.DataFrame:
    """Generate one synthetic day-0 fleet from fitted state and Markov tables."""
    aircraft = _sample_fleet(state, rng)

    for ac in aircraft:
        if ac.has_prior:
            _sample_prior_for_aircraft(state, rng, ac)
        _sample_prior_only(state, rng, ac)

        if ac.prior_only:
            continue

        ac.first_std_mins = _sample_first_std(state, rng, ac)
        ac.first_dest = _sample_first_destination(state, rng, ac, ac.first_std_mins)
        ac.first_sta_mins = _compute_first_sta(state, ac, ac.first_std_mins, ac.first_dest)
        ac.single_flight = _sample_single_flight_flag(state, rng, ac, ac.first_sta_mins)

    rows = []
    for ac in aircraft:
        rows.append({
            "AC_REG": ac.reg,
            "AC_OPER": ac.airline,
            "AC_WAKE": ac.wake,
            "PRIOR_ONLY": 1 if ac.prior_only else 0,
            "ORIGIN": "" if ac.prior_only else ac.origin,
            "DEST": "" if ac.prior_only else ac.first_dest,
            "STD_REFTZ_MINS": None if ac.prior_only else int(ac.first_std_mins),
            "STA_REFTZ_MINS": None if ac.prior_only else int(ac.first_sta_mins),
            "SINGLE_FLIGHT": 0 if ac.prior_only else int(ac.single_flight),
            "PRIOR_ORIGIN": ac.prior_origin,
            "PRIOR_DEST": ac.prior_dest,
            "PRIOR_STD_REFTZ_MINS": ac.prior_std_mins,
            "PRIOR_STA_REFTZ_MINS": ac.prior_sta_mins,
        })

    return pd.DataFrame(rows)
