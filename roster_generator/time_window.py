"""Shared utilities for REFTZ-based shifted time windows."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytz

DEFAULT_REFTZ = "UTC"
DEFAULT_WINDOW_START = "00:00"
DEFAULT_WINDOW_LENGTH_HOURS = 24
DEFAULT_ACTUAL_TIMES = False

PARAM_KEY_REFTZ = "REFTZ"
PARAM_KEY_WINDOW_START = "WINDOW_START"
PARAM_KEY_WINDOW_LENGTH_HOURS = "WINDOW_LENGTH_HOURS"
PARAM_KEY_ACTUAL_TIMES = "ACTUAL_TIMES"
ALLOWED_PARAM_KEYS = {
    PARAM_KEY_REFTZ,
    PARAM_KEY_WINDOW_START,
    PARAM_KEY_WINDOW_LENGTH_HOURS,
    PARAM_KEY_ACTUAL_TIMES,
}

WINDOW_START_PATTERN = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")


def validate_reftz(value: Any) -> str:
    """Validate and normalize a reference timezone string."""
    out = str(value).strip()
    if not out:
        raise ValueError("REFTZ must be a non-empty timezone name.")
    if out.upper() == "UTC":
        return "UTC"
    try:
        pytz.timezone(out)
    except Exception as exc:
        raise ValueError(f"Invalid REFTZ timezone: {out!r}") from exc
    return out


def validate_window_start(value: Any) -> str:
    """Validate strict HH:MM window start and return normalized string."""
    out = str(value).strip()
    if not WINDOW_START_PATTERN.fullmatch(out):
        raise ValueError(
            f"Invalid WINDOW_START={value!r}. Expected strict HH:MM (00:00..23:59)."
        )
    return out


def window_start_to_minutes(window_start: str) -> int:
    """Convert HH:MM window start to minute offset."""
    hh, mm = window_start.split(":", 1)
    return int(hh) * 60 + int(mm)


def validate_window_length_hours(value: Any) -> int:
    """Validate window length hours in [1, 24]."""
    try:
        out = int(value)
    except Exception as exc:
        raise ValueError(
            f"Invalid WINDOW_LENGTH_HOURS={value!r}. Expected integer in [1, 24]."
        ) from exc
    if out < 1 or out > 24:
        raise ValueError(f"WINDOW_LENGTH_HOURS must be within [1, 24], got {out}.")
    return out


def validate_actual_times(value: Any) -> bool:
    """Validate ACTUAL_TIMES and normalize it to bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "on"}:
            return True
        if text in {"false", "0", "no", "off"}:
            return False
    raise ValueError(
        f"Invalid ACTUAL_TIMES={value!r}. Expected boolean-like true/false or 1/0."
    )


def parse_datetime_series_to_reftz(series: pd.Series, reftz: str) -> pd.Series:
    """Parse timestamps as UTC and convert them to REFTZ timezone-naive datetimes."""
    parsed = pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
    converted = parsed.dt.tz_convert(reftz).dt.tz_localize(None)
    return converted


def shift_series_by_window_start(series: pd.Series, window_start_mins: int) -> pd.Series:
    """Shift datetimes so window start becomes minute 0 for each day."""
    return series - pd.to_timedelta(int(window_start_mins), unit="m")


def minute_of_shifted_day(series: pd.Series) -> pd.Series:
    """Minute-of-day from shifted timestamp series, in [0, 1439]."""
    return (series.dt.hour * 60 + series.dt.minute).astype("Int64")


def hour_of_shifted_day(series: pd.Series) -> pd.Series:
    """Hour-of-day from shifted timestamp series, in [0, 23]."""
    return series.dt.hour.astype("Int64")


@dataclass(frozen=True)
class WindowConfig:
    """Validated runtime window configuration."""

    reftz: str = DEFAULT_REFTZ
    window_start: str = DEFAULT_WINDOW_START
    window_length_hours: int = DEFAULT_WINDOW_LENGTH_HOURS
    actual_times: bool = DEFAULT_ACTUAL_TIMES

    @property
    def window_start_mins(self) -> int:
        return window_start_to_minutes(self.window_start)

    @property
    def window_length_mins(self) -> int:
        return int(self.window_length_hours) * 60


def _parse_scalar(raw: str) -> Any:
    text = raw.strip()
    if not text:
        return ""
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        return text[1:-1]
    if text.isdigit():
        return int(text)
    return text


def load_params_yaml(path: Path | str) -> dict[str, Any]:
    """Load simple top-level YAML key-value params from disk."""
    params_path = Path(path)
    if not params_path.exists():
        return {}

    out: dict[str, Any] = {}
    for lineno, line in enumerate(
        params_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(
                f"Invalid params.yaml line {lineno}: expected 'KEY: value', got {line!r}"
            )
        key, raw_value = line.split(":", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid params.yaml line {lineno}: empty key.")
        out[key] = _parse_scalar(raw_value)
    return out


def resolve_window_config(raw_params: dict[str, Any]) -> WindowConfig:
    """Validate params map and produce validated WindowConfig."""
    unknown = sorted(set(raw_params.keys()) - ALLOWED_PARAM_KEYS)
    if unknown:
        raise ValueError(f"Unknown keys in params.yaml: {unknown}")

    reftz = validate_reftz(raw_params.get(PARAM_KEY_REFTZ, DEFAULT_REFTZ))
    window_start = validate_window_start(
        raw_params.get(PARAM_KEY_WINDOW_START, DEFAULT_WINDOW_START)
    )
    window_length_hours = validate_window_length_hours(
        raw_params.get(PARAM_KEY_WINDOW_LENGTH_HOURS, DEFAULT_WINDOW_LENGTH_HOURS)
    )
    actual_times = validate_actual_times(
        raw_params.get(PARAM_KEY_ACTUAL_TIMES, DEFAULT_ACTUAL_TIMES)
    )
    return WindowConfig(
        reftz=reftz,
        window_start=window_start,
        window_length_hours=window_length_hours,
        actual_times=actual_times,
    )
