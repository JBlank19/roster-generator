"""Tests for REFTZ window utilities."""

import pandas as pd
import pytest

from roster_generator.time_window import (
    load_params_yaml,
    minute_of_shifted_day,
    parse_datetime_series_to_reftz,
    resolve_window_config,
    shift_series_by_window_start,
)


class TestParamsYaml:
    def test_missing_file_returns_empty_dict(self, tmp_path):
        params = load_params_yaml(tmp_path / "missing.yaml")
        assert params == {}

    def test_load_top_level_scalar_values(self, tmp_path):
        params_file = tmp_path / "params.yaml"
        params_file.write_text(
            "\n".join(
                [
                    "REFTZ: Europe/Madrid",
                    'WINDOW_START: "06:00"',
                    "WINDOW_LENGTH_HOURS: 18",
                ]
            ),
            encoding="utf-8",
        )
        params = load_params_yaml(params_file)
        assert params["REFTZ"] == "Europe/Madrid"
        assert params["WINDOW_START"] == "06:00"
        assert params["WINDOW_LENGTH_HOURS"] == 18

    def test_resolve_defaults(self):
        cfg = resolve_window_config({})
        assert cfg.reftz == "UTC"
        assert cfg.window_start == "00:00"
        assert cfg.window_length_hours == 24
        assert cfg.window_start_mins == 0
        assert cfg.window_length_mins == 1440

    def test_partial_override_keeps_defaults(self):
        cfg = resolve_window_config({"REFTZ": "Europe/Madrid"})
        assert cfg.reftz == "Europe/Madrid"
        assert cfg.window_start == "00:00"
        assert cfg.window_length_hours == 24

    def test_unknown_keys_raise(self):
        with pytest.raises(ValueError, match="Unknown keys"):
            resolve_window_config({"FOO": "bar"})

    def test_invalid_values_raise(self):
        with pytest.raises(ValueError, match="Invalid REFTZ"):
            resolve_window_config({"REFTZ": "Bad/Timezone"})
        with pytest.raises(ValueError, match="WINDOW_START"):
            resolve_window_config({"WINDOW_START": "6:00"})
        with pytest.raises(ValueError, match="WINDOW_LENGTH_HOURS"):
            resolve_window_config({"WINDOW_LENGTH_HOURS": 0})


class TestShiftedDayUtilities:
    def test_shifted_minutes_follow_window_start(self):
        series = pd.Series(
            [
                "2023-09-01 03:30:00",  # UTC
                "2023-09-01 07:15:00",  # UTC
            ]
        )
        local = parse_datetime_series_to_reftz(series, "Europe/Madrid")
        shifted = shift_series_by_window_start(local, window_start_mins=6 * 60)
        minutes = minute_of_shifted_day(shifted).astype(int).tolist()

        # September in Madrid is UTC+2:
        # 03:30 UTC -> 05:30 local -> shifted by 6h = 23:30 (previous day) => 1410
        # 07:15 UTC -> 09:15 local -> shifted by 6h = 03:15 => 195
        assert minutes == [1410, 195]
