"""Tests for roster_generator.config module.

Covers
------
- Construction: required paths, default/custom seed, default/custom suffix.
- __post_init__: string-to-Path coercion for schedule_file, analysis_dir, output_dir.
- analysis_path(): correct directory, suffix inclusion, return type.
- output_path(): correct directory, suffix inclusion, return type,
  differs from analysis_path for the same name.
"""

from pathlib import Path

import pytest

from roster_generator.config import PipelineConfig


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestPipelineConfigInit:

    def test_basic_construction(self, tmp_path):
        """Config can be created with the three required paths."""
        cfg = PipelineConfig(
            schedule_file=tmp_path / "schedule.csv",
            analysis_dir=tmp_path / "analysis",
            output_dir=tmp_path / "output",
        )
        assert cfg.schedule_file == tmp_path / "schedule.csv"
        assert cfg.analysis_dir == tmp_path / "analysis"
        assert cfg.output_dir == tmp_path / "output"

    def test_default_seed(self, tmp_path):
        """Default seed should be 42."""
        cfg = PipelineConfig(
            schedule_file=tmp_path / "s.csv",
            analysis_dir=tmp_path,
            output_dir=tmp_path,
        )
        assert cfg.seed == 42

    def test_custom_seed(self, tmp_path):
        """Seed can be overridden."""
        cfg = PipelineConfig(
            schedule_file=tmp_path / "s.csv",
            analysis_dir=tmp_path,
            output_dir=tmp_path,
            seed=123,
        )
        assert cfg.seed == 123

    def test_default_suffix_empty(self, tmp_path):
        """Default suffix should be an empty string."""
        cfg = PipelineConfig(
            schedule_file=tmp_path / "s.csv",
            analysis_dir=tmp_path,
            output_dir=tmp_path,
        )
        assert cfg.suffix == ""

    def test_custom_suffix(self, tmp_path):
        """Suffix can be overridden."""
        cfg = PipelineConfig(
            schedule_file=tmp_path / "s.csv",
            analysis_dir=tmp_path,
            output_dir=tmp_path,
            suffix="_v2",
        )
        assert cfg.suffix == "_v2"

    def test_default_time_window_values(self, tmp_path):
        """Default REFTZ window config should match UTC 00:00 24h."""
        cfg = PipelineConfig(
            schedule_file=tmp_path / "s.csv",
            analysis_dir=tmp_path,
            output_dir=tmp_path,
        )
        assert cfg.reftz == "UTC"
        assert cfg.window_start == "00:00"
        assert cfg.window_length_hours == 24
        assert cfg.window_start_mins == 0
        assert cfg.window_length_mins == 1440

    def test_custom_time_window_values(self, tmp_path):
        """Custom REFTZ/window values should be stored and derived correctly."""
        cfg = PipelineConfig(
            schedule_file=tmp_path / "s.csv",
            analysis_dir=tmp_path,
            output_dir=tmp_path,
            reftz="Europe/Madrid",
            window_start="06:30",
            window_length_hours=18,
        )
        assert cfg.reftz == "Europe/Madrid"
        assert cfg.window_start == "06:30"
        assert cfg.window_length_hours == 18
        assert cfg.window_start_mins == 390
        assert cfg.window_length_mins == 1080

    def test_invalid_reftz_raises(self, tmp_path):
        """Invalid REFTZ must raise ValueError."""
        with pytest.raises(ValueError, match="Invalid REFTZ"):
            PipelineConfig(
                schedule_file=tmp_path / "s.csv",
                analysis_dir=tmp_path,
                output_dir=tmp_path,
                reftz="Not/A_Timezone",
            )

    def test_invalid_window_start_raises(self, tmp_path):
        """Invalid WINDOW_START format must raise ValueError."""
        with pytest.raises(ValueError, match="WINDOW_START"):
            PipelineConfig(
                schedule_file=tmp_path / "s.csv",
                analysis_dir=tmp_path,
                output_dir=tmp_path,
                window_start="6:30",
            )

    def test_invalid_window_length_raises(self, tmp_path):
        """WINDOW_LENGTH_HOURS outside [1,24] must raise ValueError."""
        with pytest.raises(ValueError, match="WINDOW_LENGTH_HOURS"):
            PipelineConfig(
                schedule_file=tmp_path / "s.csv",
                analysis_dir=tmp_path,
                output_dir=tmp_path,
                window_length_hours=25,
            )


# ---------------------------------------------------------------------------
# __post_init__ — string-to-Path coercion
# ---------------------------------------------------------------------------

class TestPostInit:

    def test_string_schedule_file_coerced_to_path(self):
        """Passing a string for schedule_file should be coerced to Path."""
        cfg = PipelineConfig(
            schedule_file="/tmp/schedule.csv",
            analysis_dir="/tmp/analysis",
            output_dir="/tmp/output",
        )
        assert isinstance(cfg.schedule_file, Path)
        assert cfg.schedule_file == Path("/tmp/schedule.csv")

    def test_string_analysis_dir_coerced_to_path(self):
        """Passing a string for analysis_dir should be coerced to Path."""
        cfg = PipelineConfig(
            schedule_file="/tmp/schedule.csv",
            analysis_dir="/tmp/analysis",
            output_dir="/tmp/output",
        )
        assert isinstance(cfg.analysis_dir, Path)
        assert cfg.analysis_dir == Path("/tmp/analysis")

    def test_string_output_dir_coerced_to_path(self):
        """Passing a string for output_dir should be coerced to Path."""
        cfg = PipelineConfig(
            schedule_file="/tmp/schedule.csv",
            analysis_dir="/tmp/analysis",
            output_dir="/tmp/output",
        )
        assert isinstance(cfg.output_dir, Path)
        assert cfg.output_dir == Path("/tmp/output")

    def test_path_objects_unchanged(self, tmp_path):
        """Passing Path objects should keep them as-is."""
        cfg = PipelineConfig(
            schedule_file=tmp_path / "s.csv",
            analysis_dir=tmp_path / "a",
            output_dir=tmp_path / "o",
        )
        assert isinstance(cfg.schedule_file, Path)
        assert isinstance(cfg.analysis_dir, Path)
        assert isinstance(cfg.output_dir, Path)


# ---------------------------------------------------------------------------
# analysis_path helper
# ---------------------------------------------------------------------------

class TestAnalysisPath:

    def test_returns_csv_in_analysis_dir(self, tmp_path):
        """analysis_path('markov') should return analysis_dir/markov.csv."""
        cfg = PipelineConfig(
            schedule_file=tmp_path / "s.csv",
            analysis_dir=tmp_path / "analysis",
            output_dir=tmp_path / "output",
        )
        result = cfg.analysis_path("markov")
        assert result == tmp_path / "analysis" / "markov.csv"

    def test_includes_suffix(self, tmp_path):
        """When suffix is set, file name should include it."""
        cfg = PipelineConfig(
            schedule_file=tmp_path / "s.csv",
            analysis_dir=tmp_path / "analysis",
            output_dir=tmp_path / "output",
            suffix="_sept",
        )
        result = cfg.analysis_path("turnaround")
        assert result == tmp_path / "analysis" / "turnaround_sept.csv"

    def test_empty_suffix_no_extra_chars(self, tmp_path):
        """With default empty suffix, no underscore or extra chars appear."""
        cfg = PipelineConfig(
            schedule_file=tmp_path / "s.csv",
            analysis_dir=tmp_path / "analysis",
            output_dir=tmp_path / "output",
        )
        assert cfg.analysis_path("fleet").name == "fleet.csv"

    def test_returns_path_object(self, tmp_path):
        """Return type should be a Path."""
        cfg = PipelineConfig(
            schedule_file=tmp_path / "s.csv",
            analysis_dir=tmp_path / "analysis",
            output_dir=tmp_path / "output",
        )
        assert isinstance(cfg.analysis_path("x"), Path)


# ---------------------------------------------------------------------------
# output_path helper
# ---------------------------------------------------------------------------

class TestOutputPath:

    def test_returns_csv_in_output_dir(self, tmp_path):
        """output_path('schedule') should return output_dir/schedule.csv."""
        cfg = PipelineConfig(
            schedule_file=tmp_path / "s.csv",
            analysis_dir=tmp_path / "analysis",
            output_dir=tmp_path / "output",
        )
        result = cfg.output_path("schedule")
        assert result == tmp_path / "output" / "schedule.csv"

    def test_includes_suffix(self, tmp_path):
        """When suffix is set, file name should include it."""
        cfg = PipelineConfig(
            schedule_file=tmp_path / "s.csv",
            analysis_dir=tmp_path / "analysis",
            output_dir=tmp_path / "output",
            suffix="_v3",
        )
        result = cfg.output_path("airports")
        assert result == tmp_path / "output" / "airports_v3.csv"

    def test_empty_suffix_no_extra_chars(self, tmp_path):
        """With default empty suffix, no underscore or extra chars appear."""
        cfg = PipelineConfig(
            schedule_file=tmp_path / "s.csv",
            analysis_dir=tmp_path / "analysis",
            output_dir=tmp_path / "output",
        )
        assert cfg.output_path("fleet").name == "fleet.csv"

    def test_returns_path_object(self, tmp_path):
        """Return type should be a Path."""
        cfg = PipelineConfig(
            schedule_file=tmp_path / "s.csv",
            analysis_dir=tmp_path / "analysis",
            output_dir=tmp_path / "output",
        )
        assert isinstance(cfg.output_path("x"), Path)

    def test_output_path_differs_from_analysis_path(self, tmp_path):
        """Same name should resolve to different directories."""
        cfg = PipelineConfig(
            schedule_file=tmp_path / "s.csv",
            analysis_dir=tmp_path / "analysis",
            output_dir=tmp_path / "output",
        )
        assert cfg.analysis_path("fleet") != cfg.output_path("fleet")
        assert cfg.analysis_path("fleet").parent.name == "analysis"
        assert cfg.output_path("fleet").parent.name == "output"
