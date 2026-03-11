"""Tests for roster_generator.fleet module.

Covers
------
- generate_fleet(): sys.exit(1) on missing initial-conditions file,
  sys.exit(1) on missing required columns, output file creation,
  suffix support, column schema, deduplication by AC_REG.

Physical tests
--------------
- AC_REG values are non-empty strings (no null registrations).
- No duplicate AC_REG rows in the output (each aircraft appears once).
- AC_WAKE values belong to a finite, known set (H, M, L, J).
- AC_OPER values are non-empty strings.
- An input with N distinct registrations produces exactly N output rows.
- A registration appearing multiple times in initial_conditions is
  deduplicated to a single row.
- Output is sorted by AC_REG (deterministic order).
"""

from __future__ import annotations

import pandas as pd
import pytest

from roster_generator.auxiliary.fleet import generate_fleet, REQUIRED_COLS
from roster_generator.config import PipelineConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Valid ICAO wake-turbulence categories used in the EUR scenario (ICAO Doc 8643).
VALID_WAKE_CATEGORIES = {"H", "M", "L", "J"}


def _make_initial_conditions(*rows) -> pd.DataFrame:
    """Build a minimal initial_conditions DataFrame.

    Each element of *rows* is a (AC_REG, AC_OPER, AC_WAKE) tuple.
    """
    return pd.DataFrame(rows, columns=["AC_REG", "AC_OPER", "AC_WAKE"])


def _write_csv(path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _make_config(tmp_path, ic: pd.DataFrame | None = None, suffix: str = "") -> PipelineConfig:
    """Write a default initial_conditions file and return a PipelineConfig."""
    if ic is None:
        ic = _make_initial_conditions(
            ("EC-001", "IBE", "M"),
            ("EC-002", "VLG", "M"),
            ("EC-003", "IBE", "H"),
            ("EC-004", "RYR", "L"),
        )
    ic_path = tmp_path / "analysis" / f"initial_conditions{suffix}.csv"
    _write_csv(ic_path, ic)
    return PipelineConfig(
        schedule_file=tmp_path / "schedule.csv",
        analysis_dir=tmp_path / "analysis",
        output_dir=tmp_path / "output",
        suffix=suffix,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_IC = _make_initial_conditions(
    ("EC-001", "IBE", "M"),
    ("EC-002", "VLG", "M"),
    ("EC-001", "IBE", "M"),   # duplicate registration
    ("EC-003", "RYR", "H"),
)


# ---------------------------------------------------------------------------
# generate_fleet — software tests
# ---------------------------------------------------------------------------

class TestGenerateFleetSoftware:

    def test_exits_when_initial_conditions_missing(self, tmp_path):
        """sys.exit(1) must be raised when the initial_conditions file is absent."""
        cfg = PipelineConfig(
            schedule_file=tmp_path / "schedule.csv",
            analysis_dir=tmp_path / "analysis",
            output_dir=tmp_path / "output",
        )
        with pytest.raises(SystemExit) as exc_info:
            generate_fleet(cfg)
        assert exc_info.value.code == 1

    def test_exits_when_required_columns_missing(self, tmp_path):
        """sys.exit(1) must be raised when the initial_conditions file lacks
        one or more of AC_REG, AC_OPER, AC_WAKE."""
        # File exists but has none of the required columns
        ic = pd.DataFrame({"SOME_COL": ["foo", "bar"]})
        ic_path = tmp_path / "analysis" / "initial_conditions.csv"
        _write_csv(ic_path, ic)
        cfg = PipelineConfig(
            schedule_file=tmp_path / "schedule.csv",
            analysis_dir=tmp_path / "analysis",
            output_dir=tmp_path / "output",
        )
        with pytest.raises(SystemExit) as exc_info:
            generate_fleet(cfg)
        assert exc_info.value.code == 1

    def test_exits_when_single_required_column_missing(self, tmp_path):
        """sys.exit(1) must be raised even when only one required column is absent."""
        ic = pd.DataFrame({"AC_REG": ["EC-001"], "AC_OPER": ["IBE"]})
        # AC_WAKE is missing
        ic_path = tmp_path / "analysis" / "initial_conditions.csv"
        _write_csv(ic_path, ic)
        cfg = PipelineConfig(
            schedule_file=tmp_path / "schedule.csv",
            analysis_dir=tmp_path / "analysis",
            output_dir=tmp_path / "output",
        )
        with pytest.raises(SystemExit) as exc_info:
            generate_fleet(cfg)
        assert exc_info.value.code == 1

    def test_creates_output_file(self, tmp_path):
        """Output CSV must be created inside output_dir."""
        cfg = _make_config(tmp_path)
        generate_fleet(cfg)
        assert (tmp_path / "output" / "fleet.csv").exists()

    def test_suffix_in_filename(self, tmp_path):
        """When suffix is set it must appear in the output filename."""
        cfg = _make_config(tmp_path, suffix="_42")
        generate_fleet(cfg)
        assert (tmp_path / "output" / "fleet_42.csv").exists()

    def test_output_not_in_analysis_dir(self, tmp_path):
        """Output must land in output_dir, not analysis_dir."""
        cfg = _make_config(tmp_path)
        generate_fleet(cfg)
        assert not (tmp_path / "analysis" / "fleet.csv").exists()

    def test_output_column_schema(self, tmp_path):
        """Output CSV must have exactly columns: AC_REG, AC_OPER, AC_WAKE."""
        cfg = _make_config(tmp_path)
        generate_fleet(cfg)
        df = pd.read_csv(tmp_path / "output" / "fleet.csv")
        assert list(df.columns) == REQUIRED_COLS

    def test_deduplicates_by_ac_reg(self, tmp_path):
        """A registration appearing more than once must produce a single output row."""
        ic = _make_initial_conditions(
            ("EC-001", "IBE", "M"),
            ("EC-001", "IBE", "M"),
            ("EC-001", "IBE", "M"),
        )
        cfg = _make_config(tmp_path, ic=ic)
        generate_fleet(cfg)
        df = pd.read_csv(tmp_path / "output" / "fleet.csv")
        assert len(df) == 1
        assert df.iloc[0]["AC_REG"] == "EC-001"

    def test_all_unique_registrations_present(self, tmp_path):
        """Every distinct AC_REG must appear in the output."""
        ic = _make_initial_conditions(
            ("EC-001", "IBE", "M"),
            ("EC-002", "VLG", "M"),
            ("EC-003", "RYR", "H"),
        )
        cfg = _make_config(tmp_path, ic=ic)
        generate_fleet(cfg)
        df = pd.read_csv(tmp_path / "output" / "fleet.csv")
        assert set(df["AC_REG"]) == {"EC-001", "EC-002", "EC-003"}

    def test_output_row_count_matches_distinct_registrations(self, tmp_path):
        """Number of output rows equals the number of unique AC_REG values."""
        ic = _make_initial_conditions(
            ("EC-001", "IBE", "M"),
            ("EC-002", "VLG", "M"),
            ("EC-003", "RYR", "H"),
            ("EC-001", "IBE", "M"),   # duplicate
        )
        cfg = _make_config(tmp_path, ic=ic)
        generate_fleet(cfg)
        df = pd.read_csv(tmp_path / "output" / "fleet.csv")
        assert len(df) == 3   # EC-001, EC-002, EC-003

    def test_single_aircraft_input(self, tmp_path):
        """A single-aircraft input produces exactly one output row."""
        ic = _make_initial_conditions(("EC-001", "IBE", "M"))
        cfg = _make_config(tmp_path, ic=ic)
        generate_fleet(cfg)
        df = pd.read_csv(tmp_path / "output" / "fleet.csv")
        assert len(df) == 1
        assert df.iloc[0]["AC_REG"] == "EC-001"

    def test_output_dir_created_if_absent(self, tmp_path):
        """generate_fleet must create output_dir if it does not exist."""
        output_dir = tmp_path / "new_output_dir"
        assert not output_dir.exists()
        ic = _make_initial_conditions(("EC-001", "IBE", "M"))
        ic_path = tmp_path / "analysis" / "initial_conditions.csv"
        _write_csv(ic_path, ic)
        cfg = PipelineConfig(
            schedule_file=tmp_path / "schedule.csv",
            analysis_dir=tmp_path / "analysis",
            output_dir=output_dir,
        )
        generate_fleet(cfg)
        assert output_dir.exists()

    def test_extra_columns_in_input_are_ignored(self, tmp_path):
        """Extra columns in initial_conditions must not appear in the output."""
        ic = _make_initial_conditions(
            ("EC-001", "IBE", "M"),
            ("EC-002", "VLG", "H"),
        )
        ic["EXTRA_COL"] = "irrelevant"
        cfg = _make_config(tmp_path, ic=ic)
        generate_fleet(cfg)
        df = pd.read_csv(tmp_path / "output" / "fleet.csv")
        assert "EXTRA_COL" not in df.columns
        assert list(df.columns) == REQUIRED_COLS


# ---------------------------------------------------------------------------
# generate_fleet — physical tests
# ---------------------------------------------------------------------------

class TestGenerateFleetPhysical:

    def _run(self, tmp_path, *rows) -> pd.DataFrame:
        """Write initial_conditions, run generate_fleet, return output CSV."""
        ic = _make_initial_conditions(*rows)
        cfg = _make_config(tmp_path, ic=ic)
        generate_fleet(cfg)
        return pd.read_csv(tmp_path / "output" / "fleet.csv")

    def test_no_duplicate_registrations(self, tmp_path):
        """Each AC_REG must appear exactly once in the output."""
        df = self._run(
            tmp_path,
            ("EC-001", "IBE", "M"),
            ("EC-002", "VLG", "M"),
            ("EC-001", "IBE", "M"),
        )
        assert df["AC_REG"].nunique() == len(df)

    def test_ac_reg_non_empty(self, tmp_path):
        """No AC_REG in the output may be null or empty."""
        df = self._run(
            tmp_path,
            ("EC-001", "IBE", "M"),
            ("EC-002", "VLG", "H"),
        )
        assert df["AC_REG"].notna().all()
        assert (df["AC_REG"].str.strip() != "").all()

    def test_ac_operator_non_empty(self, tmp_path):
        """No AC_OPER in the output may be null or empty."""
        df = self._run(
            tmp_path,
            ("EC-001", "IBE", "M"),
            ("EC-002", "VLG", "M"),
        )
        assert df["AC_OPER"].notna().all()
        assert (df["AC_OPER"].str.strip() != "").all()

    def test_wake_categories_valid(self, tmp_path):
        """All AC_WAKE values must belong to the set {H, M, L, J}"""
        df = self._run(
            tmp_path,
            ("EC-001", "IBE", "M"),
            ("EC-002", "VLG", "H"),
            ("EC-003", "RYR", "L"),
        )
        invalid = set(df["AC_WAKE"]) - VALID_WAKE_CATEGORIES
        assert not invalid, f"Unexpected wake categories in output: {invalid}"

    def test_output_sorted_by_ac_reg(self, tmp_path):
        """AC_REG must be sorted in ascending order for determinism."""
        df = self._run(
            tmp_path,
            ("EC-003", "VLG", "M"),
            ("EC-001", "IBE", "H"),
            ("EC-002", "RYR", "M"),
        )
        regs = df["AC_REG"].tolist()
        assert regs == sorted(regs), f"Expected sorted AC_REG, got: {regs}"

    def test_n_distinct_registrations_yield_n_rows(self, tmp_path):
        """Output row count must equal the number of distinct AC_REG values."""
        rows = [
            ("EC-001", "IBE", "M"),
            ("EC-002", "VLG", "M"),
            ("EC-003", "RYR", "H"),
            ("EC-004", "BAW", "L"),
            ("EC-005", "EZY", "M"),
        ]
        df = self._run(tmp_path, *rows)
        assert len(df) == len({r[0] for r in rows})

    def test_registration_repeated_many_times_counts_once(self, tmp_path):
        """A registration repeated 100 times yields exactly 1 output row."""
        df = self._run(tmp_path, *[("EC-001", "IBE", "M")] * 100)
        assert len(df) == 1

    def test_all_input_registrations_represented(self, tmp_path):
        """Every distinct AC_REG from the input must be present in the output."""
        rows = [
            ("EC-001", "IBE", "M"),
            ("EC-002", "VLG", "H"),
            ("EC-003", "RYR", "L"),
        ]
        df = self._run(tmp_path, *rows)
        assert set(df["AC_REG"]) == {r[0] for r in rows}
