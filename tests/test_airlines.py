"""Tests for roster_generator.airlines module.

Covers
------
- generate_airlines(): sys.exit(1) on missing initial-conditions file,
  output file creation, suffix support, column schema, correct extraction
  of unique AC_OPER values, alphabetical sorting.

Physical tests
--------------
- airline_id column contains only non-empty strings.
- No duplicate airline_id rows in the output.
- Sorted alphabetically (expected physical invariant from the catalogue).
- An input with N distinct operators produces exactly N output rows.
- An airline appearing multiple times in initial_conditions is deduplicated
  to a single row.
"""

import pandas as pd
import pytest

from roster_generator.auxiliary.airlines import generate_airlines, AIRLINE_COL
from roster_generator.config import PipelineConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_initial_conditions(*operators) -> pd.DataFrame:
    """Build a minimal initial_conditions DataFrame with given AC_OPER values."""
    return pd.DataFrame({
        AIRLINE_COL: list(operators),
        "AC_REG": [f"EC-{i:03d}" for i in range(len(operators))],
        "DEP_ICAO": "LEMD",
    })


def _write_csv(path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _make_config(tmp_path, suffix="") -> PipelineConfig:
    ic = _make_initial_conditions("IBE", "VLG", "IBE", "RYR")
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

SAMPLE_IC = _make_initial_conditions("IBE", "RYR", "VLG", "IBE", "RYR")


# ---------------------------------------------------------------------------
# generate_airlines — software tests
# ---------------------------------------------------------------------------

class TestGenerateAirlinesSoftware:

    def test_exits_when_initial_conditions_missing(self, tmp_path):
        """sys.exit(1) must be raised when the initial_conditions file is absent."""
        cfg = PipelineConfig(
            schedule_file=tmp_path / "schedule.csv",
            analysis_dir=tmp_path / "analysis",
            output_dir=tmp_path / "output",
        )
        with pytest.raises(SystemExit) as exc_info:
            generate_airlines(cfg)
        assert exc_info.value.code == 1

    def test_creates_output_file(self, tmp_path):
        """Output CSV must be created inside output_dir."""
        cfg = _make_config(tmp_path)
        generate_airlines(cfg)
        assert (tmp_path / "output" / "airlines.csv").exists()

    def test_suffix_in_filename(self, tmp_path):
        """When suffix is set it must appear in the output filename."""
        cfg = _make_config(tmp_path, suffix="_42")
        generate_airlines(cfg)
        assert (tmp_path / "output" / "airlines_42.csv").exists()

    def test_output_not_in_analysis_dir(self, tmp_path):
        """Output must land in output_dir, not analysis_dir."""
        cfg = _make_config(tmp_path)
        generate_airlines(cfg)
        assert not (tmp_path / "analysis" / "airlines.csv").exists()

    def test_output_column_schema(self, tmp_path):
        """Output CSV must have exactly one column: airline_id."""
        cfg = _make_config(tmp_path)
        generate_airlines(cfg)
        df = pd.read_csv(tmp_path / "output" / "airlines.csv")
        assert list(df.columns) == ["airline_id"]

    def test_deduplicates_operators(self, tmp_path):
        """Repeated AC_OPER values must produce a single output row each."""
        ic = _make_initial_conditions("IBE", "IBE", "IBE")
        ic_path = tmp_path / "analysis" / "initial_conditions.csv"
        _write_csv(ic_path, ic)
        cfg = PipelineConfig(
            schedule_file=tmp_path / "schedule.csv",
            analysis_dir=tmp_path / "analysis",
            output_dir=tmp_path / "output",
        )
        generate_airlines(cfg)
        df = pd.read_csv(tmp_path / "output" / "airlines.csv")
        assert len(df) == 1
        assert df.iloc[0]["airline_id"] == "IBE"

    def test_all_unique_operators_present(self, tmp_path):
        """Every distinct AC_OPER value must appear in the output."""
        ic = _make_initial_conditions("IBE", "VLG", "RYR")
        ic_path = tmp_path / "analysis" / "initial_conditions.csv"
        _write_csv(ic_path, ic)
        cfg = PipelineConfig(
            schedule_file=tmp_path / "schedule.csv",
            analysis_dir=tmp_path / "analysis",
            output_dir=tmp_path / "output",
        )
        generate_airlines(cfg)
        df = pd.read_csv(tmp_path / "output" / "airlines.csv")
        assert set(df["airline_id"]) == {"IBE", "VLG", "RYR"}

    def test_output_row_count_matches_distinct_operators(self, tmp_path):
        """Number of output rows equals the number of unique AC_OPER values."""
        ic = _make_initial_conditions("IBE", "VLG", "RYR", "IBE", "RYR")
        ic_path = tmp_path / "analysis" / "initial_conditions.csv"
        _write_csv(ic_path, ic)
        cfg = PipelineConfig(
            schedule_file=tmp_path / "schedule.csv",
            analysis_dir=tmp_path / "analysis",
            output_dir=tmp_path / "output",
        )
        generate_airlines(cfg)
        df = pd.read_csv(tmp_path / "output" / "airlines.csv")
        assert len(df) == 3  # IBE, VLG, RYR

    def test_single_operator_input(self, tmp_path):
        """A single-operator input produces exactly one output row."""
        ic = _make_initial_conditions("BAW")
        ic_path = tmp_path / "analysis" / "initial_conditions.csv"
        _write_csv(ic_path, ic)
        cfg = PipelineConfig(
            schedule_file=tmp_path / "schedule.csv",
            analysis_dir=tmp_path / "analysis",
            output_dir=tmp_path / "output",
        )
        generate_airlines(cfg)
        df = pd.read_csv(tmp_path / "output" / "airlines.csv")
        assert len(df) == 1
        assert df.iloc[0]["airline_id"] == "BAW"

    def test_output_dir_created_if_absent(self, tmp_path):
        """generate_airlines must create output_dir if it does not exist."""
        output_dir = tmp_path / "new_output_dir"
        assert not output_dir.exists()
        ic_path = tmp_path / "analysis" / "initial_conditions.csv"
        _write_csv(ic_path, _make_initial_conditions("IBE"))
        cfg = PipelineConfig(
            schedule_file=tmp_path / "schedule.csv",
            analysis_dir=tmp_path / "analysis",
            output_dir=output_dir,
        )
        generate_airlines(cfg)
        assert output_dir.exists()


# ---------------------------------------------------------------------------
# generate_airlines — physical tests
# ---------------------------------------------------------------------------

class TestGenerateAirlinesPhysical:

    def _run(self, tmp_path, *operators) -> pd.DataFrame:
        """Write initial_conditions, run generate_airlines, return output CSV."""
        ic = _make_initial_conditions(*operators)
        ic_path = tmp_path / "analysis" / "initial_conditions.csv"
        _write_csv(ic_path, ic)
        cfg = PipelineConfig(
            schedule_file=tmp_path / "schedule.csv",
            analysis_dir=tmp_path / "analysis",
            output_dir=tmp_path / "output",
        )
        generate_airlines(cfg)
        return pd.read_csv(tmp_path / "output" / "airlines.csv")

    def test_no_duplicate_airline_ids(self, tmp_path):
        """Each airline_id must appear exactly once in the output."""
        df = self._run(tmp_path, "IBE", "VLG", "RYR", "IBE")
        assert df["airline_id"].nunique() == len(df)

    def test_airline_ids_non_empty(self, tmp_path):
        """No airline_id in the output may be null or empty."""
        df = self._run(tmp_path, "IBE", "VLG", "RYR")
        assert df["airline_id"].notna().all()
        assert (df["airline_id"].str.strip() != "").all()

    def test_output_sorted_alphabetically(self, tmp_path):
        """Airline IDs must be sorted in ascending alphabetical order."""
        df = self._run(tmp_path, "VLG", "RYR", "IBE", "BAW")
        ids = df["airline_id"].tolist()
        assert ids == sorted(ids), f"Expected sorted order, got: {ids}"

    def test_n_distinct_operators_yields_n_rows(self, tmp_path):
        """Output row count must equal the number of distinct input operators."""
        operators = ["IBE", "VLG", "RYR", "BAW", "EZY"]
        df = self._run(tmp_path, *operators)
        assert len(df) == len(set(operators))

    def test_operator_appearing_many_times_counts_once(self, tmp_path):
        """An operator repeated 100 times in the input yields exactly 1 output row."""
        df = self._run(tmp_path, *["IBE"] * 100)
        assert len(df) == 1

    def test_all_input_operators_represented(self, tmp_path):
        """Every distinct operator from the input must be present in the output."""
        operators = ["IBE", "VLG", "RYR", "BAW"]
        df = self._run(tmp_path, *operators)
        assert set(df["airline_id"]) == set(operators)
