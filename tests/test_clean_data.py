"""Tests for roster_generator.clean_data module.

Covers
------
- calculate_local_time(): summer/winter offsets, negative UTC offsets,
  tz-naive output, missing/invalid timezones, index preservation, empty input.
- clean(): output file creation, column renaming/dropping, invalid airport
  filtering, missing registration filtering, date parsing (dayfirst),
  AC_TYPE normalisation, local time offsets, AC_WAKE mapping
  output date format, tolerance for missing optional columns.
"""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from roster_generator.data_cleaning.clean_data import calculate_local_time, clean


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw_csv(tmp_path, rows, filename="raw.csv"):
    """Build a EUROCONTROL-style raw CSV from a list of row dicts.

    Mirrors the real format: DD-MM-YYYY HH:MM:SS dates, extra columns that
    clean() should ignore, and quoted fields.
    """
    base = {
        "ECTRL ID": "0",
        "ADEP Latitude": "0",
        "ADEP Longitude": "0",
        "ADES Latitude": "0",
        "ADES Longitude": "0",
        "ICAO Flight Type": "S",
        "STATFOR Market Segment": "Traditional Scheduled",
        "Requested FL": "350",
        "Actual Distance Flown (nm)": "1000",
    }
    full_rows = [{**base, **r} for r in rows]
    df = pd.DataFrame(full_rows)
    p = tmp_path / filename
    df.to_csv(p, index=False)
    return p


VALID_FLIGHT = {
    "ADEP": "LEMD",
    "ADES": "EGLL",
    "FILED OFF BLOCK TIME": "01-09-2023 08:00:00",
    "FILED ARRIVAL TIME": "01-09-2023 10:30:00",
    "ACTUAL OFF BLOCK TIME": "01-09-2023 08:05:00",
    "ACTUAL ARRIVAL TIME": "01-09-2023 10:35:00",
    "AC Type": "A320",
    "AC Operator": "IBE",
    "AC Registration": "EC-ABC",
}

VALID_FLIGHT_2 = {
    **VALID_FLIGHT,
    "ADEP": "EGLL",
    "ADES": "LFPG",
    "AC Registration": "G-EUAB",
    "AC Operator": "BAW",
    "AC Type": "A319",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def raw_csv_path(tmp_path):
    """Minimal 3-row raw CSV with realistic EUROCONTROL format."""
    return _make_raw_csv(tmp_path, [
        VALID_FLIGHT,
        VALID_FLIGHT_2,
        {**VALID_FLIGHT, "ADEP": "LFPG", "ADES": "LEMD",
         "AC Registration": "EC-DEF", "AC Type": "B738"},
    ])


# ---------------------------------------------------------------------------
# calculate_local_time
# ---------------------------------------------------------------------------

class TestCalculateLocalTime:

    def test_summer_offset_madrid_and_london(self):
        """Madrid CEST = UTC+2, London BST = UTC+1 in September."""
        df = pd.DataFrame({
            "utc": pd.to_datetime([
                "2023-09-01 10:00",
                "2023-09-01 10:00",
            ]).tz_localize("UTC"),
            "tz": ["Europe/Madrid", "Europe/London"],
        })
        result = calculate_local_time(df, "utc", "tz")
        assert result.iloc[0] == pd.Timestamp("2023-09-01 12:00")
        assert result.iloc[1] == pd.Timestamp("2023-09-01 11:00")

    def test_winter_offset_differs_from_summer(self):
        """Madrid CET = UTC+1 in January (vs UTC+2 in summer)."""
        df = pd.DataFrame({
            "utc": pd.to_datetime(["2023-01-15 10:00"]).tz_localize("UTC"),
            "tz": ["Europe/Madrid"],
        })
        result = calculate_local_time(df, "utc", "tz")
        assert result.iloc[0].hour == 11  # CET = UTC+1

    def test_negative_utc_offset(self):
        """Los Angeles in summer (PDT) is UTC-7."""
        df = pd.DataFrame({
            "utc": pd.to_datetime(["2023-09-01 00:00"]).tz_localize("UTC"),
            "tz": ["America/Los_Angeles"],
        })
        result = calculate_local_time(df, "utc", "tz")
        # 00:00 UTC → 17:00 previous day in LA (UTC-7)
        assert result.iloc[0] == pd.Timestamp("2023-08-31 17:00")

    def test_result_is_timezone_naive(self):
        """Returned series must be tz-naive (tz_localize(None) applied)."""
        df = pd.DataFrame({
            "utc": pd.to_datetime(["2023-09-01 10:00"]).tz_localize("UTC"),
            "tz": ["Europe/Madrid"],
        })
        result = calculate_local_time(df, "utc", "tz")
        assert result.iloc[0].tzinfo is None

    def test_missing_utc_column_returns_all_nat(self):
        """If the UTC column doesn't exist, every entry should be NaT."""
        df = pd.DataFrame({
            "utc": pd.to_datetime(["2023-09-01 10:00"]).tz_localize("UTC"),
            "tz": ["Europe/Madrid"],
        })
        result = calculate_local_time(df, "nonexistent_col", "tz")
        assert result.isna().all()

    def test_nan_timezone_produces_nat(self):
        """Rows where the timezone is NaN should remain NaT."""
        df = pd.DataFrame({
            "utc": pd.to_datetime(["2023-09-01 12:00"]).tz_localize("UTC"),
            "tz": [None],
        })
        result = calculate_local_time(df, "utc", "tz")
        assert result.isna().all()

    def test_invalid_tz_string_produces_nat(self):
        """An unrecognisable timezone string should not crash."""
        df = pd.DataFrame({
            "utc": pd.to_datetime(["2023-09-01 12:00"]).tz_localize("UTC"),
            "tz": ["Not/A_Timezone"],
        })
        result = calculate_local_time(df, "utc", "tz")
        assert result.isna().all()

    def test_preserves_original_index(self):
        """Result index must match the input DataFrame's non-default index."""
        idx = [5, 10, 15]
        df = pd.DataFrame({
            "utc": pd.to_datetime([
                "2023-09-01 00:00",
                "2023-09-01 00:00",
                "2023-09-01 00:00",
            ]).tz_localize("UTC"),
            "tz": ["Europe/Madrid"] * 3,
        }, index=idx)
        result = calculate_local_time(df, "utc", "tz")
        assert list(result.index) == idx

    def test_empty_dataframe(self):
        """Empty input should return an empty series without errors."""
        df = pd.DataFrame({
            "utc": pd.Series(dtype="datetime64[ns, UTC]"),
            "tz": pd.Series(dtype=str),
        })
        result = calculate_local_time(df, "utc", "tz")
        assert len(result) == 0

    def test_mixed_valid_and_nan_timezones(self):
        """Valid tz rows are converted; NaN tz rows stay NaT."""
        df = pd.DataFrame({
            "utc": pd.to_datetime([
                "2023-09-01 10:00",
                "2023-09-01 10:00",
            ]).tz_localize("UTC"),
            "tz": ["Europe/Madrid", None],
        })
        result = calculate_local_time(df, "utc", "tz")
        assert result.iloc[0] == pd.Timestamp("2023-09-01 12:00")
        assert pd.isna(result.iloc[1])


# ---------------------------------------------------------------------------
# clean — integration tests
# ---------------------------------------------------------------------------

class TestClean:

    def test_output_file_created(self, raw_csv_path, tmp_path):
        out = tmp_path / "out" / "clean.csv"
        clean(raw_csv_path, out)
        assert out.exists()

    def test_creates_nested_parent_directories(self, raw_csv_path, tmp_path):
        out = tmp_path / "a" / "b" / "c" / "clean.csv"
        clean(raw_csv_path, out)
        assert out.exists()

    def test_extra_source_columns_discarded(self, raw_csv_path, tmp_path):
        """Extra EUROCONTROL columns (ECTRL ID, lat, lon, etc.) must not
        appear in the output."""
        out = tmp_path / "clean.csv"
        clean(raw_csv_path, out)
        df = pd.read_csv(out)
        for col in ["ECTRL ID", "ADEP Latitude", "Requested FL",
                     "STATFOR Market Segment"]:
            assert col not in df.columns

    def test_columns_renamed_correctly(self, raw_csv_path, tmp_path):
        """Output must match the strict schema exactly (names and order)."""
        out = tmp_path / "clean.csv"
        clean(raw_csv_path, out)
        df = pd.read_csv(out)

        expected = [
            "DEP_ICAO",
            "ARR_ICAO",
            "STD_REFTZ",
            "STA_REFTZ",
            "ATD_REFTZ",
            "ATA_REFTZ",
            "AC_TYPE",
            "AC_OPER",
            "AC_REG",
            "AC_WAKE",
        ]
        assert list(df.columns) == expected

    def test_tz_helper_columns_dropped(self, raw_csv_path, tmp_path):
        """No timezone/local helper columns must appear in the output."""
        out = tmp_path / "clean.csv"
        clean(raw_csv_path, out)
        df = pd.read_csv(out)
        assert "DEP_TZ" not in df.columns
        assert "ARR_TZ" not in df.columns
        assert not any(col.endswith("_LOCAL") for col in df.columns)

    def test_invalid_departure_airport_dropped(self, tmp_path):
        """Rows with a non-existent ICAO departure code are removed."""
        csv = _make_raw_csv(tmp_path, [
            VALID_FLIGHT,
            {**VALID_FLIGHT, "ADEP": "ZZZZ", "AC Registration": "EC-999"},
        ])
        out = tmp_path / "clean.csv"
        clean(csv, out)
        df = pd.read_csv(out)
        assert len(df) == 1
        assert df.iloc[0]["DEP_ICAO"] == "LEMD"

    def test_invalid_arrival_airport_dropped(self, tmp_path):
        """Rows with a non-existent ICAO arrival code are removed."""
        csv = _make_raw_csv(tmp_path, [
            VALID_FLIGHT,
            {**VALID_FLIGHT, "ADES": "XXXX", "AC Registration": "EC-999"},
        ])
        out = tmp_path / "clean.csv"
        clean(csv, out)
        df = pd.read_csv(out)
        assert len(df) == 1

    def test_missing_registration_dropped(self, tmp_path):
        """Rows where AC Registration is NaN should be removed."""
        row_no_reg = {**VALID_FLIGHT, "AC Registration": None}
        # pandas will write empty string → NaN on read
        csv = _make_raw_csv(tmp_path, [VALID_FLIGHT, row_no_reg])
        out = tmp_path / "clean.csv"
        clean(csv, out)
        df = pd.read_csv(out)
        assert len(df) == 1

    def test_dayfirst_date_parsing(self, tmp_path):
        """Dates in DD-MM-YYYY format must be parsed correctly (not MM-DD)."""
        row = {
            **VALID_FLIGHT,
            # 15th of September, NOT September parsed as month=15
            "FILED OFF BLOCK TIME": "15-09-2023 08:00:00",
        }
        csv = _make_raw_csv(tmp_path, [row])
        out = tmp_path / "clean.csv"
        clean(csv, out)
        df = pd.read_csv(out, parse_dates=["STD_REFTZ"])
        assert df.iloc[0]["STD_REFTZ"].month == 9
        assert df.iloc[0]["STD_REFTZ"].day == 15

    def test_ac_type_stripped_and_uppercased(self, tmp_path):
        """AC_TYPE values should be stripped of whitespace and uppercased."""
        row = {**VALID_FLIGHT, "AC Type": "  a320  "}
        csv = _make_raw_csv(tmp_path, [row])
        out = tmp_path / "clean.csv"
        clean(csv, out)
        df = pd.read_csv(out)
        assert df.iloc[0]["AC_TYPE"] == "A320"

    def test_local_time_columns_not_emitted(self, tmp_path):
        """Cleaned output must not contain local-time columns."""
        csv = _make_raw_csv(tmp_path, [VALID_FLIGHT])
        out = tmp_path / "clean.csv"
        clean(csv, out)
        df = pd.read_csv(out)
        for col in ["STD_LOCAL", "ATD_LOCAL", "STA_LOCAL", "ATA_LOCAL"]:
            assert col not in df.columns

    def test_row_count_preserved_for_valid_data(self, raw_csv_path, tmp_path):
        """All rows with valid ICAO + registration should survive cleaning."""
        out = tmp_path / "clean.csv"
        clean(raw_csv_path, out)
        df = pd.read_csv(out)
        assert len(df) == 3

    def test_missing_required_columns_raises_error(self, tmp_path):
        """clean() must fail if any required source column is missing."""
        data = pd.DataFrame({
            "ADEP": ["LEMD"],
            "ADES": ["EGLL"],
            "FILED OFF BLOCK TIME": ["01-09-2023 08:00:00"],
            "FILED ARRIVAL TIME": ["01-09-2023 10:00:00"],
            "ACTUAL OFF BLOCK TIME": ["01-09-2023 08:05:00"],
            "ACTUAL ARRIVAL TIME": ["01-09-2023 10:05:00"],
        })
        csv = tmp_path / "raw.csv"
        data.to_csv(csv, index=False)
        out = tmp_path / "clean.csv"
        with pytest.raises(ValueError, match="Missing required source columns"):
            clean(csv, out)

    def test_ac_wake_column_present(self, tmp_path):
        """When aircraft-list is importable, AC_WAKE column should exist."""
        csv = _make_raw_csv(tmp_path, [VALID_FLIGHT])

        with patch("roster_generator.data_cleaning.clean_data._load_wake_map", return_value={"A320": "M"}):
            out = tmp_path / "clean.csv"
            clean(csv, out)
            df = pd.read_csv(out)
            assert "AC_WAKE" in df.columns

    def test_missing_aircraft_list_raises_error(self, tmp_path):
        """When aircraft-list is not importable, clean() must raise ImportError."""
        csv = _make_raw_csv(tmp_path, [VALID_FLIGHT])
        out = tmp_path / "clean.csv"

        with patch("roster_generator.data_cleaning.clean_data._load_wake_map", side_effect=ImportError):
            with pytest.raises(ImportError, match="aircraft-list not installed"):
                clean(csv, out)

    def test_wake_lm_remapped_to_m(self, tmp_path):
        """L/M wake category should be remapped to M."""
        row = {**VALID_FLIGHT, "AC Type": "TEST"}
        csv = _make_raw_csv(tmp_path, [row])

        with patch("roster_generator.data_cleaning.clean_data._load_wake_map", return_value={"TEST": "L/M"}):
            out = tmp_path / "clean.csv"
            clean(csv, out)
            df = pd.read_csv(out)
            assert df.iloc[0]["AC_WAKE"] == "M"

    def test_wake_mh_remapped_to_h(self, tmp_path):
        """M/H wake category should be remapped to H."""
        row = {**VALID_FLIGHT, "AC Type": "BIG1"}
        csv = _make_raw_csv(tmp_path, [row])

        with patch("roster_generator.data_cleaning.clean_data._load_wake_map", return_value={"BIG1": "M/H"}):
            out = tmp_path / "clean.csv"
            clean(csv, out)
            df = pd.read_csv(out)
            assert df.iloc[0]["AC_WAKE"] == "H"

    def test_output_date_format(self, tmp_path):
        """Timestamps in the CSV should follow YYYY-MM-DD HH:MM:SS."""
        csv = _make_raw_csv(tmp_path, [VALID_FLIGHT])
        out = tmp_path / "clean.csv"
        clean(csv, out)
        # Read as raw strings to check format
        df = pd.read_csv(out, dtype=str)
        std_ref = df.iloc[0]["STD_REFTZ"]
        assert std_ref.startswith("2023-09-01"), f"Unexpected format: {std_ref}"
