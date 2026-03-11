import argparse
import calendar
import csv
from datetime import date
from importlib import resources
from pathlib import Path

import pandas as pd
import airportsdata

REQUIRED_SOURCE_COLUMNS = [
    "ADEP",
    "ADES",
    "FILED OFF BLOCK TIME",
    "FILED ARRIVAL TIME",
    "ACTUAL OFF BLOCK TIME",
    "ACTUAL ARRIVAL TIME",
    "AC Type",
    "AC Operator",
    "AC Registration",
]

FINAL_COLUMNS = [
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


def calculate_local_time(df: pd.DataFrame, utc_col: str, tz_col: str) -> pd.Series:
    """
    Convert a UTC column to local time based on a timezone column.

    Iterates by unique timezone groups for performance.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the UTC timestamps and timezone info.
    utc_col : str
        Name of the column with UTC datetime values.
    tz_col : str
        Name of the column with IANA timezone strings (e.g. 'Europe/Madrid').

    Returns
    -------
    pandas.Series
        Series of timezone-naive datetimes representing local times.
        Entries with missing or invalid timezones are left as NaT.
    """
    local_series = pd.Series(pd.NaT, index=df.index)

    if utc_col not in df.columns:
        return local_series

    for tz_name, group in df.groupby(tz_col):
        if pd.isna(tz_name):
            continue
        try:
            converted = group[utc_col].dt.tz_convert(tz_name)
            converted = converted.dt.tz_localize(None)
            local_series.loc[group.index] = converted
        except Exception as e:
            print(f"[Clean Data] Error converting timezone {tz_name}: {e}")

    return local_series


def _load_wake_map() -> dict[str, str | None]:
    """
    Load ICAO wake categories from aircraft-list package data.

    Uses importlib.resources to avoid aircraft_list.aircraft_models(), which
    relies on deprecated pkg_resources.
    """
    try:
        data_file = resources.files("aircraft_list").joinpath("aircraft_model_list.csv")
    except Exception as exc:  # pragma: no cover - exercised via clean() ImportError path
        raise ImportError("aircraft-list package resources are unavailable") from exc

    try:
        with data_file.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            wake_map: dict[str, str | None] = {}
            for row in reader:
                icao = (row.get("ICAO Designator") or row.get("icao") or "").strip().upper()
                wake = (row.get("Wake Turbulence") or row.get("wake") or "").strip().upper()
                if icao:
                    wake_map[icao] = wake or None
    except FileNotFoundError as exc:  # pragma: no cover - exercised via clean() ImportError path
        raise ImportError("aircraft-list data file not found") from exc

    return wake_map


def clean(dirty_file: str | Path, clean_file: str | Path) -> None:
    """
    Clean EUROCONTROL flight data and save the result.

    Loads raw CSV data, validates airport codes, converts timestamps to UTC,
    adds wake turbulence categories, and exports a strict cleaned CSV.

    Parameters
    ----------
    dirty_file : str or pathlib.Path
        Path to the raw EUROCONTROL CSV file.
    clean_file : str or pathlib.Path
        Path where the cleaned CSV will be written. Parent directories are
        created automatically if they don't exist.
    """
    input_file = Path(dirty_file)
    output_file = Path(clean_file)

    # 1. Setup and Loading
    print("[Clean Data] Loading airport database...")
    airports = airportsdata.load('ICAO')

    print(f"[Clean Data] Loading flight data from {input_file}...")
    df = pd.read_csv(input_file)

    # 2. Validate source schema
    missing_cols = [col for col in REQUIRED_SOURCE_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"[Clean Data] Missing required source columns: {missing_cols}")

    df_clean = df[REQUIRED_SOURCE_COLUMNS].copy()

    # 3. Data Cleaning & Formatting
    # A. Process Dates (UTC)
    time_cols = [
        "FILED OFF BLOCK TIME",
        "FILED ARRIVAL TIME",
        "ACTUAL OFF BLOCK TIME",
        "ACTUAL ARRIVAL TIME",
    ]

    print("[Clean Data] Processing UTC timestamps...")
    for col in time_cols:
        df_clean[col] = pd.to_datetime(df_clean[col], dayfirst=True, errors="coerce")
        if df_clean[col].dt.tz is None:
            df_clean[col] = df_clean[col].dt.tz_localize("UTC")
        else:
            df_clean[col] = df_clean[col].dt.tz_convert("UTC")

    # B. Validate Airports
    print("[Clean Data] Validating airports against airportsdata library...")
    initial_count = len(df_clean)
    valid_icaos = set(airports.keys())

    df_clean = df_clean[
        df_clean['ADEP'].isin(valid_icaos) &
        df_clean['ADES'].isin(valid_icaos)
    ]

    print(f"[Clean Data] Dropped {initial_count - len(df_clean)} rows with invalid or unknown airport codes.")

    # C. Clean Aircraft Registration
    df_clean = df_clean.dropna(subset=["AC Registration"])

    # 4. Rename Columns
    print("[Clean Data] Renaming columns...")
    rename_map = {
        "ADEP": "DEP_ICAO",
        "ADES": "ARR_ICAO",
        "FILED OFF BLOCK TIME": "STD_REFTZ",
        "ACTUAL OFF BLOCK TIME": "ATD_REFTZ",
        "FILED ARRIVAL TIME": "STA_REFTZ",
        "ACTUAL ARRIVAL TIME": "ATA_REFTZ",
        "AC Type": "AC_TYPE",
        "AC Operator": "AC_OPER",
        "AC Registration": "AC_REG",
    }
    df_clean = df_clean.rename(columns=rename_map)

    # Add AC_WAKE (ICAO Wake Turbulence Category)
    print("[Clean Data] Adding AC_WAKE from aircraft-list (ICAO Doc 8643)...")

    if "AC_TYPE" in df_clean.columns:
        df_clean["AC_TYPE"] = df_clean["AC_TYPE"].astype(str).str.strip().str.upper()

        try:
            wake_map = _load_wake_map()

            df_clean["AC_WAKE"] = df_clean["AC_TYPE"].map(wake_map)
            df_clean["AC_WAKE"] = df_clean["AC_WAKE"].replace({"L/M": "M", "M/H": "H"})

            # Drop missing or empty AC_WAKE
            initial_count_before_wake = len(df_clean)
            df_clean = df_clean.dropna(subset=["AC_WAKE"])
            df_clean = df_clean[df_clean["AC_WAKE"].astype(str).str.strip() != ""]
            dropped_wake = initial_count_before_wake - len(df_clean)
            if dropped_wake > 0:
                print(f"[Clean Data] Dropped {dropped_wake} rows with missing or empty AC_WAKE.")

            print("[Clean Data] AC_WAKE added.")
        except ImportError as exc:
            raise ImportError(
                "[Clean Data] aircraft-list not installed. Run: pip install aircraft-list"
            ) from exc
    else:
        raise ValueError("[Clean Data] Column AC_TYPE not present in cleaned data.")

    # 5. Enforce strict output schema (exact columns and order)
    missing_final = [col for col in FINAL_COLUMNS if col not in df_clean.columns]
    if missing_final:
        raise ValueError(f"[Clean Data] Missing required output columns: {missing_final}")
    df_clean = df_clean[FINAL_COLUMNS].copy()

    # 6. Export
    print(f"[Clean Data] Saving cleaned data to {output_file}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_file, index=False, date_format='%Y-%m-%d %H:%M:%S')

    print("[Clean Data] Done.")
    print(f"[Clean Data] Final dataset contains {len(df_clean)} flights.")


def main():
    parser = argparse.ArgumentParser(description="Clean EUROCONTROL flight data.")
    parser.add_argument("--yyyymm", required=True,
                        help="Year-month in YYYYMM format, e.g. 202309")
    args = parser.parse_args()

    year = int(args.yyyymm[:4])
    month = int(args.yyyymm[4:6])
    last_day = calendar.monthrange(year, month)[1]
    month_name = date(year, month, 1).strftime("%B").lower()

    start = f"{year}{month:02d}01"
    end = f"{year}{month:02d}{last_day:02d}"

    cwd = Path.cwd()
    INPUT_FILE = cwd / "data" / f"Flights_{start}_{end}.csv"
    OUTPUT_FILE = cwd / "computed" / "clean_data" / f"{month_name}{year}.csv"

    clean(dirty_file=INPUT_FILE, clean_file=OUTPUT_FILE)


if __name__ == "__main__":
    main()
