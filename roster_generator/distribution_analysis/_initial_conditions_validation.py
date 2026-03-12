"""Validation checks for generated initial-conditions DataFrames."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd


def validate_initial_conditions(
    ic_df: pd.DataFrame,
    get_phys_ta_min: Callable[[str, str], int],
) -> None:
    """Run sanity checks on generated synthetic initial conditions."""
    if ic_df["AC_REG"].duplicated().any():
        dup = ic_df[ic_df["AC_REG"].duplicated()]["AC_REG"].head(5).tolist()
        raise ValueError(f"Duplicate AC_REG found: {dup}")

    has_prior = ic_df["PRIOR_STD_REFTZ_MINS"].notna()
    if (ic_df.loc[has_prior, "PRIOR_STD_REFTZ_MINS"].astype(float) >= 0).any():
        raise ValueError("Validation failed: prior rows with PRIOR_STD_REFTZ_MINS >= 0")

    non_prior_only = ic_df["PRIOR_ONLY"].astype(int) != 1
    required_cols = ["ORIGIN", "DEST", "STD_REFTZ_MINS", "STA_REFTZ_MINS"]
    for col in required_cols:
        if ic_df.loc[non_prior_only, col].isna().any():
            raise ValueError(f"Validation failed: missing {col} in non-prior-only rows")
    if (ic_df.loc[non_prior_only, "ORIGIN"].astype(str).str.strip() == "").any():
        raise ValueError("Validation failed: blank ORIGIN in non-prior-only rows")
    if (ic_df.loc[non_prior_only, "DEST"].astype(str).str.strip() == "").any():
        raise ValueError("Validation failed: blank DEST in non-prior-only rows")

    with_prior_and_first = (
        (ic_df["PRIOR_STD_REFTZ_MINS"].notna())
        & (ic_df["PRIOR_ONLY"].astype(int) != 1)
    )

    for row in ic_df[with_prior_and_first].itertuples(index=False):
        key = (str(row.AC_OPER), str(row.AC_WAKE))
        min_ta = get_phys_ta_min(str(row.AC_OPER), str(row.AC_WAKE))
        if int(row.STD_REFTZ_MINS) - int(row.PRIOR_STA_REFTZ_MINS) < min_ta:
            raise ValueError(
                "Validation failed: first_STD - prior_STA below phys_ta for "
                f"AC_REG={row.AC_REG}, key={key}, min_ta={min_ta}"
            )
