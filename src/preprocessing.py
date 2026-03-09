import pandas as pd

from src.config import (
    TS_COL,
    SOC_COL,
    X_RAW,
    CURRENT_COL,
    VOLTAGE_COL,
    SOH_COL,
    TIMESTAMP_FORMAT,
)


def select_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    need = [TS_COL, SOC_COL] + X_RAW + ["__source_file"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns:\n" + "\n".join(missing))

    return df[need].copy()


def parse_and_sort_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    df[TS_COL] = pd.to_datetime(
        df[TS_COL],
        format=TIMESTAMP_FORMAT,
        errors="coerce",
    )
    df = df.dropna(subset=[TS_COL]).sort_values(TS_COL).reset_index(drop=True)
    return df


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in [SOC_COL] + X_RAW:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def drop_rows_missing_core_signals(df: pd.DataFrame) -> pd.DataFrame:
    core = [SOC_COL, CURRENT_COL, VOLTAGE_COL, SOH_COL]
    return df.dropna(subset=core).reset_index(drop=True)


def prepare_base_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all basic preprocessing steps.
    """
    df = select_required_columns(df)
    df = parse_and_sort_timestamp(df)
    df = convert_numeric_columns(df)
    df = drop_rows_missing_core_signals(df)
    return df