import pandas as pd

from src.config import (
    TS_COL,
    X_RAW,
    CURRENT_COL,
    VOLTAGE_COL,
    SOH_COL,
    TIMESTAMP_FORMAT,
    TEMP_SENSOR_COLS,
    GLOBAL_TEMP_FEATURES,
    VOLTAGE_SPREAD_FEATURES,
)


def select_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the columns needed for anomaly detection.
    """
    need = [TS_COL] + X_RAW + ["__source_file"]
    missing = [c for c in need if c not in df.columns]

    if missing:
        raise ValueError("Missing required columns:\n" + "\n".join(missing))

    return df[need].copy()


def parse_and_sort_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse timestamp and sort chronologically.
    """
    df[TS_COL] = pd.to_datetime(
        df[TS_COL],
        format=TIMESTAMP_FORMAT,
        errors="coerce",
    )
    df = df.dropna(subset=[TS_COL]).sort_values(TS_COL).reset_index(drop=True)
    return df


def get_numeric_columns() -> list[str]:
    """
    Build a list of columns that should be converted to numeric.
    Excludes categorical/state columns like batteryState, stackStatus, etc.
    """
    numeric_cols = (
        TEMP_SENSOR_COLS
        + GLOBAL_TEMP_FEATURES
        + VOLTAGE_SPREAD_FEATURES
        + [
            CURRENT_COL,
            VOLTAGE_COL,
            SOH_COL,
        ]
    )

    # keep only unique values while preserving order
    seen = set()
    unique_numeric_cols = []
    for col in numeric_cols:
        if col not in seen:
            unique_numeric_cols.append(col)
            seen.add(col)

    return unique_numeric_cols


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all anomaly-relevant numeric columns to numeric dtype.
    """
    numeric_cols = get_numeric_columns()

    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def drop_rows_missing_core_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows missing the minimum critical signals needed for anomaly detection.

    For temperature anomaly detection, the truly critical signals are:
    - current
    - voltage
    - SOH
    - at least the temperature sensors
    """
    core = [CURRENT_COL, VOLTAGE_COL, SOH_COL]

    # require all main core context signals
    df = df.dropna(subset=core).reset_index(drop=True)

    # require at least one valid temperature sensor row
    temp_available_mask = df[TEMP_SENSOR_COLS].notna().any(axis=1)
    df = df[temp_available_mask].reset_index(drop=True)

    return df


def prepare_base_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all basic preprocessing steps for anomaly detection.
    """
    df = select_required_columns(df)
    df = parse_and_sort_timestamp(df)
    df = convert_numeric_columns(df)
    df = drop_rows_missing_core_signals(df)
    return df