import pandas as pd
import numpy as np

from src.config import (
    TS_COL,
    SOC_COL,
    SOH_COL,
    CURRENT_COL,
    VOLTAGE_COL,
    POWER_COL,
    CELL_VOLT_MAX_COL,
    CELL_VOLT_MIN_COL,
    COMP_CELL_VOLT_MAX_COL,
    COMP_CELL_VOLT_MIN_COL,
    TEMP_SENSOR_COLS,
    GLOBAL_TEMP_FEATURES,
    REST_A,
    ROLL_S,
)


def add_dt_feature(df: pd.DataFrame) -> pd.DataFrame:
    df["dt_s"] = df[TS_COL].diff().dt.total_seconds()
    med_dt = df["dt_s"].dropna().median()
    df["dt_s"] = df["dt_s"].fillna(med_dt)
    df.loc[df["dt_s"] <= 0, "dt_s"] = med_dt
    return df


def add_current_voltage_features(df: pd.DataFrame) -> pd.DataFrame:
    df["current_abs_A"] = df[CURRENT_COL].abs()
    df["current_sq_A2"] = df[CURRENT_COL] ** 2

    df["dI_A"] = df[CURRENT_COL].diff()
    df["dV_V"] = df[VOLTAGE_COL].diff()

    df["dI_dt_Aps"] = df["dI_A"] / df["dt_s"]
    df["dV_dt_Vps"] = df["dV_V"] / df["dt_s"]

    if POWER_COL in df.columns:
        df["power_abs_W"] = df[POWER_COL].abs()

    return df


def add_rest_flag(df: pd.DataFrame) -> pd.DataFrame:
    df["is_rest"] = (df["current_abs_A"] < REST_A).astype(int)
    return df


def add_operating_mode_features(df: pd.DataFrame) -> pd.DataFrame:
    df["is_rest"] = (df["current_abs_A"] < REST_A).astype(int)
    df["is_charging"] = (df[CURRENT_COL] > REST_A).astype(int)
    df["is_discharging"] = (df[CURRENT_COL] < -REST_A).astype(int)
    return df


def add_voltage_spread_features(df: pd.DataFrame) -> pd.DataFrame:
    if CELL_VOLT_MAX_COL in df.columns and CELL_VOLT_MIN_COL in df.columns:
        df["cell_voltage_spread_V"] = df[CELL_VOLT_MAX_COL] - df[CELL_VOLT_MIN_COL]

    if COMP_CELL_VOLT_MAX_COL in df.columns and COMP_CELL_VOLT_MIN_COL in df.columns:
        df["comp_cell_voltage_spread_V"] = (
            df[COMP_CELL_VOLT_MAX_COL] - df[COMP_CELL_VOLT_MIN_COL]
        )

    return df


def add_temperature_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create system-level temperature features from all detailed pack temp sensors.
    """
    available_temp_cols = [c for c in TEMP_SENSOR_COLS if c in df.columns]

    if not available_temp_cols:
        raise ValueError("No detailed pack temperature columns found in dataframe.")

    df["temp_mean_C"] = df[available_temp_cols].mean(axis=1)
    df["temp_std_C"] = df[available_temp_cols].std(axis=1)
    df["temp_max_C"] = df[available_temp_cols].max(axis=1)
    df["temp_min_C"] = df[available_temp_cols].min(axis=1)
    df["temp_spread_C"] = df["temp_max_C"] - df["temp_min_C"]

    return df


def add_temperature_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    df["dT_mean_C"] = df["temp_mean_C"].diff()
    df["dT_max_C"] = df["temp_max_C"].diff()
    df["dT_spread_C"] = df["temp_spread_C"].diff()

    df["dT_mean_dt_Cps"] = df["dT_mean_C"] / df["dt_s"]
    df["dT_max_dt_Cps"] = df["dT_max_C"] / df["dt_s"]
    df["dT_spread_dt_Cps"] = df["dT_spread_C"] / df["dt_s"]

    return df


def add_load_normalized_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features that relate thermal/electrical imbalance to electrical load.

    Important:
    These features are only physically meaningful when current is above a
    minimum threshold. During rest / near-zero current, they are disabled
    to avoid artificial feature explosions.
    """
    load_mask = df["current_abs_A"] >= REST_A

    df["temp_spread_per_current"] = np.where(
        load_mask,
        df["temp_spread_C"] / df["current_abs_A"],
        np.nan,
    )

    df["temp_mean_per_current"] = np.where(
        load_mask,
        df["temp_mean_C"] / df["current_abs_A"],
        np.nan,
    )

    if "cell_voltage_spread_V" in df.columns:
        df["volt_spread_per_current"] = np.where(
            load_mask,
            df["cell_voltage_spread_V"] / df["current_abs_A"],
            np.nan,
        )

    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index(TS_COL)

    win = f"{ROLL_S}s"

    df[f"I_mean_{ROLL_S}s"] = df[CURRENT_COL].rolling(win, min_periods=3).mean()
    df[f"V_mean_{ROLL_S}s"] = df[VOLTAGE_COL].rolling(win, min_periods=3).mean()

    df[f"T_mean_{ROLL_S}s"] = df["temp_mean_C"].rolling(win, min_periods=3).mean()
    df[f"T_std_{ROLL_S}s"] = df["temp_std_C"].rolling(win, min_periods=3).mean()
    df[f"T_spread_{ROLL_S}s"] = df["temp_spread_C"].rolling(win, min_periods=3).mean()

    if "cell_voltage_spread_V" in df.columns:
        df[f"Vspread_mean_{ROLL_S}s"] = (
            df["cell_voltage_spread_V"].rolling(win, min_periods=3).mean()
        )

    df = df.reset_index()
    return df


def get_feature_columns() -> list[str]:
    cols = [
        # Context
        SOH_COL,
        CURRENT_COL,
        VOLTAGE_COL,
        SOC_COL,   # context only, not target
        "dt_s",

        # Current / voltage dynamics
        "current_abs_A",
        "current_sq_A2",
        "dI_dt_Aps",
        "dV_dt_Vps",
        "is_rest",
        "is_charging",
        "is_discharging"

        # Temperature statistics
        "temp_mean_C",
        "temp_std_C",
        "temp_max_C",
        "temp_min_C",
        "temp_spread_C",

        # Temperature dynamics
        "dT_mean_dt_Cps",
        "dT_max_dt_Cps",
        "dT_spread_dt_Cps",

        # Load-normalized temperature behavior
        "temp_spread_per_current",
        "temp_mean_per_current",

        # Rolling features
        f"I_mean_{ROLL_S}s",
        f"V_mean_{ROLL_S}s",
        f"T_mean_{ROLL_S}s",
        f"T_std_{ROLL_S}s",
        f"T_spread_{ROLL_S}s",
    ]

    # Optional features
    optional_cols = [
        POWER_COL,
        "power_abs_W",
        "cell_voltage_spread_V",
        "comp_cell_voltage_spread_V",
        "volt_spread_per_current",
        f"Vspread_mean_{ROLL_S}s",
    ]

    cols.extend(optional_cols)

    # remove duplicates while preserving order
    seen = set()
    unique_cols = []
    for c in cols:
        if c not in seen:
            unique_cols.append(c)
            seen.add(c)

    return unique_cols


def build_feature_table(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Run all feature engineering steps and return:
    - engineered dataframe
    - feature column list
    """
    df = add_dt_feature(df)
    df = add_current_voltage_features(df)
    df = add_rest_flag(df)
    df = add_operating_mode_features(df)
    df = add_voltage_spread_features(df)
    df = add_temperature_statistics(df)
    df = add_temperature_dynamics(df)
    df = add_load_normalized_features(df)
    df = add_rolling_features(df)

    feature_cols = [c for c in get_feature_columns() if c in df.columns]

    # Do not drop rows just because load-normalized features are NaN during rest.
    # Only drop rows if core features are missing.
    core_required_cols = [
        c for c in feature_cols
        if c not in {
            "temp_spread_per_current",
            "temp_mean_per_current",
            "volt_spread_per_current",
        }
    ]

    df_feat = df.dropna(subset=core_required_cols).copy()

    # Fill inactive load-normalized features with 0 so downstream models can use them.
    for col in ["temp_spread_per_current", "temp_mean_per_current", "volt_spread_per_current"]:
        if col in df_feat.columns:
            df_feat[col] = df_feat[col].fillna(0.0)

    print("Rows after feature eng:", len(df_feat))
    print("Feature columns:", feature_cols)

    return df_feat, feature_cols