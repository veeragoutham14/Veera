import pandas as pd

from src.config import (
    TS_COL,
    SOC_COL,
    SOH_COL,
    CURRENT_COL,
    VOLTAGE_COL,
    POWER_COL,
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
    return df


def add_rest_flag(df: pd.DataFrame) -> pd.DataFrame:
    df["is_rest"] = (df["current_abs_A"] < REST_A).astype(int)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index(TS_COL)

    win = f"{ROLL_S}s"
    df[f"I_mean_{ROLL_S}s"] = df[CURRENT_COL].rolling(win, min_periods=3).mean()
    df[f"V_mean_{ROLL_S}s"] = df[VOLTAGE_COL].rolling(win, min_periods=3).mean()

    df = df.reset_index()
    return df


def get_feature_columns() -> list[str]:
    return [
        SOH_COL,
        CURRENT_COL,
        VOLTAGE_COL,
        POWER_COL,
        "dt_s",
        "current_abs_A",
        "current_sq_A2",
        "dI_dt_Aps",
        "dV_dt_Vps",
        f"I_mean_{ROLL_S}s",
        f"V_mean_{ROLL_S}s",
        "is_rest",
    ]


def build_feature_table(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Run all feature engineering steps and return:
    - engineered dataframe
    - feature column list
    """
    df = add_dt_feature(df)
    df = add_current_voltage_features(df)
    df = add_rest_flag(df)
    df = add_rolling_features(df)

    feature_cols = get_feature_columns()
    df_feat = df.dropna(subset=feature_cols + [SOC_COL]).copy()

    print("Rows after feature eng:", len(df_feat))
    print("Feature columns:", feature_cols)

    return df_feat, feature_cols