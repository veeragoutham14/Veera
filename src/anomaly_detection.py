import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.config import ANOMALY_CONTAMINATION, RANDOM_STATE

ANOMALY_FEATURES = [
    "temp_spread_C",
    "temp_std_C",
    "dT_mean_dt_Cps",
    "dT_max_dt_Cps",
    "cell_voltage_spread_V",
    "comp_cell_voltage_spread_V",
    "current_abs_A",
    "power_abs_W",
    "volt_spread_per_current",
    "T_spread_120s",
    "Vspread_mean_120s",
]


def get_available_anomaly_features(df: pd.DataFrame) -> list[str]:
    return [c for c in ANOMALY_FEATURES if c in df.columns]


def run_isolation_forest(
    df: pd.DataFrame,
    contamination: float = ANOMALY_CONTAMINATION,
) -> pd.DataFrame:
    feature_cols = get_available_anomaly_features(df)

    if not feature_cols:
        raise ValueError("No anomaly features found in dataframe.")

    X = df[feature_cols].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    df = df.copy()
    df["anomaly"] = model.fit_predict(X_scaled)          # -1 anomaly, 1 normal
    df["anomaly_score"] = model.decision_function(X_scaled)

    return df


def get_all_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    if "anomaly" not in df.columns:
        raise ValueError("No anomaly column found. Run anomaly detection first.")
    return df[df["anomaly"] == -1].copy()


def get_top_anomalies(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    if "anomaly_score" not in df.columns:
        raise ValueError("No anomaly_score column found. Run anomaly detection first.")
    return df.sort_values("anomaly_score", ascending=True).head(n).copy()