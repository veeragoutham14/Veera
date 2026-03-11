import numpy as np
import pandas as pd


def explain_top_anomalies(
    df: pd.DataFrame,
    feature_cols: list[str],
    top_n: int = 20,
    timestamp_col: str = "timestamp",
    anomaly_col: str = "anomaly",
    score_col: str = "anomaly_score",
) -> pd.DataFrame:
    """
    Explain the top-N strongest anomalies by comparing each anomaly row
    against the distribution of normal rows.

    Returns a dataframe with:
    - timestamp
    - anomaly_score
    - top contributing features
    - a short explanation text
    """

    required_cols = [anomaly_col, score_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    usable_features = [c for c in feature_cols if c in df.columns]
    if not usable_features:
        raise ValueError("No usable feature columns found in dataframe.")

    # Normal rows define the baseline
    df_normal = df[df[anomaly_col] == 1].copy()
    if df_normal.empty:
        raise ValueError("No normal rows found. Cannot build baseline statistics.")

    # Strongest anomalies = lowest anomaly score
    df_anom = df[df[anomaly_col] == -1].copy()
    if df_anom.empty:
        raise ValueError("No anomaly rows found.")

    df_top = df_anom.nsmallest(top_n, score_col).copy()

    # Baseline statistics from normal rows
    normal_mean = df_normal[usable_features].mean()
    normal_std = df_normal[usable_features].std().replace(0, np.nan)

    rows = []

    for idx, row in df_top.iterrows():
        # z-score-like deviation from normal rows
        deviations = ((row[usable_features] - normal_mean) / normal_std).replace([np.inf, -np.inf], np.nan)
        abs_dev = deviations.abs().sort_values(ascending=False)

        top3 = abs_dev.head(3).index.tolist()

        feature_summaries = []
        for feat in top3:
            value = row[feat]
            z = deviations[feat]

            if pd.isna(z):
                direction = "unusual"
            elif z > 0:
                direction = "high"
            else:
                direction = "low"

            feature_summaries.append(
                {
                    "feature": feat,
                    "value": value,
                    "z_score": z,
                    "direction": direction,
                }
            )

        explanation_text = build_explanation_text(row, feature_summaries, timestamp_col, score_col)

        out = {
            "row_index": idx,
            "timestamp": row[timestamp_col] if timestamp_col in row else None,
            "anomaly_score": row[score_col],
            "top_feature_1": feature_summaries[0]["feature"] if len(feature_summaries) > 0 else None,
            "top_feature_1_value": feature_summaries[0]["value"] if len(feature_summaries) > 0 else None,
            "top_feature_1_z": feature_summaries[0]["z_score"] if len(feature_summaries) > 0 else None,
            "top_feature_1_direction": feature_summaries[0]["direction"] if len(feature_summaries) > 0 else None,
            "top_feature_2": feature_summaries[1]["feature"] if len(feature_summaries) > 1 else None,
            "top_feature_2_value": feature_summaries[1]["value"] if len(feature_summaries) > 1 else None,
            "top_feature_2_z": feature_summaries[1]["z_score"] if len(feature_summaries) > 1 else None,
            "top_feature_2_direction": feature_summaries[1]["direction"] if len(feature_summaries) > 1 else None,
            "top_feature_3": feature_summaries[2]["feature"] if len(feature_summaries) > 2 else None,
            "top_feature_3_value": feature_summaries[2]["value"] if len(feature_summaries) > 2 else None,
            "top_feature_3_z": feature_summaries[2]["z_score"] if len(feature_summaries) > 2 else None,
            "top_feature_3_direction": feature_summaries[2]["direction"] if len(feature_summaries) > 2 else None,
            "explanation": explanation_text,
        }

        rows.append(out)

    return pd.DataFrame(rows)


def build_explanation_text(
    row: pd.Series,
    feature_summaries: list[dict],
    timestamp_col: str,
    score_col: str,
) -> str:
    """
    Create a short human-readable explanation.
    """
    parts = []

    if timestamp_col in row:
        parts.append(f"Anomaly at {row[timestamp_col]}")

    parts.append(f"score={row[score_col]:.6f}")

    feature_parts = []
    for item in feature_summaries:
        feat = item["feature"]
        direction = item["direction"]
        value = item["value"]

        if isinstance(value, (float, np.floating)):
            feature_parts.append(f"{feat} is {direction} ({value:.4f})")
        else:
            feature_parts.append(f"{feat} is {direction} ({value})")

    if feature_parts:
        parts.append("Main deviations: " + ", ".join(feature_parts))

    # Small domain-specific interpretation rules
    interpretation = infer_physical_interpretation(row, feature_summaries)
    if interpretation:
        parts.append("Interpretation: " + interpretation)

    return " | ".join(parts)


def infer_physical_interpretation(row: pd.Series, feature_summaries: list[dict]) -> str:
    """
    Add a simple engineering-style interpretation based on feature names.
    """
    top_feats = [f["feature"] for f in feature_summaries]

    # Convert to safe values
    current = row["current_abs_A"] if "current_abs_A" in row.index else np.nan
    temp_spread = row["temp_spread_C"] if "temp_spread_C" in row.index else np.nan
    temp_std = row["temp_std_C"] if "temp_std_C" in row.index else np.nan
    vspread = row["cell_voltage_spread_V"] if "cell_voltage_spread_V" in row.index else np.nan

    if "temp_spread_C" in top_feats and "current_abs_A" in top_feats:
        if pd.notna(current) and pd.notna(temp_spread) and current < 3:
            return "temperature imbalance appears high relative to the current load"

    if "temp_std_C" in top_feats and "temp_spread_C" in top_feats:
        return "thermal distribution across the battery pack appears unusual"

    if "cell_voltage_spread_V" in top_feats:
        return "cell voltage imbalance may be contributing"

    if "dI_dt_Aps" in top_feats or "dV_dt_Vps" in top_feats:
        return "anomaly may be linked to a fast electrical transition"

    if pd.notna(vspread) and vspread >= 0.04:
        return "cell voltage spread is elevated"

    if pd.notna(temp_spread) and pd.notna(temp_std) and temp_spread > 3.5:
        return "thermal spread across modules is elevated"

    return "multiple features jointly deviate from normal operating behaviour"