import pandas as pd
from src.config import DATA_DIR, OUT_DIR, ANOMALY_OUT_DIR
from src.io_utils import (
    load_all_csvs,
    save_feature_table,
    load_feature_table,
    save_anomaly_tables,
)
from src.preprocessing import prepare_base_dataframe
from src.feature_engineering import build_feature_table
from src.visualization import run_all_visual_checks, plot_anomalies
from src.anomaly_detection import (
    run_isolation_forest,
    get_all_anomalies,
    get_top_anomalies,
    get_available_anomaly_features
)
from src.anomaly_explainer import explain_top_anomalies


def extract() -> None:
    print("\n--- Loading raw CSV logs ---")
    df_raw = load_all_csvs(DATA_DIR)

    print("\n--- Running preprocessing ---")
    df_base = prepare_base_dataframe(df_raw)

    print("\n--- Running feature engineering ---")
    df_feat, feature_cols = build_feature_table(df_base)

    print("\n--- Saving processed dataset ---")
    save_feature_table(df_feat, OUT_DIR)

    print("\nExtraction complete.")
    print("Feature count:", len(feature_cols))


def visualize() -> None:
    print("\n--- Loading processed feature table ---")
    df_feat = load_feature_table(OUT_DIR)

    print("\n--- Running visualization checks ---")
    run_all_visual_checks(df_feat)


def detect_anomalies() -> None:
    print("\n--- Loading processed feature table ---")
    df_feat = load_feature_table(OUT_DIR)

    print("\n--- Running Isolation Forest anomaly detection ---")
    df_scored = run_isolation_forest(df_feat)

    df_anomalies_only = get_all_anomalies(df_scored)
    df_top20 = get_top_anomalies(df_scored, n=20)

    print("\nTotal samples:", len(df_scored))
    print("Detected anomalies:", len(df_anomalies_only))

    print("\n--- Top 20 strongest anomalies ---")
    cols_to_show = [
        "timestamp",
        "anomaly_score",
        "temp_spread_C",
        "temp_std_C",
        "dT_mean_dt_Cps",
        "dT_max_dt_Cps",
        "cell_voltage_spread_V",
        "current_abs_A",
        "power_abs_W",
    ]
    cols_to_show = [c for c in cols_to_show if c in df_top20.columns]
    print(df_top20[cols_to_show])

    # NEW: explain top anomalies using normal rows in df_scored
    print("\n--- Explaining top anomalies ---")
    feature_cols = get_available_anomaly_features(df_scored)

    df_explained = explain_top_anomalies(
        df=df_scored,
        feature_cols=feature_cols,
        top_n=20,
        timestamp_col="timestamp",
        anomaly_col="anomaly",
        score_col="anomaly_score",
    )

    cols_expl_to_show = [
        "timestamp",
        "anomaly_score",
        "top_feature_1",
        "top_feature_2",
        "top_feature_3",
        "explanation",
    ]
    cols_expl_to_show = [c for c in cols_expl_to_show if c in df_explained.columns]
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.max_columns", None)
    print(df_explained[cols_expl_to_show])

    print("\n--- Saving anomaly outputs ---")
    save_anomaly_tables(
        df_all=df_scored,
        df_anomalies_only=df_anomalies_only,
        df_top20=df_top20,
        out_dir=ANOMALY_OUT_DIR,
    )

    print("\n--- Plotting anomalies ---")
    plot_anomalies(df_scored, "temp_spread_C")
    if "cell_voltage_spread_V" in df_scored.columns:
        plot_anomalies(df_scored, "cell_voltage_spread_V")
    if "dT_max_dt_Cps" in df_scored.columns:
        plot_anomalies(df_scored, "dT_max_dt_Cps")


def main():
    #extract()
    #visualize()
    detect_anomalies()


if __name__ == "__main__":
    main()