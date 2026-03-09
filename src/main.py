from src.config import DATA_DIR, OUT_DIR
from src.io_utils import load_all_csvs, save_feature_table, load_feature_table
from src.preprocessing import prepare_base_dataframe
from src.feature_engineering import build_feature_table
from src.visualization import (
    print_dataset_preview,
    plot_soc_current_voltage,
    plot_dt_distribution,
    plot_voltage_vs_soc_rest,
    plot_power_redundancy,
)


def extract():
    # 1. Load raw CSV logs
    df_raw = load_all_csvs(DATA_DIR)

    # 2. Basic preprocessing
    df_base = prepare_base_dataframe(df_raw)

    # 3. Feature engineering
    df_feat, feature_cols = build_feature_table(df_base)

    # 4. Save extracted dataset
    save_feature_table(df_feat, OUT_DIR)

    print("Extraction complete.")
    return df_feat


def plot(df_feat):
    print_dataset_preview(df_feat)
    plot_soc_current_voltage(df_feat)
    plot_dt_distribution(df_feat)
    plot_voltage_vs_soc_rest(df_feat)
    plot_power_redundancy(df_feat)

def visualize():
    df_feat = load_feature_table(OUT_DIR)
    plot(df_feat)



def main():
    # Only load already-processed data
    #extract()
    visualize()


if __name__ == "__main__":
    main()