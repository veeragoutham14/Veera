import matplotlib.pyplot as plt
import pandas as pd

from src.config import (
    TS_COL,
    SOC_COL,
    CURRENT_COL,
    VOLTAGE_COL,
    POWER_COL,
    REST_A,
    ROLL_S,
)


def print_dataset_preview(df_feat: pd.DataFrame) -> None:
    print("\nPreview of extracted dataset:\n")
    print(df_feat.head(20))

    print("\nColumn names:\n")
    print(df_feat.columns)

    print("\nDataset shape:", df_feat.shape)


def plot_dt_distribution(df_feat: pd.DataFrame) -> None:
    plt.figure()
    plt.hist(df_feat["dt_s"].clip(upper=df_feat["dt_s"].quantile(0.99)), bins=60)
    plt.title("dt_s distribution (clipped at 99th percentile)")
    plt.xlabel("dt_s")
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()


def plot_core_signals_over_time(df_feat: pd.DataFrame) -> None:
    """
    Plot main physical signals over time:
    current, voltage, SOC (context), and mean temperature.
    """
    s = df_feat.iloc[::10]
    t = s[TS_COL]

    plt.figure()
    plt.plot(t, s[CURRENT_COL])
    plt.title("Current over time")
    plt.xlabel("time")
    plt.ylabel("current_A")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(t, s[VOLTAGE_COL])
    plt.title("Voltage over time")
    plt.xlabel("time")
    plt.ylabel("voltage_V")
    plt.tight_layout()
    plt.show()

    if SOC_COL in s.columns:
        plt.figure()
        plt.plot(t, s[SOC_COL])
        plt.title("SOC over time (context only)")
        plt.xlabel("time")
        plt.ylabel("soc_pct")
        plt.tight_layout()
        plt.show()

    if "temp_mean_C" in s.columns:
        plt.figure()
        plt.plot(t, s["temp_mean_C"])
        plt.title("Mean temperature over time")
        plt.xlabel("time")
        plt.ylabel("temp_mean_C")
        plt.tight_layout()
        plt.show()


def plot_temperature_statistics_over_time(df_feat: pd.DataFrame) -> None:
    """
    Plot engineered temperature statistics over time.
    These should be physically meaningful and smooth.
    """
    s = df_feat.iloc[::10]
    t = s[TS_COL]

    plt.figure()
    plt.plot(t, s["temp_mean_C"], label="temp_mean_C")
    plt.plot(t, s["temp_max_C"], label="temp_max_C")
    plt.plot(t, s["temp_min_C"], label="temp_min_C")
    plt.legend()
    plt.title("Temperature statistics over time")
    plt.xlabel("time")
    plt.ylabel("temperature (C)")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(t, s["temp_spread_C"], label="temp_spread_C")
    plt.plot(t, s["temp_std_C"], label="temp_std_C")
    plt.legend()
    plt.title("Temperature spread / std over time")
    plt.xlabel("time")
    plt.ylabel("temperature difference (C)")
    plt.tight_layout()
    plt.show()


def plot_temperature_feature_distributions(df_feat: pd.DataFrame) -> None:
    """
    Histograms of the main engineered temperature features.
    Helps check if values are realistic or broken.
    """
    plt.figure()
    plt.hist(df_feat["temp_mean_C"].dropna(), bins=80)
    plt.title("Distribution of temp_mean_C")
    plt.xlabel("temp_mean_C")
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.hist(df_feat["temp_spread_C"].dropna(), bins=80)
    plt.title("Distribution of temp_spread_C")
    plt.xlabel("temp_spread_C")
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.hist(df_feat["temp_std_C"].dropna(), bins=80)
    plt.title("Distribution of temp_std_C")
    plt.xlabel("temp_std_C")
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()


def plot_temperature_dynamics(df_feat: pd.DataFrame) -> None:
    """
    Plot temperature derivative features over time.
    Useful to see whether dT/dt behaves sensibly.
    """
    s = df_feat.iloc[::10]
    t = s[TS_COL]

    plt.figure()
    plt.plot(t, s["dT_mean_dt_Cps"], label="dT_mean_dt_Cps")
    plt.plot(t, s["dT_max_dt_Cps"], label="dT_max_dt_Cps")
    plt.plot(t, s["dT_spread_dt_Cps"], label="dT_spread_dt_Cps")
    plt.legend()
    plt.title("Temperature dynamics over time")
    plt.xlabel("time")
    plt.ylabel("C/s")
    plt.tight_layout()
    plt.show()


def plot_voltage_spread_over_time(df_feat: pd.DataFrame) -> None:
    """
    Plot cell voltage spread features over time.
    Important for cell imbalance anomaly checks.
    """
    s = df_feat.iloc[::10]
    t = s[TS_COL]

    if "cell_voltage_spread_V" in s.columns:
        plt.figure()
        plt.plot(t, s["cell_voltage_spread_V"], label="cell_voltage_spread_V")
        if "comp_cell_voltage_spread_V" in s.columns:
            plt.plot(t, s["comp_cell_voltage_spread_V"], label="comp_cell_voltage_spread_V")
        plt.legend()
        plt.title("Cell voltage spread over time")
        plt.xlabel("time")
        plt.ylabel("V")
        plt.tight_layout()
        plt.show()


def plot_load_vs_temperature_relationships(df_feat: pd.DataFrame) -> None:
    """
    Scatter plots to check whether thermal behavior correlates with load.
    These are important sanity checks for anomaly features.
    """
    sample = df_feat.sample(n=min(5000, len(df_feat)), random_state=0)

    plt.figure()
    plt.scatter(sample[CURRENT_COL], sample["temp_mean_C"], s=6)
    plt.title("Current vs mean temperature")
    plt.xlabel("current_A")
    plt.ylabel("temp_mean_C")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.scatter(sample["current_abs_A"], sample["temp_spread_C"], s=6)
    plt.title("Abs current vs temperature spread")
    plt.xlabel("current_abs_A")
    plt.ylabel("temp_spread_C")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.scatter(sample["current_abs_A"], sample["dT_mean_dt_Cps"], s=6)
    plt.title("Abs current vs dT_mean_dt_Cps")
    plt.xlabel("current_abs_A")
    plt.ylabel("dT_mean_dt_Cps")
    plt.tight_layout()
    plt.show()

    if "cell_voltage_spread_V" in sample.columns:
        plt.figure()
        plt.scatter(sample["current_abs_A"], sample["cell_voltage_spread_V"], s=6)
        plt.title("Abs current vs cell voltage spread")
        plt.xlabel("current_abs_A")
        plt.ylabel("cell_voltage_spread_V")
        plt.tight_layout()
        plt.show()


def plot_rolling_feature_checks(df_feat: pd.DataFrame) -> None:
    """
    Plot rolling features to confirm they behave like smoothed versions
    of the raw features.
    """
    s = df_feat.iloc[::10]
    t = s[TS_COL]

    if f"I_mean_{ROLL_S}s" in s.columns:
        plt.figure()
        plt.plot(t, s[CURRENT_COL], label="raw current", alpha=0.5)
        plt.plot(t, s[f"I_mean_{ROLL_S}s"], label=f"I_mean_{ROLL_S}s")
        plt.legend()
        plt.title("Current vs rolling current mean")
        plt.xlabel("time")
        plt.ylabel("A")
        plt.tight_layout()
        plt.show()

    if f"T_mean_{ROLL_S}s" in s.columns:
        plt.figure()
        plt.plot(t, s["temp_mean_C"], label="raw temp_mean_C", alpha=0.5)
        plt.plot(t, s[f"T_mean_{ROLL_S}s"], label=f"T_mean_{ROLL_S}s")
        plt.legend()
        plt.title("Temperature mean vs rolling mean")
        plt.xlabel("time")
        plt.ylabel("C")
        plt.tight_layout()
        plt.show()

    if f"T_spread_{ROLL_S}s" in s.columns:
        plt.figure()
        plt.plot(t, s["temp_spread_C"], label="raw temp_spread_C", alpha=0.5)
        plt.plot(t, s[f"T_spread_{ROLL_S}s"], label=f"T_spread_{ROLL_S}s")
        plt.legend()
        plt.title("Temperature spread vs rolling spread")
        plt.xlabel("time")
        plt.ylabel("C")
        plt.tight_layout()
        plt.show()


def plot_rest_points(df_feat: pd.DataFrame) -> None:
    """
    Keep this only as a context check.
    Rest points can be useful to understand whether low-current behavior
    is physically reasonable.
    """
    rest = df_feat[df_feat["is_rest"] == 1]

    if len(rest) > 0:
        r = rest.sample(n=min(5000, len(rest)), random_state=0)

        plt.figure()
        plt.scatter(r[VOLTAGE_COL], r["temp_mean_C"], s=6)
        plt.title("Voltage vs mean temperature (rest points)")
        plt.xlabel("voltage_V")
        plt.ylabel("temp_mean_C")
        plt.tight_layout()
        plt.show()
    else:
        print("No rest points found with REST_A =", REST_A)


def plot_power_redundancy(df_feat: pd.DataFrame) -> None:
    """
    Keep this while validating whether POWER_COL should remain in the pipeline.
    """
    if POWER_COL in df_feat.columns and df_feat[POWER_COL].notna().any():
        approx = df_feat[VOLTAGE_COL] * df_feat[CURRENT_COL]
        err = df_feat[POWER_COL] - approx

        plt.figure()
        plt.hist(
            err.dropna().clip(err.quantile(0.01), err.quantile(0.99)),
            bins=80,
        )
        plt.title("power_W - (V*I) error (clipped)")
        plt.xlabel("W")
        plt.ylabel("count")
        plt.tight_layout()
        plt.show()


def run_all_visual_checks(df_feat: pd.DataFrame) -> None:
    """
    Run all anomaly-feature sanity checks.
    """
    print_dataset_preview(df_feat)
    plot_dt_distribution(df_feat)
    plot_core_signals_over_time(df_feat)
    plot_temperature_statistics_over_time(df_feat)
    plot_temperature_feature_distributions(df_feat)
    plot_temperature_dynamics(df_feat)
    plot_voltage_spread_over_time(df_feat)
    plot_load_vs_temperature_relationships(df_feat)
    plot_rolling_feature_checks(df_feat)
    plot_rest_points(df_feat)
    plot_power_redundancy(df_feat)


def plot_anomalies(df: pd.DataFrame, feature: str) -> None:
    """
    Plot a feature over time with anomalies highlighted.
    Normal points = blue
    Anomalies = red
    """

    if "anomaly" not in df.columns:
        raise ValueError("No anomaly column found. Run anomaly detection first.")

    normal = df[df["anomaly"] == 1]
    anomalies = df[df["anomaly"] == -1]

    plt.figure(figsize=(12,6))

    plt.scatter(
        normal[TS_COL],
        normal[feature],
        s=4,
        color="blue",
        label="normal",
        alpha=0.5
    )

    plt.scatter(
        anomalies[TS_COL],
        anomalies[feature],
        s=20,
        color="red",
        label="anomaly"
    )

    plt.title(f"Anomaly detection on {feature}")
    plt.xlabel("time")
    plt.ylabel(feature)
    plt.legend()
    plt.tight_layout()
    plt.show()