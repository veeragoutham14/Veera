import matplotlib.pyplot as plt
import pandas as pd

from src.config import (
    TS_COL,
    SOC_COL,
    CURRENT_COL,
    VOLTAGE_COL,
    POWER_COL,
    REST_A,
)


def print_dataset_preview(df_feat: pd.DataFrame) -> None:
    print("\nPreview of extracted dataset:\n")
    print(df_feat.head(20))

    print("\nColumn names:\n")
    print(df_feat.columns)

    print("\nDataset shape:", df_feat.shape)


def plot_soc_current_voltage(df_feat: pd.DataFrame) -> None:
    s = df_feat.iloc[::10]
    t = s[TS_COL]

    plt.figure()
    plt.plot(t, s[SOC_COL])
    plt.title("Target SOC over time (sample)")
    plt.xlabel("time")
    plt.ylabel("soc_pct")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(t, s[CURRENT_COL])
    plt.title("Current over time (sample)")
    plt.xlabel("time")
    plt.ylabel("current_A")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(t, s[VOLTAGE_COL])
    plt.title("Voltage over time (sample)")
    plt.xlabel("time")
    plt.ylabel("voltage_V")
    plt.tight_layout()
    plt.show()


def plot_dt_distribution(df_feat: pd.DataFrame) -> None:
    plt.figure()
    plt.hist(df_feat["dt_s"].clip(upper=df_feat["dt_s"].quantile(0.99)), bins=60)
    plt.title("dt_s distribution (clipped at 99th percentile)")
    plt.xlabel("dt_s")
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()


def plot_voltage_vs_soc_rest(df_feat: pd.DataFrame) -> None:
    rest = df_feat[df_feat["is_rest"] == 1]

    if len(rest) > 0:
        r = rest.sample(n=min(5000, len(rest)), random_state=0)
        plt.figure()
        plt.scatter(r[VOLTAGE_COL], r[SOC_COL], s=6)
        plt.title("Voltage vs SOC (rest points)")
        plt.xlabel("voltage_V")
        plt.ylabel("soc_pct")
        plt.tight_layout()
        plt.show()
    else:
        print("No rest points found with REST_A =", REST_A)


def plot_power_redundancy(df_feat: pd.DataFrame) -> None:
    if df_feat[POWER_COL].notna().any():
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