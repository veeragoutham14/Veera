from pathlib import Path
import pandas as pd

from src.config import (
    CSV_GLOB,
    FEATURE_OUTPUT_FILENAME,
    ANOMALY_ALL_OUTPUT_FILENAME,
    ANOMALY_ONLY_OUTPUT_FILENAME,
    ANOMALY_TOP20_OUTPUT_FILENAME,
    TS_COL,
    TIMESTAMP_FORMAT,
)


def read_varta_blm_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        skiprows=6,
        sep=";",
        decimal=",",
        engine="python",
        on_bad_lines="skip",
    )

    df = df.rename(columns=lambda c: c.lstrip("# ").strip())
    df = df.assign(__source_file=path.name).copy()
    return df


def load_all_csvs(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob(CSV_GLOB))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    dfs = []
    bad = []

    for f in files:
        try:
            d = read_varta_blm_csv(f)
            dfs.append(d)
            print(f"Loaded {f.name}: {d.shape}")
        except Exception as e:
            bad.append((f.name, str(e)))

    if bad:
        print("\nFiles that failed to load:")
        for name, err in bad:
            print("-", name, ":", err)

    if not dfs:
        raise ValueError("No valid CSV files could be loaded.")

    df = pd.concat(dfs, ignore_index=True)
    print("\nCombined shape:", df.shape)
    return df


def save_feature_table(df: pd.DataFrame, out_dir: Path) -> Path:
    out_csv = out_dir / FEATURE_OUTPUT_FILENAME
    df.to_csv(
        out_csv,
        index=False,
        sep=";",
        decimal=",",
        
    )
    print("Saved extracted dataset to:", out_csv)
    return out_csv


def load_feature_table(out_dir: Path) -> pd.DataFrame:
    feature_path = out_dir / FEATURE_OUTPUT_FILENAME
    if not feature_path.exists():
        raise FileNotFoundError(
            f"Processed feature file not found: {feature_path}\n"
            "Run extraction first."
        )

    df = pd.read_csv(feature_path, sep=";", decimal=",")

    if TS_COL in df.columns:
        df[TS_COL] = pd.to_datetime(df[TS_COL], errors="coerce")

    print("Loaded processed feature table:", feature_path)
    print("Shape:", df.shape)
    return df


def save_anomaly_tables(
    df_all: pd.DataFrame,
    df_anomalies_only: pd.DataFrame,
    df_top20: pd.DataFrame,
    out_dir: Path,
) -> tuple[Path, Path, Path]:
    all_path = out_dir / ANOMALY_ALL_OUTPUT_FILENAME
    only_path = out_dir / ANOMALY_ONLY_OUTPUT_FILENAME
    top20_path = out_dir / ANOMALY_TOP20_OUTPUT_FILENAME

    for df, path in [
        (df_all, all_path),
        (df_anomalies_only, only_path),
        (df_top20, top20_path),
    ]:
        df.to_csv(
            path,
            index=False,
            sep=";",
            decimal=",",
        )

    print("Saved all anomaly results to:", all_path)
    print("Saved anomaly-only rows to:", only_path)
    print("Saved top 20 anomalies to:", top20_path)

    return all_path, only_path, top20_path


def load_anomaly_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Anomaly file not found: {path}")

    df = pd.read_csv(path, sep=";", decimal=",")

    if TS_COL in df.columns:
        df[TS_COL] = pd.to_datetime(df[TS_COL], errors="coerce")

    print("Loaded anomaly table:", path)
    print("Shape:", df.shape)
    return df