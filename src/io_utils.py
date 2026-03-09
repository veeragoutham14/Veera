from pathlib import Path
import pandas as pd

from src.config import CSV_GLOB, FEATURE_OUTPUT_FILENAME


def read_varta_blm_csv(path: Path) -> pd.DataFrame:
    """
    Read one VARTA BLM CSV file.
    """
    df = pd.read_csv(
        path,
        skiprows=6,
        sep=";",
        decimal=",",
        engine="python",
        on_bad_lines="skip",
    )

    # "# timestamp" -> "timestamp"
    df = df.rename(columns=lambda c: c.lstrip("# ").strip())
    df["__source_file"] = path.name
    return df


def load_all_csvs(data_dir: Path) -> pd.DataFrame:
    """
    Load and concatenate all CSV files from a directory.
    """
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
    """
    Save engineered feature table.
    """
    out_csv = out_dir / FEATURE_OUTPUT_FILENAME
    df.to_csv(out_csv, index=False, sep=";", decimal=",")
    print("Saved extracted dataset to:", out_csv)
    return out_csv