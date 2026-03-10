from pathlib import Path
import pandas as pd


# Change these paths to match your project
INPUT_DIR = Path(r"C:\Users\GouthamVeera\Documents\Veera\data\models")
EXCEL_DIR = Path(r"C:\Users\GouthamVeera\Documents\Veera\data\Excel_view")


def export_csvs_to_excel(input_dir: Path, output_dir: Path) -> None:
    """
    Convert all CSV files in input_dir to .xlsx files in output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in: {input_dir}")
        return

    print(f"Found {len(csv_files)} CSV file(s).")

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, sep=";", decimal=",")
            excel_file = output_dir / f"{csv_file.stem}.xlsx"
            df.to_excel(excel_file, index=False)
            print(f"Exported: {excel_file}")
        except Exception as e:
            print(f"Failed to export {csv_file.name}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    export_csvs_to_excel(INPUT_DIR, EXCEL_DIR)