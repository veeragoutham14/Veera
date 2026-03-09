from pathlib import Path

# =========================
# PATHS
# =========================
DATA_DIR = Path(r"C:\Users\GouthamVeera\Documents\Veera\data\raw")
OUT_DIR = Path(r"C:\Users\GouthamVeera\Documents\Veera\data\processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_GLOB = "*.csv"
FEATURE_OUTPUT_FILENAME = "features_extracted.csv"

# =========================
# CORE COLUMNS
# =========================
TS_COL = "timestamp"
SOC_COL = "batteryStackProcImage-2.soc_pct"

SOH_COL = "batteryStackProcImage-2.soh_pct"
CURRENT_COL = "batteryStackProcImage-2.current_A"
VOLTAGE_COL = "batteryStackProcImage-2.voltage_V"
POWER_COL = "batteryStackProcImage-2.power_W"

X_RAW = [
    SOH_COL,
    CURRENT_COL,
    VOLTAGE_COL,
    POWER_COL,  # optional; kept for redundancy check
]

# =========================
# FEATURE ENGINEERING PARAMS
# =========================
REST_A = 0.5
ROLL_S = 120  # seconds
TIMESTAMP_FORMAT = "%Y.%m.%d %H:%M:%S"