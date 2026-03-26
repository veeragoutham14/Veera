from pathlib import Path

# =========================
# PATHS
# =========================
DATA_DIR = Path(r"C:\Users\GouthamVeera\Documents\Veera\data\raw")

OUT_DIR = Path(r"C:\Users\GouthamVeera\Documents\Veera\data\processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ANOMALY_OUT_DIR = Path(r"C:\Users\GouthamVeera\Documents\Veera\data\models")
ANOMALY_OUT_DIR.mkdir(parents=True,exist_ok=True)

CSV_GLOB = "*.csv"
FEATURE_OUTPUT_FILENAME = "anomaly_features_extracted.csv"
ANOMALY_ALL_OUTPUT_FILENAME = "anomaly_results_all.csv"
ANOMALY_ONLY_OUTPUT_FILENAME = "anomaly_results_only.csv"
ANOMALY_TOP20_OUTPUT_FILENAME = "anomaly_top20.csv"

# =========================
# CORE COLUMNS
# =========================
TS_COL = "timestamp"

# Context / operating features
SOC_COL = "batteryStackProcImage-2.soc_pct"
SOH_COL = "batteryStackProcImage-2.soh_pct"
CURRENT_COL = "batteryStackProcImage-2.current_A"
VOLTAGE_COL = "batteryStackProcImage-2.voltage_V"
POWER_COL = "batteryStackProcImage-2.power_W" # optional 

# Voltage spread features
CELL_VOLT_MAX_COL = "batteryStackProcImage-2.cellVoltageMax_V"
CELL_VOLT_MIN_COL = "batteryStackProcImage-2.cellVoltageMin_V"
# COMP_CELL_VOLT_MAX_COL = "batteryStackProcImage-2.compensatedCellVoltageMax_V" # only for soc (OCV)
# COMP_CELL_VOLT_MIN_COL = "batteryStackProcImage-2.compensatedCellVoltageMin_V" # only for soc (OCV)

# =========================
# DETAILED PACK TEMPERATURE SENSORS
# =========================
TEMP_SENSOR_COLS = [
    # mod1 pack1
    "batteryStackProcImage-2.batteryModules.mod1.pack1.temperature0_C",
    "batteryStackProcImage-2.batteryModules.mod1.pack1.temperature1_C",
    "batteryStackProcImage-2.batteryModules.mod1.pack1.temperature2_C",
    "batteryStackProcImage-2.batteryModules.mod1.pack1.temperature3_C",

    # mod1 pack2
    "batteryStackProcImage-2.batteryModules.mod1.pack2.temperature0_C",
    "batteryStackProcImage-2.batteryModules.mod1.pack2.temperature1_C",
    "batteryStackProcImage-2.batteryModules.mod1.pack2.temperature2_C",
    "batteryStackProcImage-2.batteryModules.mod1.pack2.temperature3_C",

    # mod2 pack1
    "batteryStackProcImage-2.batteryModules.mod2.pack1.temperature0_C",
    "batteryStackProcImage-2.batteryModules.mod2.pack1.temperature1_C",
    "batteryStackProcImage-2.batteryModules.mod2.pack1.temperature2_C",
    "batteryStackProcImage-2.batteryModules.mod2.pack1.temperature3_C",

    # mod2 pack2
    "batteryStackProcImage-2.batteryModules.mod2.pack2.temperature0_C",
    "batteryStackProcImage-2.batteryModules.mod2.pack2.temperature1_C",
    "batteryStackProcImage-2.batteryModules.mod2.pack2.temperature2_C",
    "batteryStackProcImage-2.batteryModules.mod2.pack2.temperature3_C",

    # mod3 pack1
    "batteryStackProcImage-2.batteryModules.mod3.pack1.temperature0_C",
    "batteryStackProcImage-2.batteryModules.mod3.pack1.temperature1_C",
    "batteryStackProcImage-2.batteryModules.mod3.pack1.temperature2_C",
    "batteryStackProcImage-2.batteryModules.mod3.pack1.temperature3_C",

    # mod3 pack2
    "batteryStackProcImage-2.batteryModules.mod3.pack2.temperature0_C",
    "batteryStackProcImage-2.batteryModules.mod3.pack2.temperature1_C",
    "batteryStackProcImage-2.batteryModules.mod3.pack2.temperature2_C",
    "batteryStackProcImage-2.batteryModules.mod3.pack2.temperature3_C",

    # mod4 pack1
    "batteryStackProcImage-2.batteryModules.mod4.pack1.temperature0_C",
    "batteryStackProcImage-2.batteryModules.mod4.pack1.temperature1_C",
    "batteryStackProcImage-2.batteryModules.mod4.pack1.temperature2_C",
    "batteryStackProcImage-2.batteryModules.mod4.pack1.temperature3_C",

    # mod4 pack2
    "batteryStackProcImage-2.batteryModules.mod4.pack2.temperature0_C",
    "batteryStackProcImage-2.batteryModules.mod4.pack2.temperature1_C",
    "batteryStackProcImage-2.batteryModules.mod4.pack2.temperature2_C",
    "batteryStackProcImage-2.batteryModules.mod4.pack2.temperature3_C",
]

# =========================
# OPTIONAL DETAILED CELL VOLTAGE SENSORS
# =========================
# Start with the precomputed voltage spread features above.
# Add detailed per-cell voltages later if needed.

# =========================
# RAW FEATURE GROUPS
# =========================
CONTEXT_FEATURES = [
    SOH_COL,
    SOC_COL,
    CURRENT_COL,
    VOLTAGE_COL,
    POWER_COL,
]



VOLTAGE_SPREAD_FEATURES = [
    CELL_VOLT_MAX_COL,
    CELL_VOLT_MIN_COL,
    #COMP_CELL_VOLT_MAX_COL,
    #COMP_CELL_VOLT_MIN_COL,
]

X_RAW = (
    CONTEXT_FEATURES
    + VOLTAGE_SPREAD_FEATURES
    + TEMP_SENSOR_COLS
)

# =========================
# FEATURE ENGINEERING PARAMS
# =========================
REST_A = 0.5
ROLL_S = 120  # seconds
TIMESTAMP_FORMAT = "%Y.%m.%d %H:%M:%S"

# =========================
# ANOMALY DETECTION PARAMS
# =========================
ANOMALY_CONTAMINATION = 0.01   # initial guess for Isolation Forest
RANDOM_STATE = 42