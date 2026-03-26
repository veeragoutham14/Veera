import numpy as np
import pandas as pd


def get_row_mode(row: pd.Series) -> str:
    """
    Determine battery mode from one-hot mode columns.
    """
    if "is_rest" in row.index and row["is_rest"] == 1:
        return "rest"
    if "is_charging" in row.index and row["is_charging"] == 1:
        return "charging"
    if "is_discharging" in row.index and row["is_discharging"] == 1:
        return "discharging"
    return "unknown"


def get_mode_normal_subset(df_normal: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    """
    Return normal rows matching the anomaly row's mode.
    Falls back to all normal rows if mode-specific subset is too small.
    """
    mode = get_row_mode(row)

    if mode == "rest" and "is_rest" in df_normal.columns:
        subset = df_normal[df_normal["is_rest"] == 1]
    elif mode == "charging" and "is_charging" in df_normal.columns:
        subset = df_normal[df_normal["is_charging"] == 1]
    elif mode == "discharging" and "is_discharging" in df_normal.columns:
        subset = df_normal[df_normal["is_discharging"] == 1]
    else:
        subset = df_normal

    # fallback if too few rows
    if len(subset) < 30:
        return df_normal

    return subset


def build_reason_text(item: dict) -> str:
    feat = item["feature"]
    direction = item["direction"]
    value = item["value"]

    if isinstance(value, (float, np.floating)):
        return f"{feat} is {direction} ({value:.4f})"
    return f"{feat} is {direction} ({value})"


def infer_physical_interpretation(row: pd.Series, feature_summaries: list[dict]) -> str:
    """
    Infer a physically meaningful interpretation for one anomaly row.

    Improvements:
    - Uses only model features for anomaly families
    - Uses SOC only as context, not as an anomaly feature
    - Gates voltage language by raw magnitude
    - Separates ratio anomaly from actual voltage imbalance
    - Requires support before using transient language
    - Combines only physically valid families
    - Uses mode-aware wording
    """
    # -----------------------------
    # Helpers
    # -----------------------------
    def get_value(name: str) -> float:
        return row[name] if name in row.index else np.nan

    def abs_z(name: str) -> float:
        return z_map.get(name, 0.0)

    def has(name: str) -> bool:
        return name in top_feat_set

    def get_mode_label(mode: str) -> str:
        if mode == "charging":
            return "charging"
        if mode == "discharging":
            return "discharge"
        if mode == "rest":
            return "rest"
        return "operation"

    def get_soc_region(soc: float) -> str:
        if pd.isna(soc):
            return "unknown"
        if soc < 20:
            return "low"
        if soc < 80:
            return "mid"
        return "high"

    def add_soc_context(text: str, family: str) -> str:
        """
        Add SOC-aware wording refinement.
        SOC is context only, not a ranking feature.
        """
        if soc_region == "unknown":
            return text

        if family == "voltage":
            if soc_region == "high":
                return text + " near high SOC, where voltage dispersion is more sensitive"
            if soc_region == "low":
                return text + " at low SOC, where voltage differences can be amplified"
            return text + " in the mid-SOC range"

        if family == "thermal":
            if soc_region == "high" and mode == "charging":
                return text + " during high-SOC charging"
            if soc_region == "low":
                return text + " at low SOC, where resistance-related heating may be stronger"
            return text

        if family == "transient":
            if soc_region == "high" and mode == "charging":
                return text + " during high-SOC charging"
            if soc_region == "low":
                return text + " at low SOC"
            return text

        return text

    # -----------------------------
    # Top features + z-scores
    # -----------------------------
    top_feat_set = set()
    z_map: dict[str, float] = {}

    for item in feature_summaries:
        feat = item.get("feature")
        if feat is None:
            continue
        top_feat_set.add(feat)
        z = item.get("z_score", np.nan)
        z_map[feat] = abs(z) if pd.notna(z) else 0.0

    # -----------------------------
    # Allowed model features only
    # -----------------------------
    temp_spread = get_value("temp_spread_C")
    temp_std = get_value("temp_std_C")
    dT_mean = get_value("dT_mean_dt_Cps")
    dT_max = get_value("dT_max_dt_Cps")
    vspread = get_value("cell_voltage_spread_V")
    current = get_value("current_abs_A")
    vspread_per_current = get_value("volt_spread_per_current")
    t_spread_120s = get_value("T_spread_120s")
    vspread_mean_120s = get_value("Vspread_mean_120s")

    # -----------------------------
    # Context
    # -----------------------------
    mode = row["battery_mode"] if "battery_mode" in row.index else "unknown"
    mode_label = get_mode_label(mode)

    # IMPORTANT:
    # Replace "soc_pct" with your real SOC column if needed.
    # Example:
    # soc = get_value("batteryStackProcImage-2.soc_pct")
    soc = get_value("batteryStackProcImage-2.soc_pct")
    soc_region = get_soc_region(soc)

    # -----------------------------
    # Magnitude gates
    # -----------------------------
    low_current = pd.notna(current) and current < 3.0
    high_current = pd.notna(current) and current >= 10.0

    # Voltage: raw magnitude
    voltage_spread_elevated = pd.notna(vspread) and vspread >= 0.015
    voltage_spread_high = pd.notna(vspread) and vspread >= 0.025
    voltage_spread_weak = pd.isna(vspread) or vspread < 0.010

    # Voltage persistence
    voltage_persistent = pd.notna(vspread_mean_120s) and vspread_mean_120s >= 0.015
    voltage_not_persistent = pd.notna(vspread_mean_120s) and vspread_mean_120s < 0.008
    voltage_persistence_weak = pd.isna(vspread_mean_120s) or vspread_mean_120s < 0.010

    # Ratio-based voltage anomaly
    ratio_voltage_high = pd.notna(vspread_per_current) and vspread_per_current >= 0.008
    ratio_voltage_moderate = pd.notna(vspread_per_current) and vspread_per_current >= 0.005

    # Thermal
    temp_spread_elevated = pd.notna(temp_spread) and temp_spread >= 2.0
    temp_spread_high = pd.notna(temp_spread) and temp_spread >= 2.8
    temp_std_elevated = pd.notna(temp_std) and temp_std >= 0.60

    thermal_persistent = pd.notna(t_spread_120s) and t_spread_120s >= 2.2
    thermal_strongly_persistent = pd.notna(t_spread_120s) and t_spread_120s >= 2.7

    # -----------------------------
    # Physically valid family flags
    # -----------------------------
    # True voltage imbalance requires raw spread support
    valid_voltage_imbalance = voltage_spread_elevated
    valid_sustained_voltage = voltage_spread_elevated and voltage_persistent

    # Ratio anomaly only: high ratio, low current, but no actual strong raw voltage spread
    valid_ratio_voltage = (
        ratio_voltage_high
        and low_current
        and not valid_voltage_imbalance
        and voltage_persistence_weak
    )

    # Thermal imbalance
    valid_thermal_imbalance = temp_spread_elevated or temp_std_elevated
    valid_sustained_thermal = thermal_persistent and valid_thermal_imbalance
    valid_strong_thermal = thermal_strongly_persistent and (temp_spread_high or temp_std_elevated)

    # Supported transient
    derivative_pair_strong = (
        pd.notna(dT_max) and abs(dT_max) >= 0.08 and
        pd.notna(dT_mean) and abs(dT_mean) >= 0.005
    )

    derivative_single_very_strong = (
        (pd.notna(dT_max) and abs(dT_max) >= 0.10) or
        (pd.notna(dT_mean) and abs(dT_mean) >= 0.008)
    )

    transient_supported = (
        derivative_pair_strong or
        (
            derivative_single_very_strong and
            (high_current or valid_thermal_imbalance or thermal_persistent)
        )
    )

    transient_strong = (
        derivative_pair_strong and
        (high_current or valid_sustained_thermal or valid_sustained_voltage)
    )

    # -----------------------------
    # Pattern strength scores
    # Only reward physically supported patterns
    # -----------------------------
    voltage_score = 0.0
    thermal_score = 0.0
    transient_score = 0.0
    load_context_score = 0.0

    # Voltage family
    if valid_sustained_voltage:
        voltage_score += 3.0
    elif valid_voltage_imbalance:
        voltage_score += 2.2
    elif valid_ratio_voltage:
        voltage_score += 1.6
    elif ratio_voltage_moderate and low_current and has("volt_spread_per_current"):
        voltage_score += 0.8

    if has("cell_voltage_spread_V") and valid_voltage_imbalance:
        voltage_score += 0.25 * abs_z("cell_voltage_spread_V")
    if has("Vspread_mean_120s") and voltage_persistent:
        voltage_score += 0.20 * abs_z("Vspread_mean_120s")
    if has("volt_spread_per_current") and (valid_ratio_voltage or valid_voltage_imbalance):
        voltage_score += 0.20 * abs_z("volt_spread_per_current")

    # Thermal family
    if valid_strong_thermal:
        thermal_score += 3.0
    elif valid_sustained_thermal:
        thermal_score += 2.5
    elif valid_thermal_imbalance:
        thermal_score += 1.8
    elif thermal_persistent:
        thermal_score += 1.2

    if has("temp_spread_C") and temp_spread_elevated:
        thermal_score += 0.25 * abs_z("temp_spread_C")
    if has("temp_std_C") and temp_std_elevated:
        thermal_score += 0.20 * abs_z("temp_std_C")
    if has("T_spread_120s") and thermal_persistent:
        thermal_score += 0.20 * abs_z("T_spread_120s")

    # Transient family
    if transient_strong:
        transient_score += 3.0
    elif transient_supported:
        transient_score += 2.0
    elif has("dT_max_dt_Cps") or has("dT_mean_dt_Cps"):
        transient_score += 0.8

    if has("dT_max_dt_Cps"):
        transient_score += 0.25 * abs_z("dT_max_dt_Cps")
    if has("dT_mean_dt_Cps"):
        transient_score += 0.20 * abs_z("dT_mean_dt_Cps")

    # Load context family
    if has("current_abs_A"):
        load_context_score += 0.20 * abs_z("current_abs_A")
    if low_current or high_current:
        load_context_score += 0.5

    # -----------------------------
    # Determine dominant families
    # -----------------------------
    family_scores = {
        "voltage": voltage_score,
        "thermal": thermal_score,
        "transient": transient_score,
        "load": load_context_score,
    }

    sorted_families = sorted(family_scores.items(), key=lambda x: x[1], reverse=True)
    primary_family, primary_score = sorted_families[0]
    secondary_family, secondary_score = sorted_families[1]

    secondary_relevant = secondary_score >= max(1.5, 0.65 * primary_score)

    # -----------------------------
    # Build primary phrase
    # -----------------------------
    primary_phrase = None
    secondary_phrase = None

    if primary_family == "voltage":
        if valid_sustained_voltage:
            if low_current:
                primary_phrase = f"sustained cell voltage imbalance is present even under low-current {mode_label}"
            else:
                primary_phrase = "sustained cell voltage imbalance is the dominant anomaly pattern"
        elif valid_voltage_imbalance:
            if low_current:
                primary_phrase = f"cell voltage spread is unusually high relative to the low current {mode_label}"
            else:
                primary_phrase = f"cell voltage spread is elevated during {mode_label}"
        elif valid_ratio_voltage:
            primary_phrase = f"a relative voltage anomaly appears unusually strong for this low-current {mode_label} condition"
        else:
            primary_phrase = "voltage-related behaviour deviates from the mode-specific baseline"

    elif primary_family == "thermal":
        if valid_strong_thermal:
            primary_phrase = "sustained thermal imbalance across the battery pack is strongly indicated"
        elif valid_sustained_thermal:
            primary_phrase = "sustained thermal imbalance across the battery pack is observed"
        elif valid_thermal_imbalance:
            primary_phrase = "thermal non-uniformity is observed across the battery pack"
        elif thermal_persistent:
            primary_phrase = "temperature spread over time is elevated"
        else:
            primary_phrase = "thermal behaviour deviates from the mode-specific baseline"

    elif primary_family == "transient":
        if transient_strong and high_current:
            primary_phrase = f"a strong transient thermal response is observed under high {mode_label} load"
        elif transient_strong:
            primary_phrase = "temperature dynamics indicate a strong unusual transient response"
        elif transient_supported:
            primary_phrase = "temperature dynamics indicate an unusual transient response"
        else:
            primary_phrase = f"temperature-rate features deviate from the normal {mode_label} baseline"

    else:
        if low_current:
            primary_phrase = f"the anomaly occurs under unusually low-current {mode_label}"
        elif high_current:
            primary_phrase = f"the anomaly occurs under unusually high-current {mode_label}"
        else:
            primary_phrase = f"the current level appears inconsistent with normal {mode_label} behaviour"

    primary_phrase = add_soc_context(primary_phrase, primary_family)

    # -----------------------------
    # Secondary phrase
    # Only combine if physically valid
    # -----------------------------
    if secondary_relevant:
        if secondary_family == "voltage":
            if valid_sustained_voltage:
                secondary_phrase = "concurrent sustained cell voltage imbalance is also present"
            elif valid_voltage_imbalance:
                secondary_phrase = "concurrent cell voltage spread is also elevated"
            elif valid_ratio_voltage:
                secondary_phrase = f"concurrent relative voltage anomaly is also present under low-current {mode_label}"

        elif secondary_family == "thermal":
            if valid_strong_thermal:
                secondary_phrase = "concurrent strong sustained thermal imbalance is also present"
            elif valid_sustained_thermal:
                secondary_phrase = "concurrent sustained thermal imbalance is also present"
            elif valid_thermal_imbalance:
                secondary_phrase = "concurrent thermal non-uniformity is also present"

        elif secondary_family == "transient":
            if transient_strong:
                secondary_phrase = "concurrent strong transient behaviour is also visible"
            elif transient_supported:
                secondary_phrase = "concurrent transient behaviour is also visible"

        elif secondary_family == "load":
            if low_current:
                secondary_phrase = f"the event also occurs under low-current {mode_label}"
            elif high_current:
                secondary_phrase = f"the event also occurs under high-current {mode_label}"

    # -----------------------------
    # Special-case mixed patterns
    # -----------------------------
    # Voltage + transient
    if primary_family == "voltage" and secondary_relevant and secondary_family == "transient":
        if valid_sustained_voltage and transient_supported:
            return add_soc_context(
                "sustained cell voltage imbalance is present, with concurrent unusual transient thermal behaviour",
                "voltage",
            )
        if valid_voltage_imbalance and transient_supported:
            return add_soc_context(
                "cell voltage imbalance is present, with concurrent unusual transient thermal behaviour",
                "voltage",
            )
        if valid_ratio_voltage and transient_supported:
            return add_soc_context(
                f"a relative voltage anomaly is present, with concurrent unusual transient thermal behaviour during {mode_label}",
                "voltage",
            )

    if primary_family == "transient" and secondary_relevant and secondary_family == "voltage":
        if valid_sustained_voltage and transient_supported:
            if transient_strong and high_current:
                return add_soc_context(
                    f"a strong transient thermal response is observed under high {mode_label} load, with concurrent sustained cell voltage imbalance",
                    "transient",
                )
            return add_soc_context(
                "an unusual transient thermal response is dominant, with concurrent sustained cell voltage imbalance",
                "transient",
            )

        if valid_voltage_imbalance and transient_supported:
            if transient_strong and high_current:
                return add_soc_context(
                    f"a strong transient thermal response is observed under high {mode_label} load, with concurrent voltage imbalance",
                    "transient",
                )
            return add_soc_context(
                "an unusual transient thermal response is dominant, with concurrent voltage imbalance",
                "transient",
            )

        # If voltage is not physically valid, do not mention it
        if transient_strong and high_current:
            return add_soc_context(
                f"a strong transient thermal response is observed under high {mode_label} load",
                "transient",
            )
        if transient_supported:
            return add_soc_context(
                "temperature dynamics indicate an unusual transient response",
                "transient",
            )

    # Thermal + transient
    if primary_family == "thermal" and secondary_relevant and secondary_family == "transient":
        if valid_sustained_thermal and transient_supported:
            return add_soc_context(
                "sustained thermal imbalance is dominant, with concurrent unusual transient behaviour",
                "thermal",
            )
        if valid_thermal_imbalance and transient_supported:
            return add_soc_context(
                "thermal non-uniformity is dominant, with concurrent unusual transient behaviour",
                "thermal",
            )

    if primary_family == "transient" and secondary_relevant and secondary_family == "thermal":
        if valid_sustained_thermal and transient_supported:
            return add_soc_context(
                "an unusual transient thermal response is dominant, with concurrent sustained thermal imbalance",
                "transient",
            )
        if valid_thermal_imbalance and transient_supported:
            return add_soc_context(
                "an unusual transient thermal response is dominant, with concurrent thermal non-uniformity",
                "transient",
            )

    # Voltage + thermal
    if primary_family == "voltage" and secondary_relevant and secondary_family == "thermal":
        if valid_sustained_voltage and valid_sustained_thermal:
            return add_soc_context(
                "sustained cell voltage imbalance is dominant, with concurrent sustained thermal imbalance",
                "voltage",
            )
        if valid_voltage_imbalance and valid_thermal_imbalance:
            return add_soc_context(
                "voltage imbalance is dominant, with concurrent thermal non-uniformity",
                "voltage",
            )
        if valid_ratio_voltage and valid_thermal_imbalance:
            return add_soc_context(
                f"relative voltage anomaly is present under low-current {mode_label}, with concurrent thermal non-uniformity",
                "voltage",
            )

    if primary_family == "thermal" and secondary_relevant and secondary_family == "voltage":
        if valid_sustained_thermal and valid_sustained_voltage:
            return add_soc_context(
                "sustained thermal imbalance is dominant, with concurrent sustained cell voltage imbalance",
                "thermal",
            )
        if valid_thermal_imbalance and valid_voltage_imbalance:
            return add_soc_context(
                "thermal non-uniformity is dominant, with concurrent voltage imbalance",
                "thermal",
            )
        # weak ratio-only voltage evidence is intentionally ignored here

    # -----------------------------
    # General composition
    # -----------------------------
    if primary_phrase and secondary_phrase:
        return f"{primary_phrase}, and {secondary_phrase}"

    if primary_phrase:
        return primary_phrase

    return "multiple features jointly deviate from normal operating behaviour"


def add_anomaly_reasons(
    df: pd.DataFrame,
    feature_cols: list[str],
    anomaly_col: str = "anomaly",
) -> pd.DataFrame:
    """
    Add explanation columns directly to the main dataframe.

    For each anomaly row:
    - select normal rows from the same battery mode
    - compute mode-specific mean/std
    - find top 3 deviating features
    - store reasons in new columns

    Falls back to global normal rows if too few normal rows exist for that mode.
    """
    df = df.copy()

    usable_features = [c for c in feature_cols if c in df.columns]
    if not usable_features:
        raise ValueError("No usable feature columns found for explanation.")

    df_normal = df[df[anomaly_col] == 1].copy()
    if df_normal.empty:
        raise ValueError("No normal rows found. Cannot build baseline statistics.")

    # initialize empty columns
    df["battery_mode"] = df.apply(get_row_mode, axis=1)

    df["top_feature_1"] = None
    df["top_feature_2"] = None
    df["top_feature_3"] = None

    df["top anomaly reason 1"] = None
    df["top anomaly reason 2"] = None
    df["top anomaly reason 3"] = None

    df["interpretation"] = None

    anomaly_indices = df.index[df[anomaly_col] == -1]

    for idx in anomaly_indices:
        row = df.loc[idx]

        # mode-aware normal baseline
        df_mode_normal = get_mode_normal_subset(df_normal, row)

        normal_mean = df_mode_normal[usable_features].mean()
        normal_std = df_mode_normal[usable_features].std().replace(0, np.nan)

        deviations = ((row[usable_features] - normal_mean) / normal_std).replace(
            [np.inf, -np.inf], np.nan
        )

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

        if len(feature_summaries) > 0:
            df.at[idx, "top_feature_1"] = feature_summaries[0]["feature"]
            df.at[idx, "top anomaly reason 1"] = build_reason_text(feature_summaries[0])

        if len(feature_summaries) > 1:
            df.at[idx, "top_feature_2"] = feature_summaries[1]["feature"]
            df.at[idx, "top anomaly reason 2"] = build_reason_text(feature_summaries[1])

        if len(feature_summaries) > 2:
            df.at[idx, "top_feature_3"] = feature_summaries[2]["feature"]
            df.at[idx, "top anomaly reason 3"] = build_reason_text(feature_summaries[2])

        df.at[idx, "interpretation"] = infer_physical_interpretation(row, feature_summaries)

    return df