"""deep_dive_data.py
-------------------
Synthetic and multi-station telemetry generator with realistic diurnal cycles
and fault injection, adapted from weather_anomaly_detection.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


def generate_synthetic_weather_data(
    days: int = 30,
    interval_minutes: int = 5,
    station_id: str = "AWS-3",
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate realistic multi-variable weather telemetry for a given station.
    Station profiles:
    - AWS-3: Highland Station (cooler base temp, lower barometric pressure)
    - AWS-1: North Field Station (inland continental, moderate pressure)
    - AWS-2: Coastal Station (higher humidity, marine diurnal damping)
    """
    np.random.seed(seed)
    total_steps = (days * 24 * 60) // interval_minutes
    start_time = pd.Timestamp("2026-08-01 00:00:00")
    timestamps = [start_time + pd.Timedelta(minutes=i * interval_minutes) for i in range(total_steps)]

    # Station profiles
    station_clean = station_id.split()[0].upper()
    if station_clean == "AWS-3":
        base_temp = 16.5
        temp_amplitude = 6.0
        base_pressure = 920.0
        pressure_amplitude = 2.8
        base_humidity = 64.0
    elif station_clean == "AWS-2":
        base_temp = 27.0
        temp_amplitude = 4.5
        base_pressure = 1012.0
        pressure_amplitude = 1.5
        base_humidity = 78.0
    else:  # AWS-1 or default
        base_temp = 24.0
        temp_amplitude = 7.5
        base_pressure = 1008.0
        pressure_amplitude = 3.0
        base_humidity = 62.0

    temperatures = []
    pressures = []
    humidities = []

    for ts in timestamps:
        hour = ts.hour + ts.minute / 60.0

        # Diurnal thermal curve peaking ~15:00
        diurnal_t = np.sin((hour - 9.0) * np.pi / 12.0)
        temp = base_temp + temp_amplitude * diurnal_t + np.random.normal(0, 0.45)

        # Barometric semidiurnal atmospheric tide
        semi_diurnal_p = np.cos((hour - 4.0) * np.pi / 6.0)
        pressure = base_pressure + pressure_amplitude * semi_diurnal_p + np.random.normal(0, 0.25)

        # Psychrometric inverse correlation with temperature
        rel_hum = base_humidity - 1.15 * (temp - base_temp) + np.random.normal(0, 1.2)
        rel_hum = np.clip(rel_hum, 15.0, 98.0)

        temperatures.append(round(temp, 2))
        pressures.append(round(pressure, 2))
        humidities.append(round(rel_hum, 2))

    df = pd.DataFrame({
        "timestamp": timestamps,
        "station_id": station_clean,
        "temperature": temperatures,
        "pressure": pressures,
        "humidity": humidities,
    })

    return df


def inject_anomalies(
    df: pd.DataFrame,
    n_spikes: int = 8,
    n_frozen: int = 8,
    n_drift: int = 8,
    n_gaps: int = 8,
    seed: int = 42
) -> pd.DataFrame:
    """
    Inject realistic sensor fault signatures:
    1. Spikes: isolated extreme deviations
    2. Frozen: flatline readings across consecutive steps
    3. Drift: slow progressive calibration bias
    4. Communication Gaps: missing telemetry packets (NaNs)
    """
    np.random.seed(seed)
    df_inj = df.copy()
    n_rows = len(df_inj)

    is_anomaly = np.zeros(n_rows, dtype=int)
    anomaly_type = np.array(["Normal"] * n_rows, dtype=object)
    anomaly_feature = np.array(["none"] * n_rows, dtype=object)

    features = ["temperature", "pressure", "humidity"]
    used_indices = set()

    # 1. Spikes
    for _ in range(n_spikes):
        idx = np.random.randint(15, n_rows - 15)
        while idx in used_indices:
            idx = np.random.randint(15, n_rows - 15)
        used_indices.add(idx)

        feat = np.random.choice(features)
        direction = np.random.choice([-1, 1])
        mag = 12.0 if feat == "temperature" else (28.0 if feat == "pressure" else 30.0)
        df_inj.loc[idx, feat] = round(float(df_inj.loc[idx, feat]) + direction * mag, 2)
        is_anomaly[idx] = 1
        anomaly_type[idx] = "Spike"
        anomaly_feature[idx] = feat

    # 2. Frozen Sensor
    for _ in range(n_frozen):
        idx = np.random.randint(25, n_rows - 25)
        duration = np.random.randint(8, 16)
        if any((idx + k) in used_indices for k in range(duration)):
            continue
        feat = np.random.choice(features)
        freeze_val = float(df_inj.loc[idx, feat])
        for k in range(duration):
            pos = idx + k
            if pos < n_rows:
                df_inj.loc[pos, feat] = freeze_val
                is_anomaly[pos] = 1
                anomaly_type[pos] = "Frozen"
                anomaly_feature[pos] = feat
                used_indices.add(pos)

    # 3. Drift
    for _ in range(n_drift):
        idx = np.random.randint(30, n_rows - 50)
        duration = np.random.randint(20, 35)
        if any((idx + k) in used_indices for k in range(duration)):
            continue
        feat = np.random.choice(features)
        slope = np.random.choice([-1, 1]) * (0.28 if feat == "temperature" else (0.55 if feat == "pressure" else 0.85))
        for k in range(duration):
            pos = idx + k
            if pos < n_rows:
                df_inj.loc[pos, feat] = round(float(df_inj.loc[pos, feat]) + slope * k, 2)
                is_anomaly[pos] = 1
                anomaly_type[pos] = "Drift"
                anomaly_feature[pos] = feat
                used_indices.add(pos)

    # 4. Communication Gaps
    for _ in range(n_gaps):
        idx = np.random.randint(20, n_rows - 20)
        duration = np.random.randint(4, 10)
        if any((idx + k) in used_indices for k in range(duration)):
            continue
        for k in range(duration):
            pos = idx + k
            if pos < n_rows:
                for feat in features:
                    df_inj.loc[pos, feat] = np.nan
                is_anomaly[pos] = 1
                anomaly_type[pos] = "Communication gap"
                anomaly_feature[pos] = "all"
                used_indices.add(pos)

    df_inj["is_anomaly"] = is_anomaly
    df_inj["anomaly_type"] = anomaly_type
    df_inj["anomaly_feature"] = anomaly_feature

    return df_inj
