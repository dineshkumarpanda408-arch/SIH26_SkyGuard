"""
anomaly_injector.py
-------------------
Injects realistic meteorological sensor anomaly signatures into clean weather telemetry:
1. Spike: Single isolated extreme value returning immediately to normal
2. Frozen: Sensor stuck on a constant flatline reading for several consecutive steps
3. Drift: Progressive linear/non-linear calibration drift over 20-30 consecutive points
4. Communication Gap: Missing/NaN sensor data packet loss across telemetry stream

Generates ground-truth columns:
- 'is_anomaly': 1 (anomaly) or 0 (normal)
- 'anomaly_type': 'Normal', 'Spike', 'Frozen', 'Drift', 'Communication gap'
- 'anomaly_feature': Affected sensor feature ('temperature', 'pressure', 'humidity', 'all')
- 'anomaly_description': Human-readable diagnostic description
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any


def inject_anomalies(
    df: pd.DataFrame,
    n_spikes: int = 8,
    n_frozen: int = 8,
    n_drift: int = 8,
    n_gaps: int = 8,
    seed: int = 42
) -> pd.DataFrame:
    """
    Inject synthetic anomalies into weather telemetry dataset.

    Parameters:
    -----------
    df : pd.DataFrame
        Clean weather DataFrame containing ['timestamp', 'station_id', 'temperature', 'pressure', 'humidity'].
    n_spikes : int
        Number of single-point spike anomalies to inject.
    n_frozen : int
        Number of frozen sensor episodes to inject.
    n_drift : int
        Number of gradual sensor drift episodes to inject.
    n_gaps : int
        Number of communication gap episodes (null values) to inject.
    seed : int
        Random seed for reproducibility.

    Returns:
    --------
    pd.DataFrame
        DataFrame with injected anomalies and ground-truth annotation columns.
    """
    np.random.seed(seed)
    df_out = df.copy()

    total_len = len(df_out)
    features = ["temperature", "pressure", "humidity"]

    # Initialize ground-truth label columns
    df_out["is_anomaly"] = 0
    df_out["anomaly_type"] = "Normal"
    df_out["anomaly_feature"] = "None"
    df_out["anomaly_description"] = "Normal sensor telemetry"

    # Track occupied indices to avoid overlapping injections
    occupied_indices = set()

    def get_valid_start_index(length: int, buffer: int = 15) -> int:
        max_attempts = 1000
        for _ in range(max_attempts):
            start = np.random.randint(buffer, total_len - length - buffer)
            span = set(range(start - buffer, start + length + buffer))
            if not span.intersection(occupied_indices):
                occupied_indices.update(range(start, start + length))
                return start
        # Fallback if dense
        start = np.random.randint(buffer, total_len - length - buffer)
        occupied_indices.update(range(start, start + length))
        return start

    # -------------------------------------------------------------
    # 1. SPIKE ANOMALIES (Single-point extreme deviation)
    # -------------------------------------------------------------
    for _ in range(n_spikes):
        idx = get_valid_start_index(length=1, buffer=10)
        target_feat = np.random.choice(features)
        original_val = df_out.loc[idx, target_feat]

        if target_feat == "temperature":
            # Spike up or down by 14°C to 25°C
            direction = np.random.choice([-1, 1])
            delta = direction * np.random.uniform(14.0, 24.0)
            injected_val = original_val + delta
        elif target_feat == "pressure":
            # Spike by 25 to 55 hPa
            direction = np.random.choice([-1, 1])
            delta = direction * np.random.uniform(25.0, 50.0)
            injected_val = original_val + delta
        else:  # humidity
            # Jump to impossible or extreme humidity values
            injected_val = np.random.choice([np.random.uniform(0.0, 8.0), np.random.uniform(105.0, 130.0)])

        df_out.loc[idx, target_feat] = round(injected_val, 2)
        df_out.loc[idx, "is_anomaly"] = 1
        df_out.loc[idx, "anomaly_type"] = "Spike"
        df_out.loc[idx, "anomaly_feature"] = target_feat
        df_out.loc[idx, "anomaly_description"] = f"Sudden transient spike on {target_feat} ({original_val:.1f} -> {injected_val:.1f})"

    # -------------------------------------------------------------
    # 2. FROZEN SENSOR ANOMALIES (Constant flatline for 10-25 readings)
    # -------------------------------------------------------------
    for _ in range(n_frozen):
        duration = np.random.randint(10, 26)  # 50 to 125 minutes
        start_idx = get_valid_start_index(length=duration, buffer=10)
        target_feat = np.random.choice(features)
        
        # Frozen at the initial reading or fixed value
        frozen_val = df_out.loc[start_idx, target_feat]

        for offset in range(duration):
            curr_idx = start_idx + offset
            df_out.loc[curr_idx, target_feat] = frozen_val
            df_out.loc[curr_idx, "is_anomaly"] = 1
            df_out.loc[curr_idx, "anomaly_type"] = "Frozen"
            df_out.loc[curr_idx, "anomaly_feature"] = target_feat
            df_out.loc[curr_idx, "anomaly_description"] = f"Sensor output frozen at {frozen_val:.2f} across {duration} steps"

    # -------------------------------------------------------------
    # 3. DRIFT ANOMALIES (Progressive calibration loss over 20-35 points)
    # -------------------------------------------------------------
    for _ in range(n_drift):
        duration = np.random.randint(20, 36)  # 100 to 175 minutes
        start_idx = get_valid_start_index(length=duration, buffer=10)
        target_feat = np.random.choice(features)
        direction = np.random.choice([-1, 1])

        if target_feat == "temperature":
            max_drift = direction * np.random.uniform(9.0, 18.0)
        elif target_feat == "pressure":
            max_drift = direction * np.random.uniform(18.0, 35.0)
        else:  # humidity
            max_drift = direction * np.random.uniform(25.0, 50.0)

        # Gradual non-linear/linear ramp
        ramp = np.linspace(0.1, 1.0, duration) ** 1.3

        for offset in range(duration):
            curr_idx = start_idx + offset
            drift_amount = max_drift * ramp[offset]
            orig_val = df_out.loc[curr_idx, target_feat]
            injected_val = orig_val + drift_amount

            # Keep humidity somewhat in observable scale
            if target_feat == "humidity":
                injected_val = np.clip(injected_val, 0.0, 120.0)

            df_out.loc[curr_idx, target_feat] = round(injected_val, 2)
            df_out.loc[curr_idx, "is_anomaly"] = 1
            df_out.loc[curr_idx, "anomaly_type"] = "Drift"
            df_out.loc[curr_idx, "anomaly_feature"] = target_feat
            df_out.loc[curr_idx, "anomaly_description"] = f"Gradual sensor calibration drift (+{drift_amount:.2f} deviation)"

    # -------------------------------------------------------------
    # 4. COMMUNICATION GAP (Missing/null values across 6-18 readings)
    # -------------------------------------------------------------
    for _ in range(n_gaps):
        duration = np.random.randint(6, 18)  # 30 to 90 minutes
        start_idx = get_valid_start_index(length=duration, buffer=10)

        for offset in range(duration):
            curr_idx = start_idx + offset
            # Communication loss drops all sensor telemetry channels
            df_out.loc[curr_idx, "temperature"] = np.nan
            df_out.loc[curr_idx, "pressure"] = np.nan
            df_out.loc[curr_idx, "humidity"] = np.nan
            df_out.loc[curr_idx, "is_anomaly"] = 1
            df_out.loc[curr_idx, "anomaly_type"] = "Communication gap"
            df_out.loc[curr_idx, "anomaly_feature"] = "all"
            df_out.loc[curr_idx, "anomaly_description"] = f"Telemetry dropout / packet loss ({duration * 5} mins)"

    return df_out


if __name__ == "__main__":
    from data_generator import generate_synthetic_weather_data

    print("Generating base clean weather dataset...")
    df_clean = generate_synthetic_weather_data(days=30, interval_minutes=5)

    print("Injecting 4 types of synthetic anomalies...")
    df_anom = inject_anomalies(df_clean, n_spikes=8, n_frozen=8, n_drift=8, n_gaps=8)

    print(f"\nTotal Data Points: {len(df_anom)}")
    print(f"Total Anomalies Injected: {df_anom['is_anomaly'].sum()} ({(df_anom['is_anomaly'].mean() * 100):.2f}%)")
    print("\nAnomaly Breakdown by Type:")
    print(df_anom["anomaly_type"].value_counts())
    print("\nSample Injected Anomalies:")
    print(df_anom[df_anom["is_anomaly"] == 1][["timestamp", "temperature", "pressure", "humidity", "anomaly_type", "anomaly_feature"]].head(15))
