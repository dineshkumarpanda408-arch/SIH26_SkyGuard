"""Evaluation-only, realistic weather telemetry generator (isolated from production).

NOT used by the production pipeline or detector. It exists purely to build a
larger, reproducible ground-truth evaluation set with a clean temporal
train/test split (no train/test overlap -> no leakage into headline metrics).

The generator is calibrated to the real AWS-023 climatology observed in the raw
sample CSV (mean temperature, diurnal amplitude, pressure baseline, humidity
baseline) so the synthetic telemetry is physically plausible rather than toy.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# Real AWS-023 climatology (measured from the sample data)
AWS023_PROFILE = {
    "base_temp": 28.0,
    "temp_amplitude": 9.8,       # measured diurnal range ~9.9 C
    "temp_phase_hours": 8.0,     # trough ~03:00, peak ~14:00 -> sin((h-8)*2pi/24)
    "temp_noise": 0.55,
    "base_pressure": 1008.0,
    "pressure_synodic": 3.5,
    "pressure_noise": 0.25,
    "base_humidity": 70.0,
    "humidity_noise": 1.8,
    "humidity_min": 30.0,
    "humidity_max": 98.0,
}


def _series(steps, start, profile, interval_minutes, seed):
    rng = np.random.default_rng(seed)
    hours = 24.0 * np.arange(steps) / (1440 / interval_minutes) % 24.0
    days = np.arange(steps) * interval_minutes / 1440.0

    # temperature: diurnal + synoptic weather fronts + noise
    diurnal = profile["temp_amplitude"] * np.sin((hours - profile["temp_phase_hours"]) * 2 * np.pi / 24.0)
    synoptic = 2.2 * np.sin(2 * np.pi * days / 5.5) + 1.1 * np.cos(2 * np.pi * days / 11.0)
    temp = profile["base_temp"] + diurnal + synoptic + rng.normal(0, profile["temp_noise"], steps)

    # pressure: synoptic fronts + semi-diurnal tide + noise
    p_syn = profile["pressure_synodic"] * np.sin(2 * np.pi * days / 5.5 + 0.5)
    p_syn += 1.2 * np.sin(2 * np.pi * days / 13.0)
    tide = 0.9 * np.sin(hours * 4 * np.pi / 24.0 + 0.3)
    pressure = profile["base_pressure"] + p_syn + tide + rng.normal(0, profile["pressure_noise"], steps)

    # humidity: inverse diurnal + synoptic + noise, clipped to physical domain
    hum = profile["base_humidity"] - 22.0 * diurnal / profile["temp_amplitude"]
    hum += 5.0 * np.sin(2 * np.pi * days / 5.5 + np.pi / 4)
    hum += rng.normal(0, profile["humidity_noise"], steps)
    hum = np.clip(hum, profile["humidity_min"], profile["humidity_max"])

    ts = [start + timedelta(minutes=int(i * interval_minutes)) for i in range(steps)]
    return pd.DataFrame({
        "timestamp": ts,
        "temperature": np.round(temp, 2),
        "pressure": np.round(pressure, 2),
        "humidity": np.round(hum, 2),
    })


def generate_realistic_series(steps, interval_minutes=15, station_id="AWS-023", start=None, seed=42):
    """Return a realistic multi-day telemetry frame for a single station."""
    if start is None:
        start = datetime(2026, 8, 1, 0, 0, 0)
    df = _series(steps, start, AWS023_PROFILE, interval_minutes, seed)
    df["station_id"] = station_id
    return df[["timestamp", "station_id", "temperature", "pressure", "humidity"]]


def split_train_test(df, train_frac=0.6):
    """Temporal train/test split (no overlap). Returns (train, test) frames."""
    df = df.reset_index(drop=True)
    cut = int(len(df) * train_frac)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


# Hard, realistic scenarios that stress the detector beyond the uniform
# default-magnitude injections. Each is a dict used by SkyGuardApp._run_scenario.
HARD_SCENARIOS = [
    {
        "id": "small_spike",
        "label": "Small SPIKE (subtle single-point jump)",
        "kind": "single",
        "variable": "temperature",
        "anomaly_type": "SPIKE",
        "magnitude": 3.0,
        "duration": 1,
        "expect_type": "SPIKE",
    },
    {
        "id": "small_drop",
        "label": "Small DROP (subtle single-point dip)",
        "kind": "single",
        "variable": "humidity",
        "anomaly_type": "DROP",
        "magnitude": 3.5,
        "duration": 1,
        "expect_type": "DROP",
    },
    {
        "id": "gradual_drift",
        "label": "Gradual DRIFT (long, low slope)",
        "kind": "single",
        "variable": "temperature",
        "anomaly_type": "DRIFT",
        "magnitude": 6.0,
        "duration": 16,
        "expect_type": "DRIFT",
    },
    {
        "id": "long_frozen",
        "label": "Long FROZEN_SENSOR (20-reading plateau)",
        "kind": "single",
        "variable": "pressure",
        "anomaly_type": "FROZEN_SENSOR",
        "duration": 20,
        "expect_type": "FROZEN_SENSOR",
    },
    {
        "id": "short_frozen",
        "label": "Short FROZEN_SENSOR (3-reading plateau)",
        "kind": "single",
        "variable": "temperature",
        "anomaly_type": "FROZEN_SENSOR",
        "duration": 3,
        "expect_type": "FROZEN_SENSOR",
    },
    {
        "id": "partial_missing",
        "label": "Partial MISSING_DATA (isolated single gap)",
        "kind": "single",
        "variable": "pressure",
        "anomaly_type": "MISSING_DATA",
        "duration": 1,
        "expect_type": "MISSING_DATA",
    },
    {
        "id": "long_missing",
        "label": "Long MISSING (communication-failure run)",
        "kind": "single",
        "variable": "humidity",
        "anomaly_type": "MISSING_DATA",
        "duration": 12,
        "expect_type": "MISSING_DATA",
    },
    {
        "id": "multi_spike",
        "label": "Multi-sensor simultaneous SPIKE (temp+pressure)",
        "kind": "multi",
        "variables": ["temperature", "pressure"],
        "anomaly_type": "SPIKE",
        "magnitude": 6.0,
        "duration": 1,
        "expect_type": "SPIKE",
    },
    {
        "id": "multi_drift",
        "label": "Multi-sensor simultaneous DRIFT (temp+humidity)",
        "kind": "multi",
        "variables": ["temperature", "humidity"],
        "anomaly_type": "DRIFT",
        "magnitude": 5.0,
        "duration": 12,
        "expect_type": "DRIFT",
    },
    {
        "id": "sensor_fault_one_station",
        "label": "Sensor fault, all neighbors normal",
        "kind": "spatial",
        "variable": "temperature",
        "neighbor_mode": "all_normal",
        "expect_assessment": "LIKELY_SENSOR_FAULT",
    },
    {
        "id": "weather_event_many_stations",
        "label": "Weather event, many neighbors abnormal",
        "kind": "spatial",
        "variable": "temperature",
        "neighbor_mode": "many_abnormal",
        "expect_assessment": "LIKELY_WEATHER_EVENT",
    },
]

