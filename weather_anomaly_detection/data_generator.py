"""
data_generator.py
-----------------
Generates realistic synthetic multi-parameter weather telemetry for Automatic Weather Stations (AWS).
Features:
- 30 days of data at 5-minute intervals (8,640 records per station)
- Extensible multi-station schema with distinct climatological profiles for AWS-1, AWS-2, AWS-3:
    * AWS-1 (Inland Plains): Warm diurnal swings (28°C base, 60% humidity, 1013.25 hPa)
    * AWS-2 (Coastal Marine): Humid maritime climate (31.5°C base, 82% humidity, 1008.5 hPa)
    * AWS-3 (Highland Mountain): High-altitude cooler climate (19.5°C base, 74% humidity, 945.0 hPa)
- Realistic physical meteorology models:
    * Diurnal temperature cycle (trough at ~05:00, peak at ~14:00) + multi-day synoptic weather fronts
    * Inverse humidity correlation with temperature (thermodynamic psychrometric relationship)
    * Atmospheric barometric pressure variations including smooth frontal transitions & semi-diurnal thermal tides
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

# Distinct Climatological Profiles for Different Automatic Weather Stations
STATION_PROFILES: Dict[str, Dict[str, Any]] = {
    "AWS-1": {
        "name": "North Field Station (Inland Plains)",
        "base_temp": 28.0,
        "temp_amplitude": 8.0,
        "base_pressure": 1013.25,
        "base_humidity": 60.0,
        "seed_offset": 101
    },
    "AWS-2": {
        "name": "Coastal Station (Marine Climatology)",
        "base_temp": 31.5,
        "temp_amplitude": 4.5,
        "base_pressure": 1008.50,
        "base_humidity": 82.0,
        "seed_offset": 202
    },
    "AWS-3": {
        "name": "Highland Station (Mountain Climatology)",
        "base_temp": 19.5,
        "temp_amplitude": 6.0,
        "base_pressure": 945.00,
        "base_humidity": 74.0,
        "seed_offset": 303
    }
}


def generate_synthetic_weather_data(
    days: int = 30,
    interval_minutes: int = 5,
    station_id: str = "AWS-1",
    start_time: Optional[datetime] = None,
    base_temp: Optional[float] = None,
    temp_amplitude: Optional[float] = None,
    base_pressure: Optional[float] = None,
    base_humidity: Optional[float] = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic, physically consistent weather telemetry with station-specific profiles.

    Parameters:
    -----------
    days : int
        Number of days of data to generate (default 30).
    interval_minutes : int
        Telemetry reporting interval in minutes (default 5).
    station_id : str
        Station identifier (e.g., 'AWS-1', 'AWS-2', 'AWS-3').
    start_time : datetime, optional
        Start timestamp. Defaults to 30 days prior to today at 00:00:00.
    base_temp : float, optional
        Mean temperature baseline in °C (inferred from station profile if None).
    temp_amplitude : float, optional
        Day/night diurnal temperature swing in °C.
    base_pressure : float, optional
        Mean atmospheric pressure in hPa.
    base_humidity : float, optional
        Mean relative humidity baseline in %.
    seed : int
        Random seed for reproducibility.

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: ['timestamp', 'station_id', 'temperature', 'pressure', 'humidity']
    """
    # Load station-specific climatology profile if available
    profile = STATION_PROFILES.get(station_id, {})
    effective_seed = seed + profile.get("seed_offset", 0)
    np.random.seed(effective_seed)

    if base_temp is None:
        base_temp = profile.get("base_temp", 28.0)
    if temp_amplitude is None:
        temp_amplitude = profile.get("temp_amplitude", 7.5)
    if base_pressure is None:
        base_pressure = profile.get("base_pressure", 1013.25)
    if base_humidity is None:
        base_humidity = profile.get("base_humidity", 65.0)

    if start_time is None:
        start_time = datetime(2026, 8, 1, 0, 0, 0)

    total_steps = int((days * 24 * 60) / interval_minutes)
    time_index = [start_time + timedelta(minutes=i * interval_minutes) for i in range(total_steps)]

    # Time variables
    hours_of_day = np.array([t.hour + t.minute / 60.0 for t in time_index])
    day_indices = np.array([(t - start_time).total_seconds() / 86400.0 for t in time_index])

    # 1. Temperature Model (°C)
    diurnal_temp = np.sin((hours_of_day - 8.0) * (2 * np.pi / 24.0))
    synoptic_temp_wave = 2.5 * np.sin(2 * np.pi * day_indices / 5.5) + 1.2 * np.cos(2 * np.pi * day_indices / 11.0)
    temp_noise = np.random.normal(0, 0.4, size=total_steps)
    temperature = base_temp + (temp_amplitude * diurnal_temp) + synoptic_temp_wave + temp_noise

    # 2. Pressure Model (hPa)
    synoptic_pressure = -8.0 * np.sin(2 * np.pi * day_indices / 5.5 + 0.5) + 3.0 * np.sin(2 * np.pi * day_indices / 13.0)
    barometric_tide = 1.2 * np.sin(hours_of_day * (4 * np.pi / 24.0) + 0.3)
    pressure_noise = np.random.normal(0, 0.15, size=total_steps)
    pressure = base_pressure + synoptic_pressure + barometric_tide + pressure_noise

    # 3. Humidity Model (%)
    humidity_diurnal = -25.0 * diurnal_temp
    humidity_synoptic = 8.0 * np.sin(2 * np.pi * day_indices / 5.5 + np.pi / 4)
    humidity_noise = np.random.normal(0, 1.2, size=total_steps)
    humidity = base_humidity + humidity_diurnal + humidity_synoptic + humidity_noise
    humidity = np.clip(humidity, 15.0, 98.0)

    # Assemble into DataFrame
    df = pd.DataFrame({
        "timestamp": time_index,
        "station_id": station_id,
        "temperature": np.round(temperature, 2),
        "pressure": np.round(pressure, 2),
        "humidity": np.round(humidity, 2)
    })

    return df


if __name__ == "__main__":
    print("Comparing Climatological Profiles across AWS stations:")
    for stn in ["AWS-1", "AWS-2", "AWS-3"]:
        df = generate_synthetic_weather_data(days=1, station_id=stn)
        print(f"\n[{stn}] {STATION_PROFILES[stn]['name']}:")
        print(f"  Mean Temp: {df['temperature'].mean():.1f}°C (Min: {df['temperature'].min():.1f}, Max: {df['temperature'].max():.1f})")
        print(f"  Mean Pressure: {df['pressure'].mean():.1f} hPa")
        print(f"  Mean Humidity: {df['humidity'].mean():.1f}%")
