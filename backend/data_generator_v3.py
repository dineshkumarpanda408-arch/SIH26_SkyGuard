"""Generate the improved synthetic weather dataset (v3).

v3 is 60 days (vs 14) with substantially more temporal coverage and multiple
weather regimes, so the training data exposes the detector to far more genuine
variation without any row duplication.

Compared with v2 (data_generator.py, 14 days / single synoptic drive):

  * longer span: 24 stations x 60 days x 96 readings/day = 138,240 observations
  * weather REGIMES: a smooth, deterministic regime state drives warm/cool,
    dry/humid and pressure-rise/fall phases
  * episodic rain: intermittent, decaying events that dip temperature, lower
    pressure and raise humidity in a physically-consistent way
  * physical relationships preserved: temperature/humidity anti-correlation,
    pressure/temperature coupling, regional (distance-decayed) co-movement

The dataset remains SYNTHETIC: explicit in meta (synthetic=true,
dataset_version='sample_weather_v3'). Output schema is IDENTICAL to v2
(timestamp, station_id, temperature, pressure, humidity) so the training,
evaluation and dashboard pipelines keep working unchanged.

Fully reproducible: every random sequence is seeded; re-running the generator
produces byte-identical output (verified via dataset sha256).
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from data_generator import (
    STATION_META,
    STATION_ELEV,
    _station_params,
    _regional_weather,
    _haversine_km,
    CLUSTER_LAT,
    CLUSTER_LON,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT = DATA_DIR / "sample_weather_v3.csv"
OUT_META = DATA_DIR / "sample_weather_v3.meta.json"

DAYS = 60
INTERVAL_MIN = 15  # one reading every 15 minutes (identical grid as v2)

REGIME_SEED = 4242
EPISODE_SEED = 777


def _v3_regime_drivers(n):
    """Deterministic regime state, pressure trend and rain episodes.

    All three are seeded random sequences, smoothed to weather-sensible time
    scales (~2-3 days), so they ARE real temporal structure but reproducible.
    """
    rng_r = np.random.default_rng(REGIME_SEED)
    rng_e = np.random.default_rng(EPISODE_SEED)

    # smooth regime state (drives warm/cool + dry/humid swings)
    inc = rng_r.normal(0, 1.0, n).cumsum()
    reg = pd.Series(inc).rolling(192, min_periods=1).mean().ffill().to_numpy()
    reg = (reg - reg.mean()) / (reg.std() + 1e-9)

    # slow pressure trend (rise/fall phases)
    inc2 = rng_r.normal(0, 0.30, n).cumsum()
    pramp = pd.Series(inc2).rolling(288, min_periods=1).mean().ffill().to_numpy()
    pramp = pramp / (pramp.std() + 1e-9)

    # long (~monthly) wave for additional regime diversity
    days = np.arange(n) * INTERVAL_MIN / 1440.0
    lw = 1.5 * np.sin(2 * math.pi * days / 23.0) + 0.8 * np.cos(2 * math.pi * days / 9.0)

    # episodic rain: sparse starts with exponential decay, merged
    rain = np.zeros(n)
    i = 0
    while i < n:
        if rng_e.uniform() < 0.012:
            dur = int(rng_e.integers(32, 160))  # 8h to 40h episode
            peak = rng_e.uniform(0.7, 1.0)
            dec = np.exp(-np.linspace(0, 3, dur))
            dec = dec / (dec.max() + 1e-9)
            i2 = 0
            while i2 < dur and i + i2 < n:
                rain[i + i2] = max(rain[i + i2], peak * dec[i2])
                i2 += 1
        i += 1

    return reg, pramp, lw, rain


def gen_station_v3(idx, sid, name, lat, lon, regional, drivers, start, rng_station):
    """Generate one station's 60-day series (deterministic given idx + seed)."""
    n = int(DAYS * 24 * 60 / INTERVAL_MIN)
    p = _station_params(idx, sid, lat, lon)
    w_temp, w_press, w_hum, clock_h, days = regional
    reg, pramp, lw, rain = drivers
    hours = clock_h % 24.0
    load = p["regional_load"]

    # temperature: diurnal + regime-loaded regional weather + rain dip + local noise
    diurnal = p["temp_amplitude"] * np.sin((hours - p["temp_phase_h"]) * 2 * math.pi / 24.0)
    temperature = (
        p["base_temp"]
        + diurnal
        + lw
        + load * w_temp
        + 3.2 * np.tanh(reg)
        - 2.4 * rain
        + rng_station.normal(0, p["temp_noise"], n)
    )

    # pressure: synoptic + semi-diurnal tide + regime/pressure-trend loading
    p_syn = load * w_press + p["pressure_synodic"] * 0.9 * np.sin(hours * 4 * math.pi / 24.0 + 0.3)
    pressure = (
        p["base_pressure"]
        + p_syn
        - 0.10 * diurnal
        + 3.5 * pramp
        - 1.1 * reg
        - 4.6 * rain
        + rng_station.normal(0, p["pressure_noise"], n)
    )

    # humidity: inverse diurnal + regime humidity + rain episodes
    humidity = (
        p["base_humidity"]
        - 18.0 * diurnal / p["temp_amplitude"]
        + load * w_hum
        - 2.0 * reg
        + 17.0 * rain
        + rng_station.normal(0, p["humidity_noise"], n)
    )
    humidity = np.clip(humidity, 30.0, 98.0)

    timestamps = [start + pd.Timedelta(minutes=INTERVAL_MIN * i) for i in range(n)]
    return pd.DataFrame(
        {
            "station_id": sid,
            "timestamp": timestamps,
            "temperature": temperature.round(1),
            "pressure": pressure.round(1),
            "humidity": humidity.round(1),
        }
    )


def generate_frame_v3(seed=91, stations=None):
    """Build the full 24-station v3 frame (deterministic)."""
    stations = stations or STATION_META
    n = int(DAYS * 24 * 60 / INTERVAL_MIN)
    start = pd.Timestamp("2026-06-01 00:00:00")
    regional = _regional_weather(n, INTERVAL_MIN, seed)
    drivers = _v3_regime_drivers(n)
    frames = []
    for idx, (sid, name, lat, lon) in enumerate(stations):
        rng_station = np.random.default_rng(seed + 1000 * (idx + 1))
        frames.append(gen_station_v3(idx, sid, name, lat, lon, regional, drivers, start, rng_station))
    return pd.concat(frames, ignore_index=True)


def dataset_sha256(frame):
    """Deterministic SHA-256 of the frame content (stable across runs)."""
    digest = hashlib.sha256()
    digest.update(frame.sort_values(["station_id", "timestamp"]).to_csv(index=False).encode("utf-8"))
    return digest.hexdigest()


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_frame_v3()
    df.to_csv(OUT, index=False)

    meta = {
        "dataset_version": "sample_weather_v3",
        "synthetic": True,
        "generator": "backend/data_generator_v3.py",
        "seed": 91,
        "regime_seed": REGIME_SEED,
        "episode_seed": EPISODE_SEED,
        "stations": [s[0] for s in STATION_META],
        "n_stations": len(STATION_META),
        "n_observations": int(len(df)),
        "records_per_station": int(len(df) / len(STATION_META)),
        "interval_minutes": INTERVAL_MIN,
        "start": "2026-06-01 00:00:00",
        "days": DAYS,
        "variables": ["temperature", "pressure", "humidity"],
        "regimes": ["warm/cool phase", "dry/humid phase", "pressure rise/fall", "episodic rain"],
        "sha256": dataset_sha256(df),
        "note": "Synthetic, station-distinct weather telemetry (lat/elevation/diurnal/"
                "spatial-correlation + regime/rain drivers). Not real observed weather.",
    }
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    means = df.groupby("station_id")["temperature"].mean().round(2)
    spread = round(float(means.max() - means.min()), 2)
    print(f"Wrote {len(df)} readings to {OUT}")
    print(f"sha256 = {meta['sha256']}")
    print(f"records/station = {meta['records_per_station']}  station mean-temp spread = {spread} C")
    print(f"rain coverage  = {round(100 * (df['humidity'] > 80).mean(), 1)}% rows above 80% RH")


if __name__ == "__main__":
    main()