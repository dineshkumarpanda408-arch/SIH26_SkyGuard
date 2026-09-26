"""Generate the improved synthetic weather dataset (v2).

STATIONS ARE NO LONGER NEAR-IDENTICAL COPIES. Each of the 24 AWS stations gets
its own deterministic parameter set derived from its station metadata:

  * station-specific temperature/pressure/humidity baselines (latitude,
    elevation-driven pressure, humidity offset)
  * station-specific diurnal amplitude and phase (local solar time)
  * spatially correlated regional weather drivers: synoptic waves whose loading
    decays with distance from the Odisha cluster centroid, so nearby stations
    covary and distant stations do not
  * independent local sensor noise from a per-station RNG stream (seeded)

The dataset remains SYNTHETIC: it is explicit in the metadata (synthetic=true,
dataset_version='sample_weather_v2'). No real observed weather is claimed.
Output schema is unchanged (timestamp, station_id, temperature, pressure,
humidity) on the same 15-minute grid / 14-day span as v1 so the training,
evaluation and dashboard pipelines keep working.

The generator is fully reproducible: every station uses default_rng(seed +
1000*station_index), so re-running produces byte-identical output.
"""

import hashlib
import math

import numpy as np
import pandas as pd
from pathlib import Path

from app.data_generator_meta import STATION_META, STATION_ELEV

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT = DATA_DIR / "sample_weather.csv"
OUT_META = DATA_DIR / "sample_weather_v2.meta.json"

DAYS = 14
INTERVAL_MIN = 15  # one reading every 15 minutes

# Cluster centroid of the Odisha station network (approx), used for the
# distance-decayed regional weather field.
CLUSTER_LAT, CLUSTER_LON = 20.5, 84.9

REALISM_SEED = 42


def _haversine_km(a_lat, a_lon, b_lat, b_lon):
    r = 6371.0
    p1, p2 = math.radians(a_lat), math.radians(b_lat)
    dp = math.radians(b_lat - a_lat)
    dl = math.radians(b_lon - a_lon)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def _station_params(idx, sid, lat, lon):
    """Deterministic, station-specific physical parameters (no shared curve)."""
    rng = np.random.default_rng(5000 + 1000 * idx)  # stable, independent per station
    elev = STATION_ELEV.get(sid, 40)

    base_temp = 27.5 - (abs(lat) - 18.5) * 0.30 + rng.normal(0, 0.6)
    temp_amplitude = 8.0 + 0.9 * math.cos(math.radians(lat)) + rng.uniform(-0.5, 0.5)
    temp_phase_h = 9.2 + (lon - 84.0) * 0.02 + rng.uniform(-0.4, 0.4)  # local solar time
    temp_noise = rng.uniform(0.35, 0.6)

    # elevation-driven pressure baseline: standard barometric formula, simplified
    base_pressure = 1013.25 * ((1 - 2.25577e-5 * elev) ** 5.25588) + rng.normal(0, 0.8)
    pressure_synodic = 2.8 + rng.uniform(0.5, 1.2)
    pressure_noise = rng.uniform(0.18, 0.35)

    base_humidity = 68.0 + rng.normal(0, 2.5)  # per-station offset
    humidity_noise = rng.uniform(1.4, 2.4)

    # regional loading: distance-decayed from the cluster centroid
    dist_km = _haversine_km(lat, lon, CLUSTER_LAT, CLUSTER_LON)
    regional_load = float(np.exp(-dist_km / 130.0)) * rng.uniform(0.7, 1.0)

    return {
        "base_temp": base_temp,
        "temp_amplitude": temp_amplitude,
        "temp_phase_h": temp_phase_h,
        "temp_noise": temp_noise,
        "base_pressure": base_pressure,
        "pressure_synodic": pressure_synodic,
        "pressure_noise": pressure_noise,
        "base_humidity": base_humidity,
        "humidity_noise": humidity_noise,
        "regional_load": regional_load,
        "regional_seed_phase": rng.uniform(0, 2 * math.pi),
    }


def _regional_weather(n, interval_minutes, seed):
    """Shared regional synoptic drivers (common across stations, time-varying).

    Returns arrays for temperature-like, pressure-like and humidity-like
    synoptic contributions plus the shared clock. The WAVE is identical for all
    stations; each station applies its own distance-dependent load.
    """
    rng = np.random.default_rng(seed)
    clock_h = np.arange(n) * interval_minutes / 60.0
    days = np.arange(n) * interval_minutes / 1440.0
    phi = rng.uniform(0, 2 * math.pi, 3)

    w_temp = (
        2.1 * np.sin(2 * math.pi * days / 5.5 + phi[0])
        + 1.0 * np.cos(2 * math.pi * days / 11.0 + phi[1])
        + 0.5 * np.sin(2 * math.pi * days / 13.0 + phi[2])
    )
    w_press = (
        2.6 * np.sin(2 * math.pi * days / 5.5 + phi[0] + 0.5)
        + 1.1 * np.cos(2 * math.pi * days / 13.0 + phi[1])
    )
    w_hum = (
        5.0 * np.sin(2 * math.pi * days / 5.5 + phi[0] + math.pi / 4)
        + 2.0 * np.cos(2 * math.pi * days / 11.0 + phi[1])
    )
    return w_temp, w_press, w_hum, clock_h, days


def gen_station(idx, sid, name, lat, lon, regional, start, rng_station):
    """Generate one station's series (deterministic given idx + seed)."""
    n = int(DAYS * 24 * 60 / INTERVAL_MIN)
    p = _station_params(idx, sid, lat, lon)
    w_temp, w_press, w_hum, clock_h, days = regional
    hours = clock_h % 24.0
    load = p["regional_load"]

    # temperature: diurnal + distance-loaded regional weather + local noise
    diurnal = p["temp_amplitude"] * np.sin((hours - p["temp_phase_h"]) * 2 * math.pi / 24.0)
    temperature = (
        p["base_temp"]
        + diurnal
        + load * w_temp
        + rng_station.normal(0, p["temp_noise"], n)
    )

    # pressure: synoptic + semi-diurnal tide (anticorrelated with temperature cycle)
    p_syn = load * w_press + p["pressure_synodic"] * 0.9 * np.sin(hours * 4 * math.pi / 24.0 + 0.3)
    pressure = (
        p["base_pressure"]
        + p_syn
        - 0.10 * diurnal
        + rng_station.normal(0, p["pressure_noise"], n)
    )

    # humidity: inverse diurnal + regional + local noise, clipped to domain
    humidity = (
        p["base_humidity"]
        - 18.0 * diurnal / p["temp_amplitude"]
        + load * w_hum
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


def generate_frame(seed=REALISM_SEED, stations=None):
    """Build the full 24-station v2 frame (deterministic)."""
    stations = stations or STATION_META
    n = int(DAYS * 24 * 60 / INTERVAL_MIN)
    start = pd.Timestamp("2026-08-14 00:00:00")
    regional = _regional_weather(n, INTERVAL_MIN, seed)
    frames = []
    for idx, (sid, name, lat, lon) in enumerate(stations):
        rng_station = np.random.default_rng(seed + 1000 * (idx + 1))
        frames.append(gen_station(idx, sid, name, lat, lon, regional, start, rng_station))
    return pd.concat(frames, ignore_index=True)


def dataset_sha256(frame):
    """SHA-256 over a deterministic digest of the frame (stable across runs)."""
    digest = hashlib.sha256()
    digest.update(frame.sort_values(["station_id", "timestamp"]).to_csv(index=False).encode("utf-8"))
    return digest.hexdigest()


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_frame(seed=REALISM_SEED)
    df.to_csv(OUT, index=False)

    meta = {
        "dataset_version": "sample_weather_v2",
        "synthetic": True,
        "generator": "backend/data_generator.py",
        "seed": REALISM_SEED,
        "stations": [s[0] for s in STATION_META],
        "n_stations": len(STATION_META),
        "n_observations": int(len(df)),
        "records_per_station": int(len(df) / len(STATION_META)),
        "interval_minutes": INTERVAL_MIN,
        "start": "2026-08-14 00:00:00",
        "days": DAYS,
        "variables": ["temperature", "pressure", "humidity"],
        "sha256": dataset_sha256(df),
        "note": "Synthetic, station-distinct weather telemetry (lat/elevation/diurnal/"
                "spatial-correlation). Not real observed weather.",
    }
    with open(OUT_META, "w", encoding="utf-8") as f:
        import json
        json.dump(meta, f, indent=2)

    means = df.groupby("station_id")["temperature"].mean().round(2)
    spread = round(float(means.max() - means.min()), 2)
    print(f"Wrote {len(df)} readings to {OUT}")
    print(f"sha256={meta['sha256']}")
    print("Station mean temperature spread (max-min) [C]:", spread)
    print("Dispatch a 24-station distinctness check if spread > 0.5C.")


if __name__ == "__main__":
    main()