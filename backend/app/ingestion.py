"""Data ingestion layer: CSV, API, or simulator.

Normalizes raw weather observations into a pandas DataFrame with columns:
    timestamp, station_id, temperature, pressure, humidity
"""

import pandas as pd

from .config import DATA_FILE


def load_csv(csv_path=None) -> pd.DataFrame:
    """Load a weather observations CSV and normalize column names."""
    path = csv_path or DATA_FILE
    df = pd.read_csv(path)
    return normalize(df)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize columns to a canonical lowercase schema."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    mapping = {
        "station": "station_id",
        "stationid": "station_id",
        "time": "timestamp",
        "datetime": "timestamp",
        "date": "timestamp",
        "temp": "temperature",
        "t": "temperature",
        "press": "pressure",
        "p": "pressure",
        "rh": "humidity",
        "relativehumidity": "humidity",
    }
    df = df.rename(columns=mapping)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def generate_stream(df: pd.DataFrame, station_id: str):
    """Yield readings one at a time from a DataFrame (realtime streaming)."""
    for _, row in df.iterrows():
        yield {
            "station_id": station_id,
            "timestamp": row.get("timestamp"),
            "temperature": row.get("temperature"),
            "pressure": row.get("pressure"),
            "humidity": row.get("humidity"),
        }
