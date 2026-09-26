"""Preprocessing: cleaning, validation, missing-value handling."""

import numpy as np
import pandas as pd

# Physically plausible ranges for weather observations (used for validation only
# to catch grossly malformed data, NOT as a hard anomaly threshold). These are
# deliberately WIDE so that plausible sensor-malfunction readings (the anomalies
# we want to detect) are preserved; hard thresholding is the ML model's job, not
# the validation layer's.
BOUNDS = {
    "temperature": (-100.0, 200.0),
    "pressure": (500.0, 1500.0),
    "humidity": (-10.0, 120.0),
}


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean a weather DataFrame.

    Returns a cleaned DataFrame and is idempotent. Invalid numeric values are
    turned into NaN (never silently kept), and duplicate/maintained timestamps
    handled explicitly.
    """
    df = df.copy()
    for col, (lo, hi) in BOUNDS.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] < lo, col] = np.nan
            df.loc[df[col] > hi, col] = np.nan
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")
        # Deduplicate per station: the 24 stations share a common 15-minute
        # timestamp grid, so deduping on timestamp alone would collapse all
        # stations to one and silently drop 23 of them.
        if "station_id" in df.columns:
            df = df.drop_duplicates(subset=["station_id", "timestamp"], keep="last")
        else:
            df = df.drop_duplicates(subset=["timestamp"], keep="last")
    return df


def impute(df: pd.DataFrame, windows: int = 3) -> pd.DataFrame:
    """Forward/backward fill within a small window; leave gaps as NaN."""
    df = df.copy()
    for col in ("temperature", "pressure", "humidity"):
        if col not in df.columns:
            continue
        df[col] = df[col].interpolate(method="linear", limit=windows, limit_direction="both")
    return df


def missing_flags(df: pd.DataFrame) -> dict:
    """Return per-variable missing-data statistics."""
    stats = {}
    for col in ("temperature", "pressure", "humidity"):
        if col in df.columns:
            stats[col] = {
                "missing": int(df[col].isna().sum()),
                "percent": float(df[col].isna().mean()),
            }
    return stats
