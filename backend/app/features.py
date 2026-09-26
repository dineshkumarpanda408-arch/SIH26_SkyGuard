"""Feature engineering: temporal and multivariate features.

All features are computed causally (rolling windows use only past/current data),
so there is no leakage and no use of future information during inference.
"""

import numpy as np
import pandas as pd

_VARS = ("temperature", "pressure", "humidity")


def add_temporal_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    if "hour" in df.columns:
        df = df.drop(columns=["hour"])
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month

    for v in _VARS:
        if v not in df.columns:
            continue
        s = df[v]
        df[f"{v}_change"] = s.diff()
        df[f"{v}_rolling_mean"] = s.rolling(window, min_periods=1).mean().shift(1)
        df[f"{v}_rolling_std"] = s.rolling(window, min_periods=1).std().shift(1)
        df[f"{v}_roc"] = s.diff() / (s.shift(1).abs() + 1e-9)
        # recent volatility relative to the previous window mean
        df[f"{v}_volatility"] = (s - df[f"{v}_rolling_mean"]).abs() / (
            df[f"{v}_rolling_mean"].abs() + 1e-9
        )
    return df


def add_multivariate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Relationships among temperature/pressure/humidity.

    These capture physically plausible joint structure that a per-variable
    detector would miss.
    """
    df = df.copy()
    has_t = "temperature" in df.columns
    has_p = "pressure" in df.columns
    has_h = "humidity" in df.columns

    # Saturation-vapor-pressure-ish expectation of humidity given temperature
    if has_t:
        df["t_celsius"] = df["temperature"]
        df["vapor_pressure_sat"] = 6.112 * np.exp(17.67 * df["t_celsius"] / (df["t_celsius"] + 243.5))
    # Dew-point proxy combining T and RH
    if has_t and has_h:
        a = 17.27
        b = 237.7
        with np.errstate(divide="ignore", invalid="ignore"):
            gamma = (a * df["temperature"]) / (b + df["temperature"]) + np.log(
                np.clip(df["humidity"], 1e-6, None) / 100.0
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            df["dew_point"] = (b * gamma) / (a - gamma)
    # Humidity-temperature residual after considering saturation
    if has_t and has_h:
        denom = df["vapor_pressure_sat"].replace(0, np.nan)
        df["humidity_vs_vapor"] = (df["humidity"] / 100.0) - (df["pressure"] if has_p else np.nan)
        df["mixing_ratio"] = 622.0 * df["humidity"] / (100.0 * denom + 1e-9)

    # Positive pressure normally => stable-ish weather; big co-movement is unusual
    df["delta_t"] = df["temperature"].diff() if has_t else np.nan
    df["delta_p"] = df["pressure"].diff() if has_p else np.nan
    return df


def build_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Full temporal + multivariate feature matrix (used at train & inference)."""
    df = add_temporal_features(df, window=window)
    df = add_multivariate_features(df)
    return df


FEATURE_GROUPS = {
    "baseline": ["temperature", "pressure", "humidity"],
    "temporal": [
        "temperature",
        "pressure",
        "humidity",
        "temperature_change",
        "temperature_rolling_mean",
        "temperature_rolling_std",
        "pressure_change",
        "humidity_change",
        "hour",
    ],
    "full": None,  # resolved at runtime
}


def feature_columns(kind: str, df: pd.DataFrame) -> list:
    if kind == "full":
        exclude = {
            "timestamp",
            "station_id",
            "temperature",
        }
        # keep raw values + all engineered features, drop n/a helpers
        cols = [c for c in df.columns if c not in exclude and df[c].dtype in (np.float64, np.float32, np.int64, int, float)]
        return cols
    return list(FEATURE_GROUPS[kind])
