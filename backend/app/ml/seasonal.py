"""Seasonal / diurnal anomaly detection.

Weather follows a strong daily (diurnal) cycle: temperature peaks mid-afternoon
and bottoms out around dawn, humidity is roughly the inverse, and pressure has
its own semi-diurnal rhythm. A reading that is "normal" at 3 AM can be deeply
anomalous at 3 PM. This detector feeds an Isolation Forest with cyclical
hour-of-day features (sin/cos of the hour) alongside the raw variables so the
model learns the expected value of each variable *for the time of day*.

This mirrors the "anomaly detection over a timeline" approach used in the
referenced weather-isolation-forest projects: instead of a purely point-wise
compare, the model accounts for where in the diurnal cycle a reading falls.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from ..detector import _score_from_decision


def _seasonal_cols():
    return ["temperature", "pressure", "humidity", "hour_sin", "hour_cos"]


def _add_cyclical(df: pd.DataFrame, cols):
    """Add hour_sin/hour_cos cyclical features from an 'hour' column."""
    df = df.copy()
    if "hour" not in df.columns:
        # infer hour from timestamp if available
        if "timestamp" in df.columns:
            df["hour"] = df["timestamp"].dt.hour
        else:
            df["hour"] = 12
    hour = df["hour"].astype(float) % 24.0
    df["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df


class SeasonalDetector:
    """Isolation Forest on [raw vars + hour-of-day cyclical features]."""

    def __init__(self, contamination: float = 0.04):
        self.contamination = contamination
        self.features = _seasonal_cols()
        self.model = None
        self.scaler = None
        self.anchor = None
        self.train_decisions = None

    def fit(self, df, feature_cols=None):
        cols = feature_cols or self.features
        df = _add_cyclical(df, cols)
        X = df[cols].fillna(df[cols].median())
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        self.model = IsolationForest(
            n_estimators=120, contamination=self.contamination, random_state=13
        )
        self.model.fit(Xs)
        raw = self.model.decision_function(Xs)
        self.anchor = float(np.median(raw))
        self.train_decisions = np.asarray(raw)
        self.features = cols
        return self

    def score(self, df, feature_cols=None):
        cols = feature_cols or self.features
        df = _add_cyclical(df, cols)
        X = df[cols].fillna(df[cols].median())
        Xs = self.scaler.transform(X)
        raw = self.model.decision_function(Xs)
        scale = float(np.std(self.train_decisions)) + 1e-9 if self.train_decisions is not None else 1.0
        return _score_from_decision(raw, anchor=self.anchor, scale=scale)
