"""Multivariate anomaly detection.

Uses the engineered multivariate features (dew point, vapor pressure, mixing
ratio, co-movement deltas) with an isolation forest. This captures joint
inconsistencies between temperature/pressure/humidity that independent
per-variable thresholds cannot see.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from ..detector import _score_from_decision


class MultivariateDetector:
    def __init__(self, contamination: float = 0.04):
        self.contamination = contamination
        self.model = None
        self.anchor = None
        self.scaler = None
        self.train_decisions = None

    def fit(self, df, feature_cols):
        X = df[feature_cols].fillna(df[feature_cols].median())
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        self.model = IsolationForest(
            n_estimators=100, contamination=self.contamination, random_state=21
        )
        self.model.fit(Xs)
        raw = self.model.decision_function(Xs)
        self.anchor = float(np.median(raw))
        self.train_decisions = np.asarray(raw)
        return self

    def score(self, df, feature_cols):
        X = df[feature_cols].fillna(df[feature_cols].median())
        Xs = self.scaler.transform(X)
        raw = self.model.decision_function(Xs)
        scale = float(np.std(self.train_decisions)) + 1e-9 if self.train_decisions is not None else 1.0
        return _score_from_decision(raw, anchor=self.anchor, scale=scale)


MULTIVARIATE_FEATURES = [
    "dew_point",
    "vapor_pressure_sat",
    "mixing_ratio",
    "delta_t",
    "delta_p",
    "humidity_change",
]
