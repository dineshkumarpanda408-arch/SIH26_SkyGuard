"""Baseline anomaly detector using Isolation Forest.

The raw IsolationForest.decision_function output is NOT a calibrated
probability. We map it to an anomaly score in [0,1] via a sigmoid of the
standardized decision function, and we explicitly label this an *anomaly
confidence score*, not a calibrated probability.

Features are standardized (z-scored) before training/scoring so that each
weather variable contributes comparably — this is essential for the raw
detector to be sensitive to e.g. a temperature spike and for SHAP attribution
to be meaningful.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .preprocessing import validate, impute

CONTAMINATION = 0.05
ANOMALY_CONFIDENCE_THRESHOLD = 0.75


def _score_from_decision(dec, anchor=None, scale=None):
    """Convert raw decision_function into a standardized [0,1] anomaly confidence score.

    decision_function: higher = more normal. We negate, standardize, then sigmoid
    so that a score near 1 = strong anomaly. anchor pins the halfway point at a
    reference (median training decision) and scale standardizes the spread.
    """
    dec = np.asarray(dec, dtype=float)
    anchor_val = float(np.median(dec)) if anchor is None else float(anchor)
    sc = scale if scale is not None else (np.std(dec) + 1e-9)
    centered = dec - anchor_val
    z = -centered / (sc + 1e-9)
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


class IsolationForestDetector:
    def __init__(self, contamination: float = CONTAMINATION):
        self.contamination = contamination
        self.model = None
        self.anchor = None
        self.features = None
        self.scaler = None
        self.train_decisions = None

    def fit(self, df, feature_cols):
        df = validate(df)
        df = impute(df)
        X = df[feature_cols].dropna()
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        self.model = IsolationForest(
            n_estimators=100, contamination=self.contamination, random_state=42
        )
        self.model.fit(Xs)
        train_dec = self.model.decision_function(Xs)
        self.anchor = float(np.median(train_dec))
        self.train_decisions = np.asarray(train_dec)
        self.features = feature_cols
        return self

    def decision_scores(self, df):
        df = validate(df)
        df = impute(df)
        X = df[self.features].fillna(df[self.features].median())
        Xs = self.scaler.transform(X)
        raw_dec = self.model.decision_function(Xs)
        scores = _score_from_decision(raw_dec, anchor=self.anchor, scale=self._dec_scale())
        labels = (scores >= ANOMALY_CONFIDENCE_THRESHOLD).astype(int)
        return scores, labels, raw_dec

    def raw_decision(self, df):
        """Raw sklearn decision_function values (higher = more normal).

        Exposed for honest reporting: the uncalibrated sigmoid score saturates at
        extreme deviations, so callers must also surface this raw margin and the
        physical deviation magnitude instead of manufacturing extra variation.
        """
        df = validate(df)
        df = impute(df)
        X = df[self.features].fillna(df[self.features].median())
        Xs = self.scaler.transform(X)
        return np.asarray(self.model.decision_function(Xs), dtype=float)

    def get_decision(self, X_rows):
        """Anomaly confidence score in [0,1] for raw (pre-scale) feature rows.

        Used by the SHAP attribution layer: it must re-score a row
        with one feature replaced by its typical value, so it accepts a numpy
        array of unpreprocessed feature values shaped (n, n_features).
        """
        X = np.asarray([list(map(float, r)) for r in X_rows], dtype=float)
        Xs = self.scaler.transform(X)
        raw = self.model.decision_function(Xs)
        return _score_from_decision(raw, anchor=self.anchor, scale=self._dec_scale())

    def _dec_scale(self):
        return float(np.std(self.train_decisions)) + 1e-9 if self.train_decisions is not None else 1.0
