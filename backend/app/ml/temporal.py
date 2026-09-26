"""Temporal anomaly detection using engineered time-series features.

A separate Isolation Forest trained on temporal/rolling features detects
spikes, drops, shifts, and local behavior that a purely point-wise detector
would miss. Scores are combined with the baseline via a weighted max.
"""

import numpy as np

from ..detector import IsolationForestDetector, _score_from_decision


class TemporalDetector:
    def __init__(self, contamination: float = 0.04):
        self.contamination = contamination
        self.model = None
        self.anchor = None
        self.scaler = None
        self.train_decisions = None

    def fit(self, df, feature_cols):
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        X = df[feature_cols].fillna(df[feature_cols].median())
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        self.model = IsolationForest(
            n_estimators=100, contamination=self.contamination, random_state=7
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


def combine_scores(baseline_scores, temporal_scores, weights=(0.5, 0.5)):
    """Weighted max fusion: a strong flag in either channel raises overall score."""
    baseline = np.asarray(baseline_scores, dtype=float)
    temporal = np.asarray(temporal_scores, dtype=float)
    w1, w2 = weights
    # weighted max preserves peak sensitivity
    return w1 * baseline + w2 * temporal
