"""Unified ML pipeline: trains detectors, exposes inference, tracks state.

This is the single source of truth for models used by the API, so every AI
value shown in the UI is traceable to real computation here.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import MODEL_DIR
from ..detector import IsolationForestDetector
from ..features import add_temporal_features, add_multivariate_features, feature_columns
from .temporal import TemporalDetector
from .multivariate import MultivariateDetector, MULTIVARIATE_FEATURES
from .seasonal import SeasonalDetector
from ..preprocessing import validate, impute

BASELINE_COLS = ["temperature", "pressure", "humidity"]
TEMPORAL_COLS = [
    "hour",
    "temperature",
    "temperature_change",
    "temperature_rolling_mean",
    "temperature_rolling_std",
    "temperature_roc",
    "pressure",
    "pressure_change",
    "humidity",
    "humidity_change",
]

# Production decision threshold: a SYSTEM/rule-level constant applied to the
# fused model score. It is not part of the model itself and is recorded in
# model metadata.
ANOMALY_THRESHOLD = 0.72

# Calibration is disabled until a validated calibration split proves Brier /
# reliability improvement (PHASE 3, section P). Until then we never label the
# sigmoid score as a probability.
CALIBRATION_METHOD = "none"
CALIBRATION_STATUS = "uncalibrated"


class ModelManager:
    def __init__(self, station_id="AWS-000"):
        self.station_id = station_id
        self.baseline = None
        self.temporal = None
        self.multivariate = None
        self.seasonal = None
        self.full_cols = []
        self.trained = False
        self.df_train_full = None
        # ---- v3 mechanism-specific channels (additive; empty = v2 behavior) ----
        self.extra_channels = {}  # {"drift": DriftDetector, "flatline": ..., "missingness": ...}
        # weighted average; v2 = {"ml": 1.0, ...} -> identical to the old fusion
        self.fusion_weights = {"ml": 1.0, "drift": 0.0, "flat": 0.0, "miss": 0.0}
        # per-model decision threshold (defaults to the system constant; v3 tunes it)
        self.threshold = ANOMALY_THRESHOLD

    # ---- training ----
    def train(self, df: pd.DataFrame):
        """Train all production detectors on chronological (no shuffle) data.

        Feature engineering is applied per station so that rolling/diff windows
        never bleed across stations when MULTIPLE stations are trained together.
        When only a single station is present, behavior is unchanged.
        """
        df = validate(df)
        df = impute(df)
        df = df.sort_values("timestamp")

        # Build temporal + multivariate features PER STATION to avoid
        # cross-station bleed in rolling/diff windows.
        if "station_id" in df.columns and df["station_id"].nunique() > 1:
            parts = []
            for _, g in df.groupby("station_id", sort=False):
                g = g.sort_values("timestamp")
                g = add_multivariate_features(g)
                g = add_temporal_features(g)
                parts.append(g)
            df = pd.concat(parts, ignore_index=True)
            df = df.sort_values("timestamp").reset_index(drop=True)
        else:
            df = add_multivariate_features(df)
            df = add_temporal_features(df)
        self.df_train_full = df

        # baseline IsolationForest on raw vars
        self.baseline = IsolationForestDetector()
        self.baseline.fit(df, BASELINE_COLS)

        # temporal detector
        self.temporal = TemporalDetector()
        self.temporal.fit(df, TEMPORAL_COLS)

        # multivariate detector on available multivariate cols
        avail = [c for c in MULTIVARIATE_FEATURES if c in df.columns]
        if len(avail) >= 2:
            self.multivariate = MultivariateDetector()
            self.multivariate.fit(df, avail)

        # seasonal (diurnal) detector: raw vars + hour-of-day cyclical features
        self.seasonal = SeasonalDetector()
        self.seasonal.fit(df)

        self.full_cols = self._resolve_full_cols(df)
        # store background for SHAP-style explanation
        bg = df[self.full_cols].fillna(df[self.full_cols].median()).values
        self.baseline.background = bg[:50]
        self.baseline.anchor = getattr(self.baseline, "anchor", 0.0)
        dec_train = self.baseline.decision_scores(df)[0]
        self.baseline.train_decisions = np.asarray(dec_train)
        self.trained = True
        return self

    def _resolve_full_cols(self, df):
        return [c for c in df.columns if c not in ("timestamp", "station_id", "day", "month")
                and df[c].dtype in (np.float64, np.float32, np.int64, int, float)]

    def save(self, name="skyguard_model.pkl"):
        path = MODEL_DIR / name
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "baseline": self.baseline,
                    "temporal": self.temporal,
                    "multivariate": self.multivariate,
                    "seasonal": self.seasonal,
                    "full_cols": self.full_cols,
                    "station_id": self.station_id,
                    "threshold": self.threshold,
                    "fusion_weights": self.fusion_weights,
                    "extra_channels": self.extra_channels,
                },
                f,
            )
        return str(path)

    # ---- inference ----
    def prepare(self, df):
        """Shared preprocessing + feature engineering for inference."""
        df = df.copy()
        df_proc = validate(df)
        df_proc = impute(df_proc)
        df_proc = add_multivariate_features(df_proc)
        df_proc = add_temporal_features(df_proc)
        return df_proc

    def _channel_scores(self, df_proc):
        """Per-channel anomaly scores for the LAST row of a prepared frame.

        Returns (base_score, temp_score, mscore, seas_score, raw_decision)
        where raw_decision is the baseline detector's raw sklearn
        decision_function value for the last row (higher = more normal).
        """
        base_scores, _, raw_decs = self.baseline.decision_scores(df_proc)
        base_score = float(base_scores[-1]) if len(base_scores) else 0.0
        raw_decision = float(raw_decs[-1]) if len(raw_decs) else None

        temp_score = float(self.temporal.score(df_proc, TEMPORAL_COLS)[-1]) if self.temporal else 0.0

        mscore = 0.0
        if self.multivariate:
            avail = [c for c in MULTIVARIATE_FEATURES if c in df_proc.columns]
            if avail:
                mscore = float(self.multivariate.score(df_proc, avail)[-1])

        seas_score = 0.0
        if self.seasonal:
            seas_score = float(self.seasonal.score(df_proc)[-1])

        return float(base_score), float(temp_score), float(mscore), float(seas_score), raw_decision

    def fused_last(self, df_proc):
        """Fused score + component scores for the LAST row (no floors, no flags).

        Returns None when the model is not trained. This is the ONLY place the
        fused score is computed, so the API can always trace `anomaly_score`
        back to genuine trained-model output.
        """
        base_score, temp_score, mscore, seas_score, raw_decision = self._channel_scores(df_proc)
        combined = (
            0.45 * base_score
            + 0.20 * temp_score
            + 0.15 * mscore
            + 0.20 * seas_score
        )
        return {
            "score": float(combined),
            "base_score": float(base_score),
            "temporal_score": float(temp_score),
            "multivariate_score": float(mscore),
            "seasonal_score": float(seas_score),
            "raw_decision": raw_decision,
        }

    def analyze_row(self, df):
        """Analyze a single-row (or final row) DataFrame.

        Returns a dict with anomaly decision, score, component scores, rule
        flags, and calibration status.

        anomaly_score is PURELY the fused trained-model output. Missing/frozen
        conditions are reported as rule_flags and NEVER modify the score.
        """
        missing_var = None
        for v in ("temperature", "pressure", "humidity"):
            if v in df.columns and pd.isna(df[v].iloc[-1]):
                missing_var = v
                break

        frozen_var = None
        for v in ("temperature", "pressure", "humidity"):
            if v in df.columns and len(df) >= 3:
                rec_v = df[v].tail(3).dropna().values
                if len(rec_v) >= 3 and np.ptp(rec_v) < 1e-4:
                    frozen_var = v
                    break

        df_proc = self.prepare(df)
        fused = self.fused_last(df_proc)

        score = fused["score"]
        component_scores = {
            "baseline": fused["base_score"],
            "temporal": fused["temporal_score"],
            "multivariate": fused["multivariate_score"],
            "seasonal": fused["seasonal_score"],
        }

        # v3 mechanism-specific evidence (drifting / frozen / missing); fused
        # score is a weighted average of ML evidence + these channels using the
        # MODEL's configured weights (v2: ML-only, identical to before)
        if self.extra_channels:
            drift_v = float(self.extra_channels["drift"].score(df)[-1]) if len(df) else 0.0
            flat_v = float(self.extra_channels["flatline"].score(df)[-1]) if len(df) else 0.0
            miss_v = float(self.extra_channels["missingness"].score_raw(df)[-1]) if len(df) else 0.0
            roc_ch = self.extra_channels.get("roc")
            roc_v = float(roc_ch.score(df)[-1]) if (len(df) and roc_ch is not None) else 0.0
            w = self.fusion_weights
            w_roc = float(w.get("roc", 0.0))
            tot = w["ml"] + w["drift"] + w["flat"] + w["miss"] + w_roc
            score = (w["ml"] * score + w["drift"] * drift_v + w["flat"] * flat_v
                     + w["miss"] * miss_v + w_roc * roc_v) / max(tot, 1e-9)
            component_scores["drift"] = drift_v
            component_scores["flatline"] = flat_v
            component_scores["missingness"] = miss_v
            if roc_ch is not None:
                component_scores["roc"] = roc_v
            evidence = {
                "ml": float(fused["score"]),
                "drift": drift_v,
                "flatline": flat_v,
                "missingness": miss_v,
            }
            if roc_ch is not None:
                evidence["roc"] = roc_v
        else:
            evidence = {"ml": float(fused["score"])}

        threshold = getattr(self, "threshold", ANOMALY_THRESHOLD)
        is_anomaly = bool(score > threshold)

        # decide which variable is most anomalous using standardized z-scores
        # (statistical heuristic, not part of the trained model)
        variable_zscores = {}
        for v in ("temperature", "pressure", "humidity"):
            if v in df_proc.columns:
                val = float(df_proc[v].iloc[-1]) if not np.isnan(df_proc[v].iloc[-1]) else 0.0
                mean_v = float(df_proc[f"{v}_rolling_mean"].iloc[-1]) if f"{v}_rolling_mean" in df_proc.columns else float(df_proc[v].mean())
                std_v = float(df_proc[f"{v}_rolling_std"].iloc[-1]) if f"{v}_rolling_std" in df_proc.columns else (float(df_proc[v].std()) + 1e-9)
                z = abs(val - mean_v) / (std_v + 1e-9)
                variable_zscores[v] = z

        primary_feature = (
            missing_var
            or frozen_var
            or (max(variable_zscores, key=variable_zscores.get) if variable_zscores else "temperature")
        )

        return {
            "score": score,
            "is_anomaly": is_anomaly,
            "base_score": fused["base_score"],
            "temporal_score": fused["temporal_score"],
            "multivariate_score": fused["multivariate_score"],
            "seasonal_score": fused["seasonal_score"],
            "component_scores": component_scores,
            "evidence": evidence,
            "raw_decision": fused.get("raw_decision"),  # baseline raw decision (higher = more normal)
            "rule_flags": {
                "missing_imputed": missing_var is not None,
                "frozen_plateau": frozen_var is not None,
            },
            "feature": primary_feature,
            "anomaly_probability": None,
            "calibration_status": CALIBRATION_STATUS,
            "calibration_method": CALIBRATION_METHOD,
        }

    def score_series(self, df):
        """Return anomaly labels for every row (uses this model's threshold)."""
        return (self.score_series_scores(df) > getattr(self, "threshold", ANOMALY_THRESHOLD)).astype(int)

    def ml_fused_scores(self, df):
        """Vectorized ML-only fused score (the v2 fusion formula).

        PURE ML output: no missing/frozen floor overrides. `df` is the raw/
        validated frame; ML features are prepared internally (imputation is ML
        internal and never erases the RAW missingness evidence used by the
        missingness channel, which is scored separately on the raw frame).
        """
        df = df.reset_index(drop=True)
        n = len(df)
        df_proc = self.prepare(df)
        base = self.baseline.decision_scores(df_proc)[0]
        temp = self.temporal.score(df_proc, TEMPORAL_COLS) if self.temporal else np.zeros(n)
        mscore = np.zeros(n)
        if self.multivariate:
            avail = [c for c in MULTIVARIATE_FEATURES if c in df_proc.columns]
            if avail:
                mscore = self.multivariate.score(df_proc, avail)
        seas = self.seasonal.score(df_proc) if self.seasonal else np.zeros(n)
        return np.asarray(0.45 * base + 0.20 * temp + 0.15 * mscore + 0.20 * seas, dtype=float)

    def score_series_scores(self, df, raw=None):
        """Return CONTINUOUS fused anomaly scores for every row (for ROC-AUC).

        v2 behavior (no extra channels) is byte-identical to the old vectorized
        fusion (0.45/0.20/0.15/0.20). With v3 channel weights it becomes a
        weighted average of the ML channel and the dedicated drift / flatline /
        missingness evidence, so the recall of FROZEN and MISSING mechanisms is
        driven by evidence those mechanisms were designed for. `raw` may be the
        pre-impute frame for the missingness channel; defaults to `df` itself.
        """
        ml = self.ml_fused_scores(df)
        if not self.extra_channels:
            return ml
        raw_use = df if raw is None else raw
        drift = np.asarray(self.extra_channels["drift"].score(df), dtype=float)
        flat = np.asarray(self.extra_channels["flatline"].score(df), dtype=float)
        miss = np.asarray(self.extra_channels["missingness"].score_raw(raw_use), dtype=float)
        w = self.fusion_weights
        term = w["ml"] * ml + w["drift"] * drift + w["flat"] * flat + w["miss"] * miss
        tot = w["ml"] + w["drift"] + w["flat"] + w["miss"]
        roc_ch = self.extra_channels.get("roc")
        if roc_ch is not None:
            roc = np.asarray(roc_ch.score(df), dtype=float)
            term = term + w.get("roc", 0.0) * roc
            tot = tot + w.get("roc", 0.0)
        return term / max(tot, 1e-9)
