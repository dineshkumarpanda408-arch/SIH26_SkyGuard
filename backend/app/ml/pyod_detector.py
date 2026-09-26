"""
detector.py
-----------
Multivariate Weather Anomaly Detector using PyOD Isolation Forest (IForest),
coupled with 9-dimensional Temporal Trend Feature Engineering, Rule-Based
Root-Cause Diagnostics, and Sensor Health Index Tracking.

Feature Engineering:
- 3 Raw Telemetry Features: temperature, pressure, humidity
- 3 Rolling Mean Features (window=20): temperature_rolling_mean_20, pressure_rolling_mean_20, humidity_rolling_mean_20
- 3 Rolling Deviation Features: temperature_deviation_20, pressure_deviation_20, humidity_deviation_20
-> 9 Features Total trained jointly on PyOD IForest to boost slow drift detection.
"""

import numpy as np
import pandas as pd
from pyod.models.iforest import IForest
from typing import Tuple, Dict, Any, Optional, List


class WeatherAnomalyDetector:
    """
    Multivariate Isolation Forest detector with trend feature engineering for weather stations.
    """

    def __init__(
        self,
        base_features: Optional[List[str]] = None,
        rolling_window: int = 20,
        contamination: float = 0.055,
        n_estimators: int = 150,
        random_state: int = 42
    ):
        """
        Initialize the detector with 9-dimensional trend-augmented feature space.

        Parameters:
        -----------
        base_features : list, optional
            List of raw meteorological sensor telemetry columns (default: ['temperature', 'pressure', 'humidity']).
        rolling_window : int
            Rolling window size for trend feature extraction (default: 20 readings / 100 mins).
        contamination : float
            Expected proportion of outliers in the dataset.
        n_estimators : int
            Number of isolation trees in the ensemble.
        random_state : int
            Random seed for reproducibility.
        """
        self.base_features = base_features if base_features is not None else ["temperature", "pressure", "humidity"]
        self.rolling_window = rolling_window
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state

        # Construct 9-feature schema
        self.trend_features = []
        for feat in self.base_features:
            self.trend_features.append(f"{feat}_rolling_mean_{self.rolling_window}")
            self.trend_features.append(f"{feat}_deviation_{self.rolling_window}")

        self.features = self.base_features + self.trend_features  # 9 features total

        self.clf = IForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state
        )
        self.is_fitted = False

    def extract_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract rolling mean and deviation features across the base telemetry channels.
        Uses expanding window (min_periods=1) for initial points to avoid NaNs.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing all 9 features.
        """
        df_clean = df[self.base_features].copy().ffill().bfill().fillna(0.0)
        df_feat = pd.DataFrame(index=df.index)

        # 1. Base raw features
        for feat in self.base_features:
            df_feat[feat] = df_clean[feat]

        # 2. Rolling mean and deviation features (min_periods=1 ensures no NaNs at start)
        for feat in self.base_features:
            r_mean = df_clean[feat].rolling(window=self.rolling_window, min_periods=1).mean()
            dev = df_clean[feat] - r_mean
            df_feat[f"{feat}_rolling_mean_{self.rolling_window}"] = np.round(r_mean, 3)
            df_feat[f"{feat}_deviation_{self.rolling_window}"] = np.round(dev, 3)

        return df_feat[self.features]

    def _preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Preprocess features and identify missing data masks.

        Returns:
        --------
        X_mat : np.ndarray
            (N, 9) numeric matrix for PyOD IForest inference.
        gap_mask : np.ndarray
            Boolean array indicating rows with missing telemetry values.
        df_feat : pd.DataFrame
            The 9-feature DataFrame.
        """
        gap_mask = df[self.base_features].isna().any(axis=1).to_numpy()
        df_feat = self.extract_trend_features(df)
        return df_feat.to_numpy(), gap_mask, df_feat

    def fit(self, df_train: pd.DataFrame) -> "WeatherAnomalyDetector":
        """
        Fit PyOD Isolation Forest on 9-dimensional trend-augmented features.
        """
        X_train, _, _ = self._preprocess(df_train)
        self.clf.fit(X_train)
        self.is_fitted = True
        return self

    def predict(self, df_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run PyOD multivariate inference on 9 features with sensor physical validation.

        Returns:
        --------
        y_pred : np.ndarray
            Binary anomaly predictions (1 = anomaly, 0 = normal).
        scores : np.ndarray
            Continuous anomaly decision scores.
        confidence : np.ndarray
            Model prediction confidence scores (0.0 to 1.0).
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predicting. Call detector.fit() first.")

        X_test, gap_mask, df_feat = self._preprocess(df_test)

        # 1. PyOD Standard IForest Inference on all 9 features
        scores = self.clf.decision_function(X_test)
        y_pred, confidence = self.clf.predict(X_test, return_confidence=True)

        n_rows = len(df_test)
        max_score = float(np.max(scores)) if len(scores) > 0 else 1.0

        # 2. Telemetry Communication Gap Detection (Packet dropouts / NaNs)
        if gap_mask.any():
            scores[gap_mask] = np.maximum(scores[gap_mask], max_score * 1.15)
            y_pred[gap_mask] = 1
            confidence[gap_mask] = 1.0

        # 3. Frozen Sensor Detection (Flatline consecutive readings with zero variation >= 4 repeats)
        for feat in self.base_features:
            series = df_test[feat].to_numpy()
            run = 1
            for i in range(1, n_rows):
                if pd.notna(series[i]) and pd.notna(series[i-1]) and series[i] == series[i-1]:
                    run += 1
                    if run >= 4:
                        start_k = max(0, i - run + 1)
                        y_pred[start_k:i+1] = 1
                        confidence[start_k:i+1] = np.maximum(confidence[start_k:i+1], 0.95)
                        scores[start_k:i+1] = np.maximum(scores[start_k:i+1], self.clf.threshold_ * 1.1)
                else:
                    run = 1

        # 4. Sensor Drift / Baseline Divergence Detection (Calibrated above normal diurnal variation)
        for feat in self.base_features:
            dev_col = f"{feat}_deviation_{self.rolling_window}"
            dev_series = df_feat[dev_col].abs().to_numpy()
            thresh = 5.5 if feat == "temperature" else (10.0 if feat == "pressure" else 16.0)
            drift_idx = np.where(dev_series > thresh)[0]
            if len(drift_idx) > 0:
                y_pred[drift_idx] = 1
                confidence[drift_idx] = np.maximum(confidence[drift_idx], 0.88)
                scores[drift_idx] = np.maximum(scores[drift_idx], self.clf.threshold_ * 1.05)

        return y_pred, scores, confidence

    def classify_root_cause(
        self,
        df: pd.DataFrame,
        anom_indices: np.ndarray,
        window: int = 6
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Rule-based diagnostic classification to identify the likely failure mode and affected sensor channel:
        - Communication gap: Missing / NaN readings -> feature: "all"
        - Frozen: Sensor reading flatlined with zero variance -> affected feat
        - Spike: Isolated, transient single-step jump that reverts immediately -> affected feat
        - Drift: Sustained multi-step monotonic divergence from local baseline -> affected feat
        - Multivariate Outlier: Cross-sensor correlation violation -> top deviating feat
        """
        diagnoses = []
        affected_features = []
        n_rows = len(df)

        for idx in range(n_rows):
            if anom_indices[idx] == 0:
                diagnoses.append("Normal")
                affected_features.append("none")
                continue

            # 1. Communication Gap check (NaN check)
            row_vals = df.loc[idx, self.base_features]
            if row_vals.isna().any():
                diagnoses.append("Communication gap")
                affected_features.append("all")
                continue

            # 2. Frozen Sensor Check (Repeated exact values >= 4 readings)
            is_frozen = False
            frozen_feat = None
            for feat in self.base_features:
                curr_v = df.loc[idx, feat]
                prev_v = df.loc[idx - 1, feat] if idx > 0 else None
                next_v = df.loc[idx + 1, feat] if idx < n_rows - 1 else None

                if (prev_v is not None and curr_v == prev_v) or (next_v is not None and curr_v == next_v):
                    start_w = max(0, idx - 4)
                    end_w = min(n_rows, idx + 5)
                    sub = df.iloc[start_w:end_w][feat].dropna().to_numpy()
                    if len(sub) >= 4 and (np.abs(np.diff(sub)) == 0).sum() >= 3:
                        is_frozen = True
                        frozen_feat = feat
                        break

            if is_frozen:
                diagnoses.append("Frozen")
                affected_features.append(frozen_feat or "temperature")
                continue

            # 3. Spike Check (Single-point extreme deviation with immediate reversion)
            is_spike = False
            spike_feat = None
            if 0 < idx < n_rows - 1:
                for feat in self.base_features:
                    prev_val = df.loc[idx - 1, feat]
                    curr_val = df.loc[idx, feat]
                    next_val = df.loc[idx + 1, feat]

                    if pd.notna(prev_val) and pd.notna(curr_val) and pd.notna(next_val):
                        d_prev = abs(curr_val - prev_val)
                        d_next = abs(curr_val - next_val)
                        d_baseline = abs(next_val - prev_val)

                        threshold = 7.0 if feat == "temperature" else (16.0 if feat == "pressure" else 18.0)

                        if d_prev > threshold and d_next > threshold and d_baseline < (threshold * 0.5):
                            is_spike = True
                            spike_feat = feat
                            break
            if is_spike:
                diagnoses.append("Spike")
                affected_features.append(spike_feat or "temperature")
                continue

            # 4. Drift Check (Multi-point trend or sustained baseline deviation)
            is_drift = False
            drift_feat = None
            for feat in self.base_features:
                dev_col = f"{feat}_deviation_{self.rolling_window}"
                if dev_col in df.columns:
                    curr_dev = abs(df.loc[idx, dev_col])
                else:
                    s = df[feat].ffill().bfill()
                    med = s.iloc[max(0, idx - 20):min(n_rows, idx + 21)].median()
                    curr_dev = abs(s.iloc[idx] - med)

                thresh = 5.5 if feat == "temperature" else (10.0 if feat == "pressure" else 16.0)
                if curr_dev > thresh:
                    is_drift = True
                    drift_feat = feat
                    break

            if is_drift:
                diagnoses.append("Drift")
                affected_features.append(drift_feat or "temperature")
                continue

            # 5. Default: Multivariate Correlation Outlier
            diagnoses.append("Multivariate Outlier")
            # Determine top deviating feature
            devs = {}
            for feat in self.base_features:
                dev_col = f"{feat}_deviation_{self.rolling_window}"
                if dev_col in df.columns:
                    devs[feat] = abs(df.loc[idx, dev_col])
            top_f = max(devs, key=devs.get) if devs else "temperature"
            affected_features.append(top_f)

        return pd.Series(diagnoses, index=df.index), pd.Series(affected_features, index=df.index)

    def calculate_sensor_health(
        self,
        pred_is_anomaly: np.ndarray,
        window_size: int = 50,
        decay_factor: float = 1.6
    ) -> pd.Series:
        """
        Calculate rolling Sensor Health Index (0-100).
        - 100: Pristine telemetry with 0 anomalies in the recent window.
        - Penalizes with higher anomaly rates.
        """
        s = pd.Series(pred_is_anomaly, dtype=float)
        rolling_anomaly_rate = s.rolling(window=window_size, min_periods=1).mean()
        health_score = 100.0 * (1.0 - np.clip(rolling_anomaly_rate * decay_factor, 0.0, 1.0))
        return np.round(health_score, 1)

    def detect_and_diagnose(
        self,
        df: pd.DataFrame,
        health_window: int = 50
    ) -> pd.DataFrame:
        """
        Execute full pipeline: Extract 9 Trend Features -> Fit/Predict -> Classify Root Cause -> Track Sensor Health.

        Returns:
        --------
        pd.DataFrame
            Telemetry DataFrame augmented with trend features, predictions, scores, and diagnostics.
        """
        if not self.is_fitted:
            self.fit(df)

        y_pred, scores, confidence = self.predict(df)
        df_feat = self.extract_trend_features(df)

        df_result = df.copy()
        for col in self.trend_features:
            df_result[col] = df_feat[col]

        pred_types, pred_feats = self.classify_root_cause(df_result, y_pred)
        df_result["pred_is_anomaly"] = y_pred
        df_result["anomaly_score"] = np.round(scores, 4)
        df_result["confidence"] = np.round(confidence, 4)
        df_result["pred_anomaly_type"] = pred_types
        df_result["pred_anomaly_feature"] = pred_feats
        df_result["sensor_health"] = self.calculate_sensor_health(y_pred, window_size=health_window)

        return df_result
