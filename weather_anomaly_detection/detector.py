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

        # 3. Frozen Sensor Detection (Flatline consecutive readings with zero variation >= 2 repeats)
        for feat in self.base_features:
            s = df_test[feat]
            ok = s.notna()
            same_prev = ok & s.shift(1).notna() & (s == s.shift(1))
            flat_rows = (
                same_prev & s.shift(2).notna() & (s == s.shift(2))
            ) | (
                same_prev & s.shift(-1).notna() & (s == s.shift(-1))
            )
            flat_mask = flat_rows.to_numpy()
            if flat_mask.any():
                y_pred[flat_mask] = 1
                confidence[flat_mask] = np.maximum(confidence[flat_mask], 0.92)
                scores[flat_mask] = np.maximum(scores[flat_mask], self.clf.threshold_)

        # 4. Sensor Drift / Baseline Divergence Detection
        for feat in self.base_features:
            dev_col = f"{feat}_deviation_{self.rolling_window}"
            dev_series = df_feat[dev_col].abs().to_numpy()
            thresh = 2.4 if feat == "temperature" else (4.8 if feat == "pressure" else 8.0)
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
    ) -> pd.Series:
        """
        Rule-based diagnostic classification to identify the likely failure mode:
        - Communication gap: Missing / NaN readings
        - Frozen: Sensor reading flatlined with zero variance over consecutive steps
        - Spike: Isolated, transient single-step jump that reverts immediately
        - Drift: Sustained multi-step monotonic divergence from local baseline
        - Multivariate Outlier: Cross-sensor correlation violation

        Vectorized implementation of the original per-row loop (equivalent outcomes,
        dramatically faster on large telemetry windows).
        """
        n_rows = len(df)
        diagnoses = np.empty(n_rows, dtype=object)
        diagnoses[:] = "Normal"

        anom_mask = np.asarray(anom_indices, dtype=bool)
        if not anom_mask.any():
            return pd.Series(diagnoses, index=df.index)

        # 1. Communication Gap check (NaN check)
        gap = df[self.base_features].isna().any(axis=1).to_numpy()
        diagnoses = np.where(anom_mask & gap, "Communication gap", diagnoses)

        # 2. Frozen Sensor Check (row equals a consecutive neighbor)
        frozen = np.zeros(n_rows, dtype=bool)
        for feat in self.base_features:
            s = df[feat]
            ok = s.notna()
            eq_prev = ok & s.shift(1).notna() & (s == s.shift(1))
            eq_next = ok & s.shift(-1).notna() & (s == s.shift(-1))
            frozen |= (eq_prev | eq_next).to_numpy()
        pick = anom_mask & (diagnoses == "Normal") & frozen
        diagnoses = np.where(pick, "Frozen", diagnoses)

        # 3. Spike Check (Single-point extreme deviation with immediate reversion)
        spike = np.zeros(n_rows, dtype=bool)
        for feat, threshold in zip(self.base_features, (7.0, 16.0, 18.0)):
            fp = df[feat]
            prev = fp.shift(1)
            nxt = fp.shift(-1)
            v_ok = fp.notna() & prev.notna() & nxt.notna()
            d_prev = (fp - prev).abs() > threshold
            d_next = (fp - nxt).abs() > threshold
            d_baseline = (nxt - prev).abs() < (threshold * 0.5)
            spike |= (v_ok & d_prev & d_next & d_baseline).to_numpy()
        pick = anom_mask & (diagnoses == "Normal") & spike
        diagnoses = np.where(pick, "Spike", diagnoses)

        # 4. Drift Check (sustained baseline deviation)
        drift = np.zeros(n_rows, dtype=bool)
        for feat, threshold in zip(self.base_features, (2.2, 4.5, 7.5)):
            dev_col = f"{feat}_deviation_{self.rolling_window}"
            if dev_col in df.columns:
                dev = df[dev_col].abs().to_numpy()
            else:
                s = df[feat].ffill().bfill()
                med = s.rolling(41, min_periods=1, center=True).median()
                dev = (s - med).abs().to_numpy()
            drift |= dev > threshold
        pick = anom_mask & (diagnoses == "Normal") & drift
        diagnoses = np.where(pick, "Drift", diagnoses)

        # 5. Default: Multivariate Correlation Outlier
        pick = anom_mask & (diagnoses == "Normal")
        diagnoses = np.where(pick, "Multivariate Outlier", diagnoses)

        return pd.Series(diagnoses, index=df.index)

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

        df_result["pred_is_anomaly"] = y_pred
        df_result["anomaly_score"] = np.round(scores, 4)
        df_result["confidence"] = np.round(confidence, 4)
        df_result["pred_anomaly_type"] = self.classify_root_cause(df_result, y_pred)
        df_result["sensor_health"] = self.calculate_sensor_health(y_pred, window_size=health_window)

        return df_result


if __name__ == "__main__":
    from data_generator import generate_synthetic_weather_data
    from anomaly_injector import inject_anomalies

    print("1. Generating telemetry with 9-feature trend engineering...")
    df_raw = generate_synthetic_weather_data(days=30, interval_minutes=5)
    df_injected = inject_anomalies(df_raw, n_spikes=8, n_frozen=8, n_drift=8, n_gaps=8)

    detector = WeatherAnomalyDetector(contamination=0.055)
    df_processed = detector.detect_and_diagnose(df_injected)

    print(f"\nTrained 9-Feature Space:\n{detector.features}")
    print(f"\nTotal Readings: {len(df_processed)}")
    print(f"Detected Anomalies: {df_processed['pred_is_anomaly'].sum()}")
    print("\nPredicted Anomaly Types:")
    print(df_processed[df_processed["pred_is_anomaly"] == 1]["pred_anomaly_type"].value_counts())
