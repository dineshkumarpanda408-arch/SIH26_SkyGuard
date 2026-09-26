"""pyod_explainer.py
-----------------
SHAP (SHapley Additive exPlanations) Engine for PyOD Isolation Forest.
Explains 9-dimensional multivariate anomaly predictions by attributing feature importance
across raw telemetry, rolling averages, and trend deviations.

Adapted from weather_anomaly_detection for the SkyGuard AI backend.
"""

import numpy as np
import pandas as pd
import shap
from typing import Dict, List, Any, Optional
from .pyod_detector import WeatherAnomalyDetector

# Maintenance action mappings
ACTION_MAP = {
    "Spike": "Transient electrical impulse or EMI disturbance. Filter outlier and inspect power supply grounding.",
    "Frozen": "Sensor communication bus locked or transducer mechanically stuck. Perform remote reboot or check I2C/RS485 interface.",
    "Drift": "Sensor calibration offset degradation or psychrometric wick contamination. Recalibrate sensor against reference standard.",
    "Communication gap": "Telemetry packet loss. Check 4G/LoRaWAN signal strength, solar battery voltage, and gateway status.",
    "Multivariate Outlier": "Thermodynamic inconsistency detected between Temperature and Humidity correlation. Inspect ventilation fan."
}

FEATURE_LABELS = {
    "temperature": "Temp (°C)",
    "pressure": "Pressure (hPa)",
    "humidity": "Humidity (%)",
    "temperature_rolling_mean_20": "Temp Roll-Mean (20)",
    "temperature_deviation_20": "Temp Trend Deviation",
    "pressure_rolling_mean_20": "Pres Roll-Mean (20)",
    "pressure_deviation_20": "Pres Trend Deviation",
    "humidity_rolling_mean_20": "Hum Roll-Mean (20)",
    "humidity_deviation_20": "Hum Trend Deviation",
}


class AnomalyExplainer:
    """
    SHAP-based explainability layer for weather anomaly detection with 9-feature support.
    """

    def __init__(
        self,
        detector: WeatherAnomalyDetector,
        background_data: Optional[pd.DataFrame] = None,
        sample_size: int = 150
    ):
        if not detector.is_fitted:
            raise ValueError("The provided detector must be fitted before initializing AnomalyExplainer.")

        self.detector = detector
        self.features = detector.features  # 9 features
        self.model = detector.clf.detector_  # scikit-learn IsolationForest from PyOD

        if background_data is not None:
            bg_clean, _, _ = self.detector._preprocess(background_data)
            if len(bg_clean) > sample_size:
                np.random.seed(42)
                bg_sample = bg_clean[np.random.choice(len(bg_clean), sample_size, replace=False)]
            else:
                bg_sample = bg_clean
            self.explainer = shap.TreeExplainer(self.model, data=bg_sample)
        else:
            self.explainer = shap.TreeExplainer(self.model)

    def explain_point(self, row_values: Any) -> Dict[str, float]:
        """
        Compute SHAP feature importance for a single telemetry reading across all 9 features.
        """
        if isinstance(row_values, pd.Series):
            df_single = pd.DataFrame([row_values])
        elif isinstance(row_values, dict):
            df_single = pd.DataFrame([row_values])
        elif isinstance(row_values, pd.DataFrame):
            df_single = row_values.iloc[[0]]
        else:
            arr = np.asarray(row_values).reshape(1, -1)
            shap_vals = self.explainer.shap_values(arr)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[0]
            if shap_vals.ndim > 1:
                shap_vals = shap_vals[0]
            return {feat: float(shap_vals[i]) for i, feat in enumerate(self.features)}

        # Extract 9 features using detector
        X_mat, _, _ = self.detector._preprocess(df_single)
        X_mat = np.nan_to_num(X_mat, nan=0.0)

        shap_vals = self.explainer.shap_values(X_mat)

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        if shap_vals.ndim > 1:
            shap_vals = shap_vals[0]

        return {feat: float(shap_vals[i]) for i, feat in enumerate(self.features)}

    def explain_with_details(self, row_values: Any, predicted_type: str = "Drift") -> Dict[str, Any]:
        """
        Compute full SHAP feature attribution breakdown, top contributors,
        and diagnostic maintenance report.
        """
        shap_dict = self.explain_point(row_values)

        # Sort items by absolute SHAP magnitude descending
        sorted_items = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        max_abs = max(abs(v) for _, v in sorted_items) if sorted_items else 1.0

        features_list = []
        for feat, val in sorted_items:
            abs_val = abs(val)
            # category type for visual color coding
            if abs_val == max_abs:
                color_type = "primary"  # Red
            elif "deviation" in feat:
                color_type = "deviation"  # Orange
            else:
                color_type = "baseline"  # Blue

            features_list.append({
                "feature": feat,
                "label": FEATURE_LABELS.get(feat, feat),
                "shap_value": round(float(val), 4),
                "abs_magnitude": round(float(abs_val), 4),
                "color_type": color_type,
            })

        rec_action = ACTION_MAP.get(
            predicted_type,
            "Inspect sensor telemetry and verify against neighboring weather stations."
        )

        top_feature = features_list[0] if features_list else None

        return {
            "features": features_list,
            "top_contributor": top_feature["feature"] if top_feature else None,
            "top_contributor_label": top_feature["label"] if top_feature else None,
            "recommended_action": rec_action,
            "predicted_anomaly_type": predicted_type,
        }
