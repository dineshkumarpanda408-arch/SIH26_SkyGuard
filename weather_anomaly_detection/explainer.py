"""
explainer.py
------------
SHAP (SHapley Additive exPlanations) Engine for PyOD Isolation Forest.
Explains 9-dimensional multivariate anomaly predictions by attributing feature importance
across raw telemetry, rolling averages, and trend deviations.

Features:
- TreeExplainer integration directly on PyOD's underlying Isolation Forest estimator
- Local feature contribution extraction per anomaly across all 9 trend-augmented dimensions
- Plotly interactive bar chart visualization with human-readable meteorological labels
"""

import numpy as np
import pandas as pd
import shap
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
from detector import WeatherAnomalyDetector


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
        """
        Initialize the SHAP TreeExplainer on the fitted detector.

        Parameters:
        -----------
        detector : WeatherAnomalyDetector
            Fitted detector instance.
        background_data : pd.DataFrame, optional
            Baseline telemetry data to calibrate Shapley reference values.
        sample_size : int
            Number of baseline samples for tree background.
        """
        if not detector.is_fitted:
            raise ValueError("The provided detector must be fitted before initializing AnomalyExplainer.")

        self.detector = detector
        self.features = detector.features  # 9 features
        self.model = detector.clf.detector_  # scikit-learn IsolationForest from PyOD

        self.point_cache = {}
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

        Parameters:
        -----------
        row_values : dict, pd.Series, or pd.DataFrame
            Values for the telemetry point.

        Returns:
        --------
        dict
            Mapping from feature name -> SHAP contribution score.
        """
        # Fast path if row already has the 9 features extracted
        if isinstance(row_values, pd.Series) and all(f in row_values.index for f in self.features):
            idx_name = row_values.name
            if idx_name is not None and idx_name in self.point_cache:
                return self.point_cache[idx_name]

            arr = row_values[self.features].to_numpy(dtype=float).reshape(1, -1)
            arr = np.nan_to_num(arr, nan=0.0)
            shap_vals = self.explainer.shap_values(arr)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[0]
            if shap_vals.ndim > 1:
                shap_vals = shap_vals[0]
            res = {feat: float(shap_vals[i]) for i, feat in enumerate(self.features)}
            if idx_name is not None:
                self.point_cache[idx_name] = res
            return res

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

    def explain_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute SHAP values for an entire dataframe.
        """
        X_mat, _, _ = self.detector._preprocess(df)
        shap_vals = self.explainer.shap_values(X_mat)

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]

        shap_df = pd.DataFrame(shap_vals, columns=[f"shap_{f}" for f in self.features], index=df.index)
        abs_vals = np.abs(shap_vals)
        top_idx = np.argmax(abs_vals, axis=1)
        shap_df["top_contributor"] = [self.features[i] for i in top_idx]

        return shap_df

    def create_feature_importance_plot(
        self,
        shap_dict: Dict[str, float],
        title: str = "SHAP 9-Feature Contribution Breakdown",
        predicted_type: str = "Drift"
    ) -> go.Figure:
        """
        Generate an interactive Plotly bar chart visualizing 9-feature contributions.
        Features are sorted by absolute attribution magnitude.
        """
        feature_labels = {
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

        # Sort features by absolute SHAP impact
        items = list(shap_dict.items())
        items.sort(key=lambda x: abs(x[1]), reverse=True)

        features = [x[0] for x in items]
        values = [x[1] for x in items]
        abs_values = [abs(v) for v in values]
        display_names = [feature_labels.get(f, f) for f in features]

        # Highlight primary contributors
        colors = ["#EF4444" if abs(v) == max(abs_values) else ("#F97316" if "deviation" in f else "#3B82F6") for f, v in zip(features, values)]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=abs_values[::-1],
                    y=display_names[::-1],
                    orientation="h",
                    text=[f"{v:.3f}" for v in abs_values[::-1]],
                    textposition="auto",
                    marker=dict(
                        color=colors[::-1],
                        line=dict(color="#1E293B", width=1)
                    ),
                    hovertemplate="<b>%{y}</b><br>SHAP Magnitude: %{x:.4f}<extra></extra>"
                )
            ]
        )

        fig.update_layout(
            title=f"<b>{title}</b><br><sub>Predicted Diagnosis: <span style='color:#EF4444; font-weight:bold;'>{predicted_type}</span></sub>",
            xaxis_title="SHAP Importance Magnitude (|SHAP|)",
            yaxis_title="Feature / Trend Metric",
            template="plotly_white",
            height=380,
            margin=dict(l=20, r=20, t=60, b=40)
        )

        return fig


if __name__ == "__main__":
    from data_generator import generate_synthetic_weather_data
    from anomaly_injector import inject_anomalies

    print("Generating telemetry and fitting 9-feature detector...")
    df_raw = generate_synthetic_weather_data(days=30, interval_minutes=5)
    df_injected = inject_anomalies(df_raw, n_spikes=8, n_frozen=8, n_drift=8, n_gaps=8)

    detector = WeatherAnomalyDetector(contamination=0.055)
    detector.fit(df_injected)

    print("Initializing SHAP Explainer with 9 features...")
    explainer = AnomalyExplainer(detector, background_data=df_injected)

    df_diag = detector.detect_and_diagnose(df_injected)
    anom_rows = df_diag[df_diag["pred_anomaly_type"] == "Drift"]

    if len(anom_rows) > 0:
        sample_row = anom_rows.iloc[0]
        explanation = explainer.explain_point(sample_row)
        print(f"\nExplaining Drift Anomaly at {sample_row['timestamp']}:")
        print("Top SHAP Feature Contributions:")
        for feat, score in sorted(explanation.items(), key=lambda x: abs(x[1]), reverse=True):
            print(f"  - {feat:30s}: {score:+.4f}")
