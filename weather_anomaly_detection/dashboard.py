"""
dashboard.py
------------
Smart India Hackathon (SIH) Weather Sensor Anomaly Detection & Diagnostic Dashboard.
Built with Streamlit, Plotly, PyOD Isolation Forest, and SHAP.

Two-Screen Architecture:
1. Overview Screen: High-level KPI telemetry metrics, failure mode distribution,
   sensor health status, and filterable anomaly summary table.
2. Detail Screen: Synchronized multivariate time-series subplots (Temp, Pressure, Humidity),
   red anomaly markers, rolling sensor health score timeline, and interactive SHAP
   feature-importance diagnostic inspector.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

from data_generator import generate_synthetic_weather_data
from anomaly_injector import inject_anomalies
from detector import WeatherAnomalyDetector
from explainer import AnomalyExplainer

# -----------------------------------------------------------------------------
# 1. STREAMLIT PAGE CONFIG & CUSTOM STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Weather Anomaly & Diagnostic Intelligence (SIH)",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for elegant hackathon presentation UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: linear-gradient(135deg, #F8FAFC 0%, #EDF2F7 100%);
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }
    .kpi-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #64748B;
        letter-spacing: 0.05em;
    }
    .kpi-value {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0F172A;
        margin-top: 0.2rem;
    }
    .kpi-subtitle {
        font-size: 0.8rem;
        color: #94A3B8;
    }
    .badge-optimal {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
    }
    .badge-warning {
        background-color: #FEF08A;
        color: #713F12;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
    }
    .badge-critical {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. CACHED DATA PIPELINE & MODEL EXECUTION
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_and_process_weather_data(
    days: int = 30,
    interval_minutes: int = 5,
    station_id: str = "AWS-1",
    contamination: float = 0.055,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic weather telemetry, inject fault signatures,
    and run PyOD Multivariate Isolation Forest detection & diagnostics.
    """
    df_raw = generate_synthetic_weather_data(
        days=days,
        interval_minutes=interval_minutes,
        station_id=station_id,
        seed=seed
    )
    df_injected = inject_anomalies(
        df_raw,
        n_spikes=8,
        n_frozen=8,
        n_drift=8,
        n_gaps=8,
        seed=seed
    )
    detector = WeatherAnomalyDetector(contamination=contamination, random_state=seed)
    df_processed = detector.detect_and_diagnose(df_injected)
    return df_processed


@st.cache_resource(show_spinner=False)
def get_trained_explainer(
    days: int = 30,
    station_id: str = "AWS-1",
    contamination: float = 0.055,
    seed: int = 42
) -> AnomalyExplainer:
    """
    Cache fitted detector and SHAP Explainer instance for specific station.
    """
    df_raw = generate_synthetic_weather_data(days=days, interval_minutes=5, station_id=station_id, seed=seed)
    df_injected = inject_anomalies(df_raw, n_spikes=8, n_frozen=8, n_drift=8, n_gaps=8, seed=seed)
    detector = WeatherAnomalyDetector(contamination=contamination, random_state=seed)
    detector.fit(df_injected)
    explainer = AnomalyExplainer(detector, background_data=df_injected)
    return explainer


# -----------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION & CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/partly-cloudy-day.png", width=70)
st.sidebar.title("🌦️ Weather AI Hub")
st.sidebar.caption("Smart India Hackathon • Telemetry Monitoring")

screen_choice = st.sidebar.radio(
    "Select Screen:",
    ["📊 Overview Screen", "🔬 Detail & Diagnostic Screen"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Station & Simulation")

station_id = st.sidebar.selectbox(
    "Weather Station:",
    ["AWS-1 (North Field Station)", "AWS-2 (Coastal Station)", "AWS-3 (Highland Station)"],
    index=0
)
station_code = station_id.split()[0]

sim_days = st.sidebar.slider("Observation Period (Days):", min_value=7, max_value=30, value=30, step=1)
contamination_val = st.sidebar.slider("Model Contamination (Sensitivity):", min_value=0.01, max_value=0.10, value=0.055, step=0.005)
random_seed = st.sidebar.number_input("Random Seed:", min_value=1, max_value=9999, value=42)

show_ground_truth = st.sidebar.checkbox(
    "🎯 Show Ground-Truth Labels (Demo Mode)",
    value=True,
    help="Display actual injected anomaly tags alongside model predictions."
)

if st.sidebar.button("🔄 Re-run Simulation & Retrain", use_container_width=True):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info(
    "**Tech Stack:**\n"
    "• **Model:** PyOD Multivariate IForest\n"
    "• **XAI:** SHAP TreeExplainer\n"
    "• **Sensors:** Temp (°C), Pres (hPa), Hum (%)\n"
    "• **Faults:** Spike, Frozen, Drift, Comms Gap"
)

# Load data and explainer
with st.spinner("Processing weather telemetry and training PyOD models..."):
    df_data = load_and_process_weather_data(
        days=sim_days,
        station_id=station_code,
        contamination=contamination_val,
        seed=random_seed
    )
    explainer = get_trained_explainer(
        days=sim_days,
        station_id=station_code,
        contamination=contamination_val,
        seed=random_seed
    )


# -----------------------------------------------------------------------------
# 4. SCREEN 1: OVERVIEW SCREEN
# -----------------------------------------------------------------------------
if screen_choice == "📊 Overview Screen":
    st.markdown(f'<div class="main-header">Automatic Weather Station Overview: {station_code}</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time telemetry health summary, multivariate anomaly counters, and failure mode distribution.</div>', unsafe_allow_html=True)

    # Key Statistics
    total_readings = len(df_data)
    total_anomalies = int(df_data["pred_is_anomaly"].sum())
    anomaly_rate = (total_anomalies / total_readings) * 100.0 if total_readings > 0 else 0.0
    latest_health = float(df_data["sensor_health"].iloc[-1])
    avg_health = float(df_data["sensor_health"].mean())

    if latest_health >= 85.0:
        health_badge = '<span class="badge-optimal">OPTIMAL</span>'
    elif latest_health >= 60.0:
        health_badge = '<span class="badge-warning">DEGRADED</span>'
    else:
        health_badge = '<span class="badge-critical">CRITICAL</span>'

    # Top KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Telemetry Points</div>
            <div class="kpi-value">{total_readings:,}</div>
            <div class="kpi-subtitle">{sim_days} Days @ 5-min intervals</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Detected Anomalies</div>
            <div class="kpi-value" style="color:#DC2626;">{total_anomalies:,}</div>
            <div class="kpi-subtitle">Multivariate PyOD IForest</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Anomaly Rate</div>
            <div class="kpi-value">{anomaly_rate:.2f}%</div>
            <div class="kpi-subtitle">Target baseline ~ 5%</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Live Sensor Health</div>
            <div class="kpi-value">{latest_health:.1f}%</div>
            <div class="kpi-subtitle">{health_badge} (Avg: {avg_health:.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Two Charts: Distribution & Health Gauge
    chart_col1, chart_col2 = st.columns([1.1, 0.9])

    with chart_col1:
        st.subheader("📌 Detected Failure Modes & Root-Cause Distribution")
        anom_counts = df_data[df_data["pred_is_anomaly"] == 1]["pred_anomaly_type"].value_counts().reset_index()
        anom_counts.columns = ["Anomaly Type", "Count"]

        color_map = {
            "Spike": "#EF4444",
            "Frozen": "#3B82F6",
            "Drift": "#F59E0B",
            "Communication gap": "#8B5CF6",
            "Multivariate Outlier": "#EC4899"
        }
        pie_colors = [color_map.get(t, "#64748B") for t in anom_counts["Anomaly Type"]]

        fig_pie = go.Figure(data=[
            go.Pie(
                labels=anom_counts["Anomaly Type"],
                values=anom_counts["Count"],
                hole=0.48,
                marker=dict(colors=pie_colors, line=dict(color="#FFFFFF", width=2)),
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>Detected Events: %{value} (%{percent})<extra></extra>"
            )
        ])
        fig_pie.update_layout(
            template="plotly_white",
            height=320,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with chart_col2:
        st.subheader("🛡️ Station Health Index Gauge")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=latest_health,
            delta={'reference': avg_health, 'increasing': {'color': "#10B981"}, 'decreasing': {'color': "#EF4444"}},
            title={'text': "<b>Operational Health Index (0-100)</b>", 'font': {'size': 15}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                'bar': {'color': "#2563EB", 'thickness': 0.25},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#E2E8F0",
                'steps': [
                    {'range': [0, 60], 'color': '#FEE2E2'},
                    {'range': [60, 85], 'color': '#FEF08A'},
                    {'range': [85, 100], 'color': '#DEF7EC'}
                ],
                'threshold': {
                    'line': {'color': "#DC2626", 'width': 4},
                    'thickness': 0.75,
                    'value': 60
                }
            }
        ))
        fig_gauge.update_layout(height=320, margin=dict(l=30, r=30, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("---")

    # Summary Table of Flagged Anomalies
    st.subheader("📋 Flagged Telemetry Anomalies Log")
    
    # Filter controls
    tbl_col1, tbl_col2, tbl_col3 = st.columns([1, 1, 1])
    with tbl_col1:
        type_filter = st.multiselect(
            "Filter by Predicted Anomaly Type:",
            options=list(df_data["pred_anomaly_type"].unique()),
            default=[t for t in df_data["pred_anomaly_type"].unique() if t != "Normal"]
        )
    with tbl_col2:
        min_conf = st.slider("Minimum Confidence Score:", 0.0, 1.0, 0.0, 0.05)
    with tbl_col3:
        only_anoms = st.checkbox("Show Only Detected Anomalies", value=True)

    # Filter DataFrame
    df_filtered = df_data.copy()
    if only_anoms:
        df_filtered = df_filtered[df_filtered["pred_is_anomaly"] == 1]
    if type_filter:
        df_filtered = df_filtered[df_filtered["pred_anomaly_type"].isin(type_filter)]
    df_filtered = df_filtered[df_filtered["confidence"] >= min_conf]

    # Select display columns
    display_cols = [
        "timestamp", "station_id", "temperature", "pressure", "humidity",
        "pred_anomaly_type", "confidence", "anomaly_score", "sensor_health"
    ]
    if show_ground_truth and "anomaly_type" in df_filtered.columns:
        display_cols.insert(6, "anomaly_type")

    # Rename for presentation
    rename_dict = {
        "timestamp": "Timestamp",
        "station_id": "Station",
        "temperature": "Temp (°C)",
        "pressure": "Pressure (hPa)",
        "humidity": "Humidity (%)",
        "pred_anomaly_type": "Predicted Diagnosis",
        "anomaly_type": "Ground Truth",
        "confidence": "Confidence",
        "anomaly_score": "Anomaly Score",
        "sensor_health": "Health (%)"
    }

    df_display = df_filtered[display_cols].rename(columns=rename_dict)

    st.dataframe(
        df_display.style.format({
            "Temp (°C)": "{:.2f}",
            "Pressure (hPa)": "{:.2f}",
            "Humidity (%)": "{:.2f}",
            "Confidence": "{:.3f}",
            "Anomaly Score": "{:.4f}",
            "Health (%)": "{:.1f}"
        }),
        height=320,
        use_container_width=True
    )

    # CSV Download Button
    csv_buffer = io.StringIO()
    df_display.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Export Flagged Anomalies Report (CSV)",
        data=csv_buffer.getvalue(),
        file_name=f"weather_anomalies_{station_code}.csv",
        mime="text/csv"
    )


# -----------------------------------------------------------------------------
# 5. SCREEN 2: DETAIL & DIAGNOSTIC SCREEN
# -----------------------------------------------------------------------------
elif screen_choice == "🔬 Detail & Diagnostic Screen":
    st.markdown(f'<div class="main-header">Multivariate Telemetry & XAI Deep-Dive: {station_code}</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Interactive multi-parameter time-series, red anomaly flags, SHAP feature importance, and root-cause diagnostics.</div>', unsafe_allow_html=True)

    # Date range selector
    min_date = df_data["timestamp"].min().date()
    max_date = df_data["timestamp"].max().date()
    
    date_col1, date_col2 = st.columns(2)
    with date_col1:
        start_date_sel = st.date_input("Start Date:", value=min_date, min_value=min_date, max_value=max_date)
    with date_col2:
        end_date_sel = st.date_input("End Date:", value=min_date + pd.Timedelta(days=7), min_value=min_date, max_value=max_date)

    if start_date_sel > end_date_sel:
        st.error("Error: Start Date must be before or equal to End Date.")
        st.stop()

    # Filter slice
    mask_time = (df_data["timestamp"].dt.date >= start_date_sel) & (df_data["timestamp"].dt.date <= end_date_sel)
    df_sub = df_data[mask_time].copy().reset_index(drop=True)

    if len(df_sub) == 0:
        st.warning("No telemetry records found for the selected time range.")
        st.stop()

    # 4-Row Synchronized Plotly Subplots
    fig_timeseries = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "<b>1. Temperature Telemetry (°C)</b>",
            "<b>2. Barometric Pressure Telemetry (hPa)</b>",
            "<b>3. Relative Humidity Telemetry (%)</b>",
            "<b>4. Rolling Sensor Health Index (%)</b>"
        )
    )

    # Anomalies subset
    anom_sub = df_sub[df_sub["pred_is_anomaly"] == 1]

    # --- 1. Temperature Subplot ---
    fig_timeseries.add_trace(
        go.Scatter(
            x=df_sub["timestamp"],
            y=df_sub["temperature"],
            mode="lines",
            name="Temperature (°C)",
            line=dict(color="#F97316", width=1.8),
            hovertemplate="Time: %{x}<br>Temp: %{y:.2f} °C<extra></extra>"
        ),
        row=1, col=1
    )
    if len(anom_sub) > 0:
        fig_timeseries.add_trace(
            go.Scatter(
                x=anom_sub["timestamp"],
                y=anom_sub["temperature"],
                mode="markers",
                name="Flagged Anomaly",
                marker=dict(color="#DC2626", size=7, symbol="circle", line=dict(color="#7F1D1D", width=1)),
                hovertemplate="<b>ANOMALY DETECTED</b><br>Time: %{x}<br>Temp: %{y:.2f} °C<br>Diagnosis: %{text}<extra></extra>",
                text=anom_sub["pred_anomaly_type"]
            ),
            row=1, col=1
        )

    # --- 2. Pressure Subplot ---
    fig_timeseries.add_trace(
        go.Scatter(
            x=df_sub["timestamp"],
            y=df_sub["pressure"],
            mode="lines",
            name="Pressure (hPa)",
            line=dict(color="#2563EB", width=1.8),
            hovertemplate="Time: %{x}<br>Pressure: %{y:.2f} hPa<extra></extra>"
        ),
        row=2, col=1
    )
    if len(anom_sub) > 0:
        fig_timeseries.add_trace(
            go.Scatter(
                x=anom_sub["timestamp"],
                y=anom_sub["pressure"],
                mode="markers",
                name="Flagged Anomaly",
                showlegend=False,
                marker=dict(color="#DC2626", size=7, symbol="circle", line=dict(color="#7F1D1D", width=1)),
                hovertemplate="<b>ANOMALY DETECTED</b><br>Time: %{x}<br>Pressure: %{y:.2f} hPa<extra></extra>"
            ),
            row=2, col=1
        )

    # --- 3. Humidity Subplot ---
    fig_timeseries.add_trace(
        go.Scatter(
            x=df_sub["timestamp"],
            y=df_sub["humidity"],
            mode="lines",
            name="Humidity (%)",
            line=dict(color="#06B6D4", width=1.8),
            hovertemplate="Time: %{x}<br>Humidity: %{y:.2f} %<extra></extra>"
        ),
        row=3, col=1
    )
    if len(anom_sub) > 0:
        fig_timeseries.add_trace(
            go.Scatter(
                x=anom_sub["timestamp"],
                y=anom_sub["humidity"],
                mode="markers",
                name="Flagged Anomaly",
                showlegend=False,
                marker=dict(color="#DC2626", size=7, symbol="circle", line=dict(color="#7F1D1D", width=1)),
                hovertemplate="<b>ANOMALY DETECTED</b><br>Time: %{x}<br>Humidity: %{y:.2f} %<extra></extra>"
            ),
            row=3, col=1
        )

    # --- 4. Sensor Health Score Subplot ---
    fig_timeseries.add_trace(
        go.Scatter(
            x=df_sub["timestamp"],
            y=df_sub["sensor_health"],
            mode="lines",
            name="Sensor Health (%)",
            line=dict(color="#10B981", width=2.0),
            fill="tozeroy",
            fillcolor="rgba(16, 185, 129, 0.15)",
            hovertemplate="Time: %{x}<br>Health Score: %{y:.1f} %<extra></extra>"
        ),
        row=4, col=1
    )

    fig_timeseries.update_layout(
        height=720,
        template="plotly_white",
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_timeseries, use_container_width=True)

    st.markdown("---")

    # Interactive SHAP Explainability & Root-Cause Inspector
    st.subheader("🔍 Interactive Anomaly Diagnostic & SHAP Explainability Inspector")
    st.caption("Select any detected anomaly event to inspect its feature attribution and mechanical failure mode.")

    all_anoms = df_data[df_data["pred_is_anomaly"] == 1].reset_index()

    if len(all_anoms) == 0:
        st.info("No anomalies detected in the dataset with the current settings.")
    else:
        # Create user-friendly label for each anomaly option
        def make_anom_label(row):
            ts = row["timestamp"].strftime("%Y-%m-%d %H:%M")
            diag = row["pred_anomaly_type"]
            score = row["anomaly_score"]
            return f"[{ts}] - Type: {diag} | Score: {score:.3f} | Index: {row['index']}"

        anom_options = [make_anom_label(row) for _, row in all_anoms.iterrows()]
        selected_anom_str = st.selectbox("Select Anomaly Instance to Diagnose:", options=anom_options, index=0)

        # Retrieve selected row
        sel_idx = int(selected_anom_str.split("Index: ")[1])
        target_row = df_data.loc[sel_idx]

        insp_col1, insp_col2 = st.columns([1.1, 0.9])

        with insp_col1:
            st.markdown("#### 📊 SHAP Feature Attribution (TreeExplainer)")
            # Compute SHAP explanation for the point
            shap_dict = explainer.explain_point(target_row)
            fig_shap = explainer.create_feature_importance_plot(
                shap_dict,
                title="SHAP Feature Importance Breakdown",
                predicted_type=target_row["pred_anomaly_type"]
            )
            st.plotly_chart(fig_shap, use_container_width=True)

        with insp_col2:
            st.markdown("#### 🛠️ Diagnostic Root-Cause Report")
            t_val = target_row["temperature"]
            p_val = target_row["pressure"]
            h_val = target_row["humidity"]
            pred_type = target_row["pred_anomaly_type"]
            conf = target_row["confidence"]
            score = target_row["anomaly_score"]
            timestamp_str = target_row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")

            # Diagnostic guidance
            action_map = {
                "Spike": "Transient electrical impulse or EMI disturbance. Filter outlier and inspect power supply grounding.",
                "Frozen": "Sensor communication bus locked or transducer mechanically stuck. Perform remote reboot or check I2C/RS485 interface.",
                "Drift": "Sensor calibration offset degradation or psychrometric wick contamination. Recalibrate sensor against reference standard.",
                "Communication gap": "Telemetry packet loss. Check 4G/LoRaWAN signal strength, solar battery voltage, and gateway status.",
                "Multivariate Outlier": "Thermodynamic inconsistency detected between Temperature and Humidity correlation. Inspect ventilation fan."
            }
            rec_action = action_map.get(pred_type, "Inspect sensor telemetry and verify against neighboring weather stations.")

            st.markdown(f"""
            <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 1.2rem;">
                <p><b>Event Timestamp:</b> <code>{timestamp_str}</code></p>
                <p><b>Station Identifier:</b> <code>{target_row['station_id']}</code></p>
                <p><b>Observed Telemetry:</b><br>
                   • Temperature: <b>{t_val} °C</b><br>
                   • Pressure: <b>{p_val} hPa</b><br>
                   • Humidity: <b>{h_val} %</b>
                </p>
                <p><b>PyOD Anomaly Score:</b> <code>{score:.4f}</code> (Confidence: <b>{conf*100:.1f}%</b>)</p>
                <p><b>Predicted Failure Mode:</b> <span style="color:#DC2626; font-weight:700; font-size:1.1rem;">{pred_type}</span></p>
                <hr style="margin: 0.8rem 0;">
                <p><b>🔧 Recommended Maintenance Action:</b><br>
                <span style="color:#1E293B;">{rec_action}</span></p>
            </div>
            """, unsafe_allow_html=True)
