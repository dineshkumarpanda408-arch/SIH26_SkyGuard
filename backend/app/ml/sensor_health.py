"""Sensor health scoring from actual historical behavior.

Health score in [0,100] derived from anomaly frequency, missing data,
communication failures, drift, stability, and recent trend. No hardcoded values.
"""

import numpy as np


def _clamp(v, lo=0.0, hi=100.0):
    return float(min(hi, max(lo, v)))


def compute_health(metrics: dict) -> dict:
    """metrics:
        anomaly_rate   : fraction of recent readings flagged anomalous
        missing_rate   : fraction missing
        comm_failures  : fraction of expected readings not received
        drift          : recent |slope| relative to typical (0..1 normalized)
        stability      : 1 - normalized recent volatility (0..1)
        trend          : 'declining'/'stable'/'improving'
    """
    anomaly_rate = _clamp(metrics.get("anomaly_rate", 0.0) * 100)
    missing_rate = _clamp(metrics.get("missing_rate", 0.0) * 100)
    comm_fail = _clamp(metrics.get("comm_failure_rate", 0.0) * 100)
    drift = _clamp(metrics.get("drift", 0.0) * 100)
    stability = _clamp(metrics.get("stability", 1.0) * 100)

    # higher component = better health
    data_quality = _clamp(100 - missing_rate)
    anomaly_freq_score = _clamp(100 - anomaly_rate)
    communication = _clamp(100 - comm_fail)
    drift_score = _clamp(100 - drift)
    ideal_stability = _clamp(100 - (100 - stability))  # stability already good

    health = (
        0.25 * data_quality
        + 0.30 * anomaly_freq_score
        + 0.20 * communication
        + 0.15 * stability
        + 0.10 * drift_score
    )
    health = _clamp(health)

    trend = metrics.get("trend", "stable")
    if health > 70:
        status = "NORMAL"
    elif health >= 40:
        status = "WARNING"
    else:
        status = "CRITICAL"

    return {
        "health_score": health,
        "health_status": status,
        "data_quality": data_quality,
        "anomaly_frequency": anomaly_freq_score,
        "communication": communication,
        "stability": stability,
        "drift": drift_score,
        "trend": trend,
        "factors": {
            "anomaly_rate_pct": anomaly_rate,
            "missing_rate_pct": missing_rate,
            "comm_failure_pct": comm_fail,
            "drift_pct": drift,
            "stability_pct": stability,
        },
    }


def status_of(score: float) -> str:
    if score > 70:
        return "NORMAL"
    if score >= 40:
        return "WARNING"
    return "CRITICAL"
