"""Lightweight sensor degradation analysis.

Detects increasing anomaly frequency, drift, instability, missing data, and
communication problems, and reports a trend + MEDIUM/LOW/HIGH risk plus an
actionable recommended action. We explicitly do NOT claim an exact failure
date without a trained model.
"""


def _trend_label(trend: str) -> str:
    """Title-case a lowercase trend token for display (Improving/Stable/Declining)."""
    return {
        "declining": "Declining",
        "improving": "Improving",
        "stable": "Stable",
    }.get(trend, trend.capitalize())


def _recommended_action(current_health: float, trend: str, feature=None, factors=None) -> str:
    """Human-actionable next step derived from the SAME inputs already used to
    compute the risk. The health/trend calculation is unchanged; this only turns
    the existing signal into a concrete instruction.
    """
    sensor = (feature or "sensor").capitalize()

    # A named sensor feature wins: the most direct calibration instruction.
    if feature:
        if current_health < 40:
            return f"Immediately inspect/calibrate {sensor} sensor - health critically low."
        if current_health < 60:
            return f"Inspect/calibrate {sensor} sensor; continue monitoring."
        if trend == "declining":
            return f"Inspect/calibrate {sensor} sensor before health degrades further."
        return f"No action needed now; continue routine monitoring of {sensor} sensor."

    # Otherwise base the instruction on the dominant worsening factor, when supplied.
    if factors:
        # Lower health component = worse contributor. Report the weakest factor.
        weakest = min(factors, key=factors.get)
        factor_actions = {
            "comm_failure_pct": "Inspect sensor wiring/communication - check for data dropouts.",
            "missing_rate_pct": "Investigate missing data - verify sensor connectivity/reporting.",
            "anomaly_rate_pct": "Inspect/calibrate sensor - anomalous readings detected.",
            "drift_pct": "Calibrate sensor - drift detected relative to recent baseline.",
            "stability_pct": "Investigate instability - reseat/replace sensor if volatility persists.",
        }
        if weakest in factor_actions:
            return factor_actions[weakest]

    if current_health < 40:
        return "Immediately inspect/calibrate sensors at this station - health critically low."
    if current_health < 60:
        return "Inspect/calibrate sensors at this station and re-check shortly."
    if trend == "declining":
        return "Inspect/calibrate sensors at this station before health degrades further."
    return "No immediate action required; continue routine monitoring."


def analyze(current_health: float, history_scores: list, feature=None, factors=None) -> dict:
    """history_scores: chronological list of past health scores.

    `feature` (optional) names the primary sensor variable (e.g. 'temperature')
    so the recommended action can reference it. `factors` (optional) is a dict of
    component health factors used only to sharpen the recommended action when no
    single feature is named. Neither changes the risk/trend computation.
    """
    if not history_scores:
        return {
            "degradation_status": "UNKNOWN",
            "trend": "stable",
            "trend_label": "Stable",
            "risk_level": "LOW",
            "evidence": [],
            "recommended_action": "No health history available to assess degradation.",
        }

    # Linear trend over recent history
    import numpy as np

    y = np.asarray(history_scores, dtype=float)
    n = len(y)
    x = np.arange(n)
    slope = np.polyfit(x, y, 1)[0] if n >= 2 else 0.0

    if slope < -0.8:
        trend = "declining"
    elif slope > 0.8:
        trend = "improving"
    else:
        trend = "stable"

    evidence = []
    if slope < -0.8:
        evidence.append(f"Health trending downward ({slope:+.2f}/reading).")
    else:
        evidence.append("No significant downward health trend detected.")

    if current_health < 40:
        evidence.append("Low current health indicates elevated degradation risk.")
    elif current_health < 60:
        evidence.append("Moderate current health; monitor closely.")

    if current_health < 40 or trend == "declining" and current_health < 60:
        risk = "HIGH"
        degradation_status = "ACTIVE"
    elif current_health < 60 or trend == "declining":
        risk = "MEDIUM"
        degradation_status = "WATCH"
    else:
        risk = "LOW"
        degradation_status = "STABLE"

    return {
        "degradation_status": degradation_status,
        "trend": trend,
        "trend_label": _trend_label(trend),
        "risk_level": risk,
        "evidence": evidence,
        "recommended_action": _recommended_action(current_health, trend, feature=feature, factors=factors),
    }
