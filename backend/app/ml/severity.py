"""Severity scoring for anomalies.

Considers magnitude, persistence, anomaly type, confidence, sensor impact, and
communication-failure duration. Produces LOW/MEDIUM/HIGH/CRITICAL.
"""


def _maximum(a, b):
    return a if a >= b else b


def severity(anomaly_type, magnitude_norm, confidence, persistence=1, type_weight_map=None):
    """Compute severity level on standardized [0,1] confidence and magnitude scale.

    anomaly_type      : string
    magnitude_norm    : normalized deviation magnitude in [0,1]
    confidence        : anomaly confidence in [0,1]
    persistence       : number of consecutive affected readings
    type_weight_map   : base severity contribution by anomaly type
    """
    if type_weight_map is None:
        type_weight_map = TYPE_BASE
    base = type_weight_map.get(anomaly_type, 0.4)

    # Standardized composite score in [0,1]
    conf = max(0.0, min(1.0, float(confidence or 0.0)))
    mag = max(0.0, min(1.0, float(magnitude_norm or 0.0)))
    score = 0.35 * base + 0.30 * mag + 0.20 * conf
    # persistence bonus: sustained problems raise severity
    score += 0.15 * min(1.0, max(1, persistence) / 5.0)

    if anomaly_type in ("MISSING_DATA", "COMMUNICATION_FAILURE"):
        score = _maximum(score, 0.3 + min(1.0, persistence / 10.0) * 0.5)

    if score >= 0.78:
        return "CRITICAL", round(score, 4)
    if score >= 0.58:
        return "HIGH", round(score, 4)
    if score >= 0.40:
        return "MEDIUM", round(score, 4)
    return "LOW", round(score, 4)


# Base contribution of each anomaly type toward severity (0..1)
TYPE_BASE = {
    "SPIKE": 0.65,
    "DROP": 0.65,
    "FROZEN_SENSOR": 0.60,
    "DRIFT": 0.50,
    "NOISE": 0.35,
    "MISSING_DATA": 0.40,
    "COMMUNICATION_FAILURE": 0.70,
    "SUDDEN_SHIFT": 0.60,
    "MULTIVARIATE_INCONSISTENCY": 0.55,
}
