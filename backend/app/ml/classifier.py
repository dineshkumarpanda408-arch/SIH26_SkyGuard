"""Anomaly type classification based on statistical/structural evidence.

Deterministic, evidence-driven rules using the actual series behavior rather
than fabricated labels. The current (suspected anomalous) point is compared
against a robust PRIOR baseline that does not include it, so spikes/drops are
not masked by their own inflated standard deviation.
"""

import numpy as np


def classify(row, series, window=5):
    """Classify the anomaly type for a single suspected-anomalous reading.

    Parameters
    ----------
    row : current reading (dict with 'value', 'variable', 'expected_value')
    series : numpy array of the variable INCLUDING the current point as its last
             element.
    window : look-back context window.
    """
    var_value = row.get("value")
    is_missing = var_value is None or (isinstance(var_value, float) and np.isnan(var_value))

    if is_missing:
        return "MISSING_DATA"

    series = np.asarray([float(x) for x in series if x is not None and not np.isnan(x)], dtype=float)
    if len(series) < 3:
        return infer_from_magnitude(row, var_value)

    cur = float(series[-1])
    # prior baseline = window ending just before the current point
    prior = series[-(window + 1) : -1]
    if len(prior) < 2:
        prior = series[:-1]
    prev_mean = float(np.mean(prior))
    prev_std = float(np.std(prior)) + 1e-9
    typical = abs(prev_mean) + 1e-9
    rel = abs(cur - prev_mean) / prev_std

    # ---- FROZEN_SENSOR: recent series (including current) is a plateau ----
    # Checked FIRST (after MISSING_DATA) per the required root-cause priority:
    # a stuck/flat sensor must be reported as FROZEN_SENSOR and never fall
    # through to DRIFT, SPIKE or DROP. The check is dimensionless: spread is a
    # small fraction of the local baseline variability (prev_std), not of the
    # mean. Using `0.02*mean` would be an absolute threshold that grows with the
    # sensor's magnitude (e.g. pressure ~ 1000 hPa) and swallow genuine spikes.
    recent = series[-max(3, min(window, len(series))):]
    if np.ptp(recent) < max(0.05 * prev_std, 1e-4):
        return "FROZEN_SENSOR"

    # ---- SPIKE / DROP: large single-point deviation from robust prior ----
    if cur > prev_mean and rel > 4:
        return "SPIKE"
    if cur < prev_mean and rel > 4:
        return "DROP"

    # ---- DRIFT: monotonic slope across consecutive readings ----
    # Checked AFTER FROZEN_SENSOR and SPIKE/DROP (required priority). A flat
    # trailing plateau is consumed by FROZEN_SENSOR above, so only a genuinely
    # changing, monotonic series reaches this branch (e.g. humidity drifting
    # toward a value). It is still checked before SUDDEN_SHIFT and NOISE.
    if len(series) >= 4:
        recent_s = series[-min(len(series), 6):]
        diffs = np.diff(recent_s)
        if len(diffs) >= 3 and (np.all(diffs > 1e-4) or np.all(diffs < -1e-4)):
            total_change = abs(recent_s[-1] - recent_s[0])
            if total_change > 1.5 * prev_std:
                return "DRIFT"

    # ---- DRIFT fallback: small but persistent trend ----
    if len(series) >= 6:
        slope = float(np.polyfit(np.arange(len(series)), series, 1)[0])
        if abs(slope * len(series)) > 2 * (float(np.std(series)) + 1e-9):
            return "DRIFT"

    # ---- SUDDEN_SHIFT: persistent level change (early vs late window) ----
    # "Other": checked after DRIFT. A sustained level step whose post-shift span
    # is flat is already claimed by FROZEN_SENSOR above; this only fires when the
    # shift is a real level change without a trailing plateau.
    if len(series) >= window + 2:
        early = series[:window]
        late = series[-window:]
        early_std = float(np.std(early))
        ref = early_std if early_std > 1e-6 else 1e-6
        sep = abs(float(np.mean(late) - np.mean(early)))
        if sep > 2.5 * ref and rel < 4 and abs(cur - prev_mean) < 3 * prev_std:
            return "SUDDEN_SHIFT"

    if rel > 2:
        return "NOISE"
    return "NOISE"


def infer_from_magnitude(row, var_value):
    """Fallback classification when only the value is available."""
    name = str(row.get("variable", ""))
    expected = row.get("expected_value")
    try:
        expected = float(expected)
    except (TypeError, ValueError):
        expected = None
    if var_value is None or (isinstance(var_value, float) and np.isnan(var_value)):
        return "MISSING_DATA"
    if expected is not None and abs(expected) > 1e-9:
        if var_value > expected * 1.8:
            return "SPIKE"
        if var_value < expected * 0.3:
            return "DROP"
    if "temperature" in name and var_value > 45:
        return "SPIKE"
    if "temperature" in name and var_value < -20:
        return "DROP"
    return "NOISE"
