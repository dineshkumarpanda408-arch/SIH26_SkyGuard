"""Corrected (imputed) value estimation for anomalous readings.

For temperature the correction uses genuine same-hour-of-day diurnal smoothing:
the imputed value blends the recent-window mean with the historical mean of
observations taken at the SAME clock hour (local time), so the estimate respects
the daily temperature cycle. When fewer than one day of history is available it
falls back to recent-window mean imputation only, and the method label always
states exactly what was computed.

Clearly labeled as ESTIMATED, not measured. Confidence is a statistical heuristic
(1 - coefficient of variation) and is never presented as a calibrated probability.
"""

import numpy as np


def estimate(series, variable, window=6, timestamps=None) -> dict:
    """Estimate a plausible value for the current (anomalous) point.

    Parameters
    ----------
    series    : numpy array of valid recent values (without the anomalous point)
    variable  : 'temperature' | 'pressure' | 'humidity'
    window    : look-back window for the recent mean
    timestamps: optional array of pd.Timestamps aligned with `series`; used only
                for temperature diurnal smoothing (same clock-hour mean).

    Returns original/corrected/method/confidence (+ diurnal metadata).
    """
    series = np.asarray([float(x) for x in series if x is not None and not np.isnan(x)])
    if len(series) == 0:
        return {
            "original_value": None,
            "corrected_value": None,
            "correction_method": "NONE",
            "correction_confidence": 0.0,
            "diurnal_used": False,
            "n_same_hour": 0,
        }

    recent = series[-window:] if len(series) >= window else series
    mean = float(np.mean(recent))
    std = float(np.std(recent)) + 1e-9
    corrected = mean

    # confidence higher when recent values are stable (statistical heuristic)
    cv = std / (abs(mean) + 1e-9)
    confidence = float(max(0.0, min(1.0, 1.0 - cv)))

    method = "recent-window mean"
    diurnal_used = False
    n_same_hour = 0

    # temperature diurnal smoothing: genuine same-clock-hour historical mean
    if variable == "temperature" and timestamps is not None and len(series) >= 96:
        try:
            import pandas as pd
            ts = pd.to_datetime(list(timestamps))
            if len(ts) == len(series):
                target_hour = int(ts[-1].hour)  # hour of the anomalous reading
                hours = ts.hour.to_numpy()
                same = series[hours == target_hour]
                if len(same) >= 4:
                    diurnal_mean = float(np.mean(same))
                    # blend toward the diurnal mean; recent window still matters
                    w = float(min(0.6, len(same) / (len(same) + 2 * window)))
                    corrected = (1.0 - w) * mean + w * diurnal_mean
                    method = "same-hour diurnal mean blended with recent-window mean"
                    diurnal_used = True
                    n_same_hour = int(len(same))
        except Exception:
            # never crash the pipeline on a data-shape issue; fall back to
            # recent-window mean with an honest label
            corrected = mean
            method = "recent-window mean"
            diurnal_used = False

    return {
        "original_value": None,  # set by caller
        "corrected_value": round(corrected, 2),
        "correction_method": method,
        "correction_confidence": round(confidence, 2),
        "diurnal_used": diurnal_used,
        "n_same_hour": n_same_hour,
    }