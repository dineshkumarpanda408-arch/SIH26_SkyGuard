"""Anomaly simulator: injects known anomalies into a normal weather stream.

Each injected reading is labelled ground_truth=True and tagged with its type so
the evaluation gives objective metrics.
"""

import numpy as np
import pandas as pd

ANOMALY_TYPES = [
    "SPIKE",
    "DROP",
    "FROZEN_SENSOR",
    "DRIFT",
    "NOISE",
    "MISSING_DATA",
    "COMMUNICATION_FAILURE",
    "SUDDEN_SHIFT",
]

VARIABLES = ["temperature", "pressure", "humidity"]

# The simulation UI uses short, user-friendly type names. Map them to the
# canonical injector/classifier names so every injected type actually injects.
ALIASES = {
    "FROZEN": "FROZEN_SENSOR",
    "MISSING": "MISSING_DATA",
    "FAULT": "COMMUNICATION_FAILURE",
}


def inject(df: pd.DataFrame, variable: str, anomaly_type: str, magnitude=None,
           duration=1, start_idx=None, rng=None) -> pd.DataFrame:
    """Return a modified COPY of df with an anomaly injected at start_idx.

    Canonical shapes (see CLASSIFIER note: the classifier must never see the
    requested type; the shape below is the only thing the classifier sees):

      SPIKE            : baseline -> single positive jump -> baseline
      DROP             : baseline -> single negative jump -> baseline
      DRIFT            : gradual monotonic movement over `duration`
      FROZEN_SENSOR    : constant / near-constant plateau over `duration`
      MISSING_DATA     : NaN over `duration`
      COMMUNICATION_FAILURE : NaN over `duration` (same signal shape, distinct tag)
      NOISE            : increased residual volatility over `duration`
      SUDDEN_SHIFT     : persistent level shift over `duration`

    A SPIKE/DROP with duration>1 stays a single-point jump (it is deliberately
    NOT turned into a flat plateau, which would be misread as a frozen sensor).

    Ground-truth labels are added in `ground_truth` and `injected_type` columns,
    and a deterministic metadata record is attached at `df.attrs["injection_meta"]`:
    {requested_type, canonical_type, variable, magnitude_sigma, duration,
     start_idx, affected_indices, true_value}. `anomaly_type` accepts both the
    long and short names (see ALIASES). All randomness is seeded (default 42).
    """
    rng = rng or np.random.default_rng(42)
    df = df.copy()
    requested_type = anomaly_type
    anomaly_type = ALIASES.get(anomaly_type, anomaly_type)
    df["ground_truth"] = False
    df["injected_type"] = None

    if start_idx is None:
        start_idx = max(1, len(df) - duration - 2)
    end = min(len(df), start_idx + duration)
    idxs = list(range(start_idx, end))
    if not idxs:
        return df

    if variable not in df.columns:
        return df

    base = df[variable].iloc[start_idx - 1] if start_idx >= 1 else df[variable].mean()
    if base is None or np.isnan(base):
        base = float(df[variable].mean())

    # Defaults tuned so each pattern is structurally recognisable when the
    # classifier inspects the resulting series.
    if anomaly_type == "DRIFT" and duration < 4:
        duration = 4
    if anomaly_type == "FROZEN_SENSOR" and duration < 3:
        duration = 3
    end = min(len(df), start_idx + duration)
    idxs = list(range(start_idx, end))
    if not idxs:
        return df

    # SPIKE / DROP are single-point of the actual injection span regardless of
    # the requested duration so the shape stays a spike/drop, not a plateau.
    if anomaly_type in ("SPIKE", "DROP"):
        affected = [start_idx]
    else:
        affected = idxs

    # Characteristic scale = standard deviation of the variable's recent history.
    # Using this (rather than raw units or a per-variable hardcoded offset) keeps
    # SPIKE/DROP genuinely variable-agnostic: one "magnitude" unit == one sigma
    # deviation, regardless of whether the sensor is temperature/pressure/humidity.
    recent = df[variable].dropna().tail(40)
    scale = float(recent.std()) if len(recent) >= 2 else 1.0
    if not np.isfinite(scale) or scale < 1e-9:
        scale = 1.0

    if magnitude is None or magnitude == 0:
        if variable == "pressure":
            def_mag = 60.0
        elif variable == "humidity":
            def_mag = 25.0
        else:
            def_mag = 30.0
        sigma = def_mag
    else:
        sigma = abs(float(magnitude))
    dev = sigma * scale

    for i in affected:
        df.loc[i, "ground_truth"] = True
        df.loc[i, "injected_type"] = anomaly_type
        step = i - start_idx + 1

        if anomaly_type == "SPIKE":
            val = base + dev
        elif anomaly_type == "DROP":
            val = base - dev
        elif anomaly_type == "NOISE":
            val = base + rng.normal(0, dev * 0.5)
        elif anomaly_type == "FROZEN_SENSOR":
            # genuine plateau: every injected reading holds constant value
            val = float(base)
        elif anomaly_type == "SUDDEN_SHIFT":
            # persistent level shift (a step, but not a flat spike): the whole
            # affected span sits at the shifted level
            val = base + dev
        elif anomaly_type in ("MISSING_DATA", "COMMUNICATION_FAILURE"):
            val = np.nan
        elif anomaly_type == "DRIFT":
            # Keep the drift monotonic and inside the sensor's physical domain.
            # Without this cap an upward humidity drift slams into 100% and the
            # trailing flat line is misread as a frozen sensor. Choose the
            # direction with the most headroom so the slope stays visible.
            if variable == "humidity":
                up_headroom = max(0.0, 100.0 - base)
                down_headroom = max(0.0, base - 0.0)
                use_up = up_headroom >= down_headroom
                avail = up_headroom if use_up else down_headroom
                dev_eff = min(dev, avail) if avail > 0 else 0.0
                sign = 1.0 if use_up else -1.0
                rate = (dev_eff / max(duration, 1)) * sign
                val = base + rate * step
            else:
                rate = dev / max(duration, 1)
                val = base + rate * step
        else:
            val = base + dev

        # keep humidity within its physical domain (0-100 %)
        if variable == "humidity" and not np.isnan(val):
            val = min(100.0, max(0.0, float(val)))

        df.loc[i, variable] = val

    # deterministic, explicit ground-truth metadata
    df.attrs["injection_meta"] = {
        "requested_type": requested_type,
        "canonical_type": anomaly_type,
        "variable": variable,
        "magnitude_sigma": sigma,
        "duration_requested": duration,
        "start_idx": start_idx,
        "affected_indices": affected,
        "true_value": float(base),
    }
    return df
