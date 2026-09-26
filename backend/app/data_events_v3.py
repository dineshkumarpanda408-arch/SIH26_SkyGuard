"""Realistic sensor-anomaly event patterns (v3 campaign).

These generate the *pattern families* the SIH story needs: many shapes of drift,
frozen sensor and missing data. Each pattern is applied to a REAL v3 weather
sub-frame so patterns are embedded inside genuine physical weather (diurnal
cycles, regimes), never on synthetic straight lines. Ground truth is always
explicit (start/end indices + metadata), never inferred from the model.

Used for:
  1. detector parameter robustness sweeps (does detector X see pattern family Y?)
  2. documentation of the "realistic sensor anomaly data" asset
  3. tests

The production evaluation benchmark itself is NOT built here -- it stays the
unchanged seed-7 injected frame (simulator.inject) so no leakage/tuning occurs.
"""

import numpy as np
import pandas as pd

VARS = ("temperature", "pressure", "humidity")

# ---------------------------------------------------------------------------
# drift mechanisms
# ---------------------------------------------------------------------------

def _make_drift(n, shape, noise=0.02, seed=0):
    rng = np.random.default_rng(seed)
    x = np.arange(n)
    if shape == "slow_pos":
        bias = 0.015 * x
    elif shape == "slow_neg":
        bias = -0.015 * x
    elif shape == "piecewise":
        a = np.zeros(n); a[: n // 3] = 0.008 * x[: n // 3]
        a[n // 3 : 2 * n // 3] = a[n // 3 - 1]
        a[2 * n // 3 :] = a[2 * n // 3 - 1] + 0.02 * (x[2 * n // 3 :] - x[2 * n // 3 - 1])
        bias = a
    elif shape == "accelerating":
        bias = 0.0004 * x * x
    elif shape == "decelerating":
        bias = 3.0 * np.sqrt(np.clip(1.5 * x, 0, None))
        bias = bias / (bias.max() + 1e-9) * n * 0.02
    elif shape == "intermittent":
        bias = np.zeros(n)
        for start, end, rate in ((0, n // 3, 0.02), (n // 2, 2 * n // 3, 0.03)):
            bias[start:end] = rate * (x[start:end] - x[start])
    elif shape == "after_offset":
        bias = np.zeros(n)
        bias[n // 6 :] = -1.5
        bias[n // 6 :] += 0.02 * (x[n // 6 :] - x[n // 6])
    elif shape == "recovery":
        bias = np.zeros(n)
        bias[: n // 2] = 0.03 * x[: n // 2]
        bias[n // 2 :] = bias[n // 2 - 1] - 0.03 * (x[n // 2 :] - x[n // 2 - 1])
    else:  # plain (with noise baked in by caller)
        bias = 0.02 * x
    # amplitude scale so patterns are moderate (sigma units of the variable)
    scale = 0.6 if shape not in ("after_offset", "recovery") else 1.0
    noise_arr = rng.normal(0, noise, n)
    return scale * bias + noise_arr, bias


def _frozen_series(n, kind, seed=0):
    rng = np.random.default_rng(seed)
    val = rng.uniform(0.4, 0.7)
    if kind == "constant":
        return np.full(n, val)
    if kind == "tiny_noise":
        return np.full(n, val) + rng.normal(0, 1e-3, n)
    if kind == "after_jump":
        return np.full(n, val) * 1.08
    # background present but the sensor itself holds flat
    return np.full(n, val)


def _missing_mask(n, kind, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.zeros(n, dtype=bool)
    def span(length, at=None):
        at = at if at is not None else n // 3
        idx[max(0, at) : min(n, at + length)] = True
    if kind == "single":
        span(1)
    elif kind == "short_burst":
        span(4)
    elif kind == "medium_burst":
        span(16)
    elif kind == "long_outage":
        span(48)
    elif kind == "intermittent":
        for s in (n // 4, n // 2, 3 * n // 4):
            span(6, at=s)
    elif kind == "recovery":
        span(20, at=n // 3)
    else:
        span(8)
    return idx


# ---------------------------------------------------------------------------
# pattern catalog -> applied frames with explicit ground truth
# ---------------------------------------------------------------------------

DRIFT_FAMILIES = [
    "slow_pos", "slow_neg", "piecewise", "accelerating", "decelerating",
    "intermittent", "with_noise", "after_offset", "recovery",
    "embedded_in_diurnal", "short", "long",
]

FROZEN_FAMILIES = [
    "short", "medium", "long", "after_jump", "tiny_noise", "daytime", "nighttime", "changing_weather",
]

MISSING_FAMILIES = [
    "single_reading", "short_burst", "medium_burst", "long_outage",
    "single_variable", "two_variable", "comm_gap", "intermittent", "recovery",
]


def build_pattern_frame(base, var, series, truth_start, truth_end, meta):
    """Attach a modified variable + explicit ground truth to a weather frame."""
    out = base.copy()
    out = out.reset_index(drop=True)
    out[var] = series
    out["pattern_truth"] = False
    out.loc[truth_start:truth_end, "pattern_truth"] = True
    out.attrs["pattern_meta"] = meta
    return out


def generate_drift_frame(base, family, seed=0):
    """Return (frame, truth_idx, meta) for a drift pattern on `base`."""
    n = len(base)
    var = "temperature"
    if family == "short":
        n_act = 48
    elif family == "long":
        n_act = 384
    else:
        n_act = n
    truth_start, truth_end = max(8, n // 4), max(8, n // 4) + n_act - 1

    if family in ("slow_pos", "slow_neg", "piecewise", "accelerating",
                  "decelerating", "intermittent", "after_offset", "recovery"):
        shape = family
        bias, _ = _make_drift(n, shape, noise=0.0, seed=seed)
        # monotonically increasing bias on top of real weather
        series = base[var].to_numpy() + bias * base[var].abs().mean() * 0.06
        series = np.clip(series, None, 60)
    elif family == "with_noise":
        bias, _ = _make_drift(n, "slow_pos", noise=0.0, seed=seed)
        rng = np.random.default_rng(seed)
        series = base[var].to_numpy() + bias * base[var].abs().mean() * 0.06 + rng.normal(0, 0.05, n)
    elif family == "embedded_in_diurnal":
        bias, _ = _make_drift(n, "slow_pos", noise=0.0, seed=seed)
        series = base[var].to_numpy() + bias * base[var].abs().mean() * 0.04
    elif family in ("short", "long"):
        bias, _ = _make_drift(n_act, "slow_pos", noise=0.0, seed=seed)
        series = base[var].to_numpy().copy()
        series[truth_start:truth_end + 1] += bias * base[var].abs().mean() * 0.12

    meta = {
        "family": "drift", "mechanism": family, "variable": var,
        "truth_start": truth_start, "truth_end": truth_end,
        "n_truth": int(truth_end - truth_start + 1), "ground_truth": True, "seed": seed,
    }
    return build_pattern_frame(base, var, series, truth_start, truth_end, meta)


def generate_frozen_frame(base, family, seed=0):
    n = len(base)
    var = "temperature"
    if family == "short":
        n_act = 24
    elif family == "medium":
        n_act = 96
    elif family == "long":
        n_act = 240
    else:
        n_act = n // 3
    truth_start, truth_end = max(8, n // 3), max(8, n // 3) + n_act - 1

    rng = np.random.default_rng(seed)
    series = base[var].to_numpy().copy()
    hold = float(np.mean(series[truth_start - 8:truth_start])) if not np.isnan(np.mean(series[truth_start - 8:truth_start])) else float(series[truth_start])
    if family == "after_jump":
        hold = hold * 1.06
    if family in ("daytime", "nighttime"):
        hour = pd.to_datetime(base["timestamp"]).dt.hour.to_numpy()
        hold = 34.0 if family == "daytime" else 18.0
        series[truth_start:truth_end + 1] = hold
    elif family == "changing_weather":
        # background keeps moving, the sensor holds perfectly flat
        series[truth_start:truth_end + 1] = hold
    else:
        series[truth_start:truth_end + 1] = hold + (rng.normal(0, 1e-3, n_act) if family == "tiny_noise" else 0.0)

    meta = {
        "family": "frozen", "mechanism": family, "variable": var,
        "truth_start": truth_start, "truth_end": truth_end,
        "n_truth": int(truth_end - truth_start + 1), "ground_truth": True, "seed": seed,
    }
    return build_pattern_frame(base, var, series, truth_start, truth_end, meta)


def generate_missing_frame(base, family, seed=0):
    n = len(base)
    rng = np.random.default_rng(seed)
    out = base.reset_index(drop=True).copy()
    out["pattern_truth"] = False

    if family == "single_variable":
        mask = _missing_mask(n, "medium_burst", seed)
        out.loc[mask, "humidity"] = np.nan
    elif family == "two_variable":
        mask = _missing_mask(n, "medium_burst", seed)
        out.loc[mask, ["temperature", "humidity"]] = np.nan
    elif family == "comm_gap":
        # genuine timestamp gap: drop 12 consecutive readings
        at = n // 3
        drop = list(range(at, at + 12))
        out = out.drop(index=drop).reset_index(drop=True)
        mask = np.zeros(n, dtype=bool); mask[at:at + 12] = True
    else:
        mask = _missing_mask(n, family, seed)
        out.loc[mask, "humidity"] = np.nan

    truth_idx = np.where(np.asarray(mask, dtype=bool))[0]
    meta = {
        "family": "missing", "mechanism": family, "variable": "humidity",
        "truth_start": int(truth_idx.min()) if len(truth_idx) else -1,
        "truth_end": int(truth_idx.max()) if len(truth_idx) else -1,
        "n_truth": int(len(truth_idx)), "ground_truth": True, "seed": seed,
        "comm_gap": family == "comm_gap",
    }
    out.attrs["pattern_meta"] = meta
    return out


def generate_all(base, seed=3):
    """Generate every pattern family. Returns list of (family, frame, is_multi)."""
    out = []
    for fam in DRIFT_FAMILIES:
        out.append((f"drift/{fam}", generate_drift_frame(base, fam, seed)))
    for fam in FROZEN_FAMILIES:
        out.append((f"frozen/{fam}", generate_frozen_frame(base, fam, seed)))
    for fam in MISSING_FAMILIES:
        out.append((f"missing/{fam}", generate_missing_frame(base, fam, seed)))
    return out