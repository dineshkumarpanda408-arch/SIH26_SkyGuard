"""Dedicated spike/drop (rate-of-change) detector.

A real weather series changes gradually: consecutive readings move by a small,
stable amount (diurnal drift + sensor noise). A SPIKE or DROP is an *abrupt,
single-point* jump in one variable that is many multiples larger than that
station's normal per-step change. This detector isolates exactly that signal:
the magnitude of the first-difference in each variable, normalised by the
typical step magnitude learned from TRAIN data only.

It is mechanism-specific in the same way the drift / flatline / missingness
channels are: it produces per-row [0,1] evidence that a point is a rate-of-
change outlier (spike or drop) and is fused into the model score as evidence
for SPIKE / DROP anomalies. It does NOT touch any ML detector.

Score semantics:
  * normal gradual steps  -> ratio <  lo  -> 0 evidence
  * strong step (> hi x typical)          -> 1.0 evidence
  * only {temperature,pressure,humidity}, taking the max across variables,
    so a spike in any one sensor variable is enough to contribute evidence.
  * missing/NaN values and the very first row (no prior) -> 0 evidence
    (a NaN or a boundary is not a spike).
"""

import numpy as np

VARS = ("temperature", "pressure", "humidity")


class RateOfChangeDetector:
    def __init__(self, lo=5.0, hi=10.0, eps=1e-6):
        self.lo = lo
        self.hi = hi
        self.eps = eps
        self.step_scale = None  # {var: typical |x[t]-x[t-1]| learned on TRAIN}

    # ------------------------------------------------------------------
    def fit(self, df):
        df = df.reset_index(drop=True)
        self.step_scale = {}
        for v in VARS:
            if v not in df.columns:
                continue
            s = df[v].astype(float).to_numpy()
            step = np.abs(np.diff(s, prepend=np.nan))
            step = step[np.isfinite(step)]
            self.step_scale[v] = max(float(np.median(step)) if len(step) else 0.0, self.eps)
        return self

    # ------------------------------------------------------------------
    def score(self, df):
        """Return per-row spike/drop evidence in [0,1] (causal first-difference)."""
        df = df.reset_index(drop=True)
        n = len(df)
        if self.step_scale is None:
            return np.zeros(n)
        best = np.zeros(n)
        for v in VARS:
            if v not in df.columns or v not in self.step_scale:
                continue
            s = df[v].astype(float).to_numpy()
            step = np.abs(np.diff(s, prepend=np.nan))
            step[0] = 0.0
            ratio = step / (self.step_scale[v] + self.eps)
            ev = (ratio - self.lo) / max(self.hi - self.lo, 1e-9)
            ev = np.clip(ev, 0.0, 1.0)
            ev[~np.isfinite(ev)] = 0.0
            best = np.maximum(best, ev)
        return best

    def computed_by(self):
        return "statistical/rate-of-change"
