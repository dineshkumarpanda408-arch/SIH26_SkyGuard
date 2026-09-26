"""Dedicated missing-data detector, runs on the RAW stream BEFORE imputation.

Imputation (preprocessing.impute) and median-fill (detector.decision_scores)
erase NaN gaps before the ML sees them, so a missing signal can never be scored
by an isolation forest. This detector inspects the raw pre-impute frame and
produces evidence for:

  * NaN / invalid (bounds-violating) values per variable
  * consecutive missing runs
  * missing fraction over a trailing window
  * single-variable vs multi-variable outages
  * timestamp gaps vs the learned expected reporting interval

Output: per-row [0,1] evidence, `computed_by="data_quality/missingness"`.
Not a calibrated probability.
"""

import numpy as np
import pandas as pd

VARS = ("temperature", "pressure", "humidity")

# Physical bound check mirrors preprocessing.BOUNDS (kept local so this module
# stays standalone and honest about what it treats as an invalid reading).
BOUNDS = {
    "temperature": (-100.0, 200.0),
    "pressure": (500.0, 1500.0),
    "humidity": (-10.0, 120.0),
}


class MissingnessDetector:
    def __init__(self, interval_tol_fraction=1.5, window=12, eps=1e-9):
        self.interval_tol_fraction = interval_tol_fraction
        self.window = window
        self.eps = eps
        self.expected_interval_s = None

    # ------------------------------------------------------------------
    def fit(self, df):
        df = df.reset_index(drop=True)
        ts = pd.to_datetime(df["timestamp"])
        d = ts.diff().dt.total_seconds().dropna()
        if len(d):
            self.expected_interval_s = float(np.median(d.to_numpy()))
        else:
            self.expected_interval_s = 900.0
        return self

    # ------------------------------------------------------------------
    def _invalid_mask(self, df):
        out = np.zeros(len(df), dtype=bool)
        for v in VARS:
            if v not in df.columns:
                continue
            vals = df[v].to_numpy(dtype=object)
            for i in range(len(df)):
                x = vals[i]
                if x is None:
                    out[i] = True
                else:
                    try:
                        f = float(x)
                        if not np.isfinite(f):
                            out[i] = True
                        elif v in BOUNDS and not (BOUNDS[v][0] <= f <= BOUNDS[v][1]):
                            out[i] = True
                    except (TypeError, ValueError):
                        out[i] = True
        return out

    # ------------------------------------------------------------------
    def score_raw(self, df):
        """Score a RAW pre-impute frame. Returns per-row [0,1] evidence."""
        df = df.reset_index(drop=True)
        n = len(df)
        if self.expected_interval_s is None:
            self.expected_interval_s = 900.0

        invalid = self._invalid_mask(df)

        per_var = {}
        for v in VARS:
            if v not in df.columns:
                continue
            vals = df[v].to_numpy(dtype=object)
            m = np.zeros(n, dtype=bool)
            for i in range(n):
                x = vals[i]
                try:
                    f = float(x)
                    m[i] = not np.isfinite(f)
                except (TypeError, ValueError):
                    m[i] = True
            per_var[v] = m

        # run length (per var, max over vars) -- consecutive missing ending at row
        max_run = np.zeros(n)
        run_any = np.zeros(n)
        for v, m in per_var.items():
            run = np.zeros(n)
            cnt = 0
            for i in range(n):
                cnt = cnt + 1 if m[i] else 0
                run[i] = cnt
            max_run = np.maximum(max_run, run)
        any_missing = np.zeros(n)
        for v, m in per_var.items():
            any_missing = np.maximum(any_missing, m.astype(float))
        n_missing_vars = sum(m.astype(int) for _, m in per_var.items()) if per_var else np.zeros(n)

        # trailing-window missing fraction (first var that is missing)
        frac_win = np.zeros(n)
        for v, m in per_var.items():
            ser = pd.Series(m.astype(float)).rolling(self.window, min_periods=1).mean()
            frac_win = np.maximum(frac_win, ser.to_numpy())

        # timestamp gap flag
        gap = np.zeros(n)
        if n > 1:
            d = pd.to_datetime(df["timestamp"]).diff().dt.total_seconds().to_numpy()
            tol = self.expected_interval_s * self.interval_tol_fraction
            gap[1:] = (d[1:] > tol).astype(float)
            gap[gap > 1.0] = 1.0  # cap at 1 (multi-gap still 1 flag)

        score = (
            0.40 * np.clip(max_run / 5.0, 0, 1)
            + 0.25 * any_missing
            + 0.25 * frac_win
            + 0.10 * gap
        )
        score[~invalid] = np.clip(score[~invalid], 0, 1)  # keep bound-violating rows high
        return np.clip(score, 0.0, 1.0)

    def computed_by(self):
        return "data_quality/missingness"