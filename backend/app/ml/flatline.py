"""Dedicated frozen-sensor (flatline) detector.

A real weather sensor has *expected* variability (diurnal + noise). A frozen
sensor's windowed variance collapses far below the station's own normal
variability. This detector compares rolling variance against a robust
per-variable baseline standard deviation learned from TRAIN data only, and
requires PERSISTENCE (a single low-variance window is not a freeze).

Output: per-row [0,1] evidence (higher = more frozen-like),
`computed_by="statistical/flatline"`. Not a calibrated probability.
"""

import numpy as np
import pandas as pd

VARS = ("temperature", "pressure", "humidity")


def rolling_std_causal(s, window):
    """Causal rolling standard deviation (only past+current data)."""
    ser = pd.Series(s).rolling(window, min_periods=1)
    return ser.std().shift(0).to_numpy()


class FlatlineDetector:
    def __init__(self, window=5, ratio_floor=0.40, eps=1e-9):
        self.window = window
        self.ratio_floor = ratio_floor
        self.eps = eps
        self.baseline_std = None  # {var: float} robust normal variability

    # ------------------------------------------------------------------
    def fit(self, df):
        df = df.reset_index(drop=True)
        baseline = {}
        for v in VARS:
            if v not in df.columns:
                continue
            s = df[v].astype(float)
            win_std = rolling_std_causal(s.to_numpy(), self.window)
            ok = np.isfinite(win_std) & (win_std > self.eps)
            # robust: median of the windowed variability + MAD-based guard
            med = float(np.nanmedian(win_std[ok])) if ok.any() else 1.0
            baseline[v] = max(med, self.eps)
        self.baseline_std = baseline
        return self

    # ------------------------------------------------------------------
    def score(self, df):
        """Return per-row frozen evidence in [0,1] (higher = more frozen-like).

        Three causal components combined per variable, all time-aligned:
          * variance ratio : rolling std vs the station's train baseline std
          * zero-difference ratio : fraction of |delta| ~= 0 over the window
          * unique-value penalty : few distinct values over the window
        A short variance dip (natural quiet weather) is NOT enough on its own:
        persistence plus a genuinely *flat* series (zero diffs / few uniques)
        is what separates a stuck sensor from ordinary low-variance weather.
        """
        df = df.reset_index(drop=True)
        n = len(df)
        if self.baseline_std is None:
            return np.zeros(n)

        best = np.zeros(n)
        for v in VARS:
            if v not in df.columns or v not in self.baseline_std:
                continue
            vals = df[v].to_numpy(dtype=float)
            win_std = rolling_std_causal(vals, self.window)
            ratio = win_std / (self.baseline_std[v] + self.eps)
            var_ev = np.clip(1.0 - ratio / self.ratio_floor, 0.0, 1.0)
            var_ev[~np.isfinite(var_ev)] = 0.0

            # zero-difference ratio (causal window)
            delta = np.abs(np.diff(vals, prepend=np.nan))
            zero_ratio = np.array(
                pd.Series((delta <= self.eps).astype(float)).rolling(
                    self.window, min_periods=1
                ).mean().to_numpy(), dtype=float, copy=True
            )
            zero_ratio[~np.isfinite(zero_ratio)] = 0.0

            # unique-value penalty (fewer distinct values -> flatter -> frozen)
            uniq = np.array(
                pd.Series(vals).rolling(self.window, min_periods=1).apply(
                    lambda w: len(np.unique(w[~np.isnan(w)])) / max(int(self.window), 1), raw=True
                ).to_numpy(), dtype=float, copy=True
            )
            uniq[~np.isfinite(uniq)] = 1.0
            uniq_pen = np.clip(1.0 - uniq, 0.0, 1.0)

            ev = 0.45 * var_ev + 0.30 * zero_ratio + 0.25 * uniq_pen

            # persistence: consecutive rows with meaningful freeze evidence
            pers = ev >= 0.3
            run = np.zeros(n)
            cnt = 0
            for i in range(n):
                cnt = cnt + 1 if pers[i] else 0
                run[i] = cnt
            # first new row can't yet be "a plateau", so scale evidence by run
            fused = 0.6 * ev + 0.4 * np.clip(run / 8.0, 0.0, 1.0)
            fused[run < 2] = 0.0  # never flag a 1-sample plateau
            best = np.maximum(best, fused)
        return np.clip(best, 0.0, 1.0)

    def computed_by(self):
        return "statistical/flatline"