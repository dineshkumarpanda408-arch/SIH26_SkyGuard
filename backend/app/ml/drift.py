"""Dedicated drift detector: expected-weather residual + EWMA + CUSUM.

Key idea (from the SIH improvement plan): detect *degradation* in the
residual  observed - expected_weather  rather than in the raw level, so a
normal evening humidity rise is never called a drift. The expectation is a
causal per-station hour-of-day profile learned from TRAIN data only; nothing
here uses test data or future information.

Output is a per-row [0,1] evidence score (higher = more drift-like) plus a
`computed_by="statistical/drift"` label. Scores are NOT calibrated
probabilities.
"""

import numpy as np
import pandas as pd

VARS = ("temperature", "pressure", "humidity")


def _ewma(x, alpha):
    out = np.zeros(len(x))
    acc = 0.0
    for i in range(len(x)):
        acc = alpha * x[i] + (1 - alpha) * acc
        out[i] = acc
    return out


class DriftDetector:
    def __init__(self, ewma_alpha=0.2, cusum_k=0.55, cusum_h=3.0, slope_win=8, epsilon=1e-6):
        self.ewma_alpha = ewma_alpha
        self.cusum_k = cusum_k
        self.cusum_h = cusum_h
        self.slope_win = slope_win
        self.epsilon = epsilon
        self.profiles = None      # {station_id: {var: [24h means]}}
        self.profiles_global = None  # {var: [24h means]}
        self.resid_std = None     # {var: float}

    # ------------------------------------------------------------------
    def fit(self, df):
        df = df.reset_index(drop=True)
        hours = pd.to_datetime(df["timestamp"]).dt.hour.to_numpy()
        station = df["station_id"].to_numpy()

        profiles = {}
        for sid in np.unique(station):
            m = station == sid
            prof = {}
            for v in VARS:
                vals = df[v].to_numpy(dtype=float)
                hh = hours[m]
                prof[v] = [float(np.nanmean(vals[m][hh == k])) if np.count_nonzero(hh == k) else np.nan
                           for k in range(24)]
            profiles[sid] = prof

        prof_global = {}
        for v in VARS:
            vals = df[v].to_numpy(dtype=float)
            prof_global[v] = [float(np.nanmean(vals[hours == k])) if np.count_nonzero(hours == k) else np.nan
                              for k in range(24)]

        # residual std per variable (using each station's own profile where present)
        resid_std = {}
        for v in VARS:
            vals = df[v].to_numpy(dtype=float)
            exp = np.array([self._exp_val(profiles, prof_global, station[i], hours[i], v) for i in range(len(df))])
            r = vals - exp
            resid_std[v] = float(np.nanstd(r)) + self.epsilon

        self.profiles = profiles
        self.profiles_global = prof_global
        self.resid_std = resid_std
        return self

    def _exp_val(self, profiles, prof_global, sid, hour, v):
        prof = profiles.get(sid, prof_global)
        val = prof.get(v, prof_global[v])[int(hour)]
        if not np.isfinite(val):
            val = prof_global[v][int(hour)]
        if not np.isfinite(val):
            return np.nan
        return float(val)

    # ------------------------------------------------------------------
    def score(self, df):
        """Return per-row drift evidence in [0,1] (higher = more drift-like)."""
        df = df.reset_index(drop=True)
        n = len(df)
        if self.profiles is None:
            return np.zeros(n)
        hours = pd.to_datetime(df["timestamp"]).dt.hour.to_numpy()
        station = df["station_id"].to_numpy() if "station_id" in df.columns else np.full(n, "STATION")

        best = np.zeros(n)
        for v in VARS:
            vals = df[v].to_numpy(dtype=float)
            exp = np.array([self._exp_val(self.profiles, self.profiles_global, station[i], hours[i], v)
                            for i in range(n)])
            r = vals - exp
            with np.errstate(invalid="ignore", divide="ignore"):
                rz = (r - np.nanmedian(r)) / (self.resid_std[v] + self.epsilon)
            rz_n = np.where(np.isfinite(rz), rz, 0.0)

            # EWMA of the standardized residual
            ew = _ewma(rz_n, self.ewma_alpha)
            # CUSUM (Page-Hinkley-style: drift allowance k, threshold h)
            cpos = np.zeros(n); cneg = np.zeros(n)
            cp = cn = 0.0
            for i in range(n):
                cp = max(0.0, cp + (rz_n[i] - self.cusum_k))
                cn = max(0.0, cn + (-rz_n[i] - self.cusum_k))
                cpos[i] = cp; cneg[i] = cn
            cmax = np.maximum(cpos, cneg)

            # rolling slope of the residual
            slope = np.abs(np.convolve(rz_n, np.ones(self.slope_win) / self.slope_win, "same"))
            # persistence: consecutive samples where |ewma| is meaningful
            pers_raw = np.abs(ew) > 0.5
            run = np.zeros(n)
            cnt = 0
            for i in range(n):
                cnt = cnt + 1 if pers_raw[i] else 0
                run[i] = cnt

            ev = (
                0.40 * np.clip(np.abs(ew) / 2.0, 0, 1)
                + 0.35 * np.clip(cmax / self.cusum_h, 0, 1) * 0.6
                + 0.25 * np.clip(np.abs(slope) / 2.0, 0, 1)
            )
            ev = ev * np.clip(run / 2.0, 0, 1)  # require persistence before strong evidence
            best = np.maximum(best, ev)
        return np.clip(best, 0.0, 1.0)

    def computed_by(self):
        return "statistical/drift"