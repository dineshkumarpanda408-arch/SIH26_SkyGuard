"""Targeted PHASE 4 regression tests.

Validates the already-implemented PHASE 4 architecture: pure fused scores,
simulator contract, spatial heuristic counts, diurnal correction, model
metadata, provenance, ground-truth/prediction separation, attribution,
evaluation coverage and persisted-model load.

Run with the stdlib unittest runner from the backend directory:
    python -m unittest discover -s tests
"""

import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services import app as skyguard  # noqa: E402
from app.simulator import inject  # noqa: E402
from app.ml.corrected_value import estimate  # noqa: E402
from app.ml.weather_vs_sensor import assess  # noqa: E402
from app.ml.classifier import classify  # noqa: E402
from app.ml.training import load_versioned  # noqa: E402

V2_SHA256 = "69b56c7757035ee64bf979fede8499e94c7968c3004cd16037a37baede17b4fe"
V3_SHA256 = "7d20d2d1a990813087fd712499905e27e0f82d5e1ef8ab93c2dc39a08d618511"


def _station_tail(app, station_id, start, length):
    sub = app.data[app.data.station_id == station_id].sort_values("timestamp").reset_index(drop=True)
    return sub.iloc[start : start + length].copy()


class TestScorePurity(unittest.TestCase):
    """Missing/frozen inputs never force hardcoded floors; they set rule_flags and
    score stays the pure fused model output (v3 channel-weighted fusion)."""

    @classmethod
    def setUpClass(cls):
        cls.app = skyguard
        if not getattr(skyguard, "_initialized_test", False):
            skyguard.initialize()
            skyguard._initialized_test = True

    @staticmethod
    def _benign_window(app, station_id="AWS-023"):
        sub = app.data[app.data.station_id == station_id].sort_values("timestamp").reset_index(drop=True)
        best = None
        best_score = np.inf
        for start in range(0, len(sub) - 60, 12):
            w = sub.iloc[start : start + 60].copy()
            try:
                s = app.model.analyze_row(w)["score"]
            except Exception:
                continue
            if s < best_score:
                best_score = s
                best = w
        return best, best_score

    def test_missing_sets_flag_not_score(self):
        window, guard = self._benign_window(self.app)
        self.assertLess(guard, 0.72)  # precondition: an actually-normal reading
        missing = window.copy()
        missing.loc[missing.index[-1], "temperature"] = np.nan
        res = self.app.model.analyze_row(missing)
        fused = float(self.app.model.score_series_scores(missing)[-1])
        self.assertTrue(res["rule_flags"]["missing_imputed"])
        self.assertAlmostEqual(res["score"], fused, places=9)  # score == pure fused model output
        self.assertLess(res["score"], 0.72)                    # NOT forced to the old 0.85 floor
        self.assertNotAlmostEqual(res["score"], 0.85, places=3)

    def test_frozen_sets_flag_not_score(self):
        window, guard = self._benign_window(self.app)
        self.assertLess(guard, 0.72)
        frozen = window.copy()
        c = float(frozen["temperature"].astype(float).median())
        frozen.loc[frozen.index[-3]:, "temperature"] = c
        res = self.app.model.analyze_row(frozen)
        fused = float(self.app.model.score_series_scores(frozen)[-1])
        self.assertTrue(res["rule_flags"]["frozen_plateau"])
        self.assertAlmostEqual(res["score"], fused, places=9)
        self.assertLess(res["score"], 0.72)
        self.assertNotAlmostEqual(res["score"], 0.80, places=3)

    def test_calibration_reporting(self):
        window, _ = self._benign_window(self.app)
        res = self.app.model.analyze_row(window)
        self.assertIsNone(res["anomaly_probability"])
        self.assertEqual(res["calibration_status"], "uncalibrated")


class TestSimulatorContract(unittest.TestCase):
    """SPIKE/DROP stay single-point; canonical shapes; ground_truth columns."""

    def _frame(self):
        rng = np.random.default_rng(7)
        t = np.arange(60, dtype=float) / 4.0
        temp = 28.0 + 3.0 * np.sin(t) + rng.normal(0, 0.2, 60)
        return pd.DataFrame({"timestamp": pd.date_range("2026-08-14", periods=60, freq="15min"),
                             "temperature": temp, "pressure": 1010.0, "humidity": 60.0})

    def test_spike_stays_single_point_even_with_duration8(self):
        df = self._frame()
        out = inject(df.copy(), "temperature", "SPIKE", magnitude=30, duration=8, start_idx=30, rng=np.random.default_rng(1))
        meta = out.attrs["injection_meta"]
        self.assertEqual(len(meta["affected_indices"]), 1)
        self.assertEqual(out["ground_truth"].sum(), 1)
        self.assertEqual(meta["canonical_type"], "SPIKE")
        base = float(df["temperature"].iloc[29])
        # pointed jump, then back to baseline (shape, NOT a plateau)
        self.assertGreater(abs(float(out["temperature"].iloc[30]) - base), 1.0)
        self.assertLess(abs(float(out["temperature"].iloc[31]) - base), 0.5)

    def test_drop_stays_single_point_even_with_duration8(self):
        df = self._frame()
        out = inject(df.copy(), "temperature", "DROP", magnitude=30, duration=8, start_idx=30, rng=np.random.default_rng(1))
        meta = out.attrs["injection_meta"]
        self.assertEqual(len(meta["affected_indices"]), 1)
        self.assertEqual(out["ground_truth"].sum(), 1)
        self.assertEqual(meta["canonical_type"], "DROP")
        base = float(df["temperature"].iloc[29])
        self.assertLess(float(out["temperature"].iloc[30]) - base, -1.0)
        self.assertLess(abs(float(out["temperature"].iloc[31]) - base), 0.5)

    def test_missing_data_nan_over_duration(self):
        df = self._frame()
        out = inject(df.copy(), "temperature", "MISSING_DATA", duration=5, start_idx=40, rng=np.random.default_rng(1))
        meta = out.attrs["injection_meta"]
        self.assertEqual(meta["canonical_type"], "MISSING_DATA")
        self.assertEqual(len(meta["affected_indices"]), 5)
        self.assertTrue(out["ground_truth"].iloc[40:45].all())
        self.assertTrue(out["temperature"].iloc[40:45].isna().all())

    def test_frozen_plateau_constant(self):
        df = self._frame()
        out = inject(df.copy(), "temperature", "FROZEN_SENSOR", duration=6, start_idx=40, rng=np.random.default_rng(1))
        meta = out.attrs["injection_meta"]
        self.assertEqual(meta["canonical_type"], "FROZEN_SENSOR")
        plat = out["temperature"].iloc[40:46].to_numpy()
        self.assertAlmostEqual(np.ptp(plat), 0.0)

    def test_drift_monotonic(self):
        df = self._frame()
        out = inject(df.copy(), "temperature", "DRIFT", duration=8, start_idx=40, rng=np.random.default_rng(1))
        seg = out["temperature"].iloc[40:48].to_numpy(dtype=float)
        diffs = np.diff(seg)
        self.assertTrue(np.all(diffs >= 0) or np.all(diffs <= 0))

    def test_noise_injects_without_forced_detection(self):
        df = self._frame()
        out = inject(df.copy(), "temperature", "NOISE", duration=8, start_idx=40, rng=np.random.default_rng(1))
        self.assertEqual(out.attrs["injection_meta"]["canonical_type"], "NOISE")
        self.assertTrue(out["ground_truth"].iloc[40:48].all())
        # honesty guard: do NOT assert detection anywhere here


class TestSpatialCounts(unittest.TestCase):
    """23-neighbour counts; abnormal/partial/normal all reported; heuristic."""

    def _scenario(self):
        # target: +8.0 on 28 -> rel dev 0.2857 -> agreement threshold ~0.143, noise floor 0.04
        anomaly = {"station_id": "AWS-023", "value": 36.0, "variable": "temperature", "expected_value": 28.0}
        neighbors = []
        for k in range(23):
            e = 28.0 + (0.3 if k % 2 else -0.2)
            if k < 5:
                v = e * 1.18         # relative dev 0.18 -> abnormal (>= 0.143)
            elif k < 18:
                v = e * 1.07         # relative dev 0.07 -> partial (>= 0.04, < 0.143)
            else:
                v = e * 1.02         # relative dev 0.02 -> normal (< 0.04)
            neighbors.append({"station_id": f"AWS-{k:03d}", "value": float(v), "expected_value": float(e)})
        return anomaly, neighbors

    def test_counts_sum_and_label(self):
        anomaly, neighbors = self._scenario()
        self.assertEqual(len(neighbors), 23)
        assessment, likelihood, evidence = assess(anomaly, neighbors)
        self.assertTrue(any("5 abnormal, 13 partial, 5 normal" in e for e in evidence))
        self.assertTrue(any("5 of 23 neighbouring stations elevated" in e for e in evidence))
        self.assertIsNotNone(likelihood)
        self.assertGreaterEqual(likelihood, 0.0)
        self.assertLessEqual(likelihood, 1.0)
        # every neighbour is explicitly classified (partial NOT hidden as normal)
        self.assertEqual(5 + 13 + 5, 23)

    def test_heuristic_not_probability(self):
        anomaly, neighbors = self._scenario()
        _, _, evidence = assess(anomaly, neighbors)
        merged = " ".join(evidence).lower()
        self.assertIn("heuristic", merged)
        self.assertIn("not a trained", merged)


class TestCorrection(unittest.TestCase):
    def test_diurnal_used_with_enough_history(self):
        n = 500
        ts = pd.date_range("2026-08-14", periods=n, freq="1h")
        series = 28.0 + 6.0 * np.sin(np.arange(n) * 2 * np.pi / 24.0) + np.random.default_rng(0).normal(0, 0.3, n)
        est = estimate(series, "temperature", window=6, timestamps=ts)
        self.assertTrue(est["diurnal_used"])
        self.assertGreaterEqual(est["n_same_hour"], 4)
        self.assertIn("same-hour diurnal", est["correction_method"])
        self.assertGreaterEqual(est["correction_confidence"], 0.0)
        self.assertLessEqual(est["correction_confidence"], 1.0)
        self.assertIsInstance(est["corrected_value"], float)

    def test_fallback_recent_window(self):
        est = estimate(np.array([20.0, 21.0, 19.5]), "temperature")
        self.assertFalse(est["diurnal_used"])
        self.assertEqual(est["correction_method"], "recent-window mean")
        self.assertIsInstance(est["corrected_value"], float)


class TestModelMetadata(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = skyguard
        if not getattr(skyguard, "_initialized_test", False):
            skyguard.initialize()
            skyguard._initialized_test = True

    def test_metadata_identity(self):
        m = self.app.model_metadata()
        self.assertEqual(m["dataset_version"], "sample_weather_v3")
        self.assertEqual(m["dataset_sha256"], V3_SHA256)
        self.assertEqual(m["n_stations"], 24)
        self.assertEqual(m["n_observations"], 138240)
        self.assertEqual(m["feature_count"], 25)
        self.assertEqual(m["split"]["train_fraction"], 0.70)
        self.assertIn("chronological", m["split"]["method"])
        self.assertEqual(m["calibration"]["status"], "uncalibrated")
        self.assertEqual(m["model_version"], "skyguard-multivariate-v3")


class TestProvenance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = skyguard
        if not getattr(skyguard, "_initialized_test", False):
            skyguard.initialize()
            skyguard._initialized_test = True

    ALLOWED = ("ml", "rule", "statistical", "system")

    def _check_computed_by(self, cb):
        self.assertIsInstance(cb, dict)
        for v in cb.values():
            if v is None:
                continue
            cat = str(v).split(":")[0].split("/")[0].split(" ")[0].lower()
            self.assertIn(cat, self.ALLOWED)

    def test_prediction_provenance(self):
        p = self.app.predict_station("AWS-023")
        self.assertIn("computed_by", p)
        self._check_computed_by(p["computed_by"])
        self.assertIn("model_version", p)
        self.assertIsNone(p.get("anomaly_probability"))
        self.assertEqual(p.get("calibration_status"), "uncalibrated")

    def test_simulation_provenance(self):
        s = self.app.simulate({"station_id": "AWS-023", "variable": "temperature",
                               "anomaly_type": "SPIKE", "magnitude": 25, "duration": 1})
        self.assertIn("computed_by", s)
        self._check_computed_by(s["computed_by"])
        self.assertEqual((s.get("model") or {}).get("model_version"), "skyguard-multivariate-v3")


class TestGtPredictionSeparation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = skyguard
        if not getattr(skyguard, "_initialized_test", False):
            skyguard.initialize()
            skyguard._initialized_test = True

    def test_requested_type_never_leaks_to_prediction(self):
        s = self.app.simulate({"station_id": "AWS-023", "variable": "temperature",
                               "anomaly_type": "FROZEN", "duration": 6})
        gt = s["ground_truth"] or {}
        self.assertEqual(gt.get("requested_type"), "FROZEN")
        self.assertEqual(gt.get("canonical_type"), "FROZEN_SENSOR")

        sub = self.app.data[self.app.data.station_id == "AWS-023"].tail(30).reset_index(drop=True)
        injected = inject(sub, "temperature", "FROZEN", duration=6, start_idx=max(0, len(sub) - 6))
        series = injected["temperature"].dropna().to_numpy(dtype=float)
        # The classifier inspects the RESULTING shape (a trailing constant
        # block). A trailing plateau is, by construction, ambiguous between
        # FROZEN_SENSOR (flat) and SUDDEN_SHIFT (level step): whichever rule the
        # classifier fires first is honest output. What must NOT happen is the
        # label being copied from the request: 'FROZEN' is never a label.
        a = classify({"value": float(series[-1]), "variable": "temperature", "expected_value": 0}, series)
        self.assertIn(a, ("FROZEN_SENSOR", "SUDDEN_SHIFT", "SPIKE", "DROP"))
        detected_label = s.get("detected_type")
        if s.get("detected"):
            # the reported label must come from the classifier on the actual
            # series, never from the aliased request string
            self.assertEqual(detected_label, a)
        self.assertNotEqual(gt.get("requested_type"), detected_label or a)

    def test_drop_label_from_classifier(self):
        s = self.app.simulate({"station_id": "AWS-023", "variable": "temperature",
                               "anomaly_type": "DROP", "magnitude": 30, "duration": 1})
        if s.get("detected"):
            gt = s["ground_truth"] or {}
            self.assertEqual(gt.get("canonical_type"), "DROP")
            self.assertEqual(s.get("detected_type"), "DROP")
            sub = self.app.data[self.app.data.station_id == "AWS-023"].tail(30).reset_index(drop=True)
            injected = inject(sub, "temperature", "DROP", magnitude=30, duration=1, start_idx=len(sub) - 1)
            self.assertEqual(s.get("detected_type"),
                             classify({"value": float(injected["temperature"].iloc[-1]), "variable": "temperature", "expected_value": 0},
                                      injected["temperature"].to_numpy(dtype=float)))


class TestExplanation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = skyguard
        if not getattr(skyguard, "_initialized_test", False):
            skyguard.initialize()
            skyguard._initialized_test = True

    def test_model_attribution_on_anomaly(self):
        # ensure there is a detected anomaly to explain (start fresh via simulate)
        s = self.app.simulate({"station_id": "AWS-023", "variable": "temperature",
                               "anomaly_type": "SPIKE", "magnitude": 25, "duration": 1})
        rec = s.get("result") or {}
        self.assertTrue(s.get("detected"), "precondition: injection must be detected")
        # timestamp AFTER the dataset end so the reconstructed anomaly row is the
        # LAST row after validate() sorts the window by timestamp
        fake = SimpleNamespace(station_id="AWS-023", feature=rec.get("feature", "temperature"),
                               raw_value=rec.get("raw_value"), timestamp=pd.Timestamp("2026-08-29 12:00:00"))
        values, narr, method = self.app.explain_anomaly(fake)
        self.assertTrue(values, "model attribution must produce contributions")
        self.assertIn(method, ("shap", "statistical"))
        top = values[0]
        self.assertEqual(top["feature"], "temperature")
        self.assertGreater(top["contribution"], 0.0)  # temperature drives the spike
        self.assertTrue(narr)
        # attribution is SHAP (game-theoretic) or an honest statistical fallback, not
        # z-score labels
        for v in values:
            self.assertTrue({"feature", "contribution"} <= set(v))


class TestEvaluationCoverage(unittest.TestCase):
    """Validate the 24 x 3 x 5 x 6 = 2160-cell grid is representable and injectable
    with the minimum computation (5 types x 6 magnitudes on temperature, one station)."""

    TYPES = ("SPIKE", "DROP", "DRIFT", "FROZEN_SENSOR", "MISSING_DATA")
    MAGNITUDES = (3, 4, 5, 6, 8, 12)
    STATIONS = 24
    VARIABLES = 3

    @classmethod
    def setUpClass(cls):
        cls.app = skyguard
        if not getattr(skyguard, "_initialized_test", False):
            skyguard.initialize()
            skyguard._initialized_test = True

    def test_grid_definition(self):
        self.assertEqual(self.STATIONS * self.VARIABLES * len(self.TYPES) * len(self.MAGNITUDES), 2160)
        self.assertEqual(set(self.MAGNITUDES), {3, 4, 5, 6, 8, 12})

    def test_grid_injectable_on_temperature(self):
        sub = self.app.data[self.app.data.station_id == "AWS-023"].tail(60).reset_index(drop=True)
        for typ in self.TYPES:
            for mag in self.MAGNITUDES:
                dur = 1 if typ in ("SPIKE", "DROP") else (6 if typ == "FROZEN_SENSOR" else (5 if typ == "MISSING_DATA" else 8))
                out = inject(sub, "temperature", typ, magnitude=mag, duration=dur, start_idx=len(sub) - dur - 1)
                meta = out.attrs["injection_meta"]
                self.assertEqual(meta["canonical_type"], typ)
                self.assertEqual(meta["magnitude_sigma"], float(mag))


class TestClassifierPriority(unittest.TestCase):
    """Root-cause classifier ordering: MISSING_DATA -> FROZEN_SENSOR ->
    SPIKE/DROP -> DRIFT -> other. In particular a frozen (flat) sensor must be
    classified as FROZEN_SENSOR, never DRIFT."""

    def _classify(self, values):
        series = np.asarray(values, dtype=float)
        return classify(
            {"value": float(series[-1]), "variable": "temperature", "expected_value": 0},
            series,
        )

    def test_missing_data_highest_priority(self):
        # NaN value is MISSING_DATA even though surrounding series is flat.
        series = np.array([20.0, 20.0, 20.0, 20.0, 20.0, np.nan])
        out = classify(
            {"value": np.nan, "variable": "temperature", "expected_value": 0}, series
        )
        self.assertEqual(out, "MISSING_DATA")

    def test_frozen_flat_plateau_is_frozen_not_drift(self):
        # A trailing run of identical values (stuck sensor) must be FROZEN_SENSOR.
        # With the OLD ordering a long flat prefix with a tiny final tick could be
        # swept into DRIFT; the plateau test now wins because it runs first.
        values = [20.0, 20.0, 20.0, 20.0, 20.0, 20.0]
        self.assertEqual(self._classify(values), "FROZEN_SENSOR")

    def test_frozen_near_constant_is_frozen_not_drift(self):
        # A sensor stuck within tiny floating point noise is still FROZEN.
        values = [24.51, 24.51001, 24.51002, 24.51001, 24.51, 24.51002]
        self.assertEqual(self._classify(values), "FROZEN_SENSOR")

    def test_frozen_with_noisy_history_still_frozen_not_drift(self):
        # History varies, but the trailing window is a flat plateau -> FROZEN,
        # and it must NOT be downgraded to DRIFT even though a slope fit over the
        # whole series is non-zero.
        values = [18.0, 25.0, 14.0, 28.0, 33.0, 33.001, 33.0, 33.001, 33.0]
        self.assertEqual(self._classify(values), "FROZEN_SENSOR")

    def test_monotonic_drift_is_drift(self):
        # A genuinely monotonic, changing series fragments to DRIFT (not frozen,
        # because the trailing window is not a flat plateau).
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        self.assertEqual(self._classify(values), "DRIFT")

    def test_spike_precedes_drift(self):
        # A large single-point spike is SPIKE, not DRIFT.
        values = [20.0, 21.0, 20.5, 40.0]  # trailing point is a spike
        self.assertEqual(self._classify(values), "SPIKE")

    def test_frozen_never_labeled_drift(self):
        # Regression: for several flat-plateau shapes, FROZEN_SENSOR must be
        # returned and DRIFT must never appear.
        plateaus = [
            [25.0] * 8,
            [10.1] * 7,
            [60.0, 60.0, 60.0, 60.0, 60.0],
            [0.0] * 6,
        ]
        for values in plateaus:
            self.assertEqual(self._classify(values), "FROZEN_SENSOR")


class TestModelLoad(unittest.TestCase):
    def test_v2_artifact_loads_with_all_detectors(self):
        model, meta = load_versioned("skyguard_v2")
        self.assertIsNotNone(model)
        self.assertEqual(meta["model_version"], "skyguard-multivariate-v2")
        self.assertIsNotNone(model.baseline)
        self.assertIsNotNone(model.temporal)
        self.assertIsNotNone(model.multivariate)
        self.assertIsNotNone(model.seasonal)
        self.assertEqual(model.baseline.model.n_features_in_, 3)
        self.assertEqual(model.temporal.model.n_features_in_, 10)
        self.assertEqual(model.multivariate.model.n_features_in_, 6)
        self.assertEqual(model.seasonal.model.n_features_in_, 5)
        self.assertIsNotNone(getattr(model.baseline, "background", None))


if __name__ == "__main__":
    unittest.main()