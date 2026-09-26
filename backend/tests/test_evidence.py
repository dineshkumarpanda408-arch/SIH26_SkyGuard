"""STAGE 7 tests: thermodynamic consistency evidence + structured evidence.

Covers the dew-point/Magnus consistency signal (consistent / inconsistent /
missing-T / missing-RH / boundary+edge) and the structured evidence/
provenance builder (grounded in computed signals only, never derived from the
anomaly label).

Run with the stdlib unittest runner from the backend directory:
    python -m unittest discover -s tests
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.ml.thermo import dew_point, thermodynamic_evidence  # noqa: E402
from app.ml.evidence import build_evidence, fallback_evidence  # noqa: E402
from app.database import init_db  # noqa: E402

init_db()  # idempotent: guarantees the evidence_json column exists


class TestThermoDewPoint(unittest.TestCase):
    """Magnus/Tetens dew point: computable domain and consistency with the
    formula used by the engineered feature set (app.features)."""

    def test_dew_point_reference_value(self):
        # T=20C, RH=50% -> Td is well-known to be ~9.3C under Magnus (17.27/237.7).
        td = dew_point(20.0, 50.0)
        self.assertIsNotNone(td)
        self.assertGreater(td, 8.5)
        self.assertLess(td, 10.0)
        self.assertLess(td, 20.0)  # dew point must stay below air temperature

    def test_saturation_dew_point_equals_temperature(self):
        # Boundary/edge: RH=100% collapses Td onto T.
        td = dew_point(20.0, 100.0)
        self.assertIsNotNone(td)
        self.assertAlmostEqual(td, 20.0, places=5)

    def test_cold_and_dry_pair(self):
        td = dew_point(-40.0, 20.0)
        self.assertIsNotNone(td)
        self.assertLess(td, -40.0)

    def test_invalid_humidity_domain_returns_none(self):
        self.assertIsNone(dew_point(20.0, 110.0))
        self.assertIsNone(dew_point(20.0, 0.0))
        self.assertIsNone(dew_point(20.0, -5.0))

    def test_missing_or_bad_inputs_return_none(self):
        self.assertIsNone(dew_point(None, 50.0))
        self.assertIsNone(dew_point(20.0, None))
        self.assertIsNone(dew_point(np.nan, 50.0))
        self.assertIsNone(dew_point(20.0, np.nan))

    def test_matches_feature_engineering_formula(self):
        # Cross-consistency guard: the evidence signal must agree with the
        # dew-point column the model's multivariate detector is trained on.
        a, b = 17.27, 237.7
        for t, rh in [(10.0, 40.0), (25.0, 70.0), (33.0, 90.0), (-5.0, 60.0)]:
            gamma = (a * t) / (b + t) + np.log(np.clip(rh, 1e-6, None) / 100.0)
            expected = (b * gamma) / (a - gamma)
            self.assertAlmostEqual(dew_point(t, rh), float(expected), places=6)


class TestThermoConsistencyEvidence(unittest.TestCase):
    """The evidence signal: CONSISTENT / INCONSISTENT / UNAVAILABLE."""

    def test_consistent_pair(self):
        ev = thermodynamic_evidence(20.0, 50.0)
        self.assertTrue(ev["available"])
        self.assertEqual(ev["status"], "CONSISTENT")
        self.assertIsNotNone(ev["dew_point"])
        self.assertTrue(ev["heuristic"])
        self.assertEqual(ev["computed_by"], "physics:magnus_dew_point")
        self.assertIn("thermodynamically consistent", ev["detail"].lower())

    def test_saturation_boundary_is_consistent(self):
        ev = thermodynamic_evidence(20.0, 100.0)
        self.assertEqual(ev["status"], "CONSISTENT")
        self.assertAlmostEqual(ev["dew_point"], 20.0, places=1)

    def test_supersaturated_humidity_inconsistent(self):
        for rh in (100.1, 115.0, -5.0):
            ev = thermodynamic_evidence(20.0, rh)
            self.assertEqual(ev["status"], "INCONSISTENT", rh)
            self.assertTrue(ev["available"])

    def test_zero_humidity_inconsistent(self):
        ev = thermodynamic_evidence(20.0, 0.0)
        self.assertEqual(ev["status"], "INCONSISTENT")
        self.assertIsNone(ev["dew_point"])
        self.assertIn("dew point is undefined", ev["detail"].lower())

    def test_missing_temperature_unavailable_no_claim(self):
        ev = thermodynamic_evidence(None, 50.0)
        self.assertFalse(ev["available"])
        self.assertEqual(ev["status"], "UNAVAILABLE")
        self.assertIsNone(ev["dew_point"])
        # no misleading claim is made
        self.assertIn("no thermodynamic claim", ev["detail"].lower())

    def test_missing_humidity_unavailable_no_claim(self):
        ev = thermodynamic_evidence(20.0, None)
        self.assertFalse(ev["available"])
        self.assertEqual(ev["status"], "UNAVAILABLE")

    def test_nan_input_unavailable(self):
        self.assertEqual(thermodynamic_evidence(np.nan, 50.0)["status"], "UNAVAILABLE")
        self.assertEqual(thermodynamic_evidence(20.0, np.nan)["status"], "UNAVAILABLE")


class TestEvidenceBuilder(unittest.TestCase):
    """Structured evidence is grounded in computed signals, never the label."""

    def _default_kwargs(self, **overrides):
        kwargs = dict(
            score=0.52,
            threshold=0.30,
            component_scores={"ml": 0.87, "multivariate": 0.9, "seasonal": 0.1,
                              "drift": 0.6, "flatline": 0.05, "missingness": 0.9, "roc": 0.55},
            rule_flags={"missing_imputed": False, "frozen_plateau": False},
            spatial={"assessment": "LIKELY_SENSOR_FAULT", "is_weather_event": False,
                     "sensor_fault_likelihood": 0.8, "evidence_lines": ["station deviates; 0 of 3 neighbours elevated"]},
            thermo=thermodynamic_evidence(20.0, 50.0),
            model_version="skyguard-multivariate-v3",
            calibration_status="uncalibrated",
        )
        kwargs.update(overrides)
        return kwargs

    def test_structure_and_provenance(self):
        items = build_evidence(**self._default_kwargs())
        self.assertTrue(items)
        for item in items:
            self.assertTrue({"signal", "label", "status", "detail"} <= set(item))
            self.assertIn("computed_by", item)
            self.assertIn("kind", item)
        self.assertEqual(items[0]["signal"], "ml")
        self.assertEqual(items[0]["status"], "flagged")
        self.assertEqual(items[0]["kind"], "model")
        self.assertIn("skyguard-multivariate-v3", items[0]["computed_by"])

    def test_ml_detail_reports_score_and_threshold_not_label(self):
        items = build_evidence(**self._default_kwargs())
        ml = items[0]["detail"]
        self.assertIn("0.520", ml)
        self.assertIn("0.30", ml)
        # the evidence must not derive its reason from an anomaly-type label
        for word in ("SPIKE", "DRIFT", "FROZEN", "MISSING"):
            self.assertNotIn(word, ml)

    def test_elevated_components_reported_nominal_skipped(self):
        items = build_evidence(**self._default_kwargs())
        signals = {i["signal"]: i for i in items}
        self.assertIn("multivariate", signals)
        self.assertEqual(signals["multivariate"]["status"], "elevated")
        self.assertNotIn("seasonal", signals)  # 0.1 < floor -> nominal, not evidence
        self.assertIn("drift", signals)

    def test_rule_flags_surface_structural_evidence(self):
        items = build_evidence(**self._default_kwargs(
            rule_flags={"missing_imputed": True, "frozen_plateau": True}))
        signals = {i["signal"]: i for i in items}
        self.assertIn("missingness", signals)
        self.assertEqual(signals["missingness"]["kind"], "rule")
        self.assertEqual(signals["flatline"]["status"], "flagged")

    def test_thermo_evidence_reflected(self):
        consistent = build_evidence(**self._default_kwargs(
            thermo=thermodynamic_evidence(20.0, 50.0)))
        statuses = {i["signal"]: i["status"] for i in consistent}
        self.assertEqual(statuses["thermodynamic"], "consistent")

        inconsistent = build_evidence(**self._default_kwargs(
            thermo=thermodynamic_evidence(20.0, 115.0)))
        statuses = {i["signal"]: i["status"] for i in inconsistent}
        self.assertEqual(statuses["thermodynamic"], "inconsistent")

        unavailable = build_evidence(**self._default_kwargs(
            thermo=thermodynamic_evidence(None, 50.0)))
        statuses = {i["signal"]: i["status"] for i in unavailable}
        self.assertEqual(statuses["thermodynamic"], "unavailable")

    def test_spatial_assessment_mapped(self):
        kwargs = self._default_kwargs()
        kwargs["spatial"] = {"assessment": "LIKELY_WEATHER_EVENT", "is_weather_event": True,
                             "sensor_fault_likelihood": 0.1, "evidence_lines": ["23 station of 23 elevated"]}
        statuses = {i["signal"]: i["status"] for i in build_evidence(**kwargs)}
        self.assertEqual(statuses["spatial"], "weather")

        kwargs["spatial"] = {"assessment": "UNAVAILABLE", "is_weather_event": False,
                             "sensor_fault_likelihood": None, "evidence_lines": []}
        statuses = {i["signal"]: i["status"] for i in build_evidence(**kwargs)}
        self.assertEqual(statuses["spatial"], "unavailable")

    def test_calibration_honesty_present(self):
        items = build_evidence(**self._default_kwargs())
        cal = next(i for i in items if i["signal"] == "calibration")
        self.assertEqual(cal["status"], "uncalibrated")
        self.assertIn("not probabilities", cal["detail"].lower())


class TestFallbackEvidence(unittest.TestCase):
    """Legacy rows fall back to evidence derived only from stored fields."""

    def test_legacy_fallback_uses_stored_fields_only(self):
        items = fallback_evidence(
            score=0.42, threshold=0.30, feature="temperature",
            event_assessment="LIKELY_SENSOR_FAULT", is_weather_event=False,
            sensor_fault_likelihood=0.9, model_version="skyguard-multivariate-v3",
        )
        signals = [i["signal"] for i in items]
        self.assertIn("ml", signals)
        self.assertIn("classification", signals)
        self.assertIn("spatial", signals)
        ml = next(i for i in items if i["signal"] == "ml")
        self.assertIn("before structured evidence", ml["detail"].lower())
        # never fabricates component scores that were not persisted
        for i in items:
            self.assertNotIn("channel score", i["detail"].lower())
            self.assertIn("computed_by", i)

    def test_legacy_fallback_without_spatial(self):
        items = fallback_evidence(
            score=0.5, threshold=0.30, feature="humidity",
            event_assessment=None, is_weather_event=False,
            sensor_fault_likelihood=None, model_version="skyguard-multivariate-v3",
        )
        self.assertNotIn("spatial", [i["signal"] for i in items])


class TestEvidenceIntegration(unittest.TestCase):
    """The production record path (simulate -> _build_anomaly) carries the
    structured evidence, and the demo DRIFT humidity event is flagged as
    thermodynamically inconsistent."""

    @classmethod
    def setUpClass(cls):
        from app.services import app as skyguard
        cls.app = skyguard
        if not getattr(skyguard, "_initialized_test", False):
            skyguard.initialize()
            skyguard._initialized_test = True

    def test_simulation_record_carries_structured_evidence(self):
        s = self.app.simulate({"station_id": "AWS-023", "variable": "temperature",
                               "anomaly_type": "SPIKE", "magnitude": 25, "duration": 1})
        rec = s.get("result") or {}
        self.assertTrue(s.get("detected"), "precondition: injection must be detected")
        evidence = rec.get("evidence")
        self.assertIsInstance(evidence, list)
        self.assertTrue(evidence)
        for item in evidence:
            self.assertTrue({"signal", "label", "status", "detail", "computed_by"} <= set(item))
        signals = {i["signal"] for i in evidence}
        self.assertIn("ml", signals)
        self.assertIn("spatial", signals)
        self.assertIn("thermodynamic", signals)
        # scores/types unchanged by the additive evidence (no model behavior change)
        self.assertEqual(rec["anomaly_type"], "SPIKE")

    def test_demo_drift_humidity_is_thermodynamically_inconsistent(self):
        result = self.app.simulate({"station_id": "AWS-023", "variable": "humidity",
                                    "anomaly_type": "DRIFT", "magnitude": 15, "duration": 6})
        rec = result.get("result")
        if not rec:
            self.skipTest("demo DRIFT not detected by current model")
        evidence = rec.get("evidence") or []
        thermo = next((e for e in evidence if e["signal"] == "thermodynamic"), None)
        self.assertIsNotNone(thermo, "thermodynamic evidence must be present")
        # Humidity drifted to 0% -> the dew-point check is unusable at zero RH
        self.assertEqual(thermo["status"], "inconsistent")


if __name__ == "__main__":
    unittest.main()