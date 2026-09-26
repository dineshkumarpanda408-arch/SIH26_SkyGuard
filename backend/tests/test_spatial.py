"""Unit tests for the spatial weather-event vs sensor-fault reasoning.

Runs with the stdlib unittest runner (no pytest dependency):
    python -m unittest tests/test_spatial.py
or, from the backend directory:
    python -m unittest discover -s tests
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.ml.weather_vs_sensor import assess  # noqa: E402

N_NEIGHBORS = 23


def _neighbors(n_abnormal, total=N_NEIGHBORS):
    """Build a neighbour list where the first `n_abnormal` deviate comparably to
    the target and the rest stay within normal sensor noise."""
    base = 28.0
    anomaly = {"station_id": "AWS-023", "value": base + 7.6, "variable": "temperature",
               "expected_value": base}  # ~ +27% relative -> clearly anomalous
    anomaly_dev = (base + 7.6 - base) / base
    neighbors = []
    for k in range(total):
        e = base + (0.3 if k % 3 == 0 else -0.2)
        if k < n_abnormal:
            v = e * (1.0 + 0.6 * anomaly_dev)  # clearly elevated / comparable
        else:
            v = e * 1.006  # within normal noise
        neighbors.append({"station_id": f"AWS-{k:03d}", "value": float(v), "expected_value": float(e)})
    return anomaly, neighbors


class TestSpatialReasoning(unittest.TestCase):

    def test_single_station_sensor_fault(self):
        """Target anomalous, ~all 23 neighbours normal -> LIKELY_SENSOR_FAULT."""
        anomaly, neighbors = _neighbors(n_abnormal=0)
        assessment, prob, evidence = assess(anomaly, neighbors)
        self.assertEqual(assessment, "LIKELY_SENSOR_FAULT")
        self.assertIsNotNone(prob)
        self.assertGreaterEqual(prob, 0.9)
        self.assertTrue(any("normal" in e.lower() for e in evidence))

    def test_regional_weather_event(self):
        """Target anomalous, 18/23 neighbours also abnormal -> LIKELY_WEATHER_EVENT."""
        anomaly, neighbors = _neighbors(n_abnormal=18)
        assessment, prob, evidence = assess(anomaly, neighbors)
        self.assertEqual(assessment, "LIKELY_WEATHER_EVENT")
        self.assertIsNotNone(prob)
        self.assertLessEqual(prob, 0.1)
        self.assertTrue(any("weather" in e.lower() for e in evidence))

    def test_partial_agreement_weighted_to_fault(self):
        """A minority of neighbours elevated -> ambiguous, weighted toward fault."""
        anomaly, neighbors = _neighbors(n_abnormal=5)
        assessment, prob, _ = assess(anomaly, neighbors)
        self.assertEqual(assessment, "LIKELY_SENSOR_FAULT")
        self.assertIsNotNone(prob)
        self.assertGreaterEqual(prob, 0.5)

    def test_no_neighbors_unavailable(self):
        assessment, prob, evidence = assess(
            {"station_id": "AWS-023", "value": 35.6, "variable": "temperature", "expected_value": 28.0},
            [],
        )
        self.assertEqual(assessment, "UNAVAILABLE")
        self.assertIsNone(prob)
        self.assertTrue(len(evidence) == 1)

    def test_no_value_is_sensor_fault(self):
        anomaly, neighbors = _neighbors(n_abnormal=0)
        anomaly["value"] = None
        assessment, prob, _ = assess(anomaly, neighbors)
        self.assertEqual(assessment, "LIKELY_SENSOR_FAULT")
        self.assertEqual(prob, 0.9)

    def test_deterministic(self):
        anomaly, neighbors = _neighbors(n_abnormal=10)
        a1 = assess(anomaly, neighbors)
        a2 = assess(anomaly, neighbors)
        self.assertEqual(a1, a2)

    def test_evidence_reports_counts(self):
        anomaly, neighbors = _neighbors(n_abnormal=18)
        _, _, evidence = assess(anomaly, neighbors)
        merged = " ".join(evidence)
        self.assertIn("18 of", merged)


if __name__ == "__main__":
    unittest.main()
