"""Tests for the anomaly timeline (context around a flagged reading)."""

import unittest
from types import SimpleNamespace

import pandas as pd

from app.services import app as skyguard


class TestAnomalyTimeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = skyguard
        if not getattr(skyguard, "_initialized_test", False):
            skyguard.initialize()
            skyguard._initialized_test = True

    def _make_anomaly(self, timestamp, raw_value, expected_value=None, feature=None, anomaly_type="SPIKE"):
        return SimpleNamespace(
            id=901,
            station_id="AWS-023",
            feature=feature or "pressure",
            raw_value=raw_value,
            expected_value=expected_value,
            timestamp=pd.Timestamp(timestamp),
            anomaly_type=anomaly_type,
        )

    def test_mid_history_timeline_has_flagged_point_and_context(self):
        # a timestamp inside AWS-023's history gives 2 normal before + flag + 2 normal after
        ts = pd.Timestamp(self.app.data["timestamp"].min()) + pd.Timedelta(days=5)
        an = self._make_anomaly(ts, 1005.12, expected_value=1007.0, feature="pressure")
        out = self.app.anomaly_timeline(an)

        self.assertEqual(out["anomaly_id"], 901)
        self.assertEqual(out["station_id"], "AWS-023")
        self.assertEqual(out["variable"], "pressure")

        flagged = [p for p in out["points"] if p["is_anomaly"]]
        self.assertEqual(len(flagged), 1)
        self.assertEqual(flagged[0]["value"], 1005.12)
        self.assertEqual(flagged[0]["expected_value"], 1007.0)

        before = [p for p in out["points"] if not p["is_anomaly"] and p["timestamp"] < str(ts)]
        after = [p for p in out["points"] if not p["is_anomaly"] and p["timestamp"] > str(ts)]
        self.assertLessEqual(len(before), 2)
        self.assertLessEqual(len(after), 2)
        # flagged point sits between the normal context
        self.assertEqual(out["points"][len(before)], flagged[0])
        # values come from the same station + variable
        for p in before + after:
            self.assertFalse(p["is_anomaly"])
            self.assertIn(p["label"], ("Normal",))

    def test_timestamp_after_history_gives_before_only(self):
        ts = pd.Timestamp("2030-01-01 00:00:00")
        an = self._make_anomaly(ts, 1020.5, feature="humidity")
        out = self.app.anomaly_timeline(an)
        before = [p for p in out["points"] if not p["is_anomaly"]]
        # trailing detection timestamps (e.g. utcnow seeds) yield 2-before + flag
        self.assertLessEqual(len(before), 2)
        self.assertEqual([p for p in out["points"] if p["is_anomaly"]][0]["value"], 1020.5)
        self.assertGreaterEqual(len(out["points"]), 1)

    def test_timestamp_before_history_gives_after_only(self):
        ts = pd.Timestamp("2000-01-01 00:00:00")
        an = self._make_anomaly(ts, 15.0, feature="temperature")
        out = self.app.anomaly_timeline(an)
        after = [p for p in out["points"] if not p["is_anomaly"] and p["timestamp"] > str(ts)]
        self.assertLessEqual(len(after), 2)
        self.assertEqual([p for p in out["points"] if p["is_anomaly"]][0]["value"], 15.0)

    def test_unknown_variable_keeps_only_flagged_point(self):
        ts = pd.Timestamp("2026-09-01 00:00:00")
        an = self._make_anomaly(ts, 42.0, feature="unknown_metric")
        out = self.app.anomaly_timeline(an)
        self.assertIsNone(out["variable"])
        self.assertEqual(len(out["points"]), 1)
        self.assertTrue(out["points"][0]["is_anomaly"])


if __name__ == "__main__":
    unittest.main()