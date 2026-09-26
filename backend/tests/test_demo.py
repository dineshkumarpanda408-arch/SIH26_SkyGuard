"""Demo reproducibility regression tests.

Guards the honest-demo refactor: the Live-Monitor demo must replay REAL,
reproducible production detection records (re-computed from the frozen
simulator), never stored hand-picked scores. The three demo events must stay at
readings 13 / 23 / 33 with types SPIKE / DRIFT / FROZEN_SENSOR.

Run with the stdlib unittest runner from the backend directory:
    python -m unittest discover -s tests
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services import app as skyguard  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from app import models  # noqa: E402


def _parse(ts):
    # sqlite may hand back a str on some drivers
    import datetime
    if isinstance(ts, datetime.datetime):
        return ts
    return datetime.datetime.fromisoformat(str(ts))


class TestDemoSpecHygiene(unittest.TestCase):
    """DEMO_EVENT_SPECS must never carry a model verdict literal (score,
    confidence, severity, raw/expected value)."""

    def test_specs_carry_only_injection_parameters(self):
        forbidden = {"score", "confidence", "severity", "raw_value",
                     "expected_value", "prob_sensor_fault", "event_assessment"}
        self.assertEqual(
            [cfg["position"] for cfg in skyguard.DEMO_EVENT_SPECS],
            [13, 23, 33],
        )
        for cfg in skyguard.DEMO_EVENT_SPECS:
            self.assertEqual(set(cfg) & forbidden, set(), cfg)
            self.assertIn("station_id", cfg)
            self.assertIn("anomaly_type", cfg)
            self.assertIn("feature", cfg)
            self.assertIn("magnitude", cfg)
            self.assertIn("duration", cfg)


class TestDemoProductionReproducibility(unittest.TestCase):
    """Seed + episode replay must equal the real production simulator output."""

    @classmethod
    def setUpClass(cls):
        if not getattr(skyguard, "_initialized_test", False):
            skyguard.initialize()
            skyguard._initialized_test = True
        skyguard.seed_demo_anomalies()

    def _simulate_record(self, cfg):
        result = skyguard.simulate({
            "station_id": cfg["station_id"],
            "variable": cfg["feature"],
            "anomaly_type": cfg["anomaly_type"],
            "magnitude": cfg.get("magnitude"),
            "duration": cfg.get("duration", 1),
        })
        self.assertTrue(result.get("detected"), result)
        return result["result"]

    def _db_record(self, cfg):
        db = SessionLocal()
        try:
            row = (db.query(models.Anomaly)
                   .filter(models.Anomaly.station_id == cfg["station_id"],
                           models.Anomaly.anomaly_type == cfg["anomaly_type"],
                           models.Anomaly.feature == cfg["feature"])
                   .first())
            self.assertIsNotNone(row, f"missing record for {cfg}")
            return row
        finally:
            db.close()

    def test_seed_matches_live_simulation(self):
        """The stored demo record is byte-exact with the simulator's output."""
        for cfg in skyguard.DEMO_EVENT_SPECS:
            sim = self._simulate_record(cfg)
            row = self._db_record(cfg)
            self.assertEqual(row.score, sim["score"], cfg)
            self.assertEqual(row.confidence, sim["confidence"], cfg)
            self.assertEqual(row.severity, sim["severity"], cfg)
            self.assertEqual(row.anomaly_type, sim["anomaly_type"], cfg)
            self.assertEqual(row.feature, sim["feature"], cfg)
            self.assertEqual(row.raw_value, sim["raw_value"], cfg)
            self.assertEqual(row.expected_value, sim["expected_value"], cfg)
            self.assertEqual(row.event_assessment, sim["event_assessment"], cfg)
            self.assertEqual(row.is_weather_event, sim["is_weather_event"], cfg)
            # the appended provenance note marks it as a demo replay
            self.assertIn("Demo anomaly record", row.event_evidence, cfg)

    def test_simulation_is_deterministic(self):
        for cfg in skyguard.DEMO_EVENT_SPECS:
            a = self._simulate_record(cfg)
            b = self._simulate_record(cfg)
            self.assertEqual(a["score"], b["score"], cfg)
            self.assertEqual(a["anomaly_type"], b["anomaly_type"], cfg)
            self.assertEqual(a["raw_value"], b["raw_value"], cfg)

    def test_seed_is_idempotent(self):
        before = {cfg["position"]: self._db_record(cfg).score
                  for cfg in skyguard.DEMO_EVENT_SPECS}
        skyguard.seed_demo_anomalies()
        after = {cfg["position"]: self._db_record(cfg).score
                 for cfg in skyguard.DEMO_EVENT_SPECS}
        self.assertEqual(before, after)
        # re-seed keeps the same unique rows (no duplicates)
        db = SessionLocal()
        try:
            for cfg in skyguard.DEMO_EVENT_SPECS:
                n = (db.query(models.Anomaly)
                     .filter(models.Anomaly.station_id == cfg["station_id"],
                             models.Anomaly.anomaly_type == cfg["anomaly_type"],
                             models.Anomaly.feature == cfg["feature"])
                     .count())
                self.assertEqual(n, 1, cfg)
        finally:
            db.close()

    def test_episode_sequence_and_real_values(self):
        """35-reading episode: events exactly at 13/23/33 in SPIKE, DRIFT,
        FROZEN_SENSOR order; every other reading carries no event; the event
        fields equal the persisted production records."""
        episode = skyguard.demo_episode("AWS-023")
        self.assertEqual(len(episode), skyguard.DEMO_EPISODE_LENGTH)
        expected_types = {13: "SPIKE", 23: "DRIFT", 33: "FROZEN_SENSOR"}
        seen = {}
        for i, reading in enumerate(episode, start=1):
            ev = reading.get("demo_event")
            if i in expected_types:
                self.assertIsNotNone(ev, f"reading {i} should be an event")
                self.assertEqual(ev["anomaly_type"], expected_types[i], f"reading {i}")
                seen[i] = ev
                # the event value in the stream equals the persisted record
                row = self._db_record({
                    "station_id": "AWS-023",
                    "anomaly_type": expected_types[i],
                    "feature": ev["feature"],
                })
                self.assertEqual(reading[ev["feature"]], row.raw_value, f"reading {i}")
            else:
                self.assertIsNone(ev, f"reading {i} must stay Normal")
        self.assertEqual(set(seen), set(expected_types))

    def test_confidence_is_production_rounded_score(self):
        for cfg in skyguard.DEMO_EVENT_SPECS:
            row = self._db_record(cfg)
            self.assertEqual(row.confidence, round(row.score, 3), cfg)
            self.assertGreater(row.confidence, 0.0, cfg)
            self.assertLess(row.confidence, 1.0, cfg)


if __name__ == "__main__":
    unittest.main()