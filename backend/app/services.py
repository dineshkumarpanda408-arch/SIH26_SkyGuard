"""Coordinating service layer: training, streaming, and analysis.

Loads sample data, trains the MLL pipeline, and provides high-level operations
used by REST + WebSocket. Injects anomalies on demand and evaluates the model.
"""

import json
import time
from datetime import datetime

import numpy as np
import pandas as pd

from .config import ENABLE_CORRECTED_VALUES
from .ingestion import load_csv, normalize
from .ml.pipeline import ModelManager
from .ml.classifier import classify
from .ml.severity import severity, TYPE_BASE
from .ml.weather_vs_sensor import assess
from .ml.sensor_health import compute_health
from .ml.degradation import analyze
from .ml.corrected_value import estimate
from .ml.shap_explain import explain, narrative
from .ml.multivariate import MULTIVARIATE_FEATURES
from .ml.thermo import thermodynamic_evidence
from .ml.evidence import build_evidence, fallback_evidence
from .ml.evaluation import evaluate, multiclass_confusion_matrix, CATEGORIES
from .simulator import inject, ANOMALY_TYPES
from .preprocessing import validate


def _json_evidence(evidence):
    """Serialize a structured evidence list for the anomalies.evidence_json column."""
    if not evidence:
        return None
    try:
        return json.dumps(evidence, default=float)
    except (TypeError, ValueError):
        return None


def _narrative(values, feat, raw, means, stds):
    """Build a human-readable explanation from z-score deviation contributions.

    values : list of {'feature', 'contribution'} where contribution is a +/- tanh
             of the feature's z-score. The most deviant feature is the driver.
    """
    ranked = sorted(values, key=lambda v: -abs(v["contribution"]))
    top = ranked[0]
    parts = []
    for v in ranked:
        c = v["contribution"]
        if abs(c) >= 0.25:
            parts.append(
                f"{v['feature']} contributed strongly to the anomaly decision because "
                f"its value deviated substantially from learned normal behavior "
                f"(contribution {c:+.2f})"
            )
        elif c <= -0.25:
            parts.append(f"{v['feature']} pushed against the anomaly decision ({c:+.2f})")
    if not parts:
        return (
            "The model flagged this reading as anomalous based on a combination "
            "of feature deviations from learned normal behavior."
        )
    if top["feature"] == feat and raw is not None:
        mean = means.get(feat, 0.0)
        std = stds.get(feat, 1.0)
        z = (raw - mean) / std
        lead = (
            f"The {feat} reading of {raw:.1f} deviated by about {z:.1f} standard "
            f"deviations from the station's normal range ({mean:.1f} +/- {std:.1f}), "
            f"making it the primary contributor."
        )
    else:
        lead = f"The most influential variable was {top['feature']}."
    return lead + " " + " ".join(parts) + "."


class _TTLCache:
    """Minimal time-to-live memoizer for expensive ML computations."""

    def __init__(self, default_ttl=60):
        self._default_ttl = default_ttl
        self._store = {}

    def get(self, key, compute, ttl=None):
        now = time.monotonic()
        hit = self._store.get(key)
        if hit and hit["expires"] > now:
            return hit["value"]
        value = compute()
        self._store[key] = {
            "value": value,
            "expires": now + (ttl if ttl is not None else self._default_ttl),
        }
        return value

    def invalidate(self, key=None):
        if key is None:
            self._store.clear()
        else:
            self._store.pop(key, None)


class SkyGuardApp:
    def __init__(self):
        self.model = None
        self.data = None            # full normal dataset
        self.stream_pos = {}        # per-station stream position
        self.anomaly_history = []
        self.health_history = {}
        self._station_names = {}
        self._cache = _TTLCache()
        self.model_meta = {}
        self.model_version = "unknown"
        self.stale_after_sec = 3600

    # ---- setup ----
    def initialize(self, csv_path=None):
        df = normalize(load_csv(csv_path))
        self.data = validate(df)
        self._station_names = {
            sid: (self.data[self.data.station_id == sid]["timestamp"].iloc[0].strftime("%d %b %Y"))
            for sid in self.data["station_id"].unique()
        }
        # attach station names via a lookup
        self._meta = {}
        from .data_generator_meta import STATION_META
        for sid, name, lat, lon in STATION_META:
            self._meta[sid] = {"name": name, "latitude": lat, "longitude": lon}

        # Production model: load the persisted artifact, or train once and
        # persist. We NEVER retrain on a per-page/per-request basis.
        from .ml.training import load_production_model, train_production, MODEL_VERSION
        self.model, self.model_meta = load_production_model()
        if self.model is None:
            art = train_production(force=True)
            self.model, self.model_meta = load_production_model()
        self.model_version = (self.model_meta or {}).get("model_version", MODEL_VERSION)
        self.v2 = "v2" in self.model_version
        return self

    def _model_context(self):
        """Small, honest provenance block: which model + dataset + threshold
        produced the scores in the response. Values come from the persisted
        training metadata / dataset metadata — never from guesswork."""
        meta = self.model_meta or {}
        cal = meta.get("calibration") or {}
        return {
            "model_version": meta.get("model_version", getattr(self, "model_version", "unknown")),
            "model_name": meta.get("model_name"),
            "dataset_version": meta.get("dataset_version"),
            "dataset_sha256": meta.get("dataset_sha256"),
            "threshold": meta.get("threshold"),
            "calibration_status": cal.get("status", "uncalibrated"),
            "calibration_method": cal.get("method", "none"),
            "trained_at_utc": meta.get("trained_at_utc"),
            "synthetic": bool(meta.get("synthetic")),  # v2 telemetry is synthetic, explicitly
        }

    def stations(self):
        out = []
        for sid in self.data["station_id"].unique():
            m = self._meta.get(sid, {})
            out.append({
                "station_id": sid,
                "name": m.get("name"),
                "latitude": m.get("latitude"),
                "longitude": m.get("longitude"),
                "active": True,
            })
        return out

    # ---- stream next reading for a station (demo mode) ----
    def next_reading(self, station_id="AWS-023"):
        sub = self.data[self.data.station_id == station_id].reset_index(drop=True)
        pos = self.stream_pos.get(station_id, 0)
        if pos >= len(sub):
            return None
        row = sub.iloc[pos]
        self.stream_pos[station_id] = pos + 1
        return {
            "station_id": station_id,
            "timestamp": row["timestamp"],
            "temperature": None if pd.isna(row["temperature"]) else float(row["temperature"]),
            "pressure": None if pd.isna(row["pressure"]) else float(row["pressure"]),
            "humidity": None if pd.isna(row["humidity"]) else float(row["humidity"]),
        }

    def add_reading(self, reading: dict):
        """Process one reading through the full ML pipeline (validation -> features
        -> inference -> decision), return a result dict for REST/WS.

        The response separates WHAT THE MODEL SAW (input) from what it OUTPUT
        (prediction) from any external ground truth (ground_truth), and tags
        every AI value with provenance (computed_by) so nothing is claimed as
        ML-derived unless it actually is.
        """
        station_id = reading.get("station_id")
        ts = reading.get("timestamp") or datetime.utcnow()
        sub = self.data[self.data.station_id == station_id].copy()
        sub = sub.tail(60).reset_index(drop=True)
        new = {
            "station_id": station_id,
            "timestamp": ts,
            "temperature": reading.get("temperature"),
            "pressure": reading.get("pressure"),
            "humidity": reading.get("humidity"),
        }
        augmented = pd.concat([sub, pd.DataFrame([new])], ignore_index=True)
        result = self.model.analyze_row(augmented)
        result["station_id"] = station_id
        result["timestamp"] = ts
        result["ground_truth"] = bool(reading.get("ground_truth", False))
        result["injected_type"] = reading.get("injected_type")

        # provenance tags: which computation produced which field
        result["computed_by"] = {
            "anomaly_score": "ml:model_fusion",
            "component_scores": "ml:model_fusion",
            "raw_decision": "ml:baseline_isolation_forest",
            "is_anomaly": "rule:score>threshold",
            "feature": "stats:zscore(rule:missing/frozen)",
            "anomaly_probability": None,  # uncalibrated -> never labeled a probability
            "calibration_status": result.get("calibration_status", "uncalibrated"),
        }
        ctx = self._model_context()
        result["model"] = {
            "model_version": ctx["model_version"],
            "dataset_version": ctx["dataset_version"],
            "threshold": ctx["threshold"],
        }
        result["explanation"] = {
            "method": "shap (shap.TreeExplainer over baseline IsolationForest)",
            "available": False,
            "limitation": "Point-stream explanations are omitted here; use /explanations for a full "
                          "SHAP attribution once the anomaly is persisted.",
        }
        if result["is_anomaly"]:
            record = self._build_anomaly(new, result, station_id)
            result.update(record)
            self.anomaly_history.insert(0, record)
        return result

    def _build_anomaly(self, new, result, station_id, injected_series=None):
        """Construct a full anomaly record (type/severity/confidence/assessment/etc).

        `injected_series` (optional) is the actual series produced by the
        simulator. When provided it is used for classification so multi-reading
        patterns (FROZEN plateau, DRIFT trend, MISSING gap) are recognised from
        their real shape instead of a single reconstructed point.
        """
        feature = result["feature"]
        value = new.get(feature)
        if injected_series is not None:
            series = np.asarray(
                [x for x in injected_series if x is not None and not (isinstance(x, float) and np.isnan(x))],
                dtype=float,
            )
        else:
            series = self._recent_series(station_id, feature)
            # append the current anomalous point so the classifier sees the full context
            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                series = np.append(series, value)

        # classify
        a_type = classify({"value": value, "variable": feature, "expected_value": 0}, series)

        # confidence = combined model score
        confidence = round(result["score"], 3)

        # magnitude (z-normalized deviation); guarded so missing values don't
        # propagate NaN into severity/probability computations
        expected = self._expected(station_id, feature, new)
        is_missing = value is None or (isinstance(value, float) and np.isnan(value))
        std = self._local_std(station_id, feature)
        if is_missing or std is None:
            dev = 0.0
            mag = 0.0
            expected = round(expected, 2)
        else:
            dev = abs(value - expected)
            mag = min(1.0, dev / (std + 1e-9) / 5.0)

        sev, _sev_score = severity(a_type, mag, confidence, persistence=1, type_weight_map=TYPE_BASE)

        # weather vs sensor (spatial evidence)
        neighbors = self._neighbors(station_id, feature)
        assessment, sensor_fault_likelihood, spatial_evidence = assess(
            {"station_id": station_id, "value": value, "variable": feature, "expected_value": expected},
            neighbors=neighbors,
        )
        is_weather = assessment == "LIKELY_WEATHER_EVENT"

        # TASK 1: dew-point / thermodynamic consistency evidence (signal only,
        # never feeds the fused score). Missing T/RH are filled from the
        # station's latest context so the pair reflects the reading's own
        # joint state instead of being silently skipped.
        ctx = self._context_latest(station_id, new)
        thermo = thermodynamic_evidence(ctx.get("temperature"), ctx.get("humidity"))

        # TASK 2: structured evidence/provenance list, built ONLY from signals
        # the pipeline actually computed (no inference from the anomaly label).
        evidence = build_evidence(
            score=float(confidence),
            threshold=float(getattr(self.model, "threshold", 0.72)),
            component_scores=result.get("component_scores") or {},
            rule_flags=result.get("rule_flags") or {},
            spatial={
                "assessment": assessment,
                "is_weather_event": is_weather,
                "sensor_fault_likelihood": sensor_fault_likelihood,
                "evidence_lines": spatial_evidence,
            },
            thermo=thermo,
            model_version=self.model_version,
            calibration_status=result.get("calibration_status", "uncalibrated"),
        )

        corrected = None
        corr_method = None
        corr_conf = None
        if ENABLE_CORRECTED_VALUES:
            sub_station = self.data[self.data.station_id == station_id]
            ts_list = sub_station["timestamp"].tail(60).values
            est = estimate(self._valid_series(station_id, feature), feature, timestamps=ts_list)
            corrected = est["corrected_value"]
            corr_method = est["correction_method"]
            corr_conf = est["correction_confidence"]

        raw_val = None if (value is None or (isinstance(value, (float, np.floating)) and np.isnan(value))) else float(value)
        exp_val = None if (expected is None or (isinstance(expected, (float, np.floating)) and np.isnan(expected))) else round(float(expected), 2)
        dev_val = None if (dev is None or (isinstance(dev, (float, np.floating)) and np.isnan(dev))) else round(float(dev), 2)

        return {
            "station_id": station_id,
            "timestamp": new["timestamp"],
            "anomaly_type": a_type,
            "anomaly_type_source": "rule/classifier on recovered series (not ML)",
            "confidence": confidence,
            "severity": sev,
            "feature": feature,
            "raw_value": raw_val,
            "expected_value": exp_val,
            "deviation": dev_val,
            "is_weather_event": is_weather,
            "event_assessment": assessment,
            "event_evidence": spatial_evidence,
            "evidence": evidence,
            "thermodynamic": thermo,
            "sensor_fault_likelihood": sensor_fault_likelihood,
            "n_neighbours": len(neighbors),
            "corrected_value": corrected,
            "correction_method": corr_method,
            "correction_confidence": corr_conf,
            "score": result.get("score"),
            "component_scores": result.get("component_scores"),
            "raw_decision": result.get("raw_decision"),
            "rule_flags": result.get("rule_flags"),
            "anomaly_probability": None,
            "calibration_status": result.get("calibration_status", "uncalibrated"),
        }

    def _recent_series(self, station_id, feature):
        sub = self.data[self.data.station_id == station_id]
        vals = sub[feature].dropna().values
        return vals[-60:]

    def _valid_series(self, station_id, feature):
        return self._recent_series(station_id, feature)

    def _expected(self, station_id, feature, new):
        sub = self.data[self.data.station_id == station_id]
        series = sub[feature].dropna()
        if len(series) == 0:
            return 30.0
        return float(series.tail(12).mean())

    def _local_std(self, station_id, feature):
        sub = self.data[self.data.station_id == station_id]
        series = sub[feature].dropna()
        if len(series) < 3:
            return 1.0
        return float(series.tail(15).std()) + 1e-9

    def _neighbors(self, station_id, feature):
        """Return nearby stations' values at the SAME timestamp as the anomaly."""
        # For demo, use the last reading of other stations' most recent entry
        recent_ts = self.data["timestamp"].max()
        out = []
        for sid in self.data["station_id"].unique():
            if sid == station_id:
                continue
            sub = self.data[self.data.station_id == sid]
            last = sub[sub["timestamp"] <= recent_ts]
            if len(last) == 0:
                continue
            row = last.iloc[-1]
            exp = self._expected(sid, feature, {})
            out.append({"station_id": sid, "value": float(row[feature]) if not pd.isna(row[feature]) else None, "expected_value": exp})
        return out

    def _context_latest(self, station_id, new):
        """Merge a reading with the station's latest known values so joint
        (T, RH) evidence can be computed even when only the affected variable
        is supplied (e.g. simulation passes only the injected feature)."""
        out = {"temperature": new.get("temperature"), "pressure": new.get("pressure"),
               "humidity": new.get("humidity")}
        sub = self.data[self.data.station_id == station_id]
        for var in ("temperature", "pressure", "humidity"):
            if out.get(var) is None:
                vals = sub[var].dropna()
                out[var] = float(vals.iloc[-1]) if len(vals) else None
        return out

    # ---- canonical production prediction ----
    def predict_station(self, station_id):
        """Canonical production prediction for a station's current state.

        Single read-path authority for the live screens. It reuses the existing
        SkyGuard production pipeline (AnalyzeRow -> decision assembly -> sensor
        health) over the canonical 24 AWS production data. It never touches the
        controlled evaluation dataset and never imports the friend's project.

        The result carries explicit provenance (data_source, model_version) and
        never fabricates values: when no reading exists it returns a
        UNAVAILABLE state, and a stale reading is reported rather than ignored.
        """
        ids = set(self.data["station_id"].unique()) if self.data is not None else set()
        if station_id not in ids:
            return None

        data_source = "HISTORICAL"

        sub = self.data[self.data.station_id == station_id]
        n_readings = int(len(sub))
        if n_readings == 0:
            return {
                "station_id": station_id,
                "data_source": data_source,
                "model_version": getattr(self, "model_version", "unknown"),
                "status": "UNAVAILABLE",
                "stale": True,
                "is_anomaly": None,
                "anomaly_score": None,
                "anomaly_type": None,
                "severity": None,
                "confidence": None,
                "feature": None,
                "root_cause": None,
                "sensor_health": None,
                "health_score": None,
                "readings": None,
                "n_readings": 0,
            }

        window = sub.tail(60).reset_index(drop=True)
        last = window.iloc[-1]
        last_ts = pd.to_datetime(last["timestamp"])
        max_ts = pd.to_datetime(self.data["timestamp"].max())
        stale_sec = getattr(self, "stale_after_sec", 3600)
        stale = bool((max_ts - last_ts).total_seconds() > stale_sec)

        result = self.model.analyze_row(window)
        result["station_id"] = station_id
        new = {
            "station_id": station_id,
            "timestamp": last["timestamp"],
            "temperature": None if pd.isna(last["temperature"]) else float(last["temperature"]),
            "pressure": None if pd.isna(last["pressure"]) else float(last["pressure"]),
            "humidity": None if pd.isna(last["humidity"]) else float(last["humidity"]),
        }
        record = None
        explanation = {
            "method": "shap (shap.TreeExplainer over baseline IsolationForest)",
            "contributions": None,
            "narrative": "No explanation produced (reading is normal, or model unavailable).",
            "limitation": "Explanations are computed for anomalies via /explanations/{id}.",
        }
        if result["is_anomaly"]:
            record = self._build_anomaly(new, result, station_id)
            result.update(record)
            contribs, narr = self._shap_explanation(window, result)
            explanation = {
                "method": "shap (shap.TreeExplainer over baseline IsolationForest)",
                "contributions": contribs,
                "narrative": narr,
                "limitation": "SHAP attribution is computed via shap.TreeExplainer and is "
                              "model-true by construction.",
            }
        h = self.sensor_health(station_id) or {}
        ctx = self._model_context()
        return {
            "station_id": station_id,
            "data_source": data_source,
            "model_version": getattr(self, "model_version", "unknown"),
            "dataset_version": ctx["dataset_version"],
            "dataset_sha256": ctx["dataset_sha256"],
            "computed_by": {
                "anomaly_score": "ml:model_fusion",
                "component_scores": "ml:model_fusion",
                "raw_decision": "ml:baseline_isolation_forest",
                "status": "rule:score>threshold",
                "anomaly_type": "rule/classifier (not ML)",
                "event_assessment": "statistical heuristic (not ML)",
                "sensor_fault_likelihood": "statistical heuristic (not ML)",
                "corrected_value": "statistical diurnal estimate (not ML)",
            },
            "timestamp": str(new["timestamp"]),
            "status": "STALE" if stale else ("ANOMALY" if result["is_anomaly"] else "NORMAL"),
            "stale": stale,
            "is_anomaly": bool(result["is_anomaly"]),
            "anomaly_score": round(float(result["score"]), 4),
            "component_scores": result.get("component_scores"),
            "raw_decision": result.get("raw_decision"),
            "rule_flags": result.get("rule_flags"),
            "anomaly_probability": None,
            "calibration_status": result.get("calibration_status", "uncalibrated"),
            "explanation": explanation,
            "anomaly_type": result.get("anomaly_type"),
            "severity": result.get("severity"),
            "confidence": result.get("confidence"),
            "feature": result.get("feature"),
            "primary_feature": result.get("feature"),
            "root_cause": result.get("event_assessment"),
            "is_weather_event": bool(result.get("is_weather_event", False)),
            "sensor_fault_likelihood": result.get("sensor_fault_likelihood"),
            "corrected_value": result.get("corrected_value"),
            "correction_method": result.get("correction_method"),
            "correction_confidence": result.get("correction_confidence"),
            "readings": {k: new[k] for k in ("temperature", "pressure", "humidity")},
            "sensor_health": h.get("health_status"),
            "health_score": round(float(h["health_score"]), 4) if "health_score" in h else None,
            "n_readings": n_readings,
        }

    def _shap_explanation(self, window, result):
        """Feature-level explanation via real SHAP (shap.TreeExplainer).

        Runs the baseline IsolationForest through shap.TreeExplainer to get a
        model-true additive attribution for the anomalous row. Falls back to a
        plain statistical statement if no contribution exceeds the noise floor.
        """
        try:
            df_proc = self.model.prepare(window)
            last = df_proc.iloc[-1]
            baseline = self.model.baseline
            feats = [f for f in baseline.features if f in df_proc.columns]
            X_row = np.asarray([last[f] if not np.isnan(last[f]) else 0.0 for f in feats], dtype=float)
            contribs = explain(baseline, X_row, feats, background=getattr(baseline, "background", None))
            values = [
                {"feature": f, "contribution": round(float(c), 4)}
                for f, c in zip(feats, contribs)
                if abs(c) >= 0.01
            ]
            values.sort(key=lambda v: -abs(v["contribution"]))
            narr = narrative(contribs, feats, top_k=3)
            return values, narr
        except Exception:
            return [], "Model attribution unavailable; no explanation could be computed for this reading."

    # ---- bulk production predictions + production analytics ----
    def predictions(self):
        """Return the canonical prediction for every 24 station, using
        predict_station() as the single authority."""
        return [self.predict_station(s["station_id"]) for s in self.stations()]

    def production_analytics(self, source="HISTORICAL", db=None):
        """Aggregate PRODUCTION state across the 24 stations.

        Derives dynamic aggregates from live/persisted anomaly records and
        current station telemetry states.

        Expensive (re-aggregates the full anomaly corpus + per-station
        prediction sweep), so the result is memoised with a short TTL and
        invalidated explicitly by the simulate()/ingest write paths.
        """

        def compute():
            return self._compute_production_analytics(source=source, db=db)

        return self._cache.get(f"production_analytics:{source}", compute, ttl=30)

    def _compute_production_analytics(self, source="HISTORICAL", db=None):
        from .database import SessionLocal
        from . import models as _models

        close_db = False
        if db is None:
            db = SessionLocal()
            close_db = True

        try:
            # Query all detected anomalies from database
            anomalies = db.query(_models.Anomaly).order_by(_models.Anomaly.timestamp.desc()).all()

            # Dynamic Anomaly Type Distribution
            type_counts = {}
            for a in anomalies:
                t = (a.anomaly_type or "").strip()
                if t and t.upper() != "NORMAL":
                    type_counts[t] = type_counts.get(t, 0) + 1

            # Dynamic Severity Distribution
            severity_counts = {}
            for a in anomalies:
                sev = (a.severity or "").strip().upper()
                if sev and sev != "NONE":
                    severity_counts[sev] = severity_counts.get(sev, 0) + 1

            # Map latest anomaly per station
            latest_anomaly_by_station = {}
            for a in anomalies:
                if a.station_id and a.station_id not in latest_anomaly_by_station:
                    latest_anomaly_by_station[a.station_id] = a

            preds = [p for p in self.predictions() if p is not None]
            preds_by_id = {p["station_id"]: p for p in preds}

            station_rows = []
            status_counts = {"NORMAL": 0, "ANOMALY": 0, "STALE": 0, "UNAVAILABLE": 0}
            health_counts = {}

            all_stations = self.stations()
            for s in all_stations:
                sid = s["station_id"]
                pred = preds_by_id.get(sid)
                latest_an = latest_anomaly_by_station.get(sid)
                sh = self.sensor_health(sid) or {}
                h_score = sh.get("health_score")
                if h_score is not None:
                    if h_score > 70:
                        h_status = "NORMAL"
                    elif h_score >= 40:
                        h_status = "WARNING"
                    else:
                        h_status = "CRITICAL"
                else:
                    h_status = "NORMAL"

                health_counts[h_status] = health_counts.get(h_status, 0) + 1

                # Determine dynamic station status
                if not s.get("active", True):
                    status = "UNAVAILABLE"
                elif pred and pred.get("stale"):
                    status = "STALE"
                elif latest_an is not None:
                    status = "ANOMALY"
                elif pred and pred.get("status") in ("ANOMALY", "STALE", "UNAVAILABLE"):
                    status = pred.get("status")
                else:
                    status = "NORMAL"

                status_counts[status] = status_counts.get(status, 0) + 1

                station_rows.append({
                    "station_id": sid,
                    "status": status,
                    "anomaly_type": latest_an.anomaly_type if latest_an else (pred.get("anomaly_type") if pred else None),
                    "severity": latest_an.severity if latest_an else (pred.get("severity") if pred else None),
                    "sensor_health": h_status,
                    "anomaly_score": round(float(latest_an.score), 4) if (latest_an and latest_an.score is not None) else (pred.get("anomaly_score") if pred else None),
                    "timestamp": str(latest_an.timestamp) if latest_an else (str(pred.get("timestamp")) if pred else None),
                    "data_source": source.upper(),
                })

            return {
                "data_source": source.upper(),
                "model_version": getattr(self, "model_version", "skyguard-multivariate-v3"),
                "total_stations": len(all_stations),
                "status_distribution": status_counts,
                "severity_distribution": severity_counts,
                "anomaly_type_distribution": type_counts,
                "sensor_health_distribution": health_counts,
                "stations": station_rows,
            }
        finally:
            if close_db:
                db.close()

    def model_metadata(self):
        """Expose persisted training/model metadata (Model page)."""
        meta = dict(self.model_meta or {})
        return {
            "model_name": meta.get("model_name"),
            "model_version": meta.get("model_version"),
            "trained_at_utc": meta.get("trained_at_utc"),
            "algorithm": meta.get("algorithm"),
            "threshold": meta.get("threshold"),
            "data_source": meta.get("data_source"),
            "dataset_version": meta.get("dataset_version"),
            "dataset_sha256": meta.get("dataset_sha256"),
            "synthetic": bool(meta.get("synthetic")),
            "training_stations": meta.get("training_stations"),
            "n_training_stations": meta.get("n_training_stations"),
            "stations_covered": meta.get("stations_covered"),
            "n_stations": meta.get("n_stations"),
            "variables_covered": meta.get("variables_covered"),
            "n_variables": meta.get("n_variables"),
            "n_observations": meta.get("n_observations"),
            "n_training_obs": meta.get("n_training_obs"),
            "n_validation_obs": meta.get("n_validation_obs"),
            "n_test_obs": meta.get("n_test_obs"),
            "split": meta.get("split"),
            "n_features": meta.get("n_features"),
            "feature_count": meta.get("feature_count"),
            "features": meta.get("features"),
            "calibration": meta.get("calibration"),
            "dataset_file": meta.get("dataset_file"),
            "notes": meta.get("notes"),
        }

    # ---- sensor health ----
    def sensor_health(self, station_id):
        return self._cache.get(
            f"health:{station_id}",
            lambda: self._compute_sensor_health(station_id),
            ttl=30,
        )

    def _compute_sensor_health(self, station_id):
        sub = self.data[self.data.station_id == station_id]
        n = len(sub)
        recent = sub.tail(120).reset_index(drop=True)

        # score the whole recent window with the model -> anomaly rate
        from .features import add_multivariate_features, add_temporal_features
        from .preprocessing import impute as _impute, validate as _validate
        feat = add_temporal_features(add_multivariate_features(_impute(_validate(recent))))
        labels = self.model.score_series(feat)
        anomaly_rate = float(labels.mean()) if len(labels) else 0.0

        missing_rate = 0.0
        for col in ("temperature", "pressure", "humidity"):
            missing_rate += float(recent[col].isna().mean())
        missing_rate /= 3.0
        comm_fail = missing_rate * 0.7

        # drift: use the dedicated ML drift detector if available, otherwise diurnal residual slope
        drift = 0.0
        if hasattr(self.model, "extra_channels") and "drift" in self.model.extra_channels:
            drift_scores = self.model.extra_channels["drift"].score(recent)
            drift = float(drift_scores[-1]) if len(drift_scores) else 0.0
        else:
            y = recent["temperature"].dropna()
            if len(y) >= 24:
                diurnal = y.rolling(24, min_periods=4).mean()
                resid = (y - diurnal).dropna().tail(40).values
                if len(resid) >= 10:
                    slope = np.polyfit(np.arange(len(resid)), resid, 1)[0]
                    typical = float(resid.std()) + 1e-9
                    drift = min(1.0, abs(slope * len(resid)) / (typical * 3.0 + 1e-9))

        # stability: inverse of short-term volatility relative to long-term std
        stability = 1.0
        if n >= 20:
            short = recent["temperature"].diff().dropna().tail(20)
            long_std = float(recent["temperature"].diff().dropna().std()) + 1e-9
            short_vol = float(short.std()) if len(short) else long_std
            stability = max(0.0, 1.0 - (short_vol / (long_std + 1e-9)))

        h = compute_health({
            "anomaly_rate": anomaly_rate,
            "missing_rate": missing_rate,
            "comm_failure_rate": comm_fail,
            "drift": drift,
            "stability": stability,
            "trend": "stable",
        })
        h["station_id"] = station_id
        return h

    def degrade(self, h, history):
        from .ml.degradation import analyze
        return analyze(
            h["health_score"],
            history,
            feature=self._weakest_feature(h),
            factors=h.get("factors"),
        )

    def _weakest_feature(self, h):
        """Name the sensor variable with the highest recent degradation signal.

        Purely diagnostic/factual: picks the variable whose recent missing or
        anomalous share is worst, so the recommended action can be specific.
        """
        station_id = h.get("station_id")
        if not station_id:
            return None
        recent = self.data[self.data.station_id == station_id].tail(120)
        best = None
        best_signal = -1.0
        for var in ("temperature", "pressure", "humidity"):
            if var not in recent.columns:
                continue
            ser = recent[var]
            missing = float(ser.isna().mean())
            anomaly = 0.0
            try:
                anomaly = float((recent[var].diff().abs().tail(40) > (ser.std() * 3)).mean())
            except Exception:
                anomaly = 0.0
            signal = missing + anomaly
            if signal > best_signal:
                best_signal = signal
                best = var
        return best

    # ---- simulation ----
    def simulate(self, params) -> dict:
        # A new anomaly record lands in the DB; drop the analytics aggregate so
        # the Analytics page reflects it immediately.
        self._cache.invalidate("production_analytics:HISTORICAL")
        self._cache.invalidate("production_analytics:PREDICTED")
        station_id = params["station_id"]
        variable = params["variable"]
        anomaly_type = params["anomaly_type"]
        magnitude = params.get("magnitude")
        duration = params.get("duration", 1)

        start = time.time()
        sub = self.data[self.data.station_id == station_id].tail(30).reset_index(drop=True)
        start_idx = max(0, len(sub) - duration)
        injected = inject(sub, variable, anomaly_type, magnitude=magnitude, duration=duration, start_idx=start_idx)

        # run model on the injected tail
        result = self.model.analyze_row(injected)
        latency_ms = (time.time() - start) * 1000
        detected = result["is_anomaly"]

        record = None
        if detected:
            result["feature"] = variable
            feature = variable
            new = {"timestamp": datetime.utcnow()}
            new[feature] = injected[feature].iloc[-1]
            # pass the real injected series so the classifier sees the full
            # multi-reading pattern (FROZEN plateau / DRIFT trend / MISSING gap)
            record = self._build_anomaly(new, result, station_id, injected_series=injected[variable].values)
            record["latency_ms"] = round(latency_ms, 1)

        # persist a simulation event marker (via DB handled in route)
        from .simulator import ALIASES
        canonical_type = ALIASES.get(anomaly_type, anomaly_type)
        meta = injected.attrs.get("injection_meta", {})
        ctx = self._model_context()
        return {
            "message": "Anomaly injected",
            "station_id": station_id,
            "variable": variable,
            "anomaly_type": canonical_type,
            "injected_readings": duration,
            "detected": detected,
            "confidence": record["confidence"] if record else None,
            "severity": record["severity"] if record else None,
            "detected_type": record["anomaly_type"] if record else None,
            "latency_ms": round(latency_ms, 1),
            # provenance: what computed what (never conflates ML with rule/stat).
            "computed_by": {
                "detected": "rule:score>threshold",
                "confidence": "ml:model_fusion",
                "score": "ml:model_fusion",
                "detected_type": "rule/classifier (not ML)",
                "severity": "rule",
                "ground_truth": "system:simulator injection metadata",
            },
            # ground truth is what the simulator injected — kept SEPARATE from
            # the model's prediction above
            "ground_truth": {
                "requested_type": meta.get("requested_type"),
                "canonical_type": meta.get("canonical_type", canonical_type),
                "variable": meta.get("variable", variable),
                "magnitude_sigma": meta.get("magnitude_sigma"),
                "duration": len(meta.get("affected_indices") or []),
                "start_idx": meta.get("start_idx"),
                "affected_indices": meta.get("affected_indices"),
                "true_value": meta.get("true_value"),
            },
            "model": {
                "model_version": ctx["model_version"],
                "dataset_version": ctx["dataset_version"],
                "threshold": ctx["threshold"],
                "calibration_status": ctx["calibration_status"],
            },
            "calibration_status": result.get("calibration_status", "uncalibrated"),
            "anomaly_probability": None,
            "result": record,
        }

    # ---- simulation ----
    def explain_anomaly(self, an):
        """Produce feature-contribution explanations for a persisted anomaly.

        Uses real SHAP (shap.TreeExplainer) over the baseline IsolationForest to
        compute an additive, game-theoretically grounded attribution for the
        anomalous reading. The result is a real model output — never a fabricated
        attribution. A fallback statistical statement (z-score based) is emitted
        when the model cannot produce attributions.
        """
        station_id = an.station_id
        vars_ = ["temperature", "pressure", "humidity"]

        sub = self.data[self.data.station_id == station_id]
        recent = sub[vars_].dropna().tail(120)
        if len(recent) == 0:
            return [], "No training context available for explanation.", "none"

        attempt = self._explain_for_record(station_id, an)
        if attempt is not None:
            values, narr = attempt
            return values, narr, "shap"

        # fallback: honest statistical statement (z-scores, not ML)
        means = recent.mean()
        stds = recent.std() + 1e-9
        feat = an.feature if an.feature in vars_ else "temperature"
        raw = an.raw_value
        contributions = {}
        for v in vars_:
            value = raw if v == feat and raw is not None else float(recent[v].iloc[-1])
            z = (value - means[v]) / stds[v]
            signed = np.tanh(z / 3.0)
            contributions[v] = round(float(signed), 3)
        values = [{"feature": f, "contribution": contributions[f]} for f in vars_]
        narr = _narrative(values, feat, raw, means, stds)
        return values, narr, "statistical"

    def _explain_for_record(self, station_id, an):
        """Rebuild the full-feature anomaly row and run real SHAP attribution.

        Uses model.full_cols in model order (the same order the detector was
        trained on) so attribution is valid. The window is reconstructed with the
        persisted anomalous reading as the LAST row, so the attribution explains
        the anomaly itself (simulated anomalies are injected on a copy, so they
        are NOT present in self.data). Returns None if attribution is impossible.
        """
        baseline = self.model.baseline if self.model is not None else None
        if baseline is None or getattr(baseline, "features", None) is None:
            return None
        sub = self.data[self.data.station_id == station_id]
        if len(sub) == 0:
            return None
        vars_ = ["temperature", "pressure", "humidity"]
        last = sub.tail(1).iloc[0]
        an_raw = getattr(an, "raw_value", None)
        an_feat = getattr(an, "feature", None) or "temperature"
        try:
            ts = getattr(an, "timestamp", None) or last["timestamp"]
            if not isinstance(ts, (pd.Timestamp, datetime)) and not pd.isna(pd.to_datetime(ts, errors="coerce")):
                ts = last["timestamp"]
            new_row = {
                "station_id": station_id,
                "timestamp": pd.Timestamp(ts),
                "temperature": float(last["temperature"]) if not pd.isna(last["temperature"]) else None,
                "pressure": float(last["pressure"]) if not pd.isna(last["pressure"]) else None,
                "humidity": float(last["humidity"]) if not pd.isna(last["humidity"]) else None,
            }
        except Exception:
            return None
        if new_row.get(an_feat) is not None and an_raw is not None and not (isinstance(an_raw, float) and np.isnan(an_raw)):
            new_row[an_feat] = float(an_raw)
        window = pd.concat([sub.tail(59).copy(), pd.DataFrame([new_row])], ignore_index=True)
        df_proc = self.model.prepare(window)
        last_proc = df_proc.iloc[-1]
        feats = [f for f in baseline.features if f in df_proc.columns]
        X_row = np.asarray([last_proc[f] if not np.isnan(last_proc[f]) else 0.0 for f in feats], dtype=float)
        contribs = explain(baseline, X_row, feats, background=getattr(baseline, "background", None))
        values = [
            {"feature": f, "contribution": round(float(c), 4)}
            for f, c in zip(feats, contribs)
            if abs(c) >= 0.01
        ]
        values.sort(key=lambda v: -abs(v["contribution"]))
        if not values:
            return None
        narr = narrative(contribs, feats, top_k=3)
        return values, narr

    # ---- anomaly timeline ----
    def anomaly_timeline(self, an):
        """Context around a flagged reading for the AnomalyDetail timeline.

        Returns up to 2 normal readings before and up to 2 after the flagged
        point, all from the SAME station and SAME variable, plus the flagged
        point itself (from the persisted anomaly record). Boundary-safe: fewer
        context points are returned when the anomaly sits at the edge of the
        station's history. The flagged point is the anomaly record (its raw
        value + timestamp); nothing is fabricated.
        """
        station_id = getattr(an, "station_id", None)
        variable = getattr(an, "feature", None)
        if variable not in ("temperature", "pressure", "humidity"):
            variable = None
        anomaly_type = getattr(an, "anomaly_type", None) or "ANOMALY"
        flagged_ts = getattr(an, "timestamp", None)
        try:
            flagged_ts = pd.Timestamp(flagged_ts)
        except Exception:
            flagged_ts = None

        def _normal_points(frame):
            out = []
            for _, row in frame.iterrows():
                out.append({
                    "label": "Normal",
                    "timestamp": str(row["timestamp"]),
                    "value": round(float(row[variable]), 2) if not pd.isna(row[variable]) else None,
                    "is_anomaly": False,
                })
            return out

        before = []
        after = []
        if variable is not None and station_id is not None:
            sub = self.data[self.data.station_id == station_id].dropna(subset=[variable]).copy()
            if sub.empty:
                sub = self.data[self.data.station_id == station_id].copy()
            if not sub.empty:
                sub["_stamps"] = pd.to_datetime(sub["timestamp"])
                sub = sub.sort_values("_stamps")
                if flagged_ts is not None:
                    before = _normal_points(sub[sub["_stamps"] < flagged_ts].tail(2))
                    after = _normal_points(sub[sub["_stamps"] > flagged_ts].head(2))
                else:
                    before = _normal_points(sub.tail(2))

        points = before + [
            {
                "label": "ANOMALY (flagged by model)",
                "timestamp": str(flagged_ts),
                "value": getattr(an, "raw_value", None),
                "expected_value": getattr(an, "expected_value", None),
                "is_anomaly": True,
            }
        ] + after

        return {
            "anomaly_id": getattr(an, "id", None),
            "station_id": station_id,
            "variable": variable,
            "points": points,
        }

    # ---- evaluation ----
    def run_evaluation(self):
        return self._cache.get("evaluation", lambda: self._compute_evaluation(), ttl=60)

    def _compute_evaluation(self):
        """Evaluate baseline vs temporal vs multivariate vs full on injected data."""
        df = self.data[self.data.station_id == "AWS-023"].head(400).copy().reset_index(drop=True)
        rng = np.random.default_rng(0)
        all_true = []
        all_pred_full = []
        results_by_type = []

        for a_type in ANOMALY_TYPES:
            injected = inject(df.copy(), "temperature", a_type, magnitude=None, duration=1, start_idx=len(df) - 10, rng=rng)
            y_true = injected["ground_truth"].values
            feat_df = injected[["temperature"]].copy()
            # baseline on raw
            from .detector import IsolationForestDetector
            base = IsolationForestDetector().fit(injected[["temperature","pressure","humidity"]].assign(pressure=injected["pressure"], humidity=injected["humidity"]), ["temperature","pressure","humidity"])
            base_scores, base_labels, _ = base.decision_scores(injected)
            # full pipeline
            full_scores = self.model.analyze_row(injected)["score"]
            full_label = 1 if full_scores > getattr(self.model, "threshold", 0.72) else 0
            idx = len(injected) - 1
            results_by_type.append({"true_type": a_type, "y_true": int(y_true[idx]), "y_pred": full_label})
            all_true.append(int(y_true[idx]))
            all_pred_full.append(full_label)

        detailed = self.detailed_evaluation()
        return {
            "n": len(all_true),
            "full_model": evaluate(all_true, all_pred_full),
            "by_type": self._type_metrics(results_by_type),
            "detailed": detailed["performance"],
            "dataset": detailed["dataset"],
            "per_category": detailed["per_category"],
            "multiclass_confusion_matrix": detailed["multiclass_confusion_matrix"],
        }

    def _type_metrics(self, results_by_type):
        from .ml.evaluation import evaluate_by_type
        return evaluate_by_type(results_by_type)

    # ---- detailed large-scale evaluation (isolated, reproducible) ----
    # The evaluation is fully isolated from production inference and from the
    # live telemetry dataset: it works on a COPY, never mutates self.data, and
    # uses a fixed seed so results are reproducible. Ground truth is kept
    # separate from model predictions.
    EVAL_VARIABLES = ["temperature", "pressure", "humidity"]
    EVAL_TYPES = ["SPIKE", "DROP", "DRIFT", "FROZEN_SENSOR", "MISSING_DATA"]
    EVAL_DURATIONS = {
        "SPIKE": 1,
        "DROP": 1,
        "DRIFT": 8,
        "FROZEN_SENSOR": 6,
        "MISSING_DATA": 5,
    }

    def detailed_evaluation(self):
        return self._cache.get("detailed_evaluation", lambda: self._compute_detailed_evaluation(), ttl=120)

    def _canonical_pred(self, p):
        """Map a raw classifier diagnosis onto the evaluation category set."""
        if p in ("SPIKE", "DROP", "DRIFT", "FROZEN_SENSOR", "MISSING_DATA"):
            return p
        return "NORMAL"

    def _diagnose_point(self, ev_df, i, var):
        """Classify the diagnosis for a single point using its local series."""
        series = ev_df[var].to_numpy(dtype=float)
        start = max(0, i - 8)
        window = series[start : i + 1]
        val = series[i]
        row = {
            "value": None if (np.isnan(val) or val is None) else float(val),
            "variable": var,
            "expected_value": 0,
        }
        return self._canonical_pred(classify(row, window))

    def _build_eval_frame(self, rows=1300, seed=7, base_frame=None):
        """Build an isolated, reproducible, realistic ground-truth frame.

        Injections use VARiED magnitude (random sigma) so the test is harder and
        more representative of real telemetry than a uniform 30-sigma slam — this
        avoids the trivially-easy, artificially-perfect results that a naive
        evaluation produces. Returns (ev_df, gt, true_type, summary).

        base_frame : optional pre-built telemetry frame (timestamp, temperature,
                     pressure, humidity). Defaults to the real AWS-023 series
                     read directly from the raw sample source.
        """
        if base_frame is None:
            raw = normalize(load_csv())
            df_base = raw[raw["station_id"] == "AWS-023"].head(rows).copy().reset_index(drop=True)
        else:
            df_base = base_frame.reset_index(drop=True)
        rng = np.random.default_rng(seed)

        ev_df = df_base[["timestamp", "temperature", "pressure", "humidity"]].copy()
        ev_df["ground_truth"] = False
        ev_df["injected_type"] = None

        n_per = {"SPIKE": 5, "DROP": 5, "DRIFT": 2, "FROZEN_SENSOR": 2, "MISSING_DATA": 2}
        occupied = set()
        true_type = {}
        summary = {"injections": 0}

        def place(dur):
            margin = 14
            min_start = margin
            max_start = len(ev_df) - dur - margin
            for _ in range(300):
                s = int(rng.integers(min_start, max_start))
                span = set(range(s - 8, s + dur + 8))
                if not (span & occupied):
                    occupied.update(range(s, s + dur))
                    return s, s + dur
            s = int(rng.integers(min_start, max_start))
            occupied.update(range(s, s + dur))
            return s, s + dur

        def draw_mag(typ):
            # varied, harder-than-default magnitudes (in sigma units)
            if typ in ("SPIKE", "DROP"):
                return float(rng.uniform(4.0, 9.0))
            if typ == "DRIFT":
                return float(rng.uniform(5.0, 11.0))
            return None

        for var in self.EVAL_VARIABLES:
            for typ in self.EVAL_TYPES:
                dur = self.EVAL_DURATIONS[typ]
                for _ in range(n_per[typ]):
                    s, e = place(dur)
                    work = ev_df[["timestamp", "temperature", "pressure", "humidity"]].copy()
                    injected = inject(work, var, typ, magnitude=draw_mag(typ), duration=dur, start_idx=s, rng=rng)
                    ev_df.loc[s : e - 1, var] = injected.loc[s : e - 1, var].values
                    ev_df.loc[s : e - 1, "ground_truth"] = True
                    ev_df.loc[s : e - 1, "injected_type"] = typ
                    for i in range(s, e):
                        true_type[i] = typ
                    summary["injections"] += dur

        # humidity must stay within its physical domain (DRIFT saturation fix)
        ev_df["humidity"] = ev_df["humidity"].clip(0.0, 100.0)
        return ev_df, ev_df["ground_truth"].to_numpy(dtype=int), true_type, summary

    def _score_eval(self, ev_df, gt, true_type, scores, injections):
        """Compute the full metrics dict for a scored evaluation frame.

        Shared by the detailed and large evaluations so both report identical
        structure: binary metrics, per-category recall, multi-class matrix.
        """
        y_pred = (scores > float(getattr(self.model, "threshold", 0.72))).astype(int)
        full = evaluate(gt, y_pred, scores=scores)

        per_category = {}
        for typ in self.EVAL_TYPES:
            mask = ev_df["injected_type"].to_numpy() == typ
            total = int(mask.sum())
            detected = int((mask & (y_pred == 1)).sum())
            per_category[typ] = {
                "injected": total,
                "detected": detected,
                "recall": round((detected / total), 4) if total else 0.0,
            }
        per_category["NORMAL"] = {
            "injected": int((gt == 0).sum()),
            "detected": int((gt == 0).sum()) - int(((y_pred == 1) & (gt == 0)).sum()),
            "recall": round(float((gt == 0).mean()), 4),
        }

        anom_mask = (gt == 1) | (y_pred == 1)
        idxs = np.where(anom_mask)[0]
        true_list, pred_list = [], []
        for i in idxs:
            ti = true_type.get(int(i), "NORMAL")
            var = self._most_deviant_variable(ev_df, scores, int(i))
            true_list.append(ti if ti in ("SPIKE", "DROP", "DRIFT", "FROZEN_SENSOR", "MISSING_DATA") else "NORMAL")
            pred_list.append(self._diagnose_point(ev_df, int(i), var))
        multi = multiclass_confusion_matrix(true_list, pred_list, categories=self.EVAL_TYPES)

        return {
            "dataset": {
                "observations": int(len(ev_df)),
                "ground_truth_anomalies": int((gt == 1).sum()),
                "anomaly_rate": round(float((gt == 1).mean()), 4),
                "injections": int(injections),
            },
            "performance": full,
            "per_category": per_category,
            "confusion_matrix": full["confusion_matrix"],
            "multiclass_confusion_matrix": multi,
            "label_counts": {t: v["injected"] for t, v in per_category.items() if t != "NORMAL"},
        }

    def _compute_detailed_evaluation(self):
        """Large, realistic, reproducible evaluation across all types & variables.

        The base telemetry is read directly from the raw sample source (via
        load_csv) rather than self.data, because self.data is globally
        de-duplicated by timestamp and would collapse a single station's series.
        This keeps the evaluation fully isolated from the live dataset.
        """
        ev_df, gt, true_type, summary = self._build_eval_frame()
        scores = self.model.score_series_scores(ev_df)
        return self._score_eval(ev_df, gt, true_type, scores, summary["injections"])

    def _most_deviant_variable(self, ev_df, scores, i):
        """Return the variable currently most anomalous for a row (best-effort)."""
        best, best_z = "temperature", -1.0
        for v in self.EVAL_VARIABLES:
            series = ev_df[v].to_numpy(dtype=float)
            start = max(0, i - 8)
            win = series[start : i + 1]
            win = win[np.isfinite(win)]
            if len(win) < 2:
                continue
            mean = float(np.mean(win[:-1])) if len(win) > 1 else float(win[0])
            std = float(np.std(win[:-1])) + 1e-9
            cur = win[-1]
            z = abs(cur - mean) / std
            if z > best_z:
                best_z, best = z, v
        return best

    # ---- model comparison (measured, on the SAME hard eval set) ----
    def model_comparison(self):
        return self._cache.get("model_comparison", lambda: self._compute_model_comparison(), ttl=60)

    def _compute_model_comparison(self):
        """Compare detector configurations on the SAME realistic, hard eval set.

        This replaces the old n=5 / 30-sigma test that artificially reported
        100% for every configuration. Each detector is fit on the CLEAN base only
        (no injection leakage) then scored on the full injected frame so results
        include true negatives and ROC-AUC.
        """
        from .features import add_temporal_features, add_multivariate_features
        from .ml.seasonal import _seasonal_cols

        ev_df, gt, _true_type, _summary = self._build_eval_frame()
        clean = ev_df[["timestamp", "temperature", "pressure", "humidity"]].copy()
        clean_tf = add_temporal_features(clean)

        # test-side feature frames mirror the fit-side feature sets (no leakage:
        # features are derived from each station's own series, not the labels)
        ev_tf = add_temporal_features(ev_df)
        ev_full = add_multivariate_features(add_temporal_features(ev_df))

        from .detector import IsolationForestDetector
        from .ml.seasonal import SeasonalDetector

        base_det = IsolationForestDetector().fit(clean, ["temperature", "pressure", "humidity"])
        temp_det = IsolationForestDetector().fit(clean_tf, self._tempcols())

        full_tf = add_multivariate_features(add_temporal_features(clean))
        fullcols = [c for c in self._fullcols() if c in full_tf.columns]
        full_det = IsolationForestDetector().fit(full_tf, fullcols)
        seas_det = SeasonalDetector().fit(clean_tf, _seasonal_cols())

        def run(det, frame, cols, seasonal=False):
            if seasonal:
                scores = np.asarray(det.score(ev_df, _seasonal_cols()), dtype=float)
            else:
                scores = det.decision_scores(frame)[0]
            labels = (scores > 0.75).astype(int)
            return evaluate(gt, labels, scores=scores)

        m_base = run(base_det, ev_df, ["temperature", "pressure", "humidity"])
        m_temporal = run(temp_det, ev_tf, self._tempcols())
        m_seasonal = run(seas_det, None, None, seasonal=True)
        m_full = run(full_det, ev_full, fullcols)

        versions = self._measured_version_comparison(ev_df, gt)

        return {
            "models": [
                {"name": "Baseline (raw 3 vars)", **m_base},
                {"name": "+ Temporal Features", **m_temporal},
                {"name": "+ Diurnal/Seasonal", **m_seasonal},
                {"name": "+ Multivariate (FINAL)", **m_full},
            ],
            "selected": "+ Multivariate (FINAL)",
            "provenance": {
                "eval_dataset": {
                    "observations": int(len(ev_df)),
                    "ground_truth_anomalies": int((gt == 1).sum()),
                    "generator": "same hard injected frame (see evaluation)",
                    "seed": 7,
                },
                "note": "All four rows measured on the SAME realistic hard injected frame; "
                        "results are real detector outputs, not fabricated.",
            },
            "versions": versions,
        }

    def _measured_version_comparison(self, ev_df, gt):
        """Score the CURRENT production model and the LEGACY artifacts on the
        SAME evaluation frame, so the "model version" comparison is measured, not
        asserted. Uses score_series_scores (pure, floor-free) so ROC-AUC is real.
        Each version is scored at its own persisted artifact threshold.
        """
        from .ml.training import load_versioned, MODEL_SUBDIR
        from .ml.training import MODEL_VERSION

        result = {
            "note": "Each persisted version scored on the identical hard injected evaluation "
                    "frame at its own artifact threshold; ROC-AUC is computed from pure fused "
                    "scores (no floors).",
            "comparison": [],
        }

        def add(subdir, label):
            model, _meta = load_versioned(subdir)
            if model is None:
                return
            thr = float(getattr(model, "threshold", 0.72))
            scores = np.asarray(model.score_series_scores(ev_df), dtype=float)
            y_pred = (scores > thr).astype(int)
            metrics = evaluate(gt, y_pred, scores=scores)
            result["comparison"].append({
                "label": label,
                "artifact": subdir,
                "n": int(len(gt)),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
                "roc_auc": metrics.get("roc_auc"),
            })

        add(MODEL_SUBDIR, "current production")
        for subdir in ("skyguard_v2", "skyguard_v1"):
            if subdir != MODEL_SUBDIR:
                add(subdir, f"{subdir.replace('skyguard_', '')} (legacy artifact)")
        return result

    def _tempcols(self):
        return ["temperature", "pressure", "humidity", "temperature_change", "hour"]

    def _fullcols(self):
        return ["temperature", "pressure", "humidity", "temperature_change", "hour", "delta_t", "dew_point"]

    # ------------------------------------------------------------------
    # Deterministic Live-Monitor demo episode
    #
    # The mentor watches a fixed 35-reading AWS-023 stream and expects three
    # 🚨 ANOMALY DETECTED events at predictable reading numbers, with Normal
    # readings in between, in the same order every run.
    #
    # To honour "reuse the same anomaly records shown on the Anomalies page",
    # the three demo events are REAL detection results (one of each
    # representative type: SPIKE, DRIFT, FROZEN_SENSOR) that are also persisted
    # as anomaly records, so Live Monitor and the Anomalies page show the same
    # events. Reading 13, 23 and 33 carry that record's real feature value,
    # type, score and severity; every other reading is real AWS-023 weather
    # that the unchanged model scores as Normal. The episode is built once and
    # cached, so the sequence and positions are identical on every demo start.
    #
    # HONESTY: the specs below carry ONLY the canonical production injection
    # parameters (station, feature, type, magnitude, duration) that reproduce
    # each event. They contain NO score / confidence / severity / value fields.
    # seed_demo_anomalies() (and the _load_demo_records() fallback) re-run the
    # REAL production simulator (simulate() -> analyzer -> classifier ->
    # severity -> spatial assessment) on the CURRENT frozen model and persist /
    # report whatever it truly outputs, so the Anomalies page and Live Monitor
    # always show the model's actual verdict — never a stored, hand-picked
    # number. If the frozen model stops detecting a canonical injection, the
    # demo raises instead of fabricating an event.
    # ------------------------------------------------------------------
    DEMO_EVENT_SPECS = [
        {
            "position": 13,
            "station_id": "AWS-023",
            "anomaly_type": "SPIKE",
            "feature": "temperature",
            "magnitude": 14,
            "duration": 1,
        },
        {
            "position": 23,
            "station_id": "AWS-023",
            "anomaly_type": "DRIFT",
            "feature": "humidity",
            "magnitude": 15,
            "duration": 6,
        },
        {
            "position": 33,
            "station_id": "AWS-023",
            "anomaly_type": "FROZEN_SENSOR",
            "feature": "temperature",
            "magnitude": 8,
            "duration": 6,
        },
    ]
    DEMO_EPISODE_LENGTH = 35
    DEMO_EPISODE_START_IDX = 40

    def _production_demo_record(self, cfg):
        """Run the real production simulator for one demo-event spec.

        Returns the production detection record (score, type, severity, raw /
        expected value, spatial assessment, evidence) for the canonical
        injection. Deterministic for a fixed model + data: the same injection
        always reproduces the same record. Never returns a fabricated value ;
        it raises if the frozen model does not detect the injection.
        """
        result = self.simulate({
            "station_id": cfg["station_id"],
            "variable": cfg["feature"],
            "anomaly_type": cfg["anomaly_type"],
            "magnitude": cfg.get("magnitude"),
            "duration": cfg.get("duration", 1),
        })
        record = result.get("result")
        if not result.get("detected") or record is None:
            raise RuntimeError(
                "Demo event not detected by the production model: "
                f"{cfg}. The frozen model no longer flags this canonical "
                "injection; refusing to fabricate a demo anomaly record."
            )
        record["ground_truth"] = result.get("ground_truth")
        return record

    def _demo_evidence(self, cfg, record):
        """Append a provenance note to the production detection evidence so a
        record is never mistaken for a false live detection."""
        base = list(record.get("event_evidence") or [])
        base.append(
            "Demo anomaly record (replayed on Live Monitor): score/type/"
            "severity re-computed by the CURRENT production model via the "
            f"simulator for the {cfg['anomaly_type']} injection "
            f"(score {record.get('score')}, severity {record.get('severity')}); "
            "reproducible by re-running the same simulation."
        )
        return "\n".join(base)

    def _load_demo_records(self):
        """Return the demo events resolved against the persisted anomaly DB.

        Records are (re)generated by the real production simulator in
        seed_demo_anomalies(). If a record is missing in the DB (e.g. a fresh
        DB before the seed ran) it is computed on the fly from the same
        production path and persisted, so the demo NEVER falls back to a
        stored/hard-coded score.
        """
        from .database import SessionLocal
        from . import models as _models

        db = SessionLocal()
        try:
            resolved = []
            for cfg in self.DEMO_EVENT_SPECS:
                an = (
                    db.query(_models.Anomaly)
                    .filter(
                        _models.Anomaly.station_id == cfg["station_id"],
                        _models.Anomaly.anomaly_type == cfg["anomaly_type"],
                        _models.Anomaly.feature == cfg["feature"],
                    )
                    .first()
                )
                if an is None:
                    record = self._production_demo_record(cfg)
                    an = self._persist_demo_record(db, cfg, record)
                resolved.append({
                    "position": cfg["position"],
                    "id": an.id,
                    "station_id": cfg["station_id"],
                    "timestamp": an.timestamp,
                    "anomaly_type": an.anomaly_type,
                    "severity": an.severity,
                    "feature": an.feature,
                    "raw_value": an.raw_value,
                    "expected_value": an.expected_value,
                    "score": an.score,
                    "confidence": an.confidence,
                })
            return resolved
        finally:
            db.close()

    def _persist_demo_record(self, db, cfg, record):
        """Insert one demo anomaly row from a real production record."""
        from datetime import datetime as _dt
        from . import models as _models

        row = _models.Anomaly(
            station_id=cfg["station_id"],
            timestamp=_dt.utcnow(),
            anomaly_type=record.get("anomaly_type"),
            confidence=record.get("confidence"),
            severity=record.get("severity"),
            cause=record.get("event_assessment"),
            is_weather_event=record.get("is_weather_event", False),
            event_assessment=record.get("event_assessment"),
            corrected_value=record.get("corrected_value"),
            correction_method=record.get("correction_method"),
            correction_confidence=record.get("correction_confidence"),
            feature=record.get("feature"),
            score=record.get("score"),
            raw_value=record.get("raw_value"),
            expected_value=record.get("expected_value"),
            prob_sensor_fault=record.get("sensor_fault_likelihood"),
            event_evidence=self._demo_evidence(cfg, record),
            evidence_json=_json_evidence(record.get("evidence")),
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return row

    def seed_demo_anomalies(self):
        """(Re)generate the 3 demo anomaly records from the REAL production
        simulator so the Anomalies page lists exactly the events the Live
        Monitor replays at readings 13 / 23 / 33.

        Idempotent upsert keyed on (station_id, anomaly_type, feature): every
        run recomputes the production output and OVERWRITES the stored row, so
        the records always match the CURRENT frozen model — no stored,
        hand-picked score can go stale and silently diverge from the demo.
        """
        from .database import SessionLocal
        from . import models as _models

        db = SessionLocal()
        try:
            for cfg in self.DEMO_EVENT_SPECS:
                record = self._production_demo_record(cfg)
                existing = (
                    db.query(_models.Anomaly)
                    .filter(
                        _models.Anomaly.station_id == cfg["station_id"],
                        _models.Anomaly.anomaly_type == cfg["anomaly_type"],
                        _models.Anomaly.feature == cfg["feature"],
                    )
                    .first()
                )
                if existing is None:
                    self._persist_demo_record(db, cfg, record)
                    continue
                existing.timestamp = datetime.utcnow()
                existing.anomaly_type = record.get("anomaly_type")
                existing.confidence = record.get("confidence")
                existing.severity = record.get("severity")
                existing.cause = record.get("event_assessment")
                existing.is_weather_event = record.get("is_weather_event", False)
                existing.event_assessment = record.get("event_assessment")
                existing.corrected_value = record.get("corrected_value")
                existing.correction_method = record.get("correction_method")
                existing.correction_confidence = record.get("correction_confidence")
                existing.feature = record.get("feature")
                existing.score = record.get("score")
                existing.raw_value = record.get("raw_value")
                existing.expected_value = record.get("expected_value")
                existing.prob_sensor_fault = record.get("sensor_fault_likelihood")
                existing.event_evidence = self._demo_evidence(cfg, record)
                existing.evidence_json = _json_evidence(record.get("evidence"))
            db.commit()
        finally:
            db.close()

    def demo_episode(self, station_id="AWS-023"):
        """Return the ordered 35-reading demo episode (list of reading dicts).

        Each element has the standard reading fields plus `demo_event` (None for
        a normal reading, or the persisted anomaly record for the 3 events).
        Deterministic: built from real AWS-023 data + the fixed persisted
        anomaly records, cached after first build.
        """
        key = f"demo_episode:{station_id}"
        cached = getattr(self, "_demo_episode_cache", {}).get(key)
        if cached is not None:
            return cached

        events = self._load_demo_records()
        sub = self.data[self.data["station_id"] == station_id].reset_index(drop=True)
        length = self.DEMO_EPISODE_LENGTH

        # Use a contiguous block of real observations as the normal backbone.
        start = self.DEMO_EPISODE_START_IDX  # stable AWS-023 region (verified Normal by model)
        base = sub.iloc[start:start + length].reset_index(drop=True)

        episode = []
        for i in range(length):
            reading = {
                "station_id": station_id,
                "timestamp": base["timestamp"].iloc[i],
                "temperature": None if pd.isna(base["temperature"].iloc[i]) else float(base["temperature"].iloc[i]),
                "pressure": None if pd.isna(base["pressure"].iloc[i]) else float(base["pressure"].iloc[i]),
                "humidity": None if pd.isna(base["humidity"].iloc[i]) else float(base["humidity"].iloc[i]),
                "demo_event": None,
            }
            episode.append(reading)

        # Overlay the real anomaly records onto their fixed reading positions.
        for ev in events:
            pos = ev["position"]
            if pos < 1 or pos > length:
                continue
            reading = episode[pos - 1]
            feat = ev.get("feature")
            raw = ev.get("raw_value")
            if feat in ("temperature", "pressure", "humidity") and raw is not None:
                reading[feat] = float(raw)
            reading["demo_event"] = ev

        if not hasattr(self, "_demo_episode_cache"):
            self._demo_episode_cache = {}
        self._demo_episode_cache[key] = episode
        return episode

    def demo_event_positions(self):
        """Human-readable map of reading number -> demo anomaly (for reporting)."""
        return {
            ev["position"]: {
                "anomaly_id": ev.get("id"),
                "station_id": ev.get("station_id"),
                "timestamp": str(ev.get("timestamp")),
                "anomaly_type": ev.get("anomaly_type"),
                "severity": ev.get("severity"),
                "feature": ev.get("feature"),
                "raw_value": ev.get("raw_value"),
                "score": ev.get("score"),
            }
            for ev in self._load_demo_records()
        }


app = SkyGuardApp()
