"""Sensor health, analytics, model, evaluation, simulation routes."""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import models
from ..database import get_db
from ..services import app as skyguard

router = APIRouter()


def _json(evidence):
    """Serialize a structured evidence list for the anomalies.evidence_json column."""
    import json

    if not evidence:
        return None
    try:
        return json.dumps(evidence, default=float)
    except (TypeError, ValueError):
        return None


@router.get("/sensor-health/{station_id}")
def sensor_health(station_id: str):
    if station_id not in {s["station_id"] for s in skyguard.stations()}:
        raise HTTPException(404, "Station not found")
    h = dict(skyguard.sensor_health(station_id))
    score = h.get("health_score", 0.0)
    if score > 70:
        h["health_status"] = "NORMAL"
    elif score >= 40:
        h["health_status"] = "WARNING"
    else:
        h["health_status"] = "CRITICAL"
    # degradation analysis from health history
    history = [h["health_score"] for h in [h]]
    deg = skyguard.degrade(h, history)
    return {**h, "degradation": deg}


@router.get("/analytics")
def analytics(source: str = "HISTORICAL", db: Session = Depends(get_db)):
    """PRODUCTION analytics aggregated across the 24 stations.

    Deliberately does NOT expose controlled-evaluation metrics (Recall, Precision,
    F1, ROC-AUC, confusion matrix); those belong exclusively to the Evaluation page.
    """
    return skyguard.production_analytics(source=source, db=db)


@router.get("/model/metadata")
def model_metadata():
    """Persisted production model + training metadata (Model page)."""
    return skyguard.model_metadata()


@router.get("/model/info")
def model_info():
    return {
        "name": "SkyGuard AI Anomaly Engine",
        "pipeline": [
            "1. Data preprocessing / validation",
            "2. Temporal feature extraction",
            "3. Multivariate analysis",
            "4. Isolation Forest (baseline)",
            "5. Temporal isolation forest (research #1)",
            "6. Multivariate isolation forest (research #2)",
            "7. Anomaly scoring & fusion",
            "8. Classification",
            "9. Confidence scoring",
            "10. Severity assessment",
            "11. Weather-event vs sensor-fault",
            "12. Sensor health",
            "13. Corrected value imputation",
            "14. SHAP explainability",
        ],
        "variables": ["temperature", "pressure", "humidity"],
        "note": "All AI outputs are computed from the actual trained models and never hardcoded.",
    }


@router.get("/model/evaluation")
def model_evaluation():
    ev = skyguard.run_evaluation()
    comparison = skyguard.model_comparison()
    return {
        "performance": ev["full_model"],
        "by_type": ev["by_type"],
        "comparison": comparison["models"],
        "selected_model": comparison["selected"],
        "comparison_provenance": comparison.get("provenance"),
        "model_versions": comparison.get("versions"),
        "detailed": ev.get("detailed") or ev["full_model"],
        "dataset": ev.get("dataset"),
        "per_category": ev.get("per_category"),
        "multiclass_confusion_matrix": ev.get("multiclass_confusion_matrix"),
    }


@router.post("/simulation/inject")
def simulate(payload: dict, db: Session = Depends(get_db)):
    res = skyguard.simulate(payload)
    # persist a simulation event record
    ev = models.SimulationEvent(
        station_id=payload.get("station_id"),
        variable=payload.get("variable"),
        anomaly_type=payload.get("anomaly_type"),
        magnitude=payload.get("magnitude"),
        duration=payload.get("duration", 1),
        detected=res.get("detected", False),
    )
    db.add(ev)
    # persist the anomaly if detected
    record = res.get("result")
    if record:
        # evidence is a single factual string from the heuristic (not a list of
        # votes); storage keeps the legacy column name for DB compatibility, the
        # API exposes it as sensor_fault_likelihood (see anomalies._ser)
        evidence = record.get("event_evidence")
        if isinstance(evidence, (list, tuple)):
            evidence = "\n".join(evidence)
        an = models.Anomaly(
            station_id=record.get("station_id"),
            timestamp=record.get("timestamp") or datetime.utcnow(),
            anomaly_type=record["anomaly_type"],
            confidence=record["confidence"],
            severity=record["severity"],
            cause=record.get("event_assessment") or record.get("cause"),
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
            event_evidence=evidence,
            evidence_json=_json(record.get("evidence")),
        )
        db.add(an)
        db.commit()
        db.refresh(an)
        res["anomaly_id"] = an.id
        res["result"]["anomaly_id"] = an.id
    else:
        db.commit()
    return res
