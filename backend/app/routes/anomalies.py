"""Anomaly + explanation routes."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from .. import models, schemas
from ..database import get_db
from ..services import app as skyguard

router = APIRouter()


@router.get("/anomalies")
def list_anomalies(
    station_id: str | None = None,
    severity: str | None = None,
    anomaly_type: str | None = None,
    limit: int = Query(100, le=1000),
    db: Session = Depends(get_db),
):
    q = db.query(models.Anomaly).order_by(models.Anomaly.timestamp.desc())
    if station_id:
        q = q.filter(models.Anomaly.station_id == station_id)
    if severity:
        q = q.filter(models.Anomaly.severity == severity.upper())
    if anomaly_type:
        q = q.filter(models.Anomaly.anomaly_type == anomaly_type.upper())
    rows = q.limit(limit).all()
    return [_ser(r) for r in rows]


@router.get("/anomalies/{anomaly_id}")
def get_anomaly(anomaly_id: int, db: Session = Depends(get_db)):
    an = db.query(models.Anomaly).filter(models.Anomaly.id == anomaly_id).first()
    if not an:
        raise HTTPException(404, "Anomaly not found")
    return _ser(an)


@router.get("/anomalies/{anomaly_id}/timeline")
def get_anomaly_timeline(anomaly_id: int, db: Session = Depends(get_db)):
    an = db.query(models.Anomaly).filter(models.Anomaly.id == anomaly_id).first()
    if not an:
        raise HTTPException(404, "Anomaly not found")
    return skyguard.anomaly_timeline(an)


@router.get("/explanations/{anomaly_id}", response_model=schemas.ExplanationOut)
def get_explanation(anomaly_id: int, db: Session = Depends(get_db)):
    an = db.query(models.Anomaly).filter(models.Anomaly.id == anomaly_id).first()
    if not an:
        raise HTTPException(404, "Anomaly not found")
    exp = db.query(models.Explanation).filter(models.Explanation.anomaly_id == anomaly_id).first()
    if not exp:
        # build a real explanation using the model + data
        values, narrative, method = skyguard.explain_anomaly(an)
        exp = models.Explanation(anomaly_id=anomaly_id, values=json_dumps(values), narrative=narrative, method=method)
        db.add(exp)
        db.commit()
    import json
    return {
        "anomaly_id": anomaly_id,
        "values": json.loads(exp.values),
        "narrative": exp.narrative,
        "method": exp.method or "shap",
    }


def _ser(an: models.Anomaly) -> dict:
    score = round(float(an.score), 4) if an.score is not None else (round(float(an.confidence), 4) if an.confidence is not None else None)
    confidence = round(float(an.confidence), 4) if an.confidence is not None else score
    return {
        "id": an.id,
        "station_id": an.station_id,
        "timestamp": an.timestamp,
        "anomaly_type": an.anomaly_type,
        "anomaly_type_source": "rule/classifier",
        "computed_by": {
            "confidence": "ml:model_fusion",
            "score": "ml:model_fusion",
            "anomaly_type": "rule/classifier (not ML)",
            "event_assessment": "statistical heuristic (not ML)",
            "sensor_fault_likelihood": "statistical heuristic (not ML)",
            "corrected_value": "statistical diurnal estimate (not ML)",
        },
        "confidence": confidence,
        "severity": an.severity,
        "cause": an.cause,
        "is_weather_event": an.is_weather_event,
        "event_assessment": an.event_assessment,
        "corrected_value": an.corrected_value,
        "correction_method": an.correction_method,
        "correction_confidence": an.correction_confidence,
        "feature": an.feature,
        "raw_value": an.raw_value,
        "expected_value": an.expected_value,
        "score": score,
        "sensor_fault_likelihood": an.prob_sensor_fault,  # legacy column name, API name is the honest one
        "event_evidence": an.event_evidence,
        "evidence": _evidence_from_row(an),
        "anomaly_probability": None,
        "calibration_status": "uncalibrated",
    }


def _evidence_from_row(an: models.Anomaly):
    """Structured evidence list for an anomaly response.

    Prefers the persisted structured evidence (evidence_json). Falls back to a
    minimal list built from the fields actually stored on LEGACY rows so an old
    record is never presented as having richer evidence than it truly carries.
    """
    import json

    raw = getattr(an, "evidence_json", None)
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
        except (TypeError, ValueError):
            pass

    threshold = getattr(getattr(skyguard, "model", None), "threshold", 0.72)
    score = float(an.score) if an.score is not None else (float(an.confidence) if an.confidence is not None else None)
    if score is None:
        return None

    from ..ml.evidence import fallback_evidence
    return fallback_evidence(
        score=score,
        threshold=threshold,
        feature=getattr(an, "feature", None),
        event_assessment=getattr(an, "event_assessment", None),
        is_weather_event=bool(getattr(an, "is_weather_event", False)),
        sensor_fault_likelihood=getattr(an, "prob_sensor_fault", None),
        model_version=getattr(skyguard, "model_version", "skyguard-multivariate-v3"),
    )


def json_dumps(values):
    import json
    return json.dumps(values, default=float)
