"""API route handlers."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from .. import models, schemas
from ..database import get_db
from ..services import app as skyguard

router = APIRouter()


@router.get("/stations/predictions")
def all_predictions():
    """Bulk canonical production predictions for all 24 stations."""
    return skyguard.predictions()


@router.get("/stations", response_model=list[schemas.StationOut])
def list_stations(db: Session = Depends(get_db)):
    return skyguard.stations()


@router.get("/stations/{station_id}", response_model=schemas.StationOut)
def get_station(station_id: str, db: Session = Depends(get_db)):
    for s in skyguard.stations():
        if s["station_id"] == station_id:
            return s
    raise HTTPException(404, "Station not found")


@router.get("/stations/{station_id}/predict")
def predict_station(station_id: str):
    pred = skyguard.predict_station(station_id)
    if pred is None:
        raise HTTPException(404, "Station not found")
    return pred


@router.get("/readings")
def list_readings(
    station_id: str | None = None,
    limit: int = Query(200, le=5000),
    db: Session = Depends(get_db),
):
    q = db.query(models.Reading).order_by(models.Reading.timestamp.desc())
    if station_id:
        q = q.filter(models.Reading.station_id == station_id)
    rows = q.limit(limit).all()
    return [
        {
            "id": r.id,
            "station_id": r.station_id,
            "timestamp": r.timestamp,
            "temperature": r.temperature,
            "pressure": r.pressure,
            "humidity": r.humidity,
        }
        for r in rows
    ]


@router.post("/readings")
def ingest_reading(payload: schemas.ReadingIn, db: Session = Depends(get_db)):
    result = skyguard.add_reading(payload.dict())
    if result.get("is_anomaly"):
        an = models.Anomaly(
            station_id=result["station_id"],
            timestamp=result["timestamp"],
            anomaly_type=result["anomaly_type"],
            confidence=result["confidence"],
            severity=result["severity"],
            cause=result.get("event_assessment") or result.get("cause"),
            is_weather_event=result.get("is_weather_event", False),
            event_assessment=result.get("event_assessment"),
            corrected_value=result.get("corrected_value"),
            correction_method=result.get("correction_method"),
            correction_confidence=result.get("correction_confidence"),
            feature=result.get("feature"),
            score=result.get("score"),
            raw_value=result.get("raw_value"),
            expected_value=result.get("expected_value"),
            prob_sensor_fault=result.get("sensor_fault_likelihood"),
            event_evidence=("\n".join(result["event_evidence"]) if isinstance(result.get("event_evidence"), (list, tuple)) else result.get("event_evidence")),
            evidence_json=_json(result.get("evidence")),
        )
        db.add(an)
        db.commit()
        db.refresh(an)
        result["anomaly_id"] = an.id
    # persist reading
    reading = models.Reading(
        station_id=result["station_id"],
        timestamp=result["timestamp"],
        temperature=payload.temperature,
        pressure=payload.pressure,
        humidity=payload.humidity,
        ground_truth=payload.ground_truth if hasattr(payload, "ground_truth") else False,
    )
    db.add(reading)
    db.commit()
    return result


def _json(evidence):
    """Serialize a structured evidence list for the anomalies.evidence_json column."""
    import json

    if not evidence:
        return None
    try:
        return json.dumps(evidence, default=float)
    except (TypeError, ValueError):
        return None
