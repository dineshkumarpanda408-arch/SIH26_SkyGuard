import sys
from pathlib import Path

# Ensure the 'backend' folder is in sys.path so unpickling 'app.*' modules succeeds
# regardless of current working directory when uvicorn is launched.
_backend_dir = Path(__file__).resolve().parent.parent
_backend_dir_str = str(_backend_dir)
if _backend_dir_str not in sys.path:
    sys.path.insert(0, _backend_dir_str)

import asyncio

from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .config import CORS_ORIGINS, DEFAULT_STREAM_INTERVAL_SEC
from .database import init_db, SessionLocal
from . import models
from .routes import stations, anomalies, analytics, deep_dive, auth
from .routes.auth import require_auth, seed_default_user
from .services import app as skyguard
from .ws import manager

init_db()

app = FastAPI(title="SkyGuard AI — Intelligent AWS Anomaly Detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if "*" in CORS_ORIGINS or not CORS_ORIGINS else CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WeatherLock guards every data endpoint. Auth + health + live websocket stay open.
app.include_router(stations.router, prefix="/api", dependencies=[Depends(require_auth)])
app.include_router(anomalies.router, prefix="/api", dependencies=[Depends(require_auth)])
app.include_router(analytics.router, prefix="/api", dependencies=[Depends(require_auth)])
app.include_router(deep_dive.router, prefix="/api", dependencies=[Depends(require_auth)])
app.include_router(auth.router, prefix="/api")

# Initialise model with sample data
skyguard.initialize()


@app.get("/")
def root():
    return {
        "status": "online",
        "system": "SKYGUARD AI — Intelligent AWS Anomaly Detection",
        "health": "/api/health",
        "docs": "/docs",
    }


@app.get("/api/health")
def health():
    return {"status": "online", "system": "SKYGUARD AI", "model_trained": skyguard.model.trained}


@app.websocket("/api/live")
async def live(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # initial snapshot
        await websocket.send_json({"type": "snapshot", "stations": skyguard.stations()})

        # Deterministic demo episode: 35 AWS-023 readings with 3 real anomaly
        # records replayed at reading 13 / 23 / 33 (Normal readings in between).
        episode = skyguard.demo_episode("AWS-023")
        run = 0
        while episode:
            # play the ordered 35-reading episode, then repeat deterministically
            for idx, reading in enumerate(episode, start=1):
                ev = reading.get("demo_event")
                reading_number = (run * len(episode)) + idx
                if ev is not None:
                    # Replay the exact persisted anomaly record (same data the
                    # Anomalies page shows), so Live matches the detection result.
                    payload = {
                        "type": "reading",
                        "reading_number": reading_number,
                        "station_id": reading["station_id"],
                        "timestamp": str(reading["timestamp"]),
                        "temperature": reading["temperature"],
                        "pressure": reading["pressure"],
                        "humidity": reading["humidity"],
                        "is_anomaly": True,
                        "score": ev.get("score"),
                        "anomaly_type": ev.get("anomaly_type"),
                        "severity": ev.get("severity"),
                        "feature": ev.get("feature"),
                        "raw_value": ev.get("raw_value"),
                        "anomaly_id": ev.get("id"),
                        # Demo event: station + type match a real Anomalies-page
                        # record, but this timestamp is the source-data time of the
                        # replayed reading, NOT the record's DB timestamp. Honest
                        # type-based match, not an exact timestamp claim.
                        "demo": True,
                    }
                else:
                    # Real AWS-023 reading scored by the unchanged model -> Normal
                    result = skyguard.add_reading(reading)
                    payload = {
                        "type": "reading",
                        "reading_number": reading_number,
                        "station_id": reading["station_id"],
                        "timestamp": str(reading["timestamp"]),
                        "temperature": reading["temperature"],
                        "pressure": reading["pressure"],
                        "humidity": reading["humidity"],
                        "is_anomaly": bool(result.get("is_anomaly")),
                        "score": result.get("score"),
                    }
                await websocket.send_json(payload)
                await asyncio.sleep(DEFAULT_STREAM_INTERVAL_SEC)
            run += 1
    except WebSocketDisconnect:
        manager.disconnect(websocket)


def _persist_anomaly(record: dict) -> int:
    """Persist a detected anomaly (from WS live stream or ingestion) to the DB."""
    db = SessionLocal()
    try:
        an = models.Anomaly(
            station_id=record.get("station_id"),
            timestamp=record.get("timestamp"),
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
            event_evidence=("\n".join(record["event_evidence"]) if isinstance(record.get("event_evidence"), (list, tuple)) else record.get("event_evidence")),
            evidence_json=_json_evidence(record.get("evidence")),
        )
        db.add(an)
        db.commit()
        db.refresh(an)
        return an.id or 0
    finally:
        db.close()


def _json_evidence(evidence):
    """Serialize a structured evidence list for the anomalies.evidence_json column."""
    import json

    if not evidence:
        return None
    try:
        return json.dumps(evidence, default=float)
    except (TypeError, ValueError):
        return None


@app.on_event("startup")
async def startup():
    # Ensure the demo anomaly records exist so the Anomalies page lists the
    # same three events that the Live Monitor replays at readings 13/23/33.
    skyguard.seed_demo_anomalies()
    # Ensure the single WeatherLock account exists for the Visual Security Lock.
    seed_default_user()
