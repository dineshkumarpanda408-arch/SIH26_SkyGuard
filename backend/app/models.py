"""SQLAlchemy ORM models for all persisted entities.

Uses explicit Column() definitions for SQLAlchemy 2.x compatibility.
"""

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from .database import Base


class Station(Base):
    __tablename__ = "stations"

    id = Column(Integer, primary_key=True)
    station_id = Column(String(50), unique=True, index=True)
    name = Column(String(100), nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    active = Column(Boolean, default=True)


class Reading(Base):
    __tablename__ = "readings"

    id = Column(Integer, primary_key=True)
    station_id = Column(String(50), index=True)
    timestamp = Column(DateTime, index=True)
    temperature = Column(Float, nullable=True)
    pressure = Column(Float, nullable=True)
    humidity = Column(Float, nullable=True)
    is_injected = Column(Boolean, default=False)
    injected_type = Column(String(50), nullable=True)
    ground_truth = Column(Boolean, default=False)
    is_anomaly = Column(Boolean, default=False)
    anomaly_score = Column(Float, nullable=True)


class Anomaly(Base):
    __tablename__ = "anomalies"

    id = Column(Integer, primary_key=True)
    reading_id = Column(Integer, ForeignKey("readings.id"), nullable=True)
    station_id = Column(String(50), index=True)
    timestamp = Column(DateTime, index=True)
    anomaly_type = Column(String(50))
    confidence = Column(Float)
    severity = Column(String(20))
    cause = Column(String(200), nullable=True)
    is_weather_event = Column(Boolean, default=False)
    event_assessment = Column(String(50), nullable=True)
    corrected_value = Column(Float, nullable=True)
    correction_method = Column(String(100), nullable=True)
    correction_confidence = Column(Float, nullable=True)
    feature = Column(String(50))
    score = Column(Float, nullable=True)
    raw_value = Column(Float, nullable=True)
    expected_value = Column(Float, nullable=True)
    prob_sensor_fault = Column(Float, nullable=True)
    event_evidence = Column(Text, nullable=True)
    evidence_json = Column(Text, nullable=True)


class SensorHealth(Base):
    __tablename__ = "sensor_health"

    id = Column(Integer, primary_key=True)
    station_id = Column(String(50), index=True)
    timestamp = Column(DateTime, index=True)
    health_score = Column(Float)
    health_status = Column(String(20))
    data_quality = Column(Float)
    anomaly_frequency = Column(Float)
    communication = Column(Float)
    stability = Column(Float)
    drift = Column(Float)
    trend = Column(String(20))


class ModelRun(Base):
    __tablename__ = "model_runs"

    id = Column(Integer, primary_key=True)
    run_time = Column(DateTime, default=datetime.utcnow)
    model_name = Column(String(100))
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1 = Column(Float, nullable=True)
    false_positive_rate = Column(Float, nullable=True)
    false_negative_rate = Column(Float, nullable=True)
    details = Column(Text, nullable=True)


class Explanation(Base):
    __tablename__ = "explanations"

    id = Column(Integer, primary_key=True)
    anomaly_id = Column(Integer, ForeignKey("anomalies.id"), index=True)
    values = Column(Text)
    narrative = Column(Text)
    method = Column(String(40), nullable=True)


class SimulationEvent(Base):
    __tablename__ = "simulation_events"

    id = Column(Integer, primary_key=True)
    station_id = Column(String(50))
    timestamp = Column(DateTime, default=datetime.utcnow)
    variable = Column(String(50))
    anomaly_type = Column(String(50))
    magnitude = Column(Float, nullable=True)
    duration = Column(Integer, default=1)
    detected = Column(Boolean, default=False)


class User(Base):
    """Single-account WeatherLock credentials (salted PBKDF2 hashes, never plaintext)."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(80), unique=True, index=True, nullable=False)
    pin_hash = Column(String(256), nullable=False)
    pin_salt = Column(String(64), nullable=False)
    pattern_hash = Column(String(256), nullable=False)
    pattern_salt = Column(String(64), nullable=False)
    pattern_length = Column(Integer, default=4)
    created_at = Column(DateTime, default=datetime.utcnow)
