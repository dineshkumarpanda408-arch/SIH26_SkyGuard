"""Pydantic schemas for API request/response validation."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class ReadingIn(BaseModel):
    station_id: str = Field(..., min_length=1)
    timestamp: Optional[datetime] = None
    temperature: Optional[float] = None
    pressure: Optional[float] = None
    humidity: Optional[float] = None

    @classmethod
    def _check(cls, values):
        if all(
            values.get(k) is None for k in ("temperature", "pressure", "humidity")
        ):
            raise ValueError("At least one of temperature/pressure/humidity required")
        return values


class StationOut(BaseModel):
    station_id: str
    name: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    active: bool = True

    class Config:
        from_attributes = True


class ReadingOut(BaseModel):
    id: int
    station_id: str
    timestamp: datetime
    temperature: Optional[float] = None
    pressure: Optional[float] = None
    humidity: Optional[float] = None
    is_injected: bool = False
    ground_truth: bool = False
    is_anomaly: bool = False
    anomaly_score: Optional[float] = None

    class Config:
        from_attributes = True


class AnomalyOut(BaseModel):
    id: int
    station_id: str
    timestamp: datetime
    anomaly_type: str
    confidence: float
    severity: str
    cause: Optional[str] = None
    is_weather_event: bool = False
    event_assessment: Optional[str] = None
    corrected_value: Optional[float] = None
    feature: Optional[str] = None
    raw_value: Optional[float] = None
    expected_value: Optional[float] = None
    score: Optional[float] = None

    class Config:
        from_attributes = True


class SensorHealthOut(BaseModel):
    station_id: str
    health_score: float
    health_status: str
    data_quality: float
    anomaly_frequency: float
    communication: float
    stability: float
    drift: float
    trend: str

    class Config:
        from_attributes = True


class SimulationIn(BaseModel):
    station_id: str
    variable: str
    anomaly_type: str
    magnitude: Optional[float] = None
    duration: int = Field(1, ge=1, le=1000)
    start_time: Optional[datetime] = None


class SimulationOut(BaseModel):
    message: str
    injected_readings: int = 0
    detected: bool = False
    confidence: Optional[float] = None
    severity: Optional[str] = None
    anomaly_type: Optional[str] = None
    latency_ms: Optional[float] = None


class ExplanationOut(BaseModel):
    anomaly_id: int
    values: List[dict]
    narrative: str
    method: Optional[str] = None


class HealthFactorsOut(BaseModel):
    station_id: str
    health_score: float
    health_status: str
    factors: dict
    trend: str
    degradation: dict
