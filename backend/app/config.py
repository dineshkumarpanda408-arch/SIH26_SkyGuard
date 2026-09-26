"""Application configuration loaded from environment variables."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'skyguard.db'}")

DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR.parent / "data")))
DATA_FILE = DATA_DIR / os.getenv("DATA_FILE", "sample_weather.csv")

CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS", "http://localhost:5173,http://localhost:3000"
).split(",")

# Model storage
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(BASE_DIR / "models")))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Demo / simulation
DEFAULT_STREAM_INTERVAL_SEC = float(os.getenv("STREAM_INTERVAL_SEC", "2.0"))
ENABLE_CORRECTED_VALUES = os.getenv("ENABLE_CORRECTED_VALUES", "true").lower() == "true"

# Sensitivity for detection thresholds
NORMAL_CONTAMINATION = float(os.getenv("NORMAL_CONTAMINATION", "0.05"))

# Dataset provenance (persisted into model metadata, never hard-coded in API)
DATASET_VERSION = os.getenv("DATASET_VERSION", "sample_weather_v2")

# Calibration: kept disabled until a validated calibration split proves
# Brier/reliability improvement, per PHASE 3 (P). Do not enable silently.
CALIBRATION_METHOD = os.getenv("CALIBRATION_METHOD", "none")
