"""Database engine and session management (SQLAlchemy)."""

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from .config import DATABASE_URL

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    from . import models  # noqa: F401  ensure models are registered

    Base.metadata.create_all(bind=engine)
    _apply_light_migrations()


def _apply_light_migrations():
    """Idempotently add new columns to existing SQLite tables (no data loss).

    ``create_all`` only creates missing tables and will not add new columns to a
    table that already exists, so we ALTER TABLE the columns back in when they
    are absent. Safe to run on every startup.
    """
    if not DATABASE_URL.startswith("sqlite"):
        return
    import sqlalchemy as sa
    insp = sa.inspect(engine)
    tables = insp.get_table_names()
    if "anomalies" not in tables:
        return
    existing = {c["name"] for c in insp.get_columns("anomalies")}
    additions = {
        "prob_sensor_fault": "FLOAT",
        "event_evidence": "TEXT",
        "evidence_json": "TEXT",
    }
    with engine.begin() as conn:
        for col, typ in additions.items():
            if col not in existing:
                conn.exec_driver_sql(f"ALTER TABLE anomalies ADD COLUMN {col} {typ}")

    if "explanations" in tables:
        existing_exp = {c["name"] for c in insp.get_columns("explanations")}
        if "method" not in existing_exp:
            with engine.begin() as conn:
                conn.exec_driver_sql("ALTER TABLE explanations ADD COLUMN method TEXT")

