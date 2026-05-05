"""
AlterScore Database Layer
===========================
SQLAlchemy ORM + SQLite (switchable to PostgreSQL via DATABASE_URL env var).
Stores: applications, features, predictions, decisions.
"""

import os
from sqlalchemy import create_engine, Column, Integer, Float, String, JSON, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/alterscore.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    echo=False,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ─────────────────────────────────────────────
# ORM MODELS
# ─────────────────────────────────────────────

class Application(Base):
    __tablename__ = "applications"

    id = Column(Integer, primary_key=True, index=True)
    application_id = Column(String(36), unique=True, index=True)
    borrower_name = Column(String(100))
    occupation = Column(String(50))
    monthly_income_inr = Column(Float)
    state = Column(String(50), nullable=True)
    loan_purpose = Column(String(100), nullable=True)
    consents = Column(JSON)          # Which data sources were consented
    raw_data = Column(JSON)          # Ingested raw data (encrypted in prod)
    created_at = Column(DateTime, default=datetime.utcnow)


class FeatureRecord(Base):
    __tablename__ = "feature_records"

    id = Column(Integer, primary_key=True, index=True)
    application_id = Column(String(36), index=True)
    features = Column(JSON)          # All engineered features
    latent_scores = Column(JSON)     # BD, IS, RP, OR, SC
    created_at = Column(DateTime, default=datetime.utcnow)


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    application_id = Column(String(36), index=True)
    alter_score = Column(Integer)
    pd_calibrated = Column(Float)
    pd_logistic = Column(Float)
    composite_latent_score = Column(Float)
    confidence = Column(Float)
    segment_name = Column(String(50))
    shap_values = Column(JSON)
    economic_analysis = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


class Decision(Base):
    __tablename__ = "decisions"

    id = Column(Integer, primary_key=True, index=True)
    application_id = Column(String(36), index=True)
    decision = Column(String(20))        # APPROVED / CONDITIONAL / REVIEW
    tier_name = Column(String(30))
    risk_band = Column(String(5))
    credit_limit_inr = Column(Float)
    interest_rate_pct = Column(Float)
    monitoring_intensity = Column(String(30))
    fairness_flags = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    """Create all tables."""
    import os
    os.makedirs("data", exist_ok=True)
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency for DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
