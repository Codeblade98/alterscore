"""
AlterScore API Routes
======================
REST endpoints for scoring, health, metrics, and application retrieval.
"""

import uuid
import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from backend.database import get_db, Application, FeatureRecord, Prediction, Decision
from backend.schemas import (
    ScoringRequest, ScoringResponse, HealthResponse, MetricsResponse,
    LatentScores, EconomicAnalysis, SegmentInfo, FairnessFlags,
)
from ml.model import get_model

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """System health and model status."""
    try:
        model = get_model()
        loaded = model._loaded
    except Exception:
        loaded = False
    return HealthResponse(status="ok", model_loaded=loaded, version="1.0.0")


@router.get("/metrics", response_model=MetricsResponse, tags=["System"])
def get_metrics():
    """Return training metrics (AUC, KS, Brier, Gini)."""
    try:
        model = get_model()
        metrics = model.get_training_metrics()
        if not metrics:
            raise HTTPException(status_code=503, detail="Metrics not available. Run training first.")
        return MetricsResponse(**metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/score", response_model=ScoringResponse, tags=["Scoring"])
def score_borrower(
    request: ScoringRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Main scoring endpoint.
    Accepts multi-signal alternative data, returns AlterScore + decision + explanation.
    Consent flags determine which data sources are used.
    """
    application_id = str(uuid.uuid4())

    try:
        model = get_model()
        raw_data = request.to_raw_data()
        result = model.score(raw_data, monthly_income=request.borrower.monthly_income_inr)
        result["application_id"] = application_id

        # Persist in background to avoid latency
        background_tasks.add_task(
            _persist_application,
            db_url=None,
            application_id=application_id,
            request=request,
            raw_data=raw_data,
            result=result,
        )

        return ScoringResponse(
            alter_score=result["alter_score"],
            pd_calibrated=result["pd_calibrated"],
            decision=result["decision"],
            tier_name=result["tier_name"],
            risk_band=result["risk_band"],
            credit_limit_inr=result["credit_limit_inr"],
            interest_rate_pct=result["interest_rate_pct"],
            monitoring_intensity=result["monitoring_intensity"],
            confidence=result["confidence"],
            tier_color=result["tier_color"],
            latent_scores=LatentScores(**result["latent_scores"]),
            composite_latent_score=result["composite_latent_score"],
            economic_analysis=EconomicAnalysis(**result["economic_analysis"]),
            segment=SegmentInfo(**result["segment"]),
            shap_values=result["shap_values"],
            features_used=result["features_used"],
            fairness_flags=FairnessFlags(**result["fairness_flags"]),
            model_versions=result["model_versions"],
            application_id=application_id,
        )

    except Exception as e:
        logger.exception(f"Scoring failed for application {application_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Scoring error: {str(e)}")


@router.get("/applications", tags=["Applications"])
def list_applications(
    skip: int = 0,
    limit: int = 20,
    db: Session = Depends(get_db),
):
    """List recent scoring applications."""
    apps = db.query(Application).order_by(Application.created_at.desc()).offset(skip).limit(limit).all()
    decisions = {
        d.application_id: d
        for d in db.query(Decision).filter(
            Decision.application_id.in_([a.application_id for a in apps])
        ).all()
    }
    return [
        {
            "application_id": a.application_id,
            "borrower_name": a.borrower_name,
            "occupation": a.occupation,
            "monthly_income_inr": a.monthly_income_inr,
            "state": a.state,
            "loan_purpose": a.loan_purpose,
            "created_at": a.created_at.isoformat(),
            "decision": decisions.get(a.application_id, {}).decision if decisions.get(a.application_id) else None,
            "credit_limit_inr": decisions.get(a.application_id, {}).credit_limit_inr if decisions.get(a.application_id) else None,
            "tier_name": decisions.get(a.application_id, {}).tier_name if decisions.get(a.application_id) else None,
        }
        for a in apps
    ]


@router.get("/applications/{application_id}", tags=["Applications"])
def get_application(application_id: str, db: Session = Depends(get_db)):
    """Get full details for a specific application."""
    app = db.query(Application).filter(Application.application_id == application_id).first()
    if not app:
        raise HTTPException(status_code=404, detail="Application not found")

    prediction = db.query(Prediction).filter(Prediction.application_id == application_id).first()
    decision = db.query(Decision).filter(Decision.application_id == application_id).first()
    features = db.query(FeatureRecord).filter(FeatureRecord.application_id == application_id).first()

    return {
        "application": {
            "id": app.application_id,
            "borrower_name": app.borrower_name,
            "occupation": app.occupation,
            "monthly_income_inr": app.monthly_income_inr,
            "state": app.state,
            "loan_purpose": app.loan_purpose,
            "consents": app.consents,
            "created_at": app.created_at.isoformat(),
        },
        "features": features.features if features else {},
        "latent_scores": features.latent_scores if features else {},
        "prediction": {
            "alter_score": prediction.alter_score if prediction else None,
            "pd_calibrated": prediction.pd_calibrated if prediction else None,
            "segment_name": prediction.segment_name if prediction else None,
            "shap_values": prediction.shap_values if prediction else {},
            "economic_analysis": prediction.economic_analysis if prediction else {},
        } if prediction else {},
        "decision": {
            "decision": decision.decision if decision else None,
            "tier_name": decision.tier_name if decision else None,
            "risk_band": decision.risk_band if decision else None,
            "credit_limit_inr": decision.credit_limit_inr if decision else None,
            "interest_rate_pct": decision.interest_rate_pct if decision else None,
            "monitoring_intensity": decision.monitoring_intensity if decision else None,
        } if decision else {},
    }


def _persist_application(db_url, application_id, request, raw_data, result):
    """Background task: persist scoring results to database."""
    from backend.database import SessionLocal
    db = SessionLocal()
    try:
        # Application record
        app = Application(
            application_id=application_id,
            borrower_name=request.borrower.name,
            occupation=request.borrower.occupation,
            monthly_income_inr=request.borrower.monthly_income_inr,
            state=request.borrower.state,
            loan_purpose=request.borrower.loan_purpose,
            consents=request.consent.model_dump(),
            raw_data={k: "ENCRYPTED" for k in raw_data.keys()},  # Never store raw PII
        )
        db.add(app)

        # Feature record
        from ml.features import FeatureEngineer
        fe = FeatureEngineer()
        features = fe.engineer(raw_data)
        from ml.latent import compute_latent_scores
        latent = compute_latent_scores(features)
        feat_rec = FeatureRecord(
            application_id=application_id,
            features={k: v for k, v in features.items() if not k.startswith("latent_")},
            latent_scores=result["latent_scores"],
        )
        db.add(feat_rec)

        # Prediction record
        pred = Prediction(
            application_id=application_id,
            alter_score=result["alter_score"],
            pd_calibrated=result["pd_calibrated"],
            pd_logistic=result.get("pd_logistic", 0),
            composite_latent_score=result["composite_latent_score"],
            confidence=result["confidence"],
            segment_name=result["segment"]["name"],
            shap_values=result["shap_values"],
            economic_analysis=result["economic_analysis"],
        )
        db.add(pred)

        # Decision record
        dec = Decision(
            application_id=application_id,
            decision=result["decision"],
            tier_name=result["tier_name"],
            risk_band=result["risk_band"],
            credit_limit_inr=result["credit_limit_inr"],
            interest_rate_pct=result["interest_rate_pct"],
            monitoring_intensity=result["monitoring_intensity"],
            fairness_flags=result["fairness_flags"],
        )
        db.add(dec)
        db.commit()
    except Exception as e:
        logger.error(f"Failed to persist application {application_id}: {e}")
        db.rollback()
    finally:
        db.close()
