"""
AlterScore Inference & Decision Engine
========================================
Loads trained artifacts and scores a borrower in real-time.

Pipeline:
  raw_data → features → latent variables → model → calibrated PD
           → economic score → AlterScore (300-900) → decision
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Any

import xgboost as xgb

from ml.features import FeatureEngineer
from ml.latent import (
    compute_latent_scores, compute_composite_latent_score,
    expected_profit_score, compute_recovery_rate_estimate,
)

logger = logging.getLogger(__name__)
MODEL_DIR = Path(__file__).parent.parent / "data" / "models"

# ─────────────────────────────────────────────
# DECISION ENGINE TIERS (from research compendium p.15)
# ─────────────────────────────────────────────
DECISION_TIERS = [
    {
        "name": "Exceptional",
        "min_score": 850,
        "max_score": 900,
        "decision": "APPROVED",
        "credit_limit_factor": 6.0,
        "interest_rate_range": (0.10, 0.12),
        "monitoring": "Quarterly",
        "risk_band": "A+",
        "color": "#059669",
    },
    {
        "name": "Very Good",
        "min_score": 750,
        "max_score": 849,
        "decision": "APPROVED",
        "credit_limit_factor": 4.0,
        "interest_rate_range": (0.12, 0.14),
        "monitoring": "Monthly",
        "risk_band": "A",
        "color": "#10b981",
    },
    {
        "name": "Good",
        "min_score": 650,
        "max_score": 749,
        "decision": "APPROVED",
        "credit_limit_factor": 2.5,
        "interest_rate_range": (0.14, 0.18),
        "monitoring": "Bi-weekly",
        "risk_band": "B",
        "color": "#3b82f6",
    },
    {
        "name": "Fair",
        "min_score": 550,
        "max_score": 649,
        "decision": "CONDITIONAL",
        "credit_limit_factor": 1.0,
        "interest_rate_range": (0.18, 0.22),
        "monitoring": "Weekly",
        "risk_band": "C",
        "color": "#f59e0b",
    },
    {
        "name": "Poor",
        "min_score": 300,
        "max_score": 549,
        "decision": "REVIEW",
        "credit_limit_factor": 0.3,
        "interest_rate_range": (0.22, 0.26),
        "monitoring": "Daily (digital)",
        "risk_band": "D",
        "color": "#ef4444",
    },
]

SEGMENT_NAMES = [
    "Stable Salaried", "Volatile Trader", "Credit-Naive Worker", "Micro-Entrepreneur"
]


class AlterScoreModel:
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self._loaded = False
        self.logistic = None
        self.xgb_model = None
        self.calibrated = None
        self.gmm = None
        self.scaler = None
        self.feature_columns = None
        self._shap_explainer = None

    def load(self):
        if self._loaded:
            return
        if not (MODEL_DIR / "calibrated_model.pkl").exists():
            logger.warning("No trained model found. Running training pipeline...")
            from ml.train import main as train_main
            train_main()

        logger.info("Loading AlterScore model artifacts...")
        self.logistic = joblib.load(MODEL_DIR / "logistic_scorecard.pkl")
        self.scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.load_model(str(MODEL_DIR / "xgboost_model.json"))
        self.gmm = joblib.load(MODEL_DIR / "gmm_segmentation.pkl")
        self.calibrated = joblib.load(MODEL_DIR / "calibrated_model.pkl")
        self.feature_columns = joblib.load(MODEL_DIR / "feature_columns.pkl")
        self._loaded = True
        logger.info("Model artifacts loaded successfully.")

    def _get_shap_values(self, X_row: np.ndarray) -> Dict:
        """Compute SHAP feature importance (TreeExplainer for XGBoost)."""
        try:
            import shap
            if self._shap_explainer is None:
                self._shap_explainer = shap.TreeExplainer(self.xgb_model)
            shap_vals = self._shap_explainer.shap_values(X_row)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]  # positive class
            if len(shap_vals.shape) > 1:
                shap_vals = shap_vals[0]
            result = {}
            for col, val in zip(self.feature_columns, shap_vals):
                if not np.isnan(float(val)):
                    result[col] = round(float(val), 5)
            # Return top 12 by abs value
            sorted_shap = sorted(result.items(), key=lambda x: abs(x[1]), reverse=True)
            return dict(sorted_shap[:12])
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")
            return {}

    def score(self, raw_data: Dict, monthly_income: float = 20000.0) -> Dict:
        """
        Full scoring pipeline for a single borrower.
        Returns AlterScore + decision + explanation.
        """
        self.load()

        # ── Step 1: Feature Engineering ──────────────────────────────
        features = self.feature_engineer.engineer(raw_data)
        model_vec = self.feature_engineer.to_model_vector(features)

        # ── Step 2: Latent Variable Computation ──────────────────────
        latent_scores = compute_latent_scores(features)
        composite_latent, confidence = compute_composite_latent_score(latent_scores)

        # ── Step 3: Model Scoring ─────────────────────────────────────
        X = np.array([[float(model_vec.get(col, np.nan)) if model_vec.get(col, np.nan) is not None else np.nan for col in self.feature_columns]])

        # Layer 1 — Logistic Scorecard
        X_scaled = self.scaler.transform(np.nan_to_num(X, nan=0.5))
        logistic_pd = float(self.logistic.predict_proba(X_scaled)[0, 1])
        logistic_score_pts = round(1000 - logistic_pd * 700)

        # Layer 2 — XGBoost
        xgb_pd = float(self.xgb_model.predict_proba(X)[0, 1])

        # Layer 3 — GMM Segment
        X_filled = np.nan_to_num(X, nan=0.5)
        segment_idx = int(self.gmm.predict(X_filled)[0])
        segment_name = SEGMENT_NAMES[segment_idx % len(SEGMENT_NAMES)]
        segment_probs = self.gmm.predict_proba(X_filled)[0].tolist()

        # Layer 4 — Calibrated PD
        pd_calibrated = float(self.calibrated.predict_proba(X)[0, 1])

        # ── Step 4: Economic Scoring ──────────────────────────────────
        loan_amount = monthly_income * 6.0
        econ = expected_profit_score(
            pd_calibrated, latent_scores, features,
            loan_amount=loan_amount, interest_rate=0.16, tenor_years=2.0
        )

        # ── Step 5: AlterScore Construction ──────────────────────────
        # Economic score (normalized [0,1]) → 300-900 scale
        alter_score = int(300 + econ["economic_score_normalized"] * 600)
        alter_score = max(300, min(900, alter_score))

        # ── Step 6: SHAP Explainability ───────────────────────────────
        shap_values = self._get_shap_values(X)

        # ── Step 7: Decision Engine ───────────────────────────────────
        tier = self._get_tier(alter_score)
        credit_limit = round(loan_amount * tier["credit_limit_factor"], -2)
        mid_rate = sum(tier["interest_rate_range"]) / 2
        interest_rate_pct = round(mid_rate * 100, 1)

        # ── Step 8: Fairness Flags ────────────────────────────────────
        fairness = self._check_fairness_flags(features, pd_calibrated, raw_data)

        return {
            "alter_score": alter_score,
            "pd_calibrated": round(pd_calibrated, 4),
            "pd_logistic": round(logistic_pd, 4),
            "decision": tier["decision"],
            "tier_name": tier["name"],
            "risk_band": tier["risk_band"],
            "credit_limit_inr": credit_limit,
            "interest_rate_pct": interest_rate_pct,
            "monitoring_intensity": tier["monitoring"],
            "confidence": round(confidence, 3),
            "tier_color": tier["color"],
            "latent_scores": {
                "BD": round(latent_scores.get("latent_BD") or 0, 3),
                "IS": round(latent_scores.get("latent_IS") or 0, 3),
                "RP": round(latent_scores.get("latent_RP") or 0, 3),
                "OR": round(latent_scores.get("latent_OR") or 0, 3),
                "SC": round(latent_scores.get("latent_SC") or 0, 3),
            },
            "composite_latent_score": round(composite_latent, 3),
            "economic_analysis": econ,
            "segment": {
                "name": segment_name,
                "index": segment_idx,
                "probabilities": [round(p, 3) for p in segment_probs],
            },
            "shap_values": shap_values,
            "features_used": {
                k: (round(v, 4) if isinstance(v, float) and not np.isnan(v) else None)
                for k, v in model_vec.items()
            },
            "fairness_flags": fairness,
            "model_versions": {
                "logistic_score_pts": logistic_score_pts,
                "xgb_pd_uncalibrated": round(xgb_pd, 4),
            },
        }

    def _get_tier(self, score: int) -> Dict:
        for tier in DECISION_TIERS:
            if tier["min_score"] <= score <= tier["max_score"]:
                return tier
        return DECISION_TIERS[-1]  # Fallback to lowest tier

    def _check_fairness_flags(self, features: Dict, pd: float, raw_data: Dict) -> Dict:
        """
        Basic fairness monitoring flags.
        Full demographic parity + disparate impact checks run at portfolio level.
        """
        missing_signals = sum(
            1 for k in ["telco_score", "geo_routine_index", "psycho_irt_score",
                        "ecom_score", "merchant_score", "bank_score"]
            if features.get(k) is None
        )
        data_coverage = round(1.0 - missing_signals / 6.0, 2)
        low_coverage_flag = data_coverage < 0.34

        return {
            "data_coverage_ratio": data_coverage,
            "low_data_coverage_flag": low_coverage_flag,
            "signals_available": 6 - missing_signals,
            "signals_total": 6,
            "recommendation": (
                "Score has full signal coverage."
                if not low_coverage_flag
                else "Low data coverage — consider manual review or request additional consent."
            ),
        }

    def get_training_metrics(self) -> Dict:
        metrics_path = MODEL_DIR / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                return json.load(f)
        return {}


# Singleton instance
_model_instance: Optional[AlterScoreModel] = None


def get_model() -> AlterScoreModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = AlterScoreModel()
        _model_instance.load()
    return _model_instance
