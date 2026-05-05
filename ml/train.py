"""
AlterScore Model Training Script
===================================
Generates synthetic data + trains 4-layer model architecture:
  1. Logistic Scorecard (interpretable baseline)
  2. XGBoost Classifier (main performance layer)
  3. GMM Clustering (latent segmentation)
  4. Isotonic Calibration (probability calibration)

Run: python -m ml.train
"""

import os
import sys
import json
import joblib
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy.stats import ks_2samp
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = [
    "telco_days_late_mean", "telco_suspensions_count", "telco_recharge_regularity_cv",
    "telco_advance_ratio", "telco_score",
    "geo_home_stasis_pct", "geo_location_entropy", "geo_routine_index",
    "geo_unique_locations", "geo_radius_of_gyration_km",
    "psycho_irt_score", "psycho_conscientiousness", "psycho_future_orientation",
    "psycho_consistency_flag", "psycho_response_time_flag",
    "ecom_utility_spend_ratio", "ecom_purchase_interval_cv", "ecom_return_rate",
    "ecom_advance_payment_ratio", "ecom_total_transactions",
    "merchant_bayesian_rating", "merchant_review_volume", "merchant_sentiment_score",
    "merchant_rating_trend", "merchant_closure_risk",
    "bank_inflow_cv", "bank_inflow_mean", "bank_min_monthly_balance",
    "bank_avg_balance", "bank_inflow_outflow_ratio",
]

# Multi-class labels: 0=on-time, 1=early-delinquent, 2=late-delinquent, 3=default
LABEL_MAP = {0: "on_time", 1: "early_delinquent", 2: "late_delinquent", 3: "default"}


def generate_synthetic_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic synthetic training data.
    Data-generating process reflects the latent variable theory:
    borrowers are drawn from 4 archetypes (GMM segments).
    """
    rng = np.random.RandomState(seed)
    rows = []

    # 4 borrower archetypes reflecting GMM segmentation theory
    archetypes = [
        # (name, weight, BD_mean, IS_mean, RP_mean, default_prob)
        ("stable_salaried",   0.30, 0.80, 0.75, 0.65, 0.04),
        ("volatile_trader",   0.25, 0.55, 0.45, 0.50, 0.18),
        ("credit_naive",      0.25, 0.65, 0.60, 0.60, 0.10),
        ("micro_entrepreneur",0.20, 0.70, 0.55, 0.70, 0.12),
    ]

    for archetype_name, weight, bd_m, is_m, rp_m, base_pd in archetypes:
        n = int(n_samples * weight)
        for _ in range(n):
            bd = np.clip(rng.normal(bd_m, 0.15), 0.05, 0.98)
            is_ = np.clip(rng.normal(is_m, 0.18), 0.05, 0.98)
            rp = np.clip(rng.normal(rp_m, 0.15), 0.05, 0.98)

            # Feature generation tied to latent variables
            days_late = max(0, rng.exponential(5 * (1 - bd)))
            suspensions = rng.poisson(2 * (1 - bd))
            recharge_cv = np.clip(rng.beta(2 * bd, 5), 0.02, 0.95)
            advance_ratio = np.clip(rng.beta(3 * bd, 3), 0.0, 1.0)
            telco_score = np.exp(-0.3 * days_late) * (1 - 0.3 * min(suspensions / 12, 1))

            home_stasis = np.clip(rng.beta(5 * bd, 2), 0.3, 0.98)
            loc_entropy = np.clip(rng.beta(2, 5 * bd), 0.1, 0.9)
            routine_index = home_stasis * 0.6 + (1 - loc_entropy) * 0.4
            unique_locs = int(rng.poisson(8 - 4 * bd + 2))
            radius = np.clip(rng.exponential(10 / bd), 0.5, 80)

            irt = np.clip(rng.normal((bd * 0.5 + rp * 0.5), 0.12), 0.1, 0.98)
            conscient = np.clip(rng.normal(bd, 0.12), 0.1, 0.98)
            future_orient = np.clip(rng.normal(rp, 0.12), 0.1, 0.98)
            consistency = float(rng.random() < (0.85 * bd + 0.15))
            rt_flag = float(rng.random() < 0.90)

            utility_ratio = np.clip(rng.beta(3 * rp + 0.5, 2), 0.1, 0.95)
            interval_cv = np.clip(rng.beta(2, 5 * is_), 0.02, 0.95)
            return_rate = np.clip(rng.beta(1, 8 * (1 - rp + 0.1)), 0.0, 0.5)
            advance_pay = np.clip(rng.beta(3 * bd, 3), 0.0, 1.0)
            n_transactions = int(rng.poisson(12 * is_))

            has_merchant = archetype_name == "micro_entrepreneur"
            if has_merchant:
                bay_rating = np.clip(rng.normal(3.5 + bd, 0.5), 1.5, 5.0)
                review_vol = int(rng.poisson(25 * bd))
                sentiment = np.clip(rng.normal(0.3 * bd, 0.2), -1, 1)
                trend = np.clip(rng.normal(0.1 * bd, 0.05), -0.3, 0.3)
                closure = np.clip(rng.beta(1, 20 * bd), 0, 0.3)
            else:
                bay_rating, review_vol, sentiment, trend, closure = np.nan, np.nan, np.nan, np.nan, np.nan

            has_bank = rng.random() < 0.35
            if has_bank:
                inflow_cv = np.clip(rng.beta(2, 6 * is_), 0.02, 0.95)
                inflow_mean = np.clip(rng.lognormal(np.log(15000 * is_ + 5000), 0.4), 3000, 200000)
                mmb = max(0, rng.normal(inflow_mean * 0.15 * is_, inflow_mean * 0.05))
                avg_balance = inflow_mean * (0.2 + 0.3 * is_)
                io_ratio = np.clip(rng.normal(1.05 + 0.15 * is_, 0.1), 0.7, 2.0)
                salary_flag = float(archetype_name == "stable_salaried")
            else:
                inflow_cv, inflow_mean, mmb, avg_balance, io_ratio, salary_flag = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

            # Label generation: probability of default driven by latent vars
            adj_pd = base_pd * (2 - bd) * (2 - is_) * (1.5 - rp * 0.5)
            adj_pd = np.clip(adj_pd, 0.01, 0.70)
            u = rng.random()
            if u < adj_pd * 0.30:
                label = 3  # default
            elif u < adj_pd * 0.55:
                label = 2  # late delinquent
            elif u < adj_pd * 0.80:
                label = 1  # early delinquent
            else:
                label = 0  # on-time

            rows.append({
                "telco_days_late_mean": days_late, "telco_suspensions_count": float(suspensions),
                "telco_recharge_regularity_cv": recharge_cv, "telco_advance_ratio": advance_ratio,
                "telco_score": telco_score,
                "geo_home_stasis_pct": home_stasis, "geo_location_entropy": loc_entropy,
                "geo_routine_index": routine_index, "geo_unique_locations": float(unique_locs),
                "geo_radius_of_gyration_km": radius,
                "psycho_irt_score": irt, "psycho_conscientiousness": conscient,
                "psycho_future_orientation": future_orient,
                "psycho_consistency_flag": consistency, "psycho_response_time_flag": rt_flag,
                "ecom_utility_spend_ratio": utility_ratio, "ecom_purchase_interval_cv": interval_cv,
                "ecom_return_rate": return_rate, "ecom_advance_payment_ratio": advance_pay,
                "ecom_total_transactions": float(n_transactions),
                "merchant_bayesian_rating": bay_rating, "merchant_review_volume": float(review_vol) if not np.isnan(review_vol) else np.nan,
                "merchant_sentiment_score": sentiment, "merchant_rating_trend": trend,
                "merchant_closure_risk": closure,
                "bank_inflow_cv": inflow_cv, "bank_inflow_mean": inflow_mean,
                "bank_min_monthly_balance": mmb, "bank_avg_balance": avg_balance,
                "bank_inflow_outflow_ratio": io_ratio,
                "label": label,
                "archetype": archetype_name,
            })

    df = pd.DataFrame(rows)
    logger.info(f"Generated {len(df)} synthetic samples. Label distribution: {df['label'].value_counts().to_dict()}")
    return df


def train_logistic_scorecard(X_train, y_train_binary, scaler):
    """Layer 1: Interpretable logistic scorecard."""
    X_scaled = scaler.transform(np.nan_to_num(X_train, nan=0.0))
    clf = LogisticRegression(C=0.5, max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(X_scaled, y_train_binary)
    logger.info("Layer 1 (Logistic Scorecard) trained")
    return clf


def train_xgboost(X_train, y_train_binary):
    """Layer 2: XGBoost main performance model."""
    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_alpha=0.1,
        reg_lambda=1.0,
        use_label_encoder=False,
        eval_metric="auc",
        tree_method="hist",
        enable_categorical=False,
        random_state=42,
    )
    clf.fit(
        X_train, y_train_binary,
        eval_set=[(X_train, y_train_binary)],
        verbose=False,
    )
    logger.info("Layer 2 (XGBoost) trained")
    return clf


def train_gmm_segmentation(X_train, n_components=4):
    """Layer 3: GMM latent segmentation."""
    X_filled = np.nan_to_num(X_train, nan=0.5)
    gmm = GaussianMixture(n_components=n_components, covariance_type="full", random_state=42, n_init=3)
    gmm.fit(X_filled)
    logger.info(f"Layer 3 (GMM, {n_components} clusters) trained")
    return gmm


def train_calibration(xgb_model, X_train, y_train_binary):
    """Layer 4: Isotonic calibration layer."""
    calibrated = CalibratedClassifierCV(estimator=xgb_model, method="isotonic", cv=3)
    calibrated.fit(X_train, y_train_binary)
    logger.info("Layer 4 (Isotonic Calibration) trained")
    return calibrated


def evaluate_model(model, scaler, X_test, y_test, model_name):
    """Compute AUC, KS, Brier score."""
    if hasattr(model, "predict_proba"):
        if "logistic" in model_name.lower():
            X_in = scaler.transform(np.nan_to_num(X_test, nan=0.0))
        else:
            X_in = X_test
        probs = model.predict_proba(X_in)[:, 1]
    else:
        probs = np.full(len(y_test), 0.5)

    auc = roc_auc_score(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    pos_probs = probs[y_test == 1]
    neg_probs = probs[y_test == 0]
    if len(pos_probs) > 0 and len(neg_probs) > 0:
        ks_stat, _ = ks_2samp(pos_probs, neg_probs)
    else:
        ks_stat = 0.0
    gini = 2 * auc - 1

    metrics = {"auc": round(auc, 4), "ks": round(ks_stat, 4), "brier": round(brier, 4), "gini": round(gini, 4)}
    logger.info(f"{model_name} → AUC={auc:.4f}, KS={ks_stat:.4f}, Brier={brier:.4f}, Gini={gini:.4f}")
    return metrics


def main():
    logger.info("=== AlterScore Model Training Pipeline ===")
    df = generate_synthetic_data(n_samples=5000)
    df.to_csv(MODEL_DIR / "training_data.csv", index=False)

    X = df[FEATURE_COLUMNS].values
    # Binary label for primary model (default = 1, all others = 0)
    y_binary = (df["label"] >= 2).astype(int).values
    archetypes = df["archetype"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.20, random_state=42, stratify=y_binary)

    # Scaler for logistic scorecard
    scaler = StandardScaler()
    scaler.fit(np.nan_to_num(X_train, nan=0.0))

    # Train all 4 layers
    logistic = train_logistic_scorecard(X_train, y_train, scaler)
    xgb_model = train_xgboost(X_train, y_train)
    gmm = train_gmm_segmentation(X_train)
    calibrated_xgb = train_calibration(xgb_model, X_train, y_train)

    # Evaluate
    metrics = {
        "logistic_scorecard": evaluate_model(logistic, scaler, X_test, y_test, "Logistic Scorecard"),
        "xgboost": evaluate_model(xgb_model, scaler, X_test, y_test, "XGBoost"),
        "calibrated_xgboost": evaluate_model(calibrated_xgb, scaler, X_test, y_test, "Calibrated XGBoost"),
    }

    # Approval rate and inclusion metrics
    probs = calibrated_xgb.predict_proba(X_test)[:, 1]
    approval_rate = float((probs < 0.25).mean())
    thin_file_mask = np.isnan(X_test[:, 25])  # No bank data → thin-file proxy
    thin_file_approval = float((probs[thin_file_mask] < 0.30).mean()) if thin_file_mask.any() else 0.0
    metrics["business"] = {
        "approval_rate": round(approval_rate, 3),
        "thin_file_inclusion_rate": round(thin_file_approval, 3),
        "test_set_size": len(y_test),
    }

    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Persist all artifacts
    joblib.dump(logistic, MODEL_DIR / "logistic_scorecard.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    xgb_model.save_model(str(MODEL_DIR / "xgboost_model.json"))
    joblib.dump(gmm, MODEL_DIR / "gmm_segmentation.pkl")
    joblib.dump(calibrated_xgb, MODEL_DIR / "calibrated_model.pkl")
    joblib.dump(FEATURE_COLUMNS, MODEL_DIR / "feature_columns.pkl")

    logger.info("=== Training complete. All artifacts saved. ===")
    logger.info(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
