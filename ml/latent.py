"""
AlterScore Latent Variable Layer
==================================
Maps engineered features → 5 latent creditworthiness dimensions.

Each latent variable aggregates multiple signal sources using
research-calibrated weights. Missing signals are excluded from
the weighted average (not penalised), following the XGBoost
missing-value philosophy from the research compendium.

DIMENSIONS:
  BD  Behavioral Discipline       (weight in ensemble: 0.28)
  IS  Income Stability            (weight: 0.24)
  RP  Risk Preference             (weight: 0.22)
  OR  Operational Reliability     (weight: 0.14)
  SC  Structural Constraints      (weight: 0.12)
"""

from typing import Dict, Optional, Tuple
import numpy as np


LATENT_WEIGHTS = {
    "BD": 0.28,
    "IS": 0.24,
    "RP": 0.22,
    "OR": 0.14,
    "SC": 0.12,
}

# Sub-signal weights within each latent dimension
BD_WEIGHTS = {
    "latent_bd_telco": 0.35,     # Bill payment timeliness
    "latent_bd_psycho": 0.40,    # Conscientiousness × consistency
    "latent_bd_geo": 0.25,       # Routine index
}

IS_WEIGHTS = {
    "latent_is_bank": 0.60,      # Cash flow CV + MMB
    "latent_is_ecom": 0.25,      # Purchase regularity
    "latent_is_telco": 0.15,     # Recharge regularity
}

RP_WEIGHTS = {
    "latent_rp_psycho": 0.55,   # Future orientation subscale
    "latent_rp_ecom": 0.45,     # Utility vs discretionary spend ratio
}

OR_WEIGHTS = {
    "latent_or_merchant": 0.60,  # Bayesian merchant rating
    "latent_or_ecom": 0.40,      # Low return rate / transaction reliability
}

SC_WEIGHTS = {
    "latent_sc_geo": 1.00,       # Geographic stability / radius of gyration
}


def _weighted_mean(features: Dict, weight_map: Dict) -> Optional[float]:
    """
    Computes weighted mean, excluding None/NaN values.
    Returns None if no valid signals exist.
    """
    total_weight = 0.0
    weighted_sum = 0.0
    for key, w in weight_map.items():
        val = features.get(key)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            weighted_sum += float(val) * w
            total_weight += w
    if total_weight == 0.0:
        return None
    return weighted_sum / total_weight


def compute_latent_scores(features: Dict) -> Dict:
    """
    Compute all 5 latent variable scores from the feature dict.
    Each score is in [0.0, 1.0] or None if insufficient data.
    """
    bd = _weighted_mean(features, BD_WEIGHTS)
    is_ = _weighted_mean(features, IS_WEIGHTS)
    rp = _weighted_mean(features, RP_WEIGHTS)
    or_ = _weighted_mean(features, OR_WEIGHTS)
    sc = _weighted_mean(features, SC_WEIGHTS)

    return {
        "latent_BD": bd,
        "latent_IS": is_,
        "latent_RP": rp,
        "latent_OR": or_,
        "latent_SC": sc,
    }


def compute_composite_latent_score(latent_scores: Dict) -> Tuple[float, float]:
    """
    Combines latent scores into a composite creditworthiness index.
    Returns (composite_score, confidence) both in [0.0, 1.0].

    The composite is a weighted average over available latent scores.
    Confidence = fraction of maximum possible weight that is available.
    """
    total_weight = 0.0
    weighted_sum = 0.0
    max_possible_weight = sum(LATENT_WEIGHTS.values())

    for dim, weight in LATENT_WEIGHTS.items():
        val = latent_scores.get(f"latent_{dim}")
        if val is not None:
            weighted_sum += val * weight
            total_weight += weight

    if total_weight == 0.0:
        return 0.5, 0.0  # No data: default to neutral with zero confidence

    composite = weighted_sum / total_weight
    confidence = total_weight / max_possible_weight
    return max(0.0, min(1.0, composite)), confidence


def compute_recovery_rate_estimate(features: Dict, latent_scores: Dict) -> float:
    """
    Estimates recovery rate (LGD proxy) for economic scoring.
    Higher geo stability + higher OR = better recovery prospects.
    """
    geo_stasis = features.get("geo_home_stasis_pct") or 0.5
    routine = features.get("geo_routine_index") or 0.5
    or_score = latent_scores.get("latent_OR") or 0.5
    sc_score = latent_scores.get("latent_SC") or 0.5

    # Geographic stability → easier to locate for collection
    # OR → business viability → assets available for recovery
    recovery_estimate = geo_stasis * 0.30 + routine * 0.20 + or_score * 0.30 + sc_score * 0.20
    return max(0.20, min(0.85, recovery_estimate))  # Floor/cap at realistic bounds


def expected_profit_score(
    pd_calibrated: float,
    latent_scores: Dict,
    features: Dict,
    loan_amount: float = 50000.0,
    interest_rate: float = 0.16,
    tenor_years: float = 2.0,
) -> Dict:
    """
    Economic scoring: NOT just probability of default.
    Expected Profit Score = f(PD, IS, BD, Recovery_Rate)

    Research basis: ICIS 2019 — lending to 'risky but recoverable' borrowers
    can be profitable if interest revenue exceeds expected credit loss.

    Formula:
        EV_revenue   = loan * rate * tenor * (1 - PD)
        EV_loss      = loan * PD * (1 - recovery_rate)
        Expected_Profit_Ratio = (EV_revenue - EV_loss) / loan
    """
    is_score = latent_scores.get("latent_IS") or 0.5
    bd_score = latent_scores.get("latent_BD") or 0.5
    recovery_rate = compute_recovery_rate_estimate(features, latent_scores)

    ev_revenue = loan_amount * interest_rate * tenor_years * (1.0 - pd_calibrated)
    ev_loss = loan_amount * pd_calibrated * (1.0 - recovery_rate)
    ev_profit = ev_revenue - ev_loss
    profit_ratio = ev_profit / loan_amount  # Normalised EPR

    # Composite economic score: blend EPR with behavioral stability signals
    econ_score = (
        profit_ratio * 0.50          # Profit-based component
        + is_score * 0.25             # Income stability adjustment
        + bd_score * 0.25             # Behavioral discipline adjustment
    )
    # Normalise to [0, 1]
    econ_score_normalized = max(0.0, min(1.0, (econ_score + 0.5) / 1.5))

    return {
        "ev_revenue": round(ev_revenue, 2),
        "ev_loss": round(ev_loss, 2),
        "ev_profit": round(ev_profit, 2),
        "profit_ratio": round(profit_ratio, 4),
        "recovery_rate_estimate": round(recovery_rate, 3),
        "economic_score_normalized": round(econ_score_normalized, 4),
    }
