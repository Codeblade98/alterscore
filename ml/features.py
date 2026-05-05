"""
AlterScore Feature Engineering Module
======================================
Converts raw alternative data signals into engineered features.

LATENT VARIABLE MAPPING (from research compendium):
  BD (Behavioral Discipline) ← Telco payment timing, Psycho conscientiousness, Geo routine
  IS (Income Stability)      ← Bank inflow CV, Ecom purchase regularity
  RP (Risk Preference)       ← Psycho future-orientation, Ecom category composition
  OR (Operational Reliability) ← Merchant ratings, Ecom return rate
  SC (Structural Constraints) ← Geo stability, Occupation type
"""

import math
import numpy as np
from typing import Dict, Any, Optional, List


class TelcoFeatures:
    """
    Telco signals → BD (primary), IS (secondary)
    Core formula: Score_telco = exp(-k * days_late_mean) * (1 - 0.3 * suspensions_normalized)
    Research basis: Björkegren & Grüner (2017); ICIS 2019 behavioral stability proxy
    """
    K = 0.30  # Penalty constant calibrated to Indian telco market

    def compute(self, data: Dict) -> Dict:
        if not data or not data.get("payments"):
            return self._null_features()

        payments = data.get("payments", [])
        suspensions = float(data.get("suspensions", 0))
        recharge_intervals = data.get("recharge_intervals_days", [])

        days_late = [max(0.0, float(p.get("days_late", 0))) for p in payments]
        days_late_mean = float(np.mean(days_late)) if days_late else 0.0
        days_late_max = float(np.max(days_late)) if days_late else 0.0

        # Recharge regularity: lower CV = more disciplined top-ups
        if len(recharge_intervals) >= 2 and np.mean(recharge_intervals) > 0:
            recharge_cv = float(np.std(recharge_intervals) / np.mean(recharge_intervals))
        else:
            recharge_cv = 0.5

        suspensions_normalized = min(suspensions / 12.0, 1.0)
        advance_ratio = float(data.get("advance_recharge_ratio", 0.0))

        # Exponential decay scoring — research-grounded formula
        telco_score = math.exp(-self.K * days_late_mean) * (1.0 - 0.30 * suspensions_normalized)
        telco_score = max(0.0, min(1.0, telco_score))

        return {
            "telco_days_late_mean": days_late_mean,
            "telco_days_late_max": days_late_max,
            "telco_suspensions_count": suspensions,
            "telco_recharge_regularity_cv": recharge_cv,
            "telco_advance_ratio": advance_ratio,
            "telco_score": telco_score,
            "latent_bd_telco": telco_score,
            "latent_is_telco": 1.0 - min(recharge_cv, 1.0),
        }

    def _null_features(self) -> Dict:
        keys = [
            "telco_days_late_mean", "telco_days_late_max", "telco_suspensions_count",
            "telco_recharge_regularity_cv", "telco_advance_ratio", "telco_score",
            "latent_bd_telco", "latent_is_telco",
        ]
        return {k: None for k in keys}


class GeoFeatures:
    """
    Mobility signals → BD (routine stability), SC (geographic constraint)
    Routine Index = home_stasis * 0.6 + (1 - normalized_entropy) * 0.4
    Research basis: Radius of gyration + Routine Index literature
    """

    def compute(self, data: Dict) -> Dict:
        if not data or not data.get("has_data", False):
            return self._null_features()

        home_stasis_pct = float(data.get("home_stasis_pct", 0.60))
        work_stasis_pct = float(data.get("work_stasis_pct", 0.50))
        unique_locs = float(data.get("unique_locations_30d", 8))
        radius_km = float(data.get("radius_of_gyration_km", 5.0))
        location_entropy = float(data.get("location_entropy", 1.5))

        # Normalize entropy relative to maximum possible
        max_entropy = math.log2(max(unique_locs, 1)) if unique_locs > 0 else 1.0
        normalized_entropy = min(location_entropy / max_entropy, 1.0) if max_entropy > 0 else 0.5

        # Routine Index — lower entropy + higher home stasis = more predictable lifestyle
        routine_index = home_stasis_pct * 0.6 + (1.0 - normalized_entropy) * 0.4
        routine_index = max(0.0, min(1.0, routine_index))

        # SC: geographically anchored borrowers are easier to reach for collection
        sc_geo = max(0.0, 1.0 - (radius_km / 50.0))

        return {
            "geo_home_stasis_pct": home_stasis_pct,
            "geo_work_stasis_pct": work_stasis_pct,
            "geo_location_entropy": normalized_entropy,
            "geo_routine_index": routine_index,
            "geo_unique_locations": unique_locs,
            "geo_radius_of_gyration_km": radius_km,
            "latent_bd_geo": routine_index,
            "latent_sc_geo": sc_geo,
        }

    def _null_features(self) -> Dict:
        keys = [
            "geo_home_stasis_pct", "geo_work_stasis_pct", "geo_location_entropy",
            "geo_routine_index", "geo_unique_locations", "geo_radius_of_gyration_km",
            "latent_bd_geo", "latent_sc_geo",
        ]
        return {k: None for k in keys}


class PsychoFeatures:
    """
    Psychometric signals → BD (conscientiousness), RP (future orientation)
    IRT scoring with consistency multiplier for trap questions.
    Research basis: IDB WP-625 (Arraiz et al.) — LenddoEFL validated approach.
    """
    QUESTION_SCORES = {
        "q1": {"A": 3, "B": 3, "C": 2, "D": 0},   # Future orientation / savings intent
        "q2": {"A": 3, "B": 2, "C": 1, "D": 0},   # Bill payment discipline (BD)
        "q3": {"A": 3, "B": 2, "C": 1, "D": 0},   # Obligation fulfillment (BD)
        "q4_trap": {"A": 3, "B": 2, "C": 1, "D": 0},  # Trap: same construct as q2
        "q5": {"A": 3, "B": 2, "C": 2, "D": 0},   # Loan utilization intent (RP)
        "q6": {"A": 3, "B": 2, "C": 1, "D": 1},   # Decision-making style (BD)
        "q7": {"A": 3, "B": 2, "C": 1, "D": 0},   # 2-year planning horizon (RP)
    }
    CONSCIENTIOUSNESS_QS = {"q2", "q3", "q6"}
    FUTURE_ORIENTATION_QS = {"q1", "q5", "q7"}
    CONSISTENCY_THRESHOLD = 1  # Max allowed delta between q2 and q4_trap

    def compute(self, data: Dict) -> Dict:
        answers = data.get("answers", {})
        if not answers:
            return self._null_features()

        raw_scores, conscient, future = [], [], []
        for qid, scoring in self.QUESTION_SCORES.items():
            ans = answers.get(qid)
            if ans:
                s = scoring.get(str(ans).upper(), 1)
                raw_scores.append(s)
                if qid in self.CONSCIENTIOUSNESS_QS:
                    conscient.append(s)
                if qid in self.FUTURE_ORIENTATION_QS:
                    future.append(s)

        max_raw = len(self.QUESTION_SCORES) * 3
        irt_raw = sum(raw_scores) / max_raw if raw_scores else 0.5

        # Trap question consistency check — key anti-gaming mechanism
        q2_score = self.QUESTION_SCORES["q2"].get(str(answers.get("q2", "B")).upper(), 2)
        q4_score = self.QUESTION_SCORES["q4_trap"].get(str(answers.get("q4_trap", "B")).upper(), 2)
        delta = abs(q2_score - q4_score)
        consistency_multiplier = 1.0 if delta <= self.CONSISTENCY_THRESHOLD else 0.65
        consistency_flag = 1.0 if delta <= self.CONSISTENCY_THRESHOLD else 0.0

        conscientiousness = float(np.mean(conscient) / 3.0) if conscient else 0.5
        future_orientation = float(np.mean(future) / 3.0) if future else 0.5
        irt_score = irt_raw * consistency_multiplier

        avg_rt = float(data.get("avg_response_time_seconds", 10))
        rt_flag = 1.0 if 3 <= avg_rt <= 60 else 0.5

        return {
            "psycho_irt_score": irt_score,
            "psycho_conscientiousness": conscientiousness,
            "psycho_future_orientation": future_orientation,
            "psycho_consistency_flag": consistency_flag,
            "psycho_consistency_multiplier": consistency_multiplier,
            "psycho_response_time_flag": rt_flag,
            "latent_bd_psycho": conscientiousness * consistency_multiplier,
            "latent_rp_psycho": future_orientation,
        }

    def _null_features(self) -> Dict:
        keys = [
            "psycho_irt_score", "psycho_conscientiousness", "psycho_future_orientation",
            "psycho_consistency_flag", "psycho_consistency_multiplier",
            "psycho_response_time_flag", "latent_bd_psycho", "latent_rp_psycho",
        ]
        return {k: None for k in keys}


class EcomFeatures:
    """
    E-commerce signals → RP (category), IS (regularity), OR (reliability)
    Score = utility_ratio * 0.5 + (1 - interval_cv) * 0.3 + (1 - return_rate) * 0.2
    Research basis: Ant Financial / Sesame Credit — what you buy > how much you spend.
    """
    UTILITY_CATEGORIES = {
        "groceries", "utilities", "tools", "agriculture", "wholesale",
        "education", "medical", "transport", "construction", "fertilizer",
    }

    def compute(self, data: Dict) -> Dict:
        transactions = data.get("transactions", [])
        if not transactions:
            return self._null_features()

        total_spend = sum(float(t.get("amount", 0)) for t in transactions)
        utility_spend = sum(
            float(t.get("amount", 0))
            for t in transactions
            if str(t.get("category", "")).lower() in self.UTILITY_CATEGORIES
        )
        returns = sum(1 for t in transactions if t.get("returned", False))
        advance_payments = sum(1 for t in transactions if t.get("payment_method") == "advance")

        utility_ratio = utility_spend / total_spend if total_spend > 0 else 0.5
        return_rate = returns / len(transactions)
        advance_ratio = advance_payments / len(transactions)

        # Purchase interval CV — regularity is a core IS signal
        if len(transactions) >= 2:
            dates = sorted([float(t.get("day_index", i * 7)) for i, t in enumerate(transactions)])
            intervals = [dates[i + 1] - dates[i] for i in range(len(dates) - 1)]
            interval_mean = np.mean(intervals)
            interval_cv = float(np.std(intervals) / interval_mean) if interval_mean > 0 else 0.5
        else:
            interval_cv = 0.5

        ecom_score = (
            utility_ratio * 0.50
            + (1.0 - min(interval_cv, 1.0)) * 0.30
            + (1.0 - return_rate) * 0.20
        )

        return {
            "ecom_utility_spend_ratio": float(utility_ratio),
            "ecom_purchase_interval_cv": float(interval_cv),
            "ecom_return_rate": float(return_rate),
            "ecom_advance_payment_ratio": float(advance_ratio),
            "ecom_total_transactions": float(len(transactions)),
            "ecom_score": float(ecom_score),
            "latent_rp_ecom": float(utility_ratio),
            "latent_is_ecom": float(1.0 - min(interval_cv, 1.0)),
            "latent_or_ecom": float(1.0 - return_rate),
        }

    def _null_features(self) -> Dict:
        keys = [
            "ecom_utility_spend_ratio", "ecom_purchase_interval_cv", "ecom_return_rate",
            "ecom_advance_payment_ratio", "ecom_total_transactions", "ecom_score",
            "latent_rp_ecom", "latent_is_ecom", "latent_or_ecom",
        ]
        return {k: None for k in keys}


class MerchantFeatures:
    """
    Merchant ratings → OR (primary), BD (secondary via operational consistency)
    Bayesian average: (n * mean + C * prior) / (n + C)
    Research basis: Yelp/Google Maps → SME survival rate literature.
    """
    PRIOR_WEIGHT = 10  # C parameter for Bayesian shrinkage
    PRIOR_MEAN = 3.0   # Population mean rating prior

    def compute(self, data: Dict) -> Dict:
        if not data or not data.get("exists", False):
            return self._null_features()

        raw_rating = float(data.get("avg_rating", 3.5))
        review_count = int(data.get("review_count", 0))
        positive = int(data.get("positive_review_count", review_count // 2))
        negative = int(data.get("negative_review_count", 0))
        closure_mentions = int(data.get("closure_mentions", 0))
        rating_trend = float(data.get("rating_trend_6m", 0.0))

        # Bayesian average prevents gaming by low-volume merchants
        C = self.PRIOR_WEIGHT
        bayesian_rating = (review_count * raw_rating + C * self.PRIOR_MEAN) / (review_count + C)

        # Sentiment derived from review distribution
        if review_count > 0:
            sentiment = (positive / review_count) - (negative / review_count)
        else:
            sentiment = 0.0
        sentiment = max(-1.0, min(1.0, sentiment))
        sentiment_multiplier = 0.80 + 0.40 * ((sentiment + 1.0) / 2.0)  # → [0.8, 1.2]

        closure_risk = min(closure_mentions / max(review_count, 1), 1.0)
        merchant_score = (bayesian_rating / 5.0) * sentiment_multiplier * (1.0 - closure_risk * 0.30)
        merchant_score = max(0.0, min(1.0, merchant_score))

        return {
            "merchant_bayesian_rating": float(bayesian_rating),
            "merchant_review_volume": float(review_count),
            "merchant_sentiment_score": float(sentiment),
            "merchant_rating_trend": float(rating_trend),
            "merchant_closure_risk": float(closure_risk),
            "merchant_score": float(merchant_score),
            "latent_or_merchant": float(merchant_score),
        }

    def _null_features(self) -> Dict:
        keys = [
            "merchant_bayesian_rating", "merchant_review_volume", "merchant_sentiment_score",
            "merchant_rating_trend", "merchant_closure_risk", "merchant_score",
            "latent_or_merchant",
        ]
        return {k: None for k in keys}


class BankFeatures:
    """
    Cash flow signals → IS (primary), SC (secondary)
    Score = (1 - inflow_cv) * 0.6 + normalized_mmb * 0.4
    Key insight: stability > level (CV-based scoring not income-level based)
    Research basis: Sahamati AA cash-flow lending framework; MSME paper
    """
    MMB_NORMALIZATION = 5000.0  # INR threshold for full MMB score

    def compute(self, data: Dict) -> Dict:
        monthly_inflows = data.get("monthly_inflows", [])
        if not monthly_inflows or len(monthly_inflows) < 2:
            return self._null_features()

        inflows = [float(x) for x in monthly_inflows]
        inflow_mean = float(np.mean(inflows))
        inflow_cv = float(np.std(inflows) / inflow_mean) if inflow_mean > 0 else 1.0

        monthly_outflows = data.get("monthly_outflows", inflows)
        outflows = [float(x) for x in monthly_outflows]
        io_ratio = inflow_mean / float(np.mean(outflows)) if float(np.mean(outflows)) > 0 else 1.0

        mmb = float(data.get("min_monthly_balance", 0))
        avg_balance = float(data.get("avg_monthly_balance", inflow_mean * 0.3))
        salary_flag = float(data.get("salary_regularity_flag", 0))

        normalized_mmb = min(mmb / self.MMB_NORMALIZATION, 1.0)
        bank_score = (1.0 - min(inflow_cv, 1.0)) * 0.6 + normalized_mmb * 0.4

        return {
            "bank_inflow_cv": float(inflow_cv),
            "bank_inflow_mean": float(inflow_mean),
            "bank_min_monthly_balance": float(mmb),
            "bank_avg_balance": float(avg_balance),
            "bank_inflow_outflow_ratio": float(io_ratio),
            "bank_salary_flag": float(salary_flag),
            "bank_score": float(bank_score),
            "latent_is_bank": float(bank_score),
        }

    def _null_features(self) -> Dict:
        keys = [
            "bank_inflow_cv", "bank_inflow_mean", "bank_min_monthly_balance",
            "bank_avg_balance", "bank_inflow_outflow_ratio", "bank_salary_flag",
            "bank_score", "latent_is_bank",
        ]
        return {k: None for k in keys}


class FeatureEngineer:
    """
    Orchestrator: combines all signal modules into a single feature dict.
    All latent_ prefixed keys are consumed by latent.py.
    """

    def __init__(self):
        self.telco = TelcoFeatures()
        self.geo = GeoFeatures()
        self.psycho = PsychoFeatures()
        self.ecom = EcomFeatures()
        self.merchant = MerchantFeatures()
        self.bank = BankFeatures()

    def engineer(self, raw_data: Dict) -> Dict:
        features: Dict = {}
        features.update(self.telco.compute(raw_data.get("telco", {})))
        features.update(self.geo.compute(raw_data.get("geo", {})))
        features.update(self.psycho.compute(raw_data.get("psychometrics", {})))
        features.update(self.ecom.compute(raw_data.get("ecommerce", {})))
        features.update(self.merchant.compute(raw_data.get("merchant", {})))
        features.update(self.bank.compute(raw_data.get("bank", {})))
        return features

    def to_model_vector(self, features: Dict) -> Dict:
        """
        Extracts the numeric feature vector for the ML model.
        None values (missing data) are passed as NaN for XGBoost native handling.
        """
        model_features = [
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
        return {k: features.get(k, float("nan")) for k in model_features}
