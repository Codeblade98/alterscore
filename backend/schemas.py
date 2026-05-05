"""
AlterScore Pydantic Schemas
============================
Defines all API request and response structures.
Reflects the consent-based data ingestion architecture.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator


# ─────────────────────────────────────────────
# INPUT DATA SCHEMAS
# ─────────────────────────────────────────────

class TelcoPayment(BaseModel):
    days_late: float = Field(0.0, ge=0, description="Days past due date")
    month: Optional[str] = None


class TelcoData(BaseModel):
    payments: List[TelcoPayment] = []
    suspensions: int = Field(0, ge=0, description="Service suspension count (last 12 months)")
    recharge_intervals_days: List[float] = []
    advance_recharge_ratio: float = Field(0.0, ge=0, le=1)


class GeoData(BaseModel):
    has_data: bool = False
    home_stasis_pct: float = Field(0.65, ge=0, le=1)
    work_stasis_pct: float = Field(0.50, ge=0, le=1)
    location_entropy: float = Field(1.5, ge=0)
    unique_locations_30d: int = Field(8, ge=1)
    radius_of_gyration_km: float = Field(5.0, ge=0)


class PsychoAnswers(BaseModel):
    q1: Optional[str] = None   # A/B/C/D
    q2: Optional[str] = None
    q3: Optional[str] = None
    q4_trap: Optional[str] = None
    q5: Optional[str] = None
    q6: Optional[str] = None
    q7: Optional[str] = None


class PsychoData(BaseModel):
    answers: PsychoAnswers = PsychoAnswers()
    avg_response_time_seconds: float = Field(10.0, ge=1)


class EcomTransaction(BaseModel):
    amount: float = Field(..., gt=0)
    category: str = "other"
    returned: bool = False
    payment_method: str = "online"
    day_index: Optional[float] = None


class EcomData(BaseModel):
    transactions: List[EcomTransaction] = []


class MerchantData(BaseModel):
    exists: bool = False
    avg_rating: float = Field(3.5, ge=1, le=5)
    review_count: int = Field(0, ge=0)
    positive_review_count: int = Field(0, ge=0)
    negative_review_count: int = Field(0, ge=0)
    closure_mentions: int = Field(0, ge=0)
    rating_trend_6m: float = Field(0.0, ge=-1, le=1)


class BankData(BaseModel):
    monthly_inflows: List[float] = []
    monthly_outflows: List[float] = []
    min_monthly_balance: float = Field(0.0, ge=0)
    avg_monthly_balance: float = Field(0.0, ge=0)
    salary_regularity_flag: int = Field(0, ge=0, le=1)


class ConsentFlags(BaseModel):
    telco_consent: bool = True
    geo_consent: bool = True
    psycho_consent: bool = True
    ecom_consent: bool = True
    merchant_consent: bool = False
    bank_consent: bool = False


class BorrowerProfile(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    phone: Optional[str] = None
    occupation: str = "self_employed"
    monthly_income_inr: float = Field(20000.0, gt=0)
    state: Optional[str] = None
    loan_purpose: Optional[str] = None


class ScoringRequest(BaseModel):
    borrower: BorrowerProfile
    consent: ConsentFlags
    telco: Optional[TelcoData] = None
    geo: Optional[GeoData] = None
    psychometrics: Optional[PsychoData] = None
    ecommerce: Optional[EcomData] = None
    merchant: Optional[MerchantData] = None
    bank: Optional[BankData] = None

    def to_raw_data(self) -> Dict:
        """Convert to raw_data dict for feature engineering, respecting consent."""
        raw: Dict = {}
        if self.consent.telco_consent and self.telco:
            raw["telco"] = self.telco.model_dump()
        if self.consent.geo_consent and self.geo:
            raw["geo"] = self.geo.model_dump()
        if self.consent.psycho_consent and self.psychometrics:
            raw["psychometrics"] = {"answers": self.psychometrics.answers.model_dump(), "avg_response_time_seconds": self.psychometrics.avg_response_time_seconds}
        if self.consent.ecom_consent and self.ecommerce:
            raw["ecommerce"] = self.ecommerce.model_dump()
        if self.consent.merchant_consent and self.merchant:
            raw["merchant"] = self.merchant.model_dump()
        if self.consent.bank_consent and self.bank:
            raw["bank"] = self.bank.model_dump()
        return raw


# ─────────────────────────────────────────────
# RESPONSE SCHEMAS
# ─────────────────────────────────────────────

class LatentScores(BaseModel):
    BD: float  # Behavioral Discipline
    IS: float  # Income Stability
    RP: float  # Risk Preference
    OR: float  # Operational Reliability
    SC: float  # Structural Constraints


class EconomicAnalysis(BaseModel):
    ev_revenue: float
    ev_loss: float
    ev_profit: float
    profit_ratio: float
    recovery_rate_estimate: float
    economic_score_normalized: float


class SegmentInfo(BaseModel):
    name: str
    index: int
    probabilities: List[float]


class FairnessFlags(BaseModel):
    data_coverage_ratio: float
    low_data_coverage_flag: bool
    signals_available: int
    signals_total: int
    recommendation: str


class ScoringResponse(BaseModel):
    alter_score: int
    pd_calibrated: float
    decision: str
    tier_name: str
    risk_band: str
    credit_limit_inr: float
    interest_rate_pct: float
    monitoring_intensity: str
    confidence: float
    tier_color: str
    latent_scores: LatentScores
    composite_latent_score: float
    economic_analysis: EconomicAnalysis
    segment: SegmentInfo
    shap_values: Dict[str, float]
    features_used: Dict[str, Optional[float]]
    fairness_flags: FairnessFlags
    model_versions: Dict[str, Any]
    application_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str = "1.0.0"


class MetricsResponse(BaseModel):
    logistic_scorecard: Dict[str, float]
    xgboost: Dict[str, float]
    calibrated_xgboost: Dict[str, float]
    business: Dict[str, Any]
