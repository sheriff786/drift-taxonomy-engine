"""Prediction request/response schemas."""

from pydantic import BaseModel, model_validator
from typing import List, Optional, Dict

from src.config.constants import FEATURE_NAME_MAPPING, FEATURE_NAME_REVERSE


class TransactionSample(BaseModel):
    """A single transaction for fraud scoring.

    Accepts both domain names and original V1-V28 column names.
    Domain names (preferred):
        card_auth_maturity, velocity_anomaly, spending_pattern_match,
        high_risk_merchant_flag, channel_risk_score, geo_distance_indicator,
        location_consistency, time_deviation, txn_frequency_normality,
        behavioral_consistency, unusual_activity_flag, address_verification_score,
        account_balance_ratio, cardholder_verification, payment_method_risk,
        merchant_reputation, transaction_legitimacy, auth_strength_score,
        ip_risk_score, device_fingerprint, cross_border_indicator,
        recurring_pattern, entry_mode_risk, billing_match_score,
        card_present_indicator, refund_history, pin_verification_result,
        decline_history_score, transaction_amount, transaction_time
    """
    # Domain names (new interpretive names)
    card_auth_maturity: float = 0.0
    velocity_anomaly: float = 0.0
    spending_pattern_match: float = 0.0
    high_risk_merchant_flag: float = 0.0
    channel_risk_score: float = 0.0
    geo_distance_indicator: float = 0.0
    location_consistency: float = 0.0
    time_deviation: float = 0.0
    txn_frequency_normality: float = 0.0
    behavioral_consistency: float = 0.0
    unusual_activity_flag: float = 0.0
    address_verification_score: float = 0.0
    account_balance_ratio: float = 0.0
    cardholder_verification: float = 0.0
    payment_method_risk: float = 0.0
    merchant_reputation: float = 0.0
    transaction_legitimacy: float = 0.0
    auth_strength_score: float = 0.0
    ip_risk_score: float = 0.0
    device_fingerprint: float = 0.0
    cross_border_indicator: float = 0.0
    recurring_pattern: float = 0.0
    entry_mode_risk: float = 0.0
    billing_match_score: float = 0.0
    card_present_indicator: float = 0.0
    refund_history: float = 0.0
    pin_verification_result: float = 0.0
    decline_history_score: float = 0.0
    transaction_amount: float = 0.0
    transaction_time: float = 0.0

    # Legacy support: V1-V28 + Amount_scaled + Time_scaled
    V1: Optional[float] = None
    V2: Optional[float] = None
    V3: Optional[float] = None
    V4: Optional[float] = None
    V5: Optional[float] = None
    V6: Optional[float] = None
    V7: Optional[float] = None
    V8: Optional[float] = None
    V9: Optional[float] = None
    V10: Optional[float] = None
    V11: Optional[float] = None
    V12: Optional[float] = None
    V13: Optional[float] = None
    V14: Optional[float] = None
    V15: Optional[float] = None
    V16: Optional[float] = None
    V17: Optional[float] = None
    V18: Optional[float] = None
    V19: Optional[float] = None
    V20: Optional[float] = None
    V21: Optional[float] = None
    V22: Optional[float] = None
    V23: Optional[float] = None
    V24: Optional[float] = None
    V25: Optional[float] = None
    V26: Optional[float] = None
    V27: Optional[float] = None
    V28: Optional[float] = None
    Amount_scaled: Optional[float] = None
    Time_scaled: Optional[float] = None

    @model_validator(mode="after")
    def merge_legacy_fields(self):
        """If V1-V28 are provided (legacy), map them to domain names."""
        for original, domain in FEATURE_NAME_MAPPING.items():
            legacy_val = getattr(self, original, None) if hasattr(self, original) else None
            if legacy_val is not None:
                setattr(self, domain, legacy_val)
        return self

    def to_model_input(self) -> Dict[str, float]:
        """Convert to model input format (original V1-V28 column names)."""
        result = {}
        for original, domain in FEATURE_NAME_MAPPING.items():
            result[original] = getattr(self, domain)
        return result


AVAILABLE_MODELS = ["random_forest", "xgboost", "lightgbm", "logistic_regression"]


class PredictionRequest(BaseModel):
    """Batch prediction request."""
    model_config = {"protected_namespaces": ()}

    samples: List[TransactionSample]
    model_name: Optional[str] = None  # Options: random_forest, xgboost, lightgbm, logistic_regression


class PredictionResponse(BaseModel):
    """Prediction response with fraud scores."""
    model_config = {"protected_namespaces": ()}

    predictions: List[int]
    probabilities: List[float]
    model_name: str
    model_version: str
    n_samples: int
    available_models: List[str] = AVAILABLE_MODELS
    feature_names: List[str] = list(FEATURE_NAME_MAPPING.values())
