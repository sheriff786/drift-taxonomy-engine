"""Constants, enums, and threshold mappings for the drift taxonomy engine."""

from enum import Enum


class DriftType(str, Enum):
    """Classification of detected drift."""
    COVARIATE = "covariate"
    CONCEPT = "concept"
    PIPELINE = "pipeline"
    TARGET = "target"
    MIXED = "mixed"
    NONE = "none"


class Severity(str, Enum):
    """Drift severity levels."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ActionType(str, Enum):
    """Recommended operational actions."""
    MONITOR = "monitor"
    ALERT = "alert"
    INVESTIGATE = "investigate"
    INCREMENTAL_RETRAIN = "incremental_retrain"
    FULL_RETRAIN = "full_retrain"
    BLOCK = "block"


# Feature name mapping: PCA components -> interpretive domain names
# Based on correlation analysis, KS statistics, and fraud detection behavior
FEATURE_NAME_MAPPING = {
    "V1": "card_auth_maturity",
    "V2": "velocity_anomaly",
    "V3": "spending_pattern_match",
    "V4": "high_risk_merchant_flag",
    "V5": "channel_risk_score",
    "V6": "geo_distance_indicator",
    "V7": "location_consistency",
    "V8": "time_deviation",
    "V9": "txn_frequency_normality",
    "V10": "behavioral_consistency",
    "V11": "unusual_activity_flag",
    "V12": "address_verification_score",
    "V13": "account_balance_ratio",
    "V14": "cardholder_verification",
    "V15": "payment_method_risk",
    "V16": "merchant_reputation",
    "V17": "transaction_legitimacy",
    "V18": "auth_strength_score",
    "V19": "ip_risk_score",
    "V20": "device_fingerprint",
    "V21": "cross_border_indicator",
    "V22": "recurring_pattern",
    "V23": "entry_mode_risk",
    "V24": "billing_match_score",
    "V25": "card_present_indicator",
    "V26": "refund_history",
    "V27": "pin_verification_result",
    "V28": "decline_history_score",
    "Amount_scaled": "transaction_amount",
    "Time_scaled": "transaction_time",
}

# Reverse mapping: domain name -> original PCA column
FEATURE_NAME_REVERSE = {v: k for k, v in FEATURE_NAME_MAPPING.items()}

# Feature importance mapping (top features from model) - using domain names
TOP_FEATURES = [
    "cardholder_verification", "behavioral_consistency",
    "address_verification_score", "high_risk_merchant_flag",
    "unusual_activity_flag", "transaction_legitimacy",
    "spending_pattern_match", "merchant_reputation",
    "location_consistency", "velocity_anomaly"
]

# Original PCA names for backward compatibility with trained models
TOP_FEATURES_ORIGINAL = [
    "V14", "V10", "V12", "V4", "V11",
    "V17", "V3", "V16", "V7", "V2"
]

# Action priority mapping: (drift_type, severity) -> action
ACTION_MAPPING = {
    (DriftType.PIPELINE, Severity.CRITICAL): ActionType.BLOCK,
    (DriftType.PIPELINE, Severity.HIGH): ActionType.BLOCK,
    (DriftType.PIPELINE, Severity.MEDIUM): ActionType.INVESTIGATE,
    (DriftType.PIPELINE, Severity.LOW): ActionType.ALERT,
    (DriftType.CONCEPT, Severity.CRITICAL): ActionType.FULL_RETRAIN,
    (DriftType.CONCEPT, Severity.HIGH): ActionType.FULL_RETRAIN,
    (DriftType.CONCEPT, Severity.MEDIUM): ActionType.INCREMENTAL_RETRAIN,
    (DriftType.CONCEPT, Severity.LOW): ActionType.ALERT,
    (DriftType.COVARIATE, Severity.CRITICAL): ActionType.INCREMENTAL_RETRAIN,
    (DriftType.COVARIATE, Severity.HIGH): ActionType.INVESTIGATE,
    (DriftType.COVARIATE, Severity.MEDIUM): ActionType.ALERT,
    (DriftType.COVARIATE, Severity.LOW): ActionType.MONITOR,
    (DriftType.TARGET, Severity.CRITICAL): ActionType.FULL_RETRAIN,
    (DriftType.TARGET, Severity.HIGH): ActionType.INCREMENTAL_RETRAIN,
    (DriftType.TARGET, Severity.MEDIUM): ActionType.ALERT,
    (DriftType.TARGET, Severity.LOW): ActionType.MONITOR,
    (DriftType.MIXED, Severity.CRITICAL): ActionType.FULL_RETRAIN,
    (DriftType.MIXED, Severity.HIGH): ActionType.FULL_RETRAIN,
    (DriftType.MIXED, Severity.MEDIUM): ActionType.INCREMENTAL_RETRAIN,
    (DriftType.MIXED, Severity.LOW): ActionType.INVESTIGATE,
}

# Model configurations
MODEL_CONFIGS = {
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
    },
    "xgboost": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "scale_pos_weight": 580,
    },
    "lightgbm": {
        "n_estimators": 200,
        "max_depth": -1,
        "learning_rate": 0.1,
        "is_unbalance": True,
    },
    "logistic_regression": {
        "max_iter": 1000,
        "class_weight": "balanced",
        "solver": "lbfgs",
    },
}
