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


# Feature importance mapping (top features from model)
TOP_FEATURES = [
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
