"""Prediction request/response schemas."""

from pydantic import BaseModel
from typing import List, Optional


class TransactionSample(BaseModel):
    """A single transaction for fraud scoring."""
    V1: float = 0.0
    V2: float = 0.0
    V3: float = 0.0
    V4: float = 0.0
    V5: float = 0.0
    V6: float = 0.0
    V7: float = 0.0
    V8: float = 0.0
    V9: float = 0.0
    V10: float = 0.0
    V11: float = 0.0
    V12: float = 0.0
    V13: float = 0.0
    V14: float = 0.0
    V15: float = 0.0
    V16: float = 0.0
    V17: float = 0.0
    V18: float = 0.0
    V19: float = 0.0
    V20: float = 0.0
    V21: float = 0.0
    V22: float = 0.0
    V23: float = 0.0
    V24: float = 0.0
    V25: float = 0.0
    V26: float = 0.0
    V27: float = 0.0
    V28: float = 0.0
    Amount_scaled: float = 0.0
    Time_scaled: float = 0.0


AVAILABLE_MODELS = ["random_forest", "xgboost", "lightgbm", "logistic_regression"]


class PredictionRequest(BaseModel):
    """Batch prediction request."""
    samples: List[TransactionSample]
    model_name: Optional[str] = None  # Options: random_forest, xgboost, lightgbm, logistic_regression


class PredictionResponse(BaseModel):
    """Prediction response with fraud scores."""
    predictions: List[int]
    probabilities: List[float]
    model_name: str
    model_version: str
    n_samples: int
    available_models: List[str] = AVAILABLE_MODELS
