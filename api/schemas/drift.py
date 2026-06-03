"""Drift diagnosis request/response schemas."""

from pydantic import BaseModel
from typing import List, Dict, Optional, Any


class DriftSample(BaseModel):
    """A single sample for drift analysis (same features as prediction)."""
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


class DriftDiagnoseRequest(BaseModel):
    """Request body for drift diagnosis."""
    current_samples: List[DriftSample]


class DriftDiagnoseResponse(BaseModel):
    """Full drift diagnosis response."""
    drift_type: str
    severity: str
    action: str
    confidence: float
    covariate_score: float
    concept_score: float
    pipeline_score: float
    target_score: float
    drifted_features: List[str]
    pipeline_issues: List[dict]
    reasoning: str
    urgency_hours: int
    playbook: Optional[dict] = None
    diagnosed_at: str


class DriftStatusResponse(BaseModel):
    """Current drift monitoring status."""
    last_check: Optional[str] = None
    current_drift_type: str
    current_severity: str
    checks_run_24h: int
    alerts_triggered: int
