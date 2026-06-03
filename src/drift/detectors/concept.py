"""Concept drift detector based on model performance decay."""

import numpy as np
from sklearn.metrics import average_precision_score, f1_score
from dataclasses import dataclass
from typing import Any, Optional
from src.config.settings import get_settings


@dataclass
class ConceptSignal:
    """Signal from concept drift detection."""
    baseline_auprc: float
    current_auprc: float
    performance_drop: float
    baseline_f1: float
    current_f1: float
    f1_drop: float
    is_drifted: bool


class ConceptDriftDetector:
    """Detects concept drift via performance degradation."""

    def __init__(self, baseline_auprc: float = 0.87, baseline_f1: float = 0.82):
        self.settings = get_settings()
        self.baseline_auprc = baseline_auprc
        self.baseline_f1 = baseline_f1

    def detect(
        self,
        model: Any,
        X_current: np.ndarray,
        y_current: np.ndarray,
    ) -> ConceptSignal:
        """Detect concept drift by comparing current performance to baseline."""
        y_proba = model.predict_proba(X_current)[:, 1]
        y_pred = model.predict(X_current)

        current_auprc = average_precision_score(y_current, y_proba)
        current_f1 = f1_score(y_current, y_pred)

        performance_drop = (self.baseline_auprc - current_auprc) / self.baseline_auprc
        f1_drop = (self.baseline_f1 - current_f1) / self.baseline_f1

        is_drifted = performance_drop >= self.settings.performance_decay_threshold

        return ConceptSignal(
            baseline_auprc=self.baseline_auprc,
            current_auprc=current_auprc,
            performance_drop=performance_drop,
            baseline_f1=self.baseline_f1,
            current_f1=current_f1,
            f1_drop=f1_drop,
            is_drifted=is_drifted,
        )

    def compute_drift_score(self, signal: ConceptSignal) -> float:
        """Compute concept drift severity score (0-1)."""
        # Combine AUPRC drop and F1 drop with AUPRC weighted higher
        score = 0.7 * max(signal.performance_drop, 0) + 0.3 * max(signal.f1_drop, 0)
        return min(score, 1.0)
