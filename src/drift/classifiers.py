"""Drift type classification logic."""

from dataclasses import dataclass
from typing import Optional
from src.config.constants import DriftType


@dataclass
class ClassificationResult:
    """Result of drift type classification."""
    drift_type: DriftType
    confidence: float
    primary_signal: str
    reasoning: str


class DriftClassifier:
    """Classifies the type of drift based on detector signals."""

    def classify(
        self,
        covariate_score: float,
        concept_score: float,
        pipeline_score: float,
        target_score: float,
    ) -> ClassificationResult:
        """
        Classify drift type based on scores from all detectors.

        Priority logic:
        1. Pipeline issues take immediate priority (data quality)
        2. Concept drift is most operationally critical
        3. Mixed if multiple strong signals
        4. Covariate/Target otherwise
        """
        threshold = 0.3  # Minimum score to consider a drift type active

        active_types = []
        if pipeline_score >= threshold:
            active_types.append((DriftType.PIPELINE, pipeline_score))
        if concept_score >= threshold:
            active_types.append((DriftType.CONCEPT, concept_score))
        if covariate_score >= threshold:
            active_types.append((DriftType.COVARIATE, covariate_score))
        if target_score >= threshold:
            active_types.append((DriftType.TARGET, target_score))

        if not active_types:
            return ClassificationResult(
                drift_type=DriftType.NONE,
                confidence=1.0 - max(covariate_score, concept_score, pipeline_score, target_score),
                primary_signal="none",
                reasoning="No drift signals exceed classification threshold.",
            )

        # Pipeline always takes priority (data quality issue)
        if pipeline_score >= 0.6:
            return ClassificationResult(
                drift_type=DriftType.PIPELINE,
                confidence=pipeline_score,
                primary_signal="pipeline_quality",
                reasoning="Strong pipeline/data quality issues detected. Fix upstream first.",
            )

        # Multiple strong signals = mixed
        strong_signals = [(t, s) for t, s in active_types if s >= 0.4]
        if len(strong_signals) >= 2:
            max_type, max_score = max(strong_signals, key=lambda x: x[1])
            return ClassificationResult(
                drift_type=DriftType.MIXED,
                confidence=max_score,
                primary_signal=max_type.value,
                reasoning=f"Multiple drift types active: {[t.value for t, _ in strong_signals]}",
            )

        # Concept drift priority over covariate
        if concept_score >= threshold and concept_score >= covariate_score:
            return ClassificationResult(
                drift_type=DriftType.CONCEPT,
                confidence=concept_score,
                primary_signal="performance_decay",
                reasoning="Model performance has degraded, indicating concept drift.",
            )

        # Single dominant type
        dominant_type, dominant_score = max(active_types, key=lambda x: x[1])
        return ClassificationResult(
            drift_type=dominant_type,
            confidence=dominant_score,
            primary_signal=dominant_type.value,
            reasoning=f"Primary drift type: {dominant_type.value} (score={dominant_score:.3f})",
        )
