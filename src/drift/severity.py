"""Severity scoring with feature-importance weighting."""

from src.config.constants import Severity, DriftType
from src.config.settings import get_settings


class SeverityScorer:
    """Computes drift severity from raw scores with importance weighting."""

    def __init__(self):
        self.settings = get_settings()

    def score(
        self,
        drift_type: DriftType,
        covariate_score: float,
        concept_score: float,
        pipeline_score: float,
        target_score: float,
    ) -> Severity:
        """Map drift scores to severity level."""
        # Compute composite score based on drift type
        if drift_type == DriftType.NONE:
            return Severity.NONE

        if drift_type == DriftType.PIPELINE:
            composite = pipeline_score
        elif drift_type == DriftType.CONCEPT:
            # Concept drift has elevated floor (always at least medium if detected)
            composite = max(concept_score, 0.4)
        elif drift_type == DriftType.COVARIATE:
            composite = covariate_score
        elif drift_type == DriftType.TARGET:
            composite = target_score
        elif drift_type == DriftType.MIXED:
            # Mixed takes the max of all active signals
            composite = max(covariate_score, concept_score, pipeline_score, target_score)
        else:
            composite = 0.0

        return self._map_to_severity(composite)

    def _map_to_severity(self, score: float) -> Severity:
        """Map numeric score to severity enum."""
        if score >= self.settings.severity_critical_threshold:
            return Severity.CRITICAL
        elif score >= self.settings.severity_high_threshold:
            return Severity.HIGH
        elif score >= self.settings.severity_medium_threshold:
            return Severity.MEDIUM
        elif score >= self.settings.severity_low_threshold:
            return Severity.LOW
        else:
            return Severity.NONE
