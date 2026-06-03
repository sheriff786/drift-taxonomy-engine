"""Main Drift Taxonomy Engine - orchestrates all detection, classification, and action."""

import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from src.drift.detectors.covariate import CovariateDriftDetector
from src.drift.detectors.concept import ConceptDriftDetector
from src.drift.detectors.pipeline import PipelineDriftDetector
from src.drift.detectors.target import TargetDriftDetector
from src.drift.classifiers import DriftClassifier, ClassificationResult
from src.drift.severity import SeverityScorer
from src.drift.actions import ActionRecommender
from src.drift.playbook import PlaybookGenerator, DriftPlaybook
from src.config.constants import DriftType, Severity, ActionType


@dataclass
class DriftDiagnosis:
    """Complete drift diagnosis result."""
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
    playbook: Optional[dict]
    diagnosed_at: str

    def to_dict(self) -> dict:
        return asdict(self)


class DriftTaxonomyEngine:
    """
    Production drift detection engine that diagnoses drift type, severity,
    and recommends operational actions with response playbooks.
    """

    def __init__(
        self,
        feature_importances: Dict[str, float],
        baseline_auprc: float = 0.87,
        baseline_f1: float = 0.82,
    ):
        self.feature_importances = feature_importances
        self.covariate_detector = CovariateDriftDetector()
        self.concept_detector = ConceptDriftDetector(baseline_auprc, baseline_f1)
        self.pipeline_detector = PipelineDriftDetector()
        self.target_detector = TargetDriftDetector()
        self.classifier = DriftClassifier()
        self.severity_scorer = SeverityScorer()
        self.action_recommender = ActionRecommender()
        self.playbook_generator = PlaybookGenerator()

    def diagnose(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        model: Optional[Any] = None,
        reference_y: Optional[pd.Series] = None,
        current_y: Optional[pd.Series] = None,
        X_current: Optional[np.ndarray] = None,
        generate_playbook: bool = True,
    ) -> DriftDiagnosis:
        """
        Run full drift diagnosis pipeline.

        Args:
            reference: Reference (baseline) feature data
            current: Current (production) feature data
            model: Trained model for concept drift detection
            reference_y: Reference target labels (for target drift)
            current_y: Current target labels (for concept + target drift)
            X_current: Current features for model prediction
            generate_playbook: Whether to generate response playbook

        Returns:
            DriftDiagnosis with complete analysis
        """
        # 1. Pipeline drift detection
        pipeline_signals = self.pipeline_detector.detect(reference, current)
        pipeline_score = self.pipeline_detector.compute_drift_score(pipeline_signals)

        # 2. Covariate drift detection
        covariate_signals = self.covariate_detector.detect(reference, current)
        covariate_score = self.covariate_detector.compute_drift_score(
            covariate_signals, self.feature_importances
        )
        drifted_features = self.covariate_detector.get_drifted_features(covariate_signals)

        # 3. Concept drift detection (requires model + labels)
        concept_score = 0.0
        if model is not None and current_y is not None and X_current is not None:
            concept_signal = self.concept_detector.detect(model, X_current, current_y)
            concept_score = self.concept_detector.compute_drift_score(concept_signal)

        # 4. Target drift detection (requires labels)
        target_score = 0.0
        if reference_y is not None and current_y is not None:
            target_signal = self.target_detector.detect(reference_y, current_y)
            target_score = self.target_detector.compute_drift_score(target_signal)

        # 5. Classify drift type
        classification = self.classifier.classify(
            covariate_score, concept_score, pipeline_score, target_score
        )

        # 6. Score severity
        severity = self.severity_scorer.score(
            classification.drift_type,
            covariate_score, concept_score, pipeline_score, target_score,
        )

        # 7. Recommend action
        action = self.action_recommender.recommend(classification.drift_type, severity)
        urgency = self.action_recommender.get_urgency_hours(action)

        # 8. Generate playbook
        playbook = None
        if generate_playbook:
            pb = self.playbook_generator.generate(
                classification.drift_type, severity, action, urgency, drifted_features
            )
            playbook = pb.to_dict()

        return DriftDiagnosis(
            drift_type=classification.drift_type.value,
            severity=severity.value,
            action=action.value,
            confidence=classification.confidence,
            covariate_score=covariate_score,
            concept_score=concept_score,
            pipeline_score=pipeline_score,
            target_score=target_score,
            drifted_features=drifted_features,
            pipeline_issues=[
                {"type": s.issue_type, "feature": s.feature, "severity": s.severity, "detail": s.detail}
                for s in pipeline_signals
            ],
            reasoning=classification.reasoning,
            urgency_hours=urgency,
            playbook=playbook,
            diagnosed_at=datetime.now().isoformat(),
        )

    def diagnose_quick(
        self, reference: pd.DataFrame, current: pd.DataFrame
    ) -> DriftDiagnosis:
        """Quick diagnosis without concept drift (no model/labels required)."""
        return self.diagnose(
            reference=reference,
            current=current,
            model=None,
            reference_y=None,
            current_y=None,
            X_current=None,
            generate_playbook=True,
        )
