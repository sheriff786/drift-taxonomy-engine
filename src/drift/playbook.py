"""Response playbook generation for drift actions."""

from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict
from src.config.constants import DriftType, Severity, ActionType


@dataclass
class PlaybookStep:
    """A single step in a response playbook."""
    order: int
    action: str
    owner: str
    description: str


@dataclass
class DriftPlaybook:
    """Complete response playbook for a drift diagnosis."""
    drift_type: str
    severity: str
    recommended_action: str
    urgency_hours: int
    steps: List[PlaybookStep]
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)


class PlaybookGenerator:
    """Generates actionable response playbooks based on drift diagnosis."""

    def generate(
        self,
        drift_type: DriftType,
        severity: Severity,
        action: ActionType,
        urgency_hours: int,
        drifted_features: Optional[List[str]] = None,
    ) -> DriftPlaybook:
        """Generate a playbook for the diagnosed drift scenario."""
        steps = self._get_steps(drift_type, action, drifted_features or [])

        return DriftPlaybook(
            drift_type=drift_type.value,
            severity=severity.value,
            recommended_action=action.value,
            urgency_hours=urgency_hours,
            steps=steps,
        )

    def _get_steps(
        self, drift_type: DriftType, action: ActionType, drifted_features: List[str]
    ) -> List[PlaybookStep]:
        """Build step list based on action type."""
        if action == ActionType.BLOCK:
            return self._block_playbook(drift_type, drifted_features)
        elif action == ActionType.FULL_RETRAIN:
            return self._full_retrain_playbook(drift_type, drifted_features)
        elif action == ActionType.INCREMENTAL_RETRAIN:
            return self._incremental_retrain_playbook(drifted_features)
        elif action == ActionType.INVESTIGATE:
            return self._investigate_playbook(drift_type, drifted_features)
        elif action == ActionType.ALERT:
            return self._alert_playbook(drift_type)
        else:
            return self._monitor_playbook()

    def _block_playbook(self, drift_type: DriftType, features: List[str]) -> List[PlaybookStep]:
        return [
            PlaybookStep(1, "Block", "ML Platform", "Halt prediction serving immediately."),
            PlaybookStep(2, "Notify", "ML Engineer", "Page on-call engineer for pipeline failure."),
            PlaybookStep(3, "Diagnose", "Data Engineer", f"Investigate data quality for: {features[:5]}"),
            PlaybookStep(4, "Fix", "Data Engineer", "Repair upstream pipeline or data source."),
            PlaybookStep(5, "Validate", "ML Engineer", "Re-run validation checks before resuming."),
            PlaybookStep(6, "Resume", "ML Platform", "Re-enable prediction serving after fix verified."),
        ]

    def _full_retrain_playbook(self, drift_type: DriftType, features: List[str]) -> List[PlaybookStep]:
        return [
            PlaybookStep(1, "Alert", "ML Engineer", "Notify team of concept drift detection."),
            PlaybookStep(2, "Analyze", "Data Scientist", "Investigate performance decay root cause."),
            PlaybookStep(3, "Collect", "Data Engineer", "Gather fresh labeled data from recent window."),
            PlaybookStep(4, "Retrain", "ML Engineer", "Full retraining with updated data and hyperparameter search."),
            PlaybookStep(5, "Validate", "Data Scientist", "Champion/challenger comparison on holdout."),
            PlaybookStep(6, "Deploy", "ML Platform", "Promote new model version to production."),
            PlaybookStep(7, "Monitor", "ML Engineer", "Watch post-deployment metrics for 48 hours."),
        ]

    def _incremental_retrain_playbook(self, features: List[str]) -> List[PlaybookStep]:
        return [
            PlaybookStep(1, "Alert", "ML Engineer", "Notify team of drift requiring retrain."),
            PlaybookStep(2, "Update", "ML Engineer", f"Retrain with recent data, focus on: {features[:5]}"),
            PlaybookStep(3, "Validate", "Data Scientist", "Run regression tests on updated model."),
            PlaybookStep(4, "Deploy", "ML Platform", "Canary deploy updated model."),
            PlaybookStep(5, "Monitor", "ML Engineer", "Verify drift metrics stabilize post-deploy."),
        ]

    def _investigate_playbook(self, drift_type: DriftType, features: List[str]) -> List[PlaybookStep]:
        return [
            PlaybookStep(1, "Analyze", "Data Scientist", f"Investigate drift in: {features[:5]}"),
            PlaybookStep(2, "Root Cause", "Data Engineer", "Check if upstream changes explain shift."),
            PlaybookStep(3, "Decision", "ML Engineer", "Determine if retrain is needed or drift is benign."),
        ]

    def _alert_playbook(self, drift_type: DriftType) -> List[PlaybookStep]:
        return [
            PlaybookStep(1, "Log", "ML Platform", f"Log {drift_type.value} drift alert."),
            PlaybookStep(2, "Monitor", "ML Engineer", "Increase monitoring frequency for next 24h."),
        ]

    def _monitor_playbook(self) -> List[PlaybookStep]:
        return [
            PlaybookStep(1, "Continue", "ML Platform", "No action. Continue standard monitoring."),
        ]
