"""Action recommendation based on drift type and severity."""

from src.config.constants import DriftType, Severity, ActionType, ACTION_MAPPING


class ActionRecommender:
    """Maps drift classification + severity to operational actions."""

    def recommend(self, drift_type: DriftType, severity: Severity) -> ActionType:
        """Get recommended action for given drift state."""
        if drift_type == DriftType.NONE or severity == Severity.NONE:
            return ActionType.MONITOR

        key = (drift_type, severity)
        return ACTION_MAPPING.get(key, ActionType.INVESTIGATE)

    def get_action_description(self, action: ActionType) -> str:
        """Get human-readable description of recommended action."""
        descriptions = {
            ActionType.MONITOR: "Continue monitoring. No immediate action required.",
            ActionType.ALERT: "Send alert to ML team. Schedule investigation.",
            ActionType.INVESTIGATE: "Investigate root cause. Check upstream data pipelines.",
            ActionType.INCREMENTAL_RETRAIN: (
                "Retrain model incrementally with recent data. "
                "Update feature distributions."
            ),
            ActionType.FULL_RETRAIN: (
                "Full model retraining required. Reassess feature engineering, "
                "hyperparameters, and training data window."
            ),
            ActionType.BLOCK: (
                "BLOCK predictions immediately. Data quality issue detected. "
                "Fix upstream pipeline before resuming."
            ),
        }
        return descriptions.get(action, "Unknown action.")

    def get_urgency_hours(self, action: ActionType) -> int:
        """Get response time SLA in hours."""
        urgency = {
            ActionType.MONITOR: 168,    # 1 week
            ActionType.ALERT: 72,       # 3 days
            ActionType.INVESTIGATE: 24, # 1 day
            ActionType.INCREMENTAL_RETRAIN: 12,
            ActionType.FULL_RETRAIN: 4,
            ActionType.BLOCK: 1,        # immediate
        }
        return urgency.get(action, 24)
