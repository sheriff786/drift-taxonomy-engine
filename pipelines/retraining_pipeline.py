"""Auto-retraining pipeline triggered by drift detection."""

import logging
from src.config.constants import ActionType
from pipelines.training_pipeline import run_training_pipeline
from pipelines.drift_pipeline import run_drift_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_retraining_pipeline(current_data_path: str = None) -> dict:
    """
    Check for drift and automatically retrain if action threshold met.

    Returns:
        dict with drift diagnosis and retraining result (if triggered).
    """
    logger.info("Running retraining decision pipeline...")

    # 1. Run drift check
    drift_result = run_drift_pipeline(current_data_path)
    action = drift_result["action"]

    logger.info(f"Drift action: {action}")

    # 2. Decision: retrain or not
    retrain_actions = {
        ActionType.INCREMENTAL_RETRAIN.value,
        ActionType.FULL_RETRAIN.value,
    }

    if action in retrain_actions:
        logger.info(f"Retraining triggered by action: {action}")
        training_result = run_training_pipeline()
        return {
            "drift": drift_result,
            "retrained": True,
            "training_result": training_result,
        }
    elif action == ActionType.BLOCK.value:
        logger.critical("PIPELINE BLOCKED - Manual intervention required!")
        return {
            "drift": drift_result,
            "retrained": False,
            "blocked": True,
            "message": "Pipeline blocked due to data quality issues.",
        }
    else:
        logger.info(f"No retraining needed. Action: {action}")
        return {
            "drift": drift_result,
            "retrained": False,
            "message": f"Action '{action}' does not require retraining.",
        }


if __name__ == "__main__":
    result = run_retraining_pipeline()
    print(f"\nRetraining pipeline result: retrained={result.get('retrained')}")
