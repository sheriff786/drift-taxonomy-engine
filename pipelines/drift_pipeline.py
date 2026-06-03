"""Scheduled drift monitoring pipeline."""

import logging
import json
from datetime import datetime
from pathlib import Path
from src.data.ingestion import DataIngestion
from src.drift.engine import DriftTaxonomyEngine
from src.features.feature_store import FeatureStore
from src.config.settings import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_drift_pipeline(current_data_path: str = None) -> dict:
    """
    Run drift detection against reference baseline.

    Args:
        current_data_path: Path to current production data CSV.
                          If None, uses latest reference as both (no-op test).
    """
    settings = get_settings()
    logger.info("Starting drift monitoring pipeline...")

    # Load feature importances
    try:
        feature_store = FeatureStore()
        importances = feature_store.get_feature_importances()
    except FileNotFoundError:
        logger.warning("Feature store not found. Using uniform importances.")
        importances = {f"V{i}": 1.0 / 30 for i in range(1, 29)}
        importances["Amount_scaled"] = 1.0 / 30
        importances["Time_scaled"] = 1.0 / 30

    # Load reference data
    ingestion = DataIngestion()
    reference = ingestion.load_reference()
    logger.info(f"Reference data: {reference.shape}")

    # Load current data
    if current_data_path:
        import pandas as pd
        current = pd.read_csv(current_data_path)
    else:
        # Demo mode: use second half of reference
        split_idx = len(reference) // 2
        current = reference.iloc[split_idx:].reset_index(drop=True)
        reference = reference.iloc[:split_idx].reset_index(drop=True)

    logger.info(f"Current data: {current.shape}")

    # Run diagnosis
    engine = DriftTaxonomyEngine(feature_importances=importances)
    diagnosis = engine.diagnose_quick(reference=reference, current=current)

    # Save report
    report_path = settings.reports_dir / f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(diagnosis.to_dict(), f, indent=2)

    logger.info(f"Drift type: {diagnosis.drift_type}, Severity: {diagnosis.severity}")
    logger.info(f"Action: {diagnosis.action}, Urgency: {diagnosis.urgency_hours}h")
    logger.info(f"Report saved: {report_path}")

    return diagnosis.to_dict()


if __name__ == "__main__":
    result = run_drift_pipeline()
    print(f"\nDrift check complete: {result['drift_type']} / {result['severity']}")
