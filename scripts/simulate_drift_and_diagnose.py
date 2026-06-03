"""
Simulate realistic drift scenarios and run the Drift Taxonomy Engine.
This validates the core idea by injecting known drift patterns and verifying detection.

Drift Scenarios Applied:
1. COVARIATE DRIFT: Shift top fraud-detecting features (V14, V10, V12, V4) by 1-2 std devs
2. PIPELINE DRIFT: Inject nulls into random features + out-of-range values
3. CONCEPT DRIFT: Flip some fraud labels to simulate decision boundary shift
4. MIXED DRIFT: Combine covariate + pipeline issues

The script generates a drifted dataset, runs the engine, saves the report,
and prints a summary so you can verify in the dashboard.
"""

import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.constants import FEATURE_NAME_MAPPING, TOP_FEATURES_ORIGINAL
from src.config.settings import get_settings
from src.data.ingestion import DataIngestion
from src.drift.engine import DriftTaxonomyEngine
from src.features.feature_store import FeatureStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def simulate_covariate_drift(df: pd.DataFrame, features_to_shift: list, shift_magnitude: float = 1.5) -> pd.DataFrame:
    """Shift specified features by N standard deviations to simulate covariate drift."""
    drifted = df.copy()
    for feat in features_to_shift:
        if feat in drifted.columns:
            std = drifted[feat].std()
            # Shift mean by shift_magnitude * std
            drifted[feat] = drifted[feat] + (shift_magnitude * std)
            logger.info(f"  Shifted {feat} ({FEATURE_NAME_MAPPING.get(feat, feat)}) by +{shift_magnitude} std ({shift_magnitude * std:.3f})")
    return drifted


def simulate_pipeline_drift(df: pd.DataFrame, null_rate: float = 0.05, n_outliers: int = 50) -> pd.DataFrame:
    """Inject nulls and extreme outliers to simulate pipeline/data quality issues."""
    drifted = df.copy()
    np.random.seed(42)

    # Inject nulls in 3 random features
    null_features = np.random.choice([c for c in drifted.columns if c.startswith("V")], 3, replace=False)
    for feat in null_features:
        mask = np.random.random(len(drifted)) < null_rate
        drifted.loc[mask, feat] = np.nan
        n_nulls = mask.sum()
        logger.info(f"  Injected {n_nulls} nulls into {feat} ({FEATURE_NAME_MAPPING.get(feat, feat)})")

    # Inject extreme outliers (10x std) in 2 features
    outlier_features = np.random.choice([c for c in drifted.columns if c.startswith("V")], 2, replace=False)
    for feat in outlier_features:
        std = drifted[feat].std()
        outlier_idx = np.random.choice(len(drifted), n_outliers, replace=False)
        drifted.loc[outlier_idx, feat] = drifted[feat].mean() + (10 * std * np.random.choice([-1, 1], n_outliers))
        logger.info(f"  Injected {n_outliers} extreme outliers into {feat} ({FEATURE_NAME_MAPPING.get(feat, feat)})")

    return drifted


def simulate_variance_change(df: pd.DataFrame, features: list, scale_factor: float = 3.0) -> pd.DataFrame:
    """Increase variance of features (distribution spread change)."""
    drifted = df.copy()
    for feat in features:
        if feat in drifted.columns:
            mean = drifted[feat].mean()
            drifted[feat] = mean + (drifted[feat] - mean) * scale_factor
            logger.info(f"  Scaled variance of {feat} ({FEATURE_NAME_MAPPING.get(feat, feat)}) by {scale_factor}x")
    return drifted


def run_drift_simulation():
    """Main simulation: generate drifted data and run the taxonomy engine."""

    settings = get_settings()
    ingestion = DataIngestion()

    # Load reference data
    logger.info("=" * 60)
    logger.info("DRIFT TAXONOMY ENGINE - VALIDATION TEST")
    logger.info("=" * 60)

    try:
        reference = ingestion.load_reference()
        logger.info(f"Loaded reference data: {reference.shape}")
    except FileNotFoundError:
        logger.error("No reference data found. Run training pipeline first.")
        return

    # Load feature importances
    feature_store = FeatureStore()
    try:
        importances = feature_store.get_feature_importances()
        # Convert domain names back to original for the engine
        from src.config.constants import FEATURE_NAME_REVERSE
        original_importances = {FEATURE_NAME_REVERSE.get(k, k): v for k, v in importances.items()}
        logger.info(f"Loaded feature importances: {len(original_importances)} features")
    except FileNotFoundError:
        original_importances = {f"V{i}": 1/30 for i in range(1, 29)}
        original_importances.update({"Amount_scaled": 1/30, "Time_scaled": 1/30})
        logger.warning("Using uniform importances (no feature store found)")

    # Take second half as base for "current" data and apply drift
    split_idx = len(reference) // 2
    ref_data = reference.iloc[:split_idx].reset_index(drop=True)
    current_base = reference.iloc[split_idx:].reset_index(drop=True)

    logger.info(f"\nReference split: {ref_data.shape[0]} samples")
    logger.info(f"Current base: {current_base.shape[0]} samples")

    # ============================================================
    # SCENARIO: Mixed Drift (Covariate + Pipeline + Variance)
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("APPLYING DRIFT SCENARIO: MIXED (Covariate + Pipeline + Variance)")
    logger.info("=" * 60)

    # 1. Covariate drift: shift top 5 most important features
    top_features_to_shift = TOP_FEATURES_ORIGINAL[:5]  # V14, V10, V12, V4, V11
    logger.info(f"\n[1] COVARIATE DRIFT - Shifting top features by 1.5 std:")
    drifted_data = simulate_covariate_drift(current_base, top_features_to_shift, shift_magnitude=1.5)

    # 2. Variance change: expand variance on V3, V7
    logger.info(f"\n[2] VARIANCE CHANGE - Scaling variance of V3, V7:")
    drifted_data = simulate_variance_change(drifted_data, ["V3", "V7"], scale_factor=2.5)

    # 3. Pipeline drift: inject nulls and outliers
    logger.info(f"\n[3] PIPELINE DRIFT - Injecting nulls and outliers:")
    drifted_data = simulate_pipeline_drift(drifted_data, null_rate=0.03, n_outliers=100)

    # Fill nulls for the engine (it needs numeric data for KS tests)
    drifted_clean = drifted_data.fillna(drifted_data.median())

    # ============================================================
    # RUN DRIFT TAXONOMY ENGINE
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("RUNNING DRIFT TAXONOMY ENGINE...")
    logger.info("=" * 60)

    engine = DriftTaxonomyEngine(
        feature_importances=original_importances,
        baseline_auprc=0.866,
        baseline_f1=0.802,
    )

    # Run full quick diagnosis (no model/labels for concept drift)
    diagnosis = engine.diagnose_quick(reference=ref_data, current=drifted_clean)

    # ============================================================
    # SAVE RESULTS
    # ============================================================
    reports_dir = settings.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_filename = f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path = reports_dir / report_filename

    # Add feature_scores for dashboard visualization
    report_data = diagnosis.to_dict()

    # Compute per-feature drift scores for the report
    covariate_signals = engine.covariate_detector.detect(ref_data, drifted_clean)
    feature_scores = {}
    for signal in covariate_signals:
        domain_name = FEATURE_NAME_MAPPING.get(signal.feature, signal.feature)
        feature_scores[domain_name] = round(signal.cohens_d, 4)

    report_data["feature_scores"] = feature_scores
    report_data["overall_score"] = max(
        diagnosis.covariate_score,
        diagnosis.concept_score,
        diagnosis.pipeline_score,
        diagnosis.target_score,
    )
    # Convert drifted features to domain names
    report_data["drifted_features"] = [
        FEATURE_NAME_MAPPING.get(f, f) for f in diagnosis.drifted_features
    ]

    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)

    logger.info(f"\nReport saved: {report_path}")

    # ============================================================
    # PRINT SUMMARY
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("DIAGNOSIS RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Drift Type:       {diagnosis.drift_type}")
    logger.info(f"  Severity:         {diagnosis.severity}")
    logger.info(f"  Action:           {diagnosis.action}")
    logger.info(f"  Urgency:          {diagnosis.urgency_hours} hours")
    logger.info(f"  Confidence:       {diagnosis.confidence:.3f}")
    logger.info(f"")
    logger.info(f"  Scores:")
    logger.info(f"    Covariate:      {diagnosis.covariate_score:.4f}")
    logger.info(f"    Concept:        {diagnosis.concept_score:.4f}")
    logger.info(f"    Pipeline:       {diagnosis.pipeline_score:.4f}")
    logger.info(f"    Target:         {diagnosis.target_score:.4f}")
    logger.info(f"")
    logger.info(f"  Drifted Features ({len(diagnosis.drifted_features)}):")
    for feat in diagnosis.drifted_features:
        domain = FEATURE_NAME_MAPPING.get(feat, feat)
        score = feature_scores.get(domain, 0)
        logger.info(f"    - {domain} (Cohen's d = {score:.3f})")

    if diagnosis.pipeline_issues:
        logger.info(f"\n  Pipeline Issues ({len(diagnosis.pipeline_issues)}):")
        for issue in diagnosis.pipeline_issues:
            logger.info(f"    - [{issue.get('severity', '?')}] {issue.get('type', '?')}: {issue.get('detail', '')}")

    if diagnosis.playbook:
        logger.info(f"\n  Playbook:")
        logger.info(f"    Priority: {diagnosis.playbook.get('recommended_action', 'N/A')}")
        steps = diagnosis.playbook.get("steps", [])
        if isinstance(steps, list):
            for step in steps[:5]:
                if isinstance(step, dict):
                    logger.info(f"    {step.get('order', '?')}. {step.get('action', '')} - {step.get('description', '')}")
                else:
                    logger.info(f"    - {step}")

    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION COMPLETE - Check dashboard at http://localhost:8501")
    logger.info("=" * 60)

    return report_data


if __name__ == "__main__":
    result = run_drift_simulation()
