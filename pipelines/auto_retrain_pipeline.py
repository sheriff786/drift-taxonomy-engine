"""
Auto-Retrain Pipeline — Triggered when drift severity exceeds threshold.

This pipeline:
1. Detects that drift is critical/high
2. Loads fresh data (recent window)
3. Retrains all candidate models
4. Compares against current champion (AUPRC)
5. Promotes the new model if it performs better
6. Updates reference data
7. Re-runs drift check to confirm stabilization
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.config.settings import get_settings
from src.config.constants import (
    MODEL_CONFIGS, DriftType, Severity,
    FEATURE_NAME_MAPPING, TOP_FEATURES_ORIGINAL,
)
from src.data.ingestion import DataIngestion
from src.data.preprocessing import DataPreprocessor
from src.models.registry import ModelRegistry
from src.features.feature_store import FeatureStore
from src.features.feature_engineering import FeatureEngineer
from src.drift.engine import DriftTaxonomyEngine

logger = logging.getLogger(__name__)


class AutoRetrainPipeline:
    """Drift-triggered automatic model retraining pipeline."""

    # Thresholds to trigger retrain
    RETRAIN_THRESHOLDS = {
        "covariate_score": 0.5,   # Critical covariate drift
        "concept_score": 0.2,     # Significant performance decay
        "severity_trigger": ["high", "critical"],
    }

    def __init__(self):
        self.settings = get_settings()
        self.registry = ModelRegistry()
        self.ingestion = DataIngestion()
        self.preprocessor = DataPreprocessor()
        self.feature_store = FeatureStore()

    def should_retrain(self, drift_report: Dict) -> bool:
        """Check if drift report warrants automatic retraining."""
        severity = drift_report.get("severity", "none")
        covariate_score = drift_report.get("covariate_score", 0)
        concept_score = drift_report.get("concept_score", 0)

        if severity in self.RETRAIN_THRESHOLDS["severity_trigger"]:
            logger.info(f"Retrain triggered: severity={severity}")
            return True
        if covariate_score >= self.RETRAIN_THRESHOLDS["covariate_score"]:
            logger.info(f"Retrain triggered: covariate_score={covariate_score:.3f}")
            return True
        if concept_score >= self.RETRAIN_THRESHOLDS["concept_score"]:
            logger.info(f"Retrain triggered: concept_score={concept_score:.3f}")
            return True

        logger.info(f"Retrain NOT triggered: severity={severity}, cov={covariate_score:.3f}")
        return False

    def get_champion_metrics(self) -> Dict:
        """Get current champion model's metrics."""
        try:
            latest = self.registry.get_latest_model_name()
            info = self.registry.get_model_info(latest)
            return info.get("metrics", {})
        except Exception:
            return {"auprc": 0.0, "auroc": 0.0, "f1": 0.0}

    def train_challenger_models(
        self, X_train: pd.DataFrame, y_train: pd.Series,
        X_test: pd.DataFrame, y_test: pd.Series,
    ) -> Dict[str, Dict]:
        """Train all candidate models and evaluate."""
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        from sklearn.linear_model import LogisticRegression
        from imblearn.over_sampling import SMOTE

        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        logger.info(f"SMOTE applied: {len(X_train_res)} samples (from {len(X_train)})")

        models = {
            "random_forest": RandomForestClassifier(**MODEL_CONFIGS["random_forest"], random_state=42, n_jobs=-1),
            "xgboost": XGBClassifier(**MODEL_CONFIGS["xgboost"], random_state=42, eval_metric="aucpr", verbosity=0),
            "lightgbm": LGBMClassifier(**MODEL_CONFIGS["lightgbm"], random_state=42, verbose=-1),
            "logistic_regression": LogisticRegression(**MODEL_CONFIGS["logistic_regression"], random_state=42, max_iter=1000),
        }

        results = {}
        for name, model in models.items():
            logger.info(f"Training challenger: {name}")
            model.fit(X_train_res, y_train_res)

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            metrics = {
                "auprc": float(average_precision_score(y_test, y_proba)),
                "auroc": float(roc_auc_score(y_test, y_proba)),
                "f1": float(f1_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred)),
                "recall": float(recall_score(y_test, y_pred)),
            }

            results[name] = {"model": model, "metrics": metrics}
            logger.info(f"  {name}: AUPRC={metrics['auprc']:.4f}, F1={metrics['f1']:.4f}")

        return results

    def run(self, drift_report: Optional[Dict] = None, force: bool = False) -> Dict:
        """
        Execute auto-retrain pipeline.

        Args:
            drift_report: Latest drift diagnosis (if None, loads from disk)
            force: Force retrain even if thresholds not met

        Returns:
            Dict with retrain results and decisions
        """
        logger.info("=" * 60)
        logger.info("AUTO-RETRAIN PIPELINE STARTED")
        logger.info("=" * 60)

        # 1. Load or check drift report
        if drift_report is None:
            reports_dir = self.settings.reports_dir
            reports = sorted(reports_dir.glob("drift_report_*.json"), reverse=True)
            if reports:
                with open(reports[0]) as f:
                    drift_report = json.load(f)
                logger.info(f"Loaded drift report: {reports[0].name}")
            else:
                logger.warning("No drift reports found. Cannot determine retrain need.")
                return {"status": "skipped", "reason": "no_drift_report"}

        # 2. Check if retrain is needed
        if not force and not self.should_retrain(drift_report):
            return {"status": "skipped", "reason": "below_threshold", "report": drift_report}

        # 3. Load data
        logger.info("\n[Step 1] Loading training data...")
        reference = self.ingestion.load_reference()
        full_data = self.ingestion.load_data()

        # Preprocess
        processed = self.preprocessor.fit_transform(full_data)
        X, y = self.preprocessor.separate_features_target(processed)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logger.info(f"Data ready: train={len(X_train)}, test={len(X_test)}")

        # 4. Get champion metrics
        champion_metrics = self.get_champion_metrics()
        champion_auprc = champion_metrics.get("auprc", 0.0)
        logger.info(f"\n[Step 2] Champion AUPRC: {champion_auprc:.4f}")

        # 5. Train challengers
        logger.info("\n[Step 3] Training challenger models...")
        results = self.train_challenger_models(X_train, y_train, X_test, y_test)

        # 6. Find best challenger
        best_name = max(results, key=lambda k: results[k]["metrics"]["auprc"])
        best_metrics = results[best_name]["metrics"]
        best_model = results[best_name]["model"]
        logger.info(f"\n[Step 4] Best challenger: {best_name} (AUPRC={best_metrics['auprc']:.4f})")

        # 7. Compare champion vs challenger
        improvement = best_metrics["auprc"] - champion_auprc
        promote = improvement > -0.01  # Promote if not worse by >1%

        logger.info(f"\n[Step 5] Champion vs Challenger comparison:")
        logger.info(f"  Champion AUPRC:   {champion_auprc:.4f}")
        logger.info(f"  Challenger AUPRC: {best_metrics['auprc']:.4f}")
        logger.info(f"  Improvement:      {improvement:+.4f}")
        logger.info(f"  Decision:         {'PROMOTE' if promote else 'KEEP CURRENT'}")

        retrain_result = {
            "status": "completed",
            "triggered_by": drift_report.get("drift_type", "unknown"),
            "severity": drift_report.get("severity", "unknown"),
            "champion_auprc": champion_auprc,
            "challenger_name": best_name,
            "challenger_auprc": best_metrics["auprc"],
            "improvement": improvement,
            "promoted": promote,
            "all_results": {k: v["metrics"] for k, v in results.items()},
            "timestamp": datetime.now().isoformat(),
        }

        if promote:
            # 8. Register and promote
            logger.info(f"\n[Step 6] Promoting {best_name} to production...")
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

            for name, data in results.items():
                self.registry.save_model(
                    model=data["model"],
                    model_name=name,
                    metrics=data["metrics"],
                    version=version,
                )

            # Set latest
            reg_data = self.registry._load_registry()
            reg_data["latest"] = best_name
            self.registry._save_registry(reg_data)

            # 9. Update reference data
            logger.info("\n[Step 7] Updating reference data...")
            self.ingestion.save_reference(X_test)

            # Update feature store with new importances
            feature_engineer = FeatureEngineer()
            importances = feature_engineer.compute_feature_importance_weights(
                best_model, list(X_train.columns)
            )
            domain_importances = {
                FEATURE_NAME_MAPPING.get(k, k): v for k, v in importances.items()
            }
            self.feature_store.save_feature_schema(
                feature_names=[FEATURE_NAME_MAPPING.get(c, c) for c in X_train.columns],
                feature_importances=domain_importances,
            )

            # 10. Re-run drift check
            logger.info("\n[Step 8] Re-running drift check after retrain...")
            new_importances = {k: v for k, v in importances.items()}
            engine = DriftTaxonomyEngine(
                feature_importances=new_importances,
                baseline_auprc=best_metrics["auprc"],
                baseline_f1=best_metrics["f1"],
            )

            new_ref = X_test.iloc[:len(X_test)//2]
            new_cur = X_test.iloc[len(X_test)//2:]
            post_diagnosis = engine.diagnose_quick(new_ref, new_cur)
            retrain_result["post_retrain_drift"] = {
                "drift_type": post_diagnosis.drift_type,
                "severity": post_diagnosis.severity,
                "covariate_score": post_diagnosis.covariate_score,
            }
            logger.info(f"  Post-retrain drift: {post_diagnosis.drift_type} / {post_diagnosis.severity}")

            retrain_result["new_version"] = version
        else:
            logger.info("\n[Step 6] Keeping current champion (challenger not better).")

        # Save retrain report
        report_path = self.settings.reports_dir / f"retrain_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, "w") as f:
            json.dump(retrain_result, f, indent=2)
        logger.info(f"\nRetrain report saved: {report_path}")

        logger.info("\n" + "=" * 60)
        logger.info("AUTO-RETRAIN PIPELINE COMPLETE")
        logger.info("=" * 60)

        return retrain_result


def run_auto_retrain(force: bool = False) -> Dict:
    """Entry point for auto-retrain pipeline."""
    pipeline = AutoRetrainPipeline()
    return pipeline.run(force=force)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    force = "--force" in sys.argv
    result = run_auto_retrain(force=force)
    print(f"\nResult: {result['status']}")
