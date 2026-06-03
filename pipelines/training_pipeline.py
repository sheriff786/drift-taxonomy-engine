"""End-to-end training pipeline with MLflow experiment tracking."""

import logging
import mlflow
import mlflow.sklearn
from pathlib import Path
from src.data.ingestion import DataIngestion
from src.data.preprocessing import DataPreprocessor
from src.data.validation import DataValidator
from src.data.splitter import DataSplitter
from src.models.trainer import ModelTrainer
from src.models.evaluator import ModelEvaluator
from src.models.registry import ModelRegistry
from src.features.feature_engineering import FeatureEngineer
from src.features.feature_store import FeatureStore
from src.config.settings import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_training_pipeline(model_name: str = None) -> dict:
    """
    Execute the full training pipeline with MLflow tracking:
    1. Ingest data
    2. Validate
    3. Preprocess
    4. Split
    5. Apply SMOTE
    6. Train models
    7. Evaluate and select best
    8. Register model (MLflow + local registry)
    9. Save reference data and feature store
    """
    settings = get_settings()

    # Setup MLflow
    mlflow_db_path = settings.artifacts_dir / "mlflow.db"
    mlflow_db_path.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")
    mlflow.set_experiment(settings.mlflow_experiment_name)

    logger.info("Starting training pipeline...")
    logger.info(f"MLflow tracking URI: sqlite:///{mlflow_db_path}")

    # 1. Ingest
    ingestion = DataIngestion()
    df = ingestion.load_csv("creditcard.csv")
    logger.info(f"Loaded {len(df)} samples.")

    # 2. Validate
    validator = DataValidator()
    result = validator.validate(df)
    if not result.is_valid:
        raise ValueError(f"Data validation failed: {result.errors}")
    if result.warnings:
        logger.warning(f"Validation warnings: {result.warnings}")

    # 3. Preprocess
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.fit_transform(df)

    # 4. Split
    X, y = preprocessor.separate_features_target(df_processed)
    splitter = DataSplitter()
    X_train, X_test, y_train, y_test = splitter.train_test_split(X, y)
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # 5. SMOTE
    trainer = ModelTrainer()
    X_train_resampled, y_train_resampled = trainer.apply_smote(X_train, y_train)
    logger.info(f"After SMOTE: {len(X_train_resampled)} samples")

    # 6. Train
    if model_name:
        model = trainer.train_single(model_name, X_train_resampled, y_train_resampled)
        models = {model_name: model}
    else:
        models = trainer.train_all(X_train_resampled, y_train_resampled)
    logger.info(f"Trained models: {list(models.keys())}")

    # 7. Evaluate
    evaluator = ModelEvaluator()
    results = evaluator.compare_models(models, X_test.values, y_test.values)
    best_name = evaluator.select_best_model(results)
    best_model = models[best_name]
    best_metrics = results[best_name]
    logger.info(f"Best model: {best_name} (AUPRC={best_metrics.auprc:.4f})")

    # 8. Log ALL models to MLflow (each as a separate run)
    for name, model_obj in models.items():
        metrics = results[name]
        with mlflow.start_run(run_name=f"train_{name}"):
            # Log parameters
            mlflow.log_param("model_type", name)
            mlflow.log_param("test_size", settings.test_size)
            mlflow.log_param("smote_strategy", settings.smote_sampling_strategy)
            mlflow.log_param("n_train_samples", len(X_train_resampled))
            mlflow.log_param("n_test_samples", len(X_test))
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("random_state", settings.random_state)

            # Log metrics
            mlflow.log_metric("precision", metrics.precision)
            mlflow.log_metric("recall", metrics.recall)
            mlflow.log_metric("f1_score", metrics.f1)
            mlflow.log_metric("auprc", metrics.auprc)
            mlflow.log_metric("auroc", metrics.auroc)
            mlflow.log_metric("is_best", 1.0 if name == best_name else 0.0)

            # Log model artifact
            mlflow.sklearn.log_model(model_obj, artifact_path="model")

            # Tag the best model
            if name == best_name:
                mlflow.set_tag("best_model", "true")
                mlflow.set_tag("model_stage", "production")
            else:
                mlflow.set_tag("best_model", "false")
                mlflow.set_tag("model_stage", "candidate")

    logger.info(f"All {len(models)} models logged to MLflow.")

    # 9. Register ALL models locally (so API can serve any of them)
    registry = ModelRegistry()
    for name, model_obj in models.items():
        model_path = registry.save_model(
            model_obj, name, results[name].to_dict()
        )
        logger.info(f"Registered: {name} -> {model_path}")

    # Set best as latest
    reg_data = registry._load_registry()
    reg_data["latest"] = best_name
    registry._save_registry(reg_data)
    logger.info(f"Best model set as latest: {best_name}")

    # 10. Save reference and feature store
    ingestion.save_reference(X_test)

    feature_engineer = FeatureEngineer()
    importances = feature_engineer.compute_feature_importance_weights(
        best_model, list(X_train.columns)
    )
    feature_store = FeatureStore()
    feature_store.save_feature_schema(
        feature_names=list(X_train.columns),
        feature_importances=importances,
    )
    logger.info("Reference data and feature store saved.")

    return {
        "best_model": best_name,
        "metrics": best_metrics.to_dict(),
        "model_path": str(model_path),
        "all_results": {name: results[name].to_dict() for name in results},
    }


if __name__ == "__main__":
    result = run_training_pipeline()
    print(f"\nTraining complete: {result['best_model']}")
    print(f"  AUPRC: {result['metrics']['auprc']:.4f}")
    print(f"  AUROC: {result['metrics']['auroc']:.4f}")
    print(f"  F1:    {result['metrics']['f1']:.4f}")
    print(f"\nOpen MLflow UI: mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db --port 5000")
