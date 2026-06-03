"""End-to-end training pipeline."""

import logging
from src.data.ingestion import DataIngestion
from src.data.preprocessing import DataPreprocessor
from src.data.validation import DataValidator
from src.data.splitter import DataSplitter
from src.models.trainer import ModelTrainer
from src.models.evaluator import ModelEvaluator
from src.models.registry import ModelRegistry
from src.features.feature_engineering import FeatureEngineer
from src.features.feature_store import FeatureStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_training_pipeline(model_name: str = None) -> dict:
    """
    Execute the full training pipeline:
    1. Ingest data
    2. Validate
    3. Preprocess
    4. Split
    5. Apply SMOTE
    6. Train models
    7. Evaluate and select best
    8. Register model
    9. Save reference data and feature store
    """
    logger.info("Starting training pipeline...")

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

    # 8. Register
    registry = ModelRegistry()
    model_path = registry.save_model(
        best_model, best_name, best_metrics.to_dict()
    )
    logger.info(f"Model registered at: {model_path}")

    # 9. Save reference and feature store
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
    }


if __name__ == "__main__":
    result = run_training_pipeline()
    print(f"\nTraining complete: {result}")
