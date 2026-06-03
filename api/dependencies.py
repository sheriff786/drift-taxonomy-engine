"""Dependency injection for API - model loader, engine instance."""

from typing import Dict, Any
from src.models.predictor import ModelPredictor
from src.drift.engine import DriftTaxonomyEngine
from src.features.feature_store import FeatureStore
from src.data.ingestion import DataIngestion

# Global app state (populated on startup)
app_state: Dict[str, Any] = {}


def load_resources():
    """Load model and drift engine on application startup."""
    try:
        feature_store = FeatureStore()
        importances = feature_store.get_feature_importances()
    except FileNotFoundError:
        # Default importances if no training has run yet
        importances = {f"V{i}": 1.0 / 30 for i in range(1, 29)}
        importances["Amount_scaled"] = 1.0 / 30
        importances["Time_scaled"] = 1.0 / 30

    app_state["predictor"] = ModelPredictor()
    app_state["drift_engine"] = DriftTaxonomyEngine(
        feature_importances=importances
    )
    app_state["ingestion"] = DataIngestion()


def get_predictor() -> ModelPredictor:
    """Get model predictor instance."""
    return app_state["predictor"]


def get_drift_engine() -> DriftTaxonomyEngine:
    """Get drift taxonomy engine instance."""
    return app_state["drift_engine"]


def get_ingestion() -> DataIngestion:
    """Get data ingestion instance."""
    return app_state["ingestion"]
