"""Application settings loaded from environment variables."""

from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field

# Resolve project root at module level
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Global application settings with env-var overrides."""

    # Project paths
    project_root: Path = _PROJECT_ROOT
    data_dir: Path = _PROJECT_ROOT / "data"
    artifacts_dir: Path = _PROJECT_ROOT / "artifacts"
    models_dir: Path = _PROJECT_ROOT / "artifacts" / "models"
    reports_dir: Path = _PROJECT_ROOT / "artifacts" / "reports"
    references_dir: Path = _PROJECT_ROOT / "artifacts" / "references"

    # Model training
    test_size: float = 0.2
    random_state: int = 42
    smote_sampling_strategy: float = 0.5

    # Drift detection thresholds
    ks_significance_level: float = 0.001
    cohens_d_threshold: float = 0.5
    missing_rate_threshold: float = 0.05
    performance_decay_threshold: float = 0.1

    # Severity scoring
    severity_low_threshold: float = 0.2
    severity_medium_threshold: float = 0.4
    severity_high_threshold: float = 0.6
    severity_critical_threshold: float = 0.8

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4

    # MLflow
    mlflow_tracking_uri: str = "sqlite:///artifacts/mlflow.db"
    mlflow_experiment_name: str = "drift-taxonomy-engine"

    # Monitoring
    prometheus_port: int = 9090

    model_config = {"env_prefix": "DTE_", "env_file": ".env", "extra": "ignore"}


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings singleton."""
    return Settings()
