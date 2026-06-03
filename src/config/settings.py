"""Application settings loaded from environment variables."""

from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Global application settings with env-var overrides."""

    # Project paths
    project_root: Path = Path(__file__).resolve().parent.parent.parent
    data_dir: Path = Field(default=None)
    artifacts_dir: Path = Field(default=None)
    models_dir: Path = Field(default=None)
    reports_dir: Path = Field(default=None)
    references_dir: Path = Field(default=None)

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

    def model_post_init(self, __context) -> None:
        """Set derived paths after init."""
        if self.data_dir is None:
            self.data_dir = self.project_root / "data"
        if self.artifacts_dir is None:
            self.artifacts_dir = self.project_root / "artifacts"
        if self.models_dir is None:
            self.models_dir = self.artifacts_dir / "models"
        if self.reports_dir is None:
            self.reports_dir = self.artifacts_dir / "reports"
        if self.references_dir is None:
            self.references_dir = self.artifacts_dir / "references"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings singleton."""
    return Settings()
