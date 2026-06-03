"""Feature store for versioned feature metadata and schemas."""

import json
from pathlib import Path
from typing import Dict, List, Optional
from src.config.settings import get_settings


class FeatureStore:
    """Manages feature metadata, versions, and schemas."""

    def __init__(self):
        self.settings = get_settings()
        self._store_path = self.settings.artifacts_dir / "feature_store.json"

    def save_feature_schema(
        self,
        feature_names: List[str],
        feature_importances: Dict[str, float],
        version: str = "1.0.0",
    ) -> None:
        """Persist feature schema and importances."""
        schema = {
            "version": version,
            "features": feature_names,
            "importances": feature_importances,
            "n_features": len(feature_names),
        }
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._store_path, "w") as f:
            json.dump(schema, f, indent=2)

    def load_feature_schema(self) -> dict:
        """Load current feature schema."""
        if not self._store_path.exists():
            raise FileNotFoundError(
                f"Feature store not found at {self._store_path}. "
                "Run training pipeline first."
            )
        with open(self._store_path, "r") as f:
            return json.load(f)

    def get_feature_importances(self) -> Dict[str, float]:
        """Get feature importance mapping."""
        schema = self.load_feature_schema()
        return schema.get("importances", {})
