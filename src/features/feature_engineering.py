"""Feature engineering transformations."""

import pandas as pd
import numpy as np
from typing import List, Optional


class FeatureEngineer:
    """Applies domain-specific feature transformations."""

    def __init__(self, feature_columns: Optional[List[str]] = None):
        self.feature_columns = feature_columns

    def compute_feature_importance_weights(
        self, model, feature_names: List[str]
    ) -> dict:
        """Extract feature importance from trained model."""
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            # Equal weights fallback
            importances = np.ones(len(feature_names)) / len(feature_names)

        importance_dict = dict(zip(feature_names, importances))
        # Normalize to sum to 1
        total = sum(importance_dict.values())
        if total > 0:
            importance_dict = {k: v / total for k, v in importance_dict.items()}
        return importance_dict

    def select_top_features(
        self, importance_dict: dict, top_n: int = 15
    ) -> List[str]:
        """Select top N features by importance."""
        sorted_features = sorted(
            importance_dict.items(), key=lambda x: x[1], reverse=True
        )
        return [f[0] for f in sorted_features[:top_n]]
