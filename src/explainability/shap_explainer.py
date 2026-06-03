"""SHAP explainer for per-prediction feature attribution."""

import numpy as np
import pandas as pd
import shap
from typing import Dict, List, Optional
from pathlib import Path

from src.config.constants import FEATURE_NAME_MAPPING
from src.models.registry import ModelRegistry


class SHAPExplainer:
    """Generate SHAP-based explanations for model predictions."""

    def __init__(self, model_name: Optional[str] = None):
        self._registry = ModelRegistry()
        self._model_name = model_name
        self._model = None
        self._explainer = None

    def _load_model(self):
        """Load model from registry."""
        if self._model is None:
            name = self._model_name or self._registry.get_latest_model_name()
            self._model = self._registry.load_model(name)
            self._model_name = name

    def _get_explainer(self, X_background: Optional[pd.DataFrame] = None):
        """Create appropriate SHAP explainer based on model type."""
        self._load_model()

        if self._explainer is None:
            model_type = type(self._model).__name__

            if model_type in ("RandomForestClassifier", "GradientBoostingClassifier"):
                self._explainer = shap.TreeExplainer(self._model)
            elif model_type in ("XGBClassifier",):
                self._explainer = shap.TreeExplainer(self._model)
            elif model_type in ("LGBMClassifier",):
                self._explainer = shap.TreeExplainer(self._model)
            elif X_background is not None:
                # Kernel SHAP for any model (slower)
                self._explainer = shap.KernelExplainer(
                    self._model.predict_proba, X_background.iloc[:100]
                )
            else:
                # Fallback: use model's predict_proba directly
                self._explainer = shap.Explainer(self._model)

        return self._explainer

    def explain(
        self,
        X: pd.DataFrame,
        top_n: int = 5,
        use_domain_names: bool = True,
    ) -> List[Dict]:
        """
        Generate SHAP explanations for each sample.

        Args:
            X: Feature DataFrame (model input format: V1-V28, Amount_scaled, Time_scaled)
            top_n: Number of top contributing features to return
            use_domain_names: Convert feature names to domain names

        Returns:
            List of explanation dicts per sample
        """
        self._load_model()
        explainer = self._get_explainer()

        # Get SHAP values
        shap_values = explainer.shap_values(X)

        # Handle multi-class output: take fraud class (index 1)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Class 1 = fraud

        # Get predictions
        predictions = self._model.predict(X)
        probabilities = self._model.predict_proba(X)[:, 1]

        explanations = []
        feature_names = list(X.columns)

        for i in range(len(X)):
            sample_shap = shap_values[i]

            # Pair features with SHAP values
            feature_contributions = []
            for j, feat in enumerate(feature_names):
                display_name = FEATURE_NAME_MAPPING.get(feat, feat) if use_domain_names else feat
                shap_val = float(sample_shap[j])
                feature_contributions.append({
                    "feature": display_name,
                    "original_name": feat,
                    "shap_value": round(shap_val, 6),
                    "abs_shap": abs(shap_val),
                    "direction": "fraud" if shap_val > 0 else "legitimate",
                    "feature_value": round(float(X.iloc[i, j]), 4),
                })

            # Sort by absolute SHAP value
            feature_contributions.sort(key=lambda x: x["abs_shap"], reverse=True)
            top_contributors = feature_contributions[:top_n]

            # Clean up output
            for contrib in top_contributors:
                del contrib["abs_shap"]

            explanations.append({
                "sample_index": i,
                "prediction": int(predictions[i]),
                "probability": round(float(probabilities[i]), 6),
                "model_used": self._model_name,
                "top_contributors": top_contributors,
                "all_shap_values": {
                    FEATURE_NAME_MAPPING.get(feat, feat) if use_domain_names else feat:
                    round(float(sample_shap[j]), 6)
                    for j, feat in enumerate(feature_names)
                },
            })

        return explanations

    def explain_single(self, X: pd.DataFrame, top_n: int = 10) -> Dict:
        """Explain a single prediction (convenience method)."""
        results = self.explain(X.iloc[:1], top_n=top_n)
        return results[0] if results else {}

    def get_global_importance(self, X: pd.DataFrame, max_samples: int = 500) -> Dict[str, float]:
        """Compute mean absolute SHAP values across samples (global importance)."""
        self._load_model()
        explainer = self._get_explainer()

        # Limit samples for speed
        X_sample = X.iloc[:max_samples] if len(X) > max_samples else X
        shap_values = explainer.shap_values(X_sample)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_names = list(X.columns)

        importance = {}
        for j, feat in enumerate(feature_names):
            domain_name = FEATURE_NAME_MAPPING.get(feat, feat)
            importance[domain_name] = round(float(mean_abs_shap[j]), 6)

        return dict(sorted(importance.items(), key=lambda x: -x[1]))
