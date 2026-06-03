"""Covariate drift detector using KS test + Cohen's d."""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import List, Dict
from src.config.settings import get_settings


@dataclass
class CovariateSignal:
    """Signal from a single feature's covariate drift check."""
    feature: str
    ks_statistic: float
    p_value: float
    cohens_d: float
    is_drifted: bool


class CovariateDriftDetector:
    """Detects feature distribution shifts using KS test with practical significance filter."""

    def __init__(self):
        self.settings = get_settings()

    def detect(
        self, reference: pd.DataFrame, current: pd.DataFrame
    ) -> List[CovariateSignal]:
        """Run covariate drift detection across all shared numeric features."""
        signals = []
        shared_cols = [
            c for c in reference.columns
            if c in current.columns and reference[c].dtype in [np.float64, np.int64, np.float32]
        ]

        for col in shared_cols:
            ref_values = reference[col].dropna().values
            cur_values = current[col].dropna().values

            if len(ref_values) < 10 or len(cur_values) < 10:
                continue

            # KS test for statistical significance
            ks_stat, p_value = stats.ks_2samp(ref_values, cur_values)

            # Cohen's d for practical significance
            pooled_std = np.sqrt(
                (np.std(ref_values) ** 2 + np.std(cur_values) ** 2) / 2
            )
            cohens_d = (
                abs(np.mean(cur_values) - np.mean(ref_values)) / pooled_std
                if pooled_std > 0 else 0.0
            )

            # Both statistical AND practical significance required
            is_drifted = (
                p_value < self.settings.ks_significance_level
                and cohens_d >= self.settings.cohens_d_threshold
            )

            signals.append(CovariateSignal(
                feature=col,
                ks_statistic=ks_stat,
                p_value=p_value,
                cohens_d=cohens_d,
                is_drifted=is_drifted,
            ))

        return signals

    def get_drifted_features(self, signals: List[CovariateSignal]) -> List[str]:
        """Extract list of features that have drifted."""
        return [s.feature for s in signals if s.is_drifted]

    def compute_drift_score(
        self, signals: List[CovariateSignal], feature_importances: Dict[str, float]
    ) -> float:
        """Compute importance-weighted covariate drift score (0-1)."""
        if not signals:
            return 0.0

        weighted_sum = 0.0
        total_weight = 0.0

        for signal in signals:
            weight = feature_importances.get(signal.feature, 0.01)
            if signal.is_drifted:
                weighted_sum += weight * signal.cohens_d
            total_weight += weight

        return min(weighted_sum / total_weight, 1.0) if total_weight > 0 else 0.0
