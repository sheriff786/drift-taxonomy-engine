"""Target drift detector for label distribution shifts."""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from src.config.settings import get_settings


@dataclass
class TargetSignal:
    """Signal from target distribution drift check."""
    reference_rate: float
    current_rate: float
    rate_change: float
    chi2_statistic: float
    p_value: float
    is_drifted: bool


class TargetDriftDetector:
    """Detects shifts in the target (Class) distribution."""

    def __init__(self):
        self.settings = get_settings()

    def detect(
        self, reference_y: pd.Series, current_y: pd.Series
    ) -> TargetSignal:
        """Detect target distribution shift between reference and current."""
        ref_rate = reference_y.mean()
        cur_rate = current_y.mean()
        rate_change = abs(cur_rate - ref_rate) / ref_rate if ref_rate > 0 else 0.0

        # Chi-squared test on class distributions
        ref_counts = np.array([
            (reference_y == 0).sum(), (reference_y == 1).sum()
        ])
        cur_counts = np.array([
            (current_y == 0).sum(), (current_y == 1).sum()
        ])

        # Expected counts under reference distribution
        ref_probs = ref_counts / ref_counts.sum()
        expected = ref_probs * cur_counts.sum()

        chi2_stat, p_value = stats.chisquare(cur_counts, f_exp=expected)

        is_drifted = (
            p_value < self.settings.ks_significance_level
            and rate_change > 0.2  # >20% relative change in fraud rate
        )

        return TargetSignal(
            reference_rate=ref_rate,
            current_rate=cur_rate,
            rate_change=rate_change,
            chi2_statistic=chi2_stat,
            p_value=p_value,
            is_drifted=is_drifted,
        )

    def compute_drift_score(self, signal: TargetSignal) -> float:
        """Compute target drift score (0-1)."""
        if not signal.is_drifted:
            return 0.0
        return min(signal.rate_change, 1.0)
