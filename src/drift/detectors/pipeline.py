"""Pipeline drift detector for data quality anomalies."""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict
from src.config.settings import get_settings


@dataclass
class PipelineSignal:
    """Signal from pipeline drift detection."""
    issue_type: str  # missing_values, constant_column, range_violation, sign_flip, missing_column
    feature: str
    severity: str  # low, medium, high
    detail: str


class PipelineDriftDetector:
    """Detects pipeline/data quality issues that indicate upstream breakages."""

    def __init__(self):
        self.settings = get_settings()

    def detect(
        self, reference: pd.DataFrame, current: pd.DataFrame
    ) -> List[PipelineSignal]:
        """Run all pipeline quality checks."""
        signals = []
        signals.extend(self._check_missing_columns(reference, current))
        signals.extend(self._check_missing_values(current))
        signals.extend(self._check_constant_columns(reference, current))
        signals.extend(self._check_range_violations(reference, current))
        signals.extend(self._check_sign_flips(reference, current))
        return signals

    def _check_missing_columns(
        self, reference: pd.DataFrame, current: pd.DataFrame
    ) -> List[PipelineSignal]:
        """Detect columns present in reference but missing in current."""
        missing = set(reference.columns) - set(current.columns)
        return [
            PipelineSignal(
                issue_type="missing_column",
                feature=col,
                severity="high",
                detail=f"Column '{col}' present in reference but missing in current data.",
            )
            for col in missing
        ]

    def _check_missing_values(self, current: pd.DataFrame) -> List[PipelineSignal]:
        """Detect columns with excessive null values."""
        signals = []
        for col in current.columns:
            null_rate = current[col].isnull().mean()
            if null_rate > self.settings.missing_rate_threshold:
                severity = "high" if null_rate > 0.3 else "medium"
                signals.append(PipelineSignal(
                    issue_type="missing_values",
                    feature=col,
                    severity=severity,
                    detail=f"Column '{col}' has {null_rate:.1%} missing values.",
                ))
        return signals

    def _check_constant_columns(
        self, reference: pd.DataFrame, current: pd.DataFrame
    ) -> List[PipelineSignal]:
        """Detect columns that became constant (zero variance)."""
        signals = []
        for col in current.select_dtypes(include=[np.number]).columns:
            if col not in reference.columns:
                continue
            ref_std = reference[col].std()
            cur_std = current[col].std()
            if ref_std > 0 and cur_std == 0:
                signals.append(PipelineSignal(
                    issue_type="constant_column",
                    feature=col,
                    severity="high",
                    detail=f"Column '{col}' has become constant (was std={ref_std:.4f}).",
                ))
        return signals

    def _check_range_violations(
        self, reference: pd.DataFrame, current: pd.DataFrame
    ) -> List[PipelineSignal]:
        """Detect values far outside reference range."""
        signals = []
        for col in current.select_dtypes(include=[np.number]).columns:
            if col not in reference.columns:
                continue
            ref_min, ref_max = reference[col].min(), reference[col].max()
            ref_range = ref_max - ref_min
            if ref_range == 0:
                continue

            cur_min, cur_max = current[col].min(), current[col].max()
            # Check if current exceeds 3x the reference range
            if cur_max > ref_max + 3 * ref_range or cur_min < ref_min - 3 * ref_range:
                signals.append(PipelineSignal(
                    issue_type="range_violation",
                    feature=col,
                    severity="medium",
                    detail=(
                        f"Column '{col}' range [{cur_min:.2f}, {cur_max:.2f}] "
                        f"far exceeds reference [{ref_min:.2f}, {ref_max:.2f}]."
                    ),
                ))
        return signals

    def _check_sign_flips(
        self, reference: pd.DataFrame, current: pd.DataFrame
    ) -> List[PipelineSignal]:
        """Detect features where the sign distribution has flipped."""
        signals = []
        for col in current.select_dtypes(include=[np.number]).columns:
            if col not in reference.columns:
                continue
            ref_pos_rate = (reference[col] > 0).mean()
            cur_pos_rate = (current[col] > 0).mean()

            # Major sign distribution change (>50% flip)
            if abs(ref_pos_rate - cur_pos_rate) > 0.5:
                signals.append(PipelineSignal(
                    issue_type="sign_flip",
                    feature=col,
                    severity="medium",
                    detail=(
                        f"Column '{col}' positive rate changed from "
                        f"{ref_pos_rate:.1%} to {cur_pos_rate:.1%}."
                    ),
                ))
        return signals

    def compute_drift_score(self, signals: List[PipelineSignal]) -> float:
        """Compute pipeline drift score based on issue count and severity."""
        if not signals:
            return 0.0

        severity_weights = {"low": 0.2, "medium": 0.5, "high": 1.0}
        total_score = sum(severity_weights.get(s.severity, 0.5) for s in signals)
        # Normalize: 5+ high-severity issues = score 1.0
        return min(total_score / 5.0, 1.0)
