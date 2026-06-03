"""Streaming drift detection with windowed checks."""

import pandas as pd
import logging
from typing import List, Dict
from src.drift.engine import DriftTaxonomyEngine, DriftDiagnosis
from src.data.splitter import DataSplitter

logger = logging.getLogger(__name__)


class StreamingDriftMonitor:
    """Monitors drift in streaming windows, tracking escalation."""

    def __init__(self, engine: DriftTaxonomyEngine, n_windows: int = 10):
        self.engine = engine
        self.n_windows = n_windows
        self.history: List[DriftDiagnosis] = []

    def run_streaming(
        self, reference: pd.DataFrame, stream_data: pd.DataFrame
    ) -> List[Dict]:
        """Run drift checks across temporal windows."""
        splitter = DataSplitter()
        windows = splitter.temporal_split(stream_data, n_windows=self.n_windows)

        results = []
        for i, window in enumerate(windows):
            diagnosis = self.engine.diagnose_quick(reference=reference, current=window)
            self.history.append(diagnosis)

            result = {
                "window": i + 1,
                "n_samples": len(window),
                "drift_type": diagnosis.drift_type,
                "severity": diagnosis.severity,
                "action": diagnosis.action,
                "covariate_score": diagnosis.covariate_score,
            }
            results.append(result)

            logger.info(
                f"Window {i+1}/{self.n_windows}: "
                f"{diagnosis.drift_type}/{diagnosis.severity} -> {diagnosis.action}"
            )

        return results

    def get_escalation_trend(self) -> List[str]:
        """Get severity trend across windows."""
        return [d.severity for d in self.history]

    def is_escalating(self) -> bool:
        """Check if drift is getting worse over recent windows."""
        if len(self.history) < 3:
            return False

        severity_order = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
        recent = [severity_order.get(d.severity, 0) for d in self.history[-3:]]
        return recent[-1] > recent[0]
