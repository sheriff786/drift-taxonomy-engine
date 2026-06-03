"""Tests for drift taxonomy engine."""

import pytest
from src.drift.engine import DriftTaxonomyEngine
from src.drift.detectors.covariate import CovariateDriftDetector
from src.drift.detectors.pipeline import PipelineDriftDetector


class TestCovariateDriftDetector:
    def test_no_drift_same_distribution(self, sample_reference_data, sample_current_data):
        detector = CovariateDriftDetector()
        signals = detector.detect(sample_reference_data, sample_current_data)
        drifted = detector.get_drifted_features(signals)
        # Same distribution should have minimal drift
        assert len(drifted) < len(sample_reference_data.columns) / 2

    def test_detects_strong_drift(self, sample_reference_data, drifted_current_data):
        detector = CovariateDriftDetector()
        signals = detector.detect(sample_reference_data, drifted_current_data)
        drifted = detector.get_drifted_features(signals)
        # Should detect drift in V14 and V10
        assert "V14" in drifted or "V10" in drifted


class TestPipelineDriftDetector:
    def test_no_issues_clean_data(self, sample_reference_data, sample_current_data):
        detector = PipelineDriftDetector()
        signals = detector.detect(sample_reference_data, sample_current_data)
        # Clean data should have no/minimal pipeline issues
        high_severity = [s for s in signals if s.severity == "high"]
        assert len(high_severity) == 0


class TestDriftTaxonomyEngine:
    def test_diagnose_quick_no_drift(
        self, sample_reference_data, sample_current_data, feature_importances
    ):
        engine = DriftTaxonomyEngine(feature_importances=feature_importances)
        diagnosis = engine.diagnose_quick(sample_reference_data, sample_current_data)

        assert diagnosis.drift_type in ["none", "covariate", "pipeline"]
        assert diagnosis.severity in ["none", "low", "medium"]
        assert diagnosis.action is not None

    def test_diagnose_quick_with_drift(
        self, sample_reference_data, drifted_current_data, feature_importances
    ):
        engine = DriftTaxonomyEngine(feature_importances=feature_importances)
        diagnosis = engine.diagnose_quick(sample_reference_data, drifted_current_data)

        # Should detect drift
        assert diagnosis.covariate_score > 0
        assert diagnosis.diagnosed_at is not None
        assert diagnosis.playbook is not None
