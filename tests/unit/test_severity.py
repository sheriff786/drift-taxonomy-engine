"""Tests for severity scoring and action mapping."""

import pytest
from src.drift.severity import SeverityScorer
from src.drift.actions import ActionRecommender
from src.config.constants import DriftType, Severity, ActionType


class TestSeverityScorer:
    def test_no_drift_returns_none(self):
        scorer = SeverityScorer()
        result = scorer.score(DriftType.NONE, 0.0, 0.0, 0.0, 0.0)
        assert result == Severity.NONE

    def test_high_concept_score_returns_high_severity(self):
        scorer = SeverityScorer()
        result = scorer.score(DriftType.CONCEPT, 0.0, 0.7, 0.0, 0.0)
        assert result in [Severity.HIGH, Severity.CRITICAL]

    def test_concept_has_elevated_floor(self):
        scorer = SeverityScorer()
        # Even low concept score should be at least medium
        result = scorer.score(DriftType.CONCEPT, 0.0, 0.3, 0.0, 0.0)
        assert result in [Severity.MEDIUM, Severity.HIGH]


class TestActionRecommender:
    def test_no_drift_returns_monitor(self):
        recommender = ActionRecommender()
        action = recommender.recommend(DriftType.NONE, Severity.NONE)
        assert action == ActionType.MONITOR

    def test_pipeline_critical_returns_block(self):
        recommender = ActionRecommender()
        action = recommender.recommend(DriftType.PIPELINE, Severity.CRITICAL)
        assert action == ActionType.BLOCK

    def test_concept_high_returns_full_retrain(self):
        recommender = ActionRecommender()
        action = recommender.recommend(DriftType.CONCEPT, Severity.HIGH)
        assert action == ActionType.FULL_RETRAIN

    def test_urgency_hours_ordering(self):
        recommender = ActionRecommender()
        block_hours = recommender.get_urgency_hours(ActionType.BLOCK)
        monitor_hours = recommender.get_urgency_hours(ActionType.MONITOR)
        assert block_hours < monitor_hours
