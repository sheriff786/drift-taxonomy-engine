"""Tests for data preprocessing module."""

import pytest
import pandas as pd
import numpy as np
from src.data.preprocessing import DataPreprocessor
from src.data.validation import DataValidator


class TestDataPreprocessor:
    def test_fit_transform_scales_amount_and_time(self, sample_raw_data):
        preprocessor = DataPreprocessor()
        result = preprocessor.fit_transform(sample_raw_data)

        assert "Amount_scaled" in result.columns
        assert "Time_scaled" in result.columns
        assert "Amount" not in result.columns
        assert "Time" not in result.columns

    def test_transform_requires_fit(self, sample_raw_data):
        preprocessor = DataPreprocessor()
        with pytest.raises(RuntimeError):
            preprocessor.transform(sample_raw_data)

    def test_separate_features_target(self, sample_raw_data):
        preprocessor = DataPreprocessor()
        df = preprocessor.fit_transform(sample_raw_data)
        X, y = preprocessor.separate_features_target(df)

        assert "Class" not in X.columns
        assert len(y) == len(X)
        assert set(y.unique()).issubset({0, 1})


class TestDataValidator:
    def test_valid_data_passes(self, sample_raw_data):
        validator = DataValidator()
        result = validator.validate(sample_raw_data)
        assert result.is_valid

    def test_missing_columns_fail(self, sample_raw_data):
        validator = DataValidator()
        df = sample_raw_data.drop(columns=["V1", "V2"])
        result = validator.validate(df)
        assert not result.is_valid
        assert any("Missing columns" in e for e in result.errors)

    def test_null_values_detected(self, sample_raw_data):
        validator = DataValidator()
        df = sample_raw_data.copy()
        df.loc[0:10, "V1"] = np.nan
        result = validator.validate(df)
        assert not result.is_valid
