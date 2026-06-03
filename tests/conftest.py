"""Test configuration and shared fixtures."""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_reference_data():
    """Generate synthetic reference data for testing."""
    np.random.seed(42)
    n = 1000
    data = {f"V{i}": np.random.randn(n) for i in range(1, 29)}
    data["Amount_scaled"] = np.random.randn(n)
    data["Time_scaled"] = np.random.randn(n)
    return pd.DataFrame(data)


@pytest.fixture
def sample_current_data(sample_reference_data):
    """Generate current data (same distribution = no drift)."""
    np.random.seed(123)
    n = len(sample_reference_data)
    data = {f"V{i}": np.random.randn(n) for i in range(1, 29)}
    data["Amount_scaled"] = np.random.randn(n)
    data["Time_scaled"] = np.random.randn(n)
    return pd.DataFrame(data)


@pytest.fixture
def drifted_current_data(sample_reference_data):
    """Generate current data with artificial drift."""
    np.random.seed(99)
    n = len(sample_reference_data)
    data = {f"V{i}": np.random.randn(n) for i in range(1, 29)}
    # Add strong drift to V14 and V10
    data["V14"] = np.random.randn(n) + 5.0
    data["V10"] = np.random.randn(n) * 3.0
    data["Amount_scaled"] = np.random.randn(n) * 10.0
    data["Time_scaled"] = np.random.randn(n)
    return pd.DataFrame(data)


@pytest.fixture
def sample_raw_data():
    """Generate raw data with Time, Amount, and Class columns."""
    np.random.seed(42)
    n = 500
    data = {f"V{i}": np.random.randn(n) for i in range(1, 29)}
    data["Time"] = np.sort(np.random.uniform(0, 172000, n))
    data["Amount"] = np.abs(np.random.exponential(50, n))
    data["Class"] = np.zeros(n, dtype=int)
    data["Class"][:5] = 1  # 1% fraud
    return pd.DataFrame(data)


@pytest.fixture
def feature_importances():
    """Sample feature importances for testing."""
    importances = {f"V{i}": 1.0 / 30 for i in range(1, 29)}
    importances["Amount_scaled"] = 1.0 / 30
    importances["Time_scaled"] = 1.0 / 30
    # Make V14 most important
    importances["V14"] = 0.15
    importances["V10"] = 0.10
    return importances
