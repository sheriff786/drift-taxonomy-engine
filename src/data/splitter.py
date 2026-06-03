"""Data splitting utilities for train/test and reference/current."""

import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple
from src.config.settings import get_settings


class DataSplitter:
    """Handles dataset splitting for training and drift detection."""

    def __init__(self):
        self.settings = get_settings()

    def train_test_split(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Stratified train/test split preserving class ratios."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.settings.test_size,
            random_state=self.settings.random_state,
            stratify=y,
        )
        return X_train, X_test, y_train, y_test

    def reference_current_split(
        self, df: pd.DataFrame, reference_fraction: float = 0.5
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into reference (baseline) and current windows."""
        split_idx = int(len(df) * reference_fraction)
        reference = df.iloc[:split_idx].reset_index(drop=True)
        current = df.iloc[split_idx:].reset_index(drop=True)
        return reference, current

    def temporal_split(
        self, df: pd.DataFrame, time_col: str = "Time", n_windows: int = 10
    ) -> list:
        """Split data into temporal windows for streaming simulation."""
        df_sorted = df.sort_values(time_col).reset_index(drop=True)
        window_size = len(df_sorted) // n_windows
        windows = []
        for i in range(n_windows):
            start = i * window_size
            end = start + window_size if i < n_windows - 1 else len(df_sorted)
            windows.append(df_sorted.iloc[start:end].reset_index(drop=True))
        return windows
