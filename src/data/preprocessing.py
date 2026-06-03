"""Data preprocessing and feature engineering pipeline."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple

from src.config.constants import FEATURE_NAME_MAPPING, FEATURE_NAME_REVERSE


class DataPreprocessor:
    """Handles scaling, encoding, and feature engineering for fraud detection."""

    def __init__(self):
        self.amount_scaler = StandardScaler()
        self.time_scaler = StandardScaler()
        self._is_fitted = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit scalers and transform the dataset."""
        df = df.copy()
        df["Amount_scaled"] = self.amount_scaler.fit_transform(
            df[["Amount"]]
        )
        df["Time_scaled"] = self.time_scaler.fit_transform(
            df[["Time"]]
        )
        df = df.drop(columns=["Amount", "Time"])
        self._is_fitted = True
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted scalers."""
        if not self._is_fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit_transform first.")
        df = df.copy()
        df["Amount_scaled"] = self.amount_scaler.transform(df[["Amount"]])
        df["Time_scaled"] = self.time_scaler.transform(df[["Time"]])
        df = df.drop(columns=["Amount", "Time"])
        return df

    def get_feature_columns(self, df: pd.DataFrame) -> list:
        """Get feature column names (excluding target)."""
        return [col for col in df.columns if col != "Class"]

    def separate_features_target(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Split dataframe into features (X) and target (y)."""
        target_col = "Class"
        X = df.drop(columns=[target_col])
        y = df[target_col]
        return X, y

    @staticmethod
    def rename_to_domain(df: pd.DataFrame) -> pd.DataFrame:
        """Rename PCA columns (V1-V28) to interpretive domain names for display."""
        rename_map = {k: v for k, v in FEATURE_NAME_MAPPING.items() if k in df.columns}
        return df.rename(columns=rename_map)

    @staticmethod
    def rename_to_original(df: pd.DataFrame) -> pd.DataFrame:
        """Rename domain names back to PCA columns (V1-V28) for model inference."""
        rename_map = {k: v for k, v in FEATURE_NAME_REVERSE.items() if k in df.columns}
        return df.rename(columns=rename_map)
