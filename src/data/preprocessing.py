"""Data preprocessing and feature engineering pipeline."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple


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
