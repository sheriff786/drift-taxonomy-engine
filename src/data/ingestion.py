"""Data ingestion from various sources."""

import pandas as pd
from pathlib import Path
from src.config.settings import get_settings


class DataIngestion:
    """Handles loading data from CSV, parquet, or database sources."""

    def __init__(self):
        self.settings = get_settings()

    def load_csv(self, filename: str = "creditcard.csv") -> pd.DataFrame:
        """Load data from CSV file in data directory."""
        filepath = self.settings.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        return pd.read_csv(filepath)

    def load_parquet(self, filename: str) -> pd.DataFrame:
        """Load data from parquet file."""
        filepath = self.settings.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        return pd.read_parquet(filepath)

    def load_reference(self, name: str = "reference_data.parquet") -> pd.DataFrame:
        """Load reference (baseline) dataset for drift comparison."""
        filepath = self.settings.references_dir / name
        if not filepath.exists():
            raise FileNotFoundError(
                f"Reference data not found: {filepath}. "
                "Run the training pipeline first to generate reference data."
            )
        return pd.read_parquet(filepath)

    def save_reference(self, df: pd.DataFrame, name: str = "reference_data.parquet") -> Path:
        """Save reference dataset for future drift comparisons."""
        self.settings.references_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.settings.references_dir / name
        df.to_parquet(filepath, index=False)
        return filepath
