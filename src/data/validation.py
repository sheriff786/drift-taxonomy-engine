"""Data validation checks before model training or inference."""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from src.config.constants import FEATURE_NAME_MAPPING, FEATURE_NAME_REVERSE


@dataclass
class ValidationResult:
    """Result of a data validation check."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class DataValidator:
    """Validates data schema, ranges, and quality before processing."""

    EXPECTED_COLUMNS = (
        ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    )
    # Also accept domain-named columns
    EXPECTED_DOMAIN_COLUMNS = (
        ["transaction_time"] + [FEATURE_NAME_MAPPING[f"V{i}"] for i in range(1, 29)]
        + ["transaction_amount", "Class"]
    )
    EXPECTED_FEATURE_COUNT = 31  # 28 V-features + Time + Amount + Class

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Run all validation checks on input data."""
        errors = []
        warnings = []

        # Check if using domain names or original names
        uses_domain = any(col in FEATURE_NAME_MAPPING.values() for col in df.columns)
        expected = self.EXPECTED_DOMAIN_COLUMNS if uses_domain else self.EXPECTED_COLUMNS

        # Schema check
        missing_cols = set(expected) - set(df.columns)
        if missing_cols:
            # If partial match, may be using the other naming convention
            other_expected = self.EXPECTED_COLUMNS if uses_domain else self.EXPECTED_DOMAIN_COLUMNS
            other_missing = set(other_expected) - set(df.columns)
            if len(other_missing) < len(missing_cols):
                missing_cols = other_missing
                warnings.append("Detected mixed column naming. Consider using domain names consistently.")
            if missing_cols:
                errors.append(f"Missing columns: {missing_cols}")

        extra_cols = set(df.columns) - set(expected)
        if extra_cols:
            warnings.append(f"Unexpected columns (will be ignored): {extra_cols}")

        # Null check
        null_counts = df.isnull().sum()
        null_cols = null_counts[null_counts > 0]
        if len(null_cols) > 0:
            null_pct = (null_cols / len(df) * 100).to_dict()
            errors.append(f"Null values detected: {null_pct}")

        # Type check
        non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            errors.append(f"Non-numeric columns found: {non_numeric}")

        # Range check for target
        if "Class" in df.columns:
            unique_classes = df["Class"].unique()
            if not set(unique_classes).issubset({0, 1}):
                errors.append(
                    f"Target column 'Class' has unexpected values: {unique_classes}"
                )

        # Size check
        if len(df) < 100:
            warnings.append(f"Very small dataset ({len(df)} rows). Results may be unreliable.")

        # Constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            warnings.append(f"Constant columns detected: {constant_cols}")

        # Duplicate check
        n_duplicates = df.duplicated().sum()
        if n_duplicates > 0:
            warnings.append(f"Found {n_duplicates} duplicate rows.")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )
