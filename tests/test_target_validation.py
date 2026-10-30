"""
Unit tests for target-label validation, class count checks, and missing value handling.
"""
import pytest
import pandas as pd
import numpy as np
from utils.validation import separate_features_and_target, clean_academic_dataset
from src.train_model import train_academic_model

def test_missing_target_column():
    df = pd.DataFrame({
        "student_id": ["S1", "S2"],
        "attendance_rate": [80.0, 90.0]
    })
    with pytest.raises(ValueError, match="Target column 'target' not found"):
        separate_features_and_target(df, target_col="target")

def test_target_with_single_class():
    df = pd.DataFrame({
        "attendance_rate": [80.0, 85.0, 90.0, 78.0],
        "internal_assessment_score": [60.0, 70.0, 65.0, 72.0],
        "target": ["Graduate", "Graduate", "Graduate", "Graduate"]  # Only 1 class
    })
    X = df[["attendance_rate", "internal_assessment_score"]]
    y = df["target"]

    with pytest.raises(ValueError, match="Target column must contain at least 2 distinct classes"):
        train_academic_model(
            X=X,
            y=y,
            numeric_features=["attendance_rate", "internal_assessment_score"],
            categorical_features=[],
            model_type="Logistic Regression"
        )

def test_missing_values_cleaning_strategies():
    df = pd.DataFrame({
        "attendance_rate": [80.0, None, 90.0, 85.0],
        "department": ["Computer Science", None, "Information Technology", "Computer Science"],
        "failed_subjects_count": [0, 1, 0, -5]  # -5 is out-of-range (< 0)
    })

    # Test impute strategy
    clean_imputed, audit_imputed = clean_academic_dataset(df, handle_missing="impute", remove_duplicates=True)
    assert clean_imputed["attendance_rate"].isnull().sum() == 0
    assert clean_imputed["department"].isnull().sum() == 0
    assert (clean_imputed["failed_subjects_count"] >= 0).all()  # clamped -5 to 0
    assert len(clean_imputed) == 4

    # Test drop strategy
    clean_dropped, audit_dropped = clean_academic_dataset(df, handle_missing="drop", remove_duplicates=True)
    assert clean_dropped["attendance_rate"].isnull().sum() == 0
    assert len(clean_dropped) < 4

