"""
Unit tests for data validation, anomaly detection, and leakage prevention.
"""
import pytest
import pandas as pd
from utils.validation import generate_data_quality_report, separate_features_and_target, clean_academic_dataset

def test_data_quality_reporting():
    df = pd.DataFrame({
        "student_id": ["S1", "S2", "S2"],
        "attendance_rate": [85.0, -10.0, 120.0],  # Anomalies: <0 and >100
        "internal_assessment_score": [70.0, None, 65.0],
        "target": ["Graduate", "Dropout", "Dropout"]
    })

    report = generate_data_quality_report(df)
    assert report["total_rows"] == 3
    assert report["duplicate_rows"] == 0  # not full duplicates
    assert report["id_duplicates"] == 1   # S2 duplicated
    assert len(report["anomalies"]) > 0   # Detected out-of-range attendance

def test_leakage_prevention():
    df = pd.DataFrame({
        "student_id": ["S1", "S2"],
        "name": ["Alice", "Bob"],
        "gender": ["Female", "Male"],
        "attendance_rate": [80.0, 90.0],
        "target": ["Graduate", "Dropout"]
    })

    # When sensitive demographics excluded by default
    X, y, num_cols, cat_cols = separate_features_and_target(df, target_col="target", exclude_sensitive=True)
    
    # Assert Student ID and Name never leak into X
    assert "student_id" not in X.columns
    assert "name" not in X.columns
    # Assert sensitive demographics excluded
    assert "gender" not in X.columns
    assert "attendance_rate" in X.columns
