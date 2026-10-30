"""
Unit tests for data export serialization (CSV and Excel formats).
"""
import pytest
import io
import pandas as pd
from utils.export import dataframe_to_csv_bytes, dataframe_to_excel_bytes

def test_dataframe_to_csv_export():
    df = pd.DataFrame({
        "student_id": ["S01", "S02"],
        "risk_category": ["High", "Low"],
        "risk_score": [0.78, 0.15]
    })
    csv_bytes = dataframe_to_csv_bytes(df)
    assert isinstance(csv_bytes, bytes)
    assert len(csv_bytes) > 0

    # Read back to verify integrity
    read_df = pd.read_csv(io.BytesIO(csv_bytes))
    assert len(read_df) == 2
    assert list(read_df["student_id"]) == ["S01", "S02"]
    assert read_df["risk_score"].iloc[0] == 0.78

def test_dataframe_to_excel_export():
    df = pd.DataFrame({
        "student_id": ["S01", "S02"],
        "predicted_outcome": ["Dropout", "Graduate"],
        "attendance_rate": [62.0, 95.0]
    })
    excel_bytes = dataframe_to_excel_bytes(df)
    assert isinstance(excel_bytes, bytes)
    assert len(excel_bytes) > 0

    # Read back to verify integrity
    read_df = pd.read_excel(io.BytesIO(excel_bytes))
    assert len(read_df) == 2
    assert read_df["predicted_outcome"].iloc[1] == "Graduate"
    assert read_df["attendance_rate"].iloc[0] == 62.0

