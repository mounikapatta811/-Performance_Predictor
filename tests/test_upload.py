"""
Unit tests for CSV and Excel upload validation, delimiter detection, and file limits.
"""
import pytest
import io
import pandas as pd
from src.data_loader import load_file
from utils.validation import validate_file_size

def test_load_csv_comma_delimited():
    csv_content = b"student_id,attendance_rate,internal_assessment_score\nS101,88.5,72.0\nS102,75.0,60.5\n"
    file_obj = io.BytesIO(csv_content)
    df = load_file(file_obj, "roster.csv")
    assert len(df) == 2
    assert "attendance_rate" in df.columns
    assert df["attendance_rate"].iloc[0] == 88.5

def test_load_csv_semicolon_delimited():
    # Standard UCI dataset format uses semicolons
    csv_content = b"student_id;attendance_rate;internal_assessment_score\nS201;91.0;80.0\nS202;65.5;55.0\n"
    file_obj = io.BytesIO(csv_content)
    df = load_file(file_obj, "uci_data.csv")
    assert len(df) == 2
    assert "internal_assessment_score" in df.columns
    assert df["internal_assessment_score"].iloc[1] == 55.0

def test_load_excel_file():
    df_orig = pd.DataFrame({
        "student_id": ["E1", "E2"],
        "attendance_rate": [82.0, 94.5],
        "failed_subjects_count": [0, 1]
    })
    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
        df_orig.to_excel(writer, index=False)
    excel_buf.seek(0)

    loaded_df = load_file(excel_buf, "class_list.xlsx")
    assert len(loaded_df) == 2
    assert "student_id" in loaded_df.columns
    assert loaded_df["attendance_rate"].iloc[1] == 94.5

def test_unsupported_file_extension():
    content = io.BytesIO(b"some raw text data")
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_file(content, "document.pdf")

def test_file_size_validation():
    class MockFileWithLargeSize:
        def __init__(self, size_bytes):
            self.size = size_bytes

    large_file = MockFileWithLargeSize(size_bytes=60 * 1024 * 1024)  # 60MB
    with pytest.raises(ValueError, match="exceeds maximum allowed limit"):
        validate_file_size(large_file, max_mb=50.0)

    normal_file = MockFileWithLargeSize(size_bytes=5 * 1024 * 1024)   # 5MB
    assert validate_file_size(normal_file, max_mb=50.0) is True

