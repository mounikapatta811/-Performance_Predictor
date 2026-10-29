"""
Unit tests for data preprocessing and pipeline construction.
"""
import pytest
import pandas as pd
import numpy as np
from src.preprocessing import build_preprocessor, get_feature_names_out

def test_build_preprocessor():
    num_cols = ["attendance_rate", "internal_assessment_score"]
    cat_cols = ["department"]
    
    preprocessor = build_preprocessor(num_cols, cat_cols)
    assert preprocessor is not None

    df = pd.DataFrame({
        "attendance_rate": [80.0, np.nan, 95.0],
        "internal_assessment_score": [60.0, 75.0, np.nan],
        "department": ["Computer Science", "Engineering", np.nan]
    })

    transformed = preprocessor.fit_transform(df)
    assert transformed.shape[0] == 3
    # Check transformed feature names
    names = get_feature_names_out(preprocessor, num_cols, cat_cols)
    assert len(names) >= 2
    # Ensure no NaN remaining
    assert not np.isnan(transformed).any()
