"""
Unit tests for model training, prediction, and risk scoring.
"""
import pytest
import pandas as pd
import numpy as np
from src.data_loader import generate_synthetic_academic_dataset
from src.train_model import train_academic_model
from src.predict import calculate_student_risk, predict_single_student, predict_batch_students
from utils.validation import separate_features_and_target

def test_risk_scoring_logic():
    # 3-class probability case
    probs_high_risk = {"Dropout": 0.70, "Enrolled": 0.20, "Graduate": 0.10}
    score, cat = calculate_student_risk(probs_high_risk)
    assert score >= 0.65
    assert cat == "High"

    probs_low_risk = {"Dropout": 0.05, "Enrolled": 0.15, "Graduate": 0.80}
    score, cat = calculate_student_risk(probs_low_risk)
    assert score < 0.35
    assert cat == "Low"

def test_model_training_and_inference():
    # Generate small synthetic dataset
    df = generate_synthetic_academic_dataset(n_samples=60, random_state=42)
    X, y, num_cols, cat_cols = separate_features_and_target(df, target_col="target")

    # Train model
    res = train_academic_model(
        X=X,
        y=y,
        numeric_features=num_cols,
        categorical_features=cat_cols,
        model_type="Random Forest",
        test_size=0.25,
        random_state=42,
        cv_folds=3
    )

    pipeline = res["pipeline"]
    label_encoder = res["label_encoder"]
    metadata = res["metadata"]

    assert pipeline is not None
    assert "accuracy" in metadata["test_metrics"]
    assert metadata["test_samples"] > 0

    # Single student inference
    sample_student = X.iloc[0].to_dict()
    single_res = predict_single_student(pipeline, label_encoder, sample_student)
    assert single_res["predicted_outcome"] in metadata["classes"]
    assert 0.0 <= single_res["risk_score"] <= 1.0
    assert single_res["risk_category"] in ["High", "Moderate", "Low"]

    # Batch inference
    batch_res = predict_batch_students(pipeline, label_encoder, X.head(5))
    assert len(batch_res) == 5
    assert "risk_score" in batch_res.columns
    assert "risk_category" in batch_res.columns
