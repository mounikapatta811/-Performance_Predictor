"""
EduGuard AI — Model Training Module
Trains and compares Logistic Regression, Random Forest, and XGBoost models
with leak-free preprocessing pipelines and class weight handling.
"""
from typing import Dict, Any, List, Tuple, Optional
import os
import json
import joblib
from datetime import datetime
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from config import MODELS_DIR, DEFAULT_MODEL_PATH, METADATA_PATH
from src.preprocessing import build_preprocessor, get_feature_names_out
from src.evaluation import evaluate_classifier

def get_model_instance(model_type: str, random_state: int = 42):
    """Instantiate classifier with balanced class weights."""
    if model_type == "Random Forest":
        return RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=4,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        )
    elif model_type == "Logistic Regression":
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
            solver="lbfgs"
        )
    elif model_type == "XGBoost":
        if not HAS_XGBOOST:
            raise ValueError("XGBoost is not installed in the environment.")
        return XGBClassifier(
            n_estimators=120,
            max_depth=5,
            learning_rate=0.08,
            random_state=random_state,
            eval_metric="mlogloss"
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def train_academic_model(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_features: List[str],
    categorical_features: List[str],
    model_type: str = "Random Forest",
    test_size: float = 0.25,
    random_state: int = 42,
    cv_folds: int = 5
) -> Dict[str, Any]:
    """
    Train a complete ML pipeline:
    1. Stratified split into train/test
    2. Build preprocessor
    3. Fit model within Pipeline (preventing leakage)
    4. Compute train, validation, and test metrics
    5. Save artifacts
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Target label validation & cleanup
    valid_mask = y.notnull()
    if valid_mask.sum() < len(y):
        X = X[valid_mask].copy()
        y = y[valid_mask].copy()

    unique_targets = pd.Series(y).unique()
    if len(unique_targets) < 2:
        raise ValueError(
            f"Target column must contain at least 2 distinct classes. Found {len(unique_targets)}: {list(unique_targets)}"
        )

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    classes = [str(c) for c in label_encoder.classes_]

    # Stratified Split (handles class distribution properly)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded
    )

    # Preprocessor
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    classifier = get_model_instance(model_type, random_state=random_state)

    # Full Pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    # Cross-validation on training data only (Zero Leakage, safe against small samples)
    min_class_count = int(np.min(np.bincount(y_train)))
    if min_class_count >= 2:
        actual_cv_splits = min(cv_folds, min_class_count)
        cv = StratifiedKFold(n_splits=actual_cv_splits, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_macro")
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))
        cv_status = f"Stratified {actual_cv_splits}-Fold Cross-Validation"
    else:
        cv_mean = 0.0
        cv_std = 0.0
        cv_status = "Sample size per class is too small for cross-validation"

    # Fit final pipeline on training set
    pipeline.fit(X_train, y_train)

    # Training set evaluation
    y_train_pred = pipeline.predict(X_train)
    y_train_prob = pipeline.predict_proba(X_train) if hasattr(pipeline, "predict_proba") else None
    train_eval = evaluate_classifier(y_train, y_train_pred, y_train_prob, classes=list(range(len(classes))))

    # Test set evaluation
    y_test_pred = pipeline.predict(X_test)
    y_test_prob = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None
    test_eval = evaluate_classifier(y_test, y_test_pred, y_test_prob, classes=list(range(len(classes))))

    # Map numeric classes back to actual names
    test_eval["class_labels"] = classes
    train_eval["class_labels"] = classes

    # Compute class distribution in train and test sets
    train_counts_named = {classes[int(k)]: v for k, v in train_eval.get("class_counts", {}).items() if int(k) < len(classes)}
    test_counts_named = {classes[int(k)]: v for k, v in test_eval.get("class_counts", {}).items() if int(k) < len(classes)}

    # Feature names after preprocessing
    fitted_preprocessor = pipeline.named_steps["preprocessor"]
    transformed_features = get_feature_names_out(fitted_preprocessor, numeric_features, categorical_features)

    # Model metadata package
    timestamp = datetime.now().isoformat()
    metadata = {
        "model_version": f"eduguard-{model_type.lower().replace(' ', '-')}-{datetime.now().strftime('%Y%m%d%H%M')}",
        "model_type": model_type,
        "classes": classes,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "transformed_features": transformed_features,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "train_class_distribution": train_counts_named,
        "test_class_distribution": test_counts_named,
        "cv_folds": cv_folds,
        "cv_status": cv_status,
        "cv_f1_macro_mean": round(cv_mean, 4),
        "cv_f1_macro_std": round(cv_std, 4),
        "test_metrics": test_eval,
        "train_metrics": train_eval,
        "training_timestamp": timestamp,
        "random_state": random_state
    }

    # Save Pipeline and LabelEncoder
    saved_payload = {
        "pipeline": pipeline,
        "label_encoder": label_encoder,
        "metadata": metadata
    }
    joblib.dump(saved_payload, DEFAULT_MODEL_PATH)

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "pipeline": pipeline,
        "label_encoder": label_encoder,
        "metadata": metadata,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }

def load_trained_model():
    """Load the latest saved pipeline and metadata."""
    if not DEFAULT_MODEL_PATH.exists():
        return None
    return joblib.load(DEFAULT_MODEL_PATH)
