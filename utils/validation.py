"""
EduGuard AI — Validation & Data Quality Module
Validates datasets, detects anomalies, checks leakage, and builds data quality reports.
"""
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

from config import STANDARD_ACADEMIC_FIELDS, SENSITIVE_ATTRIBUTES

# Fields that should never be used as predictive features
PROTECTED_IDENTIFIERS = ["student_id", "id", "roll_no", "registration_no", "name", "email", "phone"]

def generate_data_quality_report(df: pd.DataFrame, target_col: str = "target") -> Dict[str, Any]:
    """
    Run comprehensive data quality checks on an academic dataframe.
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    
    # 1. Missing values
    missing_counts = df.isnull().sum().to_dict()
    missing_pcts = {k: round((v / total_rows) * 100, 2) for k, v in missing_counts.items() if v > 0}
    total_missing_cells = int(df.isnull().sum().sum())
    
    # 2. Duplicates
    dup_rows = int(df.duplicated().sum())
    id_duplicates = 0
    if "student_id" in df.columns:
        id_duplicates = int(df["student_id"].duplicated().sum())

    # 3. Range & Anomaly Checks on standard academic fields
    anomalies = []
    for col, spec in STANDARD_ACADEMIC_FIELDS.items():
        if col in df.columns:
            if spec["type"] in ["float", "int"]:
                # Check for negative values
                neg_count = int((df[col] < spec.get("min", 0.0)).sum())
                if neg_count > 0:
                    anomalies.append(f"Column '{col}' has {neg_count} values below valid minimum ({spec.get('min', 0)}).")
                # Check for values exceeding max
                max_val = spec.get("max", 100.0)
                max_count = int((df[col] > max_val).sum())
                if max_count > 0:
                    anomalies.append(f"Column '{col}' has {max_count} values exceeding maximum ({max_val}).")

    # 4. Target Distribution (if present)
    target_info = {}
    if target_col in df.columns:
        target_counts = df[target_col].value_counts().to_dict()
        target_info = {str(k): int(v) for k, v in target_counts.items()}

    # 5. Potential Leakage Check (checking for student identifiers in feature set)
    id_columns_found = [c for c in df.columns if c.lower() in PROTECTED_IDENTIFIERS or "student" in c.lower() and "id" in c.lower()]

    return {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "missing_counts": missing_counts,
        "missing_pcts": missing_pcts,
        "total_missing_cells": total_missing_cells,
        "duplicate_rows": dup_rows,
        "id_duplicates": id_duplicates,
        "anomalies": anomalies,
        "target_info": target_info,
        "id_columns_found": id_columns_found,
        "is_healthy": (total_missing_cells == 0 and dup_rows == 0 and len(anomalies) == 0)
    }

def clean_academic_dataset(
    df: pd.DataFrame,
    handle_missing: str = "impute",
    remove_duplicates: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Clean dataset with full transparency.
    Returns cleaned_df and audit_log of transformations applied.
    """
    cleaned_df = df.copy()
    audit_log = []
    
    # 1. Duplicate removal
    if remove_duplicates:
        initial_len = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        dropped = initial_len - len(cleaned_df)
        if dropped > 0:
            audit_log.append(f"Removed {dropped} duplicate rows.")

    # 2. Missing value handling
    missing_cols = [c for c in cleaned_df.columns if cleaned_df[c].isnull().sum() > 0]
    for c in missing_cols:
        n_missing = int(cleaned_df[c].isnull().sum())
        if pd.api.types.is_numeric_dtype(cleaned_df[c]):
            if handle_missing == "impute":
                median_val = cleaned_df[c].median()
                cleaned_df[c] = cleaned_df[c].fillna(median_val)
                audit_log.append(f"Column '{c}': Imputed {n_missing} missing values with column median ({median_val:.2f}).")
            elif handle_missing == "drop":
                cleaned_df = cleaned_df.dropna(subset=[c])
                audit_log.append(f"Column '{c}': Dropped {n_missing} rows containing missing values.")
        else:
            if handle_missing == "impute":
                mode_val = cleaned_df[c].mode()[0] if not cleaned_df[c].mode().empty else "Unknown"
                cleaned_df[c] = cleaned_df[c].fillna(mode_val)
                audit_log.append(f"Column '{c}': Imputed {n_missing} missing values with mode ('{mode_val}').")
            elif handle_missing == "drop":
                cleaned_df = cleaned_df.dropna(subset=[c])
                audit_log.append(f"Column '{c}': Dropped {n_missing} rows with missing categories.")

    # 3. Numeric bounds clamping for academic sanity
    for col, spec in STANDARD_ACADEMIC_FIELDS.items():
        if col in cleaned_df.columns and spec["type"] in ["float", "int"]:
            min_val = spec.get("min", 0.0)
            max_val = spec.get("max", 100.0)
            clipped = cleaned_df[col].clip(lower=min_val, upper=max_val)
            diffs = (cleaned_df[col] != clipped).sum()
            if diffs > 0:
                cleaned_df[col] = clipped
                audit_log.append(f"Column '{col}': Clamped {diffs} out-of-bounds values to range [{min_val}, {max_val}].")

    return cleaned_df, audit_log

def separate_features_and_target(
    df: pd.DataFrame,
    target_col: str = "target",
    exclude_sensitive: bool = True
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Safely extract predictive features and target while:
    - Never including Student ID or name (leakage prevention)
    - Optionally excluding sensitive demographic attributes (Fairness & Ethics)
    Returns (X, y, numeric_features, categorical_features).
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    y = df[target_col].copy()
    
    # Exclude target, identifiers, and sensitive columns if configured
    cols_to_drop = [target_col]
    
    # Exclude identifiers
    for col in df.columns:
        c_lower = col.lower()
        if c_lower in PROTECTED_IDENTIFIERS or ("student" in c_lower and "id" in c_lower) or c_lower in ["name", "email"]:
            cols_to_drop.append(col)

    # Exclude sensitive attributes if requested
    if exclude_sensitive:
        for s_col in SENSITIVE_ATTRIBUTES.keys():
            if s_col in df.columns and s_col not in cols_to_drop:
                cols_to_drop.append(s_col)

    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    numeric_features = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_features = [c for c in X.columns if c not in numeric_features]

    return X, y, numeric_features, categorical_features
