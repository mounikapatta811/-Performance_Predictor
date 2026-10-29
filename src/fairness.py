"""
EduGuard AI — Fairness & Responsible AI Auditing Module
Calculates demographic parity, equalized odds, group-level recall/precision/FPR,
and surfaces potential algorithmic disparities with statistical warnings.
"""
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score, confusion_matrix

def audit_group_fairness(
    df: pd.DataFrame,
    group_col: str,
    y_true_col: str = "target",
    y_pred_col: str = "predicted_outcome",
    positive_class: str = "Dropout",
    min_sample_threshold: int = 20
) -> Dict[str, Any]:
    """
    Evaluate model fairness and performance disparities across groups defined by group_col.
    """
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found in DataFrame.")
    if y_true_col not in df.columns:
        raise ValueError(f"True label column '{y_true_col}' not found in DataFrame.")
    if y_pred_col not in df.columns:
        raise ValueError(f"Predicted outcome column '{y_pred_col}' not found in DataFrame.")

    # Filter out missing rows
    valid_df = df.dropna(subset=[group_col, y_true_col, y_pred_col]).copy()
    
    # Binary conversion for fair metrics (positive_class vs rest)
    valid_df["y_true_bin"] = (valid_df[y_true_col] == positive_class).astype(int)
    valid_df["y_pred_bin"] = (valid_df[y_pred_col] == positive_class).astype(int)

    groups = valid_df[group_col].unique()
    group_metrics = []
    warnings = []

    for g in groups:
        gdf = valid_df[valid_df[group_col] == g]
        n_samples = len(gdf)
        
        if n_samples < min_sample_threshold:
            warnings.append(
                f"Group '{g}' has only {n_samples} samples. Metrics have high variance and should be interpreted with caution."
            )

        y_t = gdf["y_true_bin"].values
        y_p = gdf["y_pred_bin"].values

        # Base rates
        base_rate = float(np.mean(y_t)) if n_samples > 0 else 0.0
        selection_rate = float(np.mean(y_p)) if n_samples > 0 else 0.0

        # Confusion elements
        tn, fp, fn, tp = 0, 0, 0, 0
        if len(np.unique(y_t)) > 1 or len(np.unique(y_p)) > 1:
            cm = confusion_matrix(y_t, y_p, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
        elif len(y_t) > 0:
            if y_t[0] == 1 and y_p[0] == 1:
                tp = n_samples
            elif y_t[0] == 0 and y_p[0] == 0:
                tn = n_samples
            elif y_t[0] == 1 and y_p[0] == 0:
                fn = n_samples
            else:
                fp = n_samples

        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

        group_metrics.append({
            "group": str(g),
            "sample_count": n_samples,
            "sample_percentage": round((n_samples / len(valid_df)) * 100, 1),
            "base_rate": round(base_rate, 3),
            "selection_rate": round(selection_rate, 3),
            "recall": round(recall, 3),
            "precision": round(precision, 3),
            "false_positive_rate": round(fpr, 3),
            "false_negative_rate": round(fnr, 3),
            "is_small_sample": n_samples < min_sample_threshold
        })

    # Calculate disparity ratios (Disparate Impact ratio & Recall gap)
    metrics_df = pd.DataFrame(group_metrics)
    if len(metrics_df) >= 2:
        max_sel = metrics_df["selection_rate"].max()
        min_sel = metrics_df["selection_rate"].min()
        selection_ratio = round(min_sel / max_sel, 3) if max_sel > 0 else 1.0

        max_rec = metrics_df["recall"].max()
        min_rec = metrics_df["recall"].min()
        recall_gap = round(max_rec - min_rec, 3)

        max_fpr = metrics_df["false_positive_rate"].max()
        min_fpr = metrics_df["false_positive_rate"].min()
        fpr_gap = round(max_fpr - min_fpr, 3)
    else:
        selection_ratio = 1.0
        recall_gap = 0.0
        fpr_gap = 0.0

    # Interpret Disparate Impact (Four-Fifths Rule: 80%)
    if selection_ratio < 0.80 and len(metrics_df) >= 2:
        warnings.append(
            f"Adverse impact threshold detected: Selection rate ratio ({selection_ratio:.2f}) across '{group_col}' is below 0.80 (80% rule)."
        )

    return {
        "group_column": group_col,
        "positive_class": positive_class,
        "total_evaluated_students": len(valid_df),
        "group_metrics": group_metrics,
        "selection_ratio": selection_ratio,
        "recall_gap": recall_gap,
        "fpr_gap": fpr_gap,
        "warnings": warnings,
        "guidelines": [
            "Demographic attributes must never be used as predictive indicators of individual academic potential.",
            "Fairness audits are diagnostic tools designed to detect institutional and systemic barriers.",
            "Use predictions strictly for targeted academic support and resource allocation, never for exclusion or sanctions."
        ]
    }
