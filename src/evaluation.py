"""
EduGuard AI — Model Evaluation Module
Calculates multiclass and binary metrics, confusion matrix, and class-specific diagnostics.
"""
from typing import Dict, Any, List
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
    classes: List[str] = None
) -> Dict[str, Any]:
    """
    Compute rigorous evaluation metrics without overstating model performance.
    """
    classes_list = list(classes) if classes is not None else sorted(list(set(y_true)))
    n_classes = len(classes_list)

    # Core scores
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    
    # Macro and weighted precision/recall/f1
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes_list)
    cm_list = cm.tolist()

    # Per-class report dictionary
    report = classification_report(
        y_true,
        y_pred,
        labels=classes_list,
        output_dict=True,
        zero_division=0
    )

    # ROC-AUC calculation if probabilities are available
    roc_auc_val = None
    if y_prob is not None:
        try:
            if n_classes == 2:
                # Binary classification
                # Use probability of positive class
                if y_prob.ndim == 2:
                    prob_pos = y_prob[:, 1]
                else:
                    prob_pos = y_prob
                roc_auc_val = float(roc_auc_score(y_true, prob_pos))
            else:
                # Multiclass classification: One-vs-Rest ROC-AUC
                roc_auc_val = float(roc_auc_score(
                    y_true,
                    y_prob,
                    multi_class="ovr",
                    average="macro",
                    labels=classes_list
                ))
        except Exception:
            roc_auc_val = None

    return {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "precision_weighted": round(prec_weighted, 4),
        "recall_weighted": round(rec_weighted, 4),
        "f1_weighted": round(f1_weighted, 4),
        "roc_auc": round(roc_auc_val, 4) if roc_auc_val is not None else None,
        "confusion_matrix": cm_list,
        "classes": [str(c) for c in classes_list],
        "classification_report": report
    }
