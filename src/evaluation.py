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
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
    classes: List[str] = None
) -> Dict[str, Any]:
    """
    Compute rigorous evaluation metrics without overstating model performance.
    Includes confusion matrix, classification report, multiclass ROC-AUC,
    and ROC curve coordinates.
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

    # ROC-AUC calculation & ROC curve coordinates if probabilities are available
    roc_auc_val = None
    roc_curves_data = {}
    if y_prob is not None:
        try:
            if n_classes == 2:
                # Binary classification
                if y_prob.ndim == 2:
                    prob_pos = y_prob[:, 1]
                else:
                    prob_pos = y_prob
                roc_auc_val = float(roc_auc_score(y_true, prob_pos))
                fpr, tpr, _ = roc_curve(y_true, prob_pos)
                roc_curves_data[str(classes_list[1])] = {
                    "fpr": [round(float(x), 4) for x in fpr],
                    "tpr": [round(float(x), 4) for x in tpr],
                    "auc": round(float(roc_auc_val), 4)
                }
            else:
                # Multiclass classification: One-vs-Rest ROC-AUC
                roc_auc_val = float(roc_auc_score(
                    y_true,
                    y_prob,
                    multi_class="ovr",
                    average="macro",
                    labels=classes_list
                ))
                # Compute per-class ROC curves
                y_true_bin = label_binarize(y_true, classes=classes_list)
                for i, cls_name in enumerate(classes_list):
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                    cls_auc = float(auc(fpr, tpr))
                    # Subsample points if too many for clean JSON storage
                    step = max(1, len(fpr) // 50)
                    roc_curves_data[str(cls_name)] = {
                        "fpr": [round(float(x), 4) for x in fpr[::step]],
                        "tpr": [round(float(x), 4) for x in tpr[::step]],
                        "auc": round(cls_auc, 4)
                    }
        except Exception:
            roc_auc_val = None
            roc_curves_data = {}

    # Class support counts
    unique_labels, counts = np.unique(y_true, return_counts=True)
    class_counts = {str(k): int(v) for k, v in zip(unique_labels, counts)}

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
        "roc_curves": roc_curves_data,
        "confusion_matrix": cm_list,
        "classes": [str(c) for c in classes_list],
        "class_counts": class_counts,
        "classification_report": report
    }
