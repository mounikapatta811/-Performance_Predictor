"""
EduGuard AI — Explainable AI (XAI) Module
Provides SHAP global and local explanations with transparent fallbacks
and natural-language factor interpretations.
"""
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from src.preprocessing import get_feature_names_out

def compute_global_importance(
    pipeline,
    X_sample: pd.DataFrame,
    y_sample: np.ndarray = None,
    max_features: int = 15
) -> Dict[str, Any]:
    """
    Compute global feature importance using SHAP TreeExplainer / LinearExplainer,
    or falling back gracefully to model feature_importances_ or Permutation Importance.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    
    # Transform sample
    X_trans = preprocessor.transform(X_sample)
    
    # Feature names
    num_cols = preprocessor.transformers_[0][2] if len(preprocessor.transformers_) > 0 else []
    cat_cols = preprocessor.transformers_[1][2] if len(preprocessor.transformers_) > 1 else []
    feat_names = get_feature_names_out(preprocessor, num_cols, cat_cols)

    importance_method = "Unknown"
    scores = {}

    if HAS_SHAP:
        try:
            # Check classifier type
            c_name = classifier.__class__.__name__
            if "Forest" in c_name or "XGB" in c_name:
                explainer = shap.TreeExplainer(classifier)
                shap_values = explainer.shap_values(X_trans)
                
                # Handle multiclass vs binary
                if isinstance(shap_values, list):
                    # Multiclass: take mean absolute SHAP value across all classes
                    mean_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0).mean(axis=0)
                elif shap_values.ndim == 3:
                    mean_abs = np.abs(shap_values).mean(axis=(0, 2))
                else:
                    mean_abs = np.abs(shap_values).mean(axis=0)
                    
                importance_method = "SHAP TreeExplainer (Mean Absolute SHAP Value)"
                scores = {feat_names[i]: float(mean_abs[i]) for i in range(min(len(feat_names), len(mean_abs)))}
            else:
                # Linear model or fallback
                explainer = shap.LinearExplainer(classifier, X_trans)
                shap_values = explainer.shap_values(X_trans)
                if isinstance(shap_values, list):
                    mean_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0).mean(axis=0)
                else:
                    mean_abs = np.abs(shap_values).mean(axis=0)
                importance_method = "SHAP LinearExplainer"
                scores = {feat_names[i]: float(mean_abs[i]) for i in range(min(len(feat_names), len(mean_abs)))}
        except Exception:
            scores = {}

    # Fallback to model intrinsic importance if SHAP failed or is unavailable
    if not scores:
        if hasattr(classifier, "feature_importances_"):
            importance_method = "Model Intrinsic Feature Importance (Gini/Information Gain)"
            raw_scores = classifier.feature_importances_
            scores = {feat_names[i]: float(raw_scores[i]) for i in range(min(len(feat_names), len(raw_scores)))}
        elif hasattr(classifier, "coef_"):
            importance_method = "Logistic Regression Absolute Coefficients"
            raw_scores = np.mean(np.abs(classifier.coef_), axis=0) if classifier.coef_.ndim > 1 else np.abs(classifier.coef_)
            scores = {feat_names[i]: float(raw_scores[i]) for i in range(min(len(feat_names), len(raw_scores)))}
        elif y_sample is not None:
            importance_method = "Permutation Feature Importance Fallback"
            r = permutation_importance(pipeline, X_sample, y_sample, n_repeats=3, random_state=42)
            scores = {X_sample.columns[i]: float(r.importances_mean[i]) for i in range(len(X_sample.columns))}

    # Sort descending
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:max_features]
    
    return {
        "method": importance_method,
        "features": [item[0] for item in sorted_items],
        "values": [round(item[1], 4) for item in sorted_items]
    }

def explain_single_prediction(
    pipeline,
    label_encoder,
    student_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Explain factors driving an individual student's risk assessment.
    Returns impact scores, positive/negative drivers, and natural-language narrative.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    
    input_df = pd.DataFrame([student_dict])
    X_trans = preprocessor.transform(input_df)

    num_cols = preprocessor.transformers_[0][2] if len(preprocessor.transformers_) > 0 else []
    cat_cols = preprocessor.transformers_[1][2] if len(preprocessor.transformers_) > 1 else []
    feat_names = get_feature_names_out(preprocessor, num_cols, cat_cols)

    feature_impacts = []
    explanation_method = "Model Feature Deviation Fallback"

    # Try SHAP Local Explanation
    if HAS_SHAP:
        try:
            c_name = classifier.__class__.__name__
            if "Forest" in c_name or "XGB" in c_name:
                explainer = shap.TreeExplainer(classifier)
                shap_vals = explainer.shap_values(X_trans)
                
                # If multiclass: identify index of dropout or predicted class
                target_idx = 0
                classes = list(label_encoder.classes_)
                if "Dropout" in classes:
                    target_idx = classes.index("Dropout")

                if isinstance(shap_vals, list):
                    vals = shap_vals[target_idx][0]
                elif shap_vals.ndim == 3:
                    vals = shap_vals[0, :, target_idx]
                else:
                    vals = shap_vals[0]
                    
                explanation_method = f"SHAP Local Explainer for '{classes[target_idx]}'"
                for i in range(min(len(feat_names), len(vals))):
                    feature_impacts.append({
                        "feature": feat_names[i],
                        "impact": float(vals[i]),
                        "direction": "Increases Risk" if vals[i] > 0 else "Decreases Risk"
                    })
        except Exception:
            feature_impacts = []

    # Heuristic / Deviation fallback if SHAP did not run
    if not feature_impacts:
        # Measure deviation from normal academic thresholds
        att = student_dict.get("attendance_rate", 80.0)
        ass = student_dict.get("internal_assessment_score", 65.0)
        failed = student_dict.get("failed_subjects_count", 0)
        cu2_g = student_dict.get("curricular_units_2nd_sem_grade", 13.0)
        cu2_a = student_dict.get("curricular_units_2nd_sem_approved", 5)

        deviations = [
            ("Attendance Rate", (75.0 - att) * 0.015, "Low attendance elevated risk" if att < 75 else "Strong attendance reduced risk"),
            ("Failed Subjects Deficit", failed * 0.12, f"{failed} failed curricular units elevated risk" if failed > 0 else "Zero failed units reduced risk"),
            ("Assessment Score", (60.0 - ass) * 0.012, "Assessment score below benchmark elevated risk" if ass < 60 else "Assessment score met benchmark"),
            ("2nd Sem Units Approved", (4 - cu2_a) * 0.08, f"Only {cu2_a} units approved in 2nd semester" if cu2_a < 4 else f"{cu2_a} units successfully approved"),
            ("2nd Sem Average Grade", (12.0 - cu2_g) * 0.04, "Average grade below passing benchmark" if cu2_g < 12 else "Passing grade benchmark maintained")
        ]

        for feat, impact, desc in deviations:
            feature_impacts.append({
                "feature": feat,
                "impact": float(impact),
                "direction": "Increases Risk" if impact > 0 else "Decreases Risk",
                "note": desc
            })

    # Sort by absolute impact
    feature_impacts.sort(key=lambda x: abs(x["impact"]), reverse=True)
    top_factors = feature_impacts[:8]

    # Generate Natural Language Narrative
    increasing_factors = [f["feature"] for f in top_factors if f["impact"] > 0]
    decreasing_factors = [f["feature"] for f in top_factors if f["impact"] < 0]

    narrative_parts = []
    if increasing_factors:
        narrative_parts.append(
            f"The primary academic factors elevating estimated risk are **{', '.join(increasing_factors[:3])}**."
        )
    if decreasing_factors:
        narrative_parts.append(
            f"Conversely, strong positive performance in **{', '.join(decreasing_factors[:2])}** provides mitigating support."
        )

    if not narrative_parts:
        narrative_parts.append("Student performance indicators are well balanced around median institution benchmarks.")

    narrative_parts.append(
        "\n*Notice: Predictive associations indicate statistical correlation within model patterns, not direct causal certainty. Use these indicators to guide supportive human mentor conversations.*"
    )

    full_narrative = " ".join(narrative_parts)

    return {
        "method": explanation_method,
        "feature_impacts": top_factors,
        "increasing_factors": [f for f in top_factors if f["impact"] > 0],
        "decreasing_factors": [f for f in top_factors if f["impact"] < 0],
        "narrative": full_narrative
    }
