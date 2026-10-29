"""
EduGuard AI — Risk Prediction Module
Performs individual and batch student inference, computes calibrated risk probabilities,
categorizes risk levels, and assesses predictive uncertainty.
"""
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.stats import entropy

from config import DEFAULT_RISK_THRESHOLDS

def calculate_student_risk(
    class_probabilities: Dict[str, float],
    thresholds: Dict[str, float] = None
) -> Tuple[float, str]:
    """
    Compute estimated risk score and category from class probabilities.
    For UCI 3-class model ('Dropout', 'Enrolled', 'Graduate'):
    - Risk is primarily driven by P(Dropout) with secondary weight on academic uncertainty P(Enrolled).
    Risk Formula: P(Dropout) + 0.25 * P(Enrolled)
    """
    if thresholds is None:
        thresholds = DEFAULT_RISK_THRESHOLDS

    p_dropout = class_probabilities.get("Dropout", 0.0)
    p_enrolled = class_probabilities.get("Enrolled", 0.0)
    
    # If binary target
    if "Dropout" in class_probabilities and len(class_probabilities) == 2:
        risk_score = float(p_dropout)
    elif "Dropout" in class_probabilities:
        # 3-class model
        risk_score = float(np.clip(p_dropout + 0.25 * p_enrolled, 0.0, 1.0))
    else:
        # Generic classification: probability of the most negative class or second class
        keys = list(class_probabilities.keys())
        risk_score = float(class_probabilities[keys[0]]) if keys else 0.5

    if risk_score >= thresholds.get("high", 0.65):
        category = "High"
    elif risk_score >= thresholds.get("moderate", 0.35):
        category = "Moderate"
    else:
        category = "Low"

    return round(risk_score, 4), category

def predict_single_student(
    pipeline,
    label_encoder,
    student_dict: Dict[str, Any],
    thresholds: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Generate inference, risk assessment, and uncertainty metrics for an individual student.
    """
    # Create single-row DataFrame
    input_df = pd.DataFrame([student_dict])
    
    # Predict class and probabilities
    pred_idx = pipeline.predict(input_df)[0]
    predicted_label = str(label_encoder.inverse_transform([pred_idx])[0])

    classes = [str(c) for c in label_encoder.classes_]
    
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(input_df)[0]
        prob_dict = {classes[i]: round(float(probs[i]), 4) for i in range(len(classes))}
        
        # Uncertainty Metrics
        pred_entropy = float(entropy(probs) / np.log(len(classes))) if len(classes) > 1 else 0.0
        sorted_probs = np.sort(probs)
        margin_of_confidence = float(sorted_probs[-1] - sorted_probs[-2]) if len(probs) > 1 else 1.0
    else:
        prob_dict = {predicted_label: 1.0}
        pred_entropy = 0.0
        margin_of_confidence = 1.0

    risk_score, risk_category = calculate_student_risk(prob_dict, thresholds)

    # Uncertainty level interpretation
    if margin_of_confidence < 0.15 or pred_entropy > 0.80:
        uncertainty_level = "High Uncertainty — Prediction is borderline; mentor review strongly recommended."
    elif margin_of_confidence < 0.30:
        uncertainty_level = "Moderate Uncertainty — Multiple outcome possibilities are viable."
    else:
        uncertainty_level = "Low Uncertainty — High model agreement."

    return {
        "predicted_outcome": predicted_label,
        "risk_score": risk_score,
        "risk_percentage": round(risk_score * 100, 1),
        "risk_category": risk_category,
        "class_probabilities": prob_dict,
        "uncertainty_entropy": round(pred_entropy, 3),
        "margin_of_confidence": round(margin_of_confidence, 3),
        "uncertainty_level": uncertainty_level
    }

def predict_batch_students(
    pipeline,
    label_encoder,
    df: pd.DataFrame,
    thresholds: Dict[str, float] = None
) -> pd.DataFrame:
    """
    Run batch inference over a DataFrame of multiple students.
    Returns DataFrame with appended prediction columns.
    """
    results_df = df.copy()
    
    # Predict
    preds_idx = pipeline.predict(df)
    predicted_labels = label_encoder.inverse_transform(preds_idx)
    results_df["predicted_outcome"] = predicted_labels

    classes = [str(c) for c in label_encoder.classes_]
    
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(df)
        
        # Calculate risk scores for each row
        risk_scores = []
        risk_cats = []
        confidences = []
        
        for i in range(len(df)):
            row_prob_dict = {classes[j]: float(probs[i, j]) for j in range(len(classes))}
            r_score, r_cat = calculate_student_risk(row_prob_dict, thresholds)
            risk_scores.append(r_score)
            risk_cats.append(r_cat)
            
            s_probs = np.sort(probs[i])
            conf = float(s_probs[-1] - s_probs[-2]) if len(s_probs) > 1 else 1.0
            confidences.append(round(conf, 3))

            # Store probability columns
            for j, c in enumerate(classes):
                results_df[f"prob_{c.lower()}"] = np.round(probs[:, j], 4)

        results_df["risk_score"] = risk_scores
        results_df["risk_percentage"] = [round(r * 100, 1) for r in risk_scores]
        results_df["risk_category"] = risk_cats
        results_df["confidence_margin"] = confidences
    else:
        results_df["risk_score"] = 0.5
        results_df["risk_category"] = "Moderate"

    return results_df
