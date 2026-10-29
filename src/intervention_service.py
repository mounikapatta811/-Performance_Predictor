"""
EduGuard AI — Intervention Management Service
Handles mentor intervention workflows, status tracking, and descriptive outcome monitoring.
"""
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime

from database.db import (
    record_intervention,
    update_intervention,
    get_interventions_by_student,
    get_all_interventions,
    get_student_by_id,
    get_student_history
)

INTERVENTION_CATEGORIES = [
    "Academic Tutoring & Remedial Classes",
    "Attendance Review & Counseling",
    "Assignment & Time Management Coaching",
    "Curricular Recovery & Prerequisite Planning",
    "Digital Accessibility & Technical Support",
    "Financial Aid & Scholarship Consultation",
    "Wellness, Stress & Mental Health Support",
    "Faculty Advisor One-on-One Check-in"
]

INTERVENTION_STATUSES = [
    "Planned",
    "In Progress",
    "Completed",
    "Dismissed"
]

def log_mentor_intervention(
    student_id: str,
    category: str,
    notes: str,
    mentor_name: str,
    follow_up_date: Optional[str] = None,
    status: str = "Planned",
    follow_up_outcome: Optional[str] = None,
    intervention_date: Optional[str] = None
) -> int:
    """Record a new intervention for a student."""
    return record_intervention(
        student_id=student_id,
        category=category,
        notes=notes,
        mentor_name=mentor_name,
        follow_up_date=follow_up_date,
        status=status,
        follow_up_outcome=follow_up_outcome,
        intervention_date=intervention_date
    )

def evaluate_student_progress_post_intervention(student_id: str) -> Dict[str, Any]:
    """
    Descriptively compare a student's risk trend before and after interventions.
    Note: Explicitly marked as descriptive tracking, avoiding claims of strict causal efficacy.
    """
    interventions_df = get_interventions_by_student(student_id)
    history_df = get_student_history(student_id)
    student = get_student_by_id(student_id)

    if interventions_df.empty or history_df.empty:
        return {
            "has_data": False,
            "message": "Insufficient intervention or longitudinal history records."
        }

    first_int_date = interventions_df["intervention_date"].min()
    
    # Pre-intervention history
    pre_history = history_df[history_df["prediction_date"] <= first_int_date]
    post_history = history_df[history_df["prediction_date"] > first_int_date]

    pre_avg_risk = float(pre_history["risk_score"].mean()) if not pre_history.empty else float(history_df["risk_score"].iloc[0])
    post_avg_risk = float(post_history["risk_score"].mean()) if not post_history.empty else float(history_df["risk_score"].iloc[-1])
    current_risk = float(student.get("risk_score", 0.0)) if student else post_avg_risk

    delta = current_risk - pre_avg_risk

    if delta < -0.08:
        trend_status = "Positive Progress (Descriptive risk reduction observed)"
    elif delta > 0.08:
        trend_status = "Elevated Risk (Further mentor engagement recommended)"
    else:
        trend_status = "Stable (Risk levels consistent)"

    return {
        "has_data": True,
        "first_intervention_date": first_int_date,
        "pre_average_risk": round(pre_avg_risk, 3),
        "post_average_risk": round(post_avg_risk, 3),
        "current_risk": round(current_risk, 3),
        "risk_delta": round(delta, 3),
        "trend_status": trend_status,
        "interventions_count": len(interventions_df),
        "causal_disclaimer": "Observed shifts are descriptive and should not be interpreted as proving causal intervention impact without controlled trial evaluation."
    }
