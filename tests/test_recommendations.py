"""
Unit tests for the rule-based recommendation engine.
"""
import pytest
from src.recommendations import generate_action_recommendations

def test_attendance_and_deficit_recommendations():
    # Student with severe attendance issues and multiple failed subjects
    at_risk_student = {
        "attendance_rate": 55.0,
        "internal_assessment_score": 42.0,
        "assignment_submission_rate": 50.0,
        "failed_subjects_count": 3,
        "curricular_units_1st_sem_approved": 1,
        "curricular_units_2nd_sem_approved": 1
    }

    recs = generate_action_recommendations(at_risk_student, risk_category="High", risk_score=0.82)
    assert len(recs) >= 3
    
    categories = [r["category"] for r in recs]
    assert "Attendance Review" in categories
    assert "Academic Tutoring" in categories
    assert "Curricular Counseling" in categories
    
    # Verify non-punitive supportive phrasing
    for r in recs:
        text_lower = r["text"].lower()
        assert "disciplinary" not in text_lower
        assert "penalize" not in text_lower
        assert "dismiss" not in text_lower
        assert len(r["text"]) > 10
