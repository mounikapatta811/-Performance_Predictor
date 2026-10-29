"""
Unit tests for database interactions, parameterized queries, and persistence.
"""
import pytest
import sqlite3
from database.db import (
    init_db,
    upsert_student,
    get_student_by_id,
    record_intervention,
    get_interventions_by_student,
    log_prediction,
    get_student_history,
    clear_all_student_data,
    get_students_count
)

def test_database_crud_and_privacy():
    init_db()
    
    # 1. Upsert student
    test_student = {
        "student_id": "TEST-STU-001",
        "name": "Test Student",
        "department": "Computer Science",
        "attendance_rate": 78.5,
        "internal_assessment_score": 62.0,
        "risk_level": "Moderate",
        "risk_score": 0.45
    }
    upsert_student(test_student)

    retrieved = get_student_by_id("TEST-STU-001")
    assert retrieved is not None
    assert retrieved["name"] == "Test Student"
    assert retrieved["attendance_rate"] == 78.5

    # 2. Record intervention
    int_id = record_intervention(
        student_id="TEST-STU-001",
        category="Academic Tutoring",
        notes="Reviewed calculus concepts.",
        mentor_name="Advisor Smith",
        status="Planned"
    )
    assert int_id > 0
    ints = get_interventions_by_student("TEST-STU-001")
    assert len(ints) >= 1

    # 3. Log prediction history
    log_id = log_prediction(
        student_id="TEST-STU-001",
        prediction_date="2024-09-01",
        risk_score=0.45,
        predicted_outcome="Enrolled",
        risk_category="Moderate"
    )
    assert log_id > 0
    hist = get_student_history("TEST-STU-001")
    assert len(hist) >= 1

    # 4. Privacy Purge Test
    clear_all_student_data()
    assert get_students_count() == 0
    assert get_student_by_id("TEST-STU-001") is None
