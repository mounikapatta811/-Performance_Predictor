"""
EduGuard AI — Database Access Layer
Robust SQLite connection management, parameterized queries, and CRUD operations.
"""
import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime

from config import DATABASE_PATH, DATABASE_DIR

def get_connection() -> sqlite3.Connection:
    """Return an active connection to the SQLite database with row factory enabled."""
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DATABASE_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    """Initialize database tables using schema.sql."""
    schema_path = DATABASE_DIR / "schema.sql"
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found at {schema_path}")
    
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_sql = f.read()

    with get_connection() as conn:
        conn.executescript(schema_sql)
        conn.commit()

# --- Student Record Operations ---

def upsert_student(student_data: Dict[str, Any]) -> None:
    """Insert or update a student record."""
    fields = [
        "student_id", "name", "department", "enrollment_year", "current_semester",
        "attendance_rate", "internal_assessment_score", "assignment_submission_rate",
        "previous_semester_grade", "failed_subjects_count", "lms_activity_hours",
        "curricular_units_1st_sem_approved", "curricular_units_1st_sem_grade",
        "curricular_units_2nd_sem_approved", "curricular_units_2nd_sem_grade",
        "gender", "scholarship_holder", "tuition_fees_up_to_date", "debtor",
        "age_at_enrollment", "risk_level", "risk_score", "latest_prediction"
    ]
    
    # Filter provided keys and supply defaults
    clean_data = {}
    for f in fields:
        clean_data[f] = student_data.get(f, None)
    clean_data["updated_at"] = datetime.now().isoformat()

    sql = f"""
        INSERT INTO students ({", ".join(clean_data.keys())})
        VALUES ({", ".join(":" + k for k in clean_data.keys())})
        ON CONFLICT(student_id) DO UPDATE SET
        {", ".join(f"{k}=excluded.{k}" for k in clean_data.keys() if k != "student_id")};
    """
    with get_connection() as conn:
        conn.execute(sql, clean_data)
        conn.commit()

def upsert_students_batch(students_list: List[Dict[str, Any]]) -> int:
    """Insert or update a batch of students."""
    count = 0
    with get_connection() as conn:
        for s in students_list:
            upsert_student(s)
            count += 1
    return count

def get_all_students() -> pd.DataFrame:
    """Retrieve all students as a pandas DataFrame."""
    with get_connection() as conn:
        df = pd.read_sql_query("SELECT * FROM students ORDER BY student_id ASC", conn)
    return df

def get_student_by_id(student_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a single student record by ID."""
    with get_connection() as conn:
        cur = conn.execute("SELECT * FROM students WHERE student_id = ?", (student_id,))
        row = cur.fetchone()
        return dict(row) if row else None

def get_students_count() -> int:
    """Get total count of students."""
    with get_connection() as conn:
        cur = conn.execute("SELECT COUNT(*) FROM students")
        return cur.fetchone()[0]

def delete_student(student_id: str) -> bool:
    """Delete a student and cascade delete associated records."""
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM students WHERE student_id = ?", (student_id,))
        conn.commit()
        return cur.rowcount > 0

def clear_all_student_data() -> None:
    """Privacy/Data Deletion feature: Clears all student and intervention records."""
    with get_connection() as conn:
        conn.execute("DELETE FROM interventions;")
        conn.execute("DELETE FROM prediction_history;")
        conn.execute("DELETE FROM recommendations;")
        conn.execute("DELETE FROM students;")
        conn.commit()

# --- Prediction History & Longitudinal Tracking ---

def log_prediction(
    student_id: str,
    prediction_date: str,
    risk_score: float,
    predicted_outcome: str,
    risk_category: str,
    top_factors: Optional[Dict[str, Any]] = None,
    model_version: str = "v1.0.0",
    is_retrospective: bool = False
) -> int:
    """Log a prediction event to longitudinal history."""
    factors_json = json.dumps(top_factors or {})
    sql = """
        INSERT INTO prediction_history (
            student_id, prediction_date, risk_score, predicted_outcome,
            risk_category, top_factors, model_version, is_retrospective
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    with get_connection() as conn:
        cur = conn.execute(
            sql,
            (
                student_id, prediction_date, float(risk_score), predicted_outcome,
                risk_category, factors_json, model_version, 1 if is_retrospective else 0
            )
        )
        # Also update the student's current risk level
        conn.execute(
            """UPDATE students SET risk_score = ?, risk_level = ?, latest_prediction = ?, updated_at = CURRENT_TIMESTAMP
               WHERE student_id = ?""",
            (float(risk_score), risk_category, predicted_outcome, student_id)
        )
        conn.commit()
        return cur.lastrowid

def get_student_history(student_id: str) -> pd.DataFrame:
    """Get longitudinal prediction history for a specific student."""
    with get_connection() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM prediction_history WHERE student_id = ? ORDER BY prediction_date ASC, created_at ASC",
            conn,
            params=(student_id,)
        )
    return df

def get_all_prediction_history() -> pd.DataFrame:
    """Get all prediction history records."""
    with get_connection() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM prediction_history ORDER BY prediction_date ASC, created_at ASC",
            conn
        )
    return df

# --- Interventions Management ---

def record_intervention(
    student_id: str,
    category: str,
    notes: str,
    mentor_name: str,
    follow_up_date: Optional[str] = None,
    status: str = "Planned",
    follow_up_outcome: Optional[str] = None,
    intervention_date: Optional[str] = None
) -> int:
    """Record a new mentor intervention."""
    date_str = intervention_date or datetime.now().strftime("%Y-%m-%d")
    sql = """
        INSERT INTO interventions (
            student_id, intervention_date, category, notes,
            mentor_name, follow_up_date, status, follow_up_outcome
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    with get_connection() as conn:
        cur = conn.execute(
            sql,
            (student_id, date_str, category, notes, mentor_name, follow_up_date, status, follow_up_outcome)
        )
        conn.commit()
        return cur.lastrowid

def update_intervention(
    intervention_id: int,
    status: str,
    notes: Optional[str] = None,
    follow_up_outcome: Optional[str] = None
) -> bool:
    """Update status, notes, or outcome for an existing intervention."""
    sql = """
        UPDATE interventions
        SET status = ?, notes = COALESCE(?, notes),
            follow_up_outcome = COALESCE(?, follow_up_outcome),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    """
    with get_connection() as conn:
        cur = conn.execute(sql, (status, notes, follow_up_outcome, intervention_id))
        conn.commit()
        return cur.rowcount > 0

def get_interventions_by_student(student_id: str) -> pd.DataFrame:
    """Get intervention records for a specific student."""
    with get_connection() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM interventions WHERE student_id = ? ORDER BY intervention_date DESC, created_at DESC",
            conn,
            params=(student_id,)
        )
    return df

def get_all_interventions() -> pd.DataFrame:
    """Get all intervention records across all students."""
    with get_connection() as conn:
        df = pd.read_sql_query(
            """
            SELECT i.*, s.name as student_name, s.department, s.risk_level
            FROM interventions i
            LEFT JOIN students s ON i.student_id = s.student_id
            ORDER BY i.intervention_date DESC, i.created_at DESC
            """,
            conn
        )
    return df

# --- Action Recommendations ---

def add_recommendation(
    student_id: str,
    recommendation_text: str,
    trigger_reason: str,
    priority: str = "Medium"
) -> int:
    """Add a rule-based student support recommendation."""
    sql = """
        INSERT INTO recommendations (student_id, recommendation_text, trigger_reason, priority)
        VALUES (?, ?, ?, ?)
    """
    with get_connection() as conn:
        cur = conn.execute(sql, (student_id, recommendation_text, trigger_reason, priority))
        conn.commit()
        return cur.lastrowid

def get_recommendations_by_student(student_id: str) -> List[Dict[str, Any]]:
    """Retrieve active recommendations for a student."""
    with get_connection() as conn:
        cur = conn.execute(
            "SELECT * FROM recommendations WHERE student_id = ? ORDER BY id DESC",
            (student_id,)
        )
        return [dict(row) for row in cur.fetchall()]

def get_all_recommendations() -> pd.DataFrame:
    """Retrieve all recommendations across all students with student metadata."""
    with get_connection() as conn:
        df = pd.read_sql_query(
            """
            SELECT r.*, s.name as student_name, s.department, s.risk_level
            FROM recommendations r
            LEFT JOIN students s ON r.student_id = s.student_id
            ORDER BY r.id DESC
            """,
            conn
        )
    return df

def update_recommendation_status(rec_id: int, status: str, mentor_notes: Optional[str] = None) -> bool:
    """Update status of a recommendation (Accepted, Modified, Completed, Dismissed)."""
    with get_connection() as conn:
        cur = conn.execute(
            "UPDATE recommendations SET status = ?, mentor_notes = COALESCE(?, mentor_notes) WHERE id = ?",
            (status, mentor_notes, rec_id)
        )
        conn.commit()
        return cur.rowcount > 0

# --- Model Metadata ---

def save_model_metadata(
    model_version: str,
    model_type: str,
    target_type: str,
    metrics: Dict[str, Any],
    features: List[str],
    sample_size: int
) -> int:
    """Store trained model metadata and evaluation results."""
    sql = """
        INSERT INTO model_metadata (
            model_version, model_type, target_type, metrics_json, features_json, sample_size
        ) VALUES (?, ?, ?, ?, ?, ?)
    """
    with get_connection() as conn:
        cur = conn.execute(
            sql,
            (model_version, model_type, target_type, json.dumps(metrics), json.dumps(features), sample_size)
        )
        conn.commit()
        return cur.lastrowid

def get_latest_model_metadata() -> Optional[Dict[str, Any]]:
    """Fetch metadata of the most recently trained model."""
    with get_connection() as conn:
        cur = conn.execute(
            "SELECT * FROM model_metadata ORDER BY id DESC LIMIT 1"
        )
        row = cur.fetchone()
        if not row:
            return None
        res = dict(row)
        res["metrics"] = json.loads(res["metrics_json"]) if res.get("metrics_json") else {}
        res["features"] = json.loads(res["features_json"]) if res.get("features_json") else []
        return res
