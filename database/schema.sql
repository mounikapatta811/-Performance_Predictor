-- EduGuard AI Database Schema
-- SQLite schema for students, prediction history, interventions, and model metadata

CREATE TABLE IF NOT EXISTS students (
    student_id TEXT PRIMARY KEY,
    name TEXT DEFAULT 'Anonymous Student',
    department TEXT DEFAULT 'General Studies',
    enrollment_year INTEGER DEFAULT 2024,
    current_semester INTEGER DEFAULT 2,
    attendance_rate REAL DEFAULT 80.0,
    internal_assessment_score REAL DEFAULT 65.0,
    assignment_submission_rate REAL DEFAULT 80.0,
    previous_semester_grade REAL DEFAULT 13.0,
    failed_subjects_count INTEGER DEFAULT 0,
    lms_activity_hours REAL DEFAULT 8.0,
    curricular_units_1st_sem_approved INTEGER DEFAULT 5,
    curricular_units_1st_sem_grade REAL DEFAULT 13.0,
    curricular_units_2nd_sem_approved INTEGER DEFAULT 5,
    curricular_units_2nd_sem_grade REAL DEFAULT 13.0,
    -- Optional sensitive attributes (stored separately/auditing only)
    gender TEXT DEFAULT 'Not Disclosed',
    scholarship_holder TEXT DEFAULT 'No',
    tuition_fees_up_to_date TEXT DEFAULT 'Yes',
    debtor TEXT DEFAULT 'No',
    age_at_enrollment INTEGER DEFAULT 20,
    -- Model generated tags
    risk_level TEXT DEFAULT 'Low',
    risk_score REAL DEFAULT 0.15,
    latest_prediction TEXT DEFAULT 'Graduate',
    target TEXT DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS prediction_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    prediction_date TEXT NOT NULL,
    risk_score REAL NOT NULL,
    predicted_outcome TEXT NOT NULL,
    risk_category TEXT NOT NULL,
    top_factors TEXT,
    model_version TEXT DEFAULT 'v1.0.0',
    is_retrospective INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(student_id) REFERENCES students(student_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS interventions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    intervention_date TEXT NOT NULL,
    category TEXT NOT NULL,
    notes TEXT,
    mentor_name TEXT NOT NULL,
    follow_up_date TEXT,
    status TEXT DEFAULT 'Planned',
    follow_up_outcome TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(student_id) REFERENCES students(student_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    recommendation_text TEXT NOT NULL,
    trigger_reason TEXT NOT NULL,
    priority TEXT DEFAULT 'Medium',
    status TEXT DEFAULT 'Pending',
    mentor_notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(student_id) REFERENCES students(student_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS model_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version TEXT NOT NULL,
    model_type TEXT NOT NULL,
    target_type TEXT NOT NULL,
    metrics_json TEXT,
    features_json TEXT,
    training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    sample_size INTEGER DEFAULT 0
);

-- Indices for performance
CREATE INDEX IF NOT EXISTS idx_pred_history_student ON prediction_history(student_id);
CREATE INDEX IF NOT EXISTS idx_interventions_student ON interventions(student_id);
CREATE INDEX IF NOT EXISTS idx_recommendations_student ON recommendations(student_id);
CREATE INDEX IF NOT EXISTS idx_students_risk ON students(risk_level);
