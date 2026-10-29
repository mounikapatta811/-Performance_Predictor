"""
EduGuard AI — Configuration & Constants
Student Dropout and Academic Performance Predictor Using Explainable AI
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DATABASE_DIR = BASE_DIR / "database"
DATABASE_PATH = DATABASE_DIR / "eduguard.db"
DEFAULT_MODEL_PATH = MODELS_DIR / "latest_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

# Application Metadata
APP_TITLE = "EduGuard AI — Student Success Intelligence"
APP_SUBTITLE = "Predict early. Understand risk. Support every student."
APP_VERSION = "2.1.0"

# Operational Risk Thresholds (Configurable)
DEFAULT_RISK_THRESHOLDS = {
    "high": 0.65,      # >= 0.65 is High Risk (Immediate Support)
    "moderate": 0.35,  # 0.35 <= risk < 0.65 is Moderate Risk
    "low": 0.35        # < 0.35 is Low Risk
}

# Design System & Theme Colors
COLORS = {
    "navy_primary": "#1E3A8A",
    "navy_dark": "#0F172A",
    "navy_light": "#3B82F6",
    "teal_primary": "#0D9488",
    "teal_accent": "#14B8A6",
    "teal_light": "#CCFBF1",
    "success": "#10B981",
    "success_bg": "#ECFDF5",
    "warning": "#F59E0B",
    "warning_bg": "#FFFBEB",
    "danger": "#EF4444",
    "danger_bg": "#FEF2F2",
    "neutral_bg": "#F8FAFC",
    "card_bg": "#FFFFFF",
    "border": "#E2E8F0",
    "text_dark": "#0F172A",
    "text_muted": "#64748B",
    "white": "#FFFFFF",
}

# Standard Academic Attributes (Predictive Features)
STANDARD_ACADEMIC_FIELDS = {
    "attendance_rate": {
        "label": "Attendance Rate (%)",
        "type": "float",
        "min": 0.0,
        "max": 100.0,
        "default": 82.5,
        "description": "Semester attendance percentage",
        "sensitive": False,
    },
    "internal_assessment_score": {
        "label": "Internal Assessment Score (0-100)",
        "type": "float",
        "min": 0.0,
        "max": 100.0,
        "default": 68.0,
        "description": "Continuous internal assessment average",
        "sensitive": False,
    },
    "assignment_submission_rate": {
        "label": "Assignment Submission Rate (%)",
        "type": "float",
        "min": 0.0,
        "max": 100.0,
        "default": 85.0,
        "description": "Percentage of assignments completed on time",
        "sensitive": False,
    },
    "previous_semester_grade": {
        "label": "Previous Semester Grade (0-20 or 0-100)",
        "type": "float",
        "min": 0.0,
        "max": 100.0,
        "default": 13.5,
        "description": "Prior semester GPA or average (UCI scale 0-20 or standard 0-100)",
        "sensitive": False,
    },
    "failed_subjects_count": {
        "label": "Number of Failed / Curricular Unit Deficits",
        "type": "int",
        "min": 0,
        "max": 20,
        "default": 0,
        "description": "Number of curricular units with incomplete/failed status",
        "sensitive": False,
    },
    "lms_activity_hours": {
        "label": "Weekly LMS Activity Hours",
        "type": "float",
        "min": 0.0,
        "max": 80.0,
        "default": 9.5,
        "description": "Average weekly active hours in online learning portal",
        "sensitive": False,
    },
    "curricular_units_1st_sem_approved": {
        "label": "1st Sem Curricular Units Approved",
        "type": "int",
        "min": 0,
        "max": 25,
        "default": 5,
        "description": "Units approved in semester 1 (UCI alignment)",
        "sensitive": False,
    },
    "curricular_units_1st_sem_grade": {
        "label": "1st Sem Average Grade (0-20)",
        "type": "float",
        "min": 0.0,
        "max": 20.0,
        "default": 13.0,
        "description": "Average grade in semester 1 units",
        "sensitive": False,
    },
    "curricular_units_2nd_sem_approved": {
        "label": "2nd Sem Curricular Units Approved",
        "type": "int",
        "min": 0,
        "max": 25,
        "default": 5,
        "description": "Units approved in semester 2 (UCI alignment)",
        "sensitive": False,
    },
    "curricular_units_2nd_sem_grade": {
        "label": "2nd Sem Average Grade (0-20)",
        "type": "float",
        "min": 0.0,
        "max": 20.0,
        "default": 13.2,
        "description": "Average grade in semester 2 units",
        "sensitive": False,
    },
    "course_department": {
        "label": "Academic Department / Course",
        "type": "categorical",
        "options": [
            "Computer Science & Engineering",
            "Information Technology",
            "Electronics & Communication",
            "Business Administration",
            "Nursing & Health Sciences",
            "Design & Multimedia",
            "General Studies",
        ],
        "default": "Computer Science & Engineering",
        "sensitive": False,
    },
}

# Optional Demographic & Sensitive Attributes (Excluded from default model; available only for auditing)
SENSITIVE_ATTRIBUTES = {
    "gender": {
        "label": "Gender",
        "type": "categorical",
        "options": ["Female", "Male", "Non-Binary / Other", "Not Disclosed"],
        "sensitive": True,
    },
    "scholarship_holder": {
        "label": "Scholarship Holder",
        "type": "categorical",
        "options": ["Yes", "No"],
        "sensitive": True,
    },
    "tuition_fees_up_to_date": {
        "label": "Tuition Fees Up to Date",
        "type": "categorical",
        "options": ["Yes", "No"],
        "sensitive": True,
    },
    "debtor": {
        "label": "Debtor Status",
        "type": "categorical",
        "options": ["No", "Yes"],
        "sensitive": True,
    },
    "age_at_enrollment": {
        "label": "Age at Enrollment",
        "type": "int",
        "min": 16,
        "max": 75,
        "default": 20,
        "sensitive": True,
    },
}

# UCI Dataset Column Names Mapping to Canonical Fields
UCI_COLUMN_MAPPINGS = {
    "Target": "target",
    "Course": "course_id",
    "Daytime/evening attendance\t": "attendance_shift",
    "Daytime/evening attendance": "attendance_shift",
    "Previous qualification (grade)": "previous_qualification_grade",
    "Curricular units 1st sem (credited)": "cu_1st_credited",
    "Curricular units 1st sem (enrolled)": "cu_1st_enrolled",
    "Curricular units 1st sem (evaluations)": "cu_1st_evaluations",
    "Curricular units 1st sem (approved)": "curricular_units_1st_sem_approved",
    "Curricular units 1st sem (grade)": "curricular_units_1st_sem_grade",
    "Curricular units 1st sem (without evaluations)": "cu_1st_no_eval",
    "Curricular units 2nd sem (credited)": "cu_2nd_credited",
    "Curricular units 2nd sem (enrolled)": "cu_2nd_enrolled",
    "Curricular units 2nd sem (evaluations)": "cu_2nd_evaluations",
    "Curricular units 2nd sem (approved)": "curricular_units_2nd_sem_approved",
    "Curricular units 2nd sem (grade)": "curricular_units_2nd_sem_grade",
    "Curricular units 2nd sem (without evaluations)": "cu_2nd_no_eval",
    "Tuition fees up to date": "tuition_fees_up_to_date",
    "Scholarship holder": "scholarship_holder",
    "Age at enrollment": "age_at_enrollment",
    "Gender": "gender",
    "Debtor": "debtor",
}
