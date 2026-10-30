"""
EduGuard AI — Data Loader & Synthetic Demo Generator
Loads UCI dataset or custom academic files, maps columns, and generates realistic synthetic demo data.
"""
import io
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from datetime import datetime, timedelta

from config import (
    STANDARD_ACADEMIC_FIELDS,
    SENSITIVE_ATTRIBUTES,
    UCI_COLUMN_MAPPINGS,
    DATA_DIR
)
from utils.validation import validate_file_size

def generate_synthetic_academic_dataset(
    n_samples: int = 350,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate a high-quality, realistic synthetic academic dataset for demo and testing.
    All records are clearly generated with realistic academic distributions.
    """
    np.random.seed(random_state)
    
    departments = [
        "Computer Science & Engineering",
        "Information Technology",
        "Electronics & Communication",
        "Business Administration",
        "Nursing & Health Sciences",
        "Design & Multimedia",
    ]
    
    # Latent student academic capability index [-2.5, 2.5]
    latent_ability = np.random.normal(0.0, 1.0, n_samples)
    
    # Academic indicators correlated with latent capability
    attendance = np.clip(80 + latent_ability * 12 + np.random.normal(0, 5, n_samples), 35.0, 100.0)
    internal_assessment = np.clip(68 + latent_ability * 14 + np.random.normal(0, 6, n_samples), 25.0, 98.0)
    assignment_rate = np.clip(82 + latent_ability * 12 + np.random.normal(0, 6, n_samples), 30.0, 100.0)
    
    # Scale previous grade (0-20 scale standard in UCI, or 0-100)
    previous_grade = np.clip(13.2 + latent_ability * 2.8 + np.random.normal(0, 1.2, n_samples), 5.0, 19.5)
    
    # Failed units: inverse relationship with ability
    failed_units_raw = np.clip(np.round(np.maximum(0, -latent_ability * 2.0 + np.random.poisson(0.8, n_samples))), 0, 10).astype(int)
    
    lms_hours = np.clip(9.5 + latent_ability * 3.5 + np.random.normal(0, 2.0, n_samples), 0.5, 35.0)
    
    cu_1st_approved = np.clip(np.round(5.5 + latent_ability * 1.5 - failed_units_raw * 0.4), 0, 8).astype(int)
    cu_1st_grade = np.clip(13.0 + latent_ability * 2.5 + np.random.normal(0, 1.0, n_samples), 0.0, 20.0)
    cu_1st_grade = np.where(cu_1st_approved == 0, 0.0, cu_1st_grade)
    
    cu_2nd_approved = np.clip(np.round(cu_1st_approved + np.random.normal(0, 0.8, n_samples)), 0, 8).astype(int)
    cu_2nd_grade = np.clip(13.1 + latent_ability * 2.6 + np.random.normal(0, 1.0, n_samples), 0.0, 20.0)
    cu_2nd_grade = np.where(cu_2nd_approved == 0, 0.0, cu_2nd_grade)

    # Demographic & Sensitive Attributes (for responsible AI / fairness auditing)
    genders = np.random.choice(["Female", "Male", "Non-Binary / Other"], size=n_samples, p=[0.48, 0.49, 0.03])
    scholarship = np.random.choice(["Yes", "No"], size=n_samples, p=[0.30, 0.70])
    tuition_up_to_date = np.random.choice(["Yes", "No"], size=n_samples, p=[0.88, 0.12])
    debtor = np.where(tuition_up_to_date == "No", np.random.choice(["Yes", "No"], size=n_samples, p=[0.85, 0.15]), "No")
    age = np.clip(np.round(np.random.gamma(shape=9, scale=2.5, size=n_samples)), 18, 55).astype(int)
    
    dept_choices = np.random.choice(departments, size=n_samples)

    # Determine 3-class target based on academic standing
    # Target: Dropout, Enrolled, Graduate
    risk_score_composite = (
        (100 - attendance) * 0.25 +
        (100 - internal_assessment) * 0.25 +
        (failed_units_raw * 8) +
        (16 - cu_2nd_grade) * 2.5
    ) / 100.0
    
    target = []
    for r in risk_score_composite:
        if r > 0.65:
            target.append("Dropout")
        elif r > 0.40:
            target.append("Enrolled")
        else:
            target.append("Graduate")

    student_ids = [f"STU-2024-{1001 + i}" for i in range(n_samples)]
    names = [f"Synthetic Student #{1001 + i}" for i in range(n_samples)]

    df = pd.DataFrame({
        "student_id": student_ids,
        "name": names,
        "department": dept_choices,
        "attendance_rate": np.round(attendance, 1),
        "internal_assessment_score": np.round(internal_assessment, 1),
        "assignment_submission_rate": np.round(assignment_rate, 1),
        "previous_semester_grade": np.round(previous_grade, 1),
        "failed_subjects_count": failed_units_raw,
        "lms_activity_hours": np.round(lms_hours, 1),
        "curricular_units_1st_sem_approved": cu_1st_approved,
        "curricular_units_1st_sem_grade": np.round(cu_1st_grade, 1),
        "curricular_units_2nd_sem_approved": cu_2nd_approved,
        "curricular_units_2nd_sem_grade": np.round(cu_2nd_grade, 1),
        "gender": genders,
        "scholarship_holder": scholarship,
        "tuition_fees_up_to_date": tuition_up_to_date,
        "debtor": debtor,
        "age_at_enrollment": age,
        "target": target
    })

    return df

def generate_demo_longitudinal_records(
    students_df: pd.DataFrame,
    weeks: int = 6
) -> pd.DataFrame:
    """
    Generate longitudinal weekly tracking records for demo students to visualize trends over time.
    """
    history_records = []
    base_date = datetime.now() - timedelta(weeks=weeks)
    
    for _, student in students_df.iterrows():
        sid = student["student_id"]
        true_target = student["target"]
        
        # Base risk
        if true_target == "Dropout":
            start_risk = np.random.uniform(0.55, 0.75)
            trend = np.random.uniform(0.01, 0.04)  # deteriorating
        elif true_target == "Enrolled":
            start_risk = np.random.uniform(0.35, 0.55)
            trend = np.random.uniform(-0.01, 0.02)  # fluctuating
        else:
            start_risk = np.random.uniform(0.10, 0.30)
            trend = np.random.uniform(-0.02, 0.01)  # stable or improving
            
        current_risk = start_risk
        for w in range(weeks):
            pred_date = (base_date + timedelta(weeks=w)).strftime("%Y-%m-%d")
            noise = np.random.normal(0, 0.03)
            current_risk = float(np.clip(current_risk + trend + noise, 0.05, 0.98))
            
            if current_risk >= 0.65:
                cat = "High"
                pred_out = "Dropout"
            elif current_risk >= 0.35:
                cat = "Moderate"
                pred_out = "Enrolled"
            else:
                cat = "Low"
                pred_out = "Graduate"
                
            history_records.append({
                "student_id": sid,
                "prediction_date": pred_date,
                "risk_score": round(current_risk, 3),
                "predicted_outcome": pred_out,
                "risk_category": cat,
                "model_version": "v1.0.0-demo",
                "is_retrospective": 1
            })
            
    return pd.DataFrame(history_records)

def load_file(file_obj, filename: str) -> pd.DataFrame:
    """
    Load a CSV or Excel file safely, auto-detecting delimiters for CSV.
    """
    validate_file_size(file_obj)
    if filename.endswith(".csv"):
        # Read raw bytes to detect delimiter
        content = file_obj.read()
        if isinstance(content, str):
            content_str = content
        else:
            content_str = content.decode("utf-8", errors="replace")
            
        first_line = content_str.split("\n")[0] if "\n" in content_str else content_str
        delimiter = ";" if ";" in first_line and first_line.count(";") > first_line.count(",") else ","
        
        file_obj.seek(0)
        df = pd.read_csv(file_obj, sep=delimiter)
    elif filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_obj)
    else:
        raise ValueError("Unsupported file format. Please upload a CSV (.csv) or Excel (.xlsx) file.")
        
    return df

def map_uci_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Detect and map UCI dataset columns or similar standard names into canonical fields.
    Returns (mapped_df, mappings_used).
    """
    mapped_df = df.copy()
    applied_mappings = {}
    
    # Strip whitespace from column names
    mapped_df.columns = [c.strip() for c in mapped_df.columns]
    
    for orig_col in mapped_df.columns:
        if orig_col in UCI_COLUMN_MAPPINGS:
            canonical = UCI_COLUMN_MAPPINGS[orig_col]
            applied_mappings[orig_col] = canonical
            mapped_df.rename(columns={orig_col: canonical}, inplace=True)
            
    # Normalize target column name
    if "Target" in mapped_df.columns:
        mapped_df.rename(columns={"Target": "target"}, inplace=True)
        applied_mappings["Target"] = "target"
    elif "target" not in mapped_df.columns:
        for c in mapped_df.columns:
            if c.lower() in ["target", "status", "outcome", "dropout", "result"]:
                mapped_df.rename(columns={c: "target"}, inplace=True)
                applied_mappings[c] = "target"
                break

    return mapped_df, applied_mappings
