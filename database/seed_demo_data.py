"""
EduGuard AI — Seed Demonstration Database
Populates SQLite database with 350 student records, calibrated risk scores,
longitudinal prediction trajectories, recommendations, and mentor interventions.
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from database.db import (
    init_db,
    upsert_students_batch,
    log_prediction,
    record_intervention,
    add_recommendation,
    get_students_count
)
from src.data_loader import generate_synthetic_academic_dataset, generate_demo_longitudinal_records
from src.train_model import load_trained_model
from src.predict import predict_batch_students
from src.recommendations import generate_action_recommendations

def seed_database():
    print("Initializing database tables...")
    init_db()

    print("Loading or generating 350 synthetic student cohort...")
    df = generate_synthetic_academic_dataset(n_samples=350, random_state=42)

    # Load trained model to get accurate predictions and probabilities
    model_payload = load_trained_model()
    if model_payload:
        pipeline = model_payload["pipeline"]
        label_encoder = model_payload["label_encoder"]
        print("Scoring cohort with active machine learning pipeline...")
        scored_df = predict_batch_students(pipeline, label_encoder, df)
        scored_df["risk_level"] = scored_df["risk_category"]
        scored_df["latest_prediction"] = scored_df["predicted_outcome"]
    else:
        print("Trained model not found, using rule-based scoring...")
        scored_df = df.copy()
        scored_df["risk_level"] = scored_df["target"].map({
            "Dropout": "High", "Enrolled": "Moderate", "Graduate": "Low"
        }).fillna("Low")
        scored_df["risk_score"] = scored_df["target"].map({
            "Dropout": 0.78, "Enrolled": 0.48, "Graduate": 0.16
        }).fillna(0.2)
        scored_df["latest_prediction"] = scored_df["target"]

    students_records = []
    for _, row in scored_df.iterrows():
        s_dict = row.to_dict()
        students_records.append(s_dict)

    print(f"Upserting {len(students_records)} students into database...")
    upsert_students_batch(students_records)

    # Generate longitudinal trajectory history (6 weeks)
    print("Generating longitudinal prediction history...")
    hist_df = generate_demo_longitudinal_records(df, weeks=6)
    for _, h_row in hist_df.iterrows():
        log_prediction(
            student_id=h_row["student_id"],
            prediction_date=h_row["prediction_date"],
            risk_score=float(h_row["risk_score"]),
            predicted_outcome=str(h_row["predicted_outcome"]),
            risk_category=str(h_row["risk_category"]),
            model_version="v2.1.0-prod",
            is_retrospective=True
        )

    # Generate rule recommendations for high & moderate risk students
    print("Generating action recommendations and sample mentor interventions...")
    mentors = ["Dr. Elena Vance", "Prof. Marcus Thorne", "Advisor Sarah Lin", "Dr. David Chen", "Counselor Priya Patel"]
    
    high_mod_students = scored_df[scored_df["risk_level"].isin(["High", "Moderate"])].copy()
    
    # Add recommendations
    rec_count = 0
    for _, s_row in high_mod_students.iterrows():
        recs = generate_action_recommendations(
            s_row.to_dict(),
            risk_category=s_row["risk_level"],
            risk_score=float(s_row["risk_score"])
        )
        for r in recs:
            add_recommendation(
                student_id=s_row["student_id"],
                recommendation_text=r["text"],
                trigger_reason=r["trigger"],
                priority=r.get("priority", "Medium")
            )
            rec_count += 1

    # Add realistic mentor interventions for a subset of students
    intervention_samples = [
        ("Academic Tutoring & Remedial Classes", "Reviewed mid-term calculus exam errors and arranged twice-weekly peer tutoring with CS honor society tutor.", "In Progress", "Student attended first 2 sessions; showed better comprehension of differential equations."),
        ("Attendance Review & Counseling", "Explored commute difficulties. Student transitioned to hybrid transit pass and morning alarm schedule.", "Completed", "Attendance improved by 14% over subsequent 3 weeks."),
        ("Assignment & Time Management Coaching", "Created structured weekly milestone checklist for programming labs and database assignments.", "In Progress", "Submitted Lab 4 on time with 88% marks."),
        ("Wellness, Stress & Mental Health Support", "Discussed exam anxiety and workload stress; connected student to campus wellness center counseling services.", "Completed", "Student reported feeling more centered and engaged in group study."),
        ("Curricular Recovery & Prerequisite Planning", "Devised revised 4-year pacing plan to retake prerequisite linear algebra course in summer semester without delaying graduation.", "Planned", "Pending advisor approval for summer credit overload.")
    ]

    int_count = 0
    sample_students = high_mod_students.sample(n=min(30, len(high_mod_students)), random_state=42)
    today = datetime.now()

    for idx, (_, s_row) in enumerate(sample_students.iterrows()):
        category, notes, status, outcome = intervention_samples[idx % len(intervention_samples)]
        mentor = mentors[idx % len(mentors)]
        int_date = (today - timedelta(days=random.randint(5, 35))).strftime("%Y-%m-%d")
        follow_up = (today + timedelta(days=random.randint(3, 20))).strftime("%Y-%m-%d")

        record_intervention(
            student_id=s_row["student_id"],
            category=category,
            notes=notes,
            mentor_name=mentor,
            follow_up_date=follow_up,
            status=status,
            follow_up_outcome=outcome if status == "Completed" else None,
            intervention_date=int_date
        )
        int_count += 1

    print(f"Seeding completed successfully!")
    print(f"- Students: {get_students_count()}")
    print(f"- Recommendations: {rec_count}")
    print(f"- Interventions: {int_count}")

if __name__ == "__main__":
    seed_database()
