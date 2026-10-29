"""
EduGuard AI — Page 1: Dashboard
Overview of institutional student risk indicators, cohort KPIs, and actionable trends.
"""
import streamlit as st
import pandas as pd
from datetime import datetime

from config import DEFAULT_RISK_THRESHOLDS
from database.db import (
    get_all_students,
    get_all_interventions,
    get_all_prediction_history,
    upsert_students_batch,
    log_prediction
)
from src.data_loader import generate_synthetic_academic_dataset, generate_demo_longitudinal_records
from utils.styles import render_kpi_card, render_risk_badge
from utils.visualization import (
    plot_risk_distribution_donut,
    plot_attendance_vs_performance,
    plot_risk_by_department,
    plot_longitudinal_risk_trend
)

def render():
    st.markdown("## 📊 Executive Academic Dashboard")
    st.markdown("Real-time intelligence on student risk profiles, academic trends, and active mentor interventions.")

    students_df = get_all_students()
    interventions_df = get_all_interventions()
    history_df = get_all_prediction_history()

    # Onboarding view if no data in database
    if students_df.empty:
        st.markdown("""
        <div class="onboarding-hero">
            <h2>Welcome to EduGuard AI</h2>
            <p>EduGuard AI empowers higher education institutions to predict academic risk early, understand key contributing factors with Explainable AI, and orchestrate supportive mentor interventions.</p>
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.info("💡 **Demo Exploration Mode**\n\nLoad a clearly labeled synthetic cohort of 350 student records with longitudinal weekly tracking to test all features immediately.")
            if st.button("🚀 Load Synthetic Demo Dataset", key="btn_load_demo_dash"):
                with st.spinner("Generating synthetic student cohort and historical predictions..."):
                    demo_df = generate_synthetic_academic_dataset(n_samples=350, random_state=42)
                    
                    # Compute demo risk level
                    demo_students = []
                    for _, row in demo_df.iterrows():
                        target = row["target"]
                        if target == "Dropout":
                            risk_lvl = "High"
                            r_score = 0.76
                        elif target == "Enrolled":
                            risk_lvl = "Moderate"
                            r_score = 0.48
                        else:
                            risk_lvl = "Low"
                            r_score = 0.18

                        s_dict = row.to_dict()
                        s_dict["risk_level"] = risk_lvl
                        s_dict["risk_score"] = r_score
                        s_dict["latest_prediction"] = target
                        demo_students.append(s_dict)

                    upsert_students_batch(demo_students)

                    # Add longitudinal demo history
                    hist_df = generate_demo_longitudinal_records(demo_df, weeks=6)
                    for _, h_row in hist_df.iterrows():
                        log_prediction(
                            student_id=h_row["student_id"],
                            prediction_date=h_row["prediction_date"],
                            risk_score=h_row["risk_score"],
                            predicted_outcome=h_row["predicted_outcome"],
                            risk_category=h_row["risk_category"],
                            model_version="v1.0.0-demo",
                            is_retrospective=True
                        )

                st.success("✅ Demo student cohort and longitudinal records loaded successfully!")
                st.rerun()

        with col_b:
            st.info("📂 **Upload Institutional Dataset**\n\nUpload an official UCI dataset or your institution's CSV/Excel student record file to configure a custom model.")
            st.markdown("Navigate to the **Data Management** page to upload datasets, validate columns, and train models.")

        return

    # Real calculated metrics from loaded data
    total_students = len(students_df)
    high_risk_count = int((students_df["risk_level"] == "High").sum())
    mod_risk_count = int((students_df["risk_level"] == "Moderate").sum())
    low_risk_count = int((students_df["risk_level"] == "Low").sum())
    
    avg_attendance = float(students_df["attendance_rate"].mean()) if "attendance_rate" in students_df.columns else None
    att_display = f"{avg_attendance:.1f}%" if avg_attendance is not None else "N/A"
    
    total_interventions = len(interventions_df)

    # Top KPI Cards
    kpi_cols = st.columns(6)
    with kpi_cols[0]:
        st.markdown(render_kpi_card("Total Analyzed", f"{total_students:,}", "Active Student Cohort", "primary"), unsafe_allow_html=True)
    with kpi_cols[1]:
        st.markdown(render_kpi_card("High Risk", f"{high_risk_count:,}", "Immediate Support Needed", "danger"), unsafe_allow_html=True)
    with kpi_cols[2]:
        st.markdown(render_kpi_card("Moderate Risk", f"{mod_risk_count:,}", "Requires Monitoring", "warning"), unsafe_allow_html=True)
    with kpi_cols[3]:
        st.markdown(render_kpi_card("Low Risk", f"{low_risk_count:,}", "Satisfactory Progress", "success"), unsafe_allow_html=True)
    with kpi_cols[4]:
        st.markdown(render_kpi_card("Avg. Attendance", att_display, "Cohort Attendance", "teal"), unsafe_allow_html=True)
    with kpi_cols[5]:
        st.markdown(render_kpi_card("Interventions", f"{total_interventions:,}", "Recorded Mentor Actions", "primary"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts Row 1: Donut Distribution & Longitudinal Risk Trend
    row1_c1, row1_c2 = st.columns([1, 1.4])
    with row1_c1:
        risk_counts = {
            "High": high_risk_count,
            "Moderate": mod_risk_count,
            "Low": low_risk_count
        }
        st.plotly_chart(plot_risk_distribution_donut(risk_counts), use_container_width=True)

    with row1_c2:
        st.plotly_chart(plot_longitudinal_risk_trend(history_df), use_container_width=True)

    # Charts Row 2: Attendance vs Academic Performance & Department breakdown
    row2_c1, row2_c2 = st.columns([1.2, 1])
    with row2_c1:
        st.plotly_chart(plot_attendance_vs_performance(students_df), use_container_width=True)
    with row2_c2:
        st.plotly_chart(plot_risk_by_department(students_df), use_container_width=True)

    # Priority Mentor Review Table
    st.markdown("### 🚨 Priority Students Requiring Mentor Review")
    st.caption("Students identified with high estimated academic risk score requiring proactive outreach.")

    high_risk_students = students_df[students_df["risk_level"] == "High"].sort_values("risk_score", ascending=False).head(10)
    
    if not high_risk_students.empty:
        display_cols = [
            "student_id", "department", "attendance_rate",
            "internal_assessment_score", "failed_subjects_count", "risk_score", "risk_level"
        ]
        available_cols = [c for c in display_cols if c in high_risk_students.columns]
        
        formatted_df = high_risk_students[available_cols].copy()
        if "risk_score" in formatted_df.columns:
            formatted_df["risk_score"] = formatted_df["risk_score"].apply(lambda x: f"{x*100:.1f}%")
        if "attendance_rate" in formatted_df.columns:
            formatted_df["attendance_rate"] = formatted_df["attendance_rate"].apply(lambda x: f"{x:.1f}%")

        st.dataframe(formatted_df, use_container_width=True, hide_index=True)
    else:
        st.success("No students currently flagged in the High Risk category.")

    # Recent Intervention & Prediction Activity Feed
    st.markdown("### ⏱️ Recent Institutional Activity")
    act_c1, act_c2 = st.columns(2)
    with act_c1:
        st.markdown("#### Recent Mentor Interventions")
        if not interventions_df.empty:
            recent_ints = interventions_df.head(5)[["student_id", "intervention_date", "category", "mentor_name", "status"]]
            st.dataframe(recent_ints, use_container_width=True, hide_index=True)
        else:
            st.info("No mentor interventions logged yet. Interventions can be recorded in the Intervention Management module.")

    with act_c2:
        st.markdown("#### Recent Prediction Evaluations")
        if not history_df.empty:
            recent_preds = history_df.tail(5)[["student_id", "prediction_date", "predicted_outcome", "risk_category", "risk_score"]]
            st.dataframe(recent_preds, use_container_width=True, hide_index=True)
        else:
            st.info("No prediction history recorded.")
