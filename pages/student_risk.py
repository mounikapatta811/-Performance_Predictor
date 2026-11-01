"""
EduGuard AI — Page 2: Student Risk Analysis
In-depth cohort risk segmentation, multidimensional filters, and student directory.
"""
import streamlit as st
import pandas as pd
from datetime import datetime
from database.db import get_all_students
from utils.styles import render_kpi_card, render_risk_badge
from utils.visualization import plot_attendance_vs_performance, plot_risk_distribution_donut
from utils.export import dataframe_to_csv_bytes, dataframe_to_excel_bytes

def render():
    st.markdown("## 🔍 Student Risk Analysis & Cohort Segmentation")
    st.markdown("Explore granular academic risk distributions, isolate vulnerable student sub-cohorts, and filter by key risk indicators.")

    students_df = get_all_students()

    if students_df.empty:
        st.warning("⚠️ No student records found. Please load the synthetic demo dataset from the Dashboard or upload a dataset in Data Management.")
        return

    # Filter Sidebar / Controls
    st.markdown("### 🎛️ Filter Cohort")
    f_c1, f_c2, f_c3, f_c4 = st.columns([1, 1, 1.2, 1.2])

    with f_c1:
        risk_options = ["All Categories"] + list(students_df["risk_level"].dropna().unique())
        selected_risk = st.selectbox("Risk Category", risk_options)

    with f_c2:
        dept_options = ["All Departments"] + sorted(list(students_df["department"].dropna().unique()))
        selected_dept = st.selectbox("Department", dept_options)

    with f_c3:
        min_att, max_att = 0.0, 100.0
        if "attendance_rate" in students_df.columns:
            min_att = float(students_df["attendance_rate"].min())
            max_att = float(students_df["attendance_rate"].max())
        att_range = st.slider("Attendance Rate Range (%)", min_value=0.0, max_value=100.0, value=(0.0, 100.0))

    with f_c4:
        search_query = st.text_input("Search Student ID or Name", placeholder="e.g. STU-2024-1010")

    # Apply Filters
    filtered_df = students_df.copy()

    if selected_risk != "All Categories":
        filtered_df = filtered_df[filtered_df["risk_level"] == selected_risk]

    if selected_dept != "All Departments":
        filtered_df = filtered_df[filtered_df["department"] == selected_dept]

    if "attendance_rate" in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df["attendance_rate"] >= att_range[0]) & 
            (filtered_df["attendance_rate"] <= att_range[1])
        ]

    if search_query:
        q = search_query.strip().lower()
        id_match = filtered_df["student_id"].str.lower().str.contains(q, na=False)
        name_match = filtered_df["name"].str.lower().str.contains(q, na=False) if "name" in filtered_df.columns else False
        filtered_df = filtered_df[id_match | name_match]

    # Filtered KPI Summary Cards
    st.markdown(f"#### Filtered Cohort: **{len(filtered_df):,}** of {len(students_df):,} students")
    
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        high_in_filter = int((filtered_df["risk_level"] == "High").sum())
        st.markdown(render_kpi_card("High Risk", f"{high_in_filter}", "In selected filter", "danger"), unsafe_allow_html=True)
    with k2:
        mod_in_filter = int((filtered_df["risk_level"] == "Moderate").sum())
        st.markdown(render_kpi_card("Moderate Risk", f"{mod_in_filter}", "In selected filter", "warning"), unsafe_allow_html=True)
    with k3:
        avg_att = filtered_df["attendance_rate"].mean() if "attendance_rate" in filtered_df.columns and not filtered_df.empty else 0.0
        st.markdown(render_kpi_card("Cohort Attendance", f"{avg_att:.1f}%", "Filtered mean", "teal"), unsafe_allow_html=True)
    with k4:
        avg_score = filtered_df["internal_assessment_score"].mean() if "internal_assessment_score" in filtered_df.columns and not filtered_df.empty else 0.0
        st.markdown(render_kpi_card("Avg. Assessment", f"{avg_score:.1f}", "Passing benchmark: 50.0", "primary"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Detailed Interactive Student Table
    st.markdown("### 📋 Filtered Student Roster")
    
    columns_to_show = [
        "student_id", "name", "department", "attendance_rate",
        "internal_assessment_score", "failed_subjects_count",
        "previous_semester_grade", "risk_score", "risk_level", "latest_prediction"
    ]
    present_cols = [c for c in columns_to_show if c in filtered_df.columns]

    display_table = filtered_df[present_cols].copy()
    if "risk_score" in display_table.columns:
        display_table["risk_score"] = display_table["risk_score"].apply(lambda x: f"{x*100:.1f}%" if pd.notnull(x) else "N/A")
    if "attendance_rate" in display_table.columns:
        display_table["attendance_rate"] = display_table["attendance_rate"].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "N/A")

    st.dataframe(
        display_table,
        use_container_width=True,
        hide_index=True
    )

    # Export options for filtered cohort
    exp_c1, exp_c2, _ = st.columns([1, 1, 2])
    with exp_c1:
        csv_data = dataframe_to_csv_bytes(filtered_df[present_cols])
        st.download_button(
            "📥 Export to CSV",
            data=csv_data,
            file_name=f"eduguard_cohort_filtered_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    with exp_c2:
        excel_data = dataframe_to_excel_bytes(filtered_df[present_cols])
        st.download_button(
            "📊 Export to Excel",
            data=excel_data,
            file_name=f"eduguard_cohort_filtered_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # Visual inspection of the filtered cohort
    if len(filtered_df) > 5:
        st.markdown("### 📈 Filtered Cohort Visualizations")
        c1, c2 = st.columns([1, 1.3])
        with c1:
            rc = filtered_df["risk_level"].value_counts().to_dict()
            st.plotly_chart(plot_risk_distribution_donut(rc), use_container_width=True)
        with c2:
            st.plotly_chart(plot_attendance_vs_performance(filtered_df), use_container_width=True)
