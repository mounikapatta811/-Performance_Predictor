"""
EduGuard AI — Page 4: Batch Prediction
High-throughput inference on CSV/Excel student rosters with validation, risk scoring, and CSV export.
"""
import streamlit as st
import pandas as pd
from datetime import datetime

from config import DEFAULT_RISK_THRESHOLDS
from src.train_model import load_trained_model
from src.data_loader import load_file, map_uci_columns
from src.predict import predict_batch_students
from database.db import upsert_students_batch, log_prediction
from utils.export import dataframe_to_csv_bytes
from utils.styles import render_kpi_card, render_risk_badge

def render():
    st.markdown("## 📁 Batch Student Risk Prediction")
    st.markdown("Upload a roster of student records to generate cohort-wide predictions, calibrate risk scores, and export results.")

    model_payload = load_trained_model()
    if model_payload is None:
        st.warning("⚠️ No trained machine learning model found. Please train a model in **Data Management** or load the demo dataset first.")
        return

    pipeline = model_payload["pipeline"]
    label_encoder = model_payload["label_encoder"]
    metadata = model_payload.get("metadata", {})

    st.caption(f"Active Pipeline: **{metadata.get('model_type', 'Classifier')}** | Version: `{metadata.get('model_version', 'latest')}`")

    # File Uploader
    uploaded_file = st.file_uploader(
        "Upload Cohort File (CSV or Excel)",
        type=["csv", "xlsx", "xls"],
        help="Upload a file containing student records. Delimiters will be auto-detected."
    )

    if uploaded_file is None:
        st.info("💡 **Batch Upload Requirements:**\n\n"
                "Your file should ideally include student identifiers and key academic fields such as attendance rate, internal marks, and curricular unit completion. "
                "You can also download a sample template or test with the synthetic demo dataset from the **Dashboard**.")
        return

    # Load and map
    try:
        raw_df = load_file(uploaded_file, uploaded_file.name)
        mapped_df, applied_maps = map_uci_columns(raw_df)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return

    st.success(f"Successfully loaded **{len(mapped_df):,}** records from `{uploaded_file.name}`.")
    if applied_maps:
        with st.expander("Detected Column Mappings"):
            st.json(applied_maps)

    # Check for required identifiers
    if "student_id" not in mapped_df.columns:
        # Create synthetic ID column for tracking
        mapped_df["student_id"] = [f"BATCH-{i+1:04d}" for i in range(len(mapped_df))]
        st.info("ℹ️ No 'student_id' column was detected in the file. Temporary IDs (e.g., `BATCH-0001`) have been assigned.")

    # Run Batch Inference
    with st.spinner("Processing batch through preprocessing pipeline and model..."):
        try:
            results_df = predict_batch_students(
                pipeline=pipeline,
                label_encoder=label_encoder,
                df=mapped_df,
                thresholds=DEFAULT_RISK_THRESHOLDS
            )
        except Exception as e:
            st.error(f"Failed to generate predictions: {e}")
            st.info("Check that the uploaded file contains the necessary numerical and categorical features expected by the model.")
            return

    # Batch Analytics Summary
    st.markdown("### 📊 Batch Cohort Risk Summary")
    high_count = int((results_df["risk_category"] == "High").sum())
    mod_count = int((results_df["risk_category"] == "Moderate").sum())
    low_count = int((results_df["risk_category"] == "Low").sum())
    avg_batch_risk = float(results_df["risk_score"].mean())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(render_kpi_card("Total Evaluated", f"{len(results_df):,}", "Batch records", "primary"), unsafe_allow_html=True)
    with c2:
        st.markdown(render_kpi_card("High Risk", f"{high_count:,}", f"{(high_count/len(results_df))*100:.1f}% of cohort", "danger"), unsafe_allow_html=True)
    with c3:
        st.markdown(render_kpi_card("Moderate Risk", f"{mod_count:,}", f"{(mod_count/len(results_df))*100:.1f}% of cohort", "warning"), unsafe_allow_html=True)
    with c4:
        st.markdown(render_kpi_card("Average Risk", f"{avg_batch_risk*100:.1f}%", "Cohort Risk Score", "teal"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Filters & Search
    st.markdown("### 🔍 Search & Filter Results")
    fc1, fc2, fc3 = st.columns([1, 1.2, 1.2])

    with fc1:
        risk_filter = st.selectbox("Filter Risk Category", ["All", "High", "Moderate", "Low"])
    with fc2:
        search_id = st.text_input("Search by Student ID", placeholder="e.g. BATCH-0010")
    with fc3:
        sort_by = st.selectbox("Sort Results By", ["Risk Score (Descending)", "Risk Score (Ascending)", "Student ID"])

    display_df = results_df.copy()

    if risk_filter != "All":
        display_df = display_df[display_df["risk_category"] == risk_filter]

    if search_id:
        display_df = display_df[display_df["student_id"].str.lower().str.contains(search_id.strip().lower(), na=False)]

    if sort_by == "Risk Score (Descending)":
        display_df = display_df.sort_values("risk_score", ascending=False)
    elif sort_by == "Risk Score (Ascending)":
        display_df = display_df.sort_values("risk_score", ascending=True)
    elif sort_by == "Student ID":
        display_df = display_df.sort_values("student_id")

    # Display Table
    cols_to_display = [
        "student_id", "predicted_outcome", "risk_category", "risk_percentage", "confidence_margin"
    ]
    # Add key academic context if present
    for extra in ["attendance_rate", "internal_assessment_score", "failed_subjects_count", "department"]:
        if extra in display_df.columns:
            cols_to_display.append(extra)

    show_cols = [c for c in cols_to_display if c in display_df.columns]
    st.dataframe(display_df[show_cols], use_container_width=True, hide_index=True)

    # Export & Persistence Actions
    st.markdown("### 💾 Export & Save Batch Results")
    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        csv_bytes = dataframe_to_csv_bytes(results_df)
        st.download_button(
            label="📥 Download Batch Predictions (CSV)",
            data=csv_bytes,
            file_name=f"eduguard_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )

    with exp_col2:
        if st.button("💾 Commit Batch to Institutional Database"):
            records_to_insert = []
            for _, r in results_df.iterrows():
                r_dict = r.to_dict()
                r_dict["risk_level"] = r_dict["risk_category"]
                r_dict["latest_prediction"] = r_dict["predicted_outcome"]
                records_to_insert.append(r_dict)

            inserted = upsert_students_batch(records_to_insert)

            # Log to longitudinal history
            today_str = datetime.now().strftime("%Y-%m-%d")
            for _, r in results_df.iterrows():
                log_prediction(
                    student_id=r["student_id"],
                    prediction_date=today_str,
                    risk_score=r["risk_score"],
                    predicted_outcome=r["predicted_outcome"],
                    risk_category=r["risk_category"],
                    model_version=metadata.get("model_version", "v1.0.0"),
                    is_retrospective=False
                )

            st.success(f"✅ Successfully persisted {inserted:,} student records and prediction logs into SQLite database!")
