"""
EduGuard AI — Page 4: Batch Prediction
High-throughput inference on CSV/Excel student rosters with validation, risk scoring, and CSV export.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

from config import DEFAULT_RISK_THRESHOLDS, COLORS
from src.train_model import load_trained_model
from src.data_loader import load_file, map_uci_columns
from src.predict import predict_batch_students
from database.db import upsert_students_batch, log_prediction, get_all_students
from utils.export import dataframe_to_csv_bytes
from utils.styles import render_kpi_card, render_risk_badge

def render():
    st.markdown("## 📁 Batch Student Risk Prediction & Longitudinal Comparison")
    st.markdown("Upload student rosters to generate batch inferences, compare trajectory shifts against saved prior records, and export results.")

    model_payload = load_trained_model()
    if model_payload is None:
        st.warning("⚠️ No trained machine learning model found. Please train a model in **Data Management** or load the demo dataset first.")
        return

    pipeline = model_payload["pipeline"]
    label_encoder = model_payload["label_encoder"]
    metadata = model_payload.get("metadata", {})

    st.caption(f"Active Pipeline: **{metadata.get('model_type', 'Classifier')}** | Version: `{metadata.get('model_version', 'latest')}`")

    tab_batch, tab_compare = st.tabs(["📁 Cohort Batch Inference", "🔄 Longitudinal File Comparison"])

    with tab_batch:
        # File Uploader
        uploaded_file = st.file_uploader(
            "Upload Cohort File (CSV or Excel)",
            type=["csv", "xlsx", "xls"],
            key="batch_uploader",
            help="Upload a file containing student records. Delimiters will be auto-detected."
        )

        if uploaded_file is None:
            st.info("💡 **Batch Upload Requirements:**\n\n"
                    "Your file should ideally include student identifiers and key academic fields such as attendance rate, internal marks, and curricular unit completion. "
                    "You can also download a sample template or test with the synthetic demo dataset from the **Dashboard**.")
        else:
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

            if "student_id" not in mapped_df.columns:
                mapped_df["student_id"] = [f"BATCH-{i+1:04d}" for i in range(len(mapped_df))]
                st.info("ℹ️ No 'student_id' column was detected in the file. Temporary IDs (e.g., `BATCH-0001`) have been assigned.")

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
                    return

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
                risk_filter = st.selectbox("Filter Risk Category", ["All", "High", "Moderate", "Low"], key="b_rf")
            with fc2:
                search_id = st.text_input("Search by Student ID", placeholder="e.g. BATCH-0010", key="b_sid")
            with fc3:
                sort_by = st.selectbox("Sort Results By", ["Risk Score (Descending)", "Risk Score (Ascending)", "Student ID"], key="b_sb")

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

            cols_to_display = [
                "student_id", "predicted_outcome", "risk_category", "risk_percentage", "confidence_margin"
            ]
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
                    mime="text/csv",
                    key="dl_batch_btn"
                )

            with exp_col2:
                if st.button("💾 Commit Batch to Institutional Database", key="commit_batch_btn"):
                    records_to_insert = []
                    for _, r in results_df.iterrows():
                        r_dict = r.to_dict()
                        r_dict["risk_level"] = r_dict["risk_category"]
                        r_dict["latest_prediction"] = r_dict["predicted_outcome"]
                        records_to_insert.append(r_dict)

                    inserted = upsert_students_batch(records_to_insert)

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

    # Tab 2: Longitudinal File Comparison
    with tab_compare:
        st.markdown("### 🔄 Compare New Academic Observations with Saved Predictions")
        st.markdown(
            "Upload an updated academic file (e.g. from a subsequent semester or assessment cycle). "
            "EduGuard AI will evaluate new student risk levels and compare them against their previous saved records to isolate meaningful trajectory shifts."
        )

        existing_students = get_all_students()
        if existing_students.empty:
            st.info("No prior student records in the database. Please import student records or load demo data first.")
        else:
            comp_file = st.file_uploader(
                "Upload New Academic Checkpoint File (CSV or Excel)",
                type=["csv", "xlsx", "xls"],
                key="comp_uploader"
            )

            if comp_file:
                try:
                    c_raw = load_file(comp_file, comp_file.name)
                    c_mapped, _ = map_uci_columns(c_raw)
                except Exception as e:
                    st.error(f"Error loading comparison file: {e}")
                    return

                if "student_id" not in c_mapped.columns:
                    st.error("The comparison file must contain a 'student_id' column to match with prior saved student records.")
                    return

                with st.spinner("Calculating new risk scores and matching with historical database records..."):
                    new_preds = predict_batch_students(
                        pipeline=pipeline,
                        label_encoder=label_encoder,
                        df=c_mapped,
                        thresholds=DEFAULT_RISK_THRESHOLDS
                    )

                    # Merge with existing records by student_id
                    prior_map = {
                        row["student_id"]: {
                            "prior_risk_score": float(row.get("risk_score", 0.0)),
                            "prior_risk_level": str(row.get("risk_level", "Low")),
                            "prior_prediction": str(row.get("latest_prediction", "Graduate"))
                        }
                        for _, row in existing_students.iterrows()
                    }

                    matched_rows = []
                    for _, r in new_preds.iterrows():
                        sid = str(r["student_id"])
                        curr_risk = float(r["risk_score"])
                        curr_cat = str(r["risk_category"])
                        curr_pred = str(r["predicted_outcome"])

                        if sid in prior_map:
                            p_info = prior_map[sid]
                            p_risk = p_info["prior_risk_score"]
                            delta = curr_risk - p_risk
                            if delta >= 0.10:
                                shift_status = "Significant Increase 🚨"
                            elif delta <= -0.10:
                                shift_status = "Significant Reduction ✅"
                            else:
                                shift_status = "Stable ⚖️"

                            matched_rows.append({
                                "student_id": sid,
                                "previous_risk": round(p_risk * 100, 1),
                                "current_risk": round(curr_risk * 100, 1),
                                "risk_delta": round(delta * 100, 1),
                                "trajectory_shift": shift_status,
                                "previous_category": p_info["prior_risk_level"],
                                "current_category": curr_cat,
                                "previous_prediction": p_info["prior_prediction"],
                                "current_prediction": curr_pred,
                                "raw_risk": curr_risk,
                                "raw_delta": delta
                            })

                if not matched_rows:
                    st.warning("None of the Student IDs in the uploaded file matched existing records in the database.")
                else:
                    comp_df = pd.DataFrame(matched_rows)
                    st.success(f"Matched **{len(comp_df):,}** students with historical records in the database!")

                    # Summary KPIs
                    inc_count = int((comp_df["raw_delta"] >= 0.10).sum())
                    dec_count = int((comp_df["raw_delta"] <= -0.10).sum())
                    stable_count = len(comp_df) - inc_count - dec_count

                    ck1, ck2, ck3, ck4 = st.columns(4)
                    with ck1:
                        st.markdown(render_kpi_card("Matched Students", f"{len(comp_df):,}", "Evaluated longitudinally", "primary"), unsafe_allow_html=True)
                    with ck2:
                        st.markdown(render_kpi_card("Elevated Risk", f"{inc_count:,}", "Meaningful increase (Δ ≥ +10%)", "danger"), unsafe_allow_html=True)
                    with ck3:
                        st.markdown(render_kpi_card("Reduced Risk", f"{dec_count:,}", "Meaningful reduction (Δ ≤ -10%)", "success"), unsafe_allow_html=True)
                    with ck4:
                        st.markdown(render_kpi_card("Stable Trajectory", f"{stable_count:,}", "Change within ±10%", "teal"), unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # Delta Distribution Scatter / Histogram
                    fig_delta = px.scatter(
                        comp_df,
                        x="previous_risk",
                        y="current_risk",
                        color="trajectory_shift",
                        hover_data=["student_id", "risk_delta"],
                        color_discrete_map={
                            "Significant Increase 🚨": COLORS["danger"],
                            "Significant Reduction ✅": COLORS["success"],
                            "Stable ⚖️": COLORS["navy_primary"]
                        },
                        labels={
                            "previous_risk": "Previous Estimated Risk (%)",
                            "current_risk": "Current Estimated Risk (%)",
                            "trajectory_shift": "Trajectory Shift"
                        }
                    )
                    # Add reference line y = x (no change line)
                    fig_delta.add_shape(
                        type="line", line=dict(dash="dash", color="#94A3B8"),
                        x0=0, x1=100, y0=0, y1=100
                    )
                    fig_delta.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        title=dict(text="<b>Risk Trajectory Shift (Previous vs. Current)</b>", font=dict(size=15, color="#1E3A8A")),
                        font=dict(family="Inter, sans-serif")
                    )
                    st.plotly_chart(fig_delta, use_container_width=True)

                    # Comparison Roster Table
                    st.markdown("#### 📋 Detailed Longitudinal Student Comparison")
                    st.dataframe(
                        comp_df[[
                            "student_id", "previous_risk", "current_risk", "risk_delta",
                            "trajectory_shift", "previous_category", "current_category"
                        ]],
                        use_container_width=True,
                        hide_index=True
                    )

                    # Save checkpoint option
                    st.markdown("---")
                    cp_col1, cp_col2 = st.columns([2, 1])
                    with cp_col1:
                        st.markdown("#### 💾 Save this Evaluation as a New Longitudinal Checkpoint")
                        st.caption("Logs current predictions to database history to track future progress.")
                    with cp_col2:
                        chk_date = st.date_input("Checkpoint Date", value=datetime.today(), key="chk_dt")
                        if st.button("Commit Checkpoint to History", key="btn_commit_chk"):
                            d_str = chk_date.strftime("%Y-%m-%d")
                            for m in matched_rows:
                                log_prediction(
                                    student_id=m["student_id"],
                                    prediction_date=d_str,
                                    risk_score=m["raw_risk"],
                                    predicted_outcome=m["current_prediction"],
                                    risk_category=m["current_category"],
                                    model_version=metadata.get("model_version", "v1.0.0"),
                                    is_retrospective=False
                                )
                            st.success(f"✅ Committed {len(matched_rows):,} updated prediction checkpoints for date {d_str}!")
