"""
EduGuard AI — Page 9: Data Management & Model Training Studio
Dataset ingestion, column mapping, data quality auditing, model training, and privacy controls.
"""
import streamlit as st
import pandas as pd
from datetime import datetime

from config import STANDARD_ACADEMIC_FIELDS, SENSITIVE_ATTRIBUTES
from src.data_loader import load_file, map_uci_columns, generate_synthetic_academic_dataset, generate_demo_longitudinal_records
from utils.validation import generate_data_quality_report, clean_academic_dataset, separate_features_and_target
from src.train_model import train_academic_model, load_trained_model
from database.db import (
    upsert_students_batch,
    clear_all_student_data,
    save_model_metadata,
    log_prediction
)
from utils.export import dataframe_to_csv_bytes

def render():
    st.markdown("## 💾 Student Data Management & Model Training Studio")
    st.markdown("Upload academic rosters, audit data quality, configure training pipelines, and manage institutional data retention.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📤 Dataset Upload & Quality Audit",
        "⚙️ Train Predictive Model",
        "🧪 Synthetic Demo Mode",
        "🔒 Privacy & Data Deletion"
    ])

    # Tab 1: Dataset Upload & Quality Audit
    with tab1:
        st.markdown("### Upload Academic Roster File")
        uploaded = st.file_uploader(
            "Upload CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            key="dm_file_uploader"
        )

        if uploaded:
            try:
                raw_df = load_file(uploaded, uploaded.name)
                mapped_df, mappings = map_uci_columns(raw_df)
                st.session_state["uploaded_academic_df"] = mapped_df
                st.success(f"Loaded **{len(mapped_df):,}** rows and **{len(mapped_df.columns):,}** columns from `{uploaded.name}`.")
            except Exception as e:
                st.error(f"Error loading file: {e}")
                return

        # If data is present in session
        if "uploaded_academic_df" in st.session_state:
            df = st.session_state["uploaded_academic_df"]

            st.markdown("#### Data Preview (First 5 Rows)")
            st.dataframe(df.head(), use_container_width=True)

            # Data Quality Report
            st.markdown("#### 🩺 Data Quality Diagnostic Report")
            dq_report = generate_data_quality_report(df)

            q_c1, q_c2, q_c3, q_c4 = st.columns(4)
            with q_c1:
                st.metric("Total Records", f"{dq_report['total_rows']:,}")
            with q_c2:
                st.metric("Missing Cells", f"{dq_report['total_missing_cells']:,}")
            with q_c3:
                st.metric("Duplicate Rows", f"{dq_report['duplicate_rows']:,}")
            with q_c4:
                health_status = "Healthy" if dq_report["is_healthy"] else "Anomalies Detected"
                st.metric("Data Health", health_status)

            if dq_report["anomalies"]:
                with st.expander("⚠️ Detected Value Anomalies"):
                    for a in dq_report["anomalies"]:
                        st.warning(a)

            if dq_report["missing_pcts"]:
                with st.expander("Columns with Missing Values"):
                    st.json(dq_report["missing_pcts"])

            # Clean Data Options
            st.markdown("#### 🧹 Clean & Normalize Dataset")
            cl_c1, cl_c2 = st.columns(2)
            with cl_c1:
                missing_strategy = st.radio("Missing Value Strategy", ["Impute with column median/mode", "Drop incomplete rows"])
            with cl_c2:
                dedup = st.checkbox("Remove duplicate rows", value=True)

            if st.button("Apply Cleaning Transformations"):
                strat = "impute" if "Impute" in missing_strategy else "drop"
                clean_df, audit = clean_academic_dataset(df, handle_missing=strat, remove_duplicates=dedup)
                st.session_state["uploaded_academic_df"] = clean_df
                st.success("Transformations applied successfully!")
                with st.expander("Transformation Audit Log"):
                    for log in audit:
                        st.write(f"• {log}")

            # Export Cleaned Data
            clean_bytes = dataframe_to_csv_bytes(df)
            st.download_button(
                "📥 Download Cleaned Dataset (CSV)",
                data=clean_bytes,
                file_name=f"cleaned_student_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

    # Tab 2: Train Predictive Model
    with tab2:
        st.markdown("### ⚙️ Configure & Train Machine Learning Model")
        
        train_df = st.session_state.get("uploaded_academic_df")
        if train_df is None:
            st.info("Please upload a dataset in Tab 1, or generate synthetic demo records in Tab 3 before training.")
        else:
            with st.form("train_model_form"):
                col1, col2 = st.columns(2)
                with col1:
                    model_type = st.selectbox(
                        "Model Algorithm",
                        ["Random Forest", "Logistic Regression", "XGBoost"]
                    )
                    
                    target_candidates = [c for c in train_df.columns if c.lower() in ["target", "status", "dropout", "outcome"]] or list(train_df.columns)
                    target_col = st.selectbox("Target Classification Column", target_candidates)

                    exclude_demographics = st.checkbox(
                        "Exclude Sensitive Demographic Attributes (Fairness by Design)",
                        value=True,
                        help="Keeps gender, scholarship, and debt status out of predictive training features."
                    )

                with col2:
                    test_split = st.slider("Holdout Test Split (%)", 15, 35, 25) / 100.0
                    cv_folds = st.selectbox("Stratified Cross-Validation Folds", [3, 5, 10], index=1)
                    random_seed = st.number_input("Reproducible Random Seed", value=42, step=1)

                train_btn = st.form_submit_button("🚀 Train Model & Preprocessing Pipeline")

            if train_btn:
                try:
                    with st.spinner(f"Fitting {model_type} pipeline with leak-free preprocessing..."):
                        X, y, num_feats, cat_feats = separate_features_and_target(
                            df=train_df,
                            target_col=target_col,
                            exclude_sensitive=exclude_demographics
                        )

                        train_results = train_academic_model(
                            X=X,
                            y=y,
                            numeric_features=num_feats,
                            categorical_features=cat_feats,
                            model_type=model_type,
                            test_size=test_split,
                            random_state=int(random_seed),
                            cv_folds=cv_folds
                        )

                        # Save metadata into database
                        meta = train_results["metadata"]
                        save_model_metadata(
                            model_version=meta["model_version"],
                            model_type=meta["model_type"],
                            target_type=target_col,
                            metrics=meta["test_metrics"],
                            features=meta["numeric_features"] + meta["categorical_features"],
                            sample_size=meta["training_samples"] + meta["test_samples"]
                        )

                    st.success(f"🎉 Model **{model_type}** trained and saved successfully!")
                    
                    # Quick Metrics Preview
                    tm = meta["test_metrics"]
                    st.markdown("#### Test Set Performance Summary:")
                    mc1, mc2, mc3 = st.columns(3)
                    with mc1:
                        st.metric("Test Balanced Accuracy", f"{tm['balanced_accuracy']*100:.1f}%")
                    with mc2:
                        st.metric("Macro F1-Score", f"{tm['f1_macro']:.3f}")
                    with mc3:
                        st.metric(f"Stratified {cv_folds}-Fold CV F1", f"{meta['cv_f1_macro_mean']:.3f}")

                except Exception as e:
                    st.error(f"Training failed: {e}")

    # Tab 3: Synthetic Demo Mode
    with tab3:
        st.markdown("### 🧪 Synthetic Demo Data Generator")
        st.markdown(
            "Generate clearly labeled, artificial student records with realistic academic distributions to test all features of the application without loading private student data."
        )

        n_samples = st.slider("Number of Synthetic Student Records", 100, 1000, 350, step=50)

        if st.button("Generate & Import Synthetic Demo Cohort"):
            with st.spinner("Generating synthetic student cohort..."):
                demo_data = generate_synthetic_academic_dataset(n_samples=n_samples, random_state=42)
                st.session_state["uploaded_academic_df"] = demo_data
                
                # Import into SQLite
                demo_records = []
                for _, row in demo_data.iterrows():
                    target = row["target"]
                    risk_lvl = "High" if target == "Dropout" else ("Moderate" if target == "Enrolled" else "Low")
                    r_score = 0.78 if target == "Dropout" else (0.48 if target == "Enrolled" else 0.16)
                    d_dict = row.to_dict()
                    d_dict["risk_level"] = risk_lvl
                    d_dict["risk_score"] = r_score
                    d_dict["latest_prediction"] = target
                    demo_records.append(d_dict)

                upsert_students_batch(demo_records)

                # Generate weekly historical trends
                hist_demo = generate_demo_longitudinal_records(demo_data, weeks=6)
                for _, h in hist_demo.iterrows():
                    log_prediction(
                        student_id=h["student_id"],
                        prediction_date=h["prediction_date"],
                        risk_score=h["risk_score"],
                        predicted_outcome=h["predicted_outcome"],
                        risk_category=h["risk_category"],
                        model_version="v1.0.0-demo",
                        is_retrospective=True
                    )

            st.success(f"✅ Generated and loaded {n_samples} synthetic student records and 6 weeks of longitudinal history!")

    # Tab 4: Privacy & Data Deletion
    with tab4:
        st.markdown("### 🔒 Privacy Controls & Data Deletion")
        st.markdown("""
        To comply with institutional data governance, privacy policies, and GDPR/FERPA guidelines,
        authorized administrators can permanently purge all stored student profiles, prediction history, and mentor notes.
        """)

        st.warning("⚠️ **Caution:** This action cannot be undone. All database records will be erased.")

        confirm_text = st.text_input("Type 'DELETE ALL DATA' to confirm purge:")
        if st.button("Permanently Purge All Student Records", type="primary"):
            if confirm_text.strip() == "DELETE ALL DATA":
                clear_all_student_data()
                st.session_state.pop("uploaded_academic_df", None)
                st.success("✅ All student records, prediction history, and intervention notes successfully deleted.")
            else:
                st.error("Confirmation text did not match. Deletion aborted.")
