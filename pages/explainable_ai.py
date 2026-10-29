"""
EduGuard AI — Page 5: Explainable AI (XAI)
Transparent model explainability, SHAP global feature importances, and individual student driver analysis.
"""
import streamlit as st
import pandas as pd

from src.train_model import load_trained_model
from src.explainability import compute_global_importance, explain_single_prediction
from database.db import get_all_students, get_student_by_id
from utils.visualization import plot_feature_importance, plot_local_feature_impacts
from utils.styles import render_kpi_card

def render():
    st.markdown("## 🧠 Explainable AI (XAI) & Factor Attribution")
    st.markdown("Inspect global academic risk predictors and examine the specific drivers influencing individual student risk scores.")

    model_payload = load_trained_model()
    if model_payload is None:
        st.warning("⚠️ No trained machine learning model found. Please train a model in **Data Management** or load demo data.")
        return

    pipeline = model_payload["pipeline"]
    label_encoder = model_payload["label_encoder"]
    metadata = model_payload.get("metadata", {})

    st.markdown(
        "> ⚠️ **Predictive Association vs. Causation Principle:**\n"
        "> Model explanations reflect statistical associations detected across historical cohort data. "
        "> A high feature impact does **not** prove that a factor directly caused a student's difficulty. "
        "> Always use these indicators as diagnostic conversation starters, not definitive causal judgments."
    )

    tab1, tab2 = st.tabs(["🌐 Global Feature Importance", "👤 Individual Student Attribution"])

    with tab1:
        st.markdown("### Institutional Risk Factors (Global Importance)")
        st.markdown("Identifies which academic indicators exert the strongest cumulative influence across the student population.")

        # Load student sample to explain
        students_df = get_all_students()
        if students_df.empty:
            st.info("No student records available to sample for global explanation. Please load data from the Dashboard.")
            return

        with st.spinner("Computing global feature importance using SHAP / Model attribution..."):
            # Exclude non-features
            features_to_use = [
                c for c in students_df.columns
                if c not in ["student_id", "name", "risk_level", "risk_score", "latest_prediction", "target", "created_at", "updated_at"]
            ]
            X_sample = students_df[features_to_use].head(150)
            
            global_res = compute_global_importance(pipeline, X_sample, max_features=12)

        st.caption(f"Method: **{global_res['method']}**")
        st.plotly_chart(
            plot_feature_importance(global_res["features"], global_res["values"], "Top Global Academic Risk Predictors"),
            use_container_width=True
        )

        with st.expander("Feature Importance Table"):
            imp_df = pd.DataFrame({
                "Rank": list(range(1, len(global_res["features"]) + 1)),
                "Academic Feature": global_res["features"],
                "Importance Score": global_res["values"]
            })
            st.table(imp_df)

    with tab2:
        st.markdown("### Individual Student Risk Factor Decomposition")
        st.markdown("Isolate the exact academic signals increasing or mitigating risk for a specific student.")

        students_df = get_all_students()
        if students_df.empty:
            st.info("No students in database.")
            return

        student_ids = list(students_df["student_id"].values)
        selected_id = st.selectbox("Select Student ID to Explain", student_ids)

        if selected_id:
            s_data = get_student_by_id(selected_id)
            if s_data:
                # Top metrics banner for student
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(render_kpi_card("Student Name", s_data.get("name", "N/A"), s_data.get("department", ""), "primary"), unsafe_allow_html=True)
                with col2:
                    st.markdown(render_kpi_card("Current Risk Level", s_data.get("risk_level", "N/A"), "Model classification", "danger" if s_data.get("risk_level")=="High" else "warning"), unsafe_allow_html=True)
                with col3:
                    st.markdown(render_kpi_card("Attendance Rate", f"{s_data.get('attendance_rate', 0):.1f}%", "Semester average", "teal"), unsafe_allow_html=True)
                with col4:
                    st.markdown(render_kpi_card("Failed Units", str(s_data.get("failed_subjects_count", 0)), "Curricular deficits", "danger"), unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Compute explanation
                with st.spinner("Generating local explanation..."):
                    local_exp = explain_single_prediction(pipeline, label_encoder, s_data)

                st.markdown("#### 🗣️ Natural Language Explanation Narrative")
                st.info(local_exp["narrative"])

                st.markdown("#### 📊 Feature Impact Breakdown")
                st.caption("Bars pointing right (red) indicate factors increasing risk; bars pointing left (teal) indicate mitigating factors.")
                st.plotly_chart(
                    plot_local_feature_impacts(local_exp["feature_impacts"]),
                    use_container_width=True
                )
