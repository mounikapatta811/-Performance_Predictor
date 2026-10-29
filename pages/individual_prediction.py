"""
EduGuard AI — Page 3: Individual Student Prediction
Real-time risk scoring, explainable factor attribution, uncertainty analysis, and action recommendations.
"""
import streamlit as st
import pandas as pd
from datetime import datetime

from config import DEFAULT_RISK_THRESHOLDS, STANDARD_ACADEMIC_FIELDS
from src.train_model import load_trained_model
from src.predict import predict_single_student
from src.explainability import explain_single_prediction
from src.recommendations import generate_action_recommendations
from database.db import (
    upsert_student,
    log_prediction,
    get_student_by_id,
    add_recommendation
)
from utils.styles import render_kpi_card, render_risk_badge
from utils.visualization import plot_local_feature_impacts

def render():
    st.markdown("## 🎯 Individual Student Risk Assessment")
    st.markdown("Input student academic indicators to predict trajectory, examine contributing drivers, and generate supportive intervention plans.")

    model_payload = load_trained_model()
    if model_payload is None:
        st.warning("⚠️ No trained machine learning model found. Please train a model in the **Data Management** page or load the demo dataset.")
        return

    pipeline = model_payload["pipeline"]
    label_encoder = model_payload["label_encoder"]
    metadata = model_payload.get("metadata", {})

    st.caption(f"Active Model: **{metadata.get('model_type', 'Classifier')}** | Classes: **{', '.join(metadata.get('classes', []))}** | Version: `{metadata.get('model_version', 'latest')}`")

    # Quick prefill selector from existing student database
    st.markdown("### 📝 Input Academic Profile")
    quick_col1, quick_col2 = st.columns([2, 1])
    prefill_dict = {}
    
    with quick_col1:
        existing_id = st.text_input("Prefill from Existing Student ID (Optional)", placeholder="e.g. STU-2024-1001")
        if existing_id:
            found = get_student_by_id(existing_id.strip())
            if found:
                st.success(f"Loaded records for: {found.get('name', existing_id)}")
                prefill_dict = found
            else:
                st.info("Student ID not found in database; using default parameters.")

    with quick_col2:
        threshold_high = st.slider("High Risk Threshold (%)", 50, 85, int(DEFAULT_RISK_THRESHOLDS["high"] * 100)) / 100.0
        custom_thresholds = {
            "high": threshold_high,
            "moderate": DEFAULT_RISK_THRESHOLDS["moderate"],
            "low": DEFAULT_RISK_THRESHOLDS["low"]
        }

    # Form with Academic Inputs
    with st.form("individual_prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 📚 Attendance & Assessments")
            student_id = st.text_input("Student Identifier", value=prefill_dict.get("student_id", "STU-2024-9999"))
            name = st.text_input("Student Name (Pseudonymized)", value=prefill_dict.get("name", "Jane Doe"))
            department = st.selectbox(
                "Department / Course",
                options=STANDARD_ACADEMIC_FIELDS["course_department"]["options"],
                index=0
            )
            attendance_rate = st.slider(
                "Attendance Rate (%)",
                0.0, 100.0,
                float(prefill_dict.get("attendance_rate", 78.5)),
                step=0.5,
                help="Semester class attendance percentage."
            )
            internal_score = st.slider(
                "Internal Assessment Score (0-100)",
                0.0, 100.0,
                float(prefill_dict.get("internal_assessment_score", 62.0)),
                step=0.5,
                help="Continuous internal evaluation average."
            )

        with col2:
            st.markdown("##### 📝 Submissions & Deficits")
            sub_rate = st.slider(
                "Assignment Submission Rate (%)",
                0.0, 100.0,
                float(prefill_dict.get("assignment_submission_rate", 80.0)),
                step=1.0,
                help="Percentage of coursework assignments delivered on time."
            )
            failed_count = st.number_input(
                "Failed Curricular Units / Backlogs",
                0, 15,
                int(prefill_dict.get("failed_subjects_count", 0)),
                help="Number of subjects requiring re-examination or makeup."
            )
            lms_hours = st.slider(
                "Weekly LMS Portal Activity (Hours)",
                0.0, 50.0,
                float(prefill_dict.get("lms_activity_hours", 8.5)),
                step=0.5,
                help="Average hours engaged with course digital management system."
            )
            prev_grade = st.number_input(
                "Previous Semester Grade / GPA",
                0.0, 20.0,
                float(prefill_dict.get("previous_semester_grade", 13.0)),
                step=0.1,
                help="Prior cumulative evaluation score (standard 0-20 scale or GPA equivalent)."
            )

        with col3:
            st.markdown("##### 🎓 Curricular Progression (UCI Standard)")
            cu1_approved = st.number_input(
                "1st Sem Curricular Units Approved",
                0, 10,
                int(prefill_dict.get("curricular_units_1st_sem_approved", 5))
            )
            cu1_grade = st.number_input(
                "1st Sem Average Grade (0-20)",
                0.0, 20.0,
                float(prefill_dict.get("curricular_units_1st_sem_grade", 12.8)),
                step=0.1
            )
            cu2_approved = st.number_input(
                "2nd Sem Curricular Units Approved",
                0, 10,
                int(prefill_dict.get("curricular_units_2nd_sem_approved", 5))
            )
            cu2_grade = st.number_input(
                "2nd Sem Average Grade (0-20)",
                0.0, 20.0,
                float(prefill_dict.get("curricular_units_2nd_sem_grade", 13.0)),
                step=0.1
            )

        submit_btn = st.form_submit_button("🔍 Generate AI Risk Assessment & Recommendations")

    # If submitted or default display
    if submit_btn:
        student_input = {
            "student_id": student_id,
            "name": name,
            "department": department,
            "attendance_rate": attendance_rate,
            "internal_assessment_score": internal_score,
            "assignment_submission_rate": sub_rate,
            "failed_subjects_count": failed_count,
            "lms_activity_hours": lms_hours,
            "previous_semester_grade": prev_grade,
            "curricular_units_1st_sem_approved": cu1_approved,
            "curricular_units_1st_sem_grade": cu1_grade,
            "curricular_units_2nd_sem_approved": cu2_approved,
            "curricular_units_2nd_sem_grade": cu2_grade
        }

        # Predict
        with st.spinner("Analyzing academic profile with machine learning model..."):
            pred_result = predict_single_student(
                pipeline=pipeline,
                label_encoder=label_encoder,
                student_dict=student_input,
                thresholds=custom_thresholds
            )

            # Explain
            explanation = explain_single_prediction(
                pipeline=pipeline,
                label_encoder=label_encoder,
                student_dict=student_input
            )

            # Recommendations
            recs = generate_action_recommendations(
                student_dict=student_input,
                risk_category=pred_result["risk_category"],
                risk_score=pred_result["risk_score"]
            )

        # Output Display
        st.markdown("---")
        st.markdown("### 📊 Assessment Results")

        res_c1, res_c2, res_c3, res_c4 = st.columns(4)
        with res_c1:
            st.markdown(render_kpi_card("Predicted Trajectory", pred_result["predicted_outcome"], "Model classification", "primary"), unsafe_allow_html=True)
        with res_c2:
            variant = "danger" if pred_result["risk_category"] == "High" else ("warning" if pred_result["risk_category"] == "Moderate" else "success")
            st.markdown(render_kpi_card("Estimated Risk Score", f"{pred_result['risk_percentage']}%", "Dropout/Deficit Risk", variant), unsafe_allow_html=True)
        with res_c3:
            st.markdown(render_kpi_card("Risk Classification", pred_result["risk_category"], f"Threshold: {custom_thresholds['high']*100:.0f}%", variant), unsafe_allow_html=True)
        with res_c4:
            st.markdown(render_kpi_card("Model Confidence", f"{pred_result['margin_of_confidence'] * 100:.1f}%", "Confidence margin", "teal"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Probabilities & Uncertainty Notice
        prob_c1, prob_c2 = st.columns([1, 1])
        with prob_c1:
            st.markdown("#### Class Probability Distribution")
            prob_df = pd.DataFrame([
                {"Outcome": k, "Probability (%)": f"{v*100:.1f}%"}
                for k, v in pred_result["class_probabilities"].items()
            ])
            st.table(prob_df)

        with prob_c2:
            st.markdown("#### ⚖️ Uncertainty & Operational Context")
            st.info(f"**Uncertainty Assessment:** {pred_result['uncertainty_level']}\n\n"
                    "**Operational Notice:** Risk classifications represent statistical probabilities derived from historical cohort patterns. "
                    "They must **never** be used to deny educational opportunities or enact disciplinary measures. Always pair algorithmic flags with compassionate human mentorship.")

        # XAI Factor Explanation
        st.markdown("### 💡 Explainable AI: Contributing Drivers")
        st.markdown(f"**Explanation Method:** `{explanation['method']}`")
        st.markdown(f"> {explanation['narrative']}")

        if explanation["feature_impacts"]:
            st.plotly_chart(plot_local_feature_impacts(explanation["feature_impacts"]), use_container_width=True)

        # Action Recommendations
        st.markdown("### 🎯 Recommended Academic Support Actions")
        st.caption("Customized, non-punitive interventions generated from student indicators and model explanations.")

        for i, rec in enumerate(recs):
            p = rec["priority"]
            badge_color = "🔴 Immediate" if p == "Immediate" else ("🟠 High" if p == "High" else "🟡 Routine")
            with st.expander(f"{badge_color} Priority — **{rec['category']}**", expanded=True):
                st.markdown(f"**Recommended Action:** {rec['text']}")
                st.markdown(f"**Triggering Indicator:** *{rec['trigger']}*")
                if st.button(f"📌 Adopt Recommendation #{i+1} into Mentor Plan", key=f"btn_rec_{i}"):
                    add_recommendation(
                        student_id=student_id,
                        recommendation_text=rec["text"],
                        trigger_reason=rec["trigger"],
                        priority=rec["priority"]
                    )
                    st.success("Recommendation saved to student's intervention plan!")

        # Save to Database Section
        st.markdown("---")
        save_col1, save_col2 = st.columns([2, 1])
        with save_col1:
            st.markdown("#### 💾 Log this Assessment to Longitudinal Database")
            st.caption("Stores record for weekly risk tracking and updates student profile.")
        with save_col2:
            if st.button("Save Record & Log Prediction", key="btn_save_pred"):
                student_record = {
                    **student_input,
                    "risk_level": pred_result["risk_category"],
                    "risk_score": pred_result["risk_score"],
                    "latest_prediction": pred_result["predicted_outcome"]
                }
                upsert_student(student_record)
                log_prediction(
                    student_id=student_id,
                    prediction_date=datetime.now().strftime("%Y-%m-%d"),
                    risk_score=pred_result["risk_score"],
                    predicted_outcome=pred_result["predicted_outcome"],
                    risk_category=pred_result["risk_category"],
                    top_factors=explanation.get("feature_impacts", []),
                    model_version=metadata.get("model_version", "v1.0.0"),
                    is_retrospective=False
                )
                st.success(f"✅ Record for student {student_id} successfully logged to database!")
