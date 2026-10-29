"""
EduGuard AI — Page 10: About the Project
Architecture, technology stack, ethical framework, and UCI dataset reference.
"""
import streamlit as st
from config import APP_TITLE, APP_SUBTITLE, APP_VERSION

def render():
    st.markdown("## 📖 About EduGuard AI")
    st.markdown(f"**{APP_TITLE}** — *{APP_SUBTITLE}* (Version `{APP_VERSION}`)")

    st.markdown("""
    ### 🎯 Problem Statement & Institutional Mission
    Higher education institutions worldwide face significant student attrition challenges. 
    Early identification of students encountering academic or engagement hurdles allows faculty advisors, 
    counselors, and mentors to deliver compassionate, personalized interventions before challenges escalate.

    **EduGuard AI** is built on the philosophy that **predictive algorithms must illuminate, not condemn**. 
    By coupling machine learning classification with rigorous **Explainable AI (XAI)**, our platform 
    transforms opaque probabilities into transparent, non-punitive, and actionable mentoring strategies.
    """)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🛠️ Technology Stack")
        st.markdown("""
        - **Frontend & Dashboard:** Streamlit with custom CSS responsive design system.
        - **Backend Runtime:** Python 3.11 with object-relational modular architecture.
        - **Machine Learning:** Scikit-learn (Random Forest, Logistic Regression), XGBoost.
        - **Explainable AI:** SHAP (TreeExplainer & LinearExplainer) with fallback feature attribution.
        - **Interactive Analytics:** Plotly (`plotly.graph_objects`, `plotly.express`).
        - **Database & Persistence:** SQLite3 with foreign-key referential integrity and parameterized queries.
        - **Model Serializer:** Joblib pipeline serialization.
        - **Testing Suite:** Pytest unit test coverage.
        """)

    with col2:
        st.markdown("### 🛡️ Ethical & Responsible AI Framework")
        st.markdown("""
        - **Exclusion of Sensitive Attributes:** Demographic attributes (gender, age, scholarship, financial debt) are excluded from the default predictive feature set by design.
        - **Non-Punitive Action Principle:** Automated risk scores must never lead to academic probation, dismissal, or withdrawal of scholarships.
        - **Predictive Association vs. Causation:** Model factors represent statistical correlations within cohort patterns, not clinical or biological causal proof.
        - **Human Oversight:** Qualified faculty advisors and counselors retain complete authority over all mentoring decisions.
        """)

    st.markdown("---")

    st.markdown("### 📚 Dataset Reference: UCI Predict Students' Dropout and Academic Success")
    st.markdown("""
    EduGuard AI supports the official **UCI Predict Students' Dropout and Academic Success dataset** 
    (contributed by Realinho et al., Polytechnic Institute of Portalegre, Portugal).

    - **Target Classes:**
      1. `Dropout`: Student discontinued enrollment.
      2. `Enrolled`: Student is actively enrolled and progressing towards degree completion.
      3. `Graduate`: Student successfully completed all degree requirements.
    - **Academic Predictors:**
      - Curricular units credited, enrolled, evaluated, and approved in 1st & 2nd semesters.
      - Grade point averages in 1st & 2nd semester courses (0–20 standard grading scale).
      - Previous secondary school qualification grades.
      - Attendance shifts (daytime vs. evening).
    - **Official Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
    """)

    st.markdown("---")

    st.markdown("### ⚖️ Operational Guidelines for Mentors & Advisors")
    st.markdown("""
    1. **Reach Out Early:** Initiate contact during the first 4–6 weeks when attendance or internal assessment scores begin to dip.
    2. **Foster Open Dialogue:** Use the explainable factors (e.g. low assignment submissions or LMS engagement) as gentle discussion topics rather than accusations.
    3. **Document Milestones:** Record agreed action plans and scheduled follow-up dates in the **Intervention Management** module to ensure continuous support.
    """)
