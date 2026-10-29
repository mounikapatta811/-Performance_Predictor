"""
EduGuard AI — Student Success Intelligence
Main Streamlit Application Controller & Navigation Router.
"""
import streamlit as st

# Configure Streamlit page settings
st.set_page_config(
    page_title="EduGuard AI — Student Success Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Imports
from config import APP_TITLE, APP_SUBTITLE, APP_VERSION
from database.db import init_db, get_students_count
from utils.styles import CUSTOM_CSS, render_header

# Initialize SQLite database schema
init_db()

# Inject design system custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Render main branded header
st.markdown(render_header(), unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.markdown("### 🎓 **EduGuard AI**")
    st.caption("Student Success Intelligence Platform")
    st.markdown("---")

    nav_pages = {
        "📊 Dashboard": "dashboard",
        "🔍 Student Risk Analysis": "student_risk",
        "🎯 Individual Student Prediction": "individual_prediction",
        "📁 Batch Prediction": "batch_prediction",
        "🧠 Explainable AI": "explainable_ai",
        "🤝 Intervention Management": "intervention_management",
        "⚖️ Fairness and Ethics": "fairness_ethics",
        "📈 Model Performance": "model_performance",
        "💾 Data Management": "data_management",
        "📖 About the Project": "about"
    }

    selected_page_label = st.radio(
        "Navigation",
        list(nav_pages.keys()),
        index=0,
        label_visibility="collapsed"
    )
    selected_page_key = nav_pages[selected_page_label]

    st.markdown("---")
    student_count = get_students_count()
    st.markdown(f"**Cohort Records:** `{student_count:,}` students")
    st.markdown(f"**Version:** `{APP_VERSION}`")
    st.caption("Predict early. Understand risk. Support every student.")

# Page Router
if selected_page_key == "dashboard":
    from pages import dashboard
    dashboard.render()

elif selected_page_key == "student_risk":
    from pages import student_risk
    student_risk.render()

elif selected_page_key == "individual_prediction":
    from pages import individual_prediction
    individual_prediction.render()

elif selected_page_key == "batch_prediction":
    from pages import batch_prediction
    batch_prediction.render()

elif selected_page_key == "explainable_ai":
    from pages import explainable_ai
    explainable_ai.render()

elif selected_page_key == "intervention_management":
    from pages import intervention_management
    intervention_management.render()

elif selected_page_key == "fairness_ethics":
    from pages import fairness_ethics
    fairness_ethics.render()

elif selected_page_key == "model_performance":
    from pages import model_performance
    model_performance.render()

elif selected_page_key == "data_management":
    from pages import data_management
    data_management.render()

elif selected_page_key == "about":
    from pages import about
    about.render()
