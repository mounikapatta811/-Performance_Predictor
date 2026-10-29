"""
EduGuard AI — Page 7: Fairness, Responsible AI & Ethical Auditing
Algorithmic fairness audits, demographic parity checks, and ethical guidelines.
"""
import streamlit as st
import pandas as pd

from database.db import get_all_students
from src.fairness import audit_group_fairness
from utils.visualization import plot_group_fairness
from utils.styles import render_kpi_card

def render():
    st.markdown("## ⚖️ Fairness, Responsible AI & Ethical Auditing")
    st.markdown("Audit algorithmic parity across sensitive cohorts, verify absence of demographic bias, and uphold ethical educational standards.")

    st.markdown("""
    > ### 🛡️ Responsible AI Core Charter
    > 1. **Demographic Agnosticism:** Demographic and socioeconomic factors (e.g. Gender, Age, Debtor status) are **strictly excluded** from default predictive features.
    > 2. **Supportive Framing:** Algorithmic indicators indicate students who *"may benefit from additional institutional support"*, never labeling an individual as a "failure" or "destined to drop out".
    > 3. **Non-Punitive Action:** Algorithmic predictions must never trigger punitive actions, loss of scholarship, or exclusion from educational opportunities.
    > 4. **Human-in-the-Loop:** All automated flags serve solely as advisory cues to empower compassionate, qualified faculty mentors.
    """)

    students_df = get_all_students()
    if students_df.empty:
        st.warning("⚠️ No student records found. Please load data from the Dashboard or Data Management to perform fairness auditing.")
        return

    # Check for sensitive attributes in database
    sensitive_cols = [c for c in ["gender", "scholarship_holder", "tuition_fees_up_to_date", "debtor"] if c in students_df.columns]

    if not sensitive_cols:
        st.info("No sensitive demographic attributes present in current database. Sensitive attributes are only collected when authorized for compliance auditing.")
        return

    st.markdown("### 🔍 Select Demographic Audit Dimension")
    audit_col1, audit_col2 = st.columns([1.5, 1])

    with audit_col1:
        chosen_group = st.selectbox("Sensitive Demographic Attribute to Audit", sensitive_cols)
    with audit_col2:
        target_positive = st.selectbox("Audit Positive Support Class", ["Dropout", "Enrolled", "Graduate"])

    # Run Fairness Audit
    # We need ground truth or predicted target
    true_col = "target" if "target" in students_df.columns else "latest_prediction"
    pred_col = "latest_prediction" if "latest_prediction" in students_df.columns else "risk_level"

    try:
        audit_res = audit_group_fairness(
            df=students_df,
            group_col=chosen_group,
            y_true_col=true_col,
            y_pred_col=pred_col,
            positive_class=target_positive,
            min_sample_threshold=15
        )
    except Exception as e:
        st.error(f"Could not compute fairness audit: {e}")
        return

    # KPI Disparity Metrics
    st.markdown(f"### 📊 Parity Metrics for `{chosen_group}`")
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(render_kpi_card("Selection Ratio", f"{audit_res['selection_ratio']:.2f}", "Parity threshold >= 0.80", "teal" if audit_res['selection_ratio']>=0.8 else "warning"), unsafe_allow_html=True)
    with m2:
        st.markdown(render_kpi_card("Recall Gap", f"{audit_res['recall_gap']*100:.1f}%", "Max difference in recall", "primary"), unsafe_allow_html=True)
    with m3:
        st.markdown(render_kpi_card("FPR Disparity", f"{audit_res['fpr_gap']*100:.1f}%", "False-positive gap", "danger" if audit_res['fpr_gap']>0.15 else "success"), unsafe_allow_html=True)
    with m4:
        st.markdown(render_kpi_card("Audited Cohort", f"{audit_res['total_evaluated_students']:,}", "Validated samples", "primary"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Disparity Visualizations
    st.plotly_chart(plot_group_fairness(audit_res), use_container_width=True)

    # Detailed Table
    st.markdown("#### Granular Group Performance Breakdown")
    metrics_table = pd.DataFrame(audit_res["group_metrics"])
    rename_dict = {
        "group": "Demographic Cohort",
        "sample_count": "Sample Count",
        "sample_percentage": "Cohort %",
        "selection_rate": "Selection Rate (Identified for Support)",
        "recall": "Recall (True Positive Rate)",
        "precision": "Precision",
        "false_positive_rate": "False Positive Rate (FPR)",
        "is_small_sample": "Small Sample Alert (<15)"
    }
    st.dataframe(metrics_table.rename(columns=rename_dict), use_container_width=True, hide_index=True)

    # Disparity Warnings
    if audit_res["warnings"]:
        st.markdown("#### ⚠️ Statistical & Fairness Warnings")
        for w in audit_res["warnings"]:
            st.warning(f"⚠️ {w}")
    else:
        st.success("✅ No severe demographic parity disparities detected across this attribute.")

    # Ethical Checklist
    with st.expander("📚 Institutional Ethics Checklist & Guidance"):
        st.markdown("""
        - **Informed Consent & Transparency:** Students have the right to know when predictive analytics are utilized to inform academic advising.
        - **Right to Contest:** Students and advisors may request manual review of any algorithmic categorization.
        - **Data Minimization:** Only performance indicators necessary for support intervention should be stored.
        - **Regular Retraining & Audits:** Quarterly audits must be conducted to detect model drift or emergent disparate impact.
        """)
