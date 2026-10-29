"""
EduGuard AI — Page 6: Intervention Management
Mentor consultation logging, action plan workflow, and descriptive longitudinal tracking.
"""
import streamlit as st
import pandas as pd
from datetime import datetime, date

from database.db import (
    get_all_students,
    get_student_by_id,
    get_interventions_by_student,
    get_all_interventions,
    get_student_history,
    record_intervention,
    update_intervention
)
from src.intervention_service import (
    INTERVENTION_CATEGORIES,
    INTERVENTION_STATUSES,
    evaluate_student_progress_post_intervention
)
from utils.visualization import plot_longitudinal_risk_trend
from utils.styles import render_kpi_card

def render():
    st.markdown("## 🤝 Academic Intervention & Mentor Case Management")
    st.markdown("Track structured student support interventions, log mentor conversations, and observe longitudinal trajectory changes.")

    students_df = get_all_students()
    if students_df.empty:
        st.warning("⚠️ No student records in database. Please load data from the Dashboard or Data Management.")
        return

    tab1, tab2, tab3 = st.tabs(["📝 Log New Intervention", "📋 Active Cases & Status Updates", "📈 Longitudinal Progress Tracker"])

    # Tab 1: Log Intervention
    with tab1:
        st.markdown("### Record Mentor Action / Support Meeting")
        
        # Student selection
        student_options = {
            f"{row['student_id']} — {row.get('name', 'Anonymous')} ({row.get('risk_level', 'Low')} Risk)": row["student_id"]
            for _, row in students_df.iterrows()
        }
        selected_label = st.selectbox("Select Student", list(student_options.keys()))
        selected_sid = student_options[selected_label]
        student_data = get_student_by_id(selected_sid)

        # Context card for student
        if student_data:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(render_kpi_card("Department", student_data.get("department", "N/A"), "Enrolled Program", "primary"), unsafe_allow_html=True)
            with c2:
                st.markdown(render_kpi_card("Risk Level", student_data.get("risk_level", "Low"), f"Score: {student_data.get('risk_score', 0)*100:.1f}%", "danger" if student_data.get("risk_level")=="High" else "warning"), unsafe_allow_html=True)
            with c3:
                st.markdown(render_kpi_card("Attendance", f"{student_data.get('attendance_rate', 0):.1f}%", "Semester Average", "teal"), unsafe_allow_html=True)
            with c4:
                st.markdown(render_kpi_card("Failed Units", str(student_data.get("failed_subjects_count", 0)), "Curricular Deficits", "danger"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        with st.form("new_intervention_form"):
            int_c1, int_c2 = st.columns(2)
            with int_c1:
                int_date = st.date_input("Intervention Date", value=date.today())
                category = st.selectbox("Intervention Category", INTERVENTION_CATEGORIES)
                mentor_name = st.text_input("Responsible Mentor / Advisor", value="Prof. Sarah Jenkins")
                status = st.selectbox("Initial Status", INTERVENTION_STATUSES, index=0)

            with int_c2:
                follow_up = st.date_input("Scheduled Follow-up Date", value=date.today())
                notes = st.text_area(
                    "Meeting Notes & Agreed Action Items",
                    placeholder="Discussed attendance drop; student reported commuting challenges and transit delays. Agreed to provide flexible morning module access and connected with commuter transit assistance..."
                )
                outcome = st.text_input("Initial Outcome / Milestone (Optional)", placeholder="Action plan agreed upon with student")

            submitted = st.form_submit_button("💾 Save Intervention Record")

        if submitted:
            if not notes.strip():
                st.error("Please provide brief meeting notes or action details.")
            else:
                record_intervention(
                    student_id=selected_sid,
                    category=category,
                    notes=notes,
                    mentor_name=mentor_name,
                    follow_up_date=follow_up.strftime("%Y-%m-%d"),
                    status=status,
                    follow_up_outcome=outcome,
                    intervention_date=int_date.strftime("%Y-%m-%d")
                )
                st.success(f"✅ Successfully logged intervention for student **{selected_sid}**!")
                st.rerun()

    # Tab 2: Active Cases & Status Updates
    with tab2:
        st.markdown("### 📋 Institutional Intervention Roster")
        interventions_df = get_all_interventions()

        if interventions_df.empty:
            st.info("No intervention records logged yet.")
        else:
            # Filter by status
            status_filter = st.selectbox("Filter by Status", ["All Statuses"] + INTERVENTION_STATUSES)
            filtered_ints = interventions_df.copy()
            if status_filter != "All Statuses":
                filtered_ints = filtered_ints[filtered_ints["status"] == status_filter]

            st.dataframe(
                filtered_ints[[
                    "id", "student_id", "student_name", "department", "category",
                    "intervention_date", "mentor_name", "follow_up_date", "status"
                ]],
                use_container_width=True,
                hide_index=True
            )

            # Update status workflow
            st.markdown("#### ✏️ Update Intervention Case")
            update_c1, update_c2, update_c3 = st.columns([1, 1, 2])
            with update_c1:
                int_id_to_edit = st.selectbox("Select Case ID to Update", list(interventions_df["id"].values))
            with update_c2:
                new_status = st.selectbox("New Status", INTERVENTION_STATUSES)
            with update_c3:
                new_outcome = st.text_input("Recorded Outcome Notes", placeholder="Student attended 3 remedial classes; grade improved.")

            if st.button("Update Case Status"):
                update_intervention(int_id_to_edit, status=new_status, follow_up_outcome=new_outcome)
                st.success(f"Case #{int_id_to_edit} updated to '{new_status}'!")
                st.rerun()

    # Tab 3: Longitudinal Progress Tracker
    with tab3:
        st.markdown("### 📈 Longitudinal Trajectory & Post-Intervention Monitoring")
        st.caption("Inspect risk trend evolution across evaluation checkpoints.")

        student_ids = list(students_df["student_id"].values)
        tracker_sid = st.selectbox("Select Student to Track Progress", student_ids, key="tracker_sid")

        if tracker_sid:
            s_hist = get_student_history(tracker_sid)
            s_ints = get_interventions_by_student(tracker_sid)

            if s_hist.empty:
                st.info("No longitudinal prediction history for this student.")
            else:
                st.plotly_chart(plot_longitudinal_risk_trend(s_hist, student_id=tracker_sid), use_container_width=True)

                # Descriptive evaluation
                eval_data = evaluate_student_progress_post_intervention(tracker_sid)
                if eval_data.get("has_data"):
                    st.markdown("#### Descriptive Progress Analysis")
                    pc1, pc2, pc3 = st.columns(3)
                    with pc1:
                        st.metric("Pre-Intervention Risk", f"{eval_data['pre_average_risk']*100:.1f}%")
                    with pc2:
                        st.metric("Current Estimated Risk", f"{eval_data['current_risk']*100:.1f}%", delta=f"{eval_data['risk_delta']*100:.1f}%", delta_color="inverse")
                    with pc3:
                        st.markdown(f"**Trajectory Status:**\n{eval_data['trend_status']}")

                    st.caption(f"ℹ️ *{eval_data['causal_disclaimer']}*")

                if not s_ints.empty:
                    st.markdown("#### Recorded Interventions for this Student")
                    st.dataframe(s_ints[["intervention_date", "category", "mentor_name", "status", "notes"]], use_container_width=True, hide_index=True)
                else:
                    st.info("No interventions logged for this student yet.")
