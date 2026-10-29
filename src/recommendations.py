"""
EduGuard AI — Support Recommendations Engine
Generates constructive, non-punitive, rule-based academic support recommendations
grounded in student indicators and model explanations.
"""
from typing import List, Dict, Any

def generate_action_recommendations(
    student_dict: Dict[str, Any],
    risk_category: str = "Low",
    risk_score: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Produce non-punitive, actionable recommendations based on academic signals.
    """
    recs = []

    att = float(student_dict.get("attendance_rate", 85.0))
    internal = float(student_dict.get("internal_assessment_score", 70.0))
    sub_rate = float(student_dict.get("assignment_submission_rate", 85.0))
    failed = int(student_dict.get("failed_subjects_count", 0))
    lms_hrs = float(student_dict.get("lms_activity_hours", 10.0))
    cu1_app = int(student_dict.get("curricular_units_1st_sem_approved", 5))
    cu2_app = int(student_dict.get("curricular_units_2nd_sem_approved", 5))

    # Rule 1: Low Attendance Alert
    if att < 70.0:
        recs.append({
            "priority": "Immediate" if att < 60.0 else "High",
            "category": "Attendance Review",
            "text": "Schedule a supportive, non-punitive mentor check-in to understand potential personal, commute, or health barriers impacting class attendance.",
            "trigger": f"Attendance is at {att:.1f}% (below the 70% institutional support threshold)."
        })
    elif att < 80.0:
        recs.append({
            "priority": "Routine",
            "category": "Attendance Monitoring",
            "text": "Provide weekly attendance summary check-ins with peer advisor to prevent further decline.",
            "trigger": f"Attendance is currently {att:.1f}%."
        })

    # Rule 2: Low or Declining Internal Assessment Marks
    if internal < 50.0:
        recs.append({
            "priority": "Immediate",
            "category": "Academic Tutoring",
            "text": "Enroll student in targeted remedial peer tutoring sessions and offer faculty office-hour support.",
            "trigger": f"Internal assessment score is {internal:.1f}/100 (below passing threshold)."
        })
    elif internal < 65.0:
        recs.append({
            "priority": "Medium",
            "category": "Academic Coaching",
            "text": "Recommend study skill workshops and review formative assessment practice materials with course instructor.",
            "trigger": f"Internal assessment score is {internal:.1f}/100."
        })

    # Rule 3: Missing Assignment Submissions
    if sub_rate < 70.0:
        recs.append({
            "priority": "High",
            "category": "Assignment Planning",
            "text": "Offer structured time-management coaching and facilitate an agreed timeline with faculty for pending submissions.",
            "trigger": f"Assignment submission rate is {sub_rate:.1f}%."
        })

    # Rule 4: Multiple Failed Units / Curricular Deficits
    if failed >= 2 or (cu1_app <= 2 and cu2_app <= 2):
        recs.append({
            "priority": "Immediate",
            "category": "Curricular Counseling",
            "text": "Initiate comprehensive academic counseling session to map an individualized degree recovery and prerequisite progression plan.",
            "trigger": f"Student has {failed} failed curricular unit(s) or low completion rate ({cu2_app} approved in 2nd semester)."
        })
    elif failed == 1:
        recs.append({
            "priority": "Medium",
            "category": "Subject Remediation",
            "text": "Connect with subject lead for supplementary learning modules before the upcoming makeup evaluation cycle.",
            "trigger": "Student has 1 failed curricular unit."
        })

    # Rule 5: Low LMS Platform Engagement
    if lms_hrs < 4.0:
        recs.append({
            "priority": "Routine",
            "category": "Digital Accessibility Check",
            "text": "Verify student has reliable device and high-speed internet access for campus digital learning tools.",
            "trigger": f"LMS activity averaged {lms_hrs:.1f} hours/week."
        })

    # Rule 6: High Overall Model Risk
    if risk_category == "High" or risk_score >= 0.65:
        recs.append({
            "priority": "Immediate",
            "category": "Multidisciplinary Review",
            "text": "Convene a holistic student support advisory conference including mentor, department advisor, and counseling staff.",
            "trigger": f"Model estimated cumulative risk score is {risk_score * 100:.1f}%."
        })

    # Default positive reinforcement if doing well
    if not recs:
        recs.append({
            "priority": "Routine",
            "category": "Positive Recognition",
            "text": "Acknowledge consistent academic performance and encourage participation in peer tutoring, research, or honors projects.",
            "trigger": "All academic indicators meet or exceed institutional targets."
        })

    return recs
