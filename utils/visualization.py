"""
EduGuard AI — Interactive Visualization Module
Plotly charts themed with Navy Blue, Teal, and standard academic analytics palettes.
"""
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import COLORS

LAYOUT_TEMPLATE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#0F172A", size=12),
    margin=dict(l=40, r=30, t=50, b=40),
    hoverlabel=dict(bgcolor="#0F172A", font_size=12, font_color="#FFFFFF")
)

def plot_risk_distribution_donut(risk_counts: Dict[str, int]) -> go.Figure:
    """Plot risk distribution as a clean modern donut chart."""
    labels = list(risk_counts.keys())
    values = list(risk_counts.values())

    color_map = {
        "High": COLORS["danger"],
        "Moderate": COLORS["warning"],
        "Low": COLORS["success"]
    }
    marker_colors = [color_map.get(label, COLORS["navy_light"]) for label in labels]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.62,
        marker=dict(colors=marker_colors, line=dict(color="#FFFFFF", width=2)),
        textinfo="percent+label",
        hoverinfo="label+value+percent",
        textfont=dict(size=13, weight="bold")
    )])

    fig.update_layout(
        **LAYOUT_TEMPLATE,
        title=dict(text="<b>Student Risk Distribution</b>", font=dict(size=15, color="#1E3A8A")),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
    )
    return fig

def plot_attendance_vs_performance(df: pd.DataFrame) -> go.Figure:
    """Plot attendance rate vs internal assessment marks scatter plot."""
    plot_df = df.copy()
    if "attendance_rate" not in plot_df.columns:
        plot_df["attendance_rate"] = 80.0
    
    score_col = "internal_assessment_score" if "internal_assessment_score" in plot_df.columns else "previous_semester_grade"
    color_col = "risk_level" if "risk_level" in plot_df.columns else ("target" if "target" in plot_df.columns else None)

    color_discrete_map = {
        "High": COLORS["danger"],
        "Moderate": COLORS["warning"],
        "Low": COLORS["success"],
        "Dropout": COLORS["danger"],
        "Enrolled": COLORS["warning"],
        "Graduate": COLORS["success"]
    }

    fig = px.scatter(
        plot_df,
        x="attendance_rate",
        y=score_col,
        color=color_col,
        hover_data=["student_id"] if "student_id" in plot_df.columns else None,
        color_discrete_map=color_discrete_map,
        labels={
            "attendance_rate": "Attendance Rate (%)",
            score_col: "Academic Performance Score",
            color_col: "Academic Standing"
        },
        opacity=0.75
    )

    # Add benchmark support threshold reference lines
    fig.add_vline(x=75.0, line_dash="dash", line_color=COLORS["warning"], annotation_text="Support Threshold (75%)")
    fig.add_hline(y=50.0, line_dash="dash", line_color=COLORS["danger"], annotation_text="Passing Benchmark")

    fig.update_layout(
        **LAYOUT_TEMPLATE,
        title=dict(text="<b>Attendance vs. Academic Performance</b>", font=dict(size=15, color="#1E3A8A")),
        xaxis=dict(showgrid=True, gridcolor="#F1F5F9", range=[30, 105]),
        yaxis=dict(showgrid=True, gridcolor="#F1F5F9"),
    )
    return fig

def plot_risk_by_department(df: pd.DataFrame) -> go.Figure:
    """Plot risk category distribution grouped by academic department."""
    dept_col = "department" if "department" in df.columns else "course_department"
    risk_col = "risk_level" if "risk_level" in df.columns else "target"

    if dept_col not in df.columns or risk_col not in df.columns:
        return go.Figure()

    grouped = df.groupby([dept_col, risk_col]).size().reset_index(name="count")

    color_map = {
        "High": COLORS["danger"],
        "Moderate": COLORS["warning"],
        "Low": COLORS["success"],
        "Dropout": COLORS["danger"],
        "Enrolled": COLORS["warning"],
        "Graduate": COLORS["success"]
    }

    fig = px.bar(
        grouped,
        x=dept_col,
        y="count",
        color=risk_col,
        barmode="stack",
        color_discrete_map=color_map,
        labels={dept_col: "Department / Course", "count": "Number of Students", risk_col: "Risk Category"}
    )

    fig.update_layout(
        **LAYOUT_TEMPLATE,
        title=dict(text="<b>Student Academic Standing by Department</b>", font=dict(size=15, color="#1E3A8A")),
        xaxis=dict(tickangle=-25),
        yaxis=dict(showgrid=True, gridcolor="#F1F5F9"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5)
    )
    return fig

def plot_longitudinal_risk_trend(history_df: pd.DataFrame, student_id: str = None) -> go.Figure:
    """Plot longitudinal risk trajectories across weekly/monthly observation checkpoints."""
    if history_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No longitudinal history records available yet.<br>Load demo data or record predictions over multiple timepoints.",
            showarrow=False,
            font=dict(size=13, color="#64748B")
        )
        fig.update_layout(**LAYOUT_TEMPLATE)
        return fig

    fig = go.Figure()

    if student_id:
        sub_df = history_df[history_df["student_id"] == student_id].sort_values("prediction_date")
        if len(sub_df) == 0:
            fig.add_annotation(
                text=f"No prediction history recorded for Student <b>{student_id}</b>.",
                showarrow=False,
                font=dict(size=13, color="#64748B")
            )
            fig.update_layout(**LAYOUT_TEMPLATE)
            return fig
        elif len(sub_df) == 1:
            obs_date = sub_df["prediction_date"].iloc[0]
            obs_score = sub_df["risk_score"].iloc[0] * 100
            obs_cat = sub_df["risk_category"].iloc[0]
            fig.add_annotation(
                text=f"<b>Longitudinal data not yet available for Student {student_id}</b><br>"
                     f"Only 1 observation recorded on <b>{obs_date}</b> (Risk: {obs_score:.1f}% — {obs_cat}).<br>"
                     f"At least two observation checkpoints are needed to display a trajectory trend.",
                showarrow=False,
                font=dict(size=13, color="#1E3A8A")
            )
            fig.update_layout(**LAYOUT_TEMPLATE)
            return fig

        # Multiple points: plot line and mark retrospective vs live
        is_retro = "is_retrospective" in sub_df.columns and sub_df["is_retrospective"].any()
        retro_note = " (Retrospective Demo Data)" if is_retro else " (Live Tracking)"
        
        fig.add_trace(go.Scatter(
            x=sub_df["prediction_date"],
            y=sub_df["risk_score"] * 100,
            mode="lines+markers",
            line=dict(color=COLORS["navy_primary"], width=3),
            marker=dict(size=9, color=COLORS["teal_primary"]),
            name=f"Student {student_id}"
        ))
        title_text = f"<b>Longitudinal Risk Trajectory: {student_id}</b>{retro_note}"
    else:
        # Aggregate average institutional trend
        agg = history_df.groupby("prediction_date")["risk_score"].agg(["mean", "count"]).reset_index()
        if len(agg) < 2:
            fig.add_annotation(
                text="Longitudinal trend tracking requires observations across at least 2 distinct evaluation dates.",
                showarrow=False,
                font=dict(size=13, color="#64748B")
            )
            fig.update_layout(**LAYOUT_TEMPLATE)
            return fig

        fig.add_trace(go.Scatter(
            x=agg["prediction_date"],
            y=agg["mean"] * 100,
            mode="lines+markers",
            line=dict(color=COLORS["navy_primary"], width=3),
            marker=dict(size=8, color=COLORS["teal_accent"]),
            name="Cohort Mean Risk %"
        ))
        title_text = "<b>Cohort Weekly Academic Risk Trajectory</b>"

    # Threshold horizontal reference zones
    fig.add_hline(y=65.0, line_dash="dot", line_color=COLORS["danger"], annotation_text="High Risk Threshold (65%)")
    fig.add_hline(y=35.0, line_dash="dot", line_color=COLORS["warning"], annotation_text="Moderate Risk Threshold (35%)")

    fig.update_layout(
        **LAYOUT_TEMPLATE,
        title=dict(text=title_text, font=dict(size=15, color="#1E3A8A")),
        xaxis_title="Evaluation Checkpoint Date",
        yaxis_title="Estimated Risk Score (%)",
        yaxis=dict(range=[0, 100], showgrid=True, gridcolor="#F1F5F9")
    )
    return fig

def plot_feature_importance(features: List[str], values: List[float], title: str = "Feature Importance") -> go.Figure:
    """Plot horizontal bar chart for feature importances."""
    # Reverse so top is at top
    rev_f = features[::-1]
    rev_v = values[::-1]

    fig = go.Figure(go.Bar(
        x=rev_v,
        y=rev_f,
        orientation="h",
        marker=dict(
            color=COLORS["teal_primary"],
            line=dict(color=COLORS["navy_primary"], width=1)
        )
    ))

    fig.update_layout(
        **LAYOUT_TEMPLATE,
        title=dict(text=f"<b>{title}</b>", font=dict(size=15, color="#1E3A8A")),
        xaxis=dict(title="Importance Score", showgrid=True, gridcolor="#F1F5F9"),
        yaxis=dict(autorange="reversed" if False else None),
        height=max(400, len(features) * 28)
    )
    return fig

def plot_local_feature_impacts(feature_impacts: List[Dict[str, Any]]) -> go.Figure:
    """Plot individual prediction drivers (waterfall or diverging bar)."""
    names = [f["feature"] for f in feature_impacts][::-1]
    impacts = [f["impact"] for f in feature_impacts][::-1]
    colors = [COLORS["danger"] if imp > 0 else COLORS["teal_primary"] for imp in impacts]

    fig = go.Figure(go.Bar(
        x=impacts,
        y=names,
        orientation="h",
        marker=dict(color=colors)
    ))

    fig.update_layout(
        **LAYOUT_TEMPLATE,
        title=dict(text="<b>Individual Prediction Factor Contributions</b>", font=dict(size=15, color="#1E3A8A")),
        xaxis=dict(title="SHAP / Impact Value (Positive: Increases Risk, Negative: Mitigates Risk)", showgrid=True, gridcolor="#F1F5F9"),
        height=380
    )
    return fig

def plot_confusion_matrix(cm: List[List[int]], classes: List[str]) -> go.Figure:
    """Plot modern heatmap confusion matrix."""
    cm_arr = np.array(cm)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm_arr,
        x=classes,
        y=classes,
        colorscale=[[0, "#F8FAFC"], [0.5, "#93C5FD"], [1.0, "#1E3A8A"]],
        text=cm_arr,
        texttemplate="%{text}",
        textfont={"size": 15, "color": "auto"},
        hoverongaps=False
    ))

    fig.update_layout(
        **LAYOUT_TEMPLATE,
        title=dict(text="<b>Model Confusion Matrix (Evaluation Set)</b>", font=dict(size=15, color="#1E3A8A")),
        xaxis=dict(title="Predicted Outcome", side="bottom"),
        yaxis=dict(title="True Label", autorange="reversed"),
        width=520,
        height=450
    )
    return fig

def plot_group_fairness(fairness_dict: Dict[str, Any]) -> go.Figure:
    """Plot group-level comparison of Selection Rate and Recall."""
    group_metrics = fairness_dict.get("group_metrics", [])
    if not group_metrics:
        return go.Figure()

    df = pd.DataFrame(group_metrics)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["group"],
        y=df["selection_rate"] * 100,
        name="Identified for Support % (Selection Rate)",
        marker_color=COLORS["navy_light"]
    ))
    fig.add_trace(go.Bar(
        x=df["group"],
        y=df["recall"] * 100,
        name="Support Identification Recall %",
        marker_color=COLORS["teal_primary"]
    ))

    fig.update_layout(
        **LAYOUT_TEMPLATE,
        title=dict(text=f"<b>Fairness Comparison Across '{fairness_dict.get('group_column')}'</b>", font=dict(size=15, color="#1E3A8A")),
        barmode="group",
        yaxis=dict(title="Percentage (%)", range=[0, 105], showgrid=True, gridcolor="#F1F5F9"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
    )
    return fig

def plot_roc_curves(roc_curves_dict: Dict[str, Dict[str, Any]]) -> go.Figure:
    """Plot multiclass or binary ROC curves with diagonal reference line."""
    if not roc_curves_dict:
        fig = go.Figure()
        fig.add_annotation(text="ROC curve coordinates not available.", showarrow=False)
        fig.update_layout(**LAYOUT_TEMPLATE)
        return fig

    fig = go.Figure()
    palette = [COLORS["navy_primary"], COLORS["teal_primary"], COLORS["warning"], COLORS["danger"]]

    for i, (cls_name, curve) in enumerate(roc_curves_dict.items()):
        color = palette[i % len(palette)]
        auc_val = curve.get("auc", 0.0)
        fig.add_trace(go.Scatter(
            x=curve.get("fpr", []),
            y=curve.get("tpr", []),
            mode="lines",
            line=dict(color=color, width=2.5),
            name=f"{cls_name} (AUC = {auc_val:.3f})"
        ))

    # Diagonal reference line (random guess baseline)
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        line=dict(color="#94A3B8", dash="dash", width=1.5),
        name="Random Classifier (AUC = 0.500)",
        hoverinfo="skip"
    ))

    fig.update_layout(
        **LAYOUT_TEMPLATE,
        title=dict(text="<b>Receiver Operating Characteristic (ROC) Curves</b>", font=dict(size=15, color="#1E3A8A")),
        xaxis=dict(title="False Positive Rate (1 - Specificity)", range=[0, 1.02], showgrid=True, gridcolor="#F1F5F9"),
        yaxis=dict(title="True Positive Rate (Sensitivity / Recall)", range=[0, 1.02], showgrid=True, gridcolor="#F1F5F9"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
        width=550,
        height=450
    )
    return fig

def plot_class_distribution(train_dist: Dict[str, int], test_dist: Dict[str, int], classes: List[str]) -> go.Figure:
    """Plot grouped bar chart of class distribution in train vs test sets."""
    fig = go.Figure()
    classes_list = [str(c) for c in classes]
    train_vals = [train_dist.get(c, 0) for c in classes_list]
    test_vals = [test_dist.get(c, 0) for c in classes_list]

    fig.add_trace(go.Bar(
        x=classes_list,
        y=train_vals,
        name="Training Split",
        marker_color=COLORS["navy_primary"]
    ))
    fig.add_trace(go.Bar(
        x=classes_list,
        y=test_vals,
        name="Holdout Test Split",
        marker_color=COLORS["teal_accent"]
    ))

    fig.update_layout(
        **LAYOUT_TEMPLATE,
        title=dict(text="<b>Class Distribution (Stratified Train vs. Test)</b>", font=dict(size=15, color="#1E3A8A")),
        barmode="group",
        xaxis=dict(title="Target Outcome Class"),
        yaxis=dict(title="Number of Student Records", showgrid=True, gridcolor="#F1F5F9"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5),
        height=380
    )
    return fig
