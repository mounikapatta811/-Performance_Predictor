"""
EduGuard AI — Page 8: Model Performance & Rigorous Evaluation
Comprehensive model metrics, confusion matrix, per-class reports, and cross-validation diagnostics.
"""
import streamlit as st
import pandas as pd
import json

from src.train_model import load_trained_model
from utils.visualization import plot_confusion_matrix, plot_roc_curves, plot_class_distribution
from utils.styles import render_kpi_card

def render():
    st.markdown("## 📈 Machine Learning Model Performance")
    st.markdown("Rigorous validation diagnostics, confusion matrices, and multiclass classification metrics.")

    model_payload = load_trained_model()
    if model_payload is None:
        st.warning("⚠️ No trained model found. Please train a model in **Data Management** or load demo data.")
        return

    metadata = model_payload.get("metadata", {})
    test_metrics = metadata.get("test_metrics", {})
    train_metrics = metadata.get("train_metrics", {})
    classes = metadata.get("classes", [])

    # Overview Banner
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(render_kpi_card("Model Architecture", metadata.get("model_type", "Random Forest"), f"v{metadata.get('model_version', '1.0')}", "primary"), unsafe_allow_html=True)
    with col2:
        st.markdown(render_kpi_card("Test Balanced Accuracy", f"{test_metrics.get('balanced_accuracy', 0)*100:.1f}%", "Balanced for Imbalance", "teal"), unsafe_allow_html=True)
    with col3:
        st.markdown(render_kpi_card("Macro F1-Score", f"{test_metrics.get('f1_macro', 0):.3f}", "Test Set Macro Average", "primary"), unsafe_allow_html=True)
    with col4:
        st.markdown(render_kpi_card("Multiclass ROC-AUC", f"{test_metrics.get('roc_auc', 0):.3f}" if test_metrics.get('roc_auc') else "N/A", "One-vs-Rest AUC", "success"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Generalization & Cross-Validation Diagnostics
    st.markdown("### 🔬 Generalization & Leak-Free Cross-Validation")
    gen_c1, gen_c2, gen_c3 = st.columns(3)
    with gen_c1:
        st.metric("Training Sample Size", f"{metadata.get('training_samples', 0):,} records")
    with gen_c2:
        st.metric("Holdout Test Sample Size", f"{metadata.get('test_samples', 0):,} records")
    with gen_c3:
        cv_folds = metadata.get("cv_folds", 5)
        cv_mean = metadata.get("cv_f1_macro_mean", 0.0)
        cv_std = metadata.get("cv_f1_macro_std", 0.0)
        st.metric(f"Stratified {cv_folds}-Fold CV F1", f"{cv_mean:.3f}", delta=f"± {cv_std:.3f}")

    # Confusion Matrix & Classification Report
    st.markdown("---")
    cm_c1, cm_c2 = st.columns([1, 1.2])

    with cm_c1:
        cm_data = test_metrics.get("confusion_matrix")
        if cm_data and classes:
            st.plotly_chart(plot_confusion_matrix(cm_data, classes), use_container_width=True)
        else:
            st.info("Confusion matrix not stored in model metadata.")

    with cm_c2:
        st.markdown("#### Per-Class Classification Report (Holdout Test)")
        class_rep = test_metrics.get("classification_report", {})
        
        rep_rows = []
        for cls_name in classes:
            if cls_name in class_rep:
                metrics_c = class_rep[cls_name]
                rep_rows.append({
                    "Class": cls_name,
                    "Precision": f"{metrics_c.get('precision', 0):.3f}",
                    "Recall (Sensitivity)": f"{metrics_c.get('recall', 0):.3f}",
                    "F1-Score": f"{metrics_c.get('f1-score', 0):.3f}",
                    "Support Count": int(metrics_c.get("support", 0))
                })

        if rep_rows:
            st.table(pd.DataFrame(rep_rows))
        else:
            st.info("Per-class report unavailable.")

        st.caption("ℹ️ *High recall for the 'Dropout' class is prioritized to minimize false negatives (failing to detect a student in need of academic support).*")

    # Train vs Test Comparison (Overfitting Check)
    st.markdown("### ⚖️ Train vs. Holdout Test Generalization Check")
    comp_df = pd.DataFrame([
        {
            "Metric": "Overall Accuracy",
            "Training Set": f"{train_metrics.get('accuracy', 0)*100:.1f}%",
            "Holdout Test Set": f"{test_metrics.get('accuracy', 0)*100:.1f}%"
        },
        {
            "Metric": "Balanced Accuracy",
            "Training Set": f"{train_metrics.get('balanced_accuracy', 0)*100:.1f}%",
            "Holdout Test Set": f"{test_metrics.get('balanced_accuracy', 0)*100:.1f}%"
        },
        {
            "Metric": "Macro Precision",
            "Training Set": f"{train_metrics.get('precision_macro', 0):.3f}",
            "Holdout Test Set": f"{test_metrics.get('precision_macro', 0):.3f}"
        },
        {
            "Metric": "Macro Recall",
            "Training Set": f"{train_metrics.get('recall_macro', 0):.3f}",
            "Holdout Test Set": f"{test_metrics.get('recall_macro', 0):.3f}"
        },
        {
            "Metric": "Macro F1-Score",
            "Training Set": f"{train_metrics.get('f1_macro', 0):.3f}",
            "Holdout Test Set": f"{test_metrics.get('f1_macro', 0):.3f}"
        }
    ])
    st.table(comp_df)

    # ROC Curves & Class Distribution
    st.markdown("### 📊 ROC Curves & Cohort Class Representation")
    curve_c1, curve_c2 = st.columns([1.1, 1])

    with curve_c1:
        roc_data = test_metrics.get("roc_curves", {})
        if roc_data:
            st.plotly_chart(plot_roc_curves(roc_data), use_container_width=True)
        else:
            st.info("ROC curve coordinates are generated when class probability estimates are available.")

    with curve_c2:
        train_dist = metadata.get("train_class_distribution", {})
        test_dist = metadata.get("test_class_distribution", {})
        if train_dist or test_dist:
            st.plotly_chart(plot_class_distribution(train_dist, test_dist, classes), use_container_width=True)
        else:
            st.info("Class distribution data available upon model training.")

    # Model Features & Transformation Parameters
    with st.expander("🛠️ Active Preprocessing Pipeline & Feature Configuration"):
        st.markdown(f"**Random State Seed:** `{metadata.get('random_state', 42)}`")
        st.markdown(f"**Training Timestamp:** `{metadata.get('training_timestamp', 'N/A')}`")
        st.markdown(f"**Numeric Features ({len(metadata.get('numeric_features', []))}):**")
        st.code(", ".join(metadata.get("numeric_features", [])))
        st.markdown(f"**Categorical Features ({len(metadata.get('categorical_features', []))}):**")
        st.code(", ".join(metadata.get("categorical_features", [])))
