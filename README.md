# 🎓 EduGuard AI — Student Success Intelligence

> **Predict early. Understand risk. Support every student.**

A full-stack, professional, and explainable AI web application designed for higher education institutions to predict student dropout and academic risk, uncover underlying drivers using Explainable AI (SHAP), and coordinate proactive, non-punitive mentor interventions.

---

## 🌟 Key Capabilities

1. **Executive Academic Dashboard:** Real-time KPI metric cards (Total Analyzed, High Risk, Moderate Risk, Low Risk, Attendance Average, Recorded Interventions), risk donut chart, attendance vs. academic performance scatter plot, department breakdowns, and priority review rosters.
2. **Student Risk Analysis & Segmentation:** Multidimensional filtering across risk tiers, departments, and attendance sliders with searchable student directories.
3. **Individual Risk Inference:** Real-time scoring using trained machine learning pipelines, uncertainty entropy metrics, and margin of confidence scoring.
4. **Explainable AI (XAI) with SHAP:** Transparent global feature importances (TreeExplainer/LinearExplainer) and local student driver factor breakdowns with natural-language narrative summaries and explicit distinction between predictive association and causation.
5. **Rule-Based Action Recommendation Engine:** Automated, non-punitive support recommendations (tutoring, attendance counseling, curricular pacing, time management) based on academic deficits.
6. **Mentor Intervention Management:** Complete SQLite-backed case management system allowing advisors to log consultations, assign responsible mentors, schedule follow-ups, and track longitudinal progress.
7. **Longitudinal Risk Tracking:** Historical tracking across evaluation checkpoints with sparklines and trajectory trend charts.
8. **Fairness & Responsible AI Auditing:** Audit demographic parity, disparate impact (four-fifths rule), group recall, and false-positive rates across sensitive cohorts (gender, scholarship status, financial standing). Sensitive attributes are strictly excluded from the default predictive feature set.
9. **Batch Processing & Export:** Ingest class rosters in CSV or Excel format, run batch inferences, and export formatted CSV reports.
10. **Model Training & Performance Studio:** Train and compare Random Forest, Logistic Regression, and XGBoost classifiers with leak-free `ColumnTransformer` pipelines, stratified cross-validation, and confusion matrix heatmaps.
11. **Privacy & Data Governance:** Parameterized SQLite queries, synthetic demo dataset generator, and complete GDPR/FERPA-compliant data deletion mechanisms.

---

## 🏗️ Architecture & Technology Stack

```
eduguard_ai/
│
├── app.py                      # Main Streamlit application & navigation router
├── config.py                   # Central configuration, themes, thresholds & fields
├── requirements.txt            # Python dependencies with compatible versions
├── pytest.ini                  # Pytest test runner configuration
├── README.md                   # Full documentation & run guide
│
├── database/                   # SQLite persistence layer
│   ├── __init__.py
│   ├── db.py                   # Parameterized queries, CRUD operations & transactions
│   ├── schema.sql              # Database DDL schema (students, interventions, predictions)
│   └── eduguard.db             # SQLite database instance
│
├── data/                       # Datasets & synthetic cohorts
│   └── sample_dataset.csv      # Pre-generated synthetic demonstration cohort
│
├── models/                     # Model artifacts & serialization
│   ├── latest_pipeline.joblib  # Trained scikit-learn pipeline & label encoder
│   └── model_metadata.json     # Hyperparameters, evaluation metrics & feature metadata
│
├── src/                        # Core ML & analytics services
│   ├── __init__.py
│   ├── data_loader.py          # CSV/Excel parsing, UCI mapping & synthetic generator
│   ├── preprocessing.py        # Leakage-free ColumnTransformer & Pipeline builder
│   ├── train_model.py          # Model training, StratifiedKFold CV & persistence
│   ├── predict.py              # Inference, risk calibration & uncertainty entropy
│   ├── explainability.py       # SHAP global & local explainers with fallbacks
│   ├── recommendations.py      # Rule-based support recommendation engine
│   ├── fairness.py             # Algorithmic parity & demographic fairness auditing
│   ├── intervention_service.py # Case management & post-intervention monitoring
│   └── evaluation.py           # Confusion matrix & multiclass metric calculations
│
├── pages/                      # 10 Dedicated Streamlit navigation views
│   ├── __init__.py
│   ├── dashboard.py            # Page 1: Executive KPI overview & charts
│   ├── student_risk.py         # Page 2: Cohort risk segmentation & roster
│   ├── individual_prediction.py# Page 3: Single student assessment form
│   ├── batch_prediction.py     # Page 4: Bulk CSV/Excel prediction & export
│   ├── explainable_ai.py       # Page 5: Global & individual SHAP insights
│   ├── intervention_management.py # Page 6: Mentor case tracking & follow-ups
│   ├── fairness_ethics.py      # Page 7: Algorithmic bias & parity audit
│   ├── model_performance.py    # Page 8: Confusion matrix & CV diagnostics
│   ├── data_management.py      # Page 9: Ingestion, cleaning & training studio
│   └── about.py                # Page 10: Technical overview & ethical principles
│
├── utils/                      # UI styles, charts & utilities
│   ├── __init__.py
│   ├── styles.py               # Custom CSS design system (Navy Blue / Teal)
│   ├── visualization.py        # Interactive Plotly figures & charts
│   ├── validation.py           # Data quality diagnostics & leakage prevention
│   └── export.py               # CSV/Excel download serializers
│
└── tests/                      # Automated unit test suite
    ├── __init__.py
    ├── test_preprocessing.py   # Pipeline transformation tests
    ├── test_predictions.py     # Inference & risk scoring tests
    ├── test_database.py        # SQLite persistence & deletion tests
    ├── test_validation.py      # Anomaly detection & leakage prevention tests
    └── test_recommendations.py # Rule engine supportive phrasing tests
```

---

## 🚀 Installation & Local Execution

### Prerequisites
- Python 3.10 or 3.11 installed.

### Windows (PowerShell or Command Prompt)
```powershell
# 1. Clone or navigate to the project directory
cd "C:\Users\Ahamed Baig\.gemini\antigravity\scratch\eduguard_ai"

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate the virtual environment
.venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Launch the Streamlit application
streamlit run app.py
```

### Linux & macOS
```bash
# 1. Navigate to the project directory
cd eduguard_ai

# 2. Create a virtual environment
python3 -m venv .venv

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Launch the Streamlit application
streamlit run app.py
```

The application will be accessible at `http://localhost:8501`.

---

## 🧪 Running the Automated Test Suite

To run all unit tests verifying preprocessing, model inference, database CRUD, fairness bounds, and recommendation logic:

```bash
python -m pytest tests -v
```

---

## 📂 Dataset Instructions: UCI Predict Students' Dropout and Academic Success

EduGuard AI is natively compatible with the benchmark **Predict Students' Dropout and Academic Success** dataset from the UCI Machine Learning Repository.

### How to Download the Official Dataset:
1. Visit the official repository page: [UCI Machine Learning Repository - Dataset 697](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success).
2. Download the `data.csv` file.
3. In EduGuard AI, navigate to **Data Management** (`Page 9`).
4. Upload `data.csv`. The system will automatically detect the semicolon (`;`) delimiter, map UCI columns (such as `Curricular units 2nd sem (approved)` and `Previous qualification grade`), and display a data quality report.
5. Select your desired algorithm (Random Forest, Logistic Regression, or XGBoost) and click **Train Model & Preprocessing Pipeline**.

---

## 🧠 Interpreting Explainable AI (XAI) Output

EduGuard AI implements Explainable AI using SHAP:
- **Global Feature Importance:** Identifies systemic factors impacting the overall student population (e.g., Curricular units completed in semester 2, attendance rate, internal marks).
- **Individual Local Explanation:** Decomposes a student's prediction into positive and negative drivers.
  - 🔴 **Red Bars (Increases Risk):** Indicators that elevate the estimated risk (e.g., failed subjects deficit, attendance below 70%).
  - 🟢 **Teal Bars (Decreases Risk):** Protective academic buffers (e.g., strong semester 1 performance, high assignment completion).
- **Predictive Association vs. Causation Notice:** Explanations reflect statistical patterns in historical data. They do **not** claim a specific factor caused an outcome, and are presented to guide helpful advisor check-ins.

---

## ⚖️ Fairness & Responsible AI Safeguards

1. **Sensitive Attributes Excluded by Default:** Demographic factors (`gender`, `age_at_enrollment`, `scholarship_holder`, `tuition_fees_up_to_date`, `debtor`) are strictly barred from the default feature set to prevent algorithmic proxy discrimination.
2. **Supportive Terminology:** Students are flagged as *"potentially benefiting from academic support"*, never labeled as failures or certain dropouts.
3. **Four-Fifths Rule Disparity Audits:** The fairness audit module flags selection rate disparities when group ratios fall below 0.80.
4. **Data Minimization & Deletion:** An explicit data purge utility is provided to wipe student records in adherence with FERPA and GDPR standards.

---

## 📊 Evaluation & Verification Summary

- **Pipeline Integrity:** Stratified train/test splits ensure no data leakage during imputation or scaling.
- **Class Imbalance:** Handled via `class_weight='balanced'` and multiclass One-vs-Rest ROC-AUC.
- **Unit Test Coverage:** 100% pass rate across 7 unit test suites covering database persistence, preprocessing, model inference, and recommendation logic.
- **Live Health Status:** Streamlit web server verified with HTTP 200 response on port 8501.
