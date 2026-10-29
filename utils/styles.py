"""
EduGuard AI — Custom CSS & Design System
Light theme featuring navy blue, white, and subtle teal accents.
"""

CUSTOM_CSS = """
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #0F172A;
        background-color: #F8FAFC;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1350px;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid #1E293B;
    }

    section[data-testid="stSidebar"] .stMarkdown, 
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] span, 
    section[data-testid="stSidebar"] label {
        color: #E2E8F0 !important;
    }

    /* EduGuard Header Card */
    .eduguard-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #0F172A 100%);
        color: #FFFFFF;
        padding: 1.75rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 15px -3px rgba(15, 23, 42, 0.1), 0 4px 6px -2px rgba(15, 23, 42, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .eduguard-header h1 {
        color: #FFFFFF !important;
        font-weight: 700;
        font-size: 1.9rem;
        letter-spacing: -0.025em;
        margin-bottom: 0.35rem;
    }
    .eduguard-header p {
        color: #94A3B8 !important;
        font-size: 1.0rem;
        font-weight: 400;
        margin: 0;
    }

    /* KPI Metric Cards */
    .kpi-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px 0 rgba(0, 0, 0, 0.03);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        position: relative;
        overflow: hidden;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08), 0 2px 4px -1px rgba(0, 0, 0, 0.04);
    }
    .kpi-title {
        color: #64748B;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    .kpi-value {
        color: #0F172A;
        font-size: 2.1rem;
        font-weight: 700;
        line-height: 1.1;
        letter-spacing: -0.02em;
    }
    .kpi-subtitle {
        color: #64748B;
        font-size: 0.82rem;
        margin-top: 0.5rem;
    }

    /* Left accent borders for KPI cards */
    .kpi-primary { border-left: 4px solid #1E3A8A; }
    .kpi-teal { border-left: 4px solid #0D9488; }
    .kpi-danger { border-left: 4px solid #EF4444; }
    .kpi-warning { border-left: 4px solid #F59E0B; }
    .kpi-success { border-left: 4px solid #10B981; }

    /* Risk Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 0.3rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        line-height: 1;
    }
    .badge-high {
        background-color: #FEF2F2;
        color: #B91C1C;
        border: 1px solid #FCA5A5;
    }
    .badge-moderate {
        background-color: #FFFBEB;
        color: #B45309;
        border: 1px solid #FDE68A;
    }
    .badge-low {
        background-color: #ECFDF5;
        color: #047857;
        border: 1px solid #A7F3D0;
    }

    /* Content Card */
    .content-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04);
    }
    .content-card-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #0F172A;
        margin-bottom: 1rem;
        border-bottom: 1px solid #F1F5F9;
        padding-bottom: 0.6rem;
    }

    /* Streamlit Button Styling */
    div.stButton > button:first-child {
        background-color: #1E3A8A;
        color: #FFFFFF;
        font-weight: 500;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.25rem;
        transition: background-color 0.15s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #1E40AF;
        color: #FFFFFF;
    }
    div.stButton > button[kind="secondary"] {
        background-color: #F1F5F9;
        color: #334155;
        border: 1px solid #CBD5E1;
    }

    /* Clean Streamlit DataFrames */
    div[data-testid="stDataFrame"] {
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        overflow: hidden;
    }

    /* Info and Warning Alerts custom styling */
    div.stAlert {
        border-radius: 8px;
        border-width: 1px;
    }

    /* Onboarding Hero Banner */
    .onboarding-hero {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 2.5rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        margin: 2rem 0;
    }
    .onboarding-hero h2 {
        color: #1E3A8A;
        font-weight: 700;
        font-size: 1.8rem;
        margin-bottom: 0.75rem;
    }
    .onboarding-hero p {
        color: #475569;
        font-size: 1.05rem;
        max-width: 650px;
        margin: 0 auto 1.5rem auto;
    }
</style>
"""

def render_header(title: str = None, subtitle: str = None):
    """Render unified branded header banner."""
    from config import APP_TITLE, APP_SUBTITLE
    t = title or APP_TITLE
    s = subtitle or APP_SUBTITLE
    return f"""
    <div class="eduguard-header">
        <h1>{t}</h1>
        <p>{s}</p>
    </div>
    """

def render_kpi_card(title: str, value: Any, subtitle: str = "", variant: str = "primary") -> str:
    """Render modern KPI metric card with custom variant border."""
    return f"""
    <div class="kpi-card kpi-{variant}">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-subtitle">{subtitle}</div>
    </div>
    """

def render_risk_badge(risk_category: str) -> str:
    """Render stylized pill badge for risk classification."""
    cat = (risk_category or "low").lower()
    if "high" in cat:
        return '<span class="badge badge-high">High Risk — Support Needed</span>'
    elif "mod" in cat:
        return '<span class="badge badge-moderate">Moderate Risk — Monitor</span>'
    else:
        return '<span class="badge badge-low">Low Risk — On Track</span>'
