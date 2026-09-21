import streamlit as st
import cv2
import numpy as np

from src.database import save_embedding, load_database


# =========================================
# PAGE CONFIGURATION
# =========================================

st.set_page_config(
    page_title="Face Recognition System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================
# MODEL PATHS & THRESHOLD
# =========================================

yunet_model = "models/face_detection_yunet_2026may.onnx"
sface_model = "models/face_recognition_sface_2021dec.onnx"

THRESHOLD = 0.40


# =========================================
# LOAD MODELS
# =========================================

@st.cache_resource
def load_models():
    detector = cv2.FaceDetectorYN.create(
        yunet_model,
        "",
        (320, 320),
        0.9,
        0.3,
        5000
    )

    recognizer = cv2.FaceRecognizerSF.create(
        sface_model,
        ""
    )

    return detector, recognizer


detector, recognizer = load_models()


# =========================================
# GLOBAL CUSTOM STYLING (THEME)
# =========================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

/* Global Reset */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0B0E17 !important;
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #F1F5F9 !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

/* Sidebar Outer Styling */
[data-testid="stSidebar"] {
    background-color: #070A13 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
}

[data-testid="stSidebarNav"] {
    display: none !important;
}

/* Sidebar Header */
.sidebar-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 5px 25px 5px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    margin-bottom: 25px;
}

.sidebar-logo-icon {
    width: 38px;
    height: 38px;
    border-radius: 10px;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(139, 92, 246, 0.2));
    border: 1px solid rgba(139, 92, 246, 0.4);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 0 15px rgba(99, 102, 241, 0.2);
}

.sidebar-title {
    font-size: 18px;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: -0.2px;
}

.sidebar-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #64748B;
    margin-bottom: 12px;
    padding-left: 4px;
}

/* Hide Deploy button, 3 dots header menu, and default Streamlit header toolbar */
header[data-testid="stHeader"],
[data-testid="stAppHeader"],
[data-testid="stAppDeployButton"],
[data-testid="stToolbar"],
#MainMenu,
.stAppHeader {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Sidebar Radio Navigation Override */
[data-testid="stRadio"] > label {
    display: none !important;
}

[data-testid="stRadio"] div[role="radiogroup"] {
    gap: 10px !important;
    width: 100% !important;
}

[data-testid="stRadio"] div[role="radiogroup"] label {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 12px !important;
    padding: 12px 18px !important;
    margin: 0 !important;
    width: 100% !important;
    display: flex !important;
    align-items: center !important;
    box-sizing: border-box !important;
    cursor: pointer !important;
    transition: all 0.25s ease !important;
}

/* Hide Radio Circle / Red Dot Completely */
[data-testid="stRadio"] div[role="radiogroup"] label input[type="radio"] {
    display: none !important;
}

[data-testid="stRadio"] div[role="radiogroup"] label > div:first-child {
    display: none !important;
    width: 0 !important;
    height: 0 !important;
    opacity: 0 !important;
    visibility: hidden !important;
    margin: 0 !important;
    padding: 0 !important;
}

[data-testid="stRadio"] div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] {
    width: 100% !important;
}

[data-testid="stRadio"] div[role="radiogroup"] label p {
    color: #94A3B8 !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Hover State */
[data-testid="stRadio"] div[role="radiogroup"] label:hover {
    background: rgba(255, 255, 255, 0.08) !important;
    border-color: rgba(139, 92, 246, 0.3) !important;
}

[data-testid="stRadio"] div[role="radiogroup"] label:hover p {
    color: #F8FAFC !important;
}

/* Checked Selected State: Full Purple/Blue Gradient Card */
[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked),
[data-testid="stRadio"] div[role="radiogroup"] label[aria-checked="true"] {
    background: linear-gradient(135deg, #6366F1 0%, #7C3AED 100%) !important;
    border: 1px solid rgba(168, 85, 247, 0.6) !important;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4) !important;
}

[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) p,
[data-testid="stRadio"] div[role="radiogroup"] label[aria-checked="true"] p,
[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) [data-testid="stMarkdownContainer"] p {
    color: #FFFFFF !important;
    font-weight: 700 !important;
}

/* Container Spacing */
.main .block-container {
    padding-top: 1.8rem !important;
    padding-bottom: 3rem !important;
    max-width: 1080px !important;
}

/* Gradient Text Utilities */
.gradient-text {
    background: linear-gradient(135deg, #818CF8 0%, #C084FC 50%, #60A5FA 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Top Right Badge */
.top-badge-row {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 5px;
}

.ai-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(99, 102, 241, 0.12);
    border: 1px solid rgba(139, 92, 246, 0.3);
    padding: 6px 14px;
    border-radius: 20px;
    color: #C084FC;
    font-size: 13px;
    font-weight: 600;
    box-shadow: 0 0 15px rgba(139, 92, 246, 0.12);
}

/* Hero Section Typography */
.hero-welcome {
    text-align: center;
    text-transform: uppercase;
    letter-spacing: 3px;
    font-size: 12px;
    font-weight: 700;
    color: #818CF8;
    margin-bottom: 6px;
}

.hero-main-title {
    text-align: center;
    font-size: 46px;
    font-weight: 800;
    letter-spacing: -0.8px;
    color: #FFFFFF;
    margin-bottom: 8px;
    line-height: 1.15;
}

.hero-subtitle {
    text-align: center;
    font-size: 17px;
    font-weight: 500;
    color: #94A3B8;
    letter-spacing: 1.5px;
    margin-bottom: 18px;
}

.hero-description {
    text-align: center;
    max-width: 740px;
    margin: 0 auto 30px auto;
    font-size: 15px;
    line-height: 1.6;
    color: #94A3B8;
}

/* Feature Highlights Grid */
.feature-highlight-card {
    background: rgba(15, 20, 35, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 16px;
    padding: 16px 14px;
    text-align: center;
    transition: all 0.3s ease;
    height: 100%;
}

.feature-highlight-card:hover {
    border-color: rgba(129, 140, 248, 0.3);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.3);
}

.feature-highlight-icon {
    font-size: 24px;
    margin-bottom: 6px;
}

.feature-highlight-title {
    font-size: 14px;
    font-weight: 700;
    color: #F8FAFC;
    margin-bottom: 3px;
}

.feature-highlight-sub {
    font-size: 12px;
    color: #94A3B8;
}

/* Action Cards */
.action-card-box {
    background: rgba(13, 17, 30, 0.75);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 30px 26px 24px 26px;
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: space-between;
    transition: all 0.3s ease;
    height: 100%;
}

.action-card-box:hover {
    border-color: rgba(139, 92, 246, 0.35);
    box-shadow: 0 12px 32px rgba(99, 102, 241, 0.12);
}

.action-circle-purple {
    width: 64px;
    height: 64px;
    border-radius: 50%;
    background: radial-gradient(circle at center, #7C3AED 0%, #4C1D95 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 16px;
    box-shadow: 0 4px 20px rgba(124, 58, 237, 0.4);
}

.action-circle-blue {
    width: 64px;
    height: 64px;
    border-radius: 50%;
    background: radial-gradient(circle at center, #2563EB 0%, #1E3A8A 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 16px;
    box-shadow: 0 4px 20px rgba(37, 99, 235, 0.4);
}

.action-card-title {
    font-size: 22px;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 8px;
}

.action-card-desc {
    font-size: 14px;
    color: #94A3B8;
    line-height: 1.5;
    margin-bottom: 22px;
    min-height: 42px;
}

/* Custom Streamlit Buttons inside Action Cards */
div.stButton > button {
    width: 100% !important;
    border-radius: 12px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    border: none !important;
    transition: all 0.25s ease !important;
    cursor: pointer !important;
}

.btn-purple-wrapper div.stButton > button {
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%) !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 16px rgba(99, 102, 241, 0.35) !important;
}

.btn-purple-wrapper div.stButton > button:hover {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%) !important;
    box-shadow: 0 6px 24px rgba(99, 102, 241, 0.5) !important;
}

.btn-blue-wrapper div.stButton > button {
    background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%) !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 16px rgba(59, 130, 246, 0.35) !important;
}

.btn-blue-wrapper div.stButton > button:hover {
    background: linear-gradient(135deg, #2563EB 0%, #1E40AF 100%) !important;
    box-shadow: 0 6px 24px rgba(59, 130, 246, 0.5) !important;
}

/* Input Fields & Streamlit Widgets Theme */
.stTextInput > div > div > input {
    background-color: #0E1322 !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: #FFFFFF !important;
    border-radius: 10px !important;
}

.stSelectbox > div > div {
    background-color: #0E1322 !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: #FFFFFF !important;
    border-radius: 10px !important;
}

.stFileUploader > div {
    background-color: #0E1322 !important;
    border: 1px dashed rgba(139, 92, 246, 0.3) !important;
    border-radius: 14px !important;
}

/* Footer Styling */
.footer-wrapper {
    margin-top: 55px;
    padding-top: 25px;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
    text-align: center;
}

.footer-quote {
    font-size: 14px;
    color: #94A3B8;
    font-style: italic;
    margin-bottom: 6px;
}

.footer-sub {
    font-size: 12px;
    color: #64748B;
}

/* Sidebar Footer Branding */
.sidebar-footer-box {
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
}

.sidebar-company-title {
    font-size: 13px;
    font-weight: 700;
    color: #E2E8F0;
    display: flex;
    align-items: center;
    gap: 8px;
}

.sidebar-company-sub {
    font-size: 11px;
    color: #64748B;
    margin-top: 2px;
}

.sidebar-copyright {
    font-size: 10px;
    color: #475569;
    margin-top: 15px;
}
</style>
""", unsafe_allow_html=True)


# =========================================
# STATEFUL NAVIGATION SETUP
# =========================================

if "page" not in st.session_state:
    st.session_state["page"] = "🏠 Home"


def nav_to(target_page):
    st.session_state["page"] = target_page


page_options = [
    "🏠 Home",
    "👤 Enrollment",
    "🔎 Recognition"
]

if st.session_state["page"] not in page_options:
    st.session_state["page"] = "🏠 Home"


# =========================================
# SIDEBAR RENDER
# =========================================

with st.sidebar:
    st.markdown("""<div class="sidebar-header">
<div class="sidebar-logo-icon">
<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#C084FC" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
<path d="M4 7V5a2 2 0 0 1 2-2h2"></path>
<path d="M16 3h2a2 2 0 0 1 2 2v2"></path>
<path d="M16 21h2a2 2 0 0 0 2-2v-2"></path>
<path d="M4 17v2a2 2 0 0 0 2 2h2"></path>
<circle cx="12" cy="10" r="3"></circle>
<path d="M7 19a5 5 0 0 1 10 0"></path>
</svg>
</div>
<div class="sidebar-title">Face Recognition</div>
</div>
<div class="sidebar-label">Navigation</div>""", unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        page_options,
        key="page"
    )

    st.markdown("""<div class="sidebar-footer-box">
<div class="sidebar-company-title">
<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#818CF8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
<path d="M17.5 19.125A9 9 0 0 0 19.5 12c0-4.694-3.806-8.5-8.5-8.5S2.5 7.306 2.5 12a8.96 8.96 0 0 0 2.07 5.75"></path>
<path d="M12 12v9"></path>
<path d="m8 17 4 4 4-4"></path>
</svg>
Code Nimbus
</div>
<div class="sidebar-company-sub">Innovate • Build • Grow</div>
<div class="sidebar-copyright">© 2026 Code Nimbus Solutions.<br>All rights reserved.</div>
</div>""", unsafe_allow_html=True)


# =========================================
# HOME PAGE
# =========================================

if page == "🏠 Home":

    # ---------- TOP BADGE & HERO ----------
    st.markdown("""<div class="top-badge-row">
<div class="ai-badge">
<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#C084FC" stroke-width="2.2">
<path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm0 14a4 4 0 1 1 4-4 4 4 0 0 1-4 4z"/>
</svg>
AI/ML Project
</div>
</div>
<div class="hero-welcome">WELCOME TO</div>
<h1 class="hero-main-title">Face <span class="gradient-text">Recognition</span> System</h1>
<div class="hero-subtitle">Enroll • Identify • Verify</div>
<div class="hero-description">
A simple and efficient face recognition system that detects faces, generates facial embeddings, and identifies individuals by matching them with enrolled data using AI-powered computer vision.
</div>""", unsafe_allow_html=True)

    # ---------- VISUAL CENTERPIECE (FUTURISTIC SVG - UNINDENTED HTML) ----------
    centerpiece_svg_html = """<div style="display: flex; justify-content: center; margin: 15px 0 35px 0;">
<svg width="600" height="260" viewBox="0 0 600 260" fill="none" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto;">
<circle cx="300" cy="130" r="110" fill="url(#centerGlow)" opacity="0.45" />
<defs>
<radialGradient id="centerGlow" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(300 130) rotate(90) scale(110)">
<stop stop-color="#8B5CF6" stop-opacity="0.6"/>
<stop offset="0.6" stop-color="#3B82F6" stop-opacity="0.2"/>
<stop offset="1" stop-color="#090C15" stop-opacity="0"/>
</radialGradient>
<linearGradient id="purpleGlowCard" x1="0" y1="0" x2="48" y2="48" gradientUnits="userSpaceOnUse">
<stop stop-color="#1E1B4B" stop-opacity="0.9"/>
<stop offset="1" stop-color="#311042" stop-opacity="0.8"/>
</linearGradient>
<linearGradient id="blueGlowCard" x1="0" y1="0" x2="48" y2="48" gradientUnits="userSpaceOnUse">
<stop stop-color="#0F172A" stop-opacity="0.9"/>
<stop offset="1" stop-color="#1E293B" stop-opacity="0.8"/>
</linearGradient>
<filter id="shadowGlow" x="0" y="0" width="600" height="260" filterUnits="userSpaceOnUse">
<feDropShadow dx="0" dy="4" stdDeviation="10" flood-color="#8B5CF6" flood-opacity="0.25"/>
</filter>
</defs>
<g opacity="0.15">
<circle cx="150" cy="40" r="1.5" fill="#818CF8"/>
<circle cx="200" cy="40" r="1.5" fill="#818CF8"/>
<circle cx="400" cy="40" r="1.5" fill="#818CF8"/>
<circle cx="450" cy="40" r="1.5" fill="#818CF8"/>
<circle cx="120" cy="130" r="1.5" fill="#818CF8"/>
<circle cx="480" cy="130" r="1.5" fill="#818CF8"/>
<circle cx="150" cy="220" r="1.5" fill="#818CF8"/>
<circle cx="450" cy="220" r="1.5" fill="#818CF8"/>
</g>
<g transform="translate(140, 45)" filter="url(#shadowGlow)">
<rect width="46" height="46" rx="14" fill="url(#purpleGlowCard)" stroke="rgba(168, 85, 247, 0.4)" stroke-width="1.2"/>
<circle cx="21" cy="20" r="5" stroke="#C084FC" stroke-width="1.8" fill="none"/>
<path d="M13 32C13 28 17 26 21 26C23 26 24.5 26.5 25.5 27.5" stroke="#C084FC" stroke-width="1.8" stroke-linecap="round"/>
<path d="M29 20H33M31 18V22" stroke="#C084FC" stroke-width="1.8" stroke-linecap="round"/>
</g>
<g transform="translate(414, 45)" filter="url(#shadowGlow)">
<rect width="46" height="46" rx="14" fill="url(#blueGlowCard)" stroke="rgba(59, 130, 246, 0.4)" stroke-width="1.2"/>
<circle cx="21" cy="21" r="7" stroke="#60A5FA" stroke-width="2" fill="none"/>
<path d="M26 26L32 32" stroke="#60A5FA" stroke-width="2" stroke-linecap="round"/>
</g>
<g transform="translate(140, 165)" filter="url(#shadowGlow)">
<rect width="46" height="46" rx="14" fill="url(#blueGlowCard)" stroke="rgba(59, 130, 246, 0.4)" stroke-width="1.2"/>
<path d="M23 14L15 17.5V23.5C15 28.5 18.5 32 23 33.5C27.5 32 31 28.5 31 23.5V17.5L23 14Z" stroke="#38BDF8" stroke-width="1.8" fill="none"/>
<path d="M19.5 23.5L22 26L26.5 21" stroke="#38BDF8" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
</g>
<g transform="translate(414, 165)" filter="url(#shadowGlow)">
<rect width="46" height="46" rx="14" fill="url(#purpleGlowCard)" stroke="rgba(168, 85, 247, 0.4)" stroke-width="1.2"/>
<rect x="15" y="25" width="4" height="9" rx="1" fill="#C084FC"/>
<rect x="21" y="20" width="4" height="14" rx="1" fill="#C084FC"/>
<rect x="27" y="16" width="4" height="18" rx="1" fill="#818CF8"/>
</g>
<path d="M225 65 H205 V85" stroke="#818CF8" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M375 65 H395 V85" stroke="#818CF8" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M225 195 H205 V175" stroke="#818CF8" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M375 195 H395 V175" stroke="#818CF8" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
<g transform="translate(300, 130)">
<path d="M0 -55 C28 -55 42 -35 42 -5 C42 22 24 45 0 52 C-24 45 -42 22 -42 -5 C-42 -35 -28 -55 0 -55 Z" stroke="#60A5FA" stroke-width="1.8" stroke-dasharray="3 3" fill="none" opacity="0.75"/>
<path d="M0 -48 C22 -48 34 -30 34 -3 C34 20 18 38 0 44 C-18 38 -34 20 -34 -3 C-34 -30 -22 -48 0 -48 Z" stroke="#C084FC" stroke-width="1.2" fill="none" opacity="0.9"/>
<path d="M-30 -20 Q0 -15 30 -20" stroke="#818CF8" stroke-width="1" fill="none" opacity="0.5"/>
<path d="M-33 -5 Q0 2 33 -5" stroke="#818CF8" stroke-width="1" fill="none" opacity="0.5"/>
<path d="M-28 12 Q0 20 28 12" stroke="#818CF8" stroke-width="1" fill="none" opacity="0.5"/>
<path d="M-18 28 Q0 35 18 28" stroke="#818CF8" stroke-width="1" fill="none" opacity="0.5"/>
<path d="M0 -48 V44" stroke="#818CF8" stroke-width="1" opacity="0.4"/>
<path d="M-16 -40 C-12 -10 -12 10 -12 36" stroke="#818CF8" stroke-width="1" fill="none" opacity="0.35"/>
<path d="M16 -40 C12 -10 12 10 12 36" stroke="#818CF8" stroke-width="1" fill="none" opacity="0.35"/>
<circle cx="-14" cy="-14" r="3" fill="#38BDF8"/>
<circle cx="-14" cy="-14" r="7" stroke="#38BDF8" stroke-width="1" fill="none" opacity="0.6"/>
<circle cx="14" cy="-14" r="3" fill="#38BDF8"/>
<circle cx="14" cy="-14" r="7" stroke="#38BDF8" stroke-width="1" fill="none" opacity="0.6"/>
<circle cx="0" cy="2" r="2.5" fill="#F472B6"/>
<circle cx="-12" cy="20" r="2" fill="#C084FC"/>
<circle cx="12" cy="20" r="2" fill="#C084FC"/>
<circle cx="0" cy="22" r="2" fill="#C084FC"/>
<path d="M-14 -14 L0 2 L14 -14" stroke="#38BDF8" stroke-width="1" opacity="0.6"/>
<path d="M-12 20 L0 22 L12 20" stroke="#C084FC" stroke-width="1" opacity="0.6"/>
<path d="M0 2 L0 22" stroke="#F472B6" stroke-width="1" opacity="0.5"/>
</g>
<line x1="200" y1="120" x2="400" y2="120" stroke="url(#scanLine)" stroke-width="1.5" opacity="0.8"/>
<defs>
<linearGradient id="scanLine" x1="200" y1="0" x2="400" y2="0" gradientUnits="userSpaceOnUse">
<stop stop-color="#38BDF8" stop-opacity="0"/>
<stop offset="0.5" stop-color="#38BDF8" stop-opacity="0.9"/>
<stop offset="1" stop-color="#38BDF8" stop-opacity="0"/>
</linearGradient>
</defs>
</svg>
</div>"""

    st.markdown(centerpiece_svg_html, unsafe_allow_html=True)

    # ---------- FEATURE HIGHLIGHTS (4 COLUMNS) ----------
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""<div class="feature-highlight-card">
<div class="feature-highlight-icon">👁️</div>
<div class="feature-highlight-title">Face Detection</div>
<div class="feature-highlight-sub">Using OpenCV YuNet</div>
</div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""<div class="feature-highlight-card">
<div class="feature-highlight-icon">🧠</div>
<div class="feature-highlight-title">Face Embeddings</div>
<div class="feature-highlight-sub">128-dimensional</div>
</div>""", unsafe_allow_html=True)

    with col3:
        st.markdown("""<div class="feature-highlight-card">
<div class="feature-highlight-icon">🔎</div>
<div class="feature-highlight-title">Similarity Matching</div>
<div class="feature-highlight-sub">Cosine similarity</div>
</div>""", unsafe_allow_html=True)

    with col4:
        st.markdown("""<div class="feature-highlight-card">
<div class="feature-highlight-icon">🛡️</div>
<div class="feature-highlight-title">Unknown Rejection</div>
<div class="feature-highlight-sub">Threshold-based verification</div>
</div>""", unsafe_allow_html=True)

    st.write("")
    st.write("")

    # ---------- TWO MAIN ACTION CARDS ----------
    col_enroll, col_recognize = st.columns(2)

    with col_enroll:
        st.markdown("""<div class="action-card-box">
<div class="action-circle-purple">
<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
<path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"></path>
<circle cx="12" cy="7" r="4"></circle>
<line x1="19" y1="8" x2="19" y2="14"></line>
<line x1="16" y1="11" x2="22" y2="11"></line>
</svg>
</div>
<div class="action-card-title">Enroll</div>
<div class="action-card-desc">
Add a new person to the system by capturing their face and storing the facial embedding.
</div>
</div>""", unsafe_allow_html=True)

        st.markdown('<div class="btn-purple-wrapper">', unsafe_allow_html=True)
        st.button("Go to Enrollment →", key="home_goto_enroll",
                  on_click=nav_to, args=("👤 Enrollment",))
        st.markdown('</div>', unsafe_allow_html=True)

    with col_recognize:
        st.markdown("""<div class="action-card-box">
<div class="action-circle-blue">
<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
<circle cx="11" cy="11" r="8"></circle>
<line x1="21" y1="21" x2="16.65" y2="16.65"></line>
<path d="M11 8v6M8 11h6"></path>
</svg>
</div>
<div class="action-card-title">Recognize</div>
<div class="action-card-desc">
Identify a person by matching their face with the enrolled database.
</div>
</div>""", unsafe_allow_html=True)

        st.markdown('<div class="btn-blue-wrapper">', unsafe_allow_html=True)
        st.button("Go to Recognition →", key="home_goto_recognize",
                  on_click=nav_to, args=("🔎 Recognition",))
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------- FOOTER ----------
    st.markdown("""<div class="footer-wrapper">
<div class="footer-quote">“AI sees faces, we see possibilities.” — Code Nimbus Solutions</div>
<div class="footer-sub">© 2026 Code Nimbus Solutions. All rights reserved.</div>
</div>""", unsafe_allow_html=True)


# =========================================
# ENROLLMENT PAGE
# =========================================

elif page == "👤 Enrollment":

    st.markdown('<h1 style="font-size: 34px; font-weight: 800; color: #FFFFFF; margin-bottom: 4px;">👤 Face Enrollment</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #94A3B8; font-size: 15px; margin-bottom: 25px;">Add a person\'s face to the recognition database.</p>', unsafe_allow_html=True)

    st.divider()

    name = st.text_input(
        "Person's name"
    )

    source = st.radio(
        "Choose image source",
        [
            "📁 Upload Image",
            "📷 Camera"
        ]
    )

    uploaded_file = None
    camera_image = None

    if source == "📁 Upload Image":
        uploaded_file = st.file_uploader(
            "Upload a face image",
            type=["jpg", "jpeg", "png"]
        )
    else:
        camera_image = st.camera_input(
            "Take a photo"
        )

    st.write("")

    if st.button("➕ Add Face Sample"):
        if not name:
            st.warning("Please enter the person's name.")
            st.stop()

        if source == "📁 Upload Image":
            if uploaded_file is None:
                st.warning("Please upload an image.")
                st.stop()
            file_bytes = uploaded_file.read()
        else:
            if camera_image is None:
                st.warning("Please take a photo.")
                st.stop()
            file_bytes = camera_image.getvalue()

        image = cv2.imdecode(
            np.frombuffer(file_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR
        )

        st.image(
            image,
            channels="BGR",
            caption="Input Image"
        )

        height, width = image.shape[:2]
        detector.setInputSize((width, height))

        _, faces = detector.detect(image)

        if faces is None:
            st.error("❌ No face detected.")
            st.stop()

        if len(faces) > 1:
            st.error(
                "❌ Multiple faces detected. Please use an image containing only one person.")
            st.stop()

        face = faces[0]
        aligned_face = recognizer.alignCrop(image, face)
        embedding = recognizer.feature(aligned_face)

        save_embedding(name, embedding)

        st.success(f"✓ Face sample added for {name}!")


# =========================================
# RECOGNITION PAGE
# =========================================

elif page == "🔎 Recognition":

    st.markdown('<h1 style="font-size: 34px; font-weight: 800; color: #FFFFFF; margin-bottom: 4px;">🔎 Face Recognition</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #94A3B8; font-size: 15px; margin-bottom: 25px;">Identify a person from the enrolled database.</p>', unsafe_allow_html=True)

    st.divider()

    source = st.radio(
        "Choose image source",
        [
            "📁 Upload Image",
            "📷 Camera"
        ]
    )

    recognition_file = None
    recognition_camera = None

    if source == "📁 Upload Image":
        recognition_file = st.file_uploader(
            "Upload image for recognition",
            type=["jpg", "jpeg", "png"],
            key="recognition_upload"
        )
    else:
        recognition_camera = st.camera_input(
            "Take a photo for recognition",
            key="recognition_camera"
        )

    st.write("")

    if st.button("🔍 Recognize Face"):
        if source == "📁 Upload Image":
            if recognition_file is None:
                st.warning("Please upload an image.")
                st.stop()
            file_bytes = recognition_file.read()
        else:
            if recognition_camera is None:
                st.warning("Please take a photo.")
                st.stop()
            file_bytes = recognition_camera.getvalue()

        image = cv2.imdecode(
            np.frombuffer(file_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR
        )

        height, width = image.shape[:2]
        detector.setInputSize((width, height))

        _, faces = detector.detect(image)

        if faces is None:
            st.error("❌ No face detected.")
            st.stop()

        if len(faces) > 1:
            st.error(
                "❌ Multiple faces detected. Please use an image containing one person.")
            st.stop()

        face = faces[0]
        x, y, w, h = face[:4].astype(int)

        display_img = image.copy()
        cv2.rectangle(
            display_img,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        small_img = cv2.resize(display_img, (400, 400))
        st.image(
            cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB),
            caption="Detected Face"
        )

        aligned_face = recognizer.alignCrop(image, face)
        query_embedding = recognizer.feature(aligned_face)

        database = load_database()

        if not database:
            st.error("❌ No enrolled people found.")
            st.stop()

        best_name = "UNKNOWN"
        best_score = -1
        scores = {}

        for person_name, stored_embeddings in database.items():
            person_best_score = -1
            for stored_embedding in stored_embeddings:
                stored_embedding = np.array(
                    stored_embedding,
                    dtype=np.float32
                ).reshape(1, -1)

                score = recognizer.match(
                    query_embedding,
                    stored_embedding,
                    cv2.FaceRecognizerSF_FR_COSINE
                )
                score = float(score)

                if score > person_best_score:
                    person_best_score = score

            scores[person_name] = person_best_score

            if person_best_score > best_score:
                best_score = person_best_score
                best_name = person_name

        st.divider()
        st.subheader("Recognition Result")

        if best_score >= THRESHOLD:
            st.success(f"✓ KNOWN PERSON: {best_name}")
            st.metric(
                "Similarity Score",
                f"{best_score:.4f}"
            )
            st.write(f"**Matched Person:** {best_name}")
        else:
            st.error("❌ UNKNOWN PERSON")
            st.metric(
                "Best Similarity Score",
                f"{best_score:.4f}"
            )
            st.write("No enrolled person matched strongly enough.")

        st.caption(f"Recognition threshold: {THRESHOLD:.2f}")

        with st.expander("Show comparison details"):
            for person_name, score in scores.items():
                st.write(f"**{person_name}:** {score:.4f}")
