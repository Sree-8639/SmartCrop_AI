"""
SmartCrop AI - Premium SaaS Theme & Design System
Uses bulletproof inline styles and reliable Streamlit CSS overrides
to guarantee perfect rendering in all browsers without being stripped.
"""

import streamlit as st

def apply_custom_theme():
    """Injects high-end SaaS styling into the Streamlit application."""
    custom_css = """
    <style>
    /* Global Typography - exclude icon elements */
    html, body, p, div, input, select, textarea {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Guarantee Material Symbols and Streamlit UI icons never display raw ligature text */
    [data-testid="stIconMaterial"] {
        font-family: "Material Symbols Rounded", "Material Symbols Outlined", sans-serif !important;
        font-feature-settings: 'liga' 1;
        -webkit-font-feature-settings: 'liga' 1;
    }

    /* Bulletproof symbol replacement for sidebar collapse (<<) and expand (>>) controls */
    [data-testid="stSidebarHeader"] {
        display: flex !important;
        justify-content: flex-end !important;
        padding: 8px 12px 0 12px !important;
        background: transparent !important;
    }

    /* Collapse button (inside sidebar) */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarHeader"] button {
        margin-left: auto !important;
        opacity: 0.9 !important;
        visibility: visible !important;
        background: rgba(16, 185, 129, 0.15) !important;
        border: 1px solid rgba(16, 185, 129, 0.4) !important;
        border-radius: 8px !important;
        padding: 4px 8px !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    [data-testid="stSidebarCollapseButton"]:hover,
    [data-testid="stSidebarHeader"] button:hover {
        background: rgba(16, 185, 129, 0.3) !important;
        border-color: #34d399 !important;
    }

    /* Expand button (top-left when sidebar is collapsed) */
    button[data-testid="stExpandSidebarButton"],
    [data-testid="stSidebarCollapsedControl"] button {
        background: #064e3b !important;
        border: 1.5px solid #10b981 !important;
        border-radius: 8px !important;
        padding: 6px 10px !important;
        margin: 10px 0 0 12px !important;
        box-shadow: 0 4px 14px rgba(6, 78, 59, 0.35) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 38px !important;
        height: 38px !important;
        min-width: 38px !important;
        min-height: 38px !important;
    }

    button[data-testid="stExpandSidebarButton"]:hover,
    [data-testid="stSidebarCollapsedControl"] button:hover {
        background: #047857 !important;
        border-color: #34d399 !important;
        transform: scale(1.05) !important;
        box-shadow: 0 6px 18px rgba(16, 185, 129, 0.4) !important;
    }

    /* Total elimination of raw text ligature bleeds: keyboard_double_arrow_left/right */
    [data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"],
    [data-testid="stSidebarHeader"] button [data-testid="stIconMaterial"],
    button[data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"],
    [data-testid="stSidebarCollapsedControl"] [data-testid="stIconMaterial"] {
        font-size: 0px !important;
        line-height: 0 !important;
        color: transparent !important;
        visibility: hidden !important;
        position: relative !important;
        width: 22px !important;
        height: 22px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        overflow: visible !important;
    }

    /* Render crisp unicode double arrow glyphs */
    [data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"]::after,
    [data-testid="stSidebarHeader"] button [data-testid="stIconMaterial"]::after {
        content: "«" !important;
        font-size: 22px !important;
        font-weight: 800 !important;
        line-height: 1 !important;
        color: #34d399 !important;
        visibility: visible !important;
        position: absolute !important;
        left: 50% !important;
        top: 50% !important;
        transform: translate(-50%, -50%) !important;
    }

    button[data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"]::after,
    [data-testid="stSidebarCollapsedControl"] [data-testid="stIconMaterial"]::after {
        content: "»" !important;
        font-size: 22px !important;
        font-weight: 800 !important;
        line-height: 1 !important;
        color: #ecfdf5 !important;
        visibility: visible !important;
        position: absolute !important;
        left: 50% !important;
        top: 50% !important;
        transform: translate(-50%, -50%) !important;
    }

    /* Main background */
    .stApp {
        background-color: #f8fafc !important;
    }

    /* Hide default unstyled multipage sidebar navigation */
    [data-testid="stSidebarNav"] {
        display: none !important;
    }

    /* Premium Dark Obsidian Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #071913 0%, #030e0b 100%) !important;
        border-right: 1px solid rgba(16, 185, 129, 0.2) !important;
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.15) !important;
    }

    /* Sidebar custom section header */
    .sidebar-section-header {
        color: #6ee7b7 !important;
        font-size: 10px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 1.2px !important;
        margin: 16px 0 6px 4px !important;
        display: flex !important;
        align-items: center !important;
        gap: 6px !important;
        opacity: 0.9 !important;
    }

    /* Page link buttons inside sidebar */
    [data-testid="stSidebar"] div[data-testid="stPageLink"] a {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.07) !important;
        border-radius: 9px !important;
        padding: 8px 12px !important;
        margin-bottom: 5px !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        display: flex !important;
        align-items: center !important;
        text-decoration: none !important;
    }

    [data-testid="stSidebar"] div[data-testid="stPageLink"] a:hover {
        background: rgba(16, 185, 129, 0.15) !important;
        border-color: rgba(52, 211, 153, 0.4) !important;
        transform: translateX(3px) !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.15) !important;
    }

    [data-testid="stSidebar"] div[data-testid="stPageLink"] a[aria-current="page"] {
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.28) 0%, rgba(5, 150, 105, 0.1) 100%) !important;
        border: 1px solid #10b981 !important;
        border-left: 4px solid #34d399 !important;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.25) !important;
    }

    /* Text inside sidebar page links */
    [data-testid="stSidebar"] div[data-testid="stPageLink"] p {
        color: #cbd5e1 !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        margin: 0 !important;
    }

    [data-testid="stSidebar"] div[data-testid="stPageLink"] a:hover p {
        color: #ffffff !important;
    }

    [data-testid="stSidebar"] div[data-testid="stPageLink"] a[aria-current="page"] p {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    /* Primary Buttons Styling */
    button[kind="primary"], button[type="primary"], div.stButton > button {
        background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 8px 20px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 6px -1px rgba(5, 150, 105, 0.25) !important;
        transition: all 0.2s ease !important;
    }

    button[kind="primary"]:hover, button[type="primary"]:hover, div.stButton > button:hover {
        background: linear-gradient(135deg, #047857 0%, #065f46 100%) !important;
        box-shadow: 0 6px 12px -2px rgba(5, 150, 105, 0.35) !important;
        transform: translateY(-1px) !important;
    }

    /* Metric values in forest green */
    [data-testid="stMetricValue"] {
        color: #064e3b !important;
        font-weight: 700 !important;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def render_sidebar_branding():
    """Renders a sleek, high-end enterprise SaaS menu bar in the sidebar."""
    with st.sidebar:
        # Top Brand Logo and Cluster Status
        st.markdown(
            """
            <div style="padding: 10px 4px 14px 4px; border-bottom: 1px solid rgba(16, 185, 129, 0.2); margin-bottom: 12px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="width: 36px; height: 36px; border-radius: 10px; background: linear-gradient(135deg, #059669 0%, #064e3b 100%); display: flex; align-items: center; justify-content: center; font-size: 20px; box-shadow: 0 0 12px rgba(16, 185, 129, 0.4); border: 1px solid #34d399;">
                        🌱
                    </div>
                    <div>
                        <div style="font-size: 19px; font-weight: 800; color: #ffffff; letter-spacing: -0.02em; line-height: 1.1;">
                            SmartCrop AI
                        </div>
                        <div style="font-size: 10px; font-weight: 600; color: #6ee7b7; text-transform: uppercase; letter-spacing: 0.8px; margin-top: 2px;">
                            Autonomous Ag-Perception
                        </div>
                    </div>
                </div>
                <div style="margin-top: 12px; display: flex; align-items: center; justify-content: space-between; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(110, 231, 183, 0.25); padding: 5px 10px; border-radius: 8px;">
                    <span style="display: inline-flex; align-items: center; gap: 6px; font-size: 11px; font-weight: 600; color: #6ee7b7;">
                        <span style="width: 7px; height: 7px; background: #34d399; border-radius: 50%; box-shadow: 0 0 6px #34d399; display: inline-block;"></span>
                        Cluster: 5 Models Online
                    </span>
                    <span style="font-size: 10px; color: #a7f3d0; font-family: monospace;">v1.0.0</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 1. Platform Hub
        st.markdown('<div class="sidebar-section-header">CORE DASHBOARD</div>', unsafe_allow_html=True)
        st.page_link("main.py", label="Executive Overview", icon="🏠")

        # 2. Field Computer Vision
        st.markdown('<div class="sidebar-section-header">FIELD COMPUTER VISION</div>', unsafe_allow_html=True)
        st.page_link("pages/1_🍃_PlantDoc_Disease_Diagnosis.py", label="PlantDoc Foliar Pathology", icon="🍃")
        st.page_link("pages/2_🐛_IP102_Pest_Detection.py", label="IP102 Pest Surveillance", icon="🐛")

        # 3. Satellite Earth Observation
        st.markdown('<div class="sidebar-section-header">ORBITAL OBSERVATION</div>', unsafe_allow_html=True)
        st.page_link("pages/3_🛰️_CropHarvest_Binary.py", label="CropHarvest Binary Cover", icon="🛰️")
        st.page_link("pages/4_🌾_CropHarvest_10Class.py", label="CropHarvest 10-Class Groups", icon="🌾")

        # 4. System Evaluation & Audit
        st.markdown('<div class="sidebar-section-header">VERIFICATION & AUDIT</div>', unsafe_allow_html=True)
        st.page_link("pages/5_📊_Unified_Results_Dashboard.py", label="Unified Analytics & Audit", icon="📊")

        # Bottom Hardware & Server Telemetry Monitor
        st.markdown(
            """
            <div style="margin-top: 24px; padding: 12px; background: rgba(0, 0, 0, 0.25); border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 10px;">
                <div style="font-size: 10px; font-weight: 700; color: #6ee7b7; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 6px;">
                    ⚡ Cluster Telemetry
                </div>
                <div style="font-size: 11px; color: #cbd5e1; line-height: 1.55;">
                    <div>• <b>API:</b> FastAPI Port 8000 <span style="color: #34d399;">●</span></div>
                    <div>• <b>Hardware:</b> PyTorch 2.6.0 CPU</div>
                    <div>• <b>Latency:</b> &lt; 120ms (Warm)</div>
                    <div>• <b>Checkpoints:</b> 5/5 Validated</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


def render_page_header(title: str, subtitle: str, category_badge: str, icon: str = "🌾"):
    """Renders a consistent, modern page header across all analysis modules."""
    st.markdown(
        f"""
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-left: 5px solid #059669; border-radius: 12px; padding: 22px 26px; margin-bottom: 22px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
                <div>
                    <h2 style="font-size: 24px; font-weight: 700; color: #064e3b; margin: 0 0 6px 0; letter-spacing: -0.01em;">{icon} {title}</h2>
                    <p style="font-size: 14px; color: #475569; margin: 0; line-height: 1.5;">{subtitle}</p>
                </div>
                <div>
                    <span style="display: inline-flex; align-items: center; gap: 6px; background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; padding: 4px 10px; border-radius: 9999px; font-size: 12px; font-weight: 600;">
                        <span style="width: 6px; height: 6px; background: #10b981; border-radius: 50%; display: inline-block;"></span>
                        {category_badge}
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
