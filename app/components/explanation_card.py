"""
SmartCrop AI - Reusable AI Explanation UI Component (LiquidAI / LFM3-VL-2B)
Renders the dedicated "AI-Powered Explanation – LFM3-VL-2B" section with
grounded visual observations, reasoning synthesis, and agronomic disclaimers.
"""

from typing import Dict, List, Optional, Any, Union
import streamlit as st
from PIL import Image

from app.services.lfm3_vl_service import explain_prediction


def render_ai_explanation_section(
    module_name: str,
    prediction_result: Dict[str, Any],
    image: Optional[Union[Image.Image, bytes, str]] = None,
    structured_evidence: Optional[Dict[str, Any]] = None,
    detection_boxes: Optional[List[Dict[str, Any]]] = None,
    key_prefix: str = "lfm3_expl"
):
    """
    Renders an interactive, grounded AI explanation section below any model prediction.
    Only generates explanations when user explicitly clicks 'Generate AI Explanation'.
    """
    st.markdown("---")
    st.markdown("### 🤖 AI-Powered Explanation – LFM3-VL-2B")
    st.caption(
        "Interprets predictions from SmartCrop AI neural backbones using **LiquidAI's LFM3-VL-2B** "
        "Vision-Language Foundation Model. Strictly grounded in model outputs and visual observations."
    )

    btn_key = f"{key_prefix}_btn"
    run_explanation = st.button("🧠 Generate AI Explanation", key=btn_key, type="secondary")

    if run_explanation:
        with st.spinner("Synthesizing grounded explanation via LiquidAI LFM3-VL-2B..."):
            try:
                res = explain_prediction(
                    module_name=module_name,
                    prediction_result=prediction_result,
                    image=image,
                    structured_evidence=structured_evidence,
                    detection_boxes=detection_boxes
                )

                # Status Badge
                badge_bg = "#ecfdf5" if "Active" in res["model_status"] else "#f8fafc"
                badge_border = "#10b981" if "Active" in res["model_status"] else "#94a3b8"
                badge_text = "#065f46" if "Active" in res["model_status"] else "#334155"

                st.markdown(
                    f"""
                    <div style="background-color: {badge_bg}; border: 1px solid {badge_border}; border-radius: 6px; padding: 8px 12px; margin: 10px 0 16px 0; display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 13px; font-weight: 600; color: {badge_text};">
                            🧬 Model Engine: {res['model_status']}
                        </span>
                        <span style="font-size: 12px; color: #64748b;">
                            ⚡ Latency: {res['execution_time_ms']} ms | Grounding Verified ✅
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Explanation Markdown
                st.markdown(res["explanation"])

                # Detailed Observations Expander
                with st.expander("🔍 Observed Visual & Quantitative Evidence", expanded=True):
                    for obs in res.get("observations", []):
                        st.markdown(f"- {obs}")

                # Limitations & Safety Callout
                with st.expander("⚠️ Diagnostic Limitations & Agricultural Advisory", expanded=False):
                    for lim in res.get("limitations", []):
                        st.markdown(f"- {lim}")
                    st.caption(
                        "Disclaimer: SmartCrop AI multimodal explanations are designed for scientific decision-support. "
                        "Always corroborate with certified local agricultural extension officers before applying chemical treatments."
                    )

            except Exception as e:
                st.error(f"⚠️ AI Explanation service encountered an error: {str(e)}")
                st.info("The original prediction above remains verified and unaffected.")
