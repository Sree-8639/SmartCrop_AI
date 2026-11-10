"""
SmartCrop AI - Page 4: CropHarvest 10-Class Crop Group & Land-Cover Classification
Categorizes satellite time-series across 9 FAO crop groups and 1 land-cover class.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from app.components.theme import apply_custom_theme, render_sidebar_branding, render_page_header
from app.components.uploader import render_array_uploader
from app.components.result_card import render_prediction_summary, render_top_k_distribution
from app.components.explanation_card import render_ai_explanation_section
from app.services.cropharvest_service import predict_cropharvest_10class_real, parse_and_validate_satellite_array
from app.services.lfm3_vl_service import LFM3VLService

st.set_page_config(page_title="CropHarvest 10-Class Classification — SmartCrop AI", page_icon="🌾", layout="wide", initial_sidebar_state="expanded")

apply_custom_theme()
render_sidebar_branding()

render_page_header(
    title="CropHarvest 10-Class Crop Group & Land-Cover Categorization",
    subtitle="Categorizes 12-month multi-spectral satellite time-series across 9 official FAO crop categories plus 1 distinct non-crop land-cover category using CropHarvestBiLSTM.",
    category_badge="CropHarvestBiLSTM • 10 Classes",
    icon="🌾"
)

# Taxonomy Reference
with st.expander("🌿 Class Hierarchy Taxonomy (9 Crop Groups + 1 Land Cover)", expanded=False):
    st.markdown(
        """
        - **9 Cultivated FAO Crop Groups:** `cereals`, `oilseeds`, `leguminous`, `root_tuber`, `fruits_nuts`, `vegetables_melons`, `beverage_spice`, `sugar`, `other`
        - **1 Land-Cover Class:** `non_crop` *(Forest, water, urban infrastructure, barren, grassland, scrub)*
        """
    )

BASE_DIR = Path(__file__).resolve().parent.parent.parent
crop_dir = BASE_DIR / "datasets" / "cropharvest"
sample_arrays = []
if crop_dir.exists():
    for f in sorted(list(crop_dir.glob("*.npy"))):
        sample_arrays.append((f"Satellite Pixel: {f.name}", str(f)))
    for h in ["togo-eval.h5", "china-crop.h5"]:
        hp = crop_dir / h
        if hp.exists():
            sample_arrays.append((f"CropHarvest Region: {h}", str(hp)))

filename, raw_bytes, status_msg = render_array_uploader(
    title="Upload Satellite Time-Series (.h5 / .npy / .csv)",
    sample_arrays=sample_arrays,
    key="cropharvest_10class_uploader"
)

if filename is not None:
    st.divider()
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        run_10c = st.button("🌾 Predict Crop / Land-Cover Group", type="primary", use_container_width=True)

    if run_10c:
        with st.spinner("Executing 10-Class CropHarvest BiLSTM forward pass..."):
            try:
                result = predict_cropharvest_10class_real(file_bytes=raw_bytes, filename=filename)
                st.session_state["cropharvest_10c_result"] = result
                st.session_state["cropharvest_10c_filename"] = filename
                # Generate visual time-series chart for multimodal grounding
                try:
                    arr = parse_and_validate_satellite_array(raw_bytes, filename)
                    chart_img = LFM3VLService.render_cropharvest_timeseries_chart(arr, filename=filename, title="10-Class Crop & Land-Cover Phenology")
                    st.session_state["cropharvest_10c_chart"] = chart_img
                except Exception:
                    st.session_state["cropharvest_10c_chart"] = None
            except Exception as e:
                st.error(f"10-Class inference failure: {e}")
                st.stop()

    if "cropharvest_10c_result" in st.session_state:
        result = st.session_state["cropharvest_10c_result"]
        render_prediction_summary(
            title="Predicted Category",
            predicted_label=result.predicted_class,
            confidence=result.confidence,
            latency_ms=result.execution_time_ms,
            extra_tag=f"{result.category_type} (Class #{result.class_index})",
            is_mock=result.is_mock,
            status_notice=result.status_notice
        )

        st.markdown("---")
        render_top_k_distribution(result.top_k, title="Top-5 Multi-Class Softmax Distribution")

        # AI-Powered Explanation (LiquidAI / LFM3-VL-2B)
        chart_img = st.session_state.get("cropharvest_10c_chart")
        if chart_img is not None:
            st.image(chart_img, caption="Multi-Spectral Phenological Trajectory (Provided to LFM3-VL-2B as visual evidence)", use_container_width=True)

        render_ai_explanation_section(
            module_name="cropharvest_10class",
            prediction_result={
                "predicted_class": result.predicted_class,
                "confidence": result.confidence,
                "category_type": result.category_type,
                "class_index": result.class_index,
                "top_k": [k.model_dump() if hasattr(k, "model_dump") else k.__dict__ for k in result.top_k]
            },
            structured_evidence={
                "filename": st.session_state.get("cropharvest_10c_filename", filename),
                "channels": result.channels_verified,
                "timesteps": result.timesteps_verified
            },
            image=chart_img,
            key_prefix="cropharvest_10class"
        )
else:
    st.info("👆 Please select or upload a satellite time-series file (.npy / .h5) above to initiate 10-class analysis.")
