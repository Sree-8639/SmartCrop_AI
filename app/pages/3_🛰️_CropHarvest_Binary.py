"""
SmartCrop AI - Page 3: CropHarvest Satellite Binary Classification
Predicts whether a 12-month multi-spectral satellite time-series corresponds to actively cultivated crop land.
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
from app.services.cropharvest_service import predict_cropharvest_binary_real, parse_and_validate_satellite_array
from app.services.lfm3_vl_service import LFM3VLService

st.set_page_config(page_title="CropHarvest Binary Detection — SmartCrop AI", page_icon="🛰️", layout="wide", initial_sidebar_state="expanded")

apply_custom_theme()
render_sidebar_branding()

render_page_header(
    title="CropHarvest Satellite Binary Crop Detection",
    subtitle="Classifies geographic pixels into Crop vs Non-Crop using 12 monthly timesteps of 18-channel Earth Observation multi-spectral imagery.",
    category_badge="3-Seed TSTransformer • 82.02% Test Acc",
    icon="🛰️"
)

# Collapsible Technical Specs
with st.expander("ℹ️ Satellite Multi-Spectral Input Specifications (12 Timesteps × 18 Channels)", expanded=False):
    st.markdown(
        """
        - **Temporal Dimension:** 12 Monthly Timesteps (January - December)
        - **Spectral & Climatological Channels (18):**
          - *Sentinel-2 Optical (10):* B2 (Blue), B3 (Green), B4 (Red), B5, B6, B7, B8 (NIR), B8A, B11 (SWIR-1), B12 (SWIR-2)
          - *Sentinel-1 SAR (2):* Synthetic Aperture Radar VV & VH Backscatter
          - *ERA5-Land Climate (5):* Temperature, Precipitation, 10m U/V Wind, Dewpoint
          - *SRTM Topography (1):* Digital Elevation Model (DEM)
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
    key="cropharvest_binary_uploader"
)

if filename is not None:
    st.divider()
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        run_bin = st.button("🛰️ Evaluate Earth Observation Time-Series", type="primary", use_container_width=True)

    if run_bin:
        with st.spinner("Processing 12x18 multi-modal tensor through 3-Seed TSTransformer Ensemble..."):
            try:
                result = predict_cropharvest_binary_real(file_bytes=raw_bytes, filename=filename)
                st.session_state["cropharvest_binary_result"] = result
                st.session_state["cropharvest_binary_filename"] = filename
                # Generate visual time-series chart for multimodal grounding
                try:
                    arr = parse_and_validate_satellite_array(raw_bytes, filename)
                    chart_img = LFM3VLService.render_cropharvest_timeseries_chart(arr, filename=filename, title="Crop vs Non-Crop Phenology")
                    st.session_state["cropharvest_binary_chart"] = chart_img
                except Exception:
                    st.session_state["cropharvest_binary_chart"] = None
            except Exception as e:
                st.error(f"Binary inference failure: {e}")
                st.stop()

    if "cropharvest_binary_result" in st.session_state:
        result = st.session_state["cropharvest_binary_result"]
        render_prediction_summary(
            title="Satellite Land-Cover Classification",
            predicted_label=result.label,
            confidence=result.confidence,
            latency_ms=result.execution_time_ms,
            extra_tag="Crop Ground Truth Detection" if result.is_crop else "Non-Agricultural Terrain",
            is_mock=result.is_mock,
            status_notice=result.status_notice
        )

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Crop Probability", f"{result.crop_probability * 100:.2f}%")
        with c2:
            st.metric("Non-Crop Probability", f"{result.non_crop_probability * 100:.2f}%")

        render_top_k_distribution(result.top_k, title="Binary State Probabilities")

        st.info(
            "💡 **Ensemble Architecture:** Evaluated by 3-Seed `TSTransformer` Ensemble trained on official `is_test=true` splits. "
            "Verified test accuracy: **82.02%**, Macro-F1: **81.18%**."
        )

        # AI-Powered Explanation (LiquidAI / LFM3-VL-2B)
        chart_img = st.session_state.get("cropharvest_binary_chart")
        if chart_img is not None:
            st.image(chart_img, caption="Multi-Spectral Phenological Trajectory (Provided to LFM3-VL-2B as visual evidence)", use_container_width=True)

        render_ai_explanation_section(
            module_name="cropharvest_binary",
            prediction_result={
                "label": result.label,
                "is_crop": result.is_crop,
                "confidence": result.confidence,
                "crop_probability": result.crop_probability,
                "non_crop_probability": result.non_crop_probability,
                "timesteps_verified": result.timesteps_verified
            },
            structured_evidence={
                "filename": st.session_state.get("cropharvest_binary_filename", filename),
                "channels": result.channels_verified,
                "timesteps": result.timesteps_verified
            },
            image=chart_img,
            key_prefix="cropharvest_binary"
        )
else:
    st.info("👆 Please select or upload a satellite time-series file (.npy / .h5) above to begin analysis.")
