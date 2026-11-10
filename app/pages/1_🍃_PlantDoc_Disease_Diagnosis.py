"""
SmartCrop AI - Page 1: PlantDoc Foliar Disease Diagnosis
Provides diagnosis for 27 crop disease and health categories.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from app.components.theme import apply_custom_theme, render_sidebar_branding, render_page_header
from app.components.uploader import render_image_uploader
from app.components.result_card import render_prediction_summary, render_top_k_distribution
from app.services.plantdoc_service import predict_plantdoc_real

st.set_page_config(page_title="PlantDoc Disease Diagnosis — SmartCrop AI", page_icon="🍃", layout="wide", initial_sidebar_state="expanded")

apply_custom_theme()
render_sidebar_branding()

render_page_header(
    title="PlantDoc Foliar Disease Diagnosis",
    subtitle="AI-driven foliar pathology diagnosis across 27 crop disease and healthy leaf classes using a low-overfitting 3-Model ConvNeXt Ensemble (ConvNeXt-S@384 + ConvNeXt-B@384 + ConvNeXt-S@320).",
    category_badge="Experiment E Ensemble • 83.55% Test Acc",
    icon="🍃"
)

# Set up sample images from dataset
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sample_dir = BASE_DIR / "datasets" / "plantdoc" / "train" / "Apple leaf"
sample_images = []
if sample_dir.exists():
    for f in list(sample_dir.glob("*.jpg"))[:3]:
        sample_images.append((f"Apple Leaf Sample: {f.name}", str(f)))

img, filename, raw_bytes = render_image_uploader(
    title="Upload Plant Leaf Image",
    help_text="Upload a clear photograph of a single leaf or foliar lesion. Formats supported: JPG, JPEG, PNG, WEBP",
    sample_images=sample_images,
    key="plantdoc_uploader"
)

if img is not None:
    st.divider()
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        run_prediction = st.button("🔍 Run Disease Diagnosis", type="primary", use_container_width=True)

    if run_prediction:
        with st.spinner("Executing ConvNeXt-Small (384px) neural forward pass..."):
            try:
                result = predict_plantdoc_real(image_input=raw_bytes or img, filename=filename or "leaf.jpg")
            except Exception as e:
                st.error(f"Inference error: {e}")
                st.stop()

        render_prediction_summary(
            title="Diagnosed Condition",
            predicted_label=result.predicted_class,
            confidence=result.confidence,
            latency_ms=result.execution_time_ms,
            extra_tag=f"Severity: {result.disease_severity}",
            is_mock=result.is_mock,
            status_notice=result.status_notice
        )

        st.markdown("---")
        render_top_k_distribution(result.top_k, title="Top-5 Differential Diagnoses")

        st.info(
            "💡 **Production Verification:** Powered by Experiment E 3-Model Ensemble (`expE_convnext_small@384_s42`, "
            "`expE_convnext_base@384_s7`, and `expE_convnext_small@320_s123`). "
            "Verified test accuracy: **83.55%**, Macro-F1: **82.43%**, Validation accuracy: **80.42%**."
        )
else:
    st.info("👆 Please upload a leaf photograph or select a sample image above to initiate diagnosis.")
