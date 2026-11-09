"""
SmartCrop AI - Page 2: IP102 Agricultural Pest Diagnosis & Detection
Supports 102 insect and pest species classification and real-time YOLOv8 object detection.
"""

from pathlib import Path
import streamlit as st
import pandas as pd

from app.components.theme import apply_custom_theme, render_sidebar_branding, render_page_header
from app.components.uploader import render_image_uploader
from app.components.result_card import (
    render_prediction_summary, 
    render_top_k_distribution, 
    draw_bounding_boxes_on_image,
)
from app.services.ip102_detection_service import predict_ip102_detection_real
from app.services.ip102_service import predict_ip102_classification_real

st.set_page_config(page_title="IP102 Pest Detection & Classification — SmartCrop AI", page_icon="🐛", layout="wide", initial_sidebar_state="expanded")

apply_custom_theme()
render_sidebar_branding()

render_page_header(
    title="IP102 Agricultural Pest Surveillance",
    subtitle="Dual-engine pest surveillance across 102 agricultural insect species: Single-stage YOLOv8s bounding-box localization and deep ConvNeXt-Small taxonomic classification.",
    category_badge="YOLOv8s & ConvNeXt • 102 Species",
    icon="🐛"
)

# Set up sample images from IP102 test set
BASE_DIR = Path(__file__).resolve().parent.parent.parent
test_dir = BASE_DIR / "datasets" / "ip102" / "Detection" / "yolo" / "images" / "test"
sample_images = []
if test_dir.exists():
    for f in sorted(list(test_dir.glob("*.jpg")))[:4]:
        sample_images.append((f"IP102 Field Sample: {f.name}", str(f)))

img, filename, raw_bytes = render_image_uploader(
    title="Upload Field Pest Photograph",
    help_text="Upload a field image containing one or more insect pests. Supported formats: JPG, JPEG, PNG, WEBP",
    sample_images=sample_images,
    key="ip102_uploader"
)

if img is not None:
    st.divider()

    tab_detect, tab_classify = st.tabs([
        "🎯 Field Pest Object Detection (YOLOv8)",
        "🔬 Taxonomic Species Classification (102 Classes)"
    ])

    with tab_detect:
        st.markdown("#### Real-Time In-Field Pest Localization")
        st.caption("Detects, delineates, and counts multiple insect pests simultaneously in natural field backgrounds.")

        c_slider, c_btn = st.columns([2, 1])
        with c_slider:
            conf_thresh = st.slider("Confidence Detection Threshold", min_value=0.10, max_value=0.90, value=0.25, step=0.05)
        with c_btn:
            st.write("")
            run_det = st.button("🎯 Detect Pests in Image", type="primary", use_container_width=True)

        if run_det:
            with st.spinner("Executing Ultralytics YOLOv8s neural localization pass..."):
                try:
                    det_result = predict_ip102_detection_real(
                        image_input=raw_bytes or img, 
                        filename=filename or "pest.jpg", 
                        conf_threshold=conf_thresh
                    )
                except Exception as e:
                    st.error(f"Detection inference failure: {e}")
                    st.stop()

            col_img, col_metrics = st.columns([1, 1])
            with col_img:
                annotated_img = draw_bounding_boxes_on_image(img, det_result.detected_boxes)
                st.image(annotated_img, caption=f"YOLOv8 Localization: {det_result.total_pests_detected} targets identified", width="stretch")
            
            with col_metrics:
                st.markdown(
                    f"""
                    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.04);">
                        <div style="font-size: 13px; font-weight: 700; color: #64748b; text-transform: uppercase;">Total Detected Pests</div>
                        <div style="font-size: 32px; font-weight: 800; color: #064e3b; margin: 4px 0 12px 0;">{det_result.total_pests_detected}</div>
                        <div style="font-size: 13px; color: #475569;">
                            <strong>Model:</strong> {det_result.model_architecture}<br>
                            <strong>Inference Latency:</strong> <code>{det_result.execution_time_ms:.1f} ms</code>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.write("")

                if det_result.detected_boxes:
                    box_data = []
                    for idx, b in enumerate(det_result.detected_boxes):
                        box_data.append({
                            "Target #": idx + 1,
                            "Pest Species": b.label,
                            "Confidence": f"{b.confidence * 100:.1f}%",
                            "Coordinates (Normalized)": f"[{b.x1:.2f}, {b.y1:.2f}, {b.x2:.2f}, {b.y2:.2f}]"
                        })
                    st.dataframe(pd.DataFrame(box_data), hide_index=True, width=600)
                else:
                    st.info("No pests detected above the selected confidence threshold. Try lowering the slider value.")

    with tab_classify:
        st.markdown("#### Deep Taxonomic Species Classification")
        st.caption("Classifies the primary insect organism across the official 102 IP102 taxonomic hierarchy using ConvNeXt-Small.")

        if st.button("🔬 Classify Pest Species", type="primary"):
            with st.spinner("Analyzing pest anatomical morphology through ConvNeXt-Small..."):
                try:
                    cls_result = predict_ip102_classification_real(
                        image_input=raw_bytes or img, 
                        filename=filename or "pest.jpg"
                    )
                except Exception as e:
                    st.error(f"Classification inference failure: {e}")
                    st.stop()

            render_prediction_summary(
                title="Identified Pest Species",
                predicted_label=cls_result.predicted_pest,
                confidence=cls_result.confidence,
                latency_ms=cls_result.execution_time_ms,
                extra_tag=f"Taxonomic Index: Class #{cls_result.pest_index + 1}",
                is_mock=cls_result.is_mock,
                status_notice=cls_result.status_notice
            )
            st.markdown("---")
            render_top_k_distribution(cls_result.top_k, title="Top-5 Differential Taxonomic Matches")

else:
    st.info("👆 Please upload a field pest image or select a sample image above to initiate analysis.")
