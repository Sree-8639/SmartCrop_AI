"""
SmartCrop AI - Reusable Prediction Result UI Component
Renders scorecards, confidence meters, Top-K probability bars, bounding boxes,
latency metrics, and explicit MOCK DATA badges.
"""

from typing import List, Optional
import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from app.api.schemas import ClassScore, BoundingBox


def render_mock_badge(status_notice: str = "MOCK DATA: Kaggle training in progress"):
    """Renders a prominent, styled badge indicating prototype mock data."""
    st.markdown(
        f"""
        <div style="background-color: #fffbeb; border: 1px solid #fef3c7; border-left: 4px solid #f59e0b; padding: 10px 14px; border-radius: 6px; margin: 12px 0;">
            <div style="display: flex; align-items: center;">
                <span style="font-size: 16px; margin-right: 8px;">⚠️</span>
                <span style="font-weight: 600; color: #b45309; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px;">Prototype Notice:</span>
                <span style="margin-left: 8px; color: #78350f; font-size: 13px;">{status_notice}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_production_badge(status_notice: str = "Verified Production Checkpoint: ConvNeXt-Small Exp D (81.20% Test Accuracy)"):
    """Renders a styled badge indicating verified real model inference."""
    st.markdown(
        f"""
        <div style="background-color: #f0fdf4; border: 1px solid #bbf7d0; border-left: 4px solid #16a34a; padding: 10px 14px; border-radius: 6px; margin: 12px 0;">
            <div style="display: flex; align-items: center;">
                <span style="font-size: 16px; margin-right: 8px;">✅</span>
                <span style="font-weight: 600; color: #15803d; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px;">Live Production Inference:</span>
                <span style="margin-left: 8px; color: #166534; font-size: 13px;">{status_notice}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_prediction_summary(
    title: str,
    predicted_label: str,
    confidence: float,
    latency_ms: float,
    extra_tag: Optional[str] = None,
    is_mock: bool = True,
    status_notice: str = "MOCK DATA"
):
    """Renders the top-level classification result card."""
    if is_mock:
        render_mock_badge(status_notice)
    else:
        render_production_badge(status_notice)

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(
            f"""
            <div style="padding: 4px 0;">
                <div style="font-size: 13px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px;">{title}</div>
                <div style="font-size: 26px; font-weight: 800; color: #064e3b; margin: 4px 0 8px 0; letter-spacing: -0.02em;">{predicted_label}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if extra_tag:
            st.markdown(f"**Category:** `{extra_tag}`")

    with col2:
        conf_pct = round(confidence * 100, 1)
        st.metric(label="Model Confidence", value=f"{conf_pct}%")
        st.progress(min(1.0, max(0.0, confidence)))

    with col3:
        st.metric(label="Inference Latency", value=f"{latency_ms} ms")
        st.caption("Engine: PyTorch / TorchScript")


def render_top_k_distribution(top_k: List[ClassScore], title: str = "Top Predictions Probability Distribution"):
    """Renders a horizontal probability distribution table and progress bar."""
    st.markdown(f"#### {title}")
    
    data = []
    for item in top_k:
        row = {
            "Rank": len(data) + 1,
            "Class Name": item.class_name,
            "Probability": f"{item.confidence * 100:.2f}%",
            "Confidence Score": item.confidence
        }
        if item.category_type:
            row["Type"] = item.category_type
        data.append(row)

    df = pd.DataFrame(data)
    
    # Render with Streamlit progress bar column config
    st.dataframe(
        df[["Rank", "Class Name", "Probability", "Confidence Score"] + (["Type"] if "Type" in df.columns else [])],
        column_config={
            "Confidence Score": st.column_config.ProgressColumn(
                "Confidence",
                help="Softmax probability distribution",
                format="%.2f",
                min_value=0.0,
                max_value=1.0,
            ),
        },
        hide_index=True,
        use_container_width=True
    )


def draw_bounding_boxes_on_image(image: Image.Image, boxes: List[BoundingBox]) -> Image.Image:
    """Draws colored bounding boxes and confidence labels onto a copy of the input image."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    w, h = image.size

    colors = ["#ef4444", "#3b82f6", "#10b981", "#f59e0b", "#8b5cf6"]

    for idx, box in enumerate(boxes):
        color = colors[idx % len(colors)]
        x1 = box.x1 * w
        y1 = box.y1 * h
        x2 = box.x2 * w
        y2 = box.y2 * h

        # Draw rectangle outline
        for offset in range(3):
            draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline=color)

        label_text = f"{box.label} ({box.confidence * 100:.1f}%)"
        draw.text((x1 + 4, max(0, y1 - 15)), label_text, fill=color)

    return annotated

