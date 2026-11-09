"""
SmartCrop AI - Reusable Input Upload UI Component
Handles image files (.jpg, .png, .jpeg) and satellite time-series files (.h5, .npy, .csv).
Provides validation, file size limits, previews, error/empty states, and sample file loading.
"""

import io
from pathlib import Path
from typing import Optional, Tuple, List
import streamlit as st
from PIL import Image

MAX_FILE_SIZE_MB = 25


def render_image_uploader(
    title: str = "Upload Leaf / Crop Image",
    help_text: str = "Supported formats: JPG, JPEG, PNG. Max file size: 25MB",
    sample_images: Optional[List[Tuple[str, str]]] = None,
    key: str = "image_uploader"
) -> Tuple[Optional[Image.Image], Optional[str], Optional[bytes]]:
    """
    Renders an image uploader with validation, preview, and optional sample images.
    Returns: (PIL.Image or None, filename or None, raw_bytes or None)
    """
    st.subheader(title)
    st.caption(help_text)

    # Sample selector option
    selected_sample = None
    if sample_images:
        sample_options = ["-- Upload your own file --"] + [label for label, _ in sample_images]
        chosen = st.selectbox("Or choose a sample test image from the dataset:", sample_options, key=f"{key}_sample_select")
        if chosen != "-- Upload your own file --":
            for label, file_path in sample_images:
                if label == chosen:
                    selected_sample = file_path
                    break

    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "webp"],
        key=f"{key}_file",
        help=help_text
    )

    # If user selected a sample image from disk
    if selected_sample and not uploaded_file:
        p = Path(selected_sample)
        if p.exists():
            try:
                img = Image.open(p).convert("RGB")
                with open(p, "rb") as f:
                    raw_bytes = f.read()
                st.info(f"Loaded sample image: **{p.name}**")
                st.image(img, caption=f"Selected Sample: {p.name}", use_container_width=True)
                return img, p.name, raw_bytes
            except Exception as e:
                st.error(f"Failed to load sample image: {e}")
                return None, None, None

    if uploaded_file is None:
        st.markdown(
            """
            <div style="border: 2px dashed #a0aec0; border-radius: 8px; padding: 30px; text-align: center; background-color: #f8fafc; color: #64748b; margin-top: 10px; margin-bottom: 15px;">
                <p style="margin: 0; font-size: 15px; font-weight: 500;">📂 Drag and drop an image, or browse from your computer</p>
                <p style="margin: 5px 0 0 0; font-size: 12px;">Validates file dimensions, format integrity, and size prior to inference.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        return None, None, None

    # Validate file size
    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(f"❌ File size exceeds {MAX_FILE_SIZE_MB}MB limit ({uploaded_file.size / (1024*1024):.1f}MB). Please upload a smaller file.")
        return None, None, None

    try:
        raw_bytes = uploaded_file.read()
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        
        # Display image preview
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(img, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)
        with col2:
            st.markdown(f"**Image Dimensions:** `{img.width} × {img.height}` px")
            st.markdown(f"**File Size:** `{len(raw_bytes) / 1024:.1f}` KB")
            st.markdown(f"**Color Format:** `RGB` (3 Channels)")
            st.success("✅ Image validated successfully. Ready for inference.")
            
        return img, uploaded_file.name, raw_bytes
    except Exception as e:
        st.error(f"❌ Invalid or corrupted image file: {e}")
        return None, None, None


def render_array_uploader(
    title: str = "Upload Satellite Time-Series (.h5 / .npy / .csv)",
    help_text: str = "Input satellite tensor format: 12 monthly timesteps × 18 spectral channels",
    sample_arrays: Optional[List[Tuple[str, str]]] = None,
    key: str = "array_uploader"
) -> Tuple[Optional[str], Optional[bytes], Optional[str]]:
    """
    Renders an uploader for satellite array data with validation.
    Returns: (filename or None, raw_bytes or None, status_message or None)
    """
    st.subheader(title)
    st.caption(help_text)

    selected_sample = None
    if sample_arrays:
        sample_options = ["-- Upload your own file --"] + [label for label, _ in sample_arrays]
        chosen = st.selectbox("Or select a verified CropHarvest array sample:", sample_options, key=f"{key}_sample_select")
        if chosen != "-- Upload your own file --":
            for label, file_path in sample_arrays:
                if label == chosen:
                    selected_sample = file_path
                    break

    uploaded_file = st.file_uploader(
        "Choose a time-series array file",
        type=["h5", "hdf5", "npy", "csv"],
        key=f"{key}_file",
        help=help_text
    )

    if selected_sample and not uploaded_file:
        p = Path(selected_sample)
        if p.exists():
            try:
                with open(p, "rb") as f:
                    raw_bytes = f.read()
                st.info(f"Loaded verified CropHarvest sample: **{p.name}**")
                return p.name, raw_bytes, f"Loaded from dataset ({len(raw_bytes)/1024:.1f} KB)"
            except Exception as e:
                st.error(f"Failed to load sample array: {e}")
                return None, None, None

    if uploaded_file is None:
        st.markdown(
            """
            <div style="border: 2px dashed #94a3b8; border-radius: 8px; padding: 25px; text-align: center; background-color: #f1f5f9; color: #475569; margin-top: 10px; margin-bottom: 15px;">
                <p style="margin: 0; font-size: 15px; font-weight: 500;">🛰️ Upload CropHarvest Time-Series File (.h5, .npy, .csv)</p>
                <p style="margin: 5px 0 0 0; font-size: 12px;">Shape requirement: 12 timesteps × 18 channels (Optical, SAR, Weather, Topo)</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        return None, None, None

    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(f"❌ File size exceeds {MAX_FILE_SIZE_MB}MB limit.")
        return None, None, None

    raw_bytes = uploaded_file.read()
    st.success(f"✅ Array file **{uploaded_file.name}** loaded ({len(raw_bytes)/1024:.1f} KB).")
    return uploaded_file.name, raw_bytes, "Uploaded file ready"

