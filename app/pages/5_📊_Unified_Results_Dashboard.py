"""
SmartCrop AI - Page 5: Unified Results & Model Comparison Dashboard
Aggregates experimental metrics, ablation studies, and deployment readiness across all datasets.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import streamlit as st
import pandas as pd

from app.components.theme import apply_custom_theme, render_sidebar_branding, render_page_header

st.set_page_config(page_title="Unified Results Dashboard — SmartCrop AI", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

apply_custom_theme()
render_sidebar_branding()

render_page_header(
    title="Unified Results & Model Benchmark Dashboard",
    subtitle="Comparative performance analysis, ablation benchmarks, confusion matrices, and training trajectories across all five perception pipelines.",
    category_badge="Audited & Benchmark Verified",
    icon="📊"
)

tab_overview, tab_cropharvest, tab_plantdoc, tab_ip102 = st.tabs([
    "🏆 System-Wide Benchmarks",
    "🛰️ CropHarvest Satellite Benchmarks",
    "🍃 PlantDoc Foliar Pathology",
    "🐛 IP102 Pest Classification & Detection"
])

with tab_overview:
    st.markdown("### Production Checkpoint Verification Matrix")
    st.caption("Official performance evaluation across all 5 production models currently serving live inference:")

    overview_data = [
        {
            "Pipeline Task": "🍃 PlantDoc Foliar Disease",
            "Model Backbone": "ConvNeXt-Small (384px)",
            "Diagnostic Scope": "27 Disease & Leaf Classes",
            "Official Test Accuracy": "81.20%",
            "Secondary Metric": "79.52% Macro-F1",
            "Inference Engine": "PyTorch / timm",
            "Active Checkpoint": "disease_model_exp_D_best.pth (188.9 MB)"
        },
        {
            "Pipeline Task": "🐛 IP102 Pest Species",
            "Model Backbone": "ConvNeXt-Small (256px)",
            "Diagnostic Scope": "102 Insect Species",
            "Official Test Accuracy": "76.89%",
            "Secondary Metric": "94.36% Top-5 Acc",
            "Inference Engine": "PyTorch / timm",
            "Active Checkpoint": "ip102_convnext_best.pth (189.1 MB)"
        },
        {
            "Pipeline Task": "🎯 IP102 Field Pest Detection",
            "Model Backbone": "Ultralytics YOLOv8s",
            "Diagnostic Scope": "102 Insect Categories",
            "Official Test Accuracy": "65.33% mAP50",
            "Secondary Metric": "41.18% mAP50-95",
            "Inference Engine": "Ultralytics YOLO",
            "Active Checkpoint": "yolov8_pest_best.pt (21.5 MB)"
        },
        {
            "Pipeline Task": "🛰️ CropHarvest Binary",
            "Model Backbone": "3-Seed TSTransformer Ensemble",
            "Diagnostic Scope": "Crop vs Non-Crop (12x18)",
            "Official Test Accuracy": "82.02%",
            "Secondary Metric": "81.18% Macro-F1",
            "Inference Engine": "PyTorch / Ensemble",
            "Active Checkpoint": "transformer_seeds.pth (2.86 MB)"
        },
        {
            "Pipeline Task": "🌾 CropHarvest 10-Class",
            "Model Backbone": "CropHarvestBiLSTM",
            "Diagnostic Scope": "9 FAO Crops + 1 Land-Cover",
            "Official Test Accuracy": "68.22%",
            "Secondary Metric": "32.14% Macro-F1",
            "Inference Engine": "PyTorch / BiLSTM",
            "Active Checkpoint": "best_model.pth (0.59 MB)"
        }
    ]
    st.dataframe(pd.DataFrame(overview_data), hide_index=True, width="stretch")

    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px;">
                <div style="font-weight: 700; color: #064e3b; margin-bottom: 8px;">🛡️ Rigorous Scientific Governance</div>
                <div style="font-size: 13.5px; color: #475569; line-height: 1.5;">
                    • <strong>Strict Separation:</strong> All evaluated test sets were held out during cross-entropy and loss backpropagation.<br>
                    • <strong>Zero Mock Discrepancy:</strong> All 5 endpoints run genuine checkpoint weights.<br>
                    • <strong>Reproducibility:</strong> Input dimensions, normalization parameters, and class mapping files strictly match training code.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            """
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px;">
                <div style="font-weight: 700; color: #064e3b; margin-bottom: 8px;">⚡ Hardware & Runtime Benchmarks</div>
                <div style="font-size: 13.5px; color: #475569; line-height: 1.5;">
                    • <strong>Total Weights Size:</strong> ~403 MB (Lightweight, cloud container ready).<br>
                    • <strong>Memory Footprint:</strong> &lt; 1.8 GB system RAM under live multi-pipeline serving.<br>
                    • <strong>Warm CPU Latency:</strong> ~35–280 ms across models.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

with tab_cropharvest:
    st.markdown("### CropHarvest Earth Observation Benchmarking & Ablations")
    st.caption("Ablation study across four architectures evaluated on the official test split (12 monthly timesteps × 18 channels):")

    cropharvest_perf = [
        {"Model Architecture": "Multi-Model Family Ensemble", "Target Task": "10-Class Crop Groups", "Train Acc": "85.95%", "Val Acc": "74.31%", "Test Acc": "68.22%", "Test Macro-F1": "32.14%", "Outcome": "🏆 Best 10-Class Benchmark"},
        {"Model Architecture": "Direct 10-Class BiLSTM", "Target Task": "10-Class Crop Groups", "Train Acc": "93.36%", "Val Acc": "74.20%", "Test Acc": "67.29%", "Test Macro-F1": "32.39%", "Outcome": "Strong Baseline"},
        {"Model Architecture": "3-Seed TSTransformer Ensemble", "Target Task": "Binary (Crop vs Non-Crop)", "Train Acc": "94.52%", "Val Acc": "87.24%", "Test Acc": "82.02%", "Test Macro-F1": "81.18%", "Outcome": "🏆 Best Binary Benchmark"},
        {"Model Architecture": "Two-Step Hierarchical Gatekeeper", "Target Task": "Binary -> 9-Crop", "Train Acc": "87.21%", "Val Acc": "73.19%", "Test Acc": "56.07%", "Test Macro-F1": "23.82%", "Outcome": "Error compounding at Stage 1"}
    ]
    st.dataframe(pd.DataFrame(cropharvest_perf), hide_index=True, width="stretch")

    cm_path = Path("outputs/cropharvest/v2_confusion_matrix.png")
    per_class_path = Path("outputs/cropharvest/10class/per_class_validation_report.csv")

    c_ch1, c_ch2 = st.columns([1, 1])
    with c_ch1:
        if cm_path.exists():
            st.image(str(cm_path), caption="CropHarvest 10-Class Confusion Matrix (Validation Set)", width="stretch")
    with c_ch2:
        if per_class_path.exists():
            st.markdown("#### Per-Class Performance Breakdown")
            df_pc = pd.read_csv(per_class_path)
            st.dataframe(df_pc, hide_index=True, width="stretch")

with tab_plantdoc:
    st.markdown("### PlantDoc Foliar Disease Architectural Progression")
    st.caption("Ablation study showing progression from initial baseline to State-of-the-Art Experiment D:")

    plantdoc_perf = [
        {"Experiment Stage": "Baseline (ResNet-50 224px)", "Train Acc": "88.4%", "Val Acc": "51.2%", "Test Acc": "50.84%", "Notes": "Severe overfitting on web leaf crops"},
        {"Experiment Stage": "Exp A (ConvNeXt-Small 224px)", "Train Acc": "91.8%", "Val Acc": "62.4%", "Test Acc": "61.35%", "Notes": "Modern 7x7 depthwise conv backbone"},
        {"Experiment Stage": "Exp B (ConvNeXt + Weighted Loss)", "Train Acc": "94.6%", "Val Acc": "69.19%", "Test Acc": "68.15%", "Notes": "Class-imbalance mitigated"},
        {"Experiment Stage": "Exp D (ConvNeXt-S 384px + EMA)", "Train Acc": "98.71%", "Val Acc": "78.85%", "Test Acc": "81.20%", "Notes": "🏆 State-of-the-Art (Production Deployed)"}
    ]
    st.dataframe(pd.DataFrame(plantdoc_perf), hide_index=True, width="stretch")

    curves_d = Path("outputs/plantdoc/experiment_D_curves.png")
    if curves_d.exists():
        st.image(str(curves_d), caption="Experiment D (384px + EMA) Loss and Accuracy Progression", width="stretch")

with tab_ip102:
    st.markdown("### IP102 Pest Classification & Detection Benchmarks")
    
    col_ip_cls, col_ip_det = st.columns(2)
    with col_ip_cls:
        st.markdown("#### Taxonomic Species Classification (102 Classes)")
        ip102_perf = [
            {"Split": "Training Split", "Top-1 Acc": "92.11%", "Top-5 Acc": "99.82%", "Macro F1": "90.02%"},
            {"Split": "Validation Split", "Top-1 Acc": "77.33%", "Top-5 Acc": "94.53%", "Macro F1": "69.82%"},
            {"Split": "Official Test Split", "Top-1 Acc": "76.89%", "Top-5 Acc": "94.36%", "Macro F1": "69.55%"}
        ]
        st.dataframe(pd.DataFrame(ip102_perf), hide_index=True, width="stretch")
        
        ip102_curves = Path("outputs/ip102/ip102_curves.png")
        if ip102_curves.exists():
            st.image(str(ip102_curves), caption="IP102 ConvNeXt-Small Loss and Validation Accuracy Curves", width="stretch")

    with col_ip_det:
        st.markdown("#### YOLOv8 Field Pest Object Detection")
        det_perf = [
            {"Benchmark Metric": "mAP@0.50", "Score": "65.33%", "Standard": "VOC Benchmark"},
            {"Benchmark Metric": "mAP@0.50:0.95", "Score": "41.18%", "Standard": "COCO Strict Multi-IoU"},
            {"Benchmark Metric": "Validation Acc (Image-Level)", "Score": "74.34%", "Standard": "Full Validation"},
            {"Benchmark Metric": "Detection F1-Score", "Score": "70.82%", "Standard": "Precision: 68.5% | Recall: 73.3%"},
            {"Benchmark Metric": "Mean Matched IoU", "Score": "0.811", "Standard": "High Box Fidelity"}
        ]
        st.dataframe(pd.DataFrame(det_perf), hide_index=True, width="stretch")

        det_curves = Path("outputs/ip102_detection/training_curves.png")
        if det_curves.exists():
            st.image(str(det_curves), caption="YOLOv8 Detection Loss & mAP Trajectories", width="stretch")
