---
title: SmartCrop AI - Intelligent Agriculture Analysis
emoji: 🌱
colorFrom: green
colorTo: emerald
sdk: streamlit
sdk_version: 1.41.0
app_file: app/main.py
pinned: false
---

# SmartCrop AI: Integrated Real-Time Crop Health Monitoring System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/YOLO-Detection-00FFFF.svg)](https://docs.ultralytics.com/)
[![Base Paper](https://img.shields.io/badge/Base_Paper-Scientific_Reports_2026-green.svg)](https://doi.org/10.1038/s41598-026-42151-5)
[![Liquid AI](https://img.shields.io/badge/VLM-Liquid_AI_LFM3--VL--2B-purple.svg)](https://huggingface.co/LiquidAI)

SmartCrop AI is an explainable, multimodal precision agriculture platform. Developed as a final-year AI/ML engineering project, it extends the foundational work of **Pal et al. (Scientific Reports, 2026, DOI: [10.1038/s41598-026-42151-5](https://doi.org/10.1038/s41598-026-42151-5))** by establishing a rigorous comparative benchmark between a paper-inspired deep learning baseline and a parameter-efficient, multimodal proposed architecture.

---

## 1. Research Paradigm: Baseline vs. Proposed

| Task Domain | Paper-Inspired Baseline (Pal et al. 2026 Principles) | Proposed SmartCrop AI Architecture | Benchmark Dataset |
| :--- | :--- | :--- | :--- |
| **Foliar Disease Diagnosis** | ResNet-50 + Instance Normalization (IN) + Dual-Attention (CBAM) | EfficientNet-B0 (Transfer Learning with Compound Scaling) | **PlantDoc** (27 classes, in-field) |
| **Agricultural Pest Localization** | ResNet-50 + Instance Normalization Classification Head | Ultralytics YOLO Object Detector (Single-stage bounding box) | **IP102** (102 pest species) |
| **Earth Observation & Crop Type** | ResNet-1D / Deep MLP + Instance Normalization on time series | Tuned Random Forest / Gradient Boosting + Measured NDVI & NDMI | **NASA CropHarvest** |
| **Multimodal Advisory** | Not available in base paper | **Liquid AI LFM3-VL-2B** Bilingual Assistant (English & Telugu) | Multi-modal outputs |

> **Important Research Standards:**
> 1. Our baseline architecture is designated as a **paper-inspired implementation** capturing the architectural principles of Pal et al. (Instance Normalization, skip connections, and attention mechanisms), not an exact reproduction of proprietary datasets.
> 2. The three datasets (**PlantDoc**, **IP102**, and **CropHarvest**) originate from non-overlapping collections; they are kept strictly distinct and evaluated under identical 70% Train, 15% Validation, and 15% Test conditions.
> 3. No experimental superiority is claimed prior to empirical validation.

---

## 2. Project Directory Structure

```
SmartCrop_AI/
├── datasets/                          # Benchmark dataset storage
│   ├── plantdoc/                      # PlantDoc foliar disease images
│   ├── ip102/                         # IP102 pest imagery and bounding box annotations
│   └── cropharvest/                   # NASA CropHarvest satellite labels and feature arrays
│
├── notebooks/                         # Self-contained Kaggle Training Notebooks
│   ├── 01_PlantDoc_Baseline_and_Proposed.ipynb
│   ├── 02_IP102_Baseline_and_Proposed.ipynb
│   └── 03_CropHarvest_Baseline_and_Proposed.ipynb
│
├── models/                            # Trained Checkpoints & Exported Metadata
│   ├── disease/                       # disease_model.pth, disease_classes.json, disease_metrics.json
│   ├── pest/                          # pest_model.pt, pest_classes.json, pest_metrics.json
│   └── satellite/                     # satellite_model.pkl, satellite_features.json, satellite_metrics.json
│
├── src/                               # Modular Python Backend
│   ├── disease_detection/             # PlantDoc validation & inference
│   ├── pest_detection/                # IP102 bounding box parser & YOLO pipeline
│   ├── satellite_analysis/            # CropHarvest schema parser & spectral index math
│   ├── fusion/                        # Multimodal assessment fusion engine
│   ├── llm_advisory/                  # Liquid AI LFM3-VL-2B & Telugu translation engine
│   └── utils/                         # Reproducible 70/15/15 splitter, loggers, vision tools
│
├── app/                               # Streamlit Web Portal
├── outputs/                           # Validation manifests, split JSONs, and metrics
├── research/                          # Research documentation, methodology & templates
│   └── dataset_validation_report_template.md
│
├── dataset_config.md                  # Comprehensive dataset acquisition & split specification
├── validate_datasets.py               # Master dataset validation CLI
├── requirements.txt                   # Production library dependencies with version notes
└── README.md                          # Engineering manual and quickstart guide
```

---

## 3. Setup & Environment Initialization

### A. Environment Configuration
Create and activate an isolated Python virtual environment:
```powershell
# Using Python venv on Windows:
python -m venv venv
.\venv\Scripts\Activate.ps1

# Upgrade package manager:
python -m pip install --upgrade pip
```

### B. Dependency Installation
Install verified packages:
```powershell
pip install -r requirements.txt
```

---

## 4. Dataset Acquisition & Placement Guide

Before training, download the three datasets according to [`dataset_config.md`](file:///c:/Users/shanm/Downloads/SmartCrop_AI/dataset_config.md):

1. **PlantDoc:** Clone or download into `datasets/plantdoc/`.
2. **IP102:** Extract image and box files into `datasets/ip102/`.
3. **CropHarvest:** Place `labels.geojson` and features into `datasets/cropharvest/`.

---

## 5. Dataset Validation Suite

To verify data integrity, check for corrupted images, validate bounding-box formats, and generate the deterministic 70% / 15% / 15% train/val/test split manifests (`outputs/*_split_manifest.json`), execute:

```powershell
# Validate all three datasets simultaneously:
python validate_datasets.py --dataset all

# Or validate datasets individually:
python validate_datasets.py --dataset plantdoc
python validate_datasets.py --dataset ip102
python validate_datasets.py --dataset cropharvest
```

Validation output reports and split manifests are automatically generated inside `outputs/`.
