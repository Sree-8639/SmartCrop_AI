"""
SmartCrop AI - Configuration Package
Centralized paths, model thresholds, device configurations, and metadata.
"""

from pathlib import Path
import torch

# Base project paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = BASE_DIR
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
DATASETS_DIR = BASE_DIR / "datasets"
DOCS_DIR = BASE_DIR / "docs"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Model artifact subdirectories and file paths
DISEASE_MODEL_DIR = MODELS_DIR / "disease"
DISEASE_MODEL_PATH = DISEASE_MODEL_DIR / "disease_model.pth"
DISEASE_CLASSES_PATH = DISEASE_MODEL_DIR / "disease_classes.json"
DISEASE_METRICS_PATH = DISEASE_MODEL_DIR / "disease_metrics.json"

BASELINE_MODEL_PATH = OUTPUTS_DIR / "baseline_disease_model.pth"
COMPARISON_RESULTS_PATH = OUTPUTS_DIR / "comparison_results.json"
PLANTDOC_NOTEBOOK_PATH = NOTEBOOKS_DIR / "01_PlantDoc_Baseline_and_Proposed.ipynb"

PEST_MODEL_DIR = MODELS_DIR / "pest"
PEST_MODEL_PATH = PEST_MODEL_DIR / "pest_model.pt"
PEST_CLASSES_PATH = PEST_MODEL_DIR / "pest_classes.json"
PEST_METRICS_PATH = PEST_MODEL_DIR / "pest_metrics.json"

SATELLITE_MODEL_DIR = MODELS_DIR / "satellite"
SATELLITE_MODEL_PATH = SATELLITE_MODEL_DIR / "satellite_model.pkl"
SATELLITE_FEATURES_PATH = SATELLITE_MODEL_DIR / "satellite_features.json"
SATELLITE_METRICS_PATH = SATELLITE_MODEL_DIR / "satellite_metrics.json"

# Hardware execution device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Image processing standard sizes
DISEASE_IMG_SIZE = (224, 224)
PEST_IMG_SIZE = 640  # YOLO default input dimension

# Normalization constants (ImageNet defaults for EfficientNet-B0)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Satellite Sentinel-2 Spectral Indices Formulas & Stress Thresholds:
NDVI_THRESHOLDS = {
    "water_or_barren": (-1.0, 0.2),
    "stressed_or_sparse": (0.2, 0.45),
    "moderate_vegetation": (0.45, 0.65),
    "healthy_dense_canopy": (0.65, 1.0)
}

NDMI_THRESHOLDS = {
    "severe_water_stress": (-1.0, 0.0),
    "moderate_water_stress": (0.0, 0.2),
    "adequate_canopy_moisture": (0.2, 0.4),
    "high_moisture_surplus": (0.4, 1.0)
}

# Vision-Language Model Configuration (Liquid AI LFM3-VL-2B)
VLM_MODEL_ID = "LiquidAI/LFM-3-VL-2B"
VLM_CACHE_DIR = MODELS_DIR / "vlm"
SUPPORTED_LANGUAGES = ["English", "Telugu"]

