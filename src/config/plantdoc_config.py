"""
SmartCrop AI - PlantDoc Configuration Module
Centralized paths, model configurations, and metadata for PlantDoc disease classification.
"""

from pathlib import Path
import torch

# Base project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Models and artifacts paths
MODELS_DIR = PROJECT_ROOT / "models"
DISEASE_MODEL_DIR = MODELS_DIR / "disease"
DISEASE_MODEL_PATH = DISEASE_MODEL_DIR / "disease_model.pth"
DISEASE_CLASSES_PATH = DISEASE_MODEL_DIR / "disease_classes.json"
DISEASE_METRICS_PATH = DISEASE_MODEL_DIR / "disease_metrics.json"

# Outputs and baseline paths
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
BASELINE_MODEL_PATH = OUTPUTS_DIR / "baseline_disease_model.pth"
COMPARISON_RESULTS_PATH = OUTPUTS_DIR / "comparison_results.json"

# Notebooks
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
PLANTDOC_NOTEBOOK_PATH = NOTEBOOKS_DIR / "01_PlantDoc_Baseline_and_Proposed.ipynb"

# Dataset paths
DATASETS_DIR = PROJECT_ROOT / "datasets"
PLANTDOC_DATASET_DIR = DATASETS_DIR / "plantdoc"
PLANTDOC_MANIFEST_PATH = OUTPUTS_DIR / "plantdoc_split_manifest.json"

# Model architecture hyperparameters
NUM_CLASSES = 27
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
DROPOUT_RATE = 0.4
CLASSIFIER_HIDDEN_DIM = 512

# Normalization constants (ImageNet defaults)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Hardware execution device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

