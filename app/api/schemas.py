"""
SmartCrop AI - API Contracts and Pydantic Data Schemas
Defines request and response models for all four prediction tasks:
1. PlantDoc Leaf Disease Diagnosis
2. IP102 Agricultural Pest Classification & Detection
3. CropHarvest Satellite Binary Classification (Crop vs Non-Crop)
4. CropHarvest Satellite 10-Class Classification (9 Crops + 1 Land Cover)
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class PredictionState(str, Enum):
    IDLE = "idle"
    LOADING = "loading"
    SUCCESS = "success"
    ERROR = "error"
    EMPTY = "empty"


class ClassScore(BaseModel):
    """Represents a predicted class and its assigned probability."""
    class_name: str
    class_index: int
    confidence: float = Field(ge=0.0, le=1.0)
    category_type: Optional[str] = None  # e.g. "Crop Category" vs "Land-Cover Class"


class BoundingBox(BaseModel):
    """Normalized bounding box coordinates for object detection."""
    x1: float = Field(ge=0.0, le=1.0, description="Normalized top-left x")
    y1: float = Field(ge=0.0, le=1.0, description="Normalized top-left y")
    x2: float = Field(ge=0.0, le=1.0, description="Normalized bottom-right x")
    y2: float = Field(ge=0.0, le=1.0, description="Normalized bottom-right y")
    label: str
    class_id: int
    confidence: float = Field(ge=0.0, le=1.0)


# =============================================================================
# Task 1: PlantDoc Leaf Disease Classification
# =============================================================================
class PlantDocPredictionResponse(BaseModel):
    task: str = "PlantDoc Foliar Disease Classification"
    predicted_class: str
    confidence: float = Field(ge=0.0, le=1.0)
    top_k: List[ClassScore] = Field(default_factory=list)
    disease_severity: Optional[str] = "Unavailable (Pathology severity grading is not annotated in PlantDoc dataset)"
    is_mock: bool = False
    model_architecture: str = "ConvNeXt-Small (384px, timm)"
    execution_time_ms: float = 0.0
    status_notice: str = "Verified Production Checkpoint: ConvNeXt-Small Exp D (81.20% Test Accuracy)"


# =============================================================================
# Task 2: IP102 Pest Classification & Detection
# =============================================================================
class IP102ClassificationResponse(BaseModel):
    task: str = "IP102 Agricultural Pest Species Classification"
    predicted_pest: str
    pest_index: int
    confidence: float = Field(ge=0.0, le=1.0)
    top_k: List[ClassScore] = Field(default_factory=list)
    is_mock: bool = True
    model_architecture: str = "ConvNeXt-Small / ResNet-50"
    execution_time_ms: float = 0.0
    status_notice: str = "MOCK DATA: Kaggle training in progress"


class IP102DetectionResponse(BaseModel):
    task: str = "IP102 Agricultural Pest Localization"
    total_pests_detected: int = 0
    detected_boxes: List[BoundingBox] = Field(default_factory=list)
    is_mock: bool = True
    model_architecture: str = "Ultralytics YOLOv8"
    execution_time_ms: float = 0.0
    status_notice: str = "MOCK DATA: Kaggle training in progress"


# =============================================================================
# Task 3: CropHarvest Binary Classification (Crop vs Non-Crop)
# =============================================================================
class CropHarvestBinaryResponse(BaseModel):
    task: str = "CropHarvest Satellite Binary Crop Detection"
    is_crop: bool
    label: str = Field(description="'Crop' or 'Non-Crop'")
    crop_probability: float = Field(ge=0.0, le=1.0)
    non_crop_probability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    top_k: List[ClassScore] = Field(default_factory=list)
    timesteps_verified: int = 12
    channels_verified: int = 18
    is_mock: bool = True
    model_architecture: str = "CropHarvestBiLSTM (Binary)"
    execution_time_ms: float = 0.0
    status_notice: str = "MOCK DATA: Kaggle training in progress"


# =============================================================================
# Task 4: CropHarvest 10-Class Classification (9 Crops + 1 Land Cover)
# =============================================================================
class CropHarvest10ClassResponse(BaseModel):
    task: str = "CropHarvest 10-Class Crop and Land-Cover Classification"
    predicted_class: str
    class_index: int
    category_type: str = Field(description="'Crop Category' or 'Land-Cover Class'")
    confidence: float = Field(ge=0.0, le=1.0)
    top_k: List[ClassScore] = Field(default_factory=list)
    timesteps_verified: int = 12
    channels_verified: int = 18
    is_mock: bool = True
    model_architecture: str = "CropHarvestBiLSTM (10-Class)"
    execution_time_ms: float = 0.0
    status_notice: str = "MOCK DATA: Kaggle training in progress"


# =============================================================================
# Subsystem Status Health Check
# =============================================================================
class SubsystemStatus(BaseModel):
    subsystem_id: str
    name: str
    dataset: str
    status: str  # "Ready", "Training on Kaggle", "Pending"
    is_ready: bool
    active_checkpoint: Optional[str] = None
    test_accuracy: Optional[float] = None
    classes_count: int

