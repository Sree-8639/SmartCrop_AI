"""
SmartCrop AI - FastAPI Backend Application & Prediction Endpoints
Provides standard REST APIs for model inference, health checks, and subsystem status.
Currently connected to mock services using official dataset labels and classes,
ready for direct drop-in integration with real checkpoints when training concludes.
"""

import os
os.environ["YOLO_AUTOINSTALL"] = "False"

from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.api.schemas import (
    PlantDocPredictionResponse,
    IP102ClassificationResponse,
    IP102DetectionResponse,
    CropHarvestBinaryResponse,
    CropHarvest10ClassResponse,
    SubsystemStatus,
    ExplanationRequest,
    ExplanationResponse,
    LFM3HealthResponse,
)
from app.api.mock_service import (
    generate_plantdoc_mock,
    generate_ip102_classification_mock,
    generate_ip102_detection_mock,
    generate_cropharvest_binary_mock,
    generate_cropharvest_10class_mock,
    get_all_subsystem_health,
)
from app.services.plantdoc_service import predict_plantdoc_real
from app.services.ip102_service import predict_ip102_classification_real
from app.services.ip102_detection_service import predict_ip102_detection_real
from app.services.cropharvest_service import (
    predict_cropharvest_binary_real,
    predict_cropharvest_10class_real,
)
from app.services.lfm3_vl_service import LFM3VLService, explain_prediction

app = FastAPI(
    title="SmartCrop AI Backend API",
    description="Multi-Modal Agricultural Intelligence API covering foliar disease, pest diagnosis, and satellite crop classification.",
    version="1.0.0"
)

# CORS middleware for local frontend and developer tools
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
def root():
    return {
        "project": "SmartCrop AI",
        "description": "Multi-Modal Agricultural Intelligence Suite",
        "version": "1.0.0",
        "status": "online",
        "active_models": {
            "plantdoc": "ConvNeXt-Small 384px (Exp D Best Checkpoint - LIVE PRODUCTION INFERENCE)",
            "ip102_classification": "ConvNeXt-Small 256px (Epoch 13 EMA - LIVE PRODUCTION INFERENCE)",
            "ip102_detection": "Ultralytics YOLOv8 (65.33% mAP50 - LIVE PRODUCTION INFERENCE)",
            "cropharvest_binary": "3-Seed TSTransformer Ensemble (82.02% Test Acc - LIVE PRODUCTION INFERENCE)",
            "cropharvest_10class": "CropHarvestBiLSTM (68.22% Benchmark - LIVE PRODUCTION INFERENCE)"
        }
    }


@app.get("/api/health", response_model=List[SubsystemStatus], tags=["Health"])
def get_health():
    """Inspect model files, datasets, and return live pipeline readiness."""
    return get_all_subsystem_health()


# =============================================================================
# Task 1: PlantDoc Leaf Disease Diagnosis (LIVE PRODUCTION INFERENCE)
# =============================================================================
@app.post("/api/predict/plantdoc", response_model=PlantDocPredictionResponse, tags=["Inference"])
async def predict_plantdoc(
    file: UploadFile = File(..., description="Crop leaf image (JPG/PNG)")
):
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        raise HTTPException(status_code=400, detail="Invalid file type. Expected an image (JPG, PNG, WEBP).")
    
    # Read bytes to validate not empty
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    
    # Execute verified real inference via ConvNeXt-Small (Exp D)
    try:
        response = predict_plantdoc_real(image_input=content, filename=file.filename)
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model resource missing: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference execution failure: {str(e)}")


# =============================================================================
# Task 2: IP102 Pest Classification & Detection (LIVE PRODUCTION INFERENCE)
# =============================================================================
@app.post("/api/predict/ip102/classify", response_model=IP102ClassificationResponse, tags=["Inference"])
async def predict_ip102_classify(
    file: UploadFile = File(..., description="Agricultural pest image (JPG/PNG)")
):
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        raise HTTPException(status_code=400, detail="Invalid file type. Expected an image.")
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    try:
        response = predict_ip102_classification_real(image_input=content, filename=file.filename)
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model resource missing: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference execution failure: {str(e)}")


@app.post("/api/predict/ip102/detect", response_model=IP102DetectionResponse, tags=["Inference"])
async def predict_ip102_detect(
    file: UploadFile = File(..., description="Agricultural field image containing pests"),
    conf_threshold: float = Query(0.25, ge=0.05, le=1.0)
):
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        raise HTTPException(status_code=400, detail="Invalid file type. Expected an image.")
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    try:
        response = predict_ip102_detection_real(image_input=content, filename=file.filename, conf_threshold=conf_threshold)
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model resource missing: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference execution failure: {str(e)}")


# =============================================================================
# Task 3: CropHarvest Satellite Binary Classification (LIVE PRODUCTION INFERENCE)
# =============================================================================
@app.post("/api/predict/cropharvest/binary", response_model=CropHarvestBinaryResponse, tags=["Inference"])
async def predict_cropharvest_binary(
    file: UploadFile = File(..., description="Satellite time-series array (.h5, .npy, or .csv format, 12x18)")
):
    if not file.filename.lower().endswith(('.h5', '.hdf5', '.npy', '.npz', '.csv')):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Expected satellite time-series array (.h5, .npy, .csv)."
        )
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    try:
        response = predict_cropharvest_binary_real(file_bytes=content, filename=file.filename)
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model resource missing: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference execution failure: {str(e)}")


# =============================================================================
# Task 4: CropHarvest Satellite 10-Class Classification (LIVE PRODUCTION INFERENCE)
# =============================================================================
@app.post("/api/predict/cropharvest/10class", response_model=CropHarvest10ClassResponse, tags=["Inference"])
async def predict_cropharvest_10class(
    file: UploadFile = File(..., description="Satellite time-series array (.h5, .npy, or .csv format, 12x18)")
):
    if not file.filename.lower().endswith(('.h5', '.hdf5', '.npy', '.npz', '.csv')):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Expected satellite time-series array (.h5, .npy, .csv)."
        )
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    try:
        response = predict_cropharvest_10class_real(file_bytes=content, filename=file.filename)
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model resource missing: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference execution failure: {str(e)}")


# =============================================================================
# Task 5: Multimodal Grounded AI Explanation (LiquidAI / LFM3-VL-2B)
# =============================================================================
@app.get("/api/lfm3/health", response_model=LFM3HealthResponse, tags=["Explanation"])
def get_lfm3_health():
    """Inspects live availability, memory footprint, and status of LFM3-VL-2B."""
    service = LFM3VLService.get_instance()
    health = service.get_health()
    return health


@app.post("/api/explain", response_model=ExplanationResponse, tags=["Explanation"])
async def explain_model_prediction(
    request: ExplanationRequest
):
    """
    Generates grounded multimodal explanations interpreting predictions from any of
    the 5 SmartCrop AI models (PlantDoc, IP102 Classification, IP102 Detection,
    CropHarvest Binary, and CropHarvest 10-Class) using LiquidAI LFM3-VL-2B.
    """
    try:
        response = explain_prediction(
            module_name=request.module_name,
            prediction_result=request.prediction_result,
            image=request.image_base64,
            structured_evidence=request.structured_evidence,
            detection_boxes=request.detection_boxes
        )
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation generation failure: {str(e)}")

