"""
SmartCrop AI - IP102 Agricultural Pest Object Detection Real Inference Service
Loads and serves the verified Ultralytics YOLOv8 checkpoint for in-field pest bounding box detection:
- Checkpoint: models/pest/yolov8_pest_best.pt (21.52 MB)
- Task: Agricultural insect pest localization (102 classes)
- Benchmark: 65.33% mAP@0.50, 41.18% mAP@0.50:0.95, 30.9 ms latency
"""

import os
os.environ["YOLO_AUTOINSTALL"] = "False"

import io
import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

from app.api.schemas import IP102DetectionResponse, BoundingBox

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_PATH = BASE_DIR / "models" / "pest" / "yolov8_pest_best.pt"
CLASSES_PATH = BASE_DIR / "models" / "pest" / "classes.json"


class IP102DetectionService:
    _instance: Optional["IP102DetectionService"] = None
    _lock = threading.Lock()

    def __init__(self, checkpoint_path: Optional[Path] = None, classes_path: Optional[Path] = None):
        self.checkpoint_path = checkpoint_path or CHECKPOINT_PATH
        self.classes_path = classes_path or CLASSES_PATH
        self.model: Optional[YOLO] = None
        self.idx_to_class: Dict[int, str] = {}
        self._load_classes()
        self._load_model()

    @classmethod
    def get_instance(cls) -> "IP102DetectionService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _load_classes(self) -> None:
        if self.classes_path.exists():
            with open(self.classes_path, "r", encoding="utf-8") as f:
                classes = json.load(f)
                if isinstance(classes, list):
                    self.idx_to_class = {i: name.strip() for i, name in enumerate(classes)}
                elif isinstance(classes, dict):
                    self.idx_to_class = {int(k): str(v).strip() for k, v in classes.items()}

    def _load_model(self) -> None:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"IP102 YOLO detection checkpoint not found at: {self.checkpoint_path}")
        self.model = YOLO(str(self.checkpoint_path))
        # Sync class names with model names if empty
        if not self.idx_to_class and hasattr(self.model, "names"):
            self.idx_to_class = {int(k): str(v).strip() for k, v in self.model.names.items()}

    def predict(self, image_input: Any, filename: str = "input.jpg", conf_threshold: float = 0.25) -> IP102DetectionResponse:
        start_time = time.perf_counter()

        # Handle bytes vs PIL vs filepath
        if isinstance(image_input, (bytes, bytearray)):
            if len(image_input) == 0:
                raise ValueError("Uploaded file is empty.")
            try:
                img = Image.open(io.BytesIO(image_input))
                img.load()
            except UnidentifiedImageError:
                raise ValueError(f"Cannot identify image file '{filename}'. Unsupported or corrupted format.")
        elif isinstance(image_input, Image.Image):
            img = image_input
        elif isinstance(image_input, (str, Path)):
            img = Image.open(str(image_input))
            img.load()
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        if img.mode != "RGB":
            img = img.convert("RGB")

        orig_w, orig_h = img.size

        # Run Ultralytics YOLO inference
        results = self.model.predict(source=img, conf=conf_threshold, verbose=False)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        detected_boxes: List[BoundingBox] = []

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())

                # Normalize coordinates [0.0, 1.0]
                x1 = max(0.0, min(1.0, float(xyxy[0]) / orig_w))
                y1 = max(0.0, min(1.0, float(xyxy[1]) / orig_h))
                x2 = max(0.0, min(1.0, float(xyxy[2]) / orig_w))
                y2 = max(0.0, min(1.0, float(xyxy[3]) / orig_h))

                label = self.idx_to_class.get(cls_id, self.model.names.get(cls_id, f"Pest_{cls_id}"))

                detected_boxes.append(
                    BoundingBox(
                        x1=round(x1, 4),
                        y1=round(y1, 4),
                        x2=round(x2, 4),
                        y2=round(y2, 4),
                        label=label,
                        class_id=cls_id,
                        confidence=round(conf, 4)
                    )
                )

        return IP102DetectionResponse(
            task="IP102 Agricultural Pest Localization",
            total_pests_detected=len(detected_boxes),
            detected_boxes=detected_boxes,
            is_mock=False,
            model_architecture="Ultralytics YOLOv8 (102 classes)",
            execution_time_ms=round(elapsed_ms, 2),
            status_notice="Verified Production Checkpoint: Ultralytics YOLOv8 (65.33% mAP50, 74.34% Val Acc)"
        )


def predict_ip102_detection_real(image_input: Any, filename: str = "input.jpg", conf_threshold: float = 0.25) -> IP102DetectionResponse:
    service = IP102DetectionService.get_instance()
    return service.predict(image_input=image_input, filename=filename, conf_threshold=conf_threshold)
