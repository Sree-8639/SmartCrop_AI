"""
SmartCrop AI - PlantDoc Foliar Disease Real Inference Service
Loads and serves the verified Experiment D ConvNeXt-Small checkpoint:
- Checkpoint: models/disease/disease_model_exp_D_best.pth
- Architecture: timm.create_model("convnext_small.fb_in22k_ft_in1k", pretrained=False, num_classes=27)
- Preprocessing: 416x416 Resize -> 384x384 CenterCrop -> ImageNet Normalization
- Weights: strict=True (Epoch 16 EMA, 81.20% Test Accuracy)
"""

import io
import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import timm

from app.api.schemas import PlantDocPredictionResponse, ClassScore

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_PATH = BASE_DIR / "models" / "disease" / "disease_model_exp_D_best.pth"
CLASSES_PATH = BASE_DIR / "models" / "disease" / "disease_classes.json"

MODEL_NAME = "convnext_small.fb_in22k_ft_in1k"
NUM_CLASSES = 27
RES = 384
PRE = RES + 32  # 416
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PlantDocInferenceService:
    _instance: Optional["PlantDocInferenceService"] = None
    _lock = threading.Lock()

    def __init__(self, checkpoint_path: Optional[Path] = None, classes_path: Optional[Path] = None):
        self.checkpoint_path = checkpoint_path or CHECKPOINT_PATH
        self.classes_path = classes_path or CLASSES_PATH
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[torch.nn.Module] = None
        self.idx_to_class: Dict[int, str] = {}
        self.class_to_idx: Dict[str, int] = {}
        self.transform = transforms.Compose([
            transforms.Resize((PRE, PRE)),
            transforms.CenterCrop(RES),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self._load_classes()
        self._load_model()

    @classmethod
    def get_instance(cls) -> "PlantDocInferenceService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _load_classes(self) -> None:
        if not self.classes_path.exists():
            raise FileNotFoundError(f"PlantDoc classes registry not found at: {self.classes_path}")
        with open(self.classes_path, "r", encoding="utf-8") as f:
            raw_classes = json.load(f)

        self.idx_to_class = {}
        self.class_to_idx = {}
        if isinstance(raw_classes, dict):
            for k, v in raw_classes.items():
                idx = int(k)
                name = str(v).strip()
                self.idx_to_class[idx] = name
                self.class_to_idx[name] = idx
        elif isinstance(raw_classes, list):
            for idx, name in enumerate(raw_classes):
                cname = str(name).strip()
                self.idx_to_class[idx] = cname
                self.class_to_idx[cname] = idx
        else:
            raise ValueError(f"Unrecognized class structure in {self.classes_path}")

        if len(self.idx_to_class) != NUM_CLASSES:
            raise ValueError(f"Expected {NUM_CLASSES} classes, found {len(self.idx_to_class)}")

    def _load_model(self) -> None:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"PlantDoc Exp D checkpoint not found at: {self.checkpoint_path}")

        # Instantiate architecture
        m = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)

        # Load weights with strict=True
        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        load_result = m.load_state_dict(state_dict, strict=True)
        if len(load_result.missing_keys) > 0 or len(load_result.unexpected_keys) > 0:
            raise RuntimeError(
                f"Strict checkpoint loading mismatch! Missing: {load_result.missing_keys}, Unexpected: {load_result.unexpected_keys}"
            )

        m.eval()
        m.to(self.device)
        self.model = m

    def predict_image(self, image_input: Any, filename: str = "upload.jpg") -> PlantDocPredictionResponse:
        """
        Executes real forward-pass inference on a PIL Image, bytes, or file-like object.
        Returns a validated PlantDocPredictionResponse with is_mock=False.
        """
        if self.model is None:
            raise RuntimeError("PlantDoc model is not loaded.")

        start_time = time.perf_counter()

        # Load and validate PIL Image
        try:
            if isinstance(image_input, bytes):
                if len(image_input) == 0:
                    raise ValueError("Image bytes are empty (0 bytes).")
                img = Image.open(io.BytesIO(image_input))
            elif hasattr(image_input, "read"):
                raw = image_input.read()
                if len(raw) == 0:
                    raise ValueError("Uploaded file stream is empty.")
                img = Image.open(io.BytesIO(raw))
            elif isinstance(image_input, Image.Image):
                img = image_input
            elif isinstance(image_input, (str, Path)):
                p = Path(image_input)
                if not p.exists():
                    raise FileNotFoundError(f"Image file does not exist: {p}")
                img = Image.open(p)
            else:
                raise TypeError(f"Unsupported image input type: {type(image_input)}")

            img = img.convert("RGB")
        except UnidentifiedImageError:
            raise ValueError(f"Cannot identify image file '{filename}'. Unsupported or corrupted format.")
        except Exception as e:
            if isinstance(e, (ValueError, FileNotFoundError)):
                raise e
            raise ValueError(f"Failed to process image '{filename}': {str(e)}")

        # Preprocess tensor
        try:
            tensor = self.transform(img).unsqueeze(0).to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to apply preprocessing transforms: {str(e)}")

        # Execute forward pass
        try:
            with torch.no_grad():
                logits = self.model(tensor)
                if not torch.isfinite(logits).all():
                    raise RuntimeError("Model produced non-finite (NaN or Inf) logits.")
                probs = F.softmax(logits, dim=-1).squeeze(0)
        except Exception as e:
            raise RuntimeError(f"Error during model forward pass: {str(e)}")

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Extract Top-5 predictions
        top5_probs, top5_indices = torch.topk(probs, k=min(5, NUM_CLASSES))
        top_k: List[ClassScore] = []
        for p, idx in zip(top5_probs.tolist(), top5_indices.tolist()):
            c_name = self.idx_to_class.get(idx, f"Class {idx}")
            top_k.append(ClassScore(
                class_name=c_name,
                class_index=idx,
                confidence=round(float(p), 4),
                category_type="Foliar Pathology Condition"
            ))

        predicted_class = top_k[0].class_name
        confidence = top_k[0].confidence

        return PlantDocPredictionResponse(
            task="PlantDoc Foliar Disease Classification",
            predicted_class=predicted_class,
            confidence=confidence,
            top_k=top_k,
            disease_severity="Unavailable (Pathology severity grading is not annotated in PlantDoc dataset)",
            is_mock=False,
            model_architecture="ConvNeXt-Small (384px, timm)",
            execution_time_ms=latency_ms,
            status_notice="Verified Production Checkpoint: ConvNeXt-Small Exp D (81.20% Test Accuracy)"
        )


def predict_plantdoc_real(image_input: Any, filename: str = "upload.jpg") -> PlantDocPredictionResponse:
    """Convenience functional interface for real PlantDoc inference."""
    service = PlantDocInferenceService.get_instance()
    return service.predict_image(image_input=image_input, filename=filename)

