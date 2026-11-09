"""
SmartCrop AI - IP102 Pest Species Real Inference Service
Loads and serves the verified ConvNeXt-Small checkpoint for 102 insect and mite pest species:
- Checkpoint: models/pest/ip102_convnext_best.pth
- Architecture: timm.create_model("convnext_small.fb_in22k_ft_in1k", pretrained=False, num_classes=102)
- Preprocessing: Resize(256, 256) -> ToTensor() -> ImageNet Normalization
- Weights: strict=True (Epoch 13 EMA, 76.89% Held-Out Test Accuracy, 94.36% Top-5)
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

from app.api.schemas import IP102ClassificationResponse, ClassScore

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_PATH = BASE_DIR / "models" / "pest" / "ip102_convnext_best.pth"
CLASSES_PATH = BASE_DIR / "models" / "pest" / "classes.json"
DATASET_CLASSES_TXT = BASE_DIR / "datasets" / "ip102" / "classes.txt"

MODEL_NAME = "convnext_small.fb_in22k_ft_in1k"
NUM_CLASSES = 102
RES = 256
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class IP102ClassificationService:
    _instance: Optional["IP102ClassificationService"] = None
    _lock = threading.Lock()

    def __init__(self, checkpoint_path: Optional[Path] = None, classes_path: Optional[Path] = None):
        self.checkpoint_path = checkpoint_path or CHECKPOINT_PATH
        self.classes_path = classes_path or CLASSES_PATH
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[torch.nn.Module] = None
        self.idx_to_class: Dict[int, str] = {}
        self.class_to_idx: Dict[str, int] = {}
        self.transform = transforms.Compose([
            transforms.Resize((RES, RES)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self._load_classes()
        self._load_model()

    @classmethod
    def get_instance(cls) -> "IP102ClassificationService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _load_classes(self) -> None:
        raw_classes = None
        if self.classes_path.exists():
            with open(self.classes_path, "r", encoding="utf-8") as f:
                raw_classes = json.load(f)
        elif DATASET_CLASSES_TXT.exists():
            with open(DATASET_CLASSES_TXT, "r", encoding="utf-8") as f:
                raw_classes = []
                for line in f:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) >= 2:
                        raw_classes.append(parts[1].strip())
                    elif len(parts) == 1:
                        raw_classes.append(parts[0].strip())

        if not raw_classes:
            raise FileNotFoundError(f"IP102 classes registry not found at {self.classes_path} or {DATASET_CLASSES_TXT}")

        self.idx_to_class = {}
        self.class_to_idx = {}
        if isinstance(raw_classes, list):
            for idx, name in enumerate(raw_classes):
                cname = str(name).strip()
                self.idx_to_class[idx] = cname
                self.class_to_idx[cname] = idx
        elif isinstance(raw_classes, dict):
            for k, v in raw_classes.items():
                idx = int(k)
                cname = str(v).strip()
                self.idx_to_class[idx] = cname
                self.class_to_idx[cname] = idx

        if len(self.idx_to_class) != NUM_CLASSES:
            raise ValueError(f"Expected {NUM_CLASSES} classes, found {len(self.idx_to_class)}")

    def _load_model(self) -> None:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"IP102 checkpoint not found at: {self.checkpoint_path}")

        m = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        m.load_state_dict(state_dict, strict=True)
        m.eval()
        m.to(self.device)
        self.model = m

    def predict(self, image_input: Any, filename: str = "input.jpg", top_k: int = 5) -> IP102ClassificationResponse:
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

        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        top_k_count = min(top_k, NUM_CLASSES)
        top_probs, top_indices = torch.topk(probs, k=top_k_count)

        top_k_scores: List[ClassScore] = []
        for p, idx_t in zip(top_probs, top_indices):
            idx = int(idx_t.item())
            prob_val = float(p.item())
            top_k_scores.append(
                ClassScore(
                    class_name=self.idx_to_class.get(idx, f"Pest_{idx}"),
                    class_index=idx,
                    confidence=round(prob_val, 4),
                    category_type="Agricultural Insect / Mite Pest"
                )
            )

        top_pred = top_k_scores[0]

        return IP102ClassificationResponse(
            task="IP102 Agricultural Pest Species Classification",
            predicted_pest=top_pred.class_name,
            pest_index=top_pred.class_index,
            confidence=top_pred.confidence,
            top_k=top_k_scores,
            is_mock=False,
            model_architecture="ConvNeXt-Small (256px, timm)",
            execution_time_ms=round(elapsed_ms, 2),
            status_notice="Verified Production Checkpoint: ConvNeXt-Small (76.89% Test Acc, 94.36% Top-5)"
        )


def predict_ip102_classification_real(image_input: Any, filename: str = "input.jpg") -> IP102ClassificationResponse:
    service = IP102ClassificationService.get_instance()
    return service.predict(image_input=image_input, filename=filename)

