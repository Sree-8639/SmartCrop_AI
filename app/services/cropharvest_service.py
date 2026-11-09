"""
SmartCrop AI - CropHarvest Satellite Real Inference Service
Provides real inference for both Binary (Crop vs Non-Crop) and 10-Class Crop Classification:
- Modality: Multi-spectral Earth Observation satellite time-series (12 monthly timesteps x 18 channels)
- Binary Pipeline: 3-Seed TSTransformer Ensemble (models/cropharvest/binary/transformer_seeds.pth, 82.02% Test Acc)
- 10-Class Pipeline: CropHarvestBiLSTM (models/cropharvest/10class/best_model.pth, 68.22% Benchmark)
"""

import io
import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.api.schemas import (
    CropHarvestBinaryResponse,
    CropHarvest10ClassResponse,
    ClassScore,
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Paths
BINARY_SEEDS_PATH = BASE_DIR / "models" / "cropharvest" / "binary" / "transformer_seeds.pth"
BINARY_CLASSES_PATH = BASE_DIR / "models" / "cropharvest" / "binary" / "classes.json"

CLASS10_CKPT_PATH = BASE_DIR / "models" / "cropharvest" / "10class" / "best_model.pth"
CLASS10_CLASSES_PATH = BASE_DIR / "models" / "cropharvest" / "10class" / "classes.json"

# Normalization constants (from official CropHarvest dataset audit)
MEAN_VALUES = np.array([
    -11.402222508898584, -18.737524460893283, 1648.2152191941907, 1573.4610874943905,
    1615.5489068522484, 1858.2716898520655, 2601.3555762760366, 2965.1262098261586,
    2848.7703773525336, 3200.2996027800878, 988.3951744805588, 2388.875987761158,
    1603.9434589134196, 291.10068757630137, 0.002899168594319935, 623.3725288552483,
    4.772892120134574, 0.3286759541524565
], dtype=np.float32)

STD_VALUES = np.array([
    3.9990006137706025, 4.746129503502093, 1381.082565250503, 1291.5253102557383,
    1476.6850103027462, 1422.7793815055568, 1343.5806285212616, 1396.6833646173918,
    1336.0620748659192, 1406.3412200105693, 834.5202295842477, 1099.279298427952,
    947.4737894434511, 9.808570091118522, 0.0033269562766611444, 655.0334629731693,
    6.18644770735272, 0.17878359339732122
], dtype=np.float32)


# =============================================================================
# Model Architectures
# =============================================================================

class TSTransformer(nn.Module):
    """Satellite Time-Series Transformer architecture used in CropHarvest Kaggle training."""
    def __init__(self, in_dim=18, d_model=96, nhead=4, num_layers=3, dim_feedforward=192, num_classes=2, dropout=0.2):
        super().__init__()
        self.inp = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, 12, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(enc_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, num_classes))

    def forward(self, x):
        h = self.inp(x) + self.pos
        h = self.enc(h)
        h = self.norm(h)
        pool = torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=-1)
        return self.head(pool)


class CropHarvestBiLSTM(nn.Module):
    """Bidirectional LSTM architecture for CropHarvest 10-class phenology."""
    def __init__(self, input_dim=18, hidden_dim=64, num_layers=2, num_classes=10, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 12, 18)
        lstm_out, _ = self.lstm(x)  # (batch, 12, hidden_dim * 2)
        pooled = lstm_out.mean(dim=1)  # Temporal mean pooling
        normed = self.layer_norm(pooled)
        logits = self.classifier(normed)
        return logits


# =============================================================================
# Helper: Parse and Normalize Satellite Arrays
# =============================================================================

def parse_and_validate_satellite_array(file_bytes: bytes, filename: str) -> np.ndarray:
    """Parses binary bytes from .npy, .npz, .h5, or .csv into a (12, 18) float32 numpy array."""
    if len(file_bytes) == 0:
        raise ValueError("Uploaded file is empty.")

    lower = filename.lower()
    arr = None

    if lower.endswith(('.npy', '.npz')):
        try:
            loaded = np.load(io.BytesIO(file_bytes), allow_pickle=False)
            if isinstance(loaded, np.lib.npyio.NpzFile):
                # Take the first array in npz
                first_key = loaded.files[0]
                arr = loaded[first_key]
            else:
                arr = loaded
        except Exception as e:
            raise ValueError(f"Failed to parse numpy array from '{filename}': {str(e)}")

    elif lower.endswith(('.h5', '.hdf5')):
        try:
            import h5py
            with h5py.File(io.BytesIO(file_bytes), 'r') as hf:
                if 'x' in hf:
                    data = hf['x']
                    if data.ndim == 3:
                        arr = data[0]
                    else:
                        arr = data[()]
                else:
                    for k in hf.keys():
                        if isinstance(hf[k], h5py.Dataset):
                            ds = hf[k]
                            if ds.ndim == 2 and ds.shape in [(12, 18), (18, 12)]:
                                arr = ds[()]
                                break
                            elif ds.ndim == 3 and ds.shape[1:] in [(12, 18), (18, 12)]:
                                arr = ds[0]
                                break
                    if arr is None:
                        for k in hf.keys():
                            if isinstance(hf[k], h5py.Dataset):
                                arr = hf[k][()]
                                if arr.ndim == 3 and arr.shape[1:] == (12, 18):
                                    arr = arr[0]
                                break
            if arr is None:
                raise ValueError("No valid (12, 18) satellite dataset found inside HDF5 file.")
        except Exception as e:
            raise ValueError(f"Failed to parse HDF5 file '{filename}': {str(e)}")

    elif lower.endswith('.csv'):
        try:
            import pandas as pd
            df = pd.read_csv(io.BytesIO(file_bytes))
            # If CSV has header, convert values
            arr = df.values
        except Exception as e:
            raise ValueError(f"Failed to parse CSV array from '{filename}': {str(e)}")

    else:
        raise ValueError(f"Unsupported array extension for '{filename}'. Expected .npy, .h5, .hdf5, or .csv.")

    arr = np.asarray(arr, dtype=np.float32)

    # Dimensionality normalization
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr.squeeze(0)
        elif arr.shape[2] == 1:
            arr = arr.squeeze(2)

    if arr.ndim == 1:
        if arr.size == 216:
            arr = arr.reshape(12, 18)
        else:
            raise ValueError(f"1D array size {arr.size} cannot be reshaped to (12, 18). Expected exactly 216 numbers.")

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array of shape (12, 18), got shape {arr.shape}")

    # Check orientation (18, 12) vs (12, 18)
    if arr.shape == (18, 12):
        arr = arr.T

    if arr.shape != (12, 18):
        raise ValueError(f"Invalid satellite array shape {arr.shape}. Expected exactly 12 timesteps and 18 channels (12, 18).")

    # Check if raw vs normalized: raw bands 2-12 have values > 20.0
    if np.nanmax(np.abs(arr)) > 20.0:
        # Apply official dataset normalization: (x - mean) / std
        arr = (arr - MEAN_VALUES) / np.where(STD_VALUES == 0, 1.0, STD_VALUES)

    # Replace any NaNs or Infs with 0.0
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    return arr


# =============================================================================
# CropHarvest Inference Service Singleton
# =============================================================================

class CropHarvestInferenceService:
    _instance: Optional["CropHarvestInferenceService"] = None
    _lock = threading.Lock()

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.binary_models: List[nn.Module] = []
        self.model_10class: Optional[nn.Module] = None
        self.class10_names: Dict[int, str] = {}
        self._load_10class_classes()
        self._load_binary_models()
        self._load_10class_model()

    @classmethod
    def get_instance(cls) -> "CropHarvestInferenceService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _load_10class_classes(self) -> None:
        if CLASS10_CLASSES_PATH.exists():
            with open(CLASS10_CLASSES_PATH, "r", encoding="utf-8") as f:
                c = json.load(f)
                if isinstance(c, list):
                    self.class10_names = {i: name for i, name in enumerate(c)}
                elif isinstance(c, dict):
                    self.class10_names = {int(k): str(v) for k, v in c.items()}
        else:
            self.class10_names = {
                0: "cereals", 1: "other", 2: "fruits_nuts", 3: "vegetables_melons",
                4: "oilseeds", 5: "leguminous", 6: "root_tuber", 7: "beverage_spice",
                8: "sugar", 9: "non_crop"
            }

    def _load_binary_models(self) -> None:
        if not BINARY_SEEDS_PATH.exists():
            raise FileNotFoundError(f"Binary checkpoint not found at: {BINARY_SEEDS_PATH}")

        seeds = torch.load(BINARY_SEEDS_PATH, map_location="cpu", weights_only=False)
        self.binary_models = []
        for state_dict in seeds:
            m = TSTransformer(in_dim=18, num_classes=2)
            m.load_state_dict(state_dict, strict=True)
            m.eval()
            m.to(self.device)
            self.binary_models.append(m)

    def _load_10class_model(self) -> None:
        if not CLASS10_CKPT_PATH.exists():
            raise FileNotFoundError(f"10-class checkpoint not found at: {CLASS10_CKPT_PATH}")

        ckpt = torch.load(CLASS10_CKPT_PATH, map_location="cpu", weights_only=False)
        cfg = ckpt.get("model_config", {"input_dim": 18, "hidden_dim": 64, "num_layers": 2, "num_classes": 10})
        m = CropHarvestBiLSTM(**cfg)
        m.load_state_dict(ckpt["model_state_dict"], strict=True)
        m.eval()
        m.to(self.device)
        self.model_10class = m

    def predict_binary(self, file_bytes: bytes, filename: str = "satellite_array.npy") -> CropHarvestBinaryResponse:
        start_time = time.perf_counter()
        arr = parse_and_validate_satellite_array(file_bytes, filename)
        tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)

        # Average ensemble probabilities across 3 Transformer seeds
        with torch.inference_mode():
            all_probs = []
            for m in self.binary_models:
                logits = m(tensor)
                probs = F.softmax(logits, dim=1).squeeze(0)
                all_probs.append(probs)
            avg_probs = torch.stack(all_probs).mean(dim=0)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        p_non_crop = float(avg_probs[0].item())
        p_crop = float(avg_probs[1].item())
        is_crop = p_crop >= 0.50

        top_k = [
            ClassScore(class_name="Crop", class_index=1, confidence=round(p_crop, 4), category_type="Crop Category"),
            ClassScore(class_name="Non-Crop", class_index=0, confidence=round(p_non_crop, 4), category_type="Land-Cover Class")
        ]
        top_k.sort(key=lambda x: x.confidence, reverse=True)

        return CropHarvestBinaryResponse(
            task="CropHarvest Satellite Binary Crop Detection",
            is_crop=is_crop,
            label="Crop" if is_crop else "Non-Crop",
            crop_probability=round(p_crop, 4),
            non_crop_probability=round(p_non_crop, 4),
            confidence=round(max(p_crop, p_non_crop), 4),
            top_k=top_k,
            timesteps_verified=12,
            channels_verified=18,
            is_mock=False,
            model_architecture="3-Seed TSTransformer Ensemble (CropHarvest)",
            execution_time_ms=round(elapsed_ms, 2),
            status_notice="Verified Production Checkpoint: TSTransformer Seeds (82.02% Official Test Acc, 81.18% F1)"
        )

    def predict_10class(self, file_bytes: bytes, filename: str = "satellite_array.npy") -> CropHarvest10ClassResponse:
        start_time = time.perf_counter()
        arr = parse_and_validate_satellite_array(file_bytes, filename)
        tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.model_10class(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        top_probs, top_indices = torch.topk(probs, k=min(10, len(self.class10_names)))
        top_k: List[ClassScore] = []

        for p, idx_t in zip(top_probs, top_indices):
            idx = int(idx_t.item())
            prob_val = float(p.item())
            cname = self.class10_names.get(idx, f"Class_{idx}")
            cat_type = "Land-Cover Class" if cname == "non_crop" else "Crop Category"
            top_k.append(
                ClassScore(
                    class_name=cname,
                    class_index=idx,
                    confidence=round(prob_val, 4),
                    category_type=cat_type
                )
            )

        top_pred = top_k[0]

        return CropHarvest10ClassResponse(
            task="CropHarvest 10-Class Crop and Land-Cover Classification",
            predicted_class=top_pred.class_name,
            class_index=top_pred.class_index,
            category_type=top_pred.category_type,
            confidence=top_pred.confidence,
            top_k=top_k,
            timesteps_verified=12,
            channels_verified=18,
            is_mock=False,
            model_architecture="CropHarvestBiLSTM (10-Class)",
            execution_time_ms=round(elapsed_ms, 2),
            status_notice="Verified Production Checkpoint: CropHarvestBiLSTM (68.22% Benchmark Test Acc)"
        )


def predict_cropharvest_binary_real(file_bytes: bytes, filename: str = "array.npy") -> CropHarvestBinaryResponse:
    service = CropHarvestInferenceService.get_instance()
    return service.predict_binary(file_bytes=file_bytes, filename=filename)


def predict_cropharvest_10class_real(file_bytes: bytes, filename: str = "array.npy") -> CropHarvest10ClassResponse:
    service = CropHarvestInferenceService.get_instance()
    return service.predict_10class(file_bytes=file_bytes, filename=filename)
