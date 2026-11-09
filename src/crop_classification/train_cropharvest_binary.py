"""
SmartCrop AI - CropHarvest Binary Crop Classification Pipeline
Task: Predict Class 0 (Non-Crop) vs Class 1 (Crop) from 12-month x 18-channel satellite time series.

Architecture: 2-layer Bidirectional LSTM with temporal pooling and classification head.
Features:
- Normalized inputs using official normalizing_dict.h5 (Sentinel-1, Sentinel-2, ERA5, SRTM).
- Official benchmark split preservation: is_test=true (4,810 samples) held out strictly for final evaluation.
- Stratified 80:20 train/validation split on is_test=false pool (82,654 samples) with SEED=42.
- Class-weighted CrossEntropyLoss to address ~69/31 crop/non-crop imbalance.
- AdamW optimizer + Cosine Annealing learning rate schedule.
- Comprehensive metrics tracking: Loss, Accuracy, Macro/Weighted F1, Precision, Recall, Confusion Matrix.
- Evaluation on official global test set and out-of-domain Togo and China regional testbeds.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets" / "cropharvest"
OUTPUT_DIR = PROJECT_ROOT / "models" / "cropharvest" / "binary"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHANNELS_18 = [
    "S1_VV", "S1_VH",
    "S2_B2_Blue", "S2_B3_Green", "S2_B4_Red", "S2_B5_RE1",
    "S2_B6_RE2", "S2_B7_RE3", "S2_B8_NIR", "S2_B8A_NNIR",
    "S2_B11_SWIR1", "S2_B12_SWIR2",
    "ERA5_Total_Precipitation", "ERA5_Temperature_2m",
    "ERA5_Dewpoint_Temp", "SRTM_Elevation", "SRTM_Slope", "ERA5_Solar_Radiation"
]

CLASS_MAPPING = {0: "non_crop", 1: "crop"}


class CropHarvestBinaryDataset(Dataset):
    """
    PyTorch Dataset for CropHarvest satellite time-series arrays.
    Each sample has shape (12, 18): 12 months x 18 channels.
    """
    def __init__(
        self,
        records: List[Dict[str, Any]],
        mean: np.ndarray,
        std: np.ndarray,
        preload_into_memory: bool = False
    ):
        self.records = records
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.std = np.where(self.std == 0, 1.0, self.std)
        self.preload = preload_into_memory
        self.cached_arrays = None

        if self.preload:
            print(f"[Dataset] Preloading {len(self.records):,} arrays into RAM for fast training...")
            t0 = time.time()
            self.cached_arrays = []
            for r in self.records:
                with h5py.File(r["filepath"], "r") as h:
                    arr = h["array"][:].astype(np.float32)
                    arr_norm = (arr - self.mean) / self.std
                    arr_norm = np.nan_to_num(arr_norm, nan=0.0, posinf=0.0, neginf=0.0)
                    self.cached_arrays.append(arr_norm)
            print(f"[Dataset] Preloaded in {time.time() - t0:.1f}s.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        rec = self.records[idx]
        target = rec["is_crop"]

        if self.preload and self.cached_arrays is not None:
            data = self.cached_arrays[idx]
        else:
            with h5py.File(rec["filepath"], "r") as h:
                arr = h["array"][:].astype(np.float32)
            data = (arr - self.mean) / self.std
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        return torch.from_numpy(data), target


class CropHarvestBiLSTM(nn.Module):
    """
    Bidirectional LSTM Classifier for Satellite Time-Series Phenology Modeling.
    Input: (batch, 12, 18)
    Output: (batch, 2)
    """
    def __init__(
        self,
        input_dim: int = 18,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
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


def evaluate_binary_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, Any]:
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * len(y)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = float(accuracy_score(all_targets, all_preds))
    prec, rec, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)
    w_prec, w_rec, w_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="weighted", zero_division=0)
    per_p, per_r, per_f, per_s = precision_recall_fscore_support(all_targets, all_preds, average=None, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds).tolist()

    return {
        "loss": float(total_loss / max(1, len(all_targets))),
        "accuracy": acc,
        "macro_precision": float(prec),
        "macro_recall": float(rec),
        "macro_f1": float(f1),
        "weighted_f1": float(w_f1),
        "per_class": {
            "non_crop": {
                "precision": float(per_p[0]) if len(per_p) > 0 else 0.0,
                "recall": float(per_r[0]) if len(per_r) > 0 else 0.0,
                "f1": float(per_f[0]) if len(per_f) > 0 else 0.0,
                "support": int(per_s[0]) if len(per_s) > 0 else 0
            },
            "crop": {
                "precision": float(per_p[1]) if len(per_p) > 1 else 0.0,
                "recall": float(per_r[1]) if len(per_r) > 1 else 0.0,
                "f1": float(per_f[1]) if len(per_f) > 1 else 0.0,
                "support": int(per_s[1]) if len(per_s) > 1 else 0
            }
        },
        "confusion_matrix": cm,
        "total_samples": len(all_targets)
    }


def evaluate_testbed(
    model: nn.Module,
    h5_path: Path,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device
) -> Optional[Dict[str, Any]]:
    if not h5_path.exists():
        return None
    with h5py.File(h5_path, "r") as h:
        x_raw = h["x"][:].astype(np.float32)
        y_raw = h["y"][:].astype(np.int64)

    valid_mask = (y_raw >= 0)
    if not valid_mask.any():
        return None

    x_valid = x_raw[valid_mask]
    y_valid = y_raw[valid_mask]

    std_safe = np.where(std == 0, 1.0, std).astype(np.float32)
    x_norm = (x_valid - mean.astype(np.float32)) / std_safe
    x_norm = np.nan_to_num(x_norm, nan=0.0, posinf=0.0, neginf=0.0)

    ds = torch.utils.data.TensorDataset(torch.from_numpy(x_norm), torch.from_numpy(y_valid))
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    return evaluate_binary_model(model, loader, device)


def run_binary_pipeline(
    dataset_dir: Path = DATASET_DIR,
    output_dir: Path = OUTPUT_DIR,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    num_layers: int = 2,
    preload: bool = False,
    sanity_check: bool = False,
    device_str: Optional[str] = None
) -> Dict[str, Any]:
    print("=" * 75)
    print("  CROPHARVEST PIPELINE 1: BINARY CROP CLASSIFICATION (CROP VS NON-CROP)")
    print("=" * 75)
    print(f"Dataset Directory : {dataset_dir.resolve()}")
    print(f"Output Directory  : {output_dir.resolve()}")

    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device    : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Load Normalization Dict
    norm_path = dataset_dir / "normalizing_dict.h5"
    if not norm_path.exists():
        raise FileNotFoundError(f"Missing normalizing_dict.h5 in {dataset_dir}")
    with h5py.File(norm_path, "r") as h:
        mean = h["mean"][:]
        std = h["std"][:]

    # 2. Load and Partition Dataset Records
    arrays_dir = dataset_dir / "arrays"
    geojson_path = dataset_dir / "labels.geojson"

    with open(geojson_path, "r", encoding="utf-8") as f:
        geo_data = json.load(f)

    available_names = set(p.name for p in arrays_dir.glob("*.h5"))
    records = []

    for feat in geo_data["features"]:
        props = feat["properties"]
        fn = f"{props['index']}_{props['dataset']}.h5"
        if fn in available_names:
            records.append({
                "filename": fn,
                "filepath": str((arrays_dir / fn).resolve()),
                "index": props["index"],
                "dataset": props["dataset"],
                "lat": float(props["lat"]),
                "lon": float(props["lon"]),
                "is_crop": int(props["is_crop"]),
                "is_test_official": bool(props.get("is_test", False))
            })

    print(f"Total verified matched arrays: {len(records):,}")

    train_val_pool = [r for r in records if not r["is_test_official"]]
    test_records = [r for r in records if r["is_test_official"]]

    # Stratified 80:20 split of train_val_pool with SEED=42
    import random
    random.seed(42)
    crop_pool = [r for r in train_val_pool if r["is_crop"] == 1]
    non_crop_pool = [r for r in train_val_pool if r["is_crop"] == 0]
    random.shuffle(crop_pool)
    random.shuffle(non_crop_pool)

    n_val_crop = int(len(crop_pool) * 0.20)
    n_val_non_crop = int(len(non_crop_pool) * 0.20)

    val_records = crop_pool[:n_val_crop] + non_crop_pool[:n_val_non_crop]
    train_records = crop_pool[n_val_crop:] + non_crop_pool[n_val_non_crop:]
    random.shuffle(train_records)
    random.shuffle(val_records)

    if sanity_check:
        print("[SANITY CHECK] Subsampling 128 train and 64 val records for dry run...")
        train_records = train_records[:128]
        val_records = val_records[:64]
        test_records = test_records[:32]
        epochs = 2
        batch_size = 32

    print(f"Data Partitions: Train={len(train_records):,}, Val={len(val_records):,}, Test={len(test_records):,}")

    # Save Class Mapping
    with open(output_dir / "classes.json", "w", encoding="utf-8") as f:
        json.dump(CLASS_MAPPING, f, indent=2)

    # 3. DataLoaders
    train_ds = CropHarvestBinaryDataset(train_records, mean, std, preload_into_memory=preload)
    val_ds = CropHarvestBinaryDataset(val_records, mean, std, preload_into_memory=preload)
    test_ds = CropHarvestBinaryDataset(test_records, mean, std, preload_into_memory=preload)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Class Weights for CrossEntropyLoss
    n_c = sum(1 for r in train_records if r["is_crop"] == 1)
    n_nc = len(train_records) - n_c
    total = len(train_records)
    w_nc = total / (2.0 * max(1, n_nc))
    w_c = total / (2.0 * max(1, n_c))
    class_weights = torch.tensor([w_nc, w_c], dtype=torch.float32).to(device)
    print(f"Class Weights : Non-Crop={w_nc:.3f}, Crop={w_c:.3f}")

    # 4. Model Setup
    model = CropHarvestBiLSTM(
        input_dim=18,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=2,
        dropout=0.2
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Initialized: Bi-LSTM ({total_params:,} parameters)")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_f1 = -1.0
    best_checkpoint_path = output_dir / "best_model.pth"
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_macro_f1": []}

    print("\n" + "=" * 75)
    print(f"  STARTING TRAINING: {epochs} EPOCHS (Batch Size={batch_size}, LR={lr})")
    print("=" * 75)
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * len(by)
            preds = logits.argmax(dim=1)
            correct_train += (preds == by).sum().item()
            total_train += len(by)

        scheduler.step()
        train_loss = running_loss / total_train
        train_acc = correct_train / total_train

        # Validation Step
        val_eval = evaluate_binary_model(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_eval["loss"])
        history["val_acc"].append(val_eval["accuracy"])
        history["val_macro_f1"].append(val_eval["macro_f1"])

        is_best = val_eval["macro_f1"] > best_val_f1
        if is_best:
            best_val_f1 = val_eval["macro_f1"]
            torch.save({
                "epoch": epoch,
                "is_sanity_check": sanity_check,
                "checkpoint_status": "SANITY_CHECK_DRY_RUN (Not fully trained)" if sanity_check else "TRAINED_CHECKPOINT",
                "model_state_dict": model.state_dict(),
                "val_accuracy": val_eval["accuracy"],
                "val_macro_f1": val_eval["macro_f1"],
                "model_config": {
                    "input_dim": 18,
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                    "num_classes": 2
                }
            }, best_checkpoint_path)

        star = " ★ BEST" if is_best else ""
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_eval['loss']:.4f} | Val Acc: {val_eval['accuracy']*100:.2f}% | "
            f"Val Macro-F1: {val_eval['macro_f1']*100:.2f}%{star}"
        )

    elapsed_time = time.time() - t_start
    print(f"\nTraining completed in {elapsed_time:.1f}s. Best Val Macro-F1: {best_val_f1*100:.2f}%")

    # Load Best Model for Final Evaluation
    if best_checkpoint_path.exists():
        ckpt = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded best checkpoint from Epoch {ckpt['epoch']} for final evaluation.")

    # 1. Validation Split Final Evaluation
    val_final = evaluate_binary_model(model, val_loader, device)

    # 2. Official Global Held-Out Test Evaluation
    test_final = evaluate_binary_model(model, test_loader, device)

    # 3. Regional Testbeds
    togo_eval = evaluate_testbed(model, dataset_dir / "togo-eval.h5", mean, std, device)
    china_eval = evaluate_testbed(model, dataset_dir / "china-crop.h5", mean, std, device)

    metrics_export = {
        "pipeline": "CropHarvest_Binary_Crop_Classification",
        "task": "is_crop (0: Non-Crop, 1: Crop)",
        "architecture": "CropHarvestBiLSTM",
        "is_sanity_check": sanity_check,
        "checkpoint_status": "SANITY_CHECK_DRY_RUN (Not fully trained)" if sanity_check else "TRAINED_MODEL",
        "training_time_seconds": round(elapsed_time, 2),
        "epochs_trained": epochs,
        "best_epoch": int(ckpt.get("epoch", epochs)),
        "validation_split": val_final,
        "official_held_out_test_split": test_final,
        "togo_regional_testbed": togo_eval,
        "china_regional_testbed": china_eval,
        "history": history
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_export, f, indent=2)

    print("\n" + "=" * 75)
    print("  FINAL BINARY EVALUATION SCORECARD")
    print("=" * 75)
    print(f"{'Split / Testbed':<34} | {'Samples':<10} | {'Accuracy':<12} | {'Macro-F1':<12}")
    print("-" * 75)
    print(f"{'Validation Split (Internal)':<34} | {val_final['total_samples']:<10} | {val_final['accuracy']*100:>10.2f}% | {val_final['macro_f1']*100:>10.2f}%")
    print(f"{'Official Held-Out Test (Global)':<34} | {test_final['total_samples']:<10} | {test_final['accuracy']*100:>10.2f}% | {test_final['macro_f1']*100:>10.2f}%")
    if togo_eval:
        print(f"{'Togo Testbed (Zero-Shot Out-Domain)':<34} | {togo_eval['total_samples']:<10} | {togo_eval['accuracy']*100:>10.2f}% | {togo_eval['macro_f1']*100:>10.2f}%")
    if china_eval:
        print(f"{'China Testbed (Regional Transfer)':<34} | {china_eval['total_samples']:<10} | {china_eval['accuracy']*100:>10.2f}% | {china_eval['macro_f1']*100:>10.2f}%")
    print("=" * 75)
    print(f"Metrics JSON exported to: {metrics_path.resolve()}")
    print(f"Best checkpoint saved to: {best_checkpoint_path.resolve()}")

    return metrics_export


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CropHarvest Binary Crop Classification")
    parser.add_argument("--sanity-check", action="store_true", help="Run quick 2-epoch dry run")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Bi-LSTM hidden dimension")
    parser.add_argument("--preload", action="store_true", help="Preload training arrays into RAM")
    args = parser.parse_args()

    run_binary_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        preload=args.preload,
        sanity_check=args.sanity_check
    )

