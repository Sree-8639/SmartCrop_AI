"""
SmartCrop AI - CropHarvest Satellite Crop Classification: Training & Evaluation Pipeline
Architecture: Bidirectional LSTM (Bi-LSTM) for 12-month Satellite Multi-Spectral Time Series

Features:
- 18 Satellite/Climate/Topography channels: Sentinel-1 SAR (VV, VH), Sentinel-2 Multi-Spectral, ERA5, SRTM.
- Normalization via official normalizing_dict.h5.
- Class-weighted CrossEntropyLoss to address crop/non-crop distribution.
- AdamW optimizer + Cosine Annealing learning rate schedule.
- Tracks Accuracy, Macro-F1, Weighted-F1, per-class metrics, and confusion matrix.
- Evaluates on Validation split, Official Held-Out Test split, and Togo/China regional testbeds.
- Sanity check mode (--sanity-check) for quick verification without full training.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets" / "cropharvest"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "cropharvest"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class CropHarvestDataset(Dataset):
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
        self.std = np.where(self.std == 0, 1.0, self.std)  # prevent div by zero
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
                    self.cached_arrays.append(arr_norm)
            print(f"[Dataset] Preloaded {len(self.cached_arrays):,} arrays in {time.time() - t0:.1f}s.")

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

        # Replace any potential residual NaNs/Infs with 0
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        tensor_x = torch.from_numpy(data)  # shape: (12, 18)
        return tensor_x, target


class CropHarvestBiLSTM(nn.Module):
    """
    Bidirectional LSTM Classifier for Satellite Time-Series Phenology Modeling.
    Input: (batch, 12, 18)
    Output: (batch, num_classes)
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
        # Global temporal mean pooling
        pooled = lstm_out.mean(dim=1)  # (batch, hidden_dim * 2)
        normed = self.layer_norm(pooled)
        logits = self.classifier(normed)  # (batch, num_classes)
        return logits


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: List[str] = ["non_crop", "crop"]
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
    per_cls_prec, per_cls_rec, per_cls_f1, cls_support = precision_recall_fscore_support(
        all_targets, all_preds, average=None, zero_division=0
    )
    cm = confusion_matrix(all_targets, all_preds).tolist()

    per_class_metrics = {}
    for i, cname in enumerate(class_names):
        if i < len(per_cls_prec):
            per_class_metrics[cname] = {
                "precision": float(per_cls_prec[i]),
                "recall": float(per_cls_rec[i]),
                "f1": float(per_cls_f1[i]),
                "support": int(cls_support[i])
            }

    return {
        "loss": float(total_loss / len(all_targets)),
        "accuracy": acc,
        "macro_precision": float(prec),
        "macro_recall": float(rec),
        "macro_f1": float(f1),
        "weighted_f1": float(w_f1),
        "per_class": per_class_metrics,
        "confusion_matrix": cm,
        "total_samples": len(all_targets)
    }


def evaluate_testbed_file(
    model: nn.Module,
    h5_path: Path,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device
) -> Optional[Dict[str, Any]]:
    if not h5_path.exists():
        return None

    with h5py.File(h5_path, "r") as h:
        x = h["x"][:].astype(np.float32)
        y = h["y"][:].astype(np.int64)

    # Filter out unannotated / invalid targets (-1 in some testbeds)
    valid_mask = (y >= 0)
    if not valid_mask.any():
        return None

    x = x[valid_mask]
    y = y[valid_mask]

    std_safe = np.where(std == 0, 1.0, std).astype(np.float32)
    x_norm = (x - mean.astype(np.float32)) / std_safe
    x_norm = np.nan_to_num(x_norm, nan=0.0, posinf=0.0, neginf=0.0)

    dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_norm), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    return evaluate_model(model, loader, device, class_names=["non_crop", "crop"])


def train_cropharvest_pipeline(
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
    print("  SMARTCROP AI: CROPHARVEST MODEL TRAINING & EVALUATION PIPELINE")
    print("=" * 75)

    # Device Setup
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Load Normalizing Dict
    norm_path = dataset_dir / "normalizing_dict.h5"
    if not norm_path.exists():
        raise FileNotFoundError(f"Missing normalizing_dict.h5 in {dataset_dir}")
    with h5py.File(norm_path, "r") as h:
        mean = h["mean"][:]
        std = h["std"][:]

    # Load Split Manifest
    manifest_path = output_dir / "cropharvest_split_manifest.json"
    if not manifest_path.exists():
        from prepare_cropharvest import create_split_manifest
        print("[Notice] Manifest missing. Generating cropharvest_split_manifest.json...")
        create_split_manifest(dataset_dir, output_dir)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    records = manifest["records"]
    train_records = [r for r in records if r["split"] == "train"]
    val_records = [r for r in records if r["split"] == "val"]
    test_records = [r for r in records if r["split"] == "test"]

    if sanity_check:
        print("[SANITY CHECK MODE] Subsampling 128 train and 64 val records for dry-run verification...")
        train_records = train_records[:128]
        val_records = val_records[:64]
        test_records = test_records[:32]
        epochs = 2
        batch_size = 32

    print(f"Data Partitions: Train={len(train_records):,}, Val={len(val_records):,}, Test={len(test_records):,}")

    # Build Datasets and DataLoaders
    train_ds = CropHarvestDataset(train_records, mean, std, preload_into_memory=preload)
    val_ds = CropHarvestDataset(val_records, mean, std, preload_into_memory=preload)
    test_ds = CropHarvestDataset(test_records, mean, std, preload_into_memory=preload)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Class Weights for CrossEntropyLoss (addressing ~70/30 class imbalance)
    n_train_crop = sum(1 for r in train_records if r["is_crop"] == 1)
    n_train_non_crop = len(train_records) - n_train_crop
    total_train = len(train_records)
    weight_non_crop = total_train / (2.0 * max(1, n_train_non_crop))
    weight_crop = total_train / (2.0 * max(1, n_train_crop))
    class_weights = torch.tensor([weight_non_crop, weight_crop], dtype=torch.float32).to(device)
    print(f"Class Weights : Non-Crop={weight_non_crop:.3f}, Crop={weight_crop:.3f}")

    # Initialize Model
    model = CropHarvestBiLSTM(
        input_dim=18,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=2,
        dropout=0.2
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Initialized: Bi-LSTM (Input=18, Hidden={hidden_dim}, Layers={num_layers}, Params={total_params:,})")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_f1 = -1.0
    best_checkpoint_path = output_dir / "cropharvest_bilstm_best.pth"
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_macro_f1": []}

    print("\n" + "=" * 75)
    print(f"  STARTING TRAINING: {epochs} EPOCHS (Batch Size={batch_size}, LR={lr})")
    print("=" * 75)
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train_samples = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * len(batch_y)
            preds = logits.argmax(dim=1)
            correct_train += (preds == batch_y).sum().item()
            total_train_samples += len(batch_y)

        scheduler.step()
        epoch_train_loss = running_loss / total_train_samples
        epoch_train_acc = correct_train / total_train_samples

        # Validation Step
        val_eval = evaluate_model(model, val_loader, device)

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(val_eval["loss"])
        history["val_acc"].append(val_eval["accuracy"])
        history["val_macro_f1"].append(val_eval["macro_f1"])

        is_best = val_eval["macro_f1"] > best_val_f1
        if is_best:
            best_val_f1 = val_eval["macro_f1"]
            torch.save({
                "epoch": epoch,
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
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc * 100:.2f}% | "
            f"Val Loss: {val_eval['loss']:.4f} | Val Acc: {val_eval['accuracy'] * 100:.2f}% | "
            f"Val Macro-F1: {val_eval['macro_f1'] * 100:.2f}%{star}"
        )

    elapsed_time = time.time() - t_start
    print(f"\nTraining completed in {elapsed_time:.1f} seconds. Best Val Macro-F1: {best_val_f1 * 100:.2f}%")

    # Load Best Model for Final Comprehensive Evaluation
    if best_checkpoint_path.exists():
        ckpt = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded best checkpoint from Epoch {ckpt['epoch']} for evaluation.")

    # 1. Validation Split Final Evaluation
    val_final = evaluate_model(model, val_loader, device)

    # 2. Official Held-Out Test Split Evaluation
    test_final = evaluate_model(model, test_loader, device)

    # 3. Togo Regional Evaluation Testbed (Zero-Shot / Out-of-Domain Transfer)
    togo_eval = evaluate_testbed_file(model, dataset_dir / "togo-eval.h5", mean, std, device)

    # 4. China Regional Evaluation Testbed (Regional Transfer)
    china_eval = evaluate_testbed_file(model, dataset_dir / "china-crop.h5", mean, std, device)

    metrics_export = {
        "architecture": "CropHarvestBiLSTM",
        "training_time_seconds": round(elapsed_time, 2),
        "epochs_trained": epochs,
        "best_epoch": int(ckpt.get("epoch", epochs)),
        "validation_split": val_final,
        "official_held_out_test_split": test_final,
        "togo_regional_testbed": togo_eval,
        "china_regional_testbed": china_eval,
        "history": history
    }

    metrics_path = output_dir / "cropharvest_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_export, f, indent=2)

    print("\n" + "=" * 75)
    print("  FINAL EVALUATION METRICS SCORECARD")
    print("=" * 75)
    print(f"{'Split / Testbed':<32} | {'Samples':<10} | {'Accuracy':<12} | {'Macro-F1':<12}")
    print("-" * 75)
    print(f"{'Validation Split (Internal)':<32} | {val_final['total_samples']:<10} | {val_final['accuracy']*100:>10.2f}% | {val_final['macro_f1']*100:>10.2f}%")
    print(f"{'Official Held-Out Test (Global)':<32} | {test_final['total_samples']:<10} | {test_final['accuracy']*100:>10.2f}% | {test_final['macro_f1']*100:>10.2f}%")
    if togo_eval:
        print(f"{'Togo Testbed (Zero-Shot Out-Domain)':<32} | {togo_eval['total_samples']:<10} | {togo_eval['accuracy']*100:>10.2f}% | {togo_eval['macro_f1']*100:>10.2f}%")
    if china_eval:
        print(f"{'China Testbed (Regional Transfer)':<32} | {china_eval['total_samples']:<10} | {china_eval['accuracy']*100:>10.2f}% | {china_eval['macro_f1']*100:>10.2f}%")
    print("=" * 75)
    print(f"Metrics JSON exported to: {metrics_path.resolve()}")
    print(f"Best checkpoint saved to: {best_checkpoint_path.resolve()}")

    return metrics_export


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CropHarvest Satellite Crop Classification Training")
    parser.add_argument("--sanity-check", action="store_true", help="Run quick 2-epoch dry run with subsampled data")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Bi-LSTM hidden dimension")
    parser.add_argument("--preload", action="store_true", help="Preload training arrays into RAM")
    args = parser.parse_args()

    train_cropharvest_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        preload=args.preload,
        sanity_check=args.sanity_check
    )

