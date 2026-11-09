"""
SmartCrop AI - CropHarvest 10-Class Classification Pipeline
Task: Predict the correct class from 9 FAO crop groups plus 1 land-cover class (non_crop)
      from 12-month x 18-channel satellite Earth observation time series.

Taxonomy (10 Classes):
- 9 Agricultural Crop Categories:
  0: cereals
  1: other
  2: fruits_nuts
  3: vegetables_melons
  4: oilseeds
  5: leguminous
  6: root_tuber
  7: beverage_spice
  8: sugar
- 1 Non-Crop Land-Cover Class:
  9: non_crop (Explicitly documented as non-crop land cover: pasture, forest, urban, water)

Architecture: 2-layer Bidirectional LSTM with temporal pooling and 10-class head.
Features:
- 18 Satellite/Climate/Topography channels normalized via official normalizing_dict.h5.
- Official benchmark split preservation: is_test=true (107 samples) held out for final test evaluation.
- Stratified 80:20 train/validation split on is_test=false pool (42,496 samples) with SEED=42.
- Class-weighted CrossEntropyLoss to address class imbalance.
- AdamW optimizer + Cosine Annealing learning rate schedule.
- Tracks Loss, Accuracy, Macro/Weighted F1, Precision, Recall, and 10x10 Confusion Matrix.
- Per-class test set audit reporting classes with low/zero official test samples.
- Saves best checkpoint, class mapping, metrics JSON to models/cropharvest/10class/.
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
OUTPUT_DIR = PROJECT_ROOT / "models" / "cropharvest" / "10class"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 10 Official Classes: 9 Crop Categories + 1 Land-Cover Class
CLASSES_10 = [
    "cereals", "other", "fruits_nuts", "vegetables_melons",
    "oilseeds", "leguminous", "root_tuber", "beverage_spice", "sugar",
    "non_crop"
]

CLASS_TYPES = {
    "cereals": "Crop Category",
    "other": "Crop Category",
    "fruits_nuts": "Crop Category",
    "vegetables_melons": "Crop Category",
    "oilseeds": "Crop Category",
    "leguminous": "Crop Category",
    "root_tuber": "Crop Category",
    "beverage_spice": "Crop Category",
    "sugar": "Crop Category",
    "non_crop": "Land-Cover Class (Non-Agricultural)"
}

CLS_TO_IDX = {c: i for i, c in enumerate(CLASSES_10)}
IDX_TO_CLS = {i: c for i, c in enumerate(CLASSES_10)}


class CropHarvest10ClassDataset(Dataset):
    """
    PyTorch Dataset for CropHarvest 10-class time-series arrays.
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
        target = rec["class_idx"]

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
    Output: (batch, 10)
    """
    def __init__(
        self,
        input_dim: int = 18,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 10,
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
        pooled = lstm_out.mean(dim=1)  # Temporal mean pooling
        normed = self.layer_norm(pooled)
        logits = self.classifier(normed)
        return logits


def evaluate_10class_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: List[str] = CLASSES_10
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

    acc = float(accuracy_score(all_targets, all_preds)) if len(all_targets) > 0 else 0.0
    prec, rec, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)
    w_prec, w_rec, w_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="weighted", zero_division=0)
    per_p, per_r, per_f, per_s = precision_recall_fscore_support(all_targets, all_preds, labels=list(range(len(class_names))), average=None, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(len(class_names)))).tolist()

    per_class_metrics = {}
    for i, cname in enumerate(class_names):
        per_class_metrics[cname] = {
            "type": CLASS_TYPES[cname],
            "precision": float(per_p[i]),
            "recall": float(per_r[i]),
            "f1": float(per_f[i]),
            "support": int(per_s[i])
        }

    return {
        "loss": float(total_loss / max(1, len(all_targets))),
        "accuracy": acc,
        "macro_precision": float(prec),
        "macro_recall": float(rec),
        "macro_f1": float(f1),
        "weighted_f1": float(w_f1),
        "per_class": per_class_metrics,
        "confusion_matrix": cm,
        "total_samples": len(all_targets)
    }


def run_10class_pipeline(
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
    print("=" * 80)
    print("  CROPHARVEST PIPELINE 2: 10-CLASS CLASSIFICATION")
    print("  (9 Verified FAO Crop Groups + 1 Non-Crop Land-Cover Class)")
    print("=" * 80)
    print(f"Dataset Directory : {dataset_dir.resolve()}")
    print(f"Output Directory  : {output_dir.resolve()}")

    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device    : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Load Normalizing Dict
    norm_path = dataset_dir / "normalizing_dict.h5"
    if not norm_path.exists():
        raise FileNotFoundError(f"Missing normalizing_dict.h5 in {dataset_dir}")
    with h5py.File(norm_path, "r") as h:
        mean = h["mean"][:]
        std = h["std"][:]

    # 2. Inspect Metadata & Extract 10-Class Records
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
            cl = props.get("classification_label")
            if cl in CLS_TO_IDX:
                records.append({
                    "filename": fn,
                    "filepath": str((arrays_dir / fn).resolve()),
                    "index": props["index"],
                    "dataset": props["dataset"],
                    "lat": float(props["lat"]),
                    "lon": float(props["lon"]),
                    "class_name": cl,
                    "class_idx": CLS_TO_IDX[cl],
                    "class_type": CLASS_TYPES[cl],
                    "is_test_official": bool(props.get("is_test", False))
                })

    print(f"Total eligible 10-class records: {len(records):,}")

    train_val_pool = [r for r in records if not r["is_test_official"]]
    test_records = [r for r in records if r["is_test_official"]]

    # Save Class Mapping
    with open(output_dir / "classes.json", "w", encoding="utf-8") as f:
        json.dump(IDX_TO_CLS, f, indent=2)

    # 3. Stratified 80:20 Partition of train_val_pool with SEED=42
    import random
    random.seed(42)

    train_records = []
    val_records = []

    for c in CLASSES_10:
        c_records = [r for r in train_val_pool if r["class_name"] == c]
        random.shuffle(c_records)
        n_val = int(len(c_records) * 0.20)
        val_records.extend(c_records[:n_val])
        train_records.extend(c_records[n_val:])

    random.shuffle(train_records)
    random.shuffle(val_records)

    train_counts = Counter(r["class_name"] for r in train_records)
    val_counts = Counter(r["class_name"] for r in val_records)
    test_counts = Counter(r["class_name"] for r in test_records)

    print("\n" + "=" * 85)
    print(f"{'Class Name':<20} | {'Type':<12} | {'Train (80%)':<12} | {'Val (20%)':<12} | {'Test (Official)':<16} | {'Test Support'}")
    print("-" * 85)
    for c in CLASSES_10:
        t_cnt = test_counts.get(c, 0)
        ctype = "Land-Cover" if c == "non_crop" else "Crop Group"
        if t_cnt >= 10:
            support_status = f"Adequate ({t_cnt})"
        elif t_cnt == 0:
            support_status = "CRITICAL: 0 samples"
        else:
            support_status = f"Warning: Very Low ({t_cnt})"
        print(f"{c:<20} | {ctype:<12} | {train_counts[c]:<12,} | {val_counts[c]:<12,} | {t_cnt:<16} | {support_status}")
    print("=" * 85)
    print(f"Total Partitions: Train={len(train_records):,}, Val={len(val_records):,}, Test={len(test_records):,}")

    if sanity_check:
        print("[SANITY CHECK] Subsampling records for dry run...")
        train_records = train_records[:128]
        val_records = val_records[:64]
        test_records = test_records[:32]
        epochs = 2
        batch_size = 32

    # 4. DataLoaders
    train_ds = CropHarvest10ClassDataset(train_records, mean, std, preload_into_memory=preload)
    val_ds = CropHarvest10ClassDataset(val_records, mean, std, preload_into_memory=preload)
    test_ds = CropHarvest10ClassDataset(test_records, mean, std, preload_into_memory=preload)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # 5. Class Weights for CrossEntropyLoss
    total_t = len(train_records)
    weights = [total_t / (10.0 * max(1, train_counts[c])) for c in CLASSES_10]
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    # 6. Model Setup
    model = CropHarvestBiLSTM(
        input_dim=18,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=10,
        dropout=0.2
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Initialized: Bi-LSTM ({total_params:,} parameters, 10 output classes)")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_f1 = -1.0
    best_checkpoint_path = output_dir / "best_model.pth"
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_macro_f1": []}

    print("\n" + "=" * 80)
    print(f"  STARTING 10-CLASS TRAINING: {epochs} EPOCHS (Batch Size={batch_size}, LR={lr})")
    print("=" * 80)
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
        train_loss = running_loss / max(1, total_train)
        train_acc = correct_train / max(1, total_train)

        # Validation Step
        val_eval = evaluate_10class_model(model, val_loader, device, CLASSES_10)

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
                "class_mapping": IDX_TO_CLS,
                "model_config": {
                    "input_dim": 18,
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                    "num_classes": 10
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

    val_final = evaluate_10class_model(model, val_loader, device, CLASSES_10)
    test_final = evaluate_10class_model(model, test_loader, device, CLASSES_10)

    # Per-class test sample audit report
    test_audit = {}
    for c in CLASSES_10:
        cnt = test_counts.get(c, 0)
        test_audit[c] = {
            "type": CLASS_TYPES[c],
            "official_test_samples": cnt,
            "sufficient_for_evaluation": bool(cnt >= 10),
            "status": "Adequate" if cnt >= 10 else ("CRITICAL: 0 samples" if cnt == 0 else f"Very Low ({cnt} sample)")
        }

    metrics_export = {
        "pipeline": "CropHarvest_10Class_Classification",
        "task": "10-Class Classification (9 FAO Crop Categories + 1 Land-Cover Class)",
        "architecture": "CropHarvestBiLSTM",
        "is_sanity_check": sanity_check,
        "checkpoint_status": "SANITY_CHECK_DRY_RUN (Not fully trained)" if sanity_check else "TRAINED_MODEL",
        "classes": CLASSES_10,
        "class_types": CLASS_TYPES,
        "training_time_seconds": round(elapsed_time, 2),
        "epochs_trained": epochs,
        "best_epoch": int(ckpt.get("epoch", epochs)),
        "validation_split": val_final,
        "official_held_out_test_split": test_final,
        "test_sample_audit": test_audit,
        "history": history
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_export, f, indent=2)

    print("\n" + "=" * 80)
    print("  FINAL 10-CLASS EVALUATION SCORECARD")
    print("=" * 80)
    print(f"{'Split':<30} | {'Samples':<10} | {'Accuracy':<12} | {'Macro-F1':<12} | {'Weighted-F1':<12}")
    print("-" * 80)
    print(f"{'Validation Split (Internal)':<30} | {val_final['total_samples']:<10} | {val_final['accuracy']*100:>10.2f}% | {val_final['macro_f1']*100:>10.2f}% | {val_final['weighted_f1']*100:>10.2f}%")
    print(f"{'Official Held-Out Test (Global)':<30} | {test_final['total_samples']:<10} | {test_final['accuracy']*100:>10.2f}% | {test_final['macro_f1']*100:>10.2f}% | {test_final['weighted_f1']*100:>10.2f}%")
    print("=" * 80)
    print(f"Metrics JSON exported to: {metrics_path.resolve()}")
    print(f"Best checkpoint saved to: {best_checkpoint_path.resolve()}")

    return metrics_export


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CropHarvest 10-Class Classification Pipeline")
    parser.add_argument("--sanity-check", action="store_true", help="Run quick 2-epoch dry run")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Bi-LSTM hidden dimension")
    parser.add_argument("--preload", action="store_true", help="Preload training arrays into RAM")
    args = parser.parse_args()

    run_10class_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        preload=args.preload,
        sanity_check=args.sanity_check
    )
