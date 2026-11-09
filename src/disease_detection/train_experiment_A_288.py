"""
SmartCrop AI - Experiment 1: High-Resolution 288x288 with Two-Stage Discriminative Fine-Tuning
Isolates the effect of:
1. 288x288 input resolution (to preserve fine punctate foliar spot lesions).
2. Two-stage training schedule:
   - Stage 1: Frozen backbone, train classification head only (5 epochs, lr=1e-3).
   - Stage 2: Unfrozen backbone, differential learning rates (backbone 2e-5, head 2e-4, 25 epochs with Cosine Annealing).
3. Early stopping based on validation Macro-F1.
4. Strictly isolated test set (tuning on validation only).
"""

import os
import sys
import time
import json
import logging
import argparse
import urllib.parse
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Experiment_A_288")

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "disease"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATASETS_DIR = PROJECT_ROOT / "datasets" / "plantdoc"
MANIFEST_PATH = OUTPUTS_DIR / "plantdoc_split_manifest.json"
CLASSES_PATH = MODELS_DIR / "disease_classes.json"

TARGET_RESOLUTION = 288
RESIZE_DIM = 320
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class PlantDocDataset(Dataset):
    def __init__(self, records: List[Tuple[str, int, str]], transform=None):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        path, label, _ = self.records[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image from path: {path}. Error: {e}")

        if self.transform:
            image = self.transform(image)
        return image, label


def build_efficientnet_b0(num_classes: int = 27, dropout: float = 0.4) -> nn.Module:
    """Builds EfficientNet-B0 with custom classification head."""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(512, num_classes)
    )
    return model


def get_data_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Returns 288x288 training and validation transforms."""
    train_transforms = transforms.Compose([
        transforms.Resize((RESIZE_DIM, RESIZE_DIM)),
        transforms.RandomResizedCrop(TARGET_RESOLUTION, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((RESIZE_DIM, RESIZE_DIM)),
        transforms.CenterCrop(TARGET_RESOLUTION),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return train_transforms, val_transforms


def resolve_entry_metadata(entry: Any) -> Tuple[str, str]:
    """Extracts relative path and class name across cross-platform formats."""
    if isinstance(entry, dict):
        rel_path = entry.get("file") or entry.get("path") or entry.get("image") or ""
        cls_name = entry.get("class") or entry.get("label") or entry.get("disease") or ""
        if not cls_name and rel_path:
            norm = str(rel_path).replace("\\", "/")
            parts = [p for p in norm.split("/") if p]
            if len(parts) >= 2:
                cls_name = parts[-2]
        return str(rel_path).strip(), str(cls_name).strip()
    elif isinstance(entry, str):
        rel_path = entry.strip()
        norm = rel_path.replace("\\", "/")
        parts = [p for p in norm.split("/") if p]
        if len(parts) >= 2:
            cls_name = parts[-2].strip()
        else:
            cls_name = ""
        return rel_path, cls_name
    else:
        return str(entry).strip(), ""


def load_dataset_splits() -> Tuple[List, List, List, Dict[str, str]]:
    """Loads exact verified split manifest without modifying records."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Split manifest not found at: {MANIFEST_PATH}")
    if not CLASSES_PATH.exists():
        raise FileNotFoundError(f"Classes mapping not found at: {CLASSES_PATH}")

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        classes_dict = json.load(f)

    # Class name to index mapping
    cls_to_idx = {name: int(idx) for idx, name in classes_dict.items()}

    def build_records(partition_name: str, path_list: List[Any]) -> List[Tuple[str, int, str]]:
        records = []
        malformed = []
        unknown = []
        missing = []

        for idx, entry in enumerate(path_list):
            rel_path, cls_name = resolve_entry_metadata(entry)
            if not rel_path or not cls_name:
                malformed.append((idx, entry))
                continue
            if cls_name not in cls_to_idx:
                unknown.append((idx, cls_name))
                continue

            norm_rel = rel_path.replace("\\", "/")
            candidates = [
                DATASETS_DIR / norm_rel,
                DATASETS_DIR / rel_path,
                DATASETS_DIR / norm_rel.replace("&", "_"),
                DATASETS_DIR / urllib.parse.unquote(norm_rel),
                DATASETS_DIR / urllib.parse.unquote(norm_rel).replace("&", "_")
            ]
            full_path = None
            for cand in candidates:
                if cand.exists() and cand.is_file():
                    full_path = cand
                    break

            if full_path is None:
                missing.append((idx, rel_path, cls_name))
                continue

            records.append((str(full_path), cls_to_idx[cls_name], cls_name))

        logger.info(f"Partition {partition_name.upper()}: Total={len(path_list)}, Mapped={len(records)}, Missing={len(missing)}, UnknownCls={len(unknown)}, Malformed={len(malformed)}")
        if malformed:
            raise ValueError(f"Malformed entries in {partition_name}: {malformed[:5]}")
        if unknown:
            raise KeyError(f"Unknown classes in {partition_name}: {unknown[:5]}")
        return records

    train_records = build_records("train", manifest["train"])
    val_records = build_records("val", manifest["val"])
    test_records = build_records("test", manifest["test"])

    logger.info(f"Loaded records: Train={len(train_records)}, Val={len(val_records)}, Test={len(test_records)}")
    return train_records, val_records, test_records, classes_dict


def evaluate_model(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str) -> Dict:
    """Evaluates model performance on validation set."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())

    total_samples = len(loader.dataset)
    val_loss = running_loss / total_samples
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    val_acc = float(np.mean(all_preds == all_targets))
    prec, rec, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)
    w_f1 = precision_recall_fscore_support(all_targets, all_preds, average="weighted", zero_division=0)[2]

    return {
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_macro_f1": float(f1),
        "val_macro_precision": float(prec),
        "val_macro_recall": float(rec),
        "val_weighted_f1": float(w_f1),
        "predictions": all_preds.tolist(),
        "targets": all_targets.tolist()
    }


def train_experiment_A(
    batch_size: int = 32,
    stage1_epochs: int = 5,
    stage2_epochs: int = 25,
    patience: int = 6,
    num_workers: int = 0,
    device: str = None
):
    start_total_time = time.time()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("=" * 80)
    logger.info("SMARTCROP AI - EXPERIMENT 1: HIGH-RES 288x288 TWO-STAGE FINE-TUNING")
    logger.info("=" * 80)
    logger.info(f"Target Device: {device}")
    logger.info(f"Batch Size: {batch_size}, Image Resolution: {TARGET_RESOLUTION}x{TARGET_RESOLUTION}")

    # Inspect device and memory
    memory_info = {}
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        total_gpu = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"Initial GPU Memory: Allocated={allocated:.2f} GB, Reserved={reserved:.2f} GB, Total={total_gpu:.2f} GB")
        memory_info["initial_gpu_allocated_gb"] = allocated
        memory_info["total_gpu_memory_gb"] = total_gpu
    else:
        logger.info("Running on CPU architecture. No CUDA hardware acceleration detected.")
        memory_info["device"] = "cpu"

    # Data loaders
    train_records, val_records, test_records, classes_dict = load_dataset_splits()
    train_transforms, val_transforms = get_data_transforms()

    train_dataset = PlantDocDataset(train_records, transform=train_transforms)
    val_dataset = PlantDocDataset(val_records, transform=val_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda")
    )

    num_classes = len(classes_dict)
    model = build_efficientnet_b0(num_classes=num_classes, dropout=0.4).to(device)

    # Class weights for training loss
    train_labels = [r[1] for r in train_records]
    class_counts = [max(1, train_labels.count(c)) for c in range(num_classes)]
    total_train = len(train_labels)
    class_weights = [total_train / (num_classes * count) for count in class_counts]
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    history = {
        "stage1": [],
        "stage2": []
    }

    best_val_f1 = 0.0
    best_val_acc = 0.0
    best_epoch = -1
    best_checkpoint_path = MODELS_DIR / "disease_model_288_v1.pth"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # ==========================================================================
    # STAGE 1: Head Warmup (Freeze Backbone, Train Classifier Head Only)
    # ==========================================================================
    logger.info("-" * 80)
    logger.info(f"STAGE 1: Head Warmup ({stage1_epochs} epochs, lr=1e-3, Backbone Frozen)")
    logger.info("-" * 80)

    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    optimizer_stage1 = optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-2)

    for epoch in range(1, stage1_epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        correct = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer_stage1.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_stage1.step()

            train_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()

        epoch_train_loss = train_loss / len(train_dataset)
        epoch_train_acc = correct / len(train_dataset)

        val_metrics = evaluate_model(model, val_loader, criterion, device)
        epoch_time = time.time() - epoch_start

        logger.info(
            f"Stage 1 [Epoch {epoch:02d}/{stage1_epochs:02d}] "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | Val Acc: {val_metrics['val_accuracy']*100:.2f}% | "
            f"Val F1: {val_metrics['val_macro_f1']*100:.2f}% | Time: {epoch_time:.1f}s"
        )

        history["stage1"].append({
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "train_acc": epoch_train_acc,
            "val_loss": val_metrics["val_loss"],
            "val_acc": val_metrics["val_accuracy"],
            "val_f1": val_metrics["val_macro_f1"],
            "epoch_time_s": epoch_time
        })

        if val_metrics["val_macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["val_macro_f1"]
            best_val_acc = val_metrics["val_accuracy"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_f1": best_val_f1,
                "val_acc": best_val_acc,
                "model_name": "EfficientNetB0_288_Stage1_Warmup",
                "resolution": TARGET_RESOLUTION
            }, best_checkpoint_path)
            logger.info(f"  --> Saved new best checkpoint at Stage 1 Epoch {epoch} (Val F1: {best_val_f1*100:.2f}%)")

    # ==========================================================================
    # STAGE 2: Full Backbone Discriminative Fine-Tuning
    # ==========================================================================
    logger.info("-" * 80)
    logger.info(f"STAGE 2: Full Backbone Fine-Tuning ({stage2_epochs} epochs max, Differential LR)")
    logger.info("-" * 80)

    for param in model.parameters():
        param.requires_grad = True

    optimizer_stage2 = optim.AdamW([
        {"params": model.features.parameters(), "lr": 2e-5},
        {"params": model.classifier.parameters(), "lr": 2e-4}
    ], weight_decay=1e-2)

    scheduler_stage2 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_stage2, T_max=stage2_epochs, eta_min=1e-6
    )

    epochs_without_improvement = 0

    for epoch in range(1, stage2_epochs + 1):
        global_epoch = stage1_epochs + epoch
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        correct = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer_stage2.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_stage2.step()

            train_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()

        scheduler_stage2.step()
        epoch_train_loss = train_loss / len(train_dataset)
        epoch_train_acc = correct / len(train_dataset)

        val_metrics = evaluate_model(model, val_loader, criterion, device)
        epoch_time = time.time() - epoch_start

        logger.info(
            f"Stage 2 [Epoch {epoch:02d}/{stage2_epochs:02d} (Global {global_epoch:02d})] "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | Val Acc: {val_metrics['val_accuracy']*100:.2f}% | "
            f"Val F1: {val_metrics['val_macro_f1']*100:.2f}% | Time: {epoch_time:.1f}s"
        )

        history["stage2"].append({
            "stage_epoch": epoch,
            "global_epoch": global_epoch,
            "train_loss": epoch_train_loss,
            "train_acc": epoch_train_acc,
            "val_loss": val_metrics["val_loss"],
            "val_acc": val_metrics["val_accuracy"],
            "val_f1": val_metrics["val_macro_f1"],
            "epoch_time_s": epoch_time
        })

        if val_metrics["val_macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["val_macro_f1"]
            best_val_acc = val_metrics["val_accuracy"]
            best_epoch = global_epoch
            epochs_without_improvement = 0
            torch.save({
                "epoch": global_epoch,
                "model_state_dict": model.state_dict(),
                "val_f1": best_val_f1,
                "val_acc": best_val_acc,
                "model_name": "EfficientNetB0_288_Stage2_FineTuned",
                "resolution": TARGET_RESOLUTION
            }, best_checkpoint_path)
            logger.info(f"  --> Saved new best checkpoint at Global Epoch {global_epoch} (Val F1: {best_val_f1*100:.2f}%)")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without validation Macro-F1 improvement.")
                break

    total_training_time_s = time.time() - start_total_time
    logger.info("=" * 80)
    logger.info(f"TRAINING COMPLETE. Best Global Epoch: {best_epoch} | Best Val F1: {best_val_f1*100:.2f}% | Best Val Acc: {best_val_acc*100:.2f}%")
    logger.info(f"Total Execution Time: {total_training_time_s/60:.2f} minutes")
    logger.info("=" * 80)

    # Load best model for validation evaluation
    best_ckpt = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    final_val_metrics = evaluate_model(model, val_loader, criterion, device)

    # Classification report & confusion matrix on validation
    val_targets = final_val_metrics["targets"]
    val_preds = final_val_metrics["predictions"]
    target_names = [classes_dict[str(i)] for i in range(num_classes)]

    val_cls_report = classification_report(
        val_targets, val_preds, target_names=target_names, output_dict=True, zero_division=0
    )
    val_cm = confusion_matrix(val_targets, val_preds).tolist()

    if device == "cuda":
        peak_gpu_mem = torch.cuda.max_memory_allocated() / (1024**3)
        memory_info["peak_gpu_allocated_gb"] = peak_gpu_mem
        logger.info(f"Peak GPU Memory Allocated: {peak_gpu_mem:.2f} GB")

    # Export experiment metrics JSON
    metrics_export = {
        "experiment_name": "Experiment_1_HighRes_288x288_TwoStage_FineTuning",
        "model_architecture": "EfficientNet-B0",
        "input_resolution": [TARGET_RESOLUTION, TARGET_RESOLUTION],
        "batch_size": batch_size,
        "total_training_time_seconds": total_training_time_s,
        "total_training_time_minutes": total_training_time_s / 60.0,
        "hardware": memory_info,
        "best_epoch": best_epoch,
        "validation_metrics": {
            "val_loss": final_val_metrics["val_loss"],
            "val_accuracy": final_val_metrics["val_accuracy"],
            "val_macro_f1": final_val_metrics["val_macro_f1"],
            "val_macro_precision": final_val_metrics["val_macro_precision"],
            "val_macro_recall": final_val_metrics["val_macro_recall"],
            "val_weighted_f1": final_val_metrics["val_weighted_f1"]
        },
        "per_class_validation_report": val_cls_report,
        "validation_confusion_matrix": val_cm,
        "training_history": history
    }

    metrics_path = OUTPUTS_DIR / "plantdoc_experiment_A_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_export, f, indent=2)
    logger.info(f"Exported metrics JSON to: {metrics_path}")

    return metrics_export


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Experiment 1: High-Res 288x288 EfficientNet-B0")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--stage1_epochs", type=int, default=5, help="Stage 1 head warmup epochs")
    parser.add_argument("--stage2_epochs", type=int, default=25, help="Stage 2 fine-tuning epochs")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    args = parser.parse_args()

    train_experiment_A(
        batch_size=args.batch_size,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        patience=args.patience,
        num_workers=args.num_workers
    )

