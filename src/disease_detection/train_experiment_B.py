"""
SmartCrop AI - PlantDoc Experiment B: Advanced Training Pipeline
Final Target: >= 80% Training Accuracy AND >= 80% Validation Accuracy.

Key Technical Components:
1. High Resolution: 288x288 spatial input resolution.
2. Regularization & Augmentation:
   - Field data augmentation (ColorJitter, Affine Rotation, Horizontal/Vertical Flips).
   - Random Erasing (p=0.25) to prevent background bias.
   - Dropout (0.4) and Weight Decay (1e-2).
3. Class-Balanced Focal Loss with Label Smoothing:
   - Effective number of samples factor (beta=0.9999).
   - Focal downweighting of easy background leaves (gamma=1.5).
   - Label smoothing (epsilon=0.1) for field illumination ambiguity.
4. Two-Stage Discriminative Schedule:
   - Stage 1: 5 epochs head warmup (backbone frozen, lr=1e-3).
   - Stage 2: Up to 35 epochs fine-tuning with differential LR (backbone 2e-5, head 2e-4) and Cosine Annealing.
5. Primary Checkpoint Selection: Validation Accuracy (with Macro-F1 logged).
6. Full Epoch-by-Epoch Metric Tracking (Train Loss, Val Loss, Train Acc, Val Acc).
7. Strict empirical target verification (no fabricated claims).
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
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Experiment_B")

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


class ClassBalancedFocalLossWithSmoothing(nn.Module):
    """
    Class-Balanced Focal Loss with Label Smoothing.
    - Effective number of samples weighting: En = (1 - beta^n) / (1 - beta)
    - Focal loss attenuation for easy examples: (1 - pt)^gamma
    - Label smoothing: epsilon = 0.1
    """
    def __init__(
        self,
        samples_per_class: List[int],
        num_classes: int = 27,
        beta: float = 0.9999,
        gamma: float = 1.5,
        smoothing: float = 0.1
    ):
        super().__init__()
        effective_num = [1.0 - (beta ** n) for n in samples_per_class]
        weights = [(1.0 - beta) / max(1e-8, en) for en in effective_num]
        total_w = sum(weights)
        weights = [w / total_w * num_classes for w in weights]
        self.class_weights = torch.tensor(weights, dtype=torch.float32)
        self.gamma = gamma
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weights = self.class_weights.to(logits.device)
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - target_probs) ** self.gamma
        class_weight = weights.gather(0, targets)

        smooth_loss = -log_probs.mean(dim=-1)
        nll_loss = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        ce_loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss

        loss = focal_weight * class_weight * ce_loss
        return loss.mean()


def build_efficientnet_b0(num_classes: int = 27, dropout: float = 0.4) -> nn.Module:
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(512, num_classes)
    )
    return model


def get_data_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    train_transforms = transforms.Compose([
        transforms.Resize((RESIZE_DIM, RESIZE_DIM)),
        transforms.RandomResizedCrop(TARGET_RESOLUTION, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=25),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), value="random")
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((RESIZE_DIM, RESIZE_DIM)),
        transforms.CenterCrop(TARGET_RESOLUTION),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return train_transforms, val_transforms


def resolve_entry_metadata(entry: Any) -> Tuple[str, str]:
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
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Split manifest not found at: {MANIFEST_PATH}")
    if not CLASSES_PATH.exists():
        raise FileNotFoundError(f"Classes mapping not found at: {CLASSES_PATH}")

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        classes_dict = json.load(f)

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

    return train_records, val_records, test_records, classes_dict


def evaluate_model(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str) -> Dict:
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


def train_experiment_B(
    batch_size: int = 32,
    stage1_epochs: int = 5,
    stage2_epochs: int = 35,
    patience: int = 8,
    num_workers: int = 0,
    device: str = None
):
    start_total_time = time.time()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("=" * 80)
    logger.info("SMARTCROP AI - EXPERIMENT B: ADVANCED PIPELINE (TARGET: >=80% ACCURACY)")
    logger.info("=" * 80)
    logger.info(f"Target Device: {device} | Batch Size: {batch_size} | Resolution: {TARGET_RESOLUTION}x{TARGET_RESOLUTION}")

    train_records, val_records, test_records, classes_dict = load_dataset_splits()
    train_transforms, val_transforms = get_data_transforms()

    train_dataset = PlantDocDataset(train_records, transform=train_transforms)
    train_eval_dataset = PlantDocDataset(train_records, transform=val_transforms)
    val_dataset = PlantDocDataset(val_records, transform=val_transforms)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device == "cuda")
    )
    train_eval_loader = DataLoader(
        train_eval_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device == "cuda")
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device == "cuda")
    )

    num_classes = len(classes_dict)
    model = build_efficientnet_b0(num_classes=num_classes, dropout=0.4).to(device)

    # Class distribution for effective sample weighting
    train_labels = [r[1] for r in train_records]
    class_counts = [max(1, train_labels.count(c)) for c in range(num_classes)]

    criterion = ClassBalancedFocalLossWithSmoothing(
        samples_per_class=class_counts,
        num_classes=num_classes,
        beta=0.9999,
        gamma=1.5,
        smoothing=0.1
    )

    history = []
    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_epoch = -1
    best_checkpoint_path = MODELS_DIR / "disease_model_exp_B_best.pth"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # STAGE 1: Head Warmup (5 Epochs)
    # =========================================================================
    logger.info("-" * 80)
    logger.info(f"STAGE 1: Head Warmup ({stage1_epochs} epochs, lr=1e-3, Backbone Frozen)")
    logger.info("-" * 80)

    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    optimizer_stage1 = optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-2)

    for epoch in range(1, stage1_epochs + 1):
        t0 = time.time()
        model.train()
        model.features.eval()  # Freeze backbone BatchNorm statistics in eval mode!
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

        tr_loss = train_loss / len(train_dataset)
        aug_tr_acc = correct / len(train_dataset)
        det_tr_metrics = evaluate_model(model, train_eval_loader, criterion, device)
        det_tr_acc = det_tr_metrics["val_accuracy"]
        v_metrics = evaluate_model(model, val_loader, criterion, device)
        val_acc = v_metrics["val_accuracy"]
        val_f1 = v_metrics["val_macro_f1"]
        ep_time = time.time() - t0

        logger.info(
            f"Stage 1 [Epoch {epoch:02d}/{stage1_epochs:02d}] "
            f"Tr Loss: {tr_loss:.4f} | Loop Tr Acc (Aug): {aug_tr_acc*100:.2f}% | "
            f"Det Tr Acc: {det_tr_acc*100:.2f}% | "
            f"Val Loss: {v_metrics['val_loss']:.4f} | Val Acc: {val_acc*100:.2f}% | "
            f"Val F1: {val_f1*100:.2f}% | Time: {ep_time:.1f}s"
        )

        history.append({
            "stage": 1,
            "epoch": epoch,
            "global_epoch": epoch,
            "train_loss_aug": tr_loss,
            "train_accuracy_aug": aug_tr_acc,
            "train_accuracy_det": det_tr_acc,
            "val_loss": v_metrics["val_loss"],
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
            "time_s": ep_time
        })

        # Save best checkpoint: Validation Accuracy, with Macro-F1 tie-breaker
        is_better = False
        if val_acc > best_val_acc + 1e-5:
            is_better = True
        elif abs(val_acc - best_val_acc) <= 1e-5 and val_f1 > best_val_f1:
            is_better = True

        if is_better:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_accuracy": best_val_acc,
                "val_f1": best_val_f1,
                "det_train_accuracy": det_tr_acc,
                "experiment": "Experiment_B_Advanced",
                "resolution": TARGET_RESOLUTION,
                "num_classes": num_classes
            }, best_checkpoint_path)
            logger.info(f"  --> Saved new best checkpoint: Val Acc={best_val_acc*100:.2f}%, Val F1={best_val_f1*100:.2f}% (Epoch {epoch})")

    # =========================================================================
    # STAGE 2: Full Backbone Fine-Tuning (Up to 35 Epochs)
    # =========================================================================
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
        t0 = time.time()
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
        tr_loss = train_loss / len(train_dataset)
        aug_tr_acc = correct / len(train_dataset)
        det_tr_metrics = evaluate_model(model, train_eval_loader, criterion, device)
        det_tr_acc = det_tr_metrics["val_accuracy"]
        v_metrics = evaluate_model(model, val_loader, criterion, device)
        val_acc = v_metrics["val_accuracy"]
        val_f1 = v_metrics["val_macro_f1"]
        ep_time = time.time() - t0

        logger.info(
            f"Stage 2 [Epoch {epoch:02d}/{stage2_epochs:02d} (Global {global_epoch:02d})] "
            f"Tr Loss: {tr_loss:.4f} | Loop Tr Acc (Aug): {aug_tr_acc*100:.2f}% | "
            f"Det Tr Acc: {det_tr_acc*100:.2f}% | "
            f"Val Loss: {v_metrics['val_loss']:.4f} | Val Acc: {val_acc*100:.2f}% | "
            f"Val F1: {val_f1*100:.2f}% | Time: {ep_time:.1f}s"
        )

        history.append({
            "stage": 2,
            "epoch": epoch,
            "global_epoch": global_epoch,
            "train_loss_aug": tr_loss,
            "train_accuracy_aug": aug_tr_acc,
            "train_accuracy_det": det_tr_acc,
            "val_loss": v_metrics["val_loss"],
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
            "time_s": ep_time
        })

        # Save best checkpoint: Validation Accuracy, with Macro-F1 tie-breaker
        is_better = False
        if val_acc > best_val_acc + 1e-5:
            is_better = True
        elif abs(val_acc - best_val_acc) <= 1e-5 and val_f1 > best_val_f1:
            is_better = True

        if is_better:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            best_epoch = global_epoch
            epochs_without_improvement = 0
            torch.save({
                "epoch": global_epoch,
                "model_state_dict": model.state_dict(),
                "val_accuracy": best_val_acc,
                "val_f1": best_val_f1,
                "det_train_accuracy": det_tr_acc,
                "experiment": "Experiment_B_Advanced",
                "resolution": TARGET_RESOLUTION,
                "num_classes": num_classes
            }, best_checkpoint_path)
            logger.info(f"  --> Saved new best checkpoint: Val Acc={best_val_acc*100:.2f}%, Val F1={best_val_f1*100:.2f}% (Global Epoch {global_epoch})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without Validation Accuracy improvement.")
                break

    total_training_time_s = time.time() - start_total_time
    logger.info("=" * 80)
    logger.info(f"EXPERIMENT B FINISHED. Best Global Epoch: {best_epoch} | Best Val Acc: {best_val_acc*100:.2f}% | Val F1: {best_val_f1*100:.2f}%")
    logger.info("=" * 80)

    # Load best model for validation and deterministic train scorecard from SAME checkpoint
    best_ckpt = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    final_val_metrics = evaluate_model(model, val_loader, criterion, device)
    final_det_train_metrics = evaluate_model(model, train_eval_loader, criterion, device)

    # Determine empirical target achievement on the SAME checkpoint
    final_det_train_acc = final_det_train_metrics["val_accuracy"]
    train_target_met = (final_det_train_acc >= 0.80)
    val_target_met = (final_val_metrics["val_accuracy"] >= 0.80)
    overall_target_reached = (train_target_met and val_target_met)

    target_status = "REACHED" if overall_target_reached else "NOT_REACHED"

    # Plot training and validation curves
    epochs_range = [h["global_epoch"] for h in history]
    tr_losses = [h["train_loss_aug"] for h in history]
    va_losses = [h["val_loss"] for h in history]
    tr_accs_det = [h["train_accuracy_det"] * 100 for h in history]
    tr_accs_aug = [h["train_accuracy_aug"] * 100 for h in history]
    va_accs = [h["val_accuracy"] * 100 for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].plot(epochs_range, tr_losses, "b-o", label="Training Loss (Aug)")
    axes[0].plot(epochs_range, va_losses, "r-s", label="Validation Loss")
    axes[0].set_title("Experiment B: Loss Curves across Epochs")
    axes[0].set_xlabel("Global Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].plot(epochs_range, tr_accs_aug, "c--", label="Train Acc (Augmented Loop)")
    axes[1].plot(epochs_range, tr_accs_det, "b-o", label="Train Acc (Deterministic Eval)")
    axes[1].plot(epochs_range, va_accs, "r-s", label="Validation Accuracy (%)")
    axes[1].axhline(y=80.0, color="g", linestyle="--", label="Target: 80% Threshold")
    axes[1].set_title("Experiment B: Accuracy Curves across Epochs")
    axes[1].set_xlabel("Global Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    curves_path = OUTPUTS_DIR / "plantdoc_experiment_B_curves.png"
    plt.tight_layout()
    plt.savefig(curves_path, dpi=300)
    plt.close()
    logger.info(f"Saved epoch curves plot to: {curves_path}")

    # Generate per-class classification report
    target_names = [classes_dict[str(i)] for i in range(num_classes)]
    val_cls_report = classification_report(
        final_val_metrics["targets"], final_val_metrics["predictions"],
        target_names=target_names, output_dict=True, zero_division=0
    )
    val_cm = confusion_matrix(final_val_metrics["targets"], final_val_metrics["predictions"]).tolist()

    # Metrics export JSON
    metrics_export = {
        "experiment_name": "Experiment_B_Advanced_Regularization_and_Loss",
        "model_architecture": "EfficientNet-B0",
        "input_resolution": [TARGET_RESOLUTION, TARGET_RESOLUTION],
        "batch_size": batch_size,
        "total_training_time_minutes": total_training_time_s / 60.0,
        "best_epoch": best_epoch,
        "primary_selection_metric": "validation_accuracy",
        "tie_breaker_metric": "validation_macro_f1",
        "final_targets": {
            "deterministic_train_accuracy_target": 0.80,
            "validation_accuracy_target": 0.80,
            "measured_checkpoint_deterministic_train_acc": final_det_train_acc,
            "measured_checkpoint_val_acc": final_val_metrics["val_accuracy"],
            "train_target_met": train_target_met,
            "val_target_met": val_target_met,
            "overall_status": target_status
        },
        "validation_metrics": {
            "val_loss": final_val_metrics["val_loss"],
            "val_accuracy": final_val_metrics["val_accuracy"],
            "val_macro_f1": final_val_metrics["val_macro_f1"],
            "val_macro_precision": final_val_metrics["val_macro_precision"],
            "val_macro_recall": final_val_metrics["val_macro_recall"],
            "val_weighted_f1": final_val_metrics["val_weighted_f1"]
        },
        "comparison_with_production_224": {
            "production_val_acc": 0.6919,
            "production_val_macro_f1": 0.6837,
            "new_val_acc": final_val_metrics["val_accuracy"],
            "new_val_macro_f1": final_val_metrics["val_macro_f1"],
            "val_acc_delta": final_val_metrics["val_accuracy"] - 0.6919,
            "val_f1_delta": final_val_metrics["val_macro_f1"] - 0.6837
        },
        "per_class_report": val_cls_report,
        "confusion_matrix": val_cm,
        "training_history": history
    }

    metrics_path = OUTPUTS_DIR / "plantdoc_experiment_B_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_export, f, indent=2)
    logger.info(f"Saved Experiment B metrics JSON to: {metrics_path}")

    # Generate Markdown report
    report_md = f"""# SmartCrop AI — PlantDoc Experiment B Evaluation Report

**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Experiment:** Experiment B (High-Res 288x288, Class-Balanced Focal Loss, Label Smoothing, Random Erasing)  
**Primary Selection Metric:** **Validation Accuracy**  
**Best Checkpoint:** `{best_checkpoint_path}` (Global Epoch {best_epoch})  
**Target Status:** **{target_status}**  

---

## 1. Performance Target Audit

| Metric | Target Threshold | Measured Result | Target Met? |
| :--- | :---: | :---: | :---: |
| **Deterministic Training Accuracy (Best Checkpoint)** | &ge; 80.00% | **{final_det_train_acc*100:.2f}%** | {'✅ YES' if train_target_met else '❌ NO'} |
| **Validation Accuracy (Best Checkpoint)** | &ge; 80.00% | **{final_val_metrics['val_accuracy']*100:.2f}%** | {'✅ YES' if val_target_met else '❌ NO'} |
| **Validation Macro-F1** | N/A (Logged) | **{final_val_metrics['val_macro_f1']*100:.2f}%** | Logged |

> [!NOTE]
> All metrics are calculated directly from empirical model predictions on the official splits. No test samples were used for hyperparameter tuning.

---

## 2. Comparison with Production Baseline (224x224)

| Metric | Production Baseline (224x224, 18 Epochs) | Experiment B (288x288, CB-Focal, 40 Epochs) | Delta |
| :--- | :---: | :---: | :---: |
| **Input Resolution** | $224 \times 224$ | **$288 \times 288$** | $+64 \text{ px}$ |
| **Loss Function** | Weighted Cross-Entropy | **CB-Focal Loss (gamma=1.5, beta=0.9999, eps=0.1)** | Focal + Smooth |
| **Augmentation** | Basic Crop/Flip/Jitter | **+ Random Erasing ($p=0.25$)** | Regularized |
| **Validation Accuracy** | 69.19% | **{final_val_metrics['val_accuracy']*100:.2f}%** | **{'+' if (final_val_metrics['val_accuracy'] - 0.6919) >= 0 else ''}{(final_val_metrics['val_accuracy'] - 0.6919)*100:.2f}%** |
| **Validation Macro-F1** | 68.37% | **{final_val_metrics['val_macro_f1']*100:.2f}%** | **{'+' if (final_val_metrics['val_macro_f1'] - 0.6837) >= 0 else ''}{(final_val_metrics['val_macro_f1'] - 0.6837)*100:.2f}%** |
| **Primary Selection** | Val Macro-F1 | **Val Accuracy** | User Target Aligned |
"""

    report_path = OUTPUTS_DIR / "plantdoc_experiment_B_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    logger.info(f"Saved Experiment B Markdown report to: {report_path}")

    return metrics_export


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PlantDoc Experiment B: Advanced Pipeline")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--stage1_epochs", type=int, default=5, help="Stage 1 epochs")
    parser.add_argument("--stage2_epochs", type=int, default=35, help="Stage 2 epochs")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    args = parser.parse_args()

    train_experiment_B(
        batch_size=args.batch_size,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        patience=args.patience,
        num_workers=args.num_workers
    )

