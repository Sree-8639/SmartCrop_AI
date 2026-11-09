"""
SmartCrop AI - IP102 Insect Pest Classification Training Pipeline
Comparative Benchmark: Pretrained ResNet-50 Baseline vs. EfficientNet-B0 Proposed Architecture.

Base Research Paper Reference:
Pal, A. K., Patro, B. D. K., & Chaube, S. (2026).
Design and implementation of a deep learning framework for automated crop classification
and health diagnosis in precision agriculture. Scientific Reports, 16, 42151.

IP102 Benchmark Reference:
Wu, X., Zhan, C., Lai, Y. K., Cheng, M. M., & Yang, J. (2019).
IP102: A Large-Scale Benchmark Dataset for Insect Pest Recognition.
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 8787-8796.

Critical Methodological Standards:
- Requires authentic IP102 classification images and official split lists (train.txt, val.txt, test.txt).
  No proxy fallbacks to VOC detection images or XML objects.
- Preserves official train/val/test splits without silent replacement.
- MD5 hash deduplication removes identical images across splits; conflicting labels are strictly quarantined.
- No black-image fallbacks for corrupted images; corruptions are detected and excluded upfront.
- Class weights are calculated STRICTLY from the training split to combat long-tail imbalance.
- Model selection and early stopping rely strictly on validation metrics; test split is strictly held out.
- Automatic output directory routing (/kaggle/working/IP102_Results/ on Kaggle or outputs/ip102/ locally).
"""

import os
import sys
import json
import time
import math
import hashlib
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T
from torchvision.models import (
    resnet50, ResNet50_Weights,
    efficientnet_b0, EfficientNet_B0_Weights
)

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("SmartCrop.IP102.Classification")

# Constants
SEED = 42
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
NUM_IP102_CLASSES = 102


def set_seed(seed: int = SEED) -> None:
    """Sets deterministic seed across all pseudo-random generators."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_md5(file_path: Path, chunk_size: int = 65536) -> str:
    """Computes MD5 hash of a file for exact duplicate detection."""
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def get_default_output_dir() -> Path:
    """Routes output directory to Kaggle working storage or local repository outputs."""
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working/IP102_Results/classification")
    return PROJECT_ROOT / "outputs" / "ip102" / "classification"


def load_ip102_classes(classes_file: Path) -> List[str]:
    """
    Loads IP102 class names from classes.txt (format: '<id>  <class_name>').
    Returns a list of 102 class names ordered by 0-indexed class index 0..101.
    """
    class_names = [f"Pest_Class_{i}" for i in range(NUM_IP102_CLASSES)]
    if not classes_file.exists():
        logger.warning(f"classes.txt not found at {classes_file}. Using index naming.")
        return class_names

    try:
        with open(classes_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) >= 2 and parts[0].isdigit():
                    idx = int(parts[0]) - 1  # 1-indexed to 0-indexed
                    if 0 <= idx < NUM_IP102_CLASSES:
                        class_names[idx] = parts[1].strip()
    except Exception as e:
        logger.error(f"Failed to parse classes.txt at {classes_file}: {e}")
    return class_names


# -----------------------------------------------------------------------------
# Dataset Discovery & Strict Authentic Loading
# -----------------------------------------------------------------------------

def discover_ip102_classification_data(
    search_dirs: List[Path]
) -> Tuple[Optional[Path], str, Dict[str, Any]]:
    """
    Discovers authentic IP102 classification dataset across local or Kaggle paths.
    Supports:
      1. Class subfolders split (train, val, test subdirs with 102 class folders each)
      2. Official pre-split text files (train.txt, val.txt, test.txt) with images directory
      3. Flat 102 class subfolders
    """
    # Expand search dirs to include /kaggle/input and subdirs
    expanded_search: List[Path] = []
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        expanded_search.append(kaggle_input)
        for item in kaggle_input.iterdir():
            if item.is_dir():
                expanded_search.append(item)
    for p in search_dirs:
        if p.exists() and p not in expanded_search:
            expanded_search.append(p)

    for base_dir in expanded_search:
        candidates = [base_dir]
        try:
            for sub in base_dir.rglob("*"):
                if sub.is_dir():
                    s_low = sub.name.lower()
                    if s_low in ["classification", "ip102", "ip102-dataset", "ip02-dataset", "ip102_dataset"] or "ip102" in s_low or "ip02" in s_low:
                        candidates.append(sub)
        except Exception:
            pass

        for cand in candidates:
            if not cand.is_dir():
                continue

            # Strategy 1: Class Subfolders Split (train, val, test subdirs)
            children = {c.name.lower(): c for c in cand.iterdir() if c.is_dir()}
            if "train" in children and ("test" in children or "val" in children):
                train_dir = children["train"]
                test_dir = children.get("test")
                val_dir = children.get("val") or children.get("validation")
                train_subs = [d for d in train_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
                if len(train_subs) >= NUM_IP102_CLASSES:
                    return cand, "CLASS_SUBFOLDERS_SPLIT", {
                        "train_dir": train_dir,
                        "val_dir": val_dir,
                        "test_dir": test_dir,
                        "root": cand
                    }

            # Strategy 2: Official pre-split text files
            train_f = cand / "train.txt"
            test_f = cand / "test.txt"
            val_f = cand / "val.txt"
            if train_f.exists() and test_f.exists():
                images_dir = cand / "images"
                if not images_dir.exists():
                    images_dir = cand
                return cand, "PRE_SPLIT_TXT", {
                    "train_txt": train_f,
                    "val_txt": val_f if val_f.exists() else None,
                    "test_txt": test_f,
                    "images_dir": images_dir
                }

    return None, "NOT_FOUND", {}


def parse_official_txt(txt_path: Path, images_dir: Path) -> List[Tuple[str, int]]:
    """
    Parses official IP102 split file where each line is: '<image_rel_path> <label_idx>'.
    Validates file existence and label bounds (0..101).
    """
    samples = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                rel_path = parts[0]
                try:
                    label = int(parts[1])
                except ValueError:
                    continue

                # Ensure 0-indexed bounds [0, 101]
                if not (0 <= label < NUM_IP102_CLASSES):
                    logger.warning(f"Label {label} out of bounds in {txt_path} for {rel_path}. Skipping.")
                    continue

                full_path = images_dir / rel_path
                if not full_path.exists():
                    full_path = images_dir.parent / rel_path

                if full_path.exists():
                    samples.append((str(full_path), label))
    return samples


def validate_image_readable(image_path: str) -> bool:
    """Opens image and verifies that file is non-empty and uncorrupted."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        # Verify can actually convert to RGB
        with Image.open(image_path) as img:
            _ = img.convert("RGB")
        return True
    except Exception:
        return False


def deduplicate_and_audit_samples(
    train_raw: List[Tuple[str, int]],
    val_raw: List[Tuple[str, int]],
    test_raw: List[Tuple[str, int]]
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]], Dict[str, Any]]:
    """
    Performs full MD5 hash deduplication across train, val, and test partitions:
      - Validates every image is uncorrupted.
      - Discards identical image duplicates that appear in multiple splits (retaining train instance).
      - If identical image has conflicting class labels, ALL copies are quarantined and excluded.
      - Verifies strict disjointness across all three partitions.
    """
    logger.info("Performing image validation and MD5 cross-split deduplication...")

    hash_to_split: Dict[str, str] = {}
    hash_to_label: Dict[str, int] = {}
    hash_to_paths: Dict[str, List[str]] = {}

    corrupted_files: List[str] = []
    conflicting_hashes: Set[str] = set()
    identical_duplicates: List[Dict[str, Any]] = []

    def scan_partition(partition_name: str, samples: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        valid_items = []
        for path_str, label in samples:
            if not validate_image_readable(path_str):
                corrupted_files.append(path_str)
                continue

            h = compute_md5(Path(path_str))
            if h in hash_to_label:
                prev_label = hash_to_label[h]
                if prev_label != label:
                    conflicting_hashes.add(h)
                    hash_to_paths[h].append(path_str)
                    continue
                else:
                    # Duplicate with identical label
                    identical_duplicates.append({
                        "file": path_str,
                        "duplicate_of": hash_to_paths[h][0],
                        "split": partition_name,
                        "original_split": hash_to_split[h]
                    })
                    continue

            hash_to_split[h] = partition_name
            hash_to_label[h] = label
            hash_to_paths[h] = [path_str]
            valid_items.append((path_str, label, h))
        return valid_items

    clean_train_with_hash = scan_partition("train", train_raw)
    clean_val_with_hash = scan_partition("val", val_raw)
    clean_test_with_hash = scan_partition("test", test_raw)

    # Exclude any hash that had conflicting labels
    train_clean = [(p, l) for p, l, h in clean_train_with_hash if h not in conflicting_hashes]
    val_clean = [(p, l) for p, l, h in clean_val_with_hash if h not in conflicting_hashes]
    test_clean = [(p, l) for p, l, h in clean_test_with_hash if h not in conflicting_hashes]

    # Disjointness checks
    tr_paths = {p for p, _ in train_clean}
    va_paths = {p for p, _ in val_clean}
    te_paths = {p for p, _ in test_clean}

    assert len(tr_paths & va_paths) == 0, "Train and Val splits share overlapping paths!"
    assert len(tr_paths & te_paths) == 0, "Train and Test splits share overlapping paths!"
    assert len(va_paths & te_paths) == 0, "Val and Test splits share overlapping paths!"

    audit_report = {
        "corrupted_images_excluded": len(corrupted_files),
        "corrupted_files_sample": corrupted_files[:5],
        "identical_duplicates_removed": len(identical_duplicates),
        "conflicting_label_hashes_excluded": len(conflicting_hashes),
        "conflicting_images_excluded_count": sum(len(hash_to_paths[h]) for h in conflicting_hashes),
        "split_counts": {
            "train": len(train_clean),
            "val": len(val_clean),
            "test": len(test_clean),
            "total_clean": len(train_clean) + len(val_clean) + len(test_clean)
        },
        "disjointness_verified": True
    }

    logger.info(
        f"Deduplication Complete -> Train: {len(train_clean)}, Val: {len(val_clean)}, Test: {len(test_clean)} "
        f"(Excluded: {len(corrupted_files)} corrupt, {len(identical_duplicates)} duplicates, {len(conflicting_hashes)} conflicting)"
    )
    return train_clean, val_clean, test_clean, audit_report


# -----------------------------------------------------------------------------
# PyTorch Dataset Definition
# -----------------------------------------------------------------------------

class IP102ClassificationDataset(Dataset):
    """PyTorch Dataset for IP102 image classification with explicit error handling."""

    def __init__(self, samples: List[Tuple[str, int]], transform: Optional[T.Compose] = None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Unreadable image encountered at runtime for index {idx} ({img_path}): {e}")

        if self.transform is not None:
            image = self.transform(image)
        return image, label


def get_classification_transforms(img_size: int = 224) -> Tuple[T.Compose, T.Compose]:
    """Returns training and validation/test torchvision data transforms."""
    train_transform = T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.RandomRotation(degrees=15),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transform = T.Compose([
        T.Resize(int(img_size * 1.14)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, eval_transform


# -----------------------------------------------------------------------------
# Model Architectures
# -----------------------------------------------------------------------------

def build_resnet50_baseline(num_classes: int = NUM_IP102_CLASSES, pretrained: bool = True) -> nn.Module:
    """Builds ResNet-50 baseline model with custom classification head."""
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    return model


def build_efficientnet_b0_proposed(num_classes: int = NUM_IP102_CLASSES, pretrained: bool = True) -> nn.Module:
    """Builds EfficientNet-B0 proposed model with custom classification head."""
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    return model


# -----------------------------------------------------------------------------
# Loss & Imbalance Handling (Calculated ONLY from Train Split)
# -----------------------------------------------------------------------------

def compute_class_weights_from_train_split(
    train_samples: List[Tuple[str, int]],
    num_classes: int = NUM_IP102_CLASSES,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Computes smoothed inverse-frequency class weights ONLY from the training partition:
      w_c = clamp(N_train / (C * count_c), 0.1, 10.0)
    Ensures zero information leakage from validation or test splits.
    """
    counts = np.zeros(num_classes, dtype=np.float32)
    for _, lbl in train_samples:
        if 0 <= lbl < num_classes:
            counts[lbl] += 1.0

    total_train = max(1, len(train_samples))
    weights = np.zeros(num_classes, dtype=np.float32)

    for c in range(num_classes):
        if counts[c] > 0:
            w = total_train / (num_classes * counts[c])
            weights[c] = float(np.clip(w, 0.1, 10.0))
        else:
            weights[c] = 1.0

    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = NUM_IP102_CLASSES
) -> Dict[str, Any]:
    """Computes Top-1 Accuracy, Macro Precision, Macro Recall, Macro F1, and per-class stats."""
    acc = float(np.mean(y_true == y_pred)) if len(y_true) > 0 else 0.0

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    precs, recs, f1s = [], [], []
    per_class = {}

    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float((2 * prec * rec) / (prec + rec)) if (prec + rec) > 0 else 0.0

        support = int(tp + fn)
        if support > 0:
            precs.append(prec)
            recs.append(rec)
            f1s.append(f1)
            per_class[c] = {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "support": support
            }

    macro_prec = float(np.mean(precs)) if precs else 0.0
    macro_rec = float(np.mean(recs)) if recs else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "evaluated_classes_count": len(precs),
        "per_class": per_class,
        "confusion_matrix": cm.tolist()
    }


# -----------------------------------------------------------------------------
# Training & Validation Engine
# -----------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device
) -> Tuple[float, float]:
    """Executes one epoch of training with optional AMP."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    use_amp = (scaler is not None and device.type == "cuda")

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / max(1, total)
    avg_acc = correct / max(1, total)
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = NUM_IP102_CLASSES
) -> Tuple[float, Dict[str, Any]]:
    """Evaluates model performance returning loss and comprehensive metrics."""
    model.eval()
    total_loss, total = 0.0, 0
    all_preds, all_targets = [], []

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(labels.cpu().numpy().tolist())
        total += labels.size(0)

    avg_loss = total_loss / max(1, total)
    metrics = compute_multiclass_metrics(np.array(all_targets), np.array(all_preds), num_classes=num_classes)
    return avg_loss, metrics


def fit_model(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epochs: int,
    device: torch.device,
    output_dir: Path,
    patience: int = 5
) -> Dict[str, Any]:
    """
    Fits model with early stopping on validation macro F1.
    Strictly restores the best validation checkpoint before evaluating on the held-out test set.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = output_dir / f"best_{model_name}_ip102.pth"

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_macro_f1": []
    }

    best_val_f1 = -1.0
    best_epoch = -1
    no_improve_epochs = 0

    logger.info(f"--- Commencing Training: {model_name} ({epochs} epochs, Device: {device}) ---")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_metrics = evaluate_model(model, val_loader, criterion, device)
        val_acc = val_metrics["accuracy"]
        val_f1 = val_metrics["macro_f1"]

        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(round(tr_loss, 4))
        history["train_acc"].append(round(tr_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))
        history["val_macro_f1"].append(round(val_f1, 4))

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] - "
            f"Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Macro-F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            no_improve_epochs = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "model_name": model_name
            }, best_ckpt_path)
            logger.info(f" -> Checkpoint saved (Best Val Macro-F1: {val_f1:.4f})")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs with no improvement.")
                break

    # Restore best validation checkpoint for final test evaluation
    if best_ckpt_path.exists():
        logger.info(f"Restoring best checkpoint from epoch {best_epoch} for test evaluation...")
        checkpoint = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        logger.warning("No checkpoint found; evaluating test set with current weights.")

    # Strict evaluation on held-out test set
    test_loss, test_metrics = evaluate_model(model, test_loader, criterion, device)
    logger.info(
        f"=== Test Set Evaluation: {model_name} ===\n"
        f"Accuracy: {test_metrics['accuracy']:.4f} | "
        f"Macro Precision: {test_metrics['macro_precision']:.4f} | "
        f"Macro Recall: {test_metrics['macro_recall']:.4f} | "
        f"Macro F1: {test_metrics['macro_f1']:.4f}"
    )

    return {
        "model_name": model_name,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "test_loss": round(test_loss, 4),
        "test_metrics": test_metrics,
        "history": history,
        "checkpoint_path": str(best_ckpt_path)
    }


# -----------------------------------------------------------------------------
# Main Execution Pipeline
# -----------------------------------------------------------------------------

def run_ip102_classification_pipeline(
    data_dir: Path,
    output_dir: Optional[Path] = None,
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    models_to_train: List[str] = ["resnet50", "efficientnet_b0"],
    dry_run: bool = False
) -> Dict[str, Any]:
    """Runs complete end-to-end IP102 classification training and benchmark."""
    set_seed(SEED)
    out_dir = output_dir or get_default_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Initializing IP102 Classification Pipeline (Device: {device})")
    classes_file = data_dir / "classes.txt"
    class_names = load_ip102_classes(classes_file)

    # 1. Dataset Discovery
    search_dirs = [
        data_dir,
        Path("/kaggle/input/ip102"),
        Path("/kaggle/input/ip102-dataset"),
        Path("/kaggle/input/ip102-v1-1"),
        PROJECT_ROOT / "datasets" / "ip102"
    ]
    resolved_root, format_type, meta = discover_ip102_classification_data(search_dirs)

    if resolved_root is None or format_type == "NOT_FOUND":
        msg = (
            "DATASET_NOT_READY: IP102 classification images (75,222 images) and official split lists "
            "(train.txt, val.txt, test.txt) were not found in any search path. "
            "Please ensure IP102 v1.1 classification archive is downloaded and extracted."
        )
        logger.error(msg)
        status_report = {
            "status": "DATASET_NOT_READY",
            "searched_paths": [str(p) for p in search_dirs],
            "message": msg,
            "classes_detected": len(class_names),
            "training_executed": False
        }
        with open(out_dir / "classification_status.json", "w", encoding="utf-8") as f:
            json.dump(status_report, f, indent=2)
        return status_report

    logger.info(f"Resolved IP102 classification dataset at {resolved_root} (Format: {format_type})")

    # 2. Parse Splits
    if format_type == "PRE_SPLIT_TXT":
        train_raw = parse_official_txt(meta["train_txt"], meta["images_dir"])
        val_raw = parse_official_txt(meta["val_txt"], meta["images_dir"]) if meta.get("val_txt") else []
        test_raw = parse_official_txt(meta["test_txt"], meta["images_dir"])

        # If val.txt is missing from official archive, partition deterministically from train
        if not val_raw and len(train_raw) > 20:
            rng = np.random.RandomState(SEED)
            perm = rng.permutation(len(train_raw))
            val_sz = int(0.15 * len(train_raw))
            val_indices = set(perm[:val_sz])
            new_train = [train_raw[i] for i in range(len(train_raw)) if i not in val_indices]
            val_raw = [train_raw[i] for i in range(len(train_raw)) if i in val_indices]
            train_raw = new_train
            logger.info(f"Generated deterministic validation partition from train.txt: {len(val_raw)} items.")
    elif format_type == "CLASS_SUBFOLDERS_SPLIT":
        train_dir = meta["train_dir"]
        val_dir = meta.get("val_dir")
        test_dir = meta.get("test_dir")

        train_class_dirs = [d for d in train_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        all_numeric = all(d.name.isdigit() for d in train_class_dirs)
        folder_to_idx: Dict[str, int] = {}

        if all_numeric:
            int_ids = [int(d.name) for d in train_class_dirs]
            min_fid, max_fid = min(int_ids), max(int_ids)
            if min_fid == 0 and max_fid == NUM_IP102_CLASSES - 1:
                for d in train_class_dirs:
                    folder_to_idx[d.name] = int(d.name)
            elif min_fid == 1 and max_fid == NUM_IP102_CLASSES:
                for d in train_class_dirs:
                    folder_to_idx[d.name] = int(d.name) - 1
            else:
                sorted_dirs = sorted(train_class_dirs, key=lambda d: int(d.name))
                for i, d in enumerate(sorted_dirs):
                    folder_to_idx[d.name] = i
        else:
            sorted_dirs = sorted(train_class_dirs, key=lambda d: d.name)
            for i, d in enumerate(sorted_dirs):
                folder_to_idx[d.name] = i

        def scan_split_subfolders(s_dir: Optional[Path]) -> List[Tuple[str, int]]:
            s_samples = []
            if not s_dir or not s_dir.exists():
                return s_samples
            for f_name, c_idx in folder_to_idx.items():
                c_dir = s_dir / f_name
                if c_dir.exists() and c_dir.is_dir():
                    for img_f in c_dir.iterdir():
                        if img_f.is_file() and img_f.suffix in VALID_IMAGE_EXTENSIONS:
                            s_samples.append((str(img_f.resolve()), c_idx))
            return s_samples

        train_raw = scan_split_subfolders(train_dir)
        val_raw = scan_split_subfolders(val_dir)
        test_raw = scan_split_subfolders(test_dir)
        logger.info(f"Loaded from class subfolders -> Train: {len(train_raw)}, Val: {len(val_raw)}, Test: {len(test_raw)}")
    else:
        raise NotImplementedError(f"Unsupported dataset format: {format_type}")

    # 3. Deduplication and Leakage Prevention Audit
    train_samples, val_samples, test_samples, audit_info = deduplicate_and_audit_samples(
        train_raw, val_raw, test_raw
    )

    with open(out_dir / "deduplication_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit_info, f, indent=2)

    # 4. Check for Empty Splits
    if len(train_samples) == 0:
        raise RuntimeError("Training split is empty after validation. Cannot proceed.")
    if len(val_samples) == 0:
        raise RuntimeError("Validation split is empty after validation. Cannot proceed.")
    if len(test_samples) == 0:
        raise RuntimeError("Held-out test split is empty after validation. Cannot proceed.")

    if dry_run:
        logger.info("Dry-run mode activated: Subsetting to 2 batches for syntax/pipeline verification.")
        train_samples = train_samples[:batch_size * 2]
        val_samples = val_samples[:batch_size]
        test_samples = test_samples[:batch_size]
        epochs = 1

    # 5. DataLoaders
    train_tf, eval_tf = get_classification_transforms(img_size=224)
    train_dataset = IP102ClassificationDataset(train_samples, transform=train_tf)
    val_dataset = IP102ClassificationDataset(val_samples, transform=eval_tf)
    test_dataset = IP102ClassificationDataset(test_samples, transform=eval_tf)

    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # 6. Compute Class Weights ONLY from Train Split
    class_weights = compute_class_weights_from_train_split(train_samples, num_classes=NUM_IP102_CLASSES, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    env_meta = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "seed": SEED,
        "split_counts": audit_info["split_counts"],
        "class_weights_min": float(class_weights.min().item()),
        "class_weights_max": float(class_weights.max().item())
    }

    benchmark_results = {
        "status": "COMPLETED",
        "dataset_format": format_type,
        "environment": env_meta,
        "models": {}
    }

    # 7. Model Training Execution
    if "resnet50" in models_to_train:
        logger.info("Initializing Baseline: ResNet-50")
        model_a = build_resnet50_baseline(num_classes=NUM_IP102_CLASSES, pretrained=True).to(device)
        optimizer_a = torch.optim.AdamW(model_a.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler_a = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_a, T_max=epochs, eta_min=1e-6)
        res_a = fit_model("resnet50", model_a, train_loader, val_loader, test_loader, criterion, optimizer_a, scheduler_a, epochs, device, out_dir)
        benchmark_results["models"]["resnet50"] = res_a

    if "efficientnet_b0" in models_to_train:
        logger.info("Initializing Proposed: EfficientNet-B0")
        model_b = build_efficientnet_b0_proposed(num_classes=NUM_IP102_CLASSES, pretrained=True).to(device)
        optimizer_b = torch.optim.AdamW(model_b.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler_b = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_b, T_max=epochs, eta_min=1e-6)
        res_b = fit_model("efficientnet_b0", model_b, train_loader, val_loader, test_loader, criterion, optimizer_b, scheduler_b, epochs, device, out_dir)
        benchmark_results["models"]["efficientnet_b0"] = res_b

    # 8. Save Benchmark Report
    results_path = out_dir / "classification_benchmark_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)
    logger.info(f"IP102 classification benchmark report saved to {results_path}")

    return benchmark_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IP102 Insect Pest Classification Training")
    parser.add_argument("--data-dir", type=str, default=str(PROJECT_ROOT / "datasets" / "ip102"), help="Path to IP102 dataset")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (defaults to Kaggle working or local outputs)")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--model", type=str, choices=["resnet50", "efficientnet_b0", "both"], default="both", help="Model to train")
    parser.add_argument("--dry-run", action="store_true", help="Execute 1 batch dry run for pipeline verification")
    args = parser.parse_args()

    models = ["resnet50", "efficientnet_b0"] if args.model == "both" else [args.model]
    run_ip102_classification_pipeline(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        models_to_train=models,
        dry_run=args.dry_run
    )
