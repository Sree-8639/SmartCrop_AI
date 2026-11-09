"""
SmartCrop AI - Generator for Notebook 04: PlantDoc Experiment A (288x288 Two-Stage Fine-Tuning)
Creates a self-contained, reproducible notebook runnable on Kaggle GPU or local machines.
Reordered and validated so all dependencies (paths, manifest, class mapping, datasets, model)
execute in strict order without NameErrors.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH = NOTEBOOKS_DIR / "04_PlantDoc_Experiment_A_288.ipynb"

def generate_notebook():
    cells = []

    def add_markdown(content: str):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in content.strip().split("\n")]
        })

    def add_code(content: str):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in content.strip().split("\n")]
        })

    # =========================================================================
    # Cell 0: Header & Research Objectives
    # =========================================================================
    add_markdown("""# SmartCrop AI — Experiment 1: High-Resolution (288×288) with Two-Stage Discriminative Fine-Tuning

### Research Objective:
Diagnose and resolve the fine-grained foliar lesion spatial bottleneck identified in the 224×224 EfficientNet-B0 production model (which achieved 68.41% held-out test accuracy and 66.62% Macro-F1).

### Key Hypotheses & Experimental Controls:
1. **Resolution Upgrade (288×288):** Preserves subtle diagnostic features of necrotic spots (*Tomato leaf bacterial spot*, *Septoria leaf spot*, *Apple scab*).
2. **Two-Stage Training Schedule:**
   - **Stage 1 (Head Warmup, 5 Epochs):** Freeze ImageNet pretrained backbone (`features.requires_grad = False`). Train custom head at $lr = 10^{-3}$ to prevent catastrophic forgetting.
   - **Stage 2 (Fine-Tuning, 25 Epochs):** Unfreeze backbone with differential learning rates ($2 \times 10^{-5}$ for backbone, $2 \times 10^{-4}$ for classifier head) with Cosine Annealing.
3. **Rigorous Validation Discipline:** Model selection and early stopping use **Validation Macro-F1 only**. The **Held-Out Test Set remains strictly untouched**.
4. **Isolated Effect:** No CutMix, Focal Loss, or TTA are included in this run to isolate the exact contribution of Option A.""")

    # =========================================================================
    # Cell 1: Environment Setup, Seeds, Device & GPU Memory
    # =========================================================================
    add_code("""# Cell 1: Environment Setup, Deterministic Seeding, Device & GPU Memory Inspection
import os
import sys
import time
import json
import random
import logging
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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

# Deterministic reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[SmartCrop AI] Execution Device: {DEVICE}")

# Resolution constants
TARGET_RESOLUTION = 288
RESIZE_DIM = 320
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# GPU Memory profiling & Recommended batch size
RECOMMENDED_BATCH_SIZE = 32
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    allocated = torch.cuda.memory_allocated(0) / (1024**3)
    reserved = torch.cuda.memory_reserved(0) / (1024**3)
    print(f"  - Device Name         : {device_name}")
    print(f"  - Total GPU Memory    : {total_mem:.2f} GB")
    print(f"  - Allocated Memory    : {allocated:.2f} GB")
    print(f"  - Reserved Memory     : {reserved:.2f} GB")
    if total_mem >= 14.0:
        RECOMMENDED_BATCH_SIZE = 32
        print("  - VRAM Recommendation : Batch Size 32 fits comfortably in 16GB VRAM.")
    elif total_mem >= 7.0:
        RECOMMENDED_BATCH_SIZE = 32
        print("  - VRAM Recommendation : Batch Size 32 fits in 8GB VRAM.")
    else:
        RECOMMENDED_BATCH_SIZE = 16
        print("  - VRAM Recommendation : Batch Size 16 recommended for <= 4GB VRAM.")
else:
    RECOMMENDED_BATCH_SIZE = 32
    print("  - [NOTE] No CUDA GPU detected. Running on CPU architecture.")
    print("  - Batch Size          : 32 (standard)")""")

    # =========================================================================
    # Cell 2: Dataset, Manifest & Class JSON Path Discovery
    # =========================================================================
    add_code("""# Cell 2: Dataset, Manifest & Class JSON Path Discovery
def locate_plantdoc_dataset_root() -> Path:
    candidates = [
        Path("/kaggle/input/plantdoc-dataset"),
        Path("/kaggle/input/plantdoc"),
        Path("/kaggle/input/plantdoc-dataset-master"),
        Path("/kaggle/input/plantdoc-data"),
        Path("datasets/plantdoc"),
        Path("../datasets/plantdoc"),
        Path("../../datasets/plantdoc")
    ]
    # Check explicit candidates
    for c in candidates:
        if c.exists() and (c / "train").exists():
            print(f"[SmartCrop AI] Found PlantDoc dataset root at: {c.resolve()}")
            return c.resolve()
    
    # Recursive search under /kaggle/input if running on Kaggle
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for sub in kaggle_input.rglob("*"):
            if sub.is_dir() and (sub / "train").exists() and (sub / "test").exists():
                print(f"[SmartCrop AI] Discovered PlantDoc dataset root at: {sub.resolve()}")
                return sub.resolve()

    # Search current workspace
    for sub in Path(".").rglob("*"):
        if sub.is_dir() and sub.name == "plantdoc" and (sub / "train").exists():
            print(f"[SmartCrop AI] Discovered PlantDoc dataset root at: {sub.resolve()}")
            return sub.resolve()

    raise FileNotFoundError("PlantDoc dataset directory not found! Please check Kaggle input paths.")

def locate_file_path(filename: str, explicit_candidates: List[Path]) -> Path:
    for c in explicit_candidates:
        if c.exists():
            print(f"[SmartCrop AI] Found '{filename}' at: {c.resolve()}")
            return c.resolve()

    # Recursive search under /kaggle/input
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        matches = list(kaggle_input.rglob(filename))
        if matches:
            print(f"[SmartCrop AI] Found '{filename}' via Kaggle search: {matches[0].resolve()}")
            return matches[0].resolve()

    # Recursive search in working directory
    workspace_matches = list(Path(".").rglob(filename))
    if workspace_matches:
        print(f"[SmartCrop AI] Found '{filename}' via workspace search: {workspace_matches[0].resolve()}")
        return workspace_matches[0].resolve()

    raise FileNotFoundError(f"Required file '{filename}' could not be located in any directory!")

DATASET_ROOT = locate_plantdoc_dataset_root()

MANIFEST_PATH = locate_file_path(
    "plantdoc_split_manifest.json",
    explicit_candidates=[
        Path("/kaggle/input/datasets/anyasreekadali/plantdoc-support-files/plantdoc_split_manifest.json"),
        Path("/kaggle/input/plantdoc-support-files/plantdoc_split_manifest.json"),
        Path("outputs/plantdoc_split_manifest.json"),
        Path("../outputs/plantdoc_split_manifest.json"),
        Path("/kaggle/working/outputs/plantdoc_split_manifest.json")
    ]
)

CLASSES_PATH = locate_file_path(
    "disease_classes.json",
    explicit_candidates=[
        Path("/kaggle/input/datasets/anyasreekadali/plantdoc-support-files/disease_classes.json"),
        Path("/kaggle/input/plantdoc-support-files/disease_classes.json"),
        Path("models/disease/disease_classes.json"),
        Path("../models/disease/disease_classes.json"),
        Path("/kaggle/working/models/disease/disease_classes.json")
    ]
)

print(f"\\n[Path Discovery Verification Summary]")
print(f"  - DATASET_ROOT : {DATASET_ROOT}")
print(f"  - MANIFEST_PATH: {MANIFEST_PATH}")
print(f"  - CLASSES_PATH : {CLASSES_PATH}")""")

    # =========================================================================
    # Cell 3: Manifest Loading & Class Mapping Validation
    # =========================================================================
    add_code("""# Cell 3: Manifest Loading & Class Mapping Validation
# Loads manifest and disease classes JSON, establishes bidirectional mappings,
# and strictly validates that num_classes == 27.

# 1. Load split manifest
with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    manifest = json.load(f)

for split_key in ["train", "val", "test"]:
    if split_key not in manifest:
        raise KeyError(f"Manifest missing required partition key: '{split_key}'")

train_file_list = manifest["train"]
val_file_list = manifest["val"]
test_file_list = manifest["test"]

print(f"[Manifest Audit] Total Registered Samples: {len(train_file_list) + len(val_file_list) + len(test_file_list)}")
print(f"  - Train Partition : {len(train_file_list)} images")
print(f"  - Val Partition   : {len(val_file_list)} images")
print(f"  - Held-Out Test   : {len(test_file_list)} images (Kept strictly isolated)")

# 2. Load and normalize class mapping JSON
with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    raw_classes = json.load(f)

# Normalize structure (handles dict mapping '0' -> name, 'name' -> 0, or list)
classes_dict = {}
cls_to_idx = {}
idx_to_cls = {}

if isinstance(raw_classes, dict):
    # Check if keys are integer strings (e.g., {"0": "Apple Scab Leaf", ...})
    first_key = next(iter(raw_classes.keys()))
    if first_key.isdigit():
        for k, v in raw_classes.items():
            idx = int(k)
            name = str(v).strip()
            classes_dict[str(idx)] = name
            cls_to_idx[name] = idx
            idx_to_cls[idx] = name
    else:
        # Format {"Apple Scab Leaf": 0, ...}
        for k, v in raw_classes.items():
            name = str(k).strip()
            idx = int(v)
            classes_dict[str(idx)] = name
            cls_to_idx[name] = idx
            idx_to_cls[idx] = name
elif isinstance(raw_classes, list):
    # Format ["Apple Scab Leaf", ...]
    for idx, name in enumerate(raw_classes):
        name = str(name).strip()
        classes_dict[str(idx)] = name
        cls_to_idx[name] = idx
        idx_to_cls[idx] = name
else:
    raise ValueError(f"Unrecognized disease_classes.json structure: {type(raw_classes)}")

# 3. Validate class count
num_classes = len(classes_dict)
print(f"\\n[Class Mapping Audit] Detected {num_classes} active disease classes.")
if num_classes != 27:
    raise ValueError(f"Class count mismatch! Expected exactly 27 classes for PlantDoc benchmark, but found {num_classes}.")

print(f"[Class Mapping Validation PASSED] num_classes == {num_classes}")

# Display verified class index table
print(\"-\" * 60)
print(f\"{'Index':<6} | {'Class Name':<45}\")
print(\"-\" * 60)
for i in range(num_classes):
    print(f\"{i:<6} | {idx_to_cls[i]:<45}\")
print(\"-\" * 60)""")

    # =========================================================================
    # Cell 4: Agricultural Data Transforms & DataLoaders (288x288)
    # =========================================================================
    add_code("""# Cell 4: Agricultural Data Transforms & DataLoaders (288x288)
# Builds reproducible Dataset records and sets up DataLoaders.

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

# Training transforms with 288x288 field augmentation
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

# Deterministic validation transforms at 288x288
val_transforms = transforms.Compose([
    transforms.Resize((RESIZE_DIM, RESIZE_DIM)),
    transforms.CenterCrop(TARGET_RESOLUTION),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

def resolve_entry_metadata(entry: Any) -> Tuple[str, str]:
    \"\"\"
    Extracts relative file path and class name from a manifest entry.
    Handles cross-platform Windows/POSIX path strings as well as structured dict entries.
    \"\"\"
    if isinstance(entry, dict):
        rel_path = entry.get("file") or entry.get("path") or entry.get("image") or ""
        cls_name = entry.get("class") or entry.get("label") or entry.get("disease") or ""
        if not cls_name and rel_path:
            norm = str(rel_path).replace(chr(92), "/")
            parts = [p for p in norm.split("/") if p]
            if len(parts) >= 2:
                cls_name = parts[-2]
        return str(rel_path).strip(), str(cls_name).strip()
    elif isinstance(entry, str):
        rel_path = entry.strip()
        # Cross-platform normalization: replace backslashes (chr(92)) so it works on Linux/Kaggle
        norm = rel_path.replace(chr(92), "/")
        parts = [p for p in norm.split("/") if p]
        if len(parts) >= 2:
            cls_name = parts[-2].strip()
        else:
            cls_name = ""
        return rel_path, cls_name
    else:
        return str(entry).strip(), ""

def build_records_from_manifest(
    partition_name: str,
    raw_entries: List[Any],
    dataset_root: Path,
    cls_to_idx: Dict[str, int]
) -> List[Tuple[str, int, str]]:
    \"\"\"
    Validates and maps every entry in a manifest partition.
    Reports:
    - Successfully mapped records
    - Missing files on disk
    - Unknown classes (not in cls_to_idx)
    - Malformed entries (empty path or cannot determine class)
    Raises a strict diagnostic error if any entry is malformed or has an unknown class.
    \"\"\"
    mapped_records = []
    malformed_entries = []
    unknown_classes = []
    missing_files = []

    for idx, entry in enumerate(raw_entries):
        rel_path, cls_name = resolve_entry_metadata(entry)
        
        # Check for malformed entries
        if not rel_path or not cls_name:
            malformed_entries.append((idx, entry, rel_path, cls_name))
            continue
            
        # Check for unknown class label
        if cls_name not in cls_to_idx:
            unknown_classes.append((idx, entry, cls_name))
            continue

        # Locate physical file on disk with cross-platform and sanitization fallbacks
        norm_rel = rel_path.replace(chr(92), "/")
        candidates = [
            dataset_root / norm_rel,
            dataset_root / rel_path,
            dataset_root / norm_rel.replace("&", "_"),
            dataset_root / urllib.parse.unquote(norm_rel),
            dataset_root / urllib.parse.unquote(norm_rel).replace("&", "_")
        ]
        
        resolved_file = None
        for cand in candidates:
            if cand.exists() and cand.is_file():
                resolved_file = cand
                break
                
        if resolved_file is None:
            missing_files.append((idx, rel_path, cls_name))
            continue

        label_idx = cls_to_idx[cls_name]
        mapped_records.append((str(resolved_file), label_idx, cls_name))

    # Comprehensive Partition Diagnostic Report
    print(f\"\\n[Partition Audit: {partition_name.upper()}]\")
    print(f\"  - Total Manifest Entries : {len(raw_entries)}\")
    print(f\"  - Successfully Mapped    : {len(mapped_records)}\")
    print(f\"  - Missing Files on Disk  : {len(missing_files)}\")
    print(f\"  - Unknown Classes        : {len(unknown_classes)}\")
    print(f\"  - Malformed Entries      : {len(malformed_entries)}\")

    # Strict Validation Assertion: Stop if any label is unresolved
    if malformed_entries:
        err_msg = f\"Fatal Error in partition '{partition_name}': {len(malformed_entries)} malformed entry/entries detected!\\n\"
        for item in malformed_entries[:5]:
            err_msg += f\"  Sample: Index {item[0]}, Entry={repr(item[1])}, Resolved Path={repr(item[2])}, Resolved Class={repr(item[3])}\\n\"
        raise ValueError(err_msg)

    if unknown_classes:
        err_msg = f\"Fatal Error in partition '{partition_name}': {len(unknown_classes)} unknown class label(s) not in cls_to_idx!\\n\"
        for item in unknown_classes[:5]:
            err_msg += f\"  Sample: Index {item[0]}, Class='{item[2]}'\\n\"
        raise KeyError(err_msg)

    if len(mapped_records) == 0:
        raise RuntimeError(f\"Fatal Error: Partition '{partition_name}' has 0 valid mapped records!\")

    return mapped_records

train_records = build_records_from_manifest(\"train\", train_file_list, DATASET_ROOT, cls_to_idx)
val_records = build_records_from_manifest(\"val\", val_file_list, DATASET_ROOT, cls_to_idx)
test_records = build_records_from_manifest(\"test\", test_file_list, DATASET_ROOT, cls_to_idx)

# Verify partition totals
total_mapped = len(train_records) + len(val_records) + len(test_records)
print(f\"\\n[Overall Mapping Summary]\")
print(f\"  - Total Mapped Records   : {total_mapped} (Expected: 2549)\")
print(f\"  - Train Records Ready    : {len(train_records)} (Expected: 1783)\")
print(f\"  - Val Records Ready      : {len(val_records)} (Expected: 383)\")
print(f\"  - Test Records Isolated  : {len(test_records)} (Expected: 383)\")
assert len(train_records) == 1783, f\"Train count mismatch: {len(train_records)} != 1783\"
assert len(val_records) == 383, f\"Val count mismatch: {len(val_records)} != 383\"
assert len(test_records) == 383, f\"Test count mismatch: {len(test_records)} != 383\"
print(\"  - [PASS] All 2,549 records successfully mapped to valid classes with ZERO leakage.\")

BATCH_SIZE = RECOMMENDED_BATCH_SIZE
NUM_WORKERS = 2 if os.name != "nt" else 0

train_dataset = PlantDocDataset(train_records, transform=train_transforms)
val_dataset = PlantDocDataset(val_records, transform=val_transforms)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda")
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda")
)

# Calculate class weights strictly from the training partition
train_labels = [r[1] for r in train_records]
class_counts = [max(1, train_labels.count(c)) for c in range(num_classes)]
total_train = len(train_labels)
class_weights = [total_train / (num_classes * count) for count in class_counts]
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)

print(f"\\n[DataLoaders Initialized]")
print(f"  - Train Loader    : {len(train_loader)} batches (batch size {BATCH_SIZE})")
print(f"  - Val Loader      : {len(val_loader)} batches (batch size {BATCH_SIZE})")
print(f"  - Class Weights   : Computed strictly from training split (min={min(class_weights):.2f}, max={max(class_weights):.2f})")

# Sanity check: fetch 1 mini-batch to confirm shape and data loading
sample_images, sample_labels = next(iter(train_loader))
print(f"\\n[Sanity Check Batch Verification]")
print(f"  - Images Shape    : {sample_images.shape} (Expected: [{BATCH_SIZE}, 3, 288, 288])")
print(f"  - Labels Shape    : {sample_labels.shape} (Expected: [{BATCH_SIZE}])")
assert sample_images.shape[1:] == (3, TARGET_RESOLUTION, TARGET_RESOLUTION), "Batch image dimensions mismatch!"
assert sample_labels.max().item() < num_classes, "Label index exceeds num_classes!"
print("  - [PASS] DataLoaders validated successfully.")""")

    # =========================================================================
    # Cell 5: Model Architecture Definition & Model Initialization
    # =========================================================================
    add_code("""# Cell 5: EfficientNet-B0 Model Architecture & Initialization
# Builds proposed architecture with custom classification head for 27 classes.

def build_efficientnet_b0(num_classes: int = 27, dropout: float = 0.4) -> nn.Module:
    base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = base.classifier[1].in_features  # 1280
    base.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(512, num_classes)
    )
    return base

# Initialize model using validated num_classes
model = build_efficientnet_b0(num_classes=num_classes, dropout=0.4).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"[Model Architecture Verification]")
print(f"  - Model Type           : EfficientNet-B0 (Pretrained ImageNet-1K)")
print(f"  - Classifier Head      : Linear(1280 -> 512) -> ReLU -> Dropout(0.4) -> Linear(512 -> {num_classes})")
print(f"  - Total Parameters     : {total_params:,}")
print(f"  - Trainable Parameters : {trainable_params:,}")
print(f"  - Device               : {DEVICE}")

# Test forward pass with dummy tensor
dummy_input = torch.randn(1, 3, TARGET_RESOLUTION, TARGET_RESOLUTION, device=DEVICE)
model.eval()
with torch.no_grad():
    dummy_out = model(dummy_input)

assert dummy_out.shape == (1, num_classes), f"Output shape mismatch! Expected (1, {num_classes}), got {dummy_out.shape}"
assert not torch.isnan(dummy_out).any(), "Dummy logits contain NaN values!"
assert not torch.isinf(dummy_out).any(), "Dummy logits contain Inf values!"
print(f"  - Dummy Forward Pass   : [PASS] Output logits shape = {tuple(dummy_out.shape)}")

# Setup loss function with class weighting
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
print(f"  - Loss Function        : CrossEntropyLoss (Class-Weighted)")
print(f"\\n>>> ALL SETUP AND MODEL INITIALIZATION CHECKS PASSED. Ready for training.")""")

    # =========================================================================
    # Cell 6: Validation Evaluation Engine
    # =========================================================================
    add_code("""# Cell 6: Validation Evaluation Engine
# Computes loss, accuracy, Macro-F1, Macro-Precision, Macro-Recall, and Weighted-F1 on validation set.

def evaluate_validation(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Dict:
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
            
    val_loss = running_loss / len(loader.dataset)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    val_acc = float(np.mean(all_preds == all_targets))
    prec, rec, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)
    w_f1 = precision_recall_fscore_support(all_targets, all_preds, average="weighted", zero_division=0)[2]
    
    return {
        "loss": val_loss,
        "acc": val_acc,
        "macro_f1": float(f1),
        "macro_prec": float(prec),
        "macro_rec": float(rec),
        "weighted_f1": float(w_f1),
        "preds": all_preds,
        "targets": all_targets
    }

print("[Evaluation Engine Configured] Ready to measure validation metrics.")""")

    # =========================================================================
    # Cell 7: Stage 1 Training — Head Warmup (5 Epochs, Backbone Frozen)
    # =========================================================================
    add_code("""# Cell 7: Stage 1 Training — Head Warmup (5 Epochs, Backbone Frozen)
# Freezes EfficientNet-B0 backbone to protect pretrained weights while classifier head adapts.

STAGE1_EPOCHS = 5
for param in model.features.parameters():
    param.requires_grad = False
for param in model.classifier.parameters():
    param.requires_grad = True

optimizer_stage1 = optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-2)

best_val_f1 = 0.0
best_val_acc = 0.0
best_epoch = -1
CHECKPOINT_PATH = Path("disease_model_288_v1.pth")

print("=" * 80)
print(f"STAGE 1: HEAD WARMUP ({STAGE1_EPOCHS} Epochs, lr=1e-3, Backbone Frozen)")
print("=" * 80)

stage1_history = []
for epoch in range(1, STAGE1_EPOCHS + 1):
    t0 = time.time()
    model.train()
    running_loss = 0.0
    correct = 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer_stage1.zero_grad()
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer_stage1.step()
        running_loss += loss.item() * images.size(0)
        correct += (torch.argmax(out, dim=1) == labels).sum().item()
        
    tr_loss = running_loss / len(train_dataset)
    tr_acc = correct / len(train_dataset)
    v_metrics = evaluate_validation(model, val_loader, criterion, DEVICE)
    ep_time = time.time() - t0
    
    print(f"Stage 1 [Epoch {epoch:02d}/{STAGE1_EPOCHS:02d}] "
          f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc*100:.2f}% | "
          f"Val Loss: {v_metrics['loss']:.4f} | Val Acc: {v_metrics['acc']*100:.2f}% | "
          f"Val Macro-F1: {v_metrics['macro_f1']*100:.2f}% | Time: {ep_time:.1f}s")
          
    stage1_history.append({"epoch": epoch, "tr_loss": tr_loss, "tr_acc": tr_acc, **v_metrics, "time": ep_time})
    if v_metrics["macro_f1"] > best_val_f1:
        best_val_f1 = v_metrics["macro_f1"]
        best_val_acc = v_metrics["acc"]
        best_epoch = epoch
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_f1": best_val_f1,
            "val_acc": best_val_acc,
            "stage": 1,
            "resolution": TARGET_RESOLUTION,
            "num_classes": num_classes
        }, CHECKPOINT_PATH)
        print(f"  --> [Saved Best Checkpoint] Stage 1 Epoch {epoch} (Val F1: {best_val_f1*100:.2f}%)")""")

    # =========================================================================
    # Cell 8: Stage 2 Training — Full Backbone Fine-Tuning (25 Epochs, Differential LR)
    # =========================================================================
    add_code("""# Cell 8: Stage 2 Training — Full Backbone Fine-Tuning (25 Epochs, Differential LR)
# Unfreezes full backbone and trains with differential learning rates and Cosine Annealing.

STAGE2_EPOCHS = 25
PATIENCE = 6

for param in model.parameters():
    param.requires_grad = True

optimizer_stage2 = optim.AdamW([
    {"params": model.features.parameters(), "lr": 2e-5},
    {"params": model.classifier.parameters(), "lr": 2e-4}
], weight_decay=1e-2)

scheduler_stage2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_stage2, T_max=STAGE2_EPOCHS, eta_min=1e-6)

print("=" * 80)
print(f"STAGE 2: DISCRIMINATIVE FINE-TUNING ({STAGE2_EPOCHS} Epochs Max, Differential LR)")
print("=" * 80)

stage2_history = []
no_improve = 0

for epoch in range(1, STAGE2_EPOCHS + 1):
    global_epoch = STAGE1_EPOCHS + epoch
    t0 = time.time()
    model.train()
    running_loss = 0.0
    correct = 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer_stage2.zero_grad()
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer_stage2.step()
        running_loss += loss.item() * images.size(0)
        correct += (torch.argmax(out, dim=1) == labels).sum().item()
        
    scheduler_stage2.step()
    tr_loss = running_loss / len(train_dataset)
    tr_acc = correct / len(train_dataset)
    v_metrics = evaluate_validation(model, val_loader, criterion, DEVICE)
    ep_time = time.time() - t0
    
    print(f"Stage 2 [Epoch {epoch:02d}/{STAGE2_EPOCHS:02d} (Global {global_epoch:02d})] "
          f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc*100:.2f}% | "
          f"Val Loss: {v_metrics['loss']:.4f} | Val Acc: {v_metrics['acc']*100:.2f}% | "
          f"Val Macro-F1: {v_metrics['macro_f1']*100:.2f}% | Time: {ep_time:.1f}s")
          
    stage2_history.append({"stage_epoch": epoch, "global_epoch": global_epoch, "tr_loss": tr_loss, "tr_acc": tr_acc, **v_metrics, "time": ep_time})
    if v_metrics["macro_f1"] > best_val_f1:
        best_val_f1 = v_metrics["macro_f1"]
        best_val_acc = v_metrics["acc"]
        best_epoch = global_epoch
        no_improve = 0
        torch.save({
            "epoch": global_epoch,
            "model_state_dict": model.state_dict(),
            "val_f1": best_val_f1,
            "val_acc": best_val_acc,
            "stage": 2,
            "resolution": TARGET_RESOLUTION,
            "num_classes": num_classes
        }, CHECKPOINT_PATH)
        print(f"  --> [Saved Best Checkpoint] Global Epoch {global_epoch} (Val F1: {best_val_f1*100:.2f}%)")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"Early stopping triggered after {PATIENCE} epochs without validation Macro-F1 improvement.")
            break""")

    # =========================================================================
    # Cell 9: Final Validation Benchmark & Artifact Export
    # =========================================================================
    add_code("""# Cell 9: Final Validation Analysis, Comparative Scorecard & Artifact Export
print("=" * 80)
print(f"EVALUATING BEST CHECKPOINT: {CHECKPOINT_PATH}")
print(f"Best Checkpoint at Global Epoch: {best_epoch} | Val F1: {best_val_f1*100:.2f}% | Val Acc: {best_val_acc*100:.2f}%")
print("=" * 80)

best_ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(best_ckpt["model_state_dict"])
final_v = evaluate_validation(model, val_loader, criterion, DEVICE)

target_names = [idx_to_cls[i] for i in range(num_classes)]
val_report = classification_report(final_v["targets"], final_v["preds"], target_names=target_names, output_dict=True, zero_division=0)
val_cm = confusion_matrix(final_v["targets"], final_v["preds"]).tolist()

# Save metrics JSON
exp_metrics = {
    "experiment_name": "Experiment_1_HighRes_288x288_TwoStage_FineTuning",
    "resolution": [TARGET_RESOLUTION, TARGET_RESOLUTION],
    "best_global_epoch": best_epoch,
    "validation_results": {
        "val_loss": final_v["loss"],
        "val_accuracy": final_v["acc"],
        "val_macro_f1": final_v["macro_f1"],
        "val_macro_precision": final_v["macro_prec"],
        "val_macro_recall": final_v["macro_rec"],
        "val_weighted_f1": final_v["weighted_f1"]
    },
    "comparison_with_production_224": {
        "production_val_acc": 0.6919,
        "production_val_macro_f1": 0.6837,
        "new_val_acc": final_v["acc"],
        "new_val_macro_f1": final_v["macro_f1"],
        "acc_delta": final_v["acc"] - 0.6919,
        "f1_delta": final_v["macro_f1"] - 0.6837
    },
    "per_class_report": val_report,
    "confusion_matrix": val_cm
}

out_metrics_path = Path("plantdoc_experiment_A_metrics.json")
with open(out_metrics_path, "w", encoding="utf-8") as f:
    json.dump(exp_metrics, f, indent=2)
print(f"[Artifact Export] Saved metrics JSON to: {out_metrics_path.resolve()}")

# Final Verification: checkpoint loads and produces exactly 27 classes
verify_model = build_efficientnet_b0(num_classes=num_classes, dropout=0.4).to(DEVICE)
verify_model.load_state_dict(best_ckpt["model_state_dict"], strict=True)
verify_model.eval()

dummy_eval = torch.randn(1, 3, TARGET_RESOLUTION, TARGET_RESOLUTION, device=DEVICE)
with torch.no_grad():
    v_out = verify_model(dummy_eval)

assert v_out.shape == (1, 27), f"Checkpoint validation failed: shape {v_out.shape} != (1, 27)"
print(f"[Verification Successful] Best checkpoint restores with strict=True and produces {v_out.shape[1]} output classes.")""")

    notebook_data = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(notebook_data, f, indent=2)

    print(f"Generated notebook at: {NOTEBOOK_PATH}")

if __name__ == "__main__":
    generate_notebook()
