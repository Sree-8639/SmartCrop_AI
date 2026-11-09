"""
SmartCrop AI - Generator for PlantDoc Training Notebook
Creates notebooks/01_PlantDoc_Baseline_and_Proposed.ipynb with robust discovery,
MD5 conflicting-label deduplication, rigorous 70/15/15 splitting, paper-inspired
ResNet-50 + IN + CBAM, and proposed EfficientNet-B0 transfer learning.
"""

import json
from pathlib import Path


def build_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            },
            "accelerator": "GPU"
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    def add_md(source_text):
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source_text.strip().split("\n")]
        })

    def add_code(code_text):
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in code_text.strip().split("\n")]
        })

    # =========================================================================
    # CELL 1: Markdown Title & Research Documentation
    # =========================================================================
    add_md("""# SmartCrop AI: Foliar Disease Diagnosis Benchmark
## Comparative Research Study: Paper-Inspired Baseline (ResNet-50 + Instance Normalization + CBAM) vs. Proposed Architecture (EfficientNet-B0)

**Base Research Paper Reference:**  
*Pal, A. K., Patro, B. D. K., & Chaube, S. (2026). Design and implementation of a deep learning framework for automated crop classification and health diagnosis in precision agriculture. Scientific Reports, Nature Portfolio, 16, Article 42151. DOI: [10.1038/s41598-026-42151-5](https://doi.org/10.1038/s41598-026-42151-5).*

---

### Research Objectives & Comparative Methodology
1. **Paper-Inspired Baseline (Model A):**
   - Implements the architectural principles from Pal et al. (2026): **ResNet-50 backbone**, **Instance Normalization (IN)**, and **Convolutional Block Attention Module (CBAM)**.
   - *Technical Integrity Note on Normalization Parameters:* Instance Normalization normalizes activations per image instance rather than across mini-batches, stripping illumination style variance. When substituting `BatchNorm2d` with `InstanceNorm2d` in a pretrained ResNet-50, running batch statistics from ImageNet are discarded by mathematical definition, and affine parameters (scale $\\gamma$, shift $\\beta$) are initialized to standard baseline values ($\\gamma=1.0, \\beta=0.0$). We explicitly do not claim that pretrained normalization parameters are preserved.
2. **Proposed Architecture (Model B):**
   - **EfficientNet-B0** leveraging compound scaling (depth, width, resolution) with ImageNet transfer learning and custom classification head for resource-constrained edge/mobile deployment.
3. **Experimental Rigor & Leakage Prevention:**
   - Both models are trained and tested on the **exact same stratified 70% Train, 15% Validation, and 15% Test splits**.
   - MD5 hash deduplication is performed to eliminate identical duplicate images across splits and exclude conflicting-label images.
   - Seed is strictly fixed to `SEED = 42`.
   - Class imbalance is addressed via inverse-frequency weighted cross-entropy loss.""")

    # =========================================================================
    # CELL 2: Code - Imports, Seeds, and Device Configuration
    # =========================================================================
    add_code("""# Cell 1: Environment Setup, Deterministic Seeding, and Device Configuration
import os
import sys
import time
import json
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Policy on Image Integrity: We enforce strict validation.
# Truncated or corrupted image files are detected during initial scanning via PIL open, verify, and load,
# and excluded from the dataset before splitting. We do not silently decode corrupted images.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, models

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# Enforce strict deterministic execution
SEED = 42

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[SmartCrop AI] PyTorch Version : {torch.__version__}")
print(f"[SmartCrop AI] Torchvision     : {torchvision.__version__}")
print(f"[SmartCrop AI] Compute Device  : {DEVICE}")
if torch.cuda.is_available():
    print(f"[SmartCrop AI] GPU Model       : {torch.cuda.get_device_name(0)}")
    print(f"[SmartCrop AI] Total VRAM      : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")""")

    # =========================================================================
    # CELL 3: Code - Robust Dataset Discovery
    # =========================================================================
    add_code("""# Cell 2: Robust PlantDoc Dataset Discovery
# Inspects search root itself and descendants for flat, nested, and pre-split structures

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def discover_plantdoc_root() -> Tuple[Path, str]:
    search_roots = [
        Path("/kaggle/input"),
        Path("./datasets/plantdoc"),
        Path("../datasets/plantdoc"),
        Path("./plantdoc"),
        Path(".")
    ]
    
    candidates = []
    
    for search_base in search_roots:
        if not search_base.exists():
            continue
            
        # Inspect the search base itself and all its descendant directories
        all_dirs = [search_base] + [p for p in search_base.rglob("*") if p.is_dir() and not p.name.startswith(".")]
        
        for d in all_dirs:
            subdirs = [sd for sd in d.iterdir() if sd.is_dir() and not sd.name.startswith(".")]
            subdir_names = {sd.name.lower() for sd in subdirs}
            
            # Case A: Directory contains pre-split partitions ('train' and 'test')
            # The common dataset root is THIS parent folder (do NOT select 'train' alone!)
            if "train" in subdir_names and "test" in subdir_names:
                total_imgs = len([f for f in d.rglob("*") if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS])
                if total_imgs > 50:
                    candidates.append((d, "pre_split", total_imgs))
                    
            # Case B: Directory contains flat class folders (e.g. >= 5 subdirs each having images)
            elif len(subdirs) >= 5:
                subdirs_with_images = 0
                for sd in subdirs:
                    if any(f.suffix.lower() in VALID_EXTENSIONS for f in sd.iterdir() if f.is_file()):
                        subdirs_with_images += 1
                if subdirs_with_images >= 5:
                    total_imgs = len([f for f in d.rglob("*") if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS])
                    candidates.append((d, "flat_or_nested_classes", total_imgs))

    if not candidates:
        raise FileNotFoundError(
            "Could not automatically locate the PlantDoc dataset directory! "
            "Searched /kaggle/input and local fallbacks. Please attach the PlantDoc dataset "
            "to this Kaggle notebook via the right-hand panel: '+ Add Input' -> search 'plantdoc'."
        )
        
    # Sort candidates by total image count descending to pick the most complete root
    candidates.sort(key=lambda x: x[2], reverse=True)
    best_root, structure_type, img_count = candidates[0]
    
    print(f"[Discovery] Selected Dataset Root : {best_root}")
    print(f"[Discovery] Structure Detected    : {structure_type}")
    print(f"[Discovery] Total Images Indexed  : {img_count}")
    return best_root, structure_type

DATASET_ROOT, DATASET_STRUCTURE = discover_plantdoc_root()""")

    # =========================================================================
    # CELL 4: Code - Recursive Scan, Integrity Verification & MD5 Deduplication
    # =========================================================================
    add_code("""# Cell 3: Recursive Image Scan, Integrity Verification & Conflicting-Label Deduplication
# Computes MD5 file hashes; flags and excludes conflicting-label duplicates and corruptions

def compute_file_md5(file_path: Path, chunk_size=65536) -> str:
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()

def scan_verify_and_deduplicate(root_path: Path):
    all_files = [f for f in root_path.rglob("*") if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS]
    print(f"[Scan] Total raw image files found: {len(all_files)}")
    if len(all_files) == 0:
        raise ValueError(f"No valid image files found in {root_path}!")

    corrupted_files: List[Tuple[str, str]] = []
    seen_hashes: Dict[str, Tuple[Path, str]] = {}  # hash -> (Path, class_name)
    exact_duplicates: List[Tuple[str, str, str]] = []
    conflicting_label_duplicates: List[Tuple[str, str, str, str]] = []
    conflicting_hashes: Set[str] = set()
    valid_records: List[Tuple[Path, str, str]] = []  # (Path, class_name, hash)

    for img_path in sorted(all_files):
        # Determine class name: ignore partition names like 'train', 'test', 'val'
        parent_name = img_path.parent.name
        if parent_name.lower() in {"train", "test", "val", "validation", "images"}:
            class_name = img_path.parent.parent.name
        else:
            class_name = parent_name
        class_name = class_name.strip()

        # 1. Image integrity check: open, verify, and load raster completely
        try:
            with Image.open(img_path) as img:
                img.verify()
            with Image.open(img_path) as img:
                img.load()  # Ensure image data decodes completely without truncation
        except Exception as e:
            corrupted_files.append((str(img_path), str(e)))
            continue

        # 2. Compute file hash
        try:
            fhash = compute_file_md5(img_path)
        except Exception as e:
            corrupted_files.append((str(img_path), str(e)))
            continue

        # 3. Check for duplicates and label conflicts
        if fhash in seen_hashes:
            orig_path, orig_class = seen_hashes[fhash]
            if orig_class == class_name:
                exact_duplicates.append((str(img_path), str(orig_path), class_name))
            else:
                conflicting_label_duplicates.append((str(img_path), class_name, str(orig_path), orig_class))
                conflicting_hashes.add(fhash)
            continue

        seen_hashes[fhash] = (img_path, class_name)
        valid_records.append((img_path, class_name, fhash))

    # Filter out any records that share a hash involved in a conflicting label
    clean_class_to_files: Dict[str, List[Path]] = {}
    excluded_conflict_count = 0
    for p, c, h in valid_records:
        if h in conflicting_hashes:
            excluded_conflict_count += 1
            continue
        if c not in clean_class_to_files:
            clean_class_to_files[c] = []
        clean_class_to_files[c].append(p)

    # Save audit report to disk
    audit_report = {
        "dataset_root": str(root_path),
        "total_files_scanned": len(all_files),
        "corrupted_files_count": len(corrupted_files),
        "corrupted_files": corrupted_files[:20],
        "exact_duplicates_pruned": len(exact_duplicates),
        "conflicting_label_duplicates_detected": len(conflicting_label_duplicates),
        "conflicting_duplicates_log": conflicting_label_duplicates,
        "clean_unique_images": sum(len(v) for v in clean_class_to_files.values())
    }
    with open("plantdoc_deduplication_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)

    return clean_class_to_files, corrupted_files, exact_duplicates, conflicting_label_duplicates

class_to_files, corrupted_files, exact_dups, conflict_dups = scan_verify_and_deduplicate(DATASET_ROOT)

print(f"\\n=======================================================")
print(f" PlantDoc Preprocessing & Verification Summary")
print(f"=======================================================")
print(f" Total Unique Classes Discovered      : {len(class_to_files)}")
print(f" Total Unique Valid Images Retained   : {sum(len(v) for v in class_to_files.values())}")
print(f" Corrupted / Truncated Files Excluded : {len(corrupted_files)}")
print(f" Intra-Class Duplicates Pruned        : {len(exact_dups)}")
print(f" Conflicting-Label Duplicates Excluded: {len(conflict_dups)}")
print(f" Audit Log Saved to                   : plantdoc_deduplication_audit.json")
print(f"=======================================================\\n")""")

    # =========================================================================
    # CELL 5: Code - Class Feasibility Validation & Imbalance Weighting
    # =========================================================================
    add_code("""# Cell 4: Class Feasibility Validation & Inverse-Frequency Loss Weighting
# Validates that every class can support a 3-way split (Train, Val, Test)

MIN_REQUIRED_SAMPLES = 5  # Strict minimum to ensure at least 1 sample in val and test

unreliable_classes = []
reliable_class_to_files = {}

for cls_name, files in sorted(class_to_files.items()):
    count = len(files)
    if count < MIN_REQUIRED_SAMPLES:
        unreliable_classes.append((cls_name, count))
    else:
        reliable_class_to_files[cls_name] = files

if len(unreliable_classes) > 0:
    print(f"[WARNING] Detected {len(unreliable_classes)} class(es) with fewer than {MIN_REQUIRED_SAMPLES} samples:")
    for cls, cnt in unreliable_classes:
        print(f"   - '{cls}': only {cnt} sample(s). Cannot reliably support 3-way partition.")
    print("These underrepresented classes are excluded to guarantee unbiased generalization evaluation.\\n")

class_to_files = reliable_class_to_files
class_names = sorted(list(class_to_files.keys()))
num_classes = len(class_names)
total_valid_images = sum(len(v) for v in class_to_files.values())

if num_classes == 0:
    raise ValueError("No classes met the minimum sample threshold for training!")

print(f"[SmartCrop AI] Active Classes for Benchmark : {num_classes}")
print(f"[SmartCrop AI] Active Image Count            : {total_valid_images}")

# Class distribution DataFrame
distribution_df = pd.DataFrame([
    {"Index": i, "Class_Name": cls, "Count": len(class_to_files[cls])}
    for i, cls in enumerate(class_names)
]).sort_values(by="Count", ascending=False).reset_index(drop=True)

# Calculate inverse-frequency class weights for cross-entropy loss:
# w_c = N_total / (C * N_c)
class_counts = [len(class_to_files[cls]) for cls in class_names]
class_weights = [total_valid_images / (num_classes * c) for c in class_counts]
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

# Plot class distribution
plt.figure(figsize=(15, 6))
sns.barplot(data=distribution_df, x="Class_Name", y="Count", palette="viridis")
plt.xticks(rotation=90)
plt.title(f"PlantDoc Class Distribution (Classes: {num_classes}, Images: {total_valid_images})")
plt.xlabel("Crop Foliar Pathology Class")
plt.ylabel("Number of Unique Images")
plt.tight_layout()
plt.savefig("plantdoc_class_distribution.png", dpi=300)
plt.show()""")

    # =========================================================================
    # CELL 6: Code - Exact Proportional 70/15/15 Split with Leakage Verification
    # =========================================================================
    add_code("""# Cell 5: Exact Proportional 70/15/15 Stratified Split
# Explicitly utilizes train_ratio, val_ratio, and test_ratio.
# Explains integer rounding constraints and reports actual split percentages.

def exact_proportional_split(
    class_dict: Dict[str, List[Path]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Tuple[str, int, str]], List[Tuple[str, int, str]], List[Tuple[str, int, str]], pd.DataFrame]:
    
    assert train_ratio > 0 and val_ratio > 0 and test_ratio > 0, "All split ratios must be strictly positive!"
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Split ratios must sum to 1.0!"
    random.seed(seed)
    
    train_records = []
    val_records = []
    test_records = []
    per_class_stats = []

    for cls_idx, (cls_name, files) in enumerate(sorted(class_dict.items())):
        shuffled = list(files)
        random.shuffle(shuffled)
        n = len(shuffled)
        
        # Calculate target counts using ratio parameters
        # Because of integer rounding, exact proportions may vary slightly for small classes
        n_val = max(1, int(round(n * val_ratio)))
        n_test = max(1, int(round(n * test_ratio)))
        n_train = n - n_val - n_test
        
        # Ensure training set has at least 1 sample
        if n_train < 1:
            n_train = 1
            if n_val > n_test:
                n_val -= 1
            else:
                n_test -= 1

        train_files = shuffled[:n_train]
        val_files = shuffled[n_train:n_train + n_val]
        test_files = shuffled[n_train + n_val:]
        
        for f in train_files:
            train_records.append((str(f), cls_idx, cls_name))
        for f in val_files:
            val_records.append((str(f), cls_idx, cls_name))
        for f in test_files:
            test_records.append((str(f), cls_idx, cls_name))

        per_class_stats.append({
            "Class_Name": cls_name,
            "Total": n,
            "Train": n_train,
            "Val": n_val,
            "Test": n_test,
            "Train_Pct": round((n_train / n) * 100, 1),
            "Val_Pct": round((n_val / n) * 100, 1),
            "Test_Pct": round((n_test / n) * 100, 1)
        })

    return train_records, val_records, test_records, pd.DataFrame(per_class_stats)

train_data, val_data, test_data, split_stats_df = exact_proportional_split(
    class_to_files, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=SEED
)

actual_train_pct = (len(train_data) / total_valid_images) * 100.0
actual_val_pct = (len(val_data) / total_valid_images) * 100.0
actual_test_pct = (len(test_data) / total_valid_images) * 100.0

print(f"[Partition Summary (Target: 70% / 15% / 15%)]")
print(f"  - Training Set   : {len(train_data):>5} images (Actual: {actual_train_pct:.2f}%)")
print(f"  - Validation Set : {len(val_data):>5} images (Actual: {actual_val_pct:.2f}%)")
print(f"  - Test Set       : {len(test_data):>5} images (Actual: {actual_test_pct:.2f}%)")
print(f"  - Total Indexed  : {total_valid_images:>5} images")

# Verify zero data leakage across file paths
train_paths = {r[0] for r in train_data}
val_paths = {r[0] for r in val_data}
test_paths = {r[0] for r in test_data}

assert len(train_paths.intersection(val_paths)) == 0, "Leakage Error: Train and Val overlap!"
assert len(train_paths.intersection(test_paths)) == 0, "Leakage Error: Train and Test overlap!"
assert len(val_paths.intersection(test_paths)) == 0, "Leakage Error: Val and Test overlap!"
print("[Integrity Check] ZERO data leakage confirmed across Train, Val, and Test splits.")

# Save class index mapping JSON
class_map = {str(i): name for i, name in enumerate(class_names)}
with open("disease_classes.json", "w", encoding="utf-8") as f:
    json.dump(class_map, f, indent=2)
print("[Artifact Export] Saved disease_classes.json")""")

    # =========================================================================
    # CELL 7: Code - Sample Image Grid
    # =========================================================================
    add_code("""# Cell 6: Visualizing Verified In-Field Training Samples
def show_sample_grid(records: List[Tuple[str, int, str]], num_samples: int = 12):
    if len(records) == 0:
        raise ValueError("Cannot visualize samples from an empty dataset record list.")
    
    random.seed(SEED)
    k = min(num_samples, len(records))
    samples = random.sample(records, k)
    
    rows, cols = 3, 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, (path, _, cls_name) in enumerate(samples):
        img = Image.open(path).convert("RGB")
        axes[i].imshow(img)
        axes[i].set_title(f"{cls_name[:25]}", fontsize=9, fontweight="bold")
        axes[i].axis("off")
        
    for j in range(len(samples), len(axes)):
        axes[j].axis("off")
        
    plt.tight_layout()
    plt.savefig("plantdoc_sample_grid.png", dpi=300)
    plt.show()

show_sample_grid(train_data, num_samples=12)""")

    # =========================================================================
    # CELL 8: Code - Dataset Class & DataLoaders with Augmentation
    # =========================================================================
    add_code("""# Cell 7: Custom Dataset & Domain DataLoaders
# Applies agricultural-specific augmentations (solar intensity jitter, orientation invariance)

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
            # Explicitly raise exception instead of fabricating a synthetic black image
            raise RuntimeError(f"Failed to load image during training from path: {path}. Error: {e}")
            
        if self.transform:
            image = self.transform(image)
        return image, label

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Training transforms with field-domain augmentation
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Deterministic validation and test transforms
eval_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

BATCH_SIZE = 32
NUM_WORKERS = 2 if os.name != 'nt' else 0

train_dataset = PlantDocDataset(train_data, transform=train_transforms)
val_dataset = PlantDocDataset(val_data, transform=eval_transforms)
test_dataset = PlantDocDataset(test_data, transform=eval_transforms)

# CRITICAL RESEARCH REQUIREMENT: Both models will consume these IDENTICAL DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

if len(train_loader) == 0 or len(val_loader) == 0:
    raise RuntimeError("DataLoader initialization failed: train_loader or val_loader has 0 batches!")

print(f"[SmartCrop AI] DataLoaders successfully configured:")
print(f"  - Train Loader : {len(train_loader)} batches ({len(train_dataset)} samples)")
print(f"  - Val Loader   : {len(val_loader)} batches ({len(val_dataset)} samples)")
print(f"  - Test Loader  : {len(test_loader)} batches ({len(test_dataset)} samples)")""")

    # =========================================================================
    # CELL 9: Markdown - Baseline Theory & Normalization Parameter Note
    # =========================================================================
    add_md("""---
## Model A: Paper-Inspired Baseline (ResNet-50 + Instance Normalization + CBAM)

### Architectural Components:
1. **Instance Normalization (IN):**  
   Computes mean and variance per image channel independently across $(H, W)$ rather than across mini-batch samples. This removes individual image contrast and illumination artifacts common in outdoor foliage imagery:
   $$\\mu_{ti} = \\frac{1}{HW}\\sum_{j=1}^{H}\\sum_{k=1}^{W} x_{tijk}, \\quad \\sigma_{ti}^2 = \\frac{1}{HW}\\sum_{j=1}^{H}\\sum_{k=1}^{W} (x_{tijk} - \\mu_{ti})^2$$
   > **Scientific Research Note on Pretrained Parameter Loss:**  
   > When replacing standard `BatchNorm2d` with `InstanceNorm2d`, the ImageNet pretrained running statistics (`running_mean`, `running_var`) are discarded because InstanceNorm computes statistics dynamically on every forward pass. The learnable affine parameters (scale $\\gamma$ and shift $\\beta$) are initialized to $\\gamma=1.0, \\beta=0.0$. Consequently, the early convolutional layers lose their exact pretrained normalization calibration and require initial training adaptation. We do not claim pretrained normalization parameters are preserved.

2. **Convolutional Block Attention Module (CBAM):**  
   - **Channel Attention:** Employs shared MLP across average and max pooled spatial features to selectively highlight disease-discriminative spectral bands.
   - **Spatial Attention:** Convolves pooled channel features with a $7 \\times 7$ kernel to emphasize focal necrotic lesions while suppressing soil and background clutter.""")

    # =========================================================================
    # CELL 10: Code - Model A Architecture (ResNet-50 + IN + CBAM)
    # =========================================================================
    add_code("""# Cell 8: Architecture Definition - Model A (ResNet-50 + Instance Normalization + CBAM)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x_cat))


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
        
    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out


def replace_bn_with_in(module):
    \"\"\"
    Recursively replaces BatchNorm2d with InstanceNorm2d.
    Explicitly documents that BatchNorm running statistics from pretraining are dropped.
    \"\"\"
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_features = child.num_features
            # Initialize with affine=True so gamma and beta are learned
            in_layer = nn.InstanceNorm2d(num_features, affine=True, track_running_stats=False)
            setattr(module, name, in_layer)
        else:
            replace_bn_with_in(child)


class PaperInspiredResNet50(nn.Module):
    \"\"\"
    Baseline Architecture: ResNet-50 + Instance Normalization + CBAM.
    Inspired by Pal et al. (Scientific Reports, 2026).
    \"\"\"
    def __init__(self, num_classes: int, pretrained: bool = True):
        super(PaperInspiredResNet50, self).__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        base = models.resnet50(weights=weights)
        
        # Replace BatchNorm with InstanceNorm (discards BN running statistics)
        replace_bn_with_in(base)
        
        self.stem = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool
        )
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        
        # CBAM Attention inserted after Stage 4 residual output (2048 channels)
        self.cbam = CBAM(in_planes=2048, ratio=16, kernel_size=7)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.cbam(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

model_baseline = PaperInspiredResNet50(num_classes=num_classes, pretrained=True).to(DEVICE)
baseline_param_count = sum(p.numel() for p in model_baseline.parameters() if p.requires_grad)
print(f"[Model A] Baseline ResNet-50 + IN + CBAM initialized.")
print(f"          Trainable Parameters: {baseline_param_count:,}")""")

    # =========================================================================
    # CELL 11: Markdown - Proposed Model Theory
    # =========================================================================
    add_md("""---
## Model B: Proposed SmartCrop AI Architecture (EfficientNet-B0)

### Compound Scaling Principle:
EfficientNet scales network depth $d$, width $w$, and resolution $r$ uniformly using fixed compound coefficients:
$$\\text{Depth: } d = \\alpha^\\phi, \\quad \\text{Width: } w = \\beta^\\phi, \\quad \\text{Resolution: } r = \\gamma^\\phi$$
$$\\text{Subject to: } \\alpha \\cdot \\beta^2 \\cdot \\gamma^2 \\approx 2, \\quad \\alpha \\ge 1, \\beta \\ge 1, \\gamma \\ge 1$$

Compared to ResNet-50 (~25.6M parameters), EfficientNet-B0 (~5.3M parameters) requires significantly fewer floating-point operations (FLOPs), enabling rapid inference on agricultural edge devices without sacrificing representation capacity.""")

    # =========================================================================
    # CELL 12: Code - Model B Architecture (EfficientNet-B0)
    # =========================================================================
    add_code("""# Cell 9: Architecture Definition - Model B (Proposed: EfficientNet-B0)

class ProposedEfficientNetB0(nn.Module):
    \"\"\"
    Proposed Architecture: EfficientNet-B0 with Compound Scaling & Transfer Learning.
    \"\"\"
    def __init__(self, num_classes: int, pretrained: bool = True):
        super(ProposedEfficientNetB0, self).__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base = models.efficientnet_b0(weights=weights)
        
        self.features = base.features
        self.avgpool = base.avgpool
        in_features = base.classifier[1].in_features
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

model_proposed = ProposedEfficientNetB0(num_classes=num_classes, pretrained=True).to(DEVICE)
proposed_param_count = sum(p.numel() for p in model_proposed.parameters() if p.requires_grad)
print(f"[Model B] Proposed EfficientNet-B0 initialized.")
print(f"          Trainable Parameters: {proposed_param_count:,}")""")

    # =========================================================================
    # CELL 13: Code - Generic Training & Validation Pipeline
    # =========================================================================
    add_code("""# Cell 10: Generic Training & Validation Engine with Checkpointing & Early Stopping

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)
        
    return running_loss / total, correct / total


def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return running_loss / total, correct / total, macro_f1


def execute_training_pipeline(
    model, model_name, train_loader, val_loader, criterion, optimizer, scheduler,
    num_epochs=18, patience=5, checkpoint_path="checkpoint.pth"
):
    if len(train_loader) == 0 or len(val_loader) == 0:
        raise RuntimeError(f"Cannot execute training for {model_name}: train_loader or val_loader has 0 batches!")

    print(f"\\n=======================================================")
    print(f" Starting Training: {model_name}")
    print(f"=======================================================")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
    
    best_val_f1 = -1.0
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        v_loss, v_acc, v_f1 = validate_epoch(model, val_loader, criterion, DEVICE)
        
        if scheduler:
            scheduler.step()
            
        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)
        history["val_f1"].append(v_f1)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch [{epoch:02d}/{num_epochs:02d}] "
            f"Train Loss: {t_loss:.4f} | Train Acc: {t_acc*100:.2f}% || "
            f"Val Loss: {v_loss:.4f} | Val Acc: {v_acc*100:.2f}% | Val Macro-F1: {v_f1:.4f} | LR: {current_lr:.6f}"
        )
        
        # Checkpoint based on Validation Macro-F1
        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_f1': best_val_f1,
                'val_acc': v_acc,
                'model_name': model_name
            }, checkpoint_path)
            print(f"  --> Checkpoint saved: Validation Macro-F1 improved to {best_val_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\\n[Early Stopping] No improvement in validation Macro-F1 for {patience} consecutive epochs.")
                break
                
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed/60:.2f} mins. Peak Validation Macro-F1: {best_val_f1:.4f}\\n")
    return history""")

    # =========================================================================
    # CELL 14: Code - Train Model A (Baseline)
    # =========================================================================
    add_code("""# Cell 11: Training Execution - Model A (Baseline: ResNet-50 + IN + CBAM)
# Cosine Annealing scheduler aligned with EPOCHS=18 (T_max=EPOCHS)

EPOCHS = 18
PATIENCE = 5

criterion_baseline = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer_baseline = optim.AdamW(model_baseline.parameters(), lr=1e-4, weight_decay=1e-2)
scheduler_baseline = optim.lr_scheduler.CosineAnnealingLR(optimizer_baseline, T_max=EPOCHS, eta_min=1e-6)

history_baseline = execute_training_pipeline(
    model=model_baseline,
    model_name="Model_A_Baseline_ResNet50_IN_CBAM",
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion_baseline,
    optimizer=optimizer_baseline,
    scheduler=scheduler_baseline,
    num_epochs=EPOCHS,
    patience=PATIENCE,
    checkpoint_path="baseline_disease_model.pth"
)""")

    # =========================================================================
    # CELL 15: Code - Train Model B (Proposed)
    # =========================================================================
    add_code("""# Cell 12: Training Execution - Model B (Proposed: EfficientNet-B0)
# Cosine Annealing scheduler aligned with EPOCHS=18 (T_max=EPOCHS)

criterion_proposed = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer_proposed = optim.AdamW(model_proposed.parameters(), lr=3e-4, weight_decay=1e-2)
scheduler_proposed = optim.lr_scheduler.CosineAnnealingLR(optimizer_proposed, T_max=EPOCHS, eta_min=1e-6)

history_proposed = execute_training_pipeline(
    model=model_proposed,
    model_name="Model_B_Proposed_EfficientNetB0",
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion_proposed,
    optimizer=optimizer_proposed,
    scheduler=scheduler_proposed,
    num_epochs=EPOCHS,
    patience=PATIENCE,
    checkpoint_path="disease_model.pth"
)""")

    # =========================================================================
    # CELL 16: Code - Training Curves Visualization
    # =========================================================================
    add_code("""# Cell 13: Comparative Training and Validation Curves Export

def plot_learning_curves(history, title, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(epochs, history["train_loss"], 'o-', color='navy', label='Train Loss')
    ax1.plot(epochs, history["val_loss"], 's--', color='crimson', label='Val Loss')
    ax1.set_title(f"{title} - Cross Entropy Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()
    
    # Accuracy and Macro-F1 curves
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], 'o-', color='forestgreen', label='Train Accuracy (%)')
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], 's--', color='darkorange', label='Val Accuracy (%)')
    ax2.plot(epochs, [f * 100 for f in history["val_f1"]], '^-.', color='purple', label='Val Macro-F1 (%)')
    ax2.set_title(f"{title} - Accuracy & Macro-F1")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Percentage (%)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

plot_learning_curves(history_baseline, "Baseline (ResNet-50 + IN + CBAM)", "baseline_training_curves.png")
plot_learning_curves(history_proposed, "Proposed (EfficientNet-B0)", "proposed_training_curves.png")""")

    # =========================================================================
    # CELL 17: Code - Evaluation on Unseen Test Split (Best Checkpoint Restored)
    # =========================================================================
    add_code("""# Cell 14: Comprehensive Test Set Evaluation with Best Checkpoint Restoration
# Evaluates on the held-out test split under identical conditions.
# Explicitly passes labels=list(range(num_classes)) and matching target_names to guarantee consistent dimensions.

def evaluate_test_split(model, checkpoint_path, dataloader, class_names, device):
    # Restore the best checkpoint state dict before test evaluation
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    num_total_classes = len(class_names)
    all_class_indices = list(range(num_total_classes))

    acc = accuracy_score(all_labels, all_preds)
    prec_macro = precision_score(all_labels, all_preds, labels=all_class_indices, average='macro', zero_division=0)
    rec_macro = recall_score(all_labels, all_preds, labels=all_class_indices, average='macro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, labels=all_class_indices, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, labels=all_class_indices, average='weighted', zero_division=0)
    
    # Ensure confusion matrix dimensions always match the complete class list
    cm = confusion_matrix(all_labels, all_preds, labels=all_class_indices)
    report_dict = classification_report(
        all_labels, all_preds,
        labels=all_class_indices,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    metrics = {
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict
    }
    return metrics, cm, all_labels, all_preds

print(f"[SmartCrop AI] Restoring best checkpoint and evaluating Model A on Test Set ({len(test_data)} samples, {actual_test_pct:.2f}%)...")
metrics_baseline, cm_baseline, test_labels, preds_baseline = evaluate_test_split(
    model_baseline, "baseline_disease_model.pth", test_loader, class_names, DEVICE
)

print(f"[SmartCrop AI] Restoring best checkpoint and evaluating Model B on Test Set ({len(test_data)} samples, {actual_test_pct:.2f}%)...")
metrics_proposed, cm_proposed, _, preds_proposed = evaluate_test_split(
    model_proposed, "disease_model.pth", test_loader, class_names, DEVICE
)""")

    # =========================================================================
    # CELL 18: Code - Confusion Matrices Plotting
    # =========================================================================
    add_code("""# Cell 15: Confusion Matrices Generation & Export
# Visualizes confusion matrices with dimensions strictly matching all class names

def plot_cm(cm, class_names, title, save_path):
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        cm, annot=False, cmap='Blues',
        xticklabels=[c[:15] for c in class_names],
        yticklabels=[c[:15] for c in class_names]
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Disease Class", fontsize=12)
    plt.ylabel("True Disease Class", fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

plot_cm(cm_baseline, class_names, "Model A: ResNet-50 + IN + CBAM (Baseline) Confusion Matrix", "baseline_confusion_matrix.png")
plot_cm(cm_proposed, class_names, "Model B: EfficientNet-B0 (Proposed) Confusion Matrix", "proposed_confusion_matrix.png")""")

    # =========================================================================
    # CELL 19: Code - Profiling: Latency, Parameters, and Disk Size
    # =========================================================================
    add_code("""# Cell 16: Hardware Efficiency Profiling (Parameters, Disk Size, and GPU Latency)

def profile_latency(model, device, input_shape=(1, 3, 224, 224), warmups=20, iterations=100):
    model.eval()
    dummy = torch.randn(*input_shape).to(device)
    
    with torch.no_grad():
        for _ in range(warmups):
            _ = model(dummy)
            
    if device.type == "cuda":
        torch.cuda.synchronize()
        
    start = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
    total_time = time.time() - start
    return (total_time / iterations) * 1000.0

lat_baseline_ms = profile_latency(model_baseline, DEVICE)
size_baseline_mb = os.path.getsize("baseline_disease_model.pth") / (1024 * 1024)

lat_proposed_ms = profile_latency(model_proposed, DEVICE)
size_proposed_mb = os.path.getsize("disease_model.pth") / (1024 * 1024)

print(f"[Profiling Summary]")
print(f"Model A (Baseline): Latency = {lat_baseline_ms:.2f} ms | Params = {baseline_param_count/1e6:.2f}M | Size = {size_baseline_mb:.2f} MB")
print(f"Model B (Proposed): Latency = {lat_proposed_ms:.2f} ms | Params = {proposed_param_count/1e6:.2f}M | Size = {size_proposed_mb:.2f} MB")""")

    # =========================================================================
    # CELL 20: Code - Comparative Table & Empirical Comparison Chart
    # =========================================================================
    add_code("""# Cell 17: Empirical Comparison Table & Visualization
# Reports actual test split counts and measured test metrics

comparison_rows = [
    {
        "Architecture": "Paper-Inspired Baseline (ResNet-50 + IN + CBAM)",
        "Parameters (M)": round(baseline_param_count / 1e6, 2),
        "Size (MB)": round(size_baseline_mb, 2),
        "Latency (ms)": round(lat_baseline_ms, 2),
        "Test Samples": len(test_data),
        "Test Accuracy (%)": round(metrics_baseline["accuracy"] * 100, 2),
        "Test Macro-Precision (%)": round(metrics_baseline["precision_macro"] * 100, 2),
        "Test Macro-Recall (%)": round(metrics_baseline["recall_macro"] * 100, 2),
        "Test Macro-F1 (%)": round(metrics_baseline["f1_macro"] * 100, 2)
    },
    {
        "Architecture": "Proposed SmartCrop AI (EfficientNet-B0)",
        "Parameters (M)": round(proposed_param_count / 1e6, 2),
        "Size (MB)": round(size_proposed_mb, 2),
        "Latency (ms)": round(lat_proposed_ms, 2),
        "Test Samples": len(test_data),
        "Test Accuracy (%)": round(metrics_proposed["accuracy"] * 100, 2),
        "Test Macro-Precision (%)": round(metrics_proposed["precision_macro"] * 100, 2),
        "Test Macro-Recall (%)": round(metrics_proposed["recall_macro"] * 100, 2),
        "Test Macro-F1 (%)": round(metrics_proposed["f1_macro"] * 100, 2)
    }
]

comparison_df = pd.DataFrame(comparison_rows)
print("\\n" + "=" * 80)
print(f" EMPIRICAL RESEARCH COMPARISON TABLE (HELD-OUT TEST SET: {len(test_data)} SAMPLES, {actual_test_pct:.1f}%)")
print("=" * 80)
print(comparison_df.to_string(index=False))
print("=" * 80 + "\\n")

# Comparative Bar Charts
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
labels = ["Baseline\\n(ResNet-50+IN+CBAM)", "Proposed\\n(EfficientNet-B0)"]

# 1. Accuracy
accs = [comparison_df.loc[0, "Test Accuracy (%)"], comparison_df.loc[1, "Test Accuracy (%)"]]
axes[0].bar(labels, accs, color=['steelblue', 'seagreen'])
axes[0].set_title(f"Test Accuracy (%) [N={len(test_data)}]")
axes[0].set_ylim(0, 100)
for i, v in enumerate(accs):
    axes[0].text(i, v + 1, f"{v}%", ha='center', fontweight='bold')

# 2. Parameters
params = [comparison_df.loc[0, "Parameters (M)"], comparison_df.loc[1, "Parameters (M)"]]
axes[1].bar(labels, params, color=['slategray', 'coral'])
axes[1].set_title("Parameter Count (Millions)")
for i, v in enumerate(params):
    axes[1].text(i, v + 0.3, f"{v}M", ha='center', fontweight='bold')

# 3. Latency
lats = [comparison_df.loc[0, "Latency (ms)"], comparison_df.loc[1, "Latency (ms)"]]
axes[2].bar(labels, lats, color=['purple', 'darkcyan'])
axes[2].set_title("Single-Image GPU Latency (ms)")
for i, v in enumerate(lats):
    axes[2].text(i, v + 0.2, f"{v}ms", ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig("model_comparison_chart.png", dpi=300)
plt.show()

# Export comparison results JSON
with open("comparison_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "test_sample_count": len(test_data),
        "test_actual_percentage": actual_test_pct,
        "comparison_table": comparison_rows,
        "baseline_metrics": metrics_baseline,
        "proposed_metrics": metrics_proposed
    }, f, indent=2)
print("[Artifact Export] Saved comparison_results.json")""")

    # =========================================================================
    # CELL 21: Code - Sample Inference & Artifact Export Manifest
    # =========================================================================
    add_code("""# Cell 18: Unseen Sample Predictions & Production Artifact Export Manifest

def visualize_sample_inferences(records, model_a, model_b, class_names, device, num_samples=6):
    if len(records) == 0:
        raise ValueError("Cannot run sample inferences: records list is empty.")
    random.seed(SEED)
    k = min(num_samples, len(records))
    samples = random.sample(records, k)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    eval_t = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    model_a.eval()
    model_b.eval()
    
    for i, (path, true_idx, true_name) in enumerate(samples):
        img_raw = Image.open(path).convert("RGB")
        tensor_img = eval_t(img_raw).unsqueeze(0).to(device)
        
        with torch.no_grad():
            out_a = model_a(tensor_img)
            pred_a = torch.argmax(out_a, dim=1).item()
            conf_a = F.softmax(out_a, dim=1)[0, pred_a].item()
            
            out_b = model_b(tensor_img)
            pred_b = torch.argmax(out_b, dim=1).item()
            conf_b = F.softmax(out_b, dim=1)[0, pred_b].item()
            
        axes[i].imshow(img_raw)
        title = (
            f"Ground Truth: {true_name[:20]}\\n"
            f"Baseline: {class_names[pred_a][:18]} ({conf_a*100:.1f}%)\\n"
            f"Proposed: {class_names[pred_b][:18]} ({conf_b*100:.1f}%)"
        )
        axes[i].set_title(title, fontsize=9, loc='left')
        axes[i].axis("off")
        
    for j in range(len(samples), len(axes)):
        axes[j].axis("off")
        
    plt.tight_layout()
    plt.savefig("sample_comparative_predictions.png", dpi=300)
    plt.show()

visualize_sample_inferences(test_data, model_baseline, model_proposed, class_names, DEVICE, num_samples=6)

# Export disease_metrics.json conforming to SmartCrop AI local runtime schema
disease_metrics = {
    "model_name": "EfficientNet-B0 (Proposed)",
    "baseline_name": "ResNet-50 + IN + CBAM (Paper-Inspired)",
    "dataset": "PlantDoc",
    "evaluation_split": f"Held-out Unseen Test Set ({len(test_data)} samples, {actual_test_pct:.2f}%)",
    "num_classes": num_classes,
    "metrics": {
        "accuracy": metrics_proposed["accuracy"],
        "macro_f1": metrics_proposed["f1_macro"],
        "macro_precision": metrics_proposed["precision_macro"],
        "macro_recall": metrics_proposed["recall_macro"],
        "weighted_f1": metrics_proposed["f1_weighted"]
    },
    "efficiency": {
        "parameters": proposed_param_count,
        "checkpoint_mb": size_proposed_mb,
        "latency_ms": lat_proposed_ms
    },
    "baseline_comparison": {
        "baseline_accuracy": metrics_baseline["accuracy"],
        "baseline_macro_f1": metrics_baseline["f1_macro"],
        "baseline_parameters": baseline_param_count,
        "baseline_latency_ms": lat_baseline_ms
    }
}

with open("disease_metrics.json", "w", encoding="utf-8") as f:
    json.dump(disease_metrics, f, indent=2)

manifest_files = [
    "baseline_disease_model.pth",
    "disease_model.pth",
    "disease_classes.json",
    "disease_metrics.json",
    "comparison_results.json",
    "plantdoc_deduplication_audit.json",
    "plantdoc_class_distribution.png",
    "baseline_training_curves.png",
    "proposed_training_curves.png",
    "baseline_confusion_matrix.png",
    "proposed_confusion_matrix.png",
    "model_comparison_chart.png",
    "sample_comparative_predictions.png"
]

print("\\n" + "=" * 70)
print(" KAGGLE ARTIFACT EXPORT VERIFICATION")
print("=" * 70)
for fn in manifest_files:
    if os.path.exists(fn):
        sz = os.path.getsize(fn) / (1024 * 1024)
        print(f"  [x] {fn:<35} : {sz:.2f} MB")
    else:
        print(f"  [ ] {fn:<35} : MISSING")
print("=" * 70)
print(\"\"\"
Local Antigravity Workspace Deployment:
Download 'disease_model.pth', 'disease_classes.json', and 'disease_metrics.json'
from the /kaggle/working/ output pane and place them in SmartCrop_AI/models/disease/.
Copy 'baseline_disease_model.pth' and 'comparison_results.json' into SmartCrop_AI/outputs/.
\"\"\")""")

    # Write notebook file
    out_file = Path("notebooks/01_PlantDoc_Baseline_and_Proposed.ipynb")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"[Success] Generated notebook with {len(nb['cells'])} cells at: {out_file}")


if __name__ == "__main__":
    build_notebook()
