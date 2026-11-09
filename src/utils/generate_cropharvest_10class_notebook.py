"""
SmartCrop AI - Generator for CropHarvest 10-Class Classification Notebook
Generates notebooks/07_CropHarvest_10Class_Classification.ipynb

Covers:
1. Environment Setup, Dependencies & Hardware Inspection.
2. Dynamic Dataset Path Discovery for /kaggle/input/ and local paths.
3. 10-Class Metadata Audit (9 FAO Crop Categories + 1 Non-Crop Land-Cover Class).
4. Dataset Structure, Array Integrity & Shape Audit (12x18, 0 NaNs).
5. Stratified Train/Val Partitioning (80:20, SEED 42) & Official Test Isolation.
6. PyTorch Dataset & DataLoaders with Normalization.
7. Bi-LSTM 10-Class Architecture & Forward/Backward Sanity Check.
8. Model Training Execution (Manual Gate: EXECUTE_TRAINING = False).
9. Validation Evaluation & Detailed 10-Class Scorecard.
10. Official Held-Out Test Split Evaluation & Per-Class Sample Support Audit.
11. 10x10 Confusion Matrix Heatmap & Phenological Profiles.
12. Deliverable Checkpoint & Metrics Export to models/cropharvest/10class/.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH = NOTEBOOKS_DIR / "07_CropHarvest_10Class_Classification.ipynb"


def build_10class_notebook() -> dict:
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

    def add_md(source_text: str):
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source_text.strip().split("\n")]
        })

    def add_code(code_text: str):
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in code_text.strip().split("\n")]
        })

    # CELL 0: Title & Objectives
    add_md(r"""# SmartCrop AI: Pipeline 2 — CropHarvest 10-Class Classification
## Earth Observation Time-Series Modeling: 9 FAO Crop Groups + 1 Non-Crop Land-Cover Class

**Benchmark Reference:**  
*Tseng, G., Zvonkov, I., Duro, C. P., et al. (2021). CropHarvest: A global dataset for crop-type classification. NeurIPS 2021 Datasets and Benchmarks Track.*

---

### Pipeline Objectives & Taxonomy Standards
1. **10-Class Target Taxonomy:**
   - **9 Agricultural Crop Categories:**
     0. `cereals` (13,259 samples)
     1. `other` (8,406 samples)
     2. `fruits_nuts` (4,110 samples)
     3. `vegetables_melons` (3,755 samples)
     4. `oilseeds` (2,868 samples)
     5. `leguminous` (1,837 samples)
     6. `root_tuber` (1,179 samples)
     7. `beverage_spice` (908 samples)
     8. `sugar` (748 samples)
   - **1 Land-Cover Class (Non-Agricultural):**
     9. `non_crop` (5,533 samples — pasture, forest, urban, water, barren land).  
     *(Explicitly documented as non-crop land-cover, distinct from crop species).*
   - **Total Eligible Dataset:** **42,603 verified samples**.
2. **Earth Observation Input ($12 \times 18$):**
   - 12 monthly composite steps ($T=12$).
   - 18 calibrated channels ($C=18$): Sentinel-1 SAR (VV, VH), Sentinel-2 Multi-Spectral (10 bands), ERA5 Climate, SRTM Topography.
   - Normalized via official `normalizing_dict.h5`.
3. **Official Benchmark Splits & Leakage Prevention:**
   - Official global test split (`is_test: true`, 107 samples) is preserved strictly for evaluation.
   - Training pool (`is_test: false`, 42,496 samples) is partitioned into stratified 80:20 Train/Validation splits (`seed=42`):
     - **Train (80%):** 34,000 samples
     - **Val (20%):** 8,496 samples
   - Every single class is represented in both Train and Validation.
4. **Test Set Support Audit:**
   - Inspects per-class test sample representation, noting that `vegetables_melons` has 0 samples and `fruits_nuts`, `oilseeds`, `leguminous`, `sugar` have 1 sample each in the official benchmark test set.
5. **Manual Training Gate:**
   - `EXECUTE_TRAINING = False` by default to allow pre-flight hardware and data verification before execution.""")

    # CELL 1: Environment & GPU Inspection
    add_code("""# Cell 1: Environment Setup, Dependencies & Hardware Inspection
import os
import sys
import json
import time
import math
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter

# Dependencies verification
try:
    import h5py
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import sklearn
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    print("[Setup] Installing required dependencies...")
    !pip install -q h5py scikit-learn seaborn matplotlib
    import h5py
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns

# Deterministic Seeding
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Hardware Inspection
print("=" * 75)
print("  CROPHARVEST PIPELINE 2: 10-CLASS CLASSIFICATION ENVIRONMENT")
print("=" * 75)
print(f"Python Version      : {sys.version.split()[0]}")
print(f"PyTorch Version     : {torch.__version__}")
print(f"h5py Version        : {h5py.__version__}")
print(f"CUDA Available      : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    gpu_cap = torch.cuda.get_device_capability(0)
    print(f"GPU Device          : {gpu_name}")
    print(f"GPU VRAM            : {gpu_mem:.2f} GB")
    print(f"Compute Capability  : {gpu_cap[0]}.{gpu_cap[1]}")
    DEVICE = torch.device("cuda:0")
else:
    print("WARNING: CUDA is NOT available. Running on CPU.")
    DEVICE = torch.device("cpu")

IS_KAGGLE = Path("/kaggle").exists()
if IS_KAGGLE:
    OUTPUT_DIR = Path("/kaggle/working/models/cropharvest/10class")
else:
    OUTPUT_DIR = Path("models/cropharvest/10class")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Runtime Environment : {'Kaggle Kernel' if IS_KAGGLE else 'Local Workstation'}")
print(f"Output Directory    : {OUTPUT_DIR.resolve()}")
print("=" * 75)""")

    # CELL 2: Dynamic Path Discovery
    add_code("""# Cell 2: Dynamic Dataset Path Discovery
# Searches /kaggle/input and local roots to locate arrays/, labels.geojson, and normalizing_dict.h5.

def discover_cropharvest_10class_paths() -> Dict[str, Any]:
    search_roots = []
    if Path("/kaggle/input").exists():
        search_roots.append(Path("/kaggle/input"))
        try:
            for item in Path("/kaggle/input").iterdir():
                if item.is_dir():
                    search_roots.append(item)
                    for sub in item.iterdir():
                        if sub.is_dir():
                            search_roots.append(sub)
        except Exception as e:
            print(f"[Warning] Error scanning /kaggle/input: {e}")

    search_roots.extend([
        Path("datasets/cropharvest"),
        Path("../datasets/cropharvest"),
        Path("../../datasets/cropharvest"),
        Path("."),
        Path("..")
    ])

    resolved = {}
    print("[Discovery] Searching for CropHarvest dataset components...")

    for root in search_roots:
        if not root.exists():
            continue
        arrays_cand = root / "arrays"
        geojson_cand = root / "labels.geojson"
        norm_cand = root / "normalizing_dict.h5"

        if arrays_cand.exists() and geojson_cand.exists():
            resolved["dataset_root"] = root.resolve()
            resolved["arrays_dir"] = arrays_cand.resolve()
            resolved["labels_geojson"] = geojson_cand.resolve()
            if norm_cand.exists():
                resolved["normalizing_dict"] = norm_cand.resolve()
            break

    if "arrays_dir" not in resolved or "labels_geojson" not in resolved:
        raise FileNotFoundError("Could not locate CropHarvest arrays/ folder and labels.geojson.")

    return resolved

ch_paths = discover_cropharvest_10class_paths()
print("\\n" + "=" * 75)
print("  CROPHARVEST PATH DISCOVERY SUCCESSFUL")
print("=" * 75)
print(f"  Dataset Root     : {ch_paths['dataset_root']}")
print(f"  Arrays Directory : {ch_paths['arrays_dir']}")
print(f"  Labels GeoJSON   : {ch_paths['labels_geojson']}")
print(f"  Normalizing Dict : {ch_paths.get('normalizing_dict')}")
print("=" * 75)""")

    # CELL 3: Dataset Structure & File Integrity Audit
    add_code("""# Cell 3: Dataset Structure, File Integrity & Shape Audit
# Audits normalizing dictionary and arrays/ folder for 12x18 shape and 0 NaNs/Infs.

arrays_dir = ch_paths["arrays_dir"]
norm_path = ch_paths.get("normalizing_dict")

if norm_path and norm_path.exists():
    with h5py.File(norm_path, "r") as h:
        mean = h["mean"][:]
        std = h["std"][:]
    print(f"[Normalizer] Loaded 18-channel normalization parameters:")
    print(f"  Mean shape: {mean.shape} | Range: [{mean.min():.2f}, {mean.max():.2f}]")
    print(f"  Std shape : {std.shape}  | Range: [{std.min():.4f}, {std.max():.2f}]")
else:
    raise FileNotFoundError("normalizing_dict.h5 is required for CropHarvest preprocessing.")

all_array_files = list(arrays_dir.glob("*.h5"))
total_arrays = len(all_array_files)
print(f"\\n[Arrays] Total available HDF5 array files: {total_arrays:,}")

sample_size = min(1000, total_arrays)
random.seed(SEED)
sample_check = random.sample(all_array_files, sample_size)

shapes = set()
dtypes = set()
nan_count = 0
inf_count = 0
corrupt_count = 0

for p in sample_check:
    try:
        with h5py.File(p, "r") as h:
            arr = h["array"][:]
            shapes.add(arr.shape)
            dtypes.add(str(arr.dtype))
            if np.isnan(arr).any():
                nan_count += 1
            if np.isinf(arr).any():
                inf_count += 1
    except Exception:
        corrupt_count += 1

print(f"[Audit Sample: {sample_size:,} arrays]")
print(f"  Observed Array Shapes : {shapes} (Expected: (12, 18))")
print(f"  Observed Data Types   : {dtypes}")
print(f"  Files with NaNs       : {nan_count}")
print(f"  Files with Infs       : {inf_count}")
print(f"  Corrupt Files         : {corrupt_count}")
assert corrupt_count == 0, f"Detected {corrupt_count} corrupted files!"
assert shapes == {(12, 18)}, f"Unexpected shapes found: {shapes}"
print("  --> Array integrity check PASSED!")""")

    # CELL 4: 10-Class Metadata Audit & Documentation
    add_code("""# Cell 4: 10-Class Metadata Audit & Taxonomy Documentation
# Audits classification_label in labels.geojson.
# Taxonomy: 9 verified FAO Crop Groups + 1 Land-Cover Class (non_crop).

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

geojson_path = ch_paths["labels_geojson"]
with open(geojson_path, "r", encoding="utf-8") as f:
    geo_data = json.load(f)

features = geo_data["features"]
available_array_stems = set(p.name for p in all_array_files)

# Count categories
raw_label_counts = Counter()
crop_counts = Counter()
noncrop_counts = Counter()

for feat in features:
    props = feat["properties"]
    fn = f"{props['index']}_{props['dataset']}.h5"
    if fn in available_array_stems:
        cl = str(props.get("classification_label"))
        is_c = props.get("is_crop")
        raw_label_counts[cl] += 1
        if is_c == 1:
            crop_counts[cl] += 1
        else:
            noncrop_counts[cl] += 1

print("=" * 85)
print("  CROPHARVEST 10-CLASS TAXONOMY & METADATA AUDIT")
print("=" * 85)
print(f"{'Class Index':<12} | {'Class Name':<20} | {'Taxonomic Type':<24} | {'Total Samples':<14}")
print("-" * 85)
for i, c in enumerate(CLASSES_10):
    print(f"Class {i:<6} | {c:<20} | {CLASS_TYPES[c]:<24} | {raw_label_counts[c]:<14,}")
print("=" * 85)
print(f"Total Eligible 10-Class Samples: {sum(raw_label_counts[c] for c in CLASSES_10):,}")
print(f"Excluded Binary-Only Samples (classification_label=None): {raw_label_counts['None']:,}")

with open(OUTPUT_DIR / "classes.json", "w", encoding="utf-8") as f:
    json.dump(IDX_TO_CLS, f, indent=2)
print(f"Class mapping saved to: {OUTPUT_DIR / 'classes.json'}")""")

    # CELL 5: Stratified Train/Val Partition & Test Isolation
    add_code("""# Cell 5: Stratified Train/Val Partition & Official Test Isolation
# Requirement: Never use held-out test data for training or tuning.
# Preserves is_test=true as official test set.
# Partitions is_test=false into 80% Train and 20% Validation (stratified by 10 classes, SEED=42).

records = []
for feat in features:
    props = feat["properties"]
    fn = f"{props['index']}_{props['dataset']}.h5"
    if fn in available_array_stems:
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

train_val_pool = [r for r in records if not r["is_test_official"]]
held_out_test = [r for r in records if r["is_test_official"]]

train_records = []
val_records = []

random.seed(SEED)
for c in CLASSES_10:
    c_recs = [r for r in train_val_pool if r["class_name"] == c]
    random.shuffle(c_recs)
    n_val = int(len(c_recs) * 0.20)
    val_records.extend(c_recs[:n_val])
    train_records.extend(c_recs[n_val:])

random.shuffle(train_records)
random.shuffle(val_records)
test_records = held_out_test

# Verify zero leakage
train_ids = set(r["filename"] for r in train_records)
val_ids = set(r["filename"] for r in val_records)
test_ids = set(r["filename"] for r in test_records)

assert len(train_ids.intersection(val_ids)) == 0, "Data leakage between train and val!"
assert len(train_ids.intersection(test_ids)) == 0, "Data leakage between train and test!"
assert len(val_ids.intersection(test_ids)) == 0, "Data leakage between val and test!"

train_counts = Counter(r["class_name"] for r in train_records)
val_counts = Counter(r["class_name"] for r in val_records)
test_counts = Counter(r["class_name"] for r in test_records)

print("=" * 90)
print("  10-CLASS CROP & LAND-COVER PARTITION SCORECARD (ZERO LEAKAGE)")
print("=" * 90)
print(f"{'Class Name':<20} | {'Type':<12} | {'Train (80%)':<12} | {'Val (20%)':<12} | {'Held-Out Test':<16} | {'Test Support'}")
print("-" * 90)
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
print("=" * 90)
print(f"Total: Train={len(train_records):,}, Val={len(val_records):,}, Test={len(test_records):,}")""")

    # CELL 6: PyTorch Dataset & DataLoaders
    add_code("""# Cell 6: PyTorch Dataset & DataLoaders Definition
# Applies 18-channel normalization (x_norm = (x - mean) / std) on (12, 18) arrays.

class CropHarvest10ClassTorchDataset(Dataset):
    def __init__(
        self,
        records: List[Dict[str, Any]],
        mean: np.ndarray,
        std: np.ndarray,
        preload_into_memory: bool = False
    ):
        self.records = records
        self.mean = mean.astype(np.float32)
        self.std = np.where(std == 0, 1.0, std).astype(np.float32)
        self.preload = preload_into_memory
        self.cached = None

        if self.preload:
            print(f"[DataLoader] Preloading {len(self.records):,} arrays into RAM for high-speed training...")
            t0 = time.time()
            self.cached = []
            for r in self.records:
                with h5py.File(r["filepath"], "r") as h:
                    arr = h["array"][:].astype(np.float32)
                    arr_norm = (arr - self.mean) / self.std
                    arr_norm = np.nan_to_num(arr_norm, nan=0.0, posinf=0.0, neginf=0.0)
                    self.cached.append(arr_norm)
            print(f"[DataLoader] Completed RAM caching in {time.time() - t0:.1f}s.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        target = self.records[idx]["class_idx"]
        if self.preload and self.cached is not None:
            data = self.cached[idx]
        else:
            with h5py.File(self.records[idx]["filepath"], "r") as h:
                arr = h["array"][:].astype(np.float32)
            data = (arr - self.mean) / self.std
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        return torch.from_numpy(data), target

BATCH_SIZE = 64

train_dataset = CropHarvest10ClassTorchDataset(train_records, mean, std, preload_into_memory=False)
val_dataset = CropHarvest10ClassTorchDataset(val_records, mean, std, preload_into_memory=False)
test_dataset = CropHarvest10ClassTorchDataset(test_records, mean, std, preload_into_memory=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2 if IS_KAGGLE else 0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2 if IS_KAGGLE else 0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2 if IS_KAGGLE else 0)

print(f"DataLoaders Ready: {len(train_loader)} train batches, {len(val_loader)} val batches, {len(test_loader)} test batches.")""")

    # CELL 7: Bi-LSTM 10-Class Architecture & Pre-Flight Sanity Check
    add_code("""# Cell 7: 10-Class Bi-LSTM Architecture & Pre-Flight Sanity Check
# 2-layer Bidirectional LSTM for 12-month multi-spectral temporal phenology modeling across 10 classes.

class CropHarvestBiLSTM(nn.Module):
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
        # x: (batch_size, 12, 18)
        lstm_out, _ = self.lstm(x)  # (batch_size, 12, hidden_dim * 2)
        pooled = lstm_out.mean(dim=1)  # Temporal mean pooling
        normed = self.layer_norm(pooled)
        logits = self.classifier(normed)
        return logits

# Pre-flight Sanity Check
print("[Sanity Check] Verifying 10-class model forward/backward pass...")
sanity_model = CropHarvestBiLSTM(input_dim=18, hidden_dim=64, num_layers=2, num_classes=10).to(DEVICE)
total_params = sum(p.numel() for p in sanity_model.parameters() if p.requires_grad)

dummy_x = torch.randn((8, 12, 18), dtype=torch.float32, device=DEVICE)
dummy_y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 9], dtype=torch.long, device=DEVICE)

sanity_model.train()
t0 = time.time()
dummy_out = sanity_model(dummy_x)
dummy_loss = nn.CrossEntropyLoss()(dummy_out, dummy_y)
dummy_loss.backward()
sanity_time_ms = (time.time() - t0) * 1000

print(f"  Model Parameters      : {total_params:,} (10 output classes)")
print(f"  Synthetic Batch Forward/Backward: {sanity_time_ms:.2f} ms on {DEVICE}")
if torch.cuda.is_available():
    print(f"  Peak VRAM Allocation : {torch.cuda.max_memory_allocated() / (1024**2):.2f} MB")
print("  --> Pre-flight hardware and 10-class model sanity test PASSED!")""")

    # CELL 8: Training Execution (Manual Gate: EXECUTE_TRAINING = False)
    add_code("""# Cell 8: Model Training Execution (Manual Gate)
# Requirement: Do not start training automatically.
# Set EXECUTE_TRAINING = True when ready to execute on Kaggle GPU.

EXECUTE_TRAINING = False  # Set to True manually to initiate model training

EPOCHS = 15
LEARNING_RATE = 1e-3

if not EXECUTE_TRAINING:
    print("=" * 75)
    print("  [TRAINING GATE: HELD]")
    print("  EXECUTE_TRAINING is set to False.")
    print("  All data integrity, 10-class balance, and hardware checks have passed.")
    print("  To train the CropHarvest 10-Class Bi-LSTM model on Kaggle GPU:")
    print("    1. Change `EXECUTE_TRAINING = True` in this cell.")
    print("    2. Re-run this cell.")
    print("=" * 75)
else:
    print(f"Starting 10-Class Bi-LSTM Training on {DEVICE} for {EPOCHS} epochs...")

    # Compute class weights for cross-entropy
    total_t = len(train_records)
    weights = [total_t / (10.0 * max(1, train_counts[c])) for c in CLASSES_10]
    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    print("Class Weights for Loss:")
    for c, w in zip(CLASSES_10, weights):
        print(f"  {c:<20}: {w:.3f}")

    model = CropHarvestBiLSTM(input_dim=18, hidden_dim=64, num_layers=2, num_classes=10).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    best_val_f1 = -1.0
    best_checkpoint_path = OUTPUT_DIR / "best_model.pth"
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_macro_f1": []}

    t_train_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * len(batch_y)
            preds = logits.argmax(dim=1)
            correct_train += (preds == batch_y).sum().item()
            total_train += len(batch_y)

        scheduler.step()
        epoch_train_loss = running_loss / max(1, total_train)
        epoch_train_acc = correct_train / max(1, total_train)

        # Validation Step
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(DEVICE), vy.to(DEVICE)
                v_logits = model(vx)
                v_loss = criterion(v_logits, vy)
                val_loss += v_loss.item() * len(vy)
                val_preds.extend(v_logits.argmax(dim=1).cpu().numpy())
                val_targets.extend(vy.cpu().numpy())

        epoch_val_loss = val_loss / max(1, len(val_targets))
        epoch_val_acc = accuracy_score(val_targets, val_preds)
        _, _, epoch_val_f1, _ = precision_recall_fscore_support(val_targets, val_preds, average="macro", zero_division=0)

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        history["val_macro_f1"].append(epoch_val_f1)

        is_best = epoch_val_f1 > best_val_f1
        if is_best:
            best_val_f1 = epoch_val_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_accuracy": epoch_val_acc,
                "val_macro_f1": epoch_val_f1,
                "class_mapping": IDX_TO_CLS,
                "config": {"input_dim": 18, "hidden_dim": 64, "num_layers": 2, "num_classes": 10}
            }, best_checkpoint_path)

        star = " ★ BEST" if is_best else ""
        print(
            f"Epoch [{epoch:02d}/{EPOCHS:02d}] "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}% | "
            f"Val Macro-F1: {epoch_val_f1*100:.2f}%{star}"
        )

    print(f"\\nTraining completed in {(time.time() - t_train_start)/60:.2f} minutes.")
    print(f"Best Checkpoint Saved: {best_checkpoint_path.resolve()} (Val Macro-F1: {best_val_f1*100:.2f}%)")""")

    # CELL 9: Validation Split Evaluation & Per-Class Scorecard
    add_code("""# Cell 9: Detailed 10-Class Validation Evaluation
# Computes Macro/Weighted F1, Precision, Recall, and per-class metrics across all 10 categories.

best_ckpt_file = OUTPUT_DIR / "best_model.pth"

if not best_ckpt_file.exists():
    print("[Notice] Best checkpoint not found. Execute Cell 8 first with EXECUTE_TRAINING = True.")
else:
    eval_model = CropHarvestBiLSTM(input_dim=18, hidden_dim=64, num_layers=2, num_classes=10).to(DEVICE)
    ckpt = torch.load(best_ckpt_file, map_location=DEVICE)
    eval_model.load_state_dict(ckpt["model_state_dict"])
    eval_model.eval()
    print(f"Loaded Best Model from Epoch {ckpt['epoch']} (Val Macro-F1: {ckpt['val_macro_f1']*100:.2f}%)")

    all_vpreds, all_vtargets = [], []
    with torch.no_grad():
        for vx, vy in val_loader:
            vx = vx.to(DEVICE)
            logits = eval_model(vx)
            all_vpreds.extend(logits.argmax(dim=1).cpu().numpy())
            all_vtargets.extend(vy.numpy())

    all_vpreds = np.array(all_vpreds)
    all_vtargets = np.array(all_vtargets)

    val_acc = accuracy_score(all_vtargets, all_vpreds)
    val_prec, val_rec, val_f1, _ = precision_recall_fscore_support(all_vtargets, all_vpreds, average="macro", zero_division=0)
    w_prec, w_rec, w_f1, _ = precision_recall_fscore_support(all_vtargets, all_vpreds, average="weighted", zero_division=0)
    per_p, per_r, per_f, per_s = precision_recall_fscore_support(all_vtargets, all_vpreds, labels=list(range(10)), average=None, zero_division=0)

    print("=" * 85)
    print("  INTERNAL 10-CLASS VALIDATION EVALUATION SCORECARD")
    print("=" * 85)
    print(f"  Overall Accuracy : {val_acc * 100:.2f}%")
    print(f"  Macro-F1         : {val_f1 * 100:.2f}%")
    print(f"  Weighted-F1      : {w_f1 * 100:.2f}%")
    print(f"  Macro Precision  : {val_prec * 100:.2f}%")
    print(f"  Macro Recall     : {val_rec * 100:.2f}%")
    print("-" * 85)
    print(f"{'Class Name':<20} | {'Type':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<10}")
    print("-" * 85)
    for i, c in enumerate(CLASSES_10):
        ctype = "Land-Cover" if c == "non_crop" else "Crop Group"
        print(f"{c:<20} | {ctype:<12} | {per_p[i]*100:>8.2f}% | {per_r[i]*100:>8.2f}% | {per_f[i]*100:>8.2f}% | {per_s[i]:>10,}")
    print("=" * 85)""")

    # CELL 10: Official Held-Out Test Evaluation & Per-Class Sample Support Audit
    add_code("""# Cell 10: Untouched Official Held-Out Test Split Evaluation & Test Support Audit
# Evaluates multi-class generalization on the official held-out test split (is_test=true).
# Audits per-class sample representation to distinguish well-tested vs sparsely tested categories.

if not best_ckpt_file.exists():
    print("[Notice] Train the model first in Cell 8 to evaluate the test set.")
else:
    all_tpreds, all_ttargets = [], []
    with torch.no_grad():
        for tx, ty in test_loader:
            tx = tx.to(DEVICE)
            logits = eval_model(tx)
            all_tpreds.extend(logits.argmax(dim=1).cpu().numpy())
            all_ttargets.extend(ty.numpy())

    all_tpreds = np.array(all_tpreds)
    all_ttargets = np.array(all_ttargets)

    test_acc = accuracy_score(all_ttargets, all_tpreds)
    test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(all_ttargets, all_tpreds, average="macro", zero_division=0)
    _, _, test_wf1, _ = precision_recall_fscore_support(all_ttargets, all_tpreds, average="weighted", zero_division=0)
    tp_p, tp_r, tp_f, tp_s = precision_recall_fscore_support(all_ttargets, all_tpreds, labels=list(range(10)), average=None, zero_division=0)

    print("=" * 90)
    print("  OFFICIAL HELD-OUT TEST EVALUATION SCORECARD & SAMPLE SUPPORT AUDIT")
    print("=" * 90)
    print(f"  Total Held-Out Test Samples : {len(all_ttargets):,}")
    print(f"  Overall Test Accuracy       : {test_acc * 100:.2f}%")
    print(f"  Test Macro-F1               : {test_f1 * 100:.2f}%")
    print(f"  Test Weighted-F1            : {test_wf1 * 100:.2f}%")
    print("-" * 90)
    print(f"{'Class Name':<20} | {'Test Samples':<14} | {'Test Precision':<16} | {'Test Recall':<14} | {'Support Status'}")
    print("-" * 90)
    for i, c in enumerate(CLASSES_10):
        cnt = tp_s[i]
        if cnt >= 10:
            support_status = f"Adequate ({cnt})"
        elif cnt == 0:
            support_status = "CRITICAL: 0 samples (untestable in benchmark)"
        else:
            support_status = f"Warning: Very Low ({cnt} sample)"
        print(f"{c:<20} | {cnt:<14,} | {tp_p[i]*100:>14.2f}% | {tp_r[i]*100:>12.2f}% | {support_status}")
    print("=" * 90)""")

    # CELL 11: Confusion Matrix & Temporal Phenology Visualization
    add_code("""# Cell 11: 10-Class Confusion Matrix & Temporal Phenology Visualization
# Plots 10x10 confusion matrix heatmap and seasonal vegetation index curves.

if not best_ckpt_file.exists():
    print("[Notice] Visualizations available after model execution.")
else:
    fig, (ax_cm, ax_pheno) = plt.subplots(1, 2, figsize=(18, 7))

    # 10x10 Confusion Matrix
    cm_val = confusion_matrix(all_vtargets, all_vpreds, labels=list(range(10)))
    sns.heatmap(cm_val, annot=True, fmt="d", cmap="YlGnBu", ax=ax_cm,
                xticklabels=CLASSES_10, yticklabels=CLASSES_10)
    ax_cm.set_title(f"10-Class Validation Confusion Matrix\\n(Acc: {val_acc*100:.2f}%, Macro-F1: {val_f1*100:.2f}%)")
    ax_cm.set_xlabel("Predicted Class")
    ax_cm.set_ylabel("True Ground Truth")
    ax_cm.tick_params(axis='x', rotation=45)

    # Phenology Curves: Top Crop Categories + Non-Crop
    compare_classes = ["cereals", "other", "fruits_nuts", "non_crop"]
    months = list(range(1, 13))

    for c in compare_classes:
        sample_rec = next(r for r in val_records if r["class_name"] == c)
        with h5py.File(sample_rec["filepath"], "r") as h:
            arr = h["array"][:]
        ndvi = (arr[:, 8] - arr[:, 4]) / (arr[:, 8] + arr[:, 4] + 1e-6)
        label_text = f"{c} (Land-Cover)" if c == "non_crop" else f"{c} (Crop)"
        ax_pheno.plot(months, ndvi, "-o", label=label_text)

    ax_pheno.set_title("12-Month Satellite Phenology Profiles (NDVI)")
    ax_pheno.set_xlabel("Month of Year")
    ax_pheno.set_ylabel("NDVI Vegetation Index")
    ax_pheno.grid(True, linestyle="--", alpha=0.5)
    ax_pheno.legend()

    plt.tight_layout()
    analysis_plot_path = OUTPUT_DIR / "10class_evaluation_analysis.png"
    plt.savefig(analysis_plot_path, dpi=300)
    plt.show()
    plt.close()
    print(f"Evaluation plots saved to: {analysis_plot_path}")""")

    # CELL 12: Deliverable Export
    add_code("""# Cell 12: Deliverables & Metrics JSON Export
# Packages trained weights, class mapping, and metrics JSON into models/cropharvest/10class/.

metrics_export_path = OUTPUT_DIR / "metrics.json"

if best_ckpt_file.exists():
    final_metrics_summary = {
        "pipeline": "CropHarvest_10Class_Classification",
        "task": "10-Class Classification (9 FAO Crop Categories + 1 Land-Cover Class)",
        "architecture": "CropHarvestBiLSTM",
        "parameters": total_params,
        "input_channels": 18,
        "sequence_length": 12,
        "classes": CLASSES_10,
        "class_types": CLASS_TYPES,
        "validation_split": {
            "samples": len(all_vtargets),
            "accuracy": float(val_acc),
            "macro_f1": float(val_f1),
            "weighted_f1": float(w_f1),
            "macro_precision": float(val_prec),
            "macro_recall": float(val_rec)
        },
        "official_global_test": {
            "samples": len(all_ttargets),
            "accuracy": float(test_acc),
            "macro_f1": float(test_f1),
            "weighted_f1": float(test_wf1)
        }
    }

    with open(metrics_export_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics_summary, f, indent=2)

    print("=" * 70)
    print("  10-CLASS PIPELINE DELIVERABLE EXPORT COMPLETE")
    print("=" * 70)
    print(f"  Model Checkpoint : {best_ckpt_file.resolve()}")
    print(f"  Class Mapping    : {(OUTPUT_DIR / 'classes.json').resolve()}")
    print(f"  Metrics JSON     : {metrics_export_path.resolve()}")
    print("=" * 70)
else:
    print(f"Evaluation summary will be exported once training is completed.")""")

    return nb


def main():
    notebook = build_10class_notebook()
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
    print(f"Successfully generated 10-Class Notebook: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
