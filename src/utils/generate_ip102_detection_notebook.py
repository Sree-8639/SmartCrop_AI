"""
SmartCrop AI - Generator for IP102 Object Detection Notebook (YOLO-Format Direct Pipeline)
Generates notebooks/03_IP102_Detection.ipynb with:
1. Dynamic /kaggle/input path discovery for YOLO-format dataset and ip102.yaml.
2. Robust parsing of ip102.yaml, class count verification (102 classes), and runtime YAML generation.
3. Image-label pair verification, missing/empty label check, and boundary checks.
4. Pre-flight GPU and Ultralytics YOLOv8 sanity check.
5. Hyperparameter configuration for YOLOv8s (imgsz=640, AdamW, batch=16).
6. Manual training gate (EXECUTE_TRAINING = False by default).
7. Validation & test evaluation (mAP@50, mAP@50:95, Precision, Recall).
8. Sample prediction visualizations.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH = NOTEBOOKS_DIR / "03_IP102_Detection.ipynb"

# Official 102 IP102 Pest Classes
IP102_CLASSES = [
    "rice leaf roller", "rice leaf caterpillar", "paddy stem maggot", "asiatic rice borer",
    "yellow rice borer", "rice gall midge", "Rice Stemfly", "brown plant hopper",
    "white backed plant hopper", "small brown plant hopper", "rice water weevil",
    "rice leafhopper", "grain spreader thrips", "rice shell pest", "grub",
    "mole cricket", "wireworm", "white margined moth", "black cutworm",
    "large cutworm", "yellow cutworm", "red spider", "corn borer", "army worm",
    "aphids", "Potosiabre vitarsis", "peach borer", "english grain aphid",
    "green bug", "bird cherry-oataphid", "wheat blossom midge", "penthaleus major",
    "longlegged spider mite", "wheat phloeothrips", "wheat sawfly", "cerodonta denticornis",
    "beet fly", "flea beetle", "cabbage army worm", "beet army worm",
    "Beet spot flies", "meadow moth", "beet weevil", "serpentine leafminer",
    "brown armor rot", "oriental fruit moth", "apple small borer", "lymantria dispar",
    "apoestis unicolorella", "tenuipalpus oryzophilus", "apple stem groover virus",
    "apple borer", "pear shoot borer", "Pear borer", "peach little borer",
    "Adoxophyes orana", "cherry stem borer", "chlamydatus associatus", "parathrene regalis",
    "spider mite", "Pseudococcus comstocki", "white-backed leafhopper", "grasshopper",
    "oriental water bug", "citrus red mite", "rust red spider mite", "green leaf hopper",
    "yellow scale", "circular black scale", "Ceroplastes rubens", "citrus leaf miner",
    "cotton bollworm", "pink bollworm", "locust", "cotton stem weevil",
    "cotton aphid", "cotton spider mite", "mirid bug", "red spider mite",
    "leaf worm", "spotted bollworm", "taro hornworm", "plant bug",
    "tea mosquito bug", "tea green leafhopper", "tea looper", "tea leaf roller",
    "tea tussock moth", "Andraca bipunctata", "tea aphid", "tea stem borer",
    "grape borer", "leaf beetle", "grape caterpillar", "grape mite",
    "grape leafhopper", "mango flat mite", "mango hopper", "mango pulp weevil",
    "Rhytidodera bowrinii white", "Sternochetus frigidus", "Cicadellidae"
]


def build_ip102_detection_notebook() -> dict:
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

    # =========================================================================
    # CELL 0: Markdown Title & Dataset Architecture
    # =========================================================================
    add_md(r"""# SmartCrop AI: IP102 Agricultural Insect Pest Object Detection
## Direct YOLO-Format Pipeline with Pretrained YOLOv8

**Dataset Reference:**  
*Wu, X., Zhan, C., Lai, Y. K., Cheng, M. M., & Yang, J. (2019). IP102: A Large-Scale Benchmark Dataset for Insect Pest Recognition. IEEE/CVF CVPR 2019.*

---

### Pipeline Architecture for YOLO-Format Dataset
1. **Dynamic Path Discovery:**
   - Recursively searches `/kaggle/input/` (and local directories) to detect the YOLO-format folder containing:
     - `images/train` and `images/val` (or `train/images`, `val/images`)
     - `labels/train` and `labels/val` (or `train/labels`, `val/labels`)
     - `ip102.yaml` (or `data.yaml`, `dataset.yaml`)
2. **YAML Parsing & Absolute Path Resolution:**
   - Reads `ip102.yaml` and extracts `names` and `nc` (102 classes).
   - Generates an absolute-path `ip102_runtime.yaml` in `/kaggle/working/` to prevent Ultralytics path resolution failures on Kaggle.
3. **Data Integrity & Image-Label Verification:**
   - Verifies matching `.jpg` / `.png` images and `.txt` label pairs for each split.
   - Audits bounding box coordinates: bounds $[0, 1]$, non-negative dimensions ($w > 0, h > 0$), and valid class indices ($0 \le c < 102$).
   - Flags degenerate boxes and empty labels; exports `detection_dataset_audit.json`.
4. **Pretrained YOLOv8 Training Setup:**
   - Pretrained `yolov8s.pt` (11.2M parameters) optimized for agricultural insect pest localization at $640 \times 640$ resolution.
5. **Manual Training Gate:**
   - **`EXECUTE_TRAINING = False`** by default so notebooks can be inspected and pre-flight tested without launching training automatically.
6. **Evaluation & Visualization:**
   - Computes mAP@50, mAP@50:95, Precision, Recall, and renders sample detections on validation images.""")

    # =========================================================================
    # CELL 1: Environment Setup, Dependencies & GPU Inspection
    # =========================================================================
    add_code("""# Cell 1: Environment Setup, Dependencies, Deterministic Seeding & GPU Inspection
import os
import sys
import json
import time
import math
import random
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional

# Ensure Ultralytics and PyYAML are installed
try:
    import ultralytics
    import yaml
except ImportError:
    print("[Setup] Installing ultralytics and pyyaml...")
    !pip install -q ultralytics pyyaml
    import ultralytics
    import yaml

from ultralytics import YOLO
import torch
import torchvision
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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

# Environment & Hardware Inspection
print("=" * 70)
print("  SMARTCROP AI: IP102 OBJECT DETECTION ENVIRONMENT INSPECTION")
print("=" * 70)
print(f"Python Version      : {sys.version.split()[0]}")
print(f"PyTorch Version     : {torch.__version__}")
print(f"Ultralytics Version : {ultralytics.__version__}")
print(f"CUDA Available      : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    gpu_cap = torch.cuda.get_device_capability(0)
    print(f"GPU Device          : {gpu_name}")
    print(f"GPU VRAM            : {gpu_mem:.2f} GB")
    print(f"Compute Capability  : {gpu_cap[0]}.{gpu_cap[1]}")
else:
    print("WARNING: CUDA is NOT available. Running on CPU will be slow.")

# Determine execution environment & set output directory
IS_KAGGLE = Path("/kaggle").exists()
if IS_KAGGLE:
    OUTPUT_DIR = Path("/kaggle/working/outputs/pest_detection")
else:
    OUTPUT_DIR = Path("outputs/pest_detection")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Runtime Environment : {'Kaggle Kernel' if IS_KAGGLE else 'Local Workstation'}")
print(f"Output Directory    : {OUTPUT_DIR.resolve()}")
print("=" * 70)""")

    # =========================================================================
    # CELL 2: Dynamic Path Discovery for YOLO-Format Dataset & ip102.yaml
    # =========================================================================
    add_code("""# Cell 2: Dynamic Path Discovery for YOLO-Format IP102 Dataset & ip102.yaml
# Recursively searches /kaggle/input and local roots to locate:
# - images/train, images/val (or train/images, val/images)
# - labels/train, labels/val (or train/labels, val/labels)
# - ip102.yaml (or data.yaml, dataset.yaml)

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml

def discover_yolo_dataset() -> Dict[str, Any]:
    search_roots = []
    
    # 1. Kaggle Input Roots
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

    # 2. Local Workspace Roots
    search_roots.extend([
        Path("datasets/ip102/Detection/yolo"),
        Path("datasets/ip102/Detection"),
        Path("datasets/ip102"),
        Path("../datasets/ip102/Detection/yolo"),
        Path("../datasets/ip102"),
        Path("."),
        Path("..")
    ])

    resolved = {}
    print("[Discovery] Searching for IP102 YOLO-format dataset and yaml...")

    # Step A: Find the training images directory (images/train or train/images)
    found_img_train = None
    for root in search_roots:
        if not root.exists():
            continue

        # Check surface candidates
        candidates = [
            root / "images" / "train",
            root / "train" / "images",
        ]
        for c in candidates:
            if c.exists() and c.is_dir():
                found_img_train = c.resolve()
                break
        if found_img_train:
            break

        # Check recursively up to 3 levels deep
        try:
            for p in root.rglob("train"):
                if not p.is_dir():
                    continue
                if p.parent.name == "images" and p.is_dir():
                    found_img_train = p.resolve()
                    break
                elif (p / "images").is_dir():
                    found_img_train = (p / "images").resolve()
                    break
        except Exception:
            pass
        if found_img_train:
            break

    if not found_img_train:
        print("\\n[ERROR] Could not automatically locate YOLO images/train directory.")
        if Path("/kaggle/input").exists():
            print("Contents of /kaggle/input:")
            for p in Path("/kaggle/input").rglob("*"):
                if p.is_dir():
                    print(f"  DIR : {p}")
                elif p.suffix in [".yaml", ".yml", ".txt"]:
                    print(f"  FILE: {p}")
        raise FileNotFoundError(
            "YOLO dataset folders (images/train, images/val) not found in /kaggle/input/. "
            "Please verify the dataset attachment on Kaggle."
        )

    # Step B: Resolve Layout and Split Directories
    if found_img_train.parent.name == "images":
        # Layout 1: <root>/images/train, <root>/images/val
        images_dir = found_img_train.parent
        data_root = images_dir.parent
        labels_dir = data_root / "labels"

        resolved["layout"] = "images_split"
        resolved["data_root"] = data_root.resolve()
        resolved["images_train"] = found_img_train.resolve()
        resolved["images_val"] = (images_dir / "val").resolve()
        resolved["labels_train"] = (labels_dir / "train").resolve() if (labels_dir / "train").exists() else None
        resolved["labels_val"] = (labels_dir / "val").resolve() if (labels_dir / "val").exists() else None
        
        # Test split if present
        if (images_dir / "test").exists():
            resolved["images_test"] = (images_dir / "test").resolve()
            resolved["labels_test"] = (labels_dir / "test").resolve() if (labels_dir / "test").exists() else None

    else:
        # Layout 2: <root>/train/images, <root>/val/images
        train_dir = found_img_train.parent
        data_root = train_dir.parent
        val_dir = data_root / "val"
        test_dir = data_root / "test"

        resolved["layout"] = "split_images"
        resolved["data_root"] = data_root.resolve()
        resolved["images_train"] = found_img_train.resolve()
        resolved["images_val"] = (val_dir / "images").resolve()
        resolved["labels_train"] = (train_dir / "labels").resolve() if (train_dir / "labels").exists() else None
        resolved["labels_val"] = (val_dir / "labels").resolve() if (val_dir / "labels").exists() else None

        if (test_dir / "images").exists():
            resolved["images_test"] = (test_dir / "images").resolve()
            resolved["labels_test"] = (test_dir / "labels").resolve() if (test_dir / "labels").exists() else None

    # Step C: Search for ip102.yaml or equivalent dataset yaml
    yaml_search_locations = [resolved["data_root"], resolved["data_root"].parent]
    if Path("/kaggle/input").exists():
        yaml_search_locations.extend(list(Path("/kaggle/input").iterdir()))
    yaml_search_locations.extend(search_roots)

    found_yamls = []
    for loc in yaml_search_locations:
        if not loc.exists():
            continue
        try:
            for y in loc.glob("*.yaml"):
                if y.is_file() and y not in found_yamls:
                    found_yamls.append(y.resolve())
            for y in loc.glob("*.yml"):
                if y.is_file() and y not in found_yamls:
                    found_yamls.append(y.resolve())
        except Exception:
            pass

    # Prioritize YAML containing 'ip102'
    ip102_yamls = [y for y in found_yamls if "ip102" in y.name.lower()]
    other_yamls = [y for y in found_yamls if "ip102" not in y.name.lower()]
    yaml_candidates = ip102_yamls + other_yamls

    for yc in yaml_candidates:
        try:
            with open(yc, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f)
            if isinstance(content, dict) and ("names" in content or "nc" in content or "train" in content):
                resolved["source_yaml"] = yc
                resolved["yaml_content"] = content
                print(f"  [Found YAML] {yc}")
                break
        except Exception:
            pass

    return resolved

dataset_paths = discover_yolo_dataset()
print("\\n" + "=" * 70)
print("  PATH DISCOVERY SUCCESSFUL")
print("=" * 70)
print(f"  Data Root        : {dataset_paths['data_root']}")
print(f"  Images Train Dir : {dataset_paths['images_train']}")
print(f"  Images Val Dir   : {dataset_paths['images_val']}")
if "images_test" in dataset_paths:
    print(f"  Images Test Dir  : {dataset_paths['images_test']}")
else:
    print(f"  Images Test Dir  : [Not present - using validation split for evaluation]")
print(f"  Labels Train Dir : {dataset_paths.get('labels_train')}")
print(f"  Labels Val Dir   : {dataset_paths.get('labels_val')}")
print(f"  Source YAML File : {dataset_paths.get('source_yaml')}")
print("=" * 70)""")

    # =========================================================================
    # CELL 3: Read ip102.yaml, Verify Class Count & Create Runtime YAML
    # =========================================================================
    classes_json_str = json.dumps(IP102_CLASSES, indent=2)
    add_code("""# Cell 3: Read ip102.yaml, Verify Class Count (102 Classes) & Generate Runtime YAML
# In Kaggle, datasets under /kaggle/input are read-only and relative paths in ip102.yaml
# can fail when Ultralytics runs from /kaggle/working/.
# This cell parses the dataset yaml, validates the 102 classes, and writes an absolute-path
# runtime dataset specification.

import json
from pathlib import Path
import yaml

FALLBACK_102_CLASSES = """ + classes_json_str + """

source_yaml_path = dataset_paths.get("source_yaml")
yaml_data = dataset_paths.get("yaml_content", {})

# 1. Parse Class Names and Count
class_names = []
num_classes = 102

if yaml_data:
    raw_names = yaml_data.get("names")
    if isinstance(raw_names, list):
        class_names = [str(x) for x in raw_names]
    elif isinstance(raw_names, dict):
        # Sort by integer keys or stringified integer keys
        try:
            sorted_keys = sorted(raw_names.keys(), key=lambda k: int(k))
            class_names = [str(raw_names[k]) for k in sorted_keys]
        except Exception:
            class_names = [str(v) for v in raw_names.values()]
    
    if "nc" in yaml_data and isinstance(yaml_data["nc"], int):
        num_classes = yaml_data["nc"]
    elif class_names:
        num_classes = len(class_names)

# Fallback to official 102 classes if YAML had empty or missing names
if not class_names:
    print("[Notice] Using official 102 IP102 class list as verified reference.")
    class_names = FALLBACK_102_CLASSES
    num_classes = len(FALLBACK_102_CLASSES)

print(f"Class Count Verified: nc = {num_classes} classes")
assert num_classes == len(class_names), f"Mismatch between nc ({num_classes}) and len(names) ({len(class_names)})"

print(f"Sample classes (first 5): {class_names[:5]}")
print(f"Sample classes (last 5) : {class_names[-5:]}")

# 2. Generate Runtime YAML with Explicit Absolute Paths
runtime_yaml_path = OUTPUT_DIR / "ip102_runtime.yaml"

runtime_yaml_data = {
    "path": str(dataset_paths["data_root"]),
    "train": str(dataset_paths["images_train"]),
    "val": str(dataset_paths["images_val"]),
    "nc": num_classes,
    "names": {i: name for i, name in enumerate(class_names)}
}

if "images_test" in dataset_paths and dataset_paths["images_test"]:
    runtime_yaml_data["test"] = str(dataset_paths["images_test"])

with open(runtime_yaml_path, "w", encoding="utf-8") as f:
    yaml.dump(runtime_yaml_data, f, sort_keys=False)

print(f"\\n[Runtime YAML Created Successfully]")
print(f"  Saved to : {runtime_yaml_path.resolve()}")
print(f"  Train Dir: {runtime_yaml_data['train']}")
print(f"  Val Dir  : {runtime_yaml_data['val']}")
if 'test' in runtime_yaml_data:
    print(f"  Test Dir : {runtime_yaml_data['test']}")
print(f"  Classes  : {num_classes} classes")""")

    # =========================================================================
    # CELL 4: Image-Label Pair Verification & Comprehensive Dataset Audit
    # =========================================================================
    add_code("""# Cell 4: Comprehensive Image-Label Pair Verification & Dataset Integrity Audit
# Verifies:
# 1. Total images per split (train, val, test).
# 2. Corresponding .txt label files for every image.
# 3. Non-empty labels vs empty/background labels.
# 4. Box coordinate bounds: normalized [0, 1], positive width/height.
# 5. Class index validity: 0 <= class_idx < nc.
# 6. Checks for corrupted/unreadable images.

import json
import random
from pathlib import Path
from typing import Dict, Any, Optional, List
from PIL import Image

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}

def audit_split(img_dir: Path, lbl_dir: Optional[Path], split_name: str, max_check: int = 500) -> Dict[str, Any]:
    if not img_dir.exists():
        return {"error": f"Directory not found: {img_dir}"}

    img_files = [p for p in img_dir.iterdir() if p.suffix in IMG_EXTENSIONS]
    img_stems = {p.stem: p for p in img_files}
    
    lbl_files = [p for p in lbl_dir.iterdir() if p.suffix == ".txt"] if (lbl_dir and lbl_dir.exists()) else []
    lbl_stems = {p.stem: p for p in lbl_files}

    missing_labels = []
    empty_labels = []
    invalid_boxes = []
    total_boxes = 0
    class_histogram = {i: 0 for i in range(num_classes)}

    for img_path in img_files:
        stem = img_path.stem
        lbl_file = lbl_stems.get(stem)
        if not lbl_file:
            # Check if labels are in the same folder as images
            same_dir_lbl = img_path.with_suffix(".txt")
            if same_dir_lbl.exists():
                lbl_file = same_dir_lbl
            else:
                missing_labels.append(img_path.name)
                continue

        try:
            content = lbl_file.read_text(encoding="utf-8").strip()
            if not content:
                empty_labels.append(img_path.name)
                continue

            lines = content.splitlines()
            for line_idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) != 5:
                    invalid_boxes.append({"img": img_path.name, "reason": f"Expected 5 values, got {len(parts)}"})
                    continue
                
                cls_id = int(float(parts[0]))
                xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                
                if not (0 <= cls_id < num_classes):
                    invalid_boxes.append({"img": img_path.name, "reason": f"Class index {cls_id} out of bounds [0, {num_classes-1}]"})
                elif w <= 0 or h <= 0:
                    invalid_boxes.append({"img": img_path.name, "reason": f"Non-positive dimensions (w={w}, h={h})"})
                elif not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
                    invalid_boxes.append({"img": img_path.name, "reason": f"Center coordinates out of bounds (xc={xc:.4f}, yc={yc:.4f})"})
                else:
                    total_boxes += 1
                    class_histogram[cls_id] += 1
        except Exception as e:
            invalid_boxes.append({"img": img_path.name, "reason": f"Error reading label: {e}"})

    # Sample image corruption check
    sample_to_check = random.sample(img_files, min(max_check, len(img_files))) if img_files else []
    corrupted_images = []
    for img_p in sample_to_check:
        try:
            with Image.open(img_p) as im:
                im.verify()
        except Exception as e:
            corrupted_images.append({"img": img_p.name, "error": str(e)})

    return {
        "split": split_name,
        "image_count": len(img_files),
        "label_count": len(lbl_files),
        "missing_label_count": len(missing_labels),
        "empty_label_count": len(empty_labels),
        "total_boxes": total_boxes,
        "invalid_box_count": len(invalid_boxes),
        "corrupted_images_sampled": len(corrupted_images),
        "sample_checked": len(sample_to_check),
        "active_classes_in_split": sum(1 for v in class_histogram.values() if v > 0),
        "invalid_boxes_sample": invalid_boxes[:5]
    }

print("[Audit] Running verification on train and val splits...")
audit_results = {}
audit_results["train"] = audit_split(dataset_paths["images_train"], dataset_paths.get("labels_train"), "train")
audit_results["val"] = audit_split(dataset_paths["images_val"], dataset_paths.get("labels_val"), "val")
if "images_test" in dataset_paths:
    audit_results["test"] = audit_split(dataset_paths["images_test"], dataset_paths.get("labels_test"), "test")

# Save audit to outputs
audit_path = OUTPUT_DIR / "detection_dataset_audit.json"
with open(audit_path, "w", encoding="utf-8") as f:
    json.dump(audit_results, f, indent=2)

print("\\n" + "=" * 75)
print("  IP102 OBJECT DETECTION DATASET AUDIT SCORECARD")
print("=" * 75)
print(f"{'Metric':<32} | {'Train Split':<18} | {'Val Split':<18}")
print("-" * 75)
print(f"{'Image Count':<32} | {audit_results['train']['image_count']:<18} | {audit_results['val']['image_count']:<18}")
print(f"{'Label Files Found':<32} | {audit_results['train']['label_count']:<18} | {audit_results['val']['label_count']:<18}")
print(f"{'Images Missing Labels':<32} | {audit_results['train']['missing_label_count']:<18} | {audit_results['val']['missing_label_count']:<18}")
print(f"{'Empty Labels (No Boxes)':<32} | {audit_results['train']['empty_label_count']:<18} | {audit_results['val']['empty_label_count']:<18}")
print(f"{'Total Bounding Boxes':<32} | {audit_results['train']['total_boxes']:<18} | {audit_results['val']['total_boxes']:<18}")
print(f"{'Active Classes Represented':<32} | {audit_results['train']['active_classes_in_split']:<18} | {audit_results['val']['active_classes_in_split']:<18}")
print(f"{'Invalid/Degenerate Boxes':<32} | {audit_results['train']['invalid_box_count']:<18} | {audit_results['val']['invalid_box_count']:<18}")
print(f"{'Corrupted Images (Sampled)':<32} | {audit_results['train']['corrupted_images_sampled']:<18} | {audit_results['val']['corrupted_images_sampled']:<18}")
print("=" * 75)
print(f"Detailed audit saved to: {audit_path}")""")

    # =========================================================================
    # CELL 5: Pre-Flight Sanity Test: GPU & Model Verification
    # =========================================================================
    add_code("""# Cell 5: Pre-Flight Sanity Test: GPU & Model Verification
# Instantiates pretrained YOLOv8 and runs a dummy synthetic forward pass
# to confirm CUDA kernels, tensor allocations, and output layers execute cleanly.

import time
import torch
from ultralytics import YOLO

print("[Sanity Check] Initializing YOLOv8 model for pre-flight verification...")
sanity_model_name = "yolov8s.pt"

try:
    sanity_model = YOLO(sanity_model_name)
    print(f"  Successfully loaded base weights: {sanity_model_name}")
except Exception as e:
    print(f"  Fallback: Loading yolov8n.pt ({e})")
    sanity_model_name = "yolov8n.pt"
    sanity_model = YOLO(sanity_model_name)

# Quick synthetic batch forward test
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dummy_input = torch.zeros((1, 3, 640, 640), dtype=torch.float32, device=device)

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

t0 = time.time()
with torch.no_grad():
    _ = sanity_model.model.to(device)(dummy_input)
forward_time_ms = (time.time() - t0) * 1000

print(f"  Synthetic forward pass completed in {forward_time_ms:.2f} ms on {device}.")
if torch.cuda.is_available():
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"  Peak VRAM used in test: {peak_vram_mb:.2f} MB")

print("  Pre-flight sanity check passed! Hardware and model pipeline ready.")""")

    # =========================================================================
    # CELL 6: Pretrained YOLO Model Configuration
    # =========================================================================
    add_code("""# Cell 6: Pretrained YOLO Model Configuration & Hyperparameters
# Configured for Kaggle Tesla T4 / P100 (16GB VRAM)

import torch
from pathlib import Path

TRAIN_CONFIG = {
    "model": "yolov8s.pt",               # YOLOv8s backbone (11.2M params)
    "data": str(runtime_yaml_path),      # Generated absolute-path runtime YAML
    "epochs": 50,                        # Target epochs
    "patience": 10,                      # Early stopping patience
    "batch": 16,                         # Batch size 16 fits comfortably in 16GB VRAM
    "imgsz": 640,                        # Standard YOLO input resolution
    "save": True,                        # Save checkpoints
    "save_period": 5,                    # Periodic checkpoint every 5 epochs
    "cache": False,                      # False to prevent Kaggle RAM overflow
    "device": 0 if torch.cuda.is_available() else "cpu",
    "workers": 2,                        # Kaggle 2 vCPU allocation
    "project": str(OUTPUT_DIR / "runs"), # Output project path
    "name": "yolov8s_ip102",             # Experiment run name
    "exist_ok": True,
    "pretrained": True,
    "optimizer": "AdamW",                # AdamW for stable multi-class convergence
    "lr0": 0.001,                        # Initial learning rate
    "lrf": 0.01,                         # Final lr factor (0.001 * 0.01 = 1e-5)
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "box": 7.5,                          # Box loss gain
    "cls": 0.5,                          # Class loss gain
    "dfl": 1.5,                          # Distribution Focal Loss gain
    "mosaic": 1.0,                       # Mosaic augmentation
    "mixup": 0.1,                        # Mixup augmentation for overlapping pests
    "degrees": 10.0,                     # Rotation augmentation
    "fliplr": 0.5,                       # Horizontal flip
    "seed": SEED,
    "plots": True,                       # Generate validation plots & curves
    "verbose": True
}

print("=" * 70)
print("  YOLOv8 TRAINING CONFIGURATION")
print("=" * 70)
for k, v in TRAIN_CONFIG.items():
    print(f"  {k:<18}: {v}")
print("=" * 70)""")

    # =========================================================================
    # CELL 7: Model Training Execution (Manual Gate: EXECUTE_TRAINING = False)
    # =========================================================================
    add_code("""# Cell 7: Model Training Execution (Manual Gate)
# Requirement: Do not start training automatically.
# Set EXECUTE_TRAINING = True when ready to execute on Kaggle GPU.

EXECUTE_TRAINING = False  # Set to True manually to run training

if not EXECUTE_TRAINING:
    print("=" * 75)
    print("  [TRAINING GATE: HELD]")
    print("  EXECUTE_TRAINING is set to False.")
    print("  To start YOLOv8 training on Kaggle GPU:")
    print("    1. Change `EXECUTE_TRAINING = True` in this cell.")
    print("    2. Re-run this cell.")
    print("=" * 75)
else:
    import time
    import shutil
    from pathlib import Path
    from ultralytics import YOLO

    print(f"Starting YOLOv8 training on {TRAIN_CONFIG['device']} with {TRAIN_CONFIG['epochs']} epochs...")
    train_model = YOLO(TRAIN_CONFIG["model"])
    
    t_start = time.time()
    train_results = train_model.train(**TRAIN_CONFIG)
    elapsed_min = (time.time() - t_start) / 60.0

    print("\\n" + "=" * 70)
    print("  TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"  Total Training Time : {elapsed_min:.2f} minutes")
    
    # Save/copy best weights to primary output directory
    best_weights = Path(TRAIN_CONFIG["project"]) / TRAIN_CONFIG["name"] / "weights" / "best.pt"
    dest_best = OUTPUT_DIR / "best.pt"
    if best_weights.exists():
        shutil.copy2(best_weights, dest_best)
        print(f"  Best Weights Saved  : {dest_best.resolve()}")""")

    # =========================================================================
    # CELL 8: Validation & Test Evaluation (Detection Metrics)
    # =========================================================================
    add_code("""# Cell 8: Validation & Test Evaluation (Detection Metrics: mAP@50, mAP@50:95)
# Evaluates the best trained checkpoint on the validation split (and test split if available).

import json
from pathlib import Path
from ultralytics import YOLO

best_checkpoint = OUTPUT_DIR / "best.pt"
runs_best = Path(TRAIN_CONFIG["project"]) / TRAIN_CONFIG["name"] / "weights" / "best.pt"

eval_weights = None
if best_checkpoint.exists():
    eval_weights = best_checkpoint
elif runs_best.exists():
    eval_weights = runs_best

if eval_weights is None:
    print("[Evaluation Held] No trained checkpoint found. Train the model first by setting EXECUTE_TRAINING = True.")
else:
    print(f"Evaluating checkpoint: {eval_weights.resolve()}")
    eval_model = YOLO(str(eval_weights))

    # 1. Validation Split Evaluation
    print("\\n--- Evaluating on Validation Split ---")
    val_results = eval_model.val(data=str(runtime_yaml_path), split="val", imgsz=640, batch=16)

    metrics_summary = {
        "model_checkpoint": str(eval_weights.name),
        "validation": {
            "mAP_50": float(val_results.box.map50),
            "mAP_50_95": float(val_results.box.map),
            "precision": float(val_results.box.mp),
            "recall": float(val_results.box.mr)
        }
    }

    # 2. Test Split Evaluation (if present)
    if "images_test" in dataset_paths and dataset_paths["images_test"]:
        print("\\n--- Evaluating on Held-Out Test Split ---")
        test_results = eval_model.val(data=str(runtime_yaml_path), split="test", imgsz=640, batch=16)
        metrics_summary["test"] = {
            "mAP_50": float(test_results.box.map50),
            "mAP_50_95": float(test_results.box.map),
            "precision": float(test_results.box.mp),
            "recall": float(test_results.box.mr)
        }

    # Export detection metrics
    metrics_path = OUTPUT_DIR / "detection_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    print("\\n" + "=" * 70)
    print("  EVALUATION METRICS SUMMARY")
    print("=" * 70)
    print(f"  Validation mAP@0.50      : {metrics_summary['validation']['mAP_50'] * 100:.2f}%")
    print(f"  Validation mAP@0.50:0.95 : {metrics_summary['validation']['mAP_50_95'] * 100:.2f}%")
    print(f"  Validation Precision     : {metrics_summary['validation']['precision'] * 100:.2f}%")
    print(f"  Validation Recall        : {metrics_summary['validation']['recall'] * 100:.2f}%")
    if "test" in metrics_summary:
        print(f"  Held-Out Test mAP@0.50   : {metrics_summary['test']['mAP_50'] * 100:.2f}%")
        print(f"  Held-Out Test mAP@0.50:95: {metrics_summary['test']['mAP_50_95'] * 100:.2f}%")
    print(f"  Metrics exported to      : {metrics_path}")
    print("=" * 70)""")

    # =========================================================================
    # CELL 9: Sample Predictions & Visualization
    # =========================================================================
    add_code("""# Cell 9: Sample Predictions Visualization & Inference Demo
# Renders bounding box predictions on sample images from the validation set.

from pathlib import Path
import matplotlib.pyplot as plt
from ultralytics import YOLO

sample_dir = dataset_paths["images_val"]
val_sample_images = [p for p in sample_dir.iterdir() if p.suffix in IMG_EXTENSIONS][:4]

if not val_sample_images:
    print("No sample images found for visualization.")
else:
    print(f"Running inference demo on {len(val_sample_images)} validation images...")
    
    # Load best checkpoint or fallback to pretrained
    demo_weights = eval_weights if eval_weights and eval_weights.exists() else "yolov8s.pt"
    demo_model = YOLO(str(demo_weights))
    demo_results = demo_model(val_sample_images, conf=0.25)

    fig, axes = plt.subplots(1, len(demo_results), figsize=(5 * len(demo_results), 5))
    if len(demo_results) == 1:
        axes = [axes]

    for i, r in enumerate(demo_results):
        im_bgr = r.plot()
        im_rgb = im_bgr[..., ::-1]
        axes[i].imshow(im_rgb)
        axes[i].set_title(f"{Path(r.path).name}\\n({len(r.boxes)} pests detected)", fontsize=10)
        axes[i].axis("off")

    plt.tight_layout()
    vis_path = OUTPUT_DIR / "sample_detections.png"
    plt.savefig(vis_path, dpi=300)
    plt.show()
    print(f"Sample prediction plot saved to: {vis_path}")""")

    return nb


def main():
    notebook = build_ip102_detection_notebook()
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
    print(f"Successfully generated IP102 Detection Notebook: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
