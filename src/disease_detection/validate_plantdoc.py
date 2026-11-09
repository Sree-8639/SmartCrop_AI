"""
SmartCrop AI - PlantDoc Dataset Validation Script
Inspects directory structure, counts classes and images, detects corrupted or truncated
image files, detects exact and conflicting-label duplicates via MD5, checks class
split feasibility, records all included/excluded files with reasons, and generates reproducible 70/15/15 splits.
"""

import sys
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any
from collections import defaultdict
from PIL import Image, UnidentifiedImageError

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.split_generator import stratified_split_by_class, export_split_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Validate.PlantDoc")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_REQUIRED_SAMPLES = 5  # Minimum samples to guarantee train, val, and test assignment


def compute_file_md5(file_path: Path, chunk_size=65536) -> str:
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_plantdoc(dataset_dir: Path, output_dir: Path) -> Dict:
    """
    Validates the local PlantDoc dataset directory.
    Preserves original train/ and test/ folders, performs duplicate and conflicting-label detection,
    and produces an audited split manifest with every included and excluded file documented.
    """
    logger.info(f"Scanning PlantDoc dataset at: {dataset_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_dir.exists():
        logger.error(f"PlantDoc directory does not exist: {dataset_dir}")
        return {
            "status": "MISSING_DIRECTORY",
            "path": str(dataset_dir),
            "errors": [f"Directory {dataset_dir} does not exist."]
        }

    # Find all images once using case-insensitive check
    all_image_paths = [
        f for f in dataset_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
    ]

    if not all_image_paths:
        logger.warning(
            f"No image files found in {dataset_dir}. "
            "Please ensure PlantDoc dataset is placed inside this directory."
        )
        return {
            "status": "EMPTY_DATASET",
            "path": str(dataset_dir),
            "total_raw_images": 0,
            "classes_detected": 0,
            "errors": ["No valid image files found."]
        }

    corrupted_images: List[Dict[str, str]] = []
    resolutions: List[Tuple[int, int]] = []
    raw_class_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"train": 0, "test": 0, "total": 0})
    hash_to_instances: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    valid_records: List[Tuple[Path, str, str]] = []
    excluded_files_audit: List[Dict[str, Any]] = []

    for img_path in sorted(all_image_paths):
        parent_name = img_path.parent.name
        # Check if parent or grandparent is a split folder
        if parent_name.lower() in {"train", "test", "val", "validation", "images"}:
            split_tag = parent_name.lower()
            class_name = img_path.parent.parent.name
        else:
            grandparent = img_path.parent.parent.name
            if grandparent.lower() in {"train", "test", "val", "validation"}:
                split_tag = grandparent.lower()
            else:
                split_tag = "flat"
            class_name = parent_name

        class_name = class_name.strip()
        raw_class_counts[class_name]["total"] += 1
        if split_tag == "train":
            raw_class_counts[class_name]["train"] += 1
        elif split_tag == "test":
            raw_class_counts[class_name]["test"] += 1

        rel_path = str(img_path.relative_to(dataset_dir))

        # Integrity verification: verify headers and load full raster
        try:
            with Image.open(img_path) as img:
                img.verify()
            with Image.open(img_path) as img:
                resolutions.append(img.size)
                img.load()
        except Exception as e:
            corrupted_images.append({
                "file": rel_path,
                "class": class_name,
                "error": str(e)
            })
            excluded_files_audit.append({
                "file": rel_path,
                "class": class_name,
                "reason": "corrupted_or_truncated",
                "details": str(e)
            })
            continue

        # MD5 Hash
        try:
            fhash = compute_file_md5(img_path)
            hash_to_instances[fhash].append({
                "file": rel_path,
                "class": class_name,
                "split": split_tag
            })
            valid_records.append((img_path, class_name, fhash))
        except Exception as e:
            corrupted_images.append({
                "file": rel_path,
                "class": class_name,
                "error": f"Hashing error: {e}"
            })
            excluded_files_audit.append({
                "file": rel_path,
                "class": class_name,
                "reason": "hashing_failure",
                "details": str(e)
            })
            continue

    # Duplicate analysis
    exact_duplicates = []
    conflicting_label_duplicates = []
    conflicting_hashes: Set[str] = set()

    for fhash, instances in hash_to_instances.items():
        if len(instances) > 1:
            classes = set(inst["class"] for inst in instances)
            if len(classes) > 1:
                conflicting_label_duplicates.append({
                    "hash": fhash,
                    "instances": instances
                })
                conflicting_hashes.add(fhash)
            else:
                exact_duplicates.append({
                    "hash": fhash,
                    "instances": instances
                })

    # Log conflicting label duplicates into excluded audit
    for conf in conflicting_label_duplicates:
        for inst in conf["instances"]:
            excluded_files_audit.append({
                "file": inst["file"],
                "class": inst["class"],
                "reason": "conflicting_label_duplicate",
                "details": f"File shares identical MD5 {conf['hash']} with conflicting classes: {[x['class'] for x in conf['instances']]}"
            })

    # Build clean deduplicated class dictionary (excluding conflicting hashes)
    clean_class_to_files: Dict[str, List[Path]] = defaultdict(list)
    seen_hashes_clean: Set[str] = set()

    for p, c, h in valid_records:
        rel_p = str(p.relative_to(dataset_dir))
        if h in conflicting_hashes:
            continue
        if h in seen_hashes_clean:
            excluded_files_audit.append({
                "file": rel_p,
                "class": c,
                "reason": "exact_duplicate_pruned",
                "details": f"File is an exact intra-class or cross-split duplicate with MD5 {h}"
            })
            continue
        seen_hashes_clean.add(h)
        clean_class_to_files[c].append(p)

    # Class feasibility check (< MIN_REQUIRED_SAMPLES)
    underrepresented_classes = {}
    reliable_class_to_files = {}

    for cname, files in sorted(clean_class_to_files.items()):
        if len(files) < MIN_REQUIRED_SAMPLES:
            underrepresented_classes[cname] = len(files)
            for f in files:
                excluded_files_audit.append({
                    "file": str(f.relative_to(dataset_dir)),
                    "class": cname,
                    "reason": "underrepresented_class",
                    "details": f"Class '{cname}' has only {len(files)} sample(s) (< {MIN_REQUIRED_SAMPLES}) and cannot support a 3-way split."
                })
        else:
            reliable_class_to_files[cname] = files

    total_clean = sum(len(imgs) for imgs in reliable_class_to_files.values())
    total_classes = len(reliable_class_to_files)

    logger.info(f"Scan complete: Found {len(all_image_paths)} raw images, {total_clean} reliable clean images across {total_classes} classes.")
    if underrepresented_classes:
        logger.warning(f"Excluded {len(underrepresented_classes)} underrepresented class(es): {underrepresented_classes}")
    if conflicting_label_duplicates:
        logger.warning(f"Detected {len(conflicting_label_duplicates)} conflicting-label duplicate image(s)!")

    # Generate reproducible 70/15/15 split manifest with complete audit
    manifest_path = output_dir / "plantdoc_split_manifest.json"
    if total_clean > 0:
        class_rel_paths = {
            cls: [str(p.relative_to(dataset_dir)) for p in imgs]
            for cls, imgs in reliable_class_to_files.items()
        }
        train_set, val_set, test_set = stratified_split_by_class(class_rel_paths, 0.70, 0.15, 0.15, seed=42)
        export_split_manifest(train_set, val_set, test_set, manifest_path, excluded_files=excluded_files_audit)

    # Resolution statistics
    res_summary = {}
    if resolutions:
        widths, heights = zip(*resolutions)
        res_summary = {
            "min_resolution": [int(min(widths)), int(min(heights))],
            "max_resolution": [int(max(widths)), int(max(heights))],
            "mean_resolution": [int(sum(widths) / len(widths)), int(sum(heights) / len(heights))]
        }

    report = {
        "dataset_name": "PlantDoc",
        "dataset_path": str(dataset_dir),
        "status": "VALIDATED" if total_clean > 0 else "VALIDATION_FAILED",
        "total_raw_images": len(all_image_paths),
        "total_classes_discovered": len(raw_class_counts),
        "reliable_classes_count": total_classes,
        "reliable_clean_images": total_clean,
        "corrupted_images_count": len(corrupted_images),
        "exact_duplicates_count": len(exact_duplicates),
        "conflicting_label_duplicates_count": len(conflicting_label_duplicates),
        "underrepresented_classes": underrepresented_classes,
        "total_excluded_files": len(excluded_files_audit),
        "split_summary": {
            "train_count": len(train_set) if total_clean > 0 else 0,
            "train_pct": round(len(train_set) / total_clean * 100, 2) if total_clean > 0 else 0,
            "val_count": len(val_set) if total_clean > 0 else 0,
            "val_pct": round(len(val_set) / total_clean * 100, 2) if total_clean > 0 else 0,
            "test_count": len(test_set) if total_clean > 0 else 0,
            "test_pct": round(len(test_set) / total_clean * 100, 2) if total_clean > 0 else 0
        },
        "resolution_summary": res_summary,
        "raw_class_distribution": dict(raw_class_counts),
        "clean_class_distribution": {c: len(f) for c, f in reliable_class_to_files.items()},
        "split_manifest_saved": str(manifest_path) if total_clean > 0 else None,
        "warnings": [
            f"Underrepresented class '{c}' had only {cnt} sample(s) (< {MIN_REQUIRED_SAMPLES}) and was excluded from 3-way split."
            for c, cnt in underrepresented_classes.items()
        ] + [
            f"Found {len(conflicting_label_duplicates)} conflicting-label duplicate images across upstream train and test partitions; excluded to prevent corrupted ground-truth."
        ] if conflicting_label_duplicates else []
    }

    report_path = output_dir / "plantdoc_validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"PlantDoc Validation report saved to: {report_path}")

    return report


if __name__ == "__main__":
    plantdoc_path = PROJECT_ROOT / "datasets" / "plantdoc"
    outputs_path = PROJECT_ROOT / "outputs"
    res = validate_plantdoc(plantdoc_path, outputs_path)
    print("\nPlantDoc Validation Summary:")
    print(f"Total Raw Images: {res.get('total_raw_images')}")
    print(f"Reliable Clean Images: {res.get('reliable_clean_images')}")
    print(f"Reliable Classes: {res.get('reliable_classes_count')}")
    print(f"Total Excluded Files (with logged reasons): {res.get('total_excluded_files')}")
    print(f"Train/Val/Test Split: {res.get('split_summary')}")
