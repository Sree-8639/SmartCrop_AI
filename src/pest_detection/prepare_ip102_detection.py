"""
SmartCrop AI - IP102 Object Detection Data Preparation Pipeline
Extracts Pascal VOC archives safely, validates XML annotations, bounding boxes,
class taxonomies, image-annotation correspondence, creates reproducible train/val splits
from official trainval.txt, isolates official test.txt, converts annotations to YOLO format,
generates dataset.yaml and comprehensive dataset audit reports.
"""

import os
import sys
import json
import time
import shutil
import random
import tarfile
import logging
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("IP102.Detection.Prepare")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_VOC_DIR = PROJECT_ROOT / "datasets" / "ip102" / "Detection" / "VOC2007"
DEFAULT_CLASSES_FILE = PROJECT_ROOT / "datasets" / "ip102" / "classes.txt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "ip102" / "Detection" / "yolo"

NUM_PEST_CLASSES = 102
SEED = 42


def load_classes_txt(classes_file: Path) -> List[str]:
    """
    Loads 102 class names from classes.txt (format: '<id>  <name>').
    Returns a 102-element list ordered by 0-indexed class index 0..101.
    """
    classes = [f"Class_{i}" for i in range(NUM_PEST_CLASSES)]
    if not classes_file.exists():
        logger.warning(f"classes.txt not found at {classes_file}. Using generic class names.")
        return classes

    with open(classes_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) >= 2 and parts[0].isdigit():
                idx = int(parts[0]) - 1  # 1-indexed to 0-indexed
                if 0 <= idx < NUM_PEST_CLASSES:
                    classes[idx] = parts[1].strip()
    return classes


def inspect_tar_archive(tar_path: Path) -> Dict[str, Any]:
    """Inspects tar archive contents without full extraction."""
    if not tar_path.exists():
        raise FileNotFoundError(f"Archive not found: {tar_path}")

    logger.info(f"Inspecting tar archive: {tar_path.name} ({tar_path.stat().st_size / (1024**2):.2f} MB)...")
    with tarfile.open(tar_path, "r") as tar:
        members = tar.getmembers()
        files = [m.name for m in members if m.isfile()]
        dirs = [m.name for m in members if m.isdir()]

    return {
        "archive_path": str(tar_path.resolve()),
        "size_bytes": tar_path.stat().st_size,
        "total_members": len(members),
        "file_count": len(files),
        "dir_count": len(dirs),
        "sample_files": files[:5]
    }


def parse_voc_xml_clean(xml_content: str) -> ET.Element:
    """
    Parses Pascal VOC XML text, handling edge-cases like duplicated root elements
    (e.g., IP087000986.xml with multiple </annotation> tags).
    """
    if "</annotation>" in xml_content:
        idx = xml_content.find("</annotation>") + len("</annotation>")
        xml_content = xml_content[:idx]
    return ET.fromstring(xml_content)


def extract_single_member_to_disk(
    tar: tarfile.TarFile,
    member: tarfile.TarInfo,
    dest_path: Path
) -> None:
    """Safely extracts a single file from tar to dest_path without arbitrary path traversal."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    f_in = tar.extractfile(member)
    if f_in is None:
        raise RuntimeError(f"Could not extract file member: {member.name}")
    with open(dest_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


def prepare_detection_dataset(
    voc_dir: Path,
    classes_file: Path,
    output_dir: Path,
    val_ratio: float = 0.15,
    seed: int = SEED,
    dry_run: bool = False,
    dry_run_samples: int = 100
) -> Dict[str, Any]:
    """
    Executes full preparation pipeline:
    1. Inspects archives and split files.
    2. Validates image-annotation correspondence.
    3. Partitions trainval into train and val deterministically.
    4. Extracts images and converts annotations to YOLO format.
    5. Writes dataset.yaml and audit reports.
    """
    start_time = time.time()
    random.seed(seed)

    ann_tar = voc_dir / "Annotations.tar"
    img_tar = voc_dir / "JPEGImages.tar"
    trainval_txt = voc_dir / "ImageSets" / "Main" / "trainval.txt"
    test_txt = voc_dir / "ImageSets" / "Main" / "test.txt"

    # Pre-condition assertions
    assert ann_tar.exists(), f"Missing Annotations archive: {ann_tar}"
    assert img_tar.exists(), f"Missing JPEGImages archive: {img_tar}"
    assert trainval_txt.exists(), f"Missing trainval.txt: {trainval_txt}"
    assert test_txt.exists(), f"Missing test.txt: {test_txt}"
    assert classes_file.exists(), f"Missing classes.txt: {classes_file}"

    # Load 102 class names
    class_names = load_classes_txt(classes_file)
    logger.info(f"Loaded {len(class_names)} official IP102 class taxonomies.")

    # 1. Archive Inspection
    ann_info = inspect_tar_archive(ann_tar)
    img_info = inspect_tar_archive(img_tar)

    logger.info(f"Annotations.tar has {ann_info['file_count']} files.")
    logger.info(f"JPEGImages.tar has {img_info['file_count']} files.")

    # 2. Parse official splits
    with open(trainval_txt, "r", encoding="utf-8") as f:
        trainval_ids = [line.strip() for line in f if line.strip()]
    with open(test_txt, "r", encoding="utf-8") as f:
        test_ids = [line.strip() for line in f if line.strip()]

    logger.info(f"Official trainval IDs: {len(trainval_ids):,}")
    logger.info(f"Official held-out test IDs: {len(test_ids):,}")

    # Check split disjointness
    overlap = set(trainval_ids) & set(test_ids)
    if overlap:
        raise ValueError(f"FATAL: {len(overlap)} IDs overlap between trainval and test splits: {list(overlap)[:5]}")
    logger.info("Disjointness check: PASS (0 overlap between trainval and test).")

    # Index tar archives
    logger.info("Indexing archive file members...")
    with tarfile.open(ann_tar, "r") as tar_a:
        ann_members = {Path(m.name).stem: m for m in tar_a.getmembers() if m.isfile() and m.name.endswith(".xml")}
    with tarfile.open(img_tar, "r") as tar_i:
        img_members = {Path(m.name).stem: m for m in tar_i.getmembers() if m.isfile() and m.name.lower().endswith(".jpg")}

    # Validate image-annotation correspondence
    all_target_ids = set(trainval_ids) | set(test_ids)
    missing_ann = [img_id for img_id in all_target_ids if img_id not in ann_members]
    missing_img = [img_id for img_id in all_target_ids if img_id not in img_members]

    if missing_ann:
        raise FileNotFoundError(f"FATAL: {len(missing_ann)} images missing XML annotations! Samples: {missing_ann[:5]}")
    if missing_img:
        raise FileNotFoundError(f"FATAL: {len(missing_img)} images missing from JPEGImages.tar! Samples: {missing_img[:5]}")

    logger.info("Correspondence check: PASS (100% of 18,976 split images have matching XML and JPEG).")

    # 3. Create reproducible train / val split from trainval ONLY
    # Using fixed seed shuffle
    shuffled_trainval = list(trainval_ids)
    random.shuffle(shuffled_trainval)
    val_count = int(len(shuffled_trainval) * val_ratio)
    val_ids = set(shuffled_trainval[:val_count])
    train_ids = set(shuffled_trainval[val_count:])

    logger.info(f"Partitioned trainval ({len(trainval_ids)}): Train = {len(train_ids):,} ({len(train_ids)/len(trainval_ids)*100:.1f}%), Val = {len(val_ids):,} ({len(val_ids)/len(trainval_ids)*100:.1f}%)")
    logger.info(f"Held-out test set strictly isolated: {len(test_ids):,} images.")

    split_map = {}
    for i in train_ids:
        split_map[i] = "train"
    for i in val_ids:
        split_map[i] = "val"
    for i in test_ids:
        split_map[i] = "test"

    # Setup YOLO destination directories
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"

    for s in ["train", "val", "test"]:
        (images_dir / s).mkdir(parents=True, exist_ok=True)
        (labels_dir / s).mkdir(parents=True, exist_ok=True)

    # 4. Conversion & Extraction
    target_ids_list = list(all_target_ids)
    if dry_run:
        logger.info(f"DRY RUN ACTIVATED: Processing {dry_run_samples} samples only.")
        target_ids_list = target_ids_list[:dry_run_samples]

    audit = {
        "total_processed": len(target_ids_list),
        "split_counts": {"train": 0, "val": 0, "test": 0},
        "total_boxes": 0,
        "boxes_per_split": {"train": 0, "val": 0, "test": 0},
        "degenerate_boxes_quarantined": [],
        "zero_box_images": [],
        "class_box_counts": {i: 0 for i in range(NUM_PEST_CLASSES)},
        "dimension_mismatches": []
    }

    logger.info(f"Extracting images and converting annotations to YOLO format ({len(target_ids_list)} files)...")

    with tarfile.open(ann_tar, "r") as tar_a, tarfile.open(img_tar, "r") as tar_i:
        for idx, img_id in enumerate(target_ids_list):
            split = split_map[img_id]
            audit["split_counts"][split] += 1

            # Extract image
            img_m = img_members[img_id]
            dest_img = images_dir / split / f"{img_id}.jpg"
            extract_single_member_to_disk(tar_i, img_m, dest_img)

            # Extract and parse XML
            ann_m = ann_members[img_id]
            xml_raw = tar_a.extractfile(ann_m).read().decode("utf-8", errors="ignore")
            root = parse_voc_xml_clean(xml_raw)

            # Parse dimensions
            size_elem = root.find("size")
            width = float(size_elem.find("width").text)
            height = float(size_elem.find("height").text)

            yolo_lines = []
            objects = root.findall("object")

            if not objects:
                audit["zero_box_images"].append(img_id)

            for obj in objects:
                name_elem = obj.find("name")
                raw_name = name_elem.text.strip() if name_elem is not None else ""

                # Handle numeric string class IDs 0..101
                if raw_name.isdigit():
                    class_idx = int(raw_name)
                else:
                    # Look up in class_names
                    if raw_name in class_names:
                        class_idx = class_names.index(raw_name)
                    else:
                        raise ValueError(f"Unknown class name '{raw_name}' in {ann_m.name}")

                if not (0 <= class_idx < NUM_PEST_CLASSES):
                    raise ValueError(f"Class index {class_idx} out of range [0, 101] in {ann_m.name}")

                bndbox = obj.find("bndbox")
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)

                # Validate bounding box coordinates
                if xmax <= xmin or ymax <= ymin:
                    audit["degenerate_boxes_quarantined"].append({
                        "image_id": img_id,
                        "class_idx": class_idx,
                        "class_name": class_names[class_idx],
                        "box": [xmin, ymin, xmax, ymax],
                        "reason": "width_or_height_zero_or_negative"
                    })
                    logger.warning(f"Degenerate box quarantined in {img_id}: {[xmin, ymin, xmax, ymax]}")
                    continue

                # Clamp box coordinates to image boundaries
                xmin = max(0.0, min(xmin, width))
                ymin = max(0.0, min(ymin, height))
                xmax = max(0.0, min(xmax, width))
                ymax = max(0.0, min(ymax, height))

                box_w = xmax - xmin
                box_h = ymax - ymin
                if box_w <= 0.0 or box_h <= 0.0:
                    continue

                # Convert to normalized YOLO format: class_id x_center y_center width height
                x_center = (xmin + xmax) / 2.0 / width
                y_center = (ymin + ymax) / 2.0 / height
                norm_w = box_w / width
                norm_h = box_h / height

                yolo_lines.append(f"{class_idx} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
                audit["total_boxes"] += 1
                audit["boxes_per_split"][split] += 1
                audit["class_box_counts"][class_idx] += 1

            # Write YOLO label text file
            dest_label = labels_dir / split / f"{img_id}.txt"
            with open(dest_label, "w", encoding="utf-8") as f_lbl:
                f_lbl.write("\n".join(yolo_lines) + "\n" if yolo_lines else "")

            if (idx + 1) % 2000 == 0 or (idx + 1) == len(target_ids_list):
                logger.info(f"Processed {idx + 1}/{len(target_ids_list)} images...")

    # 5. Generate dataset.yaml for YOLO
    yaml_content = f"""# SmartCrop AI - IP102 Object Detection Dataset Specification
# Benchmark: IP102 (Wu et al., 2019)
# Format: Ultralytics YOLO

path: {output_dir.resolve().as_posix()}
train: images/train
val: images/val
test: images/test

nc: {NUM_PEST_CLASSES}
names:
"""
    for i, name in enumerate(class_names):
        yaml_content += f"  {i}: '{name}'\n"

    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    logger.info(f"Saved YOLO dataset.yaml to: {yaml_path}")

    # 6. Save JSON audit report
    elapsed_time = time.time() - start_time
    audit["elapsed_seconds"] = round(elapsed_time, 2)
    audit["yaml_path"] = str(yaml_path.resolve())
    audit["num_classes"] = NUM_PEST_CLASSES
    audit["active_classes_with_boxes"] = sum(1 for c, cnt in audit["class_box_counts"].items() if cnt > 0)

    audit_path = output_dir / "detection_dataset_audit.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    logger.info(f"Saved Dataset Audit Report to: {audit_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("SMARTCROP AI: IP102 OBJECT DETECTION DATASET PREPARATION REPORT")
    print("=" * 80)
    print(f"YOLO Root Directory     : {output_dir.resolve()}")
    print(f"Total Images Processed  : {audit['total_processed']:,}")
    print(f"  - Train Partition     : {audit['split_counts']['train']:,} images ({audit['boxes_per_split']['train']:,} boxes)")
    print(f"  - Validation Partition: {audit['split_counts']['val']:,} images ({audit['boxes_per_split']['val']:,} boxes)")
    print(f"  - Held-Out Test Set   : {audit['split_counts']['test']:,} images ({audit['boxes_per_split']['test']:,} boxes) [Isolated]")
    print(f"Total Bounding Boxes    : {audit['total_boxes']:,}")
    print(f"Degenerate Quarantined  : {len(audit['degenerate_boxes_quarantined'])} boxes")
    print(f"Active Pest Classes     : {audit['active_classes_with_boxes']} / {NUM_PEST_CLASSES}")
    print(f"dataset.yaml Path       : {yaml_path}")
    print(f"Audit Time              : {elapsed_time:.1f}s")
    print("=" * 80)

    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare IP102 Pascal VOC into YOLO Detection Format")
    parser.add_argument("--voc-dir", type=Path, default=DEFAULT_VOC_DIR, help="Path to VOC2007 directory")
    parser.add_argument("--classes-file", type=Path, default=DEFAULT_CLASSES_FILE, help="Path to classes.txt")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Destination directory for YOLO dataset")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio from trainval (default: 0.15)")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for deterministic split")
    parser.add_argument("--dry-run", action="store_true", help="Process only a small batch for verification")
    parser.add_argument("--dry-run-samples", type=int, default=50, help="Number of samples for dry run")

    args = parser.parse_args()

    prepare_detection_dataset(
        voc_dir=args.voc_dir,
        classes_file=args.classes_file,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        dry_run=args.dry_run,
        dry_run_samples=args.dry_run_samples
    )

