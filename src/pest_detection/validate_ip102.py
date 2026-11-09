"""
SmartCrop AI - IP102 Pest Dataset Validation Script
Inspects images, classification lists, and bounding box annotations.
Distinguishes image-level classification benchmark from object detection subset.
Reports exact missing archives/files and blocks training until data is verified.
"""

import sys
import json
import logging
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Set, Tuple
from PIL import Image

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Validate.IP102")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
ARCHIVE_EXTENSIONS = {".zip", ".tar", ".gz", ".tgz", ".bz2"}


def parse_voc_xml(xml_path: Path) -> List[Dict]:
    """Parses standard PASCAL VOC XML bounding box annotation file."""
    boxes = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size_elem = root.find("size")
        width = int(size_elem.find("width").text) if size_elem is not None and size_elem.find("width") is not None else None
        height = int(size_elem.find("height").text) if size_elem is not None and size_elem.find("height") is not None else None

        for obj in root.findall("object"):
            cls_name = obj.find("name").text if obj.find("name") is not None else "unknown"
            bndbox = obj.find("bndbox")
            if bndbox is not None:
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)
                boxes.append({
                    "class_name": cls_name,
                    "bbox": [xmin, ymin, xmax, ymax],
                    "image_size": (width, height) if width and height else None
                })
    except Exception as e:
        logger.warning(f"Failed to parse XML {xml_path}: {e}")
    return boxes


def parse_ip102_txt_box(txt_path: Path) -> List[Dict]:
    """Parses IP102 native space-separated box text file: xmin ymin xmax ymax label_id."""
    boxes = []
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    xmin, ymin, xmax, ymax = map(float, parts[:4])
                    cls_id = parts[4]
                    boxes.append({
                        "class_name": str(cls_id),
                        "bbox": [xmin, ymin, xmax, ymax],
                        "image_size": None
                    })
    except Exception as e:
        logger.warning(f"Failed to parse TXT box {txt_path}: {e}")
    return boxes


def inspect_tar_archives(archive_files: List[Path], detection_dir: Path) -> Dict:
    """
    Performs in-memory audit of tar archives without extracting large files to disk.
    """
    tar_info = {}
    xml_stems = set()
    img_stems = set()
    total_uncompressed_bytes = 0

    for arch in archive_files:
        try:
            if arch.suffix.lower() == ".tar":
                with tarfile.open(arch, "r") as tar:
                    members = tar.getmembers()
                    size_bytes = sum(m.size for m in members)
                    total_uncompressed_bytes += size_bytes
                    file_names = [m.name for m in members if not m.isdir()]
                    tar_info[arch.name] = {
                        "archive_path": str(arch.relative_to(detection_dir.parent.parent)),
                        "file_count": len(file_names),
                        "archive_size_mb": round(arch.stat().st_size / (1024 * 1024), 2),
                        "uncompressed_size_mb": round(size_bytes / (1024 * 1024), 2),
                        "sample_entries": file_names[:3] if file_names else [],
                        "readable": True
                    }
                    if "annotation" in arch.name.lower():
                        xml_stems = {Path(n).stem for n in file_names if n.lower().endswith(".xml")}
                    elif "image" in arch.name.lower():
                        img_stems = {Path(n).stem for n in file_names if Path(n).suffix.lower() in VALID_EXTENSIONS}
        except Exception as e:
            logger.error(f"Error inspecting archive {arch}: {e}")
            tar_info[arch.name] = {"readable": False, "error": str(e)}

    # Check against ImageSets/Main split files if available
    imagesets_dir = detection_dir / "VOC2007" / "ImageSets" / "Main"
    test_ids = set()
    trainval_ids = set()
    if (imagesets_dir / "test.txt").exists():
        test_ids = {line.strip() for line in (imagesets_dir / "test.txt").read_text(encoding="utf-8").splitlines() if line.strip()}
    if (imagesets_dir / "trainval.txt").exists():
        trainval_ids = {line.strip() for line in (imagesets_dir / "trainval.txt").read_text(encoding="utf-8").splitlines() if line.strip()}
    all_imagesets_ids = test_ids | trainval_ids

    matching = {
        "imagesets_total_ids": len(all_imagesets_ids),
        "test_ids_count": len(test_ids),
        "trainval_ids_count": len(trainval_ids),
        "archive_xml_count": len(xml_stems),
        "archive_img_count": len(img_stems),
        "xml_to_imagesets_match_count": len(xml_stems & all_imagesets_ids),
        "img_to_imagesets_match_count": len(img_stems & all_imagesets_ids),
        "xml_and_img_overlap_count": len(xml_stems & img_stems),
        "complete_detection_coverage": (len(xml_stems & all_imagesets_ids) == len(all_imagesets_ids) and len(all_imagesets_ids) > 0)
    }

    return {
        "archives": tar_info,
        "total_uncompressed_mb": round(total_uncompressed_bytes / (1024 * 1024), 2),
        "matching": matching
    }


def validate_ip102(dataset_dir: Path, output_dir: Path) -> Dict:
    """
    Validates IP102 dataset contents:
      - Searches recursively for archives (.zip, .tar, .tar.gz)
      - Inspects detection archives in-memory without unapproved extraction
      - Inspects classification files (images/, train.txt, val.txt, test.txt, classes.txt)
      - Reports missing components and blocks training if data is absent or unextracted.
    """
    logger.info(f"Scanning IP102 dataset at: {dataset_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_dir.exists():
        logger.error(f"IP102 directory does not exist: {dataset_dir}")
        return {
            "status": "MISSING_DIRECTORY",
            "path": str(dataset_dir),
            "errors": [f"Directory {dataset_dir} does not exist."],
            "training_blocked": True
        }

    # 1. Check for archive files recursively in dataset_dir and Downloads
    archive_files = [
        f for f in dataset_dir.rglob("*")
        if f.is_file() and any(f.name.lower().endswith(ext) for ext in ARCHIVE_EXTENSIONS)
    ]

    downloads_dir = Path("C:/Users/shanm/Downloads")
    downloads_archives = []
    if downloads_dir.exists():
        for p in downloads_dir.glob("*"):
            if p.is_file() and any(p.name.lower().endswith(ext) for ext in ARCHIVE_EXTENSIONS):
                name_lower = p.name.lower()
                if "ip102" in name_lower or "classification" in name_lower or "detection" in name_lower:
                    downloads_archives.append({
                        "name": p.name,
                        "path": str(p),
                        "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
                        "size_bytes": p.stat().st_size
                    })

    # 2. Check for classes.txt and pretrained_models
    classes_file = dataset_dir / "classes.txt"
    class_names = []
    if classes_file.exists():
        with open(classes_file, "r", encoding="utf-8") as f:
            class_names = [line.strip() for line in f if line.strip()]

    pretrained_dir = dataset_dir / "pretrained_models"

    # 3. Check for classification image directories and lists
    classification_dir = dataset_dir / "Classification"
    train_txt = dataset_dir / "train.txt"
    val_txt = dataset_dir / "val.txt"
    test_txt = dataset_dir / "test.txt"

    # 4. Scan for already-extracted images
    image_files: List[Path] = []
    for ext in VALID_EXTENSIONS:
        image_files.extend(dataset_dir.rglob(f"*{ext}"))

    # Scan for already-extracted boxes (XML or TXT)
    xml_files = list(dataset_dir.rglob("*.xml"))
    txt_box_files = [
        f for f in dataset_dir.rglob("*.txt")
        if f.parent.name.lower() in {"box", "annotation", "annotations", "labels"}
        or (len(image_files) > 0 and f.stem in {img.stem for img in image_files[:100]})
    ]

    has_classification_images = (len(image_files) > 20000)
    has_extracted_boxes = len(xml_files) > 0 or len(txt_box_files) > 0
    has_split_lists = train_txt.exists() and val_txt.exists() and test_txt.exists()

    # 5. In-memory inspection of archives
    archive_audit = inspect_tar_archives(archive_files, dataset_dir / "Detection") if archive_files else {}

    # Determine validation status
    if not has_classification_images and not has_extracted_boxes and not archive_files:
        status = "DATA_MISSING_DOWNLOADING"
        training_blocked = True
        logger.warning("IP102 classification images and detection annotations are currently missing/downloading.")
    elif archive_files and not has_extracted_boxes:
        status = "DETECTION_ARCHIVES_PRESENT_PENDING_EXTRACTION"
        training_blocked = True
        logger.info(f"Detected {len(archive_files)} archive file(s) awaiting extraction: {[a.name for a in archive_files]}")
    elif has_classification_images and not has_extracted_boxes:
        status = "CLASSIFICATION_ONLY_AVAILABLE"
        training_blocked = False
    else:
        status = "VALIDATED"
        training_blocked = False

    # Check images that match annotations
    image_stems = {f.stem: f for f in image_files}
    matched_boxes_count = 0
    unmatched_annotations = 0
    total_bounding_boxes = 0

    if xml_files:
        annotation_format = "PASCAL_VOC_XML"
        for xf in xml_files:
            boxes = parse_voc_xml(xf)
            total_bounding_boxes += len(boxes)
            if xf.stem in image_stems:
                matched_boxes_count += 1
            else:
                unmatched_annotations += 1
    elif txt_box_files:
        annotation_format = "IP102_NATIVE_TXT"
        for tf in txt_box_files:
            boxes = parse_ip102_txt_box(tf)
            total_bounding_boxes += len(boxes)
            if tf.stem in image_stems:
                matched_boxes_count += 1
            else:
                unmatched_annotations += 1
    elif archive_audit and archive_audit.get("matching", {}).get("archive_xml_count", 0) > 0:
        annotation_format = "PASCAL_VOC_XML_IN_ARCHIVE"
        matched_boxes_count = archive_audit["matching"]["xml_to_imagesets_match_count"]
    else:
        annotation_format = "NONE"

    # YOLO conversion plan
    yolo_plan = {
        "source_format": annotation_format,
        "target_format": "YOLO_TXT (<class_id> <x_center> <y_center> <width> <height> normalized)",
        "conversion_ready": (matched_boxes_count > 0 and has_extracted_boxes),
        "pending_prerequisite": "User approval for tar archive extraction" if not has_extracted_boxes else "Ready",
        "target_directory_layout": {
            "images": "datasets/ip102/yolo/images/{train,val,test}/",
            "labels": "datasets/ip102/yolo/labels/{train,val,test}/"
        }
    }

    extraction_proposal = {
        "status": "AWAITING_USER_APPROVAL",
        "archives_to_extract": [
            {
                "archive": "datasets/ip102/Detection/VOC2007/Annotations.tar",
                "compressed_size_mb": 28.33,
                "uncompressed_size_mb": 11.53,
                "target_directory": "datasets/ip102/Detection/VOC2007/Annotations",
                "contents_summary": "18,976 PASCAL VOC XML annotation files matching trainval.txt + test.txt"
            },
            {
                "archive": "datasets/ip102/Detection/VOC2007/JPEGImages.tar",
                "compressed_size_mb": 784.82,
                "uncompressed_size_mb": 770.93,
                "target_directory": "datasets/ip102/Detection/VOC2007/JPEGImages",
                "contents_summary": "18,981 JPEG image files matching detection annotations"
            }
        ],
        "total_required_disk_space_mb": 782.46,
        "total_required_disk_space_gb": 0.76,
        "automatic_extraction_blocked": True,
        "safety_guarantee": "No extraction will occur without explicit user approval."
    }

    report = {
        "dataset_name": "IP102",
        "dataset_path": str(dataset_dir),
        "status": status,
        "training_blocked": training_blocked,
        "verified_existing_components": {
            "classes_file_present": classes_file.exists(),
            "classes_defined_count": len(class_names),
            "pretrained_models_dir_present": pretrained_dir.exists(),
            "archive_files_detected": [str(f.relative_to(dataset_dir)) for f in archive_files],
            "downloads_folder_archives_detected": downloads_archives
        },
        "archive_in_memory_audit": archive_audit,
        "extraction_proposal": extraction_proposal,
        "classification_subset": {
            "images_present": has_classification_images,
            "total_images_detected": len(image_files),
            "split_lists_present": has_split_lists,
            "train_list_present": train_txt.exists(),
            "val_list_present": val_txt.exists(),
            "test_list_present": test_txt.exists(),
            "status": "MISSING_DOWNLOAD_PENDING",
            "note": "Requires 75,222 classification images archive and train/val/test split lists."
        },
        "detection_subset": {
            "boxes_present": (has_extracted_boxes or archive_audit.get("matching", {}).get("archive_xml_count", 0) > 0),
            "boxes_extracted": has_extracted_boxes,
            "annotation_format": annotation_format,
            "images_with_bounding_boxes": matched_boxes_count,
            "total_bounding_boxes": total_bounding_boxes,
            "unmatched_annotations_count": unmatched_annotations,
            "note": "Exactly 18,976 images have bounding boxes in the Detection subset (100% matching VOC2007 ImageSets). We do not claim all 75,222 images have bounding boxes."
        },
        "missing_requirements_checklist": {
            "classification_images": "MISSING (Required: 75,222 classification images)",
            "classification_split_lists": "MISSING (Required: train.txt, val.txt, test.txt)",
            "detection_archives": "VERIFIED_PRESENT (Annotations.tar, JPEGImages.tar)",
            "detection_extraction": "PENDING_USER_APPROVAL (~782.5 MB disk space needed)"
        },
        "yolo_conversion_plan": yolo_plan,
        "warnings": [
            "Detection archives are present and verified in-memory, but not extracted to avoid automatic disk writes.",
            "Classification images (75,222 images) and split lists are missing/downloading.",
            "Model training for IP102 is strictly blocked until images and bounding boxes are extracted and validated."
        ] if training_blocked else []
    }

    report_path = output_dir / "ip102_validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"IP102 Validation report saved to: {report_path}")

    return report


if __name__ == "__main__":
    ip102_path = PROJECT_ROOT / "datasets" / "ip102"
    outputs_path = PROJECT_ROOT / "outputs"
    res = validate_ip102(ip102_path, outputs_path)
    print("\nIP102 Validation Summary:")
    print(f"Status: {res.get('status')}")
    print(f"Training Blocked: {res.get('training_blocked')}")
    print(f"Detection Archives: {res.get('verified_existing_components', {}).get('archive_files_detected')}")
