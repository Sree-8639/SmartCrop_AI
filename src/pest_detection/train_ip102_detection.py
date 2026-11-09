"""
SmartCrop AI - IP102 Object Detection Training Pipeline
Trains a pretrained YOLO detector (YOLOv8 / YOLO11) on the authentic IP102 benchmark.
Evaluates mAP@0.50, mAP@0.50:0.95, Precision, and Recall.
Preserves official test set strictly for held-out evaluation.
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("IP102.Detection.Train")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_YAML = PROJECT_ROOT / "datasets" / "ip102" / "Detection" / "yolo" / "dataset.yaml"


def get_default_output_dir() -> Path:
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working/IP102_Results/detection")
    return PROJECT_ROOT / "outputs" / "ip102" / "detection"


def check_environment() -> Dict[str, Any]:
    """Inspects execution environment, GPU availability, and Ultralytics install."""
    import torch
    has_gpu = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if has_gpu else "None"
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3) if has_gpu else 0.0

    try:
        import ultralytics
        ultralytics_ver = ultralytics.__version__
        has_ultralytics = True
    except ImportError:
        ultralytics_ver = None
        has_ultralytics = False

    env_info = {
        "has_gpu": has_gpu,
        "device": "cuda" if has_gpu else "cpu",
        "gpu_name": gpu_name,
        "gpu_mem_gb": round(gpu_mem, 2),
        "torch_version": torch.__version__,
        "has_ultralytics": has_ultralytics,
        "ultralytics_version": ultralytics_ver
    }

    logger.info("=" * 80)
    logger.info("IP102 OBJECT DETECTION ENVIRONMENT INSPECTION")
    logger.info("=" * 80)
    logger.info(f"Execution Device : {env_info['device'].upper()}")
    if has_gpu:
        logger.info(f"GPU Model        : {gpu_name} ({env_info['gpu_mem_gb']:.2f} GB VRAM)")
    else:
        logger.info("GPU Model        : None detected (Running on host CPU)")
    logger.info(f"PyTorch Version  : {env_info['torch_version']}")
    logger.info(f"Ultralytics YOLO : {'Installed (v' + str(ultralytics_ver) + ')' if has_ultralytics else 'NOT INSTALLED (Run: pip install ultralytics)'}")
    logger.info("=" * 80)

    return env_info


def run_sanity_check(data_yaml_path: Path, model_size: str = "yolov8n.pt") -> bool:
    """Verifies that dataset.yaml exists, points to valid images, and model can be loaded."""
    logger.info(f"Running pre-flight sanity check on: {data_yaml_path}...")
    if not data_yaml_path.exists():
        logger.error(f"Sanity Check FAIL: dataset.yaml does not exist at {data_yaml_path}")
        return False

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    path_line = [l for l in lines if l.startswith("path:")]
    if not path_line:
        logger.error("Sanity Check FAIL: Missing 'path:' field in dataset.yaml")
        return False

    dataset_root = Path(path_line[0].split("path:")[1].strip())
    train_images = dataset_root / "images" / "train"
    val_images = dataset_root / "images" / "val"
    test_images = dataset_root / "images" / "test"

    if not train_images.exists():
        logger.error(f"Sanity Check FAIL: Train images directory missing: {train_images}")
        return False

    train_count = len(list(train_images.glob("*.jpg")))
    val_count = len(list(val_images.glob("*.jpg"))) if val_images.exists() else 0
    test_count = len(list(test_images.glob("*.jpg"))) if test_images.exists() else 0

    logger.info(f"Sanity Check Images -> Train: {train_count}, Val: {val_count}, Test: {test_count}")
    if train_count == 0:
        logger.error("Sanity Check FAIL: 0 training images found!")
        return False

    logger.info(">>> Pre-flight Sanity Check PASSED. Dataset is valid for YOLO training.")
    return True


def train_yolo_detector(
    data_yaml: Path,
    model_size: str = "yolov8n.pt",
    epochs: int = 30,
    batch_size: int = 16,
    img_size: int = 640,
    output_dir: Optional[Path] = None,
    device_arg: Optional[str] = None
) -> Dict[str, Any]:
    """
    Trains a pretrained YOLO detector and evaluates on validation and held-out test sets.
    Metrics tracked: mAP@50, mAP@50:95, Precision, Recall.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "Ultralytics YOLO is required for training. Please run: pip install ultralytics"
        )

    import torch
    if output_dir is None:
        output_dir = get_default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    if device_arg is None:
        device_arg = "0" if torch.cuda.is_available() else "cpu"

    logger.info("=" * 80)
    logger.info("COMMENCING IP102 YOLO OBJECT DETECTION TRAINING")
    logger.info("=" * 80)
    logger.info(f"Base Pretrained Model : {model_size}")
    logger.info(f"Dataset Specification : {data_yaml}")
    logger.info(f"Epochs               : {epochs}")
    logger.info(f"Batch Size           : {batch_size}")
    logger.info(f"Image Resolution     : {img_size}x{img_size}")
    logger.info(f"Device               : {device_arg}")
    logger.info(f"Output Directory     : {output_dir}")
    logger.info("=" * 80)

    # 1. Load pretrained model
    model = YOLO(model_size)

    # 2. Train model
    start_train_time = time.time()
    train_results = model.train(
        data=str(data_yaml.resolve()),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device_arg,
        project=str(output_dir / "runs"),
        name="ip102_yolo",
        exist_ok=True,
        save=True,
        save_period=5,
        patience=10,
        verbose=True
    )
    total_training_time = time.time() - start_train_time
    logger.info(f"Training completed in {total_training_time / 60.0:.2f} minutes.")

    # 3. Evaluate on Validation Split
    logger.info("Evaluating Best Checkpoint on Validation Partition...")
    val_results = model.val(
        data=str(data_yaml.resolve()),
        split="val",
        batch=batch_size,
        imgsz=img_size,
        device=device_arg,
        project=str(output_dir / "runs"),
        name="ip102_val_eval",
        exist_ok=True
    )

    # 4. Strict Evaluation on Held-Out Test Split (Evaluated ONLY ONCE)
    logger.info("Evaluating Best Checkpoint on Held-Out Test Partition (Preserved Official Test Set)...")
    test_results = model.val(
        data=str(data_yaml.resolve()),
        split="test",
        batch=batch_size,
        imgsz=img_size,
        device=device_arg,
        project=str(output_dir / "runs"),
        name="ip102_test_eval",
        exist_ok=True
    )

    # Extract metrics
    metrics_summary = {
        "model_architecture": model_size,
        "input_resolution": [img_size, img_size],
        "epochs": epochs,
        "batch_size": batch_size,
        "training_time_minutes": round(total_training_time / 60.0, 2),
        "validation_metrics": {
            "mAP50": round(float(val_results.box.map50), 4),
            "mAP50_95": round(float(val_results.box.map), 4),
            "precision": round(float(val_results.box.mp), 4),
            "recall": round(float(val_results.box.mr), 4)
        },
        "held_out_test_metrics": {
            "mAP50": round(float(test_results.box.map50), 4),
            "mAP50_95": round(float(test_results.box.map), 4),
            "precision": round(float(test_results.box.mp), 4),
            "recall": round(float(test_results.box.mr), 4)
        }
    }

    metrics_export_path = output_dir / "detection_metrics.json"
    with open(metrics_export_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    logger.info(f"Saved Detection Metrics to: {metrics_export_path}")

    print("\n" + "=" * 80)
    print("SMARTCROP AI: IP102 OBJECT DETECTION BENCHMARK SCORECARD")
    print("=" * 80)
    print(f"Pretrained Backbone     : {model_size}")
    print(f"Validation mAP@0.50     : {metrics_summary['validation_metrics']['mAP50'] * 100:.2f}%")
    print(f"Validation mAP@0.50:0.95: {metrics_summary['validation_metrics']['mAP50_95'] * 100:.2f}%")
    print(f"Validation Precision    : {metrics_summary['validation_metrics']['precision'] * 100:.2f}%")
    print(f"Validation Recall       : {metrics_summary['validation_metrics']['recall'] * 100:.2f}%")
    print("-" * 80)
    print(f"Held-Out Test mAP@0.50  : {metrics_summary['held_out_test_metrics']['mAP50'] * 100:.2f}%")
    print(f"Held-Out Test mAP@50:95 : {metrics_summary['held_out_test_metrics']['mAP50_95'] * 100:.2f}%")
    print(f"Held-Out Test Precision : {metrics_summary['held_out_test_metrics']['precision'] * 100:.2f}%")
    print(f"Held-Out Test Recall    : {metrics_summary['held_out_test_metrics']['recall'] * 100:.2f}%")
    print("=" * 80)

    return metrics_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IP102 YOLO Pest Object Detector")
    parser.add_argument("--data-yaml", type=Path, default=DEFAULT_DATA_YAML, help="Path to dataset.yaml")
    parser.add_argument("--model-size", type=str, default="yolov8n.pt", help="Pretrained YOLO model (yolov8n.pt, yolov8s.pt, etc.)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--img-size", type=int, default=640, help="Input resolution")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    parser.add_argument("--device", type=str, default=None, help="Device (e.g. 0 or cpu)")
    parser.add_argument("--sanity-check", action="store_true", help="Run pre-flight sanity checks only (no training)")

    args = parser.parse_args()

    env = check_environment()
    valid = run_sanity_check(args.data_yaml, args.model_size)

    if args.sanity_check:
        logger.info("Sanity check completed. Exiting without training.")
        sys.exit(0 if valid else 1)

    if not valid:
        logger.error("Dataset sanity check failed. Cannot start training.")
        sys.exit(1)

    train_yolo_detector(
        data_yaml=args.data_yaml,
        model_size=args.model_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        output_dir=args.output_dir,
        device_arg=args.device
    )
