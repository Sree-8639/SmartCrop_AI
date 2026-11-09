"""
SmartCrop AI - Master Dataset Validation Suite
Runs integrity and format validation across all three benchmark datasets:
1. PlantDoc (Leaf Disease Classification)
2. IP102 (Insect Pest Detection)
3. CropHarvest (Satellite Earth Observation)

Generates:
- outputs/plantdoc_validation_report.json
- outputs/ip102_validation_report.json
- outputs/cropharvest_validation_report.json
- outputs/overall_dataset_status.json

Usage:
    python validate_datasets.py [--dataset plantdoc|ip102|cropharvest|all]
"""

import sys
import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.disease_detection.validate_plantdoc import validate_plantdoc
from src.pest_detection.validate_ip102 import validate_ip102
from src.satellite_analysis.validate_cropharvest import validate_cropharvest


def main():
    parser = argparse.ArgumentParser(description="SmartCrop AI - Dataset Validation Suite")
    parser.add_argument(
        "--dataset",
        choices=["all", "plantdoc", "ip102", "cropharvest"],
        default="all",
        help="Which dataset to validate (default: all)"
    )
    args = parser.parse_args()

    outputs_dir = PROJECT_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print(" SMARTCROP AI — THREE DATASET INTEGRITY & INTEGRATION AUDIT")
    print("=" * 75)

    results = {}

    if args.dataset in ["all", "plantdoc"]:
        print("\n[1/3] Auditing PlantDoc Foliar Disease Dataset...")
        plantdoc_dir = PROJECT_ROOT / "datasets" / "plantdoc"
        results["plantdoc"] = validate_plantdoc(plantdoc_dir, outputs_dir)

    if args.dataset in ["all", "cropharvest"]:
        print("\n[2/3] Auditing NASA CropHarvest Satellite Earth Observation Dataset...")
        cropharvest_dir = PROJECT_ROOT / "datasets" / "cropharvest"
        results["cropharvest"] = validate_cropharvest(cropharvest_dir, outputs_dir)

    if args.dataset in ["all", "ip102"]:
        print("\n[3/3] Auditing IP102 Agricultural Insect Pest Dataset...")
        ip102_dir = PROJECT_ROOT / "datasets" / "ip102"
        results["ip102"] = validate_ip102(ip102_dir, outputs_dir)

    # Fallback to existing reports if a dataset wasn't evaluated in this specific run
    for dset in ["plantdoc", "cropharvest", "ip102"]:
        if dset not in results:
            existing_rep = outputs_dir / f"{dset}_validation_report.json"
            if existing_rep.exists():
                try:
                    with open(existing_rep, "r", encoding="utf-8") as f:
                        results[dset] = json.load(f)
                except Exception:
                    pass

    # Master overall status
    overall_status = {
        "audit_timestamp": "2026-11-06",
        "datasets": {
            "plantdoc": {
                "status": results.get("plantdoc", {}).get("status", "NOT_CHECKED"),
                "total_raw_images": results.get("plantdoc", {}).get("total_raw_images"),
                "clean_images": results.get("plantdoc", {}).get("reliable_clean_images"),
                "reliable_classes": results.get("plantdoc", {}).get("reliable_classes_count"),
                "training_ready": results.get("plantdoc", {}).get("status") == "VALIDATED"
            },
            "cropharvest": {
                "status": results.get("cropharvest", {}).get("status", "NOT_CHECKED"),
                "total_arrays_h5": results.get("cropharvest", {}).get("arrays_summary", {}).get("total_h5_files_in_arrays"),
                "total_geojson_features": results.get("cropharvest", {}).get("labels_geojson_summary", {}).get("total_features"),
                "task_sufficiency": results.get("cropharvest", {}).get("task_sufficiency", {}).get("overall_verdict"),
                "training_ready": results.get("cropharvest", {}).get("status") == "VALIDATED"
            },
            "ip102": {
                "status": results.get("ip102", {}).get("status", "NOT_CHECKED"),
                "training_blocked": results.get("ip102", {}).get("training_blocked", True),
                "missing_data": results.get("ip102", {}).get("missing_requirements_checklist", {}),
                "training_ready": not results.get("ip102", {}).get("training_blocked", True)
            }
        },
        "governance_rules": {
            "strict_isolation": "PlantDoc, IP102, and CropHarvest datasets originate from disparate collection regimes and are never merged into a single training tensor or assumed to represent the same physical field.",
            "vegetation_stress_distinction": "CropHarvest ground truth is strictly crop classification. NDVI is pre-computed; NDMI canopy moisture requires unscaling B08/B11 or real-time Sentinel-2 L2A tile access.",
            "unverified_results_prohibition": "No accuracy or performance claims are made prior to actual execution."
        }
    }

    overall_path = outputs_dir / "overall_dataset_status.json"
    with open(overall_path, "w", encoding="utf-8") as f:
        json.dump(overall_status, f, indent=2)

    print("\n" + "=" * 75)
    print(" AUDIT COMPLETE — OUTPUT SUMMARY")
    print("=" * 75)
    print(f" PlantDoc Status   : {overall_status['datasets']['plantdoc']['status']} (Clean Images: {overall_status['datasets']['plantdoc']['clean_images']}, Classes: {overall_status['datasets']['plantdoc']['reliable_classes']})")
    print(f" CropHarvest Status: {overall_status['datasets']['cropharvest']['status']} (H5 Arrays: {overall_status['datasets']['cropharvest']['total_arrays_h5']}, GeoJSON: {overall_status['datasets']['cropharvest']['total_geojson_features']})")
    print(f" IP102 Status      : {overall_status['datasets']['ip102']['status']} (Training Blocked: {overall_status['datasets']['ip102']['training_ready'] is False})")
    print(f"\n Master Reports Saved in: {outputs_dir}")
    print(f"  - {outputs_dir / 'plantdoc_validation_report.json'}")
    print(f"  - {outputs_dir / 'cropharvest_validation_report.json'}")
    print(f"  - {outputs_dir / 'ip102_validation_report.json'}")
    print(f"  - {overall_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
