"""
SmartCrop AI - Model Weights Validator and Automated Fetcher
Checks integrity of all 5 ML pipeline checkpoints.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODELS = [
    {
        "name": "PlantDoc Disease Diagnosis",
        "path": BASE_DIR / "models" / "disease" / "disease_model_exp_D_best.pth",
        "expected_min_mb": 180,
    },
    {
        "name": "IP102 Pest Species Classification",
        "path": BASE_DIR / "models" / "pest" / "ip102_convnext_best.pth",
        "expected_min_mb": 180,
    },
    {
        "name": "IP102 Pest Detection (YOLOv8)",
        "path": BASE_DIR / "models" / "pest" / "yolov8_pest_best.pt",
        "expected_min_mb": 20,
    },
    {
        "name": "CropHarvest Binary Crop Cover",
        "path": BASE_DIR / "models" / "cropharvest" / "binary" / "transformer_seeds.pth",
        "expected_min_mb": 2,
    },
    {
        "name": "CropHarvest 10-Class Categorization",
        "path": BASE_DIR / "models" / "cropharvest" / "10class" / "best_model.pth",
        "expected_min_mb": 0.5,
    },
]

def verify_all_models():
    print("==================================================================")
    print("🌱 SmartCrop AI — Model Weights Integrity Verification")
    print("==================================================================")
    all_ok = True
    for m in MODELS:
        p = m["path"]
        if not p.exists():
            print(f"❌ Missing: {m['name']} -> {p}")
            all_ok = False
        else:
            mb = p.stat().st_size / (1024 * 1024)
            if mb < m["expected_min_mb"]:
                print(f"⚠️ Truncated: {m['name']} ({mb:.2f} MB, expected >={m['expected_min_mb']} MB)")
                all_ok = False
            else:
                print(f"✅ Verified: {m['name']} ({mb:.2f} MB)")

    print("==================================================================")
    if all_ok:
        print("All 5 production checkpoints are present and verified!")
    else:
        print("Some checkpoints require attention.")
    return all_ok

if __name__ == "__main__":
    verify_all_models()

