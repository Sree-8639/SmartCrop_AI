from pathlib import Path
import json
import h5py

root = Path("datasets/cropharvest")

required = [
    "labels.geojson",
    "normalizing_dict.h5",
    "china-crop.h5",
    "togo-eval.h5",
    "Kenya_maize_2020_0_0.h5",
    "Kenya_maize_2020_1_0.h5",
    "Brazil_coffee_2021_0_0.h5",
    "Brazil_coffee_2020_0_0.h5",
    "arrays"
]

print("\n=== CROPHARVEST DATASET AUDIT ===")

missing = [f for f in required if not (root / f).exists()]

if missing:
    print("MISSING:", missing)
else:
    print("PASS: All required files/folders exist")

# Validate GeoJSON
try:
    with open(root / "labels.geojson", encoding="utf-8") as f:
        geo = json.load(f)
    print("PASS: labels.geojson is valid JSON")
    print("GeoJSON features:", len(geo.get("features", [])))
except Exception as e:
    print("FAIL: GeoJSON:", e)

# Validate normalization file
try:
    with h5py.File(root / "normalizing_dict.h5", "r") as f:
        print("PASS: normalizing_dict.h5 opens")
        print("Keys:", list(f.keys()))
except Exception as e:
    print("FAIL: normalizing_dict.h5:", e)

# Count and inspect array files
files = list((root / "arrays").glob("*.h5"))
print("\nArray HDF5 file count:", len(files))

valid = 0
invalid = []

for p in files:
    try:
        with h5py.File(p, "r") as f:
            if "array" not in f:
                invalid.append((p.name, "Missing array dataset"))
                continue
            if f["array"].shape != (12, 18):
                invalid.append((p.name, f"Unexpected shape: {f['array'].shape}"))
                continue
            valid += 1
    except Exception as e:
        invalid.append((p.name, str(e)))

print("Valid arrays:", valid)
print("Invalid arrays:", len(invalid))

if invalid:
    print("First 10 errors:", invalid[:10])
else:
    print("PASS: All array files have expected format and shape")

print("\n=== AUDIT COMPLETE ===")
