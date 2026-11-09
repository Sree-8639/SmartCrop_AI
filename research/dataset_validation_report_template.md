# SmartCrop AI - Dataset Validation & Verification Report Template

**Date:** 2026-11-06  
**Auditor:** SmartCrop AI Validation Engine  
**Experiment Configuration:** Seed 42 | Split 70% Train, 15% Validation, 15% Test  

---

## 1. PlantDoc (Foliar Disease Classification)

- **Dataset Path:** `datasets/plantdoc`
- **Validation Status:** `[ PENDING / COMPLETED ]`
- **Total Classes Detected:** `[ 0 / 27 ]`
- **Total Valid Images:** `[ N ]`
- **Corrupted / Truncated Images:** `[ N ]` (Details logged to `outputs/plantdoc_validation_report.json`)
- **Resolution Range:**
  - Minimum Resolution: `[ W x H ]`
  - Maximum Resolution: `[ W x H ]`
  - Mean Resolution: `[ W x H ]`
- **Stratified Split Manifest:** `outputs/plantdoc_split_manifest.json`
  - Training Set: `[ N_train ]` images (70%)
  - Validation Set: `[ N_val ]` images (15%)
  - Held-out Test Set: `[ N_test ]` images (15%)
- **Data Leakage Check:**
  - Hashes verified between train and test: `[ PASS / FAIL ]`
  - Zero overlapping sample identifiers across splits: `[ CONFIRMED ]`

---

## 2. IP102 (Insect Pest Detection)

- **Dataset Path:** `datasets/ip102`
- **Validation Status:** `[ PENDING / COMPLETED ]`
- **Total Images Detected:** `[ N ]`
- **Images with Bounding Box Annotations:** `[ N_bbox ]`
- **Images with Image-Level Labels Only:** `[ N_cls ]`
- **Annotation Format Detected:** `[ PASCAL_VOC_XML / IP102_NATIVE_TXT / NONE ]`
- **Target Pest Species Count:** `[ N_species ]` (Target: 102 classes)
- **YOLO Conversion Feasibility:**
  - Normalized coordinate validity: `[ VERIFIED ]`
  - Empty/zero-area bounding box check: `[ COMPLETED ]`
- **Split Distribution (Detection Subset):**
  - Train: `[ 70% ]`
  - Validation: `[ 15% ]`
  - Test: `[ 15% ]`

---

## 3. CropHarvest (Satellite Earth Observation)

- **Dataset Path:** `datasets/cropharvest`
- **Validation Status:** `[ PENDING / COMPLETED ]`
- **Labels File Type:** `[ GeoJSON / CSV / NOT FOUND ]`
- **Total Georeferenced Records:** `[ N_records ]`
- **Crop vs. Non-Crop Distribution:**
  - Crop: `[ N_crop ]`
  - Non-crop: `[ N_noncrop ]`
- **Dominant Crop Classes:** `[ List top crop varieties ]`
- **Multi-Spectral Feature Tensor Structure:**
  - Temporal Timesteps: `[ 12 months ]`
  - Feature Dimensions per Timestep: `[ 18 channels / custom ]`
- **Optical Bands Available:**
  - B04 (Red): `[ YES / NO ]`
  - B08 (NIR): `[ YES / NO ]`
  - B11 (SWIR1): `[ YES / NO ]`
- **Biophysical Index Computation Status:**
  - NDVI computable directly from array: `[ YES / NO ]`
  - NDMI computable directly from array: `[ YES / NO ]`
  - External Sentinel-2 L2A tile connection required: `[ YES / NO ]`

---

## 4. Master Sign-Off & Verification Verdict

| Dataset | File Integrity | Split Generated | Leakage Safeguard | Ready for Baseline Training |
| :--- | :---: | :---: | :---: | :---: |
| **PlantDoc** | `[ PENDING ]` | `[ PENDING ]` | `[ PENDING ]` | `[ NO ]` |
| **IP102** | `[ PENDING ]` | `[ PENDING ]` | `[ PENDING ]` | `[ NO ]` |
| **CropHarvest**| `[ PENDING ]` | `[ PENDING ]` | `[ PENDING ]` | `[ NO ]` |

> **Sign-Off Statement:** Model training code execution is blocked until all three datasets pass automated validation and empirical numbers are recorded in this document.

