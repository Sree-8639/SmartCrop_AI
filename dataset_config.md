# SmartCrop AI - Dataset Configuration & Verification Manual

This document provides empirical facts, verified schemas, and acquisition requirements for the three benchmark datasets used in SmartCrop AI.

---

## 1. Verified Dataset Status & Boundary Architecture

| Dataset | Local Files Present | Empirical Status | Validated Active Samples | Training Status |
| :--- | :--- | :--- | :--- | :--- |
| **PlantDoc** | `train/`, `test/`, `PlantDoc_Examples.png`, `LICENSE.txt` | **VALIDATED** | **2,549 clean unique images across 27 reliable classes** | **READY FOR KAGGLE TRAINING** |
| **CropHarvest** | `labels.geojson`, `normalizing_dict.h5`, `china-crop.h5`, `togo-eval.h5`, `Kenya_maize_...h5`, `arrays/` | **VALIDATED** | **113,893 features, 87,464 array `.h5` files, 46,288 crop-type labels** | **READY FOR KAGGLE TRAINING** |
| **IP102** | `classes.txt`, `README.md`, `pretrained_models/` | **DATA MISSING / DOWNLOADING** | **0 images, 0 bounding boxes extracted** | **BLOCKED PENDING ARCHIVE DOWNLOAD** |

> **Scientific Research Boundary:**  
> These three datasets originate from completely disparate collection regimes across different continents and sensor modalities. SmartCrop AI maintains **strict isolation** between their training and evaluation pipelines. Records from different datasets are never combined as if they belong to the same physical farm or time window.

---

## 2. Dataset 1: PlantDoc (Foliar Disease Classification)

### A. Empirical Audit & Verification Findings
- **Raw Class Count:** 28 distinct directory labels discovered.
  - Upstream `train/`: 28 classes, 2,336 raw files.
  - Upstream `test/`: 27 classes, 236 raw files.
  - Class `Tomato two spotted spider mites leaf` exists only in `train` with **2 images total** and 0 in `test`. Because 2 samples cannot support a 3-way partition ($\ge 1$ train, $\ge 1$ val, $\ge 1$ test), it is flagged and excluded from the 3-way benchmark.
- **Corrupted / Truncated Files:** 0 unreadable files detected during PIL verification.
- **Duplicate & Conflicting Files:**
  - **3 Cross-split identical duplicates:** Same image placed in both `train` and `test` with identical class (`Blueberry leaf`, `Tomato leaf yellow virus`, `Tomato Septoria leaf spot`).
  - **9 Conflicting-label duplicates:** Identical image files assigned contradictory labels across upstream `train` and `test` (e.g., `Corn Gray leaf spot` in train vs. `Corn leaf blight` in test). All conflicting instances are pruned from the clean benchmark to preserve ground-truth integrity.
- **Active Clean Benchmark Pool:**
  - **2,549 unique clean images across 27 reliable classes** ($\ge 54$ images per class).
  - Reproducible 70% Train (1,787 images, 70.1%), 15% Validation (383 images, 15.0%), and 15% Test (379 images, 14.9%) split generated with random seed `42` (`outputs/plantdoc_split_manifest.json`).

---

## 3. Dataset 2: CropHarvest (Satellite Earth Observation)

### A. Empirical Audit & Verification Findings
- **Labels File (`labels.geojson`):**
  - Total features: **113,893 georeferenced polygon/point records**.
  - Identifiers: `(index, dataset)` pairs — zero duplicate identifier collisions found across all 113,893 records.
  - Binary crop distribution: **80,895 crop (`is_crop = 1`)** vs. **32,998 non-crop (`is_crop = 0`)**.
  - Multi-class crop-type labels: **46,288 records** with explicit crop species labels (`label`/`classification_label`).
    - Top crops: `cotton` (3,871), `maize` (3,084), `rice` (2,917), `wheat` (1,948), `millet` (1,295), `groundnut` (956), `sorghum` (860), `soybean` (736).
- **Satellite Time-Series Arrays (`arrays/`):**
  - **87,464 individual HDF5 files** named `{index}_{dataset}.h5`.
  - Feature tensor shape: **`[12 timesteps, 18 channels]`** with `float64` dtype.
  - Normalizing statistics: `normalizing_dict.h5` contains 18 channel means and standard deviations.
- **Regional Benchmark Files:**
  - `china-crop.h5`: 4,397 records, `x` shape `(4397, 12, 18)`, `y` shape `(4397,)`.
  - `togo-eval.h5`: 306 records, `x` shape `(306, 12, 18)`, `y` shape `(306,)`.
  - `Kenya_maize_2020_0_0.h5`: 7,600 records, `x` shape `(7600, 12, 18)`.
  - `Brazil_coffee_2020_0_0.h5` & `Brazil_coffee_2021_0_0.h5`: regional coffee evaluations.
- **Task Sufficiency:**
  - **SUFFICIENT** for both binary crop classification and multi-class crop type classification.
- **Vegetation Stress Index Distinction:**
  - **NDVI:** Pre-computed and available in feature channel index 17.
  - **NDMI:** NOT pre-extracted in CropHarvest arrays. NDMI = (B08 - B11)/(B08 + B11) requires calculating from optical bands B08 (index 8) and B11 (index 11) or querying raw Sentinel-2 L2A tiles via Copernicus Data Space / Planetary Computer.

---

## 4. Dataset 3: IP102 (Insect Pest Detection)

### A. Current Status & Required Actions
- **Current Folder Contents:**
  - `classes.txt`: Defines the 102 insect pest taxonomic categories.
  - `README.md`: Official repository description.
  - `pretrained_models/`: Pretrained model notes.
- **Missing Data (Training Blocked):**
  - **Classification Archive / Directory:** `images/` containing the 75,222 field images.
  - **Detection Archive / Directory:** `box/` containing the bounding box files for the ~18,974 object detection images.
  - **Split Lists:** `train.txt`, `val.txt`, `test.txt`.
- **Action Required by User:**
  1. Download the IP102 dataset archives from [GitHub: xpwu95/IP102](https://github.com/xpwu95/IP102) or Kaggle (`jessicali9530/ip102-dataset`).
  2. Extract `images/`, `box/`, and the split text files into `SmartCrop_AI/datasets/ip102/`.
  3. Re-run `python validate_datasets.py --dataset ip102` to verify images and bounding boxes before initiating YOLO detection training.
