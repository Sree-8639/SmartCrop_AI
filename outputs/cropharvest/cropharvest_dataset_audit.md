# CropHarvest Satellite Crop Classification Dataset Audit Report
**SmartCrop AI Project — Verified Dataset Status**  
*Audit Timestamp:* `2026-11-08 16:11:37`

---

## 1. Executive Summary
- **Dataset Domain:** Global Satellite Earth Observation Time-Series for Crop Classification.
- **Satellite Features:** 18 channels (Sentinel-1 SAR, Sentinel-2 Optical Multi-Spectral, ERA5 Climate, SRTM Topography).
- **Temporal Resolution:** 12 monthly composite steps ($T=12$, $C=18$).
- **Total Local Array Files:** **87,464** individual HDF5 arrays verified in `arrays/`.
- **Labels Matching Coverage:** **100.0%** (87,464 / 87,464) exact correspondence between `arrays/` and `labels.geojson`.
- **Integrity Status:** **0 NaNs, 0 Infs, 0 Corruptions** across verified samples.
- **Supervised Classification Suitability:** **FULLY SUITABLE**. Supports both binary crop detection and multi-class crop type categorization.

---

## 2. Dataset Structure & Array Characteristics
| Property | Verified Value | Description |
| :--- | :--- | :--- |
| **Array File Count** | `87,464` | HDF5 files in `datasets/cropharvest/arrays/` |
| **Array Dimensions** | `(12, 18)` | 12 monthly timesteps $\times$ 18 satellite/weather features |
| **Data Type** | `float64` | Calibrated reflectances, backscatter (dB), and climate metrics |
| **Missing Values** | `0 NaNs, 0 Infs` | Verified across random audit samples |
| **Normalizing Constants** | `mean (18,), std (18,)` | Provided in `normalizing_dict.h5` |

---

## 3. Official Benchmark Splits
| Split Category | Sample Count | Percentage | Benchmark Role |
| :--- | :--- | :--- | :--- |
| **Train/Validation Pool (`is_test=False`)** | **82,654** | 94.5% | Global model training and internal validation |
| **Official Held-Out Test (`is_test=True`)** | **4,810** | 5.5% | Strict held-out evaluation test set |
| **Togo Regional Evaluation (`togo-eval.h5`)** | **306** | Dedicated Testbed | Zero-shot / out-of-domain transfer evaluation |
| **China Regional Evaluation (`china-crop.h5`)** | **4397** | Dedicated Testbed | Large-scale regional evaluation |

---

## 4. Class Label Distribution (Binary Crop Detection)
All **87,464** files possess unambiguous ground truth binary labels:
- **Crop (`is_crop = 1`):** 60,521 samples (69.2%)
- **Non-Crop (`is_crop = 0`):** 26,943 samples (30.8%)

---

## 5. Fine-Grained Multi-Class Crop Groups
A total of **42,603** samples contain specific FAO classification groups:
- **cereals:** 13,259 samples
- **other:** 8,406 samples
- **non_crop:** 5,533 samples
- **fruits_nuts:** 4,110 samples
- **vegetables_melons:** 3,755 samples
- **oilseeds:** 2,868 samples
- **leguminous:** 1,837 samples
- **root_tuber:** 1,179 samples
- **beverage_spice:** 908 samples
- **sugar:** 748 samples

---

## 6. Official Split Preservation & Safety Rules
1. Samples flagged with `is_test: true` ({geo['official_is_test_counts'].get(True, 0):,} samples) must **never** be used for model training or hyperparameter tuning.
2. `togo-eval.h5` (306 samples) and `china-crop.h5` (4,397 samples) must remain strictly isolated as final benchmark evaluation testbeds.
3. The training pool (`is_test: false`, 82,654 samples) is partitioned deterministically into:
   - **Training Set (80%):** ~66,123 samples
   - **Validation Set (20%):** ~16,531 samples
   using stratified sampling with fixed seed `42`.
