# SmartCrop AI — Dataset-Wise Kaggle Artifact Inventory & Pre-Integration Specification

**Audit Date:** 2026-11-09T09:20:00+05:30  
**Status:** All Kaggle training runs complete; all checkpoints downloaded, checksummed, strict-loaded, and verified with dummy forward passes.  
**Inference Policy:** All non-PlantDoc endpoints are safely locked in **MOCK MODE** (`is_mock=True`) until each model's real inference service is individually integrated and tested.

---

## 1. Master Checkpoint Specification Matrix

| Modality / Task | Model Architecture | Checkpoint File & Size | SHA256 Checksum | Class Count | Input Resolution / Tensor Shape | Matching Class Mapping | Verified Benchmark Metric |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| **PlantDoc Leaf Disease** | `convnext_small.fb_in22k_ft_in1k` | [`models/disease/disease_model_exp_D_best.pth`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/disease/disease_model_exp_D_best.pth)<br>(188.87 MB) | `b84ee660bad5fa775261168994398e4102d804836aeebb0a0c1f9c22c1cc5807` | 27 | `[B, 3, 384, 384]`<br>(Resize 416 $\to$ Crop 384) | [`models/disease/classes.json`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/disease/classes.json)<br>[`models/disease/disease_classes.json`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/disease/disease_classes.json) | **81.20%** Test Accuracy<br>(78.85% Val EMA) |
| **IP102 Pest Classification** | `convnext_small.fb_in22k_ft_in1k` | [`models/pest/ip102_convnext_best.pth`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/pest/ip102_convnext_best.pth)<br>(189.08 MB) | `3d905f95f0b5cd9a3b713ed7c52f266148f27eb83ea506e7bee909a518331440` | 102 | `[B, 3, 256, 256]`<br>(ImageNet normalized) | [`models/pest/classes.json`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/pest/classes.json)<br>[`datasets/ip102/classes.txt`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/datasets/ip102/classes.txt) | **76.89%** Test Accuracy<br>(Top-5: **94.36%**, F1: **76.47%**) |
| **IP102 Pest Detection** | Ultralytics YOLOv8 (`detect`) | [`models/pest/yolov8_pest_best.pt`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/pest/yolov8_pest_best.pt)<br>(21.52 MB) | `a0a2ab353cac1b5a161b0d65120d249eac3245f9e80a26c7ff4fea51d0c02ca0` | 102 | `[B, 3, 640, 640]`<br>(RGB in-field) | [`models/pest/classes.json`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/pest/classes.json)<br>YOLO Model Names (0–101) | **65.33%** mAP@0.50<br>(41.18% mAP50-95, 30.9 ms) |
| **CropHarvest Binary** | 3-Seed `TSTransformer`<br>& `CropHarvestBiLSTM` | [`models/cropharvest/binary/transformer_seeds.pth`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/cropharvest/binary/transformer_seeds.pth)<br>(2.86 MB) | `c1dfa05fecfd9f80f740a97893678f0d047e63b9d188a1cded0a0e99efec1dd8` | 2 | `[B, 12, 18]`<br>(12 months × 18 channels) | [`models/cropharvest/binary/classes.json`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/cropharvest/binary/classes.json)<br>(0: Non-Crop, 1: Crop) | **82.02%** Test Accuracy<br>(81.18% Macro-F1) |
| **CropHarvest 10-Class** | `CropHarvestBiLSTM`<br>& Family Ensemble | [`models/cropharvest/10class/best_model.pth`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/cropharvest/10class/best_model.pth)<br>(0.59 MB) | `8dcac4be2f7c25cbdf9dc68e185b26bcebd9d8d510790208b0ad901aa763c9e6` | 10 | `[B, 12, 18]`<br>(12 months × 18 channels) | [`models/cropharvest/10class/classes.json`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/cropharvest/10class/classes.json)<br>(9 Crops + 1 Non-Crop) | **68.22%** Test Accuracy<br>(Family Ensemble Benchmark) |

---

## 2. Dataset-Wise Inventory & Verification

### A. PlantDoc Foliar Disease Classification
* **Model Checkpoint:** `models/disease/disease_model_exp_D_best.pth` (188.87 MB)
* **Status:** **VERIFIED & LIVE INFERENCE ACTIVE**
* **Strict Loading:** Passed with 0 missing and 0 unexpected keys (`strict=True`).
* **Dummy Inference:** Passed with shape `[1, 27]`, finite logits, valid softmax sum.
* **Class Mapping:** Exactly 27 classes mapped from `0: Apple Scab Leaf` to `26: Tomato leaf yellow virus`.
* **Executed Notebook:** [`notebooks/01_PlantDoc_Classification.ipynb`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/notebooks/01_PlantDoc_Classification.ipynb) (8 outputs).
* **Metrics File:** [`models/disease/disease_metrics.json`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/disease/disease_metrics.json) (81.20% test accuracy).
* **Missing Files:** None.

---

### B. IP102 Agricultural Insect Pest Classification
* **Model Checkpoint:** `models/pest/ip102_convnext_best.pth` (189.08 MB)
* **Secondary Alias:** `models/pest/pest_model_best.pth` (byte-for-byte identical, 189.08 MB).
* **Status:** **VERIFIED & READY FOR INTEGRATION** (currently locked in mock mode).
* **Strict Loading:** Passed with 0 missing and 0 unexpected keys (`strict=True`).
* **Dummy Inference:** Passed with shape `[1, 102]`, finite logits, valid softmax sum.
* **Class Mapping:** Exactly 102 classes mapped from `0: rice leaf roller` to `101: Cicadellidae`. Saved in [`models/pest/classes.json`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/pest/classes.json).
* **Executed Notebook:** [`notebooks/02_IP102_Classification.ipynb`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/notebooks/02_IP102_Classification.ipynb) (13 outputs, 14 epochs training logs).
* **Metrics File:** [`models/pest/pest_metrics.json`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/pest/pest_metrics.json) (76.89% test accuracy, 94.36% top-5 accuracy).
* **Per-Class Breakdown:** [`outputs/ip102/ip102_per_class_val.csv`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/outputs/ip102/ip102_per_class_val.csv) (all 102 species).
* **Missing Files:** None.

---

### C. IP102 Agricultural Insect Pest Object Detection
* **Model Checkpoint:** `models/pest/yolov8_pest_best.pt` (21.52 MB)
* **Secondary Checkpoint:** `outputs/ip102_detection/last.pt` (21.52 MB).
* **Status:** **VERIFIED & READY FOR INTEGRATION** (currently locked in mock mode).
* **Strict Loading:** Passed via Ultralytics YOLO (`task='detect'`).
* **Dummy Inference:** Passed on `640 x 640` input, returning bounding box and class score structures.
* **Class Mapping:** 102 species mapped, aligned with [`models/pest/classes.json`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/pest/classes.json).
* **Executed Notebook:** [`notebooks/03_IP102_Detection.ipynb`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/notebooks/03_IP102_Detection.ipynb) (10 outputs, complete training & validation curves).
* **Metrics File:** [`models/pest/detection_metrics.json`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/pest/detection_metrics.json), [`outputs/ip102_detection/final_metrics.csv`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/outputs/ip102_detection/final_metrics.csv).
* **Visualizations:** [`outputs/ip102_detection/training_curves.png`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/outputs/ip102_detection/training_curves.png), [`outputs/ip102_detection/sample_detections.png`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/outputs/ip102_detection/sample_detections.png), [`outputs/ip102_detection/confusion_matrix_top20.png`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/outputs/ip102_detection/confusion_matrix_top20.png).
* **Missing Files:** None.

---

### D. CropHarvest Binary Crop Detection
* **Primary Checkpoint:** `models/cropharvest/binary/transformer_seeds.pth` (2.86 MB, 3-seed ensemble).
* **Secondary Checkpoint:** `models/cropharvest/binary/best_model.pth` (0.58 MB, single BiLSTM model).
* **Status:** **VERIFIED & READY FOR INTEGRATION** (currently locked in mock mode).
* **Strict Loading:** Passed for both `TSTransformer` (45 keys) and `CropHarvestBiLSTM` (22 keys).
* **Dummy Inference:** Passed on `[1, 12, 18]` satellite array, shape `[1, 2]`.
* **Class Mapping:** [`models/cropharvest/binary/classes.json`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/cropharvest/binary/classes.json) (`0: Non-Crop`, `1: Crop`).
* **Executed Notebook:** [`notebooks/09_CropHarvest_Two_Step_Pipeline.ipynb`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/notebooks/09_CropHarvest_Two_Step_Pipeline.ipynb) (9 outputs).
* **Metrics File:** [`models/cropharvest/binary/metrics.json`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/cropharvest/binary/metrics.json) (82.02% test accuracy, 81.18% macro-F1).
* **Missing Files:** None.

---

### E. CropHarvest 10-Class Classification
* **Model Checkpoint:** `models/cropharvest/10class/best_model.pth` (0.59 MB).
* **Status:** **VERIFIED & READY FOR INTEGRATION** (currently locked in mock mode).
* **Strict Loading:** Passed for `CropHarvestBiLSTM` with configuration `{input_dim: 18, hidden_dim: 64, num_layers: 2, num_classes: 10}`.
* **Dummy Inference:** Passed on `[1, 12, 18]` satellite array, shape `[1, 10]`.
* **Class Mapping:** [`models/cropharvest/10class/classes.json`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/cropharvest/10class/classes.json) (`cereals`, `other`, `fruits_nuts`, `vegetables_melons`, `oilseeds`, `leguminous`, `root_tuber`, `beverage_spice`, `sugar`, `non_crop`).
* **Metrics File:** [`models/cropharvest/10class/metrics.json`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/cropharvest/10class/metrics.json) (68.22% family ensemble benchmark, 67.29% direct BiLSTM benchmark).
* **Per-Class Breakdown:** [`outputs/cropharvest/10class/per_class_validation_report.csv`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/outputs/cropharvest/10class/per_class_validation_report.csv) (8,488 validation samples).
* **Confusion Matrix:** [`outputs/cropharvest/v2_confusion_matrix.png`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/outputs/cropharvest/v2_confusion_matrix.png).
* **Missing Files:** None.

---

## 3. Discrepancy & Class Mapping Audit

1. **Class Mapping Files Synchronized:**
   - Generated `models/pest/classes.json` containing stripped names for all 102 species.
   - Synchronized `models/disease/classes.json` with `models/disease/disease_classes.json`.
   - Identified and resolved the single trailing space in class 83 (`Tetradacus c Bactrocera minax`) from YOLO training labels.
2. **Missing Dependencies:** **None.** All required packages (`torch`, `torchvision`, `timm`, `ultralytics`, `scikit-learn`, `pandas`, `fastapi`, `streamlit`) are confirmed active.
3. **Running Sessions:** No active training processes were interrupted; all checkpoints evaluated are authentic offline files.

---

## 4. Next Integration Steps

1. **Step 1 — IP102 Classification Service:**  
   Implement `app/services/ip102_service.py` to load `models/pest/ip102_convnext_best.pth` and connect `POST /api/predict/ip102/classify`.
2. **Step 2 — IP102 Detection Service:**  
   Implement `app/services/ip102_detection_service.py` using Ultralytics YOLO to load `models/pest/yolov8_pest_best.pt` and connect `POST /api/predict/ip102/detect`.
3. **Step 3 — CropHarvest Satellite Service:**  
   Implement `app/services/cropharvest_service.py` to handle both binary (`transformer_seeds.pth`) and 10-class (`best_model.pth`) time-series arrays (.h5 / .npy).
4. **Step 4 — Regression & Verification:**  
   Re-run the full FastAPI test suite verifying that each endpoint switches from `is_mock=True` to `is_mock=False` only upon passing real inference verification.

