# SmartCrop AI — Complete Kaggle Artifact Verification & Comprehensive Audit

**Audit Date:** 2026-11-09T08:35:00+05:30  
**Audit Scope:** Comprehensive inspection, SHA256 integrity verification, strict parameter loading, and dummy inference forward-pass validation across all five Machine Learning pipelines in SmartCrop AI.

---

## 1. Executive Summary & Readiness Matrix

| Pipeline | Checkpoint Found | Architecture Identified | Strict Loading | Dummy Forward Pass | Class Mapping Verified | Metrics Verified | Missing Dependencies | Integration Readiness |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. PlantDoc Disease Classification** | `disease_model_exp_D_best.pth` (188.87 MB) | `convnext_small.fb_in22k_ft_in1k` | ✅ **PASS** (`strict=True`) | ✅ **PASS** (`[1, 27]`, finite, $\sum p=1.0$) | ✅ **VERIFIED** (27 classes) | ✅ **VERIFIED** (81.20% Test) | None (`timm`, `torch`) | 🚀 **CONNECTED & LIVE** |
| **2. IP102 Pest Classification** | `ip102_convnext_best.pth` (189.08 MB) | `convnext_small.fb_in22k_ft_in1k` | ✅ **PASS** (`strict=True`) | ✅ **PASS** (`[1, 102]`, finite, $\sum p=1.0$) | ✅ **VERIFIED** (102 species) | ✅ **VERIFIED** (76.89% Test) | None (`timm`, `torch`) | 🟢 **READY FOR INTEGRATION** |
| **3. IP102 Pest Detection** | `yolov8_pest_best.pt` (21.52 MB) | Ultralytics YOLOv8 Detection | ✅ **PASS** (Native Ultralytics) | ✅ **PASS** (`boxes` object returned) | ✅ **VERIFIED** (102 species) | ✅ **VERIFIED** (65.33% mAP50) | None (`ultralytics`) | 🟢 **READY FOR INTEGRATION** |
| **4. CropHarvest Binary Classification** | `transformer_seeds.pth` (2.86 MB) & `best_model.pth` (0.58 MB) | 3-Seed `TSTransformer` & `CropHarvestBiLSTM` | ✅ **PASS** (`strict=True`) | ✅ **PASS** (`[1, 2]`, finite, $\sum p=1.0$) | ✅ **VERIFIED** (2 classes) | ✅ **VERIFIED** (82.02% Test) | None (`torch`) | 🟢 **READY FOR INTEGRATION** |
| **5. CropHarvest 10-Class Classification** | `best_model.pth` (0.59 MB) | `CropHarvestBiLSTM` | ✅ **PASS** (`strict=True`) | ✅ **PASS** (`[1, 10]`, finite, $\sum p=1.0$) | ✅ **VERIFIED** (10 categories) | ✅ **VERIFIED** (68.22% Benchmark) | None (`torch`) | 🟢 **READY FOR INTEGRATION** |

---

## 2. In-Depth Pipeline Audit Details

---

### Pipeline 1: PlantDoc Foliar Disease Classification (Experiment D)

1. **Checkpoint & Artifact Inventory:**
   - **Primary Checkpoint:** [`models/disease/disease_model_exp_D_best.pth`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/disease/disease_model_exp_D_best.pth)
   - **File Size:** `198,041,037 bytes` (188.87 MB)
   - **SHA256 Hash:** `b84ee660bad5fa775261168994398e4102d804836aeebb0a0c1f9c22c1cc5807`
   - **Metrics Files:** [`models/disease/disease_metrics.json`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/disease/disease_metrics.json), [`outputs/plantdoc_experiment_D_boosted_metrics.json`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/outputs/plantdoc_experiment_D_boosted_metrics.json)
   - **Executed Notebook:** [`notebooks/01_PlantDoc_Classification.ipynb`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/notebooks/01_PlantDoc_Classification.ipynb) (8 outputs)
2. **Architecture & Python Dependencies:**
   - **Model Backbone:** `timm.create_model("convnext_small.fb_in22k_ft_in1k", pretrained=False, num_classes=27)`
   - **Required Packages:** `torch>=2.0.0`, `torchvision`, `timm==1.0.31`, `pillow`
3. **Checkpoint Keys & Class Count:**
   - Contains 344 parameter tensors including `stem.0.weight`, `stages.*`, `head.fc.weight` (`[27, 768]`), `head.fc.bias` (`[27]`).
   - Class count: Exactly **27 foliar classes**.
4. **Strict Checkpoint Loading Result:**
   - **VERIFIED:** `model.load_state_dict(sd, strict=True)` loaded with 0 missing and 0 unexpected keys.
5. **Dummy Forward Pass & Probability Validation:**
   - Input: `[1, 3, 384, 384]` float32 tensor normalized with ImageNet mean/std.
   - Output Logits Shape: `[1, 27]`.
   - Logits are completely finite (`torch.isfinite(out).all() == True`).
   - Softmax probabilities sum to `1.0000`.
6. **Class Mapping Verification:**
   - Verified against [`models/disease/disease_classes.json`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/disease/disease_classes.json). Index 0: `Apple Scab Leaf` $\to$ Index 26: `Tomato leaf yellow virus`.
7. **Evaluation Metrics Breakdown:**
   - **Training Accuracy:** 98.71%
   - **Validation Accuracy (EMA):** 78.85%
   - **Official Held-Out Test Accuracy:** **81.20%** (evaluated strictly on 239 held-out images).
8. **Real Model vs Dummy Check:**
   - **REAL TRAINED MODEL:** Non-zero, converged float32 weights trained over 36 epochs with EMA.
9. **Status:**
   - Already connected to real production inference in `app/services/plantdoc_service.py` via `POST /api/predict/plantdoc`.

---

### Pipeline 2: IP102 Agricultural Insect Pest Classification

1. **Checkpoint & Artifact Inventory:**
   - **Primary Checkpoint:** [`models/pest/ip102_convnext_best.pth`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/pest/ip102_convnext_best.pth)
   - **Secondary Alias:** [`models/pest/pest_model_best.pth`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/pest/pest_model_best.pth) (identical byte-for-byte)
   - **File Size:** `198,269,815 bytes` (189.08 MB)
   - **SHA256 Hash:** `3d905f95f0b5cd9a3b713ed7c52f266148f27eb83ea506e7bee909a518331440`
   - **Metrics Files:** [`models/pest/pest_metrics.json`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/pest/pest_metrics.json), [`outputs/ip102/ip102_metrics_table.csv`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/outputs/ip102/ip102_metrics_table.csv), [`models/pest/ip102_per_class_val.csv`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/pest/ip102_per_class_val.csv)
   - **Executed Notebook:** [`notebooks/02_IP102_Classification.ipynb`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/notebooks/02_IP102_Classification.ipynb) (13 outputs, full training logs)
2. **Architecture & Python Dependencies:**
   - **Model Backbone:** `timm.create_model("convnext_small.fb_in22k_ft_in1k", pretrained=False, num_classes=102)`
   - **Input Resolution:** `256 x 256`
   - **Required Packages:** `torch>=2.0.0`, `timm==1.0.31`, `pillow`, `pandas`
3. **Checkpoint Keys & Class Count:**
   - Dictionary keys: `['model_state_dict', 'source', 'epoch', 'val_acc']`.
   - Epoch: `13`, `val_acc: 0.7718`.
   - Contains 344 parameter tensors including `head.fc.weight` (`[102, 768]`), `head.fc.bias` (`[102]`).
   - Class count: Exactly **102 insect species**.
4. **Strict Checkpoint Loading Result:**
   - **VERIFIED:** `model.load_state_dict(sd, strict=True)` loaded cleanly with 0 mismatches.
5. **Dummy Forward Pass & Probability Validation:**
   - Input: `[1, 3, 256, 256]` float32 tensor normalized with ImageNet mean/std.
   - Output Logits Shape: `[1, 102]`.
   - Logits are finite (`torch.isfinite(out).all() == True`).
   - Softmax probabilities sum to `1.0000`.
6. **Class Mapping Verification:**
   - Verified against [`datasets/ip102/classes.txt`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/datasets/ip102/classes.txt) and [`outputs/ip102/ip102_per_class_val.csv`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/outputs/ip102/ip102_per_class_val.csv). Index 0: `rice leaf roller` $\to$ Index 101: `Cicadellidae`.
7. **Evaluation Metrics Breakdown:**
   - **Training Accuracy (10k clean subset):** 92.11% (Top-5: 99.82%)
   - **Validation Accuracy (Full split):** 77.33% (Top-5: 94.53%, Weighted F1: 76.85%)
   - **Official Held-Out Test Accuracy:** **76.89%** (Top-5: **94.36%**, Weighted F1: **76.47%**, Macro F1: **69.55%**).
8. **Real Model vs Dummy Check:**
   - **REAL TRAINED MODEL:** Full weights saved from converged Epoch 13 checkpoint with EMA.
9. **Status:**
   - Checkpoint fully verified locally; currently held in mock mode pending integration instruction.

---

### Pipeline 3: IP102 Agricultural Insect Pest Object Detection (YOLOv8)

1. **Checkpoint & Artifact Inventory:**
   - **Primary Checkpoint:** [`models/pest/yolov8_pest_best.pt`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/pest/yolov8_pest_best.pt)
   - **File Size:** `22,569,642 bytes` (21.52 MB)
   - **SHA256 Hash:** `a0a2ab353cac1b5a161b0d65120d249eac3245f9e80a26c7ff4fea51d0c02ca0`
   - **Last Checkpoint:** [`outputs/ip102_detection/last.pt`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/outputs/ip102_detection/last.pt) (21.52 MB)
   - **Metrics Files:** [`models/pest/detection_metrics.json`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/pest/detection_metrics.json), [`outputs/ip102_detection/final_metrics.csv`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/outputs/ip102_detection/final_metrics.csv), [`outputs/ip102_detection/per_class_AP.csv`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/outputs/ip102_detection/per_class_AP.csv)
   - **Executed Notebook:** [`notebooks/03_IP102_Detection.ipynb`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/notebooks/03_IP102_Detection.ipynb) (10 outputs)
2. **Architecture & Python Dependencies:**
   - **Model Backbone:** `Ultralytics YOLOv8` (Detection Head, 102 classes)
   - **Input Resolution:** `640 x 640`
   - **Required Packages:** `ultralytics==8.4.208`, `torch>=2.0.0`, `opencv-python`, `pillow`
3. **Checkpoint Keys & Class Count:**
   - Native PyTorch pickled dictionary containing `model` (instance of `ultralytics.nn.tasks.DetectionModel`), `epoch`, `date`, `args`.
   - Model names attribute: exactly **102 classes** (`0: rice leaf roller` $\dots$ `101: Cicadellidae`).
4. **Strict Checkpoint Loading Result:**
   - **VERIFIED:** Successfully loaded via `YOLO("models/pest/yolov8_pest_best.pt")`. Model task identified as `'detect'`.
5. **Dummy Forward Pass Validation:**
   - Input: `640 x 640 x 3` uint8 image array.
   - Evaluated without errors; successfully returns bounding box, confidence, and class prediction structures.
6. **Class Mapping Verification:**
   - Matches official IP102 dataset taxonomy with 102 species.
7. **Evaluation Metrics Breakdown:**
   - **mAP@0.50:** **65.33%**
   - **mAP@0.50:0.95:** **41.18%**
   - **Validation Accuracy (Image-Level):** 74.34%
   - **Detection F1 (IoU $\ge$ 0.5):** 70.82% (Precision: 68.51%, Recall: 73.29%)
   - **Mean Matched IoU:** 0.811
   - **Inference Latency:** 30.9 ms (32.4 FPS on edge / 6.2 ms Ultralytics val)
8. **Real Model vs Dummy Check:**
   - **REAL TRAINED MODEL:** Authentic trained YOLOv8 weights with verified AP per class.
9. **Status:**
   - Checkpoint fully verified locally; currently held in mock mode pending integration instruction.

---

### Pipeline 4: CropHarvest Binary Crop Detection (Satellite Phenology)

1. **Checkpoint & Artifact Inventory:**
   - **Primary Checkpoint:** [`models/cropharvest/binary/transformer_seeds.pth`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/cropharvest/binary/transformer_seeds.pth)
   - **File Size:** `3,003,891 bytes` (2.86 MB)
   - **SHA256 Hash:** `c1dfa05fecfd9f80f740a97893678f0d047e63b9d188a1cded0a0e99efec1dd8`
   - **Secondary Checkpoint (BiLSTM):** [`models/cropharvest/binary/best_model.pth`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/cropharvest/binary/best_model.pth) (611,637 bytes, SHA256: `bd2e6b930d4a0f813f1e904f401b8c13d18b9ed7caa9c2cd0545ab561de88169`)
   - **Metrics Files:** [`models/cropharvest/binary/metrics.json`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/cropharvest/binary/metrics.json), [`outputs/cropharvest/two_step/step1_binary_metrics_table.csv`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/outputs/cropharvest/two_step/step1_binary_metrics_table.csv)
   - **Executed Notebook:** [`notebooks/09_CropHarvest_Two_Step_Pipeline.ipynb`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/notebooks/09_CropHarvest_Two_Step_Pipeline.ipynb) (9 outputs)
2. **Architecture & Python Dependencies:**
   - **Primary Architecture:** Multi-seed `TSTransformer` (in_dim=18, d_model=96, nhead=4, 3 layers, num_classes=2)
   - **Secondary Architecture:** `CropHarvestBiLSTM` (input_dim=18, hidden_dim=64, num_layers=2, num_classes=2)
   - **Input Modality:** 12 monthly timesteps $\times$ 18 multi-spectral channels
   - **Required Packages:** `torch>=2.0.0`, `numpy`, `h5py`
3. **Checkpoint Keys & Class Count:**
   - `transformer_seeds.pth` contains a `list` of 3 distinct seed state dictionaries (45 keys each, head `[2, 96]`).
   - Class count: Exactly **2 classes** (`0: Non-Crop`, `1: Crop`).
4. **Strict Checkpoint Loading Result:**
   - **VERIFIED:** `TSTransformer.load_state_dict(seeds[0], strict=True)` loaded with 0 mismatches.
   - **VERIFIED:** `CropHarvestBiLSTM.load_state_dict(ckpt['model_state_dict'], strict=True)` loaded with 0 mismatches.
5. **Dummy Forward Pass & Probability Validation:**
   - Input: `[1, 12, 18]` float32 tensor.
   - Output Shape: `[1, 2]`.
   - Logits are finite (`torch.isfinite(out).all() == True`).
   - Softmax probabilities sum to `1.0000`.
6. **Class Mapping Verification:**
   - Verified against [`models/cropharvest/binary/classes.json`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/cropharvest/binary/classes.json) (`0: Non-Crop`, `1: Crop`).
7. **Evaluation Metrics Breakdown:**
   - **Training Accuracy:** 94.52% (Macro-F1: 93.44%)
   - **Validation Accuracy:** 87.24% (Macro-F1: 84.70%)
   - **Official Held-Out Test Accuracy:** **82.02%** (Macro-F1: **81.18%**).
8. **Real Model vs Dummy Check:**
   - **REAL TRAINED MODEL:** Full weights saved from 3-seed ensemble run.
9. **Status:**
   - Checkpoint fully verified locally; currently held in mock mode pending integration instruction.

---

### Pipeline 5: CropHarvest 10-Class Classification (9 FAO Crops + Land Cover)

1. **Checkpoint & Artifact Inventory:**
   - **Primary Checkpoint:** [`models/cropharvest/10class/best_model.pth`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/cropharvest/10class/best_model.pth)
   - **File Size:** `613,941 bytes` (0.59 MB)
   - **SHA256 Hash:** `8dcac4be2f7c25cbdf9dc68e185b26bcebd9d8d510790208b0ad901aa763c9e6`
   - **Metrics Files:** [`models/cropharvest/10class/metrics.json`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/cropharvest/10class/metrics.json), [`outputs/cropharvest/10class/v2_metrics.json`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/outputs/cropharvest/10class/v2_metrics.json), [`outputs/cropharvest/10class/per_class_validation_report.csv`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/outputs/cropharvest/10class/per_class_validation_report.csv)
   - **Confusion Matrix:** [`outputs/cropharvest/v2_confusion_matrix.png`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/outputs/cropharvest/v2_confusion_matrix.png)
   - **Benchmark Report:** [`outputs/cropharvest/CROPHARVEST_ABLATION_BENCHMARK.md`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/outputs/cropharvest/CROPHARVEST_ABLATION_BENCHMARK.md)
2. **Architecture & Python Dependencies:**
   - **Model Backbone:** `CropHarvestBiLSTM` (input_dim=18, hidden_dim=64, num_layers=2, num_classes=10)
   - **Benchmark Architecture:** Multi-Model Family Ensemble (Transformer + 1D-CNN + BiGRU + LightGBM)
   - **Input Modality:** 12 monthly timesteps $\times$ 18 multi-spectral channels
   - **Required Packages:** `torch>=2.0.0`, `numpy`, `h5py`, `pandas`
3. **Checkpoint Keys & Class Count:**
   - Checkpoint dictionary contains `['epoch', 'model_state_dict', 'val_accuracy', 'val_macro_f1', 'class_mapping', 'model_config']`.
   - `model_state_dict` has 22 tensors including `classifier.3.weight` (`[10, 64]`), `classifier.3.bias` (`[10]`).
   - Class count: Exactly **10 categories**.
4. **Strict Checkpoint Loading Result:**
   - **VERIFIED:** `CropHarvestBiLSTM.load_state_dict(ckpt['model_state_dict'], strict=True)` loaded with 0 missing and 0 unexpected keys.
5. **Dummy Forward Pass & Probability Validation:**
   - Input: `[1, 12, 18]` float32 tensor.
   - Output Shape: `[1, 10]`.
   - Logits are finite (`torch.isfinite(out).all() == True`).
   - Softmax probabilities sum to `1.0000`.
6. **Class Mapping Verification:**
   - Verified against [`models/cropharvest/10class/classes.json`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/cropharvest/10class/classes.json):
     `cereals`, `other`, `fruits_nuts`, `vegetables_melons`, `oilseeds`, `leguminous`, `root_tuber`, `beverage_spice`, `sugar`, `non_crop`.
7. **Evaluation Metrics Breakdown:**
   - **Multi-Model Family Ensemble:**
     - Training: 85.95%
     - Validation: 74.31%
     - Official Test: **68.22%** (Macro-F1: 32.14%)
   - **Direct 10-Class BiLSTM Ensemble:**
     - Training: 93.36%
     - Validation: 74.20%
     - Official Test: **67.29%** (Macro-F1: 32.39%)
8. **Real Model vs Dummy Check:**
   - **REAL TRAINED MODEL:** Full weights saved from converged BiLSTM run with 8,488 validation samples evaluated.
9. **Status:**
   - Checkpoint fully verified locally; currently held in mock mode pending integration instruction.

---

## 3. Dependency & Environment Health Check

All required libraries have been inspected and confirmed installed in the active environment:

| Library | Installed Version | Status |
| :--- | :---: | :---: |
| `torch` | 2.14.1+cpu | ✅ Installed & Verified |
| `torchvision` | 0.29.2+cpu | ✅ Installed & Verified |
| `timm` | 1.0.31 | ✅ Installed & Verified |
| `ultralytics` | 8.4.208 | ✅ Installed & Verified |
| `numpy` | 2.1.2 | ✅ Installed & Verified |
| `pandas` | 2.2.3 | ✅ Installed & Verified |
| `pydantic` | 2.12.5 | ✅ Installed & Verified |
| `fastapi` | 0.135.1 | ✅ Installed & Verified |
| `streamlit` | 1.55.0 | ✅ Installed & Verified |
| `scikit-learn` | 1.5.2 | ✅ Installed & Verified |

**Missing Dependencies:** **NONE.**

---

## 4. Current Inference Route Status in `app/api/routes.py`

* `POST /api/predict/plantdoc`: **REAL MODEL INFERENCE** (`convnext_small.fb_in22k_ft_in1k`, `is_mock=False`).
* `POST /api/predict/ip102/classify`: **MOCK MODE** (`is_mock=True`, verified weights ready for wiring).
* `POST /api/predict/ip102/detect`: **MOCK MODE** (`is_mock=True`, verified weights ready for wiring).
* `POST /api/predict/cropharvest/binary`: **MOCK MODE** (`is_mock=True`, verified weights ready for wiring).
* `POST /api/predict/cropharvest/10class`: **MOCK MODE** (`is_mock=True`, verified weights ready for wiring).

All mock safety constraints remain intact.

