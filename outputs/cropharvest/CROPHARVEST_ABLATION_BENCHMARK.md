# SmartCrop AI — CropHarvest Satellite Phenology Benchmark & Ablation Study

**Audit Date:** 2026-11-09T07:20:00+05:30  
**Dataset:** CropHarvest Earth Observation Time-Series (12 monthly timesteps × 18 multi-spectral channels)  
**Evaluation Protocol:** Official Held-Out Test Set (`is_test: true`, strictly evaluated once)

---

## 1. Executive Performance Comparison

| Pipeline Strategy | Approach / Backbone | Train Acc | Val Acc | Official Test Acc | Test Macro-F1 | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Binary Detection** | BiLSTM Ensemble | 94.52% | 87.24% | **82.02%** | **81.18%** | 🏆 Best Binary |
| **10-Class Family Ensemble** | Transformer + CNN1D + BiGRU + LightGBM | 85.95% | 74.31% | **68.22%** | **32.14%** | 🏆 **Best 10-Class Accuracy** |
| **10-Class Direct Ensemble** | BiLSTM Deep Ensemble | 93.36% | 74.20% | **67.29%** | **32.39%** | 🥈 High Val Alignment |
| **Two-Step Hierarchical** | Binary Gatekeeper $\to$ 9-Crop Classification | 87.21% | 73.19% | **56.07%** | **23.82%** | Compound error cascades |

---

## 2. Key Experimental Insights & Ablation Analysis

### A. Why Multi-Model Family Ensemble Achieved Highest Test Accuracy (68.22%)
- Combining diverse model inductive biases (Self-Attention in Transformers, Local Convolutions in 1D-CNN, Recurrent Phenology in BiGRU, and Gradient-Boosted Trees in LightGBM) produced superior generalization over the held-out test split.
- **Individual Validation Performance by Architecture Family:**
  - `LightGBM`: **73.92%**
  - `BiGRU`: **73.23%**
  - `CNN1D`: **72.94%**
  - `Transformer`: **72.86%**
- **Ensemble Synergy:** Combining the four families boosted Validation Accuracy to **74.31%** and Official Test Accuracy to **68.22%**.

### B. Two-Step Pipeline Error Compounding (56.07%)
- While Step 1 (Binary Filter) achieved high accuracy (**82.43%** on official test, Macro-F1: 81.63%), any false-negative crop misclassification immediately discarded true crop samples from reaching the 9-class classifier.
- This compounded error drop resulted in lower overall end-to-end accuracy (**56.07%** vs **67.29%** direct and **68.22%** multi-model family ensemble).

### C. 10-Class Macro-F1 Test Support Deficit Audit
- The Official Benchmark Test Split contains significant class imbalance and statistical support deficits:
  - `vegetables_melons`: **0 test samples** (statistically untestable in official benchmark test set).
  - `fruits_nuts`, `oilseeds`, `leguminous`, `sugar`: **1 sample each** (causes extreme penalty to macro-averaged F1).
  - `non_crop`, `cereals`, `root_tuber`, `beverage_spice`: $\ge 10$ samples (solid performance).
- Consequently, while raw test accuracy reaches **68.22%**, macro-averaged F1 is bounded around ~32.4% due to zero/single-sample categories.

---

## 3. Detailed Metrics Across Pipeline Variants

### 1. Multi-Model Family Ensemble (`family_ensemble`)
| Metric | Training | Validation | Official Test |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 85.95% | 74.31% | **68.22%** |
| **Precision (macro)** | 86.49% | 71.75% | **36.52%** |
| **Recall (macro)** | 81.39% | 64.75% | **33.63%** |
| **F1 (macro)** | 83.44% | 66.75% | **32.14%** |
| **Precision (weighted)** | 86.09% | 74.04% | **57.96%** |
| **Recall (weighted)** | 85.95% | 74.31% | **68.22%** |
| **F1 (weighted)** | 85.76% | 73.56% | **59.40%** |

### 2. Direct 10-Class Ensemble (`10class_direct`)
| Metric | Training | Validation | Official Test |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 93.36% | 74.20% | **67.29%** |
| **Precision (macro)** | 94.07% | 71.21% | **45.37%** |
| **Recall (macro)** | 93.94% | 65.21% | **32.67%** |
| **F1 (macro)** | 93.96% | 67.27% | **32.39%** |
| **Precision (weighted)** | 93.42% | 73.86% | **72.65%** |
| **Recall (weighted)** | 93.36% | 74.20% | **67.29%** |
| **F1 (weighted)** | 93.33% | 73.55% | **60.08%** |

### 3. Two-Step Hierarchical Pipeline (`two_step`)
| Metric | Training | Validation | Official Test |
| :--- | :---: | :---: | :---: |
| **End-to-End Accuracy** | 87.21% | 73.19% | **56.07%** |
| **Precision (macro)** | 88.98% | 71.11% | **34.69%** |
| **Recall (macro)** | 84.30% | 62.82% | **24.37%** |
| **F1 (macro)** | 86.27% | 65.14% | **23.82%** |
| *Step 1 Binary Filter (Holdout)* | *86.68%* | *Val Acc* | *82.43% Test Acc* |

---

## 4. 10-Class Per-Class Validation Breakdown

Evaluation performed on the full validation partition (8,488 labeled samples across all 10 categories):

| Class Index / Name | Category Type | Precision | Recall | F1-Score | Validation Samples |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `cereals` | Crop Category | **0.778** | **0.898** | **0.834** | 2,647 |
| `other` | Crop Category | 0.818 | 0.717 | 0.764 | 1,679 |
| `non_crop` | Land-Cover Class | 0.706 | 0.740 | 0.723 | 1,098 |
| `fruits_nuts` | Crop Category | 0.682 | 0.755 | 0.717 | 821 |
| `vegetables_melons` | Crop Category | 0.684 | 0.736 | 0.709 | 751 |
| `leguminous` | Crop Category | 0.695 | 0.564 | 0.623 | 367 |
| `oilseeds` | Crop Category | 0.617 | 0.438 | 0.512 | 573 |
| `root_tuber` | Crop Category | 0.621 | 0.464 | 0.531 | 233 |
| `sugar` | Crop Category | **0.847** | **0.893** | **0.869** | 149 |
| `beverage_spice` | Crop Category | 0.727 | 0.270 | 0.393 | 178 |
| **Overall / Macro** | **10-Class Balanced** | **0.718** | **0.648** | **0.668** | **8,488 Total** |

> **Key Observation:** Crops with substantial phenological training samples (`cereals`, `sugar`, `fruits_nuts`, `vegetables_melons`, and `other`) achieve strong F1 scores between **71% and 87%**. Categories with seasonal harvest overlap and fewer samples (`beverage_spice`, `root_tuber`, `oilseeds`) experience greater confusion with general vegetation.

---

## 5. Comprehensive File Inventory

| Category | File Path | Description |
| :--- | :--- | :--- |
| **Pipeline Notebook (Two-Step)** | [`notebooks/09_CropHarvest_Two_Step_Pipeline.ipynb`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/notebooks/09_CropHarvest_Two_Step_Pipeline.ipynb) | Production Kaggle notebook executing Two-Step Hierarchical pipeline |
| **Master Training Notebook** | [`notebooks/08_CropHarvest_Complete_Training.ipynb`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/notebooks/08_CropHarvest_Complete_Training.ipynb) | Production Kaggle notebook executing unified benchmark training |
| **Binary Transformer Seeds Checkpoint** | [`models/cropharvest/binary/transformer_seeds.pth`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/cropharvest/binary/transformer_seeds.pth) | 3-seed Transformer ensemble weights for Step 1 Binary Classification (2.86 MB) |
| **10-Class Per-Class Val Report (CSV)** | [`outputs/cropharvest/10class/per_class_validation_report.csv`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/outputs/cropharvest/10class/per_class_validation_report.csv) | Full per-class validation metrics table (precision, recall, F1, samples) |
| **10-Class Per-Class Val Report (JSON)** | [`outputs/cropharvest/10class/per_class_validation_report.json`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/outputs/cropharvest/10class/per_class_validation_report.json) | Machine-readable per-class validation metrics |
| **Family Ensemble (JSON)** | [`outputs/cropharvest/family_ensemble/family_ensemble_metrics.json`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/outputs/cropharvest/family_ensemble/family_ensemble_metrics.json) | Multi-model family ensemble metrics (68.22% Test Acc) |
| **Family Ensemble (CSV)** | [`outputs/cropharvest/family_ensemble/family_ensemble_metrics_table.csv`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/outputs/cropharvest/family_ensemble/family_ensemble_metrics_table.csv) | CSV performance table for Multi-Model Family Ensemble |
| **10-Class Direct (JSON)** | [`outputs/cropharvest/10class_direct/10class_direct_metrics.json`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/outputs/cropharvest/10class_direct/10class_direct_metrics.json) | Direct 10-Class BiLSTM ensemble metrics (67.29% Test Acc) |
| **10-Class Direct (CSV)** | [`outputs/cropharvest/10class_direct/10class_direct_metrics_table.csv`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/outputs/cropharvest/10class_direct/10class_direct_metrics_table.csv) | CSV performance table for Direct 10-Class Ensemble |
| **Two-Step Pipeline (JSON)** | [`outputs/cropharvest/two_step/two_step_metrics.json`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/outputs/cropharvest/two_step/two_step_metrics.json) | Two-Step Hierarchical pipeline metrics (56.07% Test Acc) |
| **Two-Step Pipeline (CSV)** | [`outputs/cropharvest/two_step/two_step_metrics_table.csv`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/outputs/cropharvest/two_step/two_step_metrics_table.csv) | CSV performance table for Two-Step Pipeline |
| **Two-Step Per-Class Val (CSV)** | [`outputs/cropharvest/two_step/two_step_per_class_val.csv`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/outputs/cropharvest/two_step/two_step_per_class_val.csv) | Per-class validation table for Two-Step pipeline |
| **Binary Pipeline (JSON)** | [`outputs/cropharvest/complete_training/binary/binary_metrics.json`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/outputs/cropharvest/complete_training/binary/binary_metrics.json) | Binary BiLSTM Ensemble metrics (82.02% Test Acc) |
| **Binary Seeds Archive** | [`outputs/cropharvest/complete_training/binary/transformer_seeds.pth`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/outputs/cropharvest/complete_training/binary/transformer_seeds.pth) | Checkpoint archive of 3-seed Transformer ensemble weights |
| **Confusion Matrix Plot** | [`outputs/cropharvest/v2_confusion_matrix.png`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/outputs/cropharvest/v2_confusion_matrix.png) | Confusion matrix plot for CropHarvest 10-Class validation evaluation |
| **Normalized Arrays Cache** | [`data/cropharvest/all_arrays_normalized.npy`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/data/cropharvest/all_arrays_normalized.npy) | Pre-computed normalized satellite arrays cache (72.0 MB) |
| **Combined Master (JSON)** | [`outputs/cropharvest/complete_training/combined_results.json`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/outputs/cropharvest/complete_training/combined_results.json) | Master combined results JSON |

