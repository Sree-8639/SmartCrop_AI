# SmartCrop AI — IP102 Pest Classification Benchmark Report

**Audit Date:** 2026-11-09T07:40:00+05:30  
**Model Architecture:** `convnext_small.fb_in22k_ft_in1k` (256x256 resolution)  
**Dataset:** IP102 Macro Field Pest Dataset (102 agricultural insect & mite species)  
**Training Pipeline:** 14 Epochs, AdamW with Cosine Annealing, MixUp/CutMix (p=0.4), Label Smoothing (0.1), EMA (decay=0.999), Test-Time Augmentation (TTA).

---

## 1. Executive Performance Summary

| Split | Top-1 Accuracy | Top-5 Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Training (10k clean subset)** | 92.11% | 99.82% | 90.69% | 89.72% | 90.02% | 92.03% |
| **Validation (Full)** | 77.33% | 94.53% | 71.83% | 68.95% | 69.82% | 76.85% |
| **Official Held-Out Test** | **76.89%** | **94.36%** | **71.06%** | **68.79%** | **69.55%** | **76.47%** |

> **Highlight:** The ConvNeXt-Small backbone achieves **76.89% Top-1 Test Accuracy** and **94.36% Top-5 Test Accuracy** across all 102 fine-grained pest categories under real macro in-field conditions.

---

## 2. Checkpoint Verification & Architecture Integrity

| Property | Value | Verification Status |
| :--- | :--- | :---: |
| **Active Checkpoint File** | [`models/pest/ip102_convnext_best.pth`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/pest/ip102_convnext_best.pth) | **VERIFIED** |
| **File Size** | 198,269,815 bytes (189.09 MB) | **VERIFIED** |
| **Architecture Specification** | `timm.create_model("convnext_small.fb_in22k_ft_in1k", pretrained=False, num_classes=102)` | **VERIFIED** |
| **State Dict Keys** | 344 parameters (all weights strictly aligned) | **VERIFIED** |
| **Input Shape** | `[B, 3, 256, 256]` (Normalized with ImageNet mean/std) | **VERIFIED** |
| **Best Epoch** | Epoch 13 / 14 | **VERIFIED** |

---

## 3. Notable High-Performing & Challenging Species

### Highest Performing Classes (F1 > 90%):
- `mole cricket`: **98.5% F1** (165 val samples, 100% recall)
- `Pieris canidia`: **97.9% F1** (47 val samples, 100% recall)
- `Papilio xuthus`: **96.6% F1** (45 val samples, 95.6% recall)
- `Locustoidea`: **96.3% F1** (139 val samples, 94.2% recall)
- `Ceroplastes rubens`: **96.2% F1** (26 val samples, 96.2% recall)
- `Limacodidae`: **95.6% F1** (139 val samples, 94.2% recall)
- `oides decempunctata`: **94.7% F1** (29 val samples, 93.1% recall)
- `wheat blossom midge`: **92.6% F1** (48 val samples, 91.7% recall)
- `flea beetle`: **92.6% F1** (79 val samples, 94.9% recall)
- `Cicadellidae`: **92.8% F1** (573 val samples, 93.4% recall)

### Challenging Micro-Pest Classes (Subject to resolution bounds):
- `Polyphagotars onemus latus` (Broad mite): 20.0% F1 (microscopic mite)
- `green bug`: 25.4% F1
- `therioaphis maculata Buckton`: 27.8% F1

---

## 4. File Inventory

| Category | File Path | Description |
| :--- | :--- | :--- |
| **Production Weights** | [`models/pest/ip102_convnext_best.pth`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/pest/ip102_convnext_best.pth) | Production ConvNeXt-Small checkpoint (189 MB) |
| **Standard Model Link** | [`models/pest/pest_model_best.pth`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/pest/pest_model_best.pth) | Symlink / duplicate for standard inference service loading |
| **Metrics Summary (JSON)** | [`models/pest/pest_metrics.json`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/pest/pest_metrics.json) | Full machine-readable metrics (Train, Val, Test) |
| **Metrics Table (CSV)** | [`outputs/ip102/ip102_metrics_table.csv`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/outputs/ip102/ip102_metrics_table.csv) | Tabular metric progression across train/val/test splits |
| **Per-Class Breakdown (CSV)** | [`outputs/ip102/ip102_per_class_val.csv`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/outputs/ip102/ip102_per_class_val.csv) | Precision, recall, and F1 across all 102 pest species |
| **Per-Class Breakdown (JSON)** | [`outputs/ip102/ip102_per_class_val.json`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/outputs/ip102/ip102_per_class_val.json) | JSON version of per-species metrics |
| **Training Curves Plot** | [`outputs/ip102/ip102_curves.png`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/outputs/ip102/ip102_curves.png) | Loss progression & raw vs EMA validation accuracy curves |
| **Executed Notebook** | [`notebooks/02_IP102_Classification.ipynb`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/notebooks/02_IP102_Classification.ipynb) | Production Kaggle notebook with executed cells and outputs |

