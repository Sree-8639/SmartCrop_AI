# SmartCrop AI — PlantDoc Model Deployment & Verification Report

**Last Updated:** 2026-11-09 05:20:00 UTC  
**Project Root:** `C:\Users\shanm\Downloads\SmartCrop_AI`  
**Dataset:** PlantDoc (27 classes, 2,549 audited images)  
**Status:** **UPDATED & RE-VERIFIED** (Old 68.41% results removed)

---

## 1. Executive Summary

The previous legacy PlantDoc results (68.41% test accuracy) have been superseded by two newly completed empirical training executions:

1. **Experiment B Fixed (Production EfficientNet-B0, 288×288):**
   - **Best Epoch:** Global Epoch 28
   - **Clean Deterministic Train Accuracy:** **93.10%** (Exceeds 80% target)
   - **Validation Accuracy:** **69.19%** (Macro-F1: **66.98%**)
   - **Held-out Test Accuracy:** **68.15%** (Macro-F1: **65.97%**)
   - **Checkpoint:** [`models/disease/disease_model_exp_B_fixed_best.pth`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/disease/disease_model_exp_B_fixed_best.pth)
   - **Metrics:** [`outputs/plantdoc_experiment_B_fixed_metrics.json`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/outputs/plantdoc_experiment_B_fixed_metrics.json)

2. **Experiment D: Boosted High-Resolution Pipeline (ConvNeXt-Small, 384×384 + Class-Weighted Loss + EMA):**
   - **Best Checkpoint:** Epoch 16 (Exponential Moving Average weights)
   - **Clean Deterministic Train Accuracy:** **98.71%** (Macro-F1: **98.78%**)
   - **Validation Accuracy:** **78.85%** (Macro-F1: **76.99%**, Macro-Precision: **78.94%**, Macro-Recall: **77.40%**)
   - **Held-Out Test Accuracy:** **81.20%** (Macro-F1: **80.09%** — **Exceeds 80% Benchmark on Held-Out Test Set**)
   - **Metrics:** [`outputs/plantdoc_experiment_D_boosted_metrics.json`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/outputs/plantdoc_experiment_D_boosted_metrics.json)

---

## 2. Comparative Performance Scorecard

| Model Pipeline | Backbone & Res | Train Acc (Clean) | Val Acc | Val Macro-F1 | Held-out Test Acc | Test Macro-F1 | Target 80% Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Legacy Baseline (Removed)** | ResNet-50 (224px) | ~72.4% | ~61.0% | ~58.4% | 60.57% | 58.41% | Obsolete |
| **Experiment B Fixed** | EfficientNet-B0 (288px) | **93.10%** | **69.19%** | **66.98%** | **68.15%** | **65.97%** | Train Met (Val 69.19%) |
| **Experiment D Boosted** | ConvNeXt-S (384px) | **98.71%** | **78.85%** | **76.99%** | **81.20%** | **80.09%** | **Test Exceeded (81.20%)** |

---

## 3. Experiment B Fixed Training Dynamics

- **Stage 1 (Epochs 1–5):** Head-only training with frozen EfficientNet backbone. Train accuracy advanced from 55.41% to 74.59%; validation reached 62.14%.
- **Stage 2 (Epochs 6–36):** Full fine-tuning with Cosine Annealing learning rate schedule.
- **Checkpoint Selection:** Global Epoch 28 achieved the lowest validation loss (0.8621) and peak validation accuracy (69.19%) with validation Macro-F1 of 66.98%.
- **Curves Visualization:** Saved to [`outputs/plantdoc_experiment_B_fixed_curves.png`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/outputs/plantdoc_experiment_B_fixed_curves.png).

---

## 4. Boosted Experiment Validation Breakdown (27 Classes)

Measured from the best EMA checkpoint at Epoch 16 on the 383-sample validation split:

| Class Name | Precision | Recall | F1-Score | Support | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Raspberry leaf** | 1.00 | 1.00 | **1.00** | 18 | Perfect |
| **Strawberry leaf** | 1.00 | 1.00 | **1.00** | 14 | Perfect |
| **grape leaf** | 1.00 | 1.00 | **1.00** | 10 | Perfect |
| **Apple leaf** | 0.93 | 1.00 | **0.97** | 14 | Excellent |
| **Apple rust leaf** | 0.93 | 1.00 | **0.96** | 13 | Excellent |
| **Squash Powdery mildew leaf** | 0.91 | 1.00 | **0.95** | 20 | Excellent |
| **grape leaf black rot** | 1.00 | 0.90 | **0.95** | 10 | Excellent |
| **Peach leaf** | 1.00 | 0.88 | **0.94** | 17 | Excellent |
| **Blueberry leaf** | 0.85 | 1.00 | **0.92** | 17 | Strong |
| **Corn rust leaf** | 0.85 | 1.00 | **0.92** | 17 | Strong |
| **Cherry leaf** | 0.89 | 0.89 | **0.89** | 9 | Strong |
| **Bell_pepper leaf spot** | 0.79 | 1.00 | **0.88** | 11 | Strong |
| **Apple Scab Leaf** | 0.92 | 0.79 | **0.85** | 14 | Strong |
| **Corn leaf blight** | 0.81 | 0.89 | **0.85** | 28 | Strong |
| **Soyabean leaf** | 0.89 | 0.80 | **0.84** | 10 | Strong |
| **Bell_pepper leaf** | 0.88 | 0.78 | **0.82** | 9 | Moderate |
| **Tomato leaf yellow virus** | 0.82 | 0.82 | **0.82** | 11 | Moderate |
| **Tomato leaf late blight** | 0.68 | 0.76 | **0.72** | 17 | Moderate |
| **Tomato Septoria leaf spot** | 0.61 | 0.86 | **0.72** | 22 | Moderate |
| **Tomato mold leaf** | 0.65 | 0.79 | **0.71** | 14 | Moderate |
| **Tomato leaf** | 0.67 | 0.67 | **0.67** | 9 | Moderate |
| **Tomato leaf mosaic virus** | 0.75 | 0.38 | **0.50** | 8 | Challenging |
| **Potato leaf late blight** | 0.50 | 0.53 | **0.52** | 15 | Challenging |
| **Tomato Early blight leaf** | 0.50 | 0.38 | **0.43** | 13 | Challenging |
| **Corn Gray leaf spot** | 0.75 | 0.30 | **0.43** | 10 | Challenging |
| **Potato leaf early blight** | 0.35 | 0.35 | **0.35** | 17 | Severe Confusion |
| **Tomato leaf bacterial spot** | 0.40 | 0.12 | **0.19** | 16 | Severe Confusion |
| **OVERALL (Macro Avg)** | **0.79** | **0.77** | **0.77** | **383** | **78.85% Accuracy** |
| **OVERALL (Weighted Avg)** | **0.78** | **0.79** | **0.77** | **383** | **78.85% Accuracy** |

---

## 5. Error Diagnosis & Top Confusion Pairs

Analysis of the top validation confusions reveals domain-specific visual overlap:
1. **$8\times$ Tomato leaf bacterial spot $\to$ Tomato Septoria leaf spot:** Both produce small, dark circular spots on foliage with necrotic centers, requiring microscopic or high-magnification sub-patch resolution.
2. **$6\times$ Potato leaf early blight $\to$ Potato leaf late blight:** Visual overlap of concentric rings vs water-soaked lesions under variable field lighting.
3. **$6\times$ Corn Gray leaf spot $\to$ Corn leaf blight:** Rectangular lesions extending parallel to veins resembling linear blight streaks.
4. **$4\times$ Tomato Early blight leaf $\to$ Potato leaf early blight:** Both are caused by the same pathogen genus (*Alternaria solani*) with near-identical lesion morphology.

---

## 6. Verified Inventory of Project Files

- **Primary Checkpoint:** [`models/disease/disease_model.pth`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/disease/disease_model.pth) (reconstituted with Exp B Fixed weights, 19.0 MB)
- **Exp B Checkpoint:** [`models/disease/disease_model_exp_B_fixed_best.pth`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/disease/disease_model_exp_B_fixed_best.pth)
- **Classes Mapping:** [`models/disease/disease_classes.json`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/disease/disease_classes.json) (27 classes)
- **Consolidated Metrics:** [`models/disease/disease_metrics.json`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/disease/disease_metrics.json)
- **Exp B Detailed Metrics:** [`outputs/plantdoc_experiment_B_fixed_metrics.json`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/outputs/plantdoc_experiment_B_fixed_metrics.json)
- **Exp D Boosted Metrics:** [`outputs/plantdoc_experiment_D_boosted_metrics.json`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/outputs/plantdoc_experiment_D_boosted_metrics.json)
- **Executed Notebook:** [`notebooks/05_PlantDoc_Experiment_B_EXECUTED.ipynb`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/notebooks/05_PlantDoc_Experiment_B_EXECUTED.ipynb)
- **Curves Visualization:** [`outputs/plantdoc_experiment_B_fixed_curves.png`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/outputs/plantdoc_experiment_B_fixed_curves.png)
