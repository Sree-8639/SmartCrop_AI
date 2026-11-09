# SmartCrop AI: CropHarvest Dual-Pipeline Training Report
**Generated Timestamp:** 2026-11-08 23:34:01 UTC  
**Hardware:** cuda  

## 1. Overview
This report summarizes the training and evaluation of two independent Bidirectional LSTM models for satellite time-series phenology classification using CropHarvest.

## 2. Model Performance Summary
| Model Name | Task | Classes | Val Acc | Official Test Acc | Test Macro-F1 | Checkpoint Size |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Binary BiLSTM | Crop vs Non-Crop | 2 | 87.24% | 82.02% | 81.18% | Ensemble |
| 10-Class BiLSTM | 9 Crops + 1 Land Cover | 10 | 74.20% | 67.29% | 32.39% | Ensemble |

### A. Binary Classification Detailed Metrics
| Metric | Training | Validation | Official Test |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 94.52% | 87.24% | **82.02%** |
| **Precision (macro)** | 93.52% | 84.92% | **86.01%** |
| **Recall (macro)** | 93.35% | 84.49% | **81.14%** |
| **F1 (macro)** | 93.44% | 84.70% | **81.18%** |
| **Precision (weighted)** | 94.51% | 87.17% | **85.45%** |
| **Recall (weighted)** | 94.52% | 87.24% | **82.02%** |
| **F1 (weighted)** | 94.51% | 87.20% | **81.38%** |

### B. 10-Class Classification Detailed Metrics
| Metric | Training | Validation | Official Test |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 93.36% | 74.20% | **67.29%** |
| **Precision (macro)** | 94.07% | 71.21% | **45.37%** |
| **Recall (macro)** | 93.94% | 65.21% | **32.67%** |
| **F1 (macro)** | 93.96% | 67.27% | **32.39%** |
| **Precision (weighted)** | 93.42% | 73.86% | **72.65%** |
| **Recall (weighted)** | 93.36% | 74.20% | **67.29%** |
| **F1 (weighted)** | 93.33% | 73.55% | **60.08%** |

## 3. Official Benchmark Test Support Deficit Audit (10-Class)
In the official CropHarvest benchmark test split (`is_test: true`, 107 samples):
- **vegetables_melons:** 0 samples (untestable in official benchmark test set).
- **fruits_nuts, oilseeds, leguminous, sugar:** 1 sample each (insufficient statistical support).
- **cereals, other, root_tuber, beverage_spice, non_crop:** $\ge 10$ samples (adequate support).

## 4. Geographic Autocorrelation Disclosure
The train/validation split uses stratified random sampling. All 16 training regional datasets are present in both train and validation splits, with 455 co-located coordinate observations. Therefore, validation metrics measure phenology learning, while true spatial transferability is evaluated against the held-out test split.

## 5. Artifact Verification Checklist
- [x] `binary/binary_metrics.json`: VERIFIED (Accuracy: 82.02%, Macro-F1: 81.18%)
- [x] `binary/binary_metrics_table.csv`: VERIFIED
- [x] `10class/10class_metrics.json`: VERIFIED (Accuracy: 67.29%, Macro-F1: 32.39%)
- [x] `10class/10class_metrics_table.csv`: VERIFIED
- [x] `combined_results.json`: VERIFIED
- [x] `combined_metrics_table.csv`: VERIFIED
- [x] `final_training_report.md`: VERIFIED
