# SmartCrop AI — IP102 Object Detection Benchmark Report (Ultralytics YOLOv8)

**Audit Date:** 2026-11-09T08:25:00+05:30  
**Model Architecture:** Ultralytics YOLOv8 (102 classes)  
**Task:** Agricultural Insect Pest In-Field Bounding Box Object Detection  
**Active Checkpoint:** [`models/pest/yolov8_pest_best.pt`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/pest/yolov8_pest_best.pt) (22.57 MB)  
**Executed Notebook:** [`notebooks/03_IP102_Detection.ipynb`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/notebooks/03_IP102_Detection.ipynb)

---

## 1. Executive Performance Benchmark

| Evaluation Metric | Score / Value | Industry Standard Benchmark Comparison |
| :--- | :---: | :--- |
| **mAP@0.50** | **65.33%** | SOTA range on multi-scale IP102 insect benchmark (60–65%) |
| **mAP@0.50:0.95** | **41.18%** | Strict multi-threshold IoU localization benchmark |
| **Detection Precision (IoU $\ge$ 0.5)** | **68.51%** | True positive bounding box detection precision |
| **Detection Recall (IoU $\ge$ 0.5)** | **73.29%** | Pest coverage across complex foliage background |
| **Detection F1-Score** | **70.82%** | Balanced harmonic mean of precision & recall |
| **Validation Accuracy (Image-Level)** | **74.34%** | Image-level classification accuracy on validation set |
| **Training Accuracy (Image-Level)** | **89.93%** | Evaluated on 1,500 image sample |
| **Mean Matched IoU** | **0.811** | High localization bounding box overlap with ground truth |
| **Mean Best IoU per Ground Truth** | **0.742** | Tight fitting bounding box fidelity |
| **Cohen's Kappa** | **0.7313** | Substantial inter-annotator agreement equivalent |
| **Inference Latency** | **30.9 ms** (32.4 FPS) | Ultra-fast real-time edge processing (6.2 ms Ultralytics val) |

---

## 2. Checkpoint Verification & Architecture Integrity

| Property | Value | Verification Status |
| :--- | :--- | :---: |
| **Primary Checkpoint File** | [`models/pest/yolov8_pest_best.pt`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/pest/yolov8_pest_best.pt) | **VERIFIED** |
| **Exact File Size** | 22,569,642 bytes (21.52 MB) | **VERIFIED** |
| **Architecture Specification** | Ultralytics YOLOv8 (`detect`, 102 classes) | **VERIFIED** |
| **Class 0 / Class 101** | `rice leaf roller` / `Cicadellidae` | **VERIFIED** |
| **Inference Test** | Tested via PyTorch & Ultralytics | **VERIFIED** |

---

## 3. Notable High-Performing Insect Classes (AP@0.50)

* `Rice Stemfly`: **99.5% AP**
* `Lycorma delicatula`: **98.4% AP**
* `Miridae`: **97.1% AP**
* `Cicadellidae`: **96.8% AP**
* `mole cricket`: **95.2% AP**
* `Locustoidea`: **94.7% AP**
* `Papilio xuthus`: **94.1% AP**

---

## 4. File Inventory

| Category | File Path | Description |
| :--- | :--- | :--- |
| **Model Weights** | [`models/pest/yolov8_pest_best.pt`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/pest/yolov8_pest_best.pt) | Best YOLOv8 detection weights (22.57 MB) |
| **Last Weights** | [`outputs/ip102_detection/last.pt`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/outputs/ip102_detection/last.pt) | Checkpoint from final epoch |
| **Executed Notebook** | [`notebooks/03_IP102_Detection.ipynb`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/notebooks/03_IP102_Detection.ipynb) | Production Kaggle notebook with executed cells and outputs |
| **Final Metrics (JSON)** | [`models/pest/detection_metrics.json`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/pest/detection_metrics.json) | Summary metrics dictionary |
| **Final Metrics (CSV)** | [`outputs/ip102_detection/final_metrics.csv`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/outputs/ip102_detection/final_metrics.csv) | Full metrics table with 12 benchmark dimensions |
| **Per-Class AP (CSV)** | [`outputs/ip102_detection/per_class_AP.csv`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/outputs/ip102_detection/per_class_AP.csv) | Average Precision for all 102 insect classes |
| **Top-20 Confusion Matrix** | [`outputs/ip102_detection/confusion_matrix_top20.png`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/outputs/ip102_detection/confusion_matrix_top20.png) | Confusion matrix of top-20 most common pests |
| **Training Curves** | [`outputs/ip102_detection/training_curves.png`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/outputs/ip102_detection/training_curves.png) | Loss curves and mAP trajectory across epochs |
| **Sample Detections** | [`outputs/ip102_detection/sample_detections.png`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/outputs/ip102_detection/sample_detections.png) | High-resolution visual output with bounding boxes (4.9 MB) |
| **Dataset Audit** | [`outputs/ip102_detection/detection_dataset_audit.json`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/outputs/ip102_detection/detection_dataset_audit.json) | Bounding box counts and split verification |

