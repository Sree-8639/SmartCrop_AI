# SmartCrop AI — IP102 Experiment Tracking & Target Comparison

**Target Benchmark:** $\ge 80.00\%$ Validation Top-1 Accuracy across 102 Insect Pest Categories  
**Dataset Splits:** Official CVPR 2019 Partition (Train / Validation)  
**Evaluation Standard:** Deterministic Center-Crop Evaluation on Validation Split  

---

## 1. Experiment Comparison Table

| Experiment ID | Model Architecture | Resolution | Loss Function | Learning Rate / Schedule | Val Top-1 (%) | Val Top-5 (%) | Val Macro-F1 (%) | Training Time | 80% Target Status | Checkpoint Path |
| :--- | :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Stage 0: Baseline** | EfficientNet-B0 | $224 \times 224$ | Weighted Cross-Entropy | AdamW ($3 \times 10^{-4}$), Cosine | *Pending Run* | *Pending Run* | *Pending Run* | *Pending Run* | ⏳ PENDING | `outputs/ip102/ip102_baseline.pth` |
| **Experiment A** | EfficientNet-B0 | $224 \times 224$ | Weighted Cross-Entropy | 2-Stage Diff LR (Backbone $2\times 10^{-5}$, Head $2\times 10^{-4}$) | *Pending Run* | *Pending Run* | *Pending Run* | *Pending Run* | ⏳ PENDING | `outputs/ip102/ip102_exp_A.pth` |
| **Experiment B** | EfficientNet-B0 | $224 \times 224$ | Class-Balanced Focal Loss ($\gamma=1.5, \beta=0.9999$) + Label Smooth ($0.1$) | 2-Stage Diff LR + Cosine Warmup | *Pending Run* | *Pending Run* | *Pending Run* | *Pending Run* | ⏳ PENDING | `outputs/ip102/ip102_exp_B.pth` |
| **Experiment C** | ConvNeXt-Tiny | $288 \times 288$ | Class-Balanced Focal Loss + Label Smooth | Layer Decay + AdamW ($1 \times 10^{-4}$) | *Pending Run* | *Pending Run* | *Pending Run* | *Pending Run* | ⏳ PENDING | `outputs/ip102/ip102_exp_C.pth` |
| **Experiment D (TTA)** | Best from A–C | $288 \times 288$ | N/A (Inference Benchmark) | 5-Crop Test-Time Augmentation | *Pending Run* | *Pending Run* | *Pending Run* | *Inference Only* | ⏳ PENDING | Evaluates Best Checkpoint |

> [!IMPORTANT]
> **Status Designation Rules:**
> - **REACHED:** Marked ONLY when measured Validation Top-1 Accuracy $\ge 80.00\%$.
> - **NOT REACHED:** Recorded honestly if maximum performance falls below $80.00\%$, accompanied by root-cause diagnosis.
> - **PENDING:** Current state prior to execution approval.

---

## 2. Hypothesis & Parameter Isolation Matrix

To comply with the rule *"Avoid changing multiple major variables simultaneously"*, each experiment introduces a strictly controlled intervention:

```
Baseline (EfficientNet-B0, 224, CE, Single LR)
   │
   ├─► [+ 2-Stage Schedule & Differential LR] ──────────► Experiment A
   │
   ├─► [Exp A + Class-Balanced Focal Loss & Smoothing] ─► Experiment B
   │
   ├─► [Exp B + ConvNeXt-Tiny + 288x288 Resolution] ────► Experiment C
   │
   └─► [Best Model + 5-Crop Test-Time Augmentation] ────► Experiment D
```

---

## 3. Bottleneck Analysis Framework (Contingency if < 80%)

If Experiment D concludes with Top-1 Accuracy $< 80.00\%$, the following analytical procedure will execute:

1. **Per-Taxon Breakdown:** Group error rates by upper agricultural order (Diptera, Lepidoptera, Coleoptera, Hemiptera) to determine whether specific insect orders account for the majority of errors.
2. **Life-Stage Metamorphic Confusion:** Inspect if caterpillars are misclassified into different species due to larval morphology overlap.
3. **Receptive Field Limitations:** Evaluate whether pest size relative to frame requires localized bounding-box crop integration (bridging YOLO detection into classification).

