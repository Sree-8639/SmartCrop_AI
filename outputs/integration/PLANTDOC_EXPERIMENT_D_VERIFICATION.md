# SmartCrop AI — PlantDoc Experiment D Checkpoint & Evaluation Verification Report

**Verification Date:** 2026-11-09T06:22:00+05:30  
**Audit Scope:** `models/disease/disease_model_exp_D_best.pth`, evaluation logs, split manifests, and API status logic.  
**Strict Safety Protocol:** Zero modification to existing models, training scripts, Kaggle notebooks, or datasets. No real inference connected yet.

---

## 1. Executive Verification Scorecard

| Item | Description | Verification Status | Evidence / Metrics |
| :---: | :--- | :---: | :--- |
| **Task 1** | Checkpoint Size & SHA256 Hash | **VERIFIED** | 198,041,037 bytes (188.87 MB) \| SHA256: `b84ee660bad5fa775261168994398e4102d804836aeebb0a0c1f9c22c1cc5807` |
| **Task 2** | Architecture Metadata & Key Inspection | **VERIFIED** | Model: `convnext_small.fb_in22k_ft_in1k`, Res: 384×384, Classes: 27, Source: `ema` (Epoch 16) |
| **Task 3** | Strict Checkpoint Loading (`strict=True`) | **VERIFIED** | Passed `strict=True` with `timm.create_model`. Missing keys: 0, Unexpected keys: 0 |
| **Task 4** | Dummy Inference Forward Pass | **VERIFIED** | Output shape: `[1, 27]`. All logits strictly finite (`is_finite = True`). Min: -0.927, Max: 1.432 |
| **Task 5** | Trace 81.20% Test Accuracy to Evaluation Logs | **VERIFIED** | Traced to Cell 7 of `notebooks/01_PlantDoc_Classification.ipynb` on 383 held-out test images |
| **Task 6** | Checkpoint Identity Confirmation | **VERIFIED** | `models/disease/disease_model_exp_D_best.pth` exactly matches the evaluated epoch 16 EMA state |
| **Task 7** | Held-Out Test Set Isolation Audit | **VERIFIED** | Test split strictly isolated. Evaluated ONCE at audit. Zero leakage into training or tuning |
| **Task 8** | Audit of Mock Endpoints in `app/api/routes.py` | **VERIFIED** | All 5 prediction endpoints confirmed currently using mock inference with `is_mock=True` |
| **Task 9** | Health-Status Logic Hardening | **VERIFIED** | Hardcoded 81.20 fallback removed; validation accuracy strictly barred from reporting as test accuracy |

---

## 2. Task 1: Checkpoint Physical Properties & Cryptographic Hash

- **File Path:** [`models/disease/disease_model_exp_D_best.pth`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/disease/disease_model_exp_D_best.pth)
- **File Existence:** Confirmed on local disk.
- **Exact File Size:** **198,041,037 bytes** (`188.8668 MB`)
- **Cryptographic Hash (SHA256):**
  ```
  b84ee660bad5fa775261168994398e4102d804836aeebb0a0c1f9c22c1cc5807
  ```
- **Status:** **VERIFIED**

---

## 3. Task 2: Checkpoint Metadata, Keys, and Architecture Determination

Inspection of the checkpoint dictionary via PyTorch inspection tools:
- **Top-Level Keys:** `['model_state_dict', 'epoch', 'source', 'val_accuracy', 'val_f1', 'model_name', 'resolution', 'num_classes']`
- **Embedded Metadata:**
  - `model_name`: `"convnext_small.fb_in22k_ft_in1k"`
  - `resolution`: `384`
  - `num_classes`: `27`
  - `epoch`: `16`
  - `source`: `"ema"` (Exponential Moving Average weights, decay=0.998)
  - `val_accuracy`: `0.7885117493472585` (78.85%)
  - `val_f1`: `0.7699382276915043` (76.99%)
- **Model Parameter Count:** 344 parameter tensors in `model_state_dict`.
- **Architecture Determination:**
  The weights belong to a **ConvNeXt-Small** backbone (`convnext_small.fb_in22k_ft_in1k` from `timm`) configured with a 27-class linear classification head. Torchvision's native `convnext_small` uses different internal parameter naming (`features.*` vs `stages.*`); therefore, the authoritative model builder is `timm.create_model("convnext_small.fb_in22k_ft_in1k", pretrained=False, num_classes=27)`.
- **Status:** **VERIFIED**

---

## 4. Task 3: Strict Checkpoint Loading (`strict=True`)

- **Loading Harness:**
  ```python
  import timm
  import torch

  model = timm.create_model(
      "convnext_small.fb_in22k_ft_in1k", pretrained=False, num_classes=27
  )
  ckpt = torch.load(
      "models/disease/disease_model_exp_D_best.pth",
      map_location="cpu",
      weights_only=False,
  )
  load_result = model.load_state_dict(ckpt["model_state_dict"], strict=True)
```
- **Execution Result:**
  - `load_result.missing_keys`: `[]` (0 missing keys)
  - `load_result.unexpected_keys`: `[]` (0 unexpected keys)
  - Strict loading succeeded with **zero** mismatches, warnings, or missing parameters.
- **Status:** **VERIFIED**

---

## 5. Task 4: Dummy Forward Pass & Logit Verification

- **Input Tensor:** `torch.randn(1, 3, 384, 384)`
- **Evaluation Mode:** `model.eval()`, `torch.no_grad()`
- **Output Tensor Shape:** `torch.Size([1, 27])` (matches 27 PlantDoc classes)
- **Numerical Integrity:**
  - `torch.isfinite(output).all()`: **True** (All 27 logits are strictly finite)
  - Minimum Logit: `-0.9274`
  - Maximum Logit: `+1.4322`
  - Mean Logit: `+0.1015`
  - Top-5 Softmax Probabilities: `[0.1160, 0.0949, 0.0836, 0.0602, 0.0539]` (Sum: finite, valid distribution)
- **Status:** **VERIFIED**

---

## 6. Task 5: Tracing 81.20% Test Accuracy to Evaluation Logs & Data Splits

The reported **81.20% Test Accuracy** was traced directly to the executed evaluation code and artifacts:

1. **Source Code & Cell Outputs ([`notebooks/01_PlantDoc_Classification.ipynb`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/notebooks/01_PlantDoc_Classification.ipynb)):**
   - **Cell 3 Split Manifest Parsing:**
     ```
     [TRAIN] mapped=1783 skipped=0
     [VAL]   mapped=383  skipped=0
     [TEST]  mapped=383  skipped=0
     ```
     Dataset total: 2,549 audited images across 27 classes.
   - **Cell 7 Final Audit Execution Block:**
     ```python
     test_loader = DataLoader(
         PlantDocDataset(test_records, eval_tf),
         batch_size=BATCH_SIZE,
         shuffle=False,
         num_workers=NUM_WORKERS,
         pin_memory=pin,
     )
     final_test = evaluate(final_model, test_loader, tta=use_tta)
```
   - **Embedded Cell 7 Execution Output:**
     ```
     ============================================================
     FINAL AUDIT (best checkpoint: epoch 16, weights: ema)
     ============================================================
       Det. train acc : 98.71%  [PASS vs 80%]
       Val acc (plain): 78.85%
       Val acc (TTA)  : 78.59%   -> TTA used: False
       Val acc final  : 78.85%  [FAIL vs 80%]
       Val macro-F1   : 76.99%
       Test acc       : 81.20%  (held-out, evaluated once)
       Test macro-F1  : 80.09%
       Previous: Exp B val 69.19% / test 68.15% | Exp C val 79.90% / test 79.90%
       Overall status : NOT_REACHED
     ============================================================
     Saved: plantdoc_experiment_D_curves.png, plantdoc_experiment_D_metrics.json, disease_model_exp_D_best.pth
     ```
2. **Metrics File ([`outputs/plantdoc_experiment_D_boosted_metrics.json`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/outputs/plantdoc_experiment_D_boosted_metrics.json)):**
   ```json
   "held_out_test_results": {
     "accuracy": 0.812,
     "macro_f1": 0.8009,
     "evaluation_frequency": "evaluated_once_at_final_audit"
   }
   ```
3. **Central Metrics Registry ([`models/disease/disease_metrics.json`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/disease/disease_metrics.json)):**
   ```json
   "experiment_d_boosted": {
     "architecture": "ConvNeXt-Small (timm)",
     "resolution": [384, 384],
     "best_checkpoint": "epoch_16_ema",
     "deterministic_train_accuracy": 0.9871,
     "validation_accuracy": 0.7885,
     "validation_macro_f1": 0.7699,
     "held_out_test_accuracy": 0.812,
     "held_out_test_macro_f1": 0.8009
   }
   ```
- **Status:** **VERIFIED**

---

## 7. Task 6: Identity Confirmation of Checkpoint

- The checkpoint on disk [`models/disease/disease_model_exp_D_best.pth`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/models/disease/disease_model_exp_D_best.pth) records:
  - `epoch`: `16`
  - `source`: `"ema"`
  - `val_accuracy`: `0.7885117493472585`
  - `val_f1`: `0.7699382276915043`
  - `model_name`: `"convnext_small.fb_in22k_ft_in1k"`
- These exact metrics match the final audit evaluation log in `01_PlantDoc_Classification.ipynb` which achieved **81.20% Test Accuracy**.
- **Status:** **VERIFIED**

---

## 8. Task 7: Held-Out Test Set Isolation & Leakage Audit

A line-by-line inspection of the training and evaluation logic in `notebooks/01_PlantDoc_Classification.ipynb` confirms:
1. **Training Isolation:** The training loop strictly fed `train_loader` (`1,783` images). `test_loader` was not loaded into memory during training.
2. **Model Selection Isolation:** Checkpoint saving in Cell 6 was triggered exclusively by validation improvement:
   ```python
   better = cur["acc"] > best_val_acc + 1e-5 or (
       abs(cur["acc"] - best_val_acc) <= 1e-5 and cur["macro_f1"] > best_val_f1
   )
   if better:
     best_val_acc, best_val_f1, best_epoch, best_source = (
         cur["acc"],
         cur["macro_f1"],
         epoch,
         src,
     )
     torch.save(...)
```
   The test split was completely excluded from checkpoint selection.
3. **Hyperparameter Tuning Isolation:** The decision of whether to apply Test-Time Augmentation (TTA) was decided solely on the validation split:
   ```python
   use_tta = (
       val_tta["acc"] > val_plain["acc"]
   )  # explicitly noted in code: "decided on VALIDATION only"
```
4. **Single-Pass Evaluation:** The test split was loaded and evaluated **exactly once** in Cell 7 after all training and model selection had completed.
- **Status:** **VERIFIED (Zero Test Set Leakage)**

---

## 9. Task 8: Audit of Mock Inference Endpoints in `app/api/routes.py`

Inspection of [`app/api/routes.py`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/app/api/routes.py) confirms that **real model inference has NOT been connected yet**:

| Endpoint | Target Pipeline | Current Handler Function | Response Flag | Status |
| :--- | :--- | :--- | :---: | :---: |
| `POST /api/predict/plantdoc` | PlantDoc Foliar Disease | `generate_plantdoc_mock()` | `is_mock=True` | Mock Inference Only |
| `POST /api/predict/ip102/classify` | IP102 Pest Classification | `generate_ip102_classification_mock()` | `is_mock=True` | Mock Inference Only |
| `POST /api/predict/ip102/detect` | IP102 Pest Detection | `generate_ip102_detection_mock()` | `is_mock=True` | Mock Inference Only |
| `POST /api/predict/cropharvest/binary` | CropHarvest Binary | `generate_cropharvest_binary_mock()` | `is_mock=True` | Mock Inference Only |
| `POST /api/predict/cropharvest/10class` | CropHarvest 10-Class | `generate_cropharvest_10class_mock()` | `is_mock=True` | Mock Inference Only |

All five endpoints remain strictly mocked, awaiting formal connection instructions.
- **Status:** **VERIFIED**

---

## 10. Task 9: Health-Status Logic Hardening & Fix

In [`app/api/mock_service.py`](file:///C:/Users/shanm/Downloads/SmartCrop_AI/app/api/mock_service.py), `get_all_subsystem_health()` previously included:
```python
# Previous vulnerable implementation:
plantdoc_acc = d.get("test_acc") or d.get("best_val_acc")
...
test_accuracy = plantdoc_acc or 81.20
```
This had two governance flaws:
1. It allowed validation accuracy (`best_val_acc`) to masquerade as test accuracy.
2. It used a hardcoded fallback (`81.20`) if the file was missing or unreadable.

### Corrected Implementation:
```python
plantdoc_metrics = (
    BASE_DIR / "outputs" / "plantdoc_experiment_D_boosted_metrics.json"
)
plantdoc_central_metrics = (
    BASE_DIR / "models" / "disease" / "disease_metrics.json"
)
plantdoc_test_acc: Optional[float] = None

if plantdoc_metrics.exists():
  try:
    with open(plantdoc_metrics, "r", encoding="utf-8") as f:
      d = json.load(f)
      test_val = d.get("held_out_test_results", {}).get("accuracy")
      if test_val is None:
        test_val = d.get("test_accuracy") or d.get("test_acc")
      if test_val is not None:
        plantdoc_test_acc = round(
            float(test_val) * 100
            if float(test_val) <= 1.0
            else float(test_val),
            2,
        )
  except Exception:
    plantdoc_test_acc = None

if plantdoc_test_acc is None and plantdoc_central_metrics.exists():
  try:
    with open(plantdoc_central_metrics, "r", encoding="utf-8") as f:
      d = json.load(f)
      test_val = d.get("experiment_d_boosted", {}).get(
          "held_out_test_accuracy"
      )
      if test_val is not None:
        plantdoc_test_acc = round(
            float(test_val) * 100
            if float(test_val) <= 1.0
            else float(test_val),
            2,
        )
  except Exception:
    plantdoc_test_acc = None
```
- **Verification of Fix:**
  - When test metrics are missing from the file: `plantdoc_test_acc` evaluates to `None`.
  - When validation accuracy is present without test accuracy: `plantdoc_test_acc` evaluates to `None` (validation accuracy is never substituted).
  - When the metrics file is deleted: `plantdoc_test_acc` evaluates to `None` (no hardcoded fallback).
  - When valid held-out test accuracy (`0.812`) is parsed: `plantdoc_test_acc` evaluates to `81.20%`.
- **Status:** **VERIFIED**

