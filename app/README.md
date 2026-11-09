# SmartCrop AI — Frontend and Backend Application Suite

This directory contains the production-ready frontend (Streamlit multi-page app) and backend (FastAPI service) prepared for multi-modal model integration.

---

## 🏗️ Architecture

```
app/
├── main.py                                      # Streamlit Executive Dashboard & Live Health Tracker
├── api/
│   ├── schemas.py                               # Pydantic Schemas & API contracts for all 4 prediction tasks
│   ├── mock_service.py                          # Realistic mock generators using official dataset labels
│   └── routes.py                                # FastAPI endpoints (/api/predict/*, /api/health)
├── components/
│   ├── uploader.py                              # Reusable image/array upload component with validation
│   └── result_card.py                           # Reusable prediction card, confidence meters & mock badges
└── pages/
    ├── 1_🍃_PlantDoc_Disease_Diagnosis.py       # Foliar disease diagnosis (27 classes)
    ├── 2_🐛_IP102_Pest_Detection.py             # Pest classification & YOLOv8 field detection (102 classes)
    ├── 3_🛰️_CropHarvest_Binary.py              # Satellite crop vs non-crop detection (12x18 tensor)
    ├── 4_🌾_CropHarvest_10Class.py              # Satellite 10-class crop group & land-cover classification
    └── 5_📊_Unified_Results_Dashboard.py        # Cross-pipeline metrics & ablation dashboard
```

---

## 🚀 How to Run

### 1. Launch Streamlit UI
```powershell
streamlit run app/main.py
```
Access the UI at: `http://localhost:8501`

### 2. Launch FastAPI Backend
```powershell
uvicorn app.api.routes:app --reload --port 8000
```
- Interactive Swagger API Documentation: `http://localhost:8000/docs`
- Redoc API Documentation: `http://localhost:8000/redoc`

---

## 🔒 Safety & Governance Principles
1. **Clear Mock Labeling:** Every mock output is marked with `is_mock=True` in schemas and highlighted with prominent yellow warning banners in the UI (`[MOCK DATA]`).
2. **Zero Checkpoint Pollution:** No unverified checkpoints are attached until Kaggle training concludes and test metrics are validated.
3. **Official Dataset Mappings:** PlantDoc uses 27 real classes from `disease_classes.json`; IP102 uses the 102 species from `classes.txt`; CropHarvest uses the 9 FAO crop groups plus 1 land-cover class (`non_crop`).
4. **Non-Destructive & Modular:** No existing training scripts, notebooks, datasets, or checkpoints were modified.

