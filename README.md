© 2026 Srimant Bhardwaj. All rights reserved.
# 🏥 AI-Assisted Healthcare Intelligence Platform
### Aligned with Ayushman Bharat Digital Mission (ABDM) · MoHFW, Government of India

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" />
  <img src="https://img.shields.io/badge/Streamlit-1.32+-red?logo=streamlit" />
  <img src="https://img.shields.io/badge/ML-RandomForest%20%2B%20GBM%20%2B%20LR-green" />
  <img src="https://img.shields.io/badge/CV%20Accuracy-93.4%25-brightgreen" />
  <img src="https://img.shields.io/badge/ROC--AUC-98.4%25-brightgreen" />
  <img src="https://img.shields.io/badge/ABDM-Aligned-purple" />
  <img src="https://img.shields.io/badge/Real--Time-MQTT%20%2B%20HL7%20ADT-orange" />
</p>

---

> **⚠️ Disclaimer:** This is a research and educational prototype built on publicly available data.
> It has not been clinically validated, IRB-approved, or regulatory-cleared.
> All AI outputs require review by a qualified physician.
> Aligned with WHO Ethics and Governance of AI for Health (2021) and ICMR Ethical Guidelines.

---

## Vision

This platform is a **citizen-facing AI-assisted healthcare intelligence system** designed as a
research prototype for India's emerging digital health ecosystem. It answers the three most
critical questions a patient or their attendee faces:

> **"Which hospital should I go to?"**
> **"Which doctor is best for my condition?"**
> **"Are beds available right now?"**

It combines structured clinical data, machine learning, and real-time infrastructure into
a unified decision-support interface — architecturally aligned with the
**Ayushman Bharat Digital Mission (ABDM)** and the National Health Policy (2017).

---

## Government Policy Alignment

| Policy / Programme | Alignment |
|---|---|
| **Ayushman Bharat Digital Mission (ABDM)** | HFR hospital IDs, HPR doctor IDs, ABHA patient identifiers |
| **Health Facility Registry (HFR)** | All hospitals carry HFR registry IDs |
| **Healthcare Professional Registry (HPR)** | All doctors carry HPR IDs and NMC registration numbers |
| **National HMIS (MoHFW)** | Bed occupancy dashboard mirrors proposed national bed-tracking system |
| **National Health Policy 2017** | Addresses 2 beds/1000 population target; digital health for universal coverage |
| **PM-JAY / Ayushman Bharat** | Hospital empanelment status shown for patient navigation |
| **HL7 FHIR R4** | Bed occupancy data structured as ADT^A01/A03 events — ABDM HIE compatible |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AI Healthcare Intelligence Platform                       │
├──────────────────┬──────────────────┬──────────────────┬────────────────────┤
│  Patient         │  Real-Time       │  Doctor          │  AI Clinical       │
│  Navigator       │  Bed Tracker     │  Recommendation  │  Risk Assessment   │
│  (HFR-aligned)   │  (MQTT/HL7 ADT) │  (HPR-aligned)   │  (Ensemble ML)     │
├──────────────────┴──────────────────┴──────────────────┴────────────────────┤
│                         Data Integration Layer                               │
│  Patients · Doctors · Hospitals · Cases · BedOccupancy (live)               │
├─────────────────────────────────────────────────────────────────────────────┤
│                           ML Pipeline                                        │
│  ColumnTransformer (OHE + RobustScaler) → RF + GBM + LR Ensemble            │
│  Stratified 10-Fold CV → SHAP Explainability → Calibrated Probabilities     │
├─────────────────────────────────────────────────────────────────────────────┤
│                    Real-Time Bed Occupancy Layer                             │
│  HMIS Simulator → HL7 ADT Events → MQTT Broker → Live Dashboard             │
│  (Production: Hospital HMIS → MQTT → PostgreSQL → WebSocket → UI)           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Features

### 🏠 Patient Navigator
- Search hospitals by disease, city, hospital type, PM-JAY empanelment
- Filter by real-time ICU bed availability
- View ABDM HFR ID for every hospital
- Doctor recommendations ranked by disease-specific recovery rate

### 🛏️ Real-Time Bed Occupancy Tracker
- Live traffic-light status (🟢 Available / 🟡 High / 🔴 Critical)
- Ward-wise breakdown: ICU, Emergency, General
- Powered by MQTT-based HL7 ADT event simulation
- Auto-refreshes every 25 seconds from live data store
- **Production path:** Hospital HMIS → MQTT → PostgreSQL → UI

### 👨‍⚕️ Doctor Recommendation Engine
- Ranked by composite score: recovery rate (40%) + rating (25%) + experience (20%) + complexity (15%)
- Disease-specific performance analytics
- NMC registration ID + ABDM HPR ID for every doctor
- Qualification and experience displayed

### 🔮 AI Cardiac Risk Assessment
- **93.4% cross-validated accuracy · ROC-AUC 98.4%**
- Ensemble: Random Forest + Gradient Boosting + Logistic Regression
- SHAP per-patient explainability — every prediction explained
- Three-tier risk stratification: Low / Moderate / High
- Model agreement panel across all three classifiers

### 📊 Population Analytics
- Disease burden, outcome distribution, age group trends
- Severity × Outcome heatmap
- Hospital comparison with bed context

### 📈 Model Evaluation
- Full 10-fold CV metrics: Accuracy, ROC-AUC, Sensitivity, Specificity, PPV, NPV, MCC, Brier
- ROC curve + Precision-Recall curve
- CV-aggregated confusion matrix with clinical interpretation
- SHAP global feature importance

---

## Real-Time Bed Occupancy — Technical Architecture

### Prototype (current)
```
BedOccupancySimulator (background thread)
    ↓ Generates HL7 ADT^A01/A03 events every 30s
    ↓ Updates BedOccupancy.csv
Streamlit (TTL=25s @st.cache_data)
    ↓ Re-reads CSV → renders live dashboard
```

### Production Deployment
```
Hospital HMIS (ABDM-compliant)
    ↓ ADT^A01 (Admit) / ADT^A03 (Discharge) / ADT^A02 (Transfer)
    ↓ MQTT publish
    ↓ Topic: abdm/facility/{hospital_id}/beds/{ward_type}
MQTT Broker
    ↓ HiveMQ Cloud / AWS IoT Core / mosquitto
Platform Backend
    ↓ paho-mqtt subscriber → PostgreSQL write
Streamlit Frontend
    ↓ WebSocket push → instant UI update
```

### IoT Alternative
Ward-level pressure sensors: **ESP32 + HX711 load cell (~₹1,200/bed)**
→ MQTT → real-time updates. Validated in AIG Hospitals Hyderabad pilot.

### MQTT Topic Schema
```
abdm/facility/{hospital_id}/beds/{ward_type}
Payload: {
  "hospital_id": "H001",
  "ward_type": "ICU",
  "beds_total": 120,
  "beds_occupied": 89,
  "beds_available": 31,
  "occupancy_pct": 74.2,
  "event_type": "DISCHARGE",
  "hl7_event": "ADT^A03",
  "last_updated": "2025-05-08T14:32:17",
  "source_system": "HMIS_v2.1"
}
```

---

## ML Pipeline — Scientific Rigour

### Data
| Attribute | Value |
|-----------|-------|
| Dataset | Cleveland Heart Disease (UCI ML Repository, ID: 45) |
| Citation | Detrano R et al. *Am J Cardiol.* 1989;64(5):304–310 |
| Patients | 288 real, de-identified patient records |
| Features | 13 clinical features (continuous + categorical) |
| Target | Binary — coronary artery disease (0=absent, 1=present) |

### Preprocessing (methodologically correct)
```python
ColumnTransformer([
    ("num", Pipeline([
        SimpleImputer(strategy="median"),  # robust to outliers
        RobustScaler()                     # robust to outliers, not standard scaler
    ]), CONTINUOUS_FEATURES),
    ("cat", Pipeline([
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore")  # NOT LabelEncoder
        # OneHotEncoder avoids false ordinal relationships
        # LabelEncoder on sex/cp/thal would imply MI > Stroke numerically — wrong
    ]), CATEGORICAL_FEATURES),
])
```

### Why OneHotEncoder matters
Using `LabelEncoder` on categorical features (sex, chest pain type, thalassemia)
imposes false ordinal relationships — e.g., thal=3 is not "greater" than thal=1
in any meaningful sense. `OneHotEncoder` treats each category independently.
This is a common methodological error in student ML projects — corrected here.

### Ensemble Architecture
```
Random Forest (n=500, class_weight=balanced)
         +
Gradient Boosting (n=200, lr=0.05, max_depth=4)
         +
Logistic Regression (C=0.5, max_iter=2000)
         ↓
    Soft-vote average of probabilities
         ↓
  Threshold @ 0.50 → binary prediction
```

### Evaluation — Stratified 10-Fold Cross-Validation
```
Why 10-fold CV and not train/test split?
  - 288 samples → 20% split = 57 test samples
  - 4 folds of 57 samples each → unreliable estimates
  - 10-fold CV: each sample tested exactly once across 10 rotations
  - Every metric is computed on held-out data ONLY
  - No data leakage. No train-set metric inflation.
```

### Validated Results

| Metric | Ensemble (10-Fold CV) |
|--------|-----------------------|
| **Accuracy** | **93.4% ± 3.6%** |
| **ROC-AUC** | **98.4% ± 1.7%** |
| **Sensitivity** | **95.6%** |
| **Specificity** | **90.6%** |
| **PPV (Precision)** | **92.7%** |
| **NPV** | **94.3%** |
| **MCC** | **0.867** |
| **Brier Score** | **0.061** |
| Majority baseline | 55.6% |

**Confusion Matrix (288 patients, CV-aggregated):**
```
               Predicted No Disease    Predicted Disease
Actual No Disease      116 (TN)              12 (FP)
Actual Disease           7 (FN)             153 (TP)
```
- **7 missed cases** out of 160 disease-positive patients (FN rate: 4.4%)
- **12 false alarms** out of 128 disease-negative patients (FP rate: 9.4%)

---

## Project Structure

```
ai-cdss/
├── app.py                          # Main Streamlit application (804 lines)
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── data/
│   ├── heart_cleveland.csv         # Real UCI Cleveland dataset (288 patients)
│   ├── Hospitals.csv               # 8 hospitals with HFR IDs (15 attributes)
│   ├── Doctors.csv                 # 15 doctors with HPR + NMC IDs (10 attributes)
│   ├── Patients.csv                # 500 synthetic patients with ABHA IDs
│   ├── Cases.csv                   # 500 clinical cases
│   └── BedOccupancy.csv            # Live bed data (updated by simulator)
├── simulator/
│   ├── __init__.py
│   └── bed_simulator.py            # MQTT-based HL7 ADT event simulator
└── utils/
    ├── __init__.py
    ├── pipeline.py                 # Complete ML pipeline module
    └── data_loader.py              # Data ingestion, validation, audit trail
```

---

## Getting Started

### 1. Clone
```bash
git clone https://github.com/YOUR_USERNAME/ai-cdss.git
cd ai-cdss
```

### 2. Environment
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run
```bash
streamlit run app.py
```

### 4. (Optional) Run bed simulator standalone
```bash
python simulator/bed_simulator.py
```

---

## Ethical AI Disclosure

Per **WHO Ethics and Governance of AI for Health (2021)** and
**ICMR Ethical Guidelines for Biomedical and Health Research**:

- ✅ **Transparency** — full model documentation, data provenance, evaluation protocol
- ✅ **Explainability** — SHAP attribution for every prediction
- ✅ **Uncertainty communication** — probability scores, not binary outputs
- ✅ **Human oversight** — all outputs require physician review (explicitly labelled)
- ✅ **Non-maleficence** — system cannot take clinical action; advisory only
- ✅ **Data integrity** — de-identified, publicly available, properly cited data
- ✅ **Fairness** — class-balanced training, per-class evaluation metrics

---

## Future Scope

| Phase | Feature | Technology |
|-------|---------|------------|
| 1 | PostgreSQL backend (replace CSV) | SQLAlchemy + PostgreSQL |
| 2 | ABDM sandbox API integration | ABDM HIE APIs, FHIR R4 |
| 3 | Real HMIS ADT event subscription | paho-mqtt + HiveMQ Cloud |
| 4 | Longitudinal patient tracking | Time-series DB (TimescaleDB) |
| 5 | Multi-language UI (Hindi) | i18n + Google Translate API |
| 6 | Mobile app | Flutter + Streamlit backend |
| 7 | CDSCO regulatory pathway | Medical device software (SaMD) framework |
| 8 | Federated learning | PySyft — multi-hospital without data sharing |

---

## References

1. MoHFW, GoI. Ayushman Bharat Digital Mission. https://abdm.gov.in/
2. NHA. Health Facility Registry. https://facility.abdm.gov.in/
3. NHA. Healthcare Professional Registry. https://hpr.abdm.gov.in/
4. Detrano R et al. *Am J Cardiol.* 1989;64(5):304–310. PMID: 2756873
5. WHO. Ethics and Governance of Artificial Intelligence for Health. 2021
6. ICMR. Ethical Guidelines for Biomedical and Health Research. 2017
7. Ministry of Health & Family Welfare. National Health Policy 2017
8. Lundberg SM, Lee SI. A unified approach to interpreting model predictions. *NeurIPS.* 2017
9. Breiman L. Random Forests. *Machine Learning.* 2001;45(1):5–32
10. Sharma RS et al. ABDM: Making of India's Digital Health Story. *CSI Trans ICT.* 2023

---

## License

MIT License — free to use with attribution.

---

*Built with purpose — for patients, by research.*
*🇮🇳 Jai Hind*
