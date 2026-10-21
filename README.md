# 🎗️ InnovSense – Breast Cancer Recurrence Prediction

> An AI-powered clinical decision-support system that estimates the risk of breast cancer recurrence after chemotherapy, with explainable predictions, a risk dashboard and a doctor view.

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn%20%7C%20XGBoost-orange)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-purple)

---

## 📑 Table of Contents
1. [Overview](#-overview)
2. [Features](#-features)
3. [Tech Stack](#-tech-stack)
4. [System Architecture](#-system-architecture)
5. [Project Structure](#-project-structure)
6. [Dataset](#-dataset)
7. [Machine Learning Pipeline](#-machine-learning-pipeline)
8. [Results](#-results)
9. [Screenshots](#-screenshots)
10. [Installation & Setup](#-installation--setup)
11. [API Endpoints](#-api-endpoints)
12. [Configuration](#-configuration)
13. [Docker](#-docker)
14. [Limitations & Future Work](#-limitations--future-work)
15. [Disclaimer](#-disclaimer)

---

## 🔍 Overview

**InnovSense** predicts the likelihood of breast cancer recurrence from a patient's clinical data. A **FastAPI** backend serves trained ML models (XGBoost, Random Forest, Logistic Regression), and a multi-page **Streamlit** frontend lets patients and clinicians enter data, view risk levels and understand *why* a prediction was made.

Patients are grouped into three risk levels (default thresholds):

| Risk Level | Probability |
|------------|-------------|
| 🟢 Low     | < 0.30 |
| 🟡 Medium  | 0.30 – 0.60 |
| 🔴 High    | > 0.60 |

---

## ✨ Features

- 🧮 **Recurrence risk prediction** using three ML models
- 🧠 **Explainable AI** – SHAP-based explanations, with coefficient-based and clinical-knowledge fallbacks
- 📊 **Risk dashboard** – stats, prediction timeline, risk summary, recent patients, model performance
- 👩‍⚕️ **Doctor view** for clinical review
- 🚨 **Alert service** based on risk thresholds
- 🗃️ **Patient & prediction history** stored in a database (SQLite by default, PostgreSQL via Docker)
- 📖 Auto-generated API docs (Swagger / ReDoc)
- 🐳 Docker & Windows one-click launch support

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|--------------|
| **Frontend** | Streamlit, Plotly, Pandas, NumPy, Requests |
| **Backend** | FastAPI, Uvicorn, SQLAlchemy, Pydantic, Alembic |
| **Machine Learning** | scikit-learn, XGBoost, SHAP, Joblib, Matplotlib, Seaborn |
| **Database** | SQLite (default) / PostgreSQL 13 |
| **DevOps** | Docker, Docker Compose |

---

## 🏗️ System Architecture

```
┌────────────────────┐   HTTP/JSON   ┌───────────────────────────┐
│  Streamlit UI      │ ────────────► │  FastAPI Backend          │
│  (port 8501)       │ ◄──────────── │  (port 8000)              │
│  - Patient Input   │               │  - /prediction            │
│  - Risk Dashboard  │               │  - /patients              │
│  - Doctor View     │               │  - /models                │
│  - About           │               │  - /dashboard             │
└────────────────────┘               └──────┬─────────────┬──────┘
                                            │             │
                                  ┌─────────▼───┐   ┌─────▼─────────┐
                                  │ ML Service  │   │ Database      │
                                  │ + SHAP      │   │ SQLite/Postgres│
                                  │ + Alerts    │   └───────────────┘
                                  └─────────────┘
```

---

## 📂 Project Structure

```
INNOVSENSE-PROJECT-BREAST-CANCER-PREDICTION/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI entry point
│   │   ├── api/endpoints/           # prediction, patients, models, dashboard routes
│   │   ├── core/                    # config, database, security
│   │   ├── models/                  # SQLAlchemy models (patient, prediction, user)
│   │   ├── services/                # ml_service, explainability, alert_service
│   │   └── utils/                   # preprocessing, validators
│   ├── ml_models/
│   │   ├── data_preparation.py      # Data loading & preprocessing
│   │   ├── train_model.py           # Trains XGBoost / RF / Logistic Regression
│   │   ├── train_synthetic.py       # Trains on synthetic data (for testing)
│   │   └── model_evaluation.py      # Evaluation utilities
│   ├── data/raw/METABRIC_RNA_Mutation.csv
│   ├── saved_models/                # Trained models & evaluation_results.json
│   ├── feature_importance.png
│   ├── check_features.py
│   ├── test_setup.py
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── streamlit_app.py             # Main Streamlit app
│   ├── pages/
│   │   ├── 1_Patient_Input.py
│   │   ├── 2_Risk_Dashboard.py
│   │   ├── 3_Doctor_View.py
│   │   └── About.py
│   ├── components/                  # risk_meter, charts, disclaimer
│   ├── utils/api_client.py
│   ├── notebooks/                   # exploratory_analysis, model_comparison
│   ├── tests/                       # test_api.py, test_model.py
│   ├── docker-compose.yml
│   ├── run_windows.bat
│   └── requirements.txt
└── README.md
```

---

## 📊 Dataset

- **Source:** METABRIC (Molecular Taxonomy of Breast Cancer International Consortium) – `METABRIC_RNA_Mutation.csv` (~1,900 patients).
- **Target:** binary label derived from `death_from_cancer` (`Died of Disease` = 1, otherwise 0), used as a proxy for recurrence.
- **Input features (10):**

| Clinical | Treatment | Biomarkers |
|----------|-----------|------------|
| `age_at_diagnosis` | `chemotherapy` | `er_status` |
| `tumor_size` | `hormone_therapy` | `pr_status` |
| `lymph_nodes_examined_positive` | `radio_therapy` | `her2_status` |
| `neoplasm_histologic_grade` | | |

---

## 🤖 Machine Learning Pipeline

1. **Load** the METABRIC CSV (`data_preparation.py`)
2. **Create target** from `death_from_cancer`
3. **Select** the 10 clinical features
4. **Handle missing values** and **encode** categorical variables
5. **Scale** with `StandardScaler` and split into train / test
6. **Train** XGBoost, Random Forest and Logistic Regression (`train_model.py`)
7. **Evaluate** with accuracy, precision, recall, F1, ROC-AUC, 5-fold stratified CV and confusion matrices
8. **Save** models and `evaluation_results.json` to `saved_models/`
9. **Serve** via `ml_service.py` with SHAP explanations (`explainability.py`)

### Feature Importance
<img width="2085" height="881" alt="feature_importance" src="https://github.com/user-attachments/assets/63d37e02-1b5d-4db3-88ed-3432af916521" />



Tumor size, age at diagnosis, positive lymph nodes, histologic grade and HER2 status are the most influential features across the tree-based models.

---

## 📈 Results

Evaluated on a held-out test set of 200 samples:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV Mean ± Std |
|-------|:--------:|:---------:|:------:|:--:|:-------:|:-------------:|
| XGBoost | 0.700 | 0.553 | 0.328 | 0.412 | 0.652 | 0.579 ± 0.095 |
| Random Forest | **0.715** | **0.630** | 0.266 | 0.374 | 0.649 | 0.573 ± 0.119 |
| Logistic Regression | 0.660 | 0.471 | **0.516** | **0.493** | **0.661** | 0.574 ± 0.069 |

> **Logistic Regression** is the default model (`DEFAULT_MODEL`) because it has the best recall, F1 and ROC-AUC – the most relevant metrics when missing a true recurrence case is costly.

## [Model Comparison]
<img width="1500" height="750" alt="model_comparison" src="https://github.com/user-attachments/assets/b6885672-8ce6-4601-9424-7691fc10a1c2" />


## [Confusion Matrices]
<img width="1800" height="600" alt="confusion_matrices" src="https://github.com/user-attachments/assets/9ec22ed9-32d5-4382-9c8f-73c39cc5890b" />


---

## 🖼️ Screenshots

> 📌 Run the app, take screenshots and save them in the `assets/` folder with the names below – they will appear automatically.

| Page | Screenshot |
|------|-----------|
| Home | ![Home](assets/home.png) |
| Patient Input | ![Patient Input](assets/patient_input.png) |
| Risk Dashboard | ![Risk Dashboard](assets/risk_dashboard.png) |
| Doctor View | ![Doctor View](assets/doctor_view.png) |

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.9+
- pip & virtualenv
- (Optional) Docker & Docker Compose

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/INNOVSENSE-PROJECT-BREAST-CANCER-PREDICTION.git
cd INNOVSENSE-PROJECT-BREAST-CANCER-PREDICTION
```

### 2. Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt

# Train the models (creates files in saved_models/)
cd ml_models
python train_model.py
cd ..

# Start the API
uvicorn app.main:app --reload --port 8000
```
- API: http://localhost:8000
- Swagger docs: http://localhost:8000/api/docs
- ReDoc: http://localhost:8000/api/redoc

### 3. Frontend
```bash
cd frontend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```
- App: http://localhost:8501

### 4. One-click start (Windows)
```bash
frontend\run_windows.bat
```
(Expects a `venv` inside both `backend/` and `frontend/`.)

### 5. Tests
```bash
cd frontend
pytest tests/
```

---

## 🔌 API Endpoints

Base URL: `http://localhost:8000/api/v1`

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API welcome message |
| GET | `/health` | Health check |
| POST | `/prediction/predict` | Predict recurrence risk for a patient |
| GET | `/prediction/history/{patient_id}` | Prediction history of a patient |
| GET | `/prediction/features` | Features used by the model |
| GET | `/patients/` | List patients |
| GET | `/patients/{patient_id}` | Patient details |
| GET | `/models/info` | Model information |
| GET | `/models/performance` | Model performance metrics |
| GET | `/dashboard/stats` | Overall statistics |
| GET | `/dashboard/predictions/timeline` | Predictions over time |
| GET | `/dashboard/patients/recent` | Recent patients |
| GET | `/dashboard/risk/summary` | Risk-level distribution |
| GET | `/dashboard/model/performance` | Model performance for dashboard |

---

## ⚙️ Configuration

Create a `.env` file in `backend/` to override defaults:

```env
DATABASE_URL=sqlite:///./innovsense.db
SECRET_KEY=change-this-in-production
MODEL_PATH=./saved_models/
DEFAULT_MODEL=LogisticRegression
LOW_RISK_THRESHOLD=0.3
MEDIUM_RISK_THRESHOLD=0.6
```

---

## 🐳 Docker

```bash
cd frontend
docker-compose up --build
```

| Service | Port |
|---------|------|
| Backend (FastAPI) | 8000 |
| Frontend (Streamlit) | 8501 |
| Database (PostgreSQL) | 5432 |

---

## ⚠️ Limitations & Future Work

- Current models reach only moderate performance (ROC-AUC ≈ 0.65); recall for recurrence is low for the tree-based models.
- The target is a **proxy** (death from cancer), not true recurrence.
- Training currently uses a subset of the dataset (first 1,000 rows) – using all ~1,900 records should improve results.
- Class imbalance handling (e.g. SMOTE, class weights) and hyperparameter tuning are future improvements.
- Possible additions: gene-expression/mutation features, user authentication, deployment (Render / cloud), and clinical validation.

---

## 🩺 Disclaimer

This project is for **educational and research purposes only**. It is **not** a medical device and must **not** be used for diagnosis or treatment decisions. Always consult a qualified healthcare professional.

---

## 👥 Team

**Team InnovSense** – Aakash Yadav

## 📄 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.
More about the license: [opensource.org/licenses/MIT](https://opensource.org/licenses/MIT)
