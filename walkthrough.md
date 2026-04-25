# 🚀 Fraud Detection MLOps System — Complete Execution Guide

## What You've Built

A **production-grade fraud detection system** using the IEEE-CIS dataset with a full MLOps stack:

| Component | Files | What It Does |
|-----------|-------|-------------|
| **ML Pipeline** | [run_pipeline.py](file://wsl.localhost/Ubuntu/home/tahar/MLOPS_Assignment_4/run_pipeline.py) + `src/*.py` | 7-stage pipeline: Ingestion → Validation → Preprocessing → Feature Eng. → Training → Evaluation → Deployment |
| **API Server** | [api/app.py](file://wsl.localhost/Ubuntu/home/tahar/MLOPS_Assignment_4/api/app.py) | FastAPI inference with `/predict`, `/health`, `/metrics` |
| **Monitoring** | `monitoring/` | Prometheus, Alertmanager, Grafana (3 dashboards) |
| **Containers** | `docker/` | Dockerfile.pipeline, Dockerfile.api, docker-compose.yml |
| **CI/CD** | [.github/workflows/ci.yml](file://wsl.localhost/Ubuntu/home/tahar/MLOPS_Assignment_4/.github/workflows/ci.yml) | 4-stage GitHub Actions (lint, test, build, retrain-on-dispatch) |
| **Data Download** | [data_extraction.py](file://wsl.localhost/Ubuntu/home/tahar/MLOPS_Assignment_4/data_extraction.py) | Kaggle → [data/](file://wsl.localhost/Ubuntu/home/tahar/MLOPS_Assignment_4/src/train.py#40-65) directory |

### Pipeline Architecture

```
data_extraction.py → Downloads IEEE-CIS CSVs (~1.3 GB)
         │
run_pipeline.py (orchestrator with @retry × 3 on every step)
         │
  ┌──────┼──────────────────────────────────────────────┐
  │  1. ingestion.py   → Merge transaction+identity,   │
  │                       downcast dtypes, save parquet │
  │  2. validation.py  → Schema, dtype, missing checks  │
  │  3. preprocessing.py→ Impute (median/mode), scale,  │
  │                       clip outliers, missing flags   │
  │  4. feature_eng.py → Time feats, card1 aggs,        │
  │                       freq/target encoding           │
  │  5. train.py       → 7 model variants:               │
  │       XGBoost (class-wt, SMOTE, cost-sensitive)      │
  │       LightGBM (class-wt, SMOTE)                     │
  │       RF Hybrid (class-wt+feat-sel, SMOTE+feat-sel)  │
  │  6. evaluate.py    → Metrics, confusion matrices,    │
  │                       SHAP, drift simulation,         │
  │                       comparison tables               │
  │  7. deploy.py      → Gate: recall>0.80 & AUC>0.85   │
  │                       → MLflow registry + local save  │
  └─────────────────────────────────────────────────────┘
         │
  artifacts/deployed_model/ → model.joblib + feature_columns.joblib
         │
  api/app.py → FastAPI serves predictions + Prometheus metrics
         │
  docker-compose.yml → API + Prometheus + Alertmanager + Grafana
```

### Current State

Your **data ingestion has already been completed** — the `data/processed/` directory contains:
- `train_merged.parquet` (~84 MB)
- `test_merged.parquet` (~73 MB)

The remaining pipeline steps (preprocessing → deployment) have **not yet been executed**.

---

## Step-by-Step Execution Guide

### Phase 1: Environment Setup

```bash
# Navigate to the project
cd ~/MLOPS_Assignment_4

# Create virtual environment (if not already done)
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

> [!NOTE]
> Your `venv/` directory already exists, so the venv is likely created. Just activate it and install dependencies.

---

### Phase 2: Download Dataset (Already Done ✅)

The `data/` directory already has all CSVs. If you ever need to re-download:

```bash
python data_extraction.py
```

> [!IMPORTANT]
> This requires Kaggle API credentials (`~/.kaggle/kaggle.json`). You already have the data, so skip this.

---

### Phase 3: Run the Full ML Pipeline

This is the **main execution step**. It runs all 7 stages end-to-end with MLflow tracking:

```bash
# Make sure venv is activated
source venv/bin/activate

# Run the complete pipeline
python run_pipeline.py
```

**What happens:**
1. **Ingestion** — Loads the merged parquets (already saved)
2. **Validation** — Checks schema, dtypes, missing values
3. **Preprocessing** — Imputes missing values, scales features, clips outliers
4. **Feature Engineering** — Adds time features, aggregations, frequency/target encoding
5. **Training** — Trains 7 model variants (XGBoost ×3, LightGBM ×2, RF Hybrid ×2)
6. **Evaluation** — Computes all metrics, generates SHAP plots, runs drift simulation
7. **Deployment** — Deploys best model if recall > 0.80 AND AUC > 0.85

> [!WARNING]
> **This step is memory-intensive** (the dataset is ~600K rows × 400+ columns). Make sure you have at least **8 GB of free RAM**. It may take **15-30 minutes** depending on your machine.

**Expected output files after pipeline completes:**

| Path | Contents |
|------|----------|
| `data/processed/train_preprocessed.parquet` | Preprocessed training data |
| `data/processed/test_preprocessed.parquet` | Preprocessed test data |
| `data/processed/train_featured.parquet` | Feature-engineered training data |
| `data/processed/test_featured.parquet` | Feature-engineered test data |
| `artifacts/preprocessing/*.joblib` | Saved imputers & scaler |
| `artifacts/features/*.joblib` | Saved encoders |
| `artifacts/models/*.joblib` | All 7 trained models |
| `artifacts/evaluation/*.png` | Confusion matrices & SHAP plots |
| `artifacts/evaluation/*.csv` | Model comparison tables |
| `artifacts/deployed_model/model.joblib` | The deployed model (if passes gate) |
| `mlruns/` | MLflow experiment tracking data |
| `pipeline.log` | Full pipeline log |

---

### Phase 4: View MLflow Dashboard

After the pipeline completes, inspect all experiments:

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

Open in browser: **http://localhost:5000**

You'll see:
- The `fraud-detection-experiments` experiment
- A parent run `full-pipeline` with nested runs per model
- Logged parameters, metrics (recall, precision, F1, AUC-ROC), and artifacts (confusion matrices, SHAP plots, comparison CSVs)

---

### Phase 5: Start the FastAPI Inference Server

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

**Test endpoints:**

```bash
# Health check
curl http://localhost:8000/health

# Swagger UI docs (in browser)
# http://localhost:8000/docs

# Prometheus metrics
curl http://localhost:8000/metrics

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"TransactionAmt": 100.0, "card1": 1234, "ProductCD_freq": 0.3, "hour": 14, "dayofweek": 3}}'
```

> [!IMPORTANT]
> The API loads the model from `artifacts/deployed_model/model.joblib`. This file only exists if the pipeline's deployment gate passed (recall > 0.80 AND AUC > 0.85). If the model was rejected, the `/health` endpoint will show `"status": "no_model"`.

---

### Phase 6: Launch the Monitoring Stack (Docker)

Make sure Docker is installed, then:

```bash
cd ~/MLOPS_Assignment_4/docker
docker-compose up -d
```

This starts 4 services:

| Service | URL | Credentials |
|---------|-----|-------------|
| **Fraud API** | http://localhost:8000 | — |
| **Prometheus** | http://localhost:9090 | — |
| **Alertmanager** | http://localhost:9093 | — |
| **Grafana** | http://localhost:3000 | admin / admin |

**Grafana has 3 pre-provisioned dashboards:**
1. **System Health** — API latency, request rate, error rate
2. **Model Performance** — Recall trends, fraud detection rate, confidence distribution
3. **Data Drift** — Feature drift scores, missing value trends

**Prometheus alert rules are configured for:**
- `LowFraudRecall` — recall < 0.80 for 5m → **critical**
- `HighFalsePositiveRate` — FPR > 0.20 for 5m → **warning**  
- `DataDriftDetected` — drift > 0.30 for 10m → **critical**
- `HighLatency` — p95 > 1s for 2m → **warning**
- `HighErrorRate` — >5% errors for 5m → **critical**

**Alert flow:** Prometheus → Alertmanager → Webhook (`http://fraud-api:8000/webhook/retrain`) → CI/CD pipeline

---

### Phase 7: CI/CD (GitHub Actions)

Your CI/CD pipeline (`.github/workflows/ci.yml`) runs automatically on:
- Push to `main` or `develop` branches
- Pull requests to `main`
- `repository_dispatch` event (for retraining triggers)

**4 stages:**
1. **CI** — Linting (flake8), pytest, data validation check
2. **Build** — Docker image builds for pipeline & API
3. **Deploy** — Deployment notification (main branch only)
4. **Retrain** — Triggered via `repository_dispatch` when alerts fire

**To trigger it:**
```bash
# Initialize git and push to GitHub
git init
git add .
git commit -m "Initial commit: fraud detection MLOps system"
git remote add origin https://github.com/YOUR_USERNAME/MLOPS_Assignment_4.git
git push -u origin main
```

The CI/CD pipeline will run automatically on push.

---

## Complete Execution Order Summary

```
1. source venv/bin/activate && pip install -r requirements.txt
2. python run_pipeline.py                      # Full ML pipeline (~15-30 min)
3. mlflow ui --host 0.0.0.0 --port 5000        # View experiments
4. uvicorn api.app:app --host 0.0.0.0 --port 8000  # Start API
5. cd docker && docker-compose up -d           # Start monitoring stack
6. git push origin main                         # Trigger CI/CD
```

> [!TIP]
> You can run steps 3-5 in **separate terminal windows/tabs** to keep all services live simultaneously.

---

## Key Files to Review for Evidence

| Evidence Required | Where to Find It |
|-------------------|-------------------|
| Pipeline logs | `pipeline.log` |
| Model comparison (SMOTE vs class-weight) | `artifacts/evaluation/smote_vs_classweight.csv` |
| Cost-sensitive comparison | `artifacts/evaluation/cost_sensitive_comparison.csv` |
| Full model comparison | `artifacts/evaluation/model_comparison.csv` |
| SHAP explainability | `artifacts/evaluation/shap_summary_*.png` |
| Confusion matrices | `artifacts/evaluation/cm_*.png` |
| Drift simulation | `artifacts/evaluation/drift_simulation_report.csv` |
| MLflow experiments | http://localhost:5000 |
| Grafana dashboards | http://localhost:3000 |
| Prometheus alerts | http://localhost:9090/alerts |
| CI/CD runs | GitHub Actions tab on your repository |
