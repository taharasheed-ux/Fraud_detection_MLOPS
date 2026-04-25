# 🚀 Fraud Detection MLOps System

A production-grade fraud detection system built on the IEEE-CIS dataset with end-to-end MLOps capabilities.

## 🧱 Architecture

```
┌─────────────────────────────────────────────────────┐
│                   ML Pipeline                       │
│  Ingestion → Validation → Preprocessing →           │
│  Feature Engineering → Training → Evaluation →      │
│  Conditional Deployment                             │
└───────────────┬─────────────────────────────────────┘
                │ MLflow (Tracking + Registry)
                ▼
┌─────────────────────────────────────────────────────┐
│              FastAPI Inference API                   │
│  POST /predict │ GET /health │ GET /metrics          │
└───────────────┬─────────────────────────────────────┘
                │ Prometheus Metrics
                ▼
┌─────────────────────────────────────────────────────┐
│           Monitoring & Alerting                      │
│  Prometheus → Alertmanager → Webhook → Retraining   │
│  Grafana Dashboards (System / Model / Drift)        │
└─────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
├── src/                    # ML pipeline modules
│   ├── ingestion.py        # Data loading & merging
│   ├── validation.py       # Schema & data validation
│   ├── preprocessing.py    # Imputation, scaling, outlier handling
│   ├── feature_engineering.py # Time features, aggregations, encoding
│   ├── train.py            # XGBoost, LightGBM, RF hybrid training
│   ├── evaluate.py         # Metrics, SHAP, drift simulation
│   └── deploy.py           # Conditional deployment gate
├── api/
│   └── app.py              # FastAPI inference server
├── monitoring/
│   ├── prometheus.yml      # Prometheus scrape config
│   ├── alert_rules.yml     # Alert rules (recall, FPR, drift, latency)
│   ├── alertmanager.yml    # Alertmanager webhook config
│   └── grafana/            # Dashboard & datasource provisioning
├── docker/
│   ├── Dockerfile.pipeline # Training container
│   ├── Dockerfile.api      # API container
│   └── docker-compose.yml  # Full monitoring stack
├── .github/workflows/
│   └── ci.yml              # CI/CD pipeline
├── run_pipeline.py         # Pipeline orchestrator
├── requirements.txt        # Python dependencies
└── README.md
```

## ⚡ Quick Start

### 1. Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Dataset

Place the IEEE-CIS Fraud Detection dataset files in `data/`:
- `train_transaction.csv`
- `train_identity.csv`
- `test_transaction.csv`
- `test_identity.csv`

### 3. Run the Pipeline

```bash
python run_pipeline.py
```

This executes all 7 stages: Ingestion → Validation → Preprocessing → Feature Engineering → Training → Evaluation → Deployment.

### 4. View MLflow Results

```bash
mlflow ui --host 0.0.0.0 --port 5000
# Open http://localhost:5000
```

### 5. Start the API

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
# Health: http://localhost:8000/health
# Docs:   http://localhost:8000/docs
```

### 6. Launch Monitoring Stack

```bash
cd docker
docker-compose up -d
# Grafana:      http://localhost:3000 (admin/admin)
# Prometheus:   http://localhost:9090
# Alertmanager: http://localhost:9093
```

## 🤖 Models Trained

| Model | Variants |
|-------|----------|
| **XGBoost** | Class-weight, SMOTE, Cost-sensitive (scale_pos_weight=10) |
| **LightGBM** | Class-weight, SMOTE |
| **Random Forest (Hybrid)** | Class-weight + top-50 feature selection, SMOTE + feature selection |

## 📊 Evaluation

- **Metrics**: Precision, Recall, F1-score, AUC-ROC, Confusion Matrix
- **Comparisons**: SMOTE vs Class-weight, Cost-sensitive vs Standard
- **Explainability**: SHAP summary plots for the best model
- **Drift Simulation**: Time-based 80/20 split to detect performance degradation

## 🚦 Deployment Gate

Models are deployed only if:
```python
if recall > 0.80 and auc_roc > 0.85:
    deploy_model()
```

## 🔔 Alerting

| Alert | Condition | Severity |
|-------|-----------|----------|
| LowFraudRecall | recall < 0.80 for 5m | Critical |
| HighFalsePositiveRate | FPR > 0.20 for 5m | Warning |
| DataDriftDetected | drift > 0.30 for 10m | Critical |
| HighLatency | p95 latency > 1s for 2m | Warning |

Alerts flow: **Prometheus → Alertmanager → Webhook → CI/CD → Retraining**

## 📦 Docker

```bash
# Training pipeline with resource limits
docker run --memory=4g --cpus=2 fraud-pipeline

# Full stack
cd docker && docker-compose up -d
```
