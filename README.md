# Fraud Risk Detection — Full-Stack ML Platform

End-to-end ML platform for detecting fraudulent fundraisers. Covers the full lifecycle: streaming data ingestion, distributed training (LoRA + XGBoost), ensemble inference, a React analyst dashboard, and continuous monitoring with feedback loops.

---

## Architecture

```
                        ┌─────────────────────────────┐
                        │    React Dashboard (Vite)    │
                        │  - Fraud queue review        │
                        │  - Model metrics & drift     │
                        │  - SHAP explanations         │
                        └────────────┬────────────────┘
                                     │ GraphQL (Apollo)
                        ┌────────────▼────────────────┐
                        │   Node.js BFF (Express)      │
                        │   GraphQL Gateway            │
                        │   Auth / rate limiting       │
                        └────────────┬────────────────┘
                              REST   │   gRPC
              ┌────────────────────┬─┴──────────────────┐
              │                    │                     │
   ┌──────────▼──────┐  ┌─────────▼──────┐  ┌──────────▼──────┐
   │  FastAPI ML API  │  │  Feedback Svc  │  │  Monitoring Svc  │
   │  /predict        │  │  (gRPC)        │  │  (gRPC)          │
   │  /batch_predict  │  │  label storage │  │  drift reports   │
   │  /explain        │  └────────────────┘  └──────────────────┘
   └──────────┬───────┘
              │ loads
   ┌──────────▼──────────────────────────────────────────┐
   │              Ensemble Predictor                      │
   │   LoRA Gemma-2B (text) + XGBoost (tabular)          │
   │   Weighted late fusion → fraud_score ∈ [0,1]        │
   └────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   Data & Training Layer                      │
│  Kafka → Spark (batch ETL) → Feature Store (Parquet/GCS)   │
│  Ray Train → LoRA fine-tune (GCP T4) + XGBoost (local)     │
│  MLflow → Experiment tracking, Model Registry              │
└─────────────────────────────────────────────────────────────┘
```

---

## Folder Structure

| Folder | Purpose |
|--------|---------|
| `data/` | Raw CSV data — `train_data/` and `test_data/` |
| `ml_training_service/` | Data ingestion, Spark ETL, Kafka producer, feature engineering, LoRA + XGBoost training, Optuna tuning, training pipeline |
| `ml_evaluation_pipeline/` | Metrics (AUC-ROC, PR-AUC, F1), SHAP explainability, bias/slice evaluation, eval orchestration |
| `ml_model_packaging_service/` | FastAPI REST serving, gRPC Feedback Service, Node.js GraphQL BFF, React analyst dashboard |
| `ml_observability_monitoring/` | Evidently drift detection, rolling AUC tracking, alerting, gRPC Monitoring Service |
| `ml_infra_deployment_ops/` | Docker Compose, Kafka topic setup, Spark config, GCP VM scripts, Makefile |

---

## Local Setup

### 1. Conda Environment

```bash
conda create -n fraud-risk python=3.11 -y
conda activate fraud-risk
```

Or restore from the environment file:

```bash
conda env create -f environment.yml
conda activate fraud-risk
```

### 2. Install Python Dependencies

```bash
conda activate fraud-risk
make env-install
```

This installs all `requirements.txt` files across all services.

### 3. Start Infrastructure (Kafka, Spark, MLflow)

```bash
make infra-up
```

Services started:
- Kafka: `localhost:9092`
- MLflow UI: `http://localhost:5000`
- Spark UI: `http://localhost:8080`

### 4. Create Kafka Topics

```bash
make kafka-init
```

Topics created: `fraud-risk.fund-events`, `fraud-risk.predictions`, `fraud-risk.labels`

---

## GCP Setup (for LoRA Training)

LoRA fine-tuning runs on a GCP Compute Engine T4 GPU VM (~$0.54/hr). **Always stop the VM after training.**

### One-time Setup

```bash
# Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Create GCS buckets
bash ml_infra_deployment_ops/gcp/gcs_buckets.sh

# Create GPU VM
make vm-create
```

### Train LoRA on GCP

```bash
make vm-train-lora   # SSH → train → push adapter to GCS → stop VM
```

LoRA adapter is saved to `gs://fraud-risk-models/lora-adapters/`.

---

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make env-install` | Install all Python dependencies into conda env |
| `make infra-up` | Start Kafka, Spark, MLflow via Docker Compose |
| `make infra-down` | Stop all infrastructure containers |
| `make kafka-init` | Create Kafka topics |
| `make spark-etl` | Run Spark batch ETL: raw CSVs → feature Parquet |
| `make stream-data` | Start Kafka producer (simulate live fund events) |
| `make vm-create` | Create GCP Compute Engine T4 GPU VM |
| `make vm-train-lora` | SSH into VM, run LoRA training, push adapter to GCS, stop VM |
| `make train-xgb` | Train XGBoost with Ray (local) |
| `make tune` | Run Optuna hyperparameter search |
| `make evaluate` | Full evaluation: metrics + SHAP + bias slices |
| `make serve` | Start FastAPI ML API (localhost:8000) |
| `make dev` | Start React frontend dev server (localhost:5173) |
| `make build` | Build all Docker images |
| `make pipeline` | Run full end-to-end training pipeline |
| `make test` | Run all pytest tests |

---

## Service Endpoints

| Service | Protocol | Address | Description |
|---------|----------|---------|-------------|
| FastAPI ML API | REST | `http://localhost:8000` | Predict, batch predict, explain |
| Node.js BFF | GraphQL | `http://localhost:4000/graphql` | Apollo GraphQL gateway |
| React Dashboard | HTTP | `http://localhost:5173` | Analyst fraud review UI |
| MLflow UI | HTTP | `http://localhost:5000` | Experiment tracking |
| Feedback Service | gRPC | `localhost:50051` | Submit human review labels |
| Monitoring Service | gRPC | `localhost:50052` | Drift reports, rolling metrics |
| Kafka | TCP | `localhost:9092` | Event streaming |
| Spark | HTTP | `http://localhost:8080` | Spark master UI |

### FastAPI Endpoints

```
POST /predict              — Single fund fraud prediction
POST /batch_predict        — Batch prediction (list of fund records)
GET  /explain/{fund_id}    — SHAP explanation for a specific fund
GET  /health               — Model version and health status
```

### GraphQL Schema (key queries)

```graphql
query {
  fraudQueue(minScore: 0.8, limit: 50) {
    fundId
    title
    fraudScore
    topFeatures { name value contribution }
  }
  modelMetrics(window: "7d") {
    aucRoc
    prAuc
    precisionAtK
    rollingAuc
  }
  driftReport {
    driftDetected
    driftedFeatures
    generatedAt
  }
}

mutation {
  submitReview(fundId: "12345", isFraud: true, confidence: 0.9)
}
```

---

## Training Guide

### XGBoost (runs locally)

```bash
conda activate fraud-risk
make spark-etl          # Build feature store first
make train-xgb          # Train XGBoost with Ray
```

Model registered to MLflow at `http://localhost:5000`.

### LoRA Fine-tuning (runs on GCP T4)

Dataset: `data/train_data/technique2_train_balanced_200.csv` (200 balanced samples)
Base model: `google/gemma-2b`

```bash
make vm-create          # One-time: create T4 GPU VM
make vm-train-lora      # Train + save adapter to GCS + stop VM
```

LoRA config: `ml_training_service/configs/lora_config.yaml`
- `r=16`, `lora_alpha=32`, `target_modules=[q_proj, v_proj]`
- Task: sequence classification (fraud / not fraud)

### Hyperparameter Tuning

```bash
make tune               # Optuna study for XGBoost params
```

Best params logged as MLflow run tags. Best model promoted to `Staging`.

---

## Evaluation Guide

```bash
conda activate fraud-risk
make evaluate
```

Produces:
- **AUC-ROC, PR-AUC, F1** at Youden-J optimal threshold
- **SHAP feature importance** (global bar chart + per-prediction waterfall)
- **Bias/slice report** — AUC per `category_id` and `phone_line_type`
- All artifacts logged to MLflow

Target performance: ensemble AUC > 0.82 on `fraud_data_test.csv` (25K rows).

---

## Monitoring Guide

### Data Drift

```bash
conda activate fraud-risk
python ml_observability_monitoring/drift.py
```

Uses Evidently `DataDriftPreset` to compare training distribution vs recent inference logs. Saves HTML report to `gs://fraud-risk-logs/drift-reports/`.

### Rolling Performance

```bash
python ml_observability_monitoring/performance.py
```

Computes rolling AUC, precision, recall from `gs://fraud-risk-logs/inference-logs/` when ground-truth labels are available.

### Alerts

Configured in `ml_infra_deployment_ops/configs/serving_config.yaml`. Fires when:
- Feature drift p-value < 0.05
- Rolling AUC drops > 3% from baseline

### Feedback Loop

Human reviews submitted via the React dashboard → gRPC Feedback Service → Kafka `fraud-risk.labels` topic → retraining triggered when ≥ 100 new labels accumulate.

---

## Tech Stack

| Layer | Tool |
|-------|------|
| Frontend | React 18, Vite, TypeScript, Apollo Client, Recharts |
| API Gateway | Node.js, Express, Apollo Server (GraphQL) |
| ML Serving | FastAPI, Pydantic, Uvicorn |
| Internal comms | gRPC (protobuf), grpcio |
| Streaming | Apache Kafka (Confluent), kafka-python |
| Batch ETL | Apache Spark (PySpark) |
| Distributed training | Ray Train (TransformersTrainer + XGBoostTrainer) |
| LoRA fine-tuning | HuggingFace PEFT, transformers, accelerate |
| Base LLM | google/gemma-2b |
| Tabular model | XGBoost, LightGBM |
| Experiment tracking | MLflow |
| Hyperparameter tuning | Optuna |
| Explainability | SHAP |
| Drift monitoring | Evidently AI |
| Cloud storage | Google Cloud Storage (GCS) |
| GPU training | GCP Compute Engine (n1-standard-4 + T4) |
| Containerization | Docker, Docker Compose |
| Environment | conda (fraud-risk, Python 3.11) |
