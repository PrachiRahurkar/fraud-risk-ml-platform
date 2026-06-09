# Building a Fraud Detection Platform for Fundraising Campaigns

*Source code: [github.com/PrachiRahurkar/fraud-risk-ml-platform](https://github.com/PrachiRahurkar/fraud-risk-ml-platform)*

---

## What Is It?

The **Fraud Risk Detection Platform** is a full-stack, end-to-end machine learning system that automatically flags fraudulent fundraising campaigns before they can harm donors. Given a campaign's structured metadata and free-form text — title and description — the platform produces a fraud probability score between 0 and 1, a binary decision label, and a human-readable explanation of the top signals that drove the decision.

Under the hood, the platform combines two complementary models in a late-fusion ensemble:

- **XGBoost** — trained on 12 tabular features capturing account-level and identity-verification signals: goal amount, email domain age, phone line type, KYC score, and more.
- **LoRA-fine-tuned Gemma-2B** — a 2-billion-parameter language model fine-tuned on campaign text to detect rhetorical patterns and linguistic markers common in fraudulent campaigns.

The two model scores are fused as a weighted average (`fraud_score = 0.65 × xgb_score + 0.35 × lora_score`), with the weight optimised on a held-out validation set. The full system spans a React dashboard, a GraphQL BFF, a FastAPI serving layer, a Spark-based training pipeline, and an observability stack that continuously monitors for model drift and triggers retraining when needed.

---

## Why Build This?

Fundraising fraud is uniquely difficult to catch because fraudsters invest effort in making campaigns look legitimate — plausible goals, coherent stories, real-looking contact information. A rule-based system can be gamed; a single-signal model leaves gaps. Three concrete problems drove the architecture choices:

**1. Tabular signals alone miss textual manipulation.**
A fraudster can pass email and phone checks while writing a fabricated or plagiarised campaign description. A language model catches this; a gradient-boosted tree cannot.

**2. Text alone misses account-level risk.**
An LLM cannot directly observe that an account was created yesterday, uses a disposable email, or set an unusually large funding goal. Structured features capture exactly these signals.

**3. Reviewers need explanations.**
A black-box score is not actionable. Human reviewers need to know *why* a campaign was flagged — which three features pushed the score over the threshold — so they can make an informed decision and provide feedback that improves the model.

Beyond the modelling problem, the platform needs to keep working reliably over time. Campaign language and fraud tactics evolve; a model trained once will degrade silently without a continuous monitoring loop. The observability and feedback components are therefore first-class requirements, not afterthoughts.

---

## High-Level Architecture

The platform has four horizontal layers:

1. **User Interface** — a React single-page application where fraud reviewers score individual campaigns, work through a ranked fraud queue, inspect model metrics, and read SHAP explanations. It communicates exclusively via GraphQL.

2. **API Gateway (BFF)** — a Node.js/Apollo GraphQL server that translates GraphQL operations into REST calls to the ML API and gRPC calls to the Feedback and Monitoring microservices. This layer shields the frontend from the internal service topology.

3. **ML Serving API** — a FastAPI service that owns inference. It loads the XGBoost model at startup, lazily loads the LoRA adapter on first ensemble request, runs the ensemble, and returns scores alongside pre-computed SHAP explanations.

4. **Training & Observability Pipeline** — an offline pipeline that ingests raw CSV data through Kafka and a Spark ETL job, trains both models (XGBoost locally via Ray, Gemma-2B on a GCP T4 GPU), evaluates and logs everything to MLflow, and then continuously monitors production inference logs for drift and performance degradation.

All four layers run together in Docker Compose for local development. The API, BFF, and frontend each have their own Cloud Build configuration for GCP deployment.

---

## System Design Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          TRAINING PIPELINE                          │
│                                                                     │
│  CSV Files ──► Kafka ──► Spark ETL ──► Feature Store (Parquet/GCS) │
│                                              │                      │
│                         ┌────────────────────┴──────────────────┐  │
│                         │                                        │  │
│                  XGBoost Trainer                       LoRA Trainer │
│                  (Ray Train, local)               (HuggingFace,    │
│                  Optuna 50-trial HPO               GCP T4 GPU)     │
│                  5-fold stratified CV             200 bal. samples  │
│                         │                                        │  │
│                  xgb_model.json                      LoRA adapter  │
│                         └────────────────┬───────────────────────┘  │
│                                          │                          │
│                             Evaluation Pipeline                     │
│                             SHAP store · AUC · PR-AUC · F1         │
│                                          │                          │
│                                       MLflow                        │
└──────────────────────────────────────────┼──────────────────────────┘
                                           │ models + artifacts
┌──────────────────────────────────────────▼──────────────────────────┐
│                            INFERENCE PATH                           │
│                                                                     │
│  React (5173) ──► Apollo ──► GraphQL BFF (4000)                     │
│                                      │                              │
│                                      │ REST /predict               │
│                              FastAPI ML API (8000)                  │
│                                      │                              │
│                   ┌──────────────────┴──────────────────┐          │
│                   │                                      │          │
│           XGBoost inference                    LoRA inference       │
│           (always loaded)                    (lazy-loaded)          │
│                   └──────────────────┬──────────────────┘          │
│                                      │                              │
│                           Weighted ensemble                         │
│                        0.65 × xgb + 0.35 × lora                    │
│                           SHAP store lookup (O(1))                  │
└──────────────────────────────────────┼──────────────────────────────┘
                                       │ response
┌──────────────────────────────────────▼──────────────────────────────┐
│                      OBSERVABILITY & FEEDBACK LOOP                  │
│                                                                     │
│  Inference logs (JSONL) ──► GCS ──► Evidently drift detection       │
│                                        │                            │
│                             KS test per feature                     │
│                             Alert if p < 0.05                       │
│                                        │                            │
│  Human reviews ──► gRPC Feedback Svc (50051) ──► Kafka labels       │
│                                        │                            │
│                             ≥ 100 new labels → retraining           │
│                                                                     │
│  gRPC Monitoring Svc (50052) ──► rolling AUC, precision@K           │
│                             Alert if AUC drops > 3%                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Components

### React Frontend
**Stack:** React 18, Vite, TypeScript, Apollo Client, Recharts | **Port:** 5173

Four pages cover the full reviewer workflow. *Campaign Scorer* lets a reviewer paste a campaign title and description, choose between XGBoost-only and ensemble mode, and see the fraud score, label, and top-3 SHAP features in real time. *Fraud Queue* lists all campaigns above a configurable threshold (0.7–0.8) ranked by score. *Model Metrics* displays rolling AUC, PR-AUC, and precision-at-K charts from the monitoring service. *Explainability* shows per-campaign feature contributions.

All data fetching goes through Apollo Client against the BFF's GraphQL endpoint — the frontend never calls the ML API directly.

---

### GraphQL BFF
**Stack:** Node.js 18, Express, Apollo Server 4, TypeScript | **Port:** 4000

The BFF is a thin aggregation layer. Its GraphQL schema exposes four top-level operations:

- `scoreCampaign(text, modelMode)` — mutation; POSTs to FastAPI `/predict`
- `fraudQueue(minScore, limit)` — query; returns ranked high-risk campaigns
- `modelMetrics(window)` — query; calls the gRPC Monitoring Service
- `driftReport` — query; returns the latest Evidently drift report summary

The BFF also handles CORS, rate limiting, and request authentication so neither the ML API nor the gRPC services need to deal with those concerns.

---

### FastAPI ML API
**Stack:** FastAPI, Uvicorn, Pydantic v2, grpcio | **Port:** 8000

The ML API owns inference. On startup it loads the XGBoost model and the pre-computed SHAP store (7.1 MB). The LoRA adapter is loaded lazily the first time an ensemble prediction is requested, keeping cold-start latency low for XGBoost-only use cases.

Endpoints:
- `POST /predict` — single prediction
- `POST /predict/batch` — batch of up to 1,000+ campaigns
- `GET /explain/{fund_id}` — SHAP top-3 features for a past prediction
- `GET /health` — model load status, threshold, LoRA-loaded flag

---

### Feedback Service
**Stack:** Python, gRPC, Protobuf, Kafka | **Port:** 50051

Human reviewers submit their verdict (fraud / legitimate, confidence, notes) via the React dashboard. The BFF translates this into a gRPC `SubmitLabel` call. The Feedback Service writes the label to the Kafka topic `fraud-risk.labels`. When at least 100 new labels have accumulated, retraining is triggered automatically.

---

### Monitoring Service
**Stack:** Python, gRPC, Evidently AI, Protobuf | **Port:** 50052

Two gRPC methods: `GetDriftReport()` returns the latest Evidently feature drift analysis, and `GetRollingMetrics(window)` returns AUC, precision, and recall computed over a sliding time window. Alerts fire when feature drift p-values fall below 0.05, or when rolling AUC drops more than 3% from the baseline established at the last training run.

---

### Training Pipeline
**Stack:** PySpark 3.5, Ray Train 2.20, Optuna 3.6, HuggingFace Transformers, PEFT, PyTorch 2.3, MLflow 2.13

The pipeline is orchestrated via `make pipeline`:

1. **Spark ETL** reads raw CSV, strips HTML from descriptions, imputes nulls, encodes categoricals, engineers features, and writes Parquet to GCS.
2. **XGBoost training** runs stratified 5-fold cross-validation via Ray Train, then a final full-data fit. Optuna runs 50 trials to search the hyperparameter space.
3. **LoRA fine-tuning** runs on a GCP Compute Engine instance with a T4 GPU (~$0.54/hr). The PEFT library applies low-rank adapters to the query and value projection layers of Gemma-2B.
4. **Evaluation** computes AUC, PR-AUC, F1, precision@100/500, SHAP values, and bias slices by category and phone type, then logs everything to MLflow.

---

### Infrastructure

| Component | Technology | Port |
|---|---|---|
| Frontend | React 18, Vite, Apollo Client | 5173 |
| GraphQL BFF | Node.js 18, Apollo Server 4 | 4000 |
| ML Serving API | FastAPI, Uvicorn, Pydantic v2 | 8000 |
| Feedback Service | gRPC, Protobuf, Kafka | 50051 |
| Monitoring Service | gRPC, Evidently AI | 50052 |
| Kafka | Confluent 7.6.1 | 9092 |
| Spark | PySpark 3.5.1 | 8080 (UI) |
| MLflow | 2.13.0 | 5001 |

---

## Details

### Data Schema and Feature Engineering

Training data comes from two CSV sources joined on `fund_id`: a fraud metadata table with account-level signals, and a fund text table with campaign titles and descriptions. The merger produces 8 raw features fed directly to XGBoost, plus 4 engineered features:

| Feature | Type | Description |
|---|---|---|
| `category_id` | float | Fundraiser category code |
| `goal` | float | Funding goal in dollars |
| `descr_len` | float | Description character length |
| `title_len` | float | Title character length |
| `primary_phone_checks__line_type` | categorical | mobile / non-fixed voip / unknown |
| `identity_check_score` | float | KYC identity verification score (0–500) |
| `is_disposable_email` | binary | Flag for disposable email addresses |
| `email_domain_creation_days` | float | Age of the email domain in days |
| `log_goal` | engineered | `log1p(goal)` — compresses outlier goals |
| `email_trust_score` | engineered | `domain_age / (is_disposable + 1)` |
| `text_ratio` | engineered | `descr_len / max(title_len, 1)` |
| `identity_norm` | engineered | `identity_check_score / 100` |

Numeric features are median-imputed. The phone line type is ordinally encoded with *unknown* mapped to 0.

---

### XGBoost Model

The XGBoost classifier is trained with these default hyperparameters:

```
n_estimators     = 400
max_depth        = 6
learning_rate    = 0.05
subsample        = 0.8
colsample_bytree = 0.8
eval_metric      = auc
```

Training uses stratified 5-fold cross-validation via Ray Train. After CV, the model is re-fit on the full training set and serialised as `xgb_model.json` (~2 MB). Optuna runs 50 trials to search the hyperparameter space, with 3-fold CV AUC as the objective; the best parameters are logged to MLflow alongside the model artifact.

---

### LoRA Fine-Tuning on Gemma-2B

The text model is Google's Gemma-2B (2 billion parameters). Fine-tuning uses Low-Rank Adaptation (LoRA) via the PEFT library, which keeps base model weights frozen and trains a small set of rank-decomposition matrices injected into the attention layers:

```
lora_r           = 16        # rank of update matrices
lora_alpha       = 32        # scaling factor
lora_dropout     = 0.1
target_modules   = ["q_proj", "v_proj"]
task_type        = SEQUENCE_CLASSIFICATION (2 classes)

Trainable params: 1.8M / 2.5B total  (0.07%)
```

The training set is 200 balanced samples (50% fraud, 50% legitimate). Each sample is formatted as a prompt:

```
Classify the following fundraiser as FRAUD or LEGITIMATE.

Title: {title}
Description: {description}

Answer (FRAUD or LEGITIMATE):
```

Training runs for 10 epochs on a GCP T4 GPU (~$0.54/hr). Final evaluation on a 40-sample held-out set: **AUC = 0.8525, F1 = 0.7805**. The trained adapter weights are uploaded to GCS at `gs://fraud-risk-models/lora-adapters/`.

---

### Ensemble Fusion Strategy

Both models output a fraud probability in [0, 1]. These are combined as a weighted average at inference time (late fusion):

```
fraud_score = (1 - w_lora) × xgb_score + w_lora × lora_score
```

The weight `w_lora` is optimised on the validation set using `scipy.optimize.minimize_scalar` to maximise AUC, landing at roughly 0.35 (35% LoRA, 65% XGBoost). The classification threshold is set at the Youden-J index (maximises sensitivity + specificity − 1), defaulting to 0.5. Both the weight and threshold are configurable via environment variables at deploy time.

Clients can also request XGBoost-only inference by passing `model_mode: "xgb"`, which skips the LoRA adapter entirely and roughly halves latency — useful for high-throughput batch screening.

---

### SHAP Explainability

SHAP (SHapley Additive exPlanations) values are computed once during the evaluation pipeline using XGBoost's `TreeExplainer` on the full test set (~25K samples). The results are stored in a 7.1 MB JSON file (`shap_store.json`) indexed by `fund_id`, containing the top-3 features by absolute SHAP magnitude with their direction (*fraud* or *safe*):

```json
{
  "fund_id": 12345,
  "top_features": [
    { "name": "goal",              "shap_value": 0.18,  "direction": "fraud" },
    { "name": "email_trust_score", "shap_value": 0.12,  "direction": "fraud" },
    { "name": "identity_norm",     "shap_value": -0.09, "direction": "safe"  }
  ]
}
```

At inference time, the API does an O(1) dictionary lookup — no per-request SHAP computation. This keeps explanation latency negligible even at high throughput.

---

### Feedback Loop and Continuous Retraining

The platform is designed to improve over time without manual intervention. When a reviewer submits a verdict in the dashboard, the BFF calls the gRPC Feedback Service (`SubmitLabel`), which writes the label to the Kafka topic `fraud-risk.labels`. A consumer counts new labels; when the count reaches 100, it triggers a retraining run that ingests the new ground truth, re-runs the full training pipeline, evaluates, and updates the model artifacts in GCS.

The Feedback Service also exposes `ListPendingLabels()` as a server-streaming gRPC method, which the retraining orchestrator uses to pull the latest labelled examples.

---

### Drift Detection and Alerting

Production inference requests are logged as JSONL to GCS. The Monitoring Service periodically loads recent logs and runs Evidently's `DataDriftPreset`, which applies a Kolmogorov-Smirnov test to each feature's production distribution versus its training-time distribution.

Two alert conditions trigger notifications:

- Any feature's KS test p-value drops below 0.05
- Rolling 7-day AUC (computed from inference logs paired with feedback labels) drops more than 3% below the baseline AUC recorded at the last training run

HTML drift reports are saved to `gs://fraud-risk-logs/drift-reports/` for manual inspection. The BFF surfaces a summary via the `driftReport` GraphQL query so reviewers can see at a glance whether the model is operating within expected bounds.

---

## References and Source Code

Full source code, Dockerfiles, Cloud Build configs, training notebooks, and Makefile:
[github.com/PrachiRahurkar/fraud-risk-ml-platform](https://github.com/PrachiRahurkar/fraud-risk-ml-platform)

Key files for further reading:

- `ml_model_packaging_service/api/predictor.py` — EnsemblePredictor class and ensemble fusion logic
- `ml_model_packaging_service/api/schemas.py` — Pydantic schemas for FundFeatures and PredictionResponse
- `ml_model_packaging_service/bff/src/schema/typeDefs.ts` — full GraphQL schema
- `ml_training_service/training/xgb_trainer.py` — XGBoost 5-fold CV training and Optuna HPO
- `ml_training_service/training/lora_trainer.py` — HuggingFace LoRA fine-tuning
- `ml_evaluation_pipeline/explainability.py` — SHAP store computation
- `ml_observability_monitoring/drift.py` — Evidently drift detection
- `ml_infra_deployment_ops/docker-compose.yml` — full local development stack
- `ml_infra_deployment_ops/colab/lora_training.ipynb` — interactive Colab training walkthrough
