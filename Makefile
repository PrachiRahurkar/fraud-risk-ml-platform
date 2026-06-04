## ─────────────────────────────────────────────────────────────────────────────
##  Fraud Risk ML Platform — Makefile
##  All Python targets require: conda activate fraud-risk
## ─────────────────────────────────────────────────────────────────────────────

CONDA_ENV   := fraud-risk
PYTHON      := conda run -n $(CONDA_ENV) python
PIP         := conda run -n $(CONDA_ENV) pip
MLFLOW_URI  := file:$(CURDIR)/mlruns
DATA_ROOT   := data
MODELS_DIR  := models

.PHONY: help env-create env-install infra-up infra-down kafka-init \
        spark-etl stream-data train-xgb tune evaluate serve dev \
        colab-lora build pipeline test clean

help:
	@echo ""
	@echo "  Fraud Risk ML Platform"
	@echo "  ──────────────────────────────────────────────────────"
	@echo "  Setup:"
	@echo "    make env-create      Create conda environment (fraud-risk)"
	@echo "    make env-install     Install all Python requirements"
	@echo ""
	@echo "  Infrastructure:"
	@echo "    make infra-up        Start Kafka, Spark, MLflow (Docker)"
	@echo "    make infra-down      Stop all infrastructure"
	@echo "    make kafka-init      Create Kafka topics"
	@echo ""
	@echo "  Data & Features:"
	@echo "    make spark-etl       Run Spark batch ETL (raw → Parquet)"
	@echo "    make stream-data     Start Kafka producer (simulate events)"
	@echo ""
	@echo "  Training:"
	@echo "    make train-xgb       Train XGBoost with Ray (local)"
	@echo "    make tune            Run Optuna hyperparameter search"
	@echo "    make colab-lora      Open LoRA fine-tuning notebook in Colab"
	@echo ""
	@echo "  Evaluation:"
	@echo "    make evaluate        Full eval: metrics + SHAP + bias"
	@echo ""
	@echo "  Serving:"
	@echo "    make serve           Start FastAPI ML API (localhost:8000)"
	@echo "    make dev             Start React frontend (localhost:5173)"
	@echo ""
	@echo "  Tests:"
	@echo "    make test            Run pytest"
	@echo ""
	@echo "  Full pipeline:"
	@echo "    make pipeline        ingest → features → train → eval"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────

env-create:
	conda create -n $(CONDA_ENV) python=3.11 -y
	@echo "Run: conda activate $(CONDA_ENV)"

env-install:
	$(PIP) install \
		-r ml_training_service/requirements.txt \
		-r ml_evaluation_pipeline/requirements.txt \
		-r ml_model_packaging_service/requirements.txt \
		-r ml_observability_monitoring/requirements.txt
	$(PIP) install -e .

# ── Infrastructure ────────────────────────────────────────────────────────────

infra-up:
	docker compose -f ml_infra_deployment_ops/docker-compose.yml up -d \
		zookeeper kafka spark-master spark-worker mlflow
	@echo "MLflow UI: http://localhost:5000"
	@echo "Spark UI:  http://localhost:8080"
	@echo "Kafka:     localhost:9092"

infra-down:
	docker compose -f ml_infra_deployment_ops/docker-compose.yml down

kafka-init:
	bash ml_infra_deployment_ops/kafka/topics.sh

# ── Data & Features ───────────────────────────────────────────────────────────

spark-etl:
	$(PYTHON) ml_training_service/data/spark_pipeline.py \
		--fraud-path $(DATA_ROOT)/train_data/fraud_data_train.csv \
		--fund-path  $(DATA_ROOT)/train_data/fund_data_train.csv \
		--output-path $(DATA_ROOT)/feature_store \
		--split train
	$(PYTHON) ml_training_service/data/spark_pipeline.py \
		--fraud-path $(DATA_ROOT)/test_data/fraud_data_test.csv \
		--fund-path  $(DATA_ROOT)/test_data/fund_data_test.csv \
		--output-path $(DATA_ROOT)/feature_store \
		--split test

stream-data:
	$(PYTHON) ml_training_service/data/kafka_producer.py \
		--fraud-path $(DATA_ROOT)/train_data/fraud_data_train.csv \
		--fund-path  $(DATA_ROOT)/train_data/fund_data_train.csv \
		--delay-ms 100

# ── Training ──────────────────────────────────────────────────────────────────

train-xgb:
	mkdir -p $(MODELS_DIR)
	$(PYTHON) -m ml_training_service.training.xgb_trainer \
		--train-path $(DATA_ROOT)/train_data/fraud_data_train.csv \
		--output-path $(MODELS_DIR)/xgb_model.json \
		--mlflow-uri $(MLFLOW_URI)

tune:
	$(PYTHON) -m ml_training_service.tuning.hyperopt \
		--train-path $(DATA_ROOT)/train_data/fraud_data_train.csv \
		--n-trials 50 \
		--mlflow-uri $(MLFLOW_URI)

colab-lora:
	@echo ""
	@echo "  LoRA fine-tuning runs in Google Colab (no GPU quota needed)."
	@echo ""
	@echo "  1. Open the notebook:"
	@echo "     ml_infra_deployment_ops/colab/lora_training.ipynb"
	@echo "     → Upload to https://colab.research.google.com or open via Drive"
	@echo ""
	@echo "  2. Runtime → Change runtime type → T4 GPU"
	@echo ""
	@echo "  3. Run all cells. Adapter is saved to:"
	@echo "     gs://fraud-risk-models/lora-adapters/v1/final-adapter"
	@echo ""
	@echo "  4. Download adapter locally:"
	@echo "     gsutil -m cp -r gs://fraud-risk-models/lora-adapters/v1/final-adapter models/lora-adapter/"
	@echo ""

# ── Evaluation ────────────────────────────────────────────────────────────────

evaluate:
	mkdir -p $(MODELS_DIR)
	$(PYTHON) -m ml_evaluation_pipeline.pipeline \
		--test-fraud-path $(DATA_ROOT)/test_data/fraud_data_test.csv \
		--test-fund-path  $(DATA_ROOT)/test_data/fund_data_test.csv \
		--xgb-model-path  $(MODELS_DIR)/xgb_model.json \
		--shap-store-path $(MODELS_DIR)/shap_store.json \
		--mlflow-uri $(MLFLOW_URI)

# ── Serving ───────────────────────────────────────────────────────────────────

serve:
	$(PYTHON) -m uvicorn ml_model_packaging_service.api.main:app \
		--host 0.0.0.0 --port 8000 --reload

dev:
	cd ml_model_packaging_service/frontend && npm run dev

# ── Build ─────────────────────────────────────────────────────────────────────

build:
	docker compose -f ml_infra_deployment_ops/docker-compose.yml build

# ── Pipeline ──────────────────────────────────────────────────────────────────

pipeline:
	$(PYTHON) -m ml_training_service.pipelines.training_pipeline \
		--data-root $(DATA_ROOT) \
		--feature-store-path $(DATA_ROOT)/feature_store \
		--model-output-dir $(MODELS_DIR) \
		--mlflow-uri $(MLFLOW_URI) \
		--skip-lora
	$(MAKE) evaluate

# ── Tests ─────────────────────────────────────────────────────────────────────

test:
	$(PYTHON) -m pytest tests/ -v

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ | xargs rm -rf
	find . -name "*.pyc" -delete
	rm -rf $(MODELS_DIR)/*.json $(MODELS_DIR)/*.csv
	@echo "Cleaned build artifacts."
