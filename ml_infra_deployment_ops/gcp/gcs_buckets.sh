#!/usr/bin/env bash
# Create GCS buckets for the fraud-risk platform.
# Usage: bash ml_infra_deployment_ops/gcp/gcs_buckets.sh
# Prereq: gcloud auth login && gcloud config set project YOUR_PROJECT_ID

set -euo pipefail

PROJECT="${GOOGLE_CLOUD_PROJECT:-$(gcloud config get-value project)}"
REGION="${GCS_REGION:-us-central1}"

BUCKETS=(
  "fraud-risk-data"      # raw CSVs, processed Parquet, feature store
  "fraud-risk-models"    # LoRA adapters, XGB models, SHAP store
  "fraud-risk-logs"      # inference logs, drift reports, feedback labels
)

for BUCKET in "${BUCKETS[@]}"; do
  echo "Creating bucket: gs://$BUCKET"
  gcloud storage buckets create "gs://$BUCKET" \
    --project="$PROJECT" \
    --location="$REGION" \
    --uniform-bucket-level-access \
    2>/dev/null || echo "  (already exists, skipping)"
done

echo ""
echo "Uploading raw data to gs://fraud-risk-data/raw/"
gsutil -m cp -r data/train_data gs://fraud-risk-data/raw/
gsutil -m cp -r data/test_data  gs://fraud-risk-data/raw/

echo "Done."
