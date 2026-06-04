#!/usr/bin/env bash
# SSH into the GCP GPU VM, run LoRA training, push adapter to GCS, then stop VM.
# Usage: bash ml_infra_deployment_ops/gcp/train_on_vm.sh
# Prereq: VM must be running (make vm-create)

set -euo pipefail

PROJECT="${GOOGLE_CLOUD_PROJECT:-$(gcloud config get-value project)}"
ZONE="${GCP_ZONE:-us-central1-a}"
VM_NAME="fraud-lora-trainer"
GCS_ADAPTER_PATH="gs://fraud-risk-models/lora-adapters/v1"

echo "Starting VM if stopped..."
gcloud compute instances start "$VM_NAME" --zone="$ZONE" --quiet

echo "Waiting for SSH to become available..."
sleep 30

echo "Uploading training data and code..."
gcloud compute scp \
  --recurse data/train_data \
  "$VM_NAME:/home/$(whoami)/fraud-risk/data/" \
  --zone="$ZONE"

gcloud compute scp \
  --recurse ml_training_service \
  "$VM_NAME:/home/$(whoami)/fraud-risk/" \
  --zone="$ZONE"

echo "Running LoRA training on VM..."
gcloud compute ssh "$VM_NAME" --zone="$ZONE" -- bash -s <<'REMOTE'
  cd /home/$USER/fraud-risk
  pip install peft transformers accelerate datasets mlflow google-cloud-storage -q
  conda activate fraud-risk 2>/dev/null || true
  python -m ml_training_service.training.lora_trainer \
    --mlflow-uri http://localhost:5000 || \
  python ml_training_service/training/lora_trainer.py
REMOTE

echo "Pushing LoRA adapter to GCS..."
gcloud compute ssh "$VM_NAME" --zone="$ZONE" -- \
  "gsutil -m cp -r /home/\$USER/fraud-risk/models/lora-adapter/final-adapter $GCS_ADAPTER_PATH"

echo "Stopping VM to avoid idle charges..."
gcloud compute instances stop "$VM_NAME" --zone="$ZONE" --quiet

echo "Done. Adapter saved to $GCS_ADAPTER_PATH"
echo "Download locally with: gsutil -m cp -r $GCS_ADAPTER_PATH models/lora-adapter"
