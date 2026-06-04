#!/usr/bin/env bash
# Create a GCP Compute Engine GPU VM for LoRA training.
# Cost: ~$0.54/hr (n1-standard-4 + T4). ALWAYS stop when done.
# Usage: bash ml_infra_deployment_ops/gcp/setup_vm.sh

# Updated for G2/L4 Compatibility
set -euo pipefail

PROJECT="${GOOGLE_CLOUD_PROJECT:-$(gcloud config get-value project)}"
ZONE="${GCP_ZONE:-us-west1-a}"
VM_NAME="fraud-lora-trainer"

echo "Creating GPU VM: $VM_NAME in $ZONE"
gcloud compute instances create "$VM_NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --machine-type=g2-standard-4 \
  --accelerator=type=nvidia-l4,count=1 \
  --image-family=pytorch-2-9-cu129-ubuntu-2404-nvidia-580 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE \
  --restart-on-failure \
  --metadata="install-nvidia-driver=True" \
  --scopes=storage-rw,logging-write




echo ""
echo "VM created. SSH with:"
echo "  gcloud compute ssh $VM_NAME --zone=$ZONE"
echo ""
echo "IMPORTANT: Stop the VM when training is done:"
echo "  gcloud compute instances stop $VM_NAME --zone=$ZONE"
