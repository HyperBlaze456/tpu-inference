#!/bin/bash
# =============================================================================
# Download a HuggingFace model to GCS via TPU VM
#
# Downloads the model on worker 0 with gcsfuse-mounted GCS bucket,
# so the weights land directly in GCS and are available to all workers.
#
# Usage:
#   # Download GLM-5 bf16 (default):
#   bash download_model.sh
#
#   # Download a different model:
#   bash download_model.sh --model "meta-llama/Llama-4-Maverick-17B-128E-Instruct"
#
#   # With HF token (for gated models):
#   bash download_model.sh --model "meta-llama/..." --token "hf_xxx"
#
#   # Custom local name in GCS:
#   bash download_model.sh --model "zai-org/GLM-5" --name "GLM-5-bf16"
# =============================================================================
set -euo pipefail

# --- Configuration -----------------------------------------------------------
: "${TPU_PROJECT:=prm-research}"
: "${TPU_VM_NAME:=v6-hyperblaze}"
: "${TPU_ZONE:=europe-west4-a}"
: "${SSH_USER:=root}"

GCS_MOUNT="/mnt/gcs/hf"

MODEL_ID="zai-org/GLM-5"
MODEL_NAME=""
HF_TOKEN=""
WORKER=0

# --- Parse arguments ---------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)    MODEL_ID="$2"; shift 2 ;;
        --name)     MODEL_NAME="$2"; shift 2 ;;
        --token)    HF_TOKEN="$2"; shift 2 ;;
        --worker)   WORKER="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,/^# =====/p' "$0" | head -n -1 | sed 's/^# \?//'
            exit 0
            ;;
        *)  echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Derive local directory name from model ID if not specified
if [[ -z "$MODEL_NAME" ]]; then
    MODEL_NAME="${MODEL_ID##*/}"
fi

DEST="${GCS_MOUNT}/${MODEL_NAME}"

log() { echo "[download] $(date '+%H:%M:%S') $1"; }

ssh_cmd() {
    gcloud compute tpus tpu-vm ssh \
        "${SSH_USER}@${TPU_VM_NAME}" \
        --project="${TPU_PROJECT}" \
        --zone="${TPU_ZONE}" \
        --tunnel-through-iap \
        --worker="${WORKER}" \
        --command="$1" \
        -- -o StrictHostKeyChecking=no -o ServerAliveInterval=60 \
           -o ServerAliveCountMax=120
}

log "========================================="
log "Model Download to GCS"
log "========================================="
log "Model:       ${MODEL_ID}"
log "Destination: ${DEST}"
log "Worker:      ${WORKER}"
log "========================================="

# --- Ensure pip packages -----------------------------------------------------
log "Installing huggingface_hub on worker ${WORKER}..."
ssh_cmd "pip install --quiet --upgrade huggingface_hub 2>/dev/null || \
         pip3 install --quiet --upgrade huggingface_hub"

# --- Download ----------------------------------------------------------------
log "Starting download of ${MODEL_ID} -> ${DEST}"
log "This is a ~1.5TB model (bf16 744B MoE). Will take a while..."

TOKEN_ARG=""
if [[ -n "$HF_TOKEN" ]]; then
    TOKEN_ARG="--token '${HF_TOKEN}'"
fi

# Use huggingface-cli download with local-dir pointing at GCS mount.
# This streams files directly to GCS via gcsfuse.
ssh_cmd "
mkdir -p '${DEST}'
huggingface-cli download '${MODEL_ID}' \
    --local-dir '${DEST}' \
    --local-dir-use-symlinks False \
    ${TOKEN_ARG} \
    2>&1 | tail -5
echo \"\"
echo \"Download complete. Contents:\"
ls -lh '${DEST}' | head -30
du -sh '${DEST}' 2>/dev/null || true
"

log "========================================="
log "Done! Model saved to: ${DEST}"
log "========================================="
log ""
log "To serve with vLLM:"
log "  docker exec -it ray-head bash -lc \"vllm serve '${DEST}' \\"
log "      --tensor-parallel-size 16 \\"
log "      --max-model-len 32768 \\"
log "      --disable-frontend-multiprocessing \\"
log "      --no-async-scheduling\""
