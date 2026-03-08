#!/bin/bash
# =============================================================================
# TPU VM Multihost Setup (v6e-64, 16 workers)
#
# Called by watchdog.sh when the TPU VM transitions to READY.
# Runs from the LOCAL machine, orchestrating all 16 host VMs via SSH.
#
# Steps:
#   1. Clone/pull repo on all workers
#   2. Install gcsfuse + mount GCS bucket on all workers
#   3. Build Docker image on all workers
#   4. Discover internal IPs via describe
#   5. Start Ray head on worker 0
#   6. Start Ray workers on workers 1-15
#   7. Verify cluster
#
# Environment variables (set by watchdog):
#   TPU_PROJECT, TPU_VM_NAME, TPU_ZONE, SSH_USER, ACCELERATOR_TYPE
# =============================================================================
set -euo pipefail

# --- Configuration -----------------------------------------------------------
: "${TPU_PROJECT:=prm-research}"
: "${TPU_VM_NAME:=v6-hyperblaze}"
: "${TPU_ZONE:=europe-west4-a}"
: "${SSH_USER:=root}"

REPO_URL="https://github.com/HyperBlaze456/tpu-inference.git"
REPO_BRANCH="main"
REPO_DIR="/root/tpu-inference"

GCS_BUCKET="hyperblaze-bucket-eu"
GCS_MOUNT="/mnt/gcs/hf"
GCSFUSE_VERSION="3.5.5"

DOCKER_IMAGE="tpu-inference:latest"
RAY_PORT=6379
SHM_SIZE="32G"

# HF token: set via env var or fetch from GCP secrets
HF_TOKEN="${HF_TOKEN:-}"

NUM_WORKERS=16  # v6e-64 = 16 host VMs

log() { echo "[on_ready] $(date '+%H:%M:%S') $1"; }

# --- SSH helpers -------------------------------------------------------------

ssh_cmd() {
    local worker="$1"
    local cmd="$2"
    gcloud compute tpus tpu-vm ssh \
        "${SSH_USER}@${TPU_VM_NAME}" \
        --project="${TPU_PROJECT}" \
        --zone="${TPU_ZONE}" \
        --tunnel-through-iap \
        --worker="${worker}" \
        --command="${cmd}" \
        -- -o StrictHostKeyChecking=no -o ServerAliveInterval=30 \
           -o ServerAliveCountMax=60
}

ssh_cmd_all() {
    local cmd="$1"
    gcloud compute tpus tpu-vm ssh \
        "${SSH_USER}@${TPU_VM_NAME}" \
        --project="${TPU_PROJECT}" \
        --zone="${TPU_ZONE}" \
        --tunnel-through-iap \
        --worker=all \
        --command="${cmd}" \
        -- -o StrictHostKeyChecking=no -o ServerAliveInterval=30 \
           -o ServerAliveCountMax=60
}

# --- Step 1: Clone / update repo on all workers -----------------------------

log "[1/7] Cloning/updating repo on all ${NUM_WORKERS} workers..."
ssh_cmd_all "
if [ -d '${REPO_DIR}/.git' ]; then
    cd '${REPO_DIR}'
    git fetch --all --quiet
    git checkout '${REPO_BRANCH}' --quiet
    git reset --hard 'origin/${REPO_BRANCH}' --quiet
else
    git clone --branch '${REPO_BRANCH}' '${REPO_URL}' '${REPO_DIR}'
fi
"
log "  Repo synced."

# --- Step 2: Install gcsfuse + mount GCS bucket on all workers ---------------

log "[2/7] Installing gcsfuse and mounting GCS bucket on all workers..."
ssh_cmd_all "
# Install gcsfuse if not present
if ! command -v gcsfuse &>/dev/null; then
    curl -sL -O https://github.com/GoogleCloudPlatform/gcsfuse/releases/download/v${GCSFUSE_VERSION}/gcsfuse_${GCSFUSE_VERSION}_amd64.deb
    dpkg -i gcsfuse_${GCSFUSE_VERSION}_amd64.deb || apt-get install -f -y
    rm -f gcsfuse_${GCSFUSE_VERSION}_amd64.deb
fi

# Mount if not already mounted
if ! mountpoint -q '${GCS_MOUNT}' 2>/dev/null; then
    mkdir -p '${GCS_MOUNT}'
    gcsfuse --implicit-dirs --file-mode=777 --dir-mode=777 \
        '${GCS_BUCKET}' '${GCS_MOUNT}'
fi
echo \"GCS mounted at ${GCS_MOUNT}\"
"
log "  GCS bucket mounted."

# --- Step 3: Build Docker image on all workers ------------------------------

log "[3/7] Building Docker image on all ${NUM_WORKERS} workers (this takes a while)..."
ssh_cmd_all "
cd '${REPO_DIR}'
DOCKER_BUILDKIT=1 docker build \
    -t '${DOCKER_IMAGE}' \
    -f docker/Dockerfile \
    .
"
log "  Docker image built on all workers."

# --- Step 4: Discover internal IPs via describe ------------------------------

log "[4/7] Discovering internal IPs..."
ALL_IPS=$(gcloud compute tpus tpu-vm describe "${TPU_VM_NAME}" \
    --project="${TPU_PROJECT}" \
    --zone="${TPU_ZONE}" \
    --format="value(networkEndpoints[].ipAddress)")

# Normalize separators (gcloud may use ; or ,)
ALL_IPS="${ALL_IPS//;/ }"
ALL_IPS="${ALL_IPS//,/ }"
# shellcheck disable=SC2206
IP_ARRAY=($ALL_IPS)

HEAD_IP="${IP_ARRAY[0]}"
WORKER_COUNT=${#IP_ARRAY[@]}

log "  Head IP (worker 0): ${HEAD_IP}"
log "  Total workers: ${WORKER_COUNT}"
for i in "${!IP_ARRAY[@]}"; do
    log "    worker ${i}: ${IP_ARRAY[$i]}"
done

# --- Step 5: Clean up old containers on all workers -------------------------

log "[5/7] Cleaning up old containers..."
ssh_cmd_all "
docker rm -f ray-head ray-worker 2>/dev/null || true
"

# --- Step 6: Start Ray head on worker 0 -------------------------------------

log "[6/7] Starting Ray head on worker 0 (${HEAD_IP})..."
ssh_cmd 0 "
docker run -d --name ray-head \
    --privileged \
    --net=host \
    --shm-size=${SHM_SIZE} \
    -e TPU_MULTIHOST_BACKEND=ray \
    -e JAX_PLATFORMS='' \
    -e HF_TOKEN='${HF_TOKEN}' \
    -v '${GCS_MOUNT}:${GCS_MOUNT}' \
    -v /root/.cache/huggingface:/root/.cache/huggingface \
    '${DOCKER_IMAGE}' \
    bash -lc 'ray start --head --port=${RAY_PORT} --block'
"
log "  Ray head started."

# Wait for Ray head to be ready
log "  Waiting for Ray head to initialize..."
sleep 30

# --- Step 7: Start Ray workers on workers 1-15 ------------------------------

log "[7/7] Starting Ray workers on workers 1-$((NUM_WORKERS - 1))..."
WORKER_RANGE="1-$((NUM_WORKERS - 1))"

gcloud compute tpus tpu-vm ssh \
    "${SSH_USER}@${TPU_VM_NAME}" \
    --project="${TPU_PROJECT}" \
    --zone="${TPU_ZONE}" \
    --tunnel-through-iap \
    --worker="${WORKER_RANGE}" \
    --command="
docker run -d --name ray-worker \
    --privileged \
    --net=host \
    --shm-size=${SHM_SIZE} \
    -e TPU_MULTIHOST_BACKEND=ray \
    -e JAX_PLATFORMS='' \
    -e HF_TOKEN='${HF_TOKEN}' \
    -v '${GCS_MOUNT}:${GCS_MOUNT}' \
    -v /root/.cache/huggingface:/root/.cache/huggingface \
    '${DOCKER_IMAGE}' \
    bash -lc 'ray start --address=${HEAD_IP}:${RAY_PORT} --block'
" \
    -- -o StrictHostKeyChecking=no -o ServerAliveInterval=30

log "  Ray workers started."

# --- Verify cluster ---------------------------------------------------------

log "Waiting 60s for workers to join the cluster..."
sleep 60

log "Verifying Ray cluster..."
CLUSTER_STATUS=$(ssh_cmd 0 "
docker exec ray-head bash -lc 'python3 -c \"
import ray
ray.init(address=\\\"auto\\\")
nodes = ray.nodes()
alive = [n for n in nodes if n[\\\"Alive\\\"]]
print(f\\\"Cluster: {len(alive)} nodes alive out of {len(nodes)}\\\")
\"'
" 2>&1) || true

log "  ${CLUSTER_STATUS}"

# --- Done --------------------------------------------------------------------

log "========================================="
log "Setup complete!"
log "========================================="
log ""
log "To serve a model:"
log "  gcloud compute tpus tpu-vm ssh '${SSH_USER}@${TPU_VM_NAME}' \\"
log "      --project='${TPU_PROJECT}' --zone='${TPU_ZONE}' --worker=0"
log ""
log "  docker exec -it ray-head bash -lc \"vllm serve 'MODEL_NAME' \\"
log "      --tensor-parallel-size ${NUM_WORKERS} \\"
log "      --max-model-len 32768 \\"
log "      --max-num-batched-tokens 32768 \\"
log "      --disable-frontend-multiprocessing \\"
log "      --no-async-scheduling\""
