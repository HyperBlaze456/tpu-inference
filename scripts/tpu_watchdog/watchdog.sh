#!/bin/bash
# =============================================================================
# TPU Spot VM Watchdog
#
# Monitors a spot TPU VM, auto-recreates on preemption, and runs a setup
# script when the VM becomes READY.
#
# Usage:
#   # Start watchdog (foreground):
#   bash watchdog.sh
#
#   # Start watchdog (background with logging):
#   bash watchdog.sh --daemon
#
#   # Custom poll interval (default 30s):
#   bash watchdog.sh --interval 60
#
#   # Skip auto-recreate (monitor only):
#   bash watchdog.sh --no-recreate
#
#   # Custom setup script:
#   bash watchdog.sh --setup ./my_setup.sh
#
#   # Dry run (log actions without executing):
#   bash watchdog.sh --dry-run
#
#   # Stop the daemon:
#   bash watchdog.sh --stop
# =============================================================================
set -uo pipefail

# --- Configuration -----------------------------------------------------------
TPU_PROJECT="prm-research"
TPU_VM_NAME="v6-hyperblaze"
TPU_ZONE="europe-west4-a"
ACCELERATOR_TYPE="v6e-64"
RUNTIME_VERSION="v2-alpha-tpuv6e"
SSH_USER="root"
QR_ID="${TPU_VM_NAME}-qr"     # queued-resource ID

POLL_INTERVAL=30
MAX_READY_WAIT=600
SSH_RETRY_INTERVAL=15
SETUP_TIMEOUT=3600      # 1 hour (Docker build on 16 hosts takes time)

# --- Paths -------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETUP_SCRIPT="${SCRIPT_DIR}/on_ready.sh"
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/watchdog.log"
PID_FILE="${SCRIPT_DIR}/.watchdog.pid"

# --- State tracking ----------------------------------------------------------
LAST_STATE=""
SETUP_RAN_FOR_CURRENT_LIFECYCLE=false
RECREATE_COUNT=0

# --- Parse arguments ---------------------------------------------------------
DAEMON=false
DRY_RUN=false
AUTO_RECREATE=true
STOP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --daemon)       DAEMON=true; shift ;;
        --dry-run)      DRY_RUN=true; shift ;;
        --no-recreate)  AUTO_RECREATE=false; shift ;;
        --stop)         STOP=true; shift ;;
        --interval)     POLL_INTERVAL="$2"; shift 2 ;;
        --setup)        SETUP_SCRIPT="$2"; shift 2 ;;
        --project)      TPU_PROJECT="$2"; shift 2 ;;
        --name)         TPU_VM_NAME="$2"; shift 2 ;;
        --zone)         TPU_ZONE="$2"; shift 2 ;;
        --accel)        ACCELERATOR_TYPE="$2"; shift 2 ;;
        --runtime)      RUNTIME_VERSION="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,/^# =====/p' "$0" | head -n -1 | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1 (use --help)"
            exit 1
            ;;
    esac
done

# --- Helpers -----------------------------------------------------------------

log() {
    local timestamp
    timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    local msg="[${timestamp}] $1"
    echo "$msg"
    [[ -d "$LOG_DIR" ]] && echo "$msg" >> "$LOG_FILE"
}

log_error() { log "ERROR: $1"; }
log_warn()  { log "WARN:  $1"; }
log_ok()    { log "OK:    $1"; }

notify() {
    # Hook point for notifications (Slack, Telegram, etc.)
    # Override in a separate config file to add notification logic.
    log "NOTIFY: $1"
}

# --- Stop daemon -------------------------------------------------------------

if [[ "$STOP" == true ]]; then
    if [[ -f "$PID_FILE" ]]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID"
            rm -f "$PID_FILE"
            echo "Watchdog (PID $PID) stopped."
        else
            rm -f "$PID_FILE"
            echo "Watchdog was not running (stale PID file cleaned)."
        fi
    else
        echo "No watchdog PID file found."
    fi
    exit 0
fi

# --- Daemon mode -------------------------------------------------------------

if [[ "$DAEMON" == true ]]; then
    mkdir -p "$LOG_DIR"
    log "Starting watchdog in daemon mode..."
    nohup bash "$0" \
        --interval "$POLL_INTERVAL" \
        --setup "$SETUP_SCRIPT" \
        --project "$TPU_PROJECT" \
        --name "$TPU_VM_NAME" \
        --zone "$TPU_ZONE" \
        --accel "$ACCELERATOR_TYPE" \
        --runtime "$RUNTIME_VERSION" \
        $( [[ "$AUTO_RECREATE" == false ]] && echo "--no-recreate" ) \
        $( [[ "$DRY_RUN" == true ]] && echo "--dry-run" ) \
        >> "$LOG_FILE" 2>&1 &
    DAEMON_PID=$!
    echo "$DAEMON_PID" > "$PID_FILE"
    log "Watchdog started as PID $DAEMON_PID"
    log "Log: $LOG_FILE"
    log "Stop: bash $0 --stop"
    exit 0
fi

# --- Core functions ----------------------------------------------------------

get_tpu_state() {
    local output
    output=$(gcloud compute tpus tpu-vm describe "$TPU_VM_NAME" \
        --project="$TPU_PROJECT" \
        --zone="$TPU_ZONE" \
        --format="value(state)" 2>&1)
    local rc=$?

    if [[ $rc -ne 0 ]]; then
        if echo "$output" | grep -qi "NOT_FOUND\|not found\|could not be found"; then
            echo "NOT_FOUND"
        else
            echo "ERROR"
        fi
        return
    fi
    echo "$output"
}

wait_for_ssh() {
    local elapsed=0
    log "Waiting for SSH on worker 0..."
    while [[ $elapsed -lt $MAX_READY_WAIT ]]; do
        if gcloud compute tpus tpu-vm ssh \
            "${SSH_USER}@${TPU_VM_NAME}" \
            --project="$TPU_PROJECT" \
            --zone="$TPU_ZONE" \
            --tunnel-through-iap \
            --worker=0 \
            --command="echo 'ssh_ok'" \
            -- -o ConnectTimeout=10 -o StrictHostKeyChecking=no \
            2>/dev/null | grep -q "ssh_ok"; then
            log_ok "SSH is ready on worker 0."
            return 0
        fi
        sleep "$SSH_RETRY_INTERVAL"
        elapsed=$((elapsed + SSH_RETRY_INTERVAL))
        log "SSH not ready yet (${elapsed}s / ${MAX_READY_WAIT}s)..."
    done
    log_error "SSH did not become available within ${MAX_READY_WAIT}s."
    return 1
}

run_setup() {
    if [[ ! -f "$SETUP_SCRIPT" ]]; then
        log_warn "Setup script not found: $SETUP_SCRIPT"
        return 1
    fi

    log "Running setup script: $SETUP_SCRIPT"
    notify "TPU VM '$TPU_VM_NAME' is READY (${ACCELERATOR_TYPE}). Running setup..."

    if [[ "$DRY_RUN" == true ]]; then
        log "[DRY RUN] Would execute: bash $SETUP_SCRIPT"
        return 0
    fi

    export TPU_PROJECT TPU_VM_NAME TPU_ZONE SSH_USER ACCELERATOR_TYPE
    if timeout "$SETUP_TIMEOUT" bash "$SETUP_SCRIPT"; then
        log_ok "Setup completed successfully."
        notify "Setup completed on '$TPU_VM_NAME' (${ACCELERATOR_TYPE})."
        return 0
    else
        local rc=$?
        log_error "Setup failed with exit code $rc."
        notify "Setup FAILED on '$TPU_VM_NAME' (exit $rc)."
        return $rc
    fi
}

recreate_tpu() {
    RECREATE_COUNT=$((RECREATE_COUNT + 1))
    log "Recreating spot TPU VM via queued-resources (attempt #${RECREATE_COUNT})..."
    notify "Queueing spot TPU VM '$TPU_VM_NAME' (${ACCELERATOR_TYPE}, attempt #${RECREATE_COUNT})..."

    if [[ "$DRY_RUN" == true ]]; then
        log "[DRY RUN] Would delete queued-resource + create new one"
        return 0
    fi

    # Clean up old queued-resource (if any) and old VM
    log "Cleaning up old queued-resource and VM..."
    gcloud alpha compute tpus queued-resources delete "${QR_ID}" \
        --project="$TPU_PROJECT" \
        --zone="$TPU_ZONE" \
        --quiet --force 2>/dev/null || true

    local current_state
    current_state=$(get_tpu_state)
    if [[ "$current_state" != "NOT_FOUND" ]]; then
        log "Deleting old VM (state: $current_state)..."
        gcloud compute tpus tpu-vm delete "$TPU_VM_NAME" \
            --project="$TPU_PROJECT" \
            --zone="$TPU_ZONE" \
            --quiet 2>&1 || true
    fi

    sleep 10

    # Submit queued-resource request. GCP handles retry/queueing internally
    # until capacity is available - no manual backoff needed.
    log "Submitting queued-resource request: ${QR_ID}..."
    gcloud alpha compute tpus queued-resources create "${QR_ID}" \
        --node-id="$TPU_VM_NAME" \
        --project="$TPU_PROJECT" \
        --zone="$TPU_ZONE" \
        --accelerator-type="$ACCELERATOR_TYPE" \
        --runtime-version="$RUNTIME_VERSION" \
        --best-effort \
        --quiet 2>&1
    local rc=$?

    if [[ $rc -eq 0 ]]; then
        log_ok "Queued-resource request submitted. GCP will allocate when capacity is available."
    else
        log_error "Failed to submit queued-resource request (exit $rc). Will retry next cycle."
    fi
    return $rc
}

# --- Startup banner ----------------------------------------------------------

mkdir -p "$LOG_DIR"

cat << 'BANNER'
 _____ ___  _   _  __        __    _       _         _
|_   _| _ \| | | | \ \      / /_ _| |_ ___| |__   __| | ___   __ _
  | | |  _/| |_| |  \ \ /\ / / _` | __/ __| '_ \ / _` |/ _ \ / _` |
  | | | |  |  _  |   \ V  V / (_| | || (__| | | | (_| | (_) | (_| |
  |_| |_|  |_| |_|    \_/\_/ \__,_|\__\___|_| |_|\__,_|\___/ \__, |
                                                               |___/
BANNER

log "========================================"
log "TPU Spot VM Watchdog"
log "========================================"
log "Project:       $TPU_PROJECT"
log "VM Name:       $TPU_VM_NAME"
log "Zone:          $TPU_ZONE"
log "Accelerator:   $ACCELERATOR_TYPE"
log "Runtime:       $RUNTIME_VERSION"
log "Poll interval: ${POLL_INTERVAL}s"
log "Auto-recreate: $AUTO_RECREATE"
log "Setup script:  $SETUP_SCRIPT"
log "Dry run:       $DRY_RUN"
log "========================================"

# --- Main loop ---------------------------------------------------------------

trap 'log "Watchdog interrupted. Exiting."; exit 0' SIGINT SIGTERM

while true; do
    STATE=$(get_tpu_state)

    if [[ "$STATE" != "$LAST_STATE" ]]; then
        log "State transition: ${LAST_STATE:-'(init)'} -> $STATE"
        notify "TPU '$TPU_VM_NAME' state: $STATE"

        case "$STATE" in
            READY)
                if [[ "$SETUP_RAN_FOR_CURRENT_LIFECYCLE" == false ]]; then
                    if wait_for_ssh; then
                        run_setup
                        SETUP_RAN_FOR_CURRENT_LIFECYCLE=true
                    else
                        log_warn "SSH not available, will retry next cycle."
                    fi
                else
                    log "Setup already ran for this lifecycle, skipping."
                fi
                ;;

            PREEMPTED|TERMINATED|STOPPED)
                log_warn "VM was preempted/stopped!"
                SETUP_RAN_FOR_CURRENT_LIFECYCLE=false
                if [[ "$AUTO_RECREATE" == true ]]; then
                    recreate_tpu
                else
                    log "Auto-recreate disabled. Waiting for manual intervention."
                fi
                ;;

            NOT_FOUND)
                log_warn "VM does not exist."
                SETUP_RAN_FOR_CURRENT_LIFECYCLE=false
                if [[ "$AUTO_RECREATE" == true ]]; then
                    recreate_tpu
                else
                    log "Auto-recreate disabled. Waiting for manual creation."
                fi
                ;;

            CREATING|STARTING)
                log "VM is being created/started, waiting..."
                ;;

            ERROR)
                log_error "Failed to query TPU state. Will retry."
                ;;

            *)
                log_warn "Unknown state: $STATE"
                ;;
        esac

        LAST_STATE="$STATE"
    fi

    sleep "$POLL_INTERVAL"
done
