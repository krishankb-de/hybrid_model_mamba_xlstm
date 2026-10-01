#!/bin/bash
# ============================================================================
# CHAT_UI_PLAN.md P1-A — cluster reachability probe for the chat server.
# CPU only. Times SQLite commits on /sc/home (D22), then serves the stdlib
# probe until --time runs out, writing <node>:<port> to ENDPOINT_FILE.
#
#   BIND=127.0.0.1 sbatch scripts/chat_probe_h100.sh   # path (b): ssh -J lx01 <node>
#   BIND=0.0.0.0   sbatch scripts/chat_probe_h100.sh   # path (a): forward via lx01
# ============================================================================
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM node; gx13v1: faulty GPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=01:00:00
#SBATCH --job-name=chat_probe
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs
BIND="${BIND:-127.0.0.1}"
ENDPOINT_FILE="${ENDPOINT_FILE:-$HOME/chat_sessions/probe_endpoint}"
echo "=== chat probe: node=$(hostname) bind=${BIND} job=${SLURM_JOB_ID:-?} ==="
python3 app/tunnel/probe_server.py --bind "${BIND}" --endpoint-file "${ENDPOINT_FILE}" \
  --sqlite-probe "$HOME/chat_sessions/probe.db"
