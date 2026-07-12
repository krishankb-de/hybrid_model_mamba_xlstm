#!/bin/bash
# ============================================================================
# Batch wrapper for the Stage-0 -> Phase-6 handoff check (no srun needed).
# CPU-only, fast. Auto-extracts stage0_model_only.pt from last.ckpt if absent,
# then runs verify_stage0_handoff.py (expects RESULT: PASS = 0 missing/0 unexpected).
#
# Default targets the current run dir; point at another via env:
#   RUN_DIR=outputs/h100_stage0_150m_v2_lr4e4_smoke sbatch scripts/verify_stage0_handoff.sh
#   sbatch scripts/verify_stage0_handoff.sh            # default run dir
#   cat logs/verify_handoff_<jobid>.log
# ============================================================================
#SBATCH --partition=aisc-batch
#SBATCH --account=aisc
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --job-name=verify_handoff
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -uo pipefail

VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"
RUN_DIR="${RUN_DIR:-outputs/h100_stage0_150m_v2}"
MODEL_YAML="${MODEL_YAML:-configs/model/hybrid_150m_v2.yaml}"
CKPT="${CKPT:-${RUN_DIR}/checkpoints/stage0_model_only.pt}"
export PYTHONUNBUFFERED=1

echo "=== verify Stage-0 -> Phase-6 handoff ==="; date; hostname
cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs
source "${VENV_ACTIVATE}"

# Derive the model-only checkpoint from last.ckpt if it doesn't exist yet.
if [ ! -f "${CKPT}" ]; then
  echo "[extract] ${CKPT} missing -> deriving from ${RUN_DIR}/checkpoints/last.ckpt"
  RUN_DIR="${RUN_DIR}" CKPT="${CKPT}" python - <<'PY'
import os, torch
d = os.path.join(os.environ["RUN_DIR"], "checkpoints")
src = os.path.join(d, "last.ckpt")
ck = torch.load(src, map_location="cpu", weights_only=False)
sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
st = {k[6:]: v for k, v in sd.items() if k.startswith("model.")}
torch.save({"state_dict": st}, os.environ["CKPT"])
print(f"wrote {os.environ['CKPT']}: {len(st)} keys (from {src})")
PY
fi

python scripts/verify_stage0_handoff.py "${CKPT}" "${MODEL_YAML}"
rc=$?
echo "=== verify exit code: ${rc} (0 = PASS) ==="; date
exit ${rc}
