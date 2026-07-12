#!/bin/bash
# ============================================================================
# Phase-5 failure diagnostic (job 2320933: val PPL 1165 vs gate <=15.62).
# Runs 3 evidence steps SEQUENTIALLY and stores every result in the SLURM log
# AND in outputs/h100_stage0_150m_v2/diag/*.txt. Submit once and let it queue:
#
#   sbatch scripts/diag_stage0_150m.sh
#   squeue --me
#   cat logs/diag_stage0_150m_<jobid>.log
#
# NOTE: deliberately NOT `set -e` — every step must run even if an earlier one
# fails, so we still get the loss trajectory even if the eval errors.
# ============================================================================
#SBATCH --partition=aisc-shortrun
#SBATCH --account=aisc
#SBATCH --gpus=1
#SBATCH --exclude=ga03,gx17v1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --job-name=diag_stage0_150m
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -uo pipefail

VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/hybrid_xmamba_h100}"
RUN_DIR="outputs/h100_stage0_150m_v2"
CKPT_DIR="${RUN_DIR}/checkpoints"
MODEL_CONFIG="hybrid_150m_v2"

export PYTHONUNBUFFERED=1
export HF_HOME="${SCRATCH_ROOT}/.hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "======================================================================"
echo "=== Phase-5 diagnostic  |  SLURM Job ${SLURM_JOB_ID:-?}"
date; hostname
echo "======================================================================"

cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs "${RUN_DIR}/diag"

source "${VENV_ACTIVATE}"
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"

echo ""
echo "################ STEP 1: loss + LR trajectory (tfevents) ################"
python - <<'PY' 2>&1 | tee "${RUN_DIR}/diag/trajectory.txt"
import glob, os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
base = "outputs/h100_stage0_150m_v2/logs/stage0_kd"
versions = sorted(glob.glob(os.path.join(base, "version_*")))
print("versions found:", versions)
for ver in versions:
    print("\n\n##########", ver, "##########")
    ea = EventAccumulator(ver, size_guidance={'scalars': 0}); ea.Reload()
    tags = ea.Tags()['scalars']
    print("TAGS:", tags)
    # full trajectory for val + lr + perplexity (sampled ~40 pts + last)
    for tag in tags:
        t = tag.lower()
        if any(k in t for k in ('val', 'lr', 'learning_rate', 'perplex')):
            s = ea.Scalars(tag)
            if not s:
                continue
            print(f"\n== {tag}  ({len(s)} pts) ==")
            step = max(1, len(s) // 40)
            for e in s[::step]:
                print(f"  step {e.step:>7}  {e.value:.5f}")
            print(f"  LAST step {s[-1].step:>7}  {s[-1].value:.5f}")
    # first/last of the training losses (did it descend at all?)
    for tag in tags:
        if 'total_loss' in tag.lower() or 'ce_loss' in tag.lower():
            s = ea.Scalars(tag)
            if s:
                print(f"\n== {tag}: first {s[0].value:.4f}@{s[0].step}  ->  last {s[-1].value:.4f}@{s[-1].step}")
PY

echo ""
echo "################ STEP 2: extract model-only checkpoint ################"
python - <<'PY' 2>&1 | tee "${RUN_DIR}/diag/extract.txt"
import torch, os
d = "./outputs/h100_stage0_150m_v2/checkpoints"
src = os.path.join(d, "last.ckpt")
ck = torch.load(src, map_location="cpu", weights_only=False)
sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
st = {k[6:]: v for k, v in sd.items() if k.startswith("model.")}
out = os.path.join(d, "stage0_model_only.pt")
torch.save({"state_dict": st}, out)
print(f"wrote {out}: {len(st)} keys (from {src})")
PY

echo ""
echo "################ STEP 3: authoritative PubMed val PPL ################"
python scripts/evaluate_lm.py \
  --checkpoint ./outputs/h100_stage0_150m_v2/checkpoints/stage0_model_only.pt \
  --model-config "${MODEL_CONFIG}" \
  --dataset pubmed --split validation \
  --batch-size 16 --max-length 512 \
  --output-dir "${RUN_DIR}/diag" 2>&1 | tee "${RUN_DIR}/diag/eval_ppl.txt"

echo ""
echo "======================================================================"
echo "=== diagnostic done. Results in:"
echo "===   ${RUN_DIR}/diag/trajectory.txt   (loss + LR curve — the key evidence)"
echo "===   ${RUN_DIR}/diag/extract.txt"
echo "===   ${RUN_DIR}/diag/eval_ppl.txt     (authoritative PPL)"
echo "=== ...and all of it is in this SLURM log too."
date
echo "======================================================================"
