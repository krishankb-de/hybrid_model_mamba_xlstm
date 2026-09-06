#!/bin/bash
# ============================================================================
# MAMBA3_PLAN.md M7-A2 — pre-flight for the screen arms. CPU only, ~2 minutes.
#
# Answers one question before any GPU time is spent: does this branch's code build the
# architectures it claims to, in *this* venv on *this* cluster? It checks three things that have
# each cost this project real time before:
#   1. every screen-arm config resolves and produces the intended operator (the ARCH fingerprint)
#   2. Delta at init is where it should be -- 0.02 for A1, not 0.80
#   3. the CPU test suite passes here, not just on a laptop
#
# Submit from the repo root:
#     sbatch scripts/preflight_mamba3_h100.sh
#     tail -f logs/mamba3_preflight_<jobid>.log
#
# Note: the aisc login node refuses to execute anything, python included, so this cannot be run
# as a one-liner on lx01. That is why it is an sbatch script rather than a paragraph of README.
# ============================================================================
#SBATCH --partition=aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM/Grace node, the x86 .venv cannot exec there
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --job-name=mamba3_preflight
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -euo pipefail

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/hybrid_xmamba_h100}"
VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"
RUN_TESTS="${RUN_TESTS:-true}"

echo "=== MAMBA3_PLAN.md M7-A2 pre-flight ==="
date; hostname
mkdir -p logs

cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"
echo "repo: $(pwd)"
echo "branch: $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"

export HF_HOME="${SCRATCH_ROOT}/.hf"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONUNBUFFERED=1

source "${VENV_ACTIVATE}"
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"

echo
echo "--- 1/3  architecture fingerprints (does each arm build what it claims?) ---"
# One definition of the ladder, in scripts/mamba3_arms.py, read by both this pre-flight and the
# submission (`eval "$(python scripts/mamba3_arms.py env A5)"`). That shared definition is the
# point: the first version of this file carried its own copy of three arms and filtered dataclass
# fields inline -- a *different* code path from the trainer's. It passed cleanly on 2026-09-06
# while train_stage0_distill.py was still dropping scan_impl/tfla_impl/dt_init_strategy, and job
# 2513007 trained the A1 arm with its defects intact. A pre-flight that checks a path nothing
# trains on is not a pre-flight.
python scripts/mamba3_arms.py verify --full

echo
echo "--- 2/3  Delta at init (A1's whole point: ~0.02, not ~0.80) ---"
python - <<'PYEOF'
import sys

import torch
import torch.nn.functional as F
import yaml

from hybrid_xmamba.models.configuration_hybrid import HybridConfig
from hybrid_xmamba.models.hybrid_lm import HybridLanguageModel

def delta_mean(name):
    """Mirror mamba_block.forward:117-141 on realistic input."""
    raw = yaml.safe_load(open(f"configs/model/{name}.yaml"))
    torch.manual_seed(0)
    model = HybridLanguageModel(HybridConfig.from_hydra(raw))
    block = next(l for l in model.layers if l.layer_type == "mamba")
    mixer, seq = block.mixer, 128
    ids = torch.randint(0, model.config.vocab_size, (4, seq))
    with torch.no_grad():
        x = block.norm1(model.embeddings(ids))
        xi, _ = mixer.in_proj(x).chunk(2, dim=-1)
        xc = mixer.activation(mixer.conv1d(xi.transpose(1, 2))[..., :seq].transpose(1, 2))
        dt = mixer.dt_proj(mixer.x_proj(xc)[..., : mixer.dt_rank])
        if mixer.dt_norm is not None:
            dt = mixer.dt_norm(dt)
        return F.softplus(dt).mean().item()


a0, a1 = delta_mean("hybrid_150m_v2"), delta_mean("hybrid_150m_a1")
print(f"  A0 control : Delta mean {a0:.4f}   (reference range is logU[1e-3, 1e-1])")
print(f"  A1 fixed   : Delta mean {a1:.4f}")
if not 1e-3 <= a1 <= 1.5e-1:
    print(f"\nFAIL: A1's Delta {a1:.4f} is outside [1e-3, 1.5e-1] -- the dt init did not take.")
    sys.exit(1)
print("\nOK: A1's Delta is in the reference range and A0's is not, which is the point of the arm.")
PYEOF

if [ "${RUN_TESTS}" = "true" ]; then
  echo
  echo "--- 3/3  CPU test suite on this cluster ---"
  python -m pytest tests/ -m "not cuda and not slow" -q --tb=short
fi

echo
echo "=== PRE-FLIGHT PASSED — safe to submit the screen arms ==="
date
