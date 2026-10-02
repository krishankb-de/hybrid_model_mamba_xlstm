#!/bin/bash
# ============================================================================
# CHAT_UI_PLAN.md P2-E (GPU) — on the GPU the engine's UNCACHED output must equal
# the published H100 dump (R2); the cached path (M6-D at scale, the first
# real-checkpoint measurement) is compared with the same dump and with the uncached
# arm, and measured, not gated. Decodes the first N test studies twice on this
# node through app.engine.RealEngine from the image file bytes
# (scripts/chat_engine_golden.py): --uncached, then the O(1) cache. No script arm:
# the published dump is the reference. The job exits 1 unless the uncached arm
# equals the dump on every line and every arm has N lines. A GPU of another kind
# than the dump's would show as drift, so the device is named in the log.
#   sbatch scripts/chat_engine_golden_gpu_h100.sh
# R7: the job log carries only [golden]/RESULT/ERROR/=== lines; the driver's own
# output goes to ${OUT}/<arm>.log, and a failed arm shows its traceback frames and
# exception class names with the messages masked.
# Output (DUA-covered, never commit): results/chat_golden_<job>/
# ============================================================================
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --gpus=1
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM node, x86 .venv incompatible; gx13v1: faulty GPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=chat_engine_golden_gpu
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs
SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/hybrid_xmamba_h100}"
VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"
DATA="${DATA:-/sc/home/$USER/dataset/mimic_full}"
CHECKPOINT="${CHECKPOINT:-./outputs/h100_report_gen_m3_tower13d_s42/checkpoints/last.ckpt}"
PUBLISHED="${PUBLISHED:-results/report_gen_m3_test_split_s42/hyps.txt}"
N="${N:-20}"
OUT="${OUT:-results/chat_golden_${SLURM_JOB_ID:-local}}"
export HF_HOME="${SCRATCH_ROOT}/.hf" HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
source "${VENV_ACTIVATE}"
for f in "${CHECKPOINT}" "${PUBLISHED}" "${DATA}/test.parquet"; do
  [ -f "$f" ] || { echo "ERROR: not found: $(basename "$f")"; exit 1; }
done
# A node where torch cannot see the GPU would decode on the CPU, and the CPU-vs-H100 "difference" would be read as
# the golden failing.
GPU="$(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")' 2>/dev/null || true)"
[ -n "${GPU}" ] || { echo "ERROR: CUDA unavailable on this node"; exit 1; }
mkdir -p "${OUT}"
echo "[golden] gpu ${GPU} node=$(hostname) n=${N}"

# $1 = arm, $2 = its python step's exit status, $3 = its log. Prints the two load lines (counts only) and every line the
# driver itself printed, tagged with the arm, then ends the job if the step failed, showing traceback frames and
# exception class names only: a message can echo a study path or report text. Every grep is guarded, because a filter
# with no match must not end the job under `set -eo pipefail`.
arm_done() {
  grep -E '^  (Missing keys|prefix_k =)' "$3" | sed "s/^[[:space:]]*/[golden] $1: /" || true
  grep '^\[golden\]' "$3" | sed "s/^\[golden\] /[golden] $1: /" || true
  [ "$2" -eq 0 ] && return 0
  echo "ERROR: $1 failed; traceback frames and exception names follow, messages masked"
  grep -E '^Traceback|^  File |^[A-Za-z_][A-Za-z0-9_.]*(Error|Exception)' "$3" | tail -20 \
    | sed -E 's/^/[golden] /; s/(Error|Exception):.*/\1: <msg>/' || true
  exit 1
}

rc=0
python scripts/chat_engine_golden.py --checkpoint "${CHECKPOINT}" --model-config hybrid_150m_m3_rrg \
  --parquet "${DATA}/test.parquet" --n "${N}" --threads "${SLURM_CPUS_PER_TASK:-8}" --uncached \
  --out "${OUT}/engine_uncached" > "${OUT}/engine_uncached.log" 2>&1 || rc=$?
arm_done engine_uncached "${rc}" "${OUT}/engine_uncached.log"
rc=0
python scripts/chat_engine_golden.py --checkpoint "${CHECKPOINT}" --model-config hybrid_150m_m3_rrg \
  --parquet "${DATA}/test.parquet" --n "${N}" --threads "${SLURM_CPUS_PER_TASK:-8}" \
  --out "${OUT}/engine_cached" > "${OUT}/engine_cached.log" 2>&1 || rc=$?
arm_done engine_cached "${rc}" "${OUT}/engine_cached.log"

# Two RESULT lines, each under the 300 characters `chat_remote.sh summary` keeps: the counts, then the differing rows
# (the first 20; golden.json has them all). zip() stops at the shorter list, so the observed line counts are part of the verdict.
python - "${OUT}" "${PUBLISHED}" "${N}" <<'EOF'
import json, sys
out, published, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
read = lambda p: open(p).read().splitlines()
rows = lambda a, b: [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
unc, cac, pub = read(out + "/engine_uncached/hyps.txt"), read(out + "/engine_cached/hyps.txt"), read(published)[:n]
res = {"n": n, "lengths": [len(unc), len(cac), len(pub)]}
res["lines_ok"] = res["lengths"] == [n, n, n]
res["engine_uncached_vs_published_differ"] = len(rows(unc, pub))
res["engine_cached_vs_published_differ"] = len(rows(cac, pub))
res["engine_cached_vs_uncached_differ"] = len(rows(cac, unc))
res["gate_passed"] = res["lines_ok"] and res["engine_uncached_vs_published_differ"] == 0
diff = {"rows_uncached_vs_published": rows(unc, pub), "rows_cached_vs_published": rows(cac, pub)}
print("RESULT " + json.dumps(res, separators=(",", ":")))
print("RESULT " + json.dumps({k: v[:20] for k, v in diff.items()}, separators=(",", ":")))
json.dump(dict(res, **diff), open(out + "/golden.json", "w"), indent=2)
print("=== GOLDEN GPU " + ("PASSED: engine (uncached) == published ===" if res["gate_passed"] else "FAILED ==="))
sys.exit(0 if res["gate_passed"] else 1)
EOF
