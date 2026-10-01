#!/bin/bash
# ============================================================================
# CHAT_UI_PLAN.md P1-C — can the published decoder serve a chat turn on CPU?
# Decodes the first N test studies with the published protocol on CPU (cached
# twice for determinism, uncached on fewer), then compares with the published
# H100 dump line by line. Writes ${OUT}/summary.json.
#
#   sbatch scripts/chat_cpu_decode_probe_h100.sh
# Output (DUA-covered, never commit): results/chat_cpu_probe_<job>/
# ============================================================================
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM node, x86 .venv incompatible; gx13v1: faulty GPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=chat_cpu_decode_probe
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
N_UNCACHED="${N_UNCACHED:-5}"
OUT="${OUT:-results/chat_cpu_probe_${SLURM_JOB_ID:-local}}"
export HF_HOME="${SCRATCH_ROOT}/.hf" HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
source "${VENV_ACTIVATE}"
for f in "${CHECKPOINT}" "${PUBLISHED}" "${DATA}/test.parquet"; do
  [ -f "$f" ] || { echo "ERROR: not found: $f"; exit 1; }
done
mkdir -p "${OUT}"
# Node type, for the determinism record. R7: the job log carries only [probe]/RESULT/=== lines;
# the eval script's own output (which prints report text) stays in ${OUT}/<arm>.log on the cluster.
echo "[probe] cpu $(lscpu | grep -m1 '^Model name' | cut -d: -f2 | xargs) ncpu=${SLURM_CPUS_PER_TASK:-?}"
TIME_V=(); [ -x /usr/bin/time ] && TIME_V=(/usr/bin/time -v)

run() {  # $1 = arm name; the rest are extra flags
  local arm="$1"; shift
  "${TIME_V[@]}" python scripts/evaluate_report_generation.py \
    --checkpoint "${CHECKPOINT}" --model-config hybrid_150m_m3_rrg \
    --parquet "${DATA}/test.parquet" --decode beam --beam-size 3 --max-new-tokens 100 \
    --dump-dir "${OUT}/${arm}" "$@" > "${OUT}/${arm}.log" 2>&1 \
    || { echo "ERROR: arm ${arm} failed; traceback frames follow (R7: no data lines)"
         grep -E '^Traceback|^  File |^[A-Za-z]*Error' "${OUT}/${arm}.log" | tail -20 | sed 's/^/[probe] /'; exit 1; }
  grep -E "Elapsed \(wall|Maximum resident|Missing keys|prefix_k =" "${OUT}/${arm}.log" | sed "s/^[[:space:]]*/[probe] ${arm}: /" || true
}
run cached_1 --num-samples 1 --cached-decode        # load cost, to subtract
run cached_a --num-samples "${N}" --cached-decode
run cached_b --num-samples "${N}" --cached-decode   # same node, same input: determinism
run uncached --num-samples "${N_UNCACHED}"          # the published path, on CPU

python - "${OUT}" "${PUBLISHED}" "${N}" "${N_UNCACHED}" <<'EOF'
import json, sys
out, published, n, nu = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
pub = open(published).read().splitlines()
def lines(arm):
    return open("{}/{}/hyps.txt".format(out, arm)).read().splitlines()
a, b, u = lines("cached_a"), lines("cached_b"), lines("uncached")
res = {
    "n": n, "n_uncached": nu,
    "cached_vs_cached_differ": sum(x != y for x, y in zip(a, b)),
    "cached_vs_uncached_cpu_differ": sum(x != y for x, y in zip(a[:nu], u)),
    "cached_vs_published_gpu_differ": sum(x != y for x, y in zip(a, pub[:n])),
    "uncached_vs_published_gpu_differ": sum(x != y for x, y in zip(u, pub[:nu])),
    "differing_rows": [i for i, (x, y) in enumerate(zip(a, pub[:n])) if x != y],
}
print("RESULT " + json.dumps(res))
json.dump(res, open("{}/summary.json".format(out), "w"), indent=2)
EOF
echo "=== seconds per report: (wall(cached_a) - wall(cached_1)) / (N - 1); uncached: (wall - load) / N_UNCACHED ==="
