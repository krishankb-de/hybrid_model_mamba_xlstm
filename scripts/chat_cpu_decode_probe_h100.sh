#!/bin/bash
# ============================================================================
# CHAT_UI_PLAN.md P1-C — can the published decoder serve a chat turn on CPU?
# Decodes the first N test studies with the published protocol on CPU (cached
# twice for determinism, uncached on fewer), then compares with the published
# H100 dump line by line. Writes ${OUT}/summary.json. A throwaway warm-up arm
# runs first so the one-study arm that is subtracted as "load cost" does not
# also pay the cold page-cache read. A failed arm does not stop the job: RESULT
# covers the arms that finished, marks the rest, and the job then exits 1.
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
  [ -f "$f" ] || { echo "ERROR: not found: $(basename "$f")"; exit 1; }
done
[ -x /usr/bin/time ] || { echo "ERROR: /usr/bin/time missing"; exit 1; }
mkdir -p "${OUT}"
# Node type, for the determinism record. R7: the job log carries only [probe]/RESULT/ERROR/=== lines. The eval
# script's own output (which prints report text) stays in ${OUT}/<arm>.log on the cluster; a failed arm shows its
# traceback frames and exception class names with the messages masked, because a message can echo a study path.
echo "[probe] cpu $(lscpu | grep -m1 '^Model name' | cut -d: -f2 | xargs) ncpu=${SLURM_CPUS_PER_TASK:-?}"

DONE=""; FAILED=""
run() {  # $1 = arm name; the rest are extra flags. A failed arm is recorded, not fatal.
  local arm="$1" rss_kb; shift
  if /usr/bin/time -v python scripts/evaluate_report_generation.py \
       --checkpoint "${CHECKPOINT}" --model-config hybrid_150m_m3_rrg \
       --parquet "${DATA}/test.parquet" --decode beam --beam-size 3 --max-new-tokens 100 \
       --dump-dir "${OUT}/${arm}" "$@" > "${OUT}/${arm}.log" 2>&1; then
    DONE="${DONE} ${arm}"
  else
    FAILED="${FAILED} ${arm}"
    echo "ERROR: arm ${arm} failed; traceback frames and exception names follow, messages masked"
    grep -E '^Traceback|^  File |^[A-Za-z_][A-Za-z0-9_.]*(Error|Exception)' "${OUT}/${arm}.log" | tail -20 \
      | sed -E 's/^/[probe] /; s/(Error|Exception):.*/\1: <msg>/' || true
  fi
  grep -E "Elapsed \(wall|Missing keys|prefix_k =" "${OUT}/${arm}.log" | sed "s/^[[:space:]]*/[probe] ${arm}: /" || true
  # /usr/bin/time prints KB; chat_remote.sh summary masks 8+ digit runs (>= ~9.5 GB), so print MB
  rss_kb="$(grep -E 'Maximum resident' "${OUT}/${arm}.log" | grep -oE '[0-9]+$' | tail -1 || true)"
  [ -z "${rss_kb}" ] || echo "[probe] ${arm}: peak_rss_mb=$((rss_kb / 1024))"
}
run warm --num-samples 1 --cached-decode            # throwaway: pays the cold page-cache load, never compared
run cached_1 --num-samples 1 --cached-decode        # load cost, to subtract (warm cache)
run cached_a --num-samples "${N}" --cached-decode
run cached_b --num-samples "${N}" --cached-decode   # same node, same input: determinism
run uncached --num-samples "${N_UNCACHED}"          # the published path, on CPU

# Two RESULT lines, each under the 300 characters `chat_remote.sh summary` keeps: the run's health, then the drift.
python - "${OUT}" "${PUBLISHED}" "${N}" "${N_UNCACHED}" "${DONE}" <<'EOF'
import json, os, sys
out, published, n, nu = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
done = sys.argv[5].split()
pub = open(published).read().splitlines()
def lines(arm):
    path = "{}/{}/hyps.txt".format(out, arm)
    if arm not in done or not os.path.exists(path):
        return None          # the arm did not finish: its comparisons are null, never a silent 0
    return open(path).read().splitlines()
def differ(x, y):
    return None if x is None or y is None else sum(p != q for p, q in zip(x, y))
a, b, u = lines("cached_a"), lines("cached_b"), lines("uncached")
seen = {"cached_a": None if a is None else len(a), "cached_b": None if b is None else len(b),
        "uncached": None if u is None else len(u),
        "published": len(pub), "published_n": len(pub[:n]), "published_nu": len(pub[:nu])}
health = {
    "n": n, "n_uncached": nu,
    "arms_failed": [x for x in ("warm", "cached_1", "cached_a", "cached_b", "uncached") if x not in done],
    "lines": seen,           # observed line counts: zip() truncates silently, so a short arm shows here
    "line_counts_ok": (seen["cached_a"], seen["cached_b"], seen["uncached"], seen["published_n"], seen["published_nu"]) == (n, n, nu, n, nu),
}
drift = {
    "cached_vs_cached_differ": differ(a, b),
    "cached_vs_uncached_cpu_differ": differ(None if a is None else a[:nu], u),
    "cached_vs_published_gpu_differ": differ(a, pub[:n]),
    "uncached_vs_published_gpu_differ": differ(u, pub[:nu]),
    "differing_rows": None if a is None else [i for i, (x, y) in enumerate(zip(a, pub[:n])) if x != y],
}
for part in (health, drift):
    print("RESULT " + json.dumps(part, separators=(",", ":")))
res = dict(health)
res.update(drift)
json.dump(res, open("{}/summary.json".format(out), "w"), indent=2)
EOF
echo "=== seconds per report: (wall(cached_a) - wall(cached_1)) / (N - 1); uncached: (wall(uncached) - wall(cached_1)) / N_UNCACHED; warm is a throwaway that pays the cold-cache load ==="
if [ -n "${FAILED}" ]; then echo "=== ARMS FAILED:${FAILED} ==="; exit 1; fi
