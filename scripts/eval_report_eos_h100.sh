#!/bin/bash
# ============================================================================
# CHAT_UI_PLAN.md P9-G4, job 1 of 3 (GPU, 1 x H100, about 1 h): the EOS-trained report model of P9-G3 decoded on the WHOLE official
# test split (n = 2,663) with the EOS-stop beam search of P9-G2, so that the model, not the token budget, ends each report.
#   bash scripts/chat_remote.sh sync && bash scripts/chat_remote.sh submit scripts/eval_report_eos_h100.sh
# Jobs 2 and 3 (CheXbert scoring, the comparison with the published run) wait for this one; the submit lines for all three, with
# their dependencies, are in the header of scripts/eval_report_eos_compare_h100.sh.
#
# THE DECODE IS FIXED HERE, NOT AN ENVIRONMENT LEVER. It is the command submit_v3_chain.sh's eval stage gave
# inspect_report_generation_h100.sh for the published Mamba-3 runs (hybrid_150m_m3_rrg, prefix_k 32, beam 3, the whole test split,
# empty input_ids, fp32) with exactly three differences:
#   --cached-decode    token-identical to the uncached path by test, and equal to the published GPU dump on 20 of 20 reports (P1-C);
#   --stop-at-eos      P9-G2: a beam that ends in the EOS the model learned in P9-G3 is set aside, and the search stops when the
#                      best hypothesis, finished or live, is a finished one;
#   --max-new-tokens 200   a ceiling, so that it is the model that ends a report; the published run is cut at 100.
# tests/test_willi_parity.py parses both published sources and pins all of it. sbatch exports the submitting shell, so a BEAM_SIZE
# or MAX_NEW_TOKENS left in it must not be able to change a decode. The levers are the three paths, as the rulings name them:
# CKPT, PARQUET and DUMP_DIR.
#
# TIME. --time=02:00:00 holds the cached decode even at the full 200-token budget. The cached step costs 6.35 ms per token on an
# H100 (analysis/mamba3_results.md, section 6: batch 1; the beams here ride the batch axis of one cache). Allow the same again for
# the beam bookkeeping, and about 0.4 s per report for the 32 prefix steps and the image. At 12.7 ms a step the 2,663 reports take
# 1.4 h at a mean of 120 tokens, and 2.2 h only if nearly every report ran to the budget, which P9-G4's prediction makes
# unlikely (under 10% are cut at it). A TIMEOUT is not requeued and cancels the chain: for more margin submit with
# `-- --time=04:00:00`.
#
# It refuses (an `ERROR ...` line, exit 1) when: DUMP_DIR is not a new results/chat_* directory (R8: ./results is a symlink into
# the thesis checkout, which may only get new chat_* subdirectories); the checkpoint or the parquet is missing; the training run
# that holds the checkpoint has no DONE marker (P9-G3 writes it only for a run that finished); the dump already holds hyps.txt or
# refs.txt (a finished dump is never overwritten); or no GPU is visible (the evaluator would fall back to the CPU without a word).
# A requeued, unfinished decode starts again and appends to the eval.log of the attempt it replaces: the dump is written only
# after the whole decode, so an attempt that was cut short left none.
#
# AFTER THE DECODE it reads eval.log, through scripts/report_eos_stats.py decode, and checks that the log vouches for the dump: the
# EOS stop line has the budget this job asked for, k + m = n, and hyps.txt and refs.txt hold n lines. Otherwise the job FAILS, so
# that afterok stops the chain: a decode that did not stop at the EOS is not the measurement this job exists for.
# R7: the evaluator prints MIMIC text and study ids (GENERATED:, REFERENCE:, one block per study), so its stdout and stderr go to
# ${DUMP_DIR}/eval.log and are never printed. The job log carries only === / RESULT / ERROR lines written by this script or the
# stats script, with numbers and names and no path. Read it with `bash scripts/chat_remote.sh summary logs/chat_eos_eval_<jobid>.log`.
# Output (MIMIC-derived: stays on the cluster, never committed, never copied to the laptop):
#   results/chat_report_eos_test_split_s42/{hyps.txt, refs.txt, eval.log, result.err}
# ============================================================================
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --gpus=1
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM node, x86 .venv incompatible; gx13v1: faulty GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=chat_eos_eval
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

# Provenance, before anything else: the commit and cleanliness `scripts/chat_remote.sh sync` last shipped to this tree. It writes
# .sync_stamp as "<UTC time> <40-hex commit> <clean|dirty>" and sends it last, so the stamp names a complete transfer. Only the
# commit and the flag are printed, and only for a line of exactly that shape: anything else reads as unknown.
SYNC="unknown"
if [ -f .sync_stamp ]; then
  { read -r _ sync_sha sync_flag sync_extra < .sync_stamp; } 2>/dev/null || true
  if [[ "${sync_sha:-}" =~ ^[0-9a-f]{40}$ && ( "${sync_flag:-}" == "clean" || "${sync_flag:-}" == "dirty" ) && -z "${sync_extra:-}" ]]; then
    SYNC="${sync_sha} ${sync_flag}"
  fi
fi
echo "=== sync ${SYNC} ==="

# Every line printed below starts with ===, RESULT or ERROR (the shapes `chat_remote.sh summary` shows) and names no path.
fail() { echo "ERROR $*"; exit 1; }

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/hybrid_xmamba_h100}"
VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"
CKPT="${CKPT:-${CHAT_HOME:-/sc/home/$USER/chat_sessions}/models/report_gen_m3_eos_s42/checkpoints/last.ckpt}"
PARQUET="${PARQUET:-/sc/home/$USER/dataset/mimic_full/test.parquet}"
DUMP_DIR="${DUMP_DIR:-results/chat_report_eos_test_split_s42}"

# --- the decode (see the header): plain assignments, deliberately not ${VAR:-default} -------------------------------------
MODEL_CONFIG=hybrid_150m_m3_rrg
PREFIX_K=32
DECODE=beam
BEAM_SIZE=3
MAX_NEW_TOKENS=200
NUM_SAMPLES=999999

echo "=== P9-G4 decode: ${MODEL_CONFIG} EOS-stop ${DECODE} ${BEAM_SIZE} budget ${MAX_NEW_TOKENS} cached, every study of the test split ==="
echo "=== job=${SLURM_JOB_ID:-local} restart=${SLURM_RESTART_COUNT:-0} node=$(hostname) ==="

export HF_HOME="${SCRATCH_ROOT}/.hf"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

# --- guards: nothing is created and nothing runs before every one of them has passed -------------------------------------
case "${DUMP_DIR}" in
  *..*) fail "DUMP_DIR holds a .. segment" ;;
  results/chat_?*) ;;
  *) fail "DUMP_DIR is not a new chat_ directory under results (R8)" ;;
esac
RUN_DIR="$(dirname "$(dirname "${CKPT}")")"
[ -f "${CKPT}" ] || fail "checkpoint not found"
[ -f "${RUN_DIR}/DONE" ] || fail "the training run has no DONE marker: it did not finish, or is still running"
[ -f "${PARQUET}" ] || fail "test parquet not found"
if [ -e "${DUMP_DIR}/hyps.txt" ] || [ -e "${DUMP_DIR}/refs.txt" ]; then
  fail "the dump already holds hyps.txt or refs.txt: a finished dump is never overwritten (R8)"
fi
[ -f "${VENV_ACTIVATE}" ] || fail "venv not found"
source "${VENV_ACTIVATE}"

GPU_INFO="$(python -c 'import torch; n = torch.cuda.device_count(); print(n, torch.cuda.get_device_name(0) if n else "none")' 2>/dev/null)" || GPU_INFO="0 none"
GPUS="${GPU_INFO%% *}"
[ "${GPUS}" -ge 1 ] || fail "${GPUS} GPU(s) visible, 1 needed (the sbatch header asks for --gpus=1)"
echo "=== gpus=${GPUS} (${GPU_INFO#* }) ==="

mkdir -p "${DUMP_DIR}"
if [ -e "${DUMP_DIR}/eval.log" ]; then
  echo "=== an earlier attempt left eval.log here and did not finish: decoding again, appending to it ==="
fi

# --- the decode ------------------------------------------------------------------------------------------------------------
echo "=== decoding: the evaluator's own output goes to eval.log and is never printed ==="
SECONDS=0
rc=0
python scripts/evaluate_report_generation.py \
  --checkpoint "${CKPT}" \
  --model-config "${MODEL_CONFIG}" \
  --prefix-k "${PREFIX_K}" \
  --cached-decode --stop-at-eos \
  --parquet "${PARQUET}" \
  --num-samples "${NUM_SAMPLES}" \
  --decode "${DECODE}" \
  --beam-size "${BEAM_SIZE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --dump-dir "${DUMP_DIR}" >> "${DUMP_DIR}/eval.log" 2>&1 || rc=$?
WALL_S="${SECONDS}"
if [ "${rc}" -ne 0 ]; then
  echo "ERROR decode exit=${rc}"
  exit "${rc}"
fi

# --- the result: whether the log vouches for the dump. R7: the stats script's stdout is cut down to the three line shapes before
# it reaches the log, and its stderr goes to a file. Its failure fails the job, so that the chained jobs do not run on this dump.
rc=0
RESULT_OUT="$(python scripts/report_eos_stats.py decode --log "${DUMP_DIR}/eval.log" --dump-dir "${DUMP_DIR}" --budget "${MAX_NEW_TOKENS}" --wall-s "${WALL_S}" 2>> "${DUMP_DIR}/result.err")" || rc=$?
printf '%s\n' "${RESULT_OUT}" | grep -aE '^(=== |RESULT |ERROR)' || true
[ "${rc}" -eq 0 ] || fail "result exit=${rc}: the log does not vouch for the dump"
echo "=== END decode ==="
