#!/bin/bash
# ============================================================================
# CHAT_UI_PLAN.md P9-G3 (GPU, 4 x H100, about 1.3 h): the published Mamba-3 report model trained AGAIN with ONE change,
# +dataset.report_eos_target=true (P9-G1), so that it learns to end its own report. P9-G4 evaluates the result.
#   bash scripts/chat_remote.sh sync && bash scripts/chat_remote.sh submit scripts/train_report_eos_h100.sh
#
# THE RECIPE IS FIXED HERE, NOT AN ENVIRONMENT LEVER. It is h100_report_gen_m3_tower13d_s42, value by value:
# scripts/submit_v3_chain.sh's decoder submission over train_report_generation_h100.sh's own defaults. sbatch exports the
# submitting shell, so a SEED or MAX_STEPS left in it would otherwise change a 5 H100-hour run without a word. The override
# list is that wrapper's, in the same order; the only differences are output_dir and the two additions at its end
# (tests/test_willi_parity.py pins all of it).
#
# Before training, scripts/report_eos_preflight.py runs with the same cwd, interpreter and environment as the trainer, on
# this very override list, and a failure ends the job: (a) the ImageTextDataset the trainer will import must know the flag
# and come from this tree, the rsynced chat_ui code, not from the thesis checkout the shared venv has installed, which would
# ignore it silently; (b) the composed job config must equal the published run's resolved_config apart from the new run's
# names and paths, plus the flag.
#
# R8: the run's output, its logs and Hydra's own run directory all go under ${CHAT_HOME}/models/report_gen_m3_eos_s42.
# ./outputs is a symlink into the thesis checkout: Hydra's default run directory would be created there, so hydra.run.dir
# is set, and the wrapper refuses an OUT_DIR inside an outputs directory or under that checkout. A finished run (the DONE
# marker) is never overwritten; a requeued, unfinished one restarts from scratch.
# R7: the job log carries only === / RESULT / ERROR lines written by this script. The trainer's stdout and stderr, which can
# show report text and study paths, go to ${OUT_DIR}/train.log and are never printed. Read the job with
# `bash scripts/chat_remote.sh summary logs/chat_report_eos_<jobid>.log`.
# Output (MIMIC-derived: stays on the cluster, never committed, never copied to the laptop):
#   CHAT_HOME/models/report_gen_m3_eos_s42/
# ============================================================================
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --gpus=4
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM node, x86 .venv incompatible; gx13v1: faulty GPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --job-name=chat_report_eos
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

# Every line printed below starts with ===, RESULT or ERROR (the shapes `chat_remote.sh summary` shows) and names no path.
fail() { echo "ERROR $*"; exit 1; }
# `readlink -f` needs the path to exist and is not the same everywhere; python's realpath is.
realpath_py() { python -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$1"; }

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/hybrid_xmamba_h100}"
VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"
OUT_DIR="${CHAT_HOME:-/sc/home/$USER/chat_sessions}/models/report_gen_m3_eos_s42"

# --- the recipe (see the header): plain assignments, deliberately not ${VAR:-default} -------------------------------
# From submit_v3_chain.sh's decoder submission for seed 42:
MODEL_CONFIG=hybrid_150m_m3_rrg
NUM_GPUS=4
MAX_STEPS=12000
SEED=42
SAVE_TOP_K=0
AUX_LAMBDA=0.0
PREFIX_K=32
# The Stage-0 Mamba-3 backbone the decoder starts from, and the 13D image tower:
DECODER_CKPT=./outputs/h100_stage0_150m_m3/checkpoints/last.ckpt
IMAGE_ENCODER_CKPT=./outputs/h100_kd_150m_v2_full_data_lr3e6/checkpoints/last.ckpt
# From train_report_generation_h100.sh's own defaults (NUM_GPUS > 1 there selects the multi-GPU DDP trainer):
DATASET_CONFIG=cxr_mimic_full
TRAINER_CFG=h100_multi_ddp
BATCH_SIZE=32
DECODER_LR=1e-5
HEAD_LR=3e-4
GRAD_CLIP=0.5
VIT_UNFREEZE=0
VIT_LR=1e-6
GRAD_CKPT=false
AUGMENT=false
OVERSAMPLE_RARE=false
OVERSAMPLE_WEIGHT=5.0
AUX_POS_WEIGHT_CAP=10.0
# dataset.cache_dir is read only by the Hugging Face mirror loaders, never by the local parquet build this recipe trains
# on: it is passed for parity with the published run and not created here.
MIMIC_CACHE_DIR="/sc/home/$USER/dataset/mimic_cxr_cache"
# The new run's own name, and the published run it is compared with (its metadata and its TensorBoard events):
EXPERIMENT=report_gen_m3_eos_s42
PUBLISHED_EXPERIMENT=h100_report_gen_m3_tower13d_s42
PUBLISHED_META="./outputs/${PUBLISHED_EXPERIMENT}/run_metadata.json"

# The ONE override list: the preflight composes it, the trainer runs it. train_report_generation_h100.sh's list in its
# order (its image-encoder override comes last there as well), then the three differences. No value holds whitespace or
# a glob character, so the unquoted ones are safe; the paths are quoted.
OVERRIDES=(
  model=${MODEL_CONFIG}
  dataset=${DATASET_CONFIG}
  trainer=${TRAINER_CFG}
  seed=${SEED}
  save_top_k=${SAVE_TOP_K}
  trainer.max_steps=${MAX_STEPS}
  trainer.accumulate_grad_batches=1
  trainer.val_check_interval=250
  trainer.log_every_n_steps=25
  dataset.batch_size=${BATCH_SIZE}
  dataset.eval_batch_size=${BATCH_SIZE}
  dataset.num_workers=8
  dataset.pin_memory=true
  dataset.cache_dir="${MIMIC_CACHE_DIR}"
  dataset.use_augmentation=${AUGMENT}
  dataset.oversample_rare_findings=${OVERSAMPLE_RARE}
  dataset.oversample_weight=${OVERSAMPLE_WEIGHT}
  model.prefix_k=${PREFIX_K}
  model.aux_lambda=${AUX_LAMBDA}
  model.aux_pos_weight_cap=${AUX_POS_WEIGHT_CAP}
  model.decoder_lr=${DECODER_LR}
  model.head_lr=${HEAD_LR}
  model.gradient_clip_val=${GRAD_CLIP}
  model.vit_unfreeze_blocks=${VIT_UNFREEZE}
  model.vit_lr=${VIT_LR}
  model.use_gradient_checkpointing=${GRAD_CKPT}
  decoder_checkpoint="${DECODER_CKPT}"
  experiment_name=${EXPERIMENT}
  output_dir="${OUT_DIR}"
  wandb.enabled=false
  image_encoder_checkpoint="${IMAGE_ENCODER_CKPT}"
  +dataset.report_eos_target=true
  hydra.run.dir="${OUT_DIR}/hydra"
)

echo "=== P9-G3 ${EXPERIMENT}: ${MODEL_CONFIG} seed=${SEED} max_steps=${MAX_STEPS} prefix_k=${PREFIX_K}, ${NUM_GPUS} GPU(s) (trainer=${TRAINER_CFG}), report_eos_target=true ==="
echo "=== job=${SLURM_JOB_ID:-local} restart=${SLURM_RESTART_COUNT:-0} node=$(hostname) ==="

export HF_HOME="${SCRATCH_ROOT}/.hf"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH_ROOT}/.torchinductor"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED=1
source "${VENV_ACTIVATE}"

# --- guards: no directory is created and no job step starts before the path guards have passed ----------------------
MAIN_REAL="$(dirname "$(realpath_py outputs)")"       # the thesis checkout: ./outputs is a symlink into it
OUT_REAL="$(realpath_py "${OUT_DIR}")"
case "${OUT_DIR}/" in */outputs/*) fail "OUT_DIR is inside an outputs directory" ;; esac
case "${OUT_REAL}/" in
  */outputs/*) fail "OUT_DIR resolves into an outputs directory" ;;
  "${MAIN_REAL}"/*) fail "OUT_DIR is under the thesis checkout (R8)" ;;
esac
# OUT_DIR is CHAT_HOME/models/<run>: CHAT_HOME is made by scripts/chat_cluster_setup_h100.sh, owner-only, and not here.
[ -d "$(dirname "$(dirname "${OUT_DIR}")")" ] || fail "CHAT_HOME does not exist: run scripts/chat_cluster_setup_h100.sh first"
if [ -e "${OUT_DIR}/DONE" ]; then
  fail "${EXPERIMENT} is already DONE: a finished run is never overwritten"
fi

GPU_INFO="$(python -c 'import torch; n = torch.cuda.device_count(); print(n, torch.cuda.get_device_name(0) if n else "none")' 2>/dev/null)" || GPU_INFO="0 none"
GPUS="${GPU_INFO%% *}"
[ "${GPUS}" -ge "${NUM_GPUS}" ] || fail "${GPUS} GPU(s) visible, ${NUM_GPUS} needed (the sbatch header asks for --gpus=${NUM_GPUS})"
echo "=== gpus=${GPUS} (${GPU_INFO#* }) ==="
[ -f "${DECODER_CKPT}" ] || fail "decoder checkpoint (the Stage-0 Mamba-3 backbone) not found"
[ -f "${IMAGE_ENCODER_CKPT}" ] || fail "image tower checkpoint (13D) not found"
[ -f "${PUBLISHED_META}" ] || fail "run_metadata.json of the published run not found"

if [ -d "${OUT_DIR}/checkpoints" ] || [ -f "${OUT_DIR}/train.log" ]; then
  echo "=== ${EXPERIMENT}: an earlier attempt left files here and did not finish: restarting from scratch ==="
fi
mkdir -p "${OUT_DIR}"

# --- preflight: a wrong code version or a wrong recipe costs a minute here, not 5 H100-hours ------------------------
echo "=== preflight: code version and recipe ==="
rc=0
python scripts/report_eos_preflight.py --published "${PUBLISHED_META}" -- "${OVERRIDES[@]}" > "${OUT_DIR}/preflight.log" 2>&1 || rc=$?
grep -aE '^(RESULT|ERROR) ' "${OUT_DIR}/preflight.log" | tail -n 40 || true
[ "${rc}" -eq 0 ] || fail "preflight exit=${rc}, nothing was trained"

# --- training -------------------------------------------------------------------------------------------------------
echo "=== training: ${MAX_STEPS} steps on ${NUM_GPUS} GPU(s); the trainer's own output goes to train.log and is never printed ==="
START_MARK="${OUT_DIR}/.train_started"
: > "${START_MARK}"
SECONDS=0
rc=0
python scripts/train_report_generation.py --config-name config "${OVERRIDES[@]}" >> "${OUT_DIR}/train.log" 2>&1 || rc=$?
WALL_S="${SECONDS}"
if [ "${rc}" -ne 0 ]; then
  echo "ERROR train exit=${rc}"
  exit "${rc}"
fi
# SignalCheckpointCallback saves interrupt.ckpt on SIGTERM / SIGUSR1 and then raises SystemExit(0): a preempted trainer exits 0.
# Marking that run DONE would block the requeue that has to finish it, so an interrupt.ckpt newer than this attempt's start
# means "not finished". (One from an earlier attempt is older than START_MARK and does not count.)
if [ "${OUT_DIR}/checkpoints/interrupt.ckpt" -nt "${START_MARK}" ]; then
  fail "train was cut short by a signal (interrupt.ckpt written during this attempt): not marking the run DONE"
fi

# --- result, then the DONE marker (this wrapper's last act) ----------------------------------------------------------
rc=0
python scripts/report_eos_result.py --out-dir "${OUT_DIR}" --published "${PUBLISHED_META}" --wall-s "${WALL_S}" 2>> "${OUT_DIR}/result.err" || rc=$?
[ "${rc}" -eq 0 ] || echo "ERROR result exit=${rc}"
[ -f "${OUT_DIR}/checkpoints/last.ckpt" ] || fail "training finished but last.ckpt is missing: not marking the run DONE"
printf '%s job=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${SLURM_JOB_ID:-local}" > "${OUT_DIR}/DONE"
echo "=== END ${EXPERIMENT}: DONE marker written ==="
