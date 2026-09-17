#!/bin/bash
# ============================================================================
# MAMBA3_PLAN_V2.md V1-F / V3 -- the full Mamba-3 pipeline as ONE dependency chain.
#
#     source scripts/submit_v3_chain.sh              # on the login node: sbatch + shell only
#     DRY_RUN=1 source scripts/submit_v3_chain.sh    # print every submission, submit nothing
#
# SOURCE it. Never `bash scripts/submit_v3_chain.sh` on lx01 -- the login-node guard fires
# before the first line (three incidents in the M7 campaign, MAMBA3_PLAN_V2.md Verification).
# Because it is sourced it has no errexit (that would kill the login shell), no python, and
# calls nothing but sbatch; every stage is `--dependency=afterok` on its predecessor, so a
# failed stage stops the chain instead of feeding a wrong checkpoint forward. Written to
# behave identically when sourced from bash OR zsh (no ${!var}, no unquoted-parameter
# word splitting): the login shell is whatever the user has.
#
# Stages:
#   1. Stage-0 150M, 120K steps, ARM (default A2x, V2-D decides)   train_stage0_150m_h100.sh
#   2. report-gen decoder x SEEDS (42 43 44), 4 GPUs, 13D tower    train_report_generation_h100.sh
#   3. beam-3 decode on the official test split, per seed          inspect_report_generation_h100.sh
#   4. CheXbert scoring, per seed                                   score_chexbert_h100.sh
#   5. paired bootstraps per seed: vs hybrid, vs Transformer, vs floor   bootstrap_compare_h100.sh
#
# Levers (all optional):
#   ARM=A2x|A2            STAGE0_JOB=<jobid>  reuse a Stage-0 already running/finished (skips 1)
#   STAGE0_EXPERIMENT     STAGE0_STEPS=120000 STAGE0_WARMUP=2000 VAL_EVERY=10000 STAGE0_SAVE_TOP_K=1
#   SEEDS="42 43 44"      TOWER_CKPT=<13D image tower>   PARQUET=<official test split>
#   RESULTS_PREFIX=results/report_gen_m3_test_split      (per-seed dump dirs get _s<seed>)
#   HYBRID_DUMP_<seed> / TRANSFORMER_DUMP_<seed>   incumbent dumps for the paired bootstraps. All six
#                          default to the dirs the 15B-4 seed table came from
#                          (h100_scaling_state.json seed_arms); an unset one is SKIPPED loudly.
#   EVAL_TIME=12:00:00     walltime for the beam decode. The wrapper's own 06:30:00 fitted the Mamba-1
#                          hybrid; this decoder's uncached beam speed is unmeasured and exact TFLA
#                          trained ~20% slower (1.75 vs 2.18 it/s), and a TIMEOUT cancels the chain.
#   FLOOR_DUMP=results/retrieval_floor_test_split
# ============================================================================

_v3_submit() {
  # usage: _v3_submit <label> [VAR=val ...] -- sbatch args...   -> echoes the job id
  local label="$1"; shift
  local envs=()
  while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
  [ "$1" = "--" ] && shift
  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[dry-run] ${label}: env ${envs[*]} sbatch $*" >&2
    echo "DRY${RANDOM}"
  else
    local id
    id=$(env "${envs[@]}" sbatch --parsable "$@") || { echo "FATAL: sbatch failed for ${label}" >&2; return 1; }
    id="${id%%;*}"
    echo "[submitted] ${label}: job ${id}" >&2
    echo "${id}"
  fi
}

_v3_chain() {
  local ARM="${ARM:-A2x}"
  local STAGE0_EXPERIMENT="${STAGE0_EXPERIMENT:-h100_stage0_150m_m3}"
  local SEEDS="${SEEDS:-42 43 44}"
  local TOWER_CKPT="${TOWER_CKPT:-./outputs/h100_kd_150m_v2_full_data_lr3e6/checkpoints/last.ckpt}"
  local PARQUET="${PARQUET:-/sc/home/${USER}/dataset/mimic_full/test.parquet}"
  local RESULTS_PREFIX="${RESULTS_PREFIX:-results/report_gen_m3_test_split}"
  local FLOOR_DUMP="${FLOOR_DUMP:-results/retrieval_floor_test_split}"
  local HYBRID_DUMP_42="${HYBRID_DUMP_42:-results/report_gen_tower13d_test_split}"
  local HYBRID_DUMP_43="${HYBRID_DUMP_43:-results/report_gen_hybrid_seed43_test_split}"
  local HYBRID_DUMP_44="${HYBRID_DUMP_44:-results/report_gen_hybrid_seed44_test_split}"
  local TRANSFORMER_DUMP_42="${TRANSFORMER_DUMP_42:-results/report_gen_transformer_test_split}"
  local TRANSFORMER_DUMP_43="${TRANSFORMER_DUMP_43:-results/report_gen_transformer_seed43_test_split}"
  local TRANSFORMER_DUMP_44="${TRANSFORMER_DUMP_44:-results/report_gen_transformer_seed44_test_split}"
  local EVAL_TIME="${EVAL_TIME:-12:00:00}"
  local dep_stage0

  echo "=== V3 chain: ARM=${ARM} seeds=[${SEEDS}] stage0=${STAGE0_EXPERIMENT} tower=${TOWER_CKPT} ===" >&2

  # --- 1. Stage-0 -------------------------------------------------------------------------
  if [ -n "${STAGE0_JOB:-}" ]; then
    dep_stage0="${STAGE0_JOB}"
    echo "[reuse] Stage-0 job ${dep_stage0} (stage 1 skipped)" >&2
  else
    dep_stage0=$(_v3_submit "stage0 ${ARM}" \
      "ARM=${ARM}" "STEPS=${STAGE0_STEPS:-120000}" "WARMUP_STEPS=${STAGE0_WARMUP:-2000}" \
      "VAL_EVERY=${VAL_EVERY:-10000}" "SAVE_TOP_K=${STAGE0_SAVE_TOP_K:-1}" \
      "EXPERIMENT=${STAGE0_EXPERIMENT}" \
      -- scripts/train_stage0_150m_h100.sh) || return 1
  fi
  local decoder_ckpt="./outputs/${STAGE0_EXPERIMENT}/checkpoints/last.ckpt"

  # --- 2..5 per seed ----------------------------------------------------------------------
  local seed dec ev cx dump exp
  for seed in $(echo "${SEEDS}"); do   # $(...) splits in bash and zsh; ${SEEDS} alone does not in zsh
    exp="h100_report_gen_m3_tower13d_s${seed}"
    dump="${RESULTS_PREFIX}_s${seed}"

    # 2. decoder: the 14A-5 command with MODEL_CONFIG/DECODER_CKPT swapped and the 15B levers.
    #    PREFIX_K is passed explicitly so a mismatch with run_metadata.json is a hard error.
    dec=$(_v3_submit "decoder s${seed}" \
      "MODEL_CONFIG=hybrid_150m_m3_rrg" "DECODER_CKPT=${decoder_ckpt}" \
      "NUM_GPUS=4" "MAX_STEPS=12000" "SEED=${seed}" "SAVE_TOP_K=0" "AUX_LAMBDA=0.0" "PREFIX_K=32" \
      "IMAGE_ENCODER_CKPT=${TOWER_CKPT}" "EXPERIMENT=${exp}" \
      -- --gpus=4 --dependency=afterok:${dep_stage0} scripts/train_report_generation_h100.sh) || return 1

    # 3. beam-3 eval on the official test split -- the incumbents' exact (uncached) decode path.
    ev=$(_v3_submit "eval s${seed}" \
      "MODEL_CONFIG=hybrid_150m_m3_rrg" "PREFIX_K=32" "DECODE=beam" "BEAM_SIZE=3" \
      "PARQUET=${PARQUET}" "NUM_SAMPLES=999999" "DUMP_DIR=${dump}" \
      "CHECKPOINT=./outputs/${exp}/checkpoints/last.ckpt" \
      -- --time=${EVAL_TIME} --dependency=afterok:${dec} scripts/inspect_report_generation_h100.sh) || return 1

    # 4. CheXbert (its own venv), writes chexbert_metrics.json + chexbert_labels.json
    cx=$(_v3_submit "chexbert s${seed}" "DUMP_DIR=${dump}" \
      -- --dependency=afterok:${ev} scripts/score_chexbert_h100.sh) || return 1

    # 5. paired bootstraps, per-label CIs on. Same seed on both sides = paired by seed.
    local hvar="HYBRID_DUMP_${seed}" tvar="TRANSFORMER_DUMP_${seed}" hdump="" tdump=""
    eval "hdump=\${${hvar}:-}"; eval "tdump=\${${tvar}:-}"   # portable indirection (no ${!var})
    if [ -n "${hdump}" ]; then
      _v3_submit "bootstrap s${seed} vs hybrid" \
        "A=${dump}" "B=${hdump}" "NAME_A=mamba3_s${seed}" "NAME_B=hybrid_s${seed}" "PER_LABEL=true" \
        "OUTPUT=analysis/bootstrap_m3_vs_hybrid_seed${seed}.md" \
        -- --dependency=afterok:${cx} scripts/bootstrap_compare_h100.sh >/dev/null || return 1
    else
      echo "SKIP bootstrap s${seed} vs hybrid: set ${hvar}=<15B-3 dump dir for seed ${seed}>" >&2
    fi
    if [ -n "${tdump}" ]; then
      _v3_submit "bootstrap s${seed} vs transformer" \
        "A=${dump}" "B=${tdump}" "NAME_A=mamba3_s${seed}" "NAME_B=transformer_s${seed}" "PER_LABEL=true" \
        "OUTPUT=analysis/bootstrap_m3_vs_transformer_seed${seed}.md" \
        -- --dependency=afterok:${cx} scripts/bootstrap_compare_h100.sh >/dev/null || return 1
    else
      echo "SKIP bootstrap s${seed} vs transformer: set ${tvar}=<15B-3 dump dir for seed ${seed}>" >&2
    fi
    _v3_submit "bootstrap s${seed} vs floor" \
      "A=${dump}" "B=${FLOOR_DUMP}" "NAME_A=mamba3_s${seed}" "NAME_B=retrieval_floor" "PER_LABEL=true" \
      "OUTPUT=analysis/bootstrap_m3_vs_floor_seed${seed}.md" \
      -- --dependency=afterok:${cx} scripts/bootstrap_compare_h100.sh >/dev/null || return 1
  done
  echo "=== chain submitted; watch with: squeue --me   (source scripts/mamba3_watch.sh for the arm view) ===" >&2
}

_v3_chain
