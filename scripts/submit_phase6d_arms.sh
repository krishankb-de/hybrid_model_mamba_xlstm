#!/bin/bash
# ============================================================================
# Phase 6D — factorial lever block submitter. NOT an sbatch script itself:
# run it on the login node and it sbatches one job per arm.
#
#   bash scripts/submit_phase6d_arms.sh            # dry run, prints the plan
#   bash scripts/submit_phase6d_arms.sh --submit   # actually sbatch
#   ARMS="D0 D2 D3" bash scripts/submit_phase6d_arms.sh --submit
#
# WHY FACTORIAL. Six one-at-a-time probes have come back null, which has made
# single-lever testing expensive per bit of information. D1-D3 run in parallel
# for attribution; D4 stacks them for the number. Every arm holds the same
# 384000-sample (13.93-epoch) budget and batch-derived LRs, so neither the
# epoch confound nor the LR confound from the Phase-6 sweep can recur.
#
# GATE: > 1.1pp over D0 (SE ~0.57pp at p~0.11, N=3063). Under that is noise.
# Interpret in-training retrieval as a SIGNAL ONLY — the authoritative number
# comes from evaluate_cxr_retrieval.py on the best-by-val/total_loss ckpt.
# ============================================================================

set -euo pipefail

SUBMIT=false
[ "${1:-}" = "--submit" ] && SUBMIT=true

WRAPPER="scripts/train_biomedclip_kd_150m_h100.sh"
ARMS="${ARMS:-D0 D1a D1b D1c D2 D3 D4 D5}"

# Common to every arm: bs=64 (the best Phase-6 batch), LR + MAX_STEPS derived
# from batch by the wrapper. Do NOT set BACKBONE_LR/HEAD_LR here — letting the
# wrapper derive them is what keeps the arms LR-matched.
COMMON="BATCH_SIZE=64"

# arm | description | env overrides
read -r -d '' PLAN <<'EOF' || true
D0|control: Phase-6B recipe, unmodified|
D1a|6D-1 ViT unfreeze 4 (the only lever with a measured positive)|VIT_UNFREEZE=4
D1b|6D-1 ViT unfreeze 6|VIT_UNFREEZE=6
D1c|6D-1 ViT unfreeze 12 = whole ViT-B/16 (OOM watch: full image-tower backward on top of the fp32 scan; drop to BATCH_SIZE=32 if it dies)|VIT_UNFREEZE=12
D2|6D-2 KD anchor decays 0.3 -> 0.0 over 2000 steps post-unfreeze|KD_DECAY_STEPS=2000 ALPHA_KD_FLOOR=0.0
D3|6D-3 SigLIP + multi-positive mask (one arm: same failure mode)|CLIP_LOSS=siglip MULTIPOS=true
D4|stack: best-of-D1 + D2 + D3 (edit VIT_UNFREEZE after D1 reports)|VIT_UNFREEZE=6 KD_DECAY_STEPS=2000 ALPHA_KD_FLOOR=0.0 CLIP_LOSS=siglip MULTIPOS=true
D5|6D-5 ablate the SimCSE term|GAMMA_SIMCSE=0.0
EOF

echo "============================================================"
echo "Phase 6D factorial block  (submit=${SUBMIT})"
echo "Baseline to beat: authoritative i2t R@10 = 0.1113"
echo "Gate: > 1.1pp movement. Anything less is noise."
echo "============================================================"

while IFS='|' read -r arm desc envs; do
  [ -z "${arm}" ] && continue
  case " ${ARMS} " in *" ${arm} "*) ;; *) continue ;; esac

  EXP="h100_6d_${arm}"
  CMD="${COMMON} ${envs} EXPERIMENT=${EXP} sbatch ${WRAPPER}"

  echo ""
  echo "--- ${arm}: ${desc}"
  echo "    ${CMD}"
  if [ "${SUBMIT}" = "true" ]; then
    eval "${CMD}"
  fi
done <<< "${PLAN}"

echo ""
if [ "${SUBMIT}" = "true" ]; then
  echo "Submitted. Watch the [CLIPRetrieval epoch=N] prints — they flush to the"
  echo "SLURM log; the Rich per-step bar is buffered and will look frozen."
  echo "Also watch train/false_neg_rate: it is logged whenever the dataset emits"
  echo "text_hash, regardless of MULTIPOS, and tells you empirically whether the"
  echo "multi-positive mask has anything to fix."
  echo ""
  echo "When each finishes, get the AUTHORITATIVE number (in-training peaks are"
  echo "on the same 3063 pairs — do not select on them):"
  echo "  CKPT=./outputs/h100_6d_<arm>/checkpoints/<best-by-val_total_loss> \\"
  echo "    MODE=retrieval DATASET=mimic sbatch scripts/eval_h100.sh"
else
  echo "Dry run. Re-run with --submit to launch."
fi
