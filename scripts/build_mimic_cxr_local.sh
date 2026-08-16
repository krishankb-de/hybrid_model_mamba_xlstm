#!/bin/bash
# ============================================================================
# Phase 8 (H100_SCALING_PLAN.md) — SLURM wrapper for build_mimic_cxr_local.py.
#
# WHY THIS SCRIPT EXISTS (Phase 7E, 2026-08-16)
# The login node (lx01) refuses ANY script execution, not just heavy ones —
# confirmed live: `python build_mimic_cxr_local.py meta` on lx01 was rejected
# with "This command is not allowed on the login node!" before it made a
# single request. Per docs.sc.hpi.de/cluster/Resources/{Login-Nodes,
# Storage/Data-Transfer}: external downloads belong on COMPUTE nodes via
# Slurm — cpu-interactive (8h cap) for small/short stages, cpu-batch (7-day
# cap) for the long fetch — not on a Run Node (rx01/rx02, explicitly NOT
# meant for data acquisition) and not on the AISC GPU partitions (this job
# needs zero GPU; requesting one would be pure waste on a shared resource).
#
# This is a CPU-only, no-GPU job on purpose. Do not add --gpus.
#
# COURTESY (per the Data-Transfer doc, verbatim: "Always contact helpdesk
# before transferring large datasets"): email sc-helpdesk@hpi.de before the
# full `fetch` run. ~310-400 GB is transferred (nothing is kept — see
# build_mimic_cxr_local.py's docstring), but it is still worth a heads-up on
# a cluster that explicitly polices "flooding the network" / "saturating
# connection tracking tables". WORKERS defaults to 8 concurrent HTTPS
# connections for exactly this reason — deliberately conservative, raise it
# only if the helpdesk says it's fine to.
#
# ENV: STAGE (meta|manifest|fetch|pack, required), OUT (build dir, required),
#      SIZE, VIEWS, CHUNK, WORKERS, LIMIT, EXCLUDE_HASHES, MIN_MATCH_FRAC,
#      SCRATCH_ROOT, VENV_ACTIVATE.
#
# Usage:
#   STAGE=meta     OUT=/sc/home/$USER/dataset/mimic_full sbatch scripts/build_mimic_cxr_local.sh
#   STAGE=manifest OUT=/sc/home/$USER/dataset/mimic_full sbatch scripts/build_mimic_cxr_local.sh
#   # stop, read $OUT/build_report.json (per Phase 8 usage notes) before fetch
#   STAGE=fetch    OUT=/sc/home/$USER/dataset/mimic_full LIMIT=500 sbatch scripts/build_mimic_cxr_local.sh   # smoke
#   STAGE=fetch    OUT=/sc/home/$USER/dataset/mimic_full sbatch scripts/build_mimic_cxr_local.sh             # the long one; resumable, just resubmit
#   STAGE=pack     OUT=/sc/home/$USER/dataset/mimic_full EXCLUDE_HASHES=legacy_gallery_hashes.txt sbatch scripts/build_mimic_cxr_local.sh
#
# This cluster requires an explicit Slurm account (confirmed live, 2026-08-16:
# sbatch refuses with "No Slurm account specified" otherwise). --account=aisc
# (used by every GPU script in this repo) submitted but then sat PENDING with
# reason QOSNotAllowed — the aisc account carries QOS=aisc, which is scoped to
# the AISC partitions (aisc-batch/aisc-interactive/aisc-shortrun), not the
# general cpu-batch/cpu-interactive partitions this script uses.
# `sacctmgr show user $USER withassoc format=User,Account,Partition,QOS`
# showed a second association: account=default, QOS=normal — the general-
# partition account. Defaulted to that. Override at submit time if a
# different allocation should be used:
#   sbatch --account=<other_account> scripts/build_mimic_cxr_local.sh
# ============================================================================
#SBATCH --partition=cpu-batch
#SBATCH --account=default
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --job-name=mimic_build
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -euo pipefail

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/hybrid_xmamba_h100}"
VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"

STAGE="${STAGE:?set STAGE=meta|manifest|fetch|pack}"
OUT="${OUT:?set OUT=/sc/home/\$USER/dataset/mimic_full}"
SIZE="${SIZE:-320}"
VIEWS="${VIEWS:-PA AP}"
CHUNK="${CHUNK:-2000}"
WORKERS="${WORKERS:-8}"
LIMIT="${LIMIT:-0}"
EXCLUDE_HASHES="${EXCLUDE_HASHES:-}"
MIN_MATCH_FRAC="${MIN_MATCH_FRAC:-0.95}"

echo "=== MIMIC-CXR-JPG local build (STAGE=${STAGE}) ==="
date; hostname
mkdir -p logs "${OUT}"

cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"

export PYTHONUNBUFFERED=1

source "${VENV_ACTIVATE}"

case "${STAGE}" in
  meta)
    python scripts/build_mimic_cxr_local.py meta --out "${OUT}"
    ;;
  manifest)
    python scripts/build_mimic_cxr_local.py manifest --out "${OUT}" \
      --size "${SIZE}" --views ${VIEWS} $( [ "${LIMIT}" != "0" ] && echo "--limit ${LIMIT}" )
    ;;
  fetch)
    python scripts/build_mimic_cxr_local.py fetch --out "${OUT}" \
      --size "${SIZE}" --chunk "${CHUNK}" --workers "${WORKERS}" \
      $( [ "${LIMIT}" != "0" ] && echo "--limit ${LIMIT}" )
    ;;
  pack)
    python scripts/build_mimic_cxr_local.py pack --out "${OUT}" \
      $( [ -n "${EXCLUDE_HASHES}" ] && echo "--exclude-hashes ${EXCLUDE_HASHES} --min-match-frac ${MIN_MATCH_FRAC}" )
    ;;
  *)
    echo "ERROR: unknown STAGE='${STAGE}' (expected meta|manifest|fetch|pack)"; exit 1
    ;;
esac

echo "=== END: STAGE=${STAGE} -> ${OUT} ==="
date
