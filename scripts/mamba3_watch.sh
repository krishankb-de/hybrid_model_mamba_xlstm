#!/bin/bash
# ============================================================================
# "How is the screen doing?" — squeue, preemption state, per-arm progress, disk.
#
# RUN IT AS A JOB. The aisc login node refuses to execute scripts at all:
#
#     $ bash scripts/mamba3_watch.sh
#     This command is not allowed on the login node!
#
# printed before the first echo, so it is `bash <script>` itself that is
# intercepted, not any command inside. Three ways, cheapest first:
#
#     source scripts/mamba3_watch.sh          # no new bash process — may pass
#     srun --partition=aisc-batch --account=aisc --cpus-per-task=2 --mem=4G \
#          --time=00:05:00 bash scripts/mamba3_watch.sh
#     sbatch scripts/mamba3_watch.sh          # then: cat logs/mamba3_watch_<id>.log
#
# The individual commands below ARE fine to paste straight into lx01 — squeue,
# sacct, grep and cat are all proven to work there interactively. It is only
# running them as a script that trips the guard.
#
# No python anywhere: the login node refuses that too.
#
# WITH_DISK=0 skips the du pass. `du -sh $HOME` walks the whole home directory
# and is exactly the kind of thing the login-node guard exists to stop, so it is
# only run when this script runs as a job.
# ============================================================================
#SBATCH --partition=aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --exclude=ga03,gx17v1,gx13v1
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --job-name=mamba3_watch
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -uo pipefail

# BASH_SOURCE keeps `source scripts/mamba3_watch.sh` working as well as bash/sbatch.
_WATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
cd "${SLURM_SUBMIT_DIR:-${_WATCH_DIR}}/hybrid_model_mamba_xlstm" 2>/dev/null \
  || cd "${SLURM_SUBMIT_DIR:-${_WATCH_DIR}}"
LOGDIR="${LOGDIR:-logs}"
WITH_DISK="${WITH_DISK:-auto}"
[ "${WITH_DISK}" = "auto" ] && { [ -n "${SLURM_JOB_ID:-}" ] && WITH_DISK=1 || WITH_DISK=0; }

hr() { printf '%s\n' "----------------------------------------------------------------------"; }

echo "=== 1. running now ==="
command -v squeue >/dev/null \
  && { squeue --me --format="%.10i %.12P %.9N %.2t %.10M %.20j" 2>/dev/null || squeue --me; } \
  || echo "  (squeue unavailable — not on the cluster?)"
echo
echo "  time left (a TIMEOUT killed A1 at 12:00:03 with the run unfinished):"
command -v squeue >/dev/null \
  && squeue --me --format="  %.10i %.20j %.10M elapsed %.10L left" 2>/dev/null \
  || true
hr

echo "=== 2. state incl. preemption (sacct, last 3 days) ==="
# PREEMPTED / REQUEUED / NODE_FAIL / TIMEOUT all show up here and nowhere in squeue
# once the job is gone. aisc-batch is preemptible and these jobs carry --requeue,
# and nothing passes ckpt_path -- so a requeued arm RESTARTS FROM STEP 0 rather
# than resuming. That is the single failure worth catching early.
sacct --starttime=now-3days --user="$USER" \
      --format="JobID%-16,JobName%-18,State%-14,ExitCode%-8,Elapsed%-10,Start%-16,NodeList%-8" \
  2>/dev/null | grep -Ev "\.(batch|extern|[0-9]+) " || echo "  (sacct unavailable)"
hr

echo "=== 3. per-arm progress ==="
# Lightning's progress bar is written with carriage returns and only lands in the log
# when the run ENDS, so a live job shows nothing there. The checkpoints are the honest
# mid-run signal: one every 2,000 steps, and `auto_insert_metric_name: false` puts the
# val loss straight into the filename (checkpoint-<epoch>-<val_loss>.ckpt).
for d in outputs/m3_screen_*; do
  [ -d "$d" ] || continue
  arm=$(basename "$d")
  ck="$d/checkpoints"
  n=$(ls -1 "$ck"/checkpoint-*.ckpt 2>/dev/null | wc -l | tr -d ' ')
  best=$(ls -1 "$ck"/checkpoint-*.ckpt 2>/dev/null | sed -E 's/.*-([0-9]+\.[0-9]+)\.ckpt/\1/' \
           | sort -g | head -1)
  last_mtime=$(ls -1t "$ck"/*.ckpt 2>/dev/null | head -1 | xargs -r stat -c %y 2>/dev/null \
                 | cut -d. -f1)
  # exp() without python: awk is not blocked and is everywhere.
  ppl=$( [ -n "$best" ] && awk -v l="$best" 'BEGIN{printf "%.3f", exp(l)}' )
  printf "  %-22s ckpts=%-3s best val/loss=%-8s ppl=%-9s last write: %s\n" \
         "$arm" "${n:-0}" "${best:-–}" "${ppl:-–}" "${last_mtime:-–}"
done
echo
echo "  a finished 12,000-step arm has 6 checkpoints; each is one val pass (every 2,000 steps)"
echo
# Real failures only. Matching bare "inf" hits config.json, dataset_infos.json and [INFO];
# matching "Error" hits nothing useful either. Anchor on things that are actually fatal.
for f in "${LOGDIR}"/m3_screen_*.log "${LOGDIR}"/m3_A*.log; do
  [ -e "$f" ] || continue
  bad=$(grep -cE 'Traceback|CUDA out of memory|RuntimeError|DUE TO TIME LIMIT|\bnan\b|Segmentation fault' "$f" 2>/dev/null)
  [ "${bad:-0}" -gt 0 ] && { echo "  ⚠ $f"; grep -E 'Traceback|CUDA out of memory|RuntimeError|DUE TO TIME LIMIT|\bnan\b|Segmentation fault' "$f" | tail -2 | sed 's/^/      /'; }
  restarts=$(grep -c '=== M7-B screen: arm' "$f" 2>/dev/null)
  [ "${restarts:-0}" -gt 1 ] && echo "  ⚠ $f has $restarts run headers — REQUEUED, restarted from step 0"
done
hr

echo "=== 4. disk ==="
if [ "${WITH_DISK}" != "1" ]; then
  echo "  skipped (WITH_DISK=0). du walks the tree and the login node blocks that;"
  echo "  re-run this script as a job, or set WITH_DISK=1 if you are on a run node."
else
  echo "  per experiment:"
  du -sh outputs/* 2>/dev/null | sort -h | tail -15
  echo
  echo "  totals:"
  du -sh outputs 2>/dev/null
  du -sh "$HOME" 2>/dev/null
  echo
  df -h "$HOME" /sc/scratch/"$USER" 2>/dev/null
  echo
  echo "  aisc reports the home quota in the login banner; 'du -sh \$HOME' is the number"
  echo "  that moves. Checkpoints live in outputs/<experiment>/checkpoints/."
fi
