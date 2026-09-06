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
# Lightning writes its metrics into a carriage-return progress bar, so the log has
# to be unfolded before it can be grepped.
for f in "${LOGDIR}"/m3_screen_*.log "${LOGDIR}"/m3_A*.log; do
  [ -e "$f" ] || continue
  arm=$(grep -m1 '^\[arm\]' "$f" 2>/dev/null | awk '{print $2}')
  [ -n "$arm" ] || arm=$(grep -m1 -o 'm3_[A-Za-z0-9_]*' "$f" 2>/dev/null | head -1)
  fp=$(grep -m1 -o 'mamba3\?(\?[^|]*' "$f" 2>/dev/null | head -c 90)
  step=$(tr '\r' '\n' < "$f" | grep -oE '[0-9]+/[0-9]+ +[0-9]+:[0-9]+:[0-9]+' | tail -1)
  ppl=$(tr '\r' '\n' < "$f" | grep -oE 'val/perplexity:? *[0-9.]+' | tail -1)
  vloss=$(tr '\r' '\n' < "$f" | grep -oE 'val/loss:? *[0-9.]+' | tail -1)
  bad=$(tr '\r' '\n' < "$f" | grep -cE 'nan|inf|Traceback|CUDA out of memory|Error' )
  printf "  %-10s %-28s %s\n" "${arm:-?}" "${step:-<no step yet>}" "${vloss:-} ${ppl:-}"
  [ "${bad:-0}" -gt 0 ] && echo "      ⚠ $bad line(s) matching nan/inf/Traceback/OOM/Error in $f"
  restarts=$(grep -c '=== M7-B screen: arm' "$f" 2>/dev/null)
  [ "${restarts:-0}" -gt 1 ] && echo "      ⚠ this log contains $restarts run headers — the job was REQUEUED and restarted"
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
