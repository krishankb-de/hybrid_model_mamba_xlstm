#!/bin/bash
# ============================================================================
# One command for "how is the screen doing?" — runs on the aisc LOGIN node.
#
#     bash scripts/mamba3_watch.sh
#
# It answers the three questions worth asking while arms are training:
#   1. what is running, and did anything get PREEMPTED or REQUEUED?
#   2. how far has each arm got, and is its loss still finite?
#   3. how much disk are the checkpoints eating?
#
# No python: the login node refuses to execute it. Everything here is squeue,
# sacct, grep, du and df.
# ============================================================================
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
LOGDIR="${LOGDIR:-logs}"

hr() { printf '%s\n' "----------------------------------------------------------------------"; }

echo "=== 1. running now ==="
squeue --me --format="%.10i %.12P %.9N %.2t %.10M %.20j" 2>/dev/null || squeue --me
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
du -sh outputs/* 2>/dev/null | sort -h | tail -15
echo
echo "  totals:"
du -sh outputs 2>/dev/null
du -sh "$HOME" 2>/dev/null
echo
df -h "$HOME" /sc/scratch/"$USER" 2>/dev/null
echo
echo "  (aisc reports the home quota in the login banner; 'du -sh \$HOME' above is the"
echo "   number that moves. Checkpoints live in outputs/<experiment>/checkpoints/.)"
