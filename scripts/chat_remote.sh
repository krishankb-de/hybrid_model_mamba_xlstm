#!/usr/bin/env bash
# chat_remote.sh — the Mac-side door to the cluster for CHAT_UI_PLAN.md (P0-G).
#
#   bash scripts/chat_remote.sh sync                          # rsync this tree -> $CLUSTER_REPO (never --delete)
#   bash scripts/chat_remote.sh submit <wrapper> [VAR=value ...] [-- <sbatch args>]   # prints the job id
#   bash scripts/chat_remote.sh state <jobid>                 # sacct one-liner
#   bash scripts/chat_remote.sh summary <log path relative to $CLUSTER_REPO>         # R7-safe lines only
#   bash scripts/chat_remote.sh queue                         # squeue --me
#
# Modelled on the legal-AI project's sync_to_cluster.sh, minus --delete (R8: the cluster side is
# additive only) and minus pulling logs to the Mac (R7: logs can hold MIMIC text). Reads
# CLUSTER_HOST, CLUSTER_REPO, MAIN_REPO, SCRATCH_ROOT from scripts/chat_cluster.env.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${CHAT_CLUSTER_ENV:-$REPO_ROOT/scripts/chat_cluster.env}"
[[ -f "$ENV_FILE" ]] || { echo "missing $ENV_FILE: copy scripts/chat_cluster.env.example" >&2; exit 1; }
# shellcheck source=/dev/null
source "$ENV_FILE"
: "${CLUSTER_HOST:?}" "${CLUSTER_REPO:?}" "${MAIN_REPO:?}" "${SCRATCH_ROOT:?}"

# Lines a summary may show (R7). Everything else in a log stays on the cluster.
SUMMARY_PATTERN='^(RESULT |\[(probe|golden|gallery|labels|gates|server|setup|compile)\]|=== |ERROR|Traceback|[A-Za-z]*Error:|[[:space:]]*(Elapsed \(wall|Maximum resident)|  Missing keys|  prefix_k =)'

quote() {   # single-quote one argument for the remote bash; refuse embedded single quotes
  [[ "$1" != *"'"* ]] || { echo "argument contains a single quote: $1" >&2; exit 2; }
  printf "'%s'" "$1"
}

mask() {    # 8+ digit runs (MIMIC subject/study ids) and dicom-style ids never reach the terminal
  sed -E 's/[0-9]{8,}/<num>/g; s/[0-9a-f]{8}(-[0-9a-f]{8}){4}/<id>/g' | cut -c1-300
}

cmd="${1:-}"
shift || true
case "$cmd" in
  sync)
    stamp="$(date -u +%Y-%m-%dT%H:%M:%SZ) $(git -C "$REPO_ROOT" rev-parse HEAD)"
    if [[ -n "$(git -C "$REPO_ROOT" status --porcelain -- app scripts hybrid_xmamba configs tests)" ]]; then
      stamp="$stamp dirty"
    else
      stamp="$stamp clean"
    fi
    printf '%s\n' "$stamp" > "$REPO_ROOT/.sync_stamp"
    ssh "$CLUSTER_HOST" "mkdir -p $(quote "$CLUSTER_REPO/logs")"
    rsync -az --exclude-from="$REPO_ROOT/.rsync-exclude-chat" "$REPO_ROOT/" "$CLUSTER_HOST:$CLUSTER_REPO/"
    echo "[sync] $REPO_ROOT -> $CLUSTER_HOST:$CLUSTER_REPO/ ($stamp)"
    ;;
  submit)
    wrapper="${1:?usage: submit <wrapper> [VAR=value ...] [-- <sbatch args>]}"
    shift
    [[ -f "$REPO_ROOT/$wrapper" ]] || { echo "no such wrapper: $wrapper" >&2; exit 2; }
    envs=() sb=()
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --) shift; sb=("$@"); break ;;
        *=*) envs+=("$(quote "$1")"); shift ;;
        *) echo "unexpected argument: $1" >&2; exit 2 ;;
      esac
    done
    sbq=()
    for a in "${sb[@]+"${sb[@]}"}"; do sbq+=("$(quote "$a")"); done
    ssh "$CLUSTER_HOST" "cd $(quote "$CLUSTER_REPO") && env ${envs[*]+"${envs[*]}"} sbatch --parsable ${sbq[*]+"${sbq[*]}"} $(quote "$wrapper")"
    ;;
  state)
    jid="$(quote "${1:?jobid}")"   # an assignment, so a missing argument or a refusal from quote() stops the script
    ssh "$CLUSTER_HOST" "sacct -j $jid --format=JobID,JobName%28,State,Elapsed,MaxRSS,ExitCode,NodeList -P"
    ;;
  summary)
    log="$(quote "$CLUSTER_REPO/${1:?log path}")"
    ssh "$CLUSTER_HOST" "grep -aE $(quote "$SUMMARY_PATTERN") $log | tail -n 200" | mask
    ;;
  queue)
    ssh "$CLUSTER_HOST" "squeue --me"
    ;;
  *)
    sed -n '2,10p' "$0"
    exit 2
    ;;
esac
