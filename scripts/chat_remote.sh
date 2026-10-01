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
#
# R7: `summary` shows only lines of the fixed shapes in SUMMARY_PATTERN, then masks them. RESULT / ERROR / [tag]
# lines are wrapper-authored and may carry only numbers, hashes, row indices and file basenames: never report
# text, ids or dataset paths. mask() blanks what follows <Name>Error: / <Name>Exception: (a message can echo
# report text) and replaces 8+ digit runs (MIMIC ids) but not decimals; it can not recognise free text.
# R8: CLUSTER_REPO and MAIN_REPO must be separate absolute trees: rsync writes into the first, and the second
# is the user's thesis checkout.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${CHAT_CLUSTER_ENV:-$REPO_ROOT/scripts/chat_cluster.env}"
[[ -f "$ENV_FILE" ]] || { echo "missing $ENV_FILE: copy scripts/chat_cluster.env.example" >&2; exit 1; }
# shellcheck source=/dev/null
source "$ENV_FILE"
: "${CLUSTER_HOST:?}" "${CLUSTER_REPO:?}" "${MAIN_REPO:?}" "${SCRATCH_ROOT:?}"

# R8, before any ssh or rsync. The check is lexical (a symlink aliasing one tree onto the other can not be seen from
# here), so both paths must be absolute with no "." or ".." segment, and neither may equal, contain or sit inside the other.
tidy() { printf '%s' "$1" | sed -E 's#/+#/#g; s#/$##'; }
cr="$(tidy "$CLUSTER_REPO")"
mr="$(tidy "$MAIN_REPO")"
if [[ "$cr" != /* || "$mr" != /* || "$cr/" == */./* || "$mr/" == */./* || "$cr/" == */../* || "$mr/" == */../* \
      || "$cr" == "$mr" || "$cr" == "$mr"/* || "$mr" == "$cr"/* ]]; then
  echo "CLUSTER_REPO ($CLUSTER_REPO) and MAIN_REPO ($MAIN_REPO) must be separate absolute trees without . or .. segments: rsync writes into CLUSTER_REPO, and MAIN_REPO is the thesis checkout (R8)" >&2
  exit 1
fi

# Lines a summary may show (R7). Everything else in a log stays on the cluster. An exception line starts with its class
# name, dotted or not (sqlite3.OperationalError:, urllib.error.URLError:, HTTPException:); mask() blanks the message.
SUMMARY_PATTERN='^(RESULT |\[(probe|golden|gallery|labels|gates|server|setup|compile)\]|=== |ERROR|Traceback|([A-Za-z_][A-Za-z0-9_.]*)?(Error|Exception):|[[:space:]]*(Elapsed \(wall|Maximum resident)|  Missing keys|  prefix_k =)'

# Always call quote as the right-hand side of an assignment (x="$(quote ...)"): its `exit 2` happens in a subshell, and
# inside a larger word, such as the ssh command string, errexit would not see it and an empty argument would be sent.
quote() {   # single-quote one argument for the remote bash; refuse embedded single quotes
  [[ "$1" != *"'"* ]] || { echo "argument contains a single quote: $1" >&2; exit 2; }
  printf "'%s'" "$1"
}

mask() {    # R7: exception text, dicom-style ids and 8+ digit runs (MIMIC ids, but not the tail of a decimal) never reach the terminal
  sed -E 's/(Error|Exception):.*/\1: <msg>/; s/[0-9a-f]{8}(-[0-9a-f]{8}){4}/<id>/g; s/(^|[^0-9.])[0-9]{8,}/\1<num>/g' | cut -c1-300
}

cmd="${1:-}"
shift || true
case "$cmd" in
  sync)
    qlogs="$(quote "$CLUSTER_REPO/logs")"
    stamp="$(date -u +%Y-%m-%dT%H:%M:%SZ) $(git -C "$REPO_ROOT" rev-parse HEAD)"
    if [[ -n "$(git -C "$REPO_ROOT" status --porcelain -- app scripts hybrid_xmamba configs tests)" ]]; then
      stamp="$stamp dirty"
    else
      stamp="$stamp clean"
    fi
    ssh "$CLUSTER_HOST" "mkdir -p $qlogs"
    # The stamp is provenance: it goes last and alone, so an interrupted transfer can not leave a fresh "clean" stamp over a partial tree.
    rsync -az --exclude-from="$REPO_ROOT/.rsync-exclude-chat" --exclude=/.sync_stamp "$REPO_ROOT/" "$CLUSTER_HOST:$CLUSTER_REPO/"
    printf '%s\n' "$stamp" > "$REPO_ROOT/.sync_stamp"
    rsync -az "$REPO_ROOT/.sync_stamp" "$CLUSTER_HOST:$CLUSTER_REPO/"
    echo "[sync] $REPO_ROOT -> $CLUSTER_HOST:$CLUSTER_REPO/ ($stamp)"
    ;;
  submit)
    wrapper="${1:?usage: submit <wrapper> [VAR=value ...] [-- <sbatch args>]}"
    shift
    [[ -f "$REPO_ROOT/$wrapper" ]] || { echo "no such wrapper: $wrapper" >&2; exit 2; }
    # Only NAME=value reaches `env`: a leading-dash argument would be an env option (--chdir, --split-string, ...).
    envre='^[A-Za-z_][A-Za-z0-9_]*='
    envs=() sb=()
    while [[ $# -gt 0 ]]; do
      if [[ "$1" == "--" ]]; then
        shift; sb=("$@"); break
      elif [[ "$1" =~ $envre ]]; then
        envs+=("$(quote "$1")"); shift
      else
        echo "unexpected argument: $1 (NAME=value, or -- before the sbatch arguments)" >&2; exit 2
      fi
    done
    sbq=()
    for a in "${sb[@]+"${sb[@]}"}"; do sbq+=("$(quote "$a")"); done
    qrepo="$(quote "$CLUSTER_REPO")"
    qwrapper="$(quote "$wrapper")"
    ssh "$CLUSTER_HOST" "cd $qrepo && env ${envs[*]+"${envs[*]}"} sbatch --parsable ${sbq[*]+"${sbq[*]}"} $qwrapper"
    ;;
  state)
    jid="$(quote "${1:?jobid}")"
    ssh "$CLUSTER_HOST" "sacct -j $jid --format=JobID,JobName%28,State,Elapsed,MaxRSS,ExitCode,NodeList -P"
    ;;
  summary)
    log="$(quote "$CLUSTER_REPO/${1:?log path}")"
    qpattern="$(quote "$SUMMARY_PATTERN")"
    ssh "$CLUSTER_HOST" "grep -aE $qpattern $log | tail -n 200" | mask
    ;;
  queue)
    ssh "$CLUSTER_HOST" "squeue --me"
    ;;
  *)
    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"
    exit 2
    ;;
esac
