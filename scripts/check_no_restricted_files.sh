#!/bin/bash
# CHAT_UI_PLAN.md P9-A — refuse a commit that stages MIMIC-derived or binary artefacts (R1).
#   bash scripts/check_no_restricted_files.sh        # what the pre-commit hook of scripts/install_hooks.sh runs
#
# It reads what is staged and nothing else: files added, copied, modified or renamed (a rename is reported under its new name),
# so deleting a restricted file that is already committed stays possible. A path is refused when it lies under outputs/,
# results/, uploads/, chat_sessions/, logs/ or hpi_results_logs/, or ends in .ckpt .parquet .npy .npz .db .pt .pth .safetensors
# .h5 .arrow or .feather, in any letter case (macOS does not tell Outputs/ from outputs/). The names are read NUL-separated: git
# otherwise quotes a name with non-ASCII or special characters, and the closing quote hides its extension.
#   Exit 0: nothing refused. Exit 1: refused; the message is a header and the offending paths, never what the files hold.
#   Exit 2: git could not list the staged files (not a repository, say): the check cannot vouch for a commit it cannot see.
#
# Allow-list, exact and short. PNGs are not refused: docs/chat_ui/evidence/ holds the screenshots of the UI over synthetic data,
# committed on purpose, and they are named below as well so that a wider list later cannot lock them out. The three logs are Stage-0
# language-model job logs (no MIMIC content) that were committed before this check existed: `git ls-files` shows them to be the only
# tracked files the patterns hit, and editing them has to stay possible. Any other file under hpi_results_logs/ is refused.
# tests/test_restricted_files_hook.py stages every tracked path as an edit would and fails if one of them would be refused.
set -euo pipefail
shopt -s nocasematch

is_restricted() {
  case "$1" in
    docs/chat_ui/evidence/*.png) return 1 ;;
    hpi_results_logs/h100_stage0_150m_2341991.log | hpi_results_logs/monitor_stage0_2351222.log | hpi_results_logs/verify_handoff_2351231.log) return 1 ;;
    outputs/* | results/* | uploads/* | chat_sessions/* | logs/* | hpi_results_logs/*) return 0 ;;
    *.ckpt | *.parquet | *.npy | *.npz | *.db | *.pt | *.pth | *.safetensors | *.h5 | *.arrow | *.feather) return 0 ;;
  esac
  return 1
}

# pipefail: a failing `git diff` fails the whole substitution, so a commit this cannot see is never waved through.
bad="$(git diff --cached --name-only -z --diff-filter=ACMR | while IFS= read -r -d '' path; do
  if is_restricted "${path}"; then printf '%s\n' "${path}"; fi
done)" || { echo "check_no_restricted_files: git could not list the staged files" >&2; exit 2; }

if [ -n "${bad}" ]; then
  echo "Refusing to commit restricted or binary artefacts:"
  printf '%s\n' "${bad}"
  exit 1
fi
