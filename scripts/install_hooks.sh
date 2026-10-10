#!/bin/bash
# CHAT_UI_PLAN.md P9-A — install the pre-commit hook that runs scripts/check_no_restricted_files.sh (R1).
#   bash scripts/install_hooks.sh
#
# The hook goes where git looks for hooks: .git/hooks, or core.hooksPath when that is set (`git rev-parse --git-path hooks` says
# which). Worktrees of one repository share that directory, so one install covers them. The hook is a few lines that run
# scripts/check_no_restricted_files.sh of the repository being committed and do nothing where that file is absent (another branch,
# or another repository under the same core.hooksPath), so it can never block a commit it has no rule for.
#   R8: an existing pre-commit hook that is not this one is never overwritten (nor is a symlink or a directory in its place): the
#   script prints the line that chains the check into it and exits 1. Running it again over its own hook only restores the
#   executable bit. It changes no git configuration and nothing outside the hooks directory.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECK="scripts/check_no_restricted_files.sh"
[ -f "${REPO_ROOT}/${CHECK}" ] || { echo "install_hooks: ${CHECK} is missing: nothing to install a hook for" >&2; exit 1; }

TOP="$(git -C "${REPO_ROOT}" rev-parse --show-toplevel)"
HOOKS="$(git -C "${TOP}" rev-parse --git-path hooks)"            # .git/hooks, or core.hooksPath; relative to ${TOP} when relative
case "${HOOKS}" in /*) ;; *) HOOKS="${TOP}/${HOOKS}" ;; esac
HOOK="${HOOKS}/pre-commit"

WANT='#!/bin/bash
# Installed by scripts/install_hooks.sh (CHAT_UI_PLAN.md P9-A): refuse a commit that stages restricted or binary artefacts.
# It runs scripts/check_no_restricted_files.sh of the repository being committed; where there is none, there is nothing to check.
root="$(git rev-parse --show-toplevel)" || exit 1
check="${root}/scripts/check_no_restricted_files.sh"
[ -f "${check}" ] || exit 0
exec bash "${check}"'

if [ -e "${HOOK}" ] || [ -L "${HOOK}" ]; then
  if [ -f "${HOOK}" ] && [ "$(cat "${HOOK}")" = "${WANT}" ]; then
    [ -x "${HOOK}" ] || chmod +x "${HOOK}"
    echo "pre-commit hook already installed: ${HOOK}"
    exit 0
  fi
  echo "A pre-commit hook already exists and is not this one: ${HOOK}"
  echo "It was left as it is. To make it run the check, add this line to it:"
  echo "  check=\"\$(git rev-parse --show-toplevel)/${CHECK}\"; [ ! -f \"\$check\" ] || bash \"\$check\" || exit 1"
  exit 1
fi

mkdir -p "${HOOKS}"
printf '%s\n' "${WANT}" > "${HOOK}"
chmod +x "${HOOK}"
echo "pre-commit hook installed: ${HOOK}"
