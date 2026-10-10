#!/bin/bash
# CHAT_UI_PLAN.md P9-A — refuse a commit that stages MIMIC-derived or binary artefacts (R1).
#   bash scripts/check_no_restricted_files.sh        # what the pre-commit hook of scripts/install_hooks.sh runs
#
# It reads what is staged and nothing else: files added, copied, modified, renamed or changed in type (a rename is reported under its
# new name; a symlink turned into a file, or the reverse, is a type change), so deleting a restricted file that is already committed
# stays possible. The list is asked for with --no-relative, so that a diff.relative=true in the configuration cannot hide a path
# outside the directory the check runs from (git 2.28 or newer; an older git makes the check say it cannot look, exit 2).
# A path is refused when, in any letter case (macOS does not tell Outputs/ from outputs/):
#   - one of its components is outputs, results, uploads, chat_sessions, logs or hpi_results_logs, at any depth. That is a directory
#     of that name, and also a file or a symlink: the cluster tree has symlinks called outputs and results;
#   - its name is refs.txt, hyps.txt, chexbert_labels.json or report_texts.txt, under any directory (analysis/ARCHIVE_MANIFEST.md:
#     refs.txt is verbatim MIMIC report text, and the others are what is generated from or labelled on it);
#   - it carries one of these extensions: ckpt parquet npy npz db pt pth safetensors h5 arrow feather (weights and arrays), jpg jpeg
#     dcm webp (images: MIMIC-CXR-JPG ships .jpg, the originals are DICOM), zip tar tgz gz 7z (archives), pkl pickle, and db-wal
#     db-shm sqlite sqlite3 (SQLite, whose side files hold the newest rows). PNG is not on the list: see the allow-list below.
#   The extension is refused when it ends the name or is followed by anything that is not a letter or a digit, so what a download
#   (.part), rsync (.last.ckpt.Xy12Ab), an editor (emb.npy~) or gzip (emb.npy.gz) leaves beside the file is refused too, and also when
#   it ends a directory name (a SQLite or parquet dataset is a directory of files without extensions). It is NOT refused when it only
#   starts a longer word: make.target.yaml, foo.ptx, notes.gzip.md. The four file names above follow the same rule, with an optional
#   leading dot (an rsync temporary of hyps.txt is .hyps.txt.Xy12Ab).
#   Exit 0: nothing refused. Exit 1: refused; the message is a header and the offending paths, never what the files hold.
#   Exit 2: git could not list the staged files (not a repository, say): the check cannot vouch for a commit it cannot see.
#
# Allow-list, exact and short, and checked first. PNGs are not refused: docs/chat_ui/evidence/ holds the screenshots of the UI over
# synthetic data, committed on purpose, and they are named below as well so that a wider list later cannot lock them out (nothing
# else in that directory is allowed). The three logs are Stage-0 language-model job logs (no MIMIC content) that were committed
# before this check existed: `git ls-files` shows them to be the only tracked files the rules hit, and editing them has to stay
# possible. Any other file under hpi_results_logs/ is refused. tests/test_restricted_files_hook.py stages every tracked path as an
# edit would and fails if one of them would be refused.
set -euo pipefail
shopt -s nocasematch

# The rules, as patterns for [[ =~ ]]. They are held in variables, never quoted at the match: a quoted pattern is literal in bash 3.2.
DIRECTORIES='outputs|results|uploads|chat_sessions|logs|hpi_results_logs'
DUA_NAMES='refs\.txt|hyps\.txt|chexbert_labels\.json|report_texts\.txt'
EXTENSIONS='ckpt|parquet|npy|npz|db|pt|pth|safetensors|h5|arrow|feather|jpg|jpeg|dcm|webp|zip|tar|tgz|gz|7z|pkl|pickle|db-wal|db-shm|sqlite|sqlite3'
DIR_RE='(^|/)('"${DIRECTORIES}"')(/|$)'
DUA_RE='(^|/)\.?('"${DUA_NAMES}"')([^A-Za-z0-9]|$)'
EXT_RE='\.('"${EXTENSIONS}"')([^A-Za-z0-9]|$)'

is_restricted() {
  case "$1" in
    docs/chat_ui/evidence/*.png) return 1 ;;
    hpi_results_logs/h100_stage0_150m_2341991.log | hpi_results_logs/monitor_stage0_2351222.log | hpi_results_logs/verify_handoff_2351231.log) return 1 ;;
  esac
  [[ "$1" =~ $DIR_RE ]] || [[ "$1" =~ $DUA_RE ]] || [[ "$1" =~ $EXT_RE ]]
}

# pipefail: a failing `git diff` fails the whole substitution, so a commit this cannot see is never waved through.
bad="$(git diff --cached --name-only -z --diff-filter=ACMRT --no-relative | while IFS= read -r -d '' path; do
  if is_restricted "${path}"; then printf '%s\n' "${path}"; fi
done)" || { echo "check_no_restricted_files: git could not list the staged files" >&2; exit 2; }

if [ -n "${bad}" ]; then
  echo "Refusing to commit restricted or binary artefacts:"
  printf '%s\n' "${bad}"
  exit 1
fi
