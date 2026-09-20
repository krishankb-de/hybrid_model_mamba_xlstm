#!/usr/bin/env bash
# Back-compat shim. The willi/A100 server is retired and the Python 3.9 harness with it
# (MAMBA3_PLAN_V2.md V4-B); the real harness is scripts/validate.sh. Kept because this name is
# written into older plans, commit messages and habits.
echo "note: validate_for_willi.sh is now a shim for scripts/validate.sh (willi retired, V4-B)." >&2
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/validate.sh" "$@"
