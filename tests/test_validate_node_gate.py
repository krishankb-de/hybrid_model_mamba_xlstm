"""CHAT_UI_PLAN.md P4-B fix round 1 (M4): Gate 2b of scripts/validate.sh, the browser modules' node tests.

app/static/*.js carries no package.json (everything there is served publicly), so node has to detect ES-module syntax by
itself: it does from 22.7, and from 20.19 on the 20 line. The gate therefore runs `node --test` only on such a node, and a
node that is missing, too old or without tests to run skips it with a warning that also lands in the SUMMARY, so a skipped
gate is never mistaken for a passed one. The gate block is cut out of the script and run under a fake `node`, which tests
the version rule on every version without installing any.
"""
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT_LINES = (REPO / "scripts" / "validate.sh").read_text(encoding="utf-8").splitlines()
ANSI = re.compile(r"\x1b\[[0-9;]*m")
TIMEOUT_FLAG = "--test-timeout=30000"
FAKE_NODE = """#!/bin/sh
if [ "$1" = "--version" ]; then echo "$FAKE_NODE_VERSION"; exit 0; fi
echo "$@" > "$FAKE_NODE_ARGS"
exit "$FAKE_NODE_EXIT"
"""


def _cut(first, last, include_last=False):
    """The script's lines from the one starting with `first` to the one equal to (or starting with) `last`."""
    start = next(i for i, line in enumerate(SCRIPT_LINES) if line.startswith(first))
    stop = next(i for i in range(start + 1, len(SCRIPT_LINES)) if SCRIPT_LINES[i].startswith(last))
    return "\n".join(SCRIPT_LINES[start:stop + (1 if include_last else 0)])


HELPERS = _cut("RED=", 'echo "════')               # the colours, gate_pass/gate_skip/gate_fail and print_summary
GATE = _cut("# ── Gate 2b", "fi", include_last=True)   # up to its closing fi, the first one in column 0


def _run_gate(tmp_path, version="v25.9.0", node=True, tests=True, node_exit=0):
    """Gate 2b and the SUMMARY under a fake node on an otherwise empty PATH -> (exit status, output, node's arguments)."""
    bin_dir, repo = tmp_path / "bin", tmp_path / "repo"
    bin_dir.mkdir(parents=True)
    (repo / "tests" / "frontend").mkdir(parents=True)
    if tests:
        (repo / "tests" / "frontend" / "x.test.mjs").write_text("// stands in for a test file\n")
    if node:
        (bin_dir / "node").write_text(FAKE_NODE)
        (bin_dir / "node").chmod(0o755)
    args_file = tmp_path / "node_args.txt"
    script = "set -uo pipefail\nREPO_ROOT={}\n{}\n{}\nprint_summary\n".format(shlex.quote(str(repo)), HELPERS, GATE)
    done = subprocess.run([shutil.which("bash"), "-c", script], capture_output=True, text=True,
                          env={"PATH": str(bin_dir), "FAKE_NODE_VERSION": version, "FAKE_NODE_EXIT": str(node_exit),
                               "FAKE_NODE_ARGS": str(args_file)})
    args = args_file.read_text().split() if args_file.exists() else None
    return done.returncode, ANSI.sub("", done.stdout + done.stderr), args, repo


def _summary(output):
    return output.split("SUMMARY")[1]


@pytest.mark.parametrize("version", ["v20.19.0", "v20.20.2", "v22.7.0", "v22.12.0", "v23.0.0", "v24.1.0", "v25.9.0"])
def test_gate_runs_the_tests_with_a_timeout_on_a_node_that_detects_modules(tmp_path, version):
    status, output, args, repo = _run_gate(tmp_path, version)
    assert args == ["--test", TIMEOUT_FLAG, str(repo / "tests" / "frontend" / "x.test.mjs")]
    assert "[PASS] node: frontend tests passed" in _summary(output)
    assert status == 0 and "All gates passed." in output


@pytest.mark.parametrize("version", ["v16.20.2", "v18.19.0", "v20.18.3", "v21.7.3", "v22.6.0", "v22.0.0", "garbage", "v25", ""])
def test_gate_skips_an_older_node_with_a_warning_in_the_summary(tmp_path, version):
    status, output, args, _ = _run_gate(tmp_path, version)
    assert args is None   # node --test never ran
    skipped = [line for line in _summary(output).splitlines() if "[WARN]" in line]
    assert len(skipped) == 1 and "node: frontend tests skipped" in skipped[0] and "22.7" in skipped[0], skipped
    assert version in skipped[0]   # it says which node it found
    assert status == 0 and "[PASS] node" not in output   # a skip is neither a pass nor a failure


def test_gate_skips_without_node_and_without_tests_and_says_why(tmp_path):
    status, output, args, _ = _run_gate(tmp_path / "a", node=False)
    assert args is None and status == 0
    assert "[WARN] node: frontend tests skipped (node not found)" in _summary(output)
    status, output, args, _ = _run_gate(tmp_path / "b", tests=False)
    assert args is None and status == 0
    assert "[WARN] node: frontend tests skipped (no tests/frontend/*.test.mjs)" in _summary(output)


def test_gate_fails_when_the_node_tests_fail(tmp_path):
    status, output, args, _ = _run_gate(tmp_path, node_exit=1)
    assert args is not None and "[FAIL] node: frontend tests failed" in _summary(output)
    assert status == 1 and "gate(s) failed" in output
