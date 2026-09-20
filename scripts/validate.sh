#!/usr/bin/env bash
# validate.sh — pre-push validation harness.
#
#   bash scripts/validate.sh
#
# MAMBA3_PLAN_V2.md V4-B. This replaces `validate_for_willi.sh`, which bootstrapped a conda env
# pinned to Python 3.9.23 to mirror the retired willi/A100 server and ran three static gates
# (AST parse, PEP 604 unions, PEP 585 generics) before the real ones. willi is gone; the cluster
# runs Python 3.11 and this laptop 3.14, so a 3.9 interpreter gated nothing anyone deploys to and
# cost ~7 minutes of env bootstrap per run. The three syntax rules survive as ordinary tests in
# tests/test_willi_parity.py, so the hygiene is kept and only the interpreter requirement is gone.
#
# Gates, in order:
#   1. Hydra config invariants for the 70M models
#   2. pytest -m "not cuda and not slow"
#   3. model import + CPU forward/backward over all five mixer types, every parameter gets a grad
#
# The interpreter is the first of: $PYTHON, ./.venv (cluster), ./venv (laptop), python3.

set -uo pipefail   # -e intentionally omitted: each gate captures its own exit code

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_ROOT

if [[ -z "${PYTHON:-}" ]]; then
  for CANDIDATE in "${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/venv/bin/python" "$(command -v python3 || true)"; do
    if [[ -x "${CANDIDATE}" ]]; then PYTHON="${CANDIDATE}"; break; fi
  done
fi
if [[ -z "${PYTHON:-}" || ! -x "${PYTHON}" ]]; then
  echo "No usable interpreter. Set PYTHON=/path/to/python and re-run." >&2
  exit 1
fi

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
PASS_TAG="[${GREEN}PASS${NC}]"; FAIL_TAG="[${RED}FAIL${NC}]"; WARN_TAG="[${YELLOW}WARN${NC}]"
declare -a SUMMARY_PASS=() SUMMARY_FAIL=()
gate_pass() { SUMMARY_PASS+=("$1"); echo -e "${PASS_TAG} $1"; }
gate_fail() { SUMMARY_FAIL+=("$1"); echo -e "${FAIL_TAG} $1"; }

print_summary() {
  echo ""
  echo "════════════════════════ SUMMARY ════════════════════════"
  for p in "${SUMMARY_PASS[@]:-}"; do [[ -n "$p" ]] && echo -e "  ${PASS_TAG} $p"; done
  for f in "${SUMMARY_FAIL[@]:-}"; do [[ -n "$f" ]] && echo -e "  ${FAIL_TAG} $f"; done
  echo "═════════════════════════════════════════════════════════"
  if [[ ${#SUMMARY_FAIL[@]} -gt 0 ]]; then
    echo -e "${RED}${#SUMMARY_FAIL[@]} gate(s) failed. Do not push.${NC}"; return 1
  fi
  echo -e "${GREEN}All gates passed.${NC}"; return 0
}

echo "════════════════════════════════════════════════════════"
echo " Pre-push validation"
echo " repo:   ${REPO_ROOT}"
echo " python: ${PYTHON}  ($(${PYTHON} -c 'import sys; print(sys.version.split()[0])'))"
echo "════════════════════════════════════════════════════════"

# ── Gate 1: Hydra config invariants ──────────────────────────────────────────
echo ""
echo "── Gate 1: Hydra config invariants ──"
HYDRA_SCRIPT="$(mktemp -t hydra_check.XXXXXX).py"
cat > "${HYDRA_SCRIPT}" << 'PYEOF'
import os, sys
sys.path.insert(0, os.environ["REPO_ROOT"])
os.chdir(os.environ["REPO_ROOT"])
model_name = os.environ["CHECK_MODEL"]
try:
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=os.environ["REPO_ROOT"] + "/configs", version_base="1.3"):
        cfg = compose(config_name="config", overrides=[
            f"model={model_name}", "dataset=wikitext",
            "trainer=colab_single_gpu", "experiment_name=ci_check",
        ])
    m = cfg.model
    assert m.get("dim", 512) == 512, f"dim={m.get('dim')} != 512"
    assert m.get("num_layers", 8) == 8, f"num_layers={m.get('num_layers')} != 8"
    assert m.get("vocab_size", 50257) == 50257, "vocab_size mismatch"
    assert m.get("max_position_embeddings") in (1024, 2048), "unexpected max_position_embeddings"
    assert len(list(m.get("layer_pattern", []))) > 0, "layer_pattern is empty"
    print("OK")
except Exception as exc:
    print(f"FAIL: {exc}")
    sys.exit(1)
PYEOF
for MODEL in hybrid_70m mamba_70m_baseline xlstm_70m_baseline; do
  RESULT=$(CHECK_MODEL="${MODEL}" "${PYTHON}" "${HYDRA_SCRIPT}" 2>&1 || echo "HARNESS_ERROR")
  if [[ "$RESULT" == *"OK"* ]]; then gate_pass "Hydra: ${MODEL} config invariants"
  else gate_fail "Hydra: ${MODEL} — ${RESULT}"; fi
done
rm -f "${HYDRA_SCRIPT}"

# ── Gate 2: pytest ───────────────────────────────────────────────────────────
echo ""
echo "── Gate 2: pytest (not cuda, not slow) ──"
if "${PYTHON}" -m pytest "${REPO_ROOT}/tests/" -m "not cuda and not slow" --tb=short -q 2>&1; then
  gate_pass "pytest: all non-CUDA tests passed"
else
  gate_fail "pytest: one or more tests failed"
fi

# ── Gate 3: model import + CPU forward/backward smoke ────────────────────────
# Deliberately covers all five mixer types with use_fast_path=False and use_tfla=False: those are
# the paths that each carried an undetected defect at some point in this project, and the
# every-parameter-gets-a-gradient assertion is what catches a dangling mixer parameter here
# rather than three days into a Stage-0 run.
echo ""
echo "── Gate 3: model import + CPU forward/backward smoke ──"
SMOKE_SCRIPT="$(mktemp -t smoke_check.XXXXXX).py"
cat > "${SMOKE_SCRIPT}" << 'PYEOF'
import os, sys
sys.path.insert(0, os.environ["REPO_ROOT"])
import torch
from hybrid_xmamba.models.configuration_hybrid import HybridConfig
from hybrid_xmamba.models.hybrid_lm import HybridLanguageModel

cfg = HybridConfig(
    vocab_size=50257, dim=64, num_layers=5,
    layer_pattern=["mamba", "mamba3", "mlstm", "slstm", "attention"],
    max_position_embeddings=64, use_fast_path=False, use_tfla=False,
    mamba3_d_state=16, mamba3_head_dim=32,
)
model = HybridLanguageModel(cfg)
model.train()
input_ids = torch.randint(0, 50257, (2, 32))
labels = torch.randint(0, 50257, (2, 32))
out = model(input_ids, labels=labels, return_dict=True)
assert out.loss is not None and torch.isfinite(out.loss), "loss missing or not finite"
assert torch.isfinite(out.logits).all(), "logits contain NaN/Inf"
out.loss.backward()
frozen = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
assert not frozen, f"missing gradients: {frozen[:5]}"
print(f"OK  loss={out.loss.item():.4f}  logits={tuple(out.logits.shape)}")
PYEOF
if "${PYTHON}" "${SMOKE_SCRIPT}" 2>&1; then
  gate_pass "Model import + CPU forward/backward smoke passed"
else
  gate_fail "Model import + CPU forward/backward smoke FAILED"
fi
rm -f "${SMOKE_SCRIPT}"

print_summary
