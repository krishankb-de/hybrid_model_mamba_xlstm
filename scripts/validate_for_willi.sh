#!/usr/bin/env bash
# validate_for_willi.sh — Pre-push validation harness mirroring willi A100 server env
# Usage:
#   bash scripts/validate_for_willi.sh                 # full (static + tests + dry-run)
#   bash scripts/validate_for_willi.sh --ci-static-only  # syntax/PEP checks only (CI)
#
# Requirements: conda on PATH, Python 3.9.23 available via conda.

set -uo pipefail   # -e intentionally omitted: each gate captures its own exit code

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_ROOT   # needed by inline Python subprocesses

# ── Bootstrap conda if not on PATH (non-interactive shells) ──────────────────
if ! command -v conda &>/dev/null; then
  for CONDA_INIT_SCRIPT in \
    /opt/anaconda3/etc/profile.d/conda.sh \
    /opt/miniconda3/etc/profile.d/conda.sh \
    "${HOME}/anaconda3/etc/profile.d/conda.sh" \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "${HOME}/opt/anaconda3/etc/profile.d/conda.sh"; do
    if [[ -f "$CONDA_INIT_SCRIPT" ]]; then
      # shellcheck source=/dev/null
      source "$CONDA_INIT_SCRIPT"
      break
    fi
  done
fi

CONDA_ENV="willi_parity"
PYTHON_VERSION="3.9.23"
CI_STATIC_ONLY=false

# ── Parse args ────────────────────────────────────────────────────────────────
for arg in "$@"; do
  case $arg in
    --ci-static-only) CI_STATIC_ONLY=true ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
PASS_TAG="[${GREEN}PASS${NC}]"
FAIL_TAG="[${RED}FAIL${NC}]"
WARN_TAG="[${YELLOW}WARN${NC}]"

declare -a SUMMARY_PASS=()
declare -a SUMMARY_FAIL=()

gate_pass() { SUMMARY_PASS+=("$1"); echo -e "${PASS_TAG} $1"; }
gate_fail() { SUMMARY_FAIL+=("$1"); echo -e "${FAIL_TAG} $1"; }
gate_warn() { echo -e "${WARN_TAG} $1"; }

# ── Summary (must be defined before first use) ────────────────────────────────
print_summary() {
  echo ""
  echo "════════════════════════ SUMMARY ════════════════════════"
  [[ ${#SUMMARY_PASS[@]} -gt 0 ]] && \
    for p in "${SUMMARY_PASS[@]}"; do echo -e "  ${PASS_TAG} $p"; done
  [[ ${#SUMMARY_FAIL[@]} -gt 0 ]] && \
    for f in "${SUMMARY_FAIL[@]}"; do echo -e "  ${FAIL_TAG} $f"; done
  echo "═════════════════════════════════════════════════════════"
  if [[ ${#SUMMARY_FAIL[@]} -gt 0 ]]; then
    echo -e "${RED}${#SUMMARY_FAIL[@]} gate(s) FAILED. Fix before pushing to willi.${NC}"
    exit 1
  else
    echo -e "${GREEN}All gates passed. Safe to push to a100_70m_baseline.${NC}"
    exit 0
  fi
}

# ── 1. Conda env setup ────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo "  Willi Parity Validator  (Python ${PYTHON_VERSION})"
echo "════════════════════════════════════════════════════════"

if ! command -v conda &>/dev/null; then
  echo -e "${FAIL_TAG} conda not found on PATH."
  echo "Install Miniconda/Anaconda and ensure 'conda' is available."
  exit 1
fi

if ! conda env list | grep -q "^${CONDA_ENV} \|^${CONDA_ENV}$"; then
  echo "Creating conda env '${CONDA_ENV}' with Python ${PYTHON_VERSION}..."
  conda create -y -n "${CONDA_ENV}" "python=${PYTHON_VERSION}" pip 2>&1 | tail -5
else
  echo "Reusing existing conda env '${CONDA_ENV}'."
fi

CONDA_PYTHON=$(conda run -n "${CONDA_ENV}" python -c "import sys; print(sys.executable)")
echo "Python: ${CONDA_PYTHON}"

ACTUAL_PY=$(conda run -n "${CONDA_ENV}" python -c "import sys; print('%d.%d.%d' % sys.version_info[:3])")
if [[ "$ACTUAL_PY" == "$PYTHON_VERSION" ]]; then
  gate_pass "Python version == ${PYTHON_VERSION}"
else
  gate_warn "Python version is ${ACTUAL_PY} (expected ${PYTHON_VERSION}). Willi runs 3.9.23 exactly."
fi

# ── 2. Install deps ───────────────────────────────────────────────────────────
# Filter Linux/CUDA-only packages that have no macOS wheel:
#   triton    — Linux only (Triton compiler)
#   flash-attn — requires CUDA build
#   apex       — requires CUDA build
#   ninja      — optional build tool, not needed for validation
#   py-spy     — optional profiler
#   torch-tb-profiler — optional
FILTERED_REQS="/tmp/willi_reqs_filtered_$$.txt"
grep -v -E "^\s*(triton|flash-attn|apex|ninja|py-spy|torch-tb-profiler)" \
  "${REPO_ROOT}/requirements.txt" > "${FILTERED_REQS}" 2>/dev/null || \
  cp "${REPO_ROOT}/requirements.txt" "${FILTERED_REQS}"

echo ""
echo "── Installing dependencies (Linux/CUDA-only packages skipped) ──"

# Install CPU torch explicitly first to avoid pulling the CUDA build
conda run -n "${CONDA_ENV}" pip install -q \
  "torch==2.1.2" --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -3

# Install project WITHOUT its own dependencies (setup.py lists triton which is Linux-only)
conda run -n "${CONDA_ENV}" pip install -q --no-deps -e "${REPO_ROOT}" 2>&1 | tail -3

# Install filtered requirements (triton/CUDA packages already removed)
if conda run -n "${CONDA_ENV}" pip install -q -r "${FILTERED_REQS}" 2>&1 | tail -5; then
  echo "Dependencies installed."
else
  echo "Some optional deps failed — continuing (CUDA-only packages skipped on macOS)."
fi
rm -f "${FILTERED_REQS}"

# ── 3. Gate 1: AST parse ──────────────────────────────────────────────────────
echo ""
echo "── Gate 1: AST parse (syntax correctness under Python ${ACTUAL_PY}) ──"
PARSE_ERRORS=0
while IFS= read -r -d '' f; do
  if ! conda run -n "${CONDA_ENV}" python -c "
import ast, sys
try:
    ast.parse(open('${f}').read())
except SyntaxError as e:
    print(f'SyntaxError in ${f}:{e.lineno}: {e.msg}')
    sys.exit(1)
" 2>&1; then
    PARSE_ERRORS=$((PARSE_ERRORS + 1))
  fi
done < <(find "${REPO_ROOT}/hybrid_xmamba" "${REPO_ROOT}/scripts" -name "*.py" -print0 2>/dev/null)

if [[ $PARSE_ERRORS -eq 0 ]]; then
  gate_pass "AST parse: all source files valid under Python ${ACTUAL_PY}"
else
  gate_fail "AST parse: ${PARSE_ERRORS} file(s) have syntax errors"
fi

# ── 4. Gate 2: PEP 604 union types (X | Y) ───────────────────────────────────
echo ""
echo "── Gate 2: PEP 604 type union guard (py3.9 compat) ──"
PEP604_HITS=$(grep -rn --include="*.py" \
  -E "(:\s+\w+(\[[\w,\s]+\])?\s*\|\s*\w+)|(->.*\w\s*\|\s*\w)" \
  "${REPO_ROOT}/hybrid_xmamba" "${REPO_ROOT}/scripts" 2>/dev/null || true)

if [[ -z "$PEP604_HITS" ]]; then
  gate_pass "PEP 604 guard: no X|Y union type hints found"
else
  echo "$PEP604_HITS"
  gate_fail "PEP 604 guard: found X|Y union syntax — use Optional[X] for Python 3.9"
fi

# ── 5. Gate 3: PEP 585 built-in generics (dict[...] etc.) ────────────────────
echo ""
echo "── Gate 3: PEP 585 built-in generics guard (py3.9 compat) ──"
PEP585_SCRIPT="/tmp/willi_pep585_$$.py"
cat > "${PEP585_SCRIPT}" << 'PYEOF'
import ast, sys, os
from pathlib import Path

REPO = os.environ["REPO_ROOT"]
BUILTIN_GENERICS = {"dict", "list", "tuple", "set", "type", "frozenset"}
hits = []

def check_file(path):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.annotation, ast.Subscript):
            val = node.annotation.value
            if isinstance(val, ast.Name) and val.id in BUILTIN_GENERICS:
                hits.append(f"{path}:{node.lineno}: {val.id}[...] — use typing.{val.id.capitalize()}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.returns and isinstance(node.returns, ast.Subscript):
                val = node.returns.value
                if isinstance(val, ast.Name) and val.id in BUILTIN_GENERICS:
                    hits.append(f"{path}:{node.lineno}: return annotation {val.id}[...]")
            for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
                if arg.annotation and isinstance(arg.annotation, ast.Subscript):
                    val = arg.annotation.value
                    if isinstance(val, ast.Name) and val.id in BUILTIN_GENERICS:
                        hits.append(f"{path}:{node.lineno}: arg annotation {val.id}[...]")

for root in ["hybrid_xmamba", "scripts"]:
    p = Path(REPO) / root
    if p.exists():
        for f in p.rglob("*.py"):
            check_file(f)

for h in hits:
    print(h)
sys.exit(1 if hits else 0)
PYEOF
PEP585_HITS=$(REPO_ROOT="${REPO_ROOT}" conda run -n "${CONDA_ENV}" \
  python "${PEP585_SCRIPT}" 2>&1) || true
rm -f "${PEP585_SCRIPT}"

if [[ -z "$PEP585_HITS" ]]; then
  gate_pass "PEP 585 guard: no bare built-in generic annotations found"
else
  echo "$PEP585_HITS"
  gate_fail "PEP 585 guard: use typing.Dict/List/Tuple etc. for Python 3.9"
fi

# ── Stop here in CI static-only mode ─────────────────────────────────────────
if $CI_STATIC_ONLY; then
  echo ""
  echo "═══════════ Static-only mode — stopping here ═══════════"
  print_summary
fi

# ── 6. Gate 4: Hydra config invariants ───────────────────────────────────────
echo ""
echo "── Gate 4: Hydra config invariants ──"
# Write Hydra check to a temp file (conda run doesn't forward stdin for heredocs)
HYDRA_SCRIPT="/tmp/willi_hydra_check_$$.py"
cat > "${HYDRA_SCRIPT}" << 'PYEOF'
import sys, os
sys.path.insert(0, os.environ["WILLI_REPO"])
os.chdir(os.environ["WILLI_REPO"])
model_name = os.environ["WILLI_MODEL"]
repo_root  = os.environ["WILLI_REPO"]
try:
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=repo_root + "/configs", version_base="1.3"):
        cfg = compose(config_name="config", overrides=[
            f"model={model_name}",
            "dataset=wikitext",
            "trainer=colab_single_gpu",
            "experiment_name=ci_check",
        ])
    m = cfg.model
    # Shared across all 70M models
    assert m.get("dim", 512) == 512, f"dim={m.get('dim')} != 512"
    assert m.get("num_layers", 8) == 8, f"num_layers={m.get('num_layers')} != 8"
    assert m.get("vocab_size", 50257) == 50257, "vocab_size mismatch"
    # hybrid_70m uses 1024 (MIG GPU); baselines use 2048 — both valid
    max_pos = m.get("max_position_embeddings")
    assert max_pos in (1024, 2048), f"unexpected max_pos={max_pos}"
    # layer_pattern cycles; no divisibility constraint
    pat = list(m.get("layer_pattern", []))
    assert len(pat) > 0, "layer_pattern is empty"
    print("OK")
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)
PYEOF

for MODEL in hybrid_70m mamba_70m_baseline xlstm_70m_baseline; do
  RESULT=$(WILLI_MODEL="${MODEL}" WILLI_REPO="${REPO_ROOT}" \
    conda run -n "${CONDA_ENV}" python "${HYDRA_SCRIPT}" 2>&1 || echo "HARNESS_ERROR")
  if [[ "$RESULT" == *"OK"* ]]; then
    gate_pass "Hydra: ${MODEL} config invariants"
  else
    gate_fail "Hydra: ${MODEL} — ${RESULT}"
  fi
done
rm -f "${HYDRA_SCRIPT}"

# ── 7. Gate 5: pytest (no cuda, no slow) ─────────────────────────────────────
echo ""
echo "── Gate 5: pytest (not cuda, not slow) ──"
if conda run -n "${CONDA_ENV}" \
    python -m pytest "${REPO_ROOT}/tests/" \
    -m "not cuda and not slow" \
    --tb=short -q 2>&1; then
  gate_pass "pytest: all non-CUDA tests passed"
else
  gate_fail "pytest: one or more tests failed"
fi

# ── 8. Gate 6: model import + CPU forward/backward smoke ─────────────────────
# train.py downloads datasets from HuggingFace (hangs on macOS/no-GPU).
# This gate instead directly imports the model, runs a forward+backward pass,
# and checks for finite loss and non-None gradients — no network required.
echo ""
echo "── Gate 6: model import + CPU forward/backward smoke (no dataset download) ──"
SMOKE_SCRIPT="/tmp/willi_smoke_$$.py"
cat > "${SMOKE_SCRIPT}" << 'PYEOF'
import sys, os
sys.path.insert(0, os.environ["WILLI_REPO"])
import torch
from hybrid_xmamba.models.configuration_hybrid import HybridConfig
from hybrid_xmamba.models.hybrid_lm import HybridLanguageModel

# Tiny model, CPU-safe dimensions. MAMBA3_PLAN.md M2-H: this used to be ["mamba","mamba","mlstm"],
# which meant the "every parameter receives a gradient" assertion below never saw a mamba3 or an
# sLSTM block. All four layer types are exercised now, so a mixer with a dangling parameter fails
# here rather than three days into a Stage-0 run. use_fast_path=False is deliberate: it is the
# path that carried an undetected copy of the selective-scan defect for the whole prior campaign.
cfg = HybridConfig(
    vocab_size=50257, dim=64, num_layers=4,
    layer_pattern=["mamba", "mamba3", "mlstm", "slstm"],
    max_position_embeddings=64,
    use_fast_path=False,
    use_tfla=False,
    mamba3_d_state=16,
    mamba3_head_dim=32,
)
model = HybridLanguageModel(cfg)
model.train()

batch_size, seq_len = 2, 32
input_ids = torch.randint(0, 50257, (batch_size, seq_len))
labels    = torch.randint(0, 50257, (batch_size, seq_len))

out = model(input_ids, labels=labels, return_dict=True)
assert out.loss is not None, "No loss returned"
assert torch.isfinite(out.loss), f"Loss not finite: {out.loss.item()}"
assert torch.isfinite(out.logits).all(), "Logits contain NaN/Inf"

out.loss.backward()

frozen = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
assert not frozen, f"Missing gradients: {frozen[:5]}"

print(f"OK  loss={out.loss.item():.4f}  logits={tuple(out.logits.shape)}")
PYEOF

if WILLI_REPO="${REPO_ROOT}" conda run -n "${CONDA_ENV}" \
    python "${SMOKE_SCRIPT}" 2>&1; then
  gate_pass "Model import + CPU forward/backward smoke passed"
else
  gate_fail "Model import + CPU forward/backward smoke FAILED"
fi
rm -f "${SMOKE_SCRIPT}"

print_summary
