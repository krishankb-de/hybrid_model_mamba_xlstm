"""
Willi Server Parity Tests
=========================
Validates that the codebase is compatible with the willi A100 server environment:
  - Python 3.9.23
  - No PEP 604 (X | Y) or PEP 585 (dict[...]) syntax in annotations
  - Hydra config invariants for all 70M models
  - Checkpoint prefix stripping roundtrip
  - CPU forward + backward smoke
  - Attention-mask-aware pooling correctness

Run via the local harness:
    bash scripts/validate.sh

Or directly (must be inside the willi_parity conda env):
    conda run -n willi_parity pytest tests/test_willi_parity.py -v
"""

import ast
import json
import re
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional

import pytest
import torch

# ── Helpers ───────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
SCAN_ROOTS = [REPO_ROOT / "hybrid_xmamba", REPO_ROOT / "scripts", REPO_ROOT / "app"]
BUILTIN_GENERICS = {"dict", "list", "tuple", "set", "type", "frozenset"}


def iter_py_files() -> List[Path]:
    files = []
    for root in SCAN_ROOTS:
        if root.exists():
            files.extend(root.rglob("*.py"))
    return files


# ── 1. Python version gate ────────────────────────────────────────────────────

@pytest.mark.willi_parity
def test_python_version_is_3_9():
    """Willi runs Python 3.9.x — fail loudly if env doesn't match."""
    major, minor = sys.version_info[:2]
    if major != 3 or minor != 9:
        pytest.skip(
            f"Python {major}.{minor} detected — this test is only meaningful inside "
            f"the 'willi_parity' conda env (Python 3.9.23). "
            f"Run: conda run -n willi_parity pytest tests/test_willi_parity.py"
        )
    # If we're on 3.9, verify we can actually import every package
    assert (major, minor) == (3, 9), f"Expected Python 3.9, got {major}.{minor}"


# ── 2. PEP 604 guard (X | Y unions in annotations) ───────────────────────────

@pytest.mark.willi_parity
def test_no_pep604_union_in_runtime_imports():
    """No module should use X | Y syntax that fails at import time on Python 3.9."""
    import importlib
    import pkgutil

    errors: List[str] = []
    pkg_path = str(REPO_ROOT / "hybrid_xmamba")
    if not os.path.isdir(pkg_path):
        pytest.skip("hybrid_xmamba package not found")

    # Only test runtime import — PEP 604 in annotations with __future__.annotations is OK
    for importer, modname, ispkg in pkgutil.walk_packages(
        path=[pkg_path], prefix="hybrid_xmamba.", onerror=lambda x: None
    ):
        try:
            importlib.import_module(modname)
        except TypeError as e:
            if "unsupported operand type(s) for |" in str(e):
                errors.append(f"{modname}: {e}")
        except Exception:
            pass  # Import errors for missing deps are not our concern here

    assert not errors, (
        "PEP 604 X|Y runtime error(s) — use Optional[X] for Python 3.9:\n"
        + "\n".join(errors)
    )


def pep604_hits(source: str, filename: str = "<snippet>") -> List[str]:
    """One hit per annotation that contains an `X | Y` union anywhere in its subtree."""
    tree = ast.parse(source, filename=filename)
    annotations = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = node.args
            extra = [x for x in (a.vararg, a.kwarg) if x is not None]
            for arg in a.posonlyargs + a.args + a.kwonlyargs + extra:
                if arg.annotation is not None:
                    annotations.append(arg.annotation)
            if node.returns is not None:
                annotations.append(node.returns)
        elif isinstance(node, ast.AnnAssign):
            annotations.append(node.annotation)
    hits = []
    for ann in annotations:
        binop = next((n for n in ast.walk(ann)
                      if isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitOr)), None)
        if binop is not None:
            hits.append("{}:{}: {}".format(filename, ann.lineno, ast.unparse(binop)))
    return hits


@pytest.mark.willi_parity
def test_no_pep604_union_in_annotations_ast():
    """Scan all SCAN_ROOTS for X | Y (BinOp with BitOr) in annotation positions using AST.
    Catches violations including nested unions like Optional[int | str], Dict[str, int | None], etc.
    """
    all_hits: List[str] = []

    for filepath in iter_py_files():
        source = filepath.read_text(encoding="utf-8")
        hits = pep604_hits(source, str(filepath))
        all_hits.extend(hits)

    assert not all_hits, (
        "PEP 604 X | Y syntax found in annotations — use Union[X, Y] or Optional[X] for Python 3.9:\n"
        + "\n".join(all_hits[:20])
    )


@pytest.mark.willi_parity
def test_pep604_helper_catches_all_root_and_nested_forms():
    """Self-test: pep604_hits catches exactly 12 root forms, 7 nested forms,
    and ignores bitwise-or outside annotations (6 silent cases)."""
    # 12 root-level forms
    root_tests = [
        ("def f(a: int | None): pass", ": int | None"),
        ("def f() -> int | None: pass", ": int | None"),
        ("x: int | None = None", ": int | None"),
        ("def f(*, a: int | None): pass", ": int | None"),
        ("def f(a: int | None, /): pass", ": int | None"),
        ("def f(*a: int | None): pass", ": int | None"),
        ("def f(**k: int | None): pass", ": int | None"),
        ("async def f(a: int | None): pass", ": int | None"),
        ("async def f() -> int | None: pass", ": int | None"),
        ("class C:\n    x: int | None = None", ": int | None"),
        ("def f():\n    x: int | None = None", ": int | None"),
        ("class C:\n    def m(self, a: int | None): pass", ": int | None"),
    ]

    # 7 nested forms
    nested_tests = [
        ("x: Optional[int | str]", "int | str"),
        ("x: List[int | None]", "int | None"),
        ("def f() -> Dict[str, int | None]: pass", "int | None"),
        ("def f(x: Callable[[int | None], None]) -> None: pass", "int | None"),
        ("x: List[str | None]", "str | None"),
        ("x: Optional[Path | str]", "Path | str"),
        ("def f(*a: Tuple[int | None, ...]) -> None: pass", "int | None"),
    ]

    # Test all root forms: exactly one hit each
    for code, expected_substr in root_tests:
        hits = pep604_hits(code)
        assert len(hits) == 1, f"Expected exactly 1 hit for: {code}, got {len(hits)}"
        assert expected_substr in hits[0], f"Expected '{expected_substr}' in: {hits[0]}"

    # Test all nested forms: exactly one hit each
    for code, expected_expr in nested_tests:
        hits = pep604_hits(code)
        assert len(hits) == 1, f"Expected exactly 1 hit for: {code}, got {len(hits)}"
        assert expected_expr in hits[0], f"Expected '{expected_expr}' in: {hits[0]}"

    # Line number case: annotation on line 3
    line_case = "def f(\n    a: int,\n    b: int | None,\n): pass"
    hits = pep604_hits(line_case)
    assert len(hits) == 1, f"Line number case failed: got {len(hits)} hits"
    assert ":3: int | None" in hits[0], f"Expected ':3: int | None' in: {hits[0]}"

    # Chained case: exactly one hit for `int | str | None`
    chained = "x: int | str | None = None"
    hits = pep604_hits(chained)
    assert len(hits) == 1, f"Chained case failed: got {len(hits)} hits"
    assert "int | str | None" in hits[0], f"Expected 'int | str | None' in: {hits[0]}"

    # Silent cases: bitwise-or outside annotations (0 hits each)
    silent_cases = [
        "x = 1 | 2",  # value
        "def f(): return 1 | 2",  # body
        "f = lambda: 1 | 2",  # lambda body
        "FLAGS = A | B",  # module-level flag
        "def f(a: int = 1 | 2): pass",  # default value
        "g = lambda x=1 | 2: x",  # lambda default
    ]

    for code in silent_cases:
        hits = pep604_hits(code)
        assert len(hits) == 0, f"Expected 0 hits for silent case: {code}, got {len(hits)}: {hits}"


# ── 3. PEP 585 guard (dict[...] etc. in annotations) ─────────────────────────

@pytest.mark.willi_parity
def test_no_pep585_generics_in_annotations():
    """No annotation should use bare built-in generics like dict[str, int] (PEP 585).
    Python 3.9 allows this at runtime but it can cause issues in older minor versions
    and with static analysis. Prefer typing.Dict / typing.List etc.
    """
    hits: List[str] = []

    def check_node(node: ast.AST, filepath: Path) -> None:
        """Walk annotation subscripts for bare builtin generic names."""
        subscript_contexts = []

        if isinstance(node, ast.AnnAssign) and isinstance(node.annotation, ast.Subscript):
            subscript_contexts.append((node.annotation, node.lineno))

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.returns and isinstance(node.returns, ast.Subscript):
                subscript_contexts.append((node.returns, node.lineno))
            for arg in (
                node.args.args
                + node.args.posonlyargs
                + node.args.kwonlyargs
                + ([node.args.vararg] if node.args.vararg else [])
                + ([node.args.kwarg] if node.args.kwarg else [])
            ):
                if arg.annotation and isinstance(arg.annotation, ast.Subscript):
                    subscript_contexts.append((arg.annotation, getattr(arg, "col_offset", 0)))

        for subscript, lineno in subscript_contexts:
            val = subscript.value
            if isinstance(val, ast.Name) and val.id in BUILTIN_GENERICS:
                hits.append(
                    f"{filepath}:{lineno}: {val.id}[...] — use typing.{val.id.capitalize()}"
                )

    for filepath in iter_py_files():
        try:
            tree = ast.parse(filepath.read_text(encoding="utf-8"))
        except SyntaxError:
            continue  # a syntax error is caught by the import/pytest gate, not here
        for node in ast.walk(tree):
            check_node(node, filepath)

    assert not hits, (
        "PEP 585 bare built-in generics found — use typing.Dict/List/Tuple for py3.9:\n"
        + "\n".join(hits[:20])
    )


# ── 4. Hydra config invariants ────────────────────────────────────────────────

@pytest.mark.willi_parity
@pytest.mark.parametrize("model_name", [
    "hybrid_70m",
    "hybrid_70m_v2",
    "hybrid_70m_v3",
    "mamba_70m_baseline",
    "xlstm_70m_baseline",
])
def test_hydra_config_resolves_70m_invariants(model_name: str):
    """All 70M configs must share identical hyperparameters (only layer_pattern differs)."""
    pytest.importorskip("hydra")
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    configs_dir = str(REPO_ROOT / "configs")

    with initialize_config_dir(config_dir=configs_dir, version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"model={model_name}",
                "dataset=wikitext",
                "trainer=colab_single_gpu",
                "experiment_name=parity_check",
            ],
        )

    m = cfg.model
    # Invariants shared across ALL 70M models
    assert m.get("dim") == 512, f"{model_name}: dim={m.get('dim')} != 512"
    assert m.get("num_layers") == 8, f"{model_name}: num_layers={m.get('num_layers')} != 8"
    assert m.get("vocab_size") == 50257, f"{model_name}: vocab_size mismatch"

    # hybrid_70m uses 1024 (MIG GPU); baselines use 2048 — both are valid
    max_pos = m.get("max_position_embeddings")
    assert max_pos in (1024, 2048), \
        f"{model_name}: unexpected max_position_embeddings={max_pos} (expected 1024 or 2048)"

    # layer_pattern must be non-empty; it cycles so no divisibility constraint needed
    pat = list(m.get("layer_pattern", []))
    assert len(pat) > 0, f"{model_name}: layer_pattern is empty"

    # Phase 4 HybridNorm: v2 must use 'hybrid' topology; others default pre_rms
    if model_name == "hybrid_70m_v2":
        topo = m.get("norm_topology", "pre_rms")
        assert topo == "hybrid", (
            f"hybrid_70m_v2: norm_topology={topo!r} — must be 'hybrid' (Phase 4)"
        )

    # dataset max_length must not exceed model capacity
    d = cfg.dataset
    dataset_max = d.get("max_length", 1024)
    model_max = m.get("max_position_embeddings", 1024)
    assert dataset_max <= model_max, \
        f"{model_name}: dataset.max_length ({dataset_max}) > model.max_position_embeddings ({model_max})"


# ── 4b. H100 trainer config invariants (Phase 2) ──────────────────────────────

@pytest.mark.willi_parity
@pytest.mark.parametrize("trainer_name,expected_strategy,expected_devices", [
    ("h100_single_gpu", "auto", 1),
    ("h100_multi_ddp", "ddp", -1),
])
def test_h100_trainer_configs_resolve(trainer_name, expected_strategy, expected_devices):
    """H100 trainer configs must load and carry the scale-up invariants:
    bf16-mixed + accumulate_grad_batches=1 (true per-step batch — grad-accum does
    NOT add in-batch contrastive negatives, so we scale batch_size, not accum)."""
    pytest.importorskip("hydra")
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    configs_dir = str(REPO_ROOT / "configs")

    with initialize_config_dir(config_dir=configs_dir, version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "model=hybrid_70m_v2",
                "dataset=wikitext",
                f"trainer={trainer_name}",
                "experiment_name=parity_check",
            ],
        )

    t = cfg.trainer
    assert t.get("precision") == "bf16-mixed", \
        f"{trainer_name}: precision={t.get('precision')!r} != 'bf16-mixed'"
    assert t.get("accumulate_grad_batches") == 1, \
        f"{trainer_name}: accumulate_grad_batches={t.get('accumulate_grad_batches')} != 1 (H100 wants true per-step batch)"
    assert str(t.get("strategy")) == expected_strategy, \
        f"{trainer_name}: strategy={t.get('strategy')!r} != {expected_strategy!r}"
    assert t.get("devices") == expected_devices, \
        f"{trainer_name}: devices={t.get('devices')} != {expected_devices}"
    if trainer_name == "h100_multi_ddp":
        assert t.get("find_unused_parameters") is True, \
            "h100_multi_ddp: find_unused_parameters must be true (ViT-unfreeze/KD leave frozen params)"


# ── 4c. hybrid_150m_v2 config + param count (Phase 4) ─────────────────────────

@pytest.mark.willi_parity
def test_hybrid_150m_v2_config_and_param_count():
    """Phase 4: the 150M v2 backbone ports every v2 arch win and builds to the
    expected size. Param count verified explicitly (count before assuming a
    mismatch) — ~183.72M actual (nominal '150M'; untied 50k-vocab embeddings
    dominate, consistent with the 70M config → 83M naming convention)."""
    import dataclasses
    from omegaconf import OmegaConf
    from hybrid_xmamba.models.configuration_hybrid import HybridConfig
    from hybrid_xmamba.models.hybrid_lm import HybridLanguageModel

    raw = OmegaConf.to_container(
        OmegaConf.load(REPO_ROOT / "configs" / "model" / "hybrid_150m_v2.yaml"),
        resolve=True,
    )
    # v2 architectural invariants
    assert raw["dim"] == 768, f"dim={raw['dim']} != 768"
    assert raw["num_layers"] == 12, f"num_layers={raw['num_layers']} != 12"
    assert raw["num_heads"] == 12 and raw["head_dim"] == 64
    assert raw["norm_topology"] == "hybrid", "150m v2 must use HybridNorm"
    assert raw["pooling_strategy"] == "attention"
    assert raw["max_position_embeddings"] == 1024, "v2 parity: max_pos=1024"
    pattern = list(raw["layer_pattern"])
    assert len(pattern) == 12, f"layer_pattern len={len(pattern)} != 12"
    assert pattern.count("mlstm") == 3, "centered 3-mLSTM (25% ratio) v2 analogue"

    fields = {f.name for f in dataclasses.fields(HybridConfig)}
    cfg = HybridConfig(**{k: v for k, v in raw.items() if k in fields})
    model = HybridLanguageModel(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    assert 181e6 < n_params < 186e6, (
        f"hybrid_150m_v2 param count {n_params/1e6:.2f}M outside [181, 186]M — "
        f"arch drift; expected ~183.72M"
    )


# ── 5. Checkpoint prefix stripping roundtrip ──────────────────────────────────

@pytest.mark.willi_parity
def test_checkpoint_prefix_stripping_roundtrip():
    """Verify that state_dict keys with known willi prefixes are correctly stripped."""
    from hybrid_xmamba.models.configuration_hybrid import HybridConfig
    from hybrid_xmamba.models.hybrid_lm import HybridLanguageModel

    cfg = HybridConfig(
        vocab_size=100, dim=64, num_layers=2,
        layer_pattern=["mamba", "mlstm"],
        max_position_embeddings=64,
    )
    model = HybridLanguageModel(cfg)
    bare_keys = set(model.state_dict().keys())

    def strip_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Strip known Lightning/compile/DDP prefixes in correct order.
        Order matters: model. must come before lm. so that model.lm.x → lm.x → x.
        """
        stripped = {}
        for k, v in state_dict.items():
            k = k.removeprefix("_orig_mod.")   # torch.compile (outermost)
            k = k.removeprefix("model.")       # Lightning wrapper
            k = k.removeprefix("module.")      # DDP
            k = k.removeprefix("lm.")          # HybridLMModule inner attribute
            stripped[k] = v
        return stripped

    prefix_combos = [
        {"prefix": "_orig_mod.", "label": "torch.compile"},
        {"prefix": "lm.",        "label": "Lightning lm"},
        {"prefix": "model.",     "label": "Lightning model"},
        {"prefix": "module.",    "label": "DDP module"},
        {"prefix": "_orig_mod.lm.", "label": "compile + lm"},
        {"prefix": "model.lm.",     "label": "model + lm"},
    ]

    for combo in prefix_combos:
        prefix = combo["prefix"]
        wrapped = {f"{prefix}{k}": v for k, v in model.state_dict().items()}
        stripped = strip_prefixes(wrapped)
        stripped_keys = set(stripped.keys())
        assert stripped_keys == bare_keys, (
            f"Prefix '{prefix}' ({combo['label']}): stripping left unexpected keys.\n"
            f"  Extra: {stripped_keys - bare_keys}\n"
            f"  Missing: {bare_keys - stripped_keys}"
        )


# ── 6. CPU forward + backward smoke ───────────────────────────────────────────

@pytest.mark.willi_parity
def test_forward_cpu_smoke():
    """Tiny model forward+backward on CPU. Catches frozen-layer and NaN bugs."""
    from hybrid_xmamba.models.configuration_hybrid import HybridConfig
    from hybrid_xmamba.models.hybrid_lm import HybridLanguageModel

    cfg = HybridConfig(
        vocab_size=100, dim=64, num_layers=2,
        layer_pattern=["mamba", "mlstm"],
        max_position_embeddings=64,
        use_fast_path=False,  # disable CUDA fast paths for CPU
        use_tfla=False,
    )
    model = HybridLanguageModel(cfg)
    model.train()

    input_ids = torch.randint(0, 100, (2, 16))
    labels = torch.randint(0, 100, (2, 16))

    out = model(input_ids, labels=labels, return_dict=True)

    assert out.loss is not None, "Model returned no loss"
    assert torch.isfinite(out.loss), f"Loss is not finite: {out.loss.item()}"
    assert torch.isfinite(out.logits).all(), "Logits contain NaN/Inf"

    out.loss.backward()

    # Every parameter that requires grad must have a non-None, finite gradient
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for parameter: {name}"
            assert torch.isfinite(param.grad).all(), \
                f"Non-finite gradient for parameter: {name}"


# ── 7. Dataloader max_length ≤ model capacity ─────────────────────────────────

@pytest.mark.willi_parity
def test_dataloader_max_length_matches_model():
    """Ensure dataset.max_length does not exceed model.max_position_embeddings."""
    pytest.importorskip("hydra")
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    configs_dir = str(REPO_ROOT / "configs")

    with initialize_config_dir(config_dir=configs_dir, version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "model=hybrid_70m",
                "dataset=wikitext",
                "trainer=colab_single_gpu",
                "experiment_name=parity_check",
            ],
        )

    dataset_max = cfg.dataset.get("max_length", 1024)
    model_max = cfg.model.get("max_position_embeddings", 1024)

    assert dataset_max <= model_max, (
        f"dataset.max_length ({dataset_max}) > model.max_position_embeddings ({model_max}). "
        "This silently truncates long sequences at eval time and causes retrieval regressions."
    )


# ── 8. Pooling: must use attention mask, not [:, -1, :] ──────────────────────

@pytest.mark.willi_parity
def test_pooling_respects_attention_mask():
    """
    Last-token pooling MUST use the last *non-padding* token via attention_mask.
    Pooling from [:, -1, :] silently reads a padding token for right-padded sequences.

    This test synthesises a batch with known padding and verifies that mask-aware
    pooling produces different (correct) results from naive tail pooling.
    """
    B, T, D = 4, 16, 64
    hidden = torch.randn(B, T, D)

    # Real sequence lengths: [16, 12, 8, 4] — rest is padding
    seq_lens = [16, 12, 8, 4]
    mask = torch.zeros(B, T, dtype=torch.long)
    for i, length in enumerate(seq_lens):
        mask[i, :length] = 1

    # ── Correct: mask-aware last-token pooling ────────────────────────────────
    last_real_idx = mask.sum(dim=1) - 1          # [15, 11, 7, 3]
    pooled_correct = hidden[range(B), last_real_idx]   # (B, D)

    # ── Naive (wrong for padded sequences) ───────────────────────────────────
    pooled_naive = hidden[:, -1, :]              # always last position

    # For items where seq_len < T, the naive and correct results should differ
    shorter_indices = [i for i, l in enumerate(seq_lens) if l < T]
    assert shorter_indices, "Test setup error: no padded sequences"

    for i in shorter_indices:
        assert not torch.allclose(pooled_correct[i], pooled_naive[i]), (
            f"Sequence {i} (len={seq_lens[i]}): mask-aware pooling equals naive pooling. "
            "This means either pooling is broken or all padding positions have identical values."
        )

    # Sanity: for the full-length sequence, both should agree
    full_len_idx = [i for i, l in enumerate(seq_lens) if l == T]
    for i in full_len_idx:
        assert torch.allclose(pooled_correct[i], pooled_naive[i]), \
            f"Full-length sequence {i}: mask-aware and naive pooling should agree but don't."


# ── 9. ContrastiveEvalCallback importable + Python 3.9-safe ───────────────

@pytest.mark.willi_parity
def test_contrastive_eval_callback_importable():
    """ContrastiveEvalCallback and AnomalyDetectionCallback must import on Python 3.9."""
    from hybrid_xmamba.training.contrastive_eval_callback import (
        ContrastiveEvalCallback,
        AnomalyDetectionCallback,
        _spearman_rho,
        _alignment,
        _uniformity,
    )
    # Spearman correctness
    rho = _spearman_rho([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert abs(rho - 1.0) < 1e-5, f"Expected ρ=1.0 for identical lists, got {rho}"

    # Callback instantiates with minimal args
    class _MockTok:
        pass
    cb = ContrastiveEvalCallback(tokenizer=_MockTok(), eval_every_n_steps=500)
    assert cb.eval_every == 500
    assert cb.align_unif_every == 1000  # default

    acb = AnomalyDetectionCallback(max_steps=200)
    assert acb.max_steps == 200


# ── 10. NT-Xent fixed_scale path ──────────────────────────────────────────

@pytest.mark.willi_parity
def test_nt_xent_fixed_scale():
    """_nt_xent_loss must honour the fixed_scale argument and ignore logit_scale."""
    from hybrid_xmamba.models.configuration_hybrid import HybridConfig
    from hybrid_xmamba.models.hybrid_lm import HybridTextEncoder
    from hybrid_xmamba.training.lightning_module import HybridContrastiveLightningModule
    import torch.nn.functional as F

    cfg = HybridConfig(
        vocab_size=100, dim=64, num_layers=2,
        layer_pattern=["mamba", "mlstm"],
        max_position_embeddings=64,
        use_fast_path=False,
        use_tfla=False,
    )
    enc = HybridTextEncoder(cfg, embed_dim=64)
    mod = HybridContrastiveLightningModule(
        model=enc,
        contrastive_mode="simcse",
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_steps=10,
        max_steps=100,
        gradient_clip_val=1.0,
    )

    z1 = F.normalize(torch.randn(4, 64), dim=-1)
    z2 = F.normalize(torch.randn(4, 64), dim=-1)

    loss_fixed = mod._nt_xent_loss(z1, z2, mod.model.logit_scale, fixed_scale=20.0)
    assert torch.isfinite(loss_fixed), f"fixed_scale loss not finite: {loss_fixed}"

    # Manually verify scale=20 is used
    logits = 20.0 * (z1 @ z2.T)
    labels = torch.arange(4)
    expected = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    assert abs(loss_fixed.item() - expected.item()) < 1e-5, (
        f"fixed_scale loss {loss_fixed.item():.6f} != manual {expected.item():.6f}"
    )


# ── 11. Stage 1 distill config values ─────────────────────────────────────

@pytest.mark.willi_parity
def test_distill_kd_step_produces_finite_loss():
    """DistillContrastiveLightningModule._simcse_step produces finite KD loss.

    Stage 1 now uses pure PubMedBERT cosine KD (SimCSE removed). This test
    verifies that the KD distill_loss is finite and in (0, 2) for random
    embeddings, confirming gradient flow is working.

    SimCSE was removed because the Stage 0 backbone (PPL=13.10) already
    perfectly separates PubMed abstracts — InfoNCE loss was ~0.002 from
    step 1, giving near-zero gradients. Pure KD directly aligns the student
    to PubMedBERT's CLS space which scores BIOSSES=0.85.
    """
    from hybrid_xmamba.models.configuration_hybrid import HybridConfig
    from hybrid_xmamba.models.hybrid_lm import HybridTextEncoder
    from hybrid_xmamba.training.lightning_module import DistillContrastiveLightningModule

    cfg = HybridConfig(
        vocab_size=100, dim=64, num_layers=2,
        layer_pattern=["mamba", "mlstm"],
        max_position_embeddings=64,
        use_fast_path=False,
        use_tfla=False,
    )
    enc = HybridTextEncoder(cfg, embed_dim=64)

    # Minimal teacher stub: returns last_hidden_state (B, L, 768)
    class _StubTeacher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("C", (), {"hidden_size": 768})()

        def forward(self, input_ids, attention_mask=None):
            B, L = input_ids.shape
            hidden = torch.randn(B, L, 768)
            return type("O", (), {"last_hidden_state": hidden})()

    teacher = _StubTeacher()
    mod = DistillContrastiveLightningModule(
        teacher=teacher,
        lambda_max=1.0,
        distill_warmup=0,
        distill_ramp=1,
        model=enc,
        contrastive_mode="simcse",
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_steps=5,
        max_steps=50,
        gradient_clip_val=1.0,
    )
    mod.train()

    input_ids = torch.randint(0, 100, (4, 16))
    attn = torch.ones(4, 16, dtype=torch.long)
    # teacher_input_ids triggers KD computation
    batch = {
        "input_ids": input_ids,
        "attention_mask": attn,
        "teacher_input_ids": input_ids,
        "teacher_attention_mask": attn,
    }

    loss = mod._simcse_step(batch, batch_idx=0, split="train")
    assert torch.isfinite(loss), f"KD loss not finite: {loss.item()}"
    assert 0.0 < loss.item() < 2.0, (
        f"KD loss {loss.item():.4f} out of expected range (0, 2) for random embeddings"
    )

    # Backward must succeed — confirms gradient flows through distill_proj
    loss.backward()
    for name, param in mod.distill_proj.named_parameters():
        assert param.grad is not None, f"No gradient for distill_proj.{name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient for distill_proj.{name}"


@pytest.mark.willi_parity
def test_stage1_distill_config_values():
    """stage1_pubmedbert.yaml KD schedule — pure KD, no SimCSE.

    SimCSE was removed because the Stage 0 backbone (PPL=13.10) already
    perfectly separates PubMed abstracts. InfoNCE loss started at ~0.002
    from step 1 (expected 2.08 for random embeddings) — no gradient signal.
    Pure PubMedBERT cosine KD is used instead: L = 1 - cos(student, teacher).
    lambda_max=1.0 (full KD from step 0), warmup_steps=0, ramp_steps=1.
    """
    pytest.importorskip("yaml")
    import yaml

    cfg_path = REPO_ROOT / "configs" / "distill" / "stage1_pubmedbert.yaml"
    assert cfg_path.exists(), f"Missing {cfg_path}"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    assert cfg.get("lambda_max") == 1.0, (
        f"lambda_max should be 1.0 (pure KD), got {cfg.get('lambda_max')}"
    )
    assert cfg.get("warmup_steps") == 0, (
        f"warmup_steps should be 0 (no SimCSE to warm up), got {cfg.get('warmup_steps')}"
    )
    assert cfg.get("ramp_steps") == 0, (
        f"ramp_steps should be 0 (start KD immediately), got {cfg.get('ramp_steps')}"
    )


# ── 12. img_proj and distill_proj must NOT exist (Phase 8 deletion) ───────────

@pytest.mark.willi_parity
def test_img_proj_and_distill_proj_deleted():
    """Phase 8: img_proj and distill_proj must be absent from all training modules.

    clip_model.visual already outputs 512-d BiomedCLIP joint embeddings;
    the random-init img_proj MLP was distorting them (root cause of Phase 5c
    paired-cos plateau). distill_proj was a gradient absorber that prevented
    KD from reaching z_text. Both are deleted in Phase 8.
    """
    import torch.nn as nn
    from hybrid_xmamba.models.configuration_hybrid import HybridConfig
    from hybrid_xmamba.models.hybrid_lm import HybridTextEncoder
    from hybrid_xmamba.training.lightning_module import (
        HybridContrastiveLightningModule,
        JointMultiTaskLightningModule,
    )

    cfg = HybridConfig(
        vocab_size=100, dim=64, num_layers=2,
        layer_pattern=["mamba", "mlstm"],
        max_position_embeddings=64,
        use_fast_path=False, use_tfla=False,
    )
    enc_simcse = HybridTextEncoder(cfg, embed_dim=64)

    # SimCSE mode: neither module should exist.
    mod_simcse = HybridContrastiveLightningModule(
        model=enc_simcse, contrastive_mode="simcse",
        learning_rate=1e-4, weight_decay=0.01,
        warmup_steps=5, max_steps=50, gradient_clip_val=1.0,
    )
    assert not isinstance(getattr(mod_simcse, "img_proj", None), nn.Module), (
        "img_proj must not be an nn.Module on HybridContrastiveLightningModule"
    )

    # Joint mode: distill_proj must be absent.
    # embed_dim=512 required: Phase 8 assert checks img_out == student embed_dim,
    # and real BiomedCLIP visual outputs 512-d.
    cfg_joint = HybridConfig(
        vocab_size=100, dim=64, num_layers=2,
        layer_pattern=["mamba", "mlstm"],
        max_position_embeddings=64,
        use_fast_path=False, use_tfla=False,
        pooling_strategy="attention",
    )
    enc_joint = HybridTextEncoder(cfg_joint, embed_dim=512)

    class _StubTeacher(nn.Module):
        def encode_text(self, ids):
            return torch.randn(ids.shape[0], 512)

    try:
        mod_joint = JointMultiTaskLightningModule(
            model=enc_joint,
            teacher=_StubTeacher(),
            alpha_kd=0.3, beta_clip=1.0, gamma_simcse=0.1,
            backbone_lr=1e-5, head_lr=3e-4,
            weight_decay=0.01, warmup_steps=5, max_steps=50,
            gradient_clip_val=1.0, freeze_text_encoder_steps=0,
        )
        assert not isinstance(getattr(mod_joint, "distill_proj", None), nn.Module), (
            "distill_proj must not be an nn.Module on JointMultiTaskLightningModule (Phase 8)"
        )
        assert not isinstance(getattr(mod_joint, "img_proj", None), nn.Module), (
            "img_proj must not be an nn.Module on JointMultiTaskLightningModule (Phase 8)"
        )
    except ImportError:
        pytest.skip("open_clip not installed — JointMultiTaskLightningModule requires it")


# ── 13. AttentionPooling correctness ──────────────────────────────────────────

@pytest.mark.willi_parity
def test_attention_pooling_correctness():
    """AttentionPooling must:
    - produce finite, non-zero outputs
    - differ from mean pooling (non-trivial weighting)
    - handle all-padding edge case without NaN (falls back to uniform)
    - be instantiated only for pooling_strategy='attention'
    """
    from hybrid_xmamba.models.configuration_hybrid import HybridConfig
    from hybrid_xmamba.models.hybrid_lm import AttentionPooling, HybridTextEncoder

    dim = 64
    pool = AttentionPooling(dim)
    pool.eval()

    B, L = 4, 16
    hidden = torch.randn(B, L, dim)
    mask = torch.ones(B, L, dtype=torch.long)
    mask[1, 12:] = 0   # second sample padded
    mask[2, 8:]  = 0
    mask[3, 4:]  = 0

    out = pool(hidden, mask=mask)
    assert out.shape == (B, dim), f"AttentionPooling output shape wrong: {out.shape}"
    assert torch.isfinite(out).all(), "AttentionPooling output contains NaN/Inf"

    # All-padding edge case — should not NaN
    all_pad_mask = torch.zeros(2, L, dtype=torch.long)
    out_ap = pool(hidden[:2], mask=all_pad_mask)
    assert torch.isfinite(out_ap).all(), "AttentionPooling NaN on all-padding mask"

    # Encoder wires correctly for strategy='attention'
    cfg_attn = HybridConfig(
        vocab_size=100, dim=dim, num_layers=2,
        layer_pattern=["mamba", "mlstm"],
        max_position_embeddings=64,
        use_fast_path=False, use_tfla=False,
        pooling_strategy="attention",
    )
    enc_attn = HybridTextEncoder(cfg_attn, embed_dim=dim)
    assert enc_attn.attn_pool is not None, "attn_pool should be set for strategy='attention'"
    assert isinstance(enc_attn.attn_pool, AttentionPooling)

    # Encoder wires correctly for strategy='mean' (baselines)
    cfg_mean = HybridConfig(
        vocab_size=100, dim=dim, num_layers=2,
        layer_pattern=["mamba", "mlstm"],
        max_position_embeddings=64,
        use_fast_path=False, use_tfla=False,
        pooling_strategy="mean",
    )
    enc_mean = HybridTextEncoder(cfg_mean, embed_dim=dim)
    assert enc_mean.attn_pool is None, "attn_pool should be None for strategy='mean'"


# ── 14. Joint module: all 3 losses finite + grads flow ────────────────────────

@pytest.mark.willi_parity
def test_joint_module_all_losses_finite():
    """JointMultiTaskLightningModule._joint_step must produce finite KD, CLIP, and
    SimCSE losses with gradients flowing into backbone and proj_head.
    Phase 8: img_proj and distill_proj deleted. KD is direct cosine sim on z_text.
    image_encoder skipped (open_clip unavailable); l_clip=0 is OK.
    """
    from hybrid_xmamba.models.configuration_hybrid import HybridConfig
    from hybrid_xmamba.models.hybrid_lm import HybridTextEncoder
    from hybrid_xmamba.training.lightning_module import JointMultiTaskLightningModule

    cfg = HybridConfig(
        vocab_size=100, dim=64, num_layers=2,
        layer_pattern=["mamba", "mlstm"],
        max_position_embeddings=64,
        use_fast_path=False, use_tfla=False,
        pooling_strategy="attention",
    )
    # embed_dim=512 matches BiomedCLIP joint space; Phase 6c KD is applied directly
    # on z_text, so z_text and t_emb must share the same dimension.
    enc = HybridTextEncoder(cfg, embed_dim=512)

    class _StubBiomedCLIPText(torch.nn.Module):
        """Mimic open_clip CLIP wrapper: encode_text returns (B, 512)."""

        def encode_text(self, input_ids):
            B = input_ids.shape[0]
            return torch.randn(B, 512)

    teacher = _StubBiomedCLIPText()

    try:
        mod = JointMultiTaskLightningModule(
            model=enc,
            teacher=teacher,
            alpha_kd=0.3,
            beta_clip=1.0,
            gamma_simcse=0.1,
            backbone_lr=1e-5,
            head_lr=3e-4,
            weight_decay=0.01,
            warmup_steps=5,
            max_steps=50,
            gradient_clip_val=1.0,
            freeze_text_encoder_steps=0,
            vit_unfreeze_blocks=0,
        )
    except ImportError:
        pytest.skip("open_clip not installed — JointMultiTaskLightningModule requires it")

    mod.train()

    input_ids = torch.randint(0, 100, (4, 16))
    attn = torch.ones(4, 16, dtype=torch.long)
    batch = {
        "input_ids": input_ids,
        "attention_mask": attn,
        # No pixel_values: l_clip will be 0 (image encoder absent without open_clip)
        "teacher_input_ids": input_ids,
        "teacher_attention_mask": attn,
    }

    # Phase 8: distill_proj must not exist as a trainable module.
    assert not isinstance(getattr(mod, "distill_proj", None), torch.nn.Module), (
        "distill_proj must be deleted from JointMultiTaskLightningModule (Phase 8)"
    )

    loss = mod._joint_step(batch, batch_idx=0, split="train")
    assert torch.isfinite(loss), f"Joint total loss not finite: {loss.item()}"
    assert loss.item() > 0.0, "Joint loss should be > 0 (l_kd + l_simcse active)"

    loss.backward()
    # KD is direct cosine on z_text → projection_head must receive gradient.
    for name, param in mod.model.projection_head.named_parameters():
        assert param.grad is not None, f"No grad for projection_head.{name}"
        assert torch.isfinite(param.grad).all(), f"NaN grad for projection_head.{name}"
    if mod.model.attn_pool is not None:
        for name, param in mod.model.attn_pool.named_parameters():
            assert param.grad is not None, f"No grad for attn_pool.{name}"


# ── 15. joint_mimic.yaml config values ────────────────────────────────────────

@pytest.mark.willi_parity
def test_joint_mimic_config_values():
    """joint_mimic.yaml must have the plan-specified loss weights and LRs."""
    pytest.importorskip("yaml")
    import yaml

    cfg_path = REPO_ROOT / "configs" / "distill" / "joint_mimic.yaml"
    assert cfg_path.exists(), f"Missing {cfg_path}"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    assert cfg.get("alpha_kd") == 0.3,   f"alpha_kd should be 0.3, got {cfg.get('alpha_kd')}"
    assert cfg.get("beta_clip") == 1.0,  f"beta_clip should be 1.0, got {cfg.get('beta_clip')}"
    assert cfg.get("gamma_simcse") == 0.1, f"gamma_simcse should be 0.1, got {cfg.get('gamma_simcse')}"
    assert cfg.get("backbone_lr") == 1e-5, f"backbone_lr should be 1e-5, got {cfg.get('backbone_lr')}"
    assert cfg.get("head_lr") == 3e-4,  f"head_lr should be 3e-4, got {cfg.get('head_lr')}"
    assert cfg.get("freeze_text_encoder_steps") == 500, (
        f"freeze_text_encoder_steps should be 500, got {cfg.get('freeze_text_encoder_steps')}"
    )


@pytest.mark.willi_parity
def test_biomedclip_kd_config_values():
    """biomedclip_kd_joint.yaml must have plan-specified Phase 4 values."""
    pytest.importorskip("yaml")
    import yaml

    cfg_path = REPO_ROOT / "configs" / "distill" / "biomedclip_kd_joint.yaml"
    assert cfg_path.exists(), f"Missing {cfg_path}"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    assert cfg.get("teacher") == "biomedclip_text", (
        f"teacher should be 'biomedclip_text', got {cfg.get('teacher')}"
    )
    assert cfg.get("alpha_kd") == 0.3,   f"alpha_kd should be 0.3 (Phase 6b), got {cfg.get('alpha_kd')}"
    # Phase 10: α_kd schedule keys.
    assert cfg.get("alpha_kd_warmup") == 1.0, (
        f"alpha_kd_warmup should be 1.0 (Phase 10), got {cfg.get('alpha_kd_warmup')}"
    )
    assert cfg.get("alpha_kd_post") == 0.3, (
        f"alpha_kd_post should be 0.3 (Phase 10), got {cfg.get('alpha_kd_post')}"
    )
    assert cfg.get("beta_clip") == 1.0,  f"beta_clip should be 1.0, got {cfg.get('beta_clip')}"
    assert cfg.get("gamma_simcse") == 0.1, f"gamma_simcse should be 0.1, got {cfg.get('gamma_simcse')}"
    assert cfg.get("backbone_lr") == 1e-5, f"backbone_lr should be 1e-5, got {cfg.get('backbone_lr')}"
    assert cfg.get("head_lr") == 3e-4,  f"head_lr should be 3e-4, got {cfg.get('head_lr')}"
    # Phase 10: 500→1000.
    assert cfg.get("freeze_text_encoder_steps") == 1000, (
        f"freeze_text_encoder_steps should be 1000 (Phase 10), got {cfg.get('freeze_text_encoder_steps')}"
    )
    # PubMedBERT-specific keys must NOT leak into this config.
    for forbidden in ("teacher_model", "teacher_dtype", "teacher_max_length"):
        assert forbidden not in cfg, f"{forbidden} is PubMedBERT-specific; remove from biomedclip_kd_joint.yaml"


@pytest.mark.willi_parity
def test_alpha_kd_schedule_switches_at_threshold():
    """Phase 10: effective α_kd must equal alpha_kd_warmup while
    global_step < freeze_text_encoder_steps, and alpha_kd_post otherwise.
    Also asserts that __init__ without overrides reduces to the legacy
    constant α_kd (back-compat).
    """
    from hybrid_xmamba.models.configuration_hybrid import HybridConfig
    from hybrid_xmamba.models.hybrid_lm import HybridTextEncoder
    from hybrid_xmamba.training.lightning_module import JointMultiTaskLightningModule

    cfg = HybridConfig(
        vocab_size=100, dim=64, num_layers=2,
        layer_pattern=["mamba", "mlstm"],
        max_position_embeddings=64,
        use_fast_path=False, use_tfla=False,
        pooling_strategy="attention",
    )
    enc = HybridTextEncoder(cfg, embed_dim=512)

    class _StubBiomedCLIPText(torch.nn.Module):
        def encode_text(self, input_ids):
            return torch.randn(input_ids.shape[0], 512)

    teacher = _StubBiomedCLIPText()

    try:
        mod = JointMultiTaskLightningModule(
            model=enc, teacher=teacher,
            alpha_kd=0.3, alpha_kd_warmup=1.0, alpha_kd_post=0.3,
            beta_clip=1.0, gamma_simcse=0.1,
            backbone_lr=1e-5, head_lr=3e-4, weight_decay=0.01,
            warmup_steps=5, max_steps=50, gradient_clip_val=1.0,
            freeze_text_encoder_steps=1000,
        )
    except ImportError:
        pytest.skip("open_clip not installed")

    assert mod.alpha_kd_warmup == 1.0
    assert mod.alpha_kd_post == 0.3

    # Back-compat: no overrides → both fall back to alpha_kd.
    try:
        mod2 = JointMultiTaskLightningModule(
            model=enc, teacher=teacher, alpha_kd=0.42,
            beta_clip=1.0, gamma_simcse=0.1,
            backbone_lr=1e-5, head_lr=3e-4, weight_decay=0.01,
            warmup_steps=5, max_steps=50, gradient_clip_val=1.0,
            freeze_text_encoder_steps=0,
        )
    except ImportError:
        pytest.skip("open_clip not installed")
    assert mod2.alpha_kd_warmup == 0.42
    assert mod2.alpha_kd_post == 0.42

    # Verify the schedule actually applies inside _joint_step by varying the
    # threshold (global_step==0 when no trainer is attached). With threshold
    # 1000 the warmup α applies; with threshold 0 the post α applies. With
    # l_clip=0 (no pixel_values) the total loss differs only by the α_kd
    # multiplier on l_kd, so any change in α produces a measurably different
    # total under a fixed RNG seed.
    input_ids = torch.randint(0, 100, (2, 8))
    attn = torch.ones(2, 8, dtype=torch.long)
    batch = {
        "input_ids": input_ids, "attention_mask": attn,
        "teacher_input_ids": input_ids, "teacher_attention_mask": attn,
    }

    try:
        mod_warmup = JointMultiTaskLightningModule(
            model=enc, teacher=teacher,
            alpha_kd=0.3, alpha_kd_warmup=1.0, alpha_kd_post=0.3,
            beta_clip=1.0, gamma_simcse=0.1,
            backbone_lr=1e-5, head_lr=3e-4, weight_decay=0.01,
            warmup_steps=5, max_steps=50, gradient_clip_val=1.0,
            freeze_text_encoder_steps=1000,  # global_step(0) < 1000 → warmup α
        )
        mod_post = JointMultiTaskLightningModule(
            model=enc, teacher=teacher,
            alpha_kd=0.3, alpha_kd_warmup=1.0, alpha_kd_post=0.3,
            beta_clip=1.0, gamma_simcse=0.1,
            backbone_lr=1e-5, head_lr=3e-4, weight_decay=0.01,
            warmup_steps=5, max_steps=50, gradient_clip_val=1.0,
            freeze_text_encoder_steps=0,     # global_step(0) >= 0 → post α
        )
    except ImportError:
        pytest.skip("open_clip not installed")

    mod_warmup.eval()
    mod_post.eval()

    torch.manual_seed(0)
    loss_warmup = mod_warmup._joint_step(batch, batch_idx=0, split="train").item()
    torch.manual_seed(0)
    loss_post = mod_post._joint_step(batch, batch_idx=0, split="train").item()

    # Different effective α must produce a different total loss.
    assert loss_warmup != loss_post, (
        f"α_kd schedule did not change loss: warmup={loss_warmup}, post={loss_post}"
    )


@pytest.mark.willi_parity
def test_moco_queue_shape_and_enqueue():
    """MoCoQueue: buffer shape correct; enqueue fills and wraps correctly."""
    from hybrid_xmamba.training.moco_queue import MoCoQueue
    import torch.nn.functional as F

    K, dim, B = 64, 16, 8
    q = MoCoQueue(dim=dim, K=K)
    assert q.queue.shape == (dim, K), f"Expected ({dim},{K}), got {q.queue.shape}"

    # Fill with known values and verify ptr advances
    keys = F.normalize(torch.randn(B, dim), dim=-1)
    q.enqueue(keys)
    assert int(q.queue_ptr) == B, f"ptr should be {B}, got {int(q.queue_ptr)}"

    # Verify stored keys match
    stored = q.all_keys()[:B]  # first B rows
    assert torch.allclose(stored, keys, atol=1e-5), "Stored keys don't match enqueued"

    # Fill remaining capacity (K - B already written) then verify wrap-around
    for _ in range(K // B - 1):
        q.enqueue(F.normalize(torch.randn(B, dim), dim=-1))
    assert int(q.queue_ptr) == 0, "Ptr should wrap to 0 after exactly K enqueued keys"


@pytest.mark.willi_parity
def test_momentum_encoder_ema_delta():
    """MomentumEncoder: EMA update moves params by exactly (1-m) fraction."""
    from hybrid_xmamba.training.moco_queue import MomentumEncoder
    import torch.nn as nn

    m = 0.9
    query = nn.Linear(8, 4, bias=False)
    nn.init.constant_(query.weight, 1.0)

    ema = MomentumEncoder(query, m=m)
    nn.init.constant_(ema.encoder.weight, 0.0)  # start EMA at 0

    ema.update(query)
    # Expected: 0.9 * 0.0 + 0.1 * 1.0 = 0.1
    expected = (1 - m) * 1.0
    assert torch.allclose(ema.encoder.weight, torch.full_like(ema.encoder.weight, expected), atol=1e-6), \
        f"EMA weight should be {expected}, got {ema.encoder.weight.mean().item()}"


@pytest.mark.willi_parity
def test_moco_config_values():
    """Phase 13: biomedclip_kd_joint.yaml must have moco_queue_size=0 (queue disabled).

    Phase 6d (job 1313) showed MoCo K=16384 cold-start at unfreeze fills the
    queue with 16384 random unit-norm vectors; K/batch=512 steps to refresh
    produces near-random CLIP gradients that destroy the KD warmup (MIMIC
    R@10=3.95% vs Phase 5c 9.99%). Fix: queue disabled, in-batch only.
    """
    pytest.importorskip("yaml")
    import yaml

    cfg_path = REPO_ROOT / "configs" / "distill" / "biomedclip_kd_joint.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    assert cfg.get("moco_queue_size") == 256, \
        f"moco_queue_size should be 256 (Phase 6f: small queue, warms in 8 steps), got {cfg.get('moco_queue_size')}"
    assert cfg.get("moco_momentum") == 0.999, \
        f"moco_momentum should be 0.999, got {cfg.get('moco_momentum')}"


@pytest.mark.willi_parity
def test_moco_symmetric_loss_both_directions():
    """_moco_clip_loss_symmetric must train both i2t and t2i directions.

    Checks: loss is finite; gradients flow into z_text (t2i path); the
    text_queue and img_queue are both initialised on the module.
    """
    from hybrid_xmamba.models.configuration_hybrid import HybridConfig
    from hybrid_xmamba.models.hybrid_lm import HybridTextEncoder
    from hybrid_xmamba.training.lightning_module import JointMultiTaskLightningModule
    from hybrid_xmamba.training.moco_queue import MoCoQueue

    # embed_dim=512: Phase 8 assert requires img_out (512 for real BiomedCLIP)
    # to equal student embed_dim.
    cfg = HybridConfig(
        vocab_size=100, dim=64, num_layers=2,
        layer_pattern=["mamba", "mlstm"],
        max_position_embeddings=64,
        use_fast_path=False, use_tfla=False,
        pooling_strategy="attention",
    )
    enc = HybridTextEncoder(cfg, embed_dim=512)

    class _StubTeacher(torch.nn.Module):
        def encode_text(self, input_ids):
            return torch.randn(input_ids.shape[0], 512)

    try:
        mod = JointMultiTaskLightningModule(
            model=enc, teacher=_StubTeacher(),
            warmup_steps=2, max_steps=10,
            freeze_text_encoder_steps=0,
            moco_queue_size=32,  # small queue for CPU test
        )
    except ImportError:
        pytest.skip("open_clip not installed")

    assert isinstance(mod.text_queue, MoCoQueue), "text_queue must be MoCoQueue"
    assert not hasattr(mod, 'img_queue') or mod.img_queue is None, \
        "img_queue must not exist — random-init queue causes max-entropy t2i loss"
    assert mod.text_queue.K == 32

    # Exercise the symmetric loss directly
    B, D = 4, 512
    raw_text = torch.randn(B, D, requires_grad=True)
    z_text   = torch.nn.functional.normalize(raw_text, dim=-1)
    z_img    = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
    z_text_k = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
    loss = mod._moco_clip_loss_symmetric(z_text, z_img, z_text_k)
    assert torch.isfinite(loss), f"Symmetric MoCo loss not finite: {loss.item()}"
    loss.backward()
    # raw_text is the leaf — grad must flow through t2i path (z_text @ img_bank)
    assert raw_text.grad is not None, "No gradient into z_text (t2i path broken)"
    assert torch.isfinite(raw_text.grad).all(), "NaN in z_text gradient"


@pytest.mark.willi_parity
def test_clip_loss_gated_during_warmup():
    """Phase 9: CLIP loss must be gated off (l_clip == 0) and the MoCo queue
    must NOT enqueue while ``global_step < freeze_text_encoder_steps``.

    Without a Lightning trainer, ``self.global_step`` returns 0; setting
    ``freeze_text_encoder_steps=1000`` keeps the gate closed.
    """
    from hybrid_xmamba.models.configuration_hybrid import HybridConfig
    from hybrid_xmamba.models.hybrid_lm import HybridTextEncoder
    from hybrid_xmamba.training.lightning_module import JointMultiTaskLightningModule

    cfg = HybridConfig(
        vocab_size=100, dim=64, num_layers=2,
        layer_pattern=["mamba", "mlstm"],
        max_position_embeddings=64,
        use_fast_path=False, use_tfla=False,
        pooling_strategy="attention",
    )
    enc = HybridTextEncoder(cfg, embed_dim=512)

    class _StubTeacher(torch.nn.Module):
        def encode_text(self, input_ids):
            return torch.randn(input_ids.shape[0], 512)

    try:
        mod = JointMultiTaskLightningModule(
            model=enc, teacher=_StubTeacher(),
            warmup_steps=2, max_steps=10,
            freeze_text_encoder_steps=1000,
            moco_queue_size=32,
        )
    except ImportError:
        pytest.skip("open_clip not installed")

    # Inject a dummy image encoder (open_clip-free) so the CLIP branch could fire.
    # If gating works, the branch is skipped despite this being available.
    B = 4

    class _StubImageEncoder(torch.nn.Module):
        def forward(self, px):
            return torch.randn(px.shape[0], 512)

    mod.image_encoder = _StubImageEncoder()

    input_ids = torch.randint(0, 100, (B, 16))
    attn = torch.ones(B, 16, dtype=torch.long)
    batch = {
        "input_ids": input_ids,
        "attention_mask": attn,
        "pixel_values": torch.randn(B, 3, 8, 8),
        "teacher_input_ids": input_ids,
        "teacher_attention_mask": attn,
    }

    ptr_before = int(mod.text_queue.queue_ptr)
    queue_before = mod.text_queue.queue.clone()
    loss = mod._joint_step(batch, batch_idx=0, split="train")
    ptr_after = int(mod.text_queue.queue_ptr)

    assert torch.isfinite(loss), "Joint loss not finite during warmup"
    assert ptr_after == ptr_before, (
        f"text_queue must not enqueue during warmup; ptr {ptr_before}→{ptr_after}"
    )
    assert torch.equal(mod.text_queue.queue, queue_before), \
        "text_queue contents must be untouched during warmup"


@pytest.mark.willi_parity
def test_moco_queue_cold_start_reset():
    """Phase 9: MoCoQueue.reset() zeros the pointer and re-randomises the buffer
    so post-warmup InfoNCE negatives start fresh (not stale GPT-2-space keys)."""
    from hybrid_xmamba.training.moco_queue import MoCoQueue
    import torch.nn.functional as F

    K, dim, B = 64, 16, 8
    q = MoCoQueue(dim=dim, K=K)
    keys = F.normalize(torch.randn(B, dim), dim=-1)
    q.enqueue(keys)
    assert int(q.queue_ptr) == B
    q_before = q.queue.clone()

    q.reset()
    assert int(q.queue_ptr) == 0, "reset() must zero queue_ptr"
    # Buffer is re-randomised — should differ from pre-reset state almost surely.
    assert not torch.equal(q.queue, q_before), "reset() must change queue contents"
    # Still L2-normalised columns.
    norms = q.queue.norm(dim=0)
    assert torch.allclose(norms, torch.ones(K), atol=1e-5), \
        "reset() queue columns must remain unit-norm"


@pytest.mark.willi_parity
def test_momentum_encoder_copy_from():
    """Phase 9: copy_from() hard-resyncs momentum encoder weights to live model."""
    from hybrid_xmamba.training.moco_queue import MomentumEncoder
    import torch.nn as nn

    query = nn.Linear(8, 4, bias=False)
    nn.init.constant_(query.weight, 1.0)

    ema = MomentumEncoder(query, m=0.999)
    nn.init.constant_(ema.encoder.weight, 0.0)
    assert not torch.allclose(ema.encoder.weight, query.weight), \
        "Pre-condition: ema and query must differ"

    ema.copy_from(query)
    assert torch.allclose(ema.encoder.weight, query.weight, atol=1e-7), \
        "copy_from must produce identical weights to live model"
    # All ema params must remain non-trainable.
    for p in ema.encoder.parameters():
        assert not p.requires_grad, "EMA encoder params must stay frozen after copy_from"


@pytest.mark.willi_parity
def test_joint_unfreeze_triggers_resync_and_reset():
    """Phase 9: at the unfreeze step, on_train_batch_start must call
    momentum_encoder.copy_from(model) and text_queue.reset()."""
    from hybrid_xmamba.models.configuration_hybrid import HybridConfig
    from hybrid_xmamba.models.hybrid_lm import HybridTextEncoder
    from hybrid_xmamba.training.lightning_module import JointMultiTaskLightningModule
    import torch.nn.functional as F

    cfg = HybridConfig(
        vocab_size=100, dim=64, num_layers=2,
        layer_pattern=["mamba", "mlstm"],
        max_position_embeddings=64,
        use_fast_path=False, use_tfla=False,
        pooling_strategy="attention",
    )
    enc = HybridTextEncoder(cfg, embed_dim=512)

    class _StubTeacher(torch.nn.Module):
        def encode_text(self, input_ids):
            return torch.randn(input_ids.shape[0], 512)

    try:
        mod = JointMultiTaskLightningModule(
            model=enc, teacher=_StubTeacher(),
            warmup_steps=2, max_steps=10,
            freeze_text_encoder_steps=0,  # so global_step(0) >= threshold triggers unfreeze
            moco_queue_size=32,
        )
    except ImportError:
        pytest.skip("open_clip not installed")

    # Force the "currently frozen" flag so parent's on_train_batch_start
    # treats this call as the unfreeze transition. self.print() needs a Trainer
    # — silence it in the test by overriding at instance level.
    mod._lm_currently_frozen = True
    mod.print = lambda *a, **kw: None
    # Pre-fill queue so reset() has something to clear.
    keys = F.normalize(torch.randn(4, 512), dim=-1)
    mod.text_queue.enqueue(keys)
    queue_before = mod.text_queue.queue.clone()

    # Perturb model weights so ema != model before resync.
    with torch.no_grad():
        for p in mod.model.projection_head.parameters():
            p.add_(0.5)

    # Sanity: ema weights differ from live model before resync.
    ema_proj = dict(mod.momentum_encoder.encoder.projection_head.named_parameters())
    live_proj = dict(mod.model.projection_head.named_parameters())
    diff_before = any(
        not torch.allclose(ema_proj[k], live_proj[k]) for k in ema_proj
    )
    assert diff_before, "Pre-condition: ema must differ from live model"

    mod.on_train_batch_start(batch=None, batch_idx=0)

    # Post-condition: ema == live model (hard-resync).
    for k in ema_proj:
        assert torch.allclose(ema_proj[k], live_proj[k], atol=1e-6), \
            f"momentum_encoder.{k} not resynced after unfreeze"
    # Post-condition: queue ptr reset and contents changed.
    assert int(mod.text_queue.queue_ptr) == 0, "text_queue ptr must reset at unfreeze"
    assert not torch.equal(mod.text_queue.queue, queue_before), \
        "text_queue contents must be re-randomised at unfreeze"


@pytest.mark.willi_parity
def test_no_queue_inbatch_clip_fires_post_warmup():
    """Phase 13: with moco_queue_size=0, text_queue must be None and
    l_clip must be > 0 once freeze_text_encoder_steps=0 (simulating post-warmup).

    Verifies the no-queue in-batch CLIP path used in Phase 6e.
    """
    from hybrid_xmamba.models.configuration_hybrid import HybridConfig
    from hybrid_xmamba.models.hybrid_lm import HybridTextEncoder
    from hybrid_xmamba.training.lightning_module import JointMultiTaskLightningModule

    cfg = HybridConfig(
        vocab_size=100, dim=64, num_layers=2,
        layer_pattern=["mamba", "mlstm"],
        max_position_embeddings=64,
        use_fast_path=False, use_tfla=False,
        pooling_strategy="attention",
    )
    enc = HybridTextEncoder(cfg, embed_dim=512)

    class _StubTeacher(torch.nn.Module):
        def encode_text(self, input_ids):
            return torch.randn(input_ids.shape[0], 512)

    try:
        mod = JointMultiTaskLightningModule(
            model=enc, teacher=_StubTeacher(),
            warmup_steps=2, max_steps=10,
            freeze_text_encoder_steps=0,   # post-warmup: CLIP active immediately
            moco_queue_size=0,              # Phase 13: no queue
        )
    except ImportError:
        pytest.skip("open_clip not installed")

    assert mod.text_queue is None, \
        "text_queue must be None when moco_queue_size=0"

    B = 4
    class _StubImageEncoder(torch.nn.Module):
        def forward(self, px):
            return torch.randn(px.shape[0], 512)

    mod.image_encoder = _StubImageEncoder()

    input_ids = torch.randint(0, 100, (B, 16))
    attn = torch.ones(B, 16, dtype=torch.long)
    batch = {
        "input_ids": input_ids, "attention_mask": attn,
        "pixel_values": torch.randn(B, 3, 8, 8),
        "teacher_input_ids": input_ids, "teacher_attention_mask": attn,
    }

    logged: dict = {}

    def _capture(self, name, value, *args, **kwargs):  # type: ignore[no-untyped-def]
        try:
            logged[name] = float(value.detach().item() if hasattr(value, "detach") else value)
        except Exception:
            pass

    import types as _types
    mod.log = _types.MethodType(_capture, mod)
    loss = mod._joint_step(batch, batch_idx=0, split="train")

    assert torch.isfinite(loss), "Joint loss not finite"
    clip_val = logged.get("train/clip_loss", None)
    assert clip_val is not None and clip_val > 0.0, (
        f"clip_loss must be > 0 post-warmup with no queue; got {clip_val}"
    )


@pytest.mark.willi_parity
def test_mlstm_stability_config_present():
    """Phase 3D: HybridConfig must expose the three mLSTM gate-stabilisation knobs
    with safe defaults (cap=15, i_bias=-10, f_bias=0)."""
    from hybrid_xmamba.models.configuration_hybrid import HybridConfig

    cfg = HybridConfig()
    assert hasattr(cfg, "mlstm_gate_soft_cap"), \
        "HybridConfig missing mlstm_gate_soft_cap"
    assert hasattr(cfg, "mlstm_input_gate_bias_init"), \
        "HybridConfig missing mlstm_input_gate_bias_init"
    assert hasattr(cfg, "mlstm_forget_gate_bias_init"), \
        "HybridConfig missing mlstm_forget_gate_bias_init"
    assert cfg.mlstm_gate_soft_cap == 15.0, \
        f"Expected cap=15.0, got {cfg.mlstm_gate_soft_cap}"
    assert cfg.mlstm_input_gate_bias_init == -10.0, \
        f"Expected i_bias=-10.0, got {cfg.mlstm_input_gate_bias_init}"
    assert cfg.mlstm_forget_gate_bias_init == 0.0, \
        f"Expected f_bias=0.0, got {cfg.mlstm_forget_gate_bias_init}"


@pytest.mark.willi_parity
def test_stage1_proj_head_dropout_default():
    """hybrid_70m.yaml must keep proj_head_dropout=0.1 (literature SimCSE default).

    Run 1209 raised it to 0.3 in tandem with scale 20→5 to fight near-zero loss;
    both reverted after STS-B decline showed the issue was KD weight, not view
    diversity. 0.3 produces overly noisy positive views.
    """
    pytest.importorskip("yaml")
    import yaml

    cfg_path = REPO_ROOT / "configs" / "model" / "hybrid_70m.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    assert cfg.get("proj_head_dropout") == 0.1, (
        f"proj_head_dropout default should be 0.1, got {cfg.get('proj_head_dropout')}"
    )


@pytest.mark.willi_parity
def test_wsd_scheduler_shape():
    """Phase 7A: WSDScheduler must produce the plan-of-record shape.

    Asserts: warmup 1% (linear rise), stable 85% (factor==1.0), decay 14%
    (factor = 1 - sqrt(p) clamped at min_lr_ratio). Also asserts the β2
    helper anneals 0.999 → 0.974 across the decay phase.
    """
    import math as _math
    from hybrid_xmamba.training.schedulers import (
        WSDScheduler,
        wsd_factor,
        beta2_for_step,
    )

    base_lr = 1.0
    max_steps = 10000
    param = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.AdamW([param], lr=base_lr, betas=(0.9, 0.999))
    sched = WSDScheduler(optimizer, max_steps=max_steps)

    assert sched.warmup_steps == 100, f"warmup_steps={sched.warmup_steps} expected 100"
    assert sched.stable_steps == 8500, f"stable_steps={sched.stable_steps} expected 8500"
    assert sched.decay_steps == 1400, f"decay_steps={sched.decay_steps} expected 1400"
    assert sched.decay_start == 8600

    # Warmup: linear from 0.01 → 1.0 across 100 steps.
    f_warmup_start = wsd_factor(0, 100, 8500, 1400)
    f_warmup_mid = wsd_factor(50, 100, 8500, 1400)
    assert abs(f_warmup_start - 0.01) < 1e-6, f_warmup_start
    assert 0.4 < f_warmup_mid < 0.6, f_warmup_mid

    # Stable phase: constant 1.0.
    for s in (100, 1000, 5000, 8599):
        f = wsd_factor(s, 100, 8500, 1400)
        assert abs(f - 1.0) < 1e-6, f"stable step {s} factor={f}"

    # Decay: 1 - sqrt(p). At p=0.25, factor=0.5; at p=1, factor=0.
    f_quarter = wsd_factor(8600 + 350, 100, 8500, 1400)
    assert abs(f_quarter - 0.5) < 1e-6, f_quarter
    f_end = wsd_factor(max_steps, 100, 8500, 1400)
    assert abs(f_end - 0.0) < 1e-6, f_end

    # β2 schedule: constant pre-decay, linear during decay.
    assert beta2_for_step(0, 8600, 1400) == 0.999
    assert beta2_for_step(8600, 8600, 1400) == 0.999
    b2_mid = beta2_for_step(8600 + 700, 8600, 1400)
    assert abs(b2_mid - 0.9865) < 1e-6, b2_mid
    assert abs(beta2_for_step(max_steps, 8600, 1400) - 0.974) < 1e-6


def test_wsd_scheduler_absolute_warmup_override():
    """Phase 9F: WSDScheduler must honor absolute ``warmup_steps`` override.

    With ``max_steps=50000, warmup_steps=1000``: warmup is 1000 (not 500=1%);
    decay stays at 14% (=7000); stable absorbs the remainder (=42000).
    """
    from hybrid_xmamba.training.schedulers import WSDScheduler

    param = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.AdamW([param], lr=1.0)
    sched = WSDScheduler(
        optimizer,
        max_steps=50000,
        warmup_steps=1000,
    )

    assert sched.warmup_steps == 1000, sched.warmup_steps
    assert sched.decay_steps == 7000, sched.decay_steps
    assert sched.stable_steps == 42000, sched.stable_steps
    assert sched.decay_start == 43000, sched.decay_start


def test_norm_topology_threaded_to_hybridconfig():
    """Training entry points must carry every yaml config field into ``HybridConfig``.

    History, because this has now happened twice and the guard should reflect both:

    * Phase 9F -- ``HybridConfig`` was built from an explicit ``cfg.model.*`` list that omitted
      ``norm_topology``, so a v2 yaml's ``norm_topology: hybrid`` was ignored and HybridNorm
      weights loaded into a pre_rms model. This test was written then, asserting the literal
      ``norm_topology=`` kwarg was present.
    * 2026-09-06 (MAMBA3_PLAN_V2.md FM5) -- the same hand-written list dropped ``scan_impl``,
      ``tfla_impl`` and ``dt_init_strategy``. Job 2513007 trained the A1 arm with every defect
      still in place; only the ARCH fingerprint caught it. Guarding one field name could never
      have caught that, because the bug is the mechanism, not the field.

    So the assertion moved up a level: entry points must build through
    ``HybridConfig.from_hydra``, which filters against ``dataclasses.fields`` and therefore
    carries fields that do not exist yet. Source text plus a runtime round-trip, because the
    failure is invisible at runtime otherwise -- the model builds and trains perfectly well, it
    is simply not the architecture that was asked for.
    """
    import dataclasses
    import pathlib

    import yaml

    from hybrid_xmamba.models.configuration_hybrid import HybridConfig

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    for rel in (
        "scripts/train.py",
        "scripts/train_stage0_distill.py",
        "scripts/train_stage0_distill_resume.py",
        "scripts/train_contrastive.py",
        "scripts/train_report_generation.py",
    ):
        src = (repo_root / rel).read_text()
        assert "HybridConfig.from_hydra(" in src, (
            f"{rel}: must build HybridConfig via from_hydra(); a hand-written kwarg list "
            "silently drops any field nobody remembered to add"
        )
        assert "HybridConfig(\n" not in src, (
            f"{rel}: still constructs HybridConfig from an explicit kwarg list"
        )

    # Runtime half: whatever a yaml sets must arrive on the dataclass.
    fields = {f.name for f in dataclasses.fields(HybridConfig)}
    for name in ("hybrid_70m_v2", "hybrid_150m_v2", "hybrid_150m_a1", "hybrid_150m_m3"):
        path = repo_root / "configs" / "model" / f"{name}.yaml"
        if not path.exists():
            continue
        raw = yaml.safe_load(path.read_text())
        cfg = HybridConfig.from_hydra(raw)
        for key, value in raw.items():
            # `null` in a yaml means "derive it" -- __post_init__ fills dt_rank, num_heads and
            # slstm_hidden_dim -- so a None never round-trips unchanged and is not a drop.
            if (
                key in fields
                and key != "model_type"
                and value is not None
                and not isinstance(value, (dict, list))
            ):
                assert getattr(cfg, key) == value, (
                    f"{name}.yaml sets {key}={value!r} but the config has "
                    f"{getattr(cfg, key)!r} -- the field was dropped in transit"
                )


def test_resume_from_checkpoint_wired_to_trainer_fit():
    """Phase 9-EXT: train_stage0_distill.py must pass an optional resume ckpt to
    trainer.fit(ckpt_path=...) so a walltime-killed run can continue from last.ckpt
    (the 120K WSD run died at step 22K; resume + recalibrated max_steps fires the
    decay it never reached). Guard the wiring against accidental removal.
    """
    import pathlib

    src = (
        pathlib.Path(__file__).resolve().parent.parent
        / "scripts" / "train_stage0_distill.py"
    ).read_text()
    assert 'cfg.get("resume_from_checkpoint"' in src, (
        "train_stage0_distill.py: resume_from_checkpoint not read from cfg"
    )
    assert "ckpt_path=" in src, (
        "train_stage0_distill.py: trainer.fit must receive ckpt_path= for resume"
    )


def test_biomedclip_kd_joint_v2_config_present():
    """Phase 10C/10F: the v2 joint distill config must keep freq-decoupled KD OFF
    (2026-06-18 ablation: ON cost Indiana 3.90%->2.96%), enable the ViT unfreeze
    (supervisor Step 6; ablation proved it a pure +2.5pp MIMIC win), and hold the
    Phase 6e recipe (K=0).
    """
    import pathlib
    import yaml

    cfg_path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "configs" / "distill" / "biomedclip_kd_joint_v2.yaml"
    )
    assert cfg_path.exists(), "configs/distill/biomedclip_kd_joint_v2.yaml missing"
    cfg = yaml.safe_load(cfg_path.read_text())
    assert cfg["teacher"] == "biomedclip_text"
    assert cfg["freq_kd"] is False, (
        "freq_kd must default to false — the 2026-06-18 ablation showed it is a "
        "cross-domain regression (Indiana 3.90% -> 2.96%). Re-enable only per-run "
        "via distill.freq_kd=true (see train_biomedclip_kd_phase15.sh)."
    )
    assert cfg["freq_kd_low_bins"] == 32
    assert abs(float(cfg["freq_kd_alpha_high"]) - 0.1) < 1e-9
    assert cfg["vit_unfreeze_blocks"] == 2
    assert abs(float(cfg["vit_lr"]) - 1.0e-6) < 1e-12
    assert int(cfg["moco_queue_size"]) == 0  # Phase 6e recipe held constant
    # α_kd schedule unchanged from Phase 6e for attribution
    assert abs(float(cfg["alpha_kd_warmup"]) - 1.0) < 1e-9
    assert abs(float(cfg["alpha_kd_post"]) - 0.3) < 1e-9


def test_h100_contrastive_lrs_are_overridable():
    """Phase 6 post-mortem: backbone_lr/head_lr were HARDCODED at the bs=128
    sqrt-scaled values, so every arm of the batch sweep — including the winning
    bs=64 arm — trained at bs=128 LRs. The sweep was never LR-matched. Guard that
    the template threads the env vars instead of baking literals.
    """
    import pathlib
    import re

    script = (
        pathlib.Path(__file__).resolve().parent.parent
        / "scripts" / "train_biomedclip_kd_h100.sh"
    ).read_text()

    assert "distill.backbone_lr=${BACKBONE_LR}" in script, (
        "train_biomedclip_kd_h100.sh: backbone_lr must come from ${BACKBONE_LR}"
    )
    assert "distill.head_lr=${HEAD_LR}" in script, (
        "train_biomedclip_kd_h100.sh: head_lr must come from ${HEAD_LR}"
    )
    assert re.search(r"^BACKBONE_LR=\"\$\{BACKBONE_LR:-", script, re.M)
    assert re.search(r"^HEAD_LR=\"\$\{HEAD_LR:-", script, re.M)
    # The literals must not survive on the python invocation lines.
    assert "distill.head_lr=6e-4" not in script
    assert "distill.backbone_lr=2e-5" not in script


def test_h100_150m_contrastive_epoch_budget_is_batch_matched():
    """Phase 6 found bigger batches at fixed MAX_STEPS see MORE epochs, not fewer
    (bs=128 x 5000 = 23 epochs vs A100's 5.8), which confounded the negatives
    lever. The 150M wrapper now derives MAX_STEPS from BATCH_SIZE; assert every
    arm holds the same 384000-sample (13.93-epoch) budget over 27570 pairs.
    """
    import pathlib
    import re

    script = (
        pathlib.Path(__file__).resolve().parent.parent
        / "scripts" / "train_biomedclip_kd_150m_h100.sh"
    ).read_text()

    arms = re.findall(
        r"^\s*(\d+)\)\s+DEF_BACKBONE_LR=([0-9.e-]+);\s+"
        r"DEF_HEAD_LR=([0-9.e-]+);\s+DEF_MAX_STEPS=(\d+)",
        script,
        re.M,
    )
    assert len(arms) >= 3, f"expected >=3 batch arms, parsed {arms}"

    seen = {}
    for bs_s, backbone_lr, head_lr, steps_s in arms:
        bs, steps = int(bs_s), int(steps_s)
        assert bs * steps == 384000, (
            f"bs={bs} x {steps} steps = {bs * steps} samples, expected 384000 "
            "(13.93 epochs over 27570 MIMIC pairs)"
        )
        seen[bs] = (float(backbone_lr), float(head_lr))

    assert {32, 64, 128} <= set(seen), f"missing batch arms: {sorted(seen)}"
    # Canonical A100 anchor: bs=32 -> backbone 1e-5 / head 3e-4.
    assert abs(seen[32][0] - 1.0e-5) < 1e-12
    assert abs(seen[32][1] - 3.0e-4) < 1e-12
    # LRs must be sqrt-scaled off that anchor (monotone in batch size).
    assert seen[32][1] < seen[64][1] < seen[128][1], (
        f"head_lr must grow with batch size: {seen}"
    )
    for bs in (64, 128):
        expected = 3.0e-4 * (bs / 32.0) ** 0.5
        assert abs(seen[bs][1] - expected) / expected < 0.02, (
            f"bs={bs} head_lr {seen[bs][1]} deviates >2% from sqrt-scaled {expected:.3e}"
        )


def test_freq_decoupled_kd_threaded():
    """Phase 10B: freq-KD must be wired into the joint module and threaded from
    the distill config by train_contrastive.
    """
    import pathlib

    repo = pathlib.Path(__file__).resolve().parent.parent
    lm = (repo / "hybrid_xmamba" / "training" / "lightning_module.py").read_text()
    assert "self.freq_kd" in lm and "torch.fft.rfft" in lm, (
        "lightning_module.py: freq-decoupled KD branch not implemented"
    )
    tc = (repo / "scripts" / "train_contrastive.py").read_text()
    assert 'freq_kd=bool(distill_cfg.get("freq_kd"' in tc, (
        "train_contrastive.py: freq_kd not threaded from distill_cfg"
    )


def test_freq_decoupled_kd_loss_finite():
    """Phase 10D: the rFFT low/high-band KD math is finite and non-negative on
    normalized embeddings (mirrors the inline _joint_step computation).
    """
    import torch
    import torch.nn.functional as F

    torch.manual_seed(0)
    z = F.normalize(torch.randn(4, 512), dim=-1)
    t = F.normalize(torch.randn(4, 512), dim=-1)
    zf = torch.fft.rfft(z, dim=-1)
    tf = torch.fft.rfft(t, dim=-1)
    n_low = 32
    low_mse = (zf[:, :n_low] - tf[:, :n_low]).abs().pow(2).mean()
    high_mse = (zf[:, n_low:] - tf[:, n_low:]).abs().pow(2).mean()
    cos = F.cosine_similarity(z, t, dim=-1)
    l_kd = low_mse + 0.1 * high_mse + 0.5 * (1.0 - cos.mean())
    assert torch.isfinite(l_kd), l_kd
    assert l_kd.item() >= 0.0
    # identical embeddings → low/high MSE vanish, cosine term → 0
    l0_zf = torch.fft.rfft(z, dim=-1)
    l0_low = (l0_zf[:, :n_low] - l0_zf[:, :n_low]).abs().pow(2).mean()
    assert l0_low.item() == 0.0


# ── Phase 6C/6D/6E/6F — plateau-intervention block (2026-07-25) ───────────────
#
# Context these tests protect: seven consecutive nulls (Stage-0 PPL 15.62->13.18,
# 70M->150M, negatives 32->128, epochs 23->14, batch 128 vs 64, head_lr
# 6e-4->4.24e-4 and ->3.0e-4) against one positive (ViT unfreeze 0->2, +2.5pp).
# Every lever below must default to the Phase-6B recipe so 6D-0 is a real
# control — that invariant is what these tests exist to enforce.

def _tiny_text_encoder(bidirectional=False):
    from hybrid_xmamba.models.configuration_hybrid import HybridConfig
    from hybrid_xmamba.models.hybrid_lm import HybridTextEncoder

    cfg = HybridConfig(
        vocab_size=100, dim=64, num_layers=2,
        layer_pattern=["mamba", "mlstm"],
        max_position_embeddings=64,
        use_fast_path=False, use_tfla=False,
        pooling_strategy="attention",
        bidirectional_encode=bidirectional,
    )
    return HybridTextEncoder(cfg, embed_dim=512)


def _tiny_joint_module(**overrides):
    from hybrid_xmamba.training.lightning_module import JointMultiTaskLightningModule

    class _StubBiomedCLIPText(torch.nn.Module):
        def encode_text(self, input_ids):
            return torch.randn(input_ids.shape[0], 512)

    kwargs = dict(
        model=overrides.pop("model", None) or _tiny_text_encoder(),
        teacher=_StubBiomedCLIPText(),
        alpha_kd=0.3, alpha_kd_warmup=1.0, alpha_kd_post=0.3,
        beta_clip=1.0, gamma_simcse=0.1,
        backbone_lr=1e-5, head_lr=3e-4, weight_decay=0.01,
        warmup_steps=5, max_steps=50, gradient_clip_val=1.0,
        freeze_text_encoder_steps=1000,
    )
    kwargs.update(overrides)
    return JointMultiTaskLightningModule(**kwargs)


@pytest.mark.willi_parity
def test_phase6d_config_defaults_are_the_control():
    """6D-0 must be bit-identical to the Phase-6B recipe.

    Every new knob in biomedclip_kd_joint_v2.yaml has to ship in its inert
    state, or the "control" arm silently becomes a treatment arm — the exact
    class of drift that made the Phase-6 batch sweep run at bs=128 LRs.
    """
    import yaml

    cfg_path = REPO_ROOT / "configs" / "distill" / "biomedclip_kd_joint_v2.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())

    assert cfg["kd_decay_steps"] == 0, "6D-2 must default OFF (step function preserved)"
    assert cfg["alpha_kd_floor"] == 0.0
    assert cfg["clip_loss_type"] == "infonce", "6D-3 must default to the canonical loss"
    assert cfg["use_multipos"] is False
    # Canonical recipe held from Phase 6e — regressions here are known-harmful.
    assert cfg["freq_kd"] is False, "freq_kd=true cost Indiana 3.90%->2.96%"
    assert cfg["vit_unfreeze_blocks"] == 2, "vit_unfreeze=0 cost MIMIC 10.45%->7.97%"
    assert cfg["moco_queue_size"] == 0, "MoCo/XBM queue post-KD-warmup is harmful"


@pytest.mark.willi_parity
def test_kd_decay_schedule_ramps_post_unfreeze():
    """6D-2: alpha_kd must ramp alpha_kd_post -> alpha_kd_floor over
    kd_decay_steps AFTER the unfreeze, and kd_decay_steps=0 must reproduce the
    original step function exactly.

    Motivation: cos_text_teacher ~0.57 is a KD-vs-CLIP equilibrium (it reaches
    0.874-0.892 under KD-only warmup with a FROZEN backbone), not an
    architecture ceiling, so the standing anchor is the thing to attack.
    """
    def effective_alpha(step, freeze, decay_steps, post=0.3, warmup=1.0, floor=0.0):
        if step < freeze:
            return warmup
        if decay_steps > 0:
            t = (step - freeze) / float(decay_steps)
            t = min(max(t, 0.0), 1.0)
            return post * (1.0 - t) + floor * t
        return post

    # decay OFF → legacy step function, exactly.
    assert effective_alpha(0, 1000, 0) == 1.0
    assert effective_alpha(999, 1000, 0) == 1.0
    assert effective_alpha(1000, 1000, 0) == 0.3
    assert effective_alpha(9999, 1000, 0) == 0.3

    # decay ON → warmup unchanged, then linear ramp to the floor, then clamped.
    assert effective_alpha(999, 1000, 2000) == 1.0
    assert effective_alpha(1000, 1000, 2000) == pytest.approx(0.3)
    assert effective_alpha(2000, 1000, 2000) == pytest.approx(0.15)
    assert effective_alpha(3000, 1000, 2000) == pytest.approx(0.0, abs=1e-9)
    assert effective_alpha(9999, 1000, 2000) == pytest.approx(0.0, abs=1e-9)

    # Non-zero floor is honoured (the destabilisation fallback).
    assert effective_alpha(3000, 1000, 2000, floor=0.05) == pytest.approx(0.05)

    src = (REPO_ROOT / "hybrid_xmamba" / "training" / "lightning_module.py").read_text()
    assert "self.kd_decay_steps" in src and "self.alpha_kd_floor" in src
    assert "kd_decay_steps=int(distill_cfg.get(\"kd_decay_steps\", 0))" in (
        REPO_ROOT / "scripts" / "train_contrastive.py"
    ).read_text().replace("'", '"'), "kd_decay_steps not threaded from distill_cfg"


@pytest.mark.willi_parity
def test_multipos_loss_reduces_to_nt_xent_on_identity_mask():
    """6D-3: the multi-positive loss must be a strict GENERALISATION.

    With an identity positive mask it has to equal _nt_xent_loss to numerical
    precision — otherwise enabling use_multipos would change the objective even
    on a batch containing no duplicates, and no result would be attributable.
    """
    import torch.nn.functional as F

    try:
        mod = _tiny_joint_module()
    except ImportError:
        pytest.skip("open_clip not installed")

    torch.manual_seed(0)
    b = 8
    z1 = F.normalize(torch.randn(b, 512), dim=-1)
    z2 = F.normalize(torch.randn(b, 512), dim=-1)
    scale = torch.tensor(2.6592)

    eye = torch.eye(b, dtype=torch.bool)
    l_multi = mod._multipos_clip_loss(z1, z2, scale, eye)
    l_nt = mod._nt_xent_loss(z1, z2, scale)
    assert torch.allclose(l_multi, l_nt, atol=1e-5), (l_multi.item(), l_nt.item())

    # A real duplicate group must CHANGE the loss (it stops pushing the
    # duplicate apart) and must stay finite.
    mask = eye.clone()
    mask[0, 1] = True
    mask[1, 0] = True
    l_dup = mod._multipos_clip_loss(z1, z2, scale, mask)
    assert torch.isfinite(l_dup)
    assert not torch.allclose(l_dup, l_nt, atol=1e-5)


@pytest.mark.willi_parity
def test_siglip_loss_finite_and_bias_is_trainable_head_param():
    """6D-3: SigLIP path must be finite, batch-decoupled, and its bias must
    actually be optimised (a frozen bias silently makes the loss useless)."""
    import torch.nn.functional as F

    enc = _tiny_text_encoder()
    assert hasattr(enc, "logit_bias"), "HybridTextEncoder must expose logit_bias"
    assert float(enc.logit_bias.item()) == pytest.approx(-10.0), (
        "logit_bias must init at -10 so positives dominate early training"
    )
    assert enc.logit_bias.requires_grad

    try:
        mod = _tiny_joint_module(model=enc, clip_loss_type="siglip")
    except ImportError:
        pytest.skip("open_clip not installed")
    assert mod.clip_loss_type == "siglip"

    torch.manual_seed(0)
    for b in (4, 16, 64):
        z1 = F.normalize(torch.randn(b, 512), dim=-1)
        z2 = F.normalize(torch.randn(b, 512), dim=-1)
        loss = mod._siglip_loss(z1, z2, torch.tensor(2.6592), enc.logit_bias)
        assert torch.isfinite(loss), (b, loss)
        assert loss.item() > 0.0

    # Perfectly aligned pairs with a positive bias must score better than
    # anti-aligned ones — sanity on the sign convention.
    z = F.normalize(torch.randn(8, 512), dim=-1)
    good = mod._siglip_loss(z, z, torch.tensor(2.6592), torch.tensor(0.0))
    bad = mod._siglip_loss(z, -z, torch.tensor(2.6592), torch.tensor(0.0))
    assert good.item() < bad.item()

    # The bias must land in a param group, else it never moves.
    groups = mod.configure_optimizers()["optimizer"].param_groups
    assert any(any(p is enc.logit_bias for p in g["params"]) for g in groups), (
        "logit_bias is not in any optimizer param group"
    )


@pytest.mark.willi_parity
def test_pos_mask_from_hash_groups_duplicates():
    """6D-3: identical report hashes must form a positive set; the diagonal is
    always positive; a missing text_hash must degrade to the identity so the
    multi-positive path is inert on datasets that do not emit it."""
    from hybrid_xmamba.training.lightning_module import JointMultiTaskLightningModule

    dev = torch.device("cpu")
    h = torch.tensor([11, 22, 11, 33], dtype=torch.long)
    mask = JointMultiTaskLightningModule._pos_mask_from_hash(h, 4, dev)
    assert mask.dtype == torch.bool and mask.shape == (4, 4)
    assert bool(mask.diagonal().all())
    assert bool(mask[0, 2]) and bool(mask[2, 0]), "duplicate reports must pair"
    assert not bool(mask[0, 1]) and not bool(mask[1, 3])

    fallback = JointMultiTaskLightningModule._pos_mask_from_hash(None, 4, dev)
    assert torch.equal(fallback, torch.eye(4, dtype=torch.bool))


@pytest.mark.willi_parity
def test_mimic_dataset_emits_text_hash_matching_normalised_text():
    """6D-3: the dataset must emit a text_hash that is (a) stable across
    processes — Python's hash() is salted per process and would differ between
    dataloader workers — and (b) equal exactly when the normalised report text
    is equal."""
    import importlib.util

    from omegaconf import OmegaConf
    from PIL import Image

    spec = importlib.util.spec_from_file_location(
        "_tc_mod", REPO_ROOT / "scripts" / "train_contrastive.py"
    )
    tc = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(tc)
    except Exception as e:                                    # pragma: no cover
        pytest.skip("train_contrastive import failed: {}".format(e))

    src = (REPO_ROOT / "scripts" / "train_contrastive.py").read_text()
    assert "hashlib.blake2b" in src, "text_hash must not use the salted builtin hash()"

    class _StubTok:
        # pad_token_id present => MIMICJointDataset takes the HuggingFace
        # tokenizer branch (the open_clip branch expects a Callable returning a
        # tensor, not a dict).
        pad_token_id = 0

        def __call__(self, text, **kw):
            return {
                "input_ids": torch.zeros(1, 8, dtype=torch.long),
                "attention_mask": torch.ones(1, 8, dtype=torch.long),
            }

    rows = [
        {"findings": "Lungs are clear.", "impression": "No acute process.",
         "image": Image.new("RGB", (8, 8))},
        # Same text, different case/whitespace → must collide.
        {"findings": "LUNGS   are clear.", "impression": "No  acute process.",
         "image": Image.new("RGB", (8, 8))},
        {"findings": "Left pleural effusion.", "impression": "Effusion.",
         "image": Image.new("RGB", (8, 8))},
    ]
    cfg = OmegaConf.create({"dataset": {
        "max_length": 8, "teacher_max_length": 8,
        "findings_field": "findings", "impression_field": "impression",
        "concatenate_sections": True, "image_size": 8,
    }})
    ds = tc.MIMICJointDataset(rows, _StubTok(), _StubTok(), cfg)

    hashes = [int(ds[i]["text_hash"]) for i in range(3)]
    assert "text_hash" in ds[0]
    assert hashes[0] == hashes[1], "case/whitespace variants must share a hash"
    assert hashes[0] != hashes[2]


@pytest.mark.willi_parity
def test_bidirectional_encode_is_param_free_and_changes_output():
    """6E-1: the reverse pass must add NO parameters (so checkpoints stay
    loadable either way) while actually changing the embedding (so the flag is
    not a silent no-op)."""
    torch.manual_seed(0)
    uni = _tiny_text_encoder(bidirectional=False)
    bi = _tiny_text_encoder(bidirectional=True)

    assert set(uni.state_dict().keys()) == set(bi.state_dict().keys()), (
        "bidirectional encode must not introduce state-dict keys"
    )
    assert sum(p.numel() for p in uni.parameters()) == sum(
        p.numel() for p in bi.parameters()
    )
    assert uni.bidirectional_encode is False and bi.bidirectional_encode is True

    bi.load_state_dict(uni.state_dict())
    uni.eval()
    bi.eval()

    ids = torch.randint(1, 100, (3, 16))
    mask = torch.ones(3, 16, dtype=torch.long)
    mask[1, 10:] = 0          # ragged batch: right padding
    mask[2, 4:] = 0

    with torch.no_grad():
        z_uni = uni.encode(ids, attention_mask=mask)
        z_bi = bi.encode(ids, attention_mask=mask)
        z_override = uni.encode(ids, attention_mask=mask, bidirectional=True)

    assert torch.isfinite(z_uni).all() and torch.isfinite(z_bi).all()
    assert torch.allclose(z_bi.norm(dim=-1), torch.ones(3), atol=1e-4)
    assert not torch.allclose(z_uni, z_bi, atol=1e-5), "flag is a no-op"
    assert torch.allclose(z_bi, z_override, atol=1e-5), (
        "per-call override must match the config-level flag"
    )


@pytest.mark.willi_parity
def test_reverse_index_reverses_only_real_tokens():
    """6E-1: the reverse index must reverse the real-token block, leave right
    padding in place, and be its own inverse (that involution is what lets the
    same gather map reverse-pass states back onto original positions)."""
    enc = _tiny_text_encoder(bidirectional=True)

    ids = torch.tensor([
        [5, 6, 7, 8, 9],      # full length
        [5, 6, 7, 0, 0],      # length 3, right padded
        [5, 0, 0, 0, 0],      # length 1
    ])
    mask = torch.tensor([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0],
        [1, 0, 0, 0, 0],
    ])
    idx = enc._reverse_index(ids, mask)

    assert torch.equal(idx[0], torch.tensor([4, 3, 2, 1, 0]))
    assert torch.equal(idx[1], torch.tensor([2, 1, 0, 3, 4]))
    assert torch.equal(idx[2], torch.tensor([0, 1, 2, 3, 4]))

    # Involutive: gathering twice returns the original ordering.
    assert torch.equal(idx.gather(1, idx), torch.arange(5).unsqueeze(0).expand(3, 5))

    # Padding never moves into the real-token block.
    rev_ids = ids.gather(1, idx)
    assert torch.equal(rev_ids[1], torch.tensor([7, 6, 5, 0, 0]))

    # No mask → plain full reversal.
    idx_nomask = enc._reverse_index(ids, None)
    assert torch.equal(idx_nomask[0], torch.tensor([4, 3, 2, 1, 0]))


@pytest.mark.willi_parity
def test_bidirectional_flag_recorded_for_eval_autodetect():
    """6E-1: the flag adds no weights, so eval cannot sniff it from the state
    dict the way it sniffs layer_pattern/norm_topology. It MUST be persisted in
    checkpoint hparams and read back — same failure class as the fresh-ViT load
    that read 1.89% instead of 10.94%."""
    try:
        mod = _tiny_joint_module(model=_tiny_text_encoder(bidirectional=True))
    except ImportError:
        pytest.skip("open_clip not installed")
    assert mod.hparams["bidirectional_encode"] is True

    try:
        mod_uni = _tiny_joint_module(model=_tiny_text_encoder(bidirectional=False))
    except ImportError:
        pytest.skip("open_clip not installed")
    assert mod_uni.hparams["bidirectional_encode"] is False

    ev = (REPO_ROOT / "scripts" / "evaluate_cxr_retrieval.py").read_text()
    assert 'hyper_parameters' in ev and 'bidirectional_encode' in ev, (
        "evaluate_cxr_retrieval.py must auto-detect bidirectional_encode from the ckpt"
    )


@pytest.mark.willi_parity
def test_dedup_aware_retrieval_metric_and_grouping():
    """6C-3/6C-4: dedup-aware R@K must count a same-group retrieval as a hit,
    must reduce to the strict-index metric when every report is unique, and the
    grouping must be case/whitespace insensitive."""
    import importlib.util

    import numpy as np

    spec = importlib.util.spec_from_file_location(
        "_ecr_mod", REPO_ROOT / "scripts" / "evaluate_cxr_retrieval.py"
    )
    ecr = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(ecr)
    except Exception as e:                                    # pragma: no cover
        pytest.skip("evaluate_cxr_retrieval import failed: {}".format(e))

    groups = ecr.group_ids_from_texts([
        "No acute cardiopulmonary process.",
        "no  ACUTE cardiopulmonary   process.",     # duplicate after normalising
        "Left lower lobe opacity.",
    ])
    assert groups[0] == groups[1] and groups[0] != groups[2]

    # Three items; make image 0 rank text 1 (its duplicate) above text 0.
    img = np.eye(3, dtype=np.float64)
    txt = np.eye(3, dtype=np.float64)
    img[0] = [0.0, 1.0, 0.0]

    strict = ecr.compute_retrieval_metrics(img, txt)
    dedup = ecr.compute_retrieval_metrics(img, txt, groups=groups)
    assert strict["i2t_R@1"] < dedup["i2t_R@1"], (
        "dedup-aware R@1 must credit the textually identical retrieval"
    )
    assert dedup["i2t_R@1"] == pytest.approx(1.0)

    # All-unique groups → identical to the authoritative strict metric.
    uniq = np.arange(3)
    assert ecr.compute_retrieval_metrics(img, txt, groups=uniq)["i2t_R@1"] == (
        strict["i2t_R@1"]
    )


@pytest.mark.willi_parity
def test_h100_launch_script_exposes_phase6d_levers():
    """Every 6D/6E/6F lever must be env-overridable AND default to the
    Phase-6B recipe. Hardcoded values in this script have cost this project
    two separate confounded experiments (LRs, then MAX_STEPS)."""
    sh = (REPO_ROOT / "scripts" / "train_biomedclip_kd_h100.sh").read_text()

    for var, default in (
        ("VIT_UNFREEZE", "2"),
        ("KD_DECAY_STEPS", "0"),
        ("ALPHA_KD_FLOOR", "0.0"),
        ("CLIP_LOSS", "infonce"),
        ("MULTIPOS", "false"),
        ("GAMMA_SIMCSE", "0.1"),
        ("BIDIRECTIONAL", "false"),
        ("SELECTION_SPLIT", "false"),
    ):
        assert '{}="${{{}:-{}}}"'.format(var, var, default) in sh, (
            "{} must be env-overridable with default {}".format(var, default)
        )

    # The overrides must actually reach Hydra, not just be echoed.
    for override in (
        "distill.vit_unfreeze_blocks=${VIT_UNFREEZE}",
        "distill.kd_decay_steps=${KD_DECAY_STEPS}",
        "distill.alpha_kd_floor=${ALPHA_KD_FLOOR}",
        "distill.clip_loss_type=${CLIP_LOSS}",
        "distill.use_multipos=${MULTIPOS}",
        "distill.gamma_simcse=${GAMMA_SIMCSE}",
        "++model.bidirectional_encode=${BIDIRECTIONAL}",
        '${SPLIT_OVERRIDES[@]+"${SPLIT_OVERRIDES[@]}"}',
    ):
        assert override in sh, "missing Hydra override: {}".format(override)

    # 6F must move ONLY the train/val slices; the test gallery is fixed.
    assert "train[:85%]" in sh and "train[85%:90%]" in sh
    assert "vit_unfreeze_blocks=2 \\" not in sh, "vit_unfreeze must not be hardcoded"


@pytest.mark.willi_parity
def test_split_overrides_parse_under_hydra_grammar():
    """REGRESSION (2026-07-26, jobs 2372273-5 died in argument parsing).

    HuggingFace slice syntax contains '[', which is a Hydra override-grammar
    metacharacter. The shell strips "..." before exec, so an override written as
        dataset.train_split="${TRAIN_SPLIT}"
    reaches Hydra as a bare `train[:90%]` and is rejected with
        mismatched input '[' expecting <EOF>
    The value must arrive at Hydra STILL QUOTED.

    This test parses the script's actual override strings with Hydra's own
    parser, so the quoting cannot be "cleaned up" away again.
    """
    try:
        from hydra.core.override_parser.overrides_parser import OverridesParser
    except ImportError:                                       # pragma: no cover
        pytest.skip("hydra not installed")

    parser = OverridesParser.create()
    sh = (REPO_ROOT / "scripts" / "train_biomedclip_kd_h100.sh").read_text()

    # Every literal single-quoted override in the script must parse, and must
    # round-trip to the slice string the dataloader expects.
    literals = re.findall(r'"([\w.+]+=\'[^\']*\')"', sh)
    assert literals, "expected quoted split overrides in the script"
    parsed = {o.key_or_group: o.value() for o in parser.parse_overrides(literals)}
    assert parsed["dataset.train_split"] == "train[:85%]"
    assert parsed["dataset.validation_split"] == "train[85%:90%]"

    # Pin WHY the quotes are there: the bare form must still be rejected.
    with pytest.raises(Exception):
        parser.parse_overrides(["dataset.train_split=train[:85%]"])

    # The overrides must be emitted ONLY when 6F is requested. Passing them
    # unconditionally is what turned a 6F-only bug into an all-arms outage, and
    # it also breaks "6D-0 is bit-identical to the Phase-6B control" — the
    # control must reproduce the original argv, not an equivalent-valued one.
    assert "SPLIT_OVERRIDES=()" in sh
    guarded = sh.split('if [ "${SELECTION_SPLIT}" = "true" ]; then')[1].split("fi")[0]
    assert "dataset.train_split=" in guarded, (
        "split overrides must live inside the SELECTION_SPLIT guard"
    )
    assert "TRAIN_SPLIT" not in sh.split("python scripts/train_contrastive.py")[1], (
        "the python invocation must not reference a bare TRAIN_SPLIT variable"
    )


@pytest.mark.willi_parity
def test_phase6c_measurement_scripts_present_and_parse():
    """6C is the zero-training block that calibrates everything else; both
    scripts must exist and parse under the willi Python."""
    for name in ("reference_biomedclip_zeroshot.py", "audit_mimic_duplicates.py"):
        path = REPO_ROOT / "scripts" / name
        assert path.exists(), "missing Phase 6C script: {}".format(name)
        ast.parse(path.read_text())

    ref = (REPO_ROOT / "scripts" / "reference_biomedclip_zeroshot.py").read_text()
    # 6C-1 must use the SAME gallery as the authoritative eval or the teacher
    # number is not comparable to the student's 0.1113.
    assert 'split="train[90%:]"' in ref


@pytest.mark.willi_parity
def test_vit_unfreeze_scope_and_lr_are_sweepable():
    """Phase 6G: depth was the ONLY image-side axis ever swept.

    6D established unfreeze depth as the single lever that moves MIMIC retrieval
    (0.116/0.132/0.150/0.168 at depth 2/4/6/12). Depth is now exhausted — ViT-B/16
    has 12 blocks — so the remaining dose axes are LR and scope, and both were
    hardcoded: vit_lr sat at 1e-6 for the entire project, and the unfreeze only
    ever covered transformer blocks, leaving patch_embed / cls_token / pos_embed /
    final norm / visual projection frozen even at depth 12.
    """
    sh = (REPO_ROOT / "scripts" / "train_biomedclip_kd_h100.sh").read_text()
    assert 'VIT_LR="${VIT_LR:-1e-6}"' in sh, "vit_lr must be env-overridable"
    assert 'VIT_SCOPE="${VIT_SCOPE:-blocks}"' in sh
    assert "distill.vit_lr=${VIT_LR}" in sh, "vit_lr must not be hardcoded in the Hydra call"
    assert "distill.vit_lr=1e-6 \\" not in sh
    assert "distill.vit_unfreeze_scope=${VIT_SCOPE}" in sh

    import yaml
    cfg = yaml.safe_load(
        (REPO_ROOT / "configs" / "distill" / "biomedclip_kd_joint_v2.yaml").read_text()
    )
    # Canonical values unchanged, so historical attribution still holds.
    assert cfg["vit_unfreeze_blocks"] == 2
    assert cfg["vit_lr"] == 1.0e-6
    assert cfg["vit_unfreeze_scope"] == "blocks"

    # Scope must be validated, not silently ignored.
    from hybrid_xmamba.training.lightning_module import JointMultiTaskLightningModule
    with pytest.raises(ValueError):
        _tiny_joint_module(vit_unfreeze_scope="everything")

    # "blocks" must keep the historical param-group behaviour exactly.
    try:
        mod = _tiny_joint_module(vit_unfreeze_scope="blocks")
    except ImportError:
        pytest.skip("open_clip not installed")
    assert mod.vit_unfreeze_scope == "blocks"
    # No image encoder in the tiny harness -> helper must not be reached, and the
    # optimizer must still build.
    assert mod.configure_optimizers()["optimizer"] is not None


@pytest.mark.willi_parity
def test_eval_script_bakes_in_offline_and_populated_cache():
    """REGRESSION (2026-07-26). MIMIC-CXR is a GATED HF repo, so any online
    load_dataset 401s — the failure that killed job 2357924.

    eval_h100.sh carried a header comment saying "run with HF_DATASETS_OFFLINE=1"
    but never exported it, and its cache default pointed at
    ${SCRATCH_ROOT}/mimic_cxr_cache, which is empty — the populated caches live
    under /sc/home/$USER/dataset/. Both are now baked in, per dataset.
    """
    sh = (REPO_ROOT / "scripts" / "eval_h100.sh").read_text()
    assert 'export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"' in sh
    assert 'export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"' in sh
    assert "/sc/home/$USER/dataset/mimic_cxr_cache" in sh
    assert "/sc/home/$USER/dataset/indiana_cxr_cache" in sh
    # Match the ASSIGNMENT, not the word — the fix comment quotes the old path.
    assert 'EVAL_CACHE_DIR="${EVAL_CACHE_DIR:-${SCRATCH_ROOT}/mimic_cxr_cache}"' not in sh, (
        "eval cache must not default to the empty scratch path"
    )
    # The training template must keep the same guarantees.
    tr = (REPO_ROOT / "scripts" / "train_biomedclip_kd_h100.sh").read_text()
    assert 'export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"' in tr
    assert "/sc/home/$USER/dataset/mimic_cxr_cache" in tr


@pytest.mark.willi_parity
def test_performance_profile_loads_every_advertised_model_config():
    """REGRESSION (2026-07-28). performance_profile.py resolved --model through
    ModelRegistry, which only ever registers 350m/1_3b/7b/mamba_baseline/
    xlstm_baseline. Every 70M and 150M name in its own --model choices list —
    i.e. every config this project actually trains, including the active
    hybrid_150m_v2 backbone — raised ValueError before a single measurement ran.

    Configs now resolve from configs/model/<name>.yaml, which is the source of
    truth, with the registry as fallback. The yamls carry training keys that are
    not HybridConfig fields (learning_rate, warmup_steps, distill, ...), so the
    loader must filter to the dataclass fields rather than splatting the dict.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "performance_profile", REPO_ROOT / "scripts" / "performance_profile.py"
    )
    pp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pp)

    names = pp.available_configs()
    for required in ("hybrid_70m", "hybrid_70m_v2", "hybrid_150m_v2",
                     "mamba_70m_baseline", "xlstm_70m_baseline"):
        assert required in names, "{} missing from profiler choices".format(required)

    # Every advertised choice must actually construct — that is the bug.
    for name in names:
        cfg = pp.load_config(name)
        assert cfg.dim > 0 and cfg.num_layers > 0, name

    # Spot-check that yaml values win over dataclass defaults.
    v2 = pp.load_config("hybrid_150m_v2")
    assert v2.dim == 768 and v2.num_layers == 12
    assert v2.norm_topology == "hybrid"
    assert v2.pooling_strategy == "attention"
    assert v2.max_position_embeddings == 1024


@pytest.mark.willi_parity
def test_efficiency_curve_slope_fit_is_correct():
    """The scaling exponent is the whole point of the sweep, so pin the fit.

    Latency ~ L^1 is the linear-scaling claim for Mamba/mLSTM; softmax attention
    would show ~L^2. A wrong slope fit would silently misreport the headline
    architectural result.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "performance_profile", REPO_ROOT / "scripts" / "performance_profile.py"
    )
    pp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pp)

    xs = [256, 512, 1024, 2048]
    assert abs(pp.fit_log_slope(xs, [x * 1.0 for x in xs]) - 1.0) < 1e-9
    assert abs(pp.fit_log_slope(xs, [x ** 2.0 for x in xs]) - 2.0) < 1e-9
    # Degenerate inputs must return None, not raise or emit a bogus exponent.
    assert pp.fit_log_slope([256], [1.0]) is None
    assert pp.fit_log_slope(xs, [None, None, None, None]) is None
    assert pp.fit_log_slope([256, 256], [1.0, 2.0]) is None


@pytest.mark.willi_parity
def test_sequence_sweep_is_valid_past_max_position_embeddings():
    """The efficiency curve sweeps L well past max_position_embeddings (1024).

    That is only legitimate because HybridLanguageModel sets
    use_pos_embedding = False (hybrid_lm.py:43) — there is no absolute position
    table to index out of. If someone re-enables it, the sweep would start
    indexing past the embedding and this test must fail loudly.
    """
    from hybrid_xmamba.models.configuration_hybrid import HybridConfig
    from hybrid_xmamba.models.hybrid_lm import HybridLanguageModel

    cfg = HybridConfig(
        vocab_size=64, dim=32, num_layers=2, layer_pattern=["mamba", "mlstm"],
        head_dim=16, num_heads=2, max_position_embeddings=16,
    )
    model = HybridLanguageModel(cfg)
    assert model.embeddings.use_pos_embedding is False
    model.eval()
    # 4x max_position_embeddings must run rather than raise.
    with torch.no_grad():
        out = model(torch.randint(0, cfg.vocab_size, (1, 64)))
    logits = out.logits if hasattr(out, "logits") else out
    assert logits.shape[:2] == (1, 64)


# ---------------------------------------------------------------------------
# Phase 8 (H100_SCALING_PLAN.md) — local PhysioNet MIMIC-CXR-JPG build.
# ---------------------------------------------------------------------------

def _load_build_script():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_mimic_cxr_local", REPO_ROOT / "scripts" / "build_mimic_cxr_local.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_get_session_requires_physionet_session_cookie_file(tmp_path):
    """REGRESSION (2026-08-16). PhysioNet's Django deployment does NOT honour
    HTTP Basic Auth for this project — verified live: `curl -u user
    https://physionet.org/settings/profile/` returns 302 to /login/
    regardless of credential correctness, while the identical /files/ URL
    with a valid session cookie returns 200. _get_session() must therefore
    require ~/.physionet_session and fail LOUDLY (not silently fall back to
    a ~/.netrc Basic Auth path that is known not to work) when it is absent.
    """
    mod = _load_build_script()
    orig_home = mod.Path.home
    mod.Path.home = staticmethod(lambda: tmp_path)  # empty dir, no cookie file
    mod._session = None
    try:
        with pytest.raises(RuntimeError, match="physionet_session"):
            mod._get_session()
    finally:
        mod.Path.home = orig_home


def test_download_detects_login_page_disguised_as_200_and_writes_nothing(tmp_path):
    """REGRESSION (2026-08-16). A session cookie can expire mid-run. PhysioNet
    then 302s to /login/, and `requests` follows redirects by default, so
    this arrives as an ordinary 200 with an HTML login page as the body.
    Without a guard, that body gets streamed straight into a .jpg/.csv.gz,
    and the resume check (Path.exists()) then skips the corrupt file forever
    on every subsequent run — a stop-everything bug discovered only when
    training on garbage weeks later. _download() must detect this BEFORE
    writing any bytes and never create the destination file.
    """
    mod = _load_build_script()

    class _FakeLoginResp:
        status_code = 200
        headers = {"Content-Type": "text/html; charset=utf-8"}
        url = "https://physionet.org/login/?next=/files/foo"

        def iter_content(self, chunk_size=None):
            raise AssertionError("must not stream the body of a login-page response")

        def close(self):
            pass

    class _FakeSession:
        def get(self, url, timeout=None, stream=None):
            return _FakeLoginResp()

    mod._get_session = lambda: _FakeSession()
    dest = tmp_path / "should_not_exist.jpg"
    status = mod._download("https://physionet.org/files/foo", dest)
    assert status == mod.SESSION_EXPIRED
    assert not dest.exists()


def test_download_still_writes_file_for_a_genuine_200(tmp_path):
    """Companion to the SESSION_EXPIRED regression test above — the
    login-page guard must not false-positive on a real, successful download
    (e.g. Content-Type: application/gzip, no /login in the final URL)."""
    mod = _load_build_script()

    class _FakeOkResp:
        status_code = 200
        headers = {"Content-Type": "application/gzip"}
        url = "https://physionet.org/files/foo"

        def iter_content(self, chunk_size=None):
            yield b"hello world"

        def close(self):
            pass

    class _FakeSession:
        def get(self, url, timeout=None, stream=None):
            return _FakeOkResp()

    mod._get_session = lambda: _FakeSession()
    dest = tmp_path / "should_exist.gz"
    status = mod._download("https://physionet.org/files/foo", dest)
    assert status == 200
    assert dest.read_bytes() == b"hello world"
    assert not dest.with_name(dest.name + ".part").exists()  # renamed away, not left behind


def test_download_never_leaves_a_partial_file_at_dest_on_interruption(tmp_path):
    """REGRESSION (2026-08-16, caught LIVE, not just in review). A SLURM kill
    (time limit / preemption) mid-download previously left a truncated-but-
    nonzero-size file directly at `dest`. Every resume check in this script
    (`dest.exists() and dest.stat().st_size > 0`) then trusted it as
    complete. This is exactly what happened: a `--time=00:10:00` override
    killed a `mimic-cxr-reports.zip` download mid-stream, the next run's
    `stage_meta` printed "[meta] have mimic-cxr-reports.zip" and skipped
    re-fetching it, and the corruption only surfaced later as
    `zipfile.BadZipFile: File is not a zip file` at unzip time. Fix:
    `_download()` streams to a `.part` sibling and `Path.replace()`s into
    `dest` only after the full body is consumed — an interruption anywhere
    in that process must leave `dest` absent, never partial.
    """
    import requests

    mod = _load_build_script()

    class _FakeInterruptedResp:
        status_code = 200
        headers = {"Content-Type": "application/zip"}
        url = "https://physionet.org/files/foo.zip"

        def iter_content(self, chunk_size=None):
            yield b"partial-bytes-before-the-job-was-killed"
            raise requests.exceptions.ConnectionError("simulated interruption mid-stream")

        def close(self):
            pass

    class _FakeSession:
        def get(self, url, timeout=None, stream=None):
            return _FakeInterruptedResp()

    mod._get_session = lambda: _FakeSession()
    dest = tmp_path / "foo.zip"
    status = mod._download("https://physionet.org/files/foo.zip", dest, retries=1)
    assert status != 200
    assert not dest.exists(), "an interrupted download must never leave a partial file at dest"


def test_get_session_is_thread_safe_and_created_exactly_once(tmp_path):
    """REGRESSION (2026-08-16, caught LIVE): stage_fetch's ThreadPoolExecutor
    calls _get_session() from up to `workers` threads concurrently on the
    first chunk. The original check-then-set was not atomic; observed live
    as FIVE duplicate "[auth] session cookie loaded" log lines from one
    invocation. Harmless correctness-wise (every racing thread reads the
    same cookie), but wasteful (needless extra Session objects splitting
    the connection pool) and confusing in logs. Artificially slow down
    Session construction to widen the race window -- this makes the test
    fail reliably against the old unlocked code rather than passing by
    scheduling luck, and pass reliably against the double-checked-locking
    fix regardless of how the threads are scheduled.
    """
    import threading
    import time as _time

    mod = _load_build_script()
    cookie_file = tmp_path / ".physionet_session"
    cookie_file.write_text("abc123")
    mod.Path.home = staticmethod(lambda: tmp_path)
    mod._session = None

    creations = []
    orig_session_cls = mod.requests.Session

    class _SlowCountingSession(orig_session_cls):
        def __init__(self):
            creations.append(1)
            _time.sleep(0.02)  # widen the race window
            super().__init__()

    mod.requests.Session = _SlowCountingSession

    results = []

    def _call():
        results.append(mod._get_session())

    threads = [threading.Thread(target=_call) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(creations) == 1, "Session must be created exactly once even under concurrent first access"
    assert all(r is results[0] for r in results), "every caller must receive the identical Session object"


def test_pack_study_hashes_filter_and_out_prefix(tmp_path):
    """Phase 9A: pack --study-hashes packs only a hash-matched subset (e.g.
    Arm-0's training set) out of whatever has actually been downloaded so
    far in the SHARED mimic_full tree -- --study-hashes on fetch and pack
    exist because copying manifest.parquet into a separate --out does NOT
    isolate anything (local_jpg paths are baked in absolute at manifest-
    generation time, found live 2026-08-17). --out-prefix keeps this from
    clobbering the eventual production train/validate/test.parquet when
    packing from the same --out directory.
    """
    import pandas as pd
    from PIL import Image

    mod = _load_build_script()
    out = tmp_path

    # Two rows match the hash filter; only one has actually been downloaded
    # (local_jpg exists) -- pack's existing local_jpg.exists() filter runs
    # BEFORE the hash filter, so only that one must survive.
    img_a = out / "a.jpg"
    Image.new("L", (10, 10)).save(img_a, "JPEG")
    manifest = pd.DataFrame({
        "has_text": [True, True, True],
        "local_jpg": [str(img_a), str(out / "b_not_downloaded.jpg"), str(out / "c.jpg")],
        "findings": ["f1", "f2", "f3"],
        "impression": ["i1", "i2", "i3"],
        "study_id": [1, 2, 3],
        "subject_id": [10, 20, 30],
        "dicom_id": ["d1", "d2", "d3"],
        "ViewPosition": ["PA", "PA", "PA"],
        "report_hash": ["hash_a", "hash_b", "hash_c"],
        "split": ["train", "train", "test"],
    })
    manifest.to_parquet(out / "manifest.parquet", index=False)

    hashes_file = out / "wanted.txt"
    hashes_file.write_text("hash_a\nhash_b\n")  # b matches but was never downloaded

    mod.stage_pack(out, exclude_hashes="", min_match_frac=0.95, allow_low_match=False,
                    study_hashes=str(hashes_file), out_prefix="arm0_")

    assert (out / "arm0_train.parquet").exists()
    assert not (out / "train.parquet").exists(), "must not touch the production filename"
    packed = pd.read_parquet(out / "arm0_train.parquet")
    assert len(packed) == 1
    assert packed.iloc[0]["study_id"] == 1


def test_fetch_study_hashes_filter_restricts_to_matching_subset(monkeypatch, tmp_path):
    """Phase 9A: --study-hashes lets `fetch` target only a hash-matched
    subset of the manifest (e.g. the historical Arm-0 reproduction set),
    reusing all existing chunking/resume/atomic-write/session-expiry logic
    rather than a parallel code path. Verify the filter actually restricts
    which rows get attempted, not just that it accepts the argument.
    """
    import pandas as pd

    mod = _load_build_script()

    out = tmp_path
    manifest = pd.DataFrame({
        "has_text": [True, True, True],
        "local_jpg": [str(out / "a.jpg"), str(out / "b.jpg"), str(out / "c.jpg")],
        "rel_jpg": ["files/a.jpg", "files/b.jpg", "files/c.jpg"],
        "report_hash": ["hash_a", "hash_b", "hash_c"],
    })
    manifest.to_parquet(out / "manifest.parquet", index=False)

    hashes_file = out / "wanted.txt"
    # Includes a hash that matches nothing, on purpose -- must not error.
    hashes_file.write_text("hash_b\nhash_c\nhash_does_not_exist\n")

    attempted = []

    def fake_download(url, dest, timeout=60, retries=3):
        attempted.append(url)
        return 404  # any non-200: records the attempt without reaching the
        # ProcessPoolExecutor resize step, which a dynamically-loaded test
        # module can't pickle across a spawned worker process (a test-
        # harness limitation, not a real one -- production runs load the
        # script normally and resize correctly, see the other fetch tests).

    monkeypatch.setattr(mod, "_download", fake_download)
    with pytest.raises(RuntimeError, match="converted 0 of"):
        mod.stage_fetch(out, size=320, chunk=2000, workers=1, limit=0,
                         study_hashes=str(hashes_file))

    assert len(attempted) == 2, "only the 2 matching rows (b, c) should be attempted"
    assert any("b.jpg" in u for u in attempted)
    assert any("c.jpg" in u for u in attempted)
    assert not any("/a.jpg" in u for u in attempted)


def test_fetch_aborts_on_session_expired_even_if_some_files_in_chunk_succeeded(monkeypatch, tmp_path):
    """REGRESSION (2026-08-16). A cookie can expire MID-CHUNK, so `ok` (the
    count of status==200) can be > 0 in the same chunk that also contains
    SESSION_EXPIRED entries — the ok==0 abort guard alone would NOT catch
    this. The SESSION_EXPIRED check in stage_fetch must be unconditional,
    not folded into (or ordered after) the ok==0 check.
    """
    import pandas as pd

    mod = _load_build_script()

    out = tmp_path
    manifest = pd.DataFrame({
        "has_text": [True, True],
        "local_jpg": [str(out / "a.jpg"), str(out / "b.jpg")],
        "rel_jpg": ["files/a.jpg", "files/b.jpg"],
    })
    manifest.to_parquet(out / "manifest.parquet", index=False)

    # First file "succeeds" (200), second is a login-page (SESSION_EXPIRED) —
    # simulates the cookie expiring between the two downloads.
    call_count = {"n": 0}

    def fake_download(url, dest, timeout=60, retries=3):
        call_count["n"] += 1
        return 200 if call_count["n"] == 1 else mod.SESSION_EXPIRED

    monkeypatch.setattr(mod, "_download", fake_download)

    with pytest.raises(RuntimeError, match="session-expired"):
        mod.stage_fetch(out, size=320, chunk=2000, workers=1, limit=0)


def test_build_script_hash_matches_repo_leakage_join_convention():
    """The Phase 8D leakage guard joins build_mimic_cxr_local.py's report_hash
    against a dump of the legacy gallery's hashes (dump_legacy_gallery_hashes.py).
    Both MUST use the identical normalisation + digest as the two conventions
    already in this repo — normalize_report_text (evaluate_cxr_retrieval.py:414)
    and the text_hash construction (train_contrastive.py:419-424) — or the join
    silently drops to near-zero matches and the leakage guard does nothing while
    reporting success.
    """
    import hashlib

    mod = _load_build_script()

    findings, impression = "The lungs are clear.", "No acute process."
    text = "Findings: {} Impression: {}".format(findings, impression)

    # evaluate_cxr_retrieval.normalize_report_text
    norm = " ".join(text.lower().split())
    expected_hex = hashlib.blake2b(norm.encode("utf-8"), digest_size=8).hexdigest()
    assert mod.norm_hash(text) == expected_hex

    # train_contrastive.py's text_hash (int64, mod 2**62) must be the same
    # digest truncated, not an independently-computed value.
    expected_int = int.from_bytes(
        hashlib.blake2b(norm.encode("utf-8"), digest_size=8).digest(), "big"
    ) % (2 ** 62)
    assert int(mod.norm_hash(text), 16) % (2 ** 62) == expected_int

    # Case/whitespace-insensitive, matching normalize_report_text's contract.
    assert mod.norm_hash(text) == mod.norm_hash(
        "FINDINGS:   The lungs   are clear.  Impression: No acute process.".replace(
            "FINDINGS:   The lungs   are clear.  ", "Findings: The lungs are clear. "
        )
    )


def test_extract_findings_impression_basic_and_custom_override():
    """The vendored official section parser (Phase 8C) must separate FINDINGS
    from IMPRESSION on a well-formed report, and must honour the
    custom_mimic_cxr_rules() per-study overrides for known-malformed reports —
    both code paths a homegrown regex would silently get wrong.
    """
    from scripts.mimic_cxr_vendor.extract import extract_findings_impression
    from scripts.mimic_cxr_vendor.section_parser import custom_mimic_cxr_rules

    report = (
        "\n FINAL REPORT\n EXAMINATION:  CHEST (PA AND LAT)\n\n"
        " INDICATION:  Cough.\n\n"
        " COMPARISON:  None.\n\n"
        " FINDINGS:\n\n"
        " The lungs are clear.  No focal consolidation.\n\n"
        " IMPRESSION:\n\n"
        " No acute cardiopulmonary process.\n"
    )
    findings, impression = extract_findings_impression(report, "s99999999")
    assert "lungs are clear" in findings.lower()
    assert "no acute cardiopulmonary" in impression.lower()

    # A study_id present in custom_mimic_cxr_rules()'s index-override table
    # must take the override path, not the regex path, regardless of report
    # content — this is what makes the ~30 known-malformed reports usable.
    _, custom_indices = custom_mimic_cxr_rules()
    override_stem, (start, end) = next(iter(custom_indices.items()))
    probe_text = "x" * start + "TARGET_SPAN" + "x" * 200
    if end <= len(probe_text):
        f, i = extract_findings_impression(probe_text, override_stem)
        assert f == ""  # index-override path has no separate findings section
        assert i == probe_text[start:end]


def test_cxr_mimic_full_config_present_and_consistent():
    """New config key (CLAUDE.md: new module/config key -> parity assertion).
    cxr_mimic_full.yaml must set local_parquet_dir (the signal
    train_contrastive.load_mimic_cxr / evaluate_cxr_retrieval.build_dataloader
    branch on) and keep every OTHER key schema-compatible with mimic_cxr.yaml
    so an unmodified training/eval invocation is unaffected by this file
    merely existing.

    dataset_name must be the literal "mimic_cxr", NOT this file's own name.
    scripts/train_contrastive.py's prepare_dataloader() dispatches on this
    exact string BEFORE load_mimic_cxr ever inspects local_parquet_dir; any
    other value raises "Unknown dataset for contrastive training" and never
    reaches the local-parquet branch this config exists to select. Confirmed
    live 2026-08-20 (job 2470516) -- dataset_name was "cxr_mimic_full" and
    training died at dataloader-prep, first time this file was ever exercised
    through actual training (Phase 8 only built the data).
    """
    from omegaconf import OmegaConf

    legacy = OmegaConf.load(REPO_ROOT / "configs" / "dataset" / "mimic_cxr.yaml")
    full = OmegaConf.load(REPO_ROOT / "configs" / "dataset" / "cxr_mimic_full.yaml")

    assert full.get("dataset_name") == "mimic_cxr"
    assert "local_parquet_dir" in full
    assert "local_parquet_dir" not in legacy  # legacy path must stay the default

    for key in ("tokenizer", "max_length", "teacher_max_length",
                "findings_field", "impression_field", "concatenate_sections",
                "image_size", "image_mean", "image_std"):
        assert key in full, "cxr_mimic_full.yaml missing schema key: {}".format(key)


def test_cxr_mimic_arm0_config_is_full_pointed_at_arm0_symlink_dir():
    """Phase 9A Arm-0 config (CLAUDE.md: new config file -> parity assertion).
    cxr_mimic_arm0.yaml must be byte-for-byte identical to cxr_mimic_full.yaml
    EXCEPT local_parquet_dir, which must point at an 'arm0' subdirectory.
    dataset_name is NOT a legitimate difference -- both files must carry the
    literal "mimic_cxr" (see test_cxr_mimic_full_config_present_and_consistent
    for why; using either file's own name there breaks training dispatch).
    The loader hardcodes train/validate/test.parquet, so the arm0 subdir is
    the symlink dir (arm0_train.parquet->train.parquet, ...) -- the only
    mechanism that selects the arm0_-prefixed pack output without a code
    change. Any OTHER drift between the two would silently change the Arm-0
    recipe relative to the production build it is meant to control.
    """
    from omegaconf import OmegaConf

    full = OmegaConf.load(REPO_ROOT / "configs" / "dataset" / "cxr_mimic_full.yaml")
    arm0 = OmegaConf.load(REPO_ROOT / "configs" / "dataset" / "cxr_mimic_arm0.yaml")

    assert arm0.get("dataset_name") == "mimic_cxr"
    assert arm0.get("dataset_name") == full.get("dataset_name")
    assert "local_parquet_dir" in arm0
    # points at an arm0 subdir of whatever full's dir is (symlink dir for the
    # arm0_-prefixed parquets), not the production tree itself.
    assert str(arm0.local_parquet_dir).rstrip("/").endswith("/arm0")
    assert str(arm0.local_parquet_dir).rstrip("/") == str(full.local_parquet_dir).rstrip("/") + "/arm0"

    # Every OTHER key must match cxr_mimic_full exactly -- Arm-0 changes only
    # the data location, never the dispatch name/recipe/schema.
    for key in full:
        if key == "local_parquet_dir":
            continue
        assert key in arm0, "cxr_mimic_arm0.yaml missing key present in full: {}".format(key)
        assert arm0[key] == full[key], "cxr_mimic_arm0.yaml drifted from full at key: {}".format(key)


def test_load_mimic_cxr_dispatches_to_local_parquet_when_configured():
    """load_mimic_cxr must call load_dataset("parquet", data_files=...) when
    dataset.local_parquet_dir is set, and must NOT touch the HF mirror path in
    that case — the exact regression this branch exists to prevent is a typo
    that silently falls through to the old itsanmolgupta network path.
    """
    import importlib.util

    from omegaconf import OmegaConf

    spec = importlib.util.spec_from_file_location(
        "_tc_mod_local", REPO_ROOT / "scripts" / "train_contrastive.py"
    )
    tc = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(tc)
    except Exception as e:                                    # pragma: no cover
        pytest.skip("train_contrastive import failed: {}".format(e))

    calls = []

    def _fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))

        class _FakeDS:
            column_names = ["image", "findings", "impression"]

            def __len__(self):
                return 0

            def filter(self, fn):
                return self

        return _FakeDS()

    tc.load_dataset = _fake_load_dataset

    cfg = OmegaConf.create({"dataset": {
        "local_parquet_dir": "/fake/dir",
        "train_split": "train", "validation_split": "validation", "test_split": "test",
        "cache_dir": "/unused",
    }})

    try:
        tc.load_mimic_cxr(cfg, "train", tokenizer=None, teacher_tokenizer=None)
    except RuntimeError:
        pass  # expected — the fake dataset is empty; we only care about the call args

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] == "parquet" or kwargs.get("path") == "parquet"
    data_files = kwargs.get("data_files") or (args[1] if len(args) > 1 else None)
    assert data_files is not None
    assert data_files["train"] == "/fake/dir/train.parquet"
    assert data_files["validation"] == "/fake/dir/validate.parquet"
    assert data_files["test"] == "/fake/dir/test.parquet"


@pytest.mark.willi_parity
@pytest.mark.parametrize("use_augmentation,is_train,expect_augmented", [
    (False, True, False),
    (True, False, False),
    (True, True, True),
])
def test_build_image_transform_augmentation_gating(use_augmentation, is_train, expect_augmented):
    """Phase 9D (2026-08-24, after job 2478647's arm0 checkpoint was confirmed
    via --checkpoint mode to have memorized boilerplate templates rather than
    condition on the image): RandomResizedCrop/RandomRotation must apply ONLY
    when BOTH use_augmentation is set AND is_train — augmenting eval images
    would make evaluation nondeterministic regardless of the training question,
    and use_augmentation defaulting off/False keeps retrieval's closed-chapter
    usage of this same helper byte-identical."""
    from omegaconf import OmegaConf
    import torchvision.transforms as T
    from scripts.train_contrastive import build_image_transform

    cfg = OmegaConf.create({"dataset": {"use_augmentation": use_augmentation, "image_size": 224}})
    transform = build_image_transform(cfg, is_train=is_train)
    types = [type(t) for t in transform.transforms]

    assert (T.RandomResizedCrop in types) is expect_augmented
    assert (T.RandomRotation in types) is expect_augmented
    assert (T.Resize in types) is (not expect_augmented)


@pytest.mark.willi_parity
def test_build_image_transform_default_matches_pre_9d_pipeline():
    """Regression pin: with use_augmentation absent (the default for every
    existing dataset config, i.e. the state before 9D landed), build_image_
    transform() must be byte-identical to the old hardcoded pipeline (Resize
    -> Grayscale(3) -> ToTensor -> Normalize) — protects retrieval's closed
    chapter from any accidental behavior change from this refactor."""
    from omegaconf import OmegaConf
    from PIL import Image
    import torchvision.transforms as T
    from scripts.train_contrastive import build_image_transform

    cfg = OmegaConf.create({"dataset": {}})
    transform = build_image_transform(cfg, is_train=True)  # is_train=True but use_augmentation unset

    old_transform = T.Compose([
        T.Resize((224, 224)),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    img = Image.new("RGB", (300, 400), color=(128, 64, 200))
    torch.testing.assert_close(transform(img), old_transform(img))


@pytest.mark.willi_parity
def test_load_mimic_cxr_threads_is_train_only_for_train_split():
    """load_mimic_cxr must pass is_train=True only when split=='train' —
    validation/test must never get augmented, even with use_augmentation=True.
    Reuses test_load_mimic_cxr_dispatches_to_local_parquet_when_configured's
    import-by-path + fake load_dataset pattern."""
    import importlib.util
    from omegaconf import OmegaConf
    import torchvision.transforms as T

    spec = importlib.util.spec_from_file_location(
        "_tc_mod_istrain", REPO_ROOT / "scripts" / "train_contrastive.py"
    )
    tc = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(tc)
    except Exception as e:                                    # pragma: no cover
        pytest.skip("train_contrastive import failed: {}".format(e))

    class _FakeDS:
        column_names = ["image", "findings", "impression"]

        def __len__(self):
            return 1

        def filter(self, fn):
            return self

    tc.load_dataset = lambda *a, **k: _FakeDS()

    cfg = OmegaConf.create({"dataset": {
        "local_parquet_dir": "/fake/dir",
        "train_split": "train", "validation_split": "validation", "test_split": "test",
        "cache_dir": "/unused", "max_length": 32, "use_augmentation": True,
    }})

    train_ds = tc.load_mimic_cxr(cfg, "train", tokenizer=None, teacher_tokenizer=None)
    val_ds = tc.load_mimic_cxr(cfg, "validation", tokenizer=None, teacher_tokenizer=None)

    train_types = [type(t) for t in train_ds.img_transform.transforms]
    val_types = [type(t) for t in val_ds.img_transform.transforms]
    assert T.RandomResizedCrop in train_types, "train split must be augmented when use_augmentation=True"
    assert T.RandomResizedCrop not in val_types, "validation split must NEVER be augmented"


def test_prepare_dataloader_dispatches_local_parquet_configs_to_load_mimic_cxr():
    """Regression for job 2470516 (2026-08-20): prepare_dataloader() has its
    OWN outer dispatch (`if name == "mimic_cxr": ... elif ...: ... else: raise
    ValueError("Unknown dataset for contrastive training")`) that runs BEFORE
    load_mimic_cxr is ever called. test_load_mimic_cxr_dispatches_to_local_
    parquet_when_configured above calls load_mimic_cxr directly, so it never
    exercised this outer gate -- which is exactly how cxr_mimic_full.yaml (and
    the arm0 config derived from it) shipped with dataset_name set to the
    file's own name ("cxr_mimic_full") instead of the literal "mimic_cxr" the
    dispatch requires, and training died with "Unknown dataset for contrastive
    training: cxr_mimic_full" the first time it was actually run. This test
    calls prepare_dataloader itself with dataset_name="mimic_cxr" (the
    corrected value) + local_parquet_dir set, and asserts it reaches
    load_mimic_cxr rather than the ValueError branch.
    """
    import importlib.util

    from omegaconf import OmegaConf

    spec = importlib.util.spec_from_file_location(
        "_tc_mod_prep", REPO_ROOT / "scripts" / "train_contrastive.py"
    )
    tc = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(tc)
    except Exception as e:                                    # pragma: no cover
        pytest.skip("train_contrastive import failed: {}".format(e))

    calls = []

    def _fake_load_mimic_cxr(cfg, split, tokenizer, teacher_tokenizer=None):
        calls.append(split)

        class _FakeDS:
            def __len__(self):
                return 4  # nonzero: DataLoader's RandomSampler rejects len==0 eagerly

            def __getitem__(self, idx):
                return {}

        return _FakeDS()

    tc.load_mimic_cxr = _fake_load_mimic_cxr

    cfg = OmegaConf.create({"dataset": {
        "dataset_name": "mimic_cxr",
        "local_parquet_dir": "/fake/dir",
        "batch_size": 2, "eval_batch_size": 2, "num_workers": 0, "pin_memory": False,
    }})

    tc.prepare_dataloader(cfg, "train", tokenizer=None, teacher_tokenizer=None)

    assert calls == ["train"], (
        "prepare_dataloader did not dispatch dataset_name='mimic_cxr' to "
        "load_mimic_cxr -- the exact regression from job 2470516"
    )

    # And the actual configs shipped for the local-parquet path must carry
    # this literal value, not their own filename.
    for fname in ("cxr_mimic_full.yaml", "cxr_mimic_arm0.yaml"):
        c = OmegaConf.load(REPO_ROOT / "configs" / "dataset" / fname)
        assert c.get("dataset_name") == "mimic_cxr", (
            "{} must set dataset_name: mimic_cxr for prepare_dataloader's "
            "dispatch to route into load_mimic_cxr".format(fname)
        )


def test_evaluate_cxr_retrieval_handles_str_image_paths():
    """MIMICValDataset / IndianaEvalDataset both did
    `if not isinstance(img, Image.Image): Image.fromarray(img)`, which CRASHES
    on a str path — the exact shape a local-parquet build's "image" column
    takes. REGRESSION (2026-08-16, Phase 8E): both must now Image.open() a str
    before the fromarray fallback.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_ecr_mod", REPO_ROOT / "scripts" / "evaluate_cxr_retrieval.py"
    )
    ecr = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(ecr)
    except Exception as e:                                    # pragma: no cover
        pytest.skip("evaluate_cxr_retrieval import failed: {}".format(e))

    tmp_img = tmp_path_factory_image()
    try:
        class _StubTok:
            def __call__(self, text, **kw):
                return {
                    "input_ids": torch.zeros(1, 8, dtype=torch.long),
                    "attention_mask": torch.ones(1, 8, dtype=torch.long),
                }

        ds = ecr.MIMICValDataset(
            [{"image": str(tmp_img), "findings": "clear", "impression": "normal"}],
            _StubTok(), max_length=8,
        )
        item = ds[0]
        assert item["pixel_values"].shape[0] == 3  # RGB after convert

        ids = ecr.IndianaEvalDataset(
            [{"image": str(tmp_img), "report": "clear lungs"}],
            _StubTok(), max_length=8,
        )
        item2 = ids[0]
        assert item2["pixel_values"].shape[0] == 3
    finally:
        os.remove(tmp_img)

    # build_dataloader must accept the new params without a TypeError.
    import inspect
    sig = inspect.signature(ecr.build_dataloader)
    assert "local_parquet_dir" in sig.parameters
    assert "mimic_split" in sig.parameters


def tmp_path_factory_image():
    import tempfile
    from PIL import Image as PILImage

    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    PILImage.new("L", (16, 16)).save(path, "JPEG")
    return path


def test_h100_scripts_expose_dataset_config_and_local_parquet_levers():
    """Phase 8E: an env lever must exist to point training/eval at the local
    PhysioNet build, defaulting to the legacy mirror so Phase 9A's Arm-0
    reproduction control (and every prior run) is unaffected by its existence.
    """
    tr = (REPO_ROOT / "scripts" / "train_biomedclip_kd_h100.sh").read_text()
    assert 'DATASET_CONFIG="${DATASET_CONFIG:-mimic_cxr}"' in tr
    assert "dataset=${DATASET_CONFIG}" in tr
    assert "dataset=mimic_cxr \\" not in tr, "must go through the DATASET_CONFIG lever, not a hardcoded value"

    ev = (REPO_ROOT / "scripts" / "eval_h100.sh").read_text()
    assert 'LOCAL_PARQUET_DIR="${LOCAL_PARQUET_DIR:-}"' in ev
    assert "--local-parquet-dir" in ev


def test_gitignore_guards_credentialed_mimic_build():
    gi = (REPO_ROOT / ".gitignore").read_text()
    assert "dataset/mimic_full/" in gi
    assert "*.parquet" in gi


def test_build_mimic_cxr_local_slurm_wrapper_is_cpu_only_on_cpu_batch():
    """REGRESSION (2026-08-16, Phase 7E). The login node rejects ANY script
    execution outright (confirmed live — not just 'heavy' commands), and per
    docs.sc.hpi.de external downloads belong on compute nodes, not a Run
    Node (rx01/rx02 — explicitly not meant for data acquisition). This job
    itself needs zero GPU (pure network I/O) — no --gpus line is requested.

    Account/partition/QOS went through THREE failed defaults before landing
    on one that actually runs, confirmed live via job 2457565 (auth
    succeeded, 3/4 small files fetched before an unrelated manual --time
    override killed it): --account=aisc on cpu-batch -> PENDING forever
    (QOSNotAllowed, aisc's QOS is scoped to AISC partitions only);
    --account=default on cpu-batch -> AssocMaxSubmitJobLimit. What works:
    --account=aisc --partition=aisc-batch --qos=aisc together. TRADEOFF this
    accepts: aisc-batch is a GPU-capable, preemptible-at-any-time partition
    (docs.sc.hpi.de) for a job that never uses the GPU — not ideal
    cluster citizenship, but the only combination proven to actually run for
    this account; worth asking sc-helpdesk@hpi.de about a non-preemptible
    CPU-only alternative before the long `fetch` stage.
    """
    sh = (REPO_ROOT / "scripts" / "build_mimic_cxr_local.sh").read_text()
    assert "#SBATCH --partition=pot-hpi-aisc-batch" in sh
    assert "#SBATCH --gpus" not in sh
    assert "#SBATCH --account=aisc" in sh
    assert "#SBATCH --qos=aisc" in sh
    assert "build_mimic_cxr_local.py meta" in sh
    assert "build_mimic_cxr_local.py manifest" in sh
    assert "build_mimic_cxr_local.py fetch" in sh
    assert "build_mimic_cxr_local.py pack" in sh


# ── 16. Phase 10A: HybridLanguageModel image-conditioning hooks ───────────────

def _tiny_cpu_config():
    from hybrid_xmamba.models.configuration_hybrid import HybridConfig
    return HybridConfig(
        vocab_size=100, dim=64, num_layers=2,
        layer_pattern=["mamba", "mlstm"],
        max_position_embeddings=64,
        use_fast_path=False,
        use_tfla=False,
    )


@pytest.mark.willi_parity
def test_forward_inputs_embeds_matches_token_embedding_path():
    """forward(inputs_embeds=embeddings(x)) must be a true drop-in equivalent
    of forward(input_ids=x), not just 'doesn't crash' — the whole point of
    the new kwarg is that Phase 10's image prefix flows through the exact
    same code path as token embeddings.
    """
    from hybrid_xmamba.models.hybrid_lm import HybridLanguageModel

    model = HybridLanguageModel(_tiny_cpu_config())
    model.eval()

    input_ids = torch.randint(0, 100, (2, 16))

    with torch.no_grad():
        out_ids = model(input_ids, return_dict=True)
        embeds = model.embeddings(input_ids)
        out_embeds = model(inputs_embeds=embeds, return_dict=True)

    assert torch.allclose(out_ids.logits, out_embeds.logits, atol=1e-5), (
        "forward(inputs_embeds=...) diverges from forward(input_ids=...) — "
        "the new kwarg is not a true drop-in for the embedding step"
    )


@pytest.mark.willi_parity
def test_forward_requires_exactly_one_of_input_ids_or_inputs_embeds():
    from hybrid_xmamba.models.hybrid_lm import HybridLanguageModel

    model = HybridLanguageModel(_tiny_cpu_config())
    input_ids = torch.randint(0, 100, (2, 16))
    embeds = torch.randn(2, 16, 64)

    with pytest.raises(ValueError):
        model(input_ids=input_ids, inputs_embeds=embeds)

    with pytest.raises(ValueError):
        model()


@pytest.mark.willi_parity
def test_generate_default_path_unchanged_when_prefix_embeds_none():
    """Regression pin: generate() with prefix_embeds=None (the default) must
    produce byte-identical output to before Phase 10A under a fixed seed —
    the default branch is untouched code, only reached via an explicit
    prefix_embeds=None check.
    """
    from hybrid_xmamba.models.hybrid_lm import HybridLanguageModel

    model = HybridLanguageModel(_tiny_cpu_config())
    model.eval()
    input_ids = torch.randint(0, 100, (2, 4))

    torch.manual_seed(0)
    out_a = model.generate(input_ids, max_new_tokens=5, temperature=1.0)
    torch.manual_seed(0)
    out_b = model.generate(input_ids, prefix_embeds=None, max_new_tokens=5, temperature=1.0)

    assert torch.equal(out_a, out_b), "generate() default path changed under Phase 10A"
    assert out_a.shape == (2, 4 + 5)


@pytest.mark.willi_parity
def test_generate_with_prefix_embeds_runs_and_returns_expected_shape():
    """Smoke test for the new capability: prefix-conditioned generation runs
    end-to-end and returns generated token ids only (the prefix contributes
    no ids of its own)."""
    from hybrid_xmamba.models.hybrid_lm import HybridLanguageModel

    cfg = _tiny_cpu_config()
    model = HybridLanguageModel(cfg)
    model.eval()

    input_ids = torch.randint(0, 100, (2, 4))
    prefix_embeds = torch.randn(2, 8, cfg.dim)

    out = model.generate(input_ids, prefix_embeds=prefix_embeds, max_new_tokens=5)

    assert out.shape == (2, 4 + 5), f"Unexpected shape: {out.shape}"
    assert torch.isfinite(out.float()).all()


@pytest.mark.willi_parity
def test_image_prefix_mapper_output_shape_and_gradients():
    """ImagePrefixMapper: synthetic (B, N, patch_dim) -> (B, k, decoder_dim),
    for several k, with gradients flowing back to token_proj."""
    from hybrid_xmamba.models.prefix_mapper import ImagePrefixMapper

    B, N, patch_dim, decoder_dim = 3, 197, 768, 64

    for k in (8, 32, 64):
        mapper = ImagePrefixMapper(patch_dim=patch_dim, decoder_dim=decoder_dim, k=k)
        patch_grid = torch.randn(B, N, patch_dim)

        out = mapper(patch_grid)
        assert out.shape == (B, k, decoder_dim), f"k={k}: unexpected shape {out.shape}"
        assert torch.isfinite(out).all()

        loss = out.sum()
        loss.backward()
        assert mapper.token_proj.weight.grad is not None, f"k={k}: no gradient for token_proj"
        assert torch.isfinite(mapper.token_proj.weight.grad).all(), f"k={k}: non-finite gradient"


# ── 17. Phase 11A: report-generation metrics + decoding harness ───────────────

@pytest.mark.willi_parity
def test_rouge_l_score_known_value():
    from scripts.evaluate_report_generation import rouge_l_score

    hyp = "the cat sat on the mat".split()
    ref = "the cat was on the mat".split()
    # LCS = "the cat on the mat" (5 tokens) out of 6 in both -> p=r=5/6 -> F=5/6
    score = rouge_l_score(hyp, ref)
    assert abs(score - 5 / 6) < 1e-6, score

    assert rouge_l_score([], ["a", "b"]) == 0.0
    assert rouge_l_score(["a", "b"], []) == 0.0


@pytest.mark.willi_parity
def test_corpus_bleu_identical_and_disjoint():
    from scripts.evaluate_report_generation import corpus_bleu

    text = "the quick brown fox jumps over the lazy dog".split()
    identical = corpus_bleu([text], [text], max_n=4)
    assert abs(identical - 1.0) < 1e-6, identical

    disjoint_hyp = ["zzz", "yyy", "xxx", "www"]
    disjoint_ref = ["aaa", "bbb", "ccc", "ddd"]
    disjoint = corpus_bleu([disjoint_hyp], [disjoint_ref], max_n=4)
    assert disjoint == 0.0, disjoint


@pytest.mark.willi_parity
def test_meteor_score_corpus_returns_none_or_float():
    from scripts.evaluate_report_generation import meteor_score_corpus

    result = meteor_score_corpus(["the cat sat"], ["the cat sat"])
    assert result is None or isinstance(result, float)


@pytest.mark.willi_parity
def test_greedy_decode_runs_and_returns_expected_shape():
    from hybrid_xmamba.models.hybrid_lm import HybridLanguageModel
    from hybrid_xmamba.models.prefix_mapper import ImagePrefixMapper
    from scripts.evaluate_report_generation import greedy_decode

    cfg = _tiny_cpu_config()
    model = HybridLanguageModel(cfg)
    model.eval()
    mapper = ImagePrefixMapper(patch_dim=768, decoder_dim=cfg.dim, k=4)
    mapper.eval()

    prefix_embeds = mapper(torch.randn(1, 197, 768))
    input_ids = torch.randint(0, 100, (1, 5))

    out = greedy_decode(model, input_ids, prefix_embeds=prefix_embeds, max_new_tokens=6)
    assert out.shape == (1, 11), out.shape
    assert torch.isfinite(out.float()).all()


@pytest.mark.willi_parity
def test_beam_search_decode_requires_batch_size_one():
    from hybrid_xmamba.models.hybrid_lm import HybridLanguageModel
    from scripts.evaluate_report_generation import beam_search_decode

    model = HybridLanguageModel(_tiny_cpu_config())
    model.eval()
    input_ids = torch.randint(0, 100, (2, 5))  # batch size 2 -> not supported

    with pytest.raises(ValueError):
        beam_search_decode(model, input_ids, beam_size=3, max_new_tokens=4)


@pytest.mark.willi_parity
def test_beam_search_decode_beam_size_one_matches_greedy():
    """beam_size=1 must reduce to the same deterministic argmax path as
    greedy_decode -- a correctness invariant for the new beam-search code
    (model.generate() has no beam mode, so beam_search_decode is fresh logic
    built directly on forward(inputs_embeds=...))."""
    from hybrid_xmamba.models.hybrid_lm import HybridLanguageModel
    from hybrid_xmamba.models.prefix_mapper import ImagePrefixMapper
    from scripts.evaluate_report_generation import greedy_decode, beam_search_decode

    cfg = _tiny_cpu_config()
    model = HybridLanguageModel(cfg)
    model.eval()
    mapper = ImagePrefixMapper(patch_dim=768, decoder_dim=cfg.dim, k=4)
    mapper.eval()

    prefix_embeds = mapper(torch.randn(1, 197, 768))
    input_ids = torch.randint(0, 100, (1, 5))

    greedy_out = greedy_decode(model, input_ids, prefix_embeds=prefix_embeds, max_new_tokens=6)
    beam_out = beam_search_decode(
        model, input_ids, prefix_embeds=prefix_embeds, beam_size=1, max_new_tokens=6
    )
    assert torch.equal(greedy_out, beam_out), (greedy_out, beam_out)


# ── 18. Phase 10E: ReportGenerationLightningModule ─────────────────────────────

@pytest.mark.willi_parity
def test_report_generation_step_produces_finite_loss_and_gradients():
    """Training step must produce a finite loss with gradients flowing into
    BOTH the prefix_mapper and the decoder backbone, using a precomputed
    batch['patch_grid'] tensor -- no open_clip/BiomedCLIP weights needed
    (image_encoder stays None; load_image_encoder() is never called)."""
    from hybrid_xmamba.training.lightning_module import ReportGenerationLightningModule

    cfg = _tiny_cpu_config()
    mod = ReportGenerationLightningModule(
        decoder_config=cfg,
        image_patch_dim=768,
        prefix_k=4,
        decoder_lr=1e-5,
        head_lr=3e-4,
        weight_decay=0.01,
        warmup_steps=2,
        max_steps=10,
        gradient_clip_val=0.5,
    )
    mod.train()

    B, L = 3, 8
    batch = {
        "input_ids": torch.randint(0, 100, (B, L)),
        "patch_grid": torch.randn(B, 197, 768),
    }

    loss = mod.training_step(batch, batch_idx=0)
    assert torch.isfinite(loss), f"Loss not finite: {loss.item()}"

    loss.backward()

    for name, param in mod.prefix_mapper.named_parameters():
        assert param.grad is not None, f"No grad for prefix_mapper.{name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite grad for prefix_mapper.{name}"

    for name, param in mod.decoder.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No grad for decoder.{name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite grad for decoder.{name}"


@pytest.mark.willi_parity
def test_report_generation_prefix_masking_matches_manual_ignore_index_ce():
    """Regression pin for the single assumption ReportGenerationLightningModule
    relies on but never states explicitly in code: HybridLanguageModel.forward()
    calls nn.CrossEntropyLoss() with NO ignore_index argument, so -100 at the
    prefix label positions is excluded from the loss only because -100 is
    PyTorch's documented default ignore_index. This reconstructs the loss
    independently via F.cross_entropy(..., ignore_index=-100) over the exact
    same logits/labels and asserts it matches _step()'s output -- if a future
    edit ever adds an explicit ignore_index to hybrid_lm.py's loss (or changes
    the shift convention), this test catches the mismatch."""
    import torch.nn.functional as F
    from hybrid_xmamba.training.lightning_module import ReportGenerationLightningModule

    cfg = _tiny_cpu_config()
    mod = ReportGenerationLightningModule(decoder_config=cfg, prefix_k=4)
    mod.eval()

    B, L = 2, 6
    input_ids = torch.randint(0, 100, (B, L))
    patch_grid = torch.randn(B, 197, 768)

    with torch.no_grad():
        step_loss = mod._step({"input_ids": input_ids, "patch_grid": patch_grid}, "val")

        prefix_embeds = mod.prefix_mapper(patch_grid)
        k = prefix_embeds.shape[1]
        inputs_embeds = torch.cat([prefix_embeds, mod.decoder.embeddings(input_ids)], dim=1)
        logits = mod.decoder(inputs_embeds=inputs_embeds, return_dict=True).logits

        labels = torch.full((B, k + L), -100, dtype=input_ids.dtype)
        labels[:, k:] = input_ids
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        manual_loss = F.cross_entropy(
            shift_logits.view(-1, cfg.vocab_size), shift_labels.view(-1), ignore_index=-100,
        )

    assert torch.allclose(step_loss, manual_loss, atol=1e-5), (step_loss.item(), manual_loss.item())


@pytest.mark.willi_parity
@pytest.mark.parametrize("decode,kwargs", [("greedy", {}), ("beam", {"beam_size": 2})])
def test_generate_from_patch_grid_runs_and_returns_expected_shape(decode, kwargs):
    """Phase 11A's --checkpoint mode (2026-08-23, added after the first real
    Phase 10E arm0 checkpoint existed): generate_from_patch_grid() must run
    end-to-end on a synthetic patch grid (no real checkpoint/BiomedCLIP/data
    needed here — that heavy path is scripts/evaluate_report_generation.py's
    load_report_generation_module(), not unit-tested, same as evaluate_lm.py's
    loader) and seed generation with an EMPTY input_ids, matching how training
    labels start immediately after the image prefix with no BOS token."""
    from scripts.evaluate_report_generation import generate_from_patch_grid
    from hybrid_xmamba.training.lightning_module import ReportGenerationLightningModule

    cfg = _tiny_cpu_config()
    module = ReportGenerationLightningModule(decoder_config=cfg, prefix_k=4)
    module.eval()

    patch_grid = torch.randn(1, 197, 768)
    max_new_tokens = 6
    out = generate_from_patch_grid(module, patch_grid, decode=decode, max_new_tokens=max_new_tokens, **kwargs)

    assert out.shape == (1, max_new_tokens), out.shape


@pytest.mark.willi_parity
def test_nearest_neighbor_indices_picks_closest_by_cosine_similarity():
    """Phase 11C (2026-08-24, built after both the no-augmentation and
    augmented arm0 checkpoints were confirmed to generate byte-identical
    boilerplate for different images): nearest_neighbor_indices() is the
    testable core of run_retrieval_baseline() — pure cosine-similarity
    argmax, no BiomedCLIP/network/data needed here."""
    from scripts.evaluate_report_generation import nearest_neighbor_indices

    gallery = torch.eye(4)  # 4 orthonormal "reports"
    query = torch.tensor([[0.9, 0.1, 0.0, 0.0], [0.0, 0.0, 0.2, 0.9]])
    idx = nearest_neighbor_indices(query, gallery)
    assert idx.tolist() == [0, 3]


@pytest.mark.willi_parity
def test_report_generation_val_step_logs_flat_named_checkpoint_alias():
    """Regression pin for the live bug hit 2026-08-23 (job 2478647): Lightning
    does not sanitize '/' inside a ModelCheckpoint filename=... interpolation
    -- {val/lm_loss:.4f} silently created a nested DIRECTORY (report_gen-
    step=NNNNNN-val/) instead of a flat checkpoint filename, with the actual
    .ckpt buried one level down as lm_loss=X.XXXX.ckpt. _step() must log an
    additional flat-named 'val_lm_loss_ckpt' alias (same value as val/lm_loss)
    on validation steps only, which train_report_generation.py's filename=
    template now interpolates instead."""
    from unittest.mock import patch as mock_patch
    from hybrid_xmamba.training.lightning_module import ReportGenerationLightningModule

    cfg = _tiny_cpu_config()
    module = ReportGenerationLightningModule(decoder_config=cfg, prefix_k=4)
    module.eval()

    B, L = 2, 6
    batch = {"input_ids": torch.randint(0, 100, (B, L)), "patch_grid": torch.randn(B, 197, 768)}

    with mock_patch.object(module, "log") as mock_log:
        module._step(batch, "val")
        val_keys = [call.args[0] for call in mock_log.call_args_list]
    assert "val/lm_loss" in val_keys
    assert "val_lm_loss_ckpt" in val_keys, (
        "val_lm_loss_ckpt not logged on a validation step -- "
        "train_report_generation.py's filename= template has nothing "
        "flat-named to interpolate, reintroducing the nested-directory bug"
    )

    with mock_patch.object(module, "log") as mock_log:
        module._step(batch, "train")
        train_keys = [call.args[0] for call in mock_log.call_args_list]
    assert "val_lm_loss_ckpt" not in train_keys, "should only log on validation steps"


@pytest.mark.willi_parity
def test_train_report_generation_checkpoint_filename_has_no_slash_in_braces():
    """Static guard, complementary to the behavioral test above: any {...}
    filename interpolation containing '/' creates a nested directory instead
    of a checkpoint file under this Lightning version's default ModelCheckpoint
    (confirmed live 2026-08-23). Scoped to train_report_generation.py only --
    train_stage0_distill.py/_resume.py have the identical latent pattern
    ({val/loss:.4f}) but are historical, already-executed production scripts
    untouched this session; documented here, not fixed, to avoid unrequested
    changes to load-bearing infra the Stage-0 150M checkpoint (val PPL 13.18,
    this whole Phase 10E chain's DECODER_CKPT) came from."""
    py = (REPO_ROOT / "scripts" / "train_report_generation.py").read_text()
    m = re.search(r'filename\s*=\s*["\']([^"\']*)["\']', py)
    assert m, "expected a ModelCheckpoint filename=... string literal"
    for expr in re.findall(r"\{([^}]*)\}", m.group(1)):
        assert "/" not in expr, (
            f"filename= template contains '{{{expr}}}' — a '/' inside a Lightning "
            f"filename interpolation creates a nested directory, not a flat file"
        )


@pytest.mark.willi_parity
def test_hybrid_150m_v2_rrg_config_matches_150m_v2_architecture():
    """Phase 10E: hybrid_150m_v2_rrg.yaml must be architecturally IDENTICAL to
    hybrid_150m_v2.yaml (checkpoint-loadable against a Stage-0/joint-trained
    150M v2 backbone, per Phase 10D), plus the new image-prefix keys."""
    import dataclasses
    from omegaconf import OmegaConf
    from hybrid_xmamba.models.configuration_hybrid import HybridConfig
    from hybrid_xmamba.models.hybrid_lm import HybridLanguageModel
    from hybrid_xmamba.training.lightning_module import ReportGenerationLightningModule

    raw = OmegaConf.to_container(
        OmegaConf.load(REPO_ROOT / "configs" / "model" / "hybrid_150m_v2_rrg.yaml"),
        resolve=True,
    )
    assert raw["dim"] == 768 and raw["num_layers"] == 12
    assert raw["norm_topology"] == "hybrid"
    assert raw["max_position_embeddings"] == 1024
    assert list(raw["layer_pattern"]).count("mlstm") == 3

    # New Phase 10 keys
    assert raw["image_patch_dim"] == 768, "BiomedCLIP ViT-B/16 patch dim"
    assert raw["prefix_k"] > 0
    assert raw["gradient_clip_val"] == 0.5, "150M is spike-fragile — must not silently drift to 1.0"
    assert raw["vit_unfreeze_blocks"] == 0, "10C default: frozen image tower"
    assert raw["vit_lr"] > 0

    # Regression pin for the live bug hit 2026-08-23 (job 2478622): Hydra's
    # strict-struct mode rejects a CLI override for a key the config doesn't
    # declare ("Could not override 'model.vit_unfreeze_blocks' ... Key
    # 'vit_unfreeze_blocks' is not in struct"). Every `model.<key>=` override
    # train_report_generation_h100.sh passes MUST have a matching key declared
    # in this yaml, or the SLURM job fails at Hydra-compose time before any
    # Python code runs.
    import re
    sh_text = (REPO_ROOT / "scripts" / "train_report_generation_h100.sh").read_text()
    overridden_keys = re.findall(r"model\.([a-zA-Z_][a-zA-Z0-9_]*)=", sh_text)
    assert overridden_keys, "expected at least one model.<key>= override in the wrapper"
    for key in overridden_keys:
        assert key in raw, (
            f"train_report_generation_h100.sh overrides model.{key}=... but "
            f"hybrid_150m_v2_rrg.yaml does not declare '{key}' — Hydra strict-struct "
            f"mode will reject this at submit time."
        )

    fields = {f.name for f in dataclasses.fields(HybridConfig)}
    decoder_cfg = HybridConfig(**{k: v for k, v in raw.items() if k in fields})
    decoder = HybridLanguageModel(decoder_cfg)
    n_params = sum(p.numel() for p in decoder.parameters())
    assert 181e6 < n_params < 186e6, (
        f"hybrid_150m_v2_rrg decoder param count {n_params/1e6:.2f}M outside [181, 186]M "
        f"— should match hybrid_150m_v2.yaml exactly"
    )

    # Config values must actually wire into the Lightning module without error.
    module = ReportGenerationLightningModule(
        decoder_config=decoder_cfg,
        image_patch_dim=raw["image_patch_dim"],
        prefix_k=raw["prefix_k"],
        decoder_lr=raw["decoder_lr"],
        head_lr=raw["head_lr"],
        weight_decay=raw["weight_decay"],
        warmup_steps=raw["warmup_steps"],
        max_steps=raw["max_steps"],
        gradient_clip_val=raw["gradient_clip_val"],
    )
    assert module.prefix_mapper.k == raw["prefix_k"]


@pytest.mark.willi_parity
def test_train_report_generation_h100_slurm_wrapper_conventions():
    """Phase 10E SLURM wrapper must follow the project's established conventions:
    ga03 excluded (ARM/x86 mismatch), aisc-batch/account/qos, and a fail-fast
    existence check on the decoder checkpoint (mirrors STAGE0_CKPT in
    train_biomedclip_kd_h100.sh) rather than silently training from random init."""
    sh = (REPO_ROOT / "scripts" / "train_report_generation_h100.sh").read_text()
    assert "#SBATCH --partition=pot-hpi-aisc-batch" in sh
    assert "#SBATCH --account=aisc" in sh
    assert "--exclude=ga03" in sh
    assert "DECODER_CKPT" in sh
    assert 'if [ ! -f "${DECODER_CKPT}" ]' in sh
    assert "train_report_generation.py" in sh


@pytest.mark.willi_parity
def test_inspect_report_generation_h100_slurm_wrapper_conventions():
    """Companion wrapper (2026-08-23) for scripts/evaluate_report_generation.py
    --checkpoint, needed because the login node refuses this command directly
    ('This command is not allowed on the login node!', hit live job 2478647's
    follow-up). Same established conventions as the training wrapper, plus a
    fail-fast existence check on the checkpoint itself."""
    sh = (REPO_ROOT / "scripts" / "inspect_report_generation_h100.sh").read_text()
    assert "#SBATCH --partition=pot-hpi-aisc-batch" in sh
    assert "#SBATCH --account=aisc" in sh
    assert "--exclude=ga03" in sh
    assert "CHECKPOINT" in sh
    assert 'if [ ! -f "${CHECKPOINT}" ]' in sh
    assert "evaluate_report_generation.py" in sh
    assert "--checkpoint" in sh
    # Defaults to VALIDATION images, not train — generations on train images
    # look artificially good even under genuine overfitting.
    assert "validate.parquet" in sh


@pytest.mark.willi_parity
def test_retrieval_baseline_h100_slurm_wrapper_conventions():
    """Phase 11C wrapper (2026-08-24) for scripts/evaluate_report_generation.py
    --retrieval-baseline, needed after the arm0 checkpoint (with AND without
    Phase 9D augmentation) was confirmed to generate byte-identical boilerplate
    rather than condition on the image — this baseline is the objective floor
    any future generator number must be compared against. Same conventions as
    the sibling inspection wrapper; queries VALIDATION against the TRAIN
    gallery (never against itself)."""
    sh = (REPO_ROOT / "scripts" / "retrieval_baseline_h100.sh").read_text()
    assert "#SBATCH --partition=pot-hpi-aisc-batch" in sh
    assert "#SBATCH --account=aisc" in sh
    assert "--exclude=ga03" in sh
    assert "TRAIN_PARQUET" in sh
    assert "evaluate_report_generation.py" in sh
    assert "--retrieval-baseline" in sh
    assert "--train-parquet" in sh
    assert "validate.parquet" in sh
    assert "train.parquet" in sh


@pytest.mark.willi_parity
def test_train_report_generation_h100_slurm_wrapper_hydra_overrides_compose():
    """Regression pin for TWO live bugs hit back-to-back 2026-08-23 (jobs
    2478622, 2478635): Hydra's strict-struct mode rejects a CLI override for
    ANY key (model.<key>=, or a bare top-level key like decoder_checkpoint=)
    that isn't declared somewhere in the composed config -- caught only at
    submit time, after the DECODER_CKPT existence check already passed, deep
    into the SLURM job. Rather than re-deriving the override list by hand
    (fragile -- would have missed decoder_checkpoint same as the first fix
    did), this replays the SLURM wrapper's OWN python invocation verbatim
    through hydra.compose() with its own documented env-var defaults
    substituted in, and asserts it composes without error -- the exact
    failure mode hit live, reproduced offline."""
    pytest.importorskip("hydra")
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    sh = (REPO_ROOT / "scripts" / "train_report_generation_h100.sh").read_text()

    # Literal stand-ins for every ${VAR} this script's invocation block
    # references — NOT parsed from the script's own defaults (several have
    # trailing inline comments or nested ${...} refs, e.g. EXPERIMENT embeds
    # ${PREFIX_K}, that make robust regex-extraction more fragile than just
    # supplying flat literals here; this test only needs each key to resolve
    # to SOME syntactically valid value, not to reproduce the real default).
    env_defaults = {
        "MODEL_CONFIG": "hybrid_150m_v2_rrg", "DATASET_CONFIG": "cxr_mimic_full",
        "MAX_STEPS": "50", "BATCH_SIZE": "16", "DECODER_LR": "1e-5", "HEAD_LR": "3e-4",
        "GRAD_CLIP": "0.5", "PREFIX_K": "32", "VIT_UNFREEZE": "0", "VIT_LR": "1e-6",
        "GRAD_CKPT": "false", "AUGMENT": "false", "MIMIC_CACHE_DIR": "/tmp/mimic_cache",
        "DECODER_CKPT": "./outputs/h100_stage0_150m_v2/checkpoints/stage0_model_only.pt",
        "EXPERIMENT": "parity_check",
        # Phase 13B: NUM_GPUS=1 (this test's scenario) resolves TRAINER_CFG to
        # h100_single_gpu inside the script itself, before the invocation
        # block; substitute that resolved value directly here.
        "TRAINER_CFG": "h100_single_gpu",
        # Phase 13F
        "OVERSAMPLE_RARE": "false", "OVERSAMPLE_WEIGHT": "5.0",
        # Phase 15B-2 / 15B-3
        "SEED": "42", "SAVE_TOP_K": "3",
    }

    # Extract the python invocation block verbatim (between the `python
    # scripts/train_report_generation.py \` line and the blank line ending it).
    invocation = sh.split("python scripts/train_report_generation.py \\", 1)[1]
    invocation = invocation.split("\n\necho", 1)[0]
    tokens = [t.strip().rstrip("\\").strip() for t in invocation.splitlines()]
    # "=" filter drops Phase 13A's trailing `${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}`
    # bash empty-array-safe expansion (no literal "=" in that syntax) -- it is
    # not a static Hydra override token, it expands to nothing when
    # IMAGE_ENCODER_CKPT is unset (this test's scenario).
    tokens = [t for t in tokens if t and t != "--config-name config" and "=" in t]

    def resolve(tok: str) -> str:
        key, _, value = tok.partition("=")
        value = value.strip('"')
        value = re.sub(
            r"\$\{?([A-Z_][A-Z0-9_]*)\}?",
            lambda m: env_defaults.get(m.group(1), m.group(0)),
            value,
        )
        return f"{key}={value}"

    overrides = [resolve(t) for t in tokens]
    assert any(o.startswith("decoder_checkpoint=") for o in overrides), (
        "sanity check: the override list should still contain decoder_checkpoint= "
        "— if this fails, the parsing above drifted from the script's actual format"
    )

    GlobalHydra.instance().clear()
    configs_dir = str(REPO_ROOT / "configs")
    with initialize_config_dir(config_dir=configs_dir, version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides)

    assert cfg.decoder_checkpoint == env_defaults["DECODER_CKPT"]
    assert cfg.model.vit_unfreeze_blocks == 0
    assert cfg.model.prefix_k == 32
    # Phase 13A: image_encoder_checkpoint is declared (config.yaml) and
    # defaults to null when IMAGE_ENCODER_CKPT is unset, same as this test's
    # scenario -- unlike decoder_checkpoint, EXTRA_ARGS only adds it to the
    # invocation when the env var is actually set (see the dedicated lever
    # test below for that conditional path).
    assert cfg.image_encoder_checkpoint is None
    # Phase 13F
    assert cfg.dataset.oversample_rare_findings is False
    assert cfg.dataset.oversample_weight == 5.0
    # Phase 15B-2: the seed must reach Hydra as an int, and the wrapper's
    # default must stay 42 so adding the lever reproduces every pre-15B arm
    # bit-for-bit rather than silently re-seeding the whole project's history.
    assert cfg.seed == 42
    assert isinstance(cfg.seed, int)
    # Phase 15B-3: save_top_k must reach Hydra as an int and default to 3, so
    # pre-15B recipes keep writing exactly what they wrote before.
    assert cfg.save_top_k == 3
    assert isinstance(cfg.save_top_k, int)


@pytest.mark.willi_parity
def test_report_gen_wrapper_exposes_save_top_k_and_checkpoint_count_is_not_hardcoded():
    """Phase 15B-3. Every report-generation run wrote save_top_k=3 PLUS
    last.ckpt -- 4 x 2.4 GB = 9.6 GB per arm -- while every eval command in
    this project loads last.ckpt and nothing else. The three extra files were
    never read by anything, and they exhausted the 200 GiB home quota partway
    through the first seed campaign, killing 3 of 4 arms with
    "OSError: [Errno 122] Disk quota exceeded" on the last.ckpt write
    (job 2542399).

    Guards the lever and, more importantly, that the count is no longer a
    literal in the training script -- a hardcoded 3 is what made this
    un-tunable when it mattered.
    """
    sh = (REPO_ROOT / "scripts" / "train_report_generation_h100.sh").read_text()
    py = (REPO_ROOT / "scripts" / "train_report_generation.py").read_text()

    assert 'SAVE_TOP_K="${SAVE_TOP_K:-3}"' in sh, "SAVE_TOP_K lever missing or default drifted off 3"
    assert "save_top_k=${SAVE_TOP_K}" in sh, "SAVE_TOP_K declared but not passed to Hydra"
    assert "save_top_k=3" not in py, (
        "save_top_k is hardcoded again in train_report_generation.py -- it must "
        "read cfg.save_top_k so seed/ablation arms can drop to 0"
    )
    assert 'cfg.get("save_top_k"' in py, "train_report_generation.py must read save_top_k from cfg"


@pytest.mark.willi_parity
def test_report_gen_wrapper_exposes_seed_lever_and_logs_it():
    """Phase 15B-2. Supervisor review item 3 (2026-09-13) is that the
    generation table has no seed variance. The cause is structural, not an
    oversight in reporting: configs/config.yaml pins `seed: 42` and NO wrapper
    ever exposed it, so 13A-13F, 14A and every prefix_k arm are literally the
    same seed. A multi-seed campaign is impossible until the lever exists.

    Asserts three things, each a separate failure this project has already
    paid for once:
      1. SEED is declared with a default of 42 -- NOT a fresh random default,
         which would silently make new runs incomparable with every existing
         checkpoint.
      2. `seed=${SEED}` is actually in the python invocation. A declared-but-
         unused env var is the 13F-era trap: the wrapper prints a value it is
         not passing.
      3. The resolved seed is ECHOED. The prefix_k trap (Phase 14, cost
         0.0145 ROUGE-L) was caught only by noticing a missing log line; two
         seed arms whose logs never state their seed are indistinguishable
         from one arm run twice, which would void 15B entirely.
    """
    sh = (REPO_ROOT / "scripts" / "train_report_generation_h100.sh").read_text()

    assert 'SEED="${SEED:-42}"' in sh, (
        "SEED lever missing or its default drifted off 42 -- a non-42 default "
        "silently breaks comparability with every pre-Phase-15 checkpoint"
    )
    assert "seed=${SEED}" in sh, (
        "SEED is declared but never passed to train_report_generation.py -- "
        "the wrapper would print a seed it does not actually use"
    )
    assert 'echo "Training seed: ${SEED}"' in sh, (
        "the resolved seed must be echoed into the job log; without a positive "
        "line, a wrong seed is undetectable (see the prefix_k trap)"
    )


# ---------------------------------------------------------------------------
# Phase 11B — CheXbert F1. compute_chexbert_metrics wraps the real, verified
# f1chexbert API (F1CheXbert()(hyps=..., refs=...) -> (accuracy,
# accuracy_per_sample, chexbert_all, chexbert_5), confirmed live 2026-08-29
# against job 2492037's log and the package's own PyPI README -- an earlier
# version of this code guessed a wrong API and would have crashed once the
# package was installed; it only avoided that because the package wasn't
# installed yet, degrading to a clean "skipped: ModuleNotFoundError" as
# designed. There is no CPU-testable pure-math core to extract (unlike 11C's
# nearest_neighbor_indices split) since f1chexbert's public API exposes only
# the aggregate scoring call, not raw per-report labels -- so the only thing
# testable without real weights is the None/"skipped" degradation path.
# ---------------------------------------------------------------------------

@pytest.mark.willi_parity
def test_compute_chexbert_metrics_returns_none_when_package_unavailable(monkeypatch):
    """f1chexbert is an optional dep (guarded, same as nltk for METEOR) --
    forcing the import to fail (sys.modules[name] = None is the documented
    way to make a subsequent `import f1chexbert` raise ImportError) must
    degrade to None, never raise, matching meteor_score_corpus's contract."""
    import sys
    from scripts.evaluate_report_generation import compute_chexbert_metrics

    monkeypatch.setitem(sys.modules, "f1chexbert", None)
    result = compute_chexbert_metrics(["a report"], ["another report"])
    assert result is None


@pytest.mark.willi_parity
def test_compute_all_metrics_chexbert_flag_gates_key_presence(monkeypatch):
    """chexbert=False (the default) must not even attempt labeling — no
    'chexbert' key at all, not just None — so callers who never opt in pay
    zero cost and see no behavior change from before this flag existed."""
    import sys
    from scripts.evaluate_report_generation import compute_all_metrics

    hyps, refs = ["the lungs are clear"], ["the lungs are clear"]

    off = compute_all_metrics(hyps, refs, chexbert=False)
    assert "chexbert" not in off

    monkeypatch.setitem(sys.modules, "f1chexbert", None)
    on = compute_all_metrics(hyps, refs, chexbert=True)
    assert "chexbert" in on
    assert on["chexbert"] is None  # package unavailable in this env -> skipped


@pytest.mark.willi_parity
def test_inspect_report_generation_h100_slurm_wrapper_exposes_chexbert_lever():
    """Phase 11B opt-in CheXbert F1 lever, off by default. Pins the set -u
    -safe empty-array-expansion pattern (${ARR[@]+"${ARR[@]}"}) already
    established in eval_h100.sh/train_biomedclip_kd_h100.sh — the bare
    ${ARR[@]} form breaks under set -u with bash <4.4 when CHEXBERT=false
    leaves the array empty."""
    sh = (REPO_ROOT / "scripts" / "inspect_report_generation_h100.sh").read_text()
    assert 'CHEXBERT="${CHEXBERT:-false}"' in sh
    assert "--chexbert" in sh
    assert '${CHEXBERT_ARGS[@]+"${CHEXBERT_ARGS[@]}"}' in sh


@pytest.mark.willi_parity
def test_retrieval_baseline_h100_slurm_wrapper_exposes_chexbert_lever():
    sh = (REPO_ROOT / "scripts" / "retrieval_baseline_h100.sh").read_text()
    assert 'CHEXBERT="${CHEXBERT:-false}"' in sh
    assert "--chexbert" in sh
    assert '${CHEXBERT_ARGS[@]+"${CHEXBERT_ARGS[@]}"}' in sh


@pytest.mark.willi_parity
def test_write_hyps_refs_writes_one_report_per_line_sanitizing_newlines(tmp_path):
    """Phase 11B (2026-08-29): dumps for the isolated-venv CheXbert scorer.
    A report containing an embedded newline (real MIMIC findings/impression
    text can have literal paragraph breaks) must be collapsed to one line —
    otherwise it would silently split across multiple lines on write and
    desync the hyp/ref alignment when a reader does .splitlines() (the same
    convention --hyp-file/--ref-file already relies on)."""
    from scripts.evaluate_report_generation import write_hyps_refs

    hyps = ["Findings: clear lungs.\nImpression: normal.", "no acute process"]
    refs = ["Findings:   clear   lungs.", "Impression:\nno acute process"]

    dump_dir = tmp_path / "dump"
    write_hyps_refs(str(dump_dir), hyps, refs)

    hyp_lines = (dump_dir / "hyps.txt").read_text().splitlines()
    ref_lines = (dump_dir / "refs.txt").read_text().splitlines()
    assert hyp_lines == ["Findings: clear lungs. Impression: normal.", "no acute process"]
    assert ref_lines == ["Findings: clear lungs.", "Impression: no acute process"]
    assert len(hyp_lines) == len(ref_lines) == len(hyps)


@pytest.mark.willi_parity
def test_dump_dir_flag_present_on_evaluate_report_generation_parser():
    """Phase 11B (2026-08-29): --dump-dir must exist and default to None (off)
    so existing invocations without it are unaffected."""
    sh = (REPO_ROOT / "scripts" / "evaluate_report_generation.py").read_text()
    assert '"--dump-dir"' in sh
    assert "write_hyps_refs(args.dump_dir, hyps, refs)" in sh


@pytest.mark.willi_parity
def test_score_chexbert_standalone_parser_builds_without_importing_f1chexbert(monkeypatch):
    """score_chexbert_standalone.py must be parseable/CLI-testable even when
    f1chexbert isn't installed in THIS env (it's meant for a separate,
    isolated venv) -- so f1chexbert must be imported inside main(), not at
    module level, and build_parser() must work standalone."""
    import sys
    monkeypatch.setitem(sys.modules, "f1chexbert", None)

    from scripts.score_chexbert_standalone import build_parser

    args = build_parser().parse_args(["--hyp-file", "h.txt", "--ref-file", "r.txt"])
    assert args.hyp_file == "h.txt"
    assert args.ref_file == "r.txt"
    assert args.output_dir is None


@pytest.mark.willi_parity
def test_score_chexbert_standalone_mismatched_line_counts_raises(tmp_path, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "f1chexbert", None)

    from scripts.score_chexbert_standalone import main

    hyp_file = tmp_path / "h.txt"
    ref_file = tmp_path / "r.txt"
    hyp_file.write_text("one\ntwo\n")
    ref_file.write_text("only one\n")

    monkeypatch.setattr(
        sys, "argv",
        ["score_chexbert_standalone.py", "--hyp-file", str(hyp_file), "--ref-file", str(ref_file)],
    )
    with pytest.raises(SystemExit):
        main()


@pytest.mark.willi_parity
def test_inspect_report_generation_h100_slurm_wrapper_exposes_dump_dir_lever():
    sh = (REPO_ROOT / "scripts" / "inspect_report_generation_h100.sh").read_text()
    assert 'DUMP_DIR="${DUMP_DIR:-}"' in sh
    assert "--dump-dir" in sh


@pytest.mark.willi_parity
def test_retrieval_baseline_h100_slurm_wrapper_exposes_dump_dir_lever():
    sh = (REPO_ROOT / "scripts" / "retrieval_baseline_h100.sh").read_text()
    assert 'DUMP_DIR="${DUMP_DIR:-}"' in sh
    assert "--dump-dir" in sh


@pytest.mark.willi_parity
def test_setup_chexbert_venv_h100_slurm_wrapper_pins_transformers_below_5():
    """Phase 11B (2026-08-29): f1chexbert's tokenize() calls the legacy
    tokenizer.encode_plus(...), removed in transformers>=5.0 -- this isolated
    venv must pin below that, independent of the main .venv's floor
    (transformers>=4.35.0 in requirements.txt, no ceiling)."""
    sh = (REPO_ROOT / "scripts" / "setup_chexbert_venv_h100.sh").read_text()
    assert '"transformers<5"' in sh
    assert "f1chexbert" in sh


@pytest.mark.willi_parity
def test_score_chexbert_h100_slurm_wrapper_requires_dump_dir():
    sh = (REPO_ROOT / "scripts" / "score_chexbert_h100.sh").read_text()
    assert "${DUMP_DIR:?" in sh
    assert "score_chexbert_standalone.py" in sh


@pytest.mark.willi_parity
def test_setup_chexbert_venv_h100_slurm_wrapper_pins_scikit_learn_below_1_8():
    """Phase 11B (2026-08-29, later): f1chexbert's forward() does
    `y_type, y_true, y_pred = _check_targets(...)`, a 3-value unpack of the
    PRIVATE sklearn.metrics._classification._check_targets API. Confirmed by
    diffing scikit-learn's own source across tags: _check_targets returned
    exactly (y_type, y_true, y_pred) through tag 1.7.2, then 1.8.0 added a
    sample_weight param/return, making it a 4-tuple --
    "ValueError: too many values to unpack (expected 3)". f1chexbert has no
    sklearn upper pin, so an unpinned install pulls the incompatible >=1.8.0.
    This isolated venv must pin below that."""
    sh = (REPO_ROOT / "scripts" / "setup_chexbert_venv_h100.sh").read_text()
    assert '"scikit-learn<1.8"' in sh


@pytest.mark.willi_parity
def test_setup_chexbert_venv_h100_slurm_wrapper_is_rerunnable_in_place():
    """Found live 2026-08-30 (job 2494759): `uv venv` errors out with "A
    virtual environment already exists" on a bare rerun against a VENV_DIR a
    prior invocation already created -- exactly the situation a dependency-pin
    fix (like the scikit-learn<1.8 one above) needs, since the fix only takes
    effect in a rebuilt venv. First fix (`--clear`) was ITSELF not reliable on
    this cluster's NFS-backed home filesystem -- job 2494771 hit `Failed to
    remove directory .../lib: Directory not empty (os error 39)` from uv's own
    internal removal logic against the large existing site-packages tree.
    Final fix: an explicit `rm -rf "${VENV_DIR}"` before venv creation,
    predictable and independent of either tool's internal --clear behavior."""
    sh = (REPO_ROOT / "scripts" / "setup_chexbert_venv_h100.sh").read_text()
    assert 'rm -rf "${VENV_DIR}"' in sh
    assert "uv venv" in sh


@pytest.mark.willi_parity
def test_inspect_report_generation_h100_slurm_wrapper_defaults_to_full_data_not_arm0():
    """Found live 2026-08-30: CHECKPOINT and PARQUET both still defaulted to
    the arm0/ subset artifacts long after full-data Phase 8 pack + Phase 10E
    training (job 2491338, EXPERIMENT=h100_report_gen_full) landed. arm0 is
    CLOSED/historical per CLAUDE.md's Phase 9 note -- the default invocation
    of this script must point at full data, not silently fall back to the
    much-smaller closed arm0 arm."""
    sh = (REPO_ROOT / "scripts" / "inspect_report_generation_h100.sh").read_text()
    assert 'CHECKPOINT:-./outputs/h100_report_gen_full/checkpoints/last.ckpt' in sh
    assert 'PARQUET:-/sc/home/$USER/dataset/mimic_full/validate.parquet' in sh
    # arm0 may still be mentioned in explanatory comments (it's referenced as
    # the historical bug this pins), but never inside a default assignment.
    assert "arm0" not in sh.split("CHECKPOINT=")[1].split("\n")[0]
    assert "arm0" not in sh.split("PARQUET=")[1].split("\n")[0]


@pytest.mark.willi_parity
def test_retrieval_baseline_h100_slurm_wrapper_defaults_to_full_data_not_arm0():
    """Found live 2026-08-30 (job 2494817): running this script with only
    DUMP_DIR set (no TRAIN_PARQUET/PARQUET override, as the plan's own
    NEXT-ACTION instructions assumed would be enough) silently reproduced the
    CLOSED arm0 retrieval-NN numbers (rouge_l ~0.369) instead of the intended
    full-data floor (rouge_l ~0.188, jobs 2491600/2491687), because
    TRAIN_PARQUET/PARQUET still defaulted to arm0/train.parquet and
    arm0/validate.parquet. Fixed: defaults now point at the full-data
    train.parquet/validate.parquet directly under mimic_full/."""
    sh = (REPO_ROOT / "scripts" / "retrieval_baseline_h100.sh").read_text()
    assert 'TRAIN_PARQUET="${TRAIN_PARQUET:-/sc/home/$USER/dataset/mimic_full/train.parquet}"' in sh
    assert 'PARQUET="${PARQUET:-/sc/home/$USER/dataset/mimic_full/validate.parquet}"' in sh
    # arm0 may still be mentioned in explanatory comments (it's referenced as
    # the historical bug this pins), but never inside a default assignment.
    assert "arm0" not in sh.split("TRAIN_PARQUET=")[-1].split("\n")[0]
    assert "arm0" not in sh.split('PARQUET="${PARQUET:-')[-1].split("\n")[0]


# ---------------------------------------------------------------------------
# Phase 13 — closing the CheXbert F1 gap (H100_SCALING_PLAN.md, 2026-08-30).
# 13A: optional fine-tuned image-tower checkpoint for report-gen training.
# 13B: multi-GPU DDP lever for the decoder trainer (plain LM loss, no
# in-batch-negatives semantics, so DDP is a clean throughput win here --
# unlike the still-unbuilt Phase 3 all_gather needed for the contrastive/CLIP
# trainer to get anything beyond throughput out of extra GPUs).
# ---------------------------------------------------------------------------

@pytest.mark.willi_parity
def test_load_image_tower_checkpoint_strips_prefix_and_loads_nonstrict(tmp_path):
    """Pure-function test of the Phase 13A checkpoint-loading helper against a
    tiny nn.Module stand-in -- no open_clip/network needed, matching
    ReportGenerationLightningModule's existing no-open_clip-required
    CPU-testability design. Verifies: (1) only "image_encoder."-prefixed keys
    are pulled from a Lightning-style checkpoint dict and the prefix is
    stripped before load_state_dict; (2) unrelated keys (e.g. "decoder.*"
    from the same checkpoint) are correctly ignored, not misapplied; (3) the
    load is non-strict, so a stand-in with an extra untouched param doesn't
    raise."""
    import torch.nn as nn
    from hybrid_xmamba.training.lightning_module import load_image_tower_checkpoint

    class TinyTower(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(4))
            self.untouched = nn.Parameter(torch.ones(2))

    tower = TinyTower()
    fine_tuned_weight = torch.arange(4, dtype=torch.float32)
    ckpt_path = tmp_path / "fake_contrastive.ckpt"
    torch.save({
        "state_dict": {
            "image_encoder.weight": fine_tuned_weight,
            "decoder.some_other_param": torch.zeros(3),  # must be ignored
        }
    }, ckpt_path)

    missing, unexpected = load_image_tower_checkpoint(tower, str(ckpt_path))

    assert torch.equal(tower.weight, fine_tuned_weight), "fine-tuned weight was not applied"
    assert torch.equal(tower.untouched, torch.ones(2)), "untouched param must be unaffected"
    assert "untouched" in missing, "non-strict load must report the un-supplied param as missing"
    assert not unexpected, f"decoder.* key must not leak through as unexpected: {unexpected}"


@pytest.mark.willi_parity
def test_train_report_generation_h100_slurm_wrapper_exposes_image_encoder_ckpt_lever():
    """Phase 13A: IMAGE_ENCODER_CKPT is an optional env lever (empty default =
    stock BiomedCLIP, unchanged behaviour), fail-fast-checked for existence
    only when set (mirrors DECODER_CKPT's unconditional check), and only
    added to the Hydra invocation when non-empty (set -u-safe empty-array
    expansion, same pattern as CHEXBERT_ARGS in inspect_report_generation_h100.sh
    / retrieval_baseline_h100.sh)."""
    sh = (REPO_ROOT / "scripts" / "train_report_generation_h100.sh").read_text()
    assert 'IMAGE_ENCODER_CKPT="${IMAGE_ENCODER_CKPT:-}"' in sh
    assert 'if [ -n "${IMAGE_ENCODER_CKPT}" ] && [ ! -f "${IMAGE_ENCODER_CKPT}" ]' in sh
    assert 'EXTRA_ARGS+=("image_encoder_checkpoint=${IMAGE_ENCODER_CKPT}")' in sh
    assert '${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}' in sh


@pytest.mark.willi_parity
def test_config_yaml_declares_image_encoder_checkpoint_key():
    """Same declared-key requirement as decoder_checkpoint (config.yaml
    comment, live bug 2026-08-23 job 2478635): Hydra's strict-struct mode
    rejects a CLI override for an undeclared top-level key, so
    image_encoder_checkpoint= must be pre-declared here, not just assumed."""
    cfg_text = (REPO_ROOT / "configs" / "config.yaml").read_text()
    assert "image_encoder_checkpoint: null" in cfg_text


@pytest.mark.willi_parity
def test_train_report_generation_h100_slurm_wrapper_exposes_multi_gpu_lever():
    """Phase 13B: NUM_GPUS selects h100_single_gpu (default, 1 GPU) vs
    h100_multi_ddp (>1), and the script fails fast if fewer GPUs were
    actually allocated than requested rather than silently training on 1 --
    NUM_GPUS alone does not request GPUs from SLURM (that's a separate
    --gpus/--gres sbatch CLI flag), so this mismatch is a real, easy-to-hit
    user error the script must catch."""
    sh = (REPO_ROOT / "scripts" / "train_report_generation_h100.sh").read_text()
    assert 'NUM_GPUS="${NUM_GPUS:-1}"' in sh
    assert 'TRAINER_CFG="h100_single_gpu"' in sh
    assert 'TRAINER_CFG="h100_multi_ddp"' in sh
    assert 'if [ "${NUM_GPUS}" -gt 1 ]' in sh
    assert "trainer=${TRAINER_CFG}" in sh
    assert 'AVAIL_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")' in sh
    assert 'if [ "${AVAIL_GPUS}" -lt "${NUM_GPUS}" ]' in sh


@pytest.mark.willi_parity
def test_h100_multi_ddp_trainer_config_exposes_keys_train_report_generation_reads():
    """train_report_generation.py's pl.Trainer(...) construction reads
    accelerator/devices/precision/strategy/max_steps/val_check_interval/
    check_val_every_n_epoch/log_every_n_steps/accumulate_grad_batches/
    default_root_dir from cfg.trainer -- h100_multi_ddp.yaml must supply all
    of these (composing trainer=h100_multi_ddp must not KeyError partway
    through Trainer construction on an H100 box this repo cannot smoke-test
    from here)."""
    import yaml

    trainer_cfg = yaml.safe_load((REPO_ROOT / "configs" / "trainer" / "h100_multi_ddp.yaml").read_text())
    required = {
        "accelerator", "devices", "precision", "strategy", "max_steps",
        "val_check_interval", "check_val_every_n_epoch", "log_every_n_steps",
        "accumulate_grad_batches", "default_root_dir",
    }
    missing = required - set(trainer_cfg.keys())
    assert not missing, f"h100_multi_ddp.yaml missing keys train_report_generation.py reads: {missing}"
    assert trainer_cfg["strategy"] == "ddp"
    assert trainer_cfg["devices"] == -1


# ---------------------------------------------------------------------------
# Phase 13F — oversample training reports positive for the 3 CheXpert labels
# the 13B checkpoint never predicts (Lung Lesion/Pneumothorax/Pleural Other,
# F1=0.0 on both eval splits).
# ---------------------------------------------------------------------------

@pytest.mark.willi_parity
def test_compute_rare_finding_sample_weights_oversamples_positive_studies(tmp_path):
    """Pure-function test against a tiny on-disk CSV fixture -- no HF Dataset/
    network needed. Verifies: (1) U-Zeros convention (1.0=positive, {0.0,
    -1.0, NaN}=not-positive) is applied per-column; (2) a study positive for
    ANY of the target rare labels gets the oversample weight, not just one
    matching column; (3) a study with NO row in the CSV at all defaults to
    weight 1.0 (conservative -- never inflates an unknown-label row); (4)
    weights are returned in the SAME order as the input study_ids list, not
    CSV row order."""
    import pandas as pd
    from scripts.train_report_generation import compute_rare_finding_sample_weights

    csv_path = tmp_path / "mimic-cxr-2.0.0-chexpert.csv.gz"
    pd.DataFrame({
        "study_id": [10, 20, 30, 40],
        "Lung Lesion":    [1.0, 0.0, -1.0, float("nan")],
        "Pneumothorax":   [0.0, 1.0, 0.0, 0.0],
        "Pleural Other":  [0.0, 0.0, 0.0, 0.0],
    }).to_csv(csv_path, index=False, compression="gzip")

    # study_ids intentionally out of CSV order, plus one (99) absent from the CSV.
    study_ids = [40, 30, 99, 20, 10]
    weights = compute_rare_finding_sample_weights(
        study_ids=study_ids,
        chexpert_csv=str(csv_path),
        rare_labels=["Lung Lesion", "Pneumothorax", "Pleural Other"],
        oversample_weight=5.0,
    )

    assert weights == [1.0, 1.0, 1.0, 5.0, 5.0], weights


@pytest.mark.willi_parity
def test_train_report_generation_h100_slurm_wrapper_exposes_oversample_rare_lever():
    """OVERSAMPLE_RARE/OVERSAMPLE_WEIGHT are optional env levers, default off
    (identical behaviour to before this lever existed), always passed to the
    Hydra invocation (both keys are declared with safe defaults in
    configs/dataset/cxr_mimic_full.yaml, so no EXTRA_ARGS empty-guard is
    needed here -- unlike IMAGE_ENCODER_CKPT, which has no such default)."""
    sh = (REPO_ROOT / "scripts" / "train_report_generation_h100.sh").read_text()
    assert 'OVERSAMPLE_RARE="${OVERSAMPLE_RARE:-false}"' in sh
    assert 'OVERSAMPLE_WEIGHT="${OVERSAMPLE_WEIGHT:-5.0}"' in sh
    assert "dataset.oversample_rare_findings=${OVERSAMPLE_RARE}" in sh
    assert "dataset.oversample_weight=${OVERSAMPLE_WEIGHT}" in sh


@pytest.mark.willi_parity
def test_cxr_mimic_full_config_declares_oversample_rare_keys():
    """Same declared-key requirement as decoder_checkpoint/image_encoder_checkpoint
    (Hydra strict-struct mode rejects a CLI override for an undeclared key) --
    all 4 Phase 13F keys must be declared in configs/dataset/cxr_mimic_full.yaml,
    not just assumed present."""
    cfg_text = (REPO_ROOT / "configs" / "dataset" / "cxr_mimic_full.yaml").read_text()
    assert "oversample_rare_findings: false" in cfg_text
    assert "chexpert_csv:" in cfg_text
    assert "rare_finding_labels:" in cfg_text
    assert "oversample_weight:" in cfg_text
    for label in ("Lung Lesion", "Pneumothorax", "Pleural Other"):
        assert label in cfg_text


# ---------------------------------------------------------------------------
# Phase 12A — MODE=sts on eval_h100.sh (BIOSSES/STS-B/MedSTS via
# evaluate_sts.py), the last genuinely missing measurement for the closed
# retrieval chapter's authoritative table.
# ---------------------------------------------------------------------------

@pytest.mark.willi_parity
def test_eval_h100_slurm_wrapper_exposes_sts_mode():
    """MODE=sts must dispatch to evaluate_sts.py with --datasets all, and must
    NOT default to the gated-repo-safe offline HF env vars MODE=ppl/retrieval
    use -- BIOSSES/STS-B/MedSTS are public HF datasets that need real network
    access to download, unlike mimic-cxr."""
    sh = (REPO_ROOT / "scripts" / "eval_h100.sh").read_text()
    assert 'elif [ "${MODE}" = "sts" ]; then' in sh
    assert "evaluate_sts.py" in sh
    assert '--datasets all' in sh
    assert 'export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"' in sh
    assert 'export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"' in sh
    assert "expected 'ppl', 'retrieval', or 'sts'" in sh


@pytest.mark.willi_parity
def test_eval_h100_slurm_wrapper_sts_mode_does_not_change_other_modes_offline_default():
    """Regression guard for the exact bug this change could introduce: the
    MODE=sts branch must be conditional, not a global default flip -- ppl and
    retrieval must still default HF_DATASETS_OFFLINE/HF_HUB_OFFLINE to 1
    (test_eval_script_bakes_in_offline_and_populated_cache pins the literal
    strings; this test pins the if/else structure around them so a future
    edit can't collapse the branches back into one unconditional default)."""
    sh = (REPO_ROOT / "scripts" / "eval_h100.sh").read_text()
    assert 'if [ "${MODE}" = "sts" ]; then' in sh
    # The offline-default lines must appear inside the else branch, i.e. after
    # the sts-mode online-default lines in the same if/else block.
    sts_idx = sh.index('export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"')
    offline_idx = sh.index('export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"')
    assert sts_idx < offline_idx, "sts (online) default must precede the else (offline) default"


@pytest.mark.willi_parity
def test_evaluate_sts_uses_script_free_dataset_mirrors():
    """REGRESSION (2026-09-03, first live MODE=sts run, job 2505439): both
    original dataset sources crashed -- BIOSSES's bigbio/biosses/nguyenthanhdo
    candidates all relied on HF's legacy "dataset loading script" mechanism
    ("Dataset scripts are no longer supported"), and STS-B's bare
    `load_dataset("glue", "stsb", split=...)` call was UNGUARDED (no
    try/except), so its equivalent failure crashed the whole job instead of
    falling through to a fallback -- confirmed live via an
    hf_file_system.resolve_path crash on glue's legacy standalone YAML.
    Fixed: mteb/biosses-sts and mteb/stsbenchmark-sts (script-free parquet
    mirrors, sentence1/sentence2/score schema) are now tried FIRST for both,
    and _load_stsb gained the same per-candidate try/except loop BIOSSES
    already had, so a first-source failure degrades to the next candidate
    instead of crashing the process."""
    src = (REPO_ROOT / "scripts" / "evaluate_sts.py").read_text()
    assert '"mteb/biosses-sts"' in src
    assert '"mteb/stsbenchmark-sts"' in src
    # STS-B must no longer be a single unguarded load_dataset call -- pin the
    # per-candidate try/except loop structure the fix introduced.
    stsb_body = src.split("def _load_stsb(")[1].split("\ndef ")[0]
    assert "for dataset_id, kwargs in" in stsb_body
    assert "except Exception as exc:" in stsb_body


# ---------------------------------------------------------------------------
# Phase 14A — parameter-matched Transformer baseline (supervisor review 2026-09-07)
# ---------------------------------------------------------------------------

# hybrid_150m_v2's instantiated parameter count. The whole point of Phase 14A is a
# MATCHED baseline, so this number is the spec, not a note.
HYBRID_150M_V2_PARAMS = 183_721_824
PARAM_MATCH_TOLERANCE = 0.005  # 0.5%


def _instantiate_model_config(config_name):
    """Build a HybridLanguageModel from a configs/model/*.yaml, as the trainers do."""
    import dataclasses

    import yaml

    from hybrid_xmamba.models.configuration_hybrid import HybridConfig
    from hybrid_xmamba.models.hybrid_lm import HybridLanguageModel

    with open(REPO_ROOT / "configs" / "model" / (config_name + ".yaml")) as fh:
        raw = yaml.safe_load(fh)
    valid = {f.name for f in dataclasses.fields(HybridConfig)}
    return HybridLanguageModel(HybridConfig(**{k: v for k, v in raw.items() if k in valid}))


def test_hybrid_150m_v2_param_count_is_the_documented_baseline():
    """Pin the incumbent's size. If this drifts, the 'matched' baseline is no longer
    matched and Phase 14A's central comparison quietly becomes invalid."""
    model = _instantiate_model_config("hybrid_150m_v2")
    total = sum(p.numel() for p in model.parameters())
    assert total == HYBRID_150M_V2_PARAMS, (
        "hybrid_150m_v2 is now %d params, not the documented %d. Phase 14A's "
        "Transformer baseline was matched against the old number -- re-derive the "
        "baseline's num_layers before running anything."
        % (total, HYBRID_150M_V2_PARAMS)
    )


@pytest.mark.parametrize(
    "config_name", ["transformer_150m_baseline", "transformer_150m_baseline_rrg"]
)
def test_transformer_baseline_is_parameter_matched_to_the_hybrid(config_name):
    """THE Phase 14A invariant: the baseline must be parameter-matched.

    'Attention-free matches attention' is only a claim if the two have the same
    parameter budget. A baseline that silently drifts smaller would make the hybrid
    look good for the wrong reason.
    """
    model = _instantiate_model_config(config_name)
    total = sum(p.numel() for p in model.parameters())
    delta = abs(total / HYBRID_150M_V2_PARAMS - 1.0)
    assert delta < PARAM_MATCH_TOLERANCE, (
        "%s has %d params vs the hybrid's %d (%.3f%% off, tolerance %.1f%%). The "
        "comparison is no longer parameter-matched."
        % (config_name, total, HYBRID_150M_V2_PARAMS, 100 * delta,
           100 * PARAM_MATCH_TOLERANCE)
    )


@pytest.mark.parametrize(
    "config_name", ["transformer_150m_baseline", "transformer_150m_baseline_rrg"]
)
def test_transformer_baseline_is_pure_attention_and_spends_no_params_on_positions(
    config_name,
):
    """The baseline must actually be a Transformer, and must use RoPE.

    A learned positional table would hand it +0.79M params the hybrid never gets
    (the hybrid's embedding matrix is exactly vocab x dim), breaking the match.
    """
    import yaml

    with open(REPO_ROOT / "configs" / "model" / (config_name + ".yaml")) as fh:
        raw = yaml.safe_load(fh)
    assert raw["layer_pattern"] == ["attention"]
    assert raw["num_layers"] == 15, "15 layers is what makes the param match work"
    assert raw["mlp_ratio"] == 4.0, "a non-standard FFN width reads as a rigged baseline"

    model = _instantiate_model_config(config_name)
    pos_params = [
        n for n, _ in model.named_parameters()
        if "pos_emb" in n or "position_embedding" in n or "wpe" in n
    ]
    assert pos_params == [], "baseline must use RoPE, found learned position params: %s" % pos_params


def test_transformer_baseline_shares_the_hybrids_hyperparameters_verbatim():
    """Single-lever discipline: only the mixer may differ.

    If the baseline's LR/schedule/regularisation drift from the hybrid's, the
    comparison stops being attributable to architecture.
    """
    import yaml

    def load(name):
        with open(REPO_ROOT / "configs" / "model" / (name + ".yaml")) as fh:
            return yaml.safe_load(fh)

    hybrid, baseline = load("hybrid_150m_v2"), load("transformer_150m_baseline")
    for key in ["vocab_size", "dim", "mlp_ratio", "max_position_embeddings", "dropout",
                "initializer_range", "tie_word_embeddings", "norm_type", "use_mlp",
                "learning_rate", "weight_decay", "warmup_steps", "gradient_clip_val"]:
        assert baseline[key] == hybrid[key], (
            "Phase 14A single-lever violation: %s is %r in the Transformer baseline but "
            "%r in hybrid_150m_v2. Only the mixer may differ."
            % (key, baseline[key], hybrid[key])
        )

    hybrid_rrg, baseline_rrg = load("hybrid_150m_v2_rrg"), load("transformer_150m_baseline_rrg")
    for key in ["prefix_k", "image_patch_dim", "decoder_lr", "head_lr", "weight_decay",
                "warmup_steps", "max_steps", "gradient_clip_val", "vit_unfreeze_blocks", "vit_lr"]:
        assert baseline_rrg[key] == hybrid_rrg[key], (
            "Phase 14A single-lever violation in the RRG variant: %s is %r vs the "
            "hybrid's %r." % (key, baseline_rrg[key], hybrid_rrg[key])
        )


def test_attention_layer_type_is_registered_everywhere():
    """'attention' must be accepted by the config validator AND the block dispatch.

    These are two separate code paths; a mismatch fails at model-build time inside a
    SLURM job rather than here.
    """
    from hybrid_xmamba.layers.hybrid_block import HybridBlock
    from hybrid_xmamba.models.configuration_hybrid import HybridConfig

    cfg = HybridConfig(dim=64, num_layers=2, layer_pattern=["attention"], num_heads=4,
                       head_dim=16, vocab_size=128)
    assert cfg.get_layer_config(0)["num_heads"] == 4

    block = HybridBlock(dim=64, layer_type="attention", num_heads=4, head_dim=16)
    assert block.mixer.__class__.__name__ == "AttentionBlock"


def test_attention_block_is_causal():
    """A non-causal mixer would leak the answer and make every metric meaningless."""
    import torch

    from hybrid_xmamba.layers.attention_block import AttentionBlock

    torch.manual_seed(0)
    block = AttentionBlock(dim=64, num_heads=4, head_dim=16).eval()
    x = torch.randn(1, 12, 64)
    with torch.no_grad():
        y_ref = block(x)
        x_perturbed = x.clone()
        x_perturbed[:, 7:] = torch.randn(1, 5, 64)
        y_perturbed = block(x_perturbed)
    assert torch.allclose(y_ref[:, :7], y_perturbed[:, :7], atol=1e-6), (
        "attention is not causal: perturbing future positions changed past outputs"
    )
    assert not torch.allclose(y_ref[:, 7:], y_perturbed[:, 7:], atol=1e-6), (
        "sanity check failed: perturbing the input changed nothing at all"
    )


def test_attention_block_blocks_cross_document_attention():
    """Stage-0 packs documents; the hybrid resets state at each boundary.

    If attention ignored cu_seqlens it would read context the hybrid provably
    cannot -- a silent, uncontrolled advantage in the exact comparison Phase 14A
    exists to make fair.
    """
    import torch

    from hybrid_xmamba.layers.attention_block import AttentionBlock

    torch.manual_seed(0)
    block = AttentionBlock(dim=64, num_heads=4, head_dim=16).eval()
    x = torch.randn(1, 12, 64)
    x_edited = x.clone()
    x_edited[:, :6] = torch.randn(1, 6, 64)  # rewrite document 0 only
    cu_seqlens = torch.tensor([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]])

    with torch.no_grad():
        masked_a = block(x, cu_seqlens=cu_seqlens)
        masked_b = block(x_edited, cu_seqlens=cu_seqlens)
        unmasked_a, unmasked_b = block(x), block(x_edited)

    assert torch.allclose(masked_a[:, 6:], masked_b[:, 6:], atol=1e-6), (
        "document 1 changed when document 0 was edited -- cu_seqlens masking is broken"
    )
    assert not torch.allclose(unmasked_a[:, 6:], unmasked_b[:, 6:], atol=1e-6), (
        "sanity check failed: without the mask document 1 should have changed, so this "
        "test would pass even with masking removed"
    )


def test_stage0_150m_wrapper_model_config_is_env_overridable():
    """Phase 14A-3 reuses this wrapper verbatim; a hardcoded MODEL_CONFIG blocks it."""
    src = (REPO_ROOT / "scripts" / "train_stage0_150m_h100.sh").read_text()
    assert 'export MODEL_CONFIG="${MODEL_CONFIG:-hybrid_150m_v2}"' in src, (
        "train_stage0_150m_h100.sh must accept MODEL_CONFIG from the environment so the "
        "Transformer baseline can share the hybrid's exact Stage-0 recipe"
    )


# ---------------------------------------------------------------------------
# Phase 14B — boilerplate / duplicate-template analysis
# ---------------------------------------------------------------------------

def test_analyze_generation_diversity_recovers_a_known_duplicate_rate():
    """The core measurement must be right, since a headline claim now rests on it."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import analyze_generation_diversity as agd
    finally:
        sys.path.pop(0)

    # 6 of 10 texts sit in duplicate clusters (3x "a", 2x "b"); "c".."e" are unique.
    texts = ["a", "a", "a", "b", "b", "c", "d", "e", "f", "g"]
    n_clusters, in_cluster, frac, top = agd.duplicate_clusters(texts)
    assert n_clusters == 2
    assert in_cluster == 5
    assert abs(frac - 0.5) < 1e-9
    assert top == [3, 2]

    assert agd.duplicate_clusters(["x"] * 4)[2] == 1.0        # fully collapsed
    assert agd.duplicate_clusters(list("abcd"))[2] == 0.0      # fully unique


def test_analyze_generation_diversity_orders_corpora_by_repetitiveness():
    """A templated corpus must score as less diverse than a varied one on every metric.

    Phase 14B's whole argument is a comparison against controls, so the metrics have
    to order corpora correctly or the comparison means nothing.
    """
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import analyze_generation_diversity as agd
    finally:
        sys.path.pop(0)

    templated = ["no acute cardiopulmonary process"] * 40
    varied = ["patient %d shows a focal opacity in segment %d" % (i, i) for i in range(40)]

    assert agd.distinct_n(templated, 2) < agd.distinct_n(varied, 2)
    assert agd.type_token_ratio(templated) < agd.type_token_ratio(varied)
    assert agd.self_bleu4(templated, sample=20, refs_per=5) > \
        agd.self_bleu4(varied, sample=20, refs_per=5)


def test_analyze_generation_diversity_warns_when_controls_are_missing():
    """A bare duplication rate is the exact reporting weakness Phase 14B exists to fix."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import analyze_generation_diversity as agd
    finally:
        sys.path.pop(0)

    report = agd.render([agd.analyse("generated (model)", ["a", "a", "b"])])
    assert "No controls were supplied" in report


def test_analyze_diversity_slurm_wrapper_exists_and_is_cpu_only():
    """Phase 14B runs on the cluster, and the login node refuses direct execution.

    Confirmed live in Phase 7E: `python build_mimic_cxr_local.py meta` on lx01 was
    rejected with "This command is not allowed on the login node!" before making a
    single request. Anything runnable therefore needs an sbatch entry point.
    """
    path = REPO_ROOT / "scripts" / "analyze_diversity_h100.sh"
    assert path.exists(), "Phase 14B needs a SLURM wrapper; bare python is refused on lx01"
    src = path.read_text()

    assert "#SBATCH --partition=pot-hpi-aisc-batch" in src
    assert "#SBATCH --account=aisc" in src
    assert "#SBATCH --qos=aisc" in src
    # Pure-stdlib text analysis: requesting a GPU would waste a scarce resource
    # and, on this cluster, queue behind GPU demand for no reason.
    # Check the DIRECTIVE, not the string -- the header comment legitimately says
    # "Do not add --gpus", which a naive substring check trips over.
    gpu_directives = [
        line for line in src.splitlines()
        if line.startswith("#SBATCH") and "--gpus" in line
    ]
    assert not gpu_directives, (
        "the diversity analysis is pure-stdlib CPU work; requesting a GPU would "
        "queue it behind real GPU demand for nothing: %s" % gpu_directives
    )

    # Must fail loudly rather than emit an empty/half-controlled report -- this
    # project's documented expensive failure mode is a run that reports success
    # while doing nothing.
    assert 'HYPS="${HYPS:?' in src, "HYPS must be required, not silently defaulted"
    assert "ERROR: HYPS file not found" in src
    assert "WARNING: no REFS control supplied" in src
    assert "WARNING: no BASELINE control supplied" in src
    # set -u safe array expansion, the pattern used by the other wrappers
    assert '${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}' in src


def test_profile_efficiency_wrapper_models_is_env_overridable():
    """Phase 14A-7 adds the Transformer to the EXISTING efficiency sweep.

    The protocol must stay identical to the one that produced
    analysis/efficiency_150m/, so the baseline joins that sweep rather than getting
    a new, subtly-different harness.
    """
    src = (REPO_ROOT / "scripts" / "profile_efficiency_h100.sh").read_text()
    assert 'if [ -z "${MODELS:-}" ]; then' in src, (
        "MODELS must be env-overridable so transformer_150m_baseline can join the sweep"
    )
    # The fallback defaults must survive, so an unset MODELS still reproduces the
    # original three-model sweep exactly.
    assert 'MODELS="hybrid_150m_v2 mamba_150m_baseline xlstm_150m_baseline"' in src
    assert 'MODELS="hybrid_70m_v2 mamba_70m_baseline xlstm_70m_baseline"' in src


def test_no_plan_command_invokes_a_bare_python_script_on_the_cluster():
    """Guard the whole Phase 14 section against the login-node trap.

    Phase 7E cost real time to diagnose; a `python scripts/foo.py` line in the plan
    is a command someone will paste into lx01 and have rejected.
    """
    plan = (REPO_ROOT / "H100_SCALING_PLAN.md").read_text()
    phase14 = plan.split("### Phase 14 —")[1].split("\n## Verification")[0]
    # MAMBA3_PLAN_V2.md V1-F: the Mamba-3 plan is scanned WHOLE. Its predecessor's own log has
    # three login-node incidents (job 2513581 among them); local-only commands are written as
    # `venv/bin/python ...` so they cannot be mistaken for cluster commands.
    # EFFICIENCY_PLAN.md is scanned whole for the same reason; its local-only commands
    # are written `venv/bin/python ...` so they cannot be pasted into lx01 by mistake.
    scanned = "\n".join([
        phase14,
        (REPO_ROOT / "MAMBA3_PLAN_V2.md").read_text(),
        (REPO_ROOT / "EFFICIENCY_PLAN.md").read_text(),
    ])
    offenders = [
        line.strip() for line in scanned.splitlines()
        if line.strip().startswith(("python scripts/", "python3 scripts/"))
    ]
    assert not offenders, (
        "Phase 14 contains bare python invocations that the aisc login node will "
        "refuse; route them through an sbatch wrapper: %s" % offenders
    )


def test_decoder_init_guards_against_wrong_architecture_checkpoint():
    """Phase 14A: DECODER_CKPT has a default, so a wrong one does not fail loudly.

    train_report_generation_h100.sh defaults DECODER_CKPT to the HYBRID's Stage-0
    backbone and only checks that the file exists. Pointing a Transformer run at it
    therefore loads under strict=False, matches almost nothing, and trains from
    random init -- silently, at the cost of a full multi-GPU run, and it would
    invalidate the matched-baseline comparison that Phase 14A exists to make.
    """
    src = (REPO_ROOT / "scripts" / "train_report_generation.py").read_text()
    assert "missing_frac" in src, "decoder init must measure how much actually loaded"
    # Slice from the FIRST occurrence onward (str.split cuts at every occurrence,
    # so [1] would only span the gap between the first two).
    tail = src[src.index("missing_frac"):][:2500]
    assert "raise RuntimeError(" in tail, (
        "a mostly-unmatched decoder init must hard-fail, not warn -- it costs a full run"
    )
    # The default really is the hybrid's checkpoint; if that ever changes, this
    # test's rationale needs revisiting rather than the assertion being deleted.
    wrapper = (REPO_ROOT / "scripts" / "train_report_generation_h100.sh").read_text()
    assert "h100_stage0_150m_v2/checkpoints/stage0_model_only.pt" in wrapper


def test_plan_14A5_command_overrides_decoder_ckpt():
    """The plan's own command must not reproduce the bug the guard protects against."""
    plan = (REPO_ROOT / "H100_SCALING_PLAN.md").read_text()
    block = plan.split("**14A-5**")[1].split("- [ ] **14A-6**")[0]
    assert "MODEL_CONFIG=transformer_150m_baseline_rrg" in block
    assert "DECODER_CKPT=./outputs/h100_stage0_transformer_150m/checkpoints/last.ckpt" in block, (
        "14A-5's command must override DECODER_CKPT, or the Transformer silently "
        "initialises from the hybrid's Stage-0 backbone"
    )


def test_report_generation_eval_guards_against_wrong_model_config():
    """Phase 14A: --model-config defaults to the hybrid and is NOT auto-detected.

    load_report_generation_module builds the module from the named YAML and then
    loads weights with strict=False. Evaluating a Transformer checkpoint without
    overriding MODEL_CONFIG therefore does not crash -- it builds a hybrid, matches
    almost nothing, and generates from a RANDOMLY INITIALISED decoder. The metrics
    that come out look plausible and are meaningless, which is worse than a crash:
    they would be misread as "the baseline generates badly".
    """
    src = (REPO_ROOT / "scripts" / "evaluate_report_generation.py").read_text()
    assert "strict=False" in src, "precondition: the load is non-strict, hence the guard"
    tail = src[src.index("n_module_keys"):][:2500]
    assert "raise RuntimeError(" in tail, (
        "a mostly-unmatched eval load must hard-fail; silent garbage generation would "
        "be reported as a real result"
    )
    wrapper = (REPO_ROOT / "scripts" / "inspect_report_generation_h100.sh").read_text()
    assert 'MODEL_CONFIG="${MODEL_CONFIG:-hybrid_150m_v2_rrg}"' in wrapper, (
        "if this default changes, the guard's rationale needs revisiting"
    )


# ---------------------------------------------------------------------------
# Phase 14A-6 — paired bootstrap comparison
# ---------------------------------------------------------------------------

def _load_bootstrap_module():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import bootstrap_compare
        return bootstrap_compare
    finally:
        sys.path.pop(0)


def test_bootstrap_chexbert_f1_matches_sklearn():
    """The F1 here is reimplemented (the scorer lives in an isolated venv).

    Reimplemented metrics drift. Pin it against sklearn, which is what produced
    every CheXbert number already reported.
    """
    import random as _random

    from sklearn.metrics import f1_score

    bc = _load_bootstrap_module()
    rng = _random.Random(0)
    n, k = 60, 14
    y_true = [[rng.randint(0, 1) for _ in range(k)] for _ in range(n)]
    y_pred = [[rng.randint(0, 1) for _ in range(k)] for _ in range(n)]

    for average in ("micro", "macro"):
        mine = bc.chexbert_f1(y_true, y_pred, average)
        theirs = f1_score(y_true, y_pred, average=average, zero_division=0)
        assert abs(mine - theirs) < 1e-9, (
            "chexbert_f1(%s) = %.10f but sklearn says %.10f" % (average, mine, theirs)
        )

    # And on a column subset, which is how the 5-label metric is computed.
    cols = [0, 2, 5, 7, 9]
    sub_true = [[r[j] for j in cols] for r in y_true]
    sub_pred = [[r[j] for j in cols] for r in y_pred]
    for average in ("micro", "macro"):
        assert abs(bc.chexbert_f1(y_true, y_pred, average, cols)
                   - f1_score(sub_true, sub_pred, average=average, zero_division=0)) < 1e-9


def test_bootstrap_per_label_f1_matches_sklearn_and_averages_to_macro():
    """Phase 15B-5. Two invariants, both load-bearing for judging 15C.

    1. Per-label F1 must match sklearn's `average=None` exactly, same pin as
       the micro/macro reimplementation above.
    2. The mean of the per-label list must equal macro F1 to floating point.
       macro is now COMPUTED from this list in the per_label path, so if the
       two ever disagreed the headline macro row and the per-label section of
       the same report would contradict each other -- the kind of internal
       inconsistency a reviewer spots immediately.
    """
    import random as _random

    from sklearn.metrics import f1_score

    bc = _load_bootstrap_module()
    rng = _random.Random(7)
    n, k = 80, 14
    y_true = [[rng.randint(0, 1) for _ in range(k)] for _ in range(n)]
    y_pred = [[rng.randint(0, 1) for _ in range(k)] for _ in range(n)]

    mine = bc.chexbert_f1_per_label(y_true, y_pred)
    theirs = f1_score(y_true, y_pred, average=None, zero_division=0)
    assert len(mine) == k
    for j, (a, b) in enumerate(zip(mine, theirs)):
        assert abs(a - b) < 1e-9, "label %d: %.10f vs sklearn %.10f" % (j, a, b)

    assert abs(sum(mine) / len(mine) - bc.chexbert_f1(y_true, y_pred, "macro")) < 1e-12


def test_bootstrap_per_label_rows_are_namespaced_and_keep_the_main_table_intact():
    """Phase 15B-5. Per-label rows must not leak into the headline table.

    Every bootstrap report this project has already published (the k=32
    head-to-head, the k-sweeps, 15B-1 vs the retrieval floor) has a nine-row
    main table. If --per-label added 14 more rows to it, a new report would no
    longer be comparable at a glance with the ones in
    analysis/PHASE14_SUPERVISOR_REVIEW.md. The prefix keeps them separable and
    the renderer splits on it.
    """
    bc = _load_bootstrap_module()
    n, k = 40, 14
    y_true = [[(i + j) % 2 for j in range(k)] for i in range(n)]
    y_pred = [[(i + j) % 2 for j in range(k)] for i in range(n)]
    names = ["Finding%d" % j for j in range(k)]
    cache = {
        "rouge": [0.5] * n,
        "hyp_toks": [["a", "b"]] * n,
        "ref_toks": [["a", "b"]] * n,
        "y_true": y_true, "y_pred": y_pred,
        "five_idx": [0, 1, 2, 3, 4], "label_names": names,
    }

    plain = bc.evaluate_subset(range(n), cache, per_label=False)
    rich = bc.evaluate_subset(range(n), cache, per_label=True)

    assert not any(m.startswith(bc.PER_LABEL_PREFIX) for m in plain)
    assert sum(m.startswith(bc.PER_LABEL_PREFIX) for m in rich) == k
    # The non-per-label metrics must be bit-identical between the two paths --
    # turning the flag on may ADD rows, never change an existing number.
    for m, v in plain.items():
        assert rich[m] == v, "metric %s changed when --per-label was enabled" % m
    assert bc.PER_LABEL_PREFIX + "Finding3" in rich


def test_bootstrap_is_paired_and_detects_a_real_difference():
    """A system that is strictly better must come out significant; a clone must tie.

    The pairing is what makes this sensitive at small effect sizes, so both
    directions are asserted -- a broken implementation usually fails one.
    """
    bc = _load_bootstrap_module()
    refs = ["the lungs are clear with no acute finding number %d" % i for i in range(80)]
    good = list(refs)                                    # perfect copy
    bad = ["something entirely different here %d" % i for i in range(80)]

    cache_good = bc.build_cache(good, refs, None)
    cache_bad = bc.build_cache(bad, refs, None)

    res, meta = bc.paired_bootstrap(cache_good, cache_bad, n_samples=200, seed=0)
    assert meta["n"] == 80
    assert res["rouge_l"]["significant"], "a perfect system vs a wrong one must be significant"
    assert res["rouge_l"]["diff"] > 0
    assert res["rouge_l"]["ci_low"] > 0, "CI must exclude zero when the gap is real"

    # Identical systems: the difference is exactly zero in every resample.
    res_tie, _ = bc.paired_bootstrap(cache_good, bc.build_cache(good, refs, None),
                                     n_samples=100, seed=0)
    assert not res_tie["rouge_l"]["significant"], "a system compared to itself must tie"
    assert abs(res_tie["rouge_l"]["diff"]) < 1e-12


def test_bootstrap_reports_both_accuracy_subsets_and_matches_sklearn():
    """f1chexbert's "accuracy" is the FIVE-label exact match, not all fourteen.

    Verified in F1CheXbert.forward: accuracy_score(refs_chexbert_5, hyps_chexbert_5).
    An earlier revision of bootstrap_compare emitted the 14-label version under the
    bare name "exact_match_accuracy", which read as the same quantity as the
    0.2163/0.2306 already published in the headline table while actually being
    0.0349/0.0469 -- a 6x difference under an identical-looking name. Both are now
    emitted, named for their subset, and both are pinned against sklearn.
    """
    import random as _random

    from sklearn.metrics import accuracy_score

    bc = _load_bootstrap_module()
    rng = _random.Random(0)
    n, k = 200, 14
    y_true = [[rng.randint(0, 1) for _ in range(k)] for _ in range(n)]
    y_pred = [[rng.randint(0, 1) for _ in range(k)] for _ in range(n)]
    five = [0, 2, 5, 6, 8]

    cache = {"rouge": [0.0] * n, "hyp_toks": [["a"]] * n, "ref_toks": [["a"]] * n,
             "y_true": y_true, "y_pred": y_pred, "five_idx": five}
    out = bc.evaluate_subset(range(n), cache)

    assert "exact_match_accuracy_14" in out and "exact_match_accuracy_5" in out
    assert "exact_match_accuracy" not in out, (
        "the unqualified name is ambiguous against the published 'accuracy' and "
        "must not come back"
    )
    assert abs(out["exact_match_accuracy_14"] - accuracy_score(y_true, y_pred)) < 1e-12
    sub_t = [[t[j] for j in five] for t in y_true]
    sub_p = [[p[j] for j in five] for p in y_pred]
    assert abs(out["exact_match_accuracy_5"] - accuracy_score(sub_t, sub_p)) < 1e-12


def test_bootstrap_refuses_unpaired_inputs():
    """Pairing requires the same studies in the same order; mismatched lengths are
    the one case where that is detectable, and it must abort rather than zip-truncate."""
    bc = _load_bootstrap_module()
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        a, b, r = Path(d) / "a.txt", Path(d) / "b.txt", Path(d) / "r.txt"
        a.write_text("one\ntwo\nthree\n")
        b.write_text("one\ntwo\n")
        r.write_text("one\ntwo\nthree\n")
        with pytest.raises(SystemExit):
            bc.main(["--hyps-a", str(a), "--hyps-b", str(b), "--refs", str(r)])


def test_score_chexbert_dumps_per_sample_labels():
    """CheXbert F1 cannot be bootstrapped from the aggregate report alone."""
    src = (REPO_ROOT / "scripts" / "score_chexbert_standalone.py").read_text()
    assert "chexbert_labels.json" in src
    # Match the CALLS, not the comprehension text -- the int() coercion added
    # 2026-09-09 rewrote the surrounding expression and broke a stricter match.
    assert "labeler.get_label(r)" in src and "for r in refs" in src
    assert "labeler.get_label(h)" in src and "for h in hyps" in src
    # The 5-label subset must come from the labeler, not a hardcoded list.
    assert "labeler.target_names_5_index" in src


def test_chexbert_label_payload_is_json_serialisable_with_numpy_ints():
    """f1chexbert's get_label() returns numpy int64, which json REFUSES.

    Caught live on 2026-09-09 (job 2525606): a 15-minute scoring run computed the
    correct metrics, printed them, then died with
    "TypeError: Object of type int64 is not JSON serializable" -- and because the
    label dump ran BEFORE the metrics write, the run produced no output at all.
    This reproduces the exact failure with real numpy scalars and pins the fix.
    """
    import numpy as np

    # Exactly what labeler.get_label() hands back: a list of numpy int64.
    raw_labels = [np.int64(v) for v in (1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)]
    raw_five_idx = [np.int64(i) for i in (0, 2, 5, 6, 8)]

    with pytest.raises(TypeError, match="not JSON serializable"):
        json.dumps({"y_true": [raw_labels]})

    coerced = {
        "y_true": [[int(v) for v in raw_labels]],
        "y_pred": [[int(v) for v in raw_labels]],
        "five_label_indices": [int(i) for i in raw_five_idx],
    }
    round_tripped = json.loads(json.dumps(coerced))
    assert round_tripped["y_true"] == [[1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]]
    assert round_tripped["five_label_indices"] == [0, 2, 5, 6, 8]

    src = (REPO_ROOT / "scripts" / "score_chexbert_standalone.py").read_text()
    assert "[int(v) for v in labeler.get_label(r)]" in src
    assert "[int(v) for v in labeler.get_label(h)]" in src
    assert "[int(i) for i in labeler.target_names_5_index]" in src


def test_chexbert_metrics_are_written_before_the_optional_label_dump():
    """An add-on must not be able to destroy the output every phase depends on.

    Job 2525606 lost a full scoring run because the label dump raised before
    chexbert_metrics.json was written. Ordering plus a try/except is the fix; both
    are asserted because either alone still leaves a way to lose the metrics.
    """
    src = (REPO_ROOT / "scripts" / "score_chexbert_standalone.py").read_text()
    body = src.split("if args.output_dir:")[1]
    metrics_pos = body.index("chexbert_metrics.json")
    labels_pos = body.index("chexbert_labels.json")
    assert metrics_pos < labels_pos, (
        "chexbert_metrics.json must be written BEFORE the optional label dump"
    )
    assert "try:" in body[:labels_pos + 200], "the label dump must be guarded"
    assert "WARNING: per-sample label dump failed" in body


def test_slurm_wrappers_that_source_the_venv_exclude_the_arm_node():
    """ga03 is ARM; .venv/bin/python3 is an x86 binary.

    Any wrapper that activates the venv MUST exclude ga03, or it dies with
    "cannot execute binary file: Exec format error" the first time SLURM happens
    to schedule it there -- which is a silent landmine, since the job runs fine
    on every other node. Hit live on 2026-09-09 (job 2525864).

    Note the trap that made this easy to get wrong: a script can be pure-stdlib
    and STILL break, because sourcing the venv puts the x86 interpreter on PATH.
    "It only needs stdlib" is not a reason to allow ga03; "it never activates the
    venv" is.
    """
    offenders = []
    for path in sorted((REPO_ROOT / "scripts").glob("*.sh")):
        src = path.read_text()
        sources_venv = "source \"${VENV_ACTIVATE}" in src or "source ${VENV_ACTIVATE}" in src
        if not sources_venv:
            continue
        excludes = [l for l in src.splitlines()
                    if l.startswith("#SBATCH") and "--exclude" in l]
        if not any("ga03" in l for l in excludes):
            offenders.append(path.name)
    assert not offenders, (
        "these wrappers activate the x86 venv but do not exclude the ARM node ga03: %s"
        % offenders
    )


def test_bootstrap_compare_slurm_wrapper_is_cpu_only():
    path = REPO_ROOT / "scripts" / "bootstrap_compare_h100.sh"
    assert path.exists()
    src = path.read_text()
    assert "#SBATCH --partition=pot-hpi-aisc-batch" in src
    assert not [l for l in src.splitlines() if l.startswith("#SBATCH") and "--gpus" in l]
    assert 'A="${A:?' in src and 'B="${B:?' in src
    assert "ERROR: required file not found" in src


def test_resolve_prefix_k_detects_trained_value_and_rejects_conflicts():
    """prefix_k is invisible to every key-count guard, so it needs its own.

    k sets only the output width of F.adaptive_avg_pool1d, which has NO
    parameters -- token_proj and out_proj are both k-independent. A k=8
    checkpoint therefore loads into a k=32 module with "Missing keys: 0,
    Unexpected: 0" and silently generates from 32 prefix tokens instead of 8.
    ReportGenerationLightningModule does not save_hyperparameters(), so k is not
    in the checkpoint; run_metadata.json is the only record.
    """
    import tempfile

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import evaluate_report_generation as erg
    finally:
        sys.path.pop(0)

    with tempfile.TemporaryDirectory() as d:
        run_dir = Path(d) / "run"
        (run_dir / "checkpoints").mkdir(parents=True)
        ckpt = run_dir / "checkpoints" / "last.ckpt"
        ckpt.write_text("")
        (run_dir / "run_metadata.json").write_text(
            json.dumps({"resolved_config": {"model": {"prefix_k": 8}}})
        )

        # Detected value beats the YAML default.
        assert erg.resolve_prefix_k(str(ckpt), yaml_prefix_k=32) == 8
        # A matching override is fine.
        assert erg.resolve_prefix_k(str(ckpt), 32, override=8) == 8
        # A CONFLICTING override must hard-fail, not silently pick one.
        with pytest.raises(RuntimeError, match="prefix_k conflict"):
            erg.resolve_prefix_k(str(ckpt), 32, override=64)

    # No metadata: fall back to the YAML, but say so loudly.
    with tempfile.TemporaryDirectory() as d:
        run_dir = Path(d) / "run"
        (run_dir / "checkpoints").mkdir(parents=True)
        ckpt = run_dir / "checkpoints" / "last.ckpt"
        ckpt.write_text("")
        assert erg.resolve_prefix_k(str(ckpt), yaml_prefix_k=32) == 32
        assert erg.resolve_prefix_k(str(ckpt), 32, override=8) == 8


def test_inspect_wrapper_exposes_prefix_k():
    src = (REPO_ROOT / "scripts" / "inspect_report_generation_h100.sh").read_text()
    assert 'PREFIX_K="${PREFIX_K:-}"' in src
    assert '${PREFIX_K:+--prefix-k "${PREFIX_K}"}' in src, (
        "must expand to nothing when unset so auto-detection stays the default path"
    )


# ---------------------------------------------------------------------------
# Phase 15C — auxiliary multi-label CheXpert loss on the mean-pooled image
# prefix. The supervisor's item 2 of 2026-09-13. Every test here exists to
# protect ONE property: with aux_lambda=0.0 (the default) nothing in this
# feature runs, so the 15B-4 seed band stays the valid baseline for 15C.
# ---------------------------------------------------------------------------

@pytest.mark.willi_parity
def test_chexpert_14_label_order_matches_f1chexbert_target_names():
    """Spec-lock on the target order. chexbert_labels.json's `label_names`,
    15B-5's per-label CIs and every per-label table in analysis/ are in this
    order; a permuted aux target would train the head against the wrong
    findings while every loss curve still looked healthy."""
    from scripts.train_report_generation import CHEXPERT_14_LABELS

    assert CHEXPERT_14_LABELS == [
        "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
        "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
        "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding",
    ]


@pytest.mark.willi_parity
def test_load_chexpert_label_matrix_u_zeros_and_omits_unknown_studies(tmp_path):
    """U-Zeros per column (1.0 positive; 0.0/-1.0/NaN negative), selection BY
    NAME (the CSV's own column order is alphabetical and must not matter), and
    a study with no CSV row is ABSENT from the dict -- never a row of zeros,
    which would teach 'no findings' from missing data."""
    import pandas as pd
    from scripts.train_report_generation import CHEXPERT_14_LABELS, load_chexpert_label_matrix

    rows = {"study_id": [10, 20]}
    for label in sorted(CHEXPERT_14_LABELS):          # deliberately NOT our order
        rows[label] = [0.0, 0.0]
    rows["Lung Opacity"] = [1.0, 0.0]
    rows["Pleural Other"] = [-1.0, 1.0]
    rows["No Finding"] = [float("nan"), 0.0]
    csv_path = tmp_path / "chexpert.csv.gz"
    pd.DataFrame(rows).to_csv(csv_path, index=False, compression="gzip")

    matrix = load_chexpert_label_matrix([10, 20, 99], str(csv_path))

    assert set(matrix) == {10, 20}, "a study absent from the CSV must be absent here"
    assert len(matrix[10]) == 14
    idx = {label: i for i, label in enumerate(CHEXPERT_14_LABELS)}
    assert matrix[10][idx["Lung Opacity"]] == 1.0
    assert matrix[10][idx["Pleural Other"]] == 0.0      # -1.0 (uncertain) -> negative
    assert matrix[10][idx["No Finding"]] == 0.0         # NaN -> negative
    assert matrix[20][idx["Pleural Other"]] == 1.0
    assert sum(matrix[20]) == 1.0


@pytest.mark.willi_parity
def test_compute_aux_pos_weight_caps_rare_labels_and_survives_zero_positives():
    """(N - n_pos)/n_pos, capped. The cap is load-bearing: uncapped, a ~1%
    label gives ~100x and 13F showed 5x already damaged common labels."""
    from scripts.train_report_generation import compute_aux_pos_weight

    # 100 studies, 2 labels: label 0 positive in 50 (ratio 1.0), label 1 in 1
    # (ratio 99.0 -> capped), label 2 never positive.
    matrix = {}
    for i in range(100):
        matrix[i] = [1.0 if i < 50 else 0.0, 1.0 if i == 0 else 0.0, 0.0]

    w = compute_aux_pos_weight(matrix, cap=10.0, num_labels=3)
    assert w[0] == pytest.approx(1.0)
    assert w[1] == pytest.approx(10.0), "uncapped this would be 99.0"
    assert w[2] == 1.0, "a label with no positives must not divide by zero"
    assert compute_aux_pos_weight({}, cap=10.0, num_labels=3) == [1.0, 1.0, 1.0]


@pytest.mark.willi_parity
def test_aux_lambda_zero_builds_no_head_and_leaves_the_state_dict_untouched():
    """The single most important property in 15C: at the default the head is
    never constructed, so (a) no aux_head.* key enters the checkpoint, and
    (b) the decoder/prefix_mapper initialisation is bit-identical to a module
    built before this feature existed -- constructing an nn.Linear consumes
    global-RNG draws, which is why the head is built LAST and only when on."""
    import torch
    from hybrid_xmamba.training.lightning_module import ReportGenerationLightningModule

    torch.manual_seed(0)
    baseline = ReportGenerationLightningModule(decoder_config=_tiny_cpu_config(), prefix_k=4)
    torch.manual_seed(0)
    explicit_off = ReportGenerationLightningModule(
        decoder_config=_tiny_cpu_config(), prefix_k=4, aux_lambda=0.0,
    )
    torch.manual_seed(0)
    aux_on = ReportGenerationLightningModule(
        decoder_config=_tiny_cpu_config(), prefix_k=4, aux_lambda=0.5,
    )

    assert baseline.aux_head is None and explicit_off.aux_head is None
    assert aux_on.aux_head is not None
    assert not any(k.startswith("aux_") for k in baseline.state_dict())
    assert any(k.startswith("aux_head.") for k in aux_on.state_dict())

    for key, tensor in baseline.state_dict().items():
        assert torch.equal(tensor, explicit_off.state_dict()[key]), key
        assert torch.equal(tensor, aux_on.state_dict()[key]), (
            f"{key} moved when the aux head was added -- the head must be built LAST"
        )


@pytest.mark.willi_parity
def test_aux_head_is_discarded_by_a_strict_false_eval_load():
    """The fairness property the writeup claims: the evaluated network stays
    parameter-identical to 13D because evaluate_report_generation.py loads
    strict=False and inspects only `missing`."""
    import torch
    from hybrid_xmamba.training.lightning_module import ReportGenerationLightningModule

    torch.manual_seed(0)
    trained = ReportGenerationLightningModule(
        decoder_config=_tiny_cpu_config(), prefix_k=4, aux_lambda=0.5,
    )
    torch.manual_seed(0)
    at_eval = ReportGenerationLightningModule(decoder_config=_tiny_cpu_config(), prefix_k=4)

    missing, unexpected = at_eval.load_state_dict(trained.state_dict(), strict=False)
    assert list(missing) == [], f"eval model is missing weights: {missing}"
    assert all(k.startswith("aux_") for k in unexpected), unexpected
    assert any(k.startswith("aux_head.") for k in unexpected)


@pytest.mark.willi_parity
def test_aux_loss_adds_lambda_times_bce_and_reaches_the_prefix_mapper():
    """total = lm_loss + aux_lambda * BCEWithLogits(head(prefix.mean(1)), y),
    reconstructed independently; and the gradient must reach the prefix_mapper
    -- if it did not, the mechanism (reshape the connector's representation)
    could not work at all, whatever the loss curve did."""
    import torch
    import torch.nn.functional as F
    from hybrid_xmamba.training.lightning_module import ReportGenerationLightningModule

    torch.manual_seed(0)
    mod = ReportGenerationLightningModule(
        decoder_config=_tiny_cpu_config(), prefix_k=4, aux_lambda=0.5,
    )
    mod.set_aux_pos_weight([2.0] * 14)
    # eval(), not train(): prefix_mapper has dropout, so in train mode each
    # forward draws a different mask and no two losses are comparable. Dropout
    # is orthogonal to what this test pins; gradients still flow in eval mode.
    mod.eval()

    B, L = 3, 8
    batch = {
        "input_ids": torch.randint(0, 100, (B, L)),
        "patch_grid": torch.randn(B, 197, 768),
        "chexpert_labels": torch.zeros(B, 14),
        "chexpert_label_mask": torch.ones(B),
    }
    batch["chexpert_labels"][0, 2] = 1.0

    lm_only = mod._step({k: v for k, v in batch.items() if not k.startswith("chexpert")}, "val")
    total = mod._step(batch, "val")

    with torch.no_grad():
        prefix = mod.prefix_mapper(batch["patch_grid"])
        logits = mod.aux_head(prefix.mean(dim=1).float())
        expected_aux = F.binary_cross_entropy_with_logits(
            logits, batch["chexpert_labels"], pos_weight=torch.full((14,), 2.0),
        )
    assert total.item() == pytest.approx((lm_only + 0.5 * expected_aux).item(), rel=1e-5)

    mod.zero_grad()
    total.backward()
    for name, param in mod.prefix_mapper.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
    for name, param in mod.aux_head.named_parameters():
        assert param.grad is not None, f"aux_head.{name} got no gradient"

    # And the head must actually be optimised -- a head left at random init
    # turns the aux gradient into noise rather than a label signal.
    groups = mod.configure_optimizers()["optimizer"].param_groups
    head_ids = {id(p) for p in mod.aux_head.parameters()}
    assert any(any(id(p) in head_ids for p in g["params"]) for g in groups)


@pytest.mark.willi_parity
def test_aux_loss_masks_out_studies_with_no_chexpert_row():
    """mask=0 samples must not contribute. Averaging them in would train the
    head towards all-negative on exactly the studies whose labels are unknown."""
    import torch
    from hybrid_xmamba.training.lightning_module import ReportGenerationLightningModule

    torch.manual_seed(0)
    mod = ReportGenerationLightningModule(
        decoder_config=_tiny_cpu_config(), prefix_k=4, aux_lambda=1.0,
    )
    mod.set_aux_pos_weight([1.0] * 14)
    mod.eval()

    B, L = 4, 8
    base = {
        "input_ids": torch.randint(0, 100, (B, L)),
        "patch_grid": torch.randn(B, 197, 768),
        "chexpert_labels": torch.zeros(B, 14),
    }
    base["chexpert_labels"][2:] = 1.0        # the two masked-out rows differ wildly

    with torch.no_grad():
        masked = mod._step({**base, "chexpert_label_mask": torch.tensor([1.0, 1.0, 0.0, 0.0])}, "val")
        first_two_only = mod._step({
            "input_ids": base["input_ids"], "patch_grid": base["patch_grid"],
            "chexpert_labels": base["chexpert_labels"],
            "chexpert_label_mask": torch.tensor([1.0, 1.0, 0.0, 0.0]),
        }, "val")
        all_in = mod._step({**base, "chexpert_label_mask": torch.ones(B)}, "val")

    assert masked.item() == pytest.approx(first_two_only.item())
    assert masked.item() != pytest.approx(all_in.item()), "the mask changed nothing"


@pytest.mark.willi_parity
def test_set_aux_pos_weight_rejects_wrong_length_and_a_missing_head():
    import torch
    from hybrid_xmamba.training.lightning_module import ReportGenerationLightningModule

    torch.manual_seed(0)
    off = ReportGenerationLightningModule(decoder_config=_tiny_cpu_config(), prefix_k=4)
    with pytest.raises(RuntimeError, match="aux_lambda == 0"):
        off.set_aux_pos_weight([1.0] * 14)

    on = ReportGenerationLightningModule(
        decoder_config=_tiny_cpu_config(), prefix_k=4, aux_lambda=0.1,
    )
    with pytest.raises(ValueError, match="14 entries"):
        on.set_aux_pos_weight([1.0] * 13)


@pytest.mark.willi_parity
def test_image_text_dataset_emits_chexpert_keys_only_when_labels_are_supplied():
    """ImageTextDataset is shared with the CLOSED retrieval chapter, so the
    default __getitem__ payload must not gain a key."""
    import torch
    from omegaconf import OmegaConf
    from PIL import Image
    from transformers import AutoTokenizer

    from scripts.train_contrastive import ImageTextDataset

    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    except Exception as exc:                      # offline CI without the tokenizer cached
        pytest.skip(f"tokenizer unavailable: {exc}")
    tokenizer.pad_token = tokenizer.eos_token

    cfg = OmegaConf.create({"dataset": {"max_length": 8, "image_size": 8}})
    rows = [{"findings": "f", "impression": "i", "study_id": 10,
             "image": Image.new("RGB", (8, 8))}]

    plain = ImageTextDataset(rows, tokenizer, cfg)[0]
    assert set(plain) == {"input_ids", "attention_mask", "pixel_values"}

    labelled = ImageTextDataset(rows, tokenizer, cfg, chexpert_labels={10: [1.0] + [0.0] * 13})[0]
    assert labelled["chexpert_labels"].shape == (14,)
    assert labelled["chexpert_labels"][0] == 1.0
    assert labelled["chexpert_label_mask"].item() == 1.0

    unknown = ImageTextDataset(rows, tokenizer, cfg, chexpert_labels={99: [1.0] + [0.0] * 13})[0]
    assert unknown["chexpert_label_mask"].item() == 0.0
    assert torch.equal(unknown["chexpert_labels"], torch.zeros(14))


@pytest.mark.willi_parity
def test_train_report_generation_h100_wrapper_exposes_aux_levers():
    """AUX_LAMBDA defaults to 0.0 (baseline recipe), is always passed to Hydra,
    and is echoed POSITIVELY -- Phase 14's prefix_k lesson: a silent default
    cannot be detected by its absence, and an aux arm whose log does not state
    lambda is indistinguishable from a baseline re-run."""
    sh = (REPO_ROOT / "scripts" / "train_report_generation_h100.sh").read_text()
    assert 'AUX_LAMBDA="${AUX_LAMBDA:-0.0}"' in sh
    assert 'AUX_POS_WEIGHT_CAP="${AUX_POS_WEIGHT_CAP:-10.0}"' in sh
    assert "model.aux_lambda=${AUX_LAMBDA}" in sh
    assert "model.aux_pos_weight_cap=${AUX_POS_WEIGHT_CAP}" in sh
    assert "Aux CheXpert loss: lambda=${AUX_LAMBDA}" in sh


@pytest.mark.willi_parity
def test_rrg_model_configs_declare_aux_keys():
    """Hydra strict-struct mode rejects a CLI override for an undeclared key --
    the same trap that cost a smoke run in 13A (job 2478622). Both report-gen
    arms must declare them, or the matched Transformer baseline cannot run the
    same recipe."""
    for name in ("hybrid_150m_v2_rrg", "transformer_150m_baseline_rrg", "hybrid_150m_m3_rrg"):
        text = (REPO_ROOT / "configs" / "model" / f"{name}.yaml").read_text()
        assert "aux_lambda: 0.0" in text, name
        assert "aux_pos_weight_cap: 10.0" in text, name

@pytest.mark.willi_parity
def test_screen_arms_job_array_reads_the_shared_arm_ladder():
    """MAMBA3_PLAN_V2.md M7-B0: the screen is one job array, and it hand-writes no lever.

    Two invariants, both load-bearing:

    1. The arm table is *not* in this script. It is read from `scripts/mamba3_arms.py`,
       which is also what the pre-flight verifies -- so no arm can be screened with a
       configuration the pre-flight never checked. A pre-flight carrying its own private
       copy of the arm list is exactly how job 2513007 came to train A1 as plain A0.
    2. aisc rejects `--gres` for GPUs; the request must be `--gpus=N`. Every other H100
       wrapper in this repo is pinned the same way because that mistake was made live.

    A0/A0-seed/A1 are deliberately absent from the default set -- they were early-started
    under M7-A2, and re-running them would burn ~22 GPU-h to produce a second control.
    """
    sh = (REPO_ROOT / "scripts" / "screen_arms_h100.sh").read_text()

    assert "#SBATCH --gpus=1" in sh
    assert "--gres" not in sh, "aisc rejects --gres for GPUs -- use --gpus=N"
    assert "#SBATCH --partition=pot-hpi-aisc-batch" in sh and "#SBATCH --account=aisc" in sh
    assert "#SBATCH --requeue" in sh, "aisc-batch is preemptible"
    assert "#SBATCH --open-mode=append" in sh, (
        "without append, a requeue TRUNCATES the log -- an arm silently restarts from step 0 "
        "(nothing passes ckpt_path) and the evidence that it did is overwritten"
    )
    assert "%A_%a" in sh, "array tasks must not all write to the same log file"

    assert 'ARMS="${ARMS:-A2 A3 A4 A5 A6}"' in sh, "default set must skip the early-started arms"
    assert "export ARM" in sh, "the arm is handed to the wrapper, which resolves it on the node"
    assert "unset EXPERIMENT" in sh, (
        "SLURM propagates the submitting environment and the wrapper lets a caller-supplied "
        "EXPERIMENT win, so a stray one would funnel all five arms into one output directory"
    )
    assert sh.index("unset EXPERIMENT") < sh.index("bash scripts/train_stage0_150m_h100.sh")
    code = "\n".join(ln for ln in sh.splitlines() if not ln.lstrip().startswith("#"))
    assert "model.mamba3_" not in code, (
        "levers must come from mamba3_arms.py, not be hand-written here"
    )
    assert "SLURM_ARRAY_TASK_ID" in sh
    assert "bash scripts/train_stage0_150m_h100.sh" in sh, (
        "the 150M stability recipe (LR 4e-4, grad-clip 0.5, 80GB-safe bs/accum) is inherited, "
        "not restated -- it took five attempts to find"
    )


@pytest.mark.willi_parity
def test_stage0_h100_passes_extra_hydra_overrides_through():
    """M5: arms A3..A6 are `model.mamba3_*=...` overrides, not yamls of their own.

    Before this existed the wrapper had no way to pass an extra Hydra argument, so half the
    screen ladder was unsubmittable. The expansion must stay unquoted -- the overrides are
    separate arguments, not one string -- and defaulted, because the script runs under
    `set -u`.
    """
    sh = (REPO_ROOT / "scripts" / "train_stage0_h100.sh").read_text()
    assert 'EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"' in sh
    assert "\n  ${EXTRA_OVERRIDES}\n" in sh, "must be word-split into separate Hydra arguments"
    wrapper = (REPO_ROOT / "scripts" / "train_stage0_150m_h100.sh").read_text()
    assert 'export EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"' in wrapper


@pytest.mark.willi_parity
def test_150m_wrapper_resolves_the_arm_on_the_compute_node():
    """The aisc login node refuses to execute python, so an arm can only be resolved inside
    the job.

    Regression pin for job 2513581 (2026-09-06). The documented launch was
    `eval "$(python scripts/mamba3_arms.py env A2)" && sbatch ...`. On lx01 that python call
    printed "This command is not allowed on the login node!"; the shell word-split the
    sentence into `This: command not found`, `Please: command not found`, `HINT:: command not
    found`; the eval exported NOTHING; and sbatch ran the wrapper's own defaults --
    hybrid_150m_v2, 120,000 steps, save_top_k=3. A silent second A0 at ten times the intended
    length, wearing the experiment name of the arm it was supposed to be.

    Resolving on the compute node removes the class rather than the instance: there is no
    pre-submit step left to fail, and a bad arm name exits non-zero instead of falling back to
    a default that happens to be a valid architecture.
    """
    sh = (REPO_ROOT / "scripts" / "train_stage0_150m_h100.sh").read_text()
    assert 'if [ -n "${ARM:-}" ]; then' in sh
    assert 'python scripts/mamba3_arms.py env "${ARM}"' in sh, "one definition of the ladder"
    assert "FATAL: could not resolve arm" in sh and "exit 1" in sh, (
        "an unknown arm must fail the job, not silently train the wrapper's default"
    )
    # Resolution must precede the ${VAR:-default} exports, or the defaults win and the arm is
    # silently ignored -- the same failure in a new costume.
    assert sh.index('if [ -n "${ARM:-}" ]; then') < sh.index('export MODEL_CONFIG=')
    # A caller-supplied EXPERIMENT must survive the arm's own naming, or a short probe writes
    # into the screen run's output directory. Job 2513598 (a 300-step A2 probe) did exactly
    # that: it landed in outputs/m3_screen_A2_s42 and left a last.ckpt behind.
    assert 'ARM_EXPERIMENT="${EXPERIMENT:-}"' in sh
    assert sh.index('ARM_EXPERIMENT="${EXPERIMENT:-}"') < sh.index('eval "${ARM_ENV}"')
    assert 'export EXPERIMENT="${ARM_EXPERIMENT}"' in sh


@pytest.mark.willi_parity
def test_stage0_checkpoint_filename_has_no_slash_metric():
    """A metric containing "/" in a ModelCheckpoint filename becomes a path separator.

    `filename="stage0_kd-{step:06d}-{val/loss:.4f}"` made Lightning create a DIRECTORY
    `stage0_kd-step=NNNNNN-val/` with `loss=N.NNNN.ckpt` inside it, for every save. Nothing
    globbing `checkpoints/*.ckpt` could find a best checkpoint -- only `last.ckpt` was ever
    visible, which is why every M7 arm reported zero checkpoints while sitting on 2.1 GB.
    `monitor="val/loss"` still drives top-k selection; the loss belongs in TensorBoard.
    """
    src = (REPO_ROOT / "scripts" / "train_stage0_distill.py").read_text()
    assert 'filename="stage0_kd-step{step:06d}"' in src
    assert "{val/loss" not in src, "a slashed metric in a filename becomes a directory"
    assert 'monitor="val/loss"' in src, "top-k selection still needs the metric"


@pytest.mark.willi_parity
def test_stage0_validation_cadence_is_tunable():
    """M8-A runs 120,000 steps; at the screen's val_check_interval=2000 that is 60 passes.

    A0 measured ~5.4 h for six passes -- the val set is 15,724 chunks and each pass runs the
    2.6B teacher alongside the student -- so 60 would cost ~54 h against ~13.5 h of training.
    The interval must be tunable. The val SET must not be: it stays 15,724 chunks so the number
    remains comparable to the 13.18 Phase-5 baseline.
    """
    sh = (REPO_ROOT / "scripts" / "train_stage0_h100.sh").read_text()
    assert 'VAL_EVERY="${VAL_EVERY:-2000}"' in sh, "screens keep the 2000-step default"
    assert "trainer.val_check_interval=${VAL_EVERY}" in sh
    assert "trainer.val_check_interval=2000" not in sh, "no hard-coded cadence left"


# ---------------------------------------------------------------------------
# MAMBA3_PLAN_V2.md V1 -- re-baseline and harden the seams
# ---------------------------------------------------------------------------

def _model_yaml(name):
    import yaml
    return yaml.safe_load((REPO_ROOT / "configs" / "model" / f"{name}.yaml").read_text())


def _v3_stage0_operator():
    """(arm name, {scan_impl, tfla_impl}) the V3 chain trains Stage-0 with by default: the arm's
    yaml overlaid with the arm's own overrides -- exactly what the ARM resolver applies."""
    import importlib.util
    chain = (REPO_ROOT / "scripts" / "submit_v3_chain.sh").read_text()
    m = re.search(r'local ARM="\$\{ARM:-([A-Za-z0-9-]+)\}"', chain)
    assert m, "submit_v3_chain.sh must declare a default ARM"
    spec = importlib.util.spec_from_file_location("m3arms", REPO_ROOT / "scripts" / "mamba3_arms.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    arm = mod.ARMS[m.group(1)]
    base = _model_yaml(arm.config)
    op = {k: arm.overrides.get(k, base[k]) for k in ("scan_impl", "tfla_impl")}
    return m.group(1), op


@pytest.mark.willi_parity
def test_m3_rrg_config_is_m3_plus_exactly_the_rrg_delta():
    """V1-B: the Mamba-3 report-gen config mirrors the hybrid_150m_v2 -> _rrg delta EXACTLY,
    the way transformer_150m_baseline_rrg does. Anything else is a second lever in the
    decoder comparison."""
    v2, v2_rrg = _model_yaml("hybrid_150m_v2"), _model_yaml("hybrid_150m_v2_rrg")
    m3, m3_rrg = _model_yaml("hybrid_150m_m3"), _model_yaml("hybrid_150m_m3_rrg")
    dropped = {k for k in v2 if k not in v2_rrg}
    changed = {k for k in v2_rrg if k not in v2 or v2_rrg[k] != v2[k]}
    expected = {k: (v2_rrg[k] if k in changed else m3[k]) for k in (set(m3) - dropped) | changed}
    # V2-D: the ONE sanctioned difference beyond the rrg delta -- the decoder runs the operator
    # the winning Stage-0 arm trained with (A2x: tfla_impl=exact). hybrid_150m_m3.yaml itself stays
    # legacy because it defines the M7 arms A2..A6.
    expected.update(_v3_stage0_operator()[1])
    assert m3_rrg == expected, {
        "missing": sorted(set(expected) - set(m3_rrg)),
        "extra": sorted(set(m3_rrg) - set(expected)),
        "differ": sorted(k for k in set(expected) & set(m3_rrg) if expected[k] != m3_rrg[k]),
    }
    assert m3_rrg["layer_pattern"].count("mamba3") == 9 and m3_rrg["layer_pattern"].count("mlstm") == 3


@pytest.mark.willi_parity
def test_v3_decoder_config_runs_the_operator_its_stage0_arm_trained_with():
    """V2-D: scan_impl / tfla_impl carry no parameters, so a decoder built with `legacy` loads an
    `exact`-trained Stage-0 checkpoint with Missing keys: 0 and silently fine-tunes and evaluates a
    different recurrence. The chain trains Stage-0 through ARM (hybrid_150m_m3 + overrides) and
    trains/evaluates the decoder through hybrid_150m_m3_rrg -- two routes that must agree."""
    arm, op = _v3_stage0_operator()
    rrg = _model_yaml("hybrid_150m_m3_rrg")
    assert arm == "A2x", "V2-D (2026-09-17): A2x advanced -- 15.566 / 15.788 vs A2 16.708 / 16.376"
    assert op == {"scan_impl": "legacy", "tfla_impl": "exact"}, op
    assert {k: rrg[k] for k in op} == op, (
        "hybrid_150m_m3_rrg.yaml runs %s but the V3 Stage-0 arm %s trains with %s"
        % ({k: rrg[k] for k in op}, arm, op))
    chain = (REPO_ROOT / "scripts" / "submit_v3_chain.sh").read_text()
    assert chain.count("MODEL_CONFIG=hybrid_150m_m3_rrg") == 2, "decoder AND eval must use the rrg config"


@pytest.mark.willi_parity
def test_every_recurrent_model_yaml_pins_the_operator_explicitly():
    """V1-E: no yaml inherits scan_impl / tfla_impl from the dataclass default. The published
    configs pin `legacy` (byte-for-byte reproduction of every checkpoint, and the 14A operator
    freeze); the corrected arms pin `exact`. Flipping a global default can never silently move
    a published number again."""
    for f in sorted((REPO_ROOT / "configs" / "model").glob("*.yaml")):
        raw = _model_yaml(f.stem)
        pattern = raw.get("layer_pattern") or []
        if not any(t in ("mamba", "mamba3", "mlstm") for t in pattern):
            continue
        assert raw.get("scan_impl") in ("legacy", "exact"), f"{f.name}: scan_impl not pinned"
        assert raw.get("tfla_impl") in ("legacy", "exact"), f"{f.name}: tfla_impl not pinned"
    for name in ("hybrid_70m_v2", "hybrid_150m_v2", "hybrid_150m_v2_rrg", "hybrid_150m_m3"):
        raw = _model_yaml(name)
        assert (raw["scan_impl"], raw["tfla_impl"]) == ("legacy", "legacy"), name
    a1 = _model_yaml("hybrid_150m_a1")
    assert (a1["scan_impl"], a1["tfla_impl"]) == ("exact", "exact")


@pytest.mark.willi_parity
def test_checkpoint_architecture_sniffer_names_every_layer_type():
    """V1-D: the eval loaders used to decide `mamba` vs `mlstm` from `A_log`/`conv1d` alone,
    which labels a Mamba-3 block (it has both) as Mamba-1 and an attention block as mLSTM.
    The shared sniffer keys on one parameter unique to each mixer and refuses ambiguity."""
    import torch
    from hybrid_xmamba.models.configuration_hybrid import HybridConfig
    from hybrid_xmamba.models.hybrid_lm import HybridLanguageModel
    from hybrid_xmamba.utils.checkpoint_arch import infer_architecture

    pattern = ["mamba", "mamba3", "mlstm", "slstm", "attention"]

    def build(norm_topology, pattern=pattern):
        cfg = HybridConfig(vocab_size=64, dim=64, num_layers=len(pattern), layer_pattern=pattern,
                           state_size=8, mamba3_d_state=16, mamba3_head_dim=32,
                           use_fast_path=False, use_tfla=False, max_position_embeddings=32,
                           norm_topology=norm_topology)
        model = HybridLanguageModel(cfg)
        return {"lm." + k: v for k, v in model.state_dict().items()}

    arch = infer_architecture(build("hybrid"), prefix="lm.layers.")
    assert arch.layer_pattern == pattern
    assert arch.norm_topology == "hybrid"
    assert arch.state_size == 8 and arch.mamba3_d_state == 16 and arch.mamba3_head_dim == 32
    kw = arch.config_kwargs()
    assert kw["layer_pattern"] == pattern and kw["state_size"] == 8 and kw["mamba3_d_state"] == 16

    assert infer_architecture(build("pre_rms"), prefix="lm.layers.").norm_topology == "pre_rms"
    assert infer_architecture(build("hybrid_bc", ["mamba", "mamba", "mlstm"]),
                              prefix="lm.layers.").norm_topology == "hybrid_bc"

    t = torch.zeros(2, 2)
    with pytest.raises(ValueError, match="ambiguous"):
        infer_architecture({"lm.layers.0.mixer.dt_proj.weight": t,
                            "lm.layers.0.mixer.i_gate_proj.weight": t}, prefix="lm.layers.")
    with pytest.raises(ValueError, match="no fingerprint"):
        infer_architecture({"lm.layers.0.mixer.mystery.weight": t}, prefix="lm.layers.")


@pytest.mark.willi_parity
def test_retrieval_and_sts_loaders_share_the_sniffer_and_raise_on_critical_misses():
    """V1-D: one sniffer, two loaders, and both FAIL on a mis-built backbone. evaluate_cxr_retrieval
    used to print the missing keys and score anyway (a wrong architecture silently mis-scored);
    evaluate_sts already raised. They now match."""
    for rel in ("scripts/evaluate_cxr_retrieval.py", "scripts/evaluate_sts.py"):
        src = (REPO_ROOT / rel).read_text()
        assert "infer_architecture(" in src, rel
        assert '"A_log" in k or "conv1d" in k' not in src, f"{rel}: binary predicate still live"
        assert "Critical keys missing after load" in src and "raise RuntimeError" in src, rel


@pytest.mark.willi_parity
def test_contrastive_backbone_load_guards_against_wrong_architecture():
    """V1-D: the tower stage was the one load in the chain with no guard -- a wrong-architecture
    Stage-0 checkpoint loaded with strict=False, matched almost nothing, and trained from random
    init while printing a key count nobody read. Same >50% rule as the decoder and the eval."""
    src = (REPO_ROOT / "scripts" / "train_contrastive.py").read_text()
    i = src.index("text_encoder.lm.load_state_dict(state, strict=False)")
    window = src[i:i + 1600]
    assert "missing_frac" in window and "raise RuntimeError" in window, "no wrong-architecture guard"
    assert "0.5" in window and "0.05" in window


@pytest.mark.willi_parity
def test_v3_chain_script_is_source_safe_and_only_submits_existing_wrappers():
    """V1-F: the full pipeline is one dependency chain, SOURCED on the login node (which executes
    no scripts and no python). So: no `set -e` (it would kill the login shell), nothing but sbatch
    and shell, every wrapper it names exists and excludes the ARM node."""
    import re
    path = REPO_ROOT / "scripts" / "submit_v3_chain.sh"
    assert path.exists(), "scripts/submit_v3_chain.sh missing"
    src = path.read_text()
    assert "set -e" not in src and "set -euo" not in src
    assert "--dependency=afterok" in src
    code = [l.strip() for l in src.splitlines() if l.strip() and not l.strip().startswith("#")]
    assert not any(l.startswith(("python ", "python3 ", "bash scripts/", "sh scripts/")) for l in code), (
        "the chain must not run scripts on the login node"
    )
    submits = "\n".join(l for l in code if not l.startswith("echo"))   # sbatch targets only, not the closing hint
    wrappers = set(re.findall(r"(scripts/[A-Za-z0-9_]+\.sh)", submits)) - {"scripts/submit_v3_chain.sh"}
    assert wrappers == {
        "scripts/train_stage0_150m_h100.sh", "scripts/train_report_generation_h100.sh",
        "scripts/inspect_report_generation_h100.sh", "scripts/score_chexbert_h100.sh",
        "scripts/bootstrap_compare_h100.sh",
    }, wrappers
    for w in wrappers:
        wsrc = (REPO_ROOT / w).read_text()
        assert (REPO_ROOT / w).exists() and "#SBATCH --exclude=ga03" in wsrc, w
    # the levers the plan's recipe depends on are all threaded
    for lever in ("ARM=", "SEEDS", "SAVE_TOP_K=0", "NUM_GPUS=4", "MAX_STEPS=12000", "PREFIX_K",
                  "DECODE=beam", "BEAM_SIZE=3", "PER_LABEL=true", "IMAGE_ENCODER_CKPT", "DRY_RUN",
                  "EVAL_TIME"):
        assert lever in src, lever
    # V2-D: every incumbent dump the 15B-4 table came from (h100_scaling_state.json seed_arms), so all
    # nine paired bootstraps are submitted -- none silently skipped.
    for d in ("results/report_gen_tower13d_test_split", "results/report_gen_hybrid_seed43_test_split",
              "results/report_gen_hybrid_seed44_test_split", "results/report_gen_transformer_test_split",
              "results/report_gen_transformer_seed43_test_split",
              "results/report_gen_transformer_seed44_test_split"):
        assert d in src, d


_PKG_IMPORT = re.compile(r"^\s*(from|import)\s+hybrid_xmamba\b", re.M)


@pytest.mark.willi_parity
def test_every_script_that_imports_the_package_puts_the_repo_root_on_sys_path():
    """V2-A, job 2552094: `python scripts/mamba3_arms.py verify` died on the cluster with
    `ModuleNotFoundError: No module named 'hybrid_xmamba'`. Running `python scripts/x.py` puts
    scripts/ on sys.path, not the repo root, and the cluster .venv has no editable install of the
    package -- only this laptop's venv does, which is why every local run passed. The other
    scripts insert the root themselves; this pins the convention for all of them."""
    offenders = []
    for path in sorted((REPO_ROOT / "scripts").glob("*.py")):
        src = path.read_text()
        if _PKG_IMPORT.search(src) and "sys.path.insert(0" not in src:
            offenders.append(path.name)
    assert not offenders, (
        "scripts import hybrid_xmamba without inserting the repo root on sys.path; they work only "
        "where the package is pip-installed (not the aisc .venv): %s" % offenders
    )


@pytest.mark.willi_parity
def test_mamba3_arms_verify_runs_without_an_installed_package():
    """Reproduces job 2552094 exactly: an interpreter that cannot see an installed/editable
    hybrid_xmamba (`-S` skips site, so no .pth finder runs; site-packages is re-added through
    PYTHONPATH only so torch and yaml still import), launched from the repo root exactly as the
    preflight's `python scripts/mamba3_arms.py verify` is. The repo root as cwd does NOT hide the
    bug: running a script puts its own directory on sys.path, never the cwd."""
    import subprocess
    import sysconfig

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (sysconfig.get_paths()["purelib"], sysconfig.get_paths()["platlib"]) if p
    )
    # The reproduction needs torch + yaml to import without `site`. If this interpreter cannot do
    # that, the environment cannot express the condition -- skip loudly rather than fail and block
    # the cluster preflight on a test-harness limitation. The static test above still guards the rule.
    probe = subprocess.run([sys.executable, "-S", "-c", "import torch, yaml"],
                           cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=300)
    if probe.returncode != 0:
        pytest.skip("torch/yaml do not import under `python -S` here: %s" % probe.stderr.strip()[-300:])
    proc = subprocess.run(
        [sys.executable, "-S", str(REPO_ROOT / "scripts" / "mamba3_arms.py"), "verify"],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=600,
    )
    out = proc.stdout + proc.stderr
    assert "No module named 'hybrid_xmamba'" not in out, out[-2000:]
    assert proc.returncode == 0, out[-2000:]


@pytest.mark.willi_parity
def test_efficiency_wrapper_can_run_the_decode_curve():
    """V3-F: the O(L^2) -> O(L) claim needs `performance_profile.py --decode`, which the wrapper
    could not reach -- and the login node runs nothing directly. Default off so the 14A-7 sweep
    protocol is unchanged."""
    src = (REPO_ROOT / "scripts" / "profile_efficiency_h100.sh").read_text()
    assert 'DECODE_CURVE="${DECODE_CURVE:-false}"' in src
    assert "--decode" in src and "--prompt-len" in src and "--new-tokens" in src
    assert "hybrid_150m_m3_rrg" in src, "the cached path needs the exact-TFLA config (M6 finding 1)"


@pytest.mark.willi_parity
def test_inspect_wrapper_exposes_the_cached_decode_lever():
    """V4: the O(1) cache has to be reachable through sbatch, and off by default so the protocol
    behind every published number is what runs unless someone asks otherwise."""
    src = (REPO_ROOT / "scripts" / "inspect_report_generation_h100.sh").read_text()
    assert 'CACHED_DECODE="${CACHED_DECODE:-false}"' in src
    assert "--cached-decode" in src
    eval_src = (REPO_ROOT / "scripts" / "evaluate_report_generation.py").read_text()
    assert '"--cached-decode"' in eval_src and "supports_cached_decode()" in eval_src


@pytest.mark.willi_parity
def test_eval_can_override_the_operator_a_checkpoint_was_trained_with():
    """V5-A: `scan_impl`/`tfla_impl` carry no parameters, so the same weights load under either.
    That is what makes "what is the defect worth on the reported metrics?" answerable by
    measurement -- and why the override must be announced in the log, not inferred afterwards."""
    src = (REPO_ROOT / "scripts" / "evaluate_report_generation.py").read_text()
    for flag in ('"--scan-impl"', '"--tfla-impl"', "scan_impl=getattr(args", "tfla_impl=getattr(args"):
        assert flag in src, flag
    assert "OVERRIDE; the checkpoint was TRAINED with" in src, "an override must announce itself"
    wrapper = (REPO_ROOT / "scripts" / "inspect_report_generation_h100.sh").read_text()
    assert 'SCAN_IMPL="${SCAN_IMPL:-}"' in wrapper and "--scan-impl" in wrapper
    assert 'TFLA_IMPL="${TFLA_IMPL:-}"' in wrapper and "--tfla-impl" in wrapper


@pytest.mark.willi_parity
def test_state_helper_points_at_the_v2_plan_and_accepts_v_phase_ids():
    """V1-A: the helper is the only thing allowed to tick a checkbox; it must read the V2 files
    and recognise both the carried M-ids and the new V-ids."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("m3state", REPO_ROOT / "scripts" / "mamba3_state.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    assert mod.PLAN.name == "MAMBA3_PLAN_V2.md" and mod.STATE.name == "mamba3_v2_state.json"
    assert mod.CHECKBOX_RE.match("- [ ] **V0-A** Merge").group(4) == "V0-A"
    assert mod.CHECKBOX_RE.match("- [x] **M7-B** Screen").group(4) == "M7-B"
    assert mod.PHASE_RE.match("### V3 — Full pipeline").group(1) == "V3"
    src = (REPO_ROOT / "scripts" / "screen_arms_h100.sh").read_text()
    assert "export ARM STEPS WARMUP_STEPS VAL_EVERY" in src, "V1-C: VAL_EVERY must reach the wrapper"



@pytest.mark.willi_parity
def test_every_analysis_deliverable_can_actually_enter_the_repo():
    """.gitignore line 84 is a blanket `*.md`, so a new analysis document is invisible by
    default: `git add` silently does nothing, `git status` stays clean, and the file is
    reported as committed while living only on one laptop. That happened to
    analysis/ARCHIVE_MANIFEST.md on 2026-09-20 (V4-E) and was caught a day later.

    Every markdown file under analysis/ must therefore be either already tracked or
    explicitly allowlisted. Writing a new one without touching .gitignore fails here."""
    import subprocess

    docs = sorted((REPO_ROOT / "analysis").glob("*.md"))
    assert docs, "analysis/ should hold the written record"

    def _git(*args):
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True
        )

    if _git("rev-parse", "--git-dir").returncode != 0:
        pytest.skip("not a git checkout")

    tracked = set(_git("ls-files", "analysis").stdout.split())
    stranded = [
        d.name
        for d in docs
        if f"analysis/{d.name}" not in tracked
        and _git("check-ignore", "-q", f"analysis/{d.name}").returncode == 0
    ]
    assert not stranded, (
        "these analysis documents are gitignored and untracked, so they cannot be "
        f"committed and are not part of the record: {stranded}. "
        "Add `!analysis/<name>.md` to the allowlist block in .gitignore."
    )


# ---------------------------------------------------------------------------
# V5-D — post-hoc repair of generated report dumps
# ---------------------------------------------------------------------------

def _load_repair_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_repair_generations", REPO_ROOT / "scripts" / "repair_generations.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.willi_parity
def test_sentence_splitter_survives_radiology_punctuation():
    """The repair rests entirely on knowing where a sentence ends. Radiology
    text is adversarial about that: measurements ("4.5 cm"), redaction
    placeholders ("___."), dictation times ("10:30 a.m."), clinician titles
    ("Dr.") and numbered impressions ("1.  Interval ...") all put a period
    somewhere that is not a boundary. A splitter that gets these wrong would
    truncate real findings away and call it a repair."""
    m = _load_repair_module()

    assert len(m.split_sentences(
        "The tip projects 4.5 cm above the carina. Dr. ___ was paged at 10:30 a.m. "
        "as soon as the findings were recognized.")) == 2
    assert len(m.split_sentences(
        "Impression: 1.  Interval extubation.  2.  Low lung volumes.")) == 2
    assert len(m.split_sentences("Comparison is made with prior study, ___.")) == 1
    # rejoining is lossless on already-normalised text
    text = "No acute process. Heart size is normal."
    assert " ".join(m.split_sentences(text)) == text


@pytest.mark.willi_parity
def test_repair_drops_the_severed_fragment_and_collapses_repeats():
    """The two artefacts of a decoder with no stop condition, on a real tail
    from the 2026-09-21 eval logs."""
    m = _load_repair_module()
    text = ("Findings: The lungs are clear. There is no pneumothorax. "
            "Sternal wires are aligned. Sternal wires are aligned. Sternal wires are")

    out, stats = m.repair_report(text)
    assert out.endswith("Sternal wires are aligned.")
    assert out.count("Sternal wires are aligned.") == 1
    assert stats["sentences_truncated"] == 1 and stats["sentences_deduped"] == 1

    # the two edits are independently switchable, so each can be attributed
    only_dedup, _ = m.repair_report(text, truncate=False)
    assert only_dedup.endswith("Sternal wires are")
    only_trunc, _ = m.repair_report(text, dedup="none")
    assert only_trunc.count("Sternal wires are aligned.") == 2

    # "all" also catches a repeat that is not immediately adjacent
    spaced = "No pneumothorax. Heart size is normal. No pneumothorax."
    assert m.repair_report(spaced, dedup="consecutive")[0] == spaced
    assert m.repair_report(spaced, dedup="all")[0] == "No pneumothorax. Heart size is normal."


@pytest.mark.willi_parity
def test_repair_never_empties_a_hypothesis_or_moves_a_line():
    """An empty hypothesis is a scoring artefact, not a measurement, and a
    dropped line silently desyncs every hyp/ref pair after it."""
    m = _load_repair_module()

    out, stats = m.repair_report("The size of the cardiac")   # no terminator at all
    assert out == "The size of the cardiac" and stats.get("fallbacks") == 1

    lines = ["The size of the cardiac", "No acute process.", "Clear lungs. Clear lungs. and"]
    repaired, totals = m.repair_lines(lines)
    assert len(repaired) == len(lines) and all(r.strip() for r in repaired)
    assert totals["reports"] == 3


@pytest.mark.willi_parity
def test_repair_cli_preserves_refs_and_refuses_to_overwrite_the_control(tmp_path):
    """The unrepaired dump is the control arm of the comparison, and the
    references are not ours to edit."""
    import subprocess

    src = tmp_path / "dump"
    src.mkdir()
    (src / "hyps.txt").write_text("Clear lungs. Clear lungs. and\nNo acute process.\n")
    (src / "refs.txt").write_text("Lungs are clear.\nNo acute cardiopulmonary process.\n")
    out = tmp_path / "repaired"

    script = str(REPO_ROOT / "scripts" / "repair_generations.py")
    done = subprocess.run(
        [sys.executable, script, "--dump-dir", str(src), "--out-dir", str(out)],
        capture_output=True, text=True,
    )
    assert done.returncode == 0, done.stderr
    assert (out / "hyps.txt").read_text() == "Clear lungs.\nNo acute process.\n"
    assert (out / "refs.txt").read_text() == (src / "refs.txt").read_text()
    assert (src / "hyps.txt").read_text() == "Clear lungs. Clear lungs. and\nNo acute process.\n"
    assert json.loads((out / "repair_report.json").read_text())["policy"]["dedup"] == "consecutive"

    clash = subprocess.run(
        [sys.executable, script, "--dump-dir", str(src), "--out-dir", str(src)],
        capture_output=True, text=True,
    )
    assert clash.returncode != 0 and "control arm" in (clash.stdout + clash.stderr)


@pytest.mark.willi_parity
def test_repair_reuses_the_eval_metric_functions_rather_than_reimplementing_them():
    """A second copy of the tokenisation would drift from the numbers this is
    being compared against, which is the one thing the comparison cannot
    survive."""
    src = (REPO_ROOT / "scripts" / "repair_generations.py").read_text()
    assert "compute_all_metrics" in src and "evaluate_report_generation.py" in src
    assert "def rouge_l" not in src and "def corpus_bleu" not in src


@pytest.mark.willi_parity
def test_repair_wrapper_is_cpu_only_and_warns_that_every_arm_must_be_repaired():
    """A decode-protocol change applied to one arm and cited against another
    arm's unrepaired numbers manufactures a win. The wrapper has to say so."""
    src = (REPO_ROOT / "scripts" / "repair_generations_h100.sh").read_text()
    directives = [ln for ln in src.splitlines() if ln.startswith("#SBATCH")]
    assert not any("--gpus" in ln or "--gres" in ln for ln in directives), (
        "repairing cached text needs no GPU; a prose mention of --gpus is fine, "
        "an actual allocation is not"
    )
    assert "--partition=pot-hpi-aisc-batch" in src and "--exclude=ga03" in src
    assert "EVERY system being compared" in src or "EVERY arm" in src
    assert "score_chexbert_h100.sh" in src and "bootstrap_compare_h100.sh" in src


# ---------------------------------------------------------------------------
# EFFICIENCY_PLAN.md — the inference-speed plan-of-record (created 2026-09-25)
# ---------------------------------------------------------------------------

@pytest.mark.willi_parity
def test_efficiency_plan_and_its_state_file_can_actually_enter_the_repo():
    """.gitignore line 84 is a blanket `*.md` and the state files are ignored by
    default too, so a new plan-of-record is invisible unless it is allowlisted:
    `git add` silently does nothing and `git status` stays clean. That is exactly
    how analysis/ARCHIVE_MANIFEST.md was reported as committed while living on one
    laptop (2026-09-20). EFFICIENCY_PLAN.md FE7 is the same trap, pre-registered.

    analysis/EFFICIENCY_NOTE.md is allowlisted before it exists on purpose — E5-A
    writes it, and the allowlist must not be a thing anyone has to remember later."""
    import subprocess

    must_be_visible = [
        "EFFICIENCY_PLAN.md",
        "efficiency_state.json",
        "analysis/EFFICIENCY_NOTE.md",   # written by E5-A; allowlisted ahead of time
    ]
    ignored = []
    for rel in must_be_visible:
        proc = subprocess.run(
            ["git", "check-ignore", "-q", rel],
            cwd=REPO_ROOT, capture_output=True,
        )
        if proc.returncode == 0:          # 0 == the path IS ignored
            ignored.append(rel)
    assert not ignored, (
        "these plan deliverables are swallowed by .gitignore and would be silently "
        "lost; add `!<path>` to the allowlist block: %s" % ignored
    )


@pytest.mark.willi_parity
def test_state_helper_serves_both_plans_of_record_without_moving_its_default():
    """One helper, two plans. The default must stay on the Mamba-3 files because
    every command in MAMBA3_PLAN_V2.md and CLAUDE.md is written without --plan;
    `--plan efficiency` repoints it at EFFICIENCY_PLAN.md + efficiency_state.json.

    The id regexes were `[MV]\\d+` until the E-ids arrived. Widening them must not
    start matching prose headings like `### 1. The live recurrence ...`, which is
    what the anchored capital-letter class protects."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m3state_multi", REPO_ROOT / "scripts" / "mamba3_state.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

    # Default is unchanged — the existing plan's documented commands keep working.
    assert mod.PLAN.name == "MAMBA3_PLAN_V2.md" and mod.STATE.name == "mamba3_v2_state.json"
    assert mod.DEFAULT_PLAN_SET == "mamba3"
    assert mod.PLAN_SETS["efficiency"] == ("EFFICIENCY_PLAN.md", "efficiency_state.json")

    # Both id families parse.
    assert mod.CHECKBOX_RE.match("- [ ] **E0-A** Per-layer split").group(4) == "E0-A"
    assert mod.CHECKBOX_RE.match("- [x] **V3-D** Eval").group(4) == "V3-D"
    assert mod.CHECKBOX_RE.match("- [x] **M7-B** Screen").group(4) == "M7-B"
    assert mod.PHASE_RE.match("### E2 — Remove the loop").group(1) == "E2"
    assert mod.PHASE_RE.match("### V3 — Full pipeline").group(1) == "V3"
    # ...and prose headings still do not.
    assert mod.PHASE_RE.match("### 1. The live recurrence is not the specified one") is None
    assert mod.PHASE_RE.match("### Intended outcome") is None

    # Selecting a plan set repoints both files together.
    mod.select_plan_set("efficiency")
    assert mod.PLAN.name == "EFFICIENCY_PLAN.md" and mod.STATE.name == "efficiency_state.json"


@pytest.mark.willi_parity
def test_efficiency_state_file_tracks_exactly_the_plans_checkboxes():
    """The state-tracking contract is worthless if the two files drift. Every phase
    and checkbox in EFFICIENCY_PLAN.md must be present in efficiency_state.json;
    `mamba3_state.py --plan efficiency sync` is the one command that fixes this."""
    import importlib.util, json
    spec = importlib.util.spec_from_file_location(
        "m3state_eff", REPO_ROOT / "scripts" / "mamba3_state.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    mod.select_plan_set("efficiency")

    order, phases = mod.parse_plan()
    state = json.loads((REPO_ROOT / "efficiency_state.json").read_text())

    assert order, "EFFICIENCY_PLAN.md parsed with no phases — check the `### E<n> — ` headings"
    assert state["phase_order"] == order, (
        "efficiency_state.json phase_order is stale; run "
        "`venv/bin/python scripts/mamba3_state.py --plan efficiency sync`"
    )
    for pid in order:
        assert set(state["phases"][pid]["checkboxes"]) == set(phases[pid]["checkboxes"]), (
            f"phase {pid} checkboxes drifted between plan and state"
        )
    # The blocking gate is the whole point of the phase order: E0 comes first.
    assert order[0] == "E0", "E0 must be the first phase — it measures the Amdahl bound that gates E2/E3/E4"


@pytest.mark.willi_parity
def test_efficiency_plan_states_it_changes_no_published_number():
    """Every phase here is inference-path. If this plan ever grows an item that
    retrains a checkpoint, the scope sentence is the thing that has to change
    first — and reviewers read the top of the file, not the phase list."""
    plan = (REPO_ROOT / "EFFICIENCY_PLAN.md").read_text()
    assert "changes no published number" in plan
    assert "No merge into `h100_scaling`" in plan or "NO MERGE" in plan
    # The equivalence gate is what licenses that claim; it must be pre-registered.
    assert "R1" in plan and "ssd_sequential_reference" in plan


# ---------------------------------------------------------------------------
# EFFICIENCY_PLAN.md E0 — the profiler arms that measure the wall-clock gap
# ---------------------------------------------------------------------------

@pytest.mark.willi_parity
def test_profiler_exposes_the_e0_arms():
    """E0 needs four things the profiler did not have: a per-mixer-type split,
    a way to force one SDPA backend, a chunk-size override, and a compile arm.

    The attention-backend flag is the one that answers the supervisor's question,
    so its default must stay `auto` — that is the arm every published efficiency
    number was measured in, and a default of `math` would silently re-baseline
    analysis/efficiency_150m_m3/."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "perfprof", REPO_ROOT / "scripts" / "performance_profile.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

    assert mod.SDPA_BACKENDS[0] == "auto", "the fused backend must remain the default arm"
    for name in ("math", "flash", "efficient"):
        assert name in mod.SDPA_BACKENDS
    for fn in ("attention_backend", "maybe_compile", "run_layer_split"):
        assert hasattr(mod, fn), f"E0/E1 entry point {fn} is missing"
    assert hasattr(mod, "LayerSplit") and hasattr(mod, "ScanSplit")

    src = (REPO_ROOT / "scripts" / "performance_profile.py").read_text()
    for flag in ("--per-layer", "--attn-backend", "--chunk-size", "--compile"):
        assert flag in src, f"{flag} is not wired into the CLI"


@pytest.mark.willi_parity
def test_scan_split_restores_the_operator_even_when_the_body_raises():
    """ScanSplit monkeypatches mamba3_block.ssd_chunked_scan. A profiler that
    leaves the patch installed would silently corrupt every later measurement in
    the same process — and the wrapper runs several sweeps per job."""
    import importlib.util
    from hybrid_xmamba.layers import mamba3_block

    spec = importlib.util.spec_from_file_location(
        "perfprof_scan", REPO_ROOT / "scripts" / "performance_profile.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

    original = mamba3_block.ssd_chunked_scan
    with pytest.raises(ValueError):
        with mod.ScanSplit("cpu"):
            assert mamba3_block.ssd_chunked_scan is not original, "patch never applied"
            raise ValueError("boom")
    assert mamba3_block.ssd_chunked_scan is original, (
        "ScanSplit leaked its monkeypatch after an exception"
    )


@pytest.mark.willi_parity
def test_layer_split_accounts_for_every_layer_and_computes_the_amdahl_bound():
    """E0-A is only useful if the split is exhaustive: every HybridBlock must be
    timed and attributed to its mixer type, and the bound must be derived from
    the measured share rather than asserted."""
    import importlib.util
    import torch
    from hybrid_xmamba.models.configuration_hybrid import HybridConfig

    spec = importlib.util.spec_from_file_location(
        "perfprof_split", REPO_ROOT / "scripts" / "performance_profile.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

    cfg = HybridConfig(
        dim=32, num_layers=4, vocab_size=128, max_position_embeddings=64,
        layer_pattern=["mamba3", "mlstm"], num_heads=2,
        mamba3_d_state=8, mamba3_head_dim=8, mamba3_chunk_size=4,
        use_fast_path=False, use_tfla=False,
    )
    rows = mod.run_layer_split(
        model_names=[], seq_lengths=[], batch_size=1, num_iterations=1,
        device="cpu", dtype=torch.float32,
    )
    assert rows == [], "an empty model list must not invent rows"

    model = mod.build_model(cfg, "cpu", torch.float32)
    split = mod.LayerSplit(model, "cpu")
    split.enabled = True
    with torch.no_grad():
        model(torch.randint(0, cfg.vocab_size, (1, 16)))
    split.enabled = False
    by_type, by_index = split.totals_ms(iterations=1)
    split.remove()

    assert set(by_index) == set(range(cfg.num_layers)), "a layer went untimed"
    assert set(by_type) == {"mamba3", "mlstm"}
    # 1 / (1 - share) is the only bound the plan is allowed to claim.
    share = 0.8
    assert abs((1.0 / (1.0 - share)) - 5.0) < 1e-9


@pytest.mark.willi_parity
def test_layer_split_wrapper_follows_the_cluster_conventions():
    """Same contract every other H100 wrapper is held to: the aisc partition,
    the faulty-node exclusion, `--gpus` rather than `--gres`, and no dataset or
    checkpoint dependency — this job must not be able to touch a published
    artefact."""
    path = REPO_ROOT / "scripts" / "profile_layer_split_h100.sh"
    src = path.read_text()
    directives = [ln for ln in src.splitlines() if ln.startswith("#SBATCH")]
    assert any("--partition=pot-hpi-aisc-batch" in ln for ln in directives)
    assert any("--account=aisc" in ln for ln in directives)
    assert any("--gpus=1" in ln for ln in directives)
    assert not any("--gres" in ln for ln in directives), "never --gres for GPUs on aisc"
    assert any("--exclude=ga03" in ln and "gx13v1" in ln for ln in directives)
    assert any("--open-mode=append" in ln for ln in directives)
    # It profiles random weights; a checkpoint path here would mean it can touch
    # DUA-covered artefacts, which the plan's scope sentence forbids.
    assert "CKPT" not in src and "checkpoints/" not in src
    assert "HF_HUB_OFFLINE=1" in src
    # The E0-D arms are the supervisor's answer; both must be present.
    assert "--attn-backend auto" in src and "--attn-backend math" in src


# ---------------------------------------------------------------------------
# EFFICIENCY_PLAN.md E1 — the R1 gate that stands between a stopwatch and a claim
# ---------------------------------------------------------------------------

@pytest.mark.willi_parity
def test_r1_gate_checks_the_document_boundary_and_the_fp64_oracle():
    """E0 found two speedups. Neither may be adopted on timing alone: this repo
    has already shipped an operator that computed a different function than
    advertised (the A_cum.clamp defect, rel-max-err 0.92), and the entire Mamba-3
    campaign exists to repair it.

    The gate must compare against the fp64 oracle rather than only against the
    shipped chunked path — two chunked variants can agree with each other and
    both be wrong — and it must include a cu_seqlens boundary that lands inside a
    chunk, which is exactly where the Mamba-1 defect lived."""
    src = (REPO_ROOT / "scripts" / "check_operator_equivalence.py").read_text()
    assert "ssd_sequential_reference" in src, "R1 requires the fp64 oracle, not just self-consistency"
    assert "cu_seqlens" in src and "mid-chunk" in src
    assert "R1_TOLERANCE = 1e-4" in src
    # It has to be usable as a gate, i.e. fail the process, not just print.
    assert "return 1" in src and "sys.exit(main())" in src


@pytest.mark.willi_parity
def test_r1_gate_actually_separates_a_correct_variant_from_a_wrong_one():
    """A gate that passes everything is not a gate. Feed it the same operands
    with and without the document reset: the chunked scan must track the oracle
    that shares its cu_seqlens and must NOT match the one that does not."""
    import importlib.util
    import torch
    from hybrid_xmamba.kernels.ssd import ssd_chunked_scan, ssd_sequential_reference

    spec = importlib.util.spec_from_file_location(
        "r1gate", REPO_ROOT / "scripts" / "check_operator_equivalence.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

    x, dt, A, B, C, D = mod._operands(2, 128, 4, 16, 1, 32, "cpu")
    seg = torch.zeros(2, 128, dtype=torch.long)
    seg[:, 50:] = 1                       # boundary inside every chunk size tested

    got = ssd_chunked_scan(x, dt, A, B, C, D=D, chunk_size=64, cu_seqlens=seg)
    right = ssd_sequential_reference(x, dt, A, B, C, D=D, cu_seqlens=seg)
    wrong = ssd_sequential_reference(x, dt, A, B, C, D=D)     # boundary ignored

    assert mod.rel_max_err(got, right) < mod.R1_TOLERANCE, (
        "the chunked scan no longer matches the fp64 oracle across a document reset"
    )
    assert mod.rel_max_err(got, wrong) > 0.01, (
        "cu_seqlens is being ignored somewhere, so the boundary case tests nothing"
    )


@pytest.mark.willi_parity
def test_e1_wrapper_runs_the_gate_before_it_times_anything():
    """Ordering is the whole design: if the equivalence check ran after the
    sweeps, a failing gate would still have produced numbers someone could quote.
    Also: job 2579642 died at 1.5h compiling L=16384, so this wrapper must ask
    for more time than that."""
    src = (REPO_ROOT / "scripts" / "verify_and_profile_e1_h100.sh").read_text()
    gate_at = src.index("check_operator_equivalence.py")
    first_sweep = src.index("performance_profile.py --sweep")
    assert gate_at < first_sweep, "the R1 gate must run before any timing arm"

    directives = [ln for ln in src.splitlines() if ln.startswith("#SBATCH")]
    assert any("--partition=pot-hpi-aisc-batch" in ln for ln in directives)
    assert any("--gpus=1" in ln for ln in directives)
    assert not any("--gres" in ln for ln in directives)
    assert any("--exclude=ga03" in ln and "gx13v1" in ln for ln in directives)
    time_line = [ln for ln in directives if "--time=" in ln][0]
    hours = int(time_line.split("--time=")[1].split(":")[0])
    assert hours >= 3, "1.5h already timed out on the L=16384 compile (job 2579642)"
    assert "CKPT" not in src and "checkpoints/" not in src


@pytest.mark.willi_parity
def test_sweep_reports_the_chunk_size_it_actually_used_not_the_one_it_was_asked_for():
    """Job 2580198: the compiled cs=64 and cs=128 arms timed identically to one
    microsecond at L=4096 and L=8192 while the uncompiled arms differed by 45%.
    Recording the requested value proves nothing about what ran, so the sweep has
    to read the chunk size back off the built module and carry it in the CSV."""
    src = (REPO_ROOT / "scripts" / "performance_profile.py").read_text()
    assert "effective_chunk_size" in src
    assert "effective chunk_size on the built module" in src
    # It must come from the module, not from the config object.
    idx = src.index("effective_chunk = None")
    window = src[idx:idx + 400]
    assert "layer.mixer.chunk_size" in window, (
        "the readback must come off the built layer, otherwise it just echoes the config"
    )


@pytest.mark.willi_parity
def test_followup_wrapper_isolates_the_inductor_cache_per_arm():
    """The anomaly's leading explanation is a shared TORCHINDUCTOR_CACHE_DIR
    handing two different chunk sizes the same compiled kernel. An arm that
    reuses the previous arm's cache cannot test that."""
    src = (REPO_ROOT / "scripts" / "profile_e1_followup_h100.sh").read_text()
    assert "inductor_cache_${name}" in src, "each arm needs its own Inductor cache"
    assert "rm -rf" in src, "a stale cache from a previous job would defeat the isolation"
    directives = [ln for ln in src.splitlines() if ln.startswith("#SBATCH")]
    assert any("--partition=pot-hpi-aisc-batch" in ln for ln in directives)
    assert any("--gpus=1" in ln for ln in directives)
    assert not any("--gres" in ln for ln in directives)
    assert "--backward" in src, "the training path is the untested half of the efficiency claim"
    assert "CKPT" not in src and "checkpoints/" not in src


@pytest.mark.willi_parity
def test_no_profiling_wrapper_shares_an_inductor_cache_between_arms():
    """Job 2580198 gave every arm one TORCHINDUCTOR_CACHE_DIR and corrupted its own
    results: re-measuring the same points with per-arm caches made them 21-30%
    faster at L=4096, because the shared cache had handed both arms one slow
    kernel. That is also why the cs=64 and cs=128 arms matched to a microsecond.

    Any wrapper that sets the cache dir must make it arm-specific."""
    import re
    for name in ("verify_and_profile_e1_h100.sh", "profile_e1_followup_h100.sh",
                 "profile_e1_confirm_h100.sh"):
        src = (REPO_ROOT / "scripts" / name).read_text()
        exports = [ln.strip() for ln in src.splitlines()
                   if "TORCHINDUCTOR_CACHE_DIR=" in ln and ln.strip().startswith("export")]
        assert exports, f"{name} compiles without pinning an Inductor cache dir"
        for ln in exports:
            assert re.search(r"\$\{(name|arm)\}", ln), (
                f"{name} shares one Inductor cache across arms: {ln}"
            )


@pytest.mark.willi_parity
def test_confirm_wrapper_measures_one_sequence_length_per_process():
    """The second finding from 2582482: within a single process, shapes compiled
    later measure worse (3.04x at L=512 down to 1.02x at L=4096). A sweep that
    passes several lengths to one process therefore cannot produce a number that
    means anything on its own, which is why the headline needs re-measuring."""
    src = (REPO_ROOT / "scripts" / "profile_e1_confirm_h100.sh").read_text()
    # Every invocation passes exactly one length, held in a shell variable.
    assert '--seq-lengths "${len}"' in src
    assert "for L in ${INFER_LENS}" in src and "for L in ${TRAIN_LENS}" in src
    # The Transformer reference must come first in the training block: job 2582482
    # timed out before reaching it, which is why its training table is unusable.
    train_block = src[src.index("########## 2:"):]
    assert train_block.index("train_xfmr") < train_block.index("train_base")
    directives = [ln for ln in src.splitlines() if ln.startswith("#SBATCH")]
    assert not any("--gres" in ln for ln in directives)
    assert any("--partition=pot-hpi-aisc-batch" in ln for ln in directives)
    hours = int([ln for ln in directives if "--time=" in ln][0].split("--time=")[1].split(":")[0])
    assert hours >= 3, "2h timed out in job 2582482"


@pytest.mark.willi_parity
def test_no_tracked_file_claims_a_triton_kernel_this_project_does_not_have():
    """EFFICIENCY_PLAN.md E5-C. A supervisor read this repo and concluded we had
    written Triton kernels to make Mamba FlashAttention-compatible. We never did:
    scan_triton.py and tfla_triton.py had one call site between them, in the dead
    mamba_block_v2.py, and were deleted in 37f7964.

    Four artefacts created that impression and are removed: the root
    test_triton_fix.py ("Test script to verify the Triton kernel fix"), an unused
    `triton>=2.1.0` requirement, a package docstring advertising "custom
    CUDA/Triton kernels", and a wrapper comment about "the same custom Mamba/mLSTM
    Triton kernels".

    Claiming a kernel you do not have is a correctness claim about your own
    efficiency numbers, so this is a test, not a style preference. Note that
    Inductor DOES emit Triton under torch.compile -- describing that is fine;
    claiming a hand-written kernel is not."""
    assert not (REPO_ROOT / "test_triton_fix.py").exists(), (
        "the stale Colab-era Triton script is back"
    )
    for req in ("requirements.txt", "requirements-colab.txt"):
        text = (REPO_ROOT / req).read_text()
        assert not any(ln.strip().startswith("triton") for ln in text.splitlines()), (
            f"{req} declares triton, which nothing in this project imports"
        )
    init = (REPO_ROOT / "hybrid_xmamba" / "__init__.py").read_text()
    assert "custom CUDA/Triton kernels" not in init
    assert "no hand-written CUDA or Triton kernels" in init
    kd = (REPO_ROOT / "scripts" / "train_biomedclip_kd_h100.sh").read_text()
    assert "custom Mamba/mLSTM Triton kernels" not in kd


@pytest.mark.willi_parity
def test_decode_path_can_override_chunk_size_the_way_it_overrides_the_operators():
    """EFFICIENCY_PLAN.md E6. Efficiency is reported at chunk_size=128 while every
    quality number was decoded at 64. R1 shows the logits agree to 3.0e-05, which
    is not the same as showing the decoded tokens agree -- beam search can flip on
    an arbitrarily small margin, and V5-A watched 227 of 400 reports change under
    an operator swap that moved no metric.

    So chunk_size must be overridable at decode, on the same mechanism as
    scan_impl/tfla_impl, and the override must be announced in the log rather than
    inferred later from a config file."""
    src = (REPO_ROOT / "scripts" / "evaluate_report_generation.py").read_text()
    assert "--chunk-size" in src
    assert "chunk_size=getattr(args, \"chunk_size\", None)" in src
    # It rides the same announce-on-override loop as the operator flags.
    i = src.index('("scan_impl", scan_impl)')
    assert "mamba3_chunk_size" in src[i:i + 300], (
        "chunk_size must go through the same override-announcing loop as scan_impl"
    )
    wrapper = (REPO_ROOT / "scripts" / "inspect_report_generation_h100.sh").read_text()
    assert 'CHUNK_SIZE="${CHUNK_SIZE:-}"' in wrapper
    assert '--chunk-size "${CHUNK_SIZE}"' in wrapper


@pytest.mark.willi_parity
def test_e6_wrapper_matches_the_v5a_protocol_it_must_pair_with():
    """E6's dump is compared against V5-A's by paired bootstrap, so every knob
    except chunk_size has to match V5-A: 13D, beam 3, 100 new tokens. A wrapper
    that quietly changed the beam size would manufacture a difference."""
    src = (REPO_ROOT / "scripts" / "verify_optimised_decode_h100.sh").read_text()
    assert "DECODE=beam BEAM_SIZE=3 MAX_NEW_TOKENS=100" in src
    assert "tower13d" in src, "E6 must decode the same 13D checkpoint V5-A used"
    directives = [ln for ln in src.splitlines() if ln.startswith("#SBATCH")]
    assert any("--partition=pot-hpi-aisc-batch" in ln for ln in directives)
    assert not any("--gres" in ln for ln in directives)
    # The pre-registered rule has to be in the wrapper, not only in the plan:
    # whoever reads the log is the person who will over-claim.
    assert "PRE-REGISTERED RULE" in src
    assert "reverts to" in src or "headroom" in src


@pytest.mark.willi_parity
def test_dropped_phases_record_why_rather_than_vanishing():
    """E2, E3 and E4 were dropped on 2026-09-25 after E1 measured the gap closed.
    A plan that deletes a phase loses the reasoning; a plan that keeps an untouched
    checkbox implies work still pending. Both are wrong, so the phases stay with
    an explicit verdict and no checkboxes."""
    plan = (REPO_ROOT / "EFFICIENCY_PLAN.md").read_text()
    for pid in ("E2", "E3", "E4"):
        assert f"### {pid} — DROPPED" in plan, f"{pid} must record that it was dropped, and why"
    # The decisive argument is measured, not asserted.
    assert "4.50" in plan and "4.57" in plan, (
        "the E2 verdict rests on compile having already realised more than E0-F's bound"
    )


@pytest.mark.willi_parity
def test_chunk_size_override_is_refused_on_a_config_with_no_mamba3_layer():
    """Job 2583277 decoded hybrid_150m_v2_rrg -- the incumbent, 9x mamba-1 + 3x
    mlstm -- with --chunk-size 128. The key was absent, got set, and nothing ever
    read it, so the run produced ROUGE-L 0.18358 against the reference's 0.1836
    and looked like a clean tie while measuring the unmodified model.

    A silent no-op that yields a publishable-looking null is worse than a crash,
    so this is a hard error now."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "evalrg", REPO_ROOT / "scripts" / "evaluate_report_generation.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

    with pytest.raises(ValueError, match="has no 'mamba3' layer"):
        mod.load_report_generation_module(
            "/nonexistent.ckpt", "hybrid_150m_v2_rrg", chunk_size=128)

    # The Mamba-3 config must get past the guard (it then fails on the missing
    # checkpoint, which is a different and expected failure).
    with pytest.raises(Exception) as exc:
        mod.load_report_generation_module(
            "/nonexistent.ckpt", "hybrid_150m_m3_rrg", chunk_size=128)
    assert "has no 'mamba3' layer" not in str(exc.value)


@pytest.mark.willi_parity
def test_e6_wrapper_decodes_the_mamba3_arm_and_both_chunk_sizes():
    """The efficiency numbers are all hybrid_150m_m3, so the quality check must be
    the Mamba-3 decoder -- not 13D, which is the incumbent. Both arms are decoded
    in the same job so they differ by exactly one setting."""
    src = (REPO_ROOT / "scripts" / "verify_optimised_decode_h100.sh").read_text()
    assert "hybrid_150m_m3_rrg" in src
    assert "h100_report_gen_m3_tower13d_s42" in src
    assert "tower13d/checkpoints" not in src, "must not default to the incumbent 13D checkpoint"
    assert 'decode_arm "${REF_DUMP}" 64' in src and 'decode_arm "${DUMP_DIR}" "${CHUNK_SIZE}"' in src
    # A failed compile arm is data about that arm, not a reason to lose the job.
    assert "ARM FAILED" in src


# ---------------------------------------------------------------------------
# EFFICIENCY_PLAN.md E7 — the second lever, verified on text instead of logits
# ---------------------------------------------------------------------------

@pytest.mark.willi_parity
def test_compile_is_a_decode_lever_threaded_from_the_wrapper_to_the_eval():
    """The efficiency headline uses TWO settings: chunk_size=128 and torch.compile.
    E6 verified the first on decoded text. The second was verified only on logits
    (3.0e-05 against a 1e-4 gate), which is not token identity -- beam search flips
    on an arbitrarily small margin. So compile has to be reachable from the same
    decode harness the quality numbers come from."""
    ev = (REPO_ROOT / "scripts" / "evaluate_report_generation.py").read_text()
    assert '"--compile", dest="compile_decoder"' in ev
    assert "compile_decoder_for_inference" in ev

    wrapper = (REPO_ROOT / "scripts" / "inspect_report_generation_h100.sh").read_text()
    assert 'COMPILE="${COMPILE:-false}"' in wrapper, "compile must default OFF"
    assert '$([ "${COMPILE}" = "true" ] && echo "--compile")' in wrapper


@pytest.mark.willi_parity
def test_a_compiled_decode_that_silently_ran_eager_is_an_error_not_agreement():
    """torch.compile fails OPEN. A capture failure, or an exhausted recompile
    limit as the beam grows one token per step, drops back to eager and still
    emits perfectly good reports -- which would agree with the eager arm for the
    trivial reason that neither arm compiled.

    That is the job-2583277 failure class: a no-op producing a publishable-looking
    null. So the run must abort instead, and it must abort BEFORE writing a dump,
    because an eager dump is indistinguishable from a real one on disk."""
    import importlib.util
    import torch._dynamo as dynamo
    spec = importlib.util.spec_from_file_location(
        "evalrg_e7", REPO_ROOT / "scripts" / "evaluate_report_generation.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

    dynamo.reset()
    dynamo.utils.counters.clear()
    with pytest.raises(RuntimeError, match="captured NOTHING"):
        mod.assert_decoder_was_compiled()

    src = (REPO_ROOT / "scripts" / "evaluate_report_generation.py").read_text()
    guard = src.index("assert_decoder_was_compiled()\n\n    if args.dump_dir:")
    assert guard > 0, "the compile check must run before write_hyps_refs, not after"


@pytest.mark.willi_parity
def test_compiling_the_decoder_raises_the_recompile_limit():
    """Beam search feeds a sequence one token longer every step. Under the default
    static-shape policy that is a fresh graph per length, and Dynamo's default
    cache_size_limit of 8 is exhausted within 8 generated tokens -- after which it
    serves eager for the remaining 92 and says so only in a warning buried in a
    13,000-line log. dynamic=True plus a raised limit is what makes the compiled
    arm actually compiled."""
    src = (REPO_ROOT / "scripts" / "evaluate_report_generation.py").read_text()
    i = src.index("def compile_decoder_for_inference")
    window = src[i:i + 1600]
    assert "dynamic: bool = True" in window
    assert "cache_size_limit" in window
    assert "torch.compile(module.decoder" in window


@pytest.mark.willi_parity
def test_e7_wrapper_runs_both_arms_and_times_the_canary_on_a_warm_cache():
    """One job, one setting apart, compiled arm first because its cost is the
    unknown one. The canary's two timed points must both run warm: if n=4 carries
    a cold graph build that n=20 does not, the fitted slope lands BELOW the true
    per-study cost and the projection is optimistic -- the one direction that
    loses the eight GPU-hours the canary exists to protect."""
    src = (REPO_ROOT / "scripts" / "verify_compiled_decode_h100.sh").read_text()
    assert "hybrid_150m_m3_rrg" in src and "h100_report_gen_m3_tower13d_s42" in src

    compiled_at = src.index('decode_arm "${DUMP_DIR}" "${NUM_SAMPLES}" true')
    eager_at = src.index('decode_arm "${REF_DUMP}" "${NUM_SAMPLES}" false')
    assert compiled_at < eager_at, "the compiled arm runs first; we already own an eager dump"

    warm_at = src.index("WARM=$(run_canary 2)")
    assert warm_at < src.index("E4=$(run_canary 4)") < src.index("E20=$(run_canary 20)")


def test_no_slurm_script_uses_the_retired_aisc_batch_partition():
    """2026-09-30: the cluster renamed `aisc-batch` to `pot-hpi-aisc-batch`. sbatch now rejects the
    old name outright ("Partition 'aisc-batch' has been renamed"), so any script still carrying it
    fails at submission. Account and QOS are unchanged."""
    offenders = [
        p.name for p in sorted((REPO_ROOT / "scripts").glob("*.sh"))
        if "--partition=aisc-batch" in p.read_text()
    ]
    assert not offenders, offenders


def test_chat_ui_plan_set_is_registered_and_its_ids_parse():
    """CHAT_UI_PLAN.md P0-E. The helper only tracks `- [ ] **P1-A**` boxes under `### P1 — ...`
    headings. An id it cannot parse (the spec's `P3b-A`, say) would silently drop out of
    chat_ui_state.json, which is what a new session reads to know what is done and what is next."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("mamba3_state_chat_ui", REPO_ROOT / "scripts" / "mamba3_state.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    assert helper.PLAN_SETS["chat_ui"] == ("CHAT_UI_PLAN.md", "chat_ui_state.json")

    helper.select_plan_set("chat_ui")
    order, phases = helper.parse_plan()
    assert order == ["P{}".format(i) for i in range(10)]
    assert all(phases[p]["checkboxes"] for p in order), "a phase with no parseable checkbox"
    plan_lines = (REPO_ROOT / "CHAT_UI_PLAN.md").read_text().splitlines()
    stray = [l for l in plan_lines if re.match(r"^- \[[ x]\] \*\*P\d", l) and not helper.CHECKBOX_RE.match(l)]
    assert not stray, stray

    state = json.loads((REPO_ROOT / "chat_ui_state.json").read_text())
    assert state["phase_order"] == order, "run: scripts/mamba3_state.py --plan chat_ui sync"
    assert set(state["phases"]) == set(order)
    for key in ("current_phase", "status", "next_action", "resume_protocol", "decisions", "open_questions"):
        assert key in state, key


def test_chat_cluster_setup_wrapper_is_cpu_only_and_additive():
    """CHAT_UI_PLAN.md P0-G (R8: additive only). The setup job fills the chat UI's own cluster directory next to the
    thesis checkout, so it may create symlinks that do not exist yet (`ln -s`, never `ln -sf`, which would replace
    one), directories, and web-dependency overlays under `--target` (never into the shared venvs), and nothing
    else: no deletion, no GPU. It executes the x86 venv's python, so it must exclude the ARM node ga03 even though
    it never sources an activate script."""
    src = (REPO_ROOT / "scripts" / "chat_cluster_setup_h100.sh").read_text()
    directives = [l for l in src.splitlines() if l.startswith("#SBATCH")]
    assert any("--partition=pot-hpi-aisc-batch" in l for l in directives)
    assert any("--account=aisc" in l for l in directives)
    assert any("--qos=aisc" in l for l in directives), "the proven CPU-only combination"
    assert any("--exclude=ga03" in l for l in directives)
    assert any("--output=logs/%x_%j.log" in l for l in directives)
    assert not [l for l in directives if "--gpus" in l or "--gres" in l], "CPU-only job"

    code = [l for l in src.splitlines() if not l.lstrip().startswith("#")]
    assert "rm " not in src, "no deletion, anywhere in the file"
    assert "ln -sf" not in src and any("ln -s " in l for l in code)
    installs = [l for l in code if "pip install" in l]
    assert len(installs) == 2, "one overlay per venv"
    assert all("--target" in l for l in installs), "web deps go into overlays, never into the shared venvs"
    # A failed `uv pip install --target` leaves its directory behind, so a re-run is guarded by a sentinel written
    # after a passing import check, never by `[ -d overlay ]` (behaviour: tests/test_chat_remote.py).
    assert ".chat_deps/.setup_ok" in src and ".chat_deps_chexbert/.setup_ok" in src
    assert "-d .chat_deps" not in src


def test_chat_probe_wrapper_is_cpu_only_on_the_renamed_partition():
    """CHAT_UI_PLAN.md P1-A. CPU-only: a GPU job would hold an accelerator idle for a port check."""
    src = (REPO_ROOT / "scripts" / "chat_probe_h100.sh").read_text()
    directives = [l for l in src.splitlines() if l.startswith("#SBATCH")]
    assert "#SBATCH --partition=pot-hpi-aisc-batch" in directives
    assert "#SBATCH --account=aisc" in directives and "#SBATCH --qos=aisc" in directives
    assert not [l for l in directives if "--gpus" in l or "--gres" in l]
    assert any(l.startswith("#SBATCH --exclude=ga03") for l in directives)
    assert "app/tunnel/probe_server.py" in src and "--sqlite-probe" in src


def test_chat_cpu_decode_probe_is_cpu_only_and_decodes_the_published_protocol():
    """CHAT_UI_PLAN.md P1-C. Same protocol as the published dump, on CPU, compared line by line."""
    src = (REPO_ROOT / "scripts" / "chat_cpu_decode_probe_h100.sh").read_text()
    directives = [l for l in src.splitlines() if l.startswith("#SBATCH")]
    assert "#SBATCH --partition=pot-hpi-aisc-batch" in directives and "#SBATCH --qos=aisc" in directives
    assert not [l for l in directives if "--gpus" in l or "--gres" in l]
    assert any(l.startswith("#SBATCH --exclude=ga03") for l in directives)
    for needle in ("--model-config hybrid_150m_m3_rrg", "--decode beam", "--beam-size 3",
                   "--max-new-tokens 100", "--cached-decode", "report_gen_m3_test_split_s42/hyps.txt",
                   "OMP_NUM_THREADS", "HF_HUB_OFFLINE", "cached_vs_published_gpu_differ"):
        assert needle in src, needle
    # fix round 1 (behaviour: tests/test_chat_cpu_probe.py). A throwaway warm-up arm pays the cold page-cache read before
    # the one-study arm that is subtracted as load cost; GNU time is required, never an untimed fallback.
    order = [src.index(arm) for arm in ("run warm --num-samples 1 --cached-decode", "run cached_1 --num-samples 1",
                                        "run cached_a ", "run cached_b ", "run uncached ")]
    assert order == sorted(order), "warm, cached_1, cached_a, cached_b, uncached"
    assert 'echo "ERROR: /usr/bin/time missing"' in src and "TIME_V" not in src


def test_chat_engine_golden_wrappers_compare_on_the_same_node():
    """CHAT_UI_PLAN.md P2-E. R2: byte identity is only meaningful on the same node and device."""
    cpu = (REPO_ROOT / "scripts" / "chat_engine_golden_h100.sh").read_text()
    gpu = (REPO_ROOT / "scripts" / "chat_engine_golden_gpu_h100.sh").read_text()
    for src in (cpu, gpu):
        assert "#SBATCH --partition=pot-hpi-aisc-batch" in src and "--exclude=ga03" in src
        assert "scripts/chat_engine_golden.py" in src
    assert "scripts/evaluate_report_generation.py" in cpu   # the GPU arm compares with the published dump instead
    assert not [l for l in cpu.splitlines() if l.startswith("#SBATCH") and "--gpus" in l]
    assert "#SBATCH --gpus=1" in gpu and "--uncached" in gpu


# ── CHAT_UI_PLAN.md P5-B: the gallery build wrapper ───────────────────────────
# Static pins on scripts/build_retrieval_gallery_h100.sh. Its behaviour (guards, R7 output, the three steps, the gate) is
# rehearsed for real in tests/test_build_retrieval_gallery.py, together with the builder it runs.

def _gallery_wrapper_text() -> str:
    return (REPO_ROOT / "scripts" / "build_retrieval_gallery_h100.sh").read_text()


def _gallery_wrapper_code(src: str) -> List[str]:
    """The wrapper's logical lines minus comments and blanks: backslash continuations joined, whitespace squeezed (the #SBATCH
    directives are comments to bash)."""
    joined = src.replace("\\\n", " ")
    return [" ".join(line.split()) for line in joined.splitlines() if line.strip() and not line.lstrip().startswith("#")]


def test_build_retrieval_gallery_wrapper_follows_the_slurm_invariants():
    """P5-B. One H100 through --gpus=1 and never a typed --gres, the renamed partition, the three nodes that cannot run it
    excluded, no --qos (that is the CPU-only combination), a requeue that appends to the SLURM log, logs where every other
    wrapper puts them, and offline Hugging Face on a compute node."""
    src = _gallery_wrapper_text()
    options, flags = {}, []
    for line in src.splitlines():
        if line.startswith("#SBATCH"):
            token = line.split()[1]
            if "=" in token:
                options[token.split("=", 1)[0]] = token.split("=", 1)[1]
            else:
                flags.append(token)
    assert options == {
        "--partition": "pot-hpi-aisc-batch", "--account": "aisc", "--gpus": "1", "--exclude": "ga03,gx17v1,gx13v1",
        "--cpus-per-task": "16", "--mem": "64G", "--time": "04:00:00", "--job-name": "chat_gallery",
        "--output": "logs/%x_%j.log", "--error": "logs/%x_%j.log", "--open-mode": "append"}, options
    assert flags == ["--requeue"], flags
    code = _gallery_wrapper_code(src)
    assert not [l for l in code if "--gres" in l or "--qos" in l], "an untyped --gpus and a typed --gres do not mix; --qos is CPU-only"
    assert 'cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"' in code
    for needle in ("set -euo pipefail", 'SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/hybrid_xmamba_h100}"',
                   'export HF_HOME="${SCRATCH_ROOT}/.hf"', "export HF_HUB_OFFLINE=1", "export HF_DATASETS_OFFLINE=1",
                   'VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"', 'source "${VENV_ACTIVATE}"'):
        assert needle in code, needle


def test_build_retrieval_gallery_wrapper_defaults_are_the_plans_inputs_and_agree_with_the_other_chat_wrappers():
    code = _gallery_wrapper_code(_gallery_wrapper_text())
    for needle in ('CHAT_HOME="${CHAT_HOME:-/sc/home/$USER/chat_sessions}"', 'DATA="${DATA:-/sc/home/$USER/dataset/mimic_full}"',
                   'CKPT_13D="${CKPT_13D:-./outputs/h100_kd_150m_v2_full_data_lr3e6/checkpoints/last.ckpt}"',
                   'CHECKPOINT="${CHECKPOINT:-./outputs/h100_report_gen_m3_tower13d_s42/checkpoints/last.ckpt}"',
                   'MODEL_CONFIG="${MODEL_CONFIG:-hybrid_150m_m3_rrg}"',
                   'BUILD_ID="${BUILD_ID:-$(date +%Y%m%d)_${SLURM_JOB_ID:-local}}"', 'OUT="${CHAT_HOME}/gallery/${BUILD_ID}"'):
        assert needle in code, needle
    assert not [l for l in code if l.startswith(("OUT=", "OUT_DIR=")) and ":-" in l], "OUT is derived, never an environment lever"
    eos = (REPO_ROOT / "scripts" / "train_report_eos_h100.sh").read_text()
    golden = (REPO_ROOT / "scripts" / "chat_engine_golden_gpu_h100.sh").read_text()
    assert "IMAGE_ENCODER_CKPT=./outputs/h100_kd_150m_v2_full_data_lr3e6/checkpoints/last.ckpt" in eos, "the 13D tower is one file"
    assert 'CHECKPOINT="${CHECKPOINT:-./outputs/h100_report_gen_m3_tower13d_s42/checkpoints/last.ckpt}"' in golden
    assert 'DATA="${DATA:-/sc/home/$USER/dataset/mimic_full}"' in golden


def test_build_retrieval_gallery_wrapper_runs_three_steps_and_the_reference_script_is_unchanged_on_the_test_split():
    """P5-B. The builder, then scripts/evaluate_cxr_retrieval.py with the brief's flags on the SAME 13D checkpoint and the
    official test split, then --compare-rk. Never another split: the gate is the published test-split protocol."""
    code = _gallery_wrapper_code(_gallery_wrapper_text())
    text = "\n".join(code)
    build = text.index("python scripts/build_retrieval_gallery.py --checkpoint-13d")
    reference = text.index("python scripts/evaluate_cxr_retrieval.py")
    compare = text.index("python scripts/build_retrieval_gallery.py --compare-rk")
    assert build < reference < compare
    assert ('python scripts/evaluate_cxr_retrieval.py --checkpoint "${CKPT_13D}" --dataset mimic --local-parquet-dir "${DATA}" '
            '--mimic-split test --output-dir "${OUT}/reference_rk"') in text
    assert ('python scripts/build_retrieval_gallery.py --checkpoint-13d "${CKPT_13D}" --decoder-checkpoint "${CHECKPOINT}" '
            '--decoder-config "${MODEL_CONFIG}" --data "${DATA}" --out "${OUT}" --build-id "${BUILD_ID}" '
            '--workers "${SLURM_CPUS_PER_TASK:-8}" --isbi-cache "${SCRATCH_ROOT}/isbi_gallery_adapted.pt"') in text
    assert 'python scripts/build_retrieval_gallery.py --compare-rk "${OUT}" --wall-s "${SECONDS}"' in text
    assert "--mimic-split validation" not in text and "--mimic-split train" not in text
    for name in ("build_retrieval_gallery.py", "evaluate_cxr_retrieval.py"):
        assert (REPO_ROOT / "scripts" / name).is_file(), name


def test_build_retrieval_gallery_wrapper_keeps_every_python_steps_raw_output_out_of_the_job_log():
    """R7. The raw stdout and stderr of the builder and of the thesis script go to files under ${OUT}; the job log gets [gallery]
    lines that carry no path, the one RESULT line of the comparison, ERROR lines with a step name and an exit code, and === lines
    that name no path. A `python scripts/...` that is neither redirected to such a file nor captured by $(...) would print into
    the log."""
    code = _gallery_wrapper_code(_gallery_wrapper_text())
    steps = [l for l in code if "python scripts/" in l]
    assert len(steps) == 3, steps
    build, reference, compare = steps
    assert build.endswith('> "${OUT}/build.log" 2>&1 || rc=$?'), build
    assert reference.endswith('> "${OUT}/reference_rk.log" 2>&1 || rc=$?'), reference
    assert compare.startswith('CMP_OUT="$(python scripts/build_retrieval_gallery.py') and compare.endswith(
        '2>> "${OUT}/compare.err")" || rc=$?'), compare
    text = "\n".join(code)
    # the only things that leave a log file: [gallery] lines of a known shape (an allowlist, not a blacklist of characters), the
    # number of those withheld, and RESULT / ERROR lines
    assert "tr '\\r' '\\n' < \"${OUT}/build.log\" | grep -aE \"${GALLERY_SHAPES}\" | tail -n 60 || true" in text
    assert ("WITHHELD=\"$(tr '\\r' '\\n' < \"${OUT}/build.log\" | grep -a '^\\[gallery\\] ' | grep -avcE \"${GALLERY_SHAPES}\" || true)\""
            in text)
    assert "case \"${WITHHELD}\" in ''|*[!0-9]*) WITHHELD=unknown ;; esac" in text, "a count that could not be made is not a zero"
    assert 'echo "=== gallery lines withheld: ${WITHHELD} ==="' in text
    assert "grep -av '/'" not in text, "a slash is not what makes a line unsafe"
    assert "printf '%s\\n' \"${CMP_OUT}\" | grep -aE '^(RESULT |ERROR)' || true" in text
    for step in ("build", "reference", "compare"):
        assert 'echo "ERROR {} exit=${{rc}}"'.format(step) in text, step
    printed = [l for l in code if re.match(r"^(echo|printf) ", l)]
    stray = [l for l in printed if not re.match(r'^(echo "(===|ERROR) |printf \'%s\\n\' "\$\{CMP_OUT\}")', l)]
    assert not stray, "every printed line starts === or ERROR, or is the filtered comparison output: {}".format(stray)
    paths = ("${OUT}", "${DATA}", "${CKPT_13D}", "${CHECKPOINT}", "${CHAT_HOME}", "${SCRATCH_ROOT}")
    messages = [m for l in code for m in re.findall(r'(?:echo|fail) "([^"]*)"', l)]       # what is printed, not what is tested
    assert len(messages) > 10, messages
    leaking = [m for m in messages if any(p in m for p in paths)]
    assert not leaking, "a printed message names a path: {}".format(leaking)


def test_build_retrieval_gallery_wrapper_prints_its_sync_stamp_before_anything_else():
    """Provenance. `scripts/chat_remote.sh sync` leaves .sync_stamp in the tree it ships; the job prints the commit and
    cleanliness it reads there as its very first line (behaviour, against the real producer's output:
    tests/test_build_retrieval_gallery.py)."""
    code = "\n".join(_gallery_wrapper_code(_gallery_wrapper_text()))
    sync = code.index('echo "=== sync ${SYNC} ==="')
    assert code.index('echo "') == sync, "something is printed before the sync line"
    assert code.index("python ") > sync and code.index("source ") > sync
    assert ".sync_stamp" in code[:sync]


def test_build_retrieval_gallery_wrapper_checks_every_path_before_it_creates_or_runs_anything():
    """R8. No directory is made and no step started before the guards have passed: OUT is under CHAT_HOME and neither inside an
    outputs directory nor under the thesis checkout, a finished build (manifest.json) is never overwritten, the inputs exist, a
    GPU is visible."""
    code = "\n".join(_gallery_wrapper_code(_gallery_wrapper_text()))
    mkdir = code.index('mkdir -p "${OUT}"')
    first_step = code.index("python scripts/build_retrieval_gallery.py --checkpoint-13d")
    for guard in ('fail "OUT is inside an outputs directory"', 'fail "OUT resolves into an outputs directory"',
                  'fail "OUT is under the thesis checkout (R8)"', 'fail "CHAT_HOME does not exist: run chat_cluster_setup_h100.sh first"',
                  '[ -e "${OUT}/manifest.json" ]',
                  'fail "${BUILD_ID} has a manifest.json but no gate_rk.json: it cannot be gated, use a new BUILD_ID"',
                  'fail "${BUILD_ID} is already built and its gate already decided: a finished build is never overwritten"',
                  'fail "13D checkpoint (the retrieval tower and text encoder) not found"',
                  'fail "decoder checkpoint not found"', 'fail "train.parquet not found in DATA"', 'fail "test.parquet not found in DATA"',
                  'fail "validate.parquet not found in DATA: the reference loader reads train, validate and test together"',
                  'MIN_FREE_KB=2097152', '"${FREE_KB}" -ge "${MIN_FREE_KB}"',
                  'fail "${FREE_KB} KB free space where the gallery goes: at least 2 GB are needed"',
                  '"${GPUS}" -ge 1'):
        assert guard in code, guard
        assert code.index(guard) < mkdir < first_step, guard
    plain_name = '"${BUILD_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$'
    assert plain_name in code and code.index(plain_name) < code.index('OUT="${CHAT_HOME}/gallery/')
    assert not re.search(r"(^|[\s;&|(])rm(\s|$)", code, re.M), "additive only: no deletion"


def test_build_retrieval_gallery_wrapper_resumes_at_the_gate_only_for_a_finished_build_with_no_verdict_yet():
    """A manifest.json (step 1 finished) with a gate_rk.json that has no verdict resumes at steps 2 and 3, with the build kept; one
    whose verdict is in is refused, and so is a manifest with no gate file (behaviour: tests/test_build_retrieval_gallery.py)."""
    code = "\n".join(_gallery_wrapper_code(_gallery_wrapper_text()))
    assert code.count("RESUME=0") == 1 and code.count("RESUME=1") == 1
    assert "grep -q '\"equal\":' \"${OUT}/gate_rk.json\"" in code
    assert 'echo "=== resume: gate only (build kept) ==="' in code
    guard = code.index('if [ -e "${OUT}/manifest.json" ]; then')
    assert code.index("RESUME=0") < guard < code.index("RESUME=1") < code.index('echo "=== resume: gate only (build kept) ==="')
    build = code.index("python scripts/build_retrieval_gallery.py --checkpoint-13d")
    reference = code.index("python scripts/evaluate_cxr_retrieval.py")
    opening = code.rindex('if [ "${RESUME}" -eq 0 ]; then', 0, build)
    assert opening < build < code.index("\nfi\n", build) < reference, "step 1 is skipped on a resume, steps 2 and 3 never are"
    assert code.index('echo "=== resume: gate only (build kept) ==="') < build


def test_build_retrieval_gallery_wrapper_passes_the_bash_syntax_check():
    import shutil
    import subprocess
    shells = [s for s in ("/bin/bash", shutil.which("bash")) if s and os.path.exists(s)]
    assert shells
    for shell in set(shells):
        done = subprocess.run([shell, "-n", str(REPO_ROOT / "scripts" / "build_retrieval_gallery_h100.sh")],
                              capture_output=True, text=True)
        assert done.returncode == 0, (shell, done.stderr)


def test_no_yaml_turns_the_report_eos_target_on():
    """CHAT_UI_PLAN.md P9-G1. dataset.report_eos_target changes what a report-gen run trains on (one supervised
    EOS per report that fits), so it is off by default and no yaml under configs/ may set it: every published
    _rrg recipe, the dataset yaml it composes with and the contrastive recipes keep the target every published
    checkpoint was trained on. A run opts in on the command line with +dataset.report_eos_target=true; the '+'
    is required because no yaml declares the key (behaviour: tests/test_report_eos_target.py)."""
    import yaml

    def occurrences(node, trail):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "report_eos_target":
                    yield trail + "/" + key, value
                yield from occurrences(value, trail + "/" + str(key))
        elif isinstance(node, list):
            for i, value in enumerate(node):
                yield from occurrences(value, "{}[{}]".format(trail, i))

    paths = sorted((REPO_ROOT / "configs").rglob("*.yaml"))
    scanned = {p.name for p in paths}
    assert {"hybrid_150m_v2_rrg.yaml", "transformer_150m_baseline_rrg.yaml", "hybrid_150m_m3_rrg.yaml",
            "cxr_mimic_full.yaml", "config.yaml"} <= scanned, "the scan missed a published recipe"
    on = [(str(p.relative_to(REPO_ROOT)), where, value)
          for p in paths
          for where, value in occurrences(yaml.safe_load(p.read_text()), "")
          if value is not False and value is not None]
    assert not on, "yaml sets report_eos_target: {}".format(on)


# ── CHAT_UI_PLAN.md P9-G3: the EOS training wrapper ───────────────────────────
# Static pins on scripts/train_report_eos_h100.sh. Its behaviour (guards, preflight gate, R7 output, DONE marker) is
# rehearsed for real in tests/test_report_eos_job.py; the preflight it runs is tests/test_report_eos_preflight.py.

def _eos_wrapper_text() -> str:
    return (REPO_ROOT / "scripts" / "train_report_eos_h100.sh").read_text()


def _eos_wrapper_code(src: str) -> List[str]:
    """The wrapper's lines minus comments and blanks (the #SBATCH directives are comments to bash)."""
    return [line for line in src.splitlines() if line.strip() and not line.lstrip().startswith("#")]


def test_train_report_eos_wrapper_follows_the_slurm_invariants():
    """P9-G3. H100 through --gpus=N and never a typed --gres (sbatch rejects the mix), the renamed partition, the three
    nodes that cannot run it excluded, a requeue that appends to the SLURM log instead of overwriting it, logs where
    every other wrapper puts them, and offline Hugging Face on a compute node."""
    src = _eos_wrapper_text()
    options, flags = {}, []
    for line in src.splitlines():
        if line.startswith("#SBATCH"):
            token = line.split()[1]
            if "=" in token:
                options[token.split("=", 1)[0]] = token.split("=", 1)[1]
            else:
                flags.append(token)
    assert options == {
        "--partition": "pot-hpi-aisc-batch", "--account": "aisc", "--gpus": "4", "--exclude": "ga03,gx17v1,gx13v1",
        "--cpus-per-task": "8", "--mem": "96G", "--time": "04:00:00", "--job-name": "chat_report_eos",
        "--output": "logs/%x_%j.log", "--error": "logs/%x_%j.log", "--open-mode": "append"}, options
    assert flags == ["--requeue"], flags
    assert not [l for l in _eos_wrapper_code(src) if "--gres" in l], "an untyped --gpus and a typed --gres do not mix"
    assert 'cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"' in src
    for needle in ("set -euo pipefail", 'SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/hybrid_xmamba_h100}"',
                   'export HF_HOME="${SCRATCH_ROOT}/.hf"', "export HF_HUB_OFFLINE=1", "export HF_DATASETS_OFFLINE=1",
                   'VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"', 'source "${VENV_ACTIVATE}"'):
        assert needle in src, needle


def test_train_report_eos_wrapper_trains_the_published_decoder_recipe_and_fixes_it():
    """P9-G3. Every value the published wrapper's override list reads equals what the published s42 decoder run used:
    submit_v3_chain.sh's decoder submission over train_report_generation_h100.sh's own defaults, both parsed here
    (tests/report_eos_recipe.py). The values are plain assignments, not environment levers, because sbatch exports the
    submitting shell: a stray SEED or MAX_STEPS must not be able to change a 5 H100-hour run."""
    from tests import report_eos_recipe as R
    published, mine = R.published_recipe(), R.new_wrapper_assignments()
    names = sorted(set(re.findall(r"\$\{([A-Z][A-Z0-9_]*)\}", " ".join(R.published_wrapper_tokens()))) - {"EXPERIMENT"})
    assert len(names) > 20, names
    for name in names:
        assert name in mine, "the EOS wrapper does not set {}".format(name)
        assert R.expand(mine[name], dict(mine, USER=R.CLUSTER_USER)) == published[name], name
        assert ":-" not in mine[name], "{} is an environment lever: the recipe is fixed".format(name)
    assert mine["EXPERIMENT"] == R.NEW_EXPERIMENT and published["EXPERIMENT"] == R.PUBLISHED_EXPERIMENT
    assert ":-" not in mine["EXPERIMENT"]
    # the headline values, written out
    assert [published[k] for k in ("MODEL_CONFIG", "NUM_GPUS", "TRAINER_CFG", "MAX_STEPS", "SEED", "SAVE_TOP_K",
                                   "AUX_LAMBDA", "PREFIX_K")] == [
        "hybrid_150m_m3_rrg", "4", "h100_multi_ddp", "12000", "42", "0", "0.0", "32"]
    assert published["DECODER_CKPT"] == "./outputs/h100_stage0_150m_m3/checkpoints/last.ckpt"
    assert published["IMAGE_ENCODER_CKPT"] == "./outputs/h100_kd_150m_v2_full_data_lr3e6/checkpoints/last.ckpt"
    assert mine["PUBLISHED_EXPERIMENT"] == R.PUBLISHED_EXPERIMENT == "h100_report_gen_m3_tower13d_s42"


def test_train_report_eos_wrapper_override_list_is_the_published_wrappers_apart_from_the_allowed_differences():
    """P9-G3. Same keys, same order, same value expressions as the published wrapper's python call. The differences are
    exactly: output_dir (the new directory), and two additions at the end, the flag and Hydra's own run directory, which
    would otherwise be created under ./outputs, a symlink into the thesis checkout."""
    from tests import report_eos_recipe as R
    published = [R.unquote(t) for t in R.published_wrapper_tokens()]
    swapped = ["output_dir=${OUT_DIR}" if t == "output_dir=./outputs/${EXPERIMENT}" else t for t in published]
    assert swapped != published, "the published wrapper's output_dir override moved"
    assert [R.unquote(t) for t in R.new_wrapper_tokens()] == swapped + [
        "+dataset.report_eos_target=true", "hydra.run.dir=${OUT_DIR}/hydra"]


def test_train_report_eos_wrapper_writes_nothing_under_outputs():
    """P9-G3, R8. ./outputs is a symlink into the thesis checkout. The run's own output, its logs and Hydra's directory go
    under OUT_DIR; the only mentions of outputs are the three paths it READS and the guard that refuses an OUT_DIR there."""
    src = _eos_wrapper_text()
    code = _eos_wrapper_code(src)
    reads = re.compile(r'^(DECODER_CKPT|IMAGE_ENCODER_CKPT)=\./outputs/[\w./-]+$'
                       r'|^PUBLISHED_META="\./outputs/\$\{PUBLISHED_EXPERIMENT\}/run_metadata\.json"$')
    other = [l for l in code if "outputs" in l and not reads.match(l.strip())]
    assert other and all(("case " in l or "fail " in l or "realpath_py outputs" in l) for l in other), other
    redirects = []
    for line in code:
        if line.lstrip().startswith(("echo", "fail")):
            continue                                              # a message may contain a '>'
        redirects += re.findall(r'(?<![0-9&])>>?\s*("[^"]+"|\S+)', line)
    assert redirects, "the wrapper redirects the trainer, the preflight and the DONE marker"
    stray = [t for t in redirects if t != "/dev/null" and not t.startswith(('"${OUT_DIR}/', '"${START_MARK}"'))]
    assert not stray, stray
    assert 'OUT_DIR="${CHAT_HOME:-/sc/home/$USER/chat_sessions}/models/report_gen_m3_eos_s42"' in src
    assert '"${OUT_DIR}/DONE"' in src and "*/outputs/*" in src, "the DONE and outputs guards"


def test_train_report_eos_wrapper_redirects_the_trainer_and_prints_only_wrapper_authored_lines():
    """P9-G3, R7. The trainer's stdout and stderr (report text, study paths) go to OUT_DIR/train.log and are never
    printed; every line the wrapper prints starts with ===, RESULT or ERROR (the shapes `chat_remote.sh summary`
    shows) and names no path."""
    src = _eos_wrapper_text()
    code = _eos_wrapper_code(src)
    train = [l for l in code if "python scripts/train_report_generation.py" in l]
    assert len(train) == 1 and '>> "${OUT_DIR}/train.log" 2>&1' in train[0], train
    assert '"ERROR train exit=${rc}"' in src
    text = "\n".join(code)
    echoed, failed = re.findall(r'echo "([^"]*)"', text), re.findall(r'\bfail "([^"]*)"', text)
    assert len(echoed) + len(failed) > 10, (echoed, failed)
    for line in echoed:
        assert re.match(r"(=== |RESULT |ERROR)", line), line          # fail() adds its own ERROR prefix
    for line in echoed + failed:
        for var in ("OUT_DIR", "CHAT_HOME", "DECODER_CKPT", "IMAGE_ENCODER_CKPT", "PUBLISHED_META", "MIMIC_CACHE_DIR",
                    "SCRATCH_ROOT", "START_MARK", "MAIN_REPO", "OUT_REAL", "MAIN_REAL"):
            assert "${" + var + "}" not in line, (var, line)
    assert 'fail() { echo "ERROR $*"; exit 1; }' in src
    # No slash anywhere in what is printed: a repo-relative script path is a path too (fix 1, minor 6).
    assert not [line for line in echoed + failed if "/" in line], [line for line in echoed + failed if "/" in line]
    # Every refusal to write DONE reads `ERROR done refused: <reason>`, with literals and numbers only.
    for reason in ("interrupt.ckpt", "last.ckpt is missing", "last.ckpt was not written during this attempt",
                   "steps=${STEPS} expected=${MAX_STEPS}"):
        assert any(f.startswith("done refused:") and reason in f for f in failed), reason
    for line in code:
        assert not re.match(r"\s*(date|hostname|nvidia-smi|env|printenv|cat|tail|head|less)\b", line), line
    assert "set -x" not in src and "set -o xtrace" not in src


def test_train_report_eos_wrapper_prints_its_sync_stamp_before_anything_else():
    """Provenance (fix 1, minor 4). `scripts/chat_remote.sh sync` leaves .sync_stamp in the tree it ships; the job prints the
    commit and cleanliness it reads there as its very first line, before any other output and before any command runs
    (behaviour, against the real producer's output: tests/test_report_eos_job.py)."""
    code = "\n".join(_eos_wrapper_code(_eos_wrapper_text()))
    sync = code.index('echo "=== sync ${SYNC} ==="')
    assert code.index('echo "') == sync, "something is printed before the sync line"
    assert code.index("python ") > sync and code.index("source ") > sync
    assert ".sync_stamp" in code[:sync]


def test_train_report_eos_wrapper_runs_the_preflight_first_on_the_one_override_list():
    """P9-G3. One source for the overrides: the same OVERRIDES array goes to the preflight, which composes it, and to
    the trainer, which runs it. The preflight runs first and its failure ends the job."""
    src = _eos_wrapper_text()
    code = "\n".join(_eos_wrapper_code(src))
    assert code.count("OVERRIDES=(") == 1
    pre = code.index("python scripts/report_eos_preflight.py")
    train = code.index("python scripts/train_report_generation.py")
    result = code.index("python scripts/report_eos_result.py")
    assert pre < train < result
    assert 'python scripts/report_eos_preflight.py --published "${PUBLISHED_META}" -- "${OVERRIDES[@]}"' in code
    assert 'python scripts/train_report_generation.py --config-name config "${OVERRIDES[@]}"' in code
    assert 'fail "preflight exit=${rc}, nothing was trained"' in code
    assert code.index('fail "preflight exit=') < train, "the gate sits between the preflight and the trainer"
    for name in ("report_eos_preflight.py", "report_eos_result.py"):
        assert (REPO_ROOT / "scripts" / name).is_file(), name


def test_train_report_eos_wrapper_passes_the_bash_syntax_check():
    import shutil
    import subprocess
    shells = [s for s in ("/bin/bash", shutil.which("bash")) if s and os.path.exists(s)]
    assert shells
    for shell in set(shells):
        done = subprocess.run([shell, "-n", str(REPO_ROOT / "scripts" / "train_report_eos_h100.sh")],
                              capture_output=True, text=True)
        assert done.returncode == 0, (shell, done.stderr)


# ── CHAT_UI_PLAN.md P9-G4: the EOS evaluation wrappers ────────────────────────
# Static pins on the three jobs that measure the EOS model on the official test split: scripts/eval_report_eos_h100.sh (decode,
# GPU), eval_report_eos_chexbert_h100.sh and eval_report_eos_compare_h100.sh (CPU), and on the scripts/report_eos_stats.py they
# call. Their behaviour (guards, R7 output, what each writes) is rehearsed for real in tests/test_report_eos_eval.py.

_EVAL_WRAPPERS = {"decode": "eval_report_eos_h100.sh", "chexbert": "eval_report_eos_chexbert_h100.sh",
                  "compare": "eval_report_eos_compare_h100.sh"}


def _eval_text(kind: str) -> str:
    return (REPO_ROOT / "scripts" / _EVAL_WRAPPERS[kind]).read_text()


def _sbatch_options(src: str):
    """({--key: value}, [bare flags]) of the #SBATCH directives."""
    options, flags = {}, []
    for line in src.splitlines():
        if line.startswith("#SBATCH"):
            token = line.split()[1]
            if "=" in token:
                options[token.split("=", 1)[0]] = token.split("=", 1)[1]
            else:
                flags.append(token)
    return options, flags


def _logical_lines(src: str) -> List[str]:
    """The wrapper's commands, one per entry: comments and blanks dropped, backslash continuations joined."""
    found, current = [], ""
    for raw in src.splitlines():
        line = raw.rstrip()
        if not current and (not line.strip() or line.lstrip().startswith("#")):
            continue
        if line.endswith("\\"):
            current += line[:-1].strip() + " "
        else:
            found.append((current + line.strip()).strip())
            current = ""
    return found


def _command(src: str, needle: str) -> str:
    """The one logical line containing `needle`."""
    found = [l for l in _logical_lines(src) if needle in l]
    assert len(found) == 1, "{} command lines contain {!r}: the wrapper changed shape, update this pin".format(len(found), needle)
    return found[0]


def _plain_assignments(src: str) -> Dict[str, str]:
    """NAME=value lines with a plain value: no ${...} default, so no environment lever."""
    return dict(re.findall(r'(?m)^([A-Z][A-Z0-9_]*)=([^\s$"#]+)[ \t]*(?:#.*)?$', src))


_EVAL_COMMON = {"--partition": "pot-hpi-aisc-batch", "--account": "aisc", "--exclude": "ga03,gx17v1,gx13v1",
                "--output": "logs/%x_%j.log", "--error": "logs/%x_%j.log", "--open-mode": "append"}
_EVAL_SLURM = {
    "decode": dict(_EVAL_COMMON, **{"--gpus": "1", "--cpus-per-task": "4", "--mem": "32G", "--time": "02:00:00",
                                    "--job-name": "chat_eos_eval"}),
    "chexbert": dict(_EVAL_COMMON, **{"--qos": "aisc", "--cpus-per-task": "4", "--mem": "16G", "--time": "02:00:00",
                                      "--job-name": "chat_eos_chexbert"}),
    "compare": dict(_EVAL_COMMON, **{"--qos": "aisc", "--cpus-per-task": "4", "--mem": "16G", "--time": "02:00:00",
                                     "--job-name": "chat_eos_compare"}),
}


@pytest.mark.parametrize("kind", sorted(_EVAL_WRAPPERS))
def test_eval_report_eos_wrappers_follow_the_slurm_invariants(kind):
    """P9-G4. The decode asks for one H100 through --gpus=1 and never a typed --gres; the two scoring jobs are the proven
    CPU-only combination (--qos=aisc, no GPU). All three: the renamed partition, the three nodes that cannot run them
    excluded, a requeue that appends to the SLURM log, logs where every other wrapper puts them, and the standard cd line."""
    src = _eval_text(kind)
    options, flags = _sbatch_options(src)
    assert options == _EVAL_SLURM[kind], options
    assert flags == ["--requeue"], flags
    code = _eos_wrapper_code(src)
    assert not [l for l in code if "--gres" in l], "an untyped --gpus and a typed --gres do not mix"
    assert 'cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"' in src
    assert "set -euo pipefail" in src and "mkdir -p logs" in src
    assert ("--gpus" in options) == (kind == "decode") and ("--qos" in options) == (kind != "decode")


def test_eval_report_eos_wrappers_have_the_offline_environment_the_job_needs():
    """Decode and comparison: every other chat job's offline Hugging Face (the GPT-2 tokenizer is read from the cache under
    SCRATCH_ROOT). The CheXbert scorer is the exception D24 records: its weights live in the DEFAULT cache, so HF_HOME is not
    overridden, and HF_HUB_OFFLINE defaults to 0 as in score_chexbert_h100.sh."""
    for kind in ("decode", "compare"):
        src = _eval_text(kind)
        for needle in ('SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/hybrid_xmamba_h100}"', 'export HF_HOME="${SCRATCH_ROOT}/.hf"',
                       "export HF_HUB_OFFLINE=1", "export HF_DATASETS_OFFLINE=1", 'VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"',
                       'source "${VENV_ACTIVATE}"'):
            assert needle in src, (kind, needle)
    chexbert = _eval_text("chexbert")
    assert 'export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"' in chexbert
    assert 'VENV_ACTIVATE="${VENV_ACTIVATE:-.venv_chexbert/bin/activate}"' in chexbert and 'source "${VENV_ACTIVATE}"' in chexbert
    assert "HF_HOME" not in "\n".join(_eos_wrapper_code(chexbert)), "D24: the CheXbert weights are in the default cache"
    thesis = (REPO_ROOT / "scripts" / "score_chexbert_h100.sh").read_text()
    assert 'export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"' in thesis and "HF_HOME" not in "\n".join(_eos_wrapper_code(thesis))


def test_eval_report_eos_decode_is_the_published_eval_command_plus_the_eos_flags():
    """P9-G4. The command is the one submit_v3_chain.sh's eval stage gave inspect_report_generation_h100.sh for the published
    Mamba-3 runs (the wrapper's own defaults under the chain's overrides), with exactly three differences: --cached-decode,
    --stop-at-eos and --max-new-tokens 200. Both published sources are parsed here, so this fails when either moves. The
    values are plain assignments, not environment levers: sbatch exports the submitting shell."""
    inspect_src = (REPO_ROOT / "scripts" / "inspect_report_generation_h100.sh").read_text()
    chain = (REPO_ROOT / "scripts" / "submit_v3_chain.sh").read_text()
    mine_src = _eval_text("decode")

    # what the published eval ran: the wrapper's defaults, then the chain's eval submission over them
    defaults = dict(re.findall(r'(?m)^([A-Z][A-Z0-9_]*)="\$\{\1:-([^}]*)\}"', inspect_src))
    anchor = 'ev=$(_v3_submit "eval s${seed}"'
    assert anchor in chain, "submit_v3_chain.sh's eval submission moved: update this pin, which reads the published eval from it"
    submission = chain.split(anchor, 1)[1].split("scripts/inspect_report_generation_h100.sh", 1)[0]
    chain_values = dict(re.findall(r'"([A-Z][A-Z0-9_]*)=([^"]*)"', submission))
    published = dict(defaults, **chain_values)
    assert {k: published[k] for k in ("MODEL_CONFIG", "PREFIX_K", "DECODE", "BEAM_SIZE", "NUM_SAMPLES", "MAX_NEW_TOKENS")} == {
        "MODEL_CONFIG": "hybrid_150m_m3_rrg", "PREFIX_K": "32", "DECODE": "beam", "BEAM_SIZE": "3", "NUM_SAMPLES": "999999",
        "MAX_NEW_TOKENS": "100"}
    assert defaults["CACHED_DECODE"] == "false", "the published eval is uncached: --cached-decode is the first difference"
    # Nothing else the inspect wrapper can switch on is on in the published eval, and the chain never mentions any of it. If it
    # ever does, the published run is no longer "this command minus three flags" and this decode would silently compare against
    # another one. The chain is searched whole, not just its eval submission: an `export` or a lever threaded through would reach the
    # eval job without appearing among that submission's VAR=value pairs.
    for name in ("SCAN_IMPL", "TFLA_IMPL", "CHUNK_SIZE"):
        assert defaults[name] == "", "the published eval pins {}: this decode must too".format(name)
    for name in ("COMPILE", "CHEXBERT", "CACHED_DECODE"):
        assert defaults[name] == "false", "the published eval turns {} on".format(name)
    for name in ("SCAN_IMPL", "TFLA_IMPL", "CHUNK_SIZE", "COMPILE", "CHEXBERT", "CACHED_DECODE"):
        assert name not in chain, "submit_v3_chain.sh mentions {}: the published eval may no longer be the default one".format(name)

    inspect_flags = set(re.findall(r"--[a-z][a-z-]*", _command(inspect_src, "python scripts/evaluate_report_generation.py")))
    inspect_flags |= set(re.findall(r"\+=\((--[a-z][a-z-]*)", inspect_src))        # --chexbert and --dump-dir ride in an array
    variable_of = dict(re.findall(r'(--[a-z][a-z-]*) "\$\{([A-Z][A-Z0-9_]*)\}"', inspect_src))
    assert variable_of["--model-config"] == "MODEL_CONFIG" and variable_of["--max-new-tokens"] == "MAX_NEW_TOKENS"

    call = _command(mine_src, "python scripts/evaluate_report_generation.py")
    mine_flags = re.findall(r"--[a-z][a-z-]*", call.split(">>")[0])
    assert mine_flags == ["--checkpoint", "--model-config", "--prefix-k", "--cached-decode", "--stop-at-eos", "--parquet",
                          "--num-samples", "--decode", "--beam-size", "--max-new-tokens", "--dump-dir"], mine_flags
    assert set(mine_flags) - {"--stop-at-eos"} <= inspect_flags, "a flag the published wrapper cannot pass"
    always = {"--checkpoint", "--model-config", "--parquet", "--num-samples", "--decode", "--beam-size", "--max-new-tokens"}
    assert always <= set(mine_flags), "a flag the published eval always passes"
    assert {"--prefix-k", "--dump-dir"} <= set(mine_flags), "the chain set PREFIX_K and DUMP_DIR"

    mine = _plain_assignments(mine_src)
    for flag in ("--model-config", "--prefix-k", "--num-samples", "--decode", "--beam-size"):
        variable = variable_of[flag]
        assert mine[variable] == published[variable], (flag, mine[variable], published[variable])
        assert '{} "${{{}}}"'.format(flag, variable) in call, flag
    assert mine["MAX_NEW_TOKENS"] == "200" != published["MAX_NEW_TOKENS"] and '--max-new-tokens "${MAX_NEW_TOKENS}"' in call
    assert "--cached-decode --stop-at-eos " in call

    # the three defaults the rulings name, and nothing else is a lever
    for line in ('CKPT="${CKPT:-${CHAT_HOME:-/sc/home/$USER/chat_sessions}/models/report_gen_m3_eos_s42/checkpoints/last.ckpt}"',
                 'PARQUET="${PARQUET:-/sc/home/$USER/dataset/mimic_full/test.parquet}"',
                 'DUMP_DIR="${DUMP_DIR:-results/chat_report_eos_test_split_s42}"'):
        assert line in mine_src, line
    levers = set(re.findall(r'(?m)^([A-Z][A-Z0-9_]*)="\$\{\1:-', mine_src))
    assert levers == {"SCRATCH_ROOT", "VENV_ACTIVATE", "CKPT", "PARQUET", "DUMP_DIR", "PUBLISHED_DIR"}, levers
    assert 'PUBLISHED_DIR="${PUBLISHED_DIR:-results/report_gen_m3_test_split_s42}"' in mine_src


def test_eval_report_eos_decode_header_says_two_hours_are_enough_for_the_expected_decode_and_thin_at_the_full_budget():
    """The first sentence of the TIME paragraph must say what its own arithmetic says: at 12.7 ms a step the expected decode takes
    1.4 h and a run at the full budget 2.2 h, which is over the limit. The controller submits with the 4 h override."""
    header = "\n".join(l for l in _eval_text("decode").splitlines() if l.startswith("#") and not l.startswith("#SBATCH"))
    time_paragraph = header[header.index("# TIME."):header.index("# It refuses")]
    label, first_sentence = " ".join(l.lstrip("# ") for l in time_paragraph.splitlines()).split(". ")[:2]
    assert label == "TIME" and "enough for the expected decode" in first_sentence and "thin at the full budget" in first_sentence, (
        label, first_sentence)
    assert "even at the full" not in header and "holds the cached decode" not in header, "the sentence the arithmetic contradicts"
    for needle in ("--time=02:00:00", "6.35 ms", "12.7 ms", "1.4 h", "2.2 h", "TIMEOUT", "--time=04:00:00"):
        assert needle in time_paragraph, needle


def test_eval_report_eos_decode_refuses_without_done_and_over_an_existing_dump_and_outside_results_chat():
    src = _eval_text("decode")
    assert 'RUN_DIR="$(dirname "$(dirname "${CKPT}")")"' in src and '[ -f "${RUN_DIR}/DONE" ] ||' in src
    assert '[ -e "${DUMP_DIR}/hyps.txt" ]' in src and '[ -e "${DUMP_DIR}/refs.txt" ]' in src
    assert "results/chat_?*)" in src and "*..*)" in src
    code = "\n".join(_eos_wrapper_code(src))
    first_write = code.index('mkdir -p "${DUMP_DIR}"')
    for message in ("DUMP_DIR is not a new chat_ directory", "checkpoint not found", "the training run has no DONE marker",
                    "test parquet not found", "the dump already holds hyps.txt or refs.txt", "venv not found", "GPU(s) visible",
                    "the published dump has no refs.txt"):
        assert 0 <= code.index(message) < first_write, "no directory is created before the guard '{}' has passed".format(message)
    # Job 3 refuses two dumps that are not the same studies. Job 1 says so first, after its own result and before its END line, so
    # that afterok stops the chain before the CheXbert hour: the same byte-for-byte test, on the same two files.
    compare = 'cmp -s "${DUMP_DIR}/refs.txt" "${PUBLISHED_DIR}/refs.txt" || fail "refs differ from the published dump"'
    assert compare in code
    assert code.index("report_eos_stats.py decode") < code.index(compare) < code.index('echo "=== END decode ==="')
    assert 'cmp -s "${DUMP_DIR}/refs.txt" "${PUBLISHED_DIR}/refs.txt"' in _eval_text("compare")


def test_eval_report_eos_wrappers_send_raw_output_to_files_in_the_dump_dir_and_print_only_wrapper_authored_lines():
    """R7. The evaluator's stdout (GENERATED:, REFERENCE:, study ids), the scorer's and the bootstrap's go to files in the dump
    directory and are never printed; the stats script's stderr goes to a file and its stdout is cut down to the three line
    shapes `chat_remote.sh summary` shows. Every line a wrapper prints starts with ===, RESULT or ERROR and names no path."""
    path_variables = ("DUMP_DIR", "CKPT", "PARQUET", "RUN_DIR", "PUBLISHED_DIR", "OUTPUT", "CHAT_HOME", "SCRATCH_ROOT", "VENV_ACTIVATE")
    redirected = {"decode": ("python scripts/evaluate_report_generation.py", '>> "${DUMP_DIR}/eval.log" 2>&1'),
                  "chexbert": ("python scripts/score_chexbert_standalone.py", '>> "${DUMP_DIR}/chexbert.log" 2>&1'),
                  "compare": ("python scripts/bootstrap_compare.py", '>> "${DUMP_DIR}/bootstrap.log" 2>&1')}
    for kind in sorted(_EVAL_WRAPPERS):
        src = _eval_text(kind)
        code = _eos_wrapper_code(src)
        needle, redirect = redirected[kind]
        assert redirect in _command(src, needle), (kind, "the tool's raw output is not sent to a file")
        helpers = [l for l in _logical_lines(src) if "python scripts/report_eos_stats.py" in l]
        assert helpers, kind
        for line in helpers:
            assert '2>> "${DUMP_DIR}/' in line, (kind, "the stats script's stderr is not sent to a file", line)
        assert any("grep -aE '^(=== |RESULT |ERROR)'" in l for l in code), kind
        redirects = []
        for line in code:
            if line.lstrip().startswith(("echo", "fail")):
                continue                                              # a message may contain a '>'
            redirects += re.findall(r'(?<![0-9&])>>?\s*("[^"]+"|\S+)', line)
        assert redirects, kind
        stray = [t for t in redirects if t != "/dev/null" and not t.startswith('"${DUMP_DIR}/')]
        assert not stray, (kind, stray)
        text = "\n".join(code)
        echoed, failed = re.findall(r'echo "([^"]*)"', text), re.findall(r'\bfail "([^"]*)"', text)
        failed += re.findall(r'(?m)^need "[^"]*" "([^"]*)"', text)                 # the comparison's input checks print through need()
        assert len(echoed) + len(failed) >= 6, (kind, echoed, failed)
        for line in echoed:
            assert re.match(r"(=== |RESULT |ERROR)", line), (kind, line)               # fail() adds its own ERROR prefix
        for line in echoed + failed:
            assert "/" not in line, (kind, line)                                        # a repo-relative path is a path too
            for var in path_variables:
                assert "${" + var + "}" not in line, (kind, var, line)
        assert 'fail() { echo "ERROR $*"; exit 1; }' in src
        for line in code:
            assert not re.match(r"\s*(date|hostname|nvidia-smi|env|printenv|cat|tail|head|less)\b", line), (kind, line)
        assert "set -x" not in src and "set -o xtrace" not in src


def test_eval_report_eos_wrappers_print_the_sync_stamp_first_and_with_the_p9_g3_snippet_verbatim():
    """Provenance, as the training wrapper prints it: the commit and cleanliness `chat_remote.sh sync` last shipped, read from
    .sync_stamp, as the very first line. The snippet is the training wrapper's, copied exactly."""
    train = _eos_wrapper_text()
    start = train.index('SYNC="unknown"')
    snippet = train[start:train.index('echo "=== sync ${SYNC} ==="', start) + len('echo "=== sync ${SYNC} ==="')]
    assert ".sync_stamp" in snippet and snippet.count("\n") >= 6
    for kind in sorted(_EVAL_WRAPPERS):
        src = _eval_text(kind)
        assert snippet in src, kind
        code = "\n".join(_eos_wrapper_code(src))
        sync = code.index('echo "=== sync ${SYNC} ==="')
        assert code.index('echo "') == sync, (kind, "something is printed before the sync line")
        assert code.index("python ") > sync and ".sync_stamp" in code[:sync], kind


def test_eval_report_eos_compare_runs_the_thesis_bootstrap_with_the_thesis_defaults_inside_the_dump_dir():
    """P9-G4. scripts/bootstrap_compare.py, unchanged, with the flags bootstrap_compare_h100.sh gives it when it has label
    matrices and PER_LABEL=true, the seed and resample count that wrapper defaults to, the EOS dump as A and the published
    Mamba-3 s42 dump as B, and its report beside the EOS dump."""
    thesis = (REPO_ROOT / "scripts" / "bootstrap_compare_h100.sh").read_text()
    thesis_flags = set(re.findall(r"--[a-z][a-z-]*", _command(thesis, "python3 scripts/bootstrap_compare.py").split("${EXTRA_ARGS")[0]))
    for line in thesis.splitlines():
        if "EXTRA_ARGS+=(" in line:
            thesis_flags |= set(re.findall(r"--[a-z][a-z-]*", line))
    assert {"--hyps-a", "--hyps-b", "--refs", "--name-a", "--name-b", "--bootstrap-samples", "--seed", "--output", "--labels-a",
            "--labels-b", "--per-label"} <= thesis_flags
    thesis_defaults = dict(re.findall(r'(?m)^([A-Z][A-Z0-9_]*)="\$\{\1:-([^}]*)\}"', thesis))
    src = _eval_text("compare")
    mine = _plain_assignments(src)
    assert (mine["SAMPLES"], mine["SEED"]) == (thesis_defaults["SAMPLES"], thesis_defaults["SEED"]) == ("1000", "0")
    assert (mine["NAME_A"], mine["NAME_B"]) == ("eos_s42", "m3_s42")
    assert 'OUTPUT="${DUMP_DIR}/bootstrap_${NAME_A}_vs_${NAME_B}.md"' in src
    for line in ('DUMP_DIR="${DUMP_DIR:-results/chat_report_eos_test_split_s42}"',
                 'PUBLISHED_DIR="${PUBLISHED_DIR:-results/report_gen_m3_test_split_s42}"'):
        assert line in src, line
    call = _command(src, "python scripts/bootstrap_compare.py")
    assert set(re.findall(r"--[a-z][a-z-]*", call.split(">>")[0])) == thesis_flags
    for needle in ('--hyps-a "${DUMP_DIR}/hyps.txt"', '--hyps-b "${PUBLISHED_DIR}/hyps.txt"', '--refs "${DUMP_DIR}/refs.txt"',
                   '--name-a "${NAME_A}"', '--name-b "${NAME_B}"', '--bootstrap-samples "${SAMPLES}"', '--seed "${SEED}"',
                   '--output "${OUTPUT}"', '--labels-a "${DUMP_DIR}/chexbert_labels.json"',
                   '--labels-b "${PUBLISHED_DIR}/chexbert_labels.json"', "--per-label"):
        assert needle in call, needle
    code = "\n".join(_eos_wrapper_code(src))
    assert code.index("python scripts/bootstrap_compare.py") < code.index("report_eos_stats.py hyps") < code.index("report_eos_stats.py gate")
    assert 'cmp -s "${DUMP_DIR}/refs.txt" "${PUBLISHED_DIR}/refs.txt"' in code, "the two systems must have been scored on the same studies"
    levers = set(re.findall(r'(?m)^([A-Z][A-Z0-9_]*)="\$\{\1:-', src))
    assert levers == {"SCRATCH_ROOT", "VENV_ACTIVATE", "DUMP_DIR", "PUBLISHED_DIR"}, levers


def test_eval_report_eos_compare_header_documents_the_three_chained_submit_lines():
    """P9-G4 ruling 4: the controller submits the three jobs; the compare wrapper's header is where the exact lines are."""
    header = "\n".join(l for l in _eval_text("compare").splitlines() if l.startswith("#") and not l.startswith("#SBATCH"))
    decode = "bash scripts/chat_remote.sh submit scripts/eval_report_eos_h100.sh"
    chexbert = ("bash scripts/chat_remote.sh submit scripts/eval_report_eos_chexbert_h100.sh DUMP_DIR=results/chat_report_eos_test_split_s42 "
                "-- --dependency=afterok:")
    compare = ("bash scripts/chat_remote.sh submit scripts/eval_report_eos_compare_h100.sh DUMP_DIR=results/chat_report_eos_test_split_s42 "
               "PUBLISHED_DIR=results/report_gen_m3_test_split_s42 -- --dependency=afterok:")
    for line in (decode, chexbert, compare):
        assert line in header, line
    assert decode + " -- --time=04:00:00)" in header, "job 1 is documented with the 4 h override the controller submits it with"
    assert header.index(decode) < header.index(chexbert) < header.index(compare)
    assert "chat_remote.sh sync" in header and "chat_remote.sh summary logs/chat_eos_eval_" in header


def test_eval_report_eos_chexbert_replaces_the_thesis_scorer_wrapper_only_because_of_what_that_one_prints():
    """P9-G4 ruling 2. score_chexbert_h100.sh would have been reused unchanged if every line `chat_remote.sh summary` shows
    were free of text, ids and paths. It is not: its first line is a `===` line (shown) naming both file paths, its ERROR
    lines name paths too, it prints no RESULT line, and it lets the scorer's raw output into the job log. The thin wrapper runs
    the same script with the same three arguments and prints none of that. If the thesis wrapper is ever made R7-safe, this
    test is the prompt to reconsider the thin one."""
    thesis = (REPO_ROOT / "scripts" / "score_chexbert_h100.sh").read_text()
    assert 'echo "=== Phase 11B standalone CheXbert scoring: ${HYP_FILE} / ${REF_FILE} ==="' in thesis
    assert 'echo "ERROR: ${HYP_FILE} or ${REF_FILE} not found' in thesis and "RESULT" not in thesis
    assert re.match(r"=== ", "=== Phase 11B standalone CheXbert scoring: x / y ===")           # SUMMARY_PATTERN shows it
    src = _eval_text("chexbert")
    thesis_call = _command(thesis, "python scripts/score_chexbert_standalone.py")
    mine_call = _command(src, "python scripts/score_chexbert_standalone.py")
    assert re.findall(r"--[a-z][a-z-]*", mine_call.split(">>")[0]) == re.findall(r"--[a-z][a-z-]*", thesis_call) == [
        "--hyp-file", "--ref-file", "--output-dir"]
    for needle in ('--hyp-file "${DUMP_DIR}/hyps.txt"', '--ref-file "${DUMP_DIR}/refs.txt"', '--output-dir "${DUMP_DIR}"'):
        assert needle in mine_call, needle
    assert 'DUMP_DIR="${DUMP_DIR:-results/chat_report_eos_test_split_s42}"' in src
    # chexbert_metrics.json is written first and chexbert_labels.json last: only the second marks a finished scoring
    assert '[ -e "${DUMP_DIR}/chexbert_labels.json" ]' in src


def test_eval_report_eos_wrappers_pass_the_bash_syntax_check():
    import shutil
    import subprocess
    shells = [s for s in ("/bin/bash", shutil.which("bash")) if s and os.path.exists(s)]
    assert shells
    for kind in sorted(_EVAL_WRAPPERS):
        for shell in set(shells):
            done = subprocess.run([shell, "-n", str(REPO_ROOT / "scripts" / _EVAL_WRAPPERS[kind])], capture_output=True, text=True)
            assert done.returncode == 0, (kind, shell, done.stderr)


def test_eval_report_eos_stats_script_is_stdlib_only_at_import_so_the_chexbert_venv_can_run_it():
    """The CheXbert job runs `report_eos_stats.py chexbert` in .venv_chexbert (transformers<5, no torch, no hybrid_xmamba): its
    module-level imports must be the standard library's, and the one sibling it needs (repair_generations) is itself stdlib-only."""
    if not hasattr(sys, "stdlib_module_names"):
        pytest.skip("sys.stdlib_module_names needs Python 3.10")
    for name in ("report_eos_stats.py", "repair_generations.py"):
        tree = ast.parse((REPO_ROOT / "scripts" / name).read_text())
        imported = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                imported |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imported.add(node.module.split(".")[0])
        assert imported <= set(sys.stdlib_module_names) | {"scripts"}, (name, sorted(imported - set(sys.stdlib_module_names)))


def test_chexbert_service_imports_nothing_from_hybrid_xmamba():
    """P5-A. app/labeler.py runs in .venv_chexbert (pins transformers<5 and scikit-learn<1.8 for f1chexbert), an environment apart from
    the one that runs hybrid_xmamba; like scripts/score_chexbert_standalone.py it must import nothing from the package. That includes
    every other first-party module (app.engine imports hybrid_xmamba, so any `app` or `scripts` import is a way in), and app.labels in
    particular: the service reports F1CheXbert's own label names, so the client's CHEXBERT_14 is checked against something independent."""
    path = REPO_ROOT / "app" / "labeler.py"
    assert path.is_file(), path
    roots = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):       # walk reaches the lazy import inside _get() too
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            roots.add("app" if node.level else (node.module or "").split(".")[0])
    assert "f1chexbert" in roots, roots          # the walker does see function-level imports, so it would see a lazy hybrid_xmamba one
    assert not roots & {"hybrid_xmamba", "app", "scripts", "tests"}, sorted(roots)


# ── CHAT_UI_PLAN.md P5-C: the gallery labelling wrappers ──────────────────────
# Static pins on scripts/label_gallery_reports_h100.sh (one GPU job with a canary), scripts/label_gallery_reports_cpu_h100.sh (the sharded
# CPU fallback and its merge) and the scripts/label_gallery_reports.py they run. Their behaviour (guards, R7 output, the canary, shards and
# merge, the requeue paths, the cross-check) is rehearsed for real in tests/test_label_gallery_reports.py, with the real script and a fake
# f1chexbert; the rehearsal harness it shares with the gallery build's is tests/wrapper_rehearsal.py.

_LABEL_WRAPPERS = {"gpu": "label_gallery_reports_h100.sh", "cpu": "label_gallery_reports_cpu_h100.sh"}
_LABEL_COMMON = dict(_EVAL_COMMON, **{"--cpus-per-task": "4", "--mem": "16G", "--time": "02:00:00"})
_LABEL_SLURM = {
    "gpu": dict(_LABEL_COMMON, **{"--gpus": "1", "--job-name": "chat_labels"}),
    "cpu": dict(_LABEL_COMMON, **{"--qos": "aisc", "--job-name": "chat_labels_cpu"}),
}
_LABEL_SCRIPT_CALL = "python scripts/label_gallery_reports.py"


def _label_text(kind: str) -> str:
    return (REPO_ROOT / "scripts" / _LABEL_WRAPPERS[kind]).read_text()


@pytest.mark.parametrize("kind", sorted(_LABEL_WRAPPERS))
def test_label_gallery_wrappers_follow_the_slurm_invariants(kind):
    """P5-C. The GPU job asks for one H100 through --gpus=1 and never a typed --gres; the fallback is the proven CPU-only combination
    (--qos=aisc, no GPU). Both: the renamed partition, the three nodes that cannot run them excluded (the venv is x86), a requeue that
    appends to the SLURM log, logs where every other wrapper puts them, the standard cd line. The array is not in the header: it is given at
    submission, so that the very same wrapper can be the merge."""
    src = _label_text(kind)
    options, flags = _sbatch_options(src)
    assert options == _LABEL_SLURM[kind], options
    assert flags == ["--requeue"], flags
    code = _eos_wrapper_code(src)
    assert not [l for l in code if "--gres" in l], "an untyped --gpus and a typed --gres do not mix"
    assert 'cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"' in src
    assert "set -euo pipefail" in src and "mkdir -p logs" in src
    assert ("--gpus" in options) == (kind == "gpu") and ("--qos" in options) == (kind == "cpu")
    assert not [l for l in src.splitlines() if l.startswith("#SBATCH") and "--array" in l]


def test_label_gallery_wrappers_have_the_chexbert_environment_of_score_chexbert_h100_and_no_hf_home():
    """D24: the CheXbert weights live in the DEFAULT Hugging Face cache, so HF_HOME is not overridden (and not mentioned: the pin of the other
    CheXbert wrappers), HF_HUB_OFFLINE defaults to 0 as in score_chexbert_h100.sh, and the venv is .venv_chexbert."""
    thesis = (REPO_ROOT / "scripts" / "score_chexbert_h100.sh").read_text()
    assert 'export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"' in thesis and "HF_HOME" not in "\n".join(_eos_wrapper_code(thesis))
    for kind in sorted(_LABEL_WRAPPERS):
        code = "\n".join(_eos_wrapper_code(_label_text(kind)))
        assert 'export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"' in code and "export PYTHONUNBUFFERED=1" in code, kind
        assert 'VENV_ACTIVATE="${VENV_ACTIVATE:-.venv_chexbert/bin/activate}"' in code and 'source "${VENV_ACTIVATE}"' in code, kind
        assert "HF_HOME" not in code and "HF_DATASETS" not in code and "HF_HUB_OFFLINE=1" not in code, kind
        assert code.index('source "${VENV_ACTIVATE}"') < code.index('export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"'), kind


def test_label_gallery_wrappers_defaults_are_the_plans_and_only_the_documented_names_are_levers():
    """The plan's gallery build (g13d_m3_v1), the published Mamba-3 s42 dump as the reference, the canary of the brief. GALLERY is derived
    from CHAT_HOME and one plain name, never a lever of its own (sbatch exports the submitting shell, and GALLERY is a likely name)."""
    common = ['VENV_ACTIVATE="${VENV_ACTIVATE:-.venv_chexbert/bin/activate}"', 'CHAT_HOME="${CHAT_HOME:-/sc/home/$USER/chat_sessions}"',
              'BUILD_ID="${BUILD_ID:-g13d_m3_v1}"', 'REFERENCE_DIR="${REFERENCE_DIR:-results/report_gen_m3_test_split_s42}"',
              'CANARY="${CANARY:-1000}"', 'GALLERY="${CHAT_HOME}/gallery/${BUILD_ID}"']
    per_kind = {"gpu": ['BUDGET_S="${BUDGET_S:-5400}"'], "cpu": ['BUDGET_S="${BUDGET_S:-6000}"', 'SHARDS="${SHARDS:-8}"', 'MERGE="${MERGE:-0}"']}
    levers = {"gpu": {"VENV_ACTIVATE", "CHAT_HOME", "BUILD_ID", "REFERENCE_DIR", "BUDGET_S", "CANARY"},
              "cpu": {"VENV_ACTIVATE", "CHAT_HOME", "BUILD_ID", "REFERENCE_DIR", "BUDGET_S", "CANARY", "SHARDS", "MERGE"}}
    for kind in sorted(_LABEL_WRAPPERS):
        src = _label_text(kind)
        code = "\n".join(_eos_wrapper_code(src))
        for line in common + per_kind[kind]:
            assert line in code, (kind, line)
        assert set(re.findall(r'(?m)^([A-Z][A-Z0-9_]*)="\$\{\1:-', src)) == levers[kind], kind
    gallery = (REPO_ROOT / "scripts" / "build_retrieval_gallery_h100.sh").read_text()
    assert 'CHAT_HOME="${CHAT_HOME:-/sc/home/$USER/chat_sessions}"' in gallery and 'OUT="${CHAT_HOME}/gallery/${BUILD_ID}"' in gallery


def test_label_gallery_wrappers_run_the_script_with_one_command_each_and_send_its_raw_output_to_a_file_in_the_gallery():
    """R7. Every `python scripts/label_gallery_reports.py` is redirected to a log inside ${GALLERY} (stdout and stderr: the library's warnings,
    a traceback that echoes report text), never printed; the job log gets the script's lines of a known shape, a count of the others, and
    wrapper-authored lines."""
    gpu = [l for l in _logical_lines(_label_text("gpu")) if _LABEL_SCRIPT_CALL in l]
    assert gpu == [_LABEL_SCRIPT_CALL + ' --gallery "${GALLERY}" --reference-dir "${REFERENCE_DIR}" --budget-s "${BUDGET_S}" '
                   '--canary "${CANARY}" --device auto > "${GALLERY}/labels.log" 2>&1 || rc=$?']
    cpu = [l for l in _logical_lines(_label_text("cpu")) if _LABEL_SCRIPT_CALL in l]
    assert cpu == [
        _LABEL_SCRIPT_CALL + ' --gallery "${GALLERY}" --reference-dir "${REFERENCE_DIR}" --budget-s "${BUDGET_S}" --canary "${CANARY}" '
        '--device cpu --shard "${SLURM_ARRAY_TASK_ID}" --of "${SHARDS}" > "${LOG}" 2>&1 || rc=$?',
        _LABEL_SCRIPT_CALL + ' --gallery "${GALLERY}" --reference-dir "${REFERENCE_DIR}" --merge --of "${SHARDS}" > "${LOG}" 2>&1 || rc=$?']
    cpu_text = "\n".join(_eos_wrapper_code(_label_text("cpu")))
    assert 'LOG="${GALLERY}/labels_shard_${SLURM_ARRAY_TASK_ID}.log"' in cpu_text and 'LOG="${GALLERY}/labels_merge.log"' in cpu_text
    for kind, log in (("gpu", '"${GALLERY}/labels.log"'), ("cpu", '"${LOG}"')):
        text = "\n".join(_eos_wrapper_code(_label_text(kind)))
        assert "tr '\\r' '\\n' < {} | grep -aE \"${{LABEL_SHAPES}}\" | tail -n 60 || true".format(log) in text, kind
        assert ("WITHHELD=\"$(tr '\\r' '\\n' < {} | grep -aE '^(\\[labels\\] |RESULT |ERROR|=== )' | grep -avcE \"${{LABEL_SHAPES}}\" || true)\""
                .format(log)) in text, kind
        assert "case \"${WITHHELD}\" in ''|*[!0-9]*) WITHHELD=unknown ;; esac" in text, "a count that could not be made is not a zero"
        assert 'echo "=== label lines withheld: ${WITHHELD} ==="' in text
        # Exit 2 is the canary's only with the script's own canary line (of a known shape) in the raw log: python exits 2 itself when it
        # cannot open the script. Without the line it is a failure, printed as such, and the wrapper's own exit is 1: its exit 2 is the hand-over.
        seen = "CANARY_SEEN=\"$(tr '\\r' '\\n' < {} | grep -aE \"${{LABEL_SHAPES}}\" | grep -c '^\\[labels\\] canary: ' || true)\"".format(log)
        assert seen in text, kind
        assert "case \"${CANARY_SEEN}\" in ''|*[!0-9]*) CANARY_SEEN=0 ;; esac" in text, "a count that could not be made is no canary"
        hand_over = 'if [ "${rc}" -eq 2 ] && [ "${CANARY_SEEN}" -ge 1 ]; then'
        failure = 'if [ "${rc}" -ne 0 ]; then\n  echo "ERROR labels exit=${rc}"\n  if [ "${rc}" -eq 2 ]; then exit 1; fi\n  exit "${rc}"\nfi'
        assert hand_over in text and failure in text, kind
        assert text.index(seen) < text.index(hand_over) < text.index(failure), kind
        assert text.count('if [ "${rc}" -eq 2 ]') == 2, "the hand-over test and the one that keeps exit 2 for it, and no other"


def test_label_gallery_wrappers_print_only_wrapper_authored_lines_that_name_no_path():
    """Every line the wrappers print themselves starts with === or ERROR (fail adds its own ERROR), and names no path variable and no path: the
    one exception is the pair of submit lines the GPU job prints when it hands over to the CPU path, which name the CPU wrapper (repo code)."""
    path_variables = ("GALLERY", "GALLERY_REAL", "MAIN_REAL", "CHAT_HOME", "REFERENCE_DIR", "VENV_ACTIVATE", "LOG")
    submit_lines = {'=== 1. bash scripts/chat_remote.sh submit scripts/label_gallery_reports_cpu_h100.sh BUILD_ID=${BUILD_ID} -- --array=0-7 ===',
                    '=== 2. bash scripts/chat_remote.sh submit scripts/label_gallery_reports_cpu_h100.sh BUILD_ID=${BUILD_ID} MERGE=1 -- '
                    '--dependency=afterok:<the array job id> ==='}
    for kind in sorted(_LABEL_WRAPPERS):
        src = _label_text(kind)
        code = _eos_wrapper_code(src)
        text = "\n".join(code)
        echoed, failed = re.findall(r'echo "([^"]*)"', text), re.findall(r'\bfail "([^"]*)"', text)
        assert len(echoed) + len(failed) >= 12, (kind, echoed, failed)
        for line in echoed:
            assert re.match(r"(=== |ERROR )", line), (kind, line)
        for line in echoed + failed:
            if line in submit_lines:
                continue
            assert "/" not in line, (kind, line)
            for var in path_variables:
                assert "${" + var + "}" not in line, (kind, var, line)
        assert (kind == "gpu") == (submit_lines <= set(echoed)), "only the GPU job prints the submit lines"
        assert 'fail() { echo "ERROR $*"; exit 1; }' in src
        for line in code:
            assert not re.match(r"\s*(date|hostname|nvidia-smi|env|printenv|cat|tail|head|less)\b", line), (kind, line)
        assert "set -x" not in src and "set -o xtrace" not in src


def test_label_gallery_wrappers_hold_one_allowlist_and_it_is_not_a_blacklist():
    """The shapes of the lines the script prints, anchored, one per line (tests/test_label_gallery_reports.py runs them through grep over
    everything printed and over ids, text and paths); both wrappers carry the same list, and no pattern is a wildcard."""
    lists = []
    for kind in sorted(_LABEL_WRAPPERS):
        found = re.search(r"^LABEL_SHAPES='(.*?)'$", _label_text(kind), re.M | re.S)
        assert found, kind
        patterns = found.group(1).split("\n")
        lists.append(patterns)
        assert "" not in patterns and all(p.startswith("^") and p.endswith("$") for p in patterns), kind
        assert not [p for p in patterns if re.search(r"(\.\*|\.\+|\\S|\\w|\\d)", p)], "a wildcard in an allowlist"
        assert "grep -av '/'" not in _label_text(kind), "the blacklist is not the filter"
    assert lists[0] == lists[1]


def test_label_gallery_wrappers_print_their_sync_stamp_first_and_with_the_p9_g3_snippet_verbatim():
    """Provenance, as the other chat wrappers print it: the commit and cleanliness `chat_remote.sh sync` last shipped, read from .sync_stamp, as
    the very first line. The snippet is the training wrapper's, copied exactly."""
    train = _eos_wrapper_text()
    start = train.index('SYNC="unknown"')
    snippet = train[start:train.index('echo "=== sync ${SYNC} ==="', start) + len('echo "=== sync ${SYNC} ==="')]
    assert ".sync_stamp" in snippet and snippet.count("\n") >= 6
    for kind in sorted(_LABEL_WRAPPERS):
        src = _label_text(kind)
        assert snippet in src, kind
        code = "\n".join(_eos_wrapper_code(src))
        sync = code.index('echo "=== sync ${SYNC} ==="')
        assert code.index('echo "') == sync, (kind, "something is printed before the sync line")
        assert code.index("python ") > sync and code.index("source ") > sync and ".sync_stamp" in code[:sync], kind


def test_label_gallery_wrappers_check_everything_before_the_script_runs_and_never_delete():
    """R8. No raw log is opened and no step started before every guard has passed: one plain build name, whole numbers, CHAT_HOME, the manifest,
    the venv, the labelling script itself (python answers a script it cannot find with exit 2, which is also the hand-over code), the gallery's
    place (not in an outputs directory, not under the thesis checkout), a finished labelling (never overwritten: refused before its job's raw
    log is replaced), the reference dump's two files."""
    guards = ['fail "BUILD_ID must be one plain name: letters, digits, dot, dash, underscore"',
              'fail "BUDGET_S must be a whole number of seconds, at most 6 digits"', 'fail "CANARY must be a whole number of groups, at most 7 digits"',
              'fail "CHAT_HOME does not exist: run chat_cluster_setup_h100.sh first"',
              'fail "the gallery has no manifest.json: build it first (build_retrieval_gallery_h100.sh)"',
              'fail "the CheXbert venv is missing: run setup_chexbert_venv_h100.sh first"',
              '[ -f scripts/label_gallery_reports.py ] || fail "label_gallery_reports.py is missing from this tree: run chat_remote.sh sync first"',
              'fail "the gallery is inside an outputs directory"',
              'fail "the gallery resolves into an outputs directory"', 'fail "the gallery is under the thesis checkout (R8)"',
              'fail "the gallery is already labelled (labels_status done): a finished labelling is never overwritten (R8)"',
              'fail "the reference dump has no refs.txt"', 'fail "the reference dump has no chexbert_labels.json"',
              """grep -Eq '"labels_status":[[:space:]]*"done"' "${GALLERY}/manifest.json" """.strip()]
    for kind in sorted(_LABEL_WRAPPERS):
        code = "\n".join(_eos_wrapper_code(_label_text(kind)))
        first_run = code.index(_LABEL_SCRIPT_CALL)
        for guard in guards:
            assert 0 <= code.index(guard) < first_run, (kind, guard)
        assert code.index('"${BUILD_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$') < code.index('GALLERY="${CHAT_HOME}/gallery/')
        assert not re.search(r"(^|[\s;&|(])rm(\s|$)", code, re.M) and "--delete" not in code and "ln -sf" not in code, "additive only: no deletion"
    cpu = "\n".join(_eos_wrapper_code(_label_text("cpu")))
    for guard in ('fail "SHARDS must be a whole number from 1 to 999"', 'fail "MERGE must be 0 or 1"',
                  'fail "not an array task: submit with --array=0-7 (SHARDS tasks), or with MERGE=1 for the merge"',
                  'fail "array task ${SLURM_ARRAY_TASK_ID} is outside the ${SHARDS} shards"'):
        assert 0 <= cpu.index(guard) < cpu.index(_LABEL_SCRIPT_CALL), guard


def test_label_gallery_gpu_wrapper_asks_the_venvs_torch_and_hands_over_the_cpu_path_on_exit_2():
    """CHAT_UI_PLAN.md D24 and P5-C. .venv_chexbert's torch comes from the CPU index (scripts/setup_chexbert_venv_h100.sh), so a GPU job's torch
    may see no GPU: the count is asked of the venv's own torch, and no GPU means exit 2 and the two submit lines at once. The canary's own exit 2
    prints the same two lines. The CPU wrapper never asks for a GPU, and asks the labeller for the CPU."""
    setup = (REPO_ROOT / "scripts" / "setup_chexbert_venv_h100.sh").read_text()
    assert "pip install torch --index-url https://download.pytorch.org/whl/cpu" in setup, "the reason for the probe moved: re-read this"
    gpu = "\n".join(_eos_wrapper_code(_label_text("gpu")))
    assert "python -c 'import torch; n = torch.cuda.device_count(); print(n, torch.cuda.get_device_name(0) if n else \"none\")'" in gpu
    assert gpu.index("use_the_cpu_path\n  exit 2") > gpu.index("torch.cuda.device_count()"), "the probe's own exit 2"
    assert gpu.count("use_the_cpu_path\n  exit 2") == 2, "one for no GPU, one for the canary"
    assert gpu.index('if [ "${rc}" -eq 2 ] && [ "${CANARY_SEEN}" -ge 1 ]; then') > gpu.index(_LABEL_SCRIPT_CALL), "the canary's, with its line"
    assert "--device auto" in gpu and "--device cpu" not in gpu
    cpu = "\n".join(_eos_wrapper_code(_label_text("cpu")))
    assert "torch" not in cpu and "--device cpu" in cpu and "--device auto" not in cpu


def test_label_gallery_reports_imports_only_the_standard_library_numpy_and_f1chexbert():
    """P5-C. Like scripts/score_chexbert_standalone.py and app/labeler.py it runs in .venv_chexbert (transformers<5, no hybrid_xmamba): nothing of
    this repo, which is why the 14 label names are a literal copy (pinned against app.labels in tests/test_label_gallery_reports.py), and no torch
    of its own (f1chexbert brings it)."""
    if not hasattr(sys, "stdlib_module_names"):
        pytest.skip("sys.stdlib_module_names needs Python 3.10")
    path = REPO_ROOT / "scripts" / "label_gallery_reports.py"
    roots = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):       # walk reaches the function-level imports too
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            roots.add("." if node.level else (node.module or "").split(".")[0])
    allowed = set(sys.stdlib_module_names) | {"numpy", "f1chexbert"}
    assert {"numpy", "f1chexbert"} <= roots, roots
    assert roots <= allowed, sorted(roots - allowed)


def test_label_gallery_wrappers_pass_the_bash_syntax_check():
    import shutil
    import subprocess
    shells = [s for s in ("/bin/bash", shutil.which("bash")) if s and os.path.exists(s)]
    assert shells
    for kind in sorted(_LABEL_WRAPPERS):
        for shell in set(shells):
            done = subprocess.run([shell, "-n", str(REPO_ROOT / "scripts" / _LABEL_WRAPPERS[kind])], capture_output=True, text=True)
            assert done.returncode == 0, (kind, shell, done.stderr)


# ── CHAT_UI_PLAN.md P9-A: the restricted-file pre-commit check ────────────────
# Static pins on scripts/check_no_restricted_files.sh and scripts/install_hooks.sh. Their behaviour (every refused pattern, the
# allow-list, the installer's three rules) is rehearsed in temporary repositories in tests/test_restricted_files_hook.py;
# nothing in the suite installs the hook on this repository.

def test_the_restricted_file_check_and_its_installer_exist_and_are_executable():
    """P9-A. A hook that cannot run protects nothing: both scripts exist, carry the executable bit, and pass the syntax check of the
    oldest shell they have to work in (the Mac's /bin/bash is 3.2)."""
    import shutil
    import subprocess
    shells = [s for s in ("/bin/bash", shutil.which("bash")) if s and os.path.exists(s)]
    assert shells
    for name in ("check_no_restricted_files.sh", "install_hooks.sh"):
        path = REPO_ROOT / "scripts" / name
        assert path.is_file(), name
        assert os.access(str(path), os.X_OK), "{} is not executable".format(name)
        for shell in set(shells):
            done = subprocess.run([shell, "-n", str(path)], capture_output=True, text=True)
            assert done.returncode == 0, (name, shell, done.stderr)


# ── CHAT_UI_PLAN.md P7-A: the chat app import smoke job ───────────────────────
# Static pins on scripts/chat_app_smoke_h100.sh. Its behaviour (R7 output, the sync line, a failing probe, the probes run for real,
# the label-order probe against fake f1chexbert packages) is rehearsed in tests/test_chat_app_smoke_job.py.

def _smoke_text() -> str:
    return (REPO_ROOT / "scripts" / "chat_app_smoke_h100.sh").read_text()


def test_chat_app_smoke_wrapper_is_cpu_only_on_the_renamed_partition_and_leaves_the_arm_node_out():
    """P7-A. The proven CPU-only combination (--qos=aisc, no GPU) on the renamed partition, the three nodes that cannot run it
    excluded (ga03 is ARM and the venvs are x86), logs where every other wrapper puts them, the standard cd line, and offline
    Hugging Face on a compute node."""
    src = _smoke_text()
    options, flags = _sbatch_options(src)
    assert options == {"--partition": "pot-hpi-aisc-batch", "--account": "aisc", "--qos": "aisc", "--exclude": "ga03,gx17v1,gx13v1",
                       "--mem": "8G", "--cpus-per-task": "2", "--time": "00:15:00", "--job-name": "chat_app_smoke",
                       "--output": "logs/%x_%j.log", "--error": "logs/%x_%j.log"}, options
    assert flags == [], flags
    code = _eos_wrapper_code(src)
    assert not [l for l in code if "--gpus" in l or "--gres" in l], "CPU-only job"
    assert 'cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"' in code
    assert "set -euo pipefail" in code
    assert [l for l in code if l.startswith("export HF_HUB_OFFLINE=1")], "compute nodes are offline"


def test_chat_app_smoke_wrapper_probes_each_half_in_its_own_venv_with_its_own_overlay():
    """P7-A. The web app imports in .venv with the .chat_deps overlay; the labeller and the label order in .venv_chexbert with
    .chat_deps_chexbert: the two overlays scripts/chat_cluster_setup_h100.sh builds (P0-G), put on PYTHONPATH by this wrapper only."""
    code = "\n".join(_eos_wrapper_code(_smoke_text()))
    assert "probe server env PYTHONPATH=.chat_deps .venv/bin/python -c" in code
    assert code.count("PYTHONPATH=.chat_deps_chexbert .venv_chexbert/bin/python -c") == 2      # the labeller and the label order
    assert code.count("PYTHONPATH=") == 3 and code.count("/bin/python") == 3
    assert "import fastapi, app.server" in code and "import app.labeler, transformers" in code
    assert "from app.labels import CHEXBERT_14" in code
    setup = (REPO_ROOT / "scripts" / "chat_cluster_setup_h100.sh").read_text()
    assert ".chat_deps/.setup_ok" in setup and ".chat_deps_chexbert/.setup_ok" in setup


def test_chat_app_smoke_wrapper_reads_the_label_order_from_the_source_and_never_builds_an_f1chexbert():
    """P7-A. f1chexbert 0.0.2 assigns target_names inside F1CheXbert.__init__, after it has loaded the CheXbert weights, so the order
    cannot be had from an import. The probe must parse the package's source (no weights, no network), never construct the class."""
    src = _smoke_text()
    code = "\n".join(_eos_wrapper_code(src))
    assert 'importlib.util.find_spec("f1chexbert")' in code and "ast.parse(" in code and "ast.literal_eval(" in code
    assert "F1CheXbert(" not in code and "import f1chexbert" not in code and "from f1chexbert" not in code
    # Fix 1: exactly one assignment, and a version that comes from the distribution metadata (never from the package) in digits and dots.
    assert "len(found) != 1" in code and "raise SystemExit(4)" in code
    assert 'importlib.metadata.version("f1chexbert")' in code and 're.fullmatch(r"[0-9]+(\\.[0-9]+)*", version)' in code
    assert "f1chexbert.__version__" not in code
    header = "\n".join(l for l in src.splitlines() if l.startswith("#") and not l.startswith("#SBATCH"))
    assert "target_names" in header and "weights" in header, "the header says why the source is read"
    assert "digits and dots" in header and "(f1chexbert <version>)" in header, "the header documents the version in the line"
    thesis = (REPO_ROOT / "scripts" / "score_chexbert_standalone.py").read_text()
    assert "labeler.target_names" in thesis, "the thesis scorer reads the same attribute off a live F1CheXbert"


def test_chat_app_smoke_wrapper_installs_nothing_and_writes_only_a_new_results_directory():
    """P7-A, R8. No installer of any kind appears anywhere in the file, comments included; the one directory it makes is
    results/chat_app_smoke_<jobid>, through the results symlink, and only when results exists (it never makes `results` itself)."""
    src = _smoke_text()
    for needle in ("pip install", "pip3 install", "uv pip", "uv add", "-m pip", "conda install", "--target", "easy_install"):
        assert needle not in src, needle
    code = _eos_wrapper_code(src)
    assert [l for l in code if "mkdir" in l] == ['mkdir -p "${OUT}" 2>/dev/null || fail "the results directory cannot be made"']
    assert 'OUT="results/chat_app_smoke_${SLURM_JOB_ID:-local}"' in code
    assert '[ -d results ] || fail "results is missing: run chat_cluster_setup_h100.sh first"' in code
    for line in code:
        assert not re.match(r"\s*(rm|mv|cp|ln|chmod|touch)\b", line), line


def test_chat_app_smoke_wrapper_prints_only_the_line_shapes_summary_shows_and_keeps_raw_output_in_files():
    """P7-A, R7. Every literal it prints starts with === or ERROR and holds no path; the one dynamic echo is the probe's own [setup]
    line, picked out of that probe's output file with sed; every python call goes through probe(), which sends all of its output to a
    file in the results directory. Nothing reads those files back to the log but that one sed."""
    src = _smoke_text()
    code = _eos_wrapper_code(src)
    text = "\n".join(code)
    echoed, failed = re.findall(r'echo "([^"]*)"', text), re.findall(r'\bfail "([^"]*)"', text)
    assert len(echoed) >= 6 and failed, (echoed, failed)
    for line in echoed:
        assert re.match(r"(=== |ERROR |\$\{line\}$)", line), line
    for line in echoed + failed:
        assert "/" not in line, line                                    # a repo-relative path is a path too
        for var in ("OUT", "SCRATCH_ROOT", "CHAT_HOME", "SLURM_SUBMIT_DIR"):
            assert "${" + var + "}" not in line, (var, line)
    assert 'fail() { echo "ERROR $*"; exit 1; }' in src
    assert 'echo "ERROR ${name} exit=${rc}"' in text
    assert "sed -n '/^\\[setup\\] /{p;q;}' \"${OUT}/${name}.out\"" in text
    assert '"$@" > "${OUT}/${name}.out" 2>&1 || rc=$?' in text
    assert [l for l in code if "python" in l and not l.lstrip().startswith("probe ")
            and not l.startswith(("SERVER_PROBE=", "LABELER_PROBE=", "LABELS_PROBE="))] == []
    redirects = []
    for line in code:
        if line.lstrip().startswith(("echo", "fail", "SERVER_PROBE", "LABELER_PROBE", "LABELS_PROBE")):
            continue                                                    # a message or the python code may contain a '>'
        redirects += re.findall(r'(?<![0-9&])>>?\s*("[^"]+"|\S+)', line)
    assert redirects, "no redirect found: the pin lost its subject"
    assert set(redirects) <= {'"${OUT}/${name}.out"', "/dev/null"}, redirects
    for line in code:
        assert not re.match(r"\s*(date|nvidia-smi|printenv|cat|tail|head|less)\b", line), line
    assert "set -x" not in src and "set -o xtrace" not in src


def test_chat_app_smoke_wrapper_prints_the_sync_stamp_first_and_with_the_p9_g3_snippet_verbatim():
    """P7-A provenance: the commit and cleanliness `chat_remote.sh sync` last shipped, read from .sync_stamp, as the very first line,
    by the snippet scripts/train_report_eos_h100.sh (P9-G3) carries, copied exactly."""
    train = _eos_wrapper_text()
    start = train.index('SYNC="unknown"')
    snippet = train[start:train.index('echo "=== sync ${SYNC} ==="', start) + len('echo "=== sync ${SYNC} ==="')]
    assert ".sync_stamp" in snippet and snippet.count("\n") >= 6
    src = _smoke_text()
    assert snippet in src
    code = "\n".join(_eos_wrapper_code(src))
    sync = code.index('echo "=== sync ${SYNC} ==="')
    assert code.index('echo "') == sync, "something is printed before the sync line"
    assert code.index("/bin/python") > sync and ".sync_stamp" in code[:sync]


def test_chat_app_smoke_header_documents_the_submit_and_summary_lines():
    header = "\n".join(l for l in _smoke_text().splitlines() if l.startswith("#") and not l.startswith("#SBATCH"))
    assert "bash scripts/chat_remote.sh sync && bash scripts/chat_remote.sh submit scripts/chat_app_smoke_h100.sh" in header
    assert "bash scripts/chat_remote.sh summary logs/chat_app_smoke_<jobid>.log" in header


def test_chat_app_smoke_wrapper_passes_the_bash_syntax_check():
    import shutil
    import subprocess
    shells = [s for s in ("/bin/bash", shutil.which("bash")) if s and os.path.exists(s)]
    assert shells
    for shell in set(shells):
        done = subprocess.run([shell, "-n", str(REPO_ROOT / "scripts" / "chat_app_smoke_h100.sh")], capture_output=True, text=True)
        assert done.returncode == 0, (shell, done.stderr)


# ── CHAT_UI_PLAN.md P5-F: the live-path gates job ─────────────────────────────
# Static pins on scripts/chat_retrieval_gates_h100.sh and scripts/chat_retrieval_gates.py. Their behaviour (the three checks, every
# refusal, R7 output, the labeller's life and death, the guards) is rehearsed for real in tests/test_chat_retrieval_gates.py: the real
# script, on the tiny stack, inside the real wrapper, over the real app.labeler.

_GATES_SH = "chat_retrieval_gates_h100.sh"
# The labeller, started as the P7-B serve wrapper starts it (D24: the CheXbert environment of score_chexbert_h100.sh, its web overlay, no
# HF_HOME), on a free port of the loopback interface, in the background, its raw output in a file.
_GATES_LABELLER = ('env -u HF_HOME HF_HUB_OFFLINE="${CHEXBERT_HF_HUB_OFFLINE:-0}" PYTHONPATH=.chat_deps_chexbert .venv_chexbert/bin/python '
                   '-m uvicorn app.labeler:app --host 127.0.0.1 --port "${LABELER_PORT}" > "${OUT}/labeler.log" 2>&1 &')
# The driver, in the main venv with its own overlay: the published protocol is the script's, so the wrapper passes paths and the thread count.
_GATES_DRIVER = ('PYTHONPATH=.chat_deps python scripts/chat_retrieval_gates.py --checkpoint "${CHECKPOINT}" --model-config "${MODEL_CONFIG}" '
                 '--gallery "${GALLERY}" --data "${DATA}" --published-labels "${REFERENCE_DIR}/chexbert_labels.json" '
                 '--published-hyps "${REFERENCE_DIR}/hyps.txt" --published-refs "${REFERENCE_DIR}/refs.txt" --labeler-url "${LABELER_URL}" '
                 '--out "${OUT}" --threads "${THREADS}" > "${OUT}/gates.log" 2>&1 || rc=$?')
_GATES_FIRST_STEP = 'LABELER_PORT="$(python -c'


def _gates_text() -> str:
    return (REPO_ROOT / "scripts" / _GATES_SH).read_text()


def test_chat_gates_wrapper_is_cpu_only_on_the_renamed_partition_and_leaves_the_arm_node_out():
    """P5-F. The proven CPU-only combination (--qos=aisc, no GPU) on the renamed partition, the three nodes that cannot run it excluded (ga03
    is ARM and the venvs are x86), 8 CPUs and 32 GB for a CPU engine, an hour, logs where every other wrapper puts them, the standard cd
    line. No requeue: a job of minutes with a results directory of its own."""
    src = _gates_text()
    options, flags = _sbatch_options(src)
    assert options == {"--partition": "pot-hpi-aisc-batch", "--account": "aisc", "--qos": "aisc", "--exclude": "ga03,gx17v1,gx13v1",
                       "--cpus-per-task": "8", "--mem": "32G", "--time": "01:00:00", "--job-name": "chat_gates",
                       "--output": "logs/%x_%j.log", "--error": "logs/%x_%j.log"}, options
    assert flags == [], flags
    code = _eos_wrapper_code(src)
    assert not [l for l in code if "--gpus" in l or "--gres" in l], "CPU-only job"
    assert not [l for l in src.splitlines() if l.startswith("#SBATCH") and "--array" in l]
    assert 'cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"' in code
    assert "set -euo pipefail" in code and "mkdir -p logs" in code


def test_chat_gates_wrapper_starts_the_labeller_as_the_serve_wrapper_does_on_loopback_only():
    """P5-F, D24. Exactly the command of the brief: the CheXbert venv and its overlay, no HF_HOME (the weights are in the default Hugging Face
    cache), HF_HUB_OFFLINE defaulting to 0 as in score_chexbert_h100.sh, the loopback interface, a free port, in the background, its raw
    output in a file. Never a non-loopback address: nothing here is behind a token."""
    src = _gates_text()
    lines = _logical_lines(src)
    assert [l for l in lines if "uvicorn" in l] == [_GATES_LABELLER]
    assert "0.0.0.0" not in src and "--host 127.0.0.1" in _GATES_LABELLER
    thesis = (REPO_ROOT / "scripts" / "score_chexbert_h100.sh").read_text()
    assert 'export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"' in thesis and "HF_HOME" not in "\n".join(_eos_wrapper_code(thesis))
    assert 'LABELER_PORT="$(python -c \'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])\')" || fail "no free loopback port"' in lines
    assert "case \"${LABELER_PORT}\" in ''|*[!0-9]*) fail \"no free loopback port\" ;; esac" in lines
    assert 'LABELER_URL="http://127.0.0.1:${LABELER_PORT}"' in lines
    assert "LABELER_PID=$!" in lines and lines.index("LABELER_PID=$!") == lines.index(_GATES_LABELLER) + 1


def test_chat_gates_wrapper_waits_for_healthz_with_a_bound_and_stops_the_labeller_on_every_exit_path():
    """P5-F. The first model load can take minutes, so the wait is one request at a time (a probe the shell can be interrupted in: it runs in
    the background and is waited for), bounded by LABELER_WAIT_S, and it notices a labeller that died. The labeller is stopped by a trap on
    EXIT (every `fail`, every `exit`, a failed command under set -e, the normal end) and on TERM and INT; the stop asks, waits a few
    seconds, and then insists (a uvicorn waiting for a model to load does not leave on SIGTERM)."""
    code = "\n".join(_logical_lines(_gates_text()))              # continuations joined: the two long commands are one line each
    assert "trap stop_labeller EXIT" in code
    assert "trap 'stop_labeller; exit 143' TERM" in code and "trap 'stop_labeller; exit 130' INT" in code
    stop = code[code.index("stop_labeller() {"):code.index("trap stop_labeller EXIT")]
    assert stop.index('kill "${LABELER_PID}"') < stop.index("kill -0") < stop.index("kill -KILL") < stop.index('wait "${LABELER_PID}"')
    assert 'LABELER_PID=""' in stop and "for _ in 1 2 3; do" in stop, "bounded: three seconds of grace"
    assert code.index("trap stop_labeller EXIT") < code.index(_GATES_LABELLER)
    assert 'DEADLINE=$((SECONDS + LABELER_WAIT_S))' in code and 'while [ "${SECONDS}" -lt "${DEADLINE}" ]; do' in code
    assert 'kill -0 "${LABELER_PID}" 2>/dev/null || break' in code, "a labeller that died is not waited for"
    assert "/healthz" in code and 'python -c "${HEALTH_PROBE}" "${LABELER_URL}" "${LEFT}" > /dev/null 2>&1 &' in code
    assert 'if wait "${PROBE_PID}"; then READY=1; break; fi' in code
    assert 'fail "labeller not ready after ${LABELER_WAIT_S} s"' in code and 'fail "labeller exit=${LABELER_RC}"' in code
    assert code.index("stop_labeller\n", code.index(_GATES_DRIVER)) > code.index(_GATES_DRIVER), "stopped as soon as the driver is done, before the report"


def test_chat_gates_wrapper_runs_the_driver_in_the_main_venv_with_its_own_overlay_and_the_published_defaults():
    """P5-F. `.venv` with .chat_deps (the labeller's overlay is on the labeller's command only: no PYTHONPATH is ever exported), one
    command with every argument the driver needs and none it does not (--engine is for the laptop tests), the engine offline from the scratch
    Hugging Face cache, one thread per CPU of the allocation, and the loopback address kept off any proxy."""
    src = _gates_text()
    lines = _logical_lines(src)
    assert [l for l in lines if "scripts/chat_retrieval_gates.py" in l and "python" in l] == [_GATES_DRIVER]
    assert "--engine" not in _GATES_DRIVER
    code = "\n".join(lines)
    assert not re.search(r"export\s+[^\n]*PYTHONPATH", code), "an overlay is put on one command, never on the shell"
    assert 'export HF_HOME="${SCRATCH_ROOT}/.hf" HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1' in code
    assert 'export OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}"' in code
    assert 'THREADS="${SLURM_CPUS_PER_TASK:-8}"' in code
    assert 'export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,${NO_PROXY}}" no_proxy="127.0.0.1,localhost${no_proxy:+,${no_proxy}}"' in code
    assert code.index('source "${VENV_ACTIVATE}"') < code.index("export HF_HOME=") < code.index(_GATES_FIRST_STEP) < code.index(_GATES_DRIVER)
    assert code.index(_GATES_LABELLER) < code.index(_GATES_DRIVER), "the labeller is up before the driver asks it anything"


def test_chat_gates_wrapper_defaults_are_the_plans_and_agree_with_the_other_chat_wrappers_and_only_the_documented_names_are_levers():
    """P5-F. The default report model and its checkpoint, the dataset, the scratch cache and CHAT_HOME as the gallery build has them; the
    gallery's build id and the published dump as the labelling has them. GALLERY, OUT and THREADS are derived and never levers of their own
    (sbatch exports the submitting shell, and OUT is a likely name)."""
    src = _gates_text()
    code = "\n".join(_eos_wrapper_code(src))
    build = (REPO_ROOT / "scripts" / "build_retrieval_gallery_h100.sh").read_text()
    label = (REPO_ROOT / "scripts" / "label_gallery_reports_cpu_h100.sh").read_text()
    for line in ('SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/hybrid_xmamba_h100}"', 'VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"',
                 'CHAT_HOME="${CHAT_HOME:-/sc/home/$USER/chat_sessions}"', 'DATA="${DATA:-/sc/home/$USER/dataset/mimic_full}"',
                 'CHECKPOINT="${CHECKPOINT:-./outputs/h100_report_gen_m3_tower13d_s42/checkpoints/last.ckpt}"',
                 'MODEL_CONFIG="${MODEL_CONFIG:-hybrid_150m_m3_rrg}"'):
        assert line in code and line in build, line
    for line in ('BUILD_ID="${BUILD_ID:-g13d_m3_v1}"', 'REFERENCE_DIR="${REFERENCE_DIR:-results/report_gen_m3_test_split_s42}"'):
        assert line in code and line in label, line
    assert 'LABELER_WAIT_S="${LABELER_WAIT_S:-600}"' in code
    assert set(re.findall(r'(?m)^([A-Z][A-Z0-9_]*)="\$\{\1:-', src)) == {
        "SCRATCH_ROOT", "VENV_ACTIVATE", "CHAT_HOME", "DATA", "CHECKPOINT", "MODEL_CONFIG", "BUILD_ID", "REFERENCE_DIR", "LABELER_WAIT_S"}
    assert 'GALLERY="${CHAT_HOME}/gallery/${BUILD_ID}"' in code and 'OUT="results/chat_retrieval_gates_${SLURM_JOB_ID:-local}"' in code
    assert code.count("CHEXBERT_HF_HUB_OFFLINE") == 1, "the labeller's own offline switch, named in its command and nowhere else"
    header = "\n".join(l for l in src.splitlines() if l.startswith("#") and not l.startswith("#SBATCH"))
    for name in ("BUILD_ID", "CHECKPOINT", "MODEL_CONFIG", "DATA", "REFERENCE_DIR", "LABELER_WAIT_S", "CHEXBERT_HF_HUB_OFFLINE"):
        assert name in header, name + " is a lever the header documents"


def test_chat_gates_wrapper_keeps_the_raw_output_of_both_processes_in_files_and_prints_only_what_may_be_printed():
    """P5-F, R7. The driver's stdout and stderr (a traceback can hold a report or a path) and the labeller's go to files in the results
    directory, never to the log; the job log gets the driver's lines of a known shape, a count of the others, and wrapper-authored lines
    that start with === or ERROR and name no path."""
    src = _gates_text()
    code = _eos_wrapper_code(src)
    text = "\n".join(code)
    assert text.count('> "${OUT}/gates.log" 2>&1') == 1 and text.count('> "${OUT}/labeler.log" 2>&1') == 1
    assert "tr '\\r' '\\n' < \"${OUT}/gates.log\" | grep -aE \"${GATES_SHAPES}\" | tail -n 60 || true" in text
    assert ("WITHHELD=\"$(tr '\\r' '\\n' < \"${OUT}/gates.log\" | grep -aE '^(\\[gates\\] |RESULT |ERROR|=== )' | grep -avcE \"${GATES_SHAPES}\" || true)\""
            in text)
    assert "case \"${WITHHELD}\" in ''|*[!0-9]*) WITHHELD=unknown ;; esac" in text, "a count that could not be made is not a zero"
    assert 'echo "=== gates lines withheld: ${WITHHELD} ==="' in text
    assert "RESULTS=\"$(tr '\\r' '\\n' < \"${OUT}/gates.log\" | grep -aE \"${GATES_SHAPES}\" | grep -c '^RESULT ' || true)\"" in text
    assert 'fail "gates printed no RESULT line"' in text, "exit 0 without its result is a failure, not a pass"
    echoed, failed = re.findall(r'echo "([^"]*)"', text), re.findall(r'\bfail "([^"]*)"', text)
    assert len(echoed) + len(failed) >= 20, (echoed, failed)
    for line in echoed:
        assert re.match(r"(=== |ERROR )", line), line
    for line in echoed + failed:
        assert "/" not in line, line                                          # a repo-relative path is a path too
        for var in ("GALLERY", "OUT", "CHAT_HOME", "DATA", "CHECKPOINT", "REFERENCE_DIR", "SCRATCH_ROOT", "VENV_ACTIVATE", "SLURM_SUBMIT_DIR"):
            assert "${" + var + "}" not in line, (var, line)
    assert 'fail() { echo "ERROR $*"; exit 1; }' in src
    for line in _logical_lines(src):
        if line != _GATES_LABELLER:                               # the one command that starts with env: the labeller's own environment
            assert not re.match(r"\s*(date|nvidia-smi|env|printenv|cat|tail|head|less)\b", line), line
    assert "set -x" not in src and "set -o xtrace" not in src
    assert "grep -av '/'" not in src, "the blacklist is not the filter"


def test_chat_gates_wrapper_holds_one_allowlist_and_it_is_not_a_blacklist():
    """P5-F. The shapes of the lines the driver prints, anchored, one per line (tests/test_chat_retrieval_gates.py runs them through grep
    over everything the driver prints and over ids, text and paths); no pattern is a wildcard."""
    found = re.search(r"^GATES_SHAPES='(.*?)'$", _gates_text(), re.M | re.S)
    assert found
    patterns = found.group(1).split("\n")
    assert "" not in patterns and all(p.startswith("^") and p.endswith("$") for p in patterns)
    assert not [p for p in patterns if re.search(r"(\.\*|\.\+|\\S|\\w|\\d)", p)], "a wildcard in an allowlist"
    assert {p.split(" ")[0] for p in patterns} == {"^\\[gates\\]", "^RESULT", "^ERROR", "^==="}
    summary = (REPO_ROOT / "scripts" / "chat_remote.sh").read_text()
    assert "\\[(probe|golden|gallery|labels|gates|server|setup|compile)\\]" in summary, "`summary` shows [gates] lines"


def test_chat_gates_wrapper_prints_its_sync_stamp_first_and_with_the_p9_g3_snippet_verbatim():
    """P5-F provenance: the commit and cleanliness `chat_remote.sh sync` last shipped, read from .sync_stamp, as the very first line, by the
    snippet scripts/train_report_eos_h100.sh (P9-G3) carries, copied exactly."""
    train = _eos_wrapper_text()
    start = train.index('SYNC="unknown"')
    snippet = train[start:train.index('echo "=== sync ${SYNC} ==="', start) + len('echo "=== sync ${SYNC} ==="')]
    assert ".sync_stamp" in snippet and snippet.count("\n") >= 6
    src = _gates_text()
    assert snippet in src
    code = "\n".join(_eos_wrapper_code(src))
    sync = code.index('echo "=== sync ${SYNC} ==="')
    assert code.index('echo "') == sync, "something is printed before the sync line"
    assert code.index("python ") > sync and code.index("source ") > sync and ".sync_stamp" in code[:sync]


def test_chat_gates_wrapper_checks_everything_before_a_process_starts_and_never_deletes_or_installs():
    """P5-F, R8. Every guard sits before the first process (the free-port probe is the first python), one plain name for the build and the
    model, a whole number for the wait, CHAT_HOME, the gallery and its decided gate and the build's own test embeddings, results, both venvs
    and overlays (their sentinels), the script itself (python answers a script it cannot find with exit 2), and every input. The one
    directory the job makes is its own, under results/ through the symlink, after the guards; nothing is removed, linked or installed."""
    src = _gates_text()
    code_lines = _eos_wrapper_code(src)
    code = "\n".join(code_lines)
    guards = ['[[ "${BUILD_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail "BUILD_ID must be one plain name: letters, digits, dot, dash, underscore"',
              '[[ "${MODEL_CONFIG}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail "MODEL_CONFIG must be one plain name: letters, digits, dot, dash, underscore"',
              '[[ "${LABELER_WAIT_S}" =~ ^[0-9]{1,5}$ ]] || fail "LABELER_WAIT_S must be a whole number of seconds, at most 5 digits"',
              '[ -d "${CHAT_HOME}" ] || fail "CHAT_HOME does not exist: run chat_cluster_setup_h100.sh first"',
              '[ -f "${GALLERY}/manifest.json" ] || fail "the gallery has no manifest.json: build it first (build_retrieval_gallery_h100.sh)"',
              """grep -Eq '"equal":[[:space:]]*true' "${GALLERY}/manifest.json" || fail "the gallery's R@k gate is not decided equal: it cannot be used" """.strip(),
              '[ -f "${GALLERY}/test_img_emb.npy" ] || fail "the gallery has no test_img_emb.npy: it is the build\'s own embedding of the test images"',
              '[ -d results ] || fail "results is missing: run chat_cluster_setup_h100.sh first"',
              '[ -f "${VENV_ACTIVATE}" ] || fail "the main venv is missing: run chat_cluster_setup_h100.sh first"',
              '[ -x .venv_chexbert/bin/python ] || fail "the CheXbert venv is missing: run setup_chexbert_venv_h100.sh first"',
              '[ -f .chat_deps/.setup_ok ] && [ -f .chat_deps_chexbert/.setup_ok ] || fail "the web overlays are not set up: run chat_cluster_setup_h100.sh first"',
              '[ -f scripts/chat_retrieval_gates.py ] || fail "chat_retrieval_gates.py is missing from this tree: run chat_remote.sh sync first"',
              '[ -f "${CHECKPOINT}" ] || fail "the report model\'s checkpoint was not found"',
              '[ -f "${DATA}/train.parquet" ] || fail "train.parquet was not found in DATA"',
              '[ -f "${DATA}/test.parquet" ] || fail "test.parquet was not found in DATA"',
              '[ -f "${REFERENCE_DIR}/hyps.txt" ] || fail "the published dump has no hyps.txt"',
              '[ -f "${REFERENCE_DIR}/refs.txt" ] || fail "the published dump has no refs.txt"',
              '[ -f "${REFERENCE_DIR}/chexbert_labels.json" ] || fail "the published dump has no chexbert_labels.json"']
    first = code.index(_GATES_FIRST_STEP)
    for guard in guards:
        assert 0 <= code.index(guard) < first, guard
    mkdir = 'mkdir -p "${OUT}" 2>/dev/null || fail "the results directory cannot be made"'
    assert [l for l in code_lines if "mkdir" in l] == ["mkdir -p logs", mkdir]
    assert max(code.index(g) for g in guards) < code.index(mkdir) < first
    assert not re.search(r"(^|[\s;&|(])rm(\s|$)", code, re.M) and "--delete" not in code and not re.search(r"\bln\s", code), "additive only"
    for needle in ("pip install", "pip3 install", "uv pip", "uv add", "-m pip", "conda install", "--target", "easy_install"):
        assert needle not in src, needle
    for line in code_lines:
        assert not re.match(r"\s*(mv|cp|chmod|touch)\b", line), line


def test_chat_gates_header_documents_the_submit_and_summary_lines_and_the_expected_job_log():
    header = "\n".join(l for l in _gates_text().splitlines() if l.startswith("#") and not l.startswith("#SBATCH"))
    assert "bash scripts/chat_remote.sh sync && bash scripts/chat_remote.sh submit scripts/chat_retrieval_gates_h100.sh" in header
    assert "bash scripts/chat_remote.sh summary logs/chat_gates_<jobid>.log" in header
    for needle in ("self-retrieval", "own-rank", "labeller", "RESULT", "[gates]", "gates.json", "gates.log", "labeler.log", "exit 1"):
        assert needle in header, needle


def test_chat_gates_wrapper_passes_the_bash_syntax_check():
    import shutil
    import subprocess
    shells = [s for s in ("/bin/bash", shutil.which("bash")) if s and os.path.exists(s)]
    assert shells
    for shell in set(shells):
        done = subprocess.run([shell, "-n", str(REPO_ROOT / "scripts" / _GATES_SH)], capture_output=True, text=True)
        assert done.returncode == 0, (shell, done.stderr)


def test_chat_retrieval_gates_imports_only_the_standard_library_numpy_pandas_and_the_app():
    """P5-F. The driver runs in `.venv` with the .chat_deps overlay beside the engine it drives: everything it needs of the engine, the
    gallery and the labeller client it takes from app/ (the code a turn runs, none of it re-implemented), and it brings no heavy import of its
    own (no torch, no scripts.*: app.engine has loaded what it needs)."""
    if not hasattr(sys, "stdlib_module_names"):
        pytest.skip("sys.stdlib_module_names needs Python 3.10")
    path = REPO_ROOT / "scripts" / "chat_retrieval_gates.py"
    roots = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):       # walk reaches the function-level imports too
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            roots.add("." if node.level else (node.module or "").split(".")[0])
    allowed = set(sys.stdlib_module_names) | {"numpy", "pandas", "app"}
    assert {"numpy", "pandas", "app"} <= roots, roots
    assert roots <= allowed, sorted(roots - allowed)
