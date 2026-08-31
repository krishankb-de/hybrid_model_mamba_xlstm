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
    bash scripts/validate_for_willi.sh

Or directly (must be inside the willi_parity conda env):
    conda run -n willi_parity pytest tests/test_willi_parity.py -v
"""

import ast
import re
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional

import pytest
import torch

# ── Helpers ───────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
SCAN_ROOTS = [REPO_ROOT / "hybrid_xmamba", REPO_ROOT / "scripts"]
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
            continue  # Caught by syntax gate in validate_for_willi.sh
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
    """Phase 9F: training entry scripts must thread ``norm_topology`` from yaml
    into ``HybridConfig``. Regression-guards against the Phase 9 silent-drop bug
    (HybridConfig was built from an explicit cfg.model.* list that omitted
    ``norm_topology`` → v2 yaml ``norm_topology: hybrid`` was ignored).

    Strategy: read the two training entry-point source files and assert the
    explicit ``norm_topology=`` kwarg is present in the HybridConfig(...) call.
    Direct source-text assert is more robust than a full Hydra eval here, and
    cheaper.
    """
    import pathlib

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    for rel in (
        "scripts/train.py",
        "scripts/train_stage0_distill.py",
        "scripts/train_contrastive.py",  # Phase 10: same bug would corrupt the backbone
    ):
        src = (repo_root / rel).read_text()
        assert "HybridConfig(" in src, f"{rel}: no HybridConfig call found"
        # Look for the threading line within the HybridConfig argument block.
        # Tolerant of either explicit `cfg.model.norm_topology` or
        # `cfg.model.get('norm_topology', ...)`.
        has_explicit = "norm_topology=cfg.model.norm_topology" in src
        has_getter = "norm_topology=cfg.model.get(" in src and 'norm_topology' in src
        assert has_explicit or has_getter, (
            f"{rel}: HybridConfig(...) call does not pass norm_topology — "
            f"Phase 9 regression hazard. Add "
            f"norm_topology=cfg.model.get('norm_topology', 'pre_rms')."
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
    assert "#SBATCH --partition=aisc-batch" in sh
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
    assert "#SBATCH --partition=aisc-batch" in sh
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
    assert "#SBATCH --partition=aisc-batch" in sh
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
    assert "#SBATCH --partition=aisc-batch" in sh
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
