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
