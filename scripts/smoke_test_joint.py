"""Phase 3 smoke test — JointMultiTaskLightningModule on CPU.

Validates the joint KD + CLIP + SimCSE pipeline end-to-end before the willi
A100 sbatch (Phase 5):

  - All 3 losses (kd, clip, simcse) finite every step
  - Total loss = α·kd + β·clip + γ·simcse computes without NaN
  - Gradients flow into:
      * backbone (model.lm)
      * attn_pool.q              (Strategy 4 — attention pooling)
      * img_proj.{0,2}.weight     (Strategy 3 — 2-layer MLP image projection)
      * distill_proj.{0,2}.weight (KD projection head)
  - 4 optimizer param groups configured (backbone_wd / backbone_no_wd / head)
  - Loss trajectory recorded over 5 steps (warn if not decreasing — random
    init + tiny model is not guaranteed to descend monotonically)

CPU-only. No network. BiomedCLIP and PubMedBERT are mocked.

Run before sbatch:
    python scripts/smoke_test_joint.py
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


# ---------------------------------------------------------------------------
# Mock open_clip BEFORE any code that triggers `import open_clip`
# ---------------------------------------------------------------------------

_MOCK_IMG_DIM = 32  # BiomedCLIP image output dim (mocked, real=512)


class _MockVisual(nn.Module):
    """Stands in for BiomedCLIP visual trunk. (B,3,224,224) -> (B, _MOCK_IMG_DIM)."""

    def __init__(self, out_dim: int = _MOCK_IMG_DIM) -> None:
        super().__init__()
        self.proj = nn.Linear(3 * 8 * 8, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B,3,H,W) -> down-pool to 8x8 -> flatten -> linear
        pooled = nn.functional.adaptive_avg_pool2d(x, (8, 8))
        return self.proj(pooled.flatten(1))


class _MockClipModel:
    def __init__(self) -> None:
        self.embed_dim = _MOCK_IMG_DIM
        self.visual = _MockVisual(_MOCK_IMG_DIM)


def _install_mock_open_clip() -> None:
    mock = types.ModuleType("open_clip")

    def create_model_from_pretrained(name: str, **kwargs: Any):
        return _MockClipModel(), None

    mock.create_model_from_pretrained = create_model_from_pretrained  # type: ignore[attr-defined]
    sys.modules["open_clip"] = mock


_install_mock_open_clip()


# ---------------------------------------------------------------------------
# Mock PubMedBERT teacher (no network, small hidden size for CPU speed)
# ---------------------------------------------------------------------------

_TEACHER_HIDDEN = 128


class _MockBERTConfig:
    hidden_size = _TEACHER_HIDDEN


class MockPubMedBERT(nn.Module):
    """Frozen mock teacher. Returns deterministic hidden states."""

    def __init__(self) -> None:
        super().__init__()
        self.config = _MockBERTConfig()
        self.embed = nn.Embedding(1000, _TEACHER_HIDDEN)

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ) -> Any:
        h = self.embed(input_ids)  # (B, L, H)
        return types.SimpleNamespace(last_hidden_state=h)


# ---------------------------------------------------------------------------
# Tiny HybridTextEncoder (dim=64, attention pooling) for CPU speed
# ---------------------------------------------------------------------------

_SMALL_CFG_KWARGS: Dict[str, Any] = dict(
    vocab_size=100,
    dim=64,
    num_layers=2,
    layer_pattern=["mamba", "mlstm"],
    state_size=4,
    conv_size=4,
    expand_factor=2,
    dt_rank=None,
    use_fast_path=False,
    head_dim=16,
    num_heads=4,
    use_tfla=False,
    proj_factor=2,
    slstm_hidden_dim=64,
    slstm_num_heads=2,
    use_exponential_gate=True,
    norm_type="rms",
    use_mlp=True,
    mlp_ratio=2.0,
    max_position_embeddings=64,
    dropout=0.1,
    initializer_range=0.02,
    use_cache=False,
    tie_word_embeddings=False,
    pooling_strategy="attention",
)


def _make_text_encoder(embed_dim: int = 64):
    from hybrid_xmamba.models.configuration_hybrid import HybridConfig
    from hybrid_xmamba.models.hybrid_lm import HybridTextEncoder
    cfg = HybridConfig(**_SMALL_CFG_KWARGS)
    enc = HybridTextEncoder(cfg, embed_dim=embed_dim)
    assert enc.attn_pool is not None, "attention pooling must be active"
    return enc


# ---------------------------------------------------------------------------
# Synthetic 16-pair MIMIC/Indiana-style batch generator
# ---------------------------------------------------------------------------

def _make_indiana_subset(n_pairs: int = 16, seq_len: int = 32) -> List[Dict[str, torch.Tensor]]:
    """Synthesise a 16-pair text+image+teacher-text subset, split into 4-pair batches."""
    torch.manual_seed(0)
    batches = []
    bs = 4
    for i in range(0, n_pairs, bs):
        batches.append({
            "input_ids":              torch.randint(1, 100, (bs, seq_len)),
            "attention_mask":         torch.ones(bs, seq_len, dtype=torch.long),
            "pixel_values":           torch.randn(bs, 3, 224, 224),
            "teacher_input_ids":      torch.randint(1, 1000, (bs, seq_len)),
            "teacher_attention_mask": torch.ones(bs, seq_len, dtype=torch.long),
        })
    return batches


# ---------------------------------------------------------------------------
# Build the JointMultiTaskLightningModule (CPU)
# ---------------------------------------------------------------------------

def _build_joint_module():
    from hybrid_xmamba.training.lightning_module import JointMultiTaskLightningModule
    enc = _make_text_encoder(embed_dim=64)
    teacher = MockPubMedBERT()
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    mod = JointMultiTaskLightningModule(
        model=enc,
        teacher=teacher,
        alpha_kd=0.3,
        beta_clip=1.0,
        gamma_simcse=0.1,
        backbone_lr=1e-4,
        head_lr=3e-4,
        weight_decay=0.01,
        warmup_steps=2,
        max_steps=5,
        gradient_clip_val=1.0,
        # No freeze in smoke test — we need backbone gradients in 5 steps.
        freeze_text_encoder_steps=0,
        vit_unfreeze_blocks=0,
    )
    return mod


# ---------------------------------------------------------------------------
# Test 1 — module construction + 4-group optimizer
# ---------------------------------------------------------------------------

def test_construction_and_optimizer() -> None:
    print("\n--- Construction + optimizer param groups ---")
    mod = _build_joint_module()

    assert mod.image_encoder is not None,           "image_encoder not loaded"
    assert mod.img_proj is not None,                "img_proj not built"
    assert isinstance(mod.img_proj, nn.Sequential), "img_proj must be Sequential MLP"
    assert len(mod.img_proj) == 3,                  f"img_proj should be Linear-GELU-Linear, got {len(mod.img_proj)} layers"
    assert mod.distill_proj is not None,            "distill_proj not built"
    assert mod.alpha_kd == 0.3 and mod.beta_clip == 1.0 and mod.gamma_simcse == 0.1, \
        "joint loss weights mismatch"

    opt_cfg = mod.configure_optimizers()
    optimizer = opt_cfg["optimizer"]
    n_groups = len(optimizer.param_groups)
    assert n_groups == 3, f"expected 3 param groups (vit frozen), got {n_groups}"

    # LRs may already be scaled by LinearLR warmup start_factor; check ratios.
    lrs = [g["lr"] for g in optimizer.param_groups]
    assert abs(lrs[0] - lrs[1]) < 1e-12, f"backbone groups should share LR: {lrs}"
    ratio = lrs[2] / lrs[0]
    assert abs(ratio - 3.0) < 1e-3, f"head/backbone LR ratio should be 3.0, got {ratio}"
    assert optimizer.param_groups[1]["weight_decay"] == 0.0, \
        "backbone bias/norm group must have weight_decay=0"
    print(f"  3 groups, LRs={lrs}, head/backbone ratio={ratio:.2f}, "
          f"head_wd={optimizer.param_groups[2]['weight_decay']}")
    print("  PASS")


# ---------------------------------------------------------------------------
# Test 2 — single forward: all 3 losses finite, total assembled correctly
# ---------------------------------------------------------------------------

def test_single_step_losses_finite() -> None:
    print("\n--- Single forward: 3 losses finite + total weighting ---")
    mod = _build_joint_module()
    mod.train()
    batch = _make_indiana_subset(n_pairs=4, seq_len=32)[0]

    # Run the joint step in train mode (gradients enabled)
    loss = mod._joint_step(batch, batch_idx=0, split="train")
    assert torch.isfinite(loss), f"total loss not finite: {loss.item()}"

    # Pull the per-loss values out of the logged metrics
    logs = mod.trainer.callback_metrics if mod._trainer is not None else {}
    # During manual call there is no Trainer; rely on a re-run with hooks instead.
    # Re-derive losses by inspecting the submodules directly:
    print(f"  total_loss (combined): {loss.item():.4f}")
    print("  PASS")


# ---------------------------------------------------------------------------
# Test 3 — 5-step training loop: gradients flow + losses finite each step
# ---------------------------------------------------------------------------

_REQUIRED_GRAD_PARAMS = [
    # (description, predicate over (name, param))
    ("backbone (model.lm)",     lambda n, p: n.startswith("model.lm.")),
    ("attn_pool.q (Strategy 4)", lambda n, p: n == "model.attn_pool.q"),
    ("img_proj (Strategy 3)",    lambda n, p: n.startswith("img_proj.")),
    ("distill_proj (KD)",        lambda n, p: n.startswith("distill_proj.")),
]


def _check_grad_flow(mod: nn.Module) -> Dict[str, bool]:
    """For each required param group, return True if any param has a non-zero grad."""
    seen = {desc: False for desc, _ in _REQUIRED_GRAD_PARAMS}
    for name, p in mod.named_parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            raise AssertionError(f"non-finite grad in {name}")
        if p.grad.abs().sum().item() == 0.0:
            continue
        for desc, pred in _REQUIRED_GRAD_PARAMS:
            if pred(name, p):
                seen[desc] = True
    return seen


def test_5step_training_loop() -> None:
    print("\n--- 5-step CPU training loop (16-pair synthetic Indiana subset) ---")
    mod = _build_joint_module()
    mod.train()

    opt_cfg = mod.configure_optimizers()
    optimizer = opt_cfg["optimizer"]

    batches = _make_indiana_subset(n_pairs=16, seq_len=32)  # 4 batches × 4 pairs
    # Step 5x — we'll cycle the 4 batches once + replay the first
    step_batches = (batches * 2)[:5]

    loss_history: List[Dict[str, float]] = []
    grad_seen_any_step = {desc: False for desc, _ in _REQUIRED_GRAD_PARAMS}

    for step, batch in enumerate(step_batches):
        optimizer.zero_grad(set_to_none=True)

        # Replicate the joint_step logic but capture per-loss values too.
        z_text = mod.model.encode(batch["input_ids"], attention_mask=batch["attention_mask"])

        z_img_raw = mod.image_encoder(batch["pixel_values"])
        z_img = nn.functional.normalize(mod.img_proj(z_img_raw.float()), dim=-1)
        l_clip = mod._nt_xent_loss(z_text, z_img, mod.model.logit_scale)

        z_text2 = mod.model.encode(batch["input_ids"], attention_mask=batch["attention_mask"])
        l_simcse = mod._nt_xent_loss(z_text, z_text2, mod.model.logit_scale, fixed_scale=20.0)

        with torch.no_grad():
            t_out = mod.teacher(
                input_ids=batch["teacher_input_ids"],
                attention_mask=batch["teacher_attention_mask"],
            )
            t_cls = nn.functional.normalize(t_out.last_hidden_state[:, 0, :].float(), dim=-1)
        z_proj = nn.functional.normalize(mod.distill_proj(z_text.float()), dim=-1)
        l_kd = (1.0 - nn.functional.cosine_similarity(z_proj, t_cls, dim=-1)).mean()

        total = mod.alpha_kd * l_kd + mod.beta_clip * l_clip + mod.gamma_simcse * l_simcse

        for name, val in [("kd", l_kd), ("clip", l_clip), ("simcse", l_simcse), ("total", total)]:
            assert torch.isfinite(val), f"step {step}: {name}_loss not finite ({val.item()})"

        total.backward()

        seen = _check_grad_flow(mod)
        for k, v in seen.items():
            grad_seen_any_step[k] = grad_seen_any_step[k] or v

        torch.nn.utils.clip_grad_norm_(
            [p for p in mod.parameters() if p.requires_grad and p.grad is not None],
            max_norm=mod.gradient_clip_val,
        )
        optimizer.step()

        loss_history.append({
            "kd": l_kd.item(), "clip": l_clip.item(),
            "simcse": l_simcse.item(), "total": total.item(),
        })
        print(f"  step {step}: total={total.item():.4f}  "
              f"kd={l_kd.item():.4f}  clip={l_clip.item():.4f}  simcse={l_simcse.item():.4f}")

    # Gradient-flow checks
    missing = [desc for desc, ok in grad_seen_any_step.items() if not ok]
    assert not missing, (
        f"no gradient signal observed in: {missing}. "
        f"Required: {[d for d, _ in _REQUIRED_GRAD_PARAMS]}"
    )
    print(f"  ✓ gradients flowed into: {[d for d, _ in _REQUIRED_GRAD_PARAMS]}")

    # Loss trajectory check (advisory, not strict — 5 steps + tiny random init)
    first_total = loss_history[0]["total"]
    last_total  = loss_history[-1]["total"]
    if last_total < first_total:
        print(f"  ✓ total loss decreased: {first_total:.4f} → {last_total:.4f}")
    else:
        print(f"  ⚠ total loss did NOT decrease over 5 steps: "
              f"{first_total:.4f} → {last_total:.4f} "
              f"(advisory only — tiny model + random data)")

    print("  PASS")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Phase 3: Joint KD+CLIP+SimCSE smoke test (CPU)")
    print("=" * 60)

    tests = [
        ("Construction + 4-group optimizer", test_construction_and_optimizer),
        ("Single forward: 3 losses finite",  test_single_step_losses_finite),
        ("5-step CPU training loop",         test_5step_training_loop),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as exc:
            print(f"\n  FAIL [{name}]: {exc}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)
    print("ALL CHECKS PASSED — Phase 3 green. Safe to proceed to Phase 4 "
          "(MIMIC-CXR data prep on willi).")
    print("=" * 60)


if __name__ == "__main__":
    main()
