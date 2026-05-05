"""Phase 3 smoke test — JointMultiTaskLightningModule on CPU.

Validates the joint KD + CLIP + SimCSE pipeline end-to-end before the willi
A100 sbatch (Phase 5):

  - All 3 losses (kd, clip, simcse) finite every step
  - Total loss = α·kd + β·clip + γ·simcse computes without NaN
  - Gradients flow into:
      * backbone (model.lm)
      * attn_pool.q              (Strategy 4 — attention pooling)
      * projection_head          (encoder output projection)
  - 3 optimizer param groups configured (backbone_wd / backbone_no_wd / head)
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

# Phase 8: img_proj/distill_proj deleted. clip_model.visual output dim MUST equal
# student embed_dim (both are 512 in real training). In smoke test, student is 64-d,
# so mock image encoder must also output 64-d; KD t_emb must be 64-d for direct
# cosine similarity.
_STUDENT_EMBED_DIM = 64
_MOCK_IMG_DIM  = _STUDENT_EMBED_DIM  # must == student embed_dim (Phase 8 assert)
_MOCK_TEXT_DIM = _STUDENT_EMBED_DIM  # KD: cos(z_text, t_emb) requires matching dims


class _MockVisual(nn.Module):
    """Stands in for BiomedCLIP visual trunk. (B,3,224,224) -> (B, _MOCK_IMG_DIM)."""

    def __init__(self, out_dim: int = _MOCK_IMG_DIM) -> None:
        super().__init__()
        self.proj = nn.Linear(3 * 8 * 8, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B,3,H,W) -> down-pool to 8x8 -> flatten -> linear
        pooled = nn.functional.adaptive_avg_pool2d(x, (8, 8))
        return self.proj(pooled.flatten(1))


class _MockClipModel(nn.Module):
    """Mimic the open_clip CLIP wrapper.

    Exposes:
      * .visual (for image encoding via HybridContrastiveLightningModule)
      * .encode_text(input_ids) -> (B, _MOCK_TEXT_DIM)  ← Phase 2 teacher API
    Phase 8: embed_dim must equal student embed_dim so the Phase 8 assert
    (img_out == model.embed_dim) passes.
    """

    def __init__(self) -> None:
        super().__init__()
        self.embed_dim = _MOCK_IMG_DIM  # must == _STUDENT_EMBED_DIM
        self.visual = _MockVisual(_MOCK_IMG_DIM)
        self._text_proj = nn.Linear(8, _MOCK_TEXT_DIM)  # dummy trainable param

    def encode_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        B = input_ids.shape[0]
        return torch.randn(B, _MOCK_TEXT_DIM)


def _mock_get_tokenizer(name: str) -> Any:
    """open_clip tokenizer stub: Callable[[List[str]], LongTensor (1, 32)]."""
    def _tok(texts):
        return torch.randint(1, 100, (len(texts), 32))
    return _tok


def _install_mock_open_clip() -> None:
    mock = types.ModuleType("open_clip")

    def create_model_from_pretrained(name: str, **kwargs: Any):
        return _MockClipModel(), None

    mock.create_model_from_pretrained = create_model_from_pretrained  # type: ignore[attr-defined]
    mock.get_tokenizer = _mock_get_tokenizer  # type: ignore[attr-defined]
    sys.modules["open_clip"] = mock


_install_mock_open_clip()


# ---------------------------------------------------------------------------
# Mock BiomedCLIP text teacher (Phase 2 pivot: encode_text API, 512-d)
# ---------------------------------------------------------------------------

class MockBiomedCLIPText(nn.Module):
    """Frozen mock teacher mirroring the open_clip CLIP wrapper.

    Returns (_MOCK_TEXT_DIM,) embeddings via encode_text — same as the real
    BiomedCLIP text tower. Phase 8: KD is direct cosine sim on z_text, so
    t_emb dim must equal student embed_dim (_MOCK_TEXT_DIM == _STUDENT_EMBED_DIM).
    """

    def __init__(self) -> None:
        super().__init__()
        # Dummy param so frozen-teacher grad-flow assertion can be made
        self._dummy = nn.Embedding(10, _MOCK_TEXT_DIM)

    def encode_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        B = input_ids.shape[0]
        return torch.randn(B, _MOCK_TEXT_DIM)


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


def _make_text_encoder(embed_dim: int = _STUDENT_EMBED_DIM):
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

def _build_joint_module(freeze_text_encoder_steps: int = 0, moco_queue_size: int = 0):
    from hybrid_xmamba.training.lightning_module import JointMultiTaskLightningModule
    enc = _make_text_encoder(embed_dim=64)
    teacher = MockBiomedCLIPText()
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    kwargs: Dict[str, Any] = dict(
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
        freeze_text_encoder_steps=freeze_text_encoder_steps,
        vit_unfreeze_blocks=0,
    )
    if moco_queue_size > 0:
        # MoCoQueue dim is hardcoded to 512 in JointMultiTaskLightningModule;
        # override the queue post-construction with the smoke-test embed dim.
        from hybrid_xmamba.training.moco_queue import MoCoQueue, MomentumEncoder
        kwargs["moco_queue_size"] = moco_queue_size

    mod = JointMultiTaskLightningModule(**kwargs)
    if moco_queue_size > 0:
        # Replace 512-d queue with embed_dim queue for the tiny student.
        from hybrid_xmamba.training.moco_queue import MoCoQueue, MomentumEncoder
        mod.text_queue = MoCoQueue(dim=_STUDENT_EMBED_DIM, K=moco_queue_size)
        mod.momentum_encoder = MomentumEncoder(enc, m=0.999)
    # Phase 8 sanity: dead modules must not exist
    assert not hasattr(mod, "distill_proj") or mod.distill_proj is None or (
        not isinstance(mod.distill_proj, nn.Module)
    ), "distill_proj must be deleted from JointMultiTaskLightningModule (Phase 8)"
    return mod


# ---------------------------------------------------------------------------
# Test 1 — module construction + 4-group optimizer
# ---------------------------------------------------------------------------

def test_construction_and_optimizer() -> None:
    print("\n--- Construction + optimizer param groups ---")
    mod = _build_joint_module()

    assert mod.image_encoder is not None, "image_encoder not loaded"
    # Phase 8: dead modules must be absent
    assert not hasattr(mod, "img_proj") or not isinstance(
        getattr(mod, "img_proj", None), nn.Module
    ), "img_proj must be deleted (Phase 8)"
    assert not hasattr(mod, "distill_proj") or not isinstance(
        getattr(mod, "distill_proj", None), nn.Module
    ), "distill_proj must be deleted (Phase 8)"
    assert mod.alpha_kd == 0.3 and mod.beta_clip == 1.0 and mod.gamma_simcse == 0.1, \
        "joint loss weights mismatch"

    opt_cfg = mod.configure_optimizers()
    optimizer = opt_cfg["optimizer"]
    n_groups = len(optimizer.param_groups)
    assert n_groups == 3, f"expected 3 param groups (vit frozen), got {n_groups}"

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
    print(f"  total_loss (combined): {loss.item():.4f}")

    # Verify teacher params received NO gradient (frozen)
    loss.backward()
    for name, p in mod.teacher.named_parameters():
        assert p.grad is None or p.grad.abs().sum().item() == 0.0, (
            f"teacher param '{name}' has gradient — teacher must be frozen"
        )
    print("  ✓ teacher params have no gradient (frozen)")
    print("  PASS")


# ---------------------------------------------------------------------------
# Test 3 — 5-step training loop: gradients flow + losses finite each step
# ---------------------------------------------------------------------------

# Phase 8: img_proj and distill_proj deleted — check backbone + projection_head only.
_REQUIRED_GRAD_PARAMS = [
    # (description, predicate over (name, param))
    ("backbone (model.lm)",          lambda n, p: n.startswith("model.lm.")),
    ("attn_pool.q (Strategy 4)",     lambda n, p: n == "model.attn_pool.q"),
    ("projection_head",              lambda n, p: n.startswith("model.projection_head.")),
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

        # Phase 8: img_proj bypassed — clip_model.visual already in joint space.
        z_img_raw = mod.image_encoder(batch["pixel_values"])
        z_img = nn.functional.normalize(z_img_raw.float(), dim=-1)
        l_clip = mod._nt_xent_loss(z_text, z_img, mod.model.logit_scale)

        z_text2 = mod.model.encode(batch["input_ids"], attention_mask=batch["attention_mask"])
        l_simcse = mod._nt_xent_loss(z_text, z_text2, mod.model.logit_scale, fixed_scale=20.0)

        # Phase 8: KD direct on z_text — no distill_proj.
        with torch.no_grad():
            t_emb = mod.teacher.encode_text(batch["teacher_input_ids"])  # (B, embed_dim)
            t_emb = nn.functional.normalize(t_emb.float(), dim=-1)
        l_kd = (1.0 - nn.functional.cosine_similarity(z_text.float(), t_emb, dim=-1)).mean()

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
    print(f"  ✓ gradients flowed into: {[d for d, _ in _REQUIRED_GRAD_PARAMS]} "
          f"(img_proj/distill_proj deleted — Phase 8)")

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
# Test 4 — Phase 11 CLIP gating: l_clip == 0 during warmup, non-zero after;
# MoCo queue stays empty during warmup, enqueues after unfreeze.
# ---------------------------------------------------------------------------

def test_phase11_clip_gating() -> None:
    print("\n--- Phase 11 gating: l_clip / queue gated by freeze_text_encoder_steps ---")
    # Without a Lightning trainer attached, mod.global_step == 0. So:
    #   freeze=10 → global_step(0) < 10  → CLIP gated OFF, queue untouched.
    #   freeze=0  → global_step(0) >= 0  → CLIP active, queue enqueues.
    batch = _make_indiana_subset(n_pairs=4, seq_len=32)[0]

    # --- warmup case ---
    mod_warm = _build_joint_module(freeze_text_encoder_steps=10, moco_queue_size=32)
    mod_warm.train()
    assert mod_warm.text_queue is not None, "text_queue must be initialised"
    ptr_before = int(mod_warm.text_queue.queue_ptr)
    queue_before = mod_warm.text_queue.queue.clone()

    # Capture clip_loss via the logger; simpler to recompute total and inspect via
    # rerunning with no pixel_values to compare. Instead read through _joint_step
    # and inspect the logged clip_loss by intercepting self.log.
    logged: Dict[str, float] = {}

    def _capture(self, name, value, *args, **kwargs):  # type: ignore[no-untyped-def]
        try:
            logged[name] = float(value.detach().item() if hasattr(value, "detach") else value)
        except Exception:
            pass

    import types as _types
    mod_warm.log = _types.MethodType(_capture, mod_warm)
    _ = mod_warm._joint_step(batch, batch_idx=0, split="train")
    ptr_after = int(mod_warm.text_queue.queue_ptr)

    clip_warm = logged.get("train/clip_loss", None)
    assert clip_warm is not None, "train/clip_loss not logged"
    assert clip_warm == 0.0, f"CLIP loss must be 0 during warmup, got {clip_warm}"
    assert ptr_after == ptr_before, (
        f"text_queue ptr must be untouched during warmup; {ptr_before}→{ptr_after}"
    )
    assert torch.equal(mod_warm.text_queue.queue, queue_before), \
        "text_queue contents must be untouched during warmup"
    print(f"  ✓ warmup: clip_loss={clip_warm:.4f}, queue_ptr unchanged ({ptr_before})")

    # --- post-warmup case ---
    mod_post = _build_joint_module(freeze_text_encoder_steps=0, moco_queue_size=32)
    mod_post.train()
    ptr_before2 = int(mod_post.text_queue.queue_ptr)
    logged.clear()
    mod_post.log = _types.MethodType(_capture, mod_post)
    _ = mod_post._joint_step(batch, batch_idx=0, split="train")
    ptr_after2 = int(mod_post.text_queue.queue_ptr)

    clip_post = logged.get("train/clip_loss", None)
    assert clip_post is not None and clip_post > 0.0, \
        f"CLIP loss must be > 0 post-warmup, got {clip_post}"
    assert ptr_after2 > ptr_before2, (
        f"text_queue must enqueue post-warmup; ptr {ptr_before2}→{ptr_after2}"
    )
    print(f"  ✓ post-warmup: clip_loss={clip_post:.4f}, queue_ptr "
          f"{ptr_before2}→{ptr_after2}")
    print("  PASS")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Phase 11: Joint KD+CLIP+SimCSE smoke test (CPU, gated CLIP)")
    print("=" * 60)

    tests = [
        ("Construction + 4-group optimizer", test_construction_and_optimizer),
        ("Single forward: 3 losses finite",  test_single_step_losses_finite),
        ("5-step CPU training loop",         test_5step_training_loop),
        ("Phase 11 CLIP gating + queue",     test_phase11_clip_gating),
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
    print("ALL CHECKS PASSED — Phase 8 architecture (no img_proj/distill_proj) green.")
    print("=" * 60)


if __name__ == "__main__":
    main()
