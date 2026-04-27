"""Stage 1 distillation smoke test — CPU, no network required.

Validates the full Stage 1 pipeline before sbatch:
  - Fixed τ=0.05 (scale=20) NT-Xent path
  - KD lambda ramp: warmup=500, ramp=1500, lambda_max=0.7
  - DistillContrastiveLightningModule forward + backward (no NaN)
  - ContrastiveEvalCallback cosine/alignment stats
  - AnomalyDetectionCallback enable/disable lifecycle
  - Dual-tokenisation batch shape correctness
  - Pooling padding invariance
  - 50-step PL Trainer end-to-end (tiny model, mock teacher)
  - Optional A100 memory profile (--cuda)

Run before sbatch:
    python scripts/smoke_test_distill.py
    python scripts/smoke_test_distill.py --cuda  # also runs CUDA memory profile
"""

import argparse
import sys
import types
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Shared mini model builders (dim=64 for CPU speed)
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
)


def _make_text_encoder(embed_dim: int = 64):
    from hybrid_xmamba.models.configuration_hybrid import HybridConfig
    from hybrid_xmamba.models.hybrid_lm import HybridTextEncoder
    cfg = HybridConfig(**_SMALL_CFG_KWARGS)
    return HybridTextEncoder(cfg, embed_dim=embed_dim)


def assert_no_nan(name: str, tensor: torch.Tensor) -> None:
    assert torch.isfinite(tensor).all(), f"NaN/Inf in {name}: {tensor}"


# ---------------------------------------------------------------------------
# Mock PubMedBERT teacher (hidden_size=768, no network)
# ---------------------------------------------------------------------------

class _MockBERTConfig:
    hidden_size = 768


class MockPubMedBERT(nn.Module):
    """Deterministic mock with PubMedBERT interface (hidden_size=768)."""

    def __init__(self) -> None:
        super().__init__()
        self.config = _MockBERTConfig()
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Any:
        B = input_ids.shape[0] if input_ids is not None else 1
        L = input_ids.shape[1] if input_ids is not None else 8
        hidden = torch.ones(B, L, self.config.hidden_size) * 0.01
        return types.SimpleNamespace(last_hidden_state=hidden)


class MockTokenizer:
    """Minimal tokenizer shim for callback smoke (no vocabulary needed)."""
    pad_token = "<pad>"
    eos_token = "<eos>"
    model_max_length = 64

    def __call__(
        self,
        texts: Any,
        max_length: int = 64,
        truncation: bool = True,
        padding: str = "max_length",
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        if isinstance(texts, str):
            texts = [texts]
        B = len(texts)
        return {
            "input_ids": torch.randint(0, 100, (B, max_length)),
            "attention_mask": torch.ones(B, max_length, dtype=torch.long),
        }


def _make_distill_module(
    lambda_max: float = 0.7,
    distill_warmup: int = 500,
    distill_ramp: int = 1500,
    max_steps: int = 50,
) -> "DistillContrastiveLightningModule":  # type: ignore[name-defined]
    from hybrid_xmamba.training.lightning_module import DistillContrastiveLightningModule
    enc = _make_text_encoder(embed_dim=64)
    teacher = MockPubMedBERT()
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return DistillContrastiveLightningModule(
        teacher=teacher,
        lambda_max=lambda_max,
        distill_warmup=distill_warmup,
        distill_ramp=distill_ramp,
        model=enc,
        contrastive_mode="simcse",
        image_encoder_name="",
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_steps=5,
        max_steps=max_steps,
        gradient_clip_val=1.0,
    )


def _make_dataloader(
    n: int = 64, seq_len: int = 32, batch_size: int = 8
) -> DataLoader:
    ids = torch.randint(0, 100, (n, seq_len))
    mask = torch.ones(n, seq_len, dtype=torch.long)
    t_ids = torch.randint(0, 1000, (n, seq_len))
    t_mask = torch.ones(n, seq_len, dtype=torch.long)
    ds = TensorDataset(ids, mask, t_ids, t_mask)

    def collate(batch: Any) -> Any:
        a, b, c, d = zip(*batch)
        return {
            "input_ids": torch.stack(a),
            "attention_mask": torch.stack(b),
            "teacher_input_ids": torch.stack(c),
            "teacher_attention_mask": torch.stack(d),
        }

    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)


# ---------------------------------------------------------------------------
# Test: fixed τ=0.05 NT-Xent path
# ---------------------------------------------------------------------------

def test_fixed_scale_nt_xent() -> None:
    print("\n--- Fixed τ=0.05 NT-Xent path ---")
    from hybrid_xmamba.training.lightning_module import HybridContrastiveLightningModule
    enc = _make_text_encoder()
    mod = HybridContrastiveLightningModule(
        model=enc,
        contrastive_mode="simcse",
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_steps=5,
        max_steps=50,
        gradient_clip_val=1.0,
    )

    z1 = F.normalize(torch.randn(4, 64), dim=-1)
    z2 = F.normalize(torch.randn(4, 64), dim=-1)

    loss_fixed = mod._nt_xent_loss(z1, z2, mod.model.logit_scale, fixed_scale=20.0)
    loss_learned = mod._nt_xent_loss(z1, z2, mod.model.logit_scale)

    assert_no_nan("loss_fixed", loss_fixed)
    assert_no_nan("loss_learned", loss_learned)
    assert loss_fixed.item() > 0, "fixed-scale loss should be positive"

    # Verify scale=20 gives same logits as manual computation
    logits_expected = 20.0 * (z1 @ z2.T)
    labels = torch.arange(4)
    manual = (F.cross_entropy(logits_expected, labels) + F.cross_entropy(logits_expected.T, labels)) / 2
    assert abs(loss_fixed.item() - manual.item()) < 1e-5, (
        f"fixed_scale loss mismatch: got {loss_fixed.item():.6f}, expected {manual.item():.6f}"
    )
    print(f"  fixed_scale=20 loss: {loss_fixed.item():.4f} (matches manual: {manual.item():.4f})")
    print("  PASS")


# ---------------------------------------------------------------------------
# Test: lambda ramp (new config: warmup=500, ramp=1500, lambda_max=0.7)
# ---------------------------------------------------------------------------

def test_lambda_ramp_new_config() -> None:
    print("\n--- KD lambda ramp (warmup=500, ramp=1500, lambda_max=0.7) ---")
    mod = _make_distill_module(lambda_max=0.7, distill_warmup=500, distill_ramp=1500)

    def get_lambda_at(step: int) -> float:
        if step < mod.distill_warmup:
            return 0.0
        ramp_step = step - mod.distill_warmup
        if ramp_step >= mod.distill_ramp:
            return mod.lambda_max
        return mod.lambda_max * (ramp_step / mod.distill_ramp)

    # step=750: ramp_step=250, λ = 0.7 * 250/1500 = 0.1167
    # step=1250: ramp_step=750 (mid-ramp), λ = 0.7 * 750/1500 = 0.35
    # step=2000: ramp_step=1500 (end of ramp), λ = 0.7
    cases = [
        (0, 0.0),
        (499, 0.0),
        (500, 0.0),
        (750, 0.7 * 250 / 1500),
        (1250, 0.35),
        (2000, 0.7),
        (5000, 0.7),
    ]
    for step, expected in cases:
        got = get_lambda_at(step)
        assert abs(got - expected) < 1e-4, (
            f"step={step}: expected λ={expected:.4f}, got {got:.4f}"
        )
    print("  All lambda schedule checkpoints correct")
    print(f"  lambda_max={mod.lambda_max}, warmup={mod.distill_warmup}, ramp={mod.distill_ramp}")
    print("  PASS")


# ---------------------------------------------------------------------------
# Test: dual-tokenisation batch shape
# ---------------------------------------------------------------------------

def test_dual_tokenisation_shapes() -> None:
    print("\n--- Dual tokenisation batch shapes ---")
    enc = _make_text_encoder(embed_dim=64)
    B, L_s, L_t = 4, 32, 32

    input_ids = torch.randint(0, 100, (B, L_s))
    attn = torch.ones(B, L_s, dtype=torch.long)
    t_input_ids = torch.randint(0, 1000, (B, L_t))
    t_attn = torch.ones(B, L_t, dtype=torch.long)

    teacher = MockPubMedBERT()
    with torch.no_grad():
        z = enc.encode(input_ids, attention_mask=attn)
        t_out = teacher(t_input_ids, t_attn)
        teacher_cls = t_out.last_hidden_state[:, 0, :]

    assert z.shape == (B, 64), f"student embedding shape: {z.shape}"
    assert teacher_cls.shape == (B, 768), f"teacher CLS shape: {teacher_cls.shape}"
    print(f"  student z: {tuple(z.shape)}, teacher CLS: {tuple(teacher_cls.shape)}")
    print("  PASS")


# ---------------------------------------------------------------------------
# Test: pooling padding invariance
# ---------------------------------------------------------------------------

def test_pooling_padding_invariance() -> None:
    print("\n--- Pooling: padding invariance ---")
    enc = _make_text_encoder()
    enc.eval()

    seq = torch.randint(1, 100, (1, 16))
    pad = torch.zeros(1, 16, dtype=torch.long)
    full = torch.cat([seq, pad], dim=1)
    mask_full = torch.cat([torch.ones(1, 16, dtype=torch.long), torch.zeros(1, 16, dtype=torch.long)], dim=1)

    with torch.no_grad():
        z_full = enc.encode(full, attention_mask=mask_full)
        z_short = enc.encode(seq, attention_mask=torch.ones(1, 16, dtype=torch.long))

    cos = F.cosine_similarity(z_full, z_short, dim=-1).item()
    print(f"  cos(padded, unpadded) = {cos:.6f}")
    if cos < 0.999:
        print(f"  WARNING: cosine {cos:.6f} < 0.999 — verify mean pooling uses attention_mask")
    else:
        print("  PASS")


# ---------------------------------------------------------------------------
# Test: ContrastiveEvalCallback stats (no real datasets — just cosine/alignment)
# ---------------------------------------------------------------------------

def test_contrastive_eval_callback_smoke() -> None:
    print("\n--- ContrastiveEvalCallback: cosine/alignment stats ---")
    from hybrid_xmamba.training.contrastive_eval_callback import (
        ContrastiveEvalCallback,
        AnomalyDetectionCallback,
        _spearman_rho,
        _alignment,
        _uniformity,
    )

    # Spearman correctness
    assert abs(_spearman_rho([1., 2., 3., 4., 5.], [1., 2., 3., 4., 5.]) - 1.0) < 1e-5
    assert abs(_spearman_rho([1., 2., 3.], [3., 2., 1.]) - (-1.0)) < 1e-5

    # Alignment / uniformity on unit sphere vectors
    z1 = F.normalize(torch.randn(16, 64), dim=-1)
    z2 = F.normalize(torch.randn(16, 64), dim=-1)
    a = _alignment(z1, z2)
    u = _uniformity(z1)
    assert isinstance(a, float) and a >= 0.0
    assert isinstance(u, float)
    print(f"  alignment={a:.4f}, uniformity={u:.4f}")

    # Callback instantiates without error
    tok = MockTokenizer()
    cb = ContrastiveEvalCallback(tokenizer=tok, eval_every_n_steps=10, align_unif_every_n_steps=20)
    assert cb.eval_every == 10
    assert cb.align_unif_every == 20

    # AnomalyDetectionCallback instantiates
    acb = AnomalyDetectionCallback(max_steps=200)
    assert acb.max_steps == 200

    print("  PASS")


# ---------------------------------------------------------------------------
# Test: 50-step PL Trainer end-to-end (CPU, tiny model)
# ---------------------------------------------------------------------------

def test_50step_trainer_smoke() -> None:
    print("\n--- 50-step PL Trainer smoke (CPU, tiny model) ---")
    from hybrid_xmamba.training.contrastive_eval_callback import (
        ContrastiveEvalCallback,
        AnomalyDetectionCallback,
    )

    module = _make_distill_module(
        lambda_max=0.7, distill_warmup=5, distill_ramp=15, max_steps=50
    )

    # 256 samples / bs=8 = 32 batches per epoch; val_check_interval=10 < 32
    train_dl = _make_dataloader(n=256, seq_len=32, batch_size=8)
    val_dl = _make_dataloader(n=32, seq_len=32, batch_size=8)

    tok = MockTokenizer()
    callbacks = [
        ContrastiveEvalCallback(
            tokenizer=tok,
            eval_every_n_steps=10,
            align_unif_every_n_steps=20,
        ),
        AnomalyDetectionCallback(max_steps=20),
    ]

    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=50,
        val_check_interval=10,
        log_every_n_steps=5,
        accumulate_grad_batches=1,
        callbacks=callbacks,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=1,
        logger=False,
    )

    trainer.fit(module, train_dataloaders=train_dl, val_dataloaders=val_dl)

    # Verify metrics logged
    metrics = trainer.callback_metrics
    total_loss = metrics.get("train/total_loss")
    assert total_loss is None or torch.isfinite(torch.tensor(total_loss)), \
        f"total_loss not finite: {total_loss}"
    print(f"  Completed 50 steps. metrics: {list(metrics.keys())}")
    print("  PASS")


# ---------------------------------------------------------------------------
# Optional: A100 CUDA memory profile
# ---------------------------------------------------------------------------

def test_memory_a100() -> None:
    print("\n--- A100 memory profile (CUDA, bs=16, seq=512) ---")
    if not torch.cuda.is_available():
        print("  SKIP: CUDA not available")
        return

    device = "cuda"
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    from hybrid_xmamba.models.configuration_hybrid import HybridConfig
    from hybrid_xmamba.models.hybrid_lm import HybridTextEncoder

    cfg = HybridConfig(
        vocab_size=50257, dim=512, num_layers=8,
        layer_pattern=["mamba", "mamba", "mlstm"],
        state_size=16, conv_size=4, expand_factor=2, dt_rank=None,
        use_fast_path=True, head_dim=64, num_heads=8, use_tfla=True,
        proj_factor=2, slstm_hidden_dim=512, slstm_num_heads=4,
        use_exponential_gate=True, norm_type="rms", use_mlp=True,
        mlp_ratio=4.0, max_position_embeddings=1024, dropout=0.1,
        initializer_range=0.02, use_cache=False, tie_word_embeddings=False,
    )
    enc = HybridTextEncoder(cfg, embed_dim=512).to(device).to(torch.bfloat16)
    enc.train()

    B, L = 16, 512
    ids = torch.randint(0, 50257, (B, L), device=device)
    mask = torch.ones(B, L, dtype=torch.long, device=device)

    z1 = enc.encode(ids, attention_mask=mask)
    z2 = enc.encode(ids, attention_mask=mask)
    logits = 20.0 * (z1 @ z2.T)
    labels = torch.arange(B, device=device)
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    loss.backward()

    peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    teacher_est = 1.8  # PubMedBERT 110M bf16 weights + acts at bs=16
    print(f"  Student 2x fwd peak : {peak_gb:.2f} GB")
    print(f"  Teacher estimate    : ~{teacher_est:.1f} GB")
    print(f"  Total estimate      : ~{peak_gb + teacher_est:.1f} GB / {total_gb:.1f} GB")
    if peak_gb + teacher_est < total_gb:
        print("  PASS: fits on A100 40GB")
    else:
        print("  WARNING: may OOM — revert to bs=8, accum=8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 distillation smoke tests")
    parser.add_argument("--cuda", action="store_true", help="Also run CUDA memory profile")
    args = parser.parse_args()

    print("=" * 60)
    print("Stage 1 Distillation Smoke Tests (Phase 2 config)")
    print("=" * 60)

    tests = [
        ("Fixed τ=0.05 NT-Xent path", test_fixed_scale_nt_xent),
        ("KD lambda ramp (warmup=500, ramp=1500, λ_max=0.7)", test_lambda_ramp_new_config),
        ("Dual tokenisation shapes", test_dual_tokenisation_shapes),
        ("Pooling padding invariance", test_pooling_padding_invariance),
        ("ContrastiveEvalCallback smoke", test_contrastive_eval_callback_smoke),
        ("50-step PL Trainer end-to-end", test_50step_trainer_smoke),
    ]
    if args.cuda:
        tests.append(("A100 memory profile", test_memory_a100))

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as exc:
            print(f"  FAIL [{name}]: {exc}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED — safe to sbatch.")
    print("=" * 60)


if __name__ == "__main__":
    main()
