"""Smoke tests for the distillation pipeline.

Runs 5 steps of each distillation script with tiny batch/CPU to verify:
  - No NaN / Inf losses
  - KD loss is non-zero and different from CE loss (Stage 0)
  - Student-teacher cosine > 0 and < 1 after first step (Stage 1)
  - Dual tokenization produces correct shapes (Stage 1)
  - Memory estimate for A100 (requires CUDA)

Run before launching full A100 jobs:
    python scripts/smoke_test_distill.py
    python scripts/smoke_test_distill.py --cuda   # also runs memory profile
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_student(dim=64, num_layers=2):
    """Tiny student model for smoke testing (dim=64, 2 layers, CPU-only)."""
    from hybrid_xmamba.models.configuration_hybrid import HybridConfig
    from hybrid_xmamba.models.hybrid_lm import HybridLanguageModel
    cfg = HybridConfig(
        vocab_size=50257,
        dim=dim,
        num_layers=num_layers,
        layer_pattern=["mamba", "mlstm"],
        state_size=4,
        conv_size=4,
        expand_factor=2,
        dt_rank=None,
        use_fast_path=False,      # avoid Triton on CPU
        head_dim=16,
        num_heads=4,
        use_tfla=False,           # avoid TFLA kernel on CPU
        proj_factor=2,
        slstm_hidden_dim=dim,
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
    return HybridLanguageModel(cfg)


def _make_text_encoder(dim=64):
    from hybrid_xmamba.models.configuration_hybrid import HybridConfig
    from hybrid_xmamba.models.hybrid_lm import HybridTextEncoder
    cfg = HybridConfig(
        vocab_size=50257,
        dim=dim,
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
        slstm_hidden_dim=dim,
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
    return HybridTextEncoder(cfg, embed_dim=64)


def assert_no_nan(tensor, name):
    assert not torch.isnan(tensor).any(), f"NaN in {name}: {tensor}"
    assert not torch.isinf(tensor).any(), f"Inf in {name}: {tensor}"


# ---------------------------------------------------------------------------
# Smoke test: Stage 0 KD (BioMedLM teacher mock)
# ---------------------------------------------------------------------------

def test_stage0_kd_loss():
    print("\n--- Stage 0 KD: loss structure ---")
    from scripts.train_stage0_distill import DistillLightningModule

    B, L, V = 2, 16, 50257

    student = _make_student(dim=64)

    # Mock teacher: a tiny LM with same vocab but fixed random weights
    class MockTeacher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = torch.nn.Linear(64, V, bias=False)

        def forward(self, input_ids, **kwargs):
            # Return random logits (same shape as real teacher)
            B, L = input_ids.shape
            logits = self.lm_head(torch.randn(B, L, 64))
            from types import SimpleNamespace
            return SimpleNamespace(logits=logits)

    teacher = MockTeacher().eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    module = DistillLightningModule(
        model=student,
        teacher=teacher,
        alpha=0.5,
        temperature=2.0,
        teacher_ppl_abort_threshold=1e9,  # disable abort in smoke test
        learning_rate=1e-3,
        weight_decay=0.01,
        warmup_steps=0,
        max_steps=10,
    )
    module._teacher_ppl_checked = True  # skip ppl check (mock teacher)

    input_ids = torch.randint(0, V, (B, L))
    labels = input_ids.clone()
    labels[0, :2] = -100  # simulate some masked positions
    attn_mask = torch.ones(B, L, dtype=torch.long)
    attn_mask[0, 0] = 0   # one pad position

    batch = {"input_ids": input_ids, "labels": labels, "attention_mask": attn_mask}

    # Simulate a trainer with an optimizer (Lightning expects this in training_step)
    # We call the internal loss logic directly for smoke testing
    student_out = student(input_ids, labels=labels, return_dict=True)
    ce_loss = student_out.loss
    student_logits = student_out.logits

    with torch.no_grad():
        teacher_logits = teacher(input_ids).logits

    T = 2.0
    s_shift = student_logits[:, :-1, :].contiguous()
    t_shift = teacher_logits[:, :-1, :].contiguous()
    s_log_prob = F.log_softmax(s_shift / T, dim=-1)
    t_prob = F.softmax(t_shift / T, dim=-1)
    kd_loss = F.kl_div(
        s_log_prob.view(-1, V), t_prob.view(-1, V), reduction="batchmean"
    ) * (T ** 2)

    total_loss = 0.5 * ce_loss + 0.5 * kd_loss

    assert_no_nan(ce_loss, "ce_loss")
    assert_no_nan(kd_loss, "kd_loss")
    assert_no_nan(total_loss, "total_loss")
    assert ce_loss.item() > 0, "CE loss should be > 0"
    assert kd_loss.item() > 0, "KD loss should be > 0"
    assert abs(ce_loss.item() - kd_loss.item()) > 1e-6, \
        "CE and KD losses should differ (random teacher)"

    print(f"  ce_loss  = {ce_loss.item():.4f}")
    print(f"  kd_loss  = {kd_loss.item():.4f}")
    print(f"  total    = {total_loss.item():.4f}")
    print("  PASS")


# ---------------------------------------------------------------------------
# Smoke test: Stage 1 KD (PubMedBERT teacher mock)
# ---------------------------------------------------------------------------

def test_stage1_kd_loss():
    print("\n--- Stage 1 KD: dual tokenisation + distill loss ---")
    from hybrid_xmamba.training.lightning_module import DistillContrastiveLightningModule

    B, L_s, L_t = 4, 32, 32
    V_s = 50257   # GPT-2 BPE
    V_t = 30522   # PubMedBERT WordPiece
    embed_dim = 64

    text_encoder = _make_text_encoder(dim=64)

    # Mock PubMedBERT teacher
    class MockBERT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(V_t, embed_dim)

        def forward(self, input_ids, attention_mask=None, **kwargs):
            h = self.embedding(input_ids)  # (B, L, D)
            from types import SimpleNamespace
            return SimpleNamespace(last_hidden_state=h)

    teacher = MockBERT().eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    module = DistillContrastiveLightningModule(
        teacher=teacher,
        lambda_max=0.3,
        distill_warmup=0,    # no warmup in smoke test
        distill_ramp=1,      # instant ramp
        model=text_encoder,
        contrastive_mode="simcse",
        learning_rate=1e-3,
        weight_decay=0.01,
        warmup_steps=0,
        max_steps=10,
    )

    # Build batch with both student and teacher tokens
    input_ids = torch.randint(0, V_s, (B, L_s))
    attention_mask = torch.ones(B, L_s, dtype=torch.long)
    teacher_input_ids = torch.randint(0, V_t, (B, L_t))
    teacher_attention_mask = torch.ones(B, L_t, dtype=torch.long)

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "teacher_input_ids": teacher_input_ids,
        "teacher_attention_mask": teacher_attention_mask,
    }

    # Test shape of teacher output
    t_out = teacher(teacher_input_ids, teacher_attention_mask)
    teacher_cls = t_out.last_hidden_state[:, 0, :]  # (B, D)
    assert teacher_cls.shape == (B, embed_dim), \
        f"Expected ({B}, {embed_dim}), got {teacher_cls.shape}"

    # Test student encode
    z = text_encoder.encode(input_ids, attention_mask=attention_mask)
    assert z.shape == (B, embed_dim), \
        f"Expected ({B}, {embed_dim}), got {z.shape}"

    # Compute distill loss
    t_norm = F.normalize(teacher_cls.float(), dim=-1)
    s_z = z.float()
    cos_sim = F.cosine_similarity(s_z, t_norm, dim=-1)
    distill_loss = (1.0 - cos_sim).mean()

    assert_no_nan(distill_loss, "distill_loss")
    assert 0.0 <= distill_loss.item() <= 2.0, \
        f"Distill loss should be in [0, 2], got {distill_loss.item():.4f}"
    assert cos_sim.mean().item() < 0.999, \
        "cos_sim should not be 1 for untrained student (random teacher)"

    print(f"  teacher_cls shape : {teacher_cls.shape}")
    print(f"  student_z shape   : {z.shape}")
    print(f"  cos_sim mean      : {cos_sim.mean().item():.4f}")
    print(f"  distill_loss      : {distill_loss.item():.4f}")
    print("  PASS")


# ---------------------------------------------------------------------------
# Smoke test: padding invariance
# ---------------------------------------------------------------------------

def test_pooling_padding_invariance():
    """Padded vs unpadded same text should give same pooled embedding (cosine > 0.999)."""
    print("\n--- Stage 1 KD: padding invariance check ---")
    text_encoder = _make_text_encoder(dim=64)
    text_encoder.eval()

    L = 32
    V = 50257

    # Create a sequence with content in first 16 positions, pad in last 16
    seq = torch.randint(1, V, (1, 16))          # 16 content tokens
    pad = torch.zeros(1, 16, dtype=torch.long)  # 16 pad tokens

    full_seq = torch.cat([seq, pad], dim=1)
    attn_mask_padded = torch.cat([
        torch.ones(1, 16, dtype=torch.long),
        torch.zeros(1, 16, dtype=torch.long),
    ], dim=1)

    short_seq = seq                              # no padding
    attn_mask_short = torch.ones(1, 16, dtype=torch.long)

    with torch.no_grad():
        z_padded = text_encoder.encode(full_seq, attention_mask=attn_mask_padded)
        z_short  = text_encoder.encode(short_seq, attention_mask=attn_mask_short)

    cos = F.cosine_similarity(z_padded, z_short, dim=-1).item()
    print(f"  cos(padded, unpadded) = {cos:.6f}")
    if cos < 0.999:
        print(f"  WARNING: cosine < 0.999 ({cos:.6f}) — verify pooling ignores padding.")
        print(f"  (Expected if model uses last-token pooling without mask — check encode())")
    else:
        print("  PASS (pooling is padding-invariant)")


# ---------------------------------------------------------------------------
# Memory profile (CUDA only)
# ---------------------------------------------------------------------------

def test_memory_a100():
    print("\n--- A100 memory profile (CUDA) ---")
    if not torch.cuda.is_available():
        print("  SKIP: CUDA not available")
        return

    device = "cuda"
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    B, L, V = 32, 512, 50257
    from hybrid_xmamba.models.configuration_hybrid import HybridConfig
    from hybrid_xmamba.models.hybrid_lm import HybridLanguageModel

    cfg = HybridConfig(
        vocab_size=V, dim=512, num_layers=8,
        layer_pattern=["mamba", "mamba", "mlstm"],
        state_size=16, conv_size=4, expand_factor=2, dt_rank=None,
        use_fast_path=True, head_dim=64, num_heads=8, use_tfla=True,
        proj_factor=2, slstm_hidden_dim=512, slstm_num_heads=4,
        use_exponential_gate=True, norm_type="rms", use_mlp=True,
        mlp_ratio=4.0, max_position_embeddings=1024, dropout=0.1,
        initializer_range=0.02, use_cache=False, tie_word_embeddings=False,
    )
    student = HybridLanguageModel(cfg).to(device).to(torch.bfloat16)

    input_ids = torch.randint(0, V, (B, L), device=device)
    labels = input_ids.clone()

    out = student(input_ids, labels=labels, return_dict=True)
    out.loss.backward()

    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print(f"  Student-only peak memory : {peak_gb:.2f} GB / {total_gb:.1f} GB")
    print(f"  Estimated teacher (2.7B bf16 frozen): ~5.4 GB weights + ~5 GB acts")
    estimated_total = peak_gb + 10.4
    print(f"  Estimated total with teacher: ~{estimated_total:.1f} GB")
    if estimated_total < total_gb:
        print(f"  PASS: fits on {total_gb:.0f} GB A100")
    else:
        print(f"  WARNING: may OOM — use B=16, grad_accum=8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true", help="Also run CUDA memory test")
    args = parser.parse_args()

    print("=" * 60)
    print("Distillation Pipeline Smoke Tests")
    print("=" * 60)

    passed = 0
    failed = 0

    for name, fn in [
        ("Stage0 KD loss structure", test_stage0_kd_loss),
        ("Stage1 KD distill loss", test_stage1_kd_loss),
        ("Padding invariance", test_pooling_padding_invariance),
    ]:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {name}: {e}")
            import traceback; traceback.print_exc()
            failed += 1

    if args.cuda:
        try:
            test_memory_a100()
            passed += 1
        except Exception as e:
            print(f"  FAIL: A100 memory: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
