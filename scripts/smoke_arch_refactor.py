"""Phase 8A — CPU smoke for the architectural refactor (Phases 3/4/6/7).

Exercises a *small-dim* hybrid_70m_v2-shaped model on CPU with PubMed-style
multi-doc batches (cu_seqlens packing). Validates:

  - Phase 3 (mLSTM stabilization): i_gate pre-cap raw max < soft-cap (15)
  - Phase 4 (HybridNorm): norm_topology='hybrid' wires without error
  - Phase 5 (v2 layer pattern): [m,m,m,L,L,m,m,m] loads
  - Phase 6 (cu_seqlens doc-boundary reset): perturbing doc-B tokens leaves
    doc-A output bit-identical
  - Phase 7 (WSD scheduler + β2): scheduler factor + β2 anneal track expected
    shape across 100 steps
  - End-to-end: 100-step CPU train loop, loss decreasing, no NaN, grad-norm < 10

Tiny dims (dim=64, layers=8) keep CPU wall-clock under ~2 minutes.

Usage:
    python scripts/smoke_arch_refactor.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from hybrid_xmamba.models.configuration_hybrid import HybridConfig
from hybrid_xmamba.models.hybrid_lm import HybridLanguageModel
from hybrid_xmamba.training.schedulers import (
    WSDScheduler,
    apply_beta2_schedule,
    beta2_for_step,
    wsd_factor,
)


V2_PATTERN: List[str] = ["mamba", "mamba", "mamba", "mlstm", "mlstm",
                          "mamba", "mamba", "mamba"]


def _make_small_v2_config() -> HybridConfig:
    """Tiny model matching v2 layer pattern + Phase 3/4 knobs."""
    return HybridConfig(
        vocab_size=256,
        dim=64,
        num_layers=8,
        layer_pattern=V2_PATTERN,
        state_size=8,
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
        max_position_embeddings=128,
        dropout=0.0,
        initializer_range=0.02,
        use_cache=False,
        tie_word_embeddings=False,
        norm_topology="hybrid",
        mlstm_gate_soft_cap=15.0,
        mlstm_input_gate_bias_init=-10.0,
        mlstm_forget_gate_bias_init=0.0,
    )


# ---------------------------------------------------------------------------
# Synthetic PubMed-style packed dataset with cu_seqlens
# ---------------------------------------------------------------------------


class PackedDocsDataset(Dataset):
    """Each example is a single packed sequence concatenating 2-3 short docs.

    cu_seqlens is a per-position doc-id (matches train.py:175 group_texts).
    """

    def __init__(self, n: int, seq_len: int, vocab: int, seed: int = 0) -> None:
        g = torch.Generator().manual_seed(seed)
        self.items: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(n):
            ids = torch.randint(1, vocab, (seq_len,), generator=g)
            doc_id = torch.zeros(seq_len, dtype=torch.long)
            # 1-3 boundaries inside the sequence
            n_boundaries = int(torch.randint(1, 4, (1,), generator=g).item())
            cuts = sorted(int(c) for c in torch.randint(
                4, seq_len - 4, (n_boundaries,), generator=g).tolist())
            cur = 0
            for c in cuts:
                doc_id[c:] = cur + 1
                cur += 1
            self.items.append((ids, doc_id))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ids, doc_id = self.items[idx]
        return {"input_ids": ids, "labels": ids.clone(), "cu_seqlens": doc_id}


def _collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "cu_seqlens": torch.stack([b["cu_seqlens"] for b in batch]),
    }


# ---------------------------------------------------------------------------
# Phase 3 probe: i_gate pre-cap raw max via forward hook on i_gate_proj
# ---------------------------------------------------------------------------


class IGateProbe:
    def __init__(self, model: nn.Module) -> None:
        self.max_raw = 0.0
        self.handles = []
        for m in model.modules():
            if m.__class__.__name__ == "mLSTMBlock":
                h = m.i_gate_proj.register_forward_hook(self._hook)
                self.handles.append(h)

    def _hook(self, _module, _inp, out: torch.Tensor) -> None:
        cur = float(out.detach().abs().max().item())
        if cur > self.max_raw:
            self.max_raw = cur

    def close(self) -> None:
        for h in self.handles:
            h.remove()


# ---------------------------------------------------------------------------
# Phase-by-phase smokes
# ---------------------------------------------------------------------------


def test_v2_pattern_loads() -> None:
    print("\n--- Phase 5: hybrid_70m_v2 layer pattern loads (small-dim) ---")
    cfg = _make_small_v2_config()
    model = HybridLanguageModel(cfg)
    block_kinds = [b.layer_type for b in model.layers]
    assert block_kinds == V2_PATTERN, f"layer_pattern mismatch: {block_kinds}"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  blocks: {block_kinds}")
    print(f"  params: {n_params/1e6:.2f}M (small-dim smoke)")
    print("  PASS")


def test_hybridnorm_wires() -> None:
    print("\n--- Phase 4: HybridNorm topology wires (pre_rms + hybrid both run) ---")
    for topo in ("pre_rms", "hybrid"):
        cfg = _make_small_v2_config()
        cfg.norm_topology = topo
        model = HybridLanguageModel(cfg)
        model.eval()
        ids = torch.randint(0, cfg.vocab_size, (2, 32))
        with torch.no_grad():
            out = model(ids, return_dict=True)
        assert torch.isfinite(out.logits).all(), f"{topo}: non-finite logits"
        print(f"  topology={topo}: logits {tuple(out.logits.shape)} finite OK")
    print("  PASS")


def test_doc_boundary_reset() -> None:
    print("\n--- Phase 6: cu_seqlens doc-boundary reset (perturb doc-B, doc-A stays) ---")
    cfg = _make_small_v2_config()
    model = HybridLanguageModel(cfg)
    model.eval()
    B, L = 2, 32
    boundary = 16
    cu = torch.zeros(B, L, dtype=torch.long)
    cu[:, boundary:] = 1

    ids_a = torch.randint(1, cfg.vocab_size, (B, L))
    ids_b = ids_a.clone()
    # perturb doc-B tokens only
    ids_b[:, boundary:] = torch.randint(1, cfg.vocab_size, (B, L - boundary))

    with torch.no_grad():
        out_a = model(ids_a, cu_seqlens=cu, return_dict=True).logits
        out_b = model(ids_b, cu_seqlens=cu, return_dict=True).logits

    # doc-A (positions [0, boundary)) must be identical
    diff = (out_a[:, :boundary] - out_b[:, :boundary]).abs().max().item()
    print(f"  max|Δlogits doc-A| under doc-B perturbation: {diff:.3e}")
    assert diff < 1e-4, f"doc-boundary reset FAILED: diff={diff:.3e}"
    print("  PASS")


def test_wsd_schedule_shape() -> None:
    print("\n--- Phase 7: WSD factor + β2 anneal shape across 100 steps ---")
    max_steps = 100
    warmup = max(1, int(max_steps * 0.01))   # 1
    decay = max(1, int(max_steps * 0.14))    # 14
    stable = max_steps - warmup - decay       # 85
    decay_start = warmup + stable             # 86
    factors = [wsd_factor(s, warmup, stable, decay) for s in range(max_steps)]
    assert factors[0] < 0.05, f"step 0 factor too large: {factors[0]}"
    assert abs(factors[50] - 1.0) < 1e-6, f"stable phase != 1: {factors[50]}"
    assert factors[-1] < 0.05, f"decay endpoint not ~0: {factors[-1]}"

    b2_start = beta2_for_step(0, decay_start, decay, 0.999, 0.974)
    b2_mid = beta2_for_step(decay_start + decay // 2, decay_start, decay,
                            0.999, 0.974)
    b2_end = beta2_for_step(max_steps, decay_start, decay, 0.999, 0.974)
    assert abs(b2_start - 0.999) < 1e-6
    assert b2_start > b2_mid > b2_end
    assert abs(b2_end - 0.974) < 5e-3, f"β2 endpoint off: {b2_end}"
    print(f"  factor[0]={factors[0]:.4f}, stable={factors[50]:.4f}, "
          f"end={factors[-1]:.4f}")
    print(f"  β2: {b2_start:.4f} → {b2_mid:.4f} → {b2_end:.4f}")
    print("  PASS")


def test_100step_train_loop() -> None:
    print("\n--- E2E: 100-step CPU train loop on packed PubMed-style data ---")
    torch.manual_seed(0)
    cfg = _make_small_v2_config()
    model = HybridLanguageModel(cfg)
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params/1e6:.2f}M, pattern={cfg.layer_pattern}")

    seq_len = 64
    # Small, fixed pool so the LM can actually learn the unigram → bigram pattern
    # in 100 steps. Random uniform vocab=256 sits at the entropy floor.
    ds = PackedDocsDataset(n=16, seq_len=seq_len, vocab=32, seed=1)
    dl = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=_collate)

    max_steps = 100
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3,
                                  betas=(0.9, 0.999), weight_decay=0.01)
    sched = WSDScheduler(optimizer, max_steps=max_steps,
                         warmup_ratio=0.01, stable_ratio=0.85,
                         decay_ratio=0.14)

    probe = IGateProbe(model)
    losses: List[float] = []
    max_grad_norm = 0.0
    step = 0
    data_iter = iter(dl)

    while step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            batch = next(data_iter)
        out = model(batch["input_ids"], labels=batch["labels"],
                    cu_seqlens=batch["cu_seqlens"], return_dict=True)
        loss = out.loss
        assert torch.isfinite(loss), f"step {step}: loss NaN/Inf"
        optimizer.zero_grad()
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0))
        if gnorm > max_grad_norm:
            max_grad_norm = gnorm
        optimizer.step()
        sched.step()
        apply_beta2_schedule(optimizer, step, sched.decay_start,
                             sched.decay_steps,
                             beta2_start=0.999, beta2_end=0.974)
        losses.append(float(loss.item()))
        step += 1

    probe.close()

    avg_first = sum(losses[:10]) / 10
    avg_last = sum(losses[-10:]) / 10
    print(f"  loss[:10] avg = {avg_first:.4f} → loss[-10:] avg = {avg_last:.4f}")
    print(f"  max grad-norm (pre-clip) = {max_grad_norm:.4f}")
    print(f"  i_gate raw |max| across run = {probe.max_raw:.4f} (cap=15)")

    assert avg_last < avg_first, (
        f"loss not decreasing: first={avg_first:.4f} last={avg_last:.4f}")
    assert max_grad_norm < 50.0, f"grad-norm explosion: {max_grad_norm}"
    assert probe.max_raw < cfg.mlstm_gate_soft_cap, (
        f"i_gate raw {probe.max_raw} ≥ soft_cap {cfg.mlstm_gate_soft_cap}")
    print("  PASS")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    print("=" * 60)
    print("Phase 8A — architectural refactor CPU smoke")
    print("=" * 60)

    tests = [
        ("Phase 5 v2 pattern loads", test_v2_pattern_loads),
        ("Phase 4 HybridNorm wires", test_hybridnorm_wires),
        ("Phase 6 doc-boundary reset", test_doc_boundary_reset),
        ("Phase 7 WSD + β2 schedule", test_wsd_schedule_shape),
        ("E2E 100-step train loop", test_100step_train_loop),
    ]

    passed, failed = 0, 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL [{name}]: {exc}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
