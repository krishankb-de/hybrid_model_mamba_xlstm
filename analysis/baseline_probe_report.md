# Phase 2 — Baseline Diagnostic Report (Phase 6e checkpoint)

**Date:** 2026-05-11  
**Checkpoint:** `output_willi_server/total_loss=2.6274.ckpt`  
**Device:** CPU (probes 1–3); STS-B + MIMIC pending Willi run (probe 4–5)

---

## Findings

### Probe 1 — mLSTM gate health

| Layer | Gate | Pre-activation max | mean | std | frac > 15 |
|-------|------|--------------------|------|-----|-----------|
| 2 | i_gate_raw | **14.25** | -2.09 | 2.08 | 0.0000 |
| 2 | f_gate_raw | 13.29 | 0.02 | 2.97 | 0.0000 |
| 5 | i_gate_raw | 8.37 | -2.26 | 1.34 | 0.0000 |
| 5 | f_gate_raw | 11.88 | 0.79 | 2.40 | 0.0000 |

**Verdict: PDF gap 2 CONFIRMED.** Layer-2 `i_gate` is at 14.25 — within 5% of the proposed 15.0 soft-cap. After `exp()`, this yields ≈ 1.55 × 10⁶, creating extreme weighting of a single input. No overflow observed on this batch, but margin is thin. Under longer sequences or domain-shifted inputs, saturation is likely. Phase 3 (tanh soft-cap + bias init) is high priority.

### Probe 2 — Layer-wise hidden norms

| Block | Output L2 norm |
|-------|---------------|
| 0 | 17.07 |
| 1 | 19.93 |
| 2 | 22.32 |
| 3 | 24.92 |
| 4 | 27.60 |
| 5 | 31.21 |
| 6 | 44.01 |
| 7 | **59.21** |

Norm grows **3.5×** from block 0 to block 7. Final block output norm is 59 — typical well-normalized transformers stay in the 4–12 range. This runaway growth makes the final RMSNorm's job harder (compressing 59→∼1), distorts projection head geometry, and destabilizes the contrastive loss.

**Verdict: PDF gap 3 CONFIRMED.** HybridNorm (Q/K/V pre-norm + FFN post-norm for blocks ≥ 1) is needed to regulate inter-block norm growth. Phase 4 is high priority.

### Probe 3 — Doc-contamination

| Metric | Value |
|--------|-------|
| Mean leak ratio (all blocks) | **0.184** |
| Max leak ratio (single block) | **0.432** |
| Verdict | LEAK_PRESENT |

18.4% of the document-B hidden-state energy is attributable to document-A content. The Mamba selective-scan carries SSM state across the EOS boundary; the mLSTM forget gate does not fully reset at token 50256. Block 6–7 show the worst leak (cascade).

**Verdict: PDF gap 4 CONFIRMED — largest single effect.** Cross-document contamination at 18% mean is a correctness blocker for the retrieval objective (MIMIC/Indiana). An encoder that embeds doc-B with 18% doc-A pollution cannot produce reliable paired embeddings. Phase 6 (cu_seqlens boundary reset) moves up in priority.

### Probe 4 — STS-B align/uniformity

Pending Willi run (requires HuggingFace access + GPU for reasonable throughput).

### Probe 5 — MIMIC cosine histogram

Pending Willi run (requires MIMIC cache on `/scratch/bhushkri/`).

---

## Priority Re-weighting

| PDF gap | Phase | Original priority | Probe support | Revised priority |
|---------|-------|------------------|---------------|-----------------|
| 4 — doc-boundary reset | Phase 6 | medium | **CONFIRMED critical** | **HIGHEST** |
| 3 — HybridNorm | Phase 4 | high | **CONFIRMED** | high |
| 2 — mLSTM stabilization | Phase 3 | high | **CONFIRMED** | high |
| 1 — layer pattern | Phase 5 | high | n/a (arch change) | high |
| 5 — WSD scheduler | Phase 7 | medium | n/a | medium |
| 6 — freq-decoupled KD | Phase 10 | medium | n/a | medium |

All six gaps still implemented as planned. No eliminations. Doc-boundary reset is the most
impactful single change based on direct evidence; norm growth is the most likely explanation
for the training ceiling.

---

## Actions

- [ ] Run probes 4–5 on Willi after Willi job completes
- [ ] Proceed to Phase 3 (mLSTM stabilization) — gate max at 14.25 leaves no safety margin
- [ ] Phases 3 → 4 → 6 are highest-evidence fixes; proceed in plan order
