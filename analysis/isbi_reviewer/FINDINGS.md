# ISBI reviewer-risk experiments: findings (B7, B9 complete; B8 running)

All numbers are aggregates. Raw outputs live on the cluster under analysis/isbi_reviewer/ and
results/ (MIMIC-derived files never leave it).

## B7 Sequence length (jobs 2624264, 2624265 + 2624905, 2624648/9)

**Context lengths (test split, GPT-2 tokens).** Reports: median 113, p95 214, 1.8% > 256.
89.9% of test studies have earlier studies of the same patient (median 6). All their reports
joined: median 738 tokens, p95 3,858, max 8,037. A history-aware decoder would read
32 + history + 100 tokens: median 870, 44% > 1,024, 21% > 2,048, 4.6% > 4,096, none > 8,192.
89% of reference reports contain comparison words (broad keyword match, an upper bound).

**Writing one report (prompt read once, then 100 tokens, beam 3, bf16, one H100).**
Both decoders cached (mLMamba: fixed state; Transformer: KV cache, FlashAttention kernels).

| reports | context | mLMamba ms | Transformer ms | Transformer / mLMamba | decode cache mLMamba vs Transformer |
|---|---|---|---|---|---|
| 1 | 132 (today) | 667 | 505 | 0.76 (Transformer faster) | 0.022 vs 0.054 GB |
| 1 | 1,024 | 664 | 1,080 | 1.63 | 0.022 vs 0.18 GB (8x) |
| 1 | 4,096 | 702 | 1,508 | 2.15 | 0.022 vs 0.60 GB (27x) |
| 1 | 16,384 | 723 | 3,631 | 5.02 | 0.022 vs 2.30 GB (105x) |
| 16 | 132 | 693 | 596 | 0.86 (Transformer faster) | 0.36 vs 0.86 GB |
| 16 | 1,024 | 741 | 1,322 | 1.78 | 0.36 vs 2.83 GB (8x) |
| 16 | 4,096 | 798 | 2,246 | 2.81 | 0.36 vs 9.63 GB (27x) |
| 16 | 16,384 | 1,161 | 6,905 | 5.95 | 0.36 vs 36.8 GB (103x) |

mLMamba's decode step is 6.5-7.4 ms at every context (flat); the Transformer's grows 5.0 ->
35.9 ms (1 report) and 5.9 -> 58.6 ms (16 reports). Prompt reading alone: mLMamba (compiled) is
faster at 128 tokens, SLOWER at batch 1 for 1,024-16,384 (e.g. 68 vs 43 ms at 16,384), faster at
batch 16 from 1,024 up (417 vs 642 ms at 16,384). Decoding dominates the report time.

Verdict: for today's reports (132 tokens) the reviewer is right: the Transformer writes a report
1.2-1.3x faster. With the patient's earlier reports as context (median 870, p95 ~4,000 tokens)
mLMamba writes 1.6-2.8x faster and its decode memory is 8-27x smaller.

Caveats: (1) the mLMamba end-to-end figure assumes the chunked forward hands its final state to the
decoder; the code computes that state but does not yet return it (M6 follow-up), today it would
prefill token by token. (2) Both decode steps are eager PyTorch; neither uses CUDA graphs or a
serving engine. (3) No history-aware model was trained; this measures cost, not quality.
(4) The first Transformer run (2624265) was invalid: SDPA chose cuDNN, which re-plans on the CPU
at every step (~15 ms per layer); fixed by pinning flash/efficient/math in AttentionBlock.step.

## B9 Error analysis (job 2624264)

| | wrongly added (FP / ref-negative) | missed (FN / ref-positive) | findings per report |
|---|---|---|---|
| mLMamba, 3 seeds | 0.099-0.106 | 0.579-0.629 | 2.14-2.36 |
| Transformer, 3 seeds | 0.096-0.103 | 0.591-0.626 | 2.11-2.29 |
| BiomedCLIP floor | 0.158 | 0.582 | 2.87 |
| XrayCLIP floor | 0.149 | 0.489 | 3.07 |
| reference | - | - | 3.13 |

Acute findings (mLMamba mean of 3 seeds, added / missed): pneumothorax 0.010 / 0.956 (84 positive
studies), consolidation 0.019 / 0.953, pneumonia 0.061 / 0.891, edema 0.093 / 0.706, pleural
effusion 0.145 / 0.477. Floors add pneumothorax 4-6% and miss 79-81%. Studies with any wrongly
added acute finding: 18-21% (decoders) vs 38-41% (floors); with any missed: 51-53% vs 41-53%.
Most wrongly added findings are common ones: cardiomegaly 0.42, support devices 0.32.
The Transformer matches mLMamba within about one point everywhere, so this is not an SSM effect.
Qualitative cases: results/isbi_error_analysis/qualitative_mLMamba_s42.md (cluster only).

## B8 Image encoder: pending (jobs 2624239-2624256, training expected from 11 Oct)
