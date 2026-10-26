# Attention-free text encoder + retrieval-augmented mLMamba (plan of record)

## ▶ RESUME HERE (new session: read this block, then `textenc_state.json`)

**Goal.** Answer the reviewer's text-encoder critique by building, and actually using at inference, an
**attention-free report encoder with the mLMamba layout** (Mamba-3/SSD + mLSTM), and test whether
giving the mLMamba decoder retrieved reports (retrieval-augmented generation, RAG) closes the CheXbert
gap to the retrieval floors. ISBI 2026 first (deadline 2026-10-26, minimal version), thesis later
(full version). Experimental: it may fail, so it is fully isolated from the published work.

**Reviewer critique this answers** (2026-10-14):
1. "The Mamba text encoder is untested and arguably pointless": no ablation vs PubMedBERT, it is
   discarded after Stage 1, inference uses only the ViT.
2. "Attention is not needed for retrieval" is vacuous; "nobody has implemented Mamba-based
   encoders" is false (lit check 2026-10-14: Mamba Retriever arXiv 2408.08066, M2-BERT ICML 2024,
   Hydra arXiv 2407.09941, CLIMP arXiv 2601.06891). Defensible wording: "to our knowledge the first
   attention-free report encoder for chest X-ray image-text retrieval".
3. (Not in this plan, belongs to `isbi_baselines` B10.) No comparison with published report
   generators (R2Gen, R2GenCMN, R2Gen-Mamba ISBI 2025, CvT2DistilGPT2, CXRMate); all four have
   ungated MIMIC weights.

**Where things run.**
| place | what |
|---|---|
| this worktree `../isbi_textenc_wt`, branch `isbi_textenc` (off `isbi_baselines` 0803cac) | all code, plan, state |
| cluster `hpi-hpc:~/hybrid_textenc` (git worktree, branch `isbi_textenc`) | jobs submitted from here |
| `~/hybrid_textenc/{outputs,results,analysis}` | REAL directories, NOT symlinks: nothing this plan makes lands in the published `outputs/`/`results/` |
| read-only inputs (absolute paths) | `/sc/home/$USER/dataset/mimic_full/`, `~/hybrid_mamba_xlstm/outputs/h100_stage0_150m_m3/checkpoints/last.ckpt`, `.../h100_stage0_transformer_150m/...`, the HF cache |

**Rules.**
- **R1 Isolation.** No job of this plan writes under `~/hybrid_mamba_xlstm/`, `~/hybrid_isbi/` or
  their `outputs/`/`results/`. Every wrapper takes its output root from `TEXTENC_ROOT` and refuses to
  run without it. Published checkpoints are only read.
- **R2 Shared code is additive.** Any edit to a file the published pipeline imports keeps the
  published default bit-for-bit (new flag, default off) and gets a parity test. New logic goes in new
  files where possible.
- **R3 No attention in the text encoder.** A test walks the module tree and fails on any attention
  module (`AttentionPooling`, `AttentionBlock`, `nn.MultiheadAttention`, SDPA call). The image encoder
  (BiomedCLIP ViT) still uses attention; the paper must say so.
- **R4 Pre-registered.** Success criteria below are fixed before any result; every arm is reported,
  win or lose. Never change a measured number.
- **R5 MIMIC data stays on the cluster** (retrieved reports, indices, generations). Aggregates only
  leave it.
- **R6** `bash scripts/validate.sh` exits 0 before any commit; commit/push only when the user asks.

## Design (fixed before results)

**Text encoder E-m3.** `HybridTextEncoder` with the mLMamba layout: 12 layers, width 768, mamba3 at
1-4 and 8-12, mLSTM at 5-7, `tfla_impl: exact`, initialised from the PubMed-pretrained Stage-0
mLMamba checkpoint. Pooling: **masked mean over report tokens** (exists today as
`pooling_strategy: mean`; attention-free). Diagnostic arm, thesis only: bidirectional mean (forward
pass + pass over the reversed tokens, averaged), still attention-free.
Stage 1 recipe = the published 13D tower recipe (CLIP + PubMedBERT distillation + SimCSE 0.1,
`vit_lr=3e-6`, full MIMIC train split), only the text encoder differs. Output: a new image tower +
E-m3.

**Use at inference (RAG).** For each study: (1) top-50 training studies by image-image cosine on the
512-d vector; (2) re-rank them by image-text cosine between the query image and E-m3's embedding of
each candidate report; (3) keep the top **k=3** reports. Training queries exclude studies of the same
patient (no copying one's own report). The decoder reads
`[32 image prefix][Similar case 1 report][sep]...[Similar case k report][sep] Findings: ...`,
each retrieved report truncated to 160 tokens; loss only on the target report. Context about
32 + 3x~130 + ~150, under 700 tokens.

**Arms.**
| arm | text encoder (Stage 1) | decoder | k | for |
|---|---|---|---|---|
| M-pub | published (old Mamba-1 hybrid, attention pooling) | mLMamba | 0 | existing 3-seed results, no new runs |
| M-rag | E-m3 | mLMamba | 3 | main arm |
| T-rag | E-m3 | Transformer | 3 | matched-architecture control |
| F-rr | E-m3 | none: copy top-1 after re-rank | - | text encoder at inference, cheapest test |
| ABL-T | matched Transformer text encoder (attention) | - | - | ablation (retrieval only, ISBI if queue allows) |
| ABL-P | fine-tuned PubMedBERT text encoder | - | - | ablation (retrieval only; new code; thesis if no time) |

**Pre-registered success criteria.**
- Primary: M-rag beats M-pub on CheXbert-14 micro AND macro at >=2/3 seeds (paired bootstrap, 95%).
- Secondary: M-rag vs T-rag (any direction is reported); M-rag vs XrayCLIP and BiomedCLIP floors;
  F-rr vs the BiomedCLIP floor (does the re-rank help?).
- Retrieval (T2): E-m3's image-to-text R@10 vs the published tower's on the same held-out protocol.
- A failed primary is a result, written up as such.

## Phases

### T0 — Plan, isolation, approval
- [x] **T0-A** Plan + `textenc_state.json` on branch `isbi_textenc` (own worktree); `textenc` registered in `scripts/mamba3_state.py` PLAN_SETS.
- [x] **T0-B** User approves the plan (gate: nothing below runs before this).
- [x] **T0-C** Cluster isolation: `~/hybrid_textenc` git worktree on `isbi_textenc`, real `outputs/ results/ analysis/` dirs, `.venv`/`.venv_chexbert` symlinks (environments only); a check script proves no wrapper resolves a path under `~/hybrid_mamba_xlstm/{outputs,results}` for writing.

### T1 — Attention-free mLMamba text encoder (code, CPU-tested)
- [x] **T1-A** `configs/model/hybrid_150m_m3_textenc.yaml`: mLMamba layout, `tfla_impl: exact`, `pooling_strategy: mean`; parity test that its backbone matches `hybrid_150m_m3_rrg` layer for layer.
- [x] **T1-B** Init from the Stage-0 mLMamba checkpoint: only the projection head is missing on load (test with a tiny checkpoint).
- [x] **T1-C** R3 test: no attention module or SDPA call anywhere in the text encoder; CPU forward/backward; mean pooling ignores padding (same embedding with extra pads).
- [x] **T1-D** New Stage-1 wrapper `scripts/textenc_stage1_h100.sh` (published wrapper untouched; refuses to run without `TEXTENC_ROOT`), recipe identical to the 13D tower except the text encoder; wrapper test pins the recipe values against the published one.

### T2 — Stage 1 with E-m3 + retrieval check
- [x] **T2-A** Train Stage 1 (1 GPU, ~4-7 h). Evidence: job id, loss curves, no NaN.
- [x] **T2-B** Retrieval on the published held-out protocol: i2t/t2i R@1/5/10 for E-m3 vs the published tower (same script, same studies).
- [ ] **T2-C** (if queue allows before ISBI) ABL-T: same Stage 1 with the matched Transformer text encoder; retrieval only.
- [ ] **T2-D** (thesis) ABL-P: fine-tuned PubMedBERT text encoder; needs a text-tower option in Stage 1.

### T3 — Retrieval-augmented data + the re-rank floor
- [x] **T3-A** `scripts/textenc_rag_index.py`: top-50 by image, re-rank by image-text with E-m3, keep top-k ids per study for train (same-patient excluded), val, test. Stored under `~/hybrid_textenc/results/` only. Tests on toy data: same-patient exclusion, ordering, k.
- [ ] **T3-B** F-rr: copy the top-1 re-ranked report, score text + CheXbert on the test split; bootstrap vs the BiomedCLIP floor.

### T4 — RAG decoder (code, CPU-tested)
- [x] **T4-A** Dataset path that inserts the k retrieved reports after the image prefix, loss masked to the target; default `rag_k=0` is the published path bit-for-bit (parity test).
- [x] **T4-B** Evaluation path with the same context (uncached beam, or cached with token-by-token prefill); test that both give identical tokens on a toy model.
- [x] **T4-C** Wrapper `scripts/textenc_rag_train_h100.sh` / eval wrapper, `TEXTENC_ROOT` enforced.

### T5 — Train + evaluate M-rag and T-rag
- [x] **T5-A** Seed 42 for M-rag and T-rag (Stage 3 recipe unchanged: 12k steps, 4 GPUs). Health gate: finite loss, val loss in range.
- [x] **T5-B** Seeds 43/44 for both.
- [ ] **T5-C** Beam-3 test decode, CheXbert, paired bootstraps: M-rag vs M-pub (same seed), M-rag vs T-rag, M-rag vs floors; per-label F1; error analysis (B9 script).

### T6 — Efficiency in the real RAG pipeline
- [ ] **T6-A** Decode latency and memory for both decoders at the real RAG context (k=3) and at k=10/30 (B7 bench), with the prefill caveat stated.

### T7 — Write-up
- [ ] **T7-A** `analysis/textenc/FINDINGS.md` with every arm, win or lose.
- [ ] **T7-B** Paper text only on the user's request; claim wording per the lit check (no "nobody has implemented Mamba encoders").

## Timeline risk (ISBI)
GPU queue was the bottleneck in B8 (evaluations waited up to a week on priority). Minimal ISBI scope
= T1, T2-A/B, T3, T4, T5 with M-rag + T-rag. If T5-B cannot finish by about 2026-10-23, ISBI gets
seed-42 results labelled as one seed, and the thesis gets the rest.
