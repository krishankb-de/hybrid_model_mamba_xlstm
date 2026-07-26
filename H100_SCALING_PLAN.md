# H100 Scale-Up Plan — Hybrid Mamba-xLSTM CXR Retrieval (Plan-of-Record)

> Resumable plan-of-record. Read this + `h100_scaling_state.json` (gitignored, allowlisted) at session start.
> **Builds on the COMPLETED `HYBRID_ARCH_REFACTOR_PLAN.md`** (broke the MIMIC ceiling 8.23%→10.45% i2t R@10). That plan is finished — historical reference only.
> Full approved plan: `/Users/krish/.claude/plans/i-want-to-implement-twinkling-ullman.md`.
>
> **Current phase: Phase 6C — measurement block** (Phases 1/2/4/5/6/6B done; 6B was the 7th consecutive null — see the 2026-07-25 post-mortem below before proposing any new lever. Phase 3 deferred post-Phase-6.)

## Context

Prior campaign is done on **A100 40GB** (willi/`mitarb`). Canonical model = `hybrid_70m_v2` + `freq_kd=false` + `vit_unfreeze=2` + `moco=0`. Final: **MIMIC i2t R@10 10.45%** (Target tier), **Indiana 3.90%** (intrinsic/data-bound), **Stage-0 PPL 15.62** (undertrained vs baseline 13.10).

User now has **H100 (94/141GB)** + optional 2-4 H100 node. Three A100-era ceilings are now liftable:
1. **Contrastive negatives capped at ~31** — CLIP loss is in-batch only (no `all_gather`, `moco=0`); H100 VRAM fits 128-256 true negatives (`lightning_module.py:512-543,1127-1158`). Biggest MIMIC lever; also cuts epochs on the 27.5k-pair set → less overfitting.
2. **Stage-0 undertrained** — 2.7B frozen teacher forced bs=8/40GB; curve still descending at 40K (needed ~117K). H100 fits bs=32-64 + teacher → finish it.
3. **70M cap** — 150M/350M configs exist but use the OLD `[m,m,mlstm]`+`pre_rms` (no v2 wins). H100 fits 150M v2 training.

Indiana gap is ablation-proven data-bound → only lever is diverse CXR data (user has access).

**Goal:** H100-native infra + 150M-v2 backbone + scaled contrastive negatives + multi-source CXR data → push MIMIC to stretch (≥12%) and recover Indiana (≥floor), with clean per-lever attribution.

## Success bar (tiered) — **status 2026-07-26**
- **Floor** (no regression): MIMIC i2t R@10 ≥ 10.45%; Indiana i2t ≥ 4.04% (recover); Stage-0 PPL ≤ 15.62.
- **Target**: MIMIC ≥ 12% (old stretch); Indiana ≥ 5.5%; PPL ≤ 13.76.
- **Stretch**: MIMIC ≥ 14%; Indiana ≥ 7%; PPL ≤ 13.10.

| Metric | Best | Tier reached |
|---|---|---|
| MIMIC i2t R@10 | **0.1714** (D1c, vit_unfreeze=12) | **STRETCH** ✅ |
| Indiana i2t R@10 | **0.0485** (D1c) | Floor ✅ (target 0.055 open) |
| Stage-0 val PPL | **13.18** (Phase 5) | Target ✅, ~stretch (13.10) |

The MIMIC headline is **8.23% → 10.45% (A100 refactor) → 17.14% (H100 + deep ViT adaptation)**. The single decisive intervention was image-tower adaptation depth; every text-side and objective-side lever was null.

## Resolved decisions (from user, 2026-07-07)
- **SLURM**: long training → `--partition=aisc-batch` (7-day cap); eval/smoke → `--partition=aisc-shortrun` (1-day); `--gres=gpu:h100:X` (X=1..8). 7-day cap ⇒ full Stage-0 in ONE block (no requeue juggling).
- **Model**: scale to **150M v2** (port v2 arch; fresh Stage-0).
- **Priority**: backbone + MIMIC (finish Stage-0 → push MIMIC to stretch).
- **Indiana data**: add diverse CXR; **IU-Xray EXCLUDED from training** (it IS the Indiana eval set — zero-leakage).
- **Per-source text**: free-text passthrough (MIMIC) + label-templated pseudo-reports (CheXpert/VinDr) + translated reports (PadChest).
- **150M `max_position_embeddings`**: 1024 (v2 parity; corpora are ≤512).
- **KD teacher**: BioMedLM 2.7B primary; larger-teacher fallback if PPL gate missed.
- **Python**: target py≥3.10 on H100 (verify stack; fall back to 3.9 if a dep breaks); keep 3.9-syntax hygiene (forward-compatible).
- **Multi-GPU (Phase 3)**: secondary — after single-H100 MIMIC win.

## Critical files

| File | Phase | Action |
|---|---|---|
| `H100_SCALING_PLAN.md`, `h100_scaling_state.json`, `.gitignore`, `CLAUDE.md` | 1 | plan/state/bootstrap |
| `configs/trainer/h100_single_gpu.yaml`, `h100_multi_ddp.yaml` (NEW) | 2 | H100 trainers |
| `scripts/train_stage0_h100.sh`, `train_biomedclip_kd_h100.sh`, `eval_h100.sh` (NEW) | 2 | SLURM templates (aisc-batch/shortrun) |
| `hybrid_xmamba/training/lightning_module.py:512-543,1041,1127-1158` | 3 | `all_gather` CLIP negatives |
| `configs/model/hybrid_150m_v2.yaml` (NEW) | 4 | 150M v2 arch |
| `scripts/train_stage0_150m_h100.sh` (NEW) | 5 | 150M Stage-0 |
| `scripts/train_biomedclip_kd_150m_h100.sh` (NEW) | 6 | batch-scaled contrastive |
| `configs/dataset/cxr_multi.yaml` (NEW), `scripts/train_contrastive.py:339-451` | 7 | multi-source CXR + text adapter |
| `analysis/h100_scaling_results.md` (NEW) | 8 | results |
| `tests/test_willi_parity.py`, `test_layers.py` | 2-4 | per-phase asserts |

---

## Phases

### Phase 1 — Plan-of-record + state (NO CODE) ✅ COMPLETE
- [x] **1A** — Write `H100_SCALING_PLAN.md` at repo root (this file).
- [x] **1B** — Write `h100_scaling_state.json` at repo root.
- [x] **1C** — `.gitignore`: allowlist `H100_SCALING_PLAN.md` + `!h100_scaling_state.json`.
- [x] **1D** — `CLAUDE.md` Session Bootstrap: repoint to this plan; `HYBRID_ARCH_REFACTOR_PLAN.md` → completed-historical.
- [x] **1E** — `bash scripts/validate_for_willi.sh` green: 69 passed, 5 skipped, 9/9 gates (doc-only; no regression).
- [x] **1F** — Commit on branch `h100_scaling`.

### Phase 2 — H100 infra enablement (single-GPU primary + 2-4 node file) ✅ LOCAL COMPLETE (2D/2G need H100 box)
Reuse: `torch.set_float32_matmul_precision('high')` (already in train scripts); auto-compile on sm_90 (`train.py:320-334` covers H100).
- [x] **2A** — `configs/trainer/h100_single_gpu.yaml`: `bf16-mixed`, `devices=1`, `compile_model=true`, `accumulate_grad_batches=1`.
- [x] **2B** — `configs/trainer/h100_multi_ddp.yaml` (2-4 node): `strategy=ddp`, `devices=-1`, `find_unused_parameters=true`, `accum=1`. Inert until Phase 3.
- [x] **2C** — SLURM templates `train_stage0_h100.sh` (aisc-batch 7-day, bs=64/accum=1, gc off, compile off for segmented path), `train_biomedclip_kd_h100.sh` (aisc-shortrun, **bs=128 = the in-batch-negative lever**, LR √-scaled), `eval_h100.sh` (ppl/retrieval modes). ENV-parametrized (`SCRATCH_ROOT`/`VENV_ACTIVATE`) — aisc scratch/env paths TBD.
- [ ] **2D** — (H100 box) Python env: full-stack import + 2-step train smoke on py≥3.10; fall back to 3.9 if a dep breaks.
- [x] **2E** — `requirements.txt`: flash-attn documented **opt-in** (NOT forced into base reqs — needs nvcc, breaks CPU/parity harness; ViT-only, marginal).
- [x] **2F** — `tests/test_willi_parity.py::test_h100_trainer_configs_resolve` (2 cases green).
- [ ] **2G** — (H100 box) Smoke `smoke_arch_refactor.py` on H100 (finite fwd/bwd, i_gate<cap, no NaN).
- [x] **2H** — `validate_for_willi.sh` green (9/9 gates; +2 h100 config tests); committed.

### Phase 3 — Distributed CLIP negatives (`all_gather`) — SECONDARY, deferred post-Phase-6
Only makes multi-GPU help retrieval. Single 141GB already gives 128-256 negatives → not on the critical path.
- [ ] **3A** — `lightning_module.py`: grad-preserving `_gather_across_gpus()` (autograd-aware `torch.distributed.nn.functional.all_gather`, OpenCLIP-style). Wire into `_nt_xent_loss` (`:512-543`) + `_joint_step` CLIP (`:1041`). Gate on `dist.is_initialized() and world_size>1` → identity single-GPU.
- [ ] **3B** — Test: world_size=1 numerical invariance; gather shape == world_size×B.
- [ ] **3C** — `validate_for_willi.sh` green; commit.

### Phase 4 — 150M v2 architecture config ✅ COMPLETE
`HybridConfig` supports all knobs; `create_hybrid_blocks` handles any pattern — config-only + tests.
- [x] **4A** — `configs/model/hybrid_150m_v2.yaml`: `dim=768`, `num_layers=12`, `head_dim=64`, `num_heads=12`, `norm_topology=hybrid`, `pooling_strategy=attention`, `max_position_embeddings=1024`, `learning_rate=4.0e-4` (√-width scaled; tunable→5e-4), `warmup_steps=2000`. mLSTM stabilization knobs via HybridConfig defaults (as 70m_v2 does).
- [x] **4B** — `layer_pattern`: `[m,m,m,m,L,L,L,m,m,m,m,m]` (3 mLSTM centered = 25% = v2 parity).
- [x] **4C** — `test_hybrid_150m_v2_config_and_param_count`: **183.72M actual** (nominal "150M"; untied 50k-vocab embeddings dominate — consistent with the 70M config → 83M convention). Tight band [181,186]M guards arch drift.
- [x] **4D** — `validate_for_willi.sh` green (72 passed, 9/9 gates); committed.

### Phase 5 — Stage-0 pretrain 150M v2 on H100 (backbone quality) ⏳ SCRIPT READY — run pending H100 box
- [x] **5A** — `scripts/train_stage0_150m_h100.sh` (wrapper over `train_stage0_h100.sh`; `model=hybrid_150m_v2`, bs=48/accum=1, `max_steps=120000` ~3B tokens, `warmup=2000`, WSD, aisc-batch 4-day).
- [ ] **5B** — (H100) Verify `train_stage0_distill.py` threads `norm_topology` (fixed 9F) + no hardcoded dim=512.
- [ ] **5C** — (H100) Submit. KD teacher BioMedLM 2.7B (primary); fallback larger teacher (BioGPT-Large/Meditron-7B/OpenBioLLM-8B) if PPL misses. Optional KD `alpha` {0.3,0.5} sweep.
- [ ] **5D** — (H100) Gate: `eval_h100.sh MODE=ppl` / `eval_stage0_lm.sh` (auto-detect; locked protocol). Target PPL ≤ 13.76.

### Phase 6 — Contrastive batch scaling (MIMIC-only, isolate lever) → stretch MIMIC ⏳ SCRIPT READY — run pending H100 box
Hold canonical recipe (`biomedclip_kd_joint_v2`: freq_kd=false, vit_unfreeze=2, moco=0). Change ONLY batch + backbone.
- [x] **6A** — `scripts/train_biomedclip_kd_150m_h100.sh` (wrapper; `model=hybrid_150m_v2`, `lm_checkpoint`=Phase-5, `batch_size` sweep {64,128,256} via env, `accum=1`, LR √-scaled for bs=128).
- [ ] **6B** — (H100) LR √-scale per batch: `backbone_lr→~2e-5`, `head_lr→~6e-4` at bs=128 (re-scale for other bs). Log embedding mean/std + cosine histogram (collapse watch).
- [ ] **6C** — (H100) Submit. Kill gates: `cos_text_teacher≥0.85` by 1k; `val/clip_loss<3.0`; MIMIC R@10 ≥ 10.45% by 3k.
- [x] **6D** — (H100) Gate: MIMIC i2t R@10 (`eval_h100.sh MODE=retrieval` / `evaluate_cxr_retrieval.py`). Target ≥12%. **RESULT: NULL.** bs=128/23ep 0.1084, bs=128/14ep 0.1090, bs=64/14ep **0.1113** (best). Floor 0.1045 cleared by all; target 0.12 missed by all; spread 0.29pp vs SE ~0.57pp ⇒ arms statistically indistinguishable.

### Phase 6B — LR-matched rerun (the one supported lever from the 2026-07-21 review) ⏳ SCRIPT READY
Phase-6 post-mortem found the batch sweep was **never LR-matched**: `backbone_lr`/`head_lr` were hardcoded at the bs=128 √-scaled values (`train_biomedclip_kd_h100.sh:90-91`), so the winning bs=64 arm silently trained at ~1.4x its proper LR. Combined with grad_norm ~12.3 against `gradient_clip_val=1.0` (~12x clipping every step), LR is the one untested knob with direct evidence behind it.
- [x] **6B-1** — `BACKBONE_LR`/`HEAD_LR` env-overridable in `train_biomedclip_kd_h100.sh`; 150M wrapper derives LR **and** `MAX_STEPS` from `BATCH_SIZE` (384000-sample / 13.93-epoch budget held across arms) so neither confound can recur. `EXPERIMENT` name now carries head LR so same-batch arms don't overwrite each other.
- [x] **6B-2** — Tests `test_h100_contrastive_lrs_are_overridable` + `test_h100_150m_contrastive_epoch_budget_is_batch_matched` (asserts bs×steps == 384000 and √-scaling off the bs=32 anchor). `validate_for_willi.sh` green 74 passed, 9/9 gates.
- [x] **6B-3** — (H100) Two bs=64 arms run (jobs 2359951 / 2359952). **RESULT: NULL.** √-matched `head_lr=4.24e-4` → in-training i2t R@10 peak 0.120 @ep11, final 0.116. Conservative `head_lr=3.0e-4` → peak/final 0.119. Prior best (head=6e-4) → peak 0.122, final 0.116. Spread 0.1–0.3pp vs SE ~0.57pp ⇒ **LR is not the binding constraint.** Secondary finding: lower LR removed the late-epoch retrieval rollover (arm B ends at its max, no overfit decline) without changing the plateau height — so overfitting is not binding either.
- [ ] **6B-4** — (H100) Authoritative `evaluate_cxr_retrieval.py` on both best-by-`val/total_loss` ckpts, for the record. Expected ~0.108–0.112 given in-training parity. Not a decision point.

---

## 2026-07-25 — Plateau post-mortem: what the seven nulls actually mean

**Lever tally: 7 clean nulls, 1 positive.**

| Lever | Δ i2t R@10 | Verdict |
|---|---|---|
| Stage-0 PPL 15.62 → 13.18 | flat | LM quality does not transfer to retrieval |
| 70M → 150M backbone | flat | capacity not binding |
| Negatives 32 → 128 | flat (bs=64 nominally ahead) | in-batch-negative thesis unsupported |
| Epochs 23 → 14 | +0.06pp | overfitting not binding |
| batch 128 vs 64 (epoch-matched) | 0.29pp spread | indistinguishable |
| head_lr 6e-4 → 4.24e-4 | flat | optimization not binding |
| head_lr 6e-4 → 3.0e-4 | flat | optimization not binding |
| **ViT unfreeze 0 → 2** | **+2.5pp** | **the only live lever — the only trainable thing not anchored to the teacher** |

### The `cos_text_teacher` = 0.57 claim is FALSIFIED — do not build on it

Two external reviews (2026-07-25) independently diagnosed the plateau as a **causal-vs-bidirectional representational ceiling**, evidenced by `cos_text_teacher` pinning at ~0.57 across every configuration. **Our own logs refute this:**

- Steps 0–1000 = KD-only warmup: CLIP is gated off (`lightning_module.py:1012-1016`), `α_kd_warmup=1.0`, and the **LM backbone is frozen** — only `projection_head` + `attn_pool` + `logit_scale` train (15.2M params). In that window `cos_text_teacher` reaches **0.874–0.892** (passes the ≥0.85 gate).
- It then *falls* to 0.60 → 0.57 **after** CLIP switches on at step 1000, and sits at 0.566 / 0.570 / 0.574 / 0.582 across a 2× LR range, both batch sizes, and both model scales.

A frozen causal SSM backbone with a small head hit **0.89** cosine against the bidirectional PubMedBERT teacher. 0.57 is therefore a **loss-weight equilibrium** between KD (α_post=0.3) and CLIP (β=1.0) pulling `z_text` toward targets that are themselves cos~0.5–0.7 apart in the joint space — exactly what the code comment at `:1086-1090` predicted. It is **not** an architecture ceiling, and its invariance to LR/batch/scale is the expected signature of an equilibrium, not of an optimization failure.

**Consequences carried into the phases below:**
1. `cos_text_teacher` must **never** be used as a gate on whether the text tower is architecturally adequate. Any experiment whose kill criterion is "did cos rise above 0.62" would draw a wrong conclusion.
2. **KD-anchor decay (6D-2) is promoted** — it attacks the equilibrium directly and is derived from our own data, not from literature.
3. Bidirectional encode (6E) is retained but **re-motivated**: the honest argument is report structure (the Impression at token ~300 recontextualizes the Findings at token ~40 and a causal encoder cannot propagate that backwards), *not* the cos number. This framing must survive into the writeup.

### Phase 6C — Measurement block (NO TRAINING) ✅ COMPLETE (job 2372055, 86 s wall)
Launch: `CKPT=<best 6B ckpt> sbatch scripts/run_phase6c_measurements.sh`. Ran on the head=3.0e-4 arm's best-by-`val/total_loss` ckpt (step 4750, 2.9465).

**Tower grid, N=3063, strict-index gt (authoritative protocol):**

| image tower | text tower | i2t R@1 | i2t R@5 | **i2t R@10** | t2i R@10 |
|---|---|---|---|---|---|
| stock ViT | BiomedCLIP text | 0.0039 | 0.0186 | **0.0340** | 0.0310 |
| stock ViT | student | 0.0023 | 0.0114 | **0.0232** | 0.0813 |
| fine-tuned ViT | BiomedCLIP text | 0.0065 | 0.0264 | **0.0431** | 0.0189 |
| **fine-tuned ViT** | **student** | **0.0180** | **0.0731** | **0.1172** | **0.1087** |

**Four hypotheses killed, one lever promoted:**

1. **Teacher parity — DEAD.** Stock BiomedCLIP scores **3.40%**; the student scores **11.72%**, i.e. **3.4×** the teacher. The 12% target is not above the anchor and Phase 6 is not a parity result. There is real headroom.
2. **KD anchor is a DRAG, not a ceiling.** BiomedCLIP's text tower is worth 4.31% even on the fine-tuned ViT, while the student is at 11.72%. So `alpha_kd_post=0.3` spends the whole post-warmup run pulling `z_text` toward a representation ~3× worse than the one CLIP is building. This is a stronger and more actionable version of what the external review guessed — **6D-2 is promoted to co-priority with 6D-1.**
3. **False negatives — DEAD (quantitatively).** At bs=64 only **19%** of batches contain a single false negative, mean **0.58 pairs out of a 4096-entry matrix**; at bs=128, 2.32 of 16384. Both external reviews ranked this a top-3 cause and called it "the direct explanation" for the flat negatives lever. It is not. **`MULTIPOS` dropped from the 6D-3 arm.**
4. **Metric artifact — DEAD.** Duplicates are 2.0% of the gallery (largest group 40 of 3063), oracle R@10 = **99.0%**, and dedup-aware R@10 differs from strict by 0.03pp. The templated-report ceiling both reviews warned about does not exist at this scale. Keep the strict metric as the headline.
5. **Text tower is not the weak half.** Swapping BiomedCLIP's text tower **in** costs 7.41pp (0.1172 → 0.0431). Per the pre-registered rule, **Phase 6E is deprioritised.**

**Caveat on 6C-2, state it in the writeup:** the fine-tuned ViT was co-trained with the student text tower, so that pairing is favoured by construction. The mitigating evidence is that fine-tuning still *helped* BiomedCLIP's own text tower (3.40 → 4.31), so the ViT did not drift into a space hostile to the teacher. The defensible claim is "substituting BiomedCLIP's text tower does not help", not "the student text tower is strictly better".

**Logical gap worth being precise about:** 6C-2 answers "is BiomedCLIP's text tower better than ours?" (no). It does **not** answer "would a bidirectional student beat a causal student?" — which is what 6E actually proposes. 6E is therefore *unsupported*, not *refuted*; it drops to an opportunistic cheap test after 6D, not a planned arm.

**Incidental observations for the writeup:**
- `stock ViT × student text` is strongly asymmetric (i2t 0.0232 vs t2i 0.0813) — hubness in the student text space relative to stock-ViT geometry. The co-trained pair is balanced (0.1172 / 0.1087). Evidence of genuine co-adaptation, not of a collapsed text space.
- The 6C load reported `1 missing key: ['logit_bias']` — expected and benign: the checkpoint predates the SigLIP parameter, it loads at its −10 init under `strict=False`, and `encode()` never reads it.
- Fine-tuning 2 ViT blocks lifts the *frozen-text* system 3.40 → 4.31 (+0.91pp), independently corroborating that the image side is where movement lives.

**Net: the binding constraint is the image representation.** Every text-side and optimization lever is null; the only two positives on record are both image-side (ViT unfreeze 0→2 = +2.5pp; ViT unfreeze with a frozen text tower = +0.91pp). Priority order in 6D reflects that.
- [x] **6C-1** — stock BiomedCLIP reference: **3.40%** i2t R@10.
- [x] **6C-2** — tower-swap 2×2 grid (table above).
- [x] **6C-3** — duplicate/false-negative audit: 2.2% train / 2.0% gallery duplicated, oracle R@10 99.0%, 0.58 false-neg pairs per bs=64 batch.
- [x] **6C-4** — R@1/R@5 surfaced; dedup-aware R@10 implemented and shown unnecessary (0.03pp).
Instrumentation and writeup evidence. **Per user decision 2026-07-25, 6D runs regardless of the 6C-1 result** — 6C does not gate 6D, it explains it and calibrates the writeup.
- [ ] **6C-1** — `scripts/reference_biomedclip_zeroshot.py`: stock BiomedCLIP (**its own text tower and image tower**) on the identical `train[90%:]` N=3063 protocol. Report i2t/t2i R@1/5/10 next to the 0.1113 student number. Published anchors put BiomedCLIP zero-shot at ~2–4% on comparable ~2.4k-study galleries, which would put the student at ~3× the teacher and imply real headroom — but the only number that counts is ours, on our protocol.
- [ ] **6C-2** — Tower-swap 2×2 in the same script: {student text, BiomedCLIP text} × {fine-tuned ViT, stock ViT}. Four numbers isolate which tower binds. If substituting BiomedCLIP's text tower barely moves R@10, text-side effort is misallocated and 6E should be dropped.
- [ ] **6C-3** — `scripts/audit_mimic_duplicates.py` (CPU only): exact + whitespace/case-normalised report-text grouping over `train[:90%]` and `train[90%:]`. Outputs (a) the oracle R@10 ceiling on the eval gallery under arbitrary tie-breaking, (b) expected false-negative rate per batch size {32,64,128}. Decides whether 6D-3's multi-positive mask is worth having and whether a dedup-aware R@10 belongs in the headline.
- [ ] **6C-4** — Reporting: surface i2t/t2i **R@1 and R@5** (already computed at `evaluate_cxr_retrieval.py:535-540`, just never carried into the state/writeup). R@1 ≈ 1.7% is far more sensitive to representation quality than R@10 and will show movement when R@10 does not. Add dedup-aware R@10 as a secondary metric.

### Phase 6D — Factorial lever block ✅ ARMS RUN (2026-07-26) — **ViT depth is the lever**

**In-training i2t R@10, N=3063, final @ep13 (peak in parens):**

| Arm | Lever | i2t R@10 | i2t R@1 | t2i R@10 | val/clip_loss | cos_teacher |
|---|---|---|---|---|---|---|
| D0 | control, vit=2 | 0.116 (0.120) | 0.017 | 0.113 | 2.897 | 0.570 |
| D1a | vit=4 | 0.132 (0.135) | 0.020 | 0.131 | 2.774 | 0.559 |
| D1b | vit=6 | 0.150 | 0.029 | 0.151 | 2.687 | 0.555 |
| **D1c** | **vit=12 (whole ViT-B/16)** | **0.168 (0.171)** | **0.030** | 0.165 | **2.585** | 0.544 |
| D2 | KD decay 0.3→0 over 2k | 0.118 (0.120) | 0.016 | 0.114 | 2.934 | **0.194** |
| D3 | SigLIP | 0.119 (0.120) | 0.016 | 0.120 | n/c | 0.546 |
| D5 | γ_simcse=0 | 0.122 | 0.016 | 0.113 | 2.872 | 0.586 |

**6D-1 is a decisive, monotone win.** 0.116 → 0.132 → 0.150 → 0.168 across unfreeze depth 2/4/6/12: **+5.2pp over control, ~9× the SE**. `val/clip_loss` falls monotonically alongside it (2.897 → 2.585) and R@1 nearly doubles, so this is generalization, not a tie-breaking or selection artifact — the eval images are never trained on. No OOM at bs=64 even with all 85.1M ViT params trainable. This corroborates the 6C conclusion exactly: **the image representation was the binding constraint all along.**

**6D-2 falsifies the 6C-derived KD prediction — record this honestly.** From 6C we argued the α_kd=0.3 anchor was dragging `z_text` toward a 4.31%-quality representation and that releasing it should help. The mechanism worked precisely as designed — `cos_text_teacher` collapsed 0.570 → 0.194, so the anchor genuinely released — and retrieval did **not** move (+0.2pp, well inside noise). `val/clip_loss` was marginally *worse* (2.934 vs 2.897), so at α=0.3 the KD term acts as a mild regulariser rather than a drag. **The KD-anchor hypothesis is now dead in both directions** (not a ceiling, not a drag) and should not be revisited.

**6D-3 (SigLIP, +0.3pp) and 6D-5 (γ_simcse=0, +0.6pp) are null.** Both under the 1.1pp bar. Combined with 6C-3 killing the false-negative premise, the entire objective-repair line of both external reviews is now empirically closed.

**Running tally: 10 nulls, 1 dominant lever.** Stage-0 PPL, model scale, negatives, epochs, batch, head_lr ×2, KD decay, SigLIP, SimCSE — all flat. ViT adaptation depth — monotone and large.
- [x] **6D-0** — control reproduces the Phase-6B recipe (0.116, matching the 6B arms).
- [x] **6D-1** — vit_unfreeze {4,6,12}: monotone, **the** result of this phase.
- [x] **6D-2** — KD decay: null; hypothesis retired.
- [x] **6D-3** — SigLIP: null.
- [x] **6D-5** — γ_simcse=0: null.
- [ ] **6D-4** — stack: now redundant. D2/D3/D5 are all null, so the "stack" is just D1c. Fold γ_simcse=0 into the 6G sweep as a free rider rather than running a separate arm.

### Phase 6G — ViT adaptation dose-response, continued ⏳ NEXT
Depth is exhausted at 12 (ViT-B/16 has 12 blocks), but "amount of image adaptation" = depth × LR × scope, and only depth has been swept. `vit_lr` has sat at **1e-6** the entire project — three orders of magnitude below the head LR — so the winning arm is one where the whole tower is unfrozen but barely moving.
- [ ] **6G-1** — `vit_lr` sweep at `VIT_UNFREEZE=12`: {3e-6, 1e-5, 3e-5}. Highest-value remaining experiment.
- [ ] **6G-2** — Scope: `_get_vit_blocks` returns transformer blocks only, so `patch_embed`, `cls_token`, `pos_embed`, the final norm and the visual projection stay frozen even at depth 12. Add an opt-in full-tower unfreeze.
- [x] **6G-3** — **AUTHORITATIVE, FULLY LR-MATCHED DOSE-RESPONSE — STRETCH TIER CLEARED.**

  All four arms at `bs=64`, `head_lr=4.24e-4`, `backbone_lr=1.41e-5`, 6000 steps (13.93 epochs), best-by-`val/total_loss` checkpoint, MIMIC `train[90%:]` N=3063:

  | `vit_unfreeze` | trainable ViT | i2t R@10 | i2t R@1 | i2t R@5 | t2i R@10 | paired cos | tier |
  |---|---|---|---|---|---|---|---|
  | 2 (D0) | 14.2M | 0.1107 | 0.0153 | 0.0637 | 0.1041 | 0.3824 | floor |
  | 4 (D1a) | 28.4M | 0.1319 | 0.0196 | 0.0738 | 0.1166 | 0.3962 | target |
  | 6 (D1b) | 42.5M | 0.1430 | 0.0206 | 0.0937 | 0.1394 | 0.4084 | stretch |
  | **12 (D1c)** | **85.1M** | **0.1714** | **0.0300** | **0.1032** | **0.1538** | **0.4230** | **stretch** |

  **Monotone in every column** — i2t R@1/R@5/R@10, t2i R@10, and paired cosine all rise with depth. **+6.07pp** from depth 2→12, ~10.6× the 0.57pp SE. Floor 0.1045 ✅, target 0.12 ✅, **stretch 0.14 ✅**.

  **Control validated:** D0 (0.1107) reproduces the standing Phase-6 baseline (0.1113) to within 0.06pp, so the curve is a clean single-variable sweep rather than an artefact of the earlier LR/epoch confounds.

  **Per-block returns diminish but do not saturate:** 1.06 pp/block (2→4), 0.56 (4→6), 0.47 (6→12). Depth is exhausted at 12, so the remaining dose axes are `vit_lr` and scope — hence 6G-1/6G-2.

  Historical context: the earlier vit=2 numbers (0.1113 at `head_lr=6e-4`, 0.1172 at `3.0e-4`) are *not* part of this curve — they used different head LRs. Cite the LR-matched table above.

  Note the in-training/authoritative reconciliation *inverted* versus Phase 6: authoritative 0.1714 now slightly **exceeds** the in-training final (0.168). In Phase 6 the val-loss minimum (step 4500) sat well before the retrieval peak (~6000), costing ~1pp at selection time; with the stronger image tower both curves peak together at ~4750, so selecting on `val/total_loss` no longer costs anything.
- [x] **6G-4** — **Indiana: 0.0485 i2t R@10** (t2i 0.0700, R@1 0.0094, paired cos 0.2730, N=743). Floor 0.0404 ✅ (target 0.055 not reached). **The cross-domain risk did not materialise** — unfreezing all 12 ViT blocks on 27.5K in-domain MIMIC pairs improved Indiana too, from the A100 baseline 0.0390 to 0.0485 (+0.95pp). Deep image adaptation is not an in-domain/cross-domain trade here; it is a genuine representation improvement. Phase 7 gate cleared.
- [ ] **6G-5** — **Re-run D1c with `SELECTION_SPLIT=true`.** Arm-level comparison used `val/total_loss` on `train[90%:]`, which is the eval gallery — that is test-set selection at the arm level. The effect is 9× SE so it is not noise-mining, but the headline architectural claim of the thesis should be confirmed under a clean protocol. This is what Phase 6F was built for.
Six one-at-a-time nulls have made single-lever probing expensive per bit of information. Run D1–D3 in parallel for attribution **and** D4 stacked for the number. ~3.5 h/arm on one H100. Gate: **>1.1pp over control** (SE ~0.57pp at p≈0.11, n=3063) or it is noise.
Launch: `./scripts/submit_phase6d_arms.sh` (dry run) → `--submit`, or paste its sbatch lines directly. Every lever is env-overridable in `train_biomedclip_kd_h100.sh` and **defaults to the Phase-6B recipe**, so an unmodified invocation *is* 6D-0.

> **Cluster constraint (2026-07-26):** the aisc login node refuses `bash <script>` ("This command is not allowed on the login node!"). Use `./scripts/...`, an `srun --pty bash` session, or paste the `sbatch` lines — `sbatch` itself is allowed on the login node, which is how Phase 6C was submitted.
- [ ] **6D-0** — Control: bs=64, LR-matched (`head_lr=4.24e-4`, `backbone_lr=1.41e-5`), 6000 steps, canonical recipe. Baseline for this block.
- [ ] **6D-1** — `vit_unfreeze_blocks` ∈ {4, 6, 12}. The only lever with a measured positive (+2.5pp at 0→2). Config-only — `_get_vit_blocks()` (`:441-457`) is already generalised to any depth and `configure_optimizers` (`:899-909`) already builds the 4th param group. Watch `vit_lr=1e-6` — consider layer-wise decay only if 12 destabilises.
- [ ] **6D-2** — **CO-PRIORITY after 6C.** KD-anchor decay: linear `alpha_kd_post → alpha_kd_floor` over `kd_decay_steps` post-unfreeze (default floor 0.0, decay 2000). 6C showed the anchor is not holding the student at parity — it is pulling toward a 4.31% representation while CLIP builds an 11.72% one. Watch `pos_cosine_mean` and `val/clip_loss` for space collapse; arm **D2b** is the `alpha_kd_floor=0.05` fallback.
- [ ] **6D-3** — **SigLIP only** — `MULTIPOS` dropped after 6C-3 measured 0.58 false-negative pairs per bs=64 batch (19% of batches contain even one). The multi-positive mask has nothing to fix on this dataset. SigLIP survives on its own rationale (pairwise sigmoid, no global softmax, decoupled from batch size) but with **downgraded expectations** — its headline justification in both external reviews was the false-negative problem that does not exist here. The mask stays implemented and `false_neg_rate` stays logged; both are re-usable if Phase 7 multi-source data changes the duplication profile.
- [ ] **6D-4** — Stack: best-of-6D-1 + 6D-2 + 6D-3. The shot at the 12% target.
- [ ] **6D-5** — Optional cheap ablation: `gamma_simcse=0`. SimCSE pulls the same projection head toward uniformity using two dropout views of one text, competing with CLIP. May be free gain.

### Phase 6E — Bidirectional text encode ⏸ DEPRIORITISED by 6C-2 (code shipped, unscheduled)
The pre-registered gate fired against it: swapping BiomedCLIP's text tower in **costs 7.41pp** (0.1172 → 0.0431), so the student text tower is not the weak half and text-side capacity is not where the plateau lives. Code is implemented, tested and inert behind `BIDIRECTIONAL=false` — run it opportunistically after 6D if image-side levers stall, not as a scheduled arm.
Be precise about what was and was not shown: 6C-2 answers "is BiomedCLIP's text tower better than ours?" (no). It does **not** answer "would a bidirectional student beat a causal student?" — 6E is *unsupported*, not *refuted*. Motivated by report structure, **never** by `cos_text_teacher` (falsified above).
- [ ] **6E-1** — `bidirectional` flag on `HybridTextEncoder.encode`: forward pass + pass over the length-aware reversed sequence (right padding preserved), reverse-pass states gathered back to original positions, averaged before pooling. Costs 2× text-encode FLOPs, trivial next to the ViT. **Checkpoint-compatible — no new parameters, so existing ckpts and `evaluate_cxr_retrieval.py` keep working.**
- [ ] **6E-2** — If 6E-1 wins: the in-layer version (bidirectional scan inside each Mamba/mLSTM block, concatenate directions, project back to `dim`). That is the publishable contribution; the cheap version exists to test the hypothesis before committing to it.

### Phase 6F — Eval-protocol fix (do regardless) ⏳ NOT STARTED
- [ ] **6F-1** — Carve a **disjoint selection split** out of `train[:90%]` (e.g. `train[:85%]` train / `train[85%:90%]` select / `train[90%:]` test). Today `validation_split == test_split == train[90%:]`, so any checkpoint selection is selection-on-test. Currently mitigated by selecting on `val/total_loss` rather than retrieval, but a reviewer will still flag the shared split. Fixing it also legitimises checkpoint-on-retrieval, which is otherwise permanently banned.

**REJECTED — do NOT re-litigate** (2026-07-21 recipe review + 2026-07-25 review of two external plateau analyses):
- `vit_unfreeze_blocks: 0` — already run (jobs 1942/1948/1949): MIMIC **10.45% → 7.97%**, Indiana identical 3.90%. Freezing loses in-domain and recovers nothing cross-domain.
- `freq_kd: true` — already run (jobs 1922/1923 vs 1930/1931): Indiana **3.90% → 2.96%**. Cross-domain regression; attacks the Phase-7 gate.
- Checkpoint/early-stop on `val/retrieval_i2t_R@10` — **selection-on-test** while `validation_split == test_split == train[90%:]`. Unblocked only by 6F-1.
- "The 0.1113 is a *last*-checkpoint artifact; best-val recovers ~0.9pp for free" — **factually wrong.** 0.1113 was measured on `contrastive-step=002750-val/total_loss=3.6083.ckpt`, which *is* the best-by-`val/total_loss` checkpoint (`train_contrastive.py:801-806`, `monitor=val/total_loss`, `mode=min`). The 0.120–0.122 in-training peak is on the same 3063 pairs the eval uses — chasing it is the banned selection-on-test.
- **XBM / cross-batch memory queue** — this is the MoCo queue, already ablated in-repo and found harmful post-KD-warmup; `moco_queue_size=0` is canonical and is in "lessons carried". Re-proposing it with a different name does not make it new evidence.
- **Swapping the image backbone to RAD-DINO / MedSigLIP** — breaks `assert img_out == model.embed_dim` (`:419-422`) and removes the BiomedCLIP joint space that the KD teacher targets. Not a lever; a different project. Also dilutes an MSc contribution that is about the *text* tower.
- **Image resolution 224 → 336/448/512** — BiomedCLIP ViT-B/16 position embeddings are fixed at 224; interpolating them perturbs the frozen joint space the whole design depends on. Poor cost/benefit here.
- **Two-stage text-only distillation to raise `cos_text_teacher`** — optimises a number that already reaches 0.89 under pure KD. Not the bottleneck.
- **"Switch from last-token to mean pooling"** — moot; the v2 configs use attention pooling (`pooling_strategy: attention`), never last-token.
- **`cos_text_teacher` as an architecture-adequacy gate** — falsified above. Any experiment gated on it draws a wrong conclusion.

### Phase 7 — Multi-source CXR data diversification → Indiana lever
Only proven Indiana lever. **IU-Xray EXCLUDED from training (= Indiana eval; zero-leakage).**
- [ ] **7A** — `configs/dataset/cxr_multi.yaml` (NEW): MIMIC + CheXpert + PadChest + VinDr-CXR + per-source weights.
- [ ] **7B** — `scripts/train_contrastive.py`: generalize `MIMICJointDataset`/`load_mimic_cxr` (`:339-451`) → `CXRJointDataset` with per-source **text adapter** (free-text passthrough / label→prompt template / translated-EN). Mixed-source collate; dual-tokenize unchanged.
- [ ] **7C** — Re-run Phase-6 best recipe on `cxr_multi`.
- [ ] **7D** — Gate: Indiana i2t R@10 ≥ 4.04% floor (target 5.5%) while MIMIC holds ≥ Phase-6.

### Phase 8 — Full eval + comparison + writeup
- [ ] **8A** — Authoritative MIMIC + Indiana + STS (BIOSSES/STS-B) + PubMed PPL on best 150M ckpt.
- [ ] **8B** — `analysis/h100_scaling_results.md` (NEW): 70M-A100 vs 150M-H100 table; per-lever ablation; honest limitations.
- [ ] **8C** — Update `h100_scaling_state.json` final verdict + best ckpt path.

---

## Verification (each phase gates on)
1. `bash scripts/validate_for_willi.sh` exits 0 (3.9-syntax hygiene kept, forward-compatible) + H100-env stack smoke on py≥3.10 (Phase 2).
2. New phase test passes.
3. Numerical: forward/backward finite on CPU + H100 smoke; grad-norm bounded; no NaN 50+ steps; i_gate<cap.
4. SLURM kill-gates pre-declared (Phase 5 PPL; Phase 6 cos_text_teacher/clip_loss/R@10).
5. Reconcile in-training vs authoritative eval numbers (refactor caught 4 latent bugs this way).

## Resumability contract
1. Read `H100_SCALING_PLAN.md` + `h100_scaling_state.json` at session start.
2. Resume at `h100_scaling_state.json["current_phase"]`. Checkboxes here = ground truth.
3. After every state change: tick checkbox + update `last_updated` (ISO 8601) + append 1-line `notes` entry.
4. If state JSON missing locally, regenerate from this file's checkbox state (gitignored on purpose).
5. Never re-run a checkpoint-producing phase (5,6,7) without first reading its log + logging a verdict.

## Lessons carried from the completed refactor (do not repeat)
- CLIP negatives are in-batch only — grad-accum does NOT add negatives; per-step batch is the real lever.
- `norm_topology` must be threaded into every `HybridConfig` builder (train/distill/contrastive) — silently dropped = wrong FFN forward.
- Eval must auto-detect `layer_pattern` + `norm_topology` from checkpoint; must load fine-tuned `image_encoder.*` (fresh-ViT load read 1.89% vs true 10.94%).
- MoCo queue post-KD-warmup is harmful → keep `moco_queue_size=0`.
- freq-decoupled KD hurt Indiana → stays off (canonical).
- Always reconcile in-training vs authoritative eval numbers before citing.

## Unresolved questions
- CheXpert/VinDr label→prompt template wording — finalize in Phase 7 (start CheXzero-style; iterate if MIMIC regresses).
- 150M `learning_rate`: **4.0e-4 chosen** (conservative √-width scale; bump to 5e-4 if Phase-5 loss descends slowly).
- Whether willi/A100 remains a target after H100 migration (if retired, drop py3.9 guards + `validate_for_willi.sh`).
