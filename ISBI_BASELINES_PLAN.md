# ISBI baselines plan: foundation-model retrieval floors + reviewer-risk experiments

## ▶ RESUME HERE (new session: read this block, then `isbi_baselines_state.json`)

**Goal.** Make the ISBI 2026 paper on **mLMamba** (attention-free report decoder: 9 Mamba-2/SSD +
3 mLSTM layers, 184.2M, matched 183.4M Transformer, 3 seeds, MIMIC-CXR official test split n=2,663)
robust to the supervisor's comments and to likely reviewer objections, with measured numbers only.
Deadline 2026-10-26. The paper lives OUTSIDE this worktree: main checkout
`../hybrid_model_mamba_xlstm/ISBI Paper /` (untracked; `figs/` is DUA data, never commit).
Current paper file: **`Method and Evaluation V9.tex`** (the user's; earlier: v6, v7).

**Where things run.**
| place | what |
|---|---|
| this worktree `../isbi_baselines_wt`, branch `isbi_baselines` (off `h100_efficiency` 5c45527) | all code + plan + state |
| cluster `hpi-hpc:~/hybrid_isbi` (git worktree of `~/hybrid_mamba_xlstm`, branch `isbi_baselines`) | jobs are submitted FROM here; `.venv`, `.venv_chexbert`, `outputs`, `results` are symlinks to the main checkout |
| cluster `~/hybrid_mamba_xlstm/results/`, `analysis/` | all job outputs (MIMIC-derived files stay there, rule R6) |

Code reached the cluster by `rsync` (not git). After this branch is pushed, sync the cluster with
`git -C ~/hybrid_isbi fetch && git -C ~/hybrid_isbi reset --hard origin/isbi_baselines` (files are
identical, so nothing is lost). Login node runs no project Python: everything goes through `sbatch`.

**Done (all verified, see phases below and `analysis/isbi_reviewer/FINDINGS.md`).**
- B0-B6: five retrieval floors (BiomedCLIP, CLIP, PubMedCLIP, XrayCLIP, MedSigLIP), scored + bootstrapped
  vs mLMamba; Table 3 + floor figure in the paper. XrayCLIP/MedSigLIP floors beat mLMamba on all CheXbert F1.
- B7 sequence length: reports median 113 tokens; with patient history median 870, p95 ~4,000.
  Writing a report: Transformer 1.3x faster at 132 tokens; mLMamba 1.6x / 2.1x / 5.0x faster at
  1,024 / 4,096 / 16,384 tokens of context, decode memory 8x / 27x / 105x smaller.
- B9 error analysis: decoders rarely add findings (~10% vs ~15% floors), miss ~60%;
  pneumothorax added 1.0% / missed 96%. Transformer identical -> not an SSM effect.

**Next step.** B8: XrayCLIP-encoder retraining is QUEUED (jobs 2624239-2624256, train -> beam-3 eval
-> CheXbert, mLMamba + Transformer x seeds 42/43/44; train expected to start 2026-10-10). When done:
1. `sacct -j $(seq -s, 2624239 2624256)` - all COMPLETED? (eval ~5 h each, 12 h limit)
2. check each train log prints `Report image encoder: xrayclip`, decoder `Missing keys` ~0, and the
   eval log prints `report image encoder = xrayclip` (encoder is resolved from run_metadata.json)
3. bootstraps (`scripts/bootstrap_compare_h100.sh`, PER_LABEL=true), per seed s:
   m3_xrayclip vs m3 (BiomedCLIP), m3_xrayclip vs transformer_xrayclip, m3_xrayclip vs floor_xrayclip
   (dirs: `results/report_gen_{m3,transformer}_xrayclip_test_split_s{42,43,44}`,
   `results/report_gen_m3_test_split_s*`, `results/isbi_floor_xrayclip_test_split`)
4. rerun `scripts/isbi_reviewer_cpu.sh` with SKIP_STATS=true and the B8 systems added (error analysis)
5. write results into FINDINGS.md + the paper; tick B8-B/B8-C.

**Things to check / known traps.**
- Never change a measured number in the paper (user asked twice; refused). Efficiency figure uses
  only E1-E-protocol numbers; at 2,048 tokens mLMamba is measured SLOWER (17.80 vs 16.42 ms).
- V9 errors still to fix in the paper: exact match must be 0.0452 / 0.2242 (not 0.0456 / 0.2247);
  16K memory is 7.17 vs 7.15 GB with torch.compile (7.09 is an old eager run); "memory grows more
  slowly" is unsupported (0.643 vs 0.644).
- Attention decode: `AttentionBlock.step` pins SDPA to flash/efficient/math; cuDNN re-plans every
  step (~15 ms/layer) and made the Transformer look 50x slower (job 2624265, discarded).
- Concurrent `profile_e1_confirm_h100.sh` jobs must use different SCRATCH_ROOT (they rm their cache).
- mLMamba end-to-end decode numbers assume a chunked prefill that returns the state; the code does
  not return it yet (M6 follow-up) - say so wherever the 1.6-5x figures are used.
- `*.md` is gitignored in this repo: add plan/analysis markdown with `git add -f`.
- The HF token for gated MedSigLIP is on the cluster at `~/.hf_token` (600); the local `.env` copy is
  excluded via `.git/info/exclude` - never commit it.
- No MIMIC report text off the cluster or into a chat (R6); qualitative cases:
  `results/isbi_error_analysis/qualitative_mLMamba_s42.md` (cluster only).


Branch `isbi_baselines` (off `h100_efficiency` at 5c45527). State file `isbi_baselines_state.json`.
Driven with `venv/bin/python scripts/mamba3_state.py --plan isbi show|tick|note|phase|sync`.

## Why

Supervisor review of the ISBI draft (2026-10-04):

> compare more decoder architectures in both Figure 3 and the merged Figure 4/5 ... at least four
> relevant foundation models for image-text retrieval that are based on Transformer architectures.
> You should be able to download their pretrained weights relatively easily from Hugging Face.

User decision (2026-10-04): add them as **retrieval floors**. Each foundation model's image tower
embeds the 191,462 training images and the 2,663 test images. Each test study gets the report of its
nearest training image. That report is scored like a generated report. This is the same procedure as
today's floor (stock BiomedCLIP, `scripts/evaluate_report_generation.py::run_retrieval_baseline`,
line 604), with the encoder swapped. No training. No change to any published number.

## Rules

- **R1 parity.** The new encoder option defaults to stock BiomedCLIP, and that default must reproduce
  `results/retrieval_floor_test_split/hyps.txt` **byte for byte** before any other encoder runs.
- **R2 leakage.** An encoder whose pretraining included MIMIC-CXR **test** studies is dropped or
  reported with an explicit flag. Training on the official MIMIC-CXR train split is allowed and must
  be stated in the paper (it is the same data our models use).
- **R3 same scoring.** Same CheXbert scorer (`score_chexbert_h100.sh`, f1chexbert rrg mode), same
  text metrics, same paired bootstrap (`bootstrap_compare_h100.sh`, 1000 resamples, seed 0) against
  mLMamba seeds 42/43/44. Table 2 marking rule unchanged: bold 3/3 seeds, dagger 2/3.
- **R4 same cost protocol.** bf16, batch 4, one H100, 3 warmup + 10 timed iterations, peak memory
  reset after warmup (`scripts/performance_profile.py::measure_point` conventions).
- **R5 DUA.** Galleries, hyps, refs and embeddings stay under `results/` / scratch on the cluster.
  Nothing MIMIC-derived is committed.

## Candidate encoders (to be vetted in B1)

| name | HF id | domain | notes to verify |
|---|---|---|---|
| BiomedCLIP | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` | biomedical (PMC-15M) | today's floor, R1 anchor |
| CLIP | `openai/clip-vit-base-patch16` | general web | public |
| PubMedCLIP | `flaviagiammarino/pubmed-clip-vit-base-patch32` | biomedical (ROCO) | public |
| MedSigLIP | `google/medsiglip-448` | medical incl. CXR | gated (HAI-DEF terms, user must accept on HF); check MIMIC use |
| BioViL-T | `microsoft/BiomedVLP-BioViL-T` | CXR (MIMIC-CXR) | image loader needs `hi-ml-multimodal`; check split |
| XrayCLIP | `StanfordAIMI/XrayCLIP__vit-b-16__laion2b-s34b-b88k` | CXR (MIMIC + CheXpert) | backup if one above fails; check split |

Target: at least 4 new encoders plus BiomedCLIP.

## Phases

### B0 — Plan and approval

- [x] **B0-A** Plan + state file written on branch `isbi_baselines`; `isbi` registered in `scripts/mamba3_state.py` PLAN_SETS.
- [x] **B0-B** User approves the plan and the encoder list (gate: nothing below runs before this).

### B1 — Vet the encoders

- [x] **B1-A** For each candidate: HF availability, licence, gating, loader (open_clip / transformers / hi-ml), input size and normalisation. Record in `analysis/isbi_baselines/encoders.md`.
- [x] **B1-B** For each candidate: pretraining data and whether it overlaps the MIMIC-CXR test split (R2), with the source (paper or model card section).
- [x] **B1-C** Final list of at least 4 new encoders fixed before any embedding job runs.

### B2 — Code

- [x] **B2-A** `--floor-encoder` option in `run_retrieval_baseline` (default `biomedclip`), an encoder registry (load, preprocess, embed), each encoder with its own preprocessing.
- [x] **B2-B** CPU tests: registry keys, default unchanged, nearest-neighbour core on a fake encoder; `tests/test_willi_parity.py` assertion for the new wrapper env var.
- [x] **B2-C** `scripts/isbi_floor_cost.py` (moved in from the uncommitted `~/isbi_tmp/floor_cost.py`) takes `--encoder`; R4 protocol.
- [x] **B2-D** `bash scripts/validate.sh` exits 0.

### B3 — Runs

- [x] **B3-A** R1 parity job: default encoder reproduces `results/retrieval_floor_test_split/hyps.txt` byte for byte.
- [x] **B3-B** One floor job per new encoder on the official test split (n=2663) into `results/isbi_floor_<name>_test_split/`.
- [x] **B3-C** CheXbert scoring (with label dump) for each new floor.

### B4 — Statistics

- [x] **B4-A** Paired bootstrap of each new floor against mLMamba seeds 42/43/44 (`analysis/bootstrap_m3_vs_floor_<name>_seed4{2,3,4}.md`).
- [x] **B4-B** Summary table: every model on all 10 metrics, marks per R3, in `analysis/isbi_baselines/summary.md`.

### B5 — Cost

- [x] **B5-A** One cost job per encoder (R4), into `analysis/efficiency_isbi_short/floor_cost_<name>.json`.
- [x] **B5-B** Short-length decoder points (job 2601749) and the BiomedCLIP floor cost (job 2601754) read and checked.

### B6 — Paper

- [x] **B6-A** Table 3: every floor on all 10 metrics, marked against mLMamba (R3). New floor graph: clinical scores and cost per floor, with mLMamba for reference.
- [x] **B6-B** Merged efficiency figure: decoder curves 256 to 16,384 (decoders only, per the 2026-10-04 decision).
- [x] **B6-C** Evaluation section names the encoders and their pretraining data. Results text interprets the floors.
- [x] **B6-D** Paper compiles with no undefined references and no overfull boxes.

## Decisions (user, 2026-10-04)

- Plan and encoder list approved.
- The new floors stay **out of Table 2**. They get their **own table** (Table 3: every floor on all metrics, marked against mLMamba per R3).
- The floors get a **new graph of their own** (clinical scores and cost of each floor next to mLMamba). The merged efficiency figure stays decoders only.
- MedSigLIP licence accepted by the user. The user places the HF token on the cluster at `~/.hf_token` (mode 600); jobs read it into `HF_TOKEN` and never echo it.


## Reviewer-risk experiments (opened 2026-10-08, user: "run the experiments and give me the results")

Three anticipated reviewer objections to the V9 draft. Same rules R1-R5. Additional rule **R6**: no
MIMIC report text is pulled off the cluster or into a chat; qualitative examples are written to a
cluster file for the user to read, only aggregate numbers leave it.

### B7 — Sequence length: is attention-free justified for short reports?

- [x] **B7-A** Context statistics (CPU, aggregate only): report token lengths (train/test), and per test study the number of earlier studies of the same patient and the token count of all their reports (the context a history-aware report would read).
- [x] **B7-B** KV-cache `step()` for the attention mixer so the Transformer gets a fair cached decode; test: cached beam search token-identical to the uncached path on a tiny model.
- [x] **B7-C** Decode benchmark on one H100: per-token decode latency and peak memory for both decoders, beam 3, at context lengths 132 (today's report), 1,024, 4,096 and 16,384, batch 1 and 16 reports. End-to-end time per report = prefill (forward pass, already measured) + 100 decode steps.

### B8 — Image encoder: does a chest X-ray encoder fix the clinical gap?

- [x] **B8-A** `report_image_encoder` option (default `biomedclip`, unchanged) in ReportGenerationLightningModule, train_report_generation.py, the wrapper and the evaluation loader (encoder + its image size/normalisation read back from run_metadata.json). Tests: default path unchanged, HF patch grid shapes.
- [ ] **B8-B** Stage 3 with the frozen stock XrayCLIP encoder (512 px, 1,025 patch tokens) for mLMamba and the Transformer, seeds 42/43/44, otherwise the published recipe.
- [ ] **B8-C** Beam-3 decode on the test split, CheXbert, paired bootstraps against: the BiomedCLIP-encoder mLMamba (same seed), the XrayCLIP-encoder Transformer (same seed), and the XrayCLIP floor.

### B9 — Error analysis: hallucination or omission?

- [x] **B9-A** Per finding, from the CheXbert label dumps: false-positive rate among reference-negative studies (hallucination) and false-negative rate among reference-positive studies (omission), mean positives per report, for mLMamba (3 seeds), Transformer (3 seeds), floors, and the B8 arms. Aggregate only.
- [x] **B9-B** Qualitative file on the cluster (R6): for acute findings (pneumothorax, consolidation, pneumonia, edema, pleural effusion), a few hallucination and omission cases with study ids and texts, for the user to review and excerpt.
