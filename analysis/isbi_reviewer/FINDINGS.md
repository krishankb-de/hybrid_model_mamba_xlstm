# ISBI reviewer-risk experiments: findings (B7, B8, B9, B10 complete)

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

## B8 Image encoder (train 2624239-54, eval/CheXbert 2624240-56, bootstraps 2638560-66 + 2646468-72, errors 2646473)

Stage 3 retrained with the frozen stock XrayCLIP tower (512 px) instead of the adapted BiomedCLIP
tower, otherwise the published recipe. All 6 trains: `Report image encoder: xrayclip`, missing
keys 0, 12k steps. Test split n=2,663, beam 3.

| system | 14-micro | 14-macro | 5-micro | 5-macro | ROUGE-L | BLEU-4 |
|---|---|---|---|---|---|---|
| mLMamba BiomedCLIP (3 seeds) | .448 ±.019 | .272 ±.012 | .504 | .417 | .1953 | |
| mLMamba XrayCLIP (3 seeds) | .463 ±.022 | .294 ±.013 | .500 ±.032 | .414 ±.023 | .2023 | .0622 |
| Transformer BiomedCLIP (3 seeds) | .444 ±.015 | .269 ±.011 | .503 | .417 | | |
| Transformer XrayCLIP (3 seeds) | .477 ±.016 | .303 ±.010 | .529 | .444 | .2005 | .0606 |
| XrayCLIP floor | .5095 | .3631 | .555 | .492 | .1748 | |

Per seed, mLMamba XrayCLIP: 14-micro .484/.463/.441, 14-macro .305/.296/.280, ROUGE-L .2003/.2033/.2032.
Per seed, Transformer XrayCLIP: 14-micro .491/.480/.460, 14-macro .309/.309/.292, ROUGE-L .1991/.2004/.2020.

Paired bootstraps (1000 resamples, per seed; analysis/bootstrap_m3_xrayclip_vs_*_seed4X.md):

- **XrayCLIP vs BiomedCLIP mLMamba (same seed):** wins ROUGE-L, BLEU-1, BLEU-4, CheXbert-14 micro
  (+.012 to +.018) and 14-macro (+.021 to +.024) at 3/3 seeds; 5-label micro/macro and exact match
  tie at 3/3. Per label: Support Devices +.06 to +.08 at 3/3, Pneumothorax +.12 to +.14 at 2/3,
  Lung Opacity at 2/3; Pleural Effusion loses at s43.
- **mLMamba XrayCLIP vs XrayCLIP floor:** decoder wins ROUGE-L (+.025 to +.029) and BLEU-4
  (+.020 to +.022) at 3/3, exact-match-5 at 2/3; floor wins all four CheXbert F1 at 3/3
  (14-micro −.025 to −.069, 14-macro −.058 to −.083). Per label the floor wins 8 findings at 3/3
  (Lung Opacity, Pleural Other, Lung Lesion, Consolidation, Pneumonia, Pleural Effusion,
  Atelectasis, Edema); mLMamba wins Cardiomegaly at 2/3.
- **Transformer XrayCLIP vs Transformer BiomedCLIP (same seed):** wins all four CheXbert F1 at 3/3
  (14-micro +.032 to +.035, 14-macro +.032 to +.036), BLEU-1 at 3/3, ROUGE-L and BLEU-4 at 2/3.
  The encoder helps the Transformer about twice as much on CheXbert-14 as it helps mLMamba.
- **mLMamba XrayCLIP vs Transformer XrayCLIP (same seed):** Transformer wins 5-macro at 3/3,
  14-micro and 5-micro at 2/3 (−.017 to −.019); 14-macro ties at 3/3; mLMamba wins ROUGE-L and
  BLEU-1 at 1/3 (s43); everything else ties. Per label mLMamba wins Fracture at 3/3, the
  Transformer wins Edema and Atelectasis at 2/3.

**Error analysis with XrayCLIP (analysis/isbi_reviewer/error_analysis_b8.json).**

| system | wrongly added (overall) | missed (overall) | findings/report (ref 3.13) | studies with an added acute finding | pneumothorax added / missed |
|---|---|---|---|---|---|
| mLMamba XrayCLIP s42/43/44 | .094/.086/.084 | .570/.605/.627 | 2.27/2.09/1.99 | 17.8/13.6/15.6% | 0.7–0.8% / 87–88% |
| Transformer XrayCLIP s42/43/44 | .094/.095/.094 | .563/.574/.596 | 2.30/2.27/2.19 | 22.8/18.9/19.0% | 0.9–1.3% / 82–89% |
| XrayCLIP floor | .149 | .489 | 3.07 | 40.5% | 4.2% / 78.6% |

With BiomedCLIP both decoders missed 92–100% of the 84 pneumothoraces; with XrayCLIP 82–89%
(9–15 found instead of 0–7). With the same encoder mLMamba adds an acute finding to fewer
studies than the Transformer at every seed (−3 to −5 points), but it also names fewer findings
per report and misses more, which is the same trade seen in its lower CheXbert recall: it is
the more conservative writer. No CI on these rates.

**Reading.** A chest X-ray encoder improves both decoders on text and CheXbert, so part of the
floor's CheXbert lead was the encoder. For mLMamba it closes about a quarter of the gap to the
XrayCLIP floor (14-micro/macro gap .061/.091 with BiomedCLIP, .047/.069 with XrayCLIP); the
Transformer gains more and ends closer (.033/.060). The rest is not the encoder: with the same
image features, nearest-neighbour retrieval of a whole human report still labels better than
either generator, while both generators write text closer to the reference and add far fewer
findings that are not there. With XrayCLIP the architectures are no longer level on CheXbert:
the Transformer is ahead on 5-macro at 3/3 and on 14-micro/5-micro at 2/3. The BiomedCLIP
parity result of the paper therefore holds for the published encoder only.

## B10 Published report generators on our test split (decodes 2680132, 2680170, 2680173, 2680240, 2680247; CheXbert 2680105-10, 2680149-54, 2680242-46, 2680264-70, 2680274-76; text 2680310; bootstraps 2680277-309)

Released checkpoints, each with its own preprocessing, tokenizer and released decoding settings,
on our 2,663 test studies and the same image per study (our 320x320 frontal JPEG). Refs of all
five runs match our refs row for row (0 mismatches). Setup per model: `ISBI_BASELINES_PLAN.md` B10.
- **R2Gen vocabulary.** R2Gen, R2GenCMN and R2Gen-Mamba build their word list from Chen et al.'s
  annotation file. That file is access-restricted, and the checkpoints do not store the list. We
  rebuilt the file from MIMIC (official split, one entry per image of a study with a findings
  section). It gives 270,790 / 2,130 / 3,858 images, exactly the published counts. The vocabulary
  (count >= 10) has 4,335 tokens, which matches the embedding rows (4,336) of all three checkpoints.
- **R2Gen-Mamba scan.** It runs mamba_ssm's own pure-PyTorch reference scan
  (`selective_scan_ref`), not the CUDA kernel.
- **Beam sizes.** These follow the released test scripts: R2Gen-Mamba 2 and R2Gen 2, although
  both papers say 3; R2GenCMN 3; CvT2DistilGPT2 4; CXRMate 4.

**Protocols.**
- **(a) Ours.** Refs are "Findings: ... Impression: ...", n=2,663, hypotheses as generated.
- **(b) Findings only.** n=**1,946** studies (of 2,663) whose reference has a findings section;
  ref = the findings section. Hypothesis: our decoders and CXRMate use their findings part, the
  findings-only generators their whole output. Both sides are normalised with R2Gen's
  `clean_report_mimic_cxr`.
- **(b-full) Sensitivity.** As (b), but our hypothesis (and CXRMate's) is the whole report with
  the section labels removed.

**Why (b-full) exists.** Our decoders leave the findings section empty in 1,516 (mLMamba s42) and
1,537 (Transformer s42) of 2,663 reports, while only 717 references have no findings. Among the
1,946 (b) studies, mLMamba's findings part is empty in 1,014 / 969 / 935 (s42/43/44). The clinical
content of those reports sits in the impression.

| model | input | target | protocol | n | ROUGE-L | BLEU-4 | 14-micro | 14-macro | 5-micro | 5-macro |
|---|---|---|---|---|---|---|---|---|---|---|
| mLMamba (3 seeds) | 1 frontal, 224 | F+I | (a) | 2,663 | .1953 ±.0029 | .0579 ±.0009 | .4480 ±.0188 | .2715 ±.0121 | .5044 ±.0257 | .4170 ±.0164 |
| Transformer (3 seeds) | 1 frontal, 224 | F+I | (a) | 2,663 | .1952 ±.0021 | .0575 ±.0016 | .4443 ±.0153 | .2692 ±.0106 | .5032 ±.0226 | .4170 ±.0165 |
| R2Gen-Mamba | 1 image, 224 | F | (a) | 2,663 | .1596 | .0362 | .4212 | .2525 | .4556 | .3677 |
| R2Gen | 1 image, 224 | F | (a) | 2,663 | .1507 | .0349 | .3653 | .2482 | .3540 | .3068 |
| R2GenCMN | 1 image, 224 | F | (a) | 2,663 | .1481 | .0301 | .4370 | .2828 | .4975 | .4380 |
| CvT2DistilGPT2 | 1 image, 384 | F | (a) | 2,663 | .1420 | .0225 | .2698 | .1912 | .2646 | .2304 |
| CXRMate single-TF | 1 image, 384 | F+I | (a) | 2,663 | .1711 | .0366 | .2741 | .2070 | .3068 | .2740 |
| mLMamba (3 seeds) | | F part | (b) | 1,946 | .1452 ±.0062 | .0280 ±.0046 | .2053 ±.0055 | .1340 ±.0055 | .2295 ±.0145 | .1951 ±.0138 |
| Transformer (3 seeds) | | F part | (b) | 1,946 | .1427 ±.0085 | .0256 ±.0040 | .1995 ±.0204 | .1292 ±.0166 | .2120 ±.0476 | .1786 ±.0403 |
| R2Gen-Mamba | | F | (b) | 1,946 | .2801 | .1016 | .4181 | .2549 | .4578 | .3740 |
| R2Gen | | F | (b) | 1,946 | .2636 | .0935 | .3658 | .2501 | .3592 | .3124 |
| R2GenCMN | | F | (b) | 1,946 | .2652 | .0856 | .4333 | .2836 | .4996 | .4380 |
| CvT2DistilGPT2 | | F | (b) | 1,946 | .2353 | .0679 | .2985 | .2078 | .2955 | .2560 |
| CXRMate single-TF | | F part | (b) | 1,946 | .2247 | .0648 | .2872 | .2104 | .3307 | .2898 |
| mLMamba (3 seeds) | | whole | (b-full) | 1,946 | .2586 ±.0035 | .0937 ±.0014 | .4369 ±.0190 | .2665 ±.0118 | .4973 ±.0237 | .4129 ±.0152 |
| Transformer (3 seeds) | | whole | (b-full) | 1,946 | .2589 ±.0019 | .0947 ±.0029 | .4377 ±.0137 | .2685 ±.0080 | .4981 ±.0233 | .4127 ±.0180 |
| CXRMate single-TF | | whole | (b-full) | 1,946 | .2291 | .0752 | .2944 | .2186 | .3382 | .2993 |

Under (b-full), the findings-only generators score the same as in (b). BLEU-1 under (b): mLMamba
.0857 ±.0140; R2Gen-Mamba .3283, R2Gen .3385, R2GenCMN .3256, CvT2DistilGPT2 .2289, CXRMate .2280.
Under (b-full): mLMamba .3602 ±.0018. The example-based CheXbert-14 F1 under (b) is R2Gen-Mamba
.3655, R2Gen .3105, R2GenCMN .3567, CvT2DistilGPT2 .2648 and CXRMate .2342. Under (b-full) it is
mLMamba .3702 ±.0174.

**What the papers report (MIMIC-CXR, their own test sets: findings only, all views of a study).**
All values were read from the paper PDFs on 2026-10-14.

| model | source | BLEU-4 | ROUGE-L | clinical F1 (as reported) |
|---|---|---|---|---|
| R2Gen-Mamba | arXiv 2410.18135, Table 2 | .110 | .284 | CheXbert-14 F1 .389 (P .483, R .325); averaging not stated |
| R2Gen | arXiv 2010.16056, Table 3 | .103 | .277 | CheXpert-labeler F1 .276 (P .333, R .273) |
| R2GenCMN | arXiv 2204.13258, Table 3 | .106 | .278 | CheXpert-labeler F1 .278 (P .334, R .275) |
| CvT2DistilGPT2 | arXiv 2201.09405, Tables 5/6/8 | .1236 (mean of 5 runs) | .2846 | CheXbert example F1 .3842; macro .307, micro .442 |
| CvT2DistilGPT2, released ckpt | repo README re-test, n=3,858 | .1269 | .2865 | example .3666, macro .2596, micro .4411 |
| CXRMate single-TF | arXiv 2307.09758, Fig. 5 only | — | — | macro CheXbert F1 shown only as a bar; no exact value |

Notes on the papers' own setups:
- R2Gen-Mamba's Table 1 copies R2Gen's test split (5.16K images / 3.27K reports).
- R2Gen-Mamba's baseline rows re-run R2Gen and CMN; they are not those papers' numbers.

**Is the adapter faithful?** Under (b), the R2Gen family lands close to its papers:
- R2Gen-Mamba: BLEU-4 .1016 vs .110, ROUGE-L .2801 vs .284.
- R2Gen: .0935 vs .103 and .2636 vs .277.

That supports the rebuilt vocabulary. CvT2DistilGPT2 and CXRMate land well below their published
numbers: CvT2DistilGPT2 has example F1 .2648 vs .3666 and micro .2985 vs .4411. Three likely
causes, none of them quantified:
- **Image shape.** Both models keep the aspect ratio of full-resolution JPEGs. Our 320x320 images
  are squashed, so they get upsampled to 384/448 and then cropped. The R2Gen family squashes to
  224x224 itself.
- **Short outputs.** CvT2DistilGPT2 was trained on references cut to 60 words and writes 33 words
  on average, so BLEU's brevity penalty hits it against our uncut references.
- **CXRMate's target.** It was built for findings + impression.

**Paired bootstraps, mLMamba vs each published model** (1,000 resamples, seed 0, PER_LABEL; files
`analysis/isbi_reviewer/b10/` on the cluster). Seeds are listed as s42/s43/s44. W = mLMamba wins,
L = published model wins, T = tie (CI spans 0).

| vs | protocol | ROUGE-L | BLEU-1 | BLEU-4 | 14-micro | 14-macro | 5-micro | 5-macro |
|---|---|---|---|---|---|---|---|---|
| R2Gen-Mamba | (b) | LLL | LLL | LLL | LLL | LLL | LLL | LLL |
| R2Gen | (b) | LLL | LLL | LLL | LLL | LLL | LLL | LLL |
| R2GenCMN | (b) | LLL | LLL | LLL | LLL | LLL | LLL | LLL |
| CvT2DistilGPT2 | (b) | LLL | LLL | LLL | LLL | LLL | LLL | LLL |
| CXRMate single-TF | (b) | LLL | LLL | LLL | LLL | LLL | LLL | LLL |
| R2Gen-Mamba | (b-full) | LLL | WWW | LLL | WWT | WWT | WWT | WWW |
| R2Gen | (b-full) | LTT | WWW | TTT | WWW | WWT | WWW | WWW |
| R2GenCMN | (b-full) | LTL | WWW | WWW | WTL | TTL | WTL | TLL |
| CvT2DistilGPT2 | (b-full) | WWW | WWW | WWW | WWW | WWW | WWW | WWW |
| CXRMate single-TF | (b-full) | WWW | WWW | WWW | WWW | WWW | WWW | WWW |
| CXRMate single-TF | (a) | WWW | WWW | WWW | WWW | WWW | WWW | WWW |

Size of the gaps:
- **(b), every model.** mLMamba's deficits are large. Against R2Gen-Mamba they are 14-micro
  −.2178/−.2070/−.2136 and ROUGE-L −.1417/−.1332/−.1296.
- **(b-full) vs R2Gen-Mamba.** mLMamba is ahead on 14-micro (+.0358/+.0222/−.0017) and 5-macro
  (+.0560/+.0336/+.0270). It is behind on ROUGE-L (−.0255/−.0192/−.0198) and BLEU-4
  (−.0094/−.0067/−.0074).
- **(b-full) vs R2GenCMN.** It is the strongest published model on CheXbert macro. mLMamba's
  14-macro difference is −.0074/−.0137/−.0302 (CI excludes 0 at s44 only). Its 5-macro difference
  is −.0080/−.0304/−.0370 (excluded at s43 and s44).
- **Per-label wins/losses of 14, (b) vs R2Gen-Mamba.** 0/8 at every seed.

**Reading.**
- **(b) as specified.** Extracting the findings part of our reports makes mLMamba lose every metric
  to every published model at 3/3 seeds. This is not an architecture effect: the matched
  Transformer is identical (.1995 ±.0204 14-micro). Our decoders often write an empty findings
  section and put the content in the impression. Under the findings-only convention our system
  therefore does not produce a usable findings section in about half the studies. That is a real
  limitation of the F+I training target, and a reviewer can fairly hold it against us.
- **(b-full).** Scoring the whole report against the findings reference changes the answer.
  mLMamba beats CvT2DistilGPT2 and CXRMate everywhere (3/3). It beats R2Gen on CheXbert
  (14-macro at 2/3, the other three at 3/3). Against
  R2Gen-Mamba, the closest prior work, it is ahead on clinical F1 at 2/3 seeds (3/3 on 5-macro)
  and behind on ROUGE-L and BLEU-4 at 3/3. It is level with or behind R2GenCMN on CheXbert macro.
  Two caveats apply: (b-full) still favours us, because the impression restates the findings,
  which helps CheXbert recall, and it is not the protocol the papers used.
- **(a) (our protocol).** Only CXRMate is target-matched, and mLMamba beats it on all metrics at
  3/3. The findings-only models lose on text there by construction. On CheXbert, R2GenCMN's
  14-macro (.2828) and 5-macro (.4380) are above mLMamba's 3-seed means (.2715, .4170). That is a
  point estimate with no bootstrap run.
- **Caveats.** All published models see our 320x320 squashed frontal images and one image per
  study; their papers used all views at full resolution. CvT2DistilGPT2 and CXRMate are clearly
  hurt by this here.
- **Test leakage.** R2Gen, R2GenCMN, CvT2DistilGPT2 and CXRMate state that they train on the
  MIMIC-CXR official training split, the same split ours uses. R2Gen-Mamba's annotation file is
  private, but its vocabulary equals the one rebuilt from the official training split.

### B10 addendum: combined and impression-only protocols (jobs 2680441-2680456, 2026-10-15)

Combined = our protocol (references "Findings: ... Impression: ...", n=2,663, no cleaner).
Impression-only = studies with a non-empty reference impression (n=2,061), each system's own
impression section, R2Gen cleaner on both sides. The R2Gen family and CvT2DistilGPT2 write no
impression: their whole output is scored (sensitivity only, not like for like). mLMamba and the
Transformer leave the impression empty in 135-197 of 2,061 (100-token cap); counted as written.
No bootstrap CIs on these tables yet.

| system | protocol | ROUGE-L | BLEU-4 | 14-micro | 14-macro | 5-micro | 5-macro |
|---|---|---|---|---|---|---|---|
| mLMamba (3-seed mean) | combined | .1953 | .0579 | .4480 | .2715 | .5044 | .4170 |
| Transformer (3-seed mean) | combined | .1952 | .0575 | .4443 | .2692 | .5032 | .4170 |
| R2Gen-Mamba | combined | .1596 | .0362 | .4212 | .2525 | .4556 | .3677 |
| R2GenCMN | combined | .1481 | .0301 | .4370 | .2828 | .4975 | .4380 |
| R2Gen | combined | .1507 | .0349 | .3653 | .2482 | .3540 | .3068 |
| CXRMate-single-tf | combined | .1711 | .0366 | .2741 | .2070 | .3068 | .2740 |
| CvT2DistilGPT2 | combined | .1420 | .0225 | .2698 | .1912 | .2646 | .2304 |
| XrayCLIP floor | combined | .1748 | .0411 | .5095 | .3631 | .5549 | .4921 |
| mLMamba (3-seed mean) | impression | .1848 | .0525 | .3572 | .2303 | .4134 | .3527 |
| Transformer (3-seed mean) | impression | .1860 | .0494 | .3592 | .2312 | .4187 | .3567 |
| CXRMate-single-tf | impression | .1893 | .0040 | .2621 | .1767 | .2496 | .2201 |
| R2Gen-Mamba (whole output) | impression* | .1721 | .0321 | .3753 | .2343 | .3980 | .3336 |
| R2GenCMN (whole output) | impression* | .1706 | .0283 | .3833 | .2617 | .4395 | .3957 |
| R2Gen (whole output) | impression* | .1643 | .0245 | .3270 | .2313 | .3181 | .2878 |
| CvT2DistilGPT2 (whole output) | impression* | .1540 | .0288 | .2389 | .1721 | .2275 | .2068 |
| BiomedCLIP floor | impression | .1553 | .0218 | .3141 | .2294 | .3591 | .3199 |
| XrayCLIP floor | impression | .1829 | .0355 | .3997 | .2961 | .4388 | .3917 |

\* sensitivity only. Per seed: seeds 42/43/44 in results/*_combined and *_impression_only
(cluster). Reading: on combined, mLMamba has the best text metrics of every system and the best
CheXbert-14 micro of the generators, but R2GenCMN leads on 14-macro and 5-macro and the XrayCLIP
floor leads on all CheXbert; on impression-only, mLMamba keeps the best BLEU-4 but R2GenCMN's
findings text labels better (14-micro .383 vs .357, 14-macro .262 vs .230) and the XrayCLIP floor
is best; mLMamba and the Transformer are level everywhere.
