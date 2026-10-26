# TEXTENC findings (attention-free mLMamba text encoder + retrieval-augmented decoding)

All numbers are aggregates; generations, indices and bootstrap tables stay on the cluster under
`~/hybrid_textenc/` (rule R5). Test split n=2,663 unless stated.

## T2 Stage 1 with the attention-free encoder (job 2680042)

Text encoder: mLMamba layout (mamba3 x9, mLSTM x3, `tfla_impl: exact`), Stage-0 mLMamba init
(missing keys 0), masked-mean pooling, no attention module (tests/test_textenc.py). Recipe = the
published 13D tower run; 6,000 steps, 1h56 on one H100.

In-training retrieval (validation, N=1,433, last evaluation):

| text encoder | i2t R@1 | i2t R@5 | i2t R@10 | t2i R@10 |
|---|---|---|---|---|
| E-m3 (this plan, attention-free) | .060 | .201 | .305 | .299 |
| published (Mamba-1 hybrid + attention pooling, job 2502673) | .070 | .213 | .316 | .307 |

Difference about 1 point at R@10 (binomial SE about 1.2 points at N=1,433): level, not better.

## T3 Re-ranked retrieval (index 2680062; CheXbert 2680063/64; bootstraps 2707990-96, 2708015)

Top-50 training studies by image-image cosine (new Stage-1 tower, same patient excluded), re-ranked
by image-text cosine with E-m3; the re-rank changes the top-1 for 97.5% of test studies.

| copy-the-top-1 floor | ROUGE-L | BLEU-4 | 14-micro | 14-macro | 5-micro | 5-macro |
|---|---|---|---|---|---|---|
| new tower, image-image only | .1704 | .0391 | .4648 | .3227 | .5242 | .4639 |
| new tower + E-m3 re-rank (F-rr) | .1767 | .0440 | .5115 | .3599 | .5687 | .5103 |
| XrayCLIP floor | .1748 | .0411 | .5095 | .3631 | .5549 | .4921 |
| MedSigLIP floor | .1730 | .0396 | .4890 | .3498 | .5447 | .4836 |
| BiomedCLIP floor (stock) | .1636 | .0330 | .4296 | .3014 | .4856 | .4284 |

Paired bootstraps (1,000 resamples) for F-rr:
- vs image-image only (same tower): wins ROUGE-L, BLEU-1/4, all four CheXbert (+.037 to +.047),
  exact-match-5; 8 labels up, Fracture down. **The text encoder at inference is what helps.**
- vs XrayCLIP floor: 5-macro and BLEU-1 win, all else tie (14-micro +.002, 14-macro -.003).
- vs MedSigLIP floor: wins text, 14-micro, 5-micro, 5-macro; 14-macro tie.
- vs mLMamba (published, each seed): wins all four CheXbert at 3/3 (14-micro +.045 to +.083,
  14-macro +.077 to +.101), BLEU-1 at 3/3; loses ROUGE-L and BLEU-4 at 3/3, exact-match-5 at 2/3.
- vs BiomedCLIP floor (stock, the published floor; rerun 2708015): wins all 9 metrics (14-micro +.082,
  14-macro +.059, 5-micro +.083, 5-macro +.082, ROUGE-L +.013, BLEU-4 +.011).

Reading: with only our own encoders (BiomedCLIP tower fine-tuned in Stage 1 + the attention-free
report encoder), retrieval reaches the level of the best published chest X-ray encoder (XrayCLIP,
frozen) and beats MedSigLIP. The generated reports (mLMamba) remain better on ROUGE-L/BLEU-4 and
worse on CheXbert; T5 tests whether giving the decoder these re-ranked reports combines both.

## T3-C Control: the same re-rank with the published (attention-pooling) encoders (2708031-36)

Published 13D image tower + its Stage-1 text encoder (hybrid_150m_v2: Mamba-1 + mLSTM with
AttentionPooling), same script, same top-50 / top-1 rule (re-rank changes top-1 for 97.0%).

| copy-the-top-1 floor | ROUGE-L | BLEU-4 | 14-micro | 14-macro | 5-micro | 5-macro |
|---|---|---|---|---|---|---|
| published encoders, image only | .1731 | .0404 | .4671 | .3276 | .5284 | .4682 |
| published encoders, re-ranked | .1792 | .0455 | .5075 | .3538 | .5616 | .5009 |
| attention-free encoders, image only | .1704 | .0391 | .4648 | .3227 | .5242 | .4639 |
| attention-free encoders, re-ranked | .1767 | .0440 | .5115 | .3599 | .5687 | .5103 |

Bootstraps: published re-ranked vs published image-only wins all four CheXbert (+.026 to +.040)
and text; attention-free vs published, re-ranked: all four CheXbert tie (+.004 to +.009, CIs span
0), ROUGE-L -.0025 (published wins, CI upper bound -.0001), rest tie; image-only: all CheXbert tie,
ROUGE-L -.0027 (published wins).

Reading: the CheXbert gain comes from re-ranking with a Stage-1 text encoder, not from removing
attention. The attention-free encoder does it as well as the attention-pooling one (level on every
clinical metric; 0.003 lower ROUGE-L). Defensible claim: attention is not needed for this retrieval
and re-ranking step; not: the attention-free encoder is better.

## T5 Retrieval-augmented decoders (training 2724002/08, 2732017/21/25/29; bootstraps 2792744-58)

Both decoders read the published 13D image prefix plus the top-3 re-ranked training reports (attention-free
encoder, 160 tokens each, sequence 736 + prefix); Stage-3 recipe unchanged (12k steps, 4 GPUs), beam 3,
cached decode (checked against the uncached decode on 2 studies per shard). Reference files are
byte-identical to the published result dirs. M-pub / T-pub are the published runs at the same seed.

Mean ± SD over seeds 42/43/44:

| system | ROUGE-L | BLEU-4 | 14-micro | 14-macro | 5-micro | 5-macro |
|---|---|---|---|---|---|---|
| mLMamba + retrieval (M-rag) | .1946 ± .0027 | .0570 ± .0018 | .4489 ± .0140 | .2739 ± .0041 | .5075 ± .0315 | .4226 ± .0280 |
| mLMamba published (M-pub) | .1953 ± .0029 | .0579 ± .0009 | .4480 ± .0188 | .2715 ± .0121 | .5044 ± .0257 | .4170 ± .0164 |
| Transformer + retrieval (T-rag) | .1936 ± .0032 | .0563 ± .0022 | .4506 ± .0068 | .2770 ± .0074 | .5089 ± .0115 | .4257 ± .0085 |
| Transformer published (T-pub) | .1952 ± .0021 | .0575 ± .0016 | .4443 ± .0153 | .2692 ± .0106 | .5032 ± .0226 | .4170 ± .0165 |
| re-ranked floor (F-rr, context source) | .1767 | .0440 | .5115 | .3599 | .5687 | .5103 |

Paired bootstraps (1,000 resamples, per seed; 95% CI excludes 0 = win):
- **Primary, M-rag vs M-pub: FAILS** (pre-registered: 14-micro AND 14-macro win at >=2/3 seeds; met at 1/3).
  s42 all 9 metrics tie; s43 M-pub wins 14-micro, 5-micro, 5-macro; s44 M-rag wins all four CheXbert.
  Text metrics tie at 3/3.
- T-rag vs T-pub: s42 T-pub wins ROUGE-L, BLEU-1, BLEU-4 (CheXbert tie); s43 all tie; s44 T-rag wins all
  four CheXbert and BLEU-1. Same pattern as mLMamba: the only CheXbert gain is at seed 44, the seed where
  both published decoders scored lowest (14-micro .4285 each).
- M-rag vs T-rag: s42 M-rag wins 14-micro, 5-micro, 5-macro; s43 T-rag wins 5-micro, 5-macro; s44 T-rag
  wins 14-micro, 14-macro, BLEU-1; text tie at 3/3 except that BLEU-1. No consistent winner.
- M-rag vs the re-ranked floor and vs the XrayCLIP floor: both floors win all four CheXbert at 3/3
  (14-micro -.045 to -.073, 14-macro -.082 to -.093) and BLEU-1 at 3/3; M-rag wins ROUGE-L, BLEU-4 and
  exact-match-5 at 3/3.
- Per label (M-rag vs M-pub): Edema up at 2/3 seeds, no label moves in the same direction at 3/3.

Reading: three retrieved reports in the context do not change what either decoder writes in a way
CheXbert can measure: the decoders stay about .06 below the 14-micro of the very reports they are
given. The architecture comparison is unchanged by retrieval (ties, seed-dependent direction). The
re-ranked retrieval (T3) is the useful part of this plan; feeding it to the decoder as plain context
is not. Seed SD of the retrieval runs is lower for the Transformer (14-micro .0068 vs .0153), but 3
seeds cannot support a claim about variance.
