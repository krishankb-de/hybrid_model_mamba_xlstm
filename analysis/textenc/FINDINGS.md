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
