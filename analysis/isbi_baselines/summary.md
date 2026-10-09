# ISBI floors: results (B4-B, 2026-10-04)

Official MIMIC-CXR test split, n=2663. Floors are deterministic. mLMamba = mean of seeds 42/43/44.
Marks: paired bootstrap (1000 resamples) of each floor vs mLMamba at each seed
(`analysis/bootstrap_m3_vs_floor[_<enc>]_seed4{2,3,4}.md` on the cluster). "F3" = floor better at 3/3
seeds, "F2" = 2/3, "M3" = mLMamba better at 3/3, blank = mixed/tie. Example-F1 is not bootstrapped.

| floor | ROUGE-L | BLEU-1 | BLEU-4 | 14 micro | 14 macro | 5 micro | 5 macro | ex-F1 | EM14 | EM5 |
|---|---|---|---|---|---|---|---|---|---|---|
| BiomedCLIP (published) | .1636 M3 | .2372 M3 | .0330 M3 | .4296 | .3014 F3 | .4856 | .4284 F2 | .3691 | .0263 M3 | .1735 M3 |
| CLIP ViT-B/16 | .1602 M3 | .2240 M3 | .0327 M3 | .3534 M3 | .2366 M3 | .3822 M3 | .3344 M3 | .2932 | .0161 M3 | .1495 M3 |
| PubMedCLIP | .1588 M3 | .2248 M3 | .0309 M3 | .3821 M3 | .2583 | .4209 M3 | .3697 M3 | .3205 | .0173 M3 | .1419 M3 |
| XrayCLIP | .1748 M3 | .2583 F3 | .0411 M3 | .5095 F3 | .3631 F3 | .5549 F3 | .4921 F3 | .4481 | .0421 | .2035 |
| MedSigLIP | .1730 M3 | .2530 | .0396 M3 | .4890 F3 | .3498 F3 | .5447 F2 | .4836 F3 | .4302 | .0368 | .1934 M3 |
| mLMamba | .1953 | .2484 | .0579 | .4480 | .2715 | .5044 | .4170 | .3858 | .0452 | .2242 |

**Finding.** The two chest-X-ray encoders (XrayCLIP, MedSigLIP) make floors that beat mLMamba on every
CheXbert F1 at all seeds; mLMamba keeps ROUGE-L and BLEU-4 at all seeds. The gain is in rare findings
(seed 42, XrayCLIP floor vs mLMamba: lung lesion .159 vs .000, pleural other .173 vs .033, pneumothorax
.171 vs .056). Interpretation: clinical accuracy is limited by the image representation, not the decoder.
Caveat: both encoders were pretrained on MIMIC-CXR training data; test-split exclusion is per their papers,
not verified by us. Exact-duplicate share of floor reports: BiomedCLIP .073, XrayCLIP .180, MedSigLIP .155.

**Cost** (encoder + top-10 search over 191,462 vectors, bf16, batch 4, H100, job 2602161):
BiomedCLIP 2.31 ms / 0.43 GB, CLIP 2.84 / 0.56, PubMedCLIP 2.78 / 0.56, XrayCLIP 3.25 / 0.62,
MedSigLIP 13.84 / 2.37.
