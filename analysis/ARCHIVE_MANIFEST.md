# Archive manifest — what exists on the cluster, what it cost, what may leave it

> Written 2026-09-20 (MAMBA3_PLAN_V2.md V4-E) from a full `du -sh outputs/*` on
> `/sc/home/$USER/hybrid_mamba_xlstm`. Covers every artifact of Phases 1–15 and of the Mamba-3
> campaign. **Classification is by provenance and the default is restrictive**: anything trained on
> or derived from MIMIC-CXR is treated as covered by the PhysioNet data use agreement, including
> model weights, because weights are a derived work of the data. Confirm against the DUA text
> before moving anything, and when in doubt ask the PhysioNet helpdesk rather than this file.
>
> The account expiry that originally motivated archiving was lifted on 2026-09-19. The obligation
> that remains is the DUA's, not the calendar's.

## Totals

| | size |
|---|---|
| `outputs/` after the 2026-09-20 cleanup | ~83 GB |
| freed on 2026-09-20 by deleting 15 screen arms | 75 GB |
| home quota | 200 GiB, currently ~100 GiB used |

`results/` (generated and reference report text, CheXbert label dumps) and `analysis/` (metric
tables, bootstrap reports) are small but **not** negligible in kind: see the restriction column.

## Class R — restricted, MIMIC-derived. Keep on the cluster or on approved storage only

Trained on MIMIC-CXR images or report text, or containing report text directly. Do **not** copy to
a laptop, personal cloud, or any unapproved system, and do not attach to a thesis submission.

| artifact | size | what it is | regenerable? |
|---|---|---|---|
| `h100_kd_150m_v2_full_data_lr3e6` | 2.4 G | **the image tower every report-gen arm used** (13D recipe, `vit_lr=3e-6`) | ~8 h; the single most reused artifact here |
| `h100_kd_150m_v2_full_data`, `_lr1e5`, `h100_kd_150m_v2_bs64_head4.24e-4` | 9.6 G total | the tower sweep those were selected from | yes, ~8 h each |
| `h100_report_gen_full_ext_4gpu_tower13d` | 2.4 G | **13D**, the incumbent decoder behind the published headline | ~2.5 h |
| `h100_report_gen_tower13d_seed43`, `_seed44` | 4.8 G | the hybrid's seed band (15B-3) | ~2.5 h each |
| `h100_report_gen_transformer_tower13d`, `_seed43`, `_seed44` | 7.2 G | the matched Transformer's three seeds | ~40 min each |
| `h100_report_gen_m3_tower13d_s42`, `_s43`, `_s44` | 7.2 G | **this campaign's three decoders** | ~1.3 h each, but only with the Mamba-3 backbone below |
| `h100_report_gen_aux_lam0.1_seed42`, `_lam0.5_seed42` | 4.8 G | the Phase-15C auxiliary-loss arms (negative result) | ~2.3 h each |
| `h100_6g*`, `h100_6d_*` (12 dirs) | 33.6 G | the retrieval chapter: contrastive towers behind 10.81% → 14.59% i2t R@10 | yes, but they back closed published numbers |
| `results/**` | small | `hyps.txt`, `refs.txt`, `chexbert_labels.json`. **`refs.txt` is verbatim MIMIC report text** | from the checkpoints above |
| `/sc/home/$USER/dataset/mimic_full/` | ~6 G | the packed 320px corpus and splits | ~310–400 GB of re-download |

## Class O — open provenance. Safe to archive anywhere

Trained only on PubMed abstracts with a public teacher (BioMedLM). No MIMIC data touched these.

| artifact | size | what it is | regenerable? |
|---|---|---|---|
| `h100_stage0_150m_m3` | 4.2 G | **the corrected Mamba-3 backbone**, val PPL 11.674 | 57 h of H100 |
| `m8_stage0_A2_150m` | 6.3 G | **the partially-corrected backbone**, val PPL 12.550. Orphaned by the superseded plan, harvested 2026-09-20; it is what decomposes the gain 42% / 58% | 57 h of H100 |
| `h100_stage0_transformer_150m` | 2.1 G | the matched Transformer's backbone, val PPL 11.222 | 23 h of H100 |
| `h100_stage0_150m_v2` | 705 M | the incumbent hybrid's backbone, val PPL 13.18, **weights-only** (`stage0_model_only.pt`) | 74 h of H100, and the full optimiser state is already gone |
| `analysis/**` | small | every metric table, bootstrap report and writeup in this repo | these are the record |

**The four Stage-0 backbones are the most valuable irreplaceable things here**: 211 H100-hours
between them, no patient data, and they are what every downstream claim rests on. If only one thing
is archived off-cluster, archive these four plus `analysis/`.

## Class D — safe to delete if space is ever needed again

| artifact | size | why |
|---|---|---|
| `outputs/2026-*` (19 dirs) | ~1 MB | Hydra per-run config dumps; the same information is in `run_metadata.json` beside each checkpoint |
| `m3_probe_A2` | 15 K | a 300-step probe's leftovers |
| *(already deleted)* `m3_screen_*` | 75 G | 15 screening arms; every number is in `MAMBA3_PLAN_V2.md` §3 and `analysis/mamba3_results.md` |

## If artifacts must be reduced further

In increasing order of regret:

1. The Phase-15C auxiliary-loss arms, 4.8 G. A closed negative result; the dumps and per-label
   tables in `analysis/PHASE15C_AUX_LOSS.md` are the part anyone will read.
2. The tower sweep other than `lr3e6`, 9.6 G. The selected tower is the one that matters.
3. The retrieval-era `h100_6d_*` arms, 14 G. The chapter is closed and its numbers are recorded,
   though `h100_6g7_vit2_cleansplit` and `h100_6g_cleansplit` back the headline and should stay.

Never delete: the four Stage-0 backbones, `h100_kd_150m_v2_full_data_lr3e6`, the nine report-gen
decoders that produced the three-seed tables, or anything in `results/` and `analysis/`.
