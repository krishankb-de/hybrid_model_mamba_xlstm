"""Phase 11A: report-generation evaluation harness (ROUGE-L / BLEU-1/4 / METEOR).

This script needs no PhysioNet data and no trained image-conditioned generator —
it is built during the Phase 8 fetch wait per H100_SCALING_PLAN.md's "unblocked
work" list. It provides two independent pieces, both usable today:

  1. Corpus-level text metrics (rouge_l_score, corpus_bleu, meteor_score_corpus)
     over any (hypothesis, reference) string pairs — e.g. the retrieval-NN
     baseline's output (Phase 11C, below), or real generator output.
  2. A fixed decoding harness (greedy_decode, beam_search_decode with beam=3)
     built on Phase 10A's HybridLanguageModel.forward(inputs_embeds=...) /
     .generate(prefix_embeds=...) additive hooks — exercised end-to-end here
     against a tiny random-init model + random prefix embeddings (--smoke-test)
     so the decoding code itself is proven correct before any checkpoint exists.

--checkpoint mode (added 2026-08-23, after the first real Phase 10E arm0
checkpoint existed) loads a trained ReportGenerationLightningModule from a
Lightning .ckpt and generates from real images — a qualitative check, NOT yet
the full official-subject-disjoint-test-split scoring run Phase 11D's plan
entry describes. Loader pattern follows scripts/evaluate_lm.py's convention
(defensive key-prefix stripping, missing/unexpected key counts printed).
CONFIRMED LIVE 2026-08-24 (jobs 2478647, 2482543): the arm0 checkpoint --
with and without Phase 9D image augmentation -- generates byte-identical
boilerplate for different images rather than conditioning on content.

--retrieval-baseline mode (added 2026-08-24, Phase 11C) retrieves the nearest
TRAINING report by stock BiomedCLIP pooled-image cosine similarity and emits
it verbatim — "the real floor; a generator that does not beat it has
contributed nothing" (H100_SCALING_PLAN.md). No Phase-9 best contrastive
checkpoint exists yet, so this uses stock BiomedCLIP (matches 10C's current
frozen-tower default).

--chexbert (added 2026-08-28, Phase 11B) is an opt-in flag, usable alongside
--checkpoint, --retrieval-baseline, or --hyp-file/--ref-file: micro/macro F1
over the 14 CheXpert labels + the 5-label RRG subset, via the f1chexbert
package (pip install f1chexbert; downloads chexbert.pth on first use).
UNVERIFIED against real weights as of this writing — degrades to a printed
"skipped" message on any failure rather than crashing, same pattern as
METEOR's nltk/wordnet guard. Pass --chexpert-csv (path to the Phase 8A
mimic-cxr-2.0.0-chexpert.csv.gz) alongside --chexbert in --checkpoint/
--retrieval-baseline mode to also cross-check our labeling of the reference
text against the official ground-truth labels (labeler-wiring sanity check).

Usage:
    # Metrics over precomputed hypothesis/reference pairs (one line each, aligned)
    python scripts/evaluate_report_generation.py \\
        --hyp-file preds.txt --ref-file refs.txt \\
        --output-dir results/report_gen_baseline

    # Prove the decoding harness (prefix-conditioned greedy + beam=3) runs
    # end-to-end on a tiny random model — no checkpoint, no data required
    python scripts/evaluate_report_generation.py --smoke-test

    # Qualitative check against a real trained checkpoint + real images
    python scripts/evaluate_report_generation.py \\
        --checkpoint outputs/h100_report_gen_arm0/checkpoints/last.ckpt \\
        --parquet /sc/home/$USER/dataset/mimic_full/arm0/validate.parquet \\
        --num-samples 10 --decode greedy

    # Phase 11C retrieval-NN baseline — the real floor a generator must beat
    python scripts/evaluate_report_generation.py \\
        --retrieval-baseline \\
        --train-parquet /sc/home/$USER/dataset/mimic_full/arm0/train.parquet \\
        --parquet /sc/home/$USER/dataset/mimic_full/arm0/validate.parquet \\
        --num-samples 10
"""

import sys
import json
import math
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from hybrid_xmamba.models.configuration_hybrid import HybridConfig
from hybrid_xmamba.models.hybrid_lm import HybridLanguageModel
from hybrid_xmamba.models.prefix_mapper import ImagePrefixMapper


# ---------------------------------------------------------------------------
# Decoding harness — fixed config: greedy + beam=3, fixed max_new_tokens
# ---------------------------------------------------------------------------

@torch.no_grad()
def greedy_decode(
    model: HybridLanguageModel,
    input_ids: torch.Tensor,
    prefix_embeds: Optional[torch.Tensor] = None,
    max_new_tokens: int = 100,
) -> torch.Tensor:
    """Deterministic argmax decoding. Delegates to generate()'s prefix-conditioned
    path (Phase 10A) with top_k=1, which forces the sampler onto the single
    highest-logit token at every step — reusing already-tested code rather than
    duplicating the prefix/hidden-state bookkeeping.
    """
    return model.generate(
        input_ids,
        prefix_embeds=prefix_embeds,
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        top_k=1,
    )


@torch.no_grad()
def beam_search_decode(
    model: HybridLanguageModel,
    input_ids: torch.Tensor,
    prefix_embeds: Optional[torch.Tensor] = None,
    beam_size: int = 3,
    max_new_tokens: int = 100,
    length_penalty: float = 1.0,
) -> torch.Tensor:
    """Standard beam search. Operates on ONE sample at a time (input_ids/
    prefix_embeds batch dim must be 1) — model.generate() has no beam-search
    mode, so this is new code built directly on forward(inputs_embeds=...).
    Callers with a batch loop per-sample.
    """
    if input_ids.shape[0] != 1:
        raise ValueError(
            "beam_search_decode operates on one sample at a time "
            "(got batch size {})".format(input_ids.shape[0])
        )
    device = input_ids.device

    if prefix_embeds is not None:
        base_hidden = torch.cat([prefix_embeds, model.embeddings(input_ids)], dim=1)
    else:
        base_hidden = model.embeddings(input_ids)

    # Each beam: (hidden_states, token_ids, cumulative_log_prob)
    beams = [(base_hidden, input_ids, 0.0)]

    for _ in range(max_new_tokens):
        candidates = []
        for hidden_states, token_ids, score in beams:
            logits = model.forward(inputs_embeds=hidden_states, return_dict=True).logits
            log_probs = torch.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)
            topk_logp, topk_idx = log_probs.topk(beam_size)
            for lp, idx in zip(topk_logp.tolist(), topk_idx.tolist()):
                next_token = torch.tensor([[idx]], device=device, dtype=token_ids.dtype)
                new_hidden = torch.cat([hidden_states, model.embeddings(next_token)], dim=1)
                new_tokens = torch.cat([token_ids, next_token], dim=1)
                candidates.append((new_hidden, new_tokens, score + lp))

        def _ranked(c):
            return c[2] / (c[1].shape[1] ** length_penalty)

        candidates.sort(key=_ranked, reverse=True)
        beams = candidates[:beam_size]

    best = max(beams, key=lambda c: c[2] / (c[1].shape[1] ** length_penalty))
    return best[1]


# ---------------------------------------------------------------------------
# Real-checkpoint loading + generation (--checkpoint mode). Split into a
# heavy loader (needs a real checkpoint + BiomedCLIP weights, not unit-tested)
# and a light decode step (generate_from_patch_grid, unit-tested with a
# synthetic patch_grid — same pattern as
# test_report_generation_step_produces_finite_loss_and_gradients).
# ---------------------------------------------------------------------------

def build_decoder_config(model_cfg: Dict) -> HybridConfig:
    """Filter a raw model-yaml dict down to HybridConfig's declared fields —
    same pattern as test_hybrid_150m_v2_rrg_config_matches_150m_v2_architecture."""
    import dataclasses
    fields = {f.name for f in dataclasses.fields(HybridConfig)}
    return HybridConfig(**{k: v for k, v in model_cfg.items() if k in fields})


def load_report_generation_module(checkpoint_path, model_config_name: str = "hybrid_150m_v2_rrg", device: str = "cpu"):
    """Load a trained ReportGenerationLightningModule from a Lightning .ckpt
    for inference. Mirrors evaluate_lm.py's checkpoint-loading convention
    (defensive _orig_mod. prefix strip, missing/unexpected key counts printed)."""
    from omegaconf import OmegaConf
    from hybrid_xmamba.training.lightning_module import ReportGenerationLightningModule

    model_cfg_path = PROJECT_ROOT / "configs" / "model" / f"{model_config_name}.yaml"
    raw = OmegaConf.to_container(OmegaConf.load(model_cfg_path), resolve=True)
    decoder_config = build_decoder_config(raw)

    module = ReportGenerationLightningModule(
        decoder_config=decoder_config,
        image_patch_dim=int(raw.get("image_patch_dim", 768)),
        prefix_k=int(raw.get("prefix_k", 32)),
    )
    module.load_image_encoder()  # registers image_encoder as a submodule so its
                                  # keys (saved in the .ckpt — see 10E's note that
                                  # the frozen ViT is checkpointed too) actually load.

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    state_dict = {
        (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
        for k, v in state_dict.items()
    }
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")

    module.to(device)
    module.eval()
    return module


@torch.no_grad()
def generate_from_patch_grid(
    module,
    patch_grid: torch.Tensor,
    decode: str = "greedy",
    beam_size: int = 3,
    max_new_tokens: int = 100,
) -> torch.Tensor:
    """(1, N, patch_dim) ViT patch grid -> (1, max_new_tokens) generated token
    ids. Seeds generation with an EMPTY input_ids (no BOS token) -- matches
    training exactly, where report_labels start immediately after the k
    image-prefix positions (ReportGenerationLightningModule._step); the model
    has never seen a start-of-sequence marker, so inventing one here would be
    a train/inference mismatch, not a neutral seed."""
    device = patch_grid.device
    prefix_embeds = module.prefix_mapper(patch_grid)
    input_ids = torch.zeros((patch_grid.shape[0], 0), dtype=torch.long, device=device)
    if decode == "greedy":
        return greedy_decode(module.decoder, input_ids, prefix_embeds=prefix_embeds,
                              max_new_tokens=max_new_tokens)
    if decode == "beam":
        return beam_search_decode(module.decoder, input_ids, prefix_embeds=prefix_embeds,
                                   beam_size=beam_size, max_new_tokens=max_new_tokens)
    raise ValueError(f"Unknown decode mode: {decode!r} (expected 'greedy' or 'beam')")


def run_checkpoint_inspection(args) -> None:
    """--checkpoint mode: generate from real images with a real trained
    checkpoint and print generated-vs-reference text side by side. Qualitative
    inspection tool, NOT the official-test-split 11D scoring run -- arm0's
    own parquets are a legacy-training-hash subset, not the subject-disjoint
    official split (see H100_SCALING_PLAN.md's cross_protocol_number_comparison
    warning)."""
    import pandas as pd
    from PIL import Image
    import torchvision.transforms as T
    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    module = load_report_generation_module(args.checkpoint, args.model_config, device=device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # BiomedCLIP CLIP normalisation — matches cxr_mimic_arm0.yaml/cxr_mimic_full.yaml.
    img_transform = T.Compose([
        T.Resize((224, 224)),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    df = pd.read_parquet(args.parquet)
    n = min(args.num_samples, len(df))
    print(f"Loaded {len(df)} rows from {args.parquet}; inspecting first {n}\n")

    hyps, refs, study_ids = [], [], []
    for i in range(n):
        row = df.iloc[i]
        img = Image.open(row["image"]).convert("RGB")
        pixel_values = img_transform(img).unsqueeze(0).to(device)
        patch_grid = module._patch_grid(pixel_values)
        out_ids = generate_from_patch_grid(
            module, patch_grid, decode=args.decode,
            beam_size=args.beam_size, max_new_tokens=args.max_new_tokens,
        )
        generated = tokenizer.decode(out_ids[0].tolist(), skip_special_tokens=True)
        reference = f"Findings: {row.get('findings', '')} Impression: {row.get('impression', '')}".strip()

        print(f"--- sample {i} (study_id={row.get('study_id', '?')}) ---")
        print(f"GENERATED: {generated}")
        print(f"REFERENCE: {reference}")
        print(f"ROUGE-L (single sample): {rouge_l_score(generated.split(), reference.split()):.3f}\n")
        hyps.append(generated)
        refs.append(reference)
        study_ids.append(row.get("study_id"))

    metrics = compute_all_metrics(hyps, refs, chexbert=args.chexbert, device=device)
    print(f"=== Aggregate over {n} samples ===")
    print(json.dumps(metrics, indent=2))

    if args.chexbert and args.chexpert_csv:
        print("\n=== Phase 11B ground-truth labeler cross-check ===")
        agreement = chexbert_ref_vs_ground_truth_agreement(
            refs, study_ids, args.chexpert_csv, device=device
        )
        print(json.dumps(agreement, indent=2))


# ---------------------------------------------------------------------------
# Phase 11C — retrieval-NN baseline. "Use the Phase-9 model to retrieve the
# nearest training report and emit it verbatim... this baseline is the real
# floor; a generator that does not beat it has contributed nothing"
# (H100_SCALING_PLAN.md). No Phase-9 best contrastive checkpoint exists yet
# (deferred, per the 2026-08-20 pivot decision) -- this uses STOCK BiomedCLIP
# pooled-image cosine similarity instead, matching 10C's current frozen-tower
# default; swap in a fine-tuned checkpoint later without changing this logic.
# Split the same way as the --checkpoint mode: nearest_neighbor_indices() is
# pure tensor math (CPU-unit-tested with synthetic embeddings); embedding real
# images with real BiomedCLIP is the untested heavy path, evaluate_lm.py-style.
# ---------------------------------------------------------------------------

def nearest_neighbor_indices(query_embeds: torch.Tensor, gallery_embeds: torch.Tensor) -> torch.Tensor:
    """(Q, D), (G, D) L2-normalized embeddings -> (Q,) argmax cosine-similarity
    index into the gallery for each query. Pure tensor math, no I/O — the
    testable core of run_retrieval_baseline()."""
    sims = query_embeds @ gallery_embeds.T
    return sims.argmax(dim=1)


def run_retrieval_baseline(args) -> None:
    """Phase 11C: for each query (validation) image, retrieve the training
    report whose image has the highest BiomedCLIP cosine similarity and emit
    it verbatim as the hypothesis. Same output format as --checkpoint mode
    (generated/reference/ROUGE-L per sample + aggregate metrics) for direct
    side-by-side comparison."""
    import pandas as pd
    from PIL import Image
    import open_clip
    from omegaconf import OmegaConf
    from scripts.train_contrastive import build_image_transform

    device = "cuda" if torch.cuda.is_available() else "cpu"

    biomedclip_id = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    clip_model, _ = open_clip.create_model_from_pretrained("hf-hub:" + biomedclip_id)
    visual = clip_model.visual.to(device).eval()

    # is_train=False unconditionally -- embedding for retrieval is inference,
    # never training, regardless of which split the image happens to come from.
    img_transform = build_image_transform(OmegaConf.create({"dataset": {}}), is_train=False)

    @torch.no_grad()
    def embed_images(image_paths: List[str], batch_size: int = 64) -> torch.Tensor:
        embeds = []
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i + batch_size]
            imgs = torch.stack([img_transform(Image.open(p).convert("RGB")) for p in batch]).to(device)
            feats = torch.nn.functional.normalize(visual(imgs), dim=-1)
            embeds.append(feats.cpu())
            print(f"  embedded {min(i + batch_size, len(image_paths))}/{len(image_paths)}")
        return torch.cat(embeds, dim=0)

    train_df = pd.read_parquet(args.train_parquet)
    if args.max_gallery:
        train_df = train_df.iloc[:args.max_gallery]
    query_df = pd.read_parquet(args.parquet)
    n = min(args.num_samples, len(query_df))
    query_df = query_df.iloc[:n]

    print(f"Embedding {len(train_df)} gallery (train) images...")
    gallery_embeds = embed_images(train_df["image"].tolist())
    print(f"Embedding {n} query images...")
    query_embeds = embed_images(query_df["image"].tolist())

    nn_idx = nearest_neighbor_indices(query_embeds, gallery_embeds)

    hyps, refs, study_ids = [], [], []
    for i in range(n):
        q_row = query_df.iloc[i]
        g_row = train_df.iloc[nn_idx[i].item()]
        retrieved = f"Findings: {g_row.get('findings', '')} Impression: {g_row.get('impression', '')}".strip()
        reference = f"Findings: {q_row.get('findings', '')} Impression: {q_row.get('impression', '')}".strip()

        print(f"\n--- sample {i} (query study_id={q_row.get('study_id', '?')}, "
              f"nearest train study_id={g_row.get('study_id', '?')}) ---")
        print(f"RETRIEVED: {retrieved}")
        print(f"REFERENCE: {reference}")
        print(f"ROUGE-L (single sample): {rouge_l_score(retrieved.split(), reference.split()):.3f}")
        hyps.append(retrieved)
        refs.append(reference)
        study_ids.append(q_row.get("study_id"))

    metrics = compute_all_metrics(hyps, refs, chexbert=args.chexbert, device=device)
    print(f"\n=== Aggregate retrieval-NN baseline over {n} samples ===")
    print(json.dumps(metrics, indent=2))

    if args.chexbert and args.chexpert_csv:
        print("\n=== Phase 11B ground-truth labeler cross-check ===")
        agreement = chexbert_ref_vs_ground_truth_agreement(
            refs, study_ids, args.chexpert_csv, device=device
        )
        print(json.dumps(agreement, indent=2))


# ---------------------------------------------------------------------------
# Metrics — pure Python, no network dependency (ROUGE-L, BLEU); METEOR is
# nltk-backed and gracefully skipped if wordnet isn't staged locally (compute
# nodes have no internet — same constraint noted for Phase 11B's CheXbert
# weights in H100_SCALING_PLAN.md).
# ---------------------------------------------------------------------------

def _lcs_length(a: List[str], b: List[str]) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def rouge_l_score(hyp_tokens: List[str], ref_tokens: List[str], beta: float = 1.2) -> float:
    """Sentence-level ROUGE-L F-measure (Lin, 2004)."""
    if not hyp_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(hyp_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    p = lcs / len(hyp_tokens)
    r = lcs / len(ref_tokens)
    denom = r + (beta ** 2) * p
    return ((1 + beta ** 2) * p * r) / denom if denom > 0 else 0.0


def _ngram_counts(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(hyps: List[List[str]], refs: List[List[str]], max_n: int = 4) -> float:
    """Corpus-level BLEU-N with brevity penalty, single reference per hypothesis."""
    weights = [1.0 / max_n] * max_n
    precisions = []
    for n in range(1, max_n + 1):
        match, total = 0, 0
        for hyp, ref in zip(hyps, refs):
            hyp_ngrams = _ngram_counts(hyp, n)
            ref_ngrams = _ngram_counts(ref, n)
            match += sum(min(c, ref_ngrams.get(g, 0)) for g, c in hyp_ngrams.items())
            total += sum(hyp_ngrams.values())
        precisions.append(match / total if total > 0 else 0.0)

    if any(p == 0.0 for p in precisions):
        geo_mean = 0.0
    else:
        geo_mean = math.exp(sum(w * math.log(p) for w, p in zip(weights, precisions)))

    hyp_len = sum(len(h) for h in hyps)
    ref_len = sum(len(r) for r in refs)
    if hyp_len == 0:
        bp = 0.0
    elif hyp_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / hyp_len)
    return bp * geo_mean



# ---------------------------------------------------------------------------
# Phase 11B — CheXbert F1. micro/macro F1 over the 14 CheXpert labels (+ the
# 5-label RRG-literature subset) between CheXbert-labeled generated vs
# reference reports. Split the same way as 11C: the label-matrix F1 math is
# pure counting, CPU-unit-tested with synthetic label arrays
# (binary_f1_from_labels); the actual text->label step needs real CheXbert
# weights and is untested-by-design here (evaluate_lm.py/embed_images-style)
# -- backed by the `f1chexbert` package (pip install f1chexbert; downloads
# chexbert.pth on first use). NOT verified against real weights as of this
# writing -- the first live cluster run IS the integration test. Any failure
# (package missing, download failure, API mismatch) degrades to a printed
# "skipped" message, same pattern as meteor_score_corpus's nltk/wordnet
# guard -- never crashes the surrounding ROUGE/BLEU computation.
#
# CHEXPERT_14_LABELS matches the column names in mimic-cxr-2.0.0-chexpert.csv.gz
# (fetched in Phase 8A); CHEXBERT_5_LABELS is the standard 5-label RRG subset
# (Irvin et al. 2019 CheXpert "competition" labels).
# ---------------------------------------------------------------------------

CHEXPERT_14_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia",
    "Pneumothorax", "Support Devices",
]

CHEXBERT_5_LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]


def load_chexpert_ground_truth_labels(csv_path: str, study_ids: List[int]) -> Dict[int, List[int]]:
    """mimic-cxr-2.0.0-chexpert.csv.gz -> {study_id: [0/1 x 14]} binary
    positive-vs-not-positive labels for the given study_ids (U-Zeros
    convention: 1.0 -> 1, {0.0, -1.0, NaN} -> 0 -- the standard encoding for
    RRG CheXbert-F1 reporting). Only study_ids actually present in the CSV are
    returned -- callers must handle misses."""
    import pandas as pd
    df = pd.read_csv(csv_path, usecols=["study_id"] + CHEXPERT_14_LABELS)
    df = df[df["study_id"].isin(set(study_ids))]
    out = {}
    for _, row in df.iterrows():
        out[int(row["study_id"])] = [1 if row[label] == 1.0 else 0 for label in CHEXPERT_14_LABELS]
    return out


def binary_f1_from_labels(
    pred: List[List[int]], true: List[List[int]], label_names: List[str]
) -> Dict:
    """(N, L) binary 0/1 label matrices -> per-label precision/recall/F1 plus
    micro F1 (pooled TP/FP/FN across labels) and macro F1 (mean of per-label
    F1). Pure counting, no I/O, no model -- the CPU-testable core, split out
    from the heavy real-CheXbert labeling path (label_reports_with_chexbert)
    the same way nearest_neighbor_indices is split from embed_images."""
    per_label = {}
    tp_total = fp_total = fn_total = 0
    f1s = []
    for j, name in enumerate(label_names):
        tp = fp = fn = 0
        for p_row, t_row in zip(pred, true):
            p, t = p_row[j], t_row[j]
            if p == 1 and t == 1:
                tp += 1
            elif p == 1 and t == 0:
                fp += 1
            elif p == 0 and t == 1:
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_label[name] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
        tp_total += tp
        fp_total += fp
        fn_total += fn
        f1s.append(f1)

    micro_p = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    micro_r = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    macro_f1 = sum(f1s) / len(label_names) if label_names else 0.0

    return {
        "per_label": per_label,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "num_examples": len(pred),
    }


def label_reports_with_chexbert(texts: List[str], device: str = "cpu") -> Optional[List[List[int]]]:
    """List[str] -> (N, 14) 0/1 label matrix (order = CHEXPERT_14_LABELS) via
    the CheXbert labeler (f1chexbert package -- downloads chexbert.pth on
    first use, needs network; per sc-helpdesk@hpi.de compute nodes have a
    1 Gbit/s uplink, confirmed live by the PhysioNet fetch, so this should
    work directly from an aisc-batch job with no separate staging step).
    Returns None (never raises) on ANY failure -- missing package, download
    failure, API mismatch -- so a broken integration degrades to 'skipped'
    exactly like meteor_score_corpus, rather than crashing the whole eval.
    UNVERIFIED AGAINST REAL WEIGHTS as of this writing."""
    try:
        from f1chexbert import F1CheXbert
        labeler = F1CheXbert(device=device)
        labels = []
        for t in texts:
            raw = labeler.get_label(t)
            labels.append([1 if int(v) == 1 else 0 for v in raw])
        return labels
    except Exception as e:
        print(f"  CheXbert labeling skipped: {type(e).__name__}: {e}")
        return None


def compute_chexbert_metrics(hyps: List[str], refs: List[str], device: str = "cpu") -> Optional[Dict]:
    """Label both hyps and refs with CheXbert, then micro/macro F1 over the
    14-label set and the 5-label RRG subset. Returns None if labeling fails
    (see label_reports_with_chexbert) -- callers treat this the same as a
    None METEOR score (not computed, not zero)."""
    hyp_labels = label_reports_with_chexbert(hyps, device=device)
    if hyp_labels is None:
        return None
    ref_labels = label_reports_with_chexbert(refs, device=device)
    if ref_labels is None:
        return None

    full = binary_f1_from_labels(hyp_labels, ref_labels, CHEXPERT_14_LABELS)
    idx5 = [CHEXPERT_14_LABELS.index(name) for name in CHEXBERT_5_LABELS]
    hyp5 = [[row[i] for i in idx5] for row in hyp_labels]
    ref5 = [[row[i] for i in idx5] for row in ref_labels]
    subset5 = binary_f1_from_labels(hyp5, ref5, CHEXBERT_5_LABELS)

    return {"chexbert_14": full, "chexbert_5": subset5}


def chexbert_ref_vs_ground_truth_agreement(
    refs: List[str], study_ids: List[int], chexpert_csv: str, device: str = "cpu"
) -> Optional[Dict]:
    """Cross-check: label the REFERENCE text with our own CheXbert wiring and
    compare against the OFFICIAL ground-truth labels shipped in
    mimic-cxr-2.0.0-chexpert.csv.gz for the same study -- a sanity check that
    the labeler is wired correctly (H100_SCALING_PLAN.md's 11B entry), NOT a
    generator-quality metric. Skips study_ids missing from the CSV."""
    ground_truth = load_chexpert_ground_truth_labels(chexpert_csv, study_ids)
    matched_refs, matched_gt = [], []
    for ref, sid in zip(refs, study_ids):
        if sid in ground_truth:
            matched_refs.append(ref)
            matched_gt.append(ground_truth[sid])
    if not matched_refs:
        print("  CheXbert ground-truth cross-check skipped: no study_ids matched the CSV")
        return None

    our_labels = label_reports_with_chexbert(matched_refs, device=device)
    if our_labels is None:
        return None

    agreement = binary_f1_from_labels(our_labels, matched_gt, CHEXPERT_14_LABELS)
    agreement["num_matched"] = len(matched_refs)
    agreement["num_requested"] = len(study_ids)
    return agreement


def meteor_score_corpus(hyps: List[str], refs: List[str]) -> Optional[float]:
    """Corpus-average METEOR. Returns None (not 0.0) if nltk/wordnet are
    unavailable — distinguishes "not computed" from "scored zero" for callers.
    """
    try:
        import nltk
        from nltk.translate.meteor_score import meteor_score
        nltk.data.find("corpora/wordnet")
    except (ImportError, LookupError):
        return None
    scores = [meteor_score([ref.split()], hyp.split()) for hyp, ref in zip(hyps, refs)]
    return sum(scores) / len(scores) if scores else 0.0


def compute_all_metrics(
    hyps: List[str], refs: List[str], chexbert: bool = False, device: str = "cpu"
) -> Dict[str, Optional[float]]:
    """hyps/refs: whitespace-tokenized text, one pair per example. CheXbert F1
    (Phase 11B) is opt-in via chexbert=True -- it needs real weights/network
    on first use, unlike ROUGE-L/BLEU/METEOR which are always computed."""
    hyp_toks = [h.split() for h in hyps]
    ref_toks = [r.split() for r in refs]

    rouge_l = sum(rouge_l_score(h, r) for h, r in zip(hyp_toks, ref_toks)) / max(len(hyps), 1)
    bleu1 = corpus_bleu(hyp_toks, ref_toks, max_n=1)
    bleu4 = corpus_bleu(hyp_toks, ref_toks, max_n=4)
    meteor = meteor_score_corpus(hyps, refs)

    result = {
        "rouge_l": rouge_l,
        "bleu_1": bleu1,
        "bleu_4": bleu4,
        "meteor": meteor,
        "num_examples": len(hyps),
    }
    if chexbert:
        result["chexbert"] = compute_chexbert_metrics(hyps, refs, device=device)
    return result


# ---------------------------------------------------------------------------
# Smoke test — proves the decoding harness end-to-end without a checkpoint
# ---------------------------------------------------------------------------

def run_smoke_test(max_new_tokens: int = 16, dim: int = 32, k: int = 4) -> None:
    print("Building tiny random-init HybridLanguageModel + ImagePrefixMapper...")
    cfg = HybridConfig(
        vocab_size=100, dim=dim, num_layers=2,
        layer_pattern=["mamba", "mlstm"],
        max_position_embeddings=64,
        use_fast_path=False,
        use_tfla=False,
    )
    model = HybridLanguageModel(cfg)
    model.eval()
    mapper = ImagePrefixMapper(patch_dim=768, decoder_dim=dim, k=k)
    mapper.eval()

    patch_grid = torch.randn(1, 197, 768)
    prefix_embeds = mapper(patch_grid)
    input_ids = torch.randint(0, 100, (1, 5))

    print("Running greedy_decode (prefix-conditioned, max_new_tokens={})...".format(max_new_tokens))
    greedy_out = greedy_decode(model, input_ids, prefix_embeds=prefix_embeds, max_new_tokens=max_new_tokens)
    assert greedy_out.shape == (1, 5 + max_new_tokens), greedy_out.shape
    assert torch.isfinite(greedy_out.float()).all()
    print("  OK  shape={}".format(tuple(greedy_out.shape)))

    print("Running beam_search_decode (beam=3, prefix-conditioned, max_new_tokens={})...".format(max_new_tokens))
    beam_out = beam_search_decode(model, input_ids, prefix_embeds=prefix_embeds, beam_size=3, max_new_tokens=max_new_tokens)
    assert beam_out.shape == (1, 5 + max_new_tokens), beam_out.shape
    assert torch.isfinite(beam_out.float()).all()
    print("  OK  shape={}".format(tuple(beam_out.shape)))

    print("\nSmoke test passed: prefix-conditioned greedy + beam=3 decoding both run end-to-end.")
    print("(Random weights → outputs are not meaningful text; this only proves the harness.)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 11A report-generation metrics / decoding harness")
    parser.add_argument("--hyp-file", type=str, default=None,
                        help="Path to hypotheses, one per line, aligned with --ref-file")
    parser.add_argument("--ref-file", type=str, default=None,
                        help="Path to references, one per line, aligned with --hyp-file")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run the decoding harness end-to-end on a tiny random model "
                             "(no checkpoint/data needed) instead of computing metrics")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a trained ReportGenerationLightningModule .ckpt — "
                             "generates from real images (--parquet) and prints "
                             "generated-vs-reference text instead of computing metrics")
    parser.add_argument("--model-config", type=str, default="hybrid_150m_v2_rrg",
                        help="Model config name under configs/model/ (for --checkpoint mode)")
    parser.add_argument("--parquet", type=str, default=None,
                        help="Path to a validate.parquet/test.parquet (for --checkpoint mode)")
    parser.add_argument("--num-samples", type=int, default=5,
                        help="Number of samples to generate from (for --checkpoint mode)")
    parser.add_argument("--decode", type=str, default="greedy", choices=["greedy", "beam"],
                        help="Decoding strategy (for --checkpoint mode)")
    parser.add_argument("--beam-size", type=int, default=3,
                        help="Beam size when --decode beam (for --checkpoint mode)")
    parser.add_argument("--max-new-tokens", type=int, default=100,
                        help="Max tokens to generate per sample (for --checkpoint mode)")
    parser.add_argument("--retrieval-baseline", action="store_true",
                        help="Phase 11C: retrieve the nearest training report by stock "
                             "BiomedCLIP image similarity instead of generating (--parquet "
                             "is the query/validation set; --train-parquet is the gallery)")
    parser.add_argument("--train-parquet", type=str, default=None,
                        help="Path to the training parquet used as the retrieval gallery "
                             "(for --retrieval-baseline)")
    parser.add_argument("--max-gallery", type=int, default=0,
                        help="Cap the gallery to the first N training rows, 0=all "
                             "(for --retrieval-baseline)")
    parser.add_argument("--chexbert", action="store_true",
                        help="Phase 11B: also compute CheXbert F1 (14-label + "
                             "5-label RRG subset) between hyps and refs, via the "
                             "f1chexbert package. Degrades to 'skipped' if the "
                             "package/weights aren't available (see "
                             "label_reports_with_chexbert docstring)")
    parser.add_argument("--chexpert-csv", type=str, default=None,
                        help="Path to mimic-cxr-2.0.0-chexpert.csv.gz (Phase 8A). "
                             "With --chexbert in --checkpoint/--retrieval-baseline "
                             "mode, also cross-checks our CheXbert labeling of the "
                             "reference text against these official ground-truth "
                             "labels (labeler-wiring sanity check, not a generator "
                             "quality metric)")
    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test()
        return

    if args.retrieval_baseline:
        if not args.parquet or not args.train_parquet:
            parser.error("--parquet and --train-parquet are required with --retrieval-baseline")
        run_retrieval_baseline(args)
        return

    if args.checkpoint:
        if not args.parquet:
            parser.error("--parquet is required with --checkpoint")
        run_checkpoint_inspection(args)
        return

    if not args.hyp_file or not args.ref_file:
        parser.error("--hyp-file and --ref-file are required unless --smoke-test is given")

    hyps = Path(args.hyp_file).read_text().splitlines()
    refs = Path(args.ref_file).read_text().splitlines()
    if len(hyps) != len(refs):
        parser.error("--hyp-file has {} lines but --ref-file has {}".format(len(hyps), len(refs)))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = compute_all_metrics(hyps, refs, chexbert=args.chexbert, device=device)

    sep = "=" * 60
    print("\n" + sep)
    print("  REPORT GENERATION METRICS  (n={})".format(results["num_examples"]))
    print(sep)
    print("  ROUGE-L : {:.4f}".format(results["rouge_l"]))
    print("  BLEU-1  : {:.4f}".format(results["bleu_1"]))
    print("  BLEU-4  : {:.4f}".format(results["bleu_4"]))
    if results["meteor"] is None:
        print("  METEOR  : (skipped — nltk wordnet corpus not staged locally)")
    else:
        print("  METEOR  : {:.4f}".format(results["meteor"]))
    if args.chexbert:
        cb = results.get("chexbert")
        if cb is None:
            print("  CheXbert F1 : (skipped — see printed reason above)")
        else:
            print("  CheXbert F1 (14-label) micro/macro : {:.4f} / {:.4f}".format(
                cb["chexbert_14"]["micro_f1"], cb["chexbert_14"]["macro_f1"]))
            print("  CheXbert F1 (5-label)  micro/macro : {:.4f} / {:.4f}".format(
                cb["chexbert_5"]["micro_f1"], cb["chexbert_5"]["macro_f1"]))

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        results["hyp_file"] = args.hyp_file
        results["ref_file"] = args.ref_file
        results["timestamp"] = datetime.now().isoformat()
        out_path = out_dir / "report_gen_metrics.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print("\n  Results saved to " + str(out_path))

    return results


if __name__ == "__main__":
    main()
