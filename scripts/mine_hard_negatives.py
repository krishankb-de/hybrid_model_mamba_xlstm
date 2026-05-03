"""Mine hard negative text pairs from MIMIC-CXR training split.

For each report, finds the top-K most similar *other* reports using the
v2 checkpoint text encoder.  Saves a LongTensor of shape [N, K] where
entry [i, j] is the j-th hardest negative index for report i.

Usage (run from repo root on willi):
    python scripts/mine_hard_negatives.py \
        --checkpoint ./outputs/joint_mimic_cxr_v2/checkpoints/contrastive-step=001637-val/total_loss=2.4715.ckpt \
        --cache-dir /scratch/bhushkri/mimic_cxr_cache \
        --output-file ./outputs/mimic_hard_neg_index.pt \
        --k 50 \
        --batch-size 128
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hybrid_xmamba.models.configuration_hybrid import HybridConfig
from hybrid_xmamba.models.hybrid_lm import HybridTextEncoder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIMIC_REPO = "itsanmolgupta/mimic-cxr-dataset"
TOKENIZER_NAME = "gpt2"

# ---------------------------------------------------------------------------
# Checkpoint loading (mirrors evaluate_cxr_retrieval.py)
# ---------------------------------------------------------------------------

def _load_text_encoder(ckpt_path: str, device: str) -> HybridTextEncoder:
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)

    cfg = HybridConfig(
        vocab_size=50257,
        dim=512,
        num_layers=8,
        layer_pattern=["mamba", "mamba", "mlstm"],
        max_position_embeddings=1024,
        pooling_strategy="attention",
    )
    model = HybridTextEncoder(cfg, embed_dim=512)

    model_state = {
        k[len("model."):]: v
        for k, v in state.items()
        if k.startswith("model.")
    }
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    kept = [k for k in unexpected if not any(
        k.startswith(p) for p in ("img_proj.", "image_encoder.", "distill_proj.", "teacher.")
    )]
    print(f"  Text encoder loaded. Missing={len(missing)}, Unexpected (non-model)={len(kept)}")
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def _get_text(item, findings_field: str = "findings",
              impression_field: str = "impression") -> str:
    findings = item.get(findings_field, "") or ""
    impression = item.get(impression_field, "") or ""
    return f"Findings: {findings} Impression: {impression}".strip()


def encode_corpus(
    model: HybridTextEncoder,
    tokenizer,
    ds,
    max_length: int,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """Encode all reports → L2-normalised embeddings [N, D] (CPU)."""
    all_embs: List[torch.Tensor] = []
    n = len(ds)
    for start in tqdm(range(0, n, batch_size), desc="Encoding corpus"):
        end = min(start + batch_size, n)
        batch_items = [ds[i] for i in range(start, end)]
        texts = [_get_text(item) for item in batch_items]
        enc = tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            embs = model.encode(input_ids, attention_mask=attention_mask)
        all_embs.append(embs.cpu())
    return torch.cat(all_embs, dim=0)  # [N, D]


# ---------------------------------------------------------------------------
# Hard negative mining (chunked matmul, no FAISS dependency)
# ---------------------------------------------------------------------------

def mine_hard_negatives(
    embs: torch.Tensor,
    k: int,
    chunk_size: int = 512,
    device: str = "cpu",
) -> torch.Tensor:
    """For each sample i return indices of top-K most similar *other* samples.

    Args:
        embs: L2-normalised [N, D] embeddings on CPU.
        k: Number of hard negatives to keep.
        chunk_size: Rows processed per GPU chunk.
        device: Device for matmul (use "cuda" if available).

    Returns:
        LongTensor [N, K] of hard negative indices.
    """
    N = embs.shape[0]
    embs_dev = embs.to(device)
    hard_neg_idx = torch.zeros(N, k, dtype=torch.long)

    for start in tqdm(range(0, N, chunk_size), desc="Mining hard negs"):
        end = min(start + chunk_size, N)
        q = embs_dev[start:end]           # [C, D]
        sim = q @ embs_dev.T              # [C, N]

        # Mask out self-similarity
        for local_i in range(end - start):
            sim[local_i, start + local_i] = -1e9

        # top-(K+1) to be safe; exclude self already masked
        topk_vals, topk_idx = sim.topk(k, dim=-1, largest=True, sorted=True)
        hard_neg_idx[start:end] = topk_idx.cpu()

    return hard_neg_idx  # [N, K]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Mine FAISS hard negatives for MIMIC-CXR")
    parser.add_argument("--checkpoint", required=True, help="Path to v2 joint checkpoint .ckpt")
    parser.add_argument("--cache-dir", default="/scratch/bhushkri/mimic_cxr_cache")
    parser.add_argument("--output-file", default="./outputs/mimic_hard_neg_index.pt")
    parser.add_argument("--k", type=int, default=50,
                        help="Hard negatives to store per anchor (training uses top-4)")
    parser.add_argument("--batch-size", type=int, default=128, help="Encoding batch size")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=512,
                        help="Mining chunk size (reduce if OOM)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load MIMIC train split
    print(f"Loading {MIMIC_REPO} train[:90%] ...")
    ds = load_dataset(MIMIC_REPO, split="train[:90%]", cache_dir=args.cache_dir)
    ds = ds.filter(lambda x: x.get("image") is not None)
    print(f"  {len(ds)} samples after filtering None images")

    # Load tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = _load_text_encoder(args.checkpoint, device)

    # Encode corpus
    print("\nEncoding all reports ...")
    embs = encode_corpus(model, tokenizer, ds, args.max_length, args.batch_size, device)
    print(f"  Corpus embeddings: {embs.shape}, norm mean={embs.norm(dim=-1).mean():.4f}")

    # Mine hard negatives
    print(f"\nMining top-{args.k} hard negatives per anchor ...")
    mine_device = device
    hard_neg_idx = mine_hard_negatives(
        embs, k=args.k, chunk_size=args.chunk_size, device=mine_device
    )
    print(f"  Hard neg index: {hard_neg_idx.shape}")

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    torch.save({"indices": hard_neg_idx, "n_samples": len(ds), "k": args.k}, args.output_file)
    print(f"\nSaved hard neg index to: {args.output_file}")
    print(f"  Shape: {hard_neg_idx.shape} — use top-4 columns during training (hard_neg_k=4)")


if __name__ == "__main__":
    main()
