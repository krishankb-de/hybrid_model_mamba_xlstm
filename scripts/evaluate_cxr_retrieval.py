"""Phase 6 — CXR image-text retrieval evaluation for joint checkpoint.

Computes i2t and t2i R@1/5/10 on:
  indiana  — IU-Xray test (743 pairs, cross-dataset generalization)
  mimic    — MIMIC-CXR val 10% slice (~3063 pairs, in-distribution sanity)

Usage:
    python scripts/evaluate_cxr_retrieval.py \
        --checkpoint outputs/joint_mimic_cxr/checkpoints/contrastive-step=001915-val/total_loss=1.9140.ckpt \
        --dataset indiana \
        --output-dir results/phase6_indiana

    python scripts/evaluate_cxr_retrieval.py \
        --checkpoint outputs/joint_mimic_cxr/checkpoints/contrastive-step=001915-val/total_loss=1.9140.ckpt \
        --dataset mimic \
        --output-dir results/phase6_mimic

Phase 6 decision gate (Indiana i2t R@10):
    >= 0.40          → done
    [0.25, 0.40)     → Phase 7 FAISS hard-neg mining
    < 0.25 OR text collapse → debug loss weights, return to Phase 2
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

from hybrid_xmamba.models.configuration_hybrid import HybridConfig
from hybrid_xmamba.models.hybrid_lm import HybridTextEncoder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BIOMEDCLIP_ID = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
FALLBACK_CLIP_ID = "laion/CLIP-ViT-B-16-laion2B-s34B-b88K"

IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
IMAGE_STD  = [0.26862954, 0.26130258, 0.27577711]
IMAGE_SIZE = 224

INDIANA_REPO  = "MLforHealthcare/Indiana_University_Chest_X-ray_Collection"
MIMIC_REPO    = "itsanmolgupta/mimic-cxr-dataset"


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _strip_model_prefix(state_dict: Dict) -> Tuple[Dict, Dict, Optional[Dict]]:
    """Split state_dict into text-encoder keys, img_proj keys, image_encoder keys."""
    text_enc, img_proj_keys, img_enc = {}, {}, {}
    for k, v in state_dict.items():
        k_orig = k
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        if k.startswith("model."):
            text_enc[k[len("model."):]] = v
        elif k.startswith("img_proj."):
            img_proj_keys[k[len("img_proj."):]] = v
        elif k.startswith("image_encoder."):
            img_enc[k[len("image_encoder."):]] = v
        # skip teacher.*, distill_proj.*, optimizer states
    return text_enc, img_proj_keys, img_enc


def load_text_encoder(text_state: Dict, device: str) -> HybridTextEncoder:
    num_layers = max(
        (int(m.group(1)) + 1 for k in text_state
         for m in [re.search(r"lm\.layers\.(\d+)\.", k)] if m),
        default=8,
    )
    dim = next(
        (int(v.shape[1]) for k, v in text_state.items()
         if "token_embedding.weight" in k),
        512,
    )
    # Auto-detect architecture from the checkpoint so v1 AND v2 load exact-match.
    # (Hardcoding [mamba,mamba,mlstm]+pre_rms silently mismapped the v2 backbone:
    #  wrong layer pattern + dropped HybridNorm weights → wrong retrieval numbers.)
    #   - per-layer type: mamba blocks carry a mixer.A_log; mLSTM blocks do not.
    #   - norm topology: HybridNorm adds dt_norm/B_norm/C_norm (mamba) + v_norm (mlstm).
    layer_pattern = []
    for i in range(num_layers):
        mixer_keys = [k for k in text_state if f"lm.layers.{i}.mixer." in k]
        is_mamba = any("A_log" in k or "conv1d" in k for k in mixer_keys)
        layer_pattern.append("mamba" if is_mamba else "mlstm")
    norm_topology = "hybrid" if any(
        (".dt_norm." in k or ".v_norm." in k or ".B_norm." in k or ".C_norm." in k)
        for k in text_state
    ) else "pre_rms"
    print(f"  [text encoder] detected layer_pattern={layer_pattern}, "
          f"norm_topology={norm_topology}")
    cfg = HybridConfig(
        vocab_size=50257,
        dim=dim,
        num_layers=num_layers,
        layer_pattern=layer_pattern,
        norm_topology=norm_topology,
        max_position_embeddings=1024,
        pooling_strategy="attention",
    )
    model = HybridTextEncoder(cfg, embed_dim=dim)
    missing, unexpected = model.load_state_dict(text_state, strict=False)
    if missing:
        print(f"  [text encoder] {len(missing)} missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"  [text encoder] {len(unexpected)} unexpected keys (first 5): {unexpected[:5]}")
    return model.to(device).eval()


def build_img_proj(img_proj_state: Dict, device: str) -> nn.Sequential:
    w0 = img_proj_state["0.weight"]   # (txt_out, img_out)
    w2 = img_proj_state["2.weight"]   # (txt_out, txt_out)
    img_out, txt_out = w0.shape[1], w0.shape[0]
    proj = nn.Sequential(
        nn.Linear(img_out, txt_out, bias=False),
        nn.GELU(),
        nn.Linear(txt_out, txt_out, bias=False),
    )
    proj.load_state_dict(img_proj_state, strict=True)
    return proj.to(device).eval()


def load_image_encoder(device: str):
    import open_clip

    def _get_dim(clip_model) -> int:
        if hasattr(clip_model, 'embed_dim'):
            return clip_model.embed_dim
        if hasattr(clip_model.visual, 'output_dim'):
            return clip_model.visual.output_dim
        if hasattr(clip_model.visual, 'embed_dim'):
            return clip_model.visual.embed_dim
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            return clip_model.visual.cpu()(dummy).shape[-1]

    try:
        clip_model, _ = open_clip.create_model_from_pretrained("hf-hub:" + BIOMEDCLIP_ID)
        model_name = "BiomedCLIP"
    except Exception as e:
        print(f"  BiomedCLIP load failed ({e}), falling back to {FALLBACK_CLIP_ID}")
        clip_model, _ = open_clip.create_model_from_pretrained("hf-hub:" + FALLBACK_CLIP_ID)
        model_name = "laion CLIP (fallback)"

    dim = _get_dim(clip_model)
    print(f"  ✓ {model_name} image encoder loaded (dim={dim})")
    enc = clip_model.visual.to(device).eval()
    for p in enc.parameters():
        p.requires_grad = False
    return enc


def load_models(checkpoint_path: str, device: str):
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw = ckpt.get("state_dict", ckpt)
    text_state, img_proj_state, _ = _strip_model_prefix(raw)

    if not text_state:
        raise RuntimeError(
            "No model.* keys found in checkpoint. "
            f"First 10 keys: {list(raw.keys())[:10]}"
        )
    print("  Building text encoder ...")
    text_enc = load_text_encoder(text_state, device)
    print(f"  ✓ Text encoder: {sum(p.numel() for p in text_enc.parameters())/1e6:.1f}M params")

    # Phase 8: img_proj deleted from architecture. Checkpoints from Phase 8+
    # have no img_proj.* keys — clip_model.visual already outputs 512-d joint
    # embeddings, so no projection is needed. img_proj=None → passthrough.
    if img_proj_state:
        img_proj = build_img_proj(img_proj_state, device)
        w0 = img_proj_state["0.weight"]
        print(f"  ✓ img_proj: ({w0.shape[1]} → {w0.shape[0]}) MLP")
    else:
        img_proj = None
        print("  ✓ img_proj: None (Phase 8+ checkpoint — BiomedCLIP visual already in joint space)")

    print("  Loading BiomedCLIP image encoder (fresh) ...")
    image_enc = load_image_encoder(device)

    return text_enc, img_proj, image_enc


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def _img_transform() -> T.Compose:
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.CenterCrop(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ])


class IndianaEvalDataset(Dataset):
    """IU-Xray test split: 743 (image, report) pairs.

    HF repo: MLforHealthcare/Indiana_University_Chest_X-ray_Collection
    Fields: image (PIL), report (str)
    """

    def __init__(self, hf_ds, tokenizer, max_length: int = 256):
        self.ds = hf_ds
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = _img_transform()

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        text = (item.get("report") or item.get("findings") or item.get("impression") or "").strip()

        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        img = item.get("image")
        if img is None:
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        if img.mode != "RGB":
            img = img.convert("RGB")

        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "pixel_values":   self.transform(img),
            "idx":            idx,
        }


class MIMICValDataset(Dataset):
    """MIMIC-CXR val slice (train[90%:]): ~3063 (image, findings+impression) pairs.

    HF repo: itsanmolgupta/mimic-cxr-dataset
    Fields: image (PIL), findings (str), impression (str)
    """

    def __init__(self, hf_ds, tokenizer, max_length: int = 256):
        self.ds = hf_ds
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = _img_transform()

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        findings   = (item.get("findings")   or "").strip()
        impression = (item.get("impression") or "").strip()
        text = f"Findings: {findings} Impression: {impression}".strip() if (findings or impression) else ""

        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        img = item.get("image")
        if img is None:
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        if img.mode != "RGB":
            img = img.convert("RGB")

        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "pixel_values":   self.transform(img),
            "idx":            idx,
        }


def build_dataloader(
    dataset_name: str,
    cache_dir: str,
    tokenizer,
    max_length: int = 256,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, int]:
    if dataset_name == "indiana":
        print(f"Loading Indiana/IU-Xray test split from {INDIANA_REPO} ...")
        hf_ds = load_dataset(INDIANA_REPO, split="test", cache_dir=cache_dir)
        print(f"  {len(hf_ds)} pairs, columns: {hf_ds.column_names}")
        ds = IndianaEvalDataset(hf_ds, tokenizer, max_length)
    elif dataset_name == "mimic":
        print(f"Loading MIMIC-CXR val slice (train[90%:]) from {MIMIC_REPO} ...")
        hf_ds = load_dataset(MIMIC_REPO, split="train[90%:]", cache_dir=cache_dir)
        print(f"  {len(hf_ds)} pairs, columns: {hf_ds.column_names}")
        ds = MIMICValDataset(hf_ds, tokenizer, max_length)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name!r}. Use 'indiana' or 'mimic'.")

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader, len(ds)


# ---------------------------------------------------------------------------
# Embedding + retrieval
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_dataset(
    loader: DataLoader,
    text_enc: HybridTextEncoder,
    img_proj: nn.Sequential,
    image_enc: nn.Module,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    all_img, all_txt = [], []

    for batch in tqdm(loader, desc="Encoding", unit="batch"):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values   = batch["pixel_values"].to(device)

        # Text embeddings
        z_text = text_enc.encode(input_ids, attention_mask=attention_mask)
        all_txt.append(z_text.cpu().float().numpy())

        # Image embeddings — Phase 8+: img_proj=None means BiomedCLIP visual
        # already outputs 512-d joint embeddings; just normalise directly.
        z_img_raw = image_enc(pixel_values)
        z_img = F.normalize(
            (img_proj(z_img_raw.float()) if img_proj is not None else z_img_raw.float()),
            dim=-1,
        )
        all_img.append(z_img.cpu().float().numpy())

    return np.concatenate(all_img, axis=0), np.concatenate(all_txt, axis=0)


def compute_retrieval_metrics(
    img_embs: np.ndarray,
    txt_embs: np.ndarray,
) -> Dict[str, float]:
    """Compute i2t and t2i R@1/5/10. Both embeddings must be L2-normalised."""
    sim = img_embs @ txt_embs.T     # (N, N)
    N = sim.shape[0]
    ks = [1, 5, 10]

    def _recall_at_k(mat: np.ndarray, ks: List[int]) -> Dict[int, float]:
        results = {}
        for k in ks:
            k_eff = min(k, N)
            top_k = np.argpartition(-mat, kth=k_eff - 1, axis=1)[:, :k_eff]
            gt = np.arange(N)[:, None]
            hits = (top_k == gt).any(axis=1)
            results[k] = float(hits.mean())
        return results

    i2t = _recall_at_k(sim, ks)
    t2i = _recall_at_k(sim.T, ks)

    return {
        "i2t_R@1":  i2t[1],  "i2t_R@5":  i2t[5],  "i2t_R@10": i2t[10],
        "t2i_R@1":  t2i[1],  "t2i_R@5":  t2i[5],  "t2i_R@10": t2i[10],
        "mean_R@10": (i2t[10] + t2i[10]) / 2,
        "N": N,
    }


# ---------------------------------------------------------------------------
# Decision gate
# ---------------------------------------------------------------------------

def print_decision_gate(metrics: Dict[str, float], dataset_name: str) -> None:
    r10 = metrics["i2t_R@10"]
    print("\n" + "=" * 55)
    print("PHASE 6 DECISION GATE")
    print(f"  Dataset : {dataset_name}")
    print(f"  i2t R@10: {r10:.4f}  ({r10*100:.2f}%)")
    print(f"  t2i R@10: {metrics['t2i_R@10']:.4f}  ({metrics['t2i_R@10']*100:.2f}%)")
    print()
    if r10 >= 0.40:
        print("  ✓ PASS  — R@10 >= 0.40. Proceed to STS-B check then done.")
    elif r10 >= 0.25:
        print("  ~ PARTIAL — R@10 in [0.25, 0.40). Trigger Phase 7 FAISS hard-neg mining.")
    else:
        print("  ✗ FAIL  — R@10 < 0.25. Debug loss weights; return to Phase 2.")
    print("=" * 55)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6 CXR retrieval evaluation")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to JointMultiTask .ckpt file")
    parser.add_argument("--dataset", required=True, choices=["indiana", "mimic"],
                        help="'indiana' (743 test) or 'mimic' (~3063 val)")
    parser.add_argument("--output-dir", default="results/phase6_eval",
                        help="Directory for JSON result file")
    parser.add_argument("--cache-dir", default="/scratch/bhushkri/indiana_cxr_cache",
                        help="HuggingFace cache directory")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("=" * 55)
    print("Phase 6: CXR Image-Text Retrieval Evaluation")
    print(f"  Dataset    : {args.dataset}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Device     : {args.device}")
    print("=" * 55)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available — falling back to CPU")
        args.device = "cpu"

    # Models
    text_enc, img_proj, image_enc = load_models(args.checkpoint, args.device)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Data
    cache = args.cache_dir
    if args.dataset == "mimic":
        cache = cache.replace("indiana_cxr_cache", "mimic_cxr_cache")
    loader, n_samples = build_dataloader(
        args.dataset, cache, tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"\nEncoding {n_samples} samples ...")

    # Encode
    img_embs, txt_embs = encode_dataset(loader, text_enc, img_proj, image_enc, args.device)
    print(f"  img_embs: {img_embs.shape}, txt_embs: {txt_embs.shape}")

    # Verify embeddings are normalised
    img_norms = np.linalg.norm(img_embs, axis=1)
    txt_norms = np.linalg.norm(txt_embs, axis=1)
    print(f"  img norm: {img_norms.mean():.4f} ± {img_norms.std():.4f} (should be ~1.0)")
    print(f"  txt norm: {txt_norms.mean():.4f} ± {txt_norms.std():.4f} (should be ~1.0)")

    # Cosine similarity distribution (sanity check for collapse)
    diag_cos = (img_embs * txt_embs).sum(axis=1)
    print(f"  paired cosine sim: {diag_cos.mean():.4f} ± {diag_cos.std():.4f}")

    # Metrics
    print("\nComputing retrieval metrics ...")
    metrics = compute_retrieval_metrics(img_embs, txt_embs)

    print("\n" + "-" * 55)
    print(f"Results ({args.dataset}, N={metrics['N']}):")
    print(f"  i2t  R@1 = {metrics['i2t_R@1']:.4f}   ({metrics['i2t_R@1']*100:.2f}%)")
    print(f"  i2t  R@5 = {metrics['i2t_R@5']:.4f}   ({metrics['i2t_R@5']*100:.2f}%)")
    print(f"  i2t R@10 = {metrics['i2t_R@10']:.4f}   ({metrics['i2t_R@10']*100:.2f}%)")
    print(f"  t2i  R@1 = {metrics['t2i_R@1']:.4f}   ({metrics['t2i_R@1']*100:.2f}%)")
    print(f"  t2i  R@5 = {metrics['t2i_R@5']:.4f}   ({metrics['t2i_R@5']*100:.2f}%)")
    print(f"  t2i R@10 = {metrics['t2i_R@10']:.4f}   ({metrics['t2i_R@10']*100:.2f}%)")
    print(f"  mean R@10= {metrics['mean_R@10']:.4f}   ({metrics['mean_R@10']*100:.2f}%)")

    print_decision_gate(metrics, args.dataset)

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"phase6_{args.dataset}_{stamp}.json"
    result = {
        "timestamp": stamp,
        "dataset": args.dataset,
        "checkpoint": str(args.checkpoint),
        "n_samples": n_samples,
        "metrics": metrics,
        "embedding_stats": {
            "img_norm_mean": float(img_norms.mean()),
            "txt_norm_mean": float(txt_norms.mean()),
            "paired_cosine_mean": float(diag_cos.mean()),
            "paired_cosine_std":  float(diag_cos.std()),
        },
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
