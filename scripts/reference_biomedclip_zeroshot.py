"""Phase 6C-1 / 6C-2 — teacher-ceiling reference and tower-swap ablation.

NO TRAINING. Pure forward passes on the identical evaluation protocol used by
``evaluate_cxr_retrieval.py`` (MIMIC ``train[90%:]``, N=3063, strict-index
ground truth), so every number here is directly comparable to the student's
authoritative 0.1113 i2t R@10.

Two questions, one script:

  6C-1  What does STOCK BiomedCLIP score on our protocol?
        Its own text tower AND its own image tower, nothing fine-tuned.
        This is the anchor the whole KD design is pointed at. If it scores
        ~11-13% the student is at teacher parity and the 12% target was
        arbitrary; if it scores ~2-4% (the published range on comparable
        ~2.4k-study galleries) the student is already ~3x the teacher and
        there is real headroom.

  6C-2  Which tower binds?
        A 2x2 grid over {student text, BiomedCLIP text} x {fine-tuned ViT,
        stock ViT}. If substituting BiomedCLIP's text tower barely moves
        R@10, the text tower is not the bottleneck and text-side effort
        (including Phase 6E bidirectionalisation) is misallocated.

Every cell is also reported with the dedup-aware metric (6C-3 grouping), so
the templated-report tie-breaking artifact can be separated from model quality.

Usage:
    # 6C-1 only — no checkpoint needed
    python scripts/reference_biomedclip_zeroshot.py \
        --cache-dir /sc/home/$USER/dataset/mimic_cxr_cache \
        --output-dir results/phase6c

    # 6C-1 + 6C-2 full grid
    python scripts/reference_biomedclip_zeroshot.py \
        --checkpoint outputs/h100_kd_150m_v2_bs64_head4.24e-4/checkpoints/<best>.ckpt \
        --cache-dir /sc/home/$USER/dataset/mimic_cxr_cache \
        --output-dir results/phase6c

Interpret every gap against SE ~0.57pp at p~0.11, N=3063 — under ~1.1pp is noise.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from scripts.evaluate_cxr_retrieval import (
    BIOMEDCLIP_ID,
    IMAGE_SIZE,
    MIMIC_REPO,
    _img_transform,
    compute_retrieval_metrics,
    group_ids_from_texts,
    load_image_encoder,
    load_models,
)

# BiomedCLIP's text context window is fixed at 256 tokens by its open_clip config.
TEACHER_CONTEXT = 256


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_report_text(item: Dict) -> str:
    """Reproduce the training/eval text exactly (train_contrastive.py:378-382)."""
    findings = (item.get("findings") or "").strip()
    impression = (item.get("impression") or "").strip()
    if not (findings or impression):
        return ""
    return "Findings: {} Impression: {}".format(findings, impression).strip()


class _ImageOnlyDataset(Dataset):
    """Emits pixel_values only — text is handled separately per tower."""

    def __init__(self, hf_ds):
        self.ds = hf_ds
        self.transform = _img_transform()

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx):
        img = self.ds[idx].get("image")
        if img is None:
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return {"pixel_values": self.transform(img)}


class _TextOnlyDataset(Dataset):
    """Emits pre-tokenised student token ids for the hybrid text tower."""

    def __init__(self, texts: List[str], tokenizer, max_length: int = 256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_images(image_enc, hf_ds, device: str, batch_size: int, num_workers: int) -> np.ndarray:
    loader = DataLoader(
        _ImageOnlyDataset(hf_ds), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    out = []
    for batch in tqdm(loader, desc="  images", unit="batch"):
        z = image_enc(batch["pixel_values"].to(device))
        out.append(F.normalize(z.float(), dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)


@torch.no_grad()
def encode_text_biomedclip(clip_model, tokenizer, texts: List[str], device: str,
                           batch_size: int) -> np.ndarray:
    """BiomedCLIP's own text tower -> 512-d joint space."""
    out = []
    for i in tqdm(range(0, len(texts), batch_size), desc="  text (BiomedCLIP)", unit="batch"):
        chunk = texts[i:i + batch_size]
        tokens = tokenizer(chunk).to(device)
        z = clip_model.encode_text(tokens)
        out.append(F.normalize(z.float(), dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)


@torch.no_grad()
def encode_text_student(text_enc, texts: List[str], tokenizer, device: str,
                        batch_size: int, num_workers: int, max_length: int) -> np.ndarray:
    """Student hybrid Mamba/xLSTM text tower -> already L2-normalised by encode()."""
    loader = DataLoader(
        _TextOnlyDataset(texts, tokenizer, max_length), batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    out = []
    for batch in tqdm(loader, desc="  text (student)", unit="batch"):
        z = text_enc.encode(
            batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        out.append(z.float().cpu().numpy())
    return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt_row(label: str, m: Dict[str, float]) -> str:
    return (
        "  {:<34s} i2t R@1 {:.4f} | R@5 {:.4f} | R@10 {:.4f}   "
        "t2i R@10 {:.4f}".format(
            label, m["i2t_R@1"], m["i2t_R@5"], m["i2t_R@10"], m["t2i_R@10"]
        )
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None,
                    help="Student joint checkpoint. Omit to run 6C-1 (stock BiomedCLIP) only.")
    ap.add_argument("--cache-dir", default=os.environ.get(
        "MIMIC_CACHE_DIR", str(Path.home() / "dataset" / "mimic_cxr_cache")))
    ap.add_argument("--output-dir", default="results/phase6c")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=256,
                    help="Student tokenizer context (matches mimic_cxr.yaml).")
    ap.add_argument("--student-tokenizer", default="gpt2")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Phase 6C — teacher-ceiling reference + tower-swap ablation")
    print("Protocol: {} split=train[90%:] (identical to evaluate_cxr_retrieval.py)".format(MIMIC_REPO))
    print("=" * 78)

    # --- Data -------------------------------------------------------------
    hf_ds = load_dataset(MIMIC_REPO, split="train[90%:]", cache_dir=args.cache_dir)
    texts = [build_report_text(hf_ds[i]) for i in range(len(hf_ds))]
    n = len(texts)
    groups = group_ids_from_texts(texts)
    n_groups = int(len(np.unique(groups)))
    print("\nLoaded {} pairs. Distinct normalised reports: {} "
          "({} duplicated positions, {:.1f}%).".format(
              n, n_groups, n - n_groups, 100.0 * (n - n_groups) / max(n, 1)))

    # --- Towers -----------------------------------------------------------
    import open_clip

    print("\nLoading stock BiomedCLIP (text + image towers) ...")
    clip_model, _ = open_clip.create_model_from_pretrained("hf-hub:" + BIOMEDCLIP_ID)
    clip_model = clip_model.to(device).eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)
    teacher_tok = open_clip.get_tokenizer("hf-hub:" + BIOMEDCLIP_ID)

    towers_img = {}      # type: Dict[str, np.ndarray]
    towers_txt = {}      # type: Dict[str, np.ndarray]

    print("\nEncoding with stock BiomedCLIP:")
    towers_img["stock_vit"] = encode_images(
        clip_model.visual, hf_ds, device, args.batch_size, args.num_workers)
    towers_txt["biomedclip_text"] = encode_text_biomedclip(
        clip_model, teacher_tok, texts, device, args.batch_size)

    if args.checkpoint:
        print("\nLoading student checkpoint for the tower-swap grid ...")
        text_enc, img_proj, image_enc = load_models(args.checkpoint, device)
        if img_proj is not None:
            raise SystemExit(
                "This checkpoint has a legacy img_proj head. The 6C grid assumes the "
                "Phase 8+ architecture where clip_model.visual is already in the 512-d "
                "joint space. Re-run 6C on a Phase 8+ checkpoint."
            )
        text_enc.eval()
        image_enc.eval()
        student_tok = AutoTokenizer.from_pretrained(args.student_tokenizer)
        if student_tok.pad_token is None:
            student_tok.pad_token = student_tok.eos_token

        print("\nEncoding with the student / fine-tuned towers:")
        towers_img["finetuned_vit"] = encode_images(
            image_enc, hf_ds, device, args.batch_size, args.num_workers)
        towers_txt["student_text"] = encode_text_student(
            text_enc, texts, student_tok, device,
            args.batch_size, args.num_workers, args.max_length)

    # --- Grid -------------------------------------------------------------
    results = {}         # type: Dict[str, Dict[str, float]]
    print("\n" + "=" * 78)
    print("RESULTS — strict-index ground truth (authoritative protocol)")
    print("=" * 78)
    for img_name, img_emb in towers_img.items():
        for txt_name, txt_emb in towers_txt.items():
            cell = "{} x {}".format(img_name, txt_name)
            m = compute_retrieval_metrics(img_emb, txt_emb)
            m_dedup = compute_retrieval_metrics(img_emb, txt_emb, groups=groups)
            results[cell] = {"strict": m, "dedup_aware": m_dedup}
            print(_fmt_row(cell, m))

    print("\n" + "-" * 78)
    print("Same cells, dedup-aware (a hit counts if the retrieved item shares the")
    print("query's normalised report text — separates the tie-breaking artifact)")
    print("-" * 78)
    for cell, r in results.items():
        print(_fmt_row(cell, r["dedup_aware"]))

    # --- Oracle ceiling ---------------------------------------------------
    # With strict-index ground truth, a duplicate group of size m caps
    # P(hit@k) at min(1, k/m) even for a perfect model. Compute that ceiling.
    _, counts = np.unique(groups, return_counts=True)
    sizes = counts[np.searchsorted(np.unique(groups), groups)]
    oracle = {
        "R@{}".format(k): float(np.minimum(1.0, float(k) / sizes).mean())
        for k in (1, 5, 10)
    }
    print("\nOracle ceiling under strict-index ground truth (perfect model, "
          "arbitrary tie-breaking):")
    print("  R@1 {R@1:.4f} | R@5 {R@5:.4f} | R@10 {R@10:.4f}".format(**oracle))

    # --- Interpretation ---------------------------------------------------
    stock_cell = "stock_vit x biomedclip_text"
    stock_r10 = results[stock_cell]["strict"]["i2t_R@10"]
    print("\n" + "=" * 78)
    print("6C-1 — STOCK BiomedCLIP zero-shot i2t R@10 = {:.4f} ({:.2f}%)".format(
        stock_r10, stock_r10 * 100))
    print("       Student authoritative reference: 0.1113 (11.13%)")
    if stock_r10 >= 0.10:
        print("       => Student is at TEACHER PARITY. The 12% target sits at or above")
        print("          the anchor the KD objective points at. Report Phase 6 as a")
        print("          parity result and read 6D through that lens.")
    else:
        print("       => Student is ABOVE the teacher by {:.2f}pp. Real headroom exists;".format(
            (0.1113 - stock_r10) * 100))
        print("          the plateau is not teacher parity. Proceed to 6D.")
    if len(towers_txt) > 1:
        swap = results["finetuned_vit x biomedclip_text"]["strict"]["i2t_R@10"]
        student = results["finetuned_vit x student_text"]["strict"]["i2t_R@10"]
        delta = (swap - student) * 100
        print("\n6C-2 — text-tower swap on the FINE-TUNED ViT:")
        print("       student text {:.4f} vs BiomedCLIP text {:.4f}  (delta {:+.2f}pp)".format(
            student, swap, delta))
        if abs(delta) < 1.1:
            print("       => Within noise (SE ~0.57pp). The text tower is NOT the binding")
            print("          constraint — deprioritise Phase 6E and put effort on the")
            print("          image side and the objective (6D-1/6D-3).")
        elif delta > 0:
            print("       => BiomedCLIP's text tower is materially better. The student text")
            print("          tower IS a real constraint — Phase 6E is justified.")
        else:
            print("       => The student text tower BEATS the teacher's on our data.")
            print("          Phase 6E is unjustified; the ceiling is elsewhere.")
    print("=" * 78)

    # --- Persist ----------------------------------------------------------
    payload = {
        "timestamp": datetime.now().isoformat(),
        "protocol": {
            "repo": MIMIC_REPO, "split": "train[90%:]", "N": n,
            "distinct_reports": n_groups,
            "duplicate_fraction": (n - n_groups) / max(n, 1),
        },
        "checkpoint": args.checkpoint,
        "oracle_ceiling_strict_index": oracle,
        "student_authoritative_reference_i2t_R@10": 0.1113,
        "results": results,
    }
    out_path = out_dir / "phase6c_reference.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print("\nWrote {}".format(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
