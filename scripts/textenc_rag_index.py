"""TEXTENC_PLAN.md T3: retrieval index for retrieval-augmented generation, and the re-rank floors.

For every query study (train, validate, test) and a gallery of all training studies:
  1. image-image cosine with the Stage-1 image tower (512-d, the published retrieval vector),
     top `--candidates` (default 50), never the query's own patient (or study);
  2. re-rank those candidates by image-text cosine between the query image and the Stage-1 text
     encoder's embedding of each candidate report (the text encoder used at inference);
  3. keep the top `--k` after re-ranking.

Writes, under --out (cluster only: these are MIMIC-derived, rule R5):
  rag_<split>.pt        {query_rows, cand_rows (N,C), ii (N,C), it (N,C), topk_rows (N,k), k, ...}
  floors/<name>/hyps.txt, refs.txt for the test split:
      textenc_floor_ii       copy the top-1 by image-image cosine (new tower, no text encoder)
      textenc_floor_rerank   copy the top-1 after the text re-rank        (= arm F-rr)
Rows index into the split parquets in file order.

    python scripts/textenc_rag_index.py --stage1-ckpt outputs/textenc_stage1_m3/checkpoints/last.ckpt \
        --data /sc/home/$USER/dataset/mimic_full --out results/textenc_rag
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

SPLITS = ("train", "validate", "test")


def report_text(findings, impression) -> str:
    """The Stage-1 training text (train_contrastive.py, concatenate_sections: true)."""
    return f"Findings: {findings or ''} Impression: {impression or ''}".strip()


def published_reference(findings, impression) -> str:
    """The reference string of the published evaluation (evaluate_report_generation.py), kept
    verbatim, so hyps/refs pair line by line with every published result dir."""
    return f"Findings: {findings} Impression: {impression}".strip()


@torch.no_grad()
def rerank_candidates(q_img: torch.Tensor, gal_img: torch.Tensor, gal_txt: torch.Tensor,
                      q_subject: Sequence[int], gal_subject: Sequence[int],
                      n_candidates: int, k: int, chunk: int = 1024,
                      device: Optional[str] = None) -> Dict[str, torch.Tensor]:
    """Image-image top-C (same patient excluded), then image-text re-rank, keep top-k.

    All embeddings are L2-normalised (N, D). Returns CPU tensors:
      cand (Nq, C) gallery rows by descending image-image score, ii (Nq, C) those scores,
      it (Nq, C) image-text scores of the same candidates, topk (Nq, k) gallery rows by
      descending image-text score among the candidates.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    g_img = gal_img.to(device)
    g_txt = gal_txt.to(device)
    g_subj = torch.as_tensor(list(gal_subject), device=device)
    q_subj_all = torch.as_tensor(list(q_subject))
    n_candidates = min(n_candidates, g_img.shape[0])
    k = min(k, n_candidates)
    out = {"cand": [], "ii": [], "it": [], "topk": []}
    for s in range(0, q_img.shape[0], chunk):
        q = q_img[s:s + chunk].to(device)
        sims = q @ g_img.t()                                          # (b, G)
        same = q_subj_all[s:s + chunk].to(device).unsqueeze(1) == g_subj.unsqueeze(0)
        sims = sims.masked_fill(same, float("-inf"))
        ii, cand = sims.topk(n_candidates, dim=1)                    # (b, C)
        it = torch.einsum("bd,bcd->bc", q, g_txt[cand])              # (b, C)
        order = it.argsort(dim=1, descending=True)[:, :k]
        out["cand"].append(cand.cpu())
        out["ii"].append(ii.cpu())
        out["it"].append(it.cpu())
        out["topk"].append(cand.gather(1, order).cpu())
    return {key: torch.cat(v) for key, v in out.items()}


def subject_ids(df, data_dir: Path) -> List[int]:
    if "subject_id" in df.columns:
        return [int(x) for x in df["subject_id"]]
    import pandas as pd
    meta = pd.read_csv(data_dir / "mimic-cxr-2.0.0-metadata.csv.gz", usecols=["subject_id", "study_id"])
    by_study = meta.groupby("study_id")["subject_id"].first()
    return [int(by_study[int(s)]) for s in df["study_id"]]


def load_stage1(ckpt_path: str, model_config: str, device: str):
    """(image tower, text encoder) from a HybridContrastiveLightningModule checkpoint."""
    import open_clip
    from omegaconf import OmegaConf
    from hybrid_xmamba.models.configuration_hybrid import HybridConfig
    from hybrid_xmamba.models.hybrid_lm import HybridTextEncoder
    from hybrid_xmamba.training.lightning_module import load_image_tower_checkpoint

    clip_model, _ = open_clip.create_model_from_pretrained(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    visual = clip_model.visual
    missing, unexpected = load_image_tower_checkpoint(visual, ckpt_path)
    print(f"image tower: missing {len(missing)}, unexpected {len(unexpected)}")
    if len(missing) > 0:
        raise RuntimeError(f"image tower keys missing from {ckpt_path}: {missing[:5]}")

    cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "model" / f"{model_config}.yaml")
    enc = HybridTextEncoder(HybridConfig.from_hydra(cfg), embed_dim=512)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)
    txt = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
    missing, unexpected = enc.load_state_dict(txt, strict=False)
    print(f"text encoder: missing {len(missing)}, unexpected {len(unexpected)}")
    if missing:
        raise RuntimeError(f"text encoder keys missing from {ckpt_path}: {missing[:5]}")
    return visual.to(device).eval(), enc.to(device).eval()


@torch.no_grad()
def embed_images(visual, paths: List[str], device: str, batch_size: int, workers: int) -> torch.Tensor:
    from concurrent.futures import ThreadPoolExecutor
    from omegaconf import OmegaConf
    from PIL import Image
    from scripts.train_contrastive import build_image_transform

    tf = build_image_transform(OmegaConf.create({"dataset": {}}), is_train=False)
    load = lambda p: tf(Image.open(p).convert("RGB"))
    out = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i in range(0, len(paths), batch_size):
            x = torch.stack(list(pool.map(load, paths[i:i + batch_size]))).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                z = visual(x)
            out.append(torch.nn.functional.normalize(z.float(), dim=-1).cpu())
            if (i // batch_size) % 50 == 0:
                print(f"  images {min(i + batch_size, len(paths))}/{len(paths)}", flush=True)
    return torch.cat(out)


@torch.no_grad()
def embed_texts(enc, texts: List[str], device: str, batch_size: int, max_length: int = 256) -> torch.Tensor:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    out = []
    for i in range(0, len(texts), batch_size):
        b = tok(texts[i:i + batch_size], padding=True, truncation=True, max_length=max_length,
                return_tensors="pt").to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            z = enc.encode(b["input_ids"], attention_mask=b["attention_mask"])
        out.append(torch.nn.functional.normalize(z.float(), dim=-1).cpu())
        if (i // batch_size) % 100 == 0:
            print(f"  texts {min(i + batch_size, len(texts))}/{len(texts)}", flush=True)
    return torch.cat(out)


def write_floor(out_dir: Path, hyps: List[str], refs: List[str]) -> None:
    """Same writer as the published floors and decoders, so CheXbert scoring reads it alike."""
    from scripts.evaluate_report_generation import write_hyps_refs
    write_hyps_refs(str(out_dir), hyps, refs)


def main() -> int:
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage1-ckpt", required=True)
    ap.add_argument("--model-config", default="hybrid_150m_m3_textenc")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--candidates", type=int, default=50)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--image-batch", type=int, default=256)
    ap.add_argument("--text-batch", type=int, default=256)
    ap.add_argument("--load-workers", type=int, default=16)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    visual, enc = load_stage1(args.stage1_ckpt, args.model_config, device)

    frames = {s: pd.read_parquet(data / f"{s}.parquet") for s in set(args.splits) | {"train"}}
    cache = out / "embeddings.pt"
    emb = torch.load(cache) if cache.exists() else {}
    for s, df in frames.items():
        if f"img_{s}" not in emb:
            print(f"embedding {len(df)} {s} images")
            emb[f"img_{s}"] = embed_images(visual, df["image"].tolist(), device,
                                           args.image_batch, args.load_workers)
            torch.save(emb, cache)
    if "txt_train" not in emb:
        texts = [report_text(f, i) for f, i in zip(frames["train"]["findings"], frames["train"]["impression"])]
        print(f"embedding {len(texts)} train reports")
        emb["txt_train"] = embed_texts(enc, texts, device, args.text_batch)
        torch.save(emb, cache)

    gal_subj = subject_ids(frames["train"], data)
    summary = {"candidates": args.candidates, "k": args.k, "stage1_ckpt": args.stage1_ckpt}
    for s in args.splits:
        res = rerank_candidates(emb[f"img_{s}"], emb["img_train"], emb["txt_train"],
                                subject_ids(frames[s], data), gal_subj, args.candidates, args.k)
        torch.save({"query_rows": torch.arange(len(frames[s])), "cand_rows": res["cand"],
                    "ii": res["ii"], "it": res["it"], "topk_rows": res["topk"], "k": args.k,
                    "candidates": args.candidates, "split": s}, out / f"rag_{s}.pt")
        moved = (res["topk"][:, 0] != res["cand"][:, 0]).float().mean().item()
        summary[s] = {"n": len(frames[s]), "top1_changed_by_rerank": moved}
        print(f"{s}: n={len(frames[s])}, re-rank changed the top-1 for {moved:.1%}")

    if "test" in args.splits:
        tr, te = frames["train"], frames["test"]
        res = torch.load(out / "rag_test.pt")
        refs = [published_reference(r.get("findings", ""), r.get("impression", ""))
                for _, r in te.iterrows()]
        gal = [report_text(f, i) for f, i in zip(tr["findings"], tr["impression"])]
        write_floor(out / "floors" / "textenc_floor_ii", [gal[r] for r in res["cand_rows"][:, 0].tolist()], refs)
        write_floor(out / "floors" / "textenc_floor_rerank", [gal[r] for r in res["topk_rows"][:, 0].tolist()], refs)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
