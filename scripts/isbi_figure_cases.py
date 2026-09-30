"""ISBI Fig. 1 material: one test X-ray, its k nearest training X-rays, and the hybrid decoder's report.

Writes into --out-dir:
    query.jpg, ret1.jpg .. ret<k>.jpg   square crops, ready for `ISBI Paper /figs/`
    cases.json                          study ids, cosine similarities, retrieved reports, the
                                        reference report and the generated report

The generated report is NOT decoded here. It is read from the hyps.txt the published test-split
eval already wrote (one report per line, aligned with test.parquet row order), so the figure shows
exactly the text that was scored. Pass --query-index to choose which test row is shown.

Encoder: the pooled image vector comes from the image encoder stored inside the report-generation
checkpoint, i.e. the adapted tower the generation branch conditions on (--encoder adapted). Note
that the published retrieval floor (run_retrieval_baseline in evaluate_report_generation.py) used
STOCK BiomedCLIP; pass --encoder stock to reproduce that neighbour instead.

Output images are MIMIC-CXR (PhysioNet DUA). Never commit them.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def topk_neighbours(query: torch.Tensor, gallery: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """(D,) query and (G, D) gallery, both L2-normalised -> (similarities, indices) of the k most
    similar gallery rows, most similar first. Pure tensor math, the CPU-testable core."""
    if k > gallery.shape[0]:
        raise ValueError("k={} exceeds gallery size {}".format(k, gallery.shape[0]))
    return (gallery @ query).topk(k)


def square_jpeg(src: str, dst: Path, size: int) -> None:
    """Centre-crop to a square and resize, so every panel in the figure has the same shape."""
    from PIL import Image

    img = Image.open(src).convert("L")
    side = min(img.size)
    left, top = (img.width - side) // 2, (img.height - side) // 2
    img = img.crop((left, top, left + side, top + side)).resize((size, size), Image.BICUBIC)
    img.save(dst, quality=92)


def report_text(row) -> str:
    return "Findings: {} Impression: {}".format(row.get("findings", ""), row.get("impression", "")).strip()


def build_encoder(args, device: str):
    """-> (callable mapping (B,3,224,224) to (B,D) unit vectors, transform)."""
    import torchvision.transforms as T

    # Same normalisation as run_checkpoint_inspection, i.e. what the generation branch saw.
    transform = T.Compose([
        T.Resize((224, 224)),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    if args.encoder == "adapted":
        from scripts.evaluate_report_generation import load_report_generation_module

        module = load_report_generation_module(args.checkpoint, args.model_config, device=device)
        visual = module.image_encoder
    else:
        import open_clip

        clip_model, _ = open_clip.create_model_from_pretrained(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
        visual = clip_model.visual
    visual = visual.to(device).eval()

    @torch.no_grad()
    def encode(pixels: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(visual(pixels.to(device)).float(), dim=-1).cpu()

    return encode, transform


class _Images(torch.utils.data.Dataset):
    def __init__(self, paths: List[str], transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image

        return self.transform(Image.open(self.paths[i]).convert("RGB"))


def embed_gallery(paths: List[str], encode, transform, batch_size: int, workers: int) -> torch.Tensor:
    loader = torch.utils.data.DataLoader(_Images(paths, transform), batch_size=batch_size,
                                         num_workers=workers, shuffle=False)
    out = []
    for i, pixels in enumerate(loader):
        out.append(encode(pixels))
        if i % 50 == 0:
            print("  embedded {}/{}".format(min((i + 1) * batch_size, len(paths)), len(paths)), flush=True)
    return torch.cat(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="report-generation last.ckpt (hybrid decoder)")
    p.add_argument("--model-config", default="hybrid_150m_m3_rrg")
    p.add_argument("--train-parquet", required=True)
    p.add_argument("--test-parquet", required=True)
    p.add_argument("--hyps", required=True, help="hyps.txt of the published test-split eval")
    p.add_argument("--query-index", type=int, default=0, help="row of test.parquet to show")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--encoder", choices=["adapted", "stock"], default="adapted")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cache", default=None, help="reuse/save gallery embeddings (.pt)")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    import pandas as pd

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_df = pd.read_parquet(args.train_parquet)
    test_df = pd.read_parquet(args.test_parquet)
    hyps = Path(args.hyps).read_text().splitlines()
    if len(hyps) != len(test_df):
        sys.exit("hyps.txt has {} lines but test.parquet has {} rows: not the full test-split dump, "
                 "so line i is not row i".format(len(hyps), len(test_df)))
    q_row = test_df.iloc[args.query_index]

    encode, transform = build_encoder(args, device)
    cache = Path(args.cache) if args.cache else None
    if cache is not None and cache.exists():
        saved = torch.load(cache)
        if saved["encoder"] != args.encoder or saved["n"] != len(train_df):
            sys.exit("cache {} was built for encoder={} n={}".format(cache, saved["encoder"], saved["n"]))
        gallery = saved["emb"].float()
        print("Loaded gallery embeddings from {}".format(cache))
    else:
        print("Embedding {} training images ({} encoder)...".format(len(train_df), args.encoder))
        gallery = embed_gallery(train_df["image"].tolist(), encode, transform, args.batch_size, args.workers)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"encoder": args.encoder, "n": len(train_df), "emb": gallery.half()}, cache)

    from PIL import Image

    query = encode(transform(Image.open(q_row["image"]).convert("RGB")).unsqueeze(0))[0]
    sims, idx = topk_neighbours(query, gallery, args.k)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    square_jpeg(q_row["image"], out / "query.jpg", args.image_size)
    retrieved = []
    for rank, (s, i) in enumerate(zip(sims.tolist(), idx.tolist()), start=1):
        g_row = train_df.iloc[i]
        square_jpeg(g_row["image"], out / "ret{}.jpg".format(rank), args.image_size)
        retrieved.append({"rank": rank, "study_id": str(g_row.get("study_id", "?")),
                          "cosine": round(s, 4), "report": report_text(g_row)})

    cases = {
        "encoder": args.encoder,
        "checkpoint": args.checkpoint,
        "query": {"test_row": args.query_index, "study_id": str(q_row.get("study_id", "?")),
                  "reference_report": report_text(q_row)},
        "generated_report_beam3": hyps[args.query_index],
        "retrieved": retrieved,
    }
    (out / "cases.json").write_text(json.dumps(cases, indent=2))
    print(json.dumps(cases, indent=2))


if __name__ == "__main__":
    main()
