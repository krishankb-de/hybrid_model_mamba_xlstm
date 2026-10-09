"""ISBI_BASELINES_PLAN.md B2-C: cost of one retrieval floor on one GPU.

A floor = an image encoder + cosine nearest neighbour over the 191,462-study
training index. Same protocol as scripts/performance_profile.py::measure_point
(rule R4): bf16, batch 4, 3 warmup + 10 timed iterations, peak memory reset
after warmup so it counts weights + index + activations.

Random pixels at the encoder's own input size and a random index of the real
shape: cost does not depend on values, so no MIMIC data is read.

    python scripts/isbi_floor_cost.py --encoder clip --output cost.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))  # run as `python scripts/isbi_floor_cost.py`

import torch

N_GALLERY = 191_462


def load_encoder(name: str, device: str):
    """Return (forward(pixel_values) -> features, input_size) in bf16."""
    from scripts.evaluate_report_generation import FLOOR_ENCODERS, hf_image_features

    spec = FLOOR_ENCODERS[name]
    if spec["loader"] == "open_clip":
        import open_clip
        model, _ = open_clip.create_model_from_pretrained("hf-hub:" + spec["id"])
        visual = model.visual.to(device, torch.bfloat16).eval()
        return (lambda pv: torch.nn.functional.normalize(visual(pv).float(), dim=-1)), 224

    from transformers import AutoModel
    model = AutoModel.from_pretrained(spec["id"]).to(device, torch.bfloat16).eval()
    size = model.config.vision_config.image_size
    return (lambda pv: hf_image_features(model, pv)), size


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="biomedclip")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    dev = "cuda"
    forward, size = load_encoder(args.encoder, dev)
    images = torch.randn(args.batch, 3, size, size, device=dev, dtype=torch.bfloat16)
    with torch.no_grad():
        dim = forward(images).shape[-1]
    gallery = torch.nn.functional.normalize(
        torch.randn(N_GALLERY, dim, device=dev), dim=-1).to(torch.bfloat16)

    def run(search: bool):
        with torch.no_grad():
            q = forward(images)
            if search:
                (q.to(torch.bfloat16) @ gallery.T).topk(args.k, dim=-1)

    out = {}
    for name, search in (("encode_only", False), ("floor_total", True)):
        for _ in range(args.warmup):
            run(search)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        ts = []
        for _ in range(args.iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            run(search)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        ts.sort()
        out[name] = {"latency_median_ms": 1e3 * ts[len(ts) // 2],
                     "latency_mean_ms": 1e3 * sum(ts) / len(ts),
                     "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9}
    out["meta"] = {"encoder": args.encoder, "input_size": size, "embed_dim": dim,
                   "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                   "batch": args.batch, "gallery": N_GALLERY, "k": args.k,
                   "dtype": "bf16", "warmup": args.warmup, "iters": args.iters}
    print(json.dumps(out, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
