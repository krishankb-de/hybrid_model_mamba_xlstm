"""ISBI_BASELINES_PLAN.md B7-C: cost of writing a report, token by token, after a context of L.

Both decoders use their cached decode: mLMamba carries a fixed-size recurrent state per layer, the
Transformer a KV cache that grows with L (attention_block.step, B7-B). The cache is filled to
length L directly (random contents; cost does not depend on values) instead of by a prefill, so
this isolates the per-token decode cost. Prompt reading is the forward pass, measured separately
by scripts/profile_e1_confirm_h100.sh.

Protocol: bf16 weights, random init, beam search bookkeeping as in HybridLanguageModel.
beam_search_cached (each step = step_logits + reorder_cache by beam index), `reports` reports
x `beam` beams in the batch axis, 5 warmup steps, then `steps` timed steps. Peak memory is reset
after the cache is built, so it counts weights + cache + step activations.

    python scripts/isbi_decode_bench.py --model hybrid_150m_m3 --context 4096 --reports 1
"""
import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch


def fill_cache_to(caches, context: int):
    """Advance every layer's cache as if `context` tokens had been read."""
    for cache in caches:
        if cache is None:
            continue
        if "k" in cache:                                   # attention: KV of length `context`
            b, h, _, d = cache["k"].shape
            cap = context + 256                            # room for the decoded tokens
            for key in ("k", "v"):
                buf = torch.zeros(b, h, cap, d, device=cache[key].device, dtype=cache[key].dtype)
                buf[:, :, :context].normal_()
                cache[key] = buf
        else:                                              # recurrent: values only, size fixed
            for key, value in cache.items():
                if torch.is_tensor(value) and value.is_floating_point():
                    value.normal_().mul_(0.1)
        cache["seen"] = context


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--context", type=int, required=True)
    ap.add_argument("--reports", type=int, default=1)
    ap.add_argument("--beam", type=int, default=3)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    from scripts.performance_profile import load_config
    from hybrid_xmamba.models.hybrid_lm import HybridLanguageModel

    dev = "cuda"
    cfg = load_config(args.model)
    model = HybridLanguageModel(cfg).to(dev, torch.bfloat16).eval()
    assert model.supports_cached_decode(), f"{args.model} has no cached decode"
    n = args.reports * args.beam

    with torch.no_grad():
        caches = model.allocate_inference_cache(n, device=dev, dtype=torch.bfloat16)
        fill_cache_to(caches, args.context)
        token = torch.randint(0, cfg.vocab_size, (n, 1), device=dev)
        index = torch.arange(n, device=dev)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        def one_step():
            nonlocal caches
            logits = model.step_logits(model.embeddings(token)[:, 0], caches)
            # beam bookkeeping: pick top-k per report, reorder the caches like beam search does
            perm = index.view(args.reports, args.beam).flip(-1).reshape(-1)
            caches = model.reorder_cache(caches, perm)
            return logits

        for _ in range(args.warmup):
            one_step()
        torch.cuda.synchronize()
        times = []
        for _ in range(args.steps):
            t0 = time.perf_counter()
            one_step()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    times.sort()
    cache_bytes = sum(v.numel() * v.element_size() for c in caches if c for v in c.values()
                      if torch.is_tensor(v))
    out = {
        "model": args.model, "context": args.context, "reports": args.reports, "beam": args.beam,
        "step_median_ms": 1e3 * times[len(times) // 2],
        "step_mean_ms": 1e3 * sum(times) / len(times),
        "report_100_tokens_ms": 1e3 * sum(times) * 100 / args.steps,
        "cache_gb": cache_bytes / 1e9,
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "dtype": "bf16",
        "note": "eager step(); cache filled to `context`, beam reorder included",
    }
    print(json.dumps(out, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
