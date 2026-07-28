"""Profiling script for analyzing model performance.

Two modes:

  single (default) — one (batch_size, seq_length) point. Reports forward
      latency, throughput and peak memory. This is the original behaviour.

  sweep (--sweep)  — EFFICIENCY CURVES. Sweeps sequence length (and optionally
      batch size) across one or more model configs, then fits the log-log
      slope of latency and peak memory versus sequence length.

The scaling exponent is the point of the sweep. The hybrid stack is built from
Mamba (selective SSM) and mLSTM (TFLA) layers, both of which are linear in
sequence length, so the expected exponent is ~1.0 for both latency and memory.
Softmax attention would show ~2.0 for latency. `use_pos_embedding` is False in
HybridLanguageModel (hybrid_lm.py:43), so sequence lengths beyond
`max_position_embeddings` are architecturally valid and the curve can be swept
well past the training context.

Results are written as CSV + JSON so they can be plotted for the writeup.

Examples
--------
    # Single point (original behaviour)
    python scripts/performance_profile.py --model hybrid_150m_v2 --batch_size 4

    # Inference efficiency curve, hybrid vs both single-family baselines
    python scripts/performance_profile.py --sweep \
        --models hybrid_150m_v2 mamba_150m_baseline xlstm_150m_baseline \
        --seq-lengths 256 512 1024 2048 4096 8192 \
        --batch_size 4 --dtype bf16 --output-dir analysis/efficiency

    # Include the training step (forward + backward)
    python scripts/performance_profile.py --sweep --backward ...
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import csv
import dataclasses
import json
import math
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import torch
import yaml

from hybrid_xmamba.models.configuration_hybrid import HybridConfig
from hybrid_xmamba.models.hybrid_lm import HybridLanguageModel

CONFIG_DIR = project_root / "configs" / "model"

DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


@contextmanager
def timer(name):
    """Simple timing context manager."""
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print("{}: {:.2f}ms".format(name, (end - start) * 1000))


def available_configs():
    """Names of every model config that can be profiled.

    The Hydra yamls under configs/model/ are the source of truth — the runtime
    registry only ever registered 350m/1_3b/7b/mamba_baseline/xlstm_baseline,
    so looking models up there alone made every 70M and 150M config (i.e. every
    config this project actually trains) fail with a ValueError.
    """
    names = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))
    try:
        from hybrid_xmamba.utils.registry import ModelRegistry

        for name in ModelRegistry.list_configs():
            if name not in names:
                names.append(name)
    except Exception:  # registry is optional for profiling
        pass
    return names


def load_config(name):
    """Build a HybridConfig from configs/model/<name>.yaml, else the registry.

    The yamls carry training keys (learning_rate, warmup_steps, distill, ...)
    that are not HybridConfig fields, so filter to the dataclass fields rather
    than splatting the whole dict.
    """
    cfg_path = CONFIG_DIR / "{}.yaml".format(name)
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            raw = yaml.safe_load(f) or {}
        valid = {f.name for f in dataclasses.fields(HybridConfig)}
        # model_type exists on both sides but means different things (the yaml
        # says "hybrid_lm", the dataclass "hybrid_xmamba"); keep the dataclass
        # default rather than importing the Hydra target string.
        kwargs = {k: v for k, v in raw.items() if k in valid and k != "model_type"}
        return HybridConfig(**kwargs)

    from hybrid_xmamba.utils.registry import ModelRegistry

    return ModelRegistry.get_config(name)


def _sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _reset_peak_memory(device):
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def _peak_memory_gb(device):
    if device.startswith("cuda"):
        return torch.cuda.max_memory_allocated() / 1e9
    return float("nan")


def measure_point(model, batch_size, seq_length, num_iterations, device,
                  vocab_size, backward=False, warmup=3):
    """Time one (batch_size, seq_length) point.

    Returns a dict of timings in seconds and peak memory in GB, or a dict with
    `oom=True` if the point does not fit. Peak memory is reset per point so the
    number is attributable to this point and not to the largest earlier one.
    """
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)

    def _run():
        if backward:
            model.zero_grad(set_to_none=True)
            out = model(input_ids)
            # forward() returns a CausalLMOutput dataclass, not a tensor or dict.
            if hasattr(out, "logits"):
                logits = out.logits
            elif isinstance(out, dict):
                logits = out["logits"]
            else:
                logits = out
            loss = logits.float().mean()
            loss.backward()
        else:
            with torch.no_grad():
                model(input_ids)

    try:
        for _ in range(warmup):
            _run()
        _sync(device)
        _reset_peak_memory(device)

        times = []
        for _ in range(num_iterations):
            _sync(device)
            start = time.perf_counter()
            _run()
            _sync(device)
            times.append(time.perf_counter() - start)
    except torch.cuda.OutOfMemoryError:
        model.zero_grad(set_to_none=True)
        _reset_peak_memory(device)
        return {"oom": True}
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        model.zero_grad(set_to_none=True)
        _reset_peak_memory(device)
        return {"oom": True}

    times.sort()
    mean = sum(times) / len(times)
    median = times[len(times) // 2]
    var = sum((t - mean) ** 2 for t in times) / max(len(times) - 1, 1)
    tokens = batch_size * seq_length
    model.zero_grad(set_to_none=True)
    return {
        "oom": False,
        "latency_mean_s": mean,
        "latency_median_s": median,
        "latency_std_s": math.sqrt(var),
        "latency_min_s": times[0],
        "tokens_per_s": tokens / median,
        "peak_memory_gb": _peak_memory_gb(device),
    }


def fit_log_slope(xs, ys):
    """Least-squares slope of log(y) vs log(x) — the empirical scaling exponent.

    Returns None if fewer than two finite positive points are available.
    """
    pts = [
        (math.log(x), math.log(y))
        for x, y in zip(xs, ys)
        if x > 0 and y is not None and y > 0 and math.isfinite(y)
    ]
    if len(pts) < 2:
        return None
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    denom = sum((p[0] - mx) ** 2 for p in pts)
    if denom == 0:
        return None
    return sum((p[0] - mx) * (p[1] - my) for p in pts) / denom


def build_model(config, device, dtype):
    model = HybridLanguageModel(config)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model


def profile_model(config, batch_size=4, seq_length=2048, num_iterations=10,
                  device="cuda", dtype=torch.float32, backward=False):
    """Profile a single (batch_size, seq_length) point and print a report."""
    print("=" * 80)
    print("Model Profiling")
    print("=" * 80)

    model = build_model(config, device, dtype)
    num_params = model.get_num_params(non_embedding=True)
    print("Model: {:.1f}M parameters (non-embedding)".format(num_params / 1e6))
    print("Batch size: {}".format(batch_size))
    print("Sequence length: {}".format(seq_length))
    print("Device: {}  dtype: {}".format(device, dtype))
    print("Pass: {}".format("forward+backward" if backward else "forward"))
    print()

    print("Warming up and profiling...")
    res = measure_point(model, batch_size, seq_length, num_iterations, device,
                        config.vocab_size, backward=backward)
    if res["oom"]:
        print("OUT OF MEMORY at this point.")
        return res

    print("\n" + "=" * 80)
    print("Results:")
    print("Median forward time: {:.2f}ms  (mean {:.2f} +/- {:.2f}ms)".format(
        res["latency_median_s"] * 1000,
        res["latency_mean_s"] * 1000,
        res["latency_std_s"] * 1000,
    ))
    print("Throughput: {:.0f} tokens/second ({:.2f}k)".format(
        res["tokens_per_s"], res["tokens_per_s"] / 1000))
    if device.startswith("cuda"):
        print("Peak memory allocated: {:.2f} GB".format(res["peak_memory_gb"]))
    print("=" * 80)
    return res


def run_sweep(model_names, seq_lengths, batch_sizes, num_iterations, device,
              dtype, backward, output_dir):
    """Sweep sequence length (x batch size) across models and fit exponents."""
    rows = []  # type: List[Dict[str, Any]]

    for name in model_names:
        config = load_config(name)
        model = build_model(config, device, dtype)
        num_params = model.get_num_params(non_embedding=True)
        pattern = ",".join(config.layer_pattern)
        print("\n" + "=" * 80)
        print("{}  |  {:.1f}M non-emb params  |  dim={} layers={}  |  [{}]".format(
            name, num_params / 1e6, config.dim, config.num_layers, pattern))
        print("=" * 80)

        for batch_size in batch_sizes:
            for seq_length in seq_lengths:
                res = measure_point(model, batch_size, seq_length, num_iterations,
                                    device, config.vocab_size, backward=backward)
                row = {
                    "model": name,
                    "params_non_emb_m": round(num_params / 1e6, 2),
                    "dim": config.dim,
                    "num_layers": config.num_layers,
                    "layer_pattern": pattern,
                    "device": device,
                    "dtype": str(dtype).replace("torch.", ""),
                    "pass": "forward+backward" if backward else "forward",
                    "batch_size": batch_size,
                    "seq_length": seq_length,
                    "oom": res["oom"],
                }
                if res["oom"]:
                    print("  bs={:<4} L={:<6} OOM".format(batch_size, seq_length))
                else:
                    row.update({
                        "latency_median_ms": round(res["latency_median_s"] * 1000, 3),
                        "latency_mean_ms": round(res["latency_mean_s"] * 1000, 3),
                        "latency_std_ms": round(res["latency_std_s"] * 1000, 3),
                        "tokens_per_s": round(res["tokens_per_s"], 1),
                        "peak_memory_gb": round(res["peak_memory_gb"], 4),
                    })
                    print("  bs={:<4} L={:<6} {:9.2f}ms  {:10.0f} tok/s  {:7.3f} GB".format(
                        batch_size, seq_length,
                        res["latency_median_s"] * 1000,
                        res["tokens_per_s"],
                        res["peak_memory_gb"],
                    ))
                rows.append(row)

        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    # --- scaling exponents, per (model, batch_size) -------------------------
    exponents = []
    for name in model_names:
        for batch_size in batch_sizes:
            sel = [r for r in rows
                   if r["model"] == name and r["batch_size"] == batch_size
                   and not r["oom"]]
            if len(sel) < 2:
                continue
            xs = [r["seq_length"] for r in sel]
            lat = fit_log_slope(xs, [r["latency_median_ms"] for r in sel])
            mem = fit_log_slope(xs, [r.get("peak_memory_gb") for r in sel])
            exponents.append({
                "model": name,
                "batch_size": batch_size,
                "seq_lengths": xs,
                "latency_exponent": None if lat is None else round(lat, 3),
                "memory_exponent": None if mem is None else round(mem, 3),
            })

    print("\n" + "=" * 80)
    print("SCALING EXPONENTS  (slope of log(y) vs log(seq_length))")
    print("  ~1.0 = linear in sequence length; ~2.0 = quadratic (softmax attention)")
    print("=" * 80)
    print("{:<26} {:>4}  {:>10}  {:>10}".format("model", "bs", "latency", "memory"))
    for e in exponents:
        print("{:<26} {:>4}  {:>10}  {:>10}".format(
            e["model"], e["batch_size"],
            "n/a" if e["latency_exponent"] is None else "{:.3f}".format(e["latency_exponent"]),
            "n/a" if e["memory_exponent"] is None else "{:.3f}".format(e["memory_exponent"]),
        ))
    print("=" * 80)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "model", "params_non_emb_m", "dim", "num_layers", "layer_pattern",
            "device", "dtype", "pass", "batch_size", "seq_length", "oom",
            "latency_median_ms", "latency_mean_ms", "latency_std_ms",
            "tokens_per_s", "peak_memory_gb",
        ]
        csv_path = out / "efficiency_curves.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
        json_path = out / "efficiency_curves.json"
        with open(json_path, "w") as f:
            json.dump({"points": rows, "scaling_exponents": exponents}, f, indent=2)
        print("\nWrote {}\n      {}".format(csv_path, json_path))

    return rows, exponents


def main():
    parser = argparse.ArgumentParser(description="Profile hybrid model")
    choices = available_configs()
    parser.add_argument("--model", type=str, default="hybrid_150m_v2",
                        choices=choices,
                        help="Model config to profile (single-point mode)")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep sequence lengths and fit scaling exponents")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        choices=choices,
                        help="Model configs to compare in --sweep mode "
                             "(default: the --model value)")
    parser.add_argument("--seq-lengths", type=int, nargs="+",
                        default=[256, 512, 1024, 2048, 4096],
                        help="Sequence lengths to sweep")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=None,
                        help="Batch sizes to sweep (default: the --batch_size value)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--seq_length", type=int, default=2048,
                        help="Sequence length (single-point mode)")
    parser.add_argument("--num_iterations", type=int, default=10,
                        help="Timed iterations per point")
    parser.add_argument("--dtype", type=str, default="fp32", choices=sorted(DTYPES),
                        help="Compute dtype (use bf16 on H100/A100)")
    parser.add_argument("--backward", action="store_true",
                        help="Time forward+backward (training step) instead of "
                             "forward-only inference")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Write efficiency_curves.csv/.json here (--sweep)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on")

    args = parser.parse_args()
    dtype = DTYPES[args.dtype]

    if args.sweep:
        model_names = args.models if args.models else [args.model]
        batch_sizes = args.batch_sizes if args.batch_sizes else [args.batch_size]
        run_sweep(
            model_names=model_names,
            seq_lengths=sorted(args.seq_lengths),
            batch_sizes=batch_sizes,
            num_iterations=args.num_iterations,
            device=args.device,
            dtype=dtype,
            backward=args.backward,
            output_dir=args.output_dir,
        )
    else:
        profile_model(
            config=load_config(args.model),
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            num_iterations=args.num_iterations,
            device=args.device,
            dtype=dtype,
            backward=args.backward,
        )


if __name__ == "__main__":
    main()
