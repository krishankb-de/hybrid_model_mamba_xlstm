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
                  device="cuda", dtype=torch.float32, backward=False,
                  attn_backend="auto", compile_model=False):
    """Profile a single (batch_size, seq_length) point and print a report."""
    print("=" * 80)
    print("Model Profiling")
    print("=" * 80)

    model = build_model(config, device, dtype)
    num_params = model.get_num_params(non_embedding=True)
    model, compile_s = maybe_compile(model, compile_model)
    print("Model: {:.1f}M parameters (non-embedding)".format(num_params / 1e6))
    print("Attention backend: {}   torch.compile: {}".format(
        attn_backend, "{:.1f}s to wrap".format(compile_s) if compile_model else "off"))
    print("Batch size: {}".format(batch_size))
    print("Sequence length: {}".format(seq_length))
    print("Device: {}  dtype: {}".format(device, dtype))
    print("Pass: {}".format("forward+backward" if backward else "forward"))
    print()

    print("Warming up and profiling...")
    with attention_backend(attn_backend):
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
              dtype, backward, output_dir, attn_backend="auto",
              compile_model=False, chunk_size=None):
    """Sweep sequence length (x batch size) across models and fit exponents.

    `attn_backend` (E0-D) and `compile_model` (E1-B) are recorded on every row so
    two sweeps written to different directories can be compared arm by arm; the
    published 14A-7 numbers are the `auto` / uncompiled arm.
    """
    rows = []  # type: List[Dict[str, Any]]

    for name in model_names:
        config = load_config(name)
        if chunk_size is not None and hasattr(config, "mamba3_chunk_size"):
            config.mamba3_chunk_size = chunk_size
        model = build_model(config, device, dtype)
        num_params = model.get_num_params(non_embedding=True)
        model, compile_s = maybe_compile(model, compile_model)
        if compile_model:
            print("torch.compile: wrapped in {:.1f}s (graph build happens on the "
                  "first forward of each new shape)".format(compile_s))
        pattern = ",".join(config.layer_pattern)
        print("\n" + "=" * 80)
        print("{}  |  {:.1f}M non-emb params  |  dim={} layers={}  |  [{}]".format(
            name, num_params / 1e6, config.dim, config.num_layers, pattern))
        print("=" * 80)

        for batch_size in batch_sizes:
            for seq_length in seq_lengths:
                with attention_backend(attn_backend):
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
                    "attn_backend": attn_backend,
                    "compiled": bool(compile_model),
                    "chunk_size": getattr(config, "mamba3_chunk_size", None),
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
            "device", "dtype", "pass", "batch_size", "seq_length",
            "attn_backend", "compiled", "chunk_size", "oom",
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


def profile_decode(config, prompt_len=256, new_tokens=64, batch_size=1,
                   device="cpu", dtype=torch.float32, beam_size=1):
    """MAMBA3_PLAN_V2.md M6-E: prefill, TTFT and per-token decode, cached vs full recompute.

    This repo had no decode benchmark at all before M6 -- `evaluate_lm.py` and the sweep above
    both time full-sequence forwards, which is the one thing autoregressive generation never
    does. Without a per-token number, "generation is slow" was an impression rather than a
    measurement, and the O(L^2) cost of re-running the prefix for every token was invisible.

    Reports, for each path:
        prefill / TTFT  -- seconds to the first sampled token
        decode          -- seconds per token thereafter, and its growth from the first half of
                           the run to the second (an O(1) path is flat; a recomputing one is not)
    """
    import time

    model = build_model(config, device, dtype).eval()
    vocab = config.vocab_size
    ids = torch.randint(0, vocab, (batch_size, prompt_len), device=device)

    def _time(fn):
        _sync(device)
        t0 = time.perf_counter()
        out = fn()
        _sync(device)
        return time.perf_counter() - t0, out

    print("\n" + "=" * 70)
    print("DECODE PROFILE  prompt={}  new_tokens={}  batch={}  beam={}".format(
        prompt_len, new_tokens, batch_size, beam_size))
    print("=" * 70)

    rows = {}
    with torch.no_grad():
        # ---- full recompute: what generate() does today -------------------------------
        hidden = model.embeddings(ids)
        ttft, _ = _time(lambda: model(inputs_embeds=hidden).logits[:, -1])
        per_token, halves = [], []
        seq = hidden
        for i in range(new_tokens):
            dt, logits = _time(lambda: model(inputs_embeds=seq).logits[:, -1])
            per_token.append(dt)
            seq = torch.cat([seq, model.embeddings(logits.argmax(-1, keepdim=True))], dim=1)
        rows["full recompute"] = (ttft, per_token)

        # ---- cached ------------------------------------------------------------------
        if model.supports_cached_decode():
            caches = model.allocate_inference_cache(batch_size, device=device, dtype=dtype)
            ttft_c, logits = _time(lambda: model.prefill(model.embeddings(ids), caches))
            per_token_c = []
            for i in range(new_tokens):
                nxt = logits.argmax(-1, keepdim=True)
                dt, logits = _time(
                    lambda: model.step_logits(model.embeddings(nxt)[:, 0], caches)
                )
                per_token_c.append(dt)
            rows["cached (O(1))"] = (ttft_c, per_token_c)

    print("{:<18} {:>10} {:>12} {:>12} {:>10}".format(
        "path", "TTFT s", "s/token", "2nd/1st half", "tok/s"))
    for name, (ttft, per_token) in rows.items():
        half = len(per_token) // 2
        first = sum(per_token[:half]) / max(half, 1)
        second = sum(per_token[half:]) / max(len(per_token) - half, 1)
        mean = sum(per_token) / len(per_token)
        print("{:<18} {:>10.4f} {:>12.5f} {:>12.2f}x {:>10.1f}".format(
            name, ttft, mean, second / first if first else float("nan"), 1.0 / mean))

    if len(rows) == 2:
        (f_ttft, f_tok), (c_ttft, c_tok) = rows["full recompute"], rows["cached (O(1))"]
        mean = lambda xs: sum(xs) / len(xs)
        half = len(c_tok) // 2
        growth = mean(c_tok[half:]) / mean(c_tok[:half]) if half else float("nan")
        print("\n  per-token speedup      {:.2f}x".format(mean(f_tok) / mean(c_tok)))
        print("  TTFT ratio             {:.2f}x  (>1 means the cached prefill is SLOWER: it "
              "steps".format(c_ttft / f_ttft))
        print("                              token by token -- see prefill()'s docstring)")
        print("  cached growth 2nd/1st  {:.2f}x  (1.00 = O(1) in context, which is the "
              "claim)".format(growth))
    return rows


# ---------------------------------------------------------------------------
# EFFICIENCY_PLAN.md E0 — where does the time actually go?
#
# The 14A-7 sweep above measures the whole model. It cannot say which layers or
# which operator the 4.14x wall-clock gap to FlashAttention lives in, and every
# optimisation estimate is a hypothesis until it can. E0-A splits by mixer type
# (the Amdahl bound: 3 of the 12 layers are mLSTM and no Mamba kernel touches
# them), E0-B splits the Mamba-3 block into `ssd_chunked_scan` and everything
# else, E0-C sweeps `mamba3_chunk_size` (a pure performance knob -- the chunked
# decomposition is exact for any chunk size), and E0-D re-times attention with
# the fused SDPA backends disabled, which is what separates the algorithm from
# the kernel engineering.
# ---------------------------------------------------------------------------

SDPA_BACKENDS = ("auto", "math", "flash", "efficient")


@contextmanager
def attention_backend(name):
    """Restrict `F.scaled_dot_product_attention` to one backend (E0-D).

    `auto` is PyTorch's own dispatch, i.e. the fused FlashAttention kernel on an
    H100 -- that is the arm every published efficiency number was measured in.
    `math` forces the unfused reference path, which materialises the (L, L)
    attention matrix. Comparing the two isolates how much of the Transformer's
    speed is its algorithm and how much is a hand-written kernel.

    Expect `math` to OOM at the top of the sequence ladder. That is not a bug in
    the harness: b=4, 12 heads, L=16384 needs ~26 GB in bf16 for one attention
    matrix. `measure_point` records it as `oom=True` and the sweep continues,
    and the OOM is itself the FlashAttention memory story stated as data.
    """
    if name == "auto":
        yield
        return
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:      # torch < 2.3
        flags = {
            "math": dict(enable_math=True, enable_flash=False, enable_mem_efficient=False),
            "flash": dict(enable_math=False, enable_flash=True, enable_mem_efficient=False),
            "efficient": dict(enable_math=False, enable_flash=False, enable_mem_efficient=True),
        }[name]
        with torch.backends.cuda.sdp_kernel(**flags):
            yield
        return
    backend = {
        "math": SDPBackend.MATH,
        "flash": SDPBackend.FLASH_ATTENTION,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
    }[name]
    with sdpa_kernel([backend]):
        yield


def maybe_compile(model, enabled):
    """`torch.compile` the model for inference only (E1-B).

    `compile_model=false` is pinned everywhere in this project, and
    MAMBA3_PLAN_V2.md:301 gives the reason: the *Mamba-1* block loops in Python
    over (row, segment) for document boundaries, which is "dozens of tiny kernel
    launches per layer per step". SSD handles boundaries with one masked `exp`,
    so that reason does not transfer -- but it was never re-tested. This flag is
    how it gets tested. Compile time is reported because a ten-minute compile
    for a 1.1x steady-state gain is a null under rule R3.
    """
    if not enabled:
        return model, 0.0
    start = time.perf_counter()
    compiled = torch.compile(model)
    return compiled, time.perf_counter() - start


def _mark(cuda):
    """Start/stop marker: a CUDA event on GPU, a wall clock on CPU."""
    if cuda:
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        return ev
    return time.perf_counter()


def _delta_ms(cuda, a, b):
    return a.elapsed_time(b) if cuda else (b - a) * 1000.0


class LayerSplit:
    """Per-`HybridBlock` timing, aggregated by mixer type (E0-A).

    Markers are queued on the same stream as the work, so on CUDA this costs two
    event records per layer per iteration and does not serialise the forward.
    Read the totals only after a synchronise.
    """

    def __init__(self, model, device):
        self.cuda = device.startswith("cuda")
        self.layer_types = [layer.layer_type for layer in model.layers]
        self.enabled = False
        self._open = {}
        self._records = []
        self._handles = []
        for idx, layer in enumerate(model.layers):
            self._handles.append(layer.register_forward_pre_hook(self._pre(idx)))
            self._handles.append(layer.register_forward_hook(self._post(idx)))

    def _pre(self, idx):
        def hook(module, args):
            if self.enabled:
                self._open[idx] = _mark(self.cuda)
        return hook

    def _post(self, idx):
        def hook(module, args, output):
            if self.enabled and idx in self._open:
                self._records.append((idx, self._open.pop(idx), _mark(self.cuda)))
        return hook

    def reset(self):
        self._open.clear()
        self._records.clear()

    def totals_ms(self, iterations):
        """{mixer type: mean ms per forward}, plus a per-layer-index breakdown."""
        by_type, by_index = {}, {}
        for idx, start, end in self._records:
            ms = _delta_ms(self.cuda, start, end) / max(iterations, 1)
            lt = self.layer_types[idx]
            by_type[lt] = by_type.get(lt, 0.0) + ms
            by_index[idx] = by_index.get(idx, 0.0) + ms
        return by_type, by_index

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


class ScanSplit:
    """Time every `ssd_chunked_scan` call inside the Mamba-3 blocks (E0-B).

    `mamba3_block.py` imports the symbol directly, so the patch target is that
    module's attribute, not the kernel package's. Restores on exit even if the
    body raises -- a profiler that leaves a monkeypatch behind would silently
    corrupt every later measurement in the same process.
    """

    def __init__(self, device):
        self.cuda = device.startswith("cuda")
        self.enabled = False
        self._pairs = []
        self._module = None
        self._orig = None

    def __enter__(self):
        from hybrid_xmamba.layers import mamba3_block as _m3
        self._module = _m3
        self._orig = _m3.ssd_chunked_scan

        def timed(*args, **kwargs):
            if not self.enabled:
                return self._orig(*args, **kwargs)
            start = _mark(self.cuda)
            out = self._orig(*args, **kwargs)
            self._pairs.append((start, _mark(self.cuda)))
            return out

        _m3.ssd_chunked_scan = timed
        return self

    def __exit__(self, *exc):
        if self._module is not None and self._orig is not None:
            self._module.ssd_chunked_scan = self._orig
        return False

    def reset(self):
        self._pairs.clear()

    def total_ms(self, iterations):
        if not self._pairs:
            return None
        return sum(_delta_ms(self.cuda, a, b) for a, b in self._pairs) / max(iterations, 1)


def run_layer_split(model_names, seq_lengths, batch_size, num_iterations, device,
                    dtype, attn_backend="auto", chunk_size=None, output_dir=None,
                    warmup=3):
    """E0-A/E0-B: split a forward pass by mixer type and by the SSD scan.

    Prints, per model and sequence length, the milliseconds and share of the
    forward spent in each mixer type, the share inside `ssd_chunked_scan`, and
    the Amdahl bound (E0-F) that follows: the best speedup available from making
    the Mamba-3 path free is 1 / (1 - its share).
    """
    rows = []
    cuda = device.startswith("cuda")

    for name in model_names:
        config = load_config(name)
        if chunk_size is not None and hasattr(config, "mamba3_chunk_size"):
            config.mamba3_chunk_size = chunk_size
        model = build_model(config, device, dtype)
        pattern = ",".join(config.layer_pattern)
        counts = {}
        for lt in [layer.layer_type for layer in model.layers]:
            counts[lt] = counts.get(lt, 0) + 1

        print("\n" + "=" * 80)
        print("{}  |  dim={} layers={}  |  [{}]".format(
            name, config.dim, config.num_layers, pattern))
        print("  layer counts: {}".format(
            ", ".join("{}x{}".format(v, k) for k, v in sorted(counts.items()))))
        if chunk_size is not None:
            print("  mamba3_chunk_size = {}".format(getattr(config, "mamba3_chunk_size", "n/a")))
        print("  attention backend = {}".format(attn_backend))
        print("=" * 80)

        split = LayerSplit(model, device)
        try:
            for seq_length in seq_lengths:
                input_ids = torch.randint(0, config.vocab_size,
                                          (batch_size, seq_length), device=device)
                row = {
                    "model": name, "seq_length": seq_length, "batch_size": batch_size,
                    "dtype": str(dtype).replace("torch.", ""),
                    "attn_backend": attn_backend,
                    "chunk_size": getattr(config, "mamba3_chunk_size", None),
                    "layer_counts": counts, "oom": False,
                }
                try:
                    with attention_backend(attn_backend), ScanSplit(device) as scan:
                        with torch.no_grad():
                            for _ in range(warmup):
                                model(input_ids)
                        _sync(device)
                        split.reset(); scan.reset()
                        split.enabled = scan.enabled = True
                        wall_start = time.perf_counter()
                        with torch.no_grad():
                            for _ in range(num_iterations):
                                model(input_ids)
                        _sync(device)
                        wall_ms = (time.perf_counter() - wall_start) * 1000.0 / num_iterations
                        split.enabled = scan.enabled = False
                        by_type, by_index = split.totals_ms(num_iterations)
                        scan_ms = scan.total_ms(num_iterations)
                except torch.cuda.OutOfMemoryError:
                    row["oom"] = True
                    rows.append(row)
                    print("  L={:<6} OOM".format(seq_length))
                    _reset_peak_memory(device)
                    continue
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower():
                        raise
                    row["oom"] = True
                    rows.append(row)
                    print("  L={:<6} OOM".format(seq_length))
                    _reset_peak_memory(device)
                    continue

                row.update({
                    "forward_ms": round(wall_ms, 3),
                    "by_type_ms": {k: round(v, 3) for k, v in by_type.items()},
                    "by_layer_ms": {str(k): round(v, 3) for k, v in sorted(by_index.items())},
                    "ssd_scan_ms": None if scan_ms is None else round(scan_ms, 3),
                })
                mixer_total = sum(by_type.values())
                row["outside_mixers_ms"] = round(max(wall_ms - mixer_total, 0.0), 3)

                print("  L={:<6} forward {:8.2f} ms".format(seq_length, wall_ms))
                for lt, ms in sorted(by_type.items(), key=lambda kv: -kv[1]):
                    print("      {:<10} {:8.2f} ms  {:5.1f}%  ({} layers)".format(
                        lt, ms, 100.0 * ms / wall_ms, counts.get(lt, 0)))
                print("      {:<10} {:8.2f} ms  {:5.1f}%  (embed/head/norm)".format(
                    "other", row["outside_mixers_ms"], 100.0 * row["outside_mixers_ms"] / wall_ms))
                if scan_ms is not None:
                    print("      -> of which ssd_chunked_scan: {:.2f} ms  {:.1f}% of forward".format(
                        scan_ms, 100.0 * scan_ms / wall_ms))
                    row["ssd_scan_share"] = round(scan_ms / wall_ms, 4)

                # E0-F, stated as data rather than left to the reader.
                m3_ms = by_type.get("mamba3", 0.0) + by_type.get("mamba", 0.0)
                if m3_ms > 0:
                    share = m3_ms / wall_ms
                    bound = float("inf") if share >= 1.0 else 1.0 / (1.0 - share)
                    row["amdahl_bound_if_ssd_free"] = round(bound, 3)
                    print("      Amdahl: SSD path is {:.1f}% of the forward, so making it FREE "
                          "caps the speedup at {:.2f}x".format(100.0 * share, bound))
                rows.append(row)
        finally:
            split.remove()
            del model
            if cuda:
                torch.cuda.empty_cache()

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        json_path = out / "layer_split.json"
        with open(json_path, "w") as f:
            json.dump({"points": rows}, f, indent=2)
        csv_path = out / "layer_split.csv"
        types = sorted({t for r in rows for t in r.get("by_type_ms", {})})
        fieldnames = (["model", "seq_length", "batch_size", "dtype", "attn_backend",
                       "chunk_size", "oom", "forward_ms"]
                      + ["ms_" + t for t in types]
                      + ["outside_mixers_ms", "ssd_scan_ms", "ssd_scan_share",
                         "amdahl_bound_if_ssd_free"])
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                flat = {k: v for k, v in r.items() if k in fieldnames}
                for t in types:
                    flat["ms_" + t] = r.get("by_type_ms", {}).get(t, "")
                writer.writerow({k: flat.get(k, "") for k in fieldnames})
        print("\nWrote {}\n      {}".format(csv_path, json_path))

    return rows


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
    parser.add_argument("--decode", action="store_true",
                        help="Profile autoregressive decode: prefill/TTFT and per-token "
                             "latency, cached vs full recompute (MAMBA3_PLAN_V2.md M6-E)")
    parser.add_argument("--prompt-len", type=int, default=256,
                        help="Prompt length for --decode")
    parser.add_argument("--new-tokens", type=int, default=64,
                        help="Tokens to generate for --decode")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on")
    # --- EFFICIENCY_PLAN.md E0/E1 -----------------------------------------
    parser.add_argument("--per-layer", action="store_true",
                        help="E0-A/E0-B: split the forward by mixer type and by "
                             "ssd_chunked_scan, and print the Amdahl bound")
    parser.add_argument("--attn-backend", type=str, default="auto",
                        choices=list(SDPA_BACKENDS),
                        help="E0-D: restrict scaled_dot_product_attention to one "
                             "backend. `auto` is the fused kernel every published "
                             "number used; `math` is the unfused reference and is "
                             "what separates algorithm from kernel engineering")
    parser.add_argument("--chunk-size", type=int, default=None,
                        help="E0-C: override mamba3_chunk_size. The chunked "
                             "decomposition is exact for any chunk size, so this is "
                             "a pure performance knob -- but it changes float "
                             "association, so equivalence (rule R1) is not optional")
    parser.add_argument("--compile", dest="compile_model", action="store_true",
                        help="E1-B: torch.compile the model for inference. The "
                             "reason compile is pinned off (MAMBA3_PLAN_V2.md:301) "
                             "is a Mamba-1 artifact and does not apply to SSD")

    args = parser.parse_args()
    dtype = DTYPES[args.dtype]

    if args.per_layer:
        run_layer_split(
            model_names=args.models if args.models else [args.model],
            seq_lengths=sorted(args.seq_lengths),
            batch_size=args.batch_size,
            num_iterations=args.num_iterations,
            device=args.device,
            dtype=dtype,
            attn_backend=args.attn_backend,
            chunk_size=args.chunk_size,
            output_dir=args.output_dir,
        )
    elif args.decode:
        profile_decode(
            config=load_config(args.model),
            prompt_len=args.prompt_len,
            new_tokens=args.new_tokens,
            batch_size=args.batch_size,
            device=args.device,
            dtype=dtype,
        )
    elif args.sweep:
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
            attn_backend=args.attn_backend,
            compile_model=args.compile_model,
            chunk_size=args.chunk_size,
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
            attn_backend=args.attn_backend,
            compile_model=args.compile_model,
        )


if __name__ == "__main__":
    main()
