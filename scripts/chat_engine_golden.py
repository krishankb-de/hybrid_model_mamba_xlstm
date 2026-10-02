"""CHAT_UI_PLAN.md P2-E: decode the first N test studies through app.engine.RealEngine.

    python scripts/chat_engine_golden.py --checkpoint … --model-config hybrid_150m_m3_rrg \
        --parquet …/test.parquet --n 20 --out results/chat_golden_<job>/engine_cached [--uncached]

Writes <out>/hyps.txt (one report per line, sanitised like write_hyps_refs) and <out>/timings.json.
The images go through Engine.preprocess from their file BYTES, the same path an upload takes.
Output is MIMIC-derived: never commit it. stdout carries only `[golden]` lines (numbers, no report text).
"""
import argparse
import json
import statistics
import sys
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-config", default="hybrid_150m_m3_rrg")
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--uncached", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args(argv)

    import pandas as pd
    import torch
    from app.engine import build_engine
    from app.schemas import Options

    device = "cuda" if torch.cuda.is_available() else "cpu"
    eng = build_engine("real", checkpoint=args.checkpoint, model_config=args.model_config,
                       device=device, threads=args.threads)
    card = eng.card()
    device_name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    print("[golden] device={} name={} torch={} threads={}".format(
        device, device_name, torch.__version__, torch.get_num_threads()), flush=True)
    print("[golden] card name={} prefix_k={} experiment={} cached_decode_available={} git={} dirty={} source={}".format(
        card["name"], card["prefix_k"], card["train_experiment"], card["cached_decode_available"],
        (card["git_sha"] or "None")[:7], card["git_dirty"], card["git_source"]), flush=True)

    df = pd.read_parquet(args.parquet).iloc[: args.n]
    opts = Options(cached_decode=not args.uncached)   # the rest of Options IS the published protocol (R2)
    cancel = threading.Event()                        # never set: the golden run is not interactive
    hyps, timings = [], []
    for i in range(len(df)):
        pre_res, prep = eng.preprocess(Path(df.iloc[i]["image"]).read_bytes())
        enc_res, enc = eng.encode(prep)
        gen_res, gen = eng.generate(enc, opts, lambda step, text: None, cancel)
        hyps.append(gen.report)
        row = {"row": i, "preprocess_ms": pre_res.ms, "encode_ms": enc_res.ms,
               "prefill_ms": gen_res.detail["prefill_ms"], "per_token_ms": gen_res.detail["per_token_ms"],
               "generate_ms": gen_res.ms, "tokens": gen_res.detail["tokens"]}
        timings.append(row)
        print("[golden] row {row} encode_ms={encode_ms} prefill_ms={prefill_ms} per_token_ms={per_token_ms} "
              "generate_ms={generate_ms}".format(**row), flush=True)
    if timings:
        median = lambda key: statistics.median(r[key] for r in timings)
        print("[golden] summary n={} cached={} preprocess_ms_median={:.1f} encode_ms_median={:.1f} "
              "prefill_ms_median={:.1f} per_token_ms_median={:.2f} generate_ms_median={:.1f}".format(
                  len(timings), not args.uncached, median("preprocess_ms"), median("encode_ms"),
                  median("prefill_ms"), median("per_token_ms"), median("generate_ms")), flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "hyps.txt").write_text("\n".join(" ".join(h.split()) for h in hyps) + "\n")
    (out / "timings.json").write_text(json.dumps(
        {"device": device, "device_name": device_name, "card": card, "rows": timings}, indent=2))


if __name__ == "__main__":
    main()
