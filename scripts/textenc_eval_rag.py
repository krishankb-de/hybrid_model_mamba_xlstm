"""TEXTENC_PLAN.md T4-B / T5-C: beam-3 test decoding with retrieved reports in the context.

Same decoding as the published evaluation (beam 3, max 100 new tokens, no BOS, length penalty
1.0); the prompt is the context [retrieved report 1]...[retrieved report k] after the image
prefix, exactly as in training (rag_report_generation.context_token_ids). Generation uses the
O(1)-step cache (beam_search_cached, token-identical to the uncached beam search by test); with
--check-uncached N the first N studies are also decoded uncached and the run fails on any
difference.

Sharded: --shard i --num-shards n decodes rows i, i+n, ...; --merge writes the ordered
hyps.txt/refs.txt for CheXbert scoring.

    python scripts/textenc_eval_rag.py --checkpoint outputs/<run>/checkpoints/last.ckpt \
        --model-config hybrid_150m_m3_rrg --rag-dir results/textenc_rag \
        --data /sc/home/$USER/dataset/mimic_full --dump-dir results/<name> --shard 0 --num-shards 4
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch


def shard_rows(n: int, shard: int, num_shards: int):
    return list(range(shard, n, num_shards))


def merge_shards(dump_dir: Path, num_shards: int, n: int):
    from scripts.evaluate_report_generation import write_hyps_refs

    hyps, refs = [None] * n, [None] * n
    for s in range(num_shards):
        part = json.loads((dump_dir / f"shard_{s}.json").read_text())
        for row, h, r in zip(part["rows"], part["hyps"], part["refs"]):
            hyps[row], refs[row] = h, r
    missing = [i for i, h in enumerate(hyps) if h is None]
    if missing:
        raise RuntimeError(f"{len(missing)} rows missing after merge (first {missing[:5]})")
    write_hyps_refs(str(dump_dir), hyps, refs)


def main() -> int:
    import pandas as pd
    from PIL import Image
    from transformers import AutoTokenizer
    from scripts.evaluate_report_generation import (beam_search_decode,
                                                     load_report_generation_module,
                                                     report_image_transform)
    from scripts.textenc_rag_index import published_reference, report_text
    from hybrid_xmamba.training.rag_report_generation import context_token_ids

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--model-config", default="hybrid_150m_m3_rrg")
    ap.add_argument("--rag-dir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--report-tokens", type=int, default=160)
    ap.add_argument("--beam-size", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=100)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--check-uncached", type=int, default=0)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    data, dump = Path(args.data), Path(args.dump_dir)
    test = pd.read_parquet(data / "test.parquet")
    if args.merge:
        merge_shards(dump, args.num_shards, len(test))
        print(f"merged {args.num_shards} shards -> {dump}/hyps.txt, refs.txt ({len(test)} rows)")
        return 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    module = load_report_generation_module(args.checkpoint, args.model_config, device=device)
    if not module.decoder.supports_cached_decode():
        raise RuntimeError("this decoder has no cached decode path")
    tok = AutoTokenizer.from_pretrained("gpt2")
    transform = report_image_transform(getattr(module, "image_encoder_name", "biomedclip"))
    res = torch.load(Path(args.rag_dir) / "rag_test.pt")
    topk = res["topk_rows"].tolist()
    if len(topk) != len(test):
        raise ValueError("rag_test.pt does not match test.parquet")
    tr = pd.read_parquet(data / "train.parquet", columns=["findings", "impression"])
    gallery = [report_text(f, i) for f, i in zip(tr["findings"], tr["impression"])]

    rows = shard_rows(len(test), args.shard, args.num_shards)
    out = {"rows": [], "hyps": [], "refs": []}
    checked = 0
    for n_done, i in enumerate(rows):
        row = test.iloc[i]
        ctx = context_token_ids(tok, [gallery[r] for r in topk[i][: args.k]], args.report_tokens)
        input_ids = torch.tensor([ctx], dtype=torch.long, device=device)
        with torch.no_grad():
            pixel_values = transform(Image.open(row["image"]).convert("RGB")).unsqueeze(0).to(device)
            prefix = module.prefix_mapper(module._patch_grid(pixel_values))
            ids = module.decoder.beam_search_cached(input_ids, prefix_embeds=prefix,
                                                    beam_size=args.beam_size,
                                                    max_new_tokens=args.max_new_tokens)
            if checked < args.check_uncached:
                ref_ids = beam_search_decode(module.decoder, input_ids, prefix_embeds=prefix,
                                             beam_size=args.beam_size,
                                             max_new_tokens=args.max_new_tokens)
                if not torch.equal(ids, ref_ids):
                    raise RuntimeError(f"cached and uncached decoding differ at test row {i}")
                checked += 1
        generated = tok.decode(ids[0, len(ctx):].tolist(), skip_special_tokens=True)
        out["rows"].append(i)
        out["hyps"].append(generated)
        out["refs"].append(published_reference(row.get("findings", ""), row.get("impression", "")))
        if n_done % 100 == 0:
            print(f"  shard {args.shard}: {n_done + 1}/{len(rows)}", flush=True)
    dump.mkdir(parents=True, exist_ok=True)
    (dump / f"shard_{args.shard}.json").write_text(json.dumps(out))
    print(f"shard {args.shard}: {len(rows)} studies, cached==uncached checked on {checked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
