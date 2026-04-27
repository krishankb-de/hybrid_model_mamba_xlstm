"""BEIR Retrieval Evaluation for Biomedical Text Encoder.

Evaluates nDCG@10 (and R@1/R@5/R@10) on standard BEIR biomedical benchmarks:
  - NFCorpus      : nutrition / biomedical IR, 3.6k docs, 336 queries
  - TREC-COVID    : COVID-19 biomedical IR, 171k docs, 50 queries
  - BioASQ        : biomedical QA retrieval, subset via BeIR

Falls back to PubMed article→abstract pairs if BEIR datasets unavailable.

For each benchmark, optionally runs PubMedBERT (teacher) as baseline.

Usage:
    # All BEIR benchmarks
    python scripts/evaluate_retrieval.py \\
        --checkpoint outputs/hybrid_70m_stage1_kd_pubmedbert/checkpoints/last.ckpt \\
        --benchmarks nfcorpus trec-covid \\
        --compare-pubmedbert \\
        --output-dir results/stage1_retrieval

Decision gate:
    PubMed R@10 >= 0.60  →  accept Stage 1 checkpoint.
    NFCorpus nDCG@10 >= 0.25  →  competitive with domain-specific baselines.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from tqdm import tqdm

from hybrid_xmamba.models.configuration_hybrid import HybridConfig
from hybrid_xmamba.models.hybrid_lm import HybridTextEncoder


# ---------------------------------------------------------------------------
# Checkpoint loading (identical to evaluate_sts.py)
# ---------------------------------------------------------------------------

def _strip_prefixes(state_dict: Dict) -> Dict:
    cleaned = {}
    for k, v in state_dict.items():
        k = k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k
        k = k.replace("model.", "", 1) if k.startswith("model.") else k
        k = k.replace("module.", "", 1) if k.startswith("module.") else k
        cleaned[k] = v
    return cleaned


def load_encoder(checkpoint_path: str, device: str = "cuda") -> HybridTextEncoder:
    import re
    print(f"Loading encoder from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw = ckpt.get("state_dict", ckpt)
    num_layers = max(
        (int(m.group(1)) + 1 for k in raw for m in [re.search(r"layers\.(\d+)\.", k)] if m),
        default=8,
    )
    state = _strip_prefixes(raw)
    if not any(k.startswith("projection_head.") for k in state):
        raise RuntimeError(
            f"Missing projection_head.* after prefix stripping. "
            f"First 10 keys: {list(state.keys())[:10]}"
        )
    dim = next(
        (int(v.shape[1]) for k, v in state.items() if "token_embedding.weight" in k), 512
    )
    base = ["mamba", "mamba", "mlstm"]
    cfg = HybridConfig(
        dim=dim, num_layers=num_layers,
        layer_pattern=[base[i % len(base)] for i in range(num_layers)],
        vocab_size=50257, max_position_embeddings=1024,
        state_size=16, conv_size=4, expand_factor=2, head_dim=64,
        use_tfla=True, proj_factor=2, slstm_hidden_dim=dim, slstm_num_heads=4,
        norm_type="rms", use_mlp=True, mlp_ratio=4.0, dropout=0.0,
    )
    model = HybridTextEncoder(cfg, embed_dim=512)
    missing, _ = model.load_state_dict(state, strict=False)
    critical = [k for k in missing if k.startswith(("lm.", "projection_head."))]
    if critical:
        raise RuntimeError(f"Critical keys missing: {critical[:5]}")
    model.to(device).eval()
    print(f"  Encoder loaded: {sum(p.numel() for p in model.parameters()):,} params")
    return model


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

@torch.no_grad()
def _encode_texts(
    model: HybridTextEncoder,
    tokenizer: AutoTokenizer,
    texts: List[str],
    batch_size: int = 32,
    max_length: int = 512,
    device: str = "cuda",
    desc: str = "Encoding",
) -> torch.Tensor:
    parts: List[torch.Tensor] = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"  {desc}", leave=False):
        enc = tokenizer(
            texts[i : i + batch_size],
            padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        parts.append(model.encode(ids, attention_mask=mask).cpu())
    return torch.cat(parts, dim=0)


@torch.no_grad()
def _encode_texts_bert(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    texts: List[str],
    batch_size: int = 32,
    max_length: int = 512,
    device: str = "cuda",
    desc: str = "Encoding",
) -> torch.Tensor:
    """Mean-pool PubMedBERT over non-padding tokens."""
    parts: List[torch.Tensor] = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"  {desc}", leave=False):
        enc = tokenizer(
            texts[i : i + batch_size],
            padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        out = model(input_ids=ids, attention_mask=mask)
        h = out.last_hidden_state
        m = mask.unsqueeze(-1).to(h.dtype)
        emb = (h * m).sum(1) / m.sum(1).clamp(min=1)
        parts.append(F.normalize(emb, dim=-1).float().cpu())
    return torch.cat(parts, dim=0)


# ---------------------------------------------------------------------------
# BEIR evaluation (brute-force cosine)
# ---------------------------------------------------------------------------

def _ndcg_at_k(ranked_doc_ids: List[str], relevant: Dict[str, int], k: int = 10) -> float:
    """Compute nDCG@k. relevant: {doc_id: relevance_score (>0)}."""
    dcg = 0.0
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        rel = relevant.get(doc_id, 0)
        if rel > 0:
            dcg += rel / np.log2(rank + 1)
    # Ideal DCG
    ideal_rels = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0


def _recall_at_k(ranked_doc_ids: List[str], relevant: Dict[str, int], k: int) -> float:
    top_k = set(ranked_doc_ids[:k])
    relevant_set = set(relevant.keys())
    return len(top_k & relevant_set) / max(len(relevant_set), 1)


def _beir_eval(
    query_embs: torch.Tensor,
    corpus_embs: torch.Tensor,
    query_ids: List[str],
    corpus_ids: List[str],
    qrels: Dict[str, Dict[str, int]],
    k_values: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Brute-force cosine similarity retrieval + nDCG@10, R@k."""
    if k_values is None:
        k_values = [1, 5, 10]

    print(f"  Computing similarity matrix ({len(query_ids)} × {len(corpus_ids)})...")
    # Process in batches to avoid OOM on large corpora
    batch = 256
    top_k_max = max(k_values)
    all_top_ids: List[List[str]] = []

    for qi in tqdm(range(0, len(query_ids), batch), desc="  Retrieval", leave=False):
        q_batch = query_embs[qi : qi + batch]  # (B, D)
        # Chunked dot product with corpus
        scores = q_batch @ corpus_embs.T  # (B, N_corpus)
        top_k_idx = scores.topk(min(top_k_max, scores.size(1)), dim=1).indices  # (B, k)
        for j in range(q_batch.size(0)):
            all_top_ids.append([corpus_ids[idx] for idx in top_k_idx[j].tolist()])

    # Compute metrics
    ndcg_list: List[float] = []
    recall: Dict[int, List[float]] = {k: [] for k in k_values}

    for qi, qid in enumerate(query_ids):
        relevant = qrels.get(qid, {})
        if not relevant:
            continue
        ranked = all_top_ids[qi]
        ndcg_list.append(_ndcg_at_k(ranked, relevant, k=10))
        for k in k_values:
            recall[k].append(_recall_at_k(ranked, relevant, k))

    metrics = {"nDCG@10": float(np.mean(ndcg_list)) if ndcg_list else 0.0}
    for k in k_values:
        metrics[f"R@{k}"] = float(np.mean(recall[k])) if recall[k] else 0.0
    return metrics


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def _load_beir_dataset(dataset_name: str) -> Optional[Tuple]:
    """Load corpus, queries, qrels from BeIR HF dataset.

    Returns (corpus_texts, corpus_ids, query_texts, query_ids, qrels) or None.
    qrels: {query_id: {doc_id: relevance_int}}
    """
    hf_name = f"BeIR/{dataset_name}"
    try:
        print(f"  Loading {hf_name}...")

        # Corpus
        corpus_ds = load_dataset(hf_name, "corpus", split="corpus")
        corpus_texts = [
            f"{row.get('title', '')} {row.get('text', '')}".strip()
            for row in corpus_ds
        ]
        corpus_ids = [str(row["_id"]) for row in corpus_ds]
        print(f"    Corpus: {len(corpus_ids):,} docs")

        # Queries
        query_ds = load_dataset(hf_name, "queries", split="queries")
        query_texts = [str(row.get("text") or "") for row in query_ds]
        query_ids = [str(row["_id"]) for row in query_ds]
        print(f"    Queries: {len(query_ids)}")

        # QRels — test split preferred, fallback to dev
        for qrel_split in ("test", "dev"):
            try:
                qrel_ds = load_dataset(hf_name, "default", split=qrel_split)
                qrels: Dict[str, Dict[str, int]] = {}
                for row in qrel_ds:
                    qid = str(row["query-id"])
                    did = str(row["corpus-id"])
                    rel = int(row.get("score", 1))
                    if rel > 0:
                        qrels.setdefault(qid, {})[did] = rel
                # Only keep queries that have qrels
                valid_qids = [qid for qid in query_ids if qid in qrels]
                valid_qtexts = [query_texts[i] for i, qid in enumerate(query_ids) if qid in qrels]
                print(f"    QRels: {len(qrels)} queries with relevance judgements (split={qrel_split})")
                return corpus_texts, corpus_ids, valid_qtexts, valid_qids, qrels
            except Exception:
                continue

        print(f"    WARNING: no qrels found for {dataset_name} — skipping")
        return None

    except Exception as exc:
        print(f"  {hf_name} load failed: {exc}")
        return None


def _load_pubmed_pairs(num_pairs: int = 1000) -> Optional[Tuple]:
    """PubMed article prefix → abstract pairs as fallback retrieval eval."""
    try:
        ds = load_dataset("ccdv/pubmed-summarization", split="validation")
        pairs = []
        for row in ds:
            art = str(row.get("article") or "")
            abs_ = str(row.get("abstract") or "")
            if art and abs_ and len(art) > 50 and len(abs_) > 50:
                pairs.append((art[:512], abs_))
            if len(pairs) >= num_pairs:
                break
        print(f"  PubMed fallback pairs: {len(pairs)}")
        return pairs
    except Exception as exc:
        print(f"  PubMed fallback failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# PubMed pair R@k evaluation (diagonal identity matching)
# ---------------------------------------------------------------------------

def _pubmed_recall_eval(
    model: HybridTextEncoder,
    tokenizer: AutoTokenizer,
    pairs: List[Tuple[str, str]],
    batch_size: int,
    max_length: int,
    device: str,
) -> Dict[str, float]:
    queries = [p[0] for p in pairs]
    corpus = [p[1] for p in pairs]
    q_emb = _encode_texts(model, tokenizer, queries, batch_size, max_length, device, "Queries")
    c_emb = _encode_texts(model, tokenizer, corpus, batch_size, max_length, device, "Corpus")
    sim = q_emb @ c_emb.T  # (N, N)
    N = len(pairs)
    gt = torch.arange(N)
    metrics: Dict[str, float] = {}
    for k in [1, 5, 10]:
        top_k = sim.topk(min(k, N), dim=1).indices
        hits = (top_k == gt.unsqueeze(1)).any(dim=1).float().mean().item()
        metrics[f"R@{k}"] = hits
    return metrics


# ---------------------------------------------------------------------------
# Results recording
# ---------------------------------------------------------------------------

def write_retrieval_md(
    results: Dict,
    output_dir: Path,
    checkpoint_path: str,
) -> None:
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    md_path = results_dir / "stage1_metrics.md"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        f"\n## Retrieval Results — {ts}\n",
        f"Checkpoint: `{checkpoint_path}`\n\n",
    ]
    for bench_name, bench in results.items():
        lines.append(f"### {bench_name}\n\n")
        lines.append("| Metric | Hybrid | PubMedBERT | Δ |\n")
        lines.append("|--------|--------|------------|---|\n")
        for metric, val in bench.get("hybrid", {}).items():
            base = bench.get("pubmedbert", {}).get(metric)
            if base is not None:
                lines.append(f"| {metric} | {val:.4f} | {base:.4f} | {val - base:+.4f} |\n")
            else:
                lines.append(f"| {metric} | {val:.4f} | — | — |\n")
        lines.append("\n")

    mode = "a" if md_path.exists() else "w"
    with open(md_path, mode) as f:
        if mode == "w":
            f.write("# Stage 1 Evaluation Metrics\n")
        f.writelines(lines)
    print(f"Results appended to {md_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 BEIR retrieval evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--benchmarks", nargs="+",
        default=["nfcorpus", "pubmed"],
        choices=["nfcorpus", "trec-covid", "bioasq", "pubmed", "all"],
    )
    parser.add_argument("--compare-pubmedbert", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--num-pubmed-pairs", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default="results/stage1_retrieval")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    print(f"Device: {device}\n")

    run_all = "all" in args.benchmarks
    do_nfcorpus = run_all or "nfcorpus" in args.benchmarks
    do_trec_covid = run_all or "trec-covid" in args.benchmarks
    do_bioasq = run_all or "bioasq" in args.benchmarks
    do_pubmed = run_all or "pubmed" in args.benchmarks

    # Hybrid model
    model = load_encoder(args.checkpoint, device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # PubMedBERT baseline
    teacher, teacher_tok = None, None
    if args.compare_pubmedbert:
        teacher_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        print(f"Loading PubMedBERT: {teacher_name}...")
        teacher = AutoModel.from_pretrained(teacher_name, torch_dtype=torch.bfloat16)
        teacher.to(device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher_tok = AutoTokenizer.from_pretrained(teacher_name)
        print("  Teacher loaded.")

    all_results: Dict = {}

    def _run_beir_bench(bench_name: str) -> None:
        print(f"\n=== {bench_name.upper()} ===")
        data = _load_beir_dataset(bench_name)
        if data is None:
            return
        corpus_texts, corpus_ids, query_texts, query_ids, qrels = data

        # Encode corpus + queries
        c_emb = _encode_texts(model, tokenizer, corpus_texts, args.batch_size, args.max_length, device, "Corpus")
        q_emb = _encode_texts(model, tokenizer, query_texts, args.batch_size, args.max_length, device, "Queries")

        metrics = _beir_eval(q_emb, c_emb, query_ids, corpus_ids, qrels)
        print(f"  Hybrid  nDCG@10={metrics['nDCG@10']:.4f}  R@10={metrics.get('R@10', 0):.4f}")
        row: Dict = {"hybrid": metrics}

        if teacher is not None:
            c_emb_b = _encode_texts_bert(teacher, teacher_tok, corpus_texts, args.batch_size, 512, device, "Corpus (BERT)")
            q_emb_b = _encode_texts_bert(teacher, teacher_tok, query_texts, args.batch_size, 512, device, "Queries (BERT)")
            metrics_b = _beir_eval(q_emb_b, c_emb_b, query_ids, corpus_ids, qrels)
            print(f"  PubMedBERT nDCG@10={metrics_b['nDCG@10']:.4f}  R@10={metrics_b.get('R@10', 0):.4f}")
            row["pubmedbert"] = metrics_b

        all_results[bench_name] = row

    if do_nfcorpus:
        _run_beir_bench("nfcorpus")
    if do_trec_covid:
        _run_beir_bench("trec-covid")
    if do_bioasq:
        _run_beir_bench("bioasq")

    # PubMed article→abstract pairs fallback
    if do_pubmed:
        print("\n=== PubMed article→abstract pairs ===")
        pairs = _load_pubmed_pairs(args.num_pubmed_pairs)
        if pairs:
            metrics = _pubmed_recall_eval(model, tokenizer, pairs, args.batch_size, args.max_length, device)
            print(f"  Hybrid  R@1={metrics['R@1']:.4f}  R@5={metrics['R@5']:.4f}  R@10={metrics['R@10']:.4f}")
            row = {"hybrid": metrics}
            if teacher is not None:
                t_queries = [p[0] for p in pairs]
                t_corpus = [p[1] for p in pairs]
                q_b = _encode_texts_bert(teacher, teacher_tok, t_queries, args.batch_size, 512, device, "Queries (BERT)")
                c_b = _encode_texts_bert(teacher, teacher_tok, t_corpus, args.batch_size, 512, device, "Corpus (BERT)")
                sim_b = q_b @ c_b.T
                N = len(pairs)
                gt = torch.arange(N)
                metrics_b = {}
                for k in [1, 5, 10]:
                    top_k = sim_b.topk(min(k, N), dim=1).indices
                    metrics_b[f"R@{k}"] = (top_k == gt.unsqueeze(1)).any(dim=1).float().mean().item()
                print(f"  PubMedBERT R@1={metrics_b['R@1']:.4f}  R@5={metrics_b['R@5']:.4f}  R@10={metrics_b['R@10']:.4f}")
                row["pubmedbert"] = metrics_b
            all_results["PubMed-pairs"] = row

    # Summary
    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 60)
    for bench_name, row in all_results.items():
        h = row["hybrid"]
        p = row.get("pubmedbert", {})
        ndcg = h.get("nDCG@10", h.get("R@10", 0.0))
        ndcg_b = p.get("nDCG@10", p.get("R@10")) if p else None
        suffix = f"  vs PubMedBERT={ndcg_b:.4f} (Δ={ndcg - ndcg_b:+.4f})" if ndcg_b is not None else ""
        print(f"  {bench_name:<18} primary={ndcg:.4f}{suffix}")

    # Decision gate
    pubmed_r10 = all_results.get("PubMed-pairs", {}).get("hybrid", {}).get("R@10", 0.0)
    nfcorpus_ndcg = all_results.get("nfcorpus", {}).get("hybrid", {}).get("nDCG@10", 0.0)
    gate = pubmed_r10 >= 0.60 or nfcorpus_ndcg >= 0.25
    print(f"\nDecision gate (PubMed R@10 ≥ 0.60 or NFCorpus nDCG@10 ≥ 0.25): {'PASS ✓' if gate else 'FAIL ✗'}")

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "retrieval_results.json"
    with open(json_path, "w") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "timestamp": datetime.now().isoformat(),
            "results": all_results,
            "gate_pass": gate,
        }, f, indent=2)
    print(f"JSON saved: {json_path}")

    write_retrieval_md(all_results, output_dir, args.checkpoint)


if __name__ == "__main__":
    main()
