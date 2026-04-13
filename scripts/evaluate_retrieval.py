"""Retrieval Evaluation on PubMed Abstracts.

Evaluates text-to-text retrieval performance using PubMed abstract pairs.
Reports Recall@1, Recall@5, Recall@10.

Target: R@1 > 0.60 to be comparable with BioViL-T.

Usage:
    python scripts/evaluate_retrieval.py \
        --checkpoint outputs/stage1_pubmed_simcse/checkpoints/last.ckpt \
        --num-pairs 1000 \
        --batch-size 32 \
        --output-dir outputs/eval_stage1/retrieval
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from hybrid_xmamba.models.configuration_hybrid import HybridConfig
from hybrid_xmamba.models.hybrid_lm import HybridTextEncoder


def strip_state_dict_prefixes(state_dict):
    """Strip torch.compile (_orig_mod.) and Lightning (model.) key prefixes.
    
    Handles all combinations:
    - model._orig_mod.<key>
    - _orig_mod.model.<key>
    - _orig_mod.<key>
    - model.<key>
    """
    cleaned = {}
    for k, v in state_dict.items():
        # Remove all known prefixes in order of specificity
        if k.startswith("model._orig_mod."):
            new_k = k[len("model._orig_mod."):]
        elif k.startswith("_orig_mod.model."):
            new_k = k[len("_orig_mod.model."):]
        elif k.startswith("_orig_mod."):
            new_k = k[len("_orig_mod."):]
        elif k.startswith("model."):
            new_k = k[len("model."):]
        else:
            new_k = k
        
        # BUG FIX 1: Only skip logit_scale — projection_head IS needed by encode()
        if new_k == "logit_scale":
            continue
            
        cleaned[new_k] = v
    return cleaned


def load_encoder_from_checkpoint(checkpoint_path, device="cuda"):
    """Load HybridTextEncoder from a contrastive training checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    
    # Extract state dict
    raw_state_dict = ckpt.get("state_dict", ckpt)
    
    # Count layers BEFORE stripping prefixes
    import re
    num_layers = 0
    for k in raw_state_dict.keys():
        m = re.search(r"layers\.(\d+)\.", k)
        if m:
            idx = int(m.group(1))
            if idx + 1 > num_layers:
                num_layers = idx + 1
    
    if num_layers == 0:
        num_layers = 8  # 70M model has 8 layers, not 12
    
    # BUG FIX 3: Use comprehensive prefix stripping (handles torch.compile)
    state_dict = strip_state_dict_prefixes(raw_state_dict)
    
    # After stripping 'model.' prefix, the state_dict has the correct structure:
    # - lm.* keys for the language model
    # - projection_head.* keys for the projection head
    # This matches HybridTextEncoder's structure exactly, so no further modification needed!
    
    # Infer config from checkpoint
    dim = 512  # 70M model uses 512, not 768
    for k, v in state_dict.items():
        if "token_embedding.weight" in k:
            dim = int(v.shape[1])
            break
    
    # Build config
    base = ["mamba", "mamba", "mlstm"]
    layer_pattern = [base[i % len(base)] for i in range(num_layers)]
    
    # BUG FIX 6: Match training config (max_position_embeddings=1024)
    config = HybridConfig(
        dim=dim,
        num_layers=num_layers,
        layer_pattern=layer_pattern,
        vocab_size=50257,
        max_position_embeddings=1024,  # Match training, not 512
        state_size=16,
        conv_size=4,
        expand_factor=2,
        head_dim=64,
        use_tfla=True,
        proj_factor=2,
        slstm_hidden_dim=dim,
        slstm_num_heads=4,
        norm_type="rms",
        use_mlp=True,
        mlp_ratio=4.0,
        dropout=0.0,
    )
    
    # Create model - HybridTextEncoder wraps the LM with self.lm
    model = HybridTextEncoder(config, embed_dim=512)
    
    # Load weights
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    # BUG FIX 9: Show all missing keys and add validation
    if missing:
        print(f"  Missing keys ({len(missing)}): {list(missing)[:10]}")
        # Check for critical missing keys
        critical_missing = [k for k in missing if k.startswith("lm.") or k.startswith("projection_head.")]
        if critical_missing:
            print(f"  ⚠️  WARNING: CRITICAL KEYS MISSING - METRICS WILL BE INVALID!")
            print(f"  Critical missing: {critical_missing[:5]}")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {list(unexpected)[:10]}")
    
    if not missing and not unexpected:
        print(f"  ✅ All weights loaded successfully (exact match)")
    
    model = model.to(device)
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {num_params:,} parameters ({num_params/1e6:.1f}M)")
    
    return model


def load_pubmed_pairs(num_pairs=1000, split="validation"):
    """Load PubMed abstract pairs for retrieval evaluation.
    
    BUG FIX 8: Use longer article prefix (512 chars instead of 200) for better retrieval.
    Creates pairs by matching article beginning with its abstract.
    """
    print(f"Loading PubMed dataset ({split} split)...")
    
    try:
        # Use ccdv/pubmed-summarization which is the maintained version
        dataset = load_dataset("ccdv/pubmed-summarization", split=split)
        
        pairs = []
        for i, item in enumerate(dataset):
            if len(pairs) >= num_pairs:
                break
            
            # This dataset has 'article' and 'abstract' fields
            article = item.get("article", "")
            abstract = item.get("abstract", "")
            
            if article and abstract and len(article) > 50 and len(abstract) > 50:
                # BUG FIX 8: Use 512 chars instead of 200 for better semantic overlap
                pairs.append((article[:512], abstract))
        
        print(f"  Created {len(pairs)} article→abstract pairs")
        return pairs
    
    except Exception as e:
        print(f"  Error loading PubMed from ccdv/pubmed-summarization: {e}")
        print("  Trying train split as fallback...")
        
        # Fallback to train split
        try:
            dataset = load_dataset("ccdv/pubmed-summarization", split="train")
            
            pairs = []
            for i, item in enumerate(dataset):
                if len(pairs) >= num_pairs:
                    break
                
                article = item.get("article", "")
                abstract = item.get("abstract", "")
                
                if article and abstract and len(article) > 50 and len(abstract) > 50:
                    # BUG FIX 8: Use 512 chars instead of 200
                    pairs.append((article[:512], abstract))
            
            print(f"  Created {len(pairs)} article→abstract pairs from train split")
            return pairs
        
        except Exception as e2:
            raise ValueError(
                f"Failed to load PubMed dataset. Errors:\n"
                f"  1. {e}\n"
                f"  2. {e2}\n"
                "Please ensure datasets library is installed and configured correctly."
            )


@torch.no_grad()
def encode_texts(model, tokenizer, texts, batch_size=32, max_length=256, device="cuda"):
    """Encode a list of texts into embeddings.
    
    NOTE: Default max_length=256 matches typical Stage 1 training.
    Adjust based on your training config's max_seq_length.
    """
    model.eval()
    embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize
        tokens = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        input_ids = tokens["input_ids"].to(device)
        
        # Encode
        batch_embeddings = model.encode(input_ids)
        embeddings.append(batch_embeddings.cpu())
    
    return torch.cat(embeddings, dim=0)


def compute_retrieval_metrics(query_embeddings, corpus_embeddings, k_values=[1, 5, 10]):
    """Compute Recall@K for retrieval.
    
    Args:
        query_embeddings: (N, D) tensor of query embeddings
        corpus_embeddings: (N, D) tensor of corpus embeddings
        k_values: List of K values for Recall@K
    
    Returns:
        Dictionary with Recall@K scores
    """
    # Compute similarity matrix (queries x corpus)
    # Embeddings are already L2-normalized
    similarities = query_embeddings @ corpus_embeddings.T  # (N, N)
    
    # For each query, the correct match is at the same index
    correct_indices = torch.arange(len(query_embeddings))
    
    # Get top-K predictions for each query
    results = {}
    for k in k_values:
        # Get top-K indices for each query
        top_k_indices = similarities.topk(k, dim=1).indices  # (N, K)
        
        # Check if correct index is in top-K
        correct_in_top_k = (top_k_indices == correct_indices.unsqueeze(1)).any(dim=1)
        
        # Compute recall
        recall = correct_in_top_k.float().mean().item()
        results[f"R@{k}"] = recall
    
    return results


def evaluate_retrieval(model, tokenizer, pairs, batch_size=32, device="cuda"):
    """Evaluate retrieval performance on abstract pairs."""
    # Extract queries and corpus
    queries = [pair[0] for pair in pairs]
    corpus = [pair[1] for pair in pairs]
    
    print(f"\nEvaluating retrieval on {len(pairs)} pairs...")
    
    # Encode queries
    print("Encoding queries...")
    query_embeddings = encode_texts(model, tokenizer, queries, batch_size, device=device)
    
    # Encode corpus
    print("Encoding corpus...")
    corpus_embeddings = encode_texts(model, tokenizer, corpus, batch_size, device=device)
    
    # Compute metrics
    print("Computing retrieval metrics...")
    metrics = compute_retrieval_metrics(query_embeddings, corpus_embeddings)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval performance on PubMed")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint (last.ckpt)")
    parser.add_argument("--num-pairs", type=int, default=1000,
                        help="Number of abstract pairs to evaluate")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)  # Match training max_seq_length
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model
    model = load_encoder_from_checkpoint(args.checkpoint, device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load data
    pairs = load_pubmed_pairs(num_pairs=args.num_pairs)
    
    # Evaluate
    print("\n" + "=" * 60)
    print("  RETRIEVAL EVALUATION - PubMed Abstracts")
    print("=" * 60)
    
    metrics = evaluate_retrieval(
        model, tokenizer, pairs,
        batch_size=args.batch_size,
        device=device
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("  Retrieval Metrics:")
    print(f"    Recall@1:  {metrics['R@1']:.4f}")
    print(f"    Recall@5:  {metrics['R@5']:.4f}")
    print(f"    Recall@10: {metrics['R@10']:.4f}")
    print(f"  Number of pairs: {len(pairs)}")
    print("=" * 60)
    
    # Benchmark comparison
    print("\nBenchmark Comparison:")
    print("  Target (competitive): R@1 > 0.60")
    print("  BioViL-T (reported):  R@1 ~ 0.65")
    
    if metrics['R@1'] >= 0.60:
        print(f"\n  ✓ PASS: Model achieves competitive retrieval performance!")
    else:
        print(f"\n  ✗ Below target: Consider more training or larger batch size")
    
    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = Path(args.output_dir) / "retrieval_results.json"
        
        save_results = {
            "checkpoint": str(args.checkpoint),
            "num_pairs": len(pairs),
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }
        
        with open(output_path, "w") as f:
            json.dump(save_results, f, indent=2)
        
        print(f"\n  Results saved to {output_path}")
    
    return metrics


if __name__ == "__main__":
    main()
