"""STS Benchmark Evaluation for Biomedical Text Encoder.

Evaluates semantic textual similarity using:
- BIOSSES: Biomedical Semantic Similarity Estimation System
- STS-B: General domain STS benchmark (optional)

Reports Spearman correlation coefficient.
Target: Spearman > 0.75 to be competitive with BioViL-T, PubMedBERT, BioBERT.

Usage:
    # Evaluate on BIOSSES (biomedical)
    python scripts/evaluate_sts.py \
        --checkpoint outputs/stage1_pubmed_simcse/checkpoints/last.ckpt \
        --dataset biosses \
        --batch-size 32 \
        --output-dir outputs/eval_stage1/sts

    # Evaluate on STS-B (general domain)
    python scripts/evaluate_sts.py \
        --checkpoint outputs/stage1_pubmed_simcse/checkpoints/last.ckpt \
        --dataset stsb \
        --batch-size 32 \
        --output-dir outputs/eval_stage1/sts
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
from scipy.stats import spearmanr
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
    
    # Detect if checkpoint has lm. prefix
    has_lm_prefix_in_checkpoint = any(k.startswith("lm.") for k in state_dict.keys())
    
    print(f"  Checkpoint has 'lm.' prefix: {has_lm_prefix_in_checkpoint}")
    
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
    
    # If checkpoint doesn't have lm. prefix but model expects it, add the prefix
    if not has_lm_prefix_in_checkpoint:
        print(f"  Adding 'lm.' prefix to checkpoint keys to match model structure")
        state_dict_with_prefix = {}
        for k, v in state_dict.items():
            state_dict_with_prefix[f"lm.{k}"] = v
        state_dict = state_dict_with_prefix
    
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


def load_biosses_dataset():
    """Load BIOSSES dataset from HuggingFace.
    
    BIOSSES contains 100 sentence pairs from biomedical literature
    with similarity scores from 0 (no relation) to 4 (equivalent).
    """
    try:
        # Try new API first (default config)
        dataset = load_dataset("biosses", split="train")
        
        pairs = []
        scores = []
        for item in dataset:
            pairs.append((item["sentence1"], item["sentence2"]))
            scores.append(item["score"])
        
        print(f"  Loaded BIOSSES: {len(pairs)} sentence pairs")
        return pairs, scores
    
    except Exception as e:
        print(f"  Error loading BIOSSES with default config: {e}")
        print("  Trying alternative: using STS-B as fallback...")
        
        # Fallback to STS-B (general domain, but still useful)
        try:
            dataset = load_dataset("glue", "stsb", split="validation")
            
            pairs = []
            scores = []
            for item in dataset:
                if item["sentence1"] and item["sentence2"]:
                    pairs.append((item["sentence1"], item["sentence2"]))
                    scores.append(item["label"])
            
            print(f"  Loaded STS-B (fallback): {len(pairs)} sentence pairs")
            print(f"  Note: Using general domain STS-B instead of biomedical BIOSSES")
            return pairs, scores
        
        except Exception as e2:
            raise ValueError(
                f"Failed to load STS datasets. Errors:\n"
                f"  1. BIOSSES: {e}\n"
                f"  2. STS-B: {e2}\n"
                "Please check your datasets installation."
            )


def load_stsb_dataset(split="test"):
    """Load STS-B dataset from HuggingFace."""
    try:
        dataset = load_dataset("glue", "stsb", split=split)
        
        pairs = []
        scores = []
        for item in dataset:
            if item["sentence1"] and item["sentence2"]:
                pairs.append((item["sentence1"], item["sentence2"]))
                scores.append(item["label"])
        
        print(f"  Loaded STS-B ({split}): {len(pairs)} sentence pairs")
        return pairs, scores
    
    except Exception as e:
        raise ValueError(f"Error loading STS-B: {e}")


@torch.no_grad()
def encode_sentences(model, tokenizer, sentences, batch_size=32, max_length=256, device="cuda"):
    """Encode a list of sentences into embeddings.
    
    NOTE: Default max_length=256 matches typical Stage 1 training.
    Adjust based on your training config's max_seq_length.
    """
    model.eval()
    embeddings = []
    
    for i in tqdm(range(0, len(sentences), batch_size), desc="Encoding"):
        batch_sentences = sentences[i:i + batch_size]
        
        # Tokenize
        tokens = tokenizer(
            batch_sentences,
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


def compute_similarity_scores(embeddings1, embeddings2):
    """Compute cosine similarity between two sets of embeddings."""
    # Embeddings are already L2-normalized from model.encode()
    similarities = (embeddings1 * embeddings2).sum(dim=1)
    return similarities.numpy()


def evaluate_sts(model, tokenizer, pairs, gold_scores, batch_size=32, device="cuda"):
    """Evaluate STS performance using Spearman correlation."""
    # Extract sentences
    sentences1 = [pair[0] for pair in pairs]
    sentences2 = [pair[1] for pair in pairs]
    
    # Encode
    print("Encoding sentence pairs...")
    embeddings1 = encode_sentences(model, tokenizer, sentences1, batch_size, device=device)
    embeddings2 = encode_sentences(model, tokenizer, sentences2, batch_size, device=device)
    
    # Compute similarities
    pred_scores = compute_similarity_scores(embeddings1, embeddings2)
    
    # Compute Spearman correlation
    spearman_corr, p_value = spearmanr(gold_scores, pred_scores)
    
    return {
        "spearman_correlation": float(spearman_corr),
        "p_value": float(p_value),
        "num_pairs": len(pairs),
        "pred_scores": pred_scores.tolist(),
        "gold_scores": gold_scores,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate STS performance")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint (last.ckpt)")
    parser.add_argument("--dataset", type=str, default="biosses",
                        choices=["biosses", "stsb"],
                        help="STS dataset to use")
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
    
    # Load dataset
    print(f"\nLoading {args.dataset.upper()} dataset...")
    if args.dataset == "biosses":
        pairs, scores = load_biosses_dataset()
    else:
        pairs, scores = load_stsb_dataset(split="test")
    
    # Evaluate
    print("\n" + "=" * 60)
    print(f"  STS EVALUATION - {args.dataset.upper()}")
    print("=" * 60)
    
    results = evaluate_sts(
        model, tokenizer, pairs, scores,
        batch_size=args.batch_size,
        device=device
    )
    
    # Print results
    print("\n" + "=" * 60)
    print(f"  Spearman Correlation: {results['spearman_correlation']:.4f}")
    print(f"  P-value: {results['p_value']:.6f}")
    print(f"  Number of pairs: {results['num_pairs']}")
    print("=" * 60)
    
    # Benchmark comparison
    print("\nBenchmark Comparison:")
    print("  Target (competitive): > 0.75")
    print("  BioViL-T (reported):  ~ 0.78")
    print("  PubMedBERT:           ~ 0.82")
    print("  BioBERT:              ~ 0.80")
    
    if results['spearman_correlation'] >= 0.75:
        print(f"\n  ✓ PASS: Model achieves competitive performance!")
    else:
        print(f"\n  ✗ Below target: Consider more training or data augmentation")
    
    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = Path(args.output_dir) / f"sts_{args.dataset}_results.json"
        
        save_results = {
            "checkpoint": str(args.checkpoint),
            "dataset": args.dataset,
            "spearman_correlation": results["spearman_correlation"],
            "p_value": results["p_value"],
            "num_pairs": results["num_pairs"],
            "timestamp": datetime.now().isoformat(),
        }
        
        with open(output_path, "w") as f:
            json.dump(save_results, f, indent=2)
        
        print(f"\n  Results saved to {output_path}")
    
    return results


if __name__ == "__main__":
    main()
