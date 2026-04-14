"""Comprehensive language model evaluation script.

Computes:
  1. Test perplexity on WikiText-103 (standard LM benchmark)
  2. Inference throughput (tokens/second)
  3. Peak GPU memory usage
  4. Optional: text generation samples

Usage:
    # Hybrid 70M
    python scripts/evaluate_lm.py \
        --checkpoint outputs/hybrid_70m_wikitext_a100_mig20g/checkpoints/last.ckpt \
        --model-config hybrid_70m \
        --dataset wikitext --split test \
        --batch-size 4 --max-length 1024 --num-workers 2 \
        --throughput --generate \
        --output-dir outputs/hybrid_70m_wikitext_a100_mig20g/eval_results

    # Mamba-only baseline
    python scripts/evaluate_lm.py \
        --checkpoint outputs/mamba_70m_wikitext_a100_mig20g/checkpoints/last.ckpt \
        --model-config mamba_baseline \
        --layer-pattern "mamba,mamba,mamba,mamba,mamba,mamba,mamba,mamba" \
        --dataset wikitext --split test \
        --batch-size 4 --max-length 1024 --num-workers 2 \
        --throughput \
        --output-dir outputs/mamba_70m_wikitext_a100_mig20g/eval_results

    # xLSTM-only baseline
    python scripts/evaluate_lm.py \
        --checkpoint outputs/xlstm_70m_wikitext_a100_mig20g/checkpoints/last.ckpt \
        --model-config xlstm_baseline \
        --layer-pattern "mlstm,mlstm,mlstm,mlstm,mlstm,mlstm,mlstm,mlstm" \
        --dataset wikitext --split test \
        --batch-size 4 --max-length 1024 --num-workers 2 \
        --throughput \
        --output-dir outputs/xlstm_70m_wikitext_a100_mig20g/eval_results
"""

import os
import re
import sys
import json
import time
import math
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from hybrid_xmamba.models.configuration_hybrid import HybridConfig
from hybrid_xmamba.models.hybrid_lm import HybridLanguageModel

# Enable TF32 for A100
torch.set_float32_matmul_precision('high')


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def collate_fn(batch):
    result = {}
    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor):
            result[key] = torch.stack([item[key] for item in batch])
        else:
            result[key] = torch.tensor([item[key] for item in batch])
    return result

def prepare_test_data(dataset_name, tokenizer, max_length=1024, batch_size=4,
                      split="test", num_workers=2):
    """BUG FIX 7: Limit dataset size and use num_proc=1 to avoid disk space issues."""
    if dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-103-v1", split=split)
        text_column = "text"
    elif dataset_name == "pubmed":
        # Load PubMed dataset for perplexity evaluation
        try:
            dataset = load_dataset("ccdv/pubmed-summarization", split=split)
        except Exception as e:
            # BUG FIX 7: Better error handling (not bare except)
            print(f"  Warning: {split} split not found ({e}), using validation split")
            dataset = load_dataset("ccdv/pubmed-summarization", split="validation")
        
        # BUG FIX 7: Limit to 1000 samples to avoid disk space issues
        if len(dataset) > 1000:
            print(f"  Limiting PubMed dataset from {len(dataset)} to 1000 samples")
            dataset = dataset.select(range(1000))
        
        text_column = "article"  # PubMed uses 'article' field
    else:
        raise ValueError("Unsupported dataset: " + dataset_name)

    def tokenize_fn(examples):
        return tokenizer(examples[text_column], truncation=False, return_attention_mask=False)

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total = len(concatenated["input_ids"])
        total = (total // max_length) * max_length
        return {
            k: [t[i:i + max_length] for i in range(0, total, max_length)]
            for k, t in concatenated.items()
        }

    # BUG FIX 7: Use num_proc=1 to avoid parallel Arrow cache writes
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names,
                            num_proc=1)
    tokenized = tokenized.map(group_texts, batched=True, num_proc=1)
    loader = DataLoader(tokenized, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
    return loader, len(tokenized)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_perplexity(model, dataloader, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    pbar = tqdm(dataloader, desc="Evaluating perplexity")
    for i, batch in enumerate(pbar):
        if max_batches and i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask")
        labels = input_ids.clone()
        # If a padding mask is present (padded source), exclude pad positions
        # from the loss and the token count. Packed data has no mask → unchanged.
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            labels[attention_mask == 0] = -100
            num_tokens = int(attention_mask.sum().item()) - input_ids.shape[0]
            num_tokens = max(num_tokens, 1)
        else:
            num_tokens = input_ids.shape[0] * (input_ids.shape[1] - 1)
        outputs = model(input_ids, labels=labels, return_dict=True)
        loss = outputs.loss
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens
        pbar.set_postfix({
            "loss": "{:.4f}".format(total_loss / total_tokens),
            "ppl": "{:.2f}".format(math.exp(min(total_loss / total_tokens, 20))),
        })

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return {
        "perplexity": perplexity,
        "loss": avg_loss,
        "bits_per_byte": avg_loss / math.log(2),
        "num_tokens": total_tokens,
    }

@torch.no_grad()
def measure_throughput(model, device, seq_lengths=None, batch_size=4, warmup=3, trials=10):
    """Measure inference throughput. seq_lengths capped at 1024 for MIG 20GB safety."""
    if seq_lengths is None:
        seq_lengths = [128, 256, 512, 1024]
    model.eval()
    results = {}
    for seq_len in seq_lengths:
        input_ids = torch.randint(0, 50257, (batch_size, seq_len), device=device)
        for _ in range(warmup):
            _ = model(input_ids, return_dict=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(trials):
            _ = model(input_ids, return_dict=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        total_tokens = batch_size * seq_len * trials
        tokens_per_sec = total_tokens / elapsed
        results[seq_len] = {
            "tokens_per_second": tokens_per_sec,
            "ms_per_token": (elapsed / total_tokens) * 1000,
            "ms_per_batch": (elapsed / trials) * 1000,
        }
        print("  seq_len={:>5}: {:>12,.0f} tok/s  ({:.1f} ms/batch)".format(
            seq_len, tokens_per_sec, results[seq_len]["ms_per_batch"]))
    return results

@torch.no_grad()
def generate_samples(model, tokenizer, device, prompts=None, max_new_tokens=100):
    model.eval()
    if prompts is None:
        prompts = [
            "The history of artificial intelligence",
            "In recent years, language models have",
            "The key difference between transformers and",
        ]
    results = []
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        output = model.generate(
            input_ids, max_new_tokens=max_new_tokens, temperature=0.8, top_k=50
        )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        results.append({"prompt": prompt, "generated": generated_text})
        print("\n  Prompt: " + prompt)
        print("  Generated: " + generated_text[:200] + "...")
    return results

# ---------------------------------------------------------------------------
# Checkpoint loading — the main fix
# ---------------------------------------------------------------------------

def strip_state_dict_prefixes(state_dict):
    """Strip torch.compile (_orig_mod.) and Lightning (model.) key prefixes.

    torch.compile wraps all keys as:
      _orig_mod.<original_key>
    Lightning wraps the model as self.model, so keys become:
      model.<original_key>
    When both are active (training used torch.compile inside Lightning), keys are:
      model._orig_mod.<original_key>
    or just:
      _orig_mod.<original_key>

    This function normalises all variants to bare <original_key>.
    
    BUG FIX 2: Also strip inner 'lm.' prefix for Stage 1 checkpoints.
    """
    cleaned = {}
    for k, v in state_dict.items():
        # Remove all known prefixes in order of specificity
        if k.startswith("model._orig_mod."):
            new_k = k[len("model._orig_mod.") :]
        elif k.startswith("_orig_mod.model."):
            new_k = k[len("_orig_mod.model.") :]
        elif k.startswith("_orig_mod."):
            new_k = k[len("_orig_mod.") :]
        elif k.startswith("model."):
            new_k = k[len("model.") :]
        else:
            new_k = k
        
        # Skip projection_head and logit_scale (not part of language model)
        if new_k.startswith("projection_head.") or new_k == "logit_scale":
            continue
        
        # BUG FIX 2: Strip inner 'lm.' prefix for Stage 1 checkpoints
        if new_k.startswith("lm."):
            new_k = new_k[len("lm."):]
            
        cleaned[new_k] = v
    return cleaned

def infer_config_from_state_dict(state_dict, layer_pattern_override=None):
    """Infer dim and num_layers from state dict key shapes."""
    # Infer embedding dim from token embedding weight shape
    dim = 512  # safe default for 70m models
    for k, v in state_dict.items():
        if "token_embedding.weight" in k:
            dim = int(v.shape[1])
            break

    # Count actual number of layers from layer index in keys
    num_layers = 0
    for k in state_dict.keys():
        m = re.search(r"layers\.(\d+)\.", k)
        if m:
            idx = int(m.group(1))
            if idx + 1 > num_layers:
                num_layers = idx + 1
    if num_layers == 0:
        num_layers = 8  # fallback

    # Build layer pattern
    if layer_pattern_override:
        layer_pattern = layer_pattern_override
    else:
        # Default hybrid repeating pattern: mamba, mamba, mlstm
        base = ["mamba", "mamba", "mlstm"]
        layer_pattern = [base[i % len(base)] for i in range(num_layers)]

    return dim, num_layers, layer_pattern

def load_model_from_checkpoint(checkpoint_path, device="cuda",
                                layer_pattern_override=None, max_length=1024):
    """Load a trained model from a PyTorch Lightning checkpoint.

    Correctly handles:
    - torch.compile (_orig_mod.) key prefix
    - Lightning (model.) key prefix
    - Combination of both (model._orig_mod.)
    - Missing hyper_parameters (infers arch from state dict shapes)
    """
    print("Loading checkpoint from " + str(checkpoint_path) + "...")
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # ------------------------------------------------------------------
    # Step 1: strip all key prefixes so we have bare model keys
    # ------------------------------------------------------------------
    raw_state_dict = ckpt.get("state_dict", ckpt)
    state_dict = strip_state_dict_prefixes(raw_state_dict)

    # Sanity check
    has_embedding = any("token_embedding.weight" in k for k in state_dict.keys())
    if not has_embedding:
        print("  WARNING: embedding key not found after prefix stripping.")
        print("  First 5 raw keys: " + str(list(raw_state_dict.keys())[:5]))
        print("  First 5 cleaned keys: " + str(list(state_dict.keys())[:5]))

    # ------------------------------------------------------------------
    # Step 2: infer architecture
    # ------------------------------------------------------------------
    dim, num_layers, layer_pattern = infer_config_from_state_dict(
        state_dict, layer_pattern_override=layer_pattern_override
    )
    print("  Config used: dim={}, num_layers={}, layer_pattern={}".format(
        dim, num_layers, layer_pattern))

    # ------------------------------------------------------------------
    # Step 3: build config matching training
    # ------------------------------------------------------------------
    config = HybridConfig(
        dim=dim,
        num_layers=num_layers,
        layer_pattern=layer_pattern,
        vocab_size=50257,
        max_position_embeddings=max_length,
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
    model = HybridLanguageModel(config)

    # ------------------------------------------------------------------
    # Step 4: load weights
    # ------------------------------------------------------------------
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print("  Missing keys ({}): {}...".format(len(missing), missing[:3]))
    if unexpected:
        print("  Unexpected keys ({}): {}...".format(len(unexpected), unexpected[:3]))
    if not missing and not unexpected:
        print("  Weights loaded successfully (exact match).")
    elif len(missing) == 0:
        print("  Weights loaded — {} unexpected keys ignored.".format(len(unexpected)))
    else:
        print("  WARNING: {} missing keys — model may not have loaded correctly!".format(
            len(missing)))

    model = model.to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print("  Model loaded: {:,} parameters ({:.1f}M)".format(num_params, num_params / 1e6))
    return model, num_params

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate hybrid/mamba/xlstm language model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to last.ckpt")
    parser.add_argument("--model-config", type=str, default="hybrid_70m",
                        help="Label for results JSON (hybrid_70m / mamba_baseline / xlstm_baseline)")
    parser.add_argument("--layer-pattern", type=str, default=None,
                        help="Comma-separated layer types, e.g. "
                             "mamba,mamba,mlstm,mamba,mamba,mlstm,mamba,mamba for hybrid, "
                             "mamba,mamba,mamba,mamba,mamba,mamba,mamba,mamba for mamba-only, "
                             "mlstm,mlstm,mlstm,mlstm,mlstm,mlstm,mlstm,mlstm for xlstm-only. "
                             "If omitted, inferred from checkpoint state dict.")
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1024,
                        help="Sequence length — must match training (default 1024)")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--throughput", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Parse layer pattern if provided
    layer_pattern_override = None
    if args.layer_pattern:
        layer_pattern_override = [x.strip() for x in args.layer_pattern.split(",")]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Device: " + str(device))
    if device.type == "cuda":
        print("GPU: " + torch.cuda.get_device_name())
        print("VRAM: {:.1f} GB".format(torch.cuda.get_device_properties(0).total_memory / 1e9))

    model, num_params = load_model_from_checkpoint(
        args.checkpoint, device,
        layer_pattern_override=layer_pattern_override,
        max_length=args.max_length,
    )

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    sep = "=" * 60
    print("\n" + sep)
    print("  PERPLEXITY EVALUATION (" + args.split + " set)")
    print(sep)

    dataloader, num_samples = prepare_test_data(
        args.dataset, tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
        split=args.split,
        num_workers=args.num_workers,
    )
    print("  Samples: {}  |  Batches: {}".format(num_samples, len(dataloader)))

    ppl_results = evaluate_perplexity(model, dataloader, device, args.max_batches)
    print("\n  +-----------------------------------+")
    print("  |  Perplexity : {:>14.2f}  |".format(ppl_results["perplexity"]))
    print("  |  Loss       : {:>14.4f}  |".format(ppl_results["loss"]))
    print("  |  BPB        : {:>14.4f}  |".format(ppl_results["bits_per_byte"]))
    print("  |  Tokens     : {:>14,}  |".format(ppl_results["num_tokens"]))
    print("  +-----------------------------------+")

    throughput_results = {}
    if args.throughput:
        print("\n" + sep)
        print("  THROUGHPUT MEASUREMENT")
        print(sep)
        throughput_results = measure_throughput(model, device, batch_size=args.batch_size)

    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print("\n  Peak GPU memory: {:.2f} GB".format(peak_mem))
    else:
        peak_mem = 0

    gen_results = []
    if args.generate:
        print("\n" + sep)
        print("  TEXT GENERATION SAMPLES")
        print(sep)
        gen_results = generate_samples(model, tokenizer, device)

    all_results = {
        "model_config": args.model_config,
        "checkpoint": str(args.checkpoint),
        "dataset": args.dataset,
        "split": args.split,
        "total_params": num_params,
        "test_perplexity": ppl_results["perplexity"],
        "test_loss": ppl_results["loss"],
        "bits_per_byte": ppl_results["bits_per_byte"],
        "num_tokens_evaluated": ppl_results["num_tokens"],
        "peak_gpu_memory_gb": peak_mem,
        "throughput": throughput_results,
        "generation_samples": gen_results,
        "timestamp": datetime.now().isoformat(),
    }

    if 1024 in throughput_results:
        all_results["tokens_per_second"] = throughput_results[1024]["tokens_per_second"]

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = Path(args.output_dir) / "results.json"
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print("\n  Results saved to " + str(output_path))

    return all_results


if __name__ == "__main__":
    main()
