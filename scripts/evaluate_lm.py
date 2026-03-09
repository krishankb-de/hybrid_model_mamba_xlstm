"""Comprehensive language model evaluation script.

Computes:
  1. Test perplexity on WikiText-103 (standard LM benchmark)
  2. Inference throughput (tokens/second)
  3. Peak GPU memory usage
  4. Optional: text generation samples

Usage:
    python scripts/evaluate_lm.py \
        --checkpoint outputs/hybrid_150m_wikitext/checkpoints/last.ckpt \
        --model-config hybrid_150m \
        --dataset wikitext \
        --split test

    # Compare all three models
    python scripts/evaluate_lm.py \
        --checkpoint outputs/hybrid_150m_wikitext/checkpoints/last.ckpt \
        --model-config hybrid_150m \
        --generate
"""

import os
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from hybrid_xmamba.models.configuration_hybrid import HybridConfig
from hybrid_xmamba.models.hybrid_lm import HybridLanguageModel
from hybrid_xmamba.training.lightning_module import HybridLightningModule

# Enable TF32 for A100
torch.set_float32_matmul_precision('high')

def collate_fn(batch):
    """Collate function for tokenized data."""
    result = {}
    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor):
            result[key] = torch.stack([item[key] for item in batch])
        else:
            result[key] = torch.tensor([item[key] for item in batch])
    return result

def prepare_test_data(dataset_name, tokenizer, max_length=2048, batch_size=16,
                      split="test", num_workers=4):
    """Prepare test dataloader with text packing (same as training)."""
    if dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-103-v1", split=split)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=False, return_attention_mask=False)

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total = len(concatenated["input_ids"])
        total = (total // max_length) * max_length
        return {
            k: [t[i:i + max_length] for i in range(0, total, max_length)]
            for k, t in concatenated.items()
        }

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names,
                            num_proc=num_workers)
    tokenized = tokenized.map(group_texts, batched=True, num_proc=num_workers)

    loader = DataLoader(tokenized, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
    return loader, len(tokenized)

@torch.no_grad()
def evaluate_perplexity(model, dataloader, device, max_batches=None):
    """Compute perplexity on a test set.

    Uses standard cross-entropy loss with teacher forcing (same as training).

    Returns:
        dict with 'perplexity', 'loss', 'num_tokens'
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    pbar = tqdm(dataloader, desc="Evaluating perplexity")
    for i, batch in enumerate(pbar):
        if max_batches and i >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        labels = input_ids.clone()

        outputs = model(input_ids, labels=labels, return_dict=True)
        loss = outputs.loss

        # Count actual tokens (seq_len - 1 per sample due to shift)
        num_tokens = (input_ids.shape[0] * (input_ids.shape[1] - 1))
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

        pbar.set_postfix({
            "loss": f"{total_loss / total_tokens:.4f}",
            "ppl": f"{math.exp(total_loss / total_tokens):.2f}",
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
def measure_throughput(model, device, seq_lengths=[128, 256, 512, 1024, 2048],
                       batch_size=8, warmup=3, trials=10):
    """Measure inference throughput at different sequence lengths.

    Returns:
        dict mapping seq_len -> tokens_per_second
    """
    model.eval()
    results = {}

    for seq_len in seq_lengths:
        input_ids = torch.randint(0, 50257, (batch_size, seq_len), device=device)

        # Warmup
        for _ in range(warmup):
            _ = model(input_ids, return_dict=True)
        torch.cuda.synchronize()

        # Timed runs
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
        print(f"  seq_len={seq_len:>5}: {tokens_per_sec:>12,.0f} tok/s  "
              f"({results[seq_len]['ms_per_batch']:.1f} ms/batch)")

    return results

@torch.no_grad()
def generate_samples(model, tokenizer, device, prompts=None, max_new_tokens=100):
    """Generate text samples for qualitative evaluation."""
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
        print(f"\n  Prompt: {prompt}")
        print(f"  Generated: {generated_text[:200]}...")

    return results

def load_model_from_checkpoint(checkpoint_path, device="cuda"):
    """Load a trained model from a Lightning checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")

    # Load the Lightning module
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Extract model config from hyperparameters or reconstruct
    if "hyper_parameters" in ckpt:
        hp = ckpt["hyper_parameters"]
    else:
        hp = {}

    # Try to load via Lightning module
    try:
        lightning_module = HybridLightningModule.load_from_checkpoint(
            checkpoint_path, map_location=device
        )
        model = lightning_module.model
    except Exception as e:
        print(f"  Warning: Could not load Lightning module ({e})")
        print(f"  Trying to extract model state dict directly...")

        # Fallback: extract state dict and build model manually
        state_dict = ckpt.get("state_dict", ckpt)
        # Remove 'model.' prefix if present (from Lightning wrapping)
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                cleaned_state_dict[k[6:]] = v
            else:
                cleaned_state_dict[k] = v

        # Build model with default config (user should pass --model-config)
        config = HybridConfig()
        model = HybridLanguageModel(config)
        model.load_state_dict(cleaned_state_dict, strict=False)

    model = model.to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {num_params:,} parameters ({num_params/1e6:.1f}M)")

    return model, num_params

def main():
    parser = argparse.ArgumentParser(description="Evaluate hybrid language model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.ckpt)")
    parser.add_argument("--model-config", type=str, default="hybrid_150m",
                        help="Model config name (for reference only)")
    parser.add_argument("--dataset", type=str, default="wikitext",
                        help="Dataset to evaluate on")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split (test/validation)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Evaluation batch size")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Max batches for evaluation (None = full dataset)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save results JSON")
    parser.add_argument("--throughput", action="store_true",
                        help="Measure inference throughput")
    parser.add_argument("--generate", action="store_true",
                        help="Generate text samples")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load model
    model, num_params = load_model_from_checkpoint(args.checkpoint, device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Perplexity evaluation ----
    print(f"
{'='*60}")
    print(f"  PERPLEXITY EVALUATION ({args.split} set)")
    print(f"{'='*60}")

    dataloader, num_samples = prepare_test_data(
        args.dataset, tokenizer, max_length=2048,
        batch_size=args.batch_size, split=args.split
    )
    print(f"  Samples: {num_samples}  |  Batches: {len(dataloader)}")

    ppl_results = evaluate_perplexity(model, dataloader, device, args.max_batches)
    print(f"\n  ┌─────────────────────────────────┐")
    print(f"  │  Perplexity : {ppl_results['perplexity']:>14.2f}  │")
    print(f"  │  Loss       : {ppl_results['loss']:>14.4f}  │")
    print(f"  │  BPB        : {ppl_results['bits_per_byte']:>14.4f}  │")
    print(f"  │  Tokens     : {ppl_results['num_tokens']:>14,}  │")
    print(f"  └─────────────────────────────────┘")

    # ---- Throughput measurement ----
    throughput_results = {}
    if args.throughput:
        print(f"
{'='*60}")
        print(f"  THROUGHPUT MEASUREMENT")
        print(f"{'='*60}")
        throughput_results = measure_throughput(model, device, batch_size=args.batch_size)

    # ---- Peak memory ----
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"\n  Peak GPU memory: {peak_mem:.2f} GB")
    else:
        peak_mem = 0

    # ---- Text generation ----
    gen_results = []
    if args.generate:
        print(f"
{'='*60}")
        print(f"  TEXT GENERATION SAMPLES")
        print(f"{'='*60}")
        gen_results = generate_samples(model, tokenizer, device)

    # ---- Save results ----
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

    # Add tokens/sec at seq_len=2048 for the comparison table
    if 2048 in throughput_results:
        all_results["tokens_per_second"] = throughput_results[2048]["tokens_per_second"]

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = Path(args.output_dir) / "results.json"
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  Results saved to {output_path}")

    return all_results


if __name__ == "__main__":
    main()