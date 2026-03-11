"""Comprehensive language model evaluation script.

Computes:
  1. Test perplexity on WikiText-103 (standard LM benchmark)
  2. Inference throughput (tokens/second)
  3. Peak GPU memory usage
  4. Optional: text generation samples

Usage:
    python scripts/evaluate_lm.py \
        --checkpoint outputs/hybrid_70m_wikitext_a100_mig20g/checkpoints/last.ckpt \
        --model-config hybrid_70m \
        --dataset wikitext \
        --split test
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

# Enable TF32 for A100
torch.set_float32_matmul_precision('high')

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
    if dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-103-v1", split=split)
    else:
        raise ValueError("Unsupported dataset: " + dataset_name)

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
        num_tokens = input_ids.shape[0] * (input_ids.shape[1] - 1)
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

def load_model_from_checkpoint(checkpoint_path, device="cuda"):
    """Load model from a PyTorch Lightning checkpoint.

    Reconstructs HybridConfig from the checkpoint's saved cfg dict,
    then loads the state dict so the correct trained weights are used.
    """
    print("Loading checkpoint from " + str(checkpoint_path) + "...")
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # ----------------------------------------------------------------
    # Step 1: reconstruct HybridConfig from what was saved in the ckpt
    # ----------------------------------------------------------------
    cfg_kwargs = {}

    # The training script saves the Hydra cfg under 'cfg' in hyper_parameters
    hp = ckpt.get("hyper_parameters", {})
    saved_cfg = hp.get("cfg", None)

    if saved_cfg is not None:
        # Extract model sub-config (OmegaConf DictConfig or plain dict)
        try:
            from omegaconf import OmegaConf
            model_cfg = OmegaConf.to_container(saved_cfg.get("model", {}), resolve=True)
        except Exception:
            model_cfg = dict(saved_cfg.get("model", {}))

        # Map known fields to HybridConfig kwargs
        field_map = {
            "vocab_size": "vocab_size",
            "dim": "dim",
            "num_layers": "num_layers",
            "layer_pattern": "layer_pattern",
            "state_size": "state_size",
            "conv_size": "conv_size",
            "expand_factor": "expand_factor",
            "dt_rank": "dt_rank",
            "use_fast_path": "use_fast_path",
            "head_dim": "head_dim",
            "num_heads": "num_heads",
            "use_tfla": "use_tfla",
            "proj_factor": "proj_factor",
            "slstm_hidden_dim": "slstm_hidden_dim",
            "slstm_num_heads": "slstm_num_heads",
            "use_exponential_gate": "use_exponential_gate",
            "norm_type": "norm_type",
            "use_mlp": "use_mlp",
            "mlp_ratio": "mlp_ratio",
            "max_position_embeddings": "max_position_embeddings",
            "dropout": "dropout",
            "initializer_range": "initializer_range",
            "use_cache": "use_cache",
            "tie_word_embeddings": "tie_word_embeddings",
        }
        for src_key, dst_key in field_map.items():
            if src_key in model_cfg and model_cfg[src_key] is not None:
                cfg_kwargs[dst_key] = model_cfg[src_key]

    if not cfg_kwargs:
        # Fallback: try to infer from state dict shape
        print("  Warning: could not read config from checkpoint hyper_parameters.")
        print("  Inferring architecture from state_dict shapes...")
        state_dict = ckpt.get("state_dict", ckpt)
        # Count layers by finding the deepest layer index
        layer_indices = set()
        for k in state_dict.keys():
            parts = k.split(".")
            # keys look like: model.layers.0.mamba.in_proj.weight
            if len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers":
                try:
                    layer_indices.add(int(parts[2]))
                except ValueError:
                    pass
        if layer_indices:
            cfg_kwargs["num_layers"] = max(layer_indices) + 1
        # Try to get dim from embedding weight
        for k, v in state_dict.items():
            if "token_embedding.weight" in k:
                cfg_kwargs["vocab_size"] = v.shape[0]
                cfg_kwargs["dim"] = v.shape[1]
                break

    # Build config and model
    config = HybridConfig(**cfg_kwargs)
    model = HybridLanguageModel(config)

    # ----------------------------------------------------------------
    # Step 2: load state dict (strip "model." prefix added by Lightning)
    # ----------------------------------------------------------------
    state_dict = ckpt.get("state_dict", ckpt)
    cleaned = {}
    for k, v in state_dict.items():
        # Lightning wraps everything under self.model, so keys are "model.xxx"
        if k.startswith("model."):
            cleaned[k[6:]] = v
        else:
            cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print("  Missing keys ({}): {}...".format(len(missing), missing[:3]))
    if unexpected:
        print("  Unexpected keys ({}): {}...".format(len(unexpected), unexpected[:3]))

    model = model.to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print("  Config used: dim={}, num_layers={}, layer_pattern={}".format(
        config.dim, config.num_layers, config.layer_pattern))
    print("  Model loaded: {:,} parameters ({:.1f}M)".format(num_params, num_params / 1e6))
    return model, num_params

def main():
    parser = argparse.ArgumentParser(description="Evaluate hybrid language model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model-config", type=str, default="hybrid_70m")
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--throughput", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Device: " + str(device))
    if device.type == "cuda":
        print("GPU: " + torch.cuda.get_device_name())
        print("VRAM: {:.1f} GB".format(torch.cuda.get_device_properties(0).total_memory / 1e9))

    model, num_params = load_model_from_checkpoint(args.checkpoint, device)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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