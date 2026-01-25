"""Profiling script for analyzing model performance.

Profiles:
- Memory usage
- Throughput (tokens/second)
- Kernel performance
- Layer-wise timing
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import torch
import time
from contextlib import contextmanager

from hybrid_xmamba.models.configuration_hybrid import HybridConfig
from hybrid_xmamba.models.hybrid_lm import HybridLanguageModel


@contextmanager
def timer(name: str):
    """Simple timing context manager."""
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"{name}: {(end - start) * 1000:.2f}ms")


def profile_model(
    config: HybridConfig,
    batch_size: int = 4,
    seq_length: int = 2048,
    num_iterations: int = 10,
    device: str = "cuda",
):
    """Profile model performance.
    
    Args:
        config: Model configuration
        batch_size: Batch size for profiling
        seq_length: Sequence length
        num_iterations: Number of iterations to average
        device: Device to run on
    """
    print("=" * 80)
    print("Model Profiling")
    print("=" * 80)
    
    # Create model
    model = HybridLanguageModel(config).to(device)
    model.eval()
    
    # Print model info
    num_params = model.get_num_params(non_embedding=True)
    print(f"Model: {num_params/1e6:.1f}M parameters")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_length}")
    print(f"Device: {device}")
    print()
    
    # Create dummy input
    input_ids = torch.randint(
        0, config.vocab_size,
        (batch_size, seq_length),
        device=device
    )
    
    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_ids)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Profile forward pass
    print("\nProfiling forward pass...")
    forward_times = []
    
    with torch.no_grad():
        for i in range(num_iterations):
            if device == "cuda":
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            outputs = model(input_ids)
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            forward_times.append(end - start)
            
            print(f"Iteration {i+1}: {forward_times[-1]*1000:.2f}ms")
    
    # Compute statistics
    avg_time = sum(forward_times) / len(forward_times)
    tokens_per_sec = (batch_size * seq_length) / avg_time
    
    print("\n" + "=" * 80)
    print("Results:")
    print(f"Average forward time: {avg_time*1000:.2f}ms")
    print(f"Throughput: {tokens_per_sec:.0f} tokens/second")
    print(f"Throughput: {tokens_per_sec/1000:.2f}k tokens/second")
    print("=" * 80)
    
    # Memory profiling (if CUDA)
    if device == "cuda":
        memory_allocated = torch.cuda.max_memory_allocated() / 1e9
        memory_reserved = torch.cuda.max_memory_reserved() / 1e9
        print(f"\nMemory allocated: {memory_allocated:.2f} GB")
        print(f"Memory reserved: {memory_reserved:.2f} GB")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Profile hybrid model")
    parser.add_argument(
        "--model",
        type=str,
        default="hybrid_350m",
        choices=["hybrid_350m", "hybrid_7b", "mamba_baseline", "xlstm_baseline"],
        help="Model configuration to profile",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=2048,
        help="Sequence length",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=10,
        help="Number of iterations",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    
    args = parser.parse_args()
    
    # Load model config from registry
    from hybrid_xmamba.utils.registry import ModelRegistry
    config = ModelRegistry.get_config(args.model)
    
    # Profile
    profile_model(
        config=config,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        num_iterations=args.num_iterations,
        device=args.device,
    )


if __name__ == "__main__":
    main()
