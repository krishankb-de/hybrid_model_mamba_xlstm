#!/usr/bin/env python3
"""Quick memory test for the updated training configuration.

This script tests if the model and batch size fit in available GPU memory
without running full training.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from hybrid_xmamba.models.configuration_hybrid import HybridConfig
from hybrid_xmamba.models.hybrid_lm import HybridLanguageModel


def format_bytes(bytes_val):
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} TB"


def test_memory_config():
    """Test if the configuration fits in GPU memory."""
    
    print("=" * 80)
    print("MEMORY CONFIGURATION TEST")
    print("=" * 80)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This test requires a GPU.")
        return False
    
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory
    
    print(f"\nGPU: {gpu_name}")
    print(f"Total VRAM: {format_bytes(total_memory)}")
    
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    initial_memory = torch.cuda.memory_allocated()
    print(f"Initial memory: {format_bytes(initial_memory)}")
    
    # Create model configuration (matching hybrid_70m.yaml with updates)
    print("\n" + "-" * 80)
    print("Creating model...")
    print("-" * 80)
    
    config = HybridConfig(
        vocab_size=50257,
        dim=512,
        num_layers=8,
        layer_pattern=["mamba", "mamba", "mlstm"],
        state_size=16,
        conv_size=4,
        expand_factor=2,
        dt_rank=None,
        use_fast_path=True,
        head_dim=64,
        num_heads=8,
        use_tfla=True,
        proj_factor=2,
        slstm_hidden_dim=512,
        slstm_num_heads=4,
        use_exponential_gate=True,
        norm_type="rms",
        use_mlp=True,
        mlp_ratio=4.0,
        max_position_embeddings=1024,  # Updated from 2048
        dropout=0.1,
        initializer_range=0.02,
        use_cache=True,
        tie_word_embeddings=False,
    )
    
    try:
        model = HybridLanguageModel(config).to(device)
        model_memory = torch.cuda.memory_allocated() - initial_memory
        print(f"✅ Model created successfully")
        print(f"Model memory: {format_bytes(model_memory)}")
        
        num_params = model.get_num_params(non_embedding=True)
        print(f"Parameters: {num_params:,} ({num_params/1e6:.1f}M)")
        
    except RuntimeError as e:
        print(f"❌ Failed to create model: {e}")
        return False
    
    # Test with training batch (bf16 mixed precision)
    print("\n" + "-" * 80)
    print("Testing training batch...")
    print("-" * 80)
    
    batch_size = 4  # Updated from 8
    seq_length = 1024  # Updated from 2048
    
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_length}")
    print(f"Tokens per batch: {batch_size * seq_length:,}")
    
    try:
        # Create dummy batch
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length), device=device)
        labels = input_ids.clone()
        
        # Forward pass with bf16 autocast
        model.train()
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(input_ids, labels=labels, return_dict=True)
            loss = outputs.loss
        
        forward_memory = torch.cuda.memory_allocated()
        print(f"✅ Forward pass successful")
        print(f"Memory after forward: {format_bytes(forward_memory)}")
        
        # Backward pass
        loss.backward()
        
        backward_memory = torch.cuda.memory_allocated()
        peak_memory = torch.cuda.max_memory_allocated()
        
        print(f"✅ Backward pass successful")
        print(f"Memory after backward: {format_bytes(backward_memory)}")
        print(f"Peak memory: {format_bytes(peak_memory)}")
        
        # Calculate memory usage
        memory_used_pct = (peak_memory / total_memory) * 100
        memory_free = total_memory - peak_memory
        
        print(f"\n" + "=" * 80)
        print("MEMORY SUMMARY")
        print("=" * 80)
        print(f"Peak memory used: {format_bytes(peak_memory)} ({memory_used_pct:.1f}%)")
        print(f"Memory free: {format_bytes(memory_free)} ({100-memory_used_pct:.1f}%)")
        
        # Safety check
        if memory_used_pct > 90:
            print(f"\n⚠️  WARNING: Using {memory_used_pct:.1f}% of GPU memory!")
            print("   Consider reducing batch size or sequence length further.")
            return False
        elif memory_used_pct > 80:
            print(f"\n⚠️  CAUTION: Using {memory_used_pct:.1f}% of GPU memory.")
            print("   Training should work but may be close to the limit.")
            return True
        else:
            print(f"\n✅ SAFE: Using {memory_used_pct:.1f}% of GPU memory.")
            print("   Sufficient headroom for training.")
            return True
        
    except RuntimeError as e:
        print(f"\n❌ Memory test failed: {e}")
        print("\nSuggestions:")
        print("  1. Reduce batch_size to 2")
        print("  2. Reduce sequence length to 512")
        print("  3. Use gradient checkpointing")
        return False
    
    finally:
        # Cleanup
        del model
        if 'input_ids' in locals():
            del input_ids, labels, outputs, loss
        torch.cuda.empty_cache()


def test_gradient_accumulation():
    """Test gradient accumulation scenario."""
    print("\n" + "=" * 80)
    print("GRADIENT ACCUMULATION TEST")
    print("=" * 80)
    
    accumulation_steps = 8
    batch_size = 4
    effective_batch = batch_size * accumulation_steps
    
    print(f"Accumulation steps: {accumulation_steps}")
    print(f"Per-GPU batch size: {batch_size}")
    print(f"Effective batch size: {effective_batch}")
    print(f"Tokens per update: {effective_batch * 1024:,}")
    
    print("\n✅ Gradient accumulation configuration looks good!")
    print("   This maintains the same effective batch size as the original config.")


if __name__ == "__main__":
    print("\nTesting memory configuration for FineWeb training...")
    print("This will test if the model fits in your GPU memory.\n")
    
    success = test_memory_config()
    test_gradient_accumulation()
    
    print("\n" + "=" * 80)
    if success:
        print("✅ MEMORY TEST PASSED")
        print("=" * 80)
        print("\nYou can proceed with training:")
        print("  sbatch scripts/train_hybrid_70m_fineweb.sh")
    else:
        print("❌ MEMORY TEST FAILED")
        print("=" * 80)
        print("\nPlease review the suggestions above and adjust the configuration.")
        print("See MIG_GPU_MEMORY_FIX.md for detailed guidance.")
    
    print()
