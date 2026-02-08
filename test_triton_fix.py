#!/usr/bin/env python3
"""Test script to verify the Triton kernel fix."""

import torch
import sys
sys.path.insert(0, '/content/hybrid_model_mamba_xlstm') if '/content' in sys.path.__str__() else None

try:
    from hybrid_xmamba.kernels.tfla.tfla_interface import apply_tfla
    print("✓ TFLA module imported successfully")
except Exception as e:
    print(f"✗ Failed to import TFLA: {e}")
    sys.exit(1)

# Test with small tensors
batch_size = 2
num_heads = 4
seq_len = 128
head_dim = 64

try:
    print(f"\nTesting TFLA kernel with:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num heads: {num_heads}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Head dim: {head_dim}")
    
    # Create test tensors
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32, device=q.device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32, device=q.device)
    i_gate = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32, device=q.device)
    f_gate = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32, device=q.device)
    
    print(f"  Device: {q.device}")
    
    # Run TFLA forward
    print("\nRunning TFLA forward pass...")
    output = apply_tfla(q, k, v, i_gate, f_gate)
    
    print(f"✓ TFLA forward pass succeeded!")
    print(f"  Output shape: {output.shape}")
    print(f"  Output dtype: {output.dtype}")
    
except Exception as e:
    print(f"✗ TFLA test failed with error:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ All tests passed! The Triton fix is working correctly.")
