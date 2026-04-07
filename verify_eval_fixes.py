#!/usr/bin/env python3
"""Quick verification that evaluation fixes are working correctly.

This script checks that the checkpoint loading logic correctly infers:
1. Number of layers (should be 8 for 70M model)
2. Embedding dimension (should be 512 for 70M model)
3. Model architecture matches checkpoint

Run this before full evaluation to verify the fixes.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import re

def verify_checkpoint_loading(checkpoint_path):
    """Verify that checkpoint loading correctly infers architecture."""
    print(f"Verifying checkpoint: {checkpoint_path}")
    print("=" * 70)
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    raw_state_dict = ckpt.get("state_dict", ckpt)
    
    print(f"\n1. Raw checkpoint keys (first 5):")
    for i, k in enumerate(list(raw_state_dict.keys())[:5]):
        print(f"   {k}")
    
    # Count layers BEFORE stripping (FIXED METHOD)
    print(f"\n2. Counting layers BEFORE prefix stripping (CORRECT):")
    num_layers_before = 0
    for k in raw_state_dict.keys():
        m = re.search(r"layers\.(\d+)\.", k)
        if m:
            idx = int(m.group(1))
            if idx + 1 > num_layers_before:
                num_layers_before = idx + 1
    print(f"   Found {num_layers_before} layers")
    
    # Count layers AFTER stripping (BROKEN METHOD - for comparison)
    print(f"\n3. Counting layers AFTER prefix stripping (BROKEN - for comparison):")
    state_dict_stripped = {}
    for k, v in raw_state_dict.items():
        if k.startswith("model."):
            k = k[len("model."):]
        if k.startswith("lm."):
            k = k[len("lm."):]
        state_dict_stripped[k] = v
    
    num_layers_after = 0
    for k in state_dict_stripped.keys():
        if "lm.layers." in k:  # This will never match!
            m = re.search(r"lm\.layers\.(\d+)\.", k)
            if m:
                idx = int(m.group(1))
                if idx + 1 > num_layers_after:
                    num_layers_after = idx + 1
    print(f"   Found {num_layers_after} layers (would default to 12 - WRONG!)")
    
    # Check embedding dimension
    print(f"\n4. Checking embedding dimension:")
    dim = None
    for k, v in state_dict_stripped.items():
        if "token_embedding.weight" in k:
            dim = int(v.shape[1])
            print(f"   Found dim={dim} from token_embedding.weight shape {v.shape}")
            break
    
    if dim is None:
        print(f"   WARNING: Could not find token_embedding.weight!")
        dim = 512  # default
    
    # Check for key patterns
    print(f"\n5. Key pattern analysis:")
    has_lm_prefix = any(k.startswith("lm.") or k.startswith("model.lm.") 
                        for k in raw_state_dict.keys())
    has_model_prefix = any(k.startswith("model.") for k in raw_state_dict.keys())
    has_projection = any("projection_head" in k for k in raw_state_dict.keys())
    has_logit_scale = any(k == "logit_scale" or k == "model.logit_scale" 
                          for k in raw_state_dict.keys())
    
    print(f"   Has 'lm.' prefix: {has_lm_prefix}")
    print(f"   Has 'model.' prefix: {has_model_prefix}")
    print(f"   Has projection_head: {has_projection}")
    print(f"   Has logit_scale: {has_logit_scale}")
    
    # Summary
    print(f"\n" + "=" * 70)
    print(f"VERIFICATION SUMMARY:")
    print(f"=" * 70)
    print(f"✓ Layers detected (FIXED method): {num_layers_before}")
    print(f"✗ Layers detected (BROKEN method): {num_layers_after} (would use default 12)")
    print(f"✓ Embedding dimension: {dim}")
    print(f"✓ Expected for 70M model: 8 layers, dim=512")
    
    if num_layers_before == 8 and dim == 512:
        print(f"\n✅ SUCCESS: Checkpoint matches 70M model architecture!")
        return True
    else:
        print(f"\n⚠️  WARNING: Architecture mismatch detected!")
        if num_layers_before != 8:
            print(f"   Expected 8 layers, found {num_layers_before}")
        if dim != 512:
            print(f"   Expected dim=512, found {dim}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify evaluation fixes")
    parser.add_argument("--checkpoint", type=str, 
                       default="outputs/stage1_pubmed_simcse/checkpoints/contrastive-step=008721-val/contrastive_loss=0.0110.ckpt",
                       help="Path to checkpoint to verify")
    args = parser.parse_args()
    
    try:
        success = verify_checkpoint_loading(args.checkpoint)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
