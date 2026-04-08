#!/usr/bin/env python3
"""Debug script to inspect checkpoint key structure."""

import sys
import torch
from pathlib import Path

def inspect_checkpoint(checkpoint_path):
    """Inspect the actual keys in the checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    print("=" * 80)
    
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    
    # Get state dict
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        print(f"\nCheckpoint has 'state_dict' key")
    else:
        state_dict = ckpt
        print(f"\nCheckpoint IS the state dict (no wrapper)")
    
    print(f"Total keys: {len(state_dict)}")
    
    # Analyze key patterns
    print("\n" + "=" * 80)
    print("KEY PATTERN ANALYSIS")
    print("=" * 80)
    
    has_model_prefix = sum(1 for k in state_dict.keys() if k.startswith("model."))
    has_lm_prefix = sum(1 for k in state_dict.keys() if k.startswith("lm.") or k.startswith("model.lm."))
    has_bare_embeddings = sum(1 for k in state_dict.keys() if k.startswith("embeddings."))
    has_bare_layers = sum(1 for k in state_dict.keys() if k.startswith("layers."))
    has_projection = sum(1 for k in state_dict.keys() if "projection_head" in k)
    has_logit_scale = sum(1 for k in state_dict.keys() if k == "logit_scale" or k == "model.logit_scale")
    
    print(f"\nKeys starting with 'model.': {has_model_prefix}")
    print(f"Keys starting with 'lm.' or 'model.lm.': {has_lm_prefix}")
    print(f"Keys starting with 'embeddings.': {has_bare_embeddings}")
    print(f"Keys starting with 'layers.': {has_bare_layers}")
    print(f"Keys containing 'projection_head': {has_projection}")
    print(f"Keys named 'logit_scale': {has_logit_scale}")
    
    # Show first 20 keys
    print("\n" + "=" * 80)
    print("FIRST 20 KEYS")
    print("=" * 80)
    for i, key in enumerate(list(state_dict.keys())[:20]):
        shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else "scalar"
        print(f"{i+1:2d}. {key:60s} {shape}")
    
    # Check for specific patterns
    print("\n" + "=" * 80)
    print("SPECIFIC KEY CHECKS")
    print("=" * 80)
    
    # Check embedding key
    embedding_keys = [k for k in state_dict.keys() if "token_embedding.weight" in k]
    if embedding_keys:
        print(f"\nToken embedding key(s):")
        for k in embedding_keys:
            print(f"  - {k}: shape {state_dict[k].shape}")
    else:
        print(f"\nNo token embedding key found!")
    
    # Check layer keys
    layer_keys = [k for k in state_dict.keys() if "layers.0." in k]
    if layer_keys:
        print(f"\nFirst layer keys (layers.0.*):")
        for k in layer_keys[:5]:
            print(f"  - {k}")
    else:
        print(f"\nNo 'layers.0.' keys found!")
    
    # Determine the correct prefix structure
    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    
    if has_bare_embeddings > 0 and has_bare_layers > 0:
        print("\n✓ Checkpoint has BARE keys (no lm. prefix)")
        print("  Keys like: embeddings.token_embedding.weight, layers.0.norm1.weight")
        print("  This is from Stage 1 contrastive training")
    elif has_lm_prefix > 0:
        print("\n✓ Checkpoint has LM-PREFIXED keys")
        print("  Keys like: lm.embeddings.token_embedding.weight, lm.layers.0.norm1.weight")
        print("  This is from standard language model training")
    else:
        print("\n⚠ Unclear key structure!")
    
    if has_model_prefix > 0:
        print(f"\n✓ Checkpoint has Lightning 'model.' wrapper prefix")
        print(f"  {has_model_prefix} keys start with 'model.'")
    
    if has_projection > 0:
        print(f"\n✓ Checkpoint has projection_head (contrastive training)")
    
    if has_logit_scale > 0:
        print(f"\n✓ Checkpoint has logit_scale (contrastive training)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug checkpoint key structure")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to checkpoint")
    args = parser.parse_args()
    
    try:
        inspect_checkpoint(args.checkpoint)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
