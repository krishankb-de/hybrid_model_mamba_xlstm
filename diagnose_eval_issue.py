#!/usr/bin/env python3
"""Comprehensive diagnostic for evaluation loading issues."""

import sys
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from hybrid_xmamba.models.configuration_hybrid import HybridConfig
from hybrid_xmamba.models.hybrid_lm import HybridTextEncoder


def diagnose_checkpoint_loading(checkpoint_path):
    """Diagnose why checkpoint loading is failing."""
    print("=" * 80)
    print("CHECKPOINT LOADING DIAGNOSTIC")
    print("=" * 80)
    print(f"\nCheckpoint: {checkpoint_path}\n")
    
    # Load checkpoint
    print("1. Loading checkpoint...")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    raw_state_dict = ckpt.get("state_dict", ckpt)
    print(f"   Total keys in checkpoint: {len(raw_state_dict)}")
    
    # Show first 10 keys
    print(f"\n   First 10 keys:")
    for i, k in enumerate(list(raw_state_dict.keys())[:10]):
        print(f"     {i+1}. {k}")
    
    # Count layers BEFORE stripping
    print(f"\n2. Counting layers...")
    num_layers = 0
    for k in raw_state_dict.keys():
        m = re.search(r"layers\.(\d+)\.", k)
        if m:
            idx = int(m.group(1))
            if idx + 1 > num_layers:
                num_layers = idx + 1
    print(f"   Found {num_layers} layers")
    
    # Strip prefixes
    print(f"\n3. Stripping prefixes...")
    state_dict = {}
    stripped_model = 0
    stripped_lm = 0
    skipped_projection = 0
    skipped_logit = 0
    
    for k, v in raw_state_dict.items():
        original_k = k
        
        if k.startswith("model."):
            k = k[len("model."):]
            stripped_model += 1
        
        if k.startswith("projection_head.") or k == "logit_scale":
            if "projection_head" in k:
                skipped_projection += 1
            else:
                skipped_logit += 1
            continue
        
        if k.startswith("lm."):
            k = k[len("lm."):]
            stripped_lm += 1
            
        state_dict[k] = v
    
    print(f"   Stripped 'model.' prefix: {stripped_model} keys")
    print(f"   Stripped 'lm.' prefix: {stripped_lm} keys")
    print(f"   Skipped projection_head: {skipped_projection} keys")
    print(f"   Skipped logit_scale: {skipped_logit} keys")
    print(f"   Final state_dict size: {len(state_dict)} keys")
    
    # Infer dim
    print(f"\n4. Inferring embedding dimension...")
    dim = 512
    for k, v in state_dict.items():
        if "token_embedding.weight" in k:
            dim = int(v.shape[1])
            print(f"   Found token_embedding.weight: shape {v.shape} -> dim={dim}")
            break
    
    # Build config
    print(f"\n5. Building model config...")
    base = ["mamba", "mamba", "mlstm"]
    layer_pattern = [base[i % len(base)] for i in range(num_layers)]
    
    config = HybridConfig(
        dim=dim,
        num_layers=num_layers,
        layer_pattern=layer_pattern,
        vocab_size=50257,
        max_position_embeddings=512,
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
    
    print(f"   Config: dim={dim}, num_layers={num_layers}")
    print(f"   Layer pattern: {layer_pattern}")
    
    # Create model
    print(f"\n6. Creating HybridTextEncoder...")
    model = HybridTextEncoder(config, embed_dim=512)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {num_params:,} ({num_params/1e6:.1f}M)")
    
    # Get model's expected keys
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())
    
    print(f"\n   Model expects {len(model_keys)} keys")
    print(f"   Checkpoint has {len(checkpoint_keys)} keys")
    
    # Load weights
    print(f"\n7. Loading weights...")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    print(f"   Missing keys: {len(missing)}")
    print(f"   Unexpected keys: {len(unexpected)}")
    
    if missing:
        print(f"\n   First 10 missing keys:")
        for i, k in enumerate(list(missing)[:10]):
            print(f"     {i+1}. {k}")
    
    if unexpected:
        print(f"\n   First 10 unexpected keys:")
        for i, k in enumerate(list(unexpected)[:10]):
            print(f"     {i+1}. {k}")
    
    # Analyze the mismatch
    print(f"\n8. Analyzing key mismatch...")
    
    # Check if it's a simple prefix issue
    missing_set = set(missing)
    unexpected_set = set(unexpected)
    
    # Try adding/removing common prefixes
    prefixes_to_try = ["lm.", "encoder.", "model.", ""]
    
    for prefix in prefixes_to_try:
        if prefix:
            # Try adding prefix to unexpected keys
            matches = sum(1 for uk in unexpected_set if f"{prefix}{uk}" in missing_set)
            if matches > 0:
                print(f"   ⚠ {matches} unexpected keys would match if we ADD '{prefix}' prefix")
            
            # Try removing prefix from unexpected keys
            matches = sum(1 for uk in unexpected_set 
                         if uk.startswith(prefix) and uk[len(prefix):] in missing_set)
            if matches > 0:
                print(f"   ⚠ {matches} unexpected keys would match if we REMOVE '{prefix}' prefix")
    
    # Check for exact matches with different prefixes
    missing_basenames = {k.split(".")[-1] for k in missing_set}
    unexpected_basenames = {k.split(".")[-1] for k in unexpected_set}
    common_basenames = missing_basenames & unexpected_basenames
    
    if common_basenames:
        print(f"   ⚠ {len(common_basenames)} keys have same basename but different paths")
        print(f"     Examples: {list(common_basenames)[:5]}")
    
    # Final verdict
    print(f"\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    
    if len(missing) == 0 and len(unexpected) == 0:
        print("\n✅ SUCCESS: All weights loaded correctly!")
        return True
    elif len(missing) > 100:
        print(f"\n❌ CRITICAL: {len(missing)} missing keys - model architecture mismatch!")
        print(f"\nLikely causes:")
        print(f"  1. Wrong model class (using HybridTextEncoder vs HybridLanguageModel)")
        print(f"  2. Wrong embed_dim parameter")
        print(f"  3. Checkpoint from different training stage")
        return False
    elif len(missing) < 10:
        print(f"\n⚠ WARNING: {len(missing)} missing keys - might be OK")
        print(f"  These could be optimizer states or non-essential parameters")
        return True
    else:
        print(f"\n⚠ UNCLEAR: {len(missing)} missing, {len(unexpected)} unexpected")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnose evaluation loading issues")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to checkpoint")
    args = parser.parse_args()
    
    try:
        success = diagnose_checkpoint_loading(args.checkpoint)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
