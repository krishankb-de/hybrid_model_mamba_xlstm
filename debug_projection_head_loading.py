#!/usr/bin/env python3
"""Debug why projection_head is not loading correctly."""

import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from hybrid_xmamba.models.configuration_hybrid import HybridConfig
from hybrid_xmamba.models.hybrid_lm import HybridTextEncoder


def main():
    checkpoint_path = "outputs/stage1_pubmed_simcse/checkpoints/contrastive-step=008721-val/contrastive_loss=0.0110.ckpt"
    
    print("=" * 80)
    print("PROJECTION HEAD LOADING DEBUG")
    print("=" * 80)
    print()
    
    # Load checkpoint
    print("1. Loading checkpoint...")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    raw_state_dict = ckpt.get("state_dict", ckpt)
    print(f"   Total keys: {len(raw_state_dict)}")
    print()
    
    # Show projection_head keys in checkpoint
    print("2. Projection head keys in checkpoint:")
    proj_keys = [k for k in raw_state_dict.keys() if "projection" in k.lower()]
    for k in sorted(proj_keys):
        print(f"   {k}: {raw_state_dict[k].shape}")
    print()
    
    # Show all top-level keys (not model.lm.*)
    print("3. Top-level keys in checkpoint (not model.lm.*):")
    top_keys = [k for k in raw_state_dict.keys() if not k.startswith("model.lm.")]
    for k in sorted(top_keys):
        print(f"   {k}")
    print()
    
    # Create model
    print("4. Creating HybridTextEncoder...")
    config = HybridConfig(
        dim=512,
        num_layers=8,
        layer_pattern=["mamba", "mamba", "mlstm", "mamba", "mamba", "mlstm", "mamba", "mamba"],
        vocab_size=50257,
        max_position_embeddings=1024,
        state_size=16,
        conv_size=4,
        expand_factor=2,
        head_dim=64,
        use_tfla=True,
        proj_factor=2,
        slstm_hidden_dim=512,
        slstm_num_heads=4,
        norm_type="rms",
        use_mlp=True,
        mlp_ratio=4.0,
        dropout=0.0,
    )
    
    model = HybridTextEncoder(config, embed_dim=512)
    print(f"   Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    print()
    
    # Show projection_head keys in model
    print("5. Projection head keys in model:")
    model_proj_keys = [k for k in model.state_dict().keys() if "projection" in k.lower()]
    for k in sorted(model_proj_keys):
        print(f"   {k}: {model.state_dict()[k].shape}")
    print()
    
    # Show all top-level keys in model (not lm.*)
    print("6. Top-level keys in model (not lm.*):")
    model_top_keys = [k for k in model.state_dict().keys() if not k.startswith("lm.")]
    for k in sorted(model_top_keys):
        print(f"   {k}")
    print()
    
    # Now simulate the loading process from evaluate_sts.py
    print("7. Simulating evaluate_sts.py loading process...")
    print()
    
    # Step 1: Strip model. prefix
    print("   Step 1: Strip 'model.' prefix")
    state_dict = {}
    for k, v in raw_state_dict.items():
        if k.startswith("model."):
            new_k = k[len("model."):]
        else:
            new_k = k
        
        # Skip only logit_scale
        if new_k == "logit_scale":
            print(f"      Skipping: {new_k}")
            continue
        
        state_dict[new_k] = v
    
    print(f"   After stripping: {len(state_dict)} keys")
    print()
    
    # Show what we have after stripping
    print("   Keys after stripping (projection_head only):")
    for k in sorted(state_dict.keys()):
        if "projection" in k.lower():
            print(f"      {k}: {state_dict[k].shape}")
    print()
    
    # Step 2: Check if lm. prefix exists
    has_lm_prefix = any(k.startswith("lm.") for k in state_dict.keys())
    print(f"   Step 2: has_lm_prefix = {has_lm_prefix}")
    print()
    
    # Step 3: Add lm. prefix if needed (THIS IS THE BUG!)
    if not has_lm_prefix:
        print("   Step 3: Adding 'lm.' prefix to ALL keys")
        state_dict_with_prefix = {}
        for k, v in state_dict.items():
            state_dict_with_prefix[f"lm.{k}"] = v
        state_dict = state_dict_with_prefix
        
        print("   Keys after adding prefix (projection_head only):")
        for k in sorted(state_dict.keys()):
            if "projection" in k.lower():
                print(f"      {k}: {state_dict[k].shape}")
        print()
    else:
        print("   Step 3: NOT adding 'lm.' prefix (checkpoint already has it)")
        print()
    
    # Step 4: Try loading
    print("8. Attempting to load state_dict...")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    print(f"   Missing keys: {len(missing)}")
    if missing:
        print("   Missing keys list:")
        for k in sorted(missing):
            print(f"      {k}")
    print()
    
    print(f"   Unexpected keys: {len(unexpected)}")
    if unexpected:
        print("   Unexpected keys list:")
        for k in sorted(unexpected)[:10]:
            print(f"      {k}")
    print()
    
    # Analysis
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print()
    
    if len(missing) == 0:
        print("✅ SUCCESS: All keys loaded correctly!")
    else:
        print("❌ FAILURE: Keys are missing")
        print()
        print("Root cause:")
        
        # Check if projection_head keys are in missing
        missing_proj = [k for k in missing if "projection" in k.lower()]
        if missing_proj:
            print(f"  - {len(missing_proj)} projection_head keys are missing")
            print(f"  - This means the projection head is NOT loaded")
            print(f"  - Model will use random/untrained projection head")
            print()
            print("Why this happened:")
            print("  - Checkpoint has: model.projection_head.*")
            print("  - After stripping 'model.': projection_head.*")
            print("  - Model expects: projection_head.*")
            print("  - Should match, but doesn't!")
            print()
            print("Possible causes:")
            print("  1. The 'lm.' prefix logic is interfering")
            print("  2. The projection_head keys are being modified incorrectly")
            print("  3. There's a mismatch in how we detect/handle mixed prefixes")


if __name__ == "__main__":
    main()
