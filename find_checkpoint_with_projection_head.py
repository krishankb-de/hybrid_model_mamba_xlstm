#!/usr/bin/env python3
"""Find checkpoints that have projection_head keys."""

import torch
import glob
from pathlib import Path

def check_checkpoint(ckpt_path):
    """Check if checkpoint has projection_head."""
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        
        # Check for projection_head keys
        proj_keys = [k for k in state_dict.keys() if "projection_head" in k]
        has_proj = len(proj_keys) > 0
        
        # Count total keys
        total_keys = len(state_dict)
        
        return has_proj, proj_keys, total_keys
    except Exception as e:
        return None, None, None

def main():
    print("="*80)
    print("SEARCHING FOR CHECKPOINTS WITH PROJECTION_HEAD")
    print("="*80)
    
    found_checkpoints = []
    
    # Check main checkpoints
    print("\n1. Checking main checkpoints...")
    main_ckpts = glob.glob("outputs/stage1_pubmed_simcse/checkpoints/*.ckpt")
    
    for ckpt_path in sorted(main_ckpts):
        has_proj, proj_keys, total_keys = check_checkpoint(ckpt_path)
        
        if has_proj is None:
            print(f"  ❌ Error loading: {Path(ckpt_path).name}")
            continue
        
        status = "✅" if has_proj else "❌"
        print(f"  {status} {Path(ckpt_path).name:30s} - {total_keys} keys, projection_head: {has_proj}")
        
        if has_proj:
            found_checkpoints.append((ckpt_path, proj_keys, total_keys))
    
    # Check intermediate checkpoints
    print("\n2. Checking intermediate checkpoints...")
    intermediate_dirs = glob.glob("outputs/stage1_pubmed_simcse/checkpoints/contrastive-step=*/")
    
    for dir_path in sorted(intermediate_dirs)[-10:]:  # Check last 10
        ckpt_files = glob.glob(f"{dir_path}/*.ckpt")
        if not ckpt_files:
            continue
        
        ckpt_path = ckpt_files[0]
        has_proj, proj_keys, total_keys = check_checkpoint(ckpt_path)
        
        if has_proj is None:
            continue
        
        status = "✅" if has_proj else "❌"
        dir_name = Path(dir_path).name
        print(f"  {status} {dir_name:40s} - {total_keys} keys, projection_head: {has_proj}")
        
        if has_proj:
            found_checkpoints.append((ckpt_path, proj_keys, total_keys))
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if found_checkpoints:
        print(f"\n✅ FOUND {len(found_checkpoints)} checkpoint(s) with projection_head!")
        print("\nCheckpoints with projection_head:")
        for ckpt_path, proj_keys, total_keys in found_checkpoints:
            print(f"\n  📁 {ckpt_path}")
            print(f"     Total keys: {total_keys}")
            print(f"     Projection keys: {proj_keys}")
        
        print("\n" + "="*80)
        print("RECOMMENDED ACTION")
        print("="*80)
        print(f"\nUse this checkpoint for evaluation:")
        print(f"  {found_checkpoints[0][0]}")
        print("\nRun diagnostic:")
        print(f"  python diagnose_eval_issue.py --checkpoint {found_checkpoints[0][0]}")
        print("\nRun evaluation:")
        print(f"  python scripts/evaluate_sts.py --checkpoint {found_checkpoints[0][0]} \\")
        print(f"      --dataset biosses --batch-size 32 --max-length 256 \\")
        print(f"      --output-dir outputs/eval_stage1/sts")
    else:
        print("\n❌ NO CHECKPOINTS FOUND WITH PROJECTION_HEAD!")
        print("\nThis means:")
        print("  1. Training did not save the projection_head")
        print("  2. Only the LM backbone was saved")
        print("  3. Contrastive training did not work as expected")
        
        print("\n" + "="*80)
        print("POSSIBLE CAUSES")
        print("="*80)
        print("\n1. Training script saved only model.lm instead of full model")
        print("2. Checkpoint saving was interrupted")
        print("3. Wrong model class was used during training")
        
        print("\n" + "="*80)
        print("SOLUTIONS")
        print("="*80)
        print("\n1. Check training script (scripts/train_contrastive.py)")
        print("   - Verify it uses HybridTextEncoder")
        print("   - Verify checkpoint callback saves full model")
        
        print("\n2. Re-run training with proper checkpoint saving")
        
        print("\n3. Or evaluate LM backbone only (perplexity):")
        print("   python scripts/evaluate_lm.py \\")
        print("       --checkpoint outputs/stage1_pubmed_simcse/checkpoints/last.ckpt \\")
        print("       --dataset pubmed --split validation --batch-size 4 \\")
        print("       --output-dir outputs/eval_stage1/lm")
    
    return len(found_checkpoints) > 0

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
