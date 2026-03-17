#!/usr/bin/env python
"""Quick test to verify training configuration works."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf

print("Testing PyTorch Lightning Trainer configuration...")
print(f"PyTorch version: {torch.__version__}")
print(f"Lightning version: {pl.__version__}")

# Test 1: max_steps=-1 with max_epochs=4 (epoch-based training)
print("\n" + "="*60)
print("Test 1: Epoch-based training (max_steps=-1, max_epochs=4)")
print("="*60)
try:
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=4,
        max_steps=-1,
        enable_checkpointing=False,
        logger=False,
    )
    print("✓ SUCCESS: Trainer created with max_steps=-1, max_epochs=4")
    print(f"  max_epochs: {trainer.max_epochs}")
    print(f"  max_steps: {trainer.max_steps}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

# Test 2: max_steps=10000 with max_epochs=None (step-based training)
print("\n" + "="*60)
print("Test 2: Step-based training (max_steps=10000, max_epochs=None)")
print("="*60)
try:
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=None,
        max_steps=10000,
        enable_checkpointing=False,
        logger=False,
    )
    print("✓ SUCCESS: Trainer created with max_steps=10000, max_epochs=None")
    print(f"  max_epochs: {trainer.max_epochs}")
    print(f"  max_steps: {trainer.max_steps}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

# Test 3: Load actual config
print("\n" + "="*60)
print("Test 3: Load FineWeb training config")
print("="*60)
try:
    from hydra import compose, initialize_config_dir
    from pathlib import Path
    
    config_dir = str(Path(__file__).parent / "configs")
    
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                "model=hybrid_70m",
                "dataset=fineweb",
                "trainer=a100_single_gpu",
                "trainer.max_epochs=4",
                "trainer.max_steps=-1",
            ]
        )
    
    print("✓ SUCCESS: Config loaded")
    print(f"  trainer.max_epochs: {cfg.trainer.max_epochs}")
    print(f"  trainer.max_steps: {cfg.trainer.max_steps}")
    print(f"  dataset.batch_size: {cfg.dataset.batch_size}")
    print(f"  trainer.accumulate_grad_batches: {cfg.trainer.accumulate_grad_batches}")
    
    # Try creating trainer with loaded config
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=cfg.trainer.max_epochs,
        max_steps=cfg.trainer.max_steps,
        enable_checkpointing=False,
        logger=False,
    )
    print("✓ SUCCESS: Trainer created from Hydra config")
    
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("All tests passed! Configuration is correct.")
print("="*60)
