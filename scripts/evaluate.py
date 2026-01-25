"""Evaluation script for hybrid models.

Example usage:
    # Evaluate a checkpoint
    python scripts/evaluate.py checkpoint_path=outputs/experiment/checkpoints/last.ckpt
    
    # Evaluate on specific dataset
    python scripts/evaluate.py checkpoint_path=... dataset=c4
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import torch
from transformers import AutoTokenizer

from hybrid_xmamba.training.lightning_module import HybridLightningModule
from scripts.train import prepare_dataloader


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main evaluation function.
    
    Args:
        cfg: Hydra configuration
    """
    # Check if checkpoint path provided
    if not hasattr(cfg, 'checkpoint_path') or cfg.checkpoint_path is None:
        raise ValueError("Please provide checkpoint_path argument")
    
    print("=" * 80)
    print("Evaluation Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)
    
    # Set seed
    pl.seed_everything(cfg.seed, workers=True)
    
    # Load checkpoint
    print(f"Loading checkpoint from {cfg.checkpoint_path}...")
    lightning_module = HybridLightningModule.load_from_checkpoint(cfg.checkpoint_path)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.dataset.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare test data
    print("Preparing evaluation data...")
    test_dataloader = prepare_dataloader(cfg, "test", tokenizer)
    
    # Create trainer
    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=1,  # Use single device for evaluation
        precision=cfg.trainer.precision,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )
    
    # Evaluate
    print("Running evaluation...")
    results = trainer.test(lightning_module, dataloaders=test_dataloader)
    
    # Print results
    print("=" * 80)
    print("Evaluation Results:")
    for key, value in results[0].items():
        print(f"{key}: {value:.4f}")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    main()
