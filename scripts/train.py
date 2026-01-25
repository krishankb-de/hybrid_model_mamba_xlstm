"""Training script for hybrid models using PyTorch Lightning and Hydra.

Example usage:
    # Train 350M model on WikiText
    python scripts/train.py model=hybrid_350m dataset=wikitext trainer=single_gpu
    
    # Train 7B model with FSDP
    python scripts/train.py model=hybrid_7b dataset=c4 trainer=gpu_fsdp
    
    # Override specific parameters
    python scripts/train.py model=hybrid_350m learning_rate=1e-4 batch_size=16
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
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch as torch_module

from hybrid_xmamba.models.configuration_hybrid import HybridConfig
from hybrid_xmamba.models.hybrid_lm import HybridLanguageModel
from hybrid_xmamba.training.lightning_module import HybridLightningModule
from hybrid_xmamba.utils.registry import ModelRegistry


def collate_fn(batch):
    """Custom collate function for handling tokenized data.
    
    Converts lists of token IDs to tensors and stacks them.
    """
    if isinstance(batch[0], dict):
        # If batch items are already dicts (from return_tensors="pt")
        result = {}
        for key in batch[0].keys():
            # Stack tensors for this key
            if isinstance(batch[0][key], torch_module.Tensor):
                result[key] = torch_module.stack([item[key] for item in batch])
            else:
                result[key] = torch_module.tensor([item[key] for item in batch])
        return result
    else:
        # Fallback for other formats
        return torch_module.utils.data._utils.default_collate(batch)


def prepare_dataloader(cfg: DictConfig, split: str, tokenizer):
    """Prepare dataloader for training/validation.
    
    Args:
        cfg: Hydra configuration
        split: Dataset split ('train', 'validation', 'test')
        tokenizer: Tokenizer instance
        
    Returns:
        DataLoader instance
    """
    dataset_name = cfg.dataset.dataset_name
    
    # Load dataset
    if dataset_name == "wikitext":
        dataset = load_dataset(
            "wikitext",
            cfg.dataset.dataset_version,
            split=cfg.dataset.get(f"{split}_split", split),
            cache_dir=cfg.dataset.cache_dir,
        )
    elif dataset_name == "c4":
        dataset = load_dataset(
            "c4",
            "en",
            split=cfg.dataset.get(f"{split}_split", split),
            streaming=cfg.dataset.streaming,
            cache_dir=cfg.dataset.cache_dir,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=cfg.dataset.max_length,
            return_tensors="pt",
            padding="max_length",
        )
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=cfg.dataset.get("preprocessing_num_workers", 4),
    )
    
    # Create dataloader
    batch_size = cfg.dataset.batch_size if split == "train" else cfg.dataset.eval_batch_size
    
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory,
        collate_fn=collate_fn,
    )
    
    return dataloader


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function.
    
    Args:
        cfg: Hydra configuration
    """
    # Print configuration
    print("=" * 80)
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)
    
    # Set seed for reproducibility
    pl.seed_everything(cfg.seed, workers=True)
    
    # Create output directories
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.dataset.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create model configuration
    model_config = HybridConfig(
        vocab_size=cfg.model.vocab_size,
        dim=cfg.model.dim,
        num_layers=cfg.model.num_layers,
        layer_pattern=cfg.model.layer_pattern,
        state_size=cfg.model.state_size,
        conv_size=cfg.model.conv_size,
        expand_factor=cfg.model.expand_factor,
        dt_rank=cfg.model.dt_rank,
        use_fast_path=cfg.model.use_fast_path,
        head_dim=cfg.model.head_dim,
        num_heads=cfg.model.num_heads,
        use_tfla=cfg.model.use_tfla,
        proj_factor=cfg.model.proj_factor,
        slstm_hidden_dim=cfg.model.slstm_hidden_dim,
        slstm_num_heads=cfg.model.slstm_num_heads,
        use_exponential_gate=cfg.model.use_exponential_gate,
        norm_type=cfg.model.norm_type,
        use_mlp=cfg.model.use_mlp,
        mlp_ratio=cfg.model.mlp_ratio,
        max_position_embeddings=cfg.model.max_position_embeddings,
        dropout=cfg.model.dropout,
        initializer_range=cfg.model.initializer_range,
        use_cache=cfg.model.use_cache,
        tie_word_embeddings=cfg.model.tie_word_embeddings,
    )
    
    # Create model
    print(f"Creating {cfg.model.model_type} model...")
    model = HybridLanguageModel(model_config)
    
    # Print model info
    num_params = model.get_num_params(non_embedding=True)
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.1f}M)")
    print(f"Layer pattern: {model.get_layer_types()}")
    
    # Create Lightning module
    lightning_module = HybridLightningModule(
        model=model,
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        warmup_steps=cfg.model.warmup_steps,
        max_steps=cfg.model.max_steps,
        gradient_clip_val=cfg.model.gradient_clip_val,
    )
    
    # Setup callbacks
    callbacks = []
    
    # Checkpointing
    if cfg.trainer.enable_checkpointing:
        checkpoint_callback = ModelCheckpoint(
            dirpath=cfg.checkpoint_dir,
            monitor=cfg.callbacks.checkpoint.monitor,
            mode=cfg.callbacks.checkpoint.mode,
            save_top_k=cfg.callbacks.checkpoint.save_top_k,
            save_last=cfg.callbacks.checkpoint.save_last,
            filename=cfg.callbacks.checkpoint.filename,
            every_n_train_steps=cfg.callbacks.checkpoint.every_n_train_steps,
        )
        callbacks.append(checkpoint_callback)
    
    # Learning rate monitor
    if cfg.callbacks.lr_monitor.enabled:
        lr_monitor = LearningRateMonitor(
            logging_interval=cfg.callbacks.lr_monitor.logging_interval
        )
        callbacks.append(lr_monitor)
    
    # Early stopping (if enabled)
    if cfg.callbacks.early_stopping.enabled:
        early_stop = EarlyStopping(
            monitor=cfg.callbacks.early_stopping.monitor,
            patience=cfg.callbacks.early_stopping.patience,
            mode=cfg.callbacks.early_stopping.mode,
            min_delta=cfg.callbacks.early_stopping.min_delta,
        )
        callbacks.append(early_stop)
    
    # Setup loggers
    loggers = []
    
    # Weights & Biases
    if cfg.wandb.enabled:
        wandb_logger = WandbLogger(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            entity=cfg.wandb.entity,
            tags=cfg.wandb.tags,
            save_dir=cfg.log_dir,
        )
        loggers.append(wandb_logger)
    
    # TensorBoard
    tb_logger = TensorBoardLogger(
        save_dir=cfg.log_dir,
        name="tensorboard",
    )
    loggers.append(tb_logger)
    
    # Create trainer
    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        strategy=cfg.trainer.strategy,
        max_epochs=cfg.trainer.max_epochs,
        max_steps=cfg.trainer.max_steps,
        val_check_interval=cfg.trainer.val_check_interval,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        gradient_clip_val=cfg.model.gradient_clip_val,
        callbacks=callbacks,
        logger=loggers,
        enable_checkpointing=cfg.trainer.enable_checkpointing,
        enable_progress_bar=cfg.trainer.enable_progress_bar,
        enable_model_summary=cfg.trainer.enable_model_summary,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        deterministic=cfg.deterministic,
        benchmark=cfg.benchmark,
        default_root_dir=cfg.trainer.default_root_dir,
        profiler=cfg.trainer.profiler,
    )
    
    # Prepare data
    print("Preparing data...")
    train_dataloader = prepare_dataloader(cfg, "train", tokenizer)
    val_dataloader = prepare_dataloader(cfg, "validation", tokenizer)
    
    # Train
    print("Starting training...")
    trainer.fit(
        lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    
    print("Training complete!")


if __name__ == "__main__":
    main()
