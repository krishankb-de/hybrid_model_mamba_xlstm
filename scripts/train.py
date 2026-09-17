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
from hybrid_xmamba.training.signal_callbacks import SignalCheckpointCallback
from hybrid_xmamba.utils.registry import ModelRegistry
from hybrid_xmamba.utils.run_metadata import write_run_metadata

# ============================================================
# PERFORMANCE: Enable Tensor Core utilization on A100/H100
# This trades off a tiny bit of float32 precision for ~2x matmul speed
# ============================================================
torch.set_float32_matmul_precision('high')


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
    elif dataset_name == "fineweb":
        # Load FineWeb dataset
        dataset = load_dataset(
            "HuggingFaceFW/fineweb",
            name=cfg.dataset.dataset_version,
            split=cfg.dataset.get(f"{split}_split", split),
            streaming=cfg.dataset.streaming,
            cache_dir=cfg.dataset.cache_dir,
        )
        
        # For FineWeb, create train/val split if needed
        if split == "validation":
            # Use a small slice of training data for validation
            val_size = cfg.dataset.get("val_size", 0.01)
            val_max_samples = cfg.dataset.get("val_max_samples", 1000)
            
            # Take first N samples for validation
            if hasattr(dataset, 'take'):
                dataset = dataset.take(val_max_samples)
            else:
                dataset = dataset.select(range(min(val_max_samples, len(dataset))))
        elif split == "train":
            # Calculate how many samples we need for target tokens
            target_tokens = cfg.dataset.get("target_tokens", None)
            if target_tokens:
                seq_length = cfg.dataset.get("max_seq_length", cfg.dataset.max_length)
                # Estimate: assume average document is ~500 tokens
                # We'll pack them into seq_length chunks
                estimated_samples_needed = int(target_tokens / 500 * 1.2)  # 20% buffer
                print(f"Limiting dataset to ~{estimated_samples_needed:,} samples for {target_tokens:,} tokens")
                
                if hasattr(dataset, 'take'):
                    dataset = dataset.take(estimated_samples_needed)
                elif len(dataset) > estimated_samples_needed:
                    dataset = dataset.select(range(estimated_samples_needed))
    elif dataset_name == "pubmed":
        hf_split = "train" if split == "train" else "validation"
        dataset = load_dataset(
            "ccdv/pubmed-summarization",
            split=hf_split,
            cache_dir=cfg.dataset.cache_dir,
        )
        if split == "validation":
            val_max = cfg.dataset.get("val_max_samples", 2000)
            dataset = dataset.select(range(min(val_max, len(dataset))))
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Tokenization function - use text packing (concatenate + chunk)
    # instead of padding every sample to max_length (wastes compute on padding tokens)
    # Phase 6A: append eos_token_id after each doc so group_texts can emit cu_seqlens.
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id

    def tokenize_function(examples):
        # Handle different text field names across datasets
        if "text" in examples:
            text_field = "text"
        elif "article" in examples:
            text_field = "article"
        else:
            text_field = next(k for k in examples.keys() if isinstance(examples[k][0], str))

        # Tokenize without padding - we'll pack sequences next
        tokenized = tokenizer(
            examples[text_field],
            truncation=False,
            return_attention_mask=False,
        )
        if eos_id is not None:
            tokenized["input_ids"] = [ids + [eos_id] for ids in tokenized["input_ids"]]
        return tokenized

    # Group texts into chunks for efficient training (text packing)
    # This eliminates wasted compute on padding tokens
    # Respect max_seq_length override (e.g. +dataset.max_seq_length=128)
    # falling back to max_length from the dataset config
    seq_length = cfg.dataset.get("max_seq_length", cfg.dataset.max_length)
    print(f"Using sequence length: {seq_length} (max_length={cfg.dataset.max_length})")

    def group_texts(examples):
        # Concatenate all texts
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        # Drop the remainder that doesn't fill a complete chunk
        total_length = (total_length // seq_length) * seq_length
        # Split into chunks of seq_length
        result = {
            k: [t[i : i + seq_length] for i in range(0, total_length, seq_length)]
            for k, t in concatenated.items()
        }
        # Phase 6A: cu_seqlens — per-position doc-id within each chunk.
        # Run-length encoded by EOS tokens; consumed by mixer per-segment wrappers.
        if eos_id is not None and total_length > 0:
            ids_flat = concatenated["input_ids"][:total_length]
            doc_id, doc_ids_flat = 0, []
            for tok in ids_flat:
                doc_ids_flat.append(doc_id)
                if tok == eos_id:
                    doc_id += 1
            result["cu_seqlens"] = [
                doc_ids_flat[i : i + seq_length] for i in range(0, total_length, seq_length)
            ]
        return result
    
    # Tokenize dataset
    # For streaming datasets, don't use num_proc
    map_kwargs = {
        "batched": True,
        "remove_columns": dataset.column_names if hasattr(dataset, 'column_names') else None,
    }
    if not cfg.dataset.streaming:
        map_kwargs["num_proc"] = cfg.dataset.get("preprocessing_num_workers", 4)
    
    tokenized_dataset = dataset.map(tokenize_function, **map_kwargs)
    
    # Pack sequences into fixed-length chunks (no padding waste)
    pack_kwargs = {"batched": True}
    if not cfg.dataset.streaming:
        pack_kwargs["num_proc"] = cfg.dataset.get("preprocessing_num_workers", 4)
    
    tokenized_dataset = tokenized_dataset.map(group_texts, **pack_kwargs)
    
    # Create dataloader
    batch_size = cfg.dataset.batch_size if split == "train" else cfg.dataset.eval_batch_size
    
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory,
        collate_fn=collate_fn,
        persistent_workers=True if cfg.dataset.num_workers > 0 else False,
        prefetch_factor=2 if cfg.dataset.num_workers > 0 else None,
        drop_last=(split == "train"),  # Avoid small trailing batches
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

    # Reproducibility snapshot (git SHA + resolved cfg + argv)
    write_run_metadata(cfg, cfg.output_dir, extra={"entrypoint": "train.py"})
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.dataset.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # pooling indexes last non-pad via attention_mask
    # Override tokenizer's default model_max_length (GPT-2 = 1024) to match
    # our actual model capacity. This suppresses the spurious warning:
    #   "Token indices sequence length is longer than the specified maximum
    #    sequence length for this model (N > 1024)"
    # We tokenize with truncation=False for text packing, so this limit
    # only controls the warning threshold, not actual truncation.
    tokenizer.model_max_length = cfg.model.max_position_embeddings
    
    # Create model configuration
    # MAMBA3_PLAN.md M2-F: every dataclass field present in the yaml is carried
    # through automatically. Do not go back to listing fields by hand -- that is
    # how norm_topology (Phase 9) and scan_impl (job 2513007) were silently lost.
    model_config = HybridConfig.from_hydra(cfg.model)
    
    # Create model
    print(f"Creating {cfg.model.model_type} model...")
    model = HybridLanguageModel(model_config)
    
    # Print model info
    num_params = model.get_num_params(non_embedding=True)
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.1f}M)")
    print(f"Layer pattern: {model.get_layer_types()}")
    
    # Determine precision for logging
    precision = cfg.trainer.precision
    if 'bf16' in str(precision) or '16' in str(precision):
        print(f"Using {precision} Automatic Mixed Precision (AMP)")
    
    # Create Lightning module with torch.compile — only on A100+ (sm_80+)
    # torch.compile on T4/V100 (sm_75) causes 10-30 min compilation on first step
    # and max_autotune_gemm fails due to insufficient SMs
    use_compile = False
    if torch.cuda.is_available() and hasattr(torch, 'compile'):
        # Check config flag first (allows explicit override)
        compile_flag = cfg.trainer.get("compile_model", None)
        if compile_flag is not None:
            use_compile = bool(compile_flag)
        else:
            # Auto-detect: only enable on Ampere+ GPUs (sm_80+)
            capability = torch.cuda.get_device_capability()
            use_compile = capability[0] >= 8  # A100=8.0, H100=9.0, T4=7.5
            if not use_compile:
                gpu_name = torch.cuda.get_device_name()
                print(f"Skipping torch.compile — {gpu_name} (sm_{capability[0]}{capability[1]}) "
                      f"is below sm_80. Use trainer.compile_model=true to force.")
    if use_compile:
        print("Enabling torch.compile for optimized GPU kernels (Ampere+ detected)...")
    
    # Phase 7 WSD wiring: pull optional scheduler knobs from cfg.model (cosine default).
    _mcfg = cfg.model
    scheduler_name = _mcfg.get("scheduler_name", "cosine") if hasattr(_mcfg, "get") else getattr(_mcfg, "scheduler_name", "cosine")
    beta2_schedule = _mcfg.get("beta2_schedule", False) if hasattr(_mcfg, "get") else getattr(_mcfg, "beta2_schedule", False)
    beta2_start = _mcfg.get("beta2_start", 0.999) if hasattr(_mcfg, "get") else getattr(_mcfg, "beta2_start", 0.999)
    beta2_end = _mcfg.get("beta2_end", 0.974) if hasattr(_mcfg, "get") else getattr(_mcfg, "beta2_end", 0.974)

    lightning_module = HybridLightningModule(
        model=model,
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        warmup_steps=cfg.model.warmup_steps,
        max_steps=cfg.model.max_steps,
        gradient_clip_val=cfg.model.gradient_clip_val,
        compile_model=use_compile,
        scheduler_name=scheduler_name,
        beta2_schedule=beta2_schedule,
        beta2_start=beta2_start,
        beta2_end=beta2_end,
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
    
    # SLURM / SIGTERM-safe checkpoint (Willi 36-48h walltime)
    callbacks.append(SignalCheckpointCallback(checkpoint_dir=cfg.checkpoint_dir))

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
    # PyTorch Lightning expects max_steps to be -1 (unlimited) or a positive integer
    # When max_steps=-1, training is controlled by max_epochs
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
        # gradient_clip_val intentionally omitted: clipping is done in
        # HybridLightningModule.on_before_optimizer_step to avoid conflict
        # with foreach AdamW under bf16-mixed AMP.
        callbacks=callbacks,
        logger=loggers,
        enable_checkpointing=cfg.trainer.enable_checkpointing,
        enable_progress_bar=cfg.trainer.enable_progress_bar,
        enable_model_summary=cfg.trainer.enable_model_summary,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        limit_train_batches=cfg.trainer.get("limit_train_batches", 1.0),
        limit_val_batches=cfg.trainer.get("limit_val_batches", 1.0),
        deterministic=cfg.deterministic,
        benchmark=cfg.benchmark,
        default_root_dir=cfg.trainer.default_root_dir,
        profiler=cfg.trainer.profiler,
    )
    
    # Prepare data
    print("Preparing data...")
    train_dataloader = prepare_dataloader(cfg, "train", tokenizer)
    val_dataloader = prepare_dataloader(cfg, "validation", tokenizer)
    
    # Print training info
    print("\n" + "="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)
    print(f"Dataset: {cfg.dataset.dataset_name}")
    if cfg.dataset.dataset_name == "fineweb":
        target_tokens = cfg.dataset.get("target_tokens", "N/A")
        print(f"Target tokens: {target_tokens:,}" if isinstance(target_tokens, int) else f"Target tokens: {target_tokens}")
    print(f"Batch size: {cfg.dataset.batch_size}")
    print(f"Gradient accumulation: {cfg.trainer.accumulate_grad_batches}")
    print(f"Effective batch size: {cfg.dataset.batch_size * cfg.trainer.accumulate_grad_batches}")
    print(f"Sequence length: {cfg.dataset.max_length}")
    print(f"Tokens per batch: {cfg.dataset.batch_size * cfg.trainer.accumulate_grad_batches * cfg.dataset.max_length:,}")
    
    # Determine training mode
    if cfg.trainer.max_steps > 0:
        print(f"Training mode: Step-based")
        print(f"Max steps: {cfg.trainer.max_steps:,}")
        total_tokens = cfg.trainer.max_steps * cfg.dataset.batch_size * cfg.trainer.accumulate_grad_batches * cfg.dataset.max_length
        print(f"Total tokens: {total_tokens:,} ({total_tokens/1e9:.2f}B)")
    elif cfg.trainer.max_epochs > 0:
        print(f"Training mode: Epoch-based")
        print(f"Max epochs: {cfg.trainer.max_epochs}")
        try:
            num_batches = len(train_dataloader)
            total_steps = num_batches * cfg.trainer.max_epochs
            print(f"Steps per epoch: {num_batches:,}")
            print(f"Total training steps: {total_steps:,}")
            total_tokens = total_steps * cfg.dataset.batch_size * cfg.trainer.accumulate_grad_batches * cfg.dataset.max_length
            print(f"Total tokens (all epochs): {total_tokens:,} ({total_tokens/1e9:.2f}B)")
        except:
            print("Steps per epoch: (calculated at runtime)")
    print("="*80 + "\n")
    
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
