"""Contrastive fine-tuning script for HybridTextEncoder.

Supports two modes controlled by dataset.contrastive_mode:

  simcse  — Stage 1: self-supervised text-only training on PubMed.
             No images needed.  Two dropout views of each text are
             contrasted against in-batch negatives.

  clip    — Stage 2: supervised image-text alignment on Indiana CXR / ROCO.
             Frozen BiomedCLIP image encoder provides image embeddings;
             only the HybridTextEncoder is trained.

Example commands
----------------
# Stage 1 — SimCSE on PubMed (text encoder only)
python scripts/train_contrastive.py \\
    --config-name config_70m \\
    dataset=pubmed \\
    trainer=a100_single_gpu \\
    trainer.max_steps=10000 \\
    contrastive_mode=simcse

# Stage 2 — CLIP alignment on Indiana CXR
python scripts/train_contrastive.py \\
    --config-name config_70m \\
    dataset=indiana_cxr \\
    trainer=a100_single_gpu \\
    trainer.max_steps=5000 \\
    contrastive_mode=clip \\
    lm_checkpoint=./outputs/pubmed_simcse/checkpoints/last.ckpt
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from PIL import Image
import torchvision.transforms as T

from hybrid_xmamba.models.configuration_hybrid import HybridConfig
from hybrid_xmamba.models.hybrid_lm import HybridTextEncoder
from hybrid_xmamba.training.lightning_module import HybridContrastiveLightningModule

torch.set_float32_matmul_precision("high")


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

class TextOnlyDataset(Dataset):
    """Token-packed text dataset for SimCSE (no images).

    Tokenises and packs texts into fixed-length chunks exactly like the
    main train.py script, so the same cache can be reused.
    """

    def __init__(self, hf_dataset, tokenizer, max_length: int):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get("text") or item.get("abstract") or ""
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {"input_ids": enc["input_ids"].squeeze(0)}


class ImageTextDataset(Dataset):
    """Paired image-text dataset for CLIP-style alignment.

    Expects a HuggingFace dataset with fields:
      - image (PIL or path)
      - findings / impression (text)
    """

    def __init__(self, hf_dataset, tokenizer, cfg):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = cfg.dataset.max_length
        self.concat = cfg.dataset.get("concatenate_sections", True)
        self.findings_field = cfg.dataset.get("findings_field", "findings")
        self.impression_field = cfg.dataset.get("impression_field", "impression")

        mean = cfg.dataset.get("image_mean", [0.48145466, 0.4578275, 0.40821073])
        std  = cfg.dataset.get("image_std",  [0.26862954, 0.26130258, 0.27577711])
        size = cfg.dataset.get("image_size", 224)

        self.img_transform = T.Compose([
            T.Resize((size, size)),
            T.Grayscale(num_output_channels=3),   # CXR is grayscale; BiomedCLIP expects 3-ch
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # --- text ---
        findings  = item.get(self.findings_field, "") or ""
        impression = item.get(self.impression_field, "") or ""
        if self.concat:
            text = f"Findings: {findings} Impression: {impression}".strip()
        else:
            text = findings or impression

        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)

        # --- image ---
        img = item.get("image")
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif not isinstance(img, Image.Image):
            img = Image.fromarray(img).convert("RGB")
        else:
            img = img.convert("RGB")
        pixel_values = self.img_transform(img)

        return {"input_ids": input_ids, "pixel_values": pixel_values}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pubmed(cfg, split: str, tokenizer):
    """Load PubMed abstracts from HuggingFace."""
    # "pubmed" on HF is available via the "ncbi/pubmed" dataset ID
    ds = load_dataset(
        "ncbi/pubmed",
        split="train",
        streaming=cfg.dataset.streaming,
        cache_dir=cfg.dataset.cache_dir,
        trust_remote_code=True,
    )
    # Carve out validation from the tail of the training stream
    val_n = cfg.dataset.get("val_max_samples", 2000)
    if split == "validation":
        if hasattr(ds, "take"):
            ds = ds.skip(0).take(val_n)   # first N as val proxy
        else:
            ds = ds.select(range(val_n))
    else:
        if not cfg.dataset.streaming:
            target_tokens = cfg.dataset.get("target_tokens", 500_000_000)
            n = int(target_tokens / 200)   # rough estimate: ~200 tokens/abstract
            if len(ds) > n:
                ds = ds.select(range(val_n, val_n + n))

    return TextOnlyDataset(ds, tokenizer, cfg.dataset.max_length)


def load_indiana_cxr(cfg, split: str, tokenizer):
    """Load Indiana University CXR from HuggingFace."""
    ds = load_dataset(
        "Corran/IU-Xray",
        split=split,
        cache_dir=cfg.dataset.cache_dir,
        trust_remote_code=True,
    )
    return ImageTextDataset(ds, tokenizer, cfg)


def prepare_dataloader(cfg, split: str, tokenizer):
    name = cfg.dataset.dataset_name
    if name == "pubmed":
        dataset = load_pubmed(cfg, split, tokenizer)
    elif name == "indiana_cxr":
        dataset = load_indiana_cxr(cfg, split, tokenizer)
    else:
        raise ValueError(f"Unknown dataset for contrastive training: {name}")

    batch_size = cfg.dataset.batch_size if split == "train" else cfg.dataset.eval_batch_size
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory,
        drop_last=(split == "train"),
        persistent_workers=(cfg.dataset.num_workers > 0),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print("=" * 80)
    print("Contrastive Training Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    pl.seed_everything(cfg.seed, workers=True)
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.dataset.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = cfg.model.max_position_embeddings

    # Model config
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

    # Build text encoder
    text_encoder = HybridTextEncoder(model_config, embed_dim=512)

    # Optionally load weights from a prior LM checkpoint
    lm_ckpt = cfg.get("lm_checkpoint", None)
    if lm_ckpt:
        print(f"Loading LM backbone weights from: {lm_ckpt}")
        ckpt = torch.load(lm_ckpt, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        # Strip Lightning "model." prefix if present
        state = {k.replace("model.", "", 1): v for k, v in state.items()}
        missing, unexpected = text_encoder.lm.load_state_dict(state, strict=False)
        print(f"  Loaded. Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")

    num_params = sum(p.numel() for p in text_encoder.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,} ({num_params/1e6:.1f}M)")

    # Contrastive mode — prefer dataset config, fall back to CLI override
    contrastive_mode = cfg.dataset.get("contrastive_mode", cfg.get("contrastive_mode", "simcse"))
    image_encoder_name = cfg.dataset.get(
        "image_encoder_name",
        "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    )

    lightning_module = HybridContrastiveLightningModule(
        model=text_encoder,
        contrastive_mode=contrastive_mode,
        image_encoder_name=image_encoder_name,
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        warmup_steps=cfg.model.warmup_steps,
        max_steps=cfg.trainer.max_steps,
        gradient_clip_val=cfg.model.gradient_clip_val,
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.checkpoint_dir,
            monitor="val/contrastive_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            filename="contrastive-{step:06d}-{val/contrastive_loss:.4f}",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Loggers
    loggers = [TensorBoardLogger(save_dir=cfg.log_dir, name="tensorboard")]
    if cfg.wandb.enabled:
        loggers.append(WandbLogger(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            entity=cfg.wandb.entity,
            save_dir=cfg.log_dir,
        ))

    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        strategy=cfg.trainer.strategy,
        max_steps=cfg.trainer.max_steps,
        val_check_interval=cfg.trainer.val_check_interval,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        gradient_clip_val=cfg.model.gradient_clip_val,
        callbacks=callbacks,
        logger=loggers,
        enable_checkpointing=True,
        enable_progress_bar=True,
        num_sanity_val_steps=2,
        default_root_dir=cfg.trainer.default_root_dir,
    )

    print("Preparing dataloaders...")
    train_dl = prepare_dataloader(cfg, "train", tokenizer)
    val_dl   = prepare_dataloader(cfg, "validation", tokenizer)

    print(f"Contrastive mode : {contrastive_mode}")
    print(f"Dataset          : {cfg.dataset.dataset_name}")
    print(f"Max steps        : {cfg.trainer.max_steps:,}")
    print("Starting contrastive training...")

    trainer.fit(lightning_module, train_dataloaders=train_dl, val_dataloaders=val_dl)
    print("Done.")


if __name__ == "__main__":
    main()
