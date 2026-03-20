"""PyTorch Lightning module for training hybrid models.

Provides a Lightning wrapper for easy distributed training with minimal boilerplate.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Optional, Dict, Any
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from hybrid_xmamba.models.hybrid_lm import HybridLanguageModel
from hybrid_xmamba.training.optimizer import configure_optimizer
from hybrid_xmamba.training.metrics import compute_perplexity


class HybridLightningModule(pl.LightningModule):
    """PyTorch Lightning module for hybrid models.
    
    Wraps the hybrid model with training logic, optimizer configuration,
    and metric tracking.
    
    Args:
        model: The hybrid model to train
        learning_rate: Initial learning rate
        weight_decay: Weight decay for AdamW
        warmup_steps: Number of warmup steps
        max_steps: Maximum training steps
        optimizer_name: Name of optimizer ('adamw', 'adam', 'sgd')
        scheduler_name: Name of scheduler ('cosine', 'linear', 'constant')
        gradient_clip_val: Gradient clipping value
        compile_model: Whether to compile model with torch.compile
    """
    
    def __init__(
        self,
        model: HybridLanguageModel,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        warmup_steps: int = 2000,
        max_steps: int = 100000,
        optimizer_name: str = "adamw",
        scheduler_name: str = "cosine",
        gradient_clip_val: float = 1.0,
        compile_model: bool = False,
    ):
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['model'])
        
        # Model
        self.model = model
        
        # Compile model if requested (PyTorch 2.0+)
        if compile_model:
            self.model = torch.compile(self.model)
        
        # Training parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.optimizer_name = optimizer_name
        self.scheduler_name = scheduler_name
        self.gradient_clip_val = gradient_clip_val
    
    def forward(self, input_ids: torch.Tensor, **kwargs):
        """Forward pass."""
        return self.model(input_ids, **kwargs)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step.
        
        Args:
            batch: Dictionary with 'input_ids' and 'labels'
            batch_idx: Batch index
            
        Returns:
            Loss tensor
        """
        input_ids = batch['input_ids']
        labels = batch.get('labels', input_ids)
        
        # Forward pass
        outputs = self.model(input_ids, labels=labels, return_dict=True)
        loss = outputs.loss
        
        # Compute perplexity
        perplexity = compute_perplexity(loss)
        
        # Log metrics
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/perplexity', perplexity, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step.
        
        Args:
            batch: Dictionary with 'input_ids' and 'labels'
            batch_idx: Batch index
            
        Returns:
            Loss tensor
        """
        input_ids = batch['input_ids']
        labels = batch.get('labels', input_ids)
        
        # Forward pass
        outputs = self.model(input_ids, labels=labels, return_dict=True)
        loss = outputs.loss
        
        # Compute perplexity
        perplexity = compute_perplexity(loss)
        
        # Log metrics
        self.log('val/loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/perplexity', perplexity, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step.
        
        Args:
            batch: Dictionary with 'input_ids' and 'labels'
            batch_idx: Batch index
            
        Returns:
            Loss tensor
        """
        input_ids = batch['input_ids']
        labels = batch.get('labels', input_ids)
        
        # Forward pass
        outputs = self.model(input_ids, labels=labels, return_dict=True)
        loss = outputs.loss
        
        # Compute perplexity
        perplexity = compute_perplexity(loss)
        
        # Log metrics
        self.log('test/loss', loss, on_step=False, on_epoch=True)
        self.log('test/perplexity', perplexity, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Configure optimizer
        optimizer = configure_optimizer(
            self.model,
            optimizer_name=self.optimizer_name,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        # Configure scheduler
        if self.scheduler_name == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.max_steps - self.warmup_steps,
                eta_min=self.learning_rate * 0.1,
            )
        elif self.scheduler_name == "linear":
            from torch.optim.lr_scheduler import LinearLR
            scheduler = LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=self.max_steps - self.warmup_steps,
            )
        else:  # constant
            from torch.optim.lr_scheduler import ConstantLR
            scheduler = ConstantLR(optimizer, factor=1.0)
        
        # Add warmup
        if self.warmup_steps > 0:
            from torch.optim.lr_scheduler import LinearLR, SequentialLR
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=self.warmup_steps,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, scheduler],
                milestones=[self.warmup_steps],
            )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
    
    def on_before_optimizer_step(self, optimizer):
        """Hook called before optimizer step (for gradient norm logging only).
        
        Note: Actual gradient clipping is handled by Lightning Trainer's
        gradient_clip_val parameter. We only log the norm here.
        """
        # Log gradient norm for monitoring (without clipping again)
        if self.gradient_clip_val > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters(), 
                max_norm=float('inf'),  # Don't clip, just compute norm
            )
            self.log('train/grad_norm', grad_norm, on_step=True)


class HybridContrastiveLightningModule(HybridLightningModule):
    """Lightning module for contrastive (SimCSE / CLIP-style) fine-tuning.

    Supports two training modes selected by ``contrastive_mode``:

    * ``"simcse"``  — self-supervised: two dropout-augmented views of the
      same text are pulled together; in-batch negatives are pushed apart.
      No image encoder needed.  Use this for Stage 1 (text-only).

    * ``"clip"``    — supervised: text embeddings are aligned to frozen
      image embeddings produced by a pretrained BiomedCLIP / BioViL
      image encoder.  Use this for Stage 2 (image-text alignment).

    The contrastive loss is symmetric NT-Xent (InfoNCE) with a learnable
    temperature, identical to the original CLIP formulation.

    Args:
        model:              HybridTextEncoder instance.
        contrastive_mode:   ``"simcse"`` or ``"clip"``.
        image_encoder_name: HuggingFace model ID for the image encoder
                            (only used when contrastive_mode=="clip").
                            Defaults to BiomedCLIP.
        image_embed_dim:    Embedding dimension of the image encoder output.
        learning_rate:      Initial learning rate.
        weight_decay:       AdamW weight decay.
        warmup_steps:       LR warmup steps.
        max_steps:          Total training steps.
        gradient_clip_val:  Gradient clipping norm.
    """

    def __init__(
        self,
        model,                          # HybridTextEncoder
        contrastive_mode: str = "simcse",
        image_encoder_name: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        max_steps: int = 10000,
        gradient_clip_val: float = 1.0,
    ):
        # Pass a dummy HybridLanguageModel so the parent __init__ is happy;
        # we override forward / training_step completely below.
        super().__init__(
            model=model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            gradient_clip_val=gradient_clip_val,
        )
        self.contrastive_mode = contrastive_mode.lower()
        self.image_encoder = None

        if self.contrastive_mode == "clip":
            # open_clip_torch is an optional dependency (pip install open-clip-torch)
            try:
                from open_clip import create_model_from_pretrained  # type: ignore[import-untyped]
                self.image_encoder, _ = create_model_from_pretrained(image_encoder_name)
                self.image_encoder.eval()
                # Freeze image encoder completely
                for p in self.image_encoder.parameters():
                    p.requires_grad = False
                # Linear bridge if dimensions differ
                img_out = self.image_encoder.visual.output_dim
                txt_out = model.embed_dim
                self.img_proj = (
                    torch.nn.Linear(img_out, txt_out, bias=False)
                    if img_out != txt_out else torch.nn.Identity()
                )
            except ImportError:
                raise ImportError(
                    "open_clip_torch is required for clip mode. "
                    "Install with: pip install open-clip-torch"
                )

    # ------------------------------------------------------------------
    # Contrastive loss (symmetric NT-Xent / InfoNCE)
    # ------------------------------------------------------------------
    def _nt_xent_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        logit_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Symmetric InfoNCE loss.

        Args:
            z1: Normalised embeddings from view-1 (B, D)
            z2: Normalised embeddings from view-2 (B, D)
            logit_scale: Learnable temperature (scalar)

        Returns:
            Scalar loss
        """
        scale = logit_scale.exp().clamp(max=100.0)
        logits = scale * z1 @ z2.T           # (B, B)
        labels = torch.arange(len(z1), device=z1.device)
        loss_12 = torch.nn.functional.cross_entropy(logits, labels)
        loss_21 = torch.nn.functional.cross_entropy(logits.T, labels)
        return (loss_12 + loss_21) / 2.0

    # ------------------------------------------------------------------
    # Training / validation steps
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        if self.contrastive_mode == "simcse":
            return self._simcse_step(batch, batch_idx, split="train")
        else:
            return self._clip_step(batch, batch_idx, split="train")

    def validation_step(self, batch, batch_idx):
        if self.contrastive_mode == "simcse":
            return self._simcse_step(batch, batch_idx, split="val")
        else:
            return self._clip_step(batch, batch_idx, split="val")

    def _simcse_step(self, batch, batch_idx, split: str):
        """SimCSE step: same input_ids passed twice through the encoder.

        Different dropout masks produce two views → contrastive loss.
        Batch must contain ``input_ids`` (B, L).
        """
        input_ids = batch["input_ids"]
        self.model.lm.train()           # keep dropout active for both views
        z1 = self.model.encode(input_ids)
        z2 = self.model.encode(input_ids)
        loss = self._nt_xent_loss(z1, z2, self.model.logit_scale)
        self.log(f"{split}/contrastive_loss", loss, prog_bar=True,
                 on_step=(split == "train"), on_epoch=True)
        return loss

    def _clip_step(self, batch, batch_idx, split: str):
        """CLIP alignment step.

        Batch must contain:
          - ``input_ids``   (B, L)   — tokenised report
          - ``pixel_values`` (B, C, H, W) — preprocessed image
        """
        input_ids = batch["input_ids"]
        pixel_values = batch["pixel_values"]

        # Text embeddings (trained)
        z_text = self.model.encode(input_ids)

        # Image embeddings (frozen)
        with torch.no_grad():
            z_img_raw = self.image_encoder.encode_image(pixel_values)
        z_img = torch.nn.functional.normalize(
            self.img_proj(z_img_raw.float()), dim=-1
        )

        loss = self._nt_xent_loss(z_text, z_img, self.model.logit_scale)
        self.log(f"{split}/contrastive_loss", loss, prog_bar=True,
                 on_step=(split == "train"), on_epoch=True)
        return loss

    # ------------------------------------------------------------------
    # Retrieval metrics (R@1, R@5, R@10) logged at validation end
    # ------------------------------------------------------------------
    def on_validation_epoch_end(self):
        pass  # Hook for future retrieval evaluation — extend as needed


class MQARLightningModule(HybridLightningModule):
    """Lightning module specialized for MQAR (Multi-Query Associative Recall) task.
    
    MQAR is a benchmark task for testing long-range memory capabilities.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step with MQAR-specific metrics."""
        loss = super().training_step(batch, batch_idx)
        
        # Additional MQAR-specific logging could go here
        # e.g., accuracy on query tokens
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step with MQAR accuracy computation."""
        from hybrid_xmamba.training.metrics import compute_mqar_accuracy
        
        input_ids = batch['input_ids']
        labels = batch.get('labels', input_ids)
        
        # Forward pass
        outputs = self.model(input_ids, labels=labels, return_dict=True)
        loss = outputs.loss
        
        # Compute MQAR accuracy
        accuracy = compute_mqar_accuracy(outputs.logits, labels, batch.get('query_positions'))
        
        # Log metrics
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/mqar_accuracy', accuracy, prog_bar=True)
        
        return loss
