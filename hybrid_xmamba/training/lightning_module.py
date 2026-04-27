"""PyTorch Lightning module for training hybrid models.

Provides a Lightning wrapper for easy distributed training with minimal boilerplate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        """Clip gradients and log norm before optimizer step.

        Clipping is done here (not via Trainer gradient_clip_val) because
        foreach=True AdamW is compatible with manual clipping but the
        fused variant conflicts with Lightning's external AMP unscaling.
        """
        if self.gradient_clip_val > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters(),
                max_norm=self.gradient_clip_val,
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
            # Load BiomedCLIP image encoder from HuggingFace
            try:
                import open_clip
                # BiomedCLIP uses a specific loading method
                # Try loading from HuggingFace hub directly
                model_name = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
                
                print(f"Loading BiomedCLIP image encoder from: {model_name}")
                
                # Load the full BiomedCLIP model
                self.image_encoder, _ = open_clip.create_model_from_pretrained(
                    'hf-hub:' + model_name
                )
                
                # Extract just the visual encoder
                self.image_encoder = self.image_encoder.visual
                self.image_encoder.eval()
                
                # Freeze image encoder completely
                for p in self.image_encoder.parameters():
                    p.requires_grad = False
                
                # Linear bridge if dimensions differ
                img_out = self.image_encoder.output_dim
                txt_out = model.embed_dim
                self.img_proj = (
                    torch.nn.Linear(img_out, txt_out, bias=False)
                    if img_out != txt_out else torch.nn.Identity()
                )
                
                print(f"✓ BiomedCLIP image encoder loaded successfully")
                print(f"  Image output dim: {img_out}")
                print(f"  Text output dim: {txt_out}")
                
            except Exception as e:
                print(f"Error loading BiomedCLIP: {e}")
                print("Trying alternative loading method...")
                
                # Fallback: Use a standard CLIP model
                try:
                    self.image_encoder, _ = open_clip.create_model_from_pretrained(
                        'hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K'
                    )
                    self.image_encoder = self.image_encoder.visual
                    self.image_encoder.eval()
                    for p in self.image_encoder.parameters():
                        p.requires_grad = False
                    
                    img_out = self.image_encoder.output_dim
                    txt_out = model.embed_dim
                    self.img_proj = (
                        torch.nn.Linear(img_out, txt_out, bias=False)
                        if img_out != txt_out else torch.nn.Identity()
                    )
                    print(f"✓ Fallback CLIP model loaded successfully")
                except Exception as e2:
                    raise ImportError(
                        f"Failed to load image encoder. Original error: {e}\n"
                        f"Fallback error: {e2}\n"
                        "Please ensure open-clip-torch is installed: pip install open-clip-torch"
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
        """Symmetric InfoNCE loss with numerical stability.

        Args:
            z1: Normalised embeddings from view-1 (B, D)
            z2: Normalised embeddings from view-2 (B, D)
            logit_scale: Learnable temperature (scalar)

        Returns:
            Scalar loss
        """
        # Clamp logit_scale: exp(2.6592)≈14.3 (CLIP init), cap at 100 to prevent overflow
        scale = logit_scale.exp().clamp(min=1.0, max=100.0)

        # Similarity matrix (B, B); z1 and z2 are already L2-normalised by encode()
        logits = scale * (z1 @ z2.T)

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

        Two dropout masks produce two views → contrastive loss.
        Batch must contain ``input_ids`` (B, L) and ``attention_mask`` (B, L).
        """
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")

        # Keep dropout active in both backbone and projection head for both
        # views — the dropout in projection_head is the stochastic augmentation
        # that SimCSE relies on to create distinct z1 / z2 views.  Without this,
        # Lightning's validation eval() call leaves projection_head in eval mode,
        # dropout is off, z1 == z2 exactly, and InfoNCE collapses to ~0,
        # making val/contrastive_loss useless for ModelCheckpoint.
        self.model.lm.train()
        self.model.projection_head.train()
        z1 = self.model.encode(input_ids, attention_mask=attention_mask)
        z2 = self.model.encode(input_ids, attention_mask=attention_mask)

        loss = self._nt_xent_loss(z1, z2, self.model.logit_scale)
        self.log(f"{split}/contrastive_loss", loss, prog_bar=True,
                 on_step=(split == "train"), on_epoch=True)
        if split == "train":
            pos_cos = (z1 * z2).sum(dim=-1).mean()
            self.log("train/pos_cosine_mean", pos_cos,
                     on_step=True, on_epoch=False, prog_bar=False)
        return loss

    def _clip_step(self, batch, batch_idx, split: str):
        """CLIP alignment step.

        Batch must contain:
          - ``input_ids``   (B, L)   — tokenised report
          - ``attention_mask`` (B, L)
          - ``pixel_values`` (B, C, H, W) — preprocessed image
        """
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        pixel_values = batch["pixel_values"]

        # Text embeddings (trained)
        z_text = self.model.encode(input_ids, attention_mask=attention_mask)

        # Image embeddings (frozen)
        with torch.no_grad():
            # The image encoder is already just the visual part
            z_img_raw = self.image_encoder(pixel_values)
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


class DistillContrastiveLightningModule(HybridContrastiveLightningModule):
    """HybridContrastiveLightningModule extended with PubMedBERT embedding KD.

    During SimCSE Stage 1, a frozen PubMedBERT teacher provides CLS embeddings.
    A cosine distillation loss is added alongside the SimCSE InfoNCE loss.

    Loss:
        L = L_SimCSE + lambda(step) * (1 - cos(student_z1, teacher_cls))

    lambda ramps from 0 to lambda_max over [warmup_steps, warmup_steps + ramp_steps]
    to prevent early collapse onto the teacher manifold.

    The teacher uses WordPiece tokenization (PubMedBERT) while the student uses
    GPT-2 BPE — dual tokenization is handled in the dataloader. Both see the same
    raw text; only pooled CLS vs. pooled student embeddings are compared.

    Args:
        teacher:            Frozen AutoModel (PubMedBERT encoder).
        lambda_max:         Maximum distillation weight after ramp-up.
        distill_warmup:     Steps with lambda=0 (pure SimCSE) before ramp starts.
        distill_ramp:       Steps over which lambda linearly ramps 0 → lambda_max.
        **kwargs:           Passed to HybridContrastiveLightningModule.
    """

    def __init__(
        self,
        teacher: torch.nn.Module,
        lambda_max: float = 0.3,
        distill_warmup: int = 500,
        distill_ramp: int = 500,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Exclude teacher from Lightning's hyperparameter pickle — it is a
        # large frozen nn.Module and some environments inject non-picklable
        # forward hooks that crash torch.save / checkpoint writing.
        self.save_hyperparameters(ignore=['teacher', 'model'])
        self.teacher = teacher
        self.lambda_max = lambda_max
        self.distill_warmup = distill_warmup
        self.distill_ramp = distill_ramp
        # Project student embeddings into teacher's dim before cosine distillation.
        # student: embed_dim=512 (hybrid_70m), teacher: hidden_size=768 (PubMedBERT)
        student_dim = self.model.embed_dim
        teacher_dim = teacher.config.hidden_size
        self.distill_proj = nn.Linear(student_dim, teacher_dim, bias=False)

    def _get_distill_lambda(self) -> float:
        """Compute current distillation weight based on global step."""
        step = self.global_step
        if step < self.distill_warmup:
            return 0.0
        ramp_step = step - self.distill_warmup
        if ramp_step >= self.distill_ramp:
            return self.lambda_max
        return self.lambda_max * (ramp_step / self.distill_ramp)

    def _simcse_step(self, batch, batch_idx, split: str):
        """SimCSE step with optional PubMedBERT cosine distillation loss."""
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")

        self.model.lm.train()
        self.model.projection_head.train()  # keep projection dropout active for both views
        z1 = self.model.encode(input_ids, attention_mask=attention_mask)
        z2 = self.model.encode(input_ids, attention_mask=attention_mask)

        simcse_loss = self._nt_xent_loss(z1, z2, self.model.logit_scale)

        # --- Distillation loss (teacher CLS → student z1) ---
        distill_lambda = self._get_distill_lambda() if split == "train" else 0.0
        distill_loss = torch.tensor(0.0, device=simcse_loss.device)

        if distill_lambda > 0.0 and "teacher_input_ids" in batch:
            t_input_ids = batch["teacher_input_ids"]
            t_attn_mask = batch.get("teacher_attention_mask")
            with torch.no_grad():
                teacher_out = self.teacher(
                    input_ids=t_input_ids,
                    attention_mask=t_attn_mask,
                )
                # CLS token (position 0) — standard for BERT-style encoders
                teacher_cls = teacher_out.last_hidden_state[:, 0, :]
                teacher_cls = F.normalize(teacher_cls.float(), dim=-1)

            z1_proj = F.normalize(self.distill_proj(z1.float()), dim=-1)
            student_teacher_cos = F.cosine_similarity(
                z1_proj, teacher_cls, dim=-1
            )  # (B,)
            distill_loss = (1.0 - student_teacher_cos).mean()

        total_loss = simcse_loss + distill_lambda * distill_loss

        # --- Logging ---
        self.log(f"{split}/simcse_loss",   simcse_loss,   prog_bar=True,
                 on_step=(split == "train"), on_epoch=True)
        self.log(f"{split}/distill_loss",  distill_loss,  prog_bar=False,
                 on_step=(split == "train"), on_epoch=True)
        self.log(f"{split}/total_loss",    total_loss,    prog_bar=True,
                 on_step=(split == "train"), on_epoch=True)
        if split == "train":
            self.log("train/distill_lambda", distill_lambda, on_step=True)
            pos_cos = (z1 * z2).sum(dim=-1).mean()
            self.log("train/pos_cosine_mean", pos_cos, on_step=True, prog_bar=False)
            # Log student↔teacher alignment when distillation is active
            if distill_lambda > 0.0 and "teacher_input_ids" in batch:
                self.log("train/student_teacher_cos",
                         student_teacher_cos.mean().item(), on_step=True)

        return total_loss


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
