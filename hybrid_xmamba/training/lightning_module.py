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
        beta2_schedule: bool = False,
        beta2_start: float = 0.999,
        beta2_end: float = 0.974,
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
        # WSD β2 schedule: linearly anneal β2 from beta2_start → beta2_end
        # across the WSD decay phase. Only active when scheduler_name == "wsd".
        self.beta2_schedule = bool(beta2_schedule)
        self.beta2_start = float(beta2_start)
        self.beta2_end = float(beta2_end)
        # Populated by _build_wsd_scheduler when scheduler_name == "wsd".
        self._wsd_decay_start: Optional[int] = None
        self._wsd_decay_steps: Optional[int] = None
    
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
        cu_seqlens = batch.get('cu_seqlens')

        # Forward pass
        outputs = self.model(input_ids, labels=labels, cu_seqlens=cu_seqlens, return_dict=True)
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
        cu_seqlens = batch.get('cu_seqlens')

        # Forward pass
        outputs = self.model(input_ids, labels=labels, cu_seqlens=cu_seqlens, return_dict=True)
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
        cu_seqlens = batch.get('cu_seqlens')

        # Forward pass
        outputs = self.model(input_ids, labels=labels, cu_seqlens=cu_seqlens, return_dict=True)
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
        if self.scheduler_name == "wsd":
            # WSD owns its own warmup; do not wrap with the warmup SequentialLR below.
            scheduler = self._build_wsd_scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
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

    # ------------------------------------------------------------------
    # WSD helpers
    # ------------------------------------------------------------------
    def _build_wsd_scheduler(self, optimizer: Optimizer):
        """Construct a WSDScheduler and record decay-phase metadata for β2."""
        from hybrid_xmamba.training.schedulers import WSDScheduler
        sched = WSDScheduler(optimizer, max_steps=self.max_steps)
        self._wsd_decay_start = sched.decay_start
        self._wsd_decay_steps = sched.decay_steps
        return sched

    def on_train_batch_start(self, batch, batch_idx):
        """Apply β2 schedule when WSD + beta2_schedule are active."""
        if not (self.beta2_schedule and self.scheduler_name == "wsd"):
            return
        if self._wsd_decay_start is None or self._wsd_decay_steps is None:
            return
        from hybrid_xmamba.training.schedulers import apply_beta2_schedule
        optimizer = self.trainer.optimizers[0]
        b2 = apply_beta2_schedule(
            optimizer,
            step=int(self.global_step),
            decay_start=self._wsd_decay_start,
            decay_steps=self._wsd_decay_steps,
            beta2_start=self.beta2_start,
            beta2_end=self.beta2_end,
        )
        self.log('train/adam_beta2', b2, on_step=True)


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
        freeze_text_encoder_steps: int = 0,
        vit_unfreeze_blocks: int = 0,
        vit_lr: float = 1e-6,
        scheduler_name: str = "cosine",
        beta2_schedule: bool = False,
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
            scheduler_name=scheduler_name,
            beta2_schedule=beta2_schedule,
        )
        self.contrastive_mode = contrastive_mode.lower()
        self.image_encoder = None
        # Phase 8: img_proj removed. clip_model.visual already projects to the
        # 512-d BiomedCLIP joint space; the legacy random-init MLP was distorting
        # it (root cause of Phase 5c paired-cos plateau).
        self._img_out = None
        self.vit_unfreeze_blocks = int(vit_unfreeze_blocks)
        self.vit_lr = float(vit_lr)
        # Stage 2 safeguard: freeze the LM backbone for the first N steps so the
        # newly-initialised image-text projection can stabilise before yanking
        # the freshly-trained text encoder weights toward image space.
        # 0 = no freeze (Stage 1 default). Recommended Stage 2 value: 500.
        self.freeze_text_encoder_steps = int(freeze_text_encoder_steps)
        self._lm_currently_frozen = False
        if self.freeze_text_encoder_steps > 0:
            for p in self.model.lm.parameters():
                p.requires_grad = False
            self._lm_currently_frozen = True

        if self.contrastive_mode == "clip":
            import open_clip

            def _get_dim(clip_model, visual) -> int:
                """Get image encoder output dim robustly across open_clip versions.

                TimmModel (BiomedCLIP) dropped .output_dim in newer open_clip.
                Reading embed_dim from the parent CLIP model before extracting
                .visual is the safest cross-version approach.
                """
                if hasattr(clip_model, 'embed_dim'):
                    return clip_model.embed_dim
                if hasattr(visual, 'output_dim'):
                    return visual.output_dim
                if hasattr(visual, 'embed_dim'):
                    return visual.embed_dim
                with torch.no_grad():
                    dummy = torch.zeros(1, 3, 224, 224)
                    return visual.cpu()(dummy).shape[-1]

            def _load_clip_encoder(model_id: str):
                clip_model, _ = open_clip.create_model_from_pretrained(
                    'hf-hub:' + model_id
                )
                img_out = _get_dim(clip_model, clip_model.visual)
                return clip_model.visual, img_out

            biomedclip_id = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            fallback_id   = "laion/CLIP-ViT-B-16-laion2B-s34B-b88K"

            print(f"Loading BiomedCLIP image encoder from: {biomedclip_id}")
            try:
                self.image_encoder, img_out = _load_clip_encoder(biomedclip_id)
                print(f"✓ BiomedCLIP loaded. Image dim: {img_out}, "
                      f"Text dim: {model.embed_dim}")
            except Exception as e:
                print(f"BiomedCLIP load error: {e} — falling back to {fallback_id}")
                try:
                    self.image_encoder, img_out = _load_clip_encoder(fallback_id)
                    print(f"✓ Fallback CLIP loaded. Image dim: {img_out}")
                except Exception as e2:
                    raise ImportError(
                        f"Failed to load image encoder.\n"
                        f"BiomedCLIP error: {e}\nFallback error: {e2}\n"
                        "Install open-clip-torch: pip install open-clip-torch"
                    )

            # Freeze entire image encoder; selective unfreeze handled below.
            self.image_encoder.eval()
            for p in self.image_encoder.parameters():
                p.requires_grad = False

            # Phase 8: img_proj deleted. clip_model.visual outputs are already
            # in the 512-d BiomedCLIP joint space (same as student projection_head
            # output dim), so a projection here both adds nothing and distorts.
            self._img_out = img_out
            assert img_out == model.embed_dim, (
                f"image-encoder out dim ({img_out}) must equal student embed_dim "
                f"({model.embed_dim}) — both should be 512 for BiomedCLIP."
            )

            # Optionally unfreeze the last N ViT transformer blocks with low LR.
            if self.vit_unfreeze_blocks > 0:
                blocks = self._get_vit_blocks()
                for blk in blocks[-self.vit_unfreeze_blocks:]:
                    for p in blk.parameters():
                        p.requires_grad = True
                self.image_encoder.train()
                n_unfreeze = sum(
                    p.numel() for blk in blocks[-self.vit_unfreeze_blocks:]
                    for p in blk.parameters()
                )
                print(f"✓ Unfreezing last {self.vit_unfreeze_blocks} ViT blocks "
                      f"({n_unfreeze/1e6:.1f}M params, lr={self.vit_lr})")

    # ------------------------------------------------------------------
    # ViT block discovery helper
    # ------------------------------------------------------------------
    def _get_vit_blocks(self):
        """Return a list of transformer blocks from the image encoder.

        Handles both open_clip VisionTransformer (.transformer.resblocks)
        and TimmModel backbones (.trunk.blocks).
        """
        enc = self.image_encoder
        if hasattr(enc, 'transformer') and hasattr(enc.transformer, 'resblocks'):
            return list(enc.transformer.resblocks)
        if hasattr(enc, 'trunk') and hasattr(enc.trunk, 'blocks'):
            return list(enc.trunk.blocks)
        if hasattr(enc, 'blocks'):
            return list(enc.blocks)
        raise AttributeError(
            f"Cannot find transformer blocks in image encoder {type(enc)}. "
            "Expected .transformer.resblocks, .trunk.blocks, or .blocks."
        )

    # ------------------------------------------------------------------
    # Optimizer: 3 param groups when ViT unfreeze is active
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        if self.vit_unfreeze_blocks > 0 and self.image_encoder is not None:
            # Group 1: main model + projection head at base LR.
            # Phase 8: img_proj deleted — only the student model is trained here.
            main_params = list(self.model.parameters())
            # Group 2: unfrozen ViT blocks at vit_lr
            blocks = self._get_vit_blocks()
            vit_params = [
                p for blk in blocks[-self.vit_unfreeze_blocks:]
                for p in blk.parameters() if p.requires_grad
            ]
            param_groups = [
                {"params": main_params, "lr": self.learning_rate,
                 "weight_decay": self.weight_decay},
                {"params": vit_params, "lr": self.vit_lr,
                 "weight_decay": self.weight_decay},
            ]
            optimizer = torch.optim.AdamW(param_groups)
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

        if self.scheduler_name == "wsd":
            scheduler = self._build_wsd_scheduler(optimizer)
        else:
            from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
            warmup = LinearLR(
                optimizer, start_factor=0.01, end_factor=1.0,
                total_iters=max(1, self.warmup_steps),
            )
            cosine = CosineAnnealingLR(
                optimizer,
                T_max=max(1, self.max_steps - self.warmup_steps),
                eta_min=self.learning_rate * 0.1,
            )
            scheduler = SequentialLR(
                optimizer, schedulers=[warmup, cosine],
                milestones=[self.warmup_steps],
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    # ------------------------------------------------------------------
    # Contrastive loss (symmetric NT-Xent / InfoNCE)
    # ------------------------------------------------------------------
    def _nt_xent_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        logit_scale: torch.Tensor,
        fixed_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Symmetric InfoNCE loss with numerical stability.

        Args:
            z1: Normalised embeddings from view-1 (B, D)
            z2: Normalised embeddings from view-2 (B, D)
            logit_scale: Learnable temperature (scalar) — used only when fixed_scale is None
            fixed_scale: If given, use this constant scale (1/τ) instead of logit_scale.
                         Set to 20.0 for SimCSE (τ=0.05) to prevent collapse.

        Returns:
            Scalar loss
        """
        if fixed_scale is not None:
            scale = torch.tensor(fixed_scale, dtype=z1.dtype, device=z1.device)
        else:
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
    def on_train_batch_start(self, batch, batch_idx):
        # Unfreeze the LM backbone once the warmup window passes.
        if (
            self._lm_currently_frozen
            and self.global_step >= self.freeze_text_encoder_steps
        ):
            for p in self.model.lm.parameters():
                p.requires_grad = True
            self._lm_currently_frozen = False
            self.print(
                f"[freeze_text_encoder_steps={self.freeze_text_encoder_steps}] "
                f"Unfreezing LM backbone at global_step={self.global_step}"
            )

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

        # During training: force full model into train mode so projection_head
        # Dropout(0.1) is active — this is the source of view diversity (z1 != z2).
        # During validation: respect Lightning's eval() context; val SimCSE loss will
        # be near-zero (expected) since dropout is off. Use ContrastiveEvalCallback
        # Spearman metrics instead for meaningful validation signal.
        if split == "train":
            self.model.train()
        z1 = self.model.encode(input_ids, attention_mask=attention_mask)
        z2 = self.model.encode(input_ids, attention_mask=attention_mask)

        # scale=20 (τ=0.05): standard SimCSE temperature. Lower scale (τ=0.2) makes
        # the loss trivially small for a pretrained backbone — high scale forces the
        # model to discriminate fine-grained differences even when embeddings are spread.
        loss = self._nt_xent_loss(z1, z2, self.model.logit_scale, fixed_scale=20.0)
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

        # Image embeddings — frozen unless vit_unfreeze_blocks > 0
        if self.vit_unfreeze_blocks > 0:
            z_img_raw = self.image_encoder(pixel_values)
        else:
            with torch.no_grad():
                z_img_raw = self.image_encoder(pixel_values)
        # Phase 8: img_proj bypassed — clip_model.visual already returns the
        # 512-d BiomedCLIP joint embedding.
        z_img = torch.nn.functional.normalize(z_img_raw.float(), dim=-1)

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
        # ramp_steps=0 means start at lambda_max immediately (no division by zero)
        if self.distill_ramp <= 0 or ramp_step >= self.distill_ramp:
            return self.lambda_max
        return self.lambda_max * (ramp_step / self.distill_ramp)

    def _simcse_step(self, batch, batch_idx, split: str):
        """Pure PubMedBERT cosine distillation step (SimCSE removed).

        Why SimCSE was removed:
        The Stage 0 backbone (PPL=13.10) already perfectly separates PubMed
        abstracts in embedding space. InfoNCE loss with bs=8 in-batch negatives
        starts at ~0.002 from step 1 (expected: 2.08 for random embeddings).
        The gradient signal is effectively zero — SimCSE cannot improve geometry
        that is already excellent. All observed 'collapse' was this: the backbone
        had nothing to learn from SimCSE.

        Pure KD directly aligns the student projection head to PubMedBERT's CLS
        space (which scores BIOSSES=0.85). Expected student BIOSSES: 0.72-0.80.
        """
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")

        if split == "train":
            self.model.train()
        z1 = self.model.encode(input_ids, attention_mask=attention_mask)

        # --- Pure distillation loss: align student to PubMedBERT CLS ---
        distill_loss = torch.tensor(0.0, device=z1.device)
        student_teacher_cos = torch.tensor(0.0, device=z1.device)

        if "teacher_input_ids" in batch:
            t_input_ids = batch["teacher_input_ids"]
            t_attn_mask = batch.get("teacher_attention_mask")
            with torch.no_grad():
                teacher_out = self.teacher(
                    input_ids=t_input_ids,
                    attention_mask=t_attn_mask,
                )
                teacher_cls = teacher_out.last_hidden_state[:, 0, :]
                teacher_cls = F.normalize(teacher_cls.float(), dim=-1)

            z1_proj = F.normalize(self.distill_proj(z1.float()), dim=-1)
            student_teacher_cos = F.cosine_similarity(z1_proj, teacher_cls, dim=-1)
            distill_loss = (1.0 - student_teacher_cos).mean()

        total_loss = distill_loss

        # --- Logging ---
        self.log(f"{split}/distill_loss", distill_loss, prog_bar=True,
                 on_step=(split == "train"), on_epoch=True)
        self.log(f"{split}/total_loss",   total_loss,   prog_bar=True,
                 on_step=(split == "train"), on_epoch=True)
        if split == "train":
            self.log("train/student_teacher_cos",
                     student_teacher_cos.mean().item(), on_step=True, prog_bar=True)

        return total_loss


def _load_biomedclip_text_teacher():
    """Load BiomedCLIP wrapper exposing ``encode_text`` as the KD teacher.

    Returns the full open_clip CLIP wrapper (frozen, eval). The 512-d
    post-projection joint embedding is shared with the image tower by
    construction — this is the reason for the Phase 4 pivot.
    """
    import open_clip
    model, _ = open_clip.create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class JointMultiTaskLightningModule(HybridContrastiveLightningModule):
    """Joint KD + CLIP + SimCSE training on MIMIC-CXR.

    Loss = α·L_KD(BiomedCLIP-text) + β·L_CLIP(BiomedCLIP) + γ·L_SimCSE

    Batch must contain: input_ids, attention_mask, pixel_values,
    teacher_input_ids, teacher_attention_mask.

    4 param groups:
      1. backbone weight matrices  — lr=backbone_lr, wd=weight_decay
      2. backbone bias/norm        — lr=backbone_lr, wd=0
      3. head params               — lr=head_lr, wd=weight_decay
      4. ViT unfrozen blocks       — lr=vit_lr (only if vit_unfreeze_blocks > 0)
    """

    def __init__(
        self,
        model,
        teacher: nn.Module,
        alpha_kd: float = 0.3,
        alpha_kd_warmup: Optional[float] = None,
        alpha_kd_post: Optional[float] = None,
        beta_clip: float = 1.0,
        gamma_simcse: float = 0.1,
        backbone_lr: float = 1e-5,
        head_lr: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        max_steps: int = 10000,
        gradient_clip_val: float = 1.0,
        freeze_text_encoder_steps: int = 500,
        vit_unfreeze_blocks: int = 0,
        vit_lr: float = 1e-6,
        moco_queue_size: int = 0,
        moco_momentum: float = 0.999,
        scheduler_name: str = "cosine",
        beta2_schedule: bool = False,
    ):
        # contrastive_mode="clip" so parent loads the BiomedCLIP visual encoder.
        # Phase 8: img_proj deleted from parent — visual output is used directly.
        super().__init__(
            model=model,
            contrastive_mode="clip",
            learning_rate=backbone_lr,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            gradient_clip_val=gradient_clip_val,
            freeze_text_encoder_steps=freeze_text_encoder_steps,
            vit_unfreeze_blocks=vit_unfreeze_blocks,
            vit_lr=vit_lr,
            scheduler_name=scheduler_name,
            beta2_schedule=beta2_schedule,
        )
        self.save_hyperparameters(ignore=['model', 'teacher'])
        self.teacher = teacher
        self.alpha_kd = alpha_kd
        # Phase 10: α_kd schedule. CLIP is gated off during warmup (Phase 9),
        # so α_kd_warmup can be high (default 1.0) without gradient conflict.
        # Post-unfreeze, CLIP engages and α_kd drops back to the safe value
        # (default == alpha_kd, typically 0.3). If neither override is given,
        # behaviour reduces to the pre-Phase-10 constant α_kd.
        self.alpha_kd_warmup = float(alpha_kd_warmup) if alpha_kd_warmup is not None else float(alpha_kd)
        self.alpha_kd_post = float(alpha_kd_post) if alpha_kd_post is not None else float(alpha_kd)
        self.beta_clip = beta_clip
        self.gamma_simcse = gamma_simcse
        self.backbone_lr = backbone_lr
        self.head_lr = head_lr

        # Phase 8: distill_proj deleted. KD applies directly on z_text vs the
        # 512-d BiomedCLIP joint embedding (same dim as student projection_head),
        # so no projection is needed. Removing the dead module saves ~500 K
        # optimizer-state params and eliminates a future regression surface.

        # MoCo text queue only.  Disabled when moco_queue_size=0.
        # text_queue: momentum text keys — negatives for i2t direction.
        # No img_queue: image encoder is frozen, so its outputs are
        # deterministic — in-batch negatives for t2i are consistent and
        # sufficient. An img_queue filled with random init vectors at
        # startup produces max-entropy t2i loss (log(K+1)≈9.7) that
        # dominates and destroys the useful i2t signal.
        self.moco_queue_size = moco_queue_size
        if moco_queue_size > 0:
            from hybrid_xmamba.training.moco_queue import MoCoQueue, MomentumEncoder
            self.text_queue = MoCoQueue(dim=512, K=moco_queue_size)
            self.momentum_encoder = MomentumEncoder(model, m=moco_momentum)
        else:
            self.text_queue = None
            self.momentum_encoder = None

    # ------------------------------------------------------------------
    # Optimizer: 4 param groups
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        def _is_no_decay(name):
            return name.endswith('.bias') or 'norm' in name.lower()

        backbone_wd, backbone_no_wd = [], []
        for name, p in self.model.lm.named_parameters():
            if not p.requires_grad:
                continue
            if _is_no_decay(name):
                backbone_no_wd.append(p)
            else:
                backbone_wd.append(p)

        # Phase 8: distill_proj + img_proj deleted — head_params is now just
        # projection_head, attn_pool, and logit_scale.
        head_params = (
            list(self.model.projection_head.parameters())
            + (list(self.model.attn_pool.parameters())
               if self.model.attn_pool is not None else [])
            + [self.model.logit_scale]
        )

        param_groups = [
            {"params": backbone_wd,    "lr": self.backbone_lr, "weight_decay": self.weight_decay},
            {"params": backbone_no_wd, "lr": self.backbone_lr, "weight_decay": 0.0},
            {"params": head_params,    "lr": self.head_lr,     "weight_decay": self.weight_decay},
        ]

        if self.vit_unfreeze_blocks > 0 and self.image_encoder is not None:
            blocks = self._get_vit_blocks()
            vit_params = [
                p for blk in blocks[-self.vit_unfreeze_blocks:]
                for p in blk.parameters() if p.requires_grad
            ]
            if vit_params:
                param_groups.append(
                    {"params": vit_params, "lr": self.vit_lr,
                     "weight_decay": self.weight_decay}
                )

        optimizer = torch.optim.AdamW(param_groups)

        if self.scheduler_name == "wsd":
            scheduler = self._build_wsd_scheduler(optimizer)
        else:
            from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
            warmup = LinearLR(
                optimizer, start_factor=0.01, end_factor=1.0,
                total_iters=max(1, self.warmup_steps),
            )
            cosine = CosineAnnealingLR(
                optimizer,
                T_max=max(1, self.max_steps - self.warmup_steps),
                eta_min=self.backbone_lr * 0.1,
            )
            scheduler = SequentialLR(
                optimizer, schedulers=[warmup, cosine],
                milestones=[self.warmup_steps],
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------
    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Strip dead-module keys from Phase ≤6c checkpoints for back-compat.

        Phase 5c/6a-c checkpoints contain img_proj.* and distill_proj.* keys
        that no longer exist in Phase 8 model. Removing them here lets Lightning
        resume without a strict-load KeyError.
        """
        sd = checkpoint.get("state_dict", {})
        dead = [k for k in sd if k.startswith(("img_proj.", "distill_proj."))]
        for k in dead:
            del sd[k]
        if dead:
            self.print(
                f"[on_load_checkpoint] stripped {len(dead)} dead keys "
                f"(img_proj/distill_proj) — loading Phase ≤6c checkpoint."
            )

    def on_train_batch_start(self, batch, batch_idx):
        """Phase 9: at the unfreeze step, hard-resync the momentum encoder
        from the live model and reset the MoCo queue.

        During the KD-only warmup CLIP is gated off, the queue is not enqueued
        and the EMA encoder is not updated. At step == freeze_text_encoder_steps
        the live model has been pulled into BiomedCLIP-text space; the EMA
        copy is still ~Stage-0 weights and the queue still holds its random-
        init unit vectors. Hard-copying weights and resetting the queue makes
        the first post-unfreeze batch's negatives non-stale.
        """
        was_frozen = self._lm_currently_frozen
        super().on_train_batch_start(batch, batch_idx)
        just_unfroze = was_frozen and not self._lm_currently_frozen
        if just_unfroze and self.momentum_encoder is not None:
            self.momentum_encoder.copy_from(self.model)
            if self.text_queue is not None:
                self.text_queue.reset()
            self.print(
                f"[Phase 9 resync] global_step={self.global_step}: "
                "copied live model → momentum_encoder; reset text_queue."
            )

    def training_step(self, batch, batch_idx):
        loss = self._joint_step(batch, batch_idx, split="train")
        # EMA update after every optimiser step — but only once CLIP is gated on.
        # During KD-only warmup the queue is not enqueued, so EMA drift is wasted
        # work and would force the post-unfreeze hard-resync to do strictly more
        # corrective work. Skipping it keeps the EMA copy at its init state until
        # the resync happens at the unfreeze boundary.
        if (
            self.momentum_encoder is not None
            and self.global_step >= self.freeze_text_encoder_steps
        ):
            self.momentum_encoder.update(self.model)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._joint_step(batch, batch_idx, split="val")

    def _joint_step(self, batch, batch_idx, split):
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        pixel_values = batch.get("pixel_values")

        if split == "train":
            self.model.train()

        # Text embedding view 1 — reused for CLIP, KD, and SimCSE anchor
        z_text = self.model.encode(input_ids, attention_mask=attention_mask)

        # L_CLIP (with optional MoCo queue for Phase 5).
        # Phase 9: CLIP loss is gated off during the KD-only warmup. Until
        # global_step >= freeze_text_encoder_steps, z_text is still in (or
        # near) GPT-2 space — running CLIP at this point pollutes proj_head
        # gradients and fills the MoCo queue with stale keys that produce
        # random InfoNCE gradients post-unfreeze (the 6a/6b/6c failure mode).
        clip_gated_on = self.global_step >= self.freeze_text_encoder_steps
        if (
            pixel_values is not None
            and self.image_encoder is not None
            and (split != "train" or clip_gated_on)
        ):
            if self.vit_unfreeze_blocks > 0:
                z_img_raw = self.image_encoder(pixel_values)
            else:
                with torch.no_grad():
                    z_img_raw = self.image_encoder(pixel_values)
            # Phase 6: clip_model.visual already includes BiomedCLIP's image
            # projection → output is already in the 512-d joint space.
            # img_proj (random-init MLP) was distorting those clean embeddings,
            # keeping paired cosine stuck at 0.22-0.29 across all runs.
            z_img = F.normalize(z_img_raw.float(), dim=-1)

            if self.text_queue is not None and split == "train":
                # MoCo i2t + in-batch t2i:
                #   i2t: z_img  queries → [z_text_k | text_queue] (16K+ negatives)
                #   t2i: z_text queries → z_img in-batch only
                # Image encoder is frozen → in-batch image keys are deterministic;
                # no img_queue needed (and it would start full of random noise).
                z_text_k = self.momentum_encoder.encode(
                    batch["input_ids"], batch.get("attention_mask")
                )  # (B, 512) momentum text key
                l_clip = self._moco_clip_loss_symmetric(z_text, z_img, z_text_k)
                self.text_queue.enqueue(z_text_k)
            else:
                l_clip = self._nt_xent_loss(z_text, z_img, self.model.logit_scale)
        else:
            l_clip = torch.tensor(0.0, device=z_text.device)

        # L_SimCSE — second dropout view
        z_text2 = self.model.encode(input_ids, attention_mask=attention_mask)
        l_simcse = self._nt_xent_loss(
            z_text, z_text2, self.model.logit_scale, fixed_scale=20.0
        )

        # L_KD — direct cosine distillation on z_text toward BiomedCLIP text
        # tower (512-d joint space, same as image tower by construction).
        # Phase 8: distill_proj deleted. KD trains projection_head directly.
        l_kd = torch.tensor(0.0, device=z_text.device)
        cos_text_teacher = torch.tensor(0.0, device=z_text.device)
        if "teacher_input_ids" in batch:
            t_ids = batch["teacher_input_ids"]
            with torch.no_grad():
                t_emb = self.teacher.encode_text(t_ids)  # (B, 512)
                t_emb = F.normalize(t_emb.float(), dim=-1)
            cos_per_sample = F.cosine_similarity(z_text.float(), t_emb, dim=-1)
            l_kd = (1.0 - cos_per_sample).mean()
            cos_text_teacher = cos_per_sample.detach().mean()

        # Phase 10: α_kd schedule. During the gated warmup (CLIP off), KD
        # owns the gradient — boost α_kd_warmup to drive z_text into BCT
        # space fast. Post-unfreeze, decay to α_kd_post to avoid the 6a/6b/6c
        # KD-vs-CLIP gradient conflict (text and image targets cos~0.5–0.7
        # apart in joint space).
        effective_alpha_kd = (
            self.alpha_kd_warmup
            if self.global_step < self.freeze_text_encoder_steps
            else self.alpha_kd_post
        )

        total = (
            effective_alpha_kd * l_kd
            + self.beta_clip * l_clip
            + self.gamma_simcse * l_simcse
        )

        on_step = (split == "train")
        self.log(f"{split}/kd_loss",     l_kd,     prog_bar=False, on_step=on_step, on_epoch=True)
        self.log(f"{split}/clip_loss",   l_clip,   prog_bar=True,  on_step=on_step, on_epoch=True)
        self.log(f"{split}/simcse_loss", l_simcse, prog_bar=False, on_step=on_step, on_epoch=True)
        self.log(f"{split}/total_loss",  total,    prog_bar=True,  on_step=on_step, on_epoch=True)
        # Phase 10: cos_text_teacher = cos(z_text, BCT). Kill-job signal:
        # must rise from ~0 to ≥0.7 by step 800 of warmup. If <0.5 by 800
        # → α_kd_warmup or proj_head LR is wrong.
        self.log(
            f"{split}/cos_text_teacher", cos_text_teacher,
            prog_bar=(split == "train"), on_step=on_step, on_epoch=True,
        )
        if split == "train":
            self.log(
                "train/effective_alpha_kd",
                float(effective_alpha_kd),
                on_step=True, on_epoch=False,
            )
        if split == "train":
            pos_cos = (z_text * z_text2).sum(dim=-1).mean()
            self.log("train/pos_cosine_mean", pos_cos, on_step=True, on_epoch=False)

        return total

    def _moco_clip_loss_symmetric(
        self,
        z_text: torch.Tensor,   # (B, 512) online text query, L2-normed
        z_img: torch.Tensor,    # (B, 512) frozen image embedding, L2-normed
        z_text_k: torch.Tensor, # (B, 512) momentum text key, L2-normed
    ) -> torch.Tensor:
        """Hybrid MoCo InfoNCE — large text queue for i2t, in-batch for t2i.

        i2t: z_img  queries against [z_text_k | text_queue]  → 16K+ text negatives
        t2i: z_text queries against z_img (in-batch only)    → B-1 image negatives

        i2t gets the MoCo boost (16K negatives vs 31).
        t2i uses in-batch, which is safe because the image encoder is frozen
        and its outputs are deterministic — no warmup problem.

        No img_queue: a queue seeded with random unit vectors produces a
        t2i InfoNCE loss at the theoretical maximum log(K+1)≈9.7, flooding
        the optimiser with noise and preventing any useful learning.
        """
        scale = self.model.logit_scale.exp().clamp(1.0, 100.0)
        labels = torch.arange(z_img.shape[0], device=z_img.device)

        # i2t: image queries → text key bank (B + K keys from queue)
        text_bank = torch.cat(
            [z_text_k, self.text_queue.all_keys().to(z_text_k.device)], dim=0
        )  # (B+K, 512)
        l_i2t = F.cross_entropy(scale * z_img @ text_bank.T, labels)

        # t2i: text queries → in-batch image keys only (frozen encoder = consistent)
        l_t2i = F.cross_entropy(scale * z_text @ z_img.T, labels)

        return 0.5 * (l_i2t + l_t2i)


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
