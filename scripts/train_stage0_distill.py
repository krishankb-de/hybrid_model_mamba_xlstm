"""Stage 0: LM pretraining on PubMed with BioMedLM knowledge distillation.

BioMedLM (stanford-crfm/BioMedLM, 2.7B) uses a custom BPE tokenizer with
vocab_size≈28,895 — different from the student's GPT-2 vocab (50,257).
Logit-level KD is therefore impossible (vocab and embedding dims differ).

Instead we use **hidden-state KD**:
  1. Each batch item is tokenized independently by both tokenizers.
  2. Teacher processes its own token IDs → mean-pool last hidden state (dim=2560).
  3. Student processes GPT-2 token IDs → mean-pool last hidden state (dim=512).
  4. A linear projection (512→2560) maps student rep into teacher's space.
  5. KD loss = 1 − mean cosine-similarity(projected_student, teacher).

Total loss:
    L = (1 - alpha) * CE(student_logits, targets)
      + alpha * (1 - cosine_sim(proj(student_hidden), teacher_hidden))

A100 40GB memory budget (bf16, B=8, L=512, grad_accum=8, torch.compile off):
    Student 70M train          : ~1.0 GB  (weights + grads + Adam + acts)
    Teacher 2.7B frozen bf16   : ~5.4 GB weights + ~6-8 GB activations
    Total peak                 : ~18-22 GB on A100 40GB

Example:
    python scripts/train_stage0_distill.py \\
        model=hybrid_70m dataset=pubmed \\
        trainer=a100_single_gpu \\
        distill=stage0_biomedlm \\
        trainer.max_steps=46000 \\
        dataset.batch_size=8 trainer.accumulate_grad_batches=8 \\
        trainer.compile_model=false \\
        experiment_name=hybrid_70m_stage0_kd_pubmed
"""

import os
import sys
import math
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from hybrid_xmamba.models.configuration_hybrid import HybridConfig
from hybrid_xmamba.models.hybrid_lm import HybridLanguageModel
from hybrid_xmamba.training.lightning_module import HybridLightningModule
from hybrid_xmamba.training.metrics import compute_perplexity
from hybrid_xmamba.training.signal_callbacks import SignalCheckpointCallback
from hybrid_xmamba.utils.run_metadata import write_run_metadata

torch.set_float32_matmul_precision("high")


# ---------------------------------------------------------------------------
# Distillation-aware Lightning module
# ---------------------------------------------------------------------------

class DistillLightningModule(HybridLightningModule):
    """HybridLightningModule extended with hidden-state knowledge distillation.

    BioMedLM uses its own BPE vocabulary (≈28,895 tokens) — different from the
    student's GPT-2 vocabulary (50,257). Logit-level KD is therefore impossible.

    Instead we use hidden-state KD:
      - Each batch carries dual tokenizations: GPT-2 for student, BioMedLM for
        teacher (keys: teacher_input_ids, teacher_attention_mask).
      - Teacher's mean-pooled last hidden state (dim=teacher_hidden_dim) is
        matched to the student's projected hidden state via cosine-similarity loss.

    Args:
        model:              HybridLanguageModel (student)
        teacher:            Frozen AutoModelForCausalLM (BioMedLM)
        alpha:              KD loss weight. 0 = pure CE, 1 = pure KD.
        teacher_hidden_dim: Dimensionality of teacher's last hidden state (2560 for 2.7B).
        student_dim:        Dimensionality of student's last hidden state (512 for 70M).
        **kwargs:           Passed to HybridLightningModule.
    """

    def __init__(
        self,
        model: HybridLanguageModel,
        teacher: torch.nn.Module,
        alpha: float = 0.5,
        teacher_hidden_dim: int = 2560,
        student_dim: int = 512,
        **kwargs,
    ):
        super().__init__(model=model, **kwargs)
        # Exclude teacher from hyper_parameters — it is a large nn.Module and Lightning AI's
        # environment injects non-picklable forward hooks that break torch.save.
        self.save_hyperparameters(ignore=['model', 'teacher'])
        self.teacher = teacher
        self.alpha = alpha
        # Linear projection: student_dim → teacher_hidden_dim (no bias — cosine loss)
        self.kd_projection = torch.nn.Linear(student_dim, teacher_hidden_dim, bias=False)

    def on_save_checkpoint(self, checkpoint):
        """Remove teacher state from checkpoint — teacher is frozen and not needed for resume."""
        # Also clear any non-picklable forward hooks injected by the environment (Lightning AI)
        self.teacher._forward_hooks.clear()
        self.teacher._forward_pre_hooks.clear()
        # Drop teacher weights from checkpoint to keep file size small
        checkpoint["state_dict"] = {
            k: v for k, v in checkpoint["state_dict"].items()
            if not k.startswith("teacher.")
        }

    def on_load_checkpoint(self, checkpoint):
        """Teacher was stripped from checkpoint at save time.
        Re-inject teacher state from the live (already loaded) teacher module so
        Lightning's strict state_dict load finds all expected keys.
        """
        for key, param in self.teacher.state_dict().items():
            checkpoint["state_dict"][f"teacher.{key}"] = param

    def configure_optimizers(self):
        """Include kd_projection in the optimizer alongside the student model."""
        from hybrid_xmamba.training.optimizer import get_parameter_groups
        from torch.optim import AdamW

        # Build param groups from student model (with weight decay splits)
        param_groups = get_parameter_groups(self.model, weight_decay=self.weight_decay)
        # kd_projection has only a weight matrix — add to the decay group
        param_groups[0]["params"] += list(self.kd_projection.parameters())

        optimizer = AdamW(
            param_groups,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            foreach=torch.cuda.is_available(),
        )

        # Phase 7 WSD path: delegate to parent helper so β2 annealing metadata is set
        if self.scheduler_name == "wsd":
            sched = self._build_wsd_scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1},
            }

        # Default: cosine decay + linear warmup
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=max(1, self.max_steps - self.warmup_steps),
            eta_min=self.learning_rate * 0.1,
        )
        if self.warmup_steps > 0:
            warmup = LinearLR(
                optimizer, start_factor=0.01, end_factor=1.0,
                total_iters=self.warmup_steps,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[self.warmup_steps],
            )
        else:
            scheduler = cosine

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch.get("labels", input_ids)
        teacher_input_ids = batch.get("teacher_input_ids", None)
        teacher_attn_mask = batch.get("teacher_attention_mask", None)
        student_attn_mask = batch.get("attention_mask", None)

        has_teacher_tokens = teacher_input_ids is not None

        # --- Teacher forward (no grad, bf16) — MUST run first so its transient
        #     activations are freed before student allocates backward buffers. ---
        if has_teacher_tokens:
            with torch.no_grad():
                teacher_out = self.teacher(
                    teacher_input_ids,
                    attention_mask=teacher_attn_mask,
                    output_hidden_states=True,
                )
                t_hidden = teacher_out.hidden_states[-1]  # (B, L_t, D_teacher)
                if teacher_attn_mask is not None:
                    t_mask = teacher_attn_mask.unsqueeze(-1).float()
                    t_pooled = (t_hidden * t_mask).sum(1) / t_mask.sum(1).clamp(min=1)
                else:
                    t_pooled = t_hidden.mean(1)
                t_pooled = t_pooled.detach()  # (B, D_teacher)

        # --- Student forward (with hidden states for KD) ---
        student_out = self.model(
            input_ids, labels=labels, return_dict=True,
            output_hidden_states=has_teacher_tokens,
        )
        ce_loss = student_out.loss

        # --- Hidden-state KD loss ---
        if has_teacher_tokens:
            s_hidden = student_out.hidden_states[-1]  # (B, L_s, D_student)
            if student_attn_mask is not None:
                s_mask = student_attn_mask.unsqueeze(-1).float()
                s_pooled = (s_hidden * s_mask).sum(1) / s_mask.sum(1).clamp(min=1)
            else:
                s_pooled = s_hidden.mean(1)

            s_proj = self.kd_projection(s_pooled)  # (B, D_teacher)
            kd_loss = 1.0 - F.cosine_similarity(
                s_proj, t_pooled.to(s_proj.dtype), dim=-1
            ).mean()
        else:
            kd_loss = torch.tensor(0.0, device=ce_loss.device)

        # --- Total loss ---
        total_loss = (1.0 - self.alpha) * ce_loss + self.alpha * kd_loss

        # --- NaN guard: skip batch instead of aborting ---
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            if not hasattr(self, "_nan_count"):
                self._nan_count = 0
            self._nan_count += 1
            print(
                f"[WARN] NaN/Inf at step {self.global_step} "
                f"(ce={ce_loss.item():.4f}, kd={kd_loss.item():.4f}), "
                f"skipping batch ({self._nan_count} total)"
            )
            if self._nan_count >= 50:
                raise RuntimeError(f"NaN/Inf detected {self._nan_count} times. Aborting.")
            return torch.tensor(0.0, device=ce_loss.device, requires_grad=True)

        # --- Logging ---
        ppl = compute_perplexity(ce_loss)
        self.log("train/ce_loss",    ce_loss,    prog_bar=True,  on_step=True, on_epoch=True)
        self.log("train/kd_loss",    kd_loss,    prog_bar=False, on_step=True, on_epoch=True)
        self.log("train/total_loss", total_loss, prog_bar=True,  on_step=True, on_epoch=True)
        self.log("train/perplexity", ppl,        prog_bar=True,  on_step=True, on_epoch=True)
        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True)

        return total_loss


# ---------------------------------------------------------------------------
# Data loading — packed sequences with dual tokenization for KD
# ---------------------------------------------------------------------------


def build_dataloader(cfg, split: str, tokenizer, teacher_tokenizer=None, teacher_max_length: int = 512):
    """Build packed DataLoader using article text (not just abstracts).

    Student tokens are packed via tokenize → concatenate → chunk (same as
    train.py).  Teacher tokenization is generated on-the-fly in the collate
    function by decoding each student chunk back to text and re-tokenizing
    with the teacher tokenizer.  This keeps student/teacher text aligned
    without cross-tokenizer packing complexity.
    """
    print(f"Loading PubMed ({split})...")
    hf_split = "train" if split == "train" else "validation"
    ds = load_dataset(
        "ccdv/pubmed-summarization",
        split=hf_split,
        streaming=cfg.dataset.get("streaming", False),
        cache_dir=cfg.dataset.get("cache_dir", None),
    )
    val_n = cfg.dataset.get("val_max_samples", 2000)
    if split == "validation" and not cfg.dataset.get("streaming", False):
        ds = ds.select(range(min(val_n, len(ds))))

    seq_length = cfg.dataset.get("max_seq_length", cfg.dataset.max_length)
    num_proc = cfg.dataset.get("preprocessing_num_workers", 2)

    def tokenize_fn(examples):
        texts = examples.get("article", examples.get("abstract", examples.get("text", [""])))
        return tokenizer(texts, truncation=False, return_attention_mask=False)

    ds = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names, num_proc=num_proc)

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = (len(concatenated["input_ids"]) // seq_length) * seq_length
        result = {
            k: [t[i : i + seq_length] for i in range(0, total_length, seq_length)]
            for k, t in concatenated.items()
        }
        return result

    ds = ds.map(group_texts, batched=True, num_proc=num_proc)

    n_samples = len(ds)
    tokens_total = n_samples * seq_length
    print(f"  Packed {split}: {n_samples:,} chunks × {seq_length} = {tokens_total/1e6:.0f}M tokens")

    def collate_fn(batch):
        input_ids = torch.stack([torch.tensor(item["input_ids"], dtype=torch.long) for item in batch])
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()

        result = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

        if teacher_tokenizer is not None:
            texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            t_enc = teacher_tokenizer(
                texts, max_length=teacher_max_length, truncation=True,
                padding="max_length", return_tensors="pt",
            )
            result["teacher_input_ids"] = t_enc["input_ids"]
            result["teacher_attention_mask"] = t_enc["attention_mask"]

        return result

    bs = cfg.dataset.batch_size if split == "train" else cfg.dataset.eval_batch_size
    nw = cfg.dataset.get("num_workers", 2)
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=(split == "train"),
        num_workers=nw,
        pin_memory=cfg.dataset.get("pin_memory", True),
        drop_last=(split == "train"),
        persistent_workers=(nw > 0),
        collate_fn=collate_fn,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print("=" * 80)
    print("Stage 0 Distillation Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Resolve distill config — require it to be set
    distill_cfg = cfg.get("distill", None)
    if distill_cfg is None:
        raise ValueError(
            "distill config not set. Run with distill=stage0_biomedlm\n"
            "  python scripts/train_stage0_distill.py ... distill=stage0_biomedlm"
        )

    pl.seed_everything(cfg.seed, workers=True)
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    write_run_metadata(cfg, cfg.output_dir, extra={"entrypoint": "train_stage0_distill.py"})

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        import subprocess
        subprocess.run(["nvidia-smi"], check=False)

    # ---------------------------------------------------------------------------
    # Tokenizer (GPT-2 BPE, for student)
    # ---------------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(cfg.dataset.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # labels/attention-mask alignment requires right-pad
    tokenizer.model_max_length = cfg.model.max_position_embeddings
    print(f"Student tokenizer vocab size: {tokenizer.vocab_size}")

    # ---------------------------------------------------------------------------
    # Student model
    # ---------------------------------------------------------------------------
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
        norm_topology=cfg.model.get("norm_topology", "pre_rms"),
        use_mlp=cfg.model.use_mlp,
        mlp_ratio=cfg.model.mlp_ratio,
        max_position_embeddings=cfg.model.max_position_embeddings,
        dropout=cfg.model.dropout,
        initializer_range=cfg.model.initializer_range,
        use_cache=cfg.model.use_cache,
        tie_word_embeddings=cfg.model.tie_word_embeddings,
    )
    student = HybridLanguageModel(model_config)

    # Optionally warm-start from existing Stage 0 checkpoint
    lm_ckpt = cfg.get("lm_checkpoint", None)
    if lm_ckpt:
        print(f"Warm-starting student from: {lm_ckpt}")
        ckpt = torch.load(lm_ckpt, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        state = {k.replace("model.", "", 1): v for k, v in state.items()}
        missing, unexpected = student.load_state_dict(state, strict=False)
        print(f"  Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    num_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"Student trainable params: {num_params:,} ({num_params/1e6:.1f}M)")

    # ---------------------------------------------------------------------------
    # Teacher model (BioMedLM, frozen bf16)
    # ---------------------------------------------------------------------------
    teacher_name = distill_cfg.teacher_model
    teacher_dtype_str = distill_cfg.get("teacher_dtype", "bfloat16")
    teacher_dtype = torch.bfloat16 if teacher_dtype_str == "bfloat16" else torch.float16

    print(f"\nLoading teacher: {teacher_name} ({teacher_dtype_str})...")
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_name,
        torch_dtype=teacher_dtype,
        low_cpu_mem_usage=True,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher = teacher.to(device)
    torch.cuda.empty_cache()       # free CPU→GPU copy residual before student alloc

    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"Teacher params: {teacher_params:,} ({teacher_params/1e9:.2f}B), frozen.")

    # Teacher hidden dim: GPT-2 style uses n_embd; fallback to hidden_size
    teacher_hidden_dim = getattr(teacher.config, "n_embd", None) or getattr(teacher.config, "hidden_size", 2560)
    print(f"Teacher hidden dim: {teacher_hidden_dim}")

    # ---------------------------------------------------------------------------
    # Teacher tokenizer (BioMedLM has its own BPE vocab ≠ GPT-2)
    # Teacher max_length capped at teacher's n_positions (usually 1024)
    # ---------------------------------------------------------------------------
    teacher_max_length = min(
        cfg.dataset.max_length,
        getattr(teacher.config, "n_positions", 1024),
    )
    print(f"Loading teacher tokenizer: {teacher_name}")
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_name)
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    teacher_tokenizer.padding_side = "right"
    print(f"Teacher tokenizer vocab size: {teacher_tokenizer.vocab_size}, max_length: {teacher_max_length}")

    # ---------------------------------------------------------------------------
    # Lightning module
    # ---------------------------------------------------------------------------
    _mcfg = cfg.model
    _scheduler_name = _mcfg.get("scheduler_name", "cosine") if hasattr(_mcfg, "get") else getattr(_mcfg, "scheduler_name", "cosine")
    _beta2_schedule = _mcfg.get("beta2_schedule", False) if hasattr(_mcfg, "get") else getattr(_mcfg, "beta2_schedule", False)
    _beta2_start = _mcfg.get("beta2_start", 0.999) if hasattr(_mcfg, "get") else getattr(_mcfg, "beta2_start", 0.999)
    _beta2_end = _mcfg.get("beta2_end", 0.974) if hasattr(_mcfg, "get") else getattr(_mcfg, "beta2_end", 0.974)

    lightning_module = DistillLightningModule(
        model=student,
        teacher=teacher,
        alpha=float(distill_cfg.alpha),
        teacher_hidden_dim=teacher_hidden_dim,
        student_dim=cfg.model.dim,
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        warmup_steps=cfg.model.warmup_steps,
        max_steps=cfg.trainer.max_steps,
        gradient_clip_val=cfg.model.gradient_clip_val,
        compile_model=cfg.trainer.get("compile", False),
        scheduler_name=_scheduler_name,
        beta2_schedule=_beta2_schedule,
        beta2_start=_beta2_start,
        beta2_end=_beta2_end,
    )

    # ---------------------------------------------------------------------------
    # Data (dual tokenization: student=GPT-2, teacher=BioMedLM)
    # ---------------------------------------------------------------------------
    train_loader = build_dataloader(cfg, "train", tokenizer,
                                    teacher_tokenizer=teacher_tokenizer,
                                    teacher_max_length=teacher_max_length)
    val_loader   = build_dataloader(cfg, "validation", tokenizer)
    # Val loader has no teacher tokens — KD loss is skipped for validation

    # ---------------------------------------------------------------------------
    # Callbacks & Loggers
    # ---------------------------------------------------------------------------
    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.checkpoint_dir,
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            every_n_train_steps=cfg.callbacks.checkpoint.get("every_n_train_steps", 2000),
            filename="stage0_kd-{step:06d}-{val/loss:.4f}",
        ),
        LearningRateMonitor(logging_interval="step"),
        SignalCheckpointCallback(checkpoint_dir=cfg.checkpoint_dir),
    ]

    loggers = [TensorBoardLogger(save_dir=cfg.log_dir, name="stage0_kd")]
    if cfg.wandb.enabled:
        loggers.append(WandbLogger(
            project=cfg.wandb.get("project", "hybrid_mamba_xlstm"),
            name=cfg.get("experiment_name", "stage0_kd"),
            config=OmegaConf.to_container(cfg, resolve=True),
        ))

    # ---------------------------------------------------------------------------
    # Trainer
    # ---------------------------------------------------------------------------
    trainer = pl.Trainer(
        max_steps=cfg.trainer.max_steps,
        max_epochs=cfg.trainer.get("max_epochs", 3),
        accelerator=cfg.trainer.get("accelerator", "auto"),
        devices=cfg.trainer.get("devices", 1),
        precision=cfg.trainer.get("precision", "bf16-mixed"),
        accumulate_grad_batches=cfg.trainer.get("accumulate_grad_batches", 4),
        val_check_interval=cfg.trainer.get("val_check_interval", 1000),
        log_every_n_steps=cfg.trainer.get("log_every_n_steps", 25),
        callbacks=callbacks,
        logger=loggers,
        enable_progress_bar=True,
    )

    print(f"\nStarting Stage 0 distillation (alpha={distill_cfg.alpha}, hidden-state KD)")
    print(f"Output: {cfg.output_dir}")
    # Optional resume: restores model/optimizer/global_step from a full Lightning
    # .ckpt. The WSDScheduler is rebuilt from the (possibly new) cfg.trainer.max_steps
    # in configure_optimizers, so the decay phase re-anchors to the new total — this
    # lets a resumed run anneal the LR even if the original run never reached decay.
    resume_ckpt = cfg.get("resume_from_checkpoint", None)
    if resume_ckpt:
        print(f"Resuming from checkpoint: {resume_ckpt}")
    trainer.fit(lightning_module, train_loader, val_loader, ckpt_path=resume_ckpt)
    print("\nStage 0 distillation complete.")
    print(f"Checkpoint saved to: {cfg.checkpoint_dir}")


if __name__ == "__main__":
    main()
