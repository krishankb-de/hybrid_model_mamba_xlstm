"""Stage 0: LM pretraining on PubMed with BioMedLM knowledge distillation.

BioMedLM (stanford-crfm/BioMedLM, 2.7B) shares the GPT-2 BPE 50257 vocab
with the student — logit KD requires zero tokenizer realignment.

Loss:
    L = (1 - alpha) * CE(student_logits, targets)
      + alpha * T^2 * KL(softmax(student/T) || softmax(teacher/T))

A100 memory budget (bf16, B=32, L=512):
    Student 70M train : ~0.84 GB (weights + grads + Adam)
    Teacher 2.7B frozen bf16 : ~5.4 GB weights + ~4-6 GB acts
    Student activations : ~3 GB
    Total : ~15-20 GB  (fits 40 GB with headroom)

Example:
    python scripts/train_stage0_distill.py \\
        --config-name config_70m \\
        dataset=pubmed \\
        trainer=a100_single_gpu \\
        distill=stage0_biomedlm \\
        trainer.max_steps=40000 \\
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

torch.set_float32_matmul_precision("high")


# ---------------------------------------------------------------------------
# Distillation-aware Lightning module
# ---------------------------------------------------------------------------

class DistillLightningModule(HybridLightningModule):
    """HybridLightningModule extended with logit knowledge distillation.

    The teacher (BioMedLM) is frozen, eval-mode, bf16. Its logits are used
    to compute a KL divergence loss alongside the standard cross-entropy.

    Args:
        model:          HybridLanguageModel (student)
        teacher:        Frozen AutoModelForCausalLM (BioMedLM)
        alpha:          KD loss weight. 0 = pure CE, 1 = pure KD.
        temperature:    Softmax temperature for logit smoothing.
        teacher_ppl_abort_threshold: Abort if teacher PPL exceeds this.
        **kwargs:       Passed to HybridLightningModule.
    """

    def __init__(
        self,
        model: HybridLanguageModel,
        teacher: torch.nn.Module,
        alpha: float = 0.5,
        temperature: float = 2.0,
        teacher_ppl_abort_threshold: float = 50.0,
        **kwargs,
    ):
        super().__init__(model=model, **kwargs)
        self.teacher = teacher
        self.alpha = alpha
        self.temperature = temperature
        self.teacher_ppl_abort_threshold = teacher_ppl_abort_threshold
        self._teacher_ppl_checked = False  # only abort-check once at step 0

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch.get("labels", input_ids)

        # --- Student forward ---
        student_out = self.model(input_ids, labels=labels, return_dict=True)
        ce_loss = student_out.loss
        student_logits = student_out.logits   # (B, L, V)

        # --- Teacher forward (no grad, bf16) ---
        with torch.no_grad():
            teacher_logits = self.teacher(input_ids).logits  # (B, L, V)

        # --- Sanity: abort if teacher PPL is implausibly high at step 0 ---
        if not self._teacher_ppl_checked and batch_idx == 0:
            self._teacher_ppl_checked = True
            shift_labels = input_ids[:, 1:].contiguous()
            shift_logits = teacher_logits[:, :-1, :].contiguous()
            teacher_ce = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            teacher_ppl = compute_perplexity(teacher_ce)
            if teacher_ppl > self.teacher_ppl_abort_threshold:
                raise RuntimeError(
                    f"Teacher perplexity ({teacher_ppl:.1f}) exceeds abort threshold "
                    f"({self.teacher_ppl_abort_threshold}). Check teacher loading and vocab."
                )

        # --- KD loss: KL on shifted logits (standard LM teacher-student) ---
        T = self.temperature
        # Align shapes: student & teacher both (B, L-1, V) for next-token prediction
        s_shift = student_logits[:, :-1, :].contiguous()   # (B, L-1, V)
        t_shift = teacher_logits[:, :-1, :].contiguous()   # (B, L-1, V)

        s_log_prob = F.log_softmax(s_shift / T, dim=-1)
        t_prob = F.softmax(t_shift / T, dim=-1)

        # Mask out padding positions (label == -100 maps to pad or EOS in most configs)
        # We use the attention mask to skip pad tokens if present.
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            # Shift mask same as logits: positions 0..L-2 predict tokens 1..L-1
            mask = attention_mask[:, 1:].contiguous().bool()  # (B, L-1)
            kd_loss = F.kl_div(
                s_log_prob[mask],   # (N, V)
                t_prob[mask],       # (N, V)
                reduction="batchmean",
            )
        else:
            kd_loss = F.kl_div(
                s_log_prob.view(-1, s_log_prob.size(-1)),
                t_prob.view(-1, t_prob.size(-1)),
                reduction="batchmean",
            )

        kd_loss = kd_loss * (T ** 2)   # scale back (Hinton et al. 2015)

        # --- Total loss ---
        total_loss = (1.0 - self.alpha) * ce_loss + self.alpha * kd_loss

        # --- Abort on NaN ---
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            raise RuntimeError(
                f"NaN/Inf detected: ce_loss={ce_loss.item():.4f}, "
                f"kd_loss={kd_loss.item():.4f}. Aborting."
            )

        # --- Logging ---
        ppl = compute_perplexity(ce_loss)
        self.log("train/ce_loss",    ce_loss,    prog_bar=True,  on_step=True, on_epoch=True)
        self.log("train/kd_loss",    kd_loss,    prog_bar=False, on_step=True, on_epoch=True)
        self.log("train/total_loss", total_loss, prog_bar=True,  on_step=True, on_epoch=True)
        self.log("train/perplexity", ppl,        prog_bar=True,  on_step=True, on_epoch=True)
        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True)

        return total_loss


# ---------------------------------------------------------------------------
# Data loading (mirrors train_stage0_lm_pubmed.sh config)
# ---------------------------------------------------------------------------

class PubMedLMDataset(torch.utils.data.Dataset):
    """PubMed abstracts tokenised for causal LM (next-token prediction)."""

    def __init__(self, hf_dataset, tokenizer, max_length: int):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get("abstract") or item.get("article") or item.get("text") or ""
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        # For causal LM: labels = input_ids, padding positions masked to -100
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def build_dataloader(cfg, split: str, tokenizer):
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

    dataset = PubMedLMDataset(ds, tokenizer, cfg.dataset.max_length)
    bs = cfg.dataset.batch_size if split == "train" else cfg.dataset.eval_batch_size
    return DataLoader(
        dataset,
        batch_size=bs,
        shuffle=(split == "train"),
        num_workers=cfg.dataset.get("num_workers", 4),
        pin_memory=cfg.dataset.get("pin_memory", True),
        drop_last=(split == "train"),
        persistent_workers=(cfg.dataset.get("num_workers", 4) > 0),
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

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        import subprocess
        subprocess.run(["nvidia-smi"], check=False)

    # ---------------------------------------------------------------------------
    # Tokenizer (GPT-2 BPE, shared with BioMedLM)
    # ---------------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(cfg.dataset.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = cfg.model.max_position_embeddings
    print(f"Tokenizer vocab size: {tokenizer.vocab_size} (should be 50257)")

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

    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"Teacher params: {teacher_params:,} ({teacher_params/1e9:.2f}B), frozen.")

    # ---------------------------------------------------------------------------
    # Lightning module
    # ---------------------------------------------------------------------------
    lightning_module = DistillLightningModule(
        model=student,
        teacher=teacher,
        alpha=float(distill_cfg.alpha),
        temperature=float(distill_cfg.temperature),
        teacher_ppl_abort_threshold=float(distill_cfg.get("teacher_ppl_abort_threshold", 50.0)),
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        warmup_steps=cfg.model.warmup_steps,
        max_steps=cfg.trainer.max_steps,
        gradient_clip_val=cfg.model.gradient_clip_val,
        compile_model=cfg.trainer.get("compile", False),
    )

    # ---------------------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------------------
    train_loader = build_dataloader(cfg, "train", tokenizer)
    val_loader   = build_dataloader(cfg, "validation", tokenizer)

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
        accelerator=cfg.trainer.get("accelerator", "auto"),
        devices=cfg.trainer.get("devices", 1),
        precision=cfg.trainer.get("precision", "bf16-mixed"),
        accumulate_grad_batches=cfg.trainer.get("accumulate_grad_batches", 4),
        gradient_clip_val=cfg.model.gradient_clip_val,
        val_check_interval=cfg.trainer.get("val_check_interval", 1000),
        log_every_n_steps=cfg.trainer.get("log_every_n_steps", 25),
        callbacks=callbacks,
        logger=loggers,
        enable_progress_bar=True,
    )

    print(f"\nStarting Stage 0 distillation (alpha={distill_cfg.alpha}, T={distill_cfg.temperature})")
    print(f"Output: {cfg.output_dir}")
    trainer.fit(lightning_module, train_loader, val_loader)
    print("\nStage 0 distillation complete.")
    print(f"Checkpoint saved to: {cfg.checkpoint_dir}")


if __name__ == "__main__":
    main()
