"""Phase 10E: image-conditioned radiology report generation training script.

Reuses scripts/train_contrastive.py's MIMIC-CXR local-parquet loading
(load_mimic_cxr -> ImageTextDataset) directly rather than duplicating it —
that gives {input_ids, attention_mask, pixel_values} batches from the exact
same Phase 8 build, image transform pipeline, and findings/impression text
convention the retrieval chapter already used. Deliberately bypasses
prepare_dataloader()'s dataset_name=="mimic_cxr" dispatch string (see
configs/dataset/cxr_mimic_full.yaml's header comment) so a report-gen
dataset config doesn't have to fight that exact-string requirement.

STILL BLOCKED on: the Phase 8 local MIMIC-CXR-JPG build finishing (job
2461245) and a decoder checkpoint (--decoder-checkpoint / decoder_checkpoint
cfg key, e.g. the Phase-5/Stage-0 150M checkpoint per 10D) — this script is
infrastructure, not yet a runnable recipe, per H100_SCALING_PLAN.md's 10E.

Example (once data + a decoder checkpoint exist):
    python scripts/train_report_generation.py \\
        model=hybrid_150m_v2_rrg dataset=cxr_mimic_full \\
        trainer=h100_single_gpu trainer.max_steps=10000 \\
        decoder_checkpoint=./outputs/h100_stage0_150m_v2/checkpoints/stage0_model_only.pt \\
        experiment_name=report_gen_150m_v1
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer

from hybrid_xmamba.models.configuration_hybrid import HybridConfig
from hybrid_xmamba.training.lightning_module import ReportGenerationLightningModule
from hybrid_xmamba.training.signal_callbacks import SignalCheckpointCallback
from hybrid_xmamba.utils.run_metadata import write_run_metadata

from scripts.train_contrastive import load_mimic_cxr

torch.set_float32_matmul_precision("high")


def compute_rare_finding_sample_weights(
    study_ids: List[int],
    chexpert_csv: str,
    rare_labels: List[str],
    oversample_weight: float,
) -> List[float]:
    """Phase 13F: per-row WeightedRandomSampler weights that oversample
    training reports whose ground-truth CheXpert label is positive for one
    of `rare_labels` -- e.g. Lung Lesion/Pneumothorax/Pleural Other, the 3
    labels the 13B checkpoint never predicts at all (F1=0.0 on both eval
    splits). U-Zeros convention (mimic-cxr-2.0.0-chexpert.csv.gz): 1.0 ->
    positive, {0.0, -1.0, NaN} -> not-positive -- matches the convention the
    now-removed evaluate_report_generation.py ground-truth cross-check used
    (git history, commit cadcc6b). A study_id with no row in the CSV (should
    not happen for the official build, but not asserted here) gets weight
    1.0, i.e. NOT oversampled -- conservative, never inflates an unknown-label
    row above baseline.

    Pure function (no dataset/dataloader construction) so it's CPU-unit-
    testable against a tiny on-disk CSV fixture, independent of the actual
    MIMIC-CXR-JPG images/HF Dataset machinery.
    """
    import pandas as pd

    df = pd.read_csv(chexpert_csv, usecols=["study_id"] + rare_labels)
    df = df[df["study_id"].isin(set(study_ids))]
    positive_study_ids = {
        int(row["study_id"])
        for _, row in df.iterrows()
        if any(row[label] == 1.0 for label in rare_labels)
    }
    return [oversample_weight if sid in positive_study_ids else 1.0 for sid in study_ids]


# Phase 15C-1 — the CheXbert-14 label order, as f1chexbert's
# `target_names` emits it. The aux targets MUST be in this order: 15B-5's
# per-label CIs, chexbert_labels.json's `label_names`, and every per-label
# table in analysis/ are all in it, and a silently permuted target vector
# would train the head against the wrong findings while every loss curve
# still looked healthy. The CheXpert CSV's own columns are alphabetical;
# load_chexpert_label_matrix() selects BY NAME, so the CSV's order is
# irrelevant and this list is the single source of truth.
CHEXPERT_14_LABELS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding",
]


def load_chexpert_label_matrix(
    study_ids: List[int],
    chexpert_csv: str,
    labels: Optional[List[str]] = None,
) -> Dict[int, List[float]]:
    """Phase 15C-1: study_id -> 14 floats in CHEXPERT_14_LABELS order.

    U-Zeros convention (1.0 -> positive; {0.0, -1.0, NaN} -> negative),
    matching compute_rare_finding_sample_weights() verbatim so 15C and 13F
    stay directly comparable (user decision 2026-09-13, chosen over masking
    uncertains out).

    A study_id absent from the CSV is simply ABSENT from the returned dict --
    never a row of zeros. ImageTextDataset turns that into mask 0.0, so the
    aux loss skips the sample instead of learning "no findings" from missing
    data. Same conservative shape as 13F's weight-1.0 default.

    Pure function (no dataset/dataloader construction) so it is CPU-unit-
    testable against a tiny on-disk CSV fixture.
    """
    import pandas as pd

    labels = list(labels) if labels else list(CHEXPERT_14_LABELS)
    df = pd.read_csv(chexpert_csv, usecols=["study_id"] + labels)
    df = df[df["study_id"].isin(set(study_ids))]
    return {
        int(row["study_id"]): [1.0 if row[label] == 1.0 else 0.0 for label in labels]
        for _, row in df.iterrows()
    }


def compute_aux_pos_weight(
    label_matrix: Dict[int, List[float]],
    cap: float = 10.0,
    num_labels: int = 14,
) -> List[float]:
    """Phase 15C-2: BCEWithLogitsLoss pos_weight per label, = (N - n_pos)/n_pos, capped.

    The cap is load-bearing, not decoration. Uncapped, Pleural Other's ~1%
    prevalence gives ~100x, and 13F showed 5x oversampling already damaged
    four-plus common high-support labels. cap=10.0 is the pre-registered
    default and stays FIXED across 15C's arms: aux_lambda is the only lever
    that moves, so any effect is attributable to it alone.

    A label with zero positives in the train split gets weight 1.0 (no signal
    to amplify), never a division by zero.
    """
    n = len(label_matrix)
    if n == 0:
        return [1.0] * num_labels
    counts = [0.0] * num_labels
    for vec in label_matrix.values():
        for i, v in enumerate(vec):
            counts[i] += float(v)
    out = []
    for n_pos in counts:
        if n_pos <= 0:
            out.append(1.0)
        else:
            out.append(min((n - n_pos) / n_pos, float(cap)))
    return out


def prepare_report_gen_dataloader(cfg: DictConfig, split: str, tokenizer) -> DataLoader:
    """{input_ids, attention_mask, pixel_values} batches from the Phase 8
    local MIMIC-CXR-JPG parquet build, via train_contrastive.py's
    load_mimic_cxr() with no teacher tokenizer (report-gen needs no KD teacher).
    """
    dataset = load_mimic_cxr(cfg, split, tokenizer, teacher_tokenizer=None)
    batch_size = cfg.dataset.batch_size if split == "train" else cfg.dataset.eval_batch_size

    # Phase 15C-1 — attach aux targets iff the aux loss is actually on. One
    # lever (model.aux_lambda) turns on the head, the targets and the batch
    # key together; at the default 0.0 nothing here runs and the batch keys
    # are exactly what every arm in 15B-4's seed band trained on.
    if float(cfg.model.get("aux_lambda", 0.0)) > 0:
        split_study_ids = [int(s) for s in dataset.data["study_id"]]
        dataset.chexpert_labels = load_chexpert_label_matrix(
            study_ids=split_study_ids, chexpert_csv=cfg.dataset.chexpert_csv,
        )
        dataset.chexpert_num_labels = len(CHEXPERT_14_LABELS)
        # Rows, not studies: a study with several images is several rows, and
        # it is the row coverage the aux loss actually sees.
        covered = sum(1 for s in split_study_ids if s in dataset.chexpert_labels)
        print(f"  [aux] {split}: CheXpert targets cover {covered}/{len(split_study_ids)} rows "
              f"({covered / max(len(split_study_ids), 1):.1%}); the rest are masked out of the aux loss")

    sampler = None
    shuffle = (split == "train")
    if split == "train" and cfg.dataset.get("oversample_rare_findings", False):
        study_ids = list(dataset.data["study_id"])
        weights = compute_rare_finding_sample_weights(
            study_ids=study_ids,
            chexpert_csv=cfg.dataset.chexpert_csv,
            rare_labels=list(cfg.dataset.get("rare_finding_labels", [])),
            oversample_weight=float(cfg.dataset.get("oversample_weight", 1.0)),
        )
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False  # sampler and shuffle are mutually exclusive on DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory,
        drop_last=(split == "train"),
        persistent_workers=(cfg.dataset.num_workers > 0),
    )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed, workers=True)
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    write_run_metadata(cfg, cfg.output_dir, extra={"entrypoint": "train_report_generation.py"})

    tokenizer = AutoTokenizer.from_pretrained(cfg.dataset.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = cfg.model.max_position_embeddings

    # MAMBA3_PLAN_V2.md M2-F: every dataclass field present in the yaml is carried
    # through automatically. Do not go back to listing fields by hand -- that is
    # how norm_topology (Phase 9) and scan_impl (job 2513007) were silently lost.
    decoder_config = HybridConfig.from_hydra(cfg.model,
        use_gradient_checkpointing=cfg.model.get("use_gradient_checkpointing", False),
    )

    module = ReportGenerationLightningModule(
        decoder_config=decoder_config,
        image_patch_dim=int(cfg.model.get("image_patch_dim", 768)),
        prefix_k=int(cfg.model.get("prefix_k", 32)),
        vit_unfreeze_blocks=int(cfg.model.get("vit_unfreeze_blocks", 0)),
        decoder_lr=float(cfg.model.get("decoder_lr", 1e-5)),
        head_lr=float(cfg.model.get("head_lr", 3e-4)),
        weight_decay=cfg.model.weight_decay,
        warmup_steps=cfg.model.warmup_steps,
        max_steps=cfg.trainer.max_steps,
        gradient_clip_val=cfg.model.gradient_clip_val,
        # Phase 15C-2. Default 0.0 => no head is built, nothing is logged, and
        # the state dict matches every arm in the 15B-4 seed band exactly.
        aux_lambda=float(cfg.model.get("aux_lambda", 0.0)),
        aux_num_labels=len(CHEXPERT_14_LABELS),
    )

    # Optional decoder init (Phase 10D — the Stage-0/joint-trained 150M backbone).
    # Mirrors train_contrastive.py's lm_checkpoint loading convention exactly.
    decoder_ckpt = cfg.get("decoder_checkpoint", None)
    if decoder_ckpt:
        print(f"Loading decoder backbone weights from: {decoder_ckpt}")
        ckpt = torch.load(decoder_ckpt, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        state = {k.replace("model.", "", 1): v for k, v in state.items()}
        state = {(k[3:] if k.startswith("lm.") else k): v for k, v in state.items()}
        missing, unexpected = module.decoder.load_state_dict(state, strict=False)
        print(f"  Loaded. Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")

        # Phase 14A guard. DECODER_CKPT has a DEFAULT (the hybrid's Stage-0
        # backbone), and the wrapper's existence check passes for it whatever
        # model you are training -- so pointing a Transformer run at the hybrid's
        # checkpoint does NOT fail, it loads under strict=False, matches almost
        # nothing, and trains from random init. That is silent, costs a full
        # multi-GPU run, and would quietly invalidate the matched-baseline
        # comparison. "A run that reports success while doing nothing is the
        # expensive failure mode" -- H100_SCALING_PLAN.md's own lesson.
        n_decoder_keys = len(module.decoder.state_dict())
        missing_frac = len(missing) / max(n_decoder_keys, 1)
        if missing_frac > 0.5:
            raise RuntimeError(
                f"Decoder init matched almost nothing: {len(missing)}/{n_decoder_keys} "
                f"keys missing ({missing_frac:.0%}). This checkpoint is very likely for a "
                f"DIFFERENT architecture than model={cfg.model.get('model_type')} with "
                f"layer_pattern={list(cfg.model.get('layer_pattern', []))}. "
                f"Check DECODER_CKPT ({decoder_ckpt}) -- it defaults to the hybrid's "
                f"Stage-0 backbone and must be overridden for any other architecture."
            )
        if missing_frac > 0.05:
            print(
                f"  WARNING: {len(missing)}/{n_decoder_keys} decoder keys ({missing_frac:.0%}) "
                f"were NOT initialised from the checkpoint. Verify this is intended before "
                f"trusting the run."
            )

    image_encoder_ckpt = cfg.get("image_encoder_checkpoint", None)
    if image_encoder_ckpt:
        print(f"Image tower: fine-tuned checkpoint {image_encoder_ckpt}")
    module.load_image_encoder(
        vit_lr=float(cfg.model.get("vit_lr", 1e-6)),
        image_encoder_checkpoint=image_encoder_ckpt,
    )

    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,} ({num_params/1e6:.1f}M)")

    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.checkpoint_dir,
            monitor="val/lm_loss",
            mode="min",
            # Phase 15B-3: was hardcoded 3. See configs/config.yaml's save_top_k
            # for why -- 4 x 2.4 GB per run exhausted the home quota and killed
            # 3 of 4 seed arms, and no eval in this project has ever loaded
            # anything but last.ckpt. save_top_k=0 keeps last.ckpt only.
            save_top_k=int(cfg.get("save_top_k", 3)),
            save_last=True,
            # NOT {val/lm_loss:.4f} -- Lightning does not sanitize '/' inside a
            # filename interpolation, so that produced a literal nested
            # directory instead of a flat checkpoint file (confirmed live
            # 2026-08-23, job 2478647). val_lm_loss_ckpt is a flat-named alias
            # logged in ReportGenerationLightningModule._step() for exactly
            # this; monitor= above is unaffected (a plain dict-key lookup).
            filename="report_gen-{step:06d}-{val_lm_loss_ckpt:.4f}",
        ),
        LearningRateMonitor(logging_interval="step"),
        SignalCheckpointCallback(checkpoint_dir=cfg.checkpoint_dir),
    ]

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
        check_val_every_n_epoch=cfg.trainer.get("check_val_every_n_epoch", 1),
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        # gradient_clip_val omitted: clipping handled in on_before_optimizer_step
        # (see ReportGenerationLightningModule / HybridLightningModule for why).
        callbacks=callbacks,
        logger=loggers,
        enable_checkpointing=True,
        enable_progress_bar=True,
        num_sanity_val_steps=2,
        default_root_dir=cfg.trainer.default_root_dir,
    )

    print("Preparing dataloaders...")
    train_dl = prepare_report_gen_dataloader(cfg, "train", tokenizer)
    val_dl = prepare_report_gen_dataloader(cfg, "validation", tokenizer)

    # Phase 15C-2: prevalence comes from the TRAIN split only -- computing it
    # on validate/test would leak the split the arm is judged on into the
    # objective. Done after the dataloaders (that is where the label matrix
    # is built) and before fit(), so the head never takes a step at ones.
    if module.aux_lambda > 0:
        pos_weight = compute_aux_pos_weight(
            train_dl.dataset.chexpert_labels,
            cap=float(cfg.model.get("aux_pos_weight_cap", 10.0)),
            num_labels=len(CHEXPERT_14_LABELS),
        )
        module.set_aux_pos_weight(pos_weight)
        # Printed POSITIVELY, per Phase 14's prefix_k lesson: a silent default
        # cannot be detected by its absence, and an aux arm whose log does not
        # state lambda is indistinguishable from a baseline re-run.
        print(f"Aux CheXpert loss: lambda={module.aux_lambda} "
              f"cap={float(cfg.model.get('aux_pos_weight_cap', 10.0))}")
        for label, w in zip(CHEXPERT_14_LABELS, pos_weight):
            print(f"  aux pos_weight  {label:<28} {w:6.2f}")
    else:
        print("Aux CheXpert loss: OFF (aux_lambda=0.0) — baseline recipe, aux_head not built")

    print(f"Dataset      : {cfg.dataset.dataset_name}")
    print(f"Prefix k     : {module.prefix_k}")
    print(f"Max steps    : {cfg.trainer.max_steps:,}")

    resume_ckpt = cfg.get("resume_from_checkpoint", None)
    if resume_ckpt:
        print(f"Resuming training from: {resume_ckpt}")

    trainer.fit(module, train_dataloaders=train_dl, val_dataloaders=val_dl, ckpt_path=resume_ckpt)


if __name__ == "__main__":
    main()
