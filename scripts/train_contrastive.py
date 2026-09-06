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

import hashlib
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

from transformers import AutoModel
from hybrid_xmamba.models.configuration_hybrid import HybridConfig
from hybrid_xmamba.models.hybrid_lm import HybridTextEncoder
from hybrid_xmamba.training.lightning_module import (
    HybridContrastiveLightningModule,
    DistillContrastiveLightningModule,
    JointMultiTaskLightningModule,
)
from hybrid_xmamba.training.signal_callbacks import SignalCheckpointCallback
from hybrid_xmamba.training.contrastive_eval_callback import (
    ContrastiveEvalCallback,
    AnomalyDetectionCallback,
    CLIPRetrievalCallback,
)
from hybrid_xmamba.utils.run_metadata import write_run_metadata

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
        # Handle different field names across datasets
        text = (
            item.get("text") or 
            item.get("abstract") or 
            item.get("article") or
            ""
        )
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


class TextOnlyDatasetWithTeacher(Dataset):
    """Like TextOnlyDataset but also emits teacher (PubMedBERT) tokenization.

    Adds ``teacher_input_ids`` and ``teacher_attention_mask`` to each batch.
    The student and teacher see the same raw text but use different tokenizers
    (GPT-2 BPE vs. WordPiece). Only pooled embeddings are compared — no
    token-level alignment needed.
    """

    def __init__(self, hf_dataset, student_tokenizer, teacher_tokenizer,
                 max_length: int, teacher_max_length: int = 512):
        self.data = hf_dataset
        self.student_tok = student_tokenizer
        self.teacher_tok = teacher_tokenizer
        self.max_length = max_length
        self.teacher_max_length = teacher_max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = (
            item.get("text") or
            item.get("abstract") or
            item.get("article") or
            ""
        )
        # Student tokenization (GPT-2 BPE)
        s_enc = self.student_tok(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        # Teacher tokenization (PubMedBERT WordPiece)
        t_enc = self.teacher_tok(
            text,
            max_length=self.teacher_max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": s_enc["input_ids"].squeeze(0),
            "attention_mask": s_enc["attention_mask"].squeeze(0),
            "teacher_input_ids": t_enc["input_ids"].squeeze(0),
            "teacher_attention_mask": t_enc["attention_mask"].squeeze(0),
        }


def build_image_transform(cfg, is_train: bool = False) -> "T.Compose":
    """CXR image transform, shared by ImageTextDataset and MIMICJointDataset.

    Phase 9D's "untested free lever" (H100_SCALING_PLAN.md — 6G-1 measured the
    retrieval ViT outright memorising this exact arm0-sized pool with zero
    augmentation): when cfg.dataset.use_augmentation is set AND is_train, the
    deterministic Resize is replaced with RandomResizedCrop + mild rotation.

    Applied to Phase 10E's report-generation decoder first (2026-08-24, after
    job 2478647's arm0 checkpoint was confirmed via --checkpoint mode to have
    memorized boilerplate templates rather than condition on the image), NOT
    to retrieval — use_augmentation defaults to False and is undeclared/false
    in cxr_mimic_full.yaml and cxr_mimic_arm0.yaml, so every existing retrieval
    call site (that closed chapter) stays BYTE-IDENTICAL unless a future arm
    explicitly opts in. is_train also gates it off for val/test — augmenting
    eval images would make evaluation nondeterministic and incomparable run to
    run, which is never wanted regardless of the training-time question.
    """
    mean = cfg.dataset.get("image_mean", [0.48145466, 0.4578275, 0.40821073])
    std = cfg.dataset.get("image_std", [0.26862954, 0.26130258, 0.27577711])
    size = cfg.dataset.get("image_size", 224)
    use_augmentation = cfg.dataset.get("use_augmentation", False)

    if use_augmentation and is_train:
        spatial = [
            T.RandomResizedCrop(size, scale=(0.8, 1.0)),
            T.RandomRotation(degrees=7),
        ]
    else:
        spatial = [T.Resize((size, size))]

    return T.Compose(spatial + [
        T.Grayscale(num_output_channels=3),   # CXR is grayscale; BiomedCLIP expects 3-ch
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


class ImageTextDataset(Dataset):
    """Paired image-text dataset for CLIP-style alignment.

    Expects a HuggingFace dataset with fields:
      - image (PIL or path)
      - findings / impression (text)
    """

    def __init__(self, hf_dataset, tokenizer, cfg, is_train: bool = False):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = cfg.dataset.max_length
        self.concat = cfg.dataset.get("concatenate_sections", True)
        self.findings_field = cfg.dataset.get("findings_field", "findings")
        self.impression_field = cfg.dataset.get("impression_field", "impression")
        self.img_transform = build_image_transform(cfg, is_train=is_train)

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
        attention_mask = enc["attention_mask"].squeeze(0)

        # --- image ---
        img = item.get("image")
        
        # Handle different image formats
        if img is None:
            raise ValueError(f"Image is None for sample (should have been filtered)")
        elif isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif not isinstance(img, Image.Image):
            img = Image.fromarray(img).convert("RGB")
        else:
            img = img.convert("RGB")
            
        pixel_values = self.img_transform(img)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pubmed(cfg, split: str, tokenizer, teacher_tokenizer=None, teacher_max_length: int = 512):
    """Load PubMed abstracts from HuggingFace.

    Uses ccdv/pubmed-summarization which is in Parquet format (no loading script).
    This dataset contains ~133k PubMed articles with abstracts.

    If ``teacher_tokenizer`` is provided, returns a TextOnlyDatasetWithTeacher
    that emits both student and teacher token tensors per sample.
    """
    print(f"Loading PubMed dataset for {split} split...")

    # Use pubmed-summarization dataset (Parquet format, no loading script)
    ds = load_dataset(
        "ccdv/pubmed-summarization",
        split="train" if split == "train" else "validation",
        streaming=cfg.dataset.streaming,
        cache_dir=cfg.dataset.cache_dir,
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

    print(f"Dataset loaded: {len(ds) if hasattr(ds, '__len__') else 'streaming'} samples")

    if teacher_tokenizer is not None:
        return TextOnlyDatasetWithTeacher(
            ds, tokenizer, teacher_tokenizer,
            max_length=cfg.dataset.max_length,
            teacher_max_length=teacher_max_length,
        )
    return TextOnlyDataset(ds, tokenizer, cfg.dataset.max_length)


class IUXrayPathDataset(Dataset):
    """Handles dz-osamu/IU-Xray schema: images are repo file paths, text is in 'response'."""

    def __init__(self, hf_dataset, repo_local: str, tokenizer, cfg):
        self.tokenizer = tokenizer
        self.max_length = cfg.dataset.max_length
        self.repo_local = repo_local
        self.report_field = cfg.dataset.get("report_field", "response")
        self.image_index = cfg.dataset.get("image_index", 0)

        mean = cfg.dataset.get("image_mean", [0.48145466, 0.4578275, 0.40821073])
        std  = cfg.dataset.get("image_std",  [0.26862954, 0.26130258, 0.27577711])
        size = cfg.dataset.get("image_size", 224)
        self.img_transform = T.Compose([
            T.Resize((size, size)),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

        # Pre-filter to samples where the image file actually exists
        self.samples = []
        for i in range(len(hf_dataset)):
            row = hf_dataset[i]
            paths = row.get("images", [])
            if not paths:
                continue
            abs_path = os.path.join(repo_local, paths[self.image_index].lstrip("/"))
            if os.path.exists(abs_path):
                self.samples.append((row, abs_path))
        print(f"IUXrayPathDataset: {len(self.samples)} samples with valid image files")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row, abs_path = self.samples[idx]
        text = row.get(self.report_field, "") or ""
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        img = Image.open(abs_path).convert("RGB")
        pixel_values = self.img_transform(img)
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "pixel_values": pixel_values,
        }


class MIMICJointDataset(Dataset):
    """MIMIC-CXR image-text dataset for joint KD+CLIP+SimCSE training.

    Emits student tokens, image pixel_values, and teacher (PubMedBERT) tokens
    in a single batch so all three losses share one forward pass.
    """

    def __init__(self, hf_dataset, student_tokenizer, teacher_tokenizer, cfg, is_train: bool = False):
        self.data = hf_dataset
        self.student_tok = student_tokenizer
        # BiomedCLIP open_clip tokenizer is Callable[[List[str]], LongTensor]
        # (no `pad_token_id`, no kwargs). HuggingFace AutoTokenizer has both.
        # Wrap the open_clip path so __getitem__ stays uniform.
        self._teacher_is_hf = hasattr(teacher_tokenizer, "pad_token_id")
        self.teacher_tok = teacher_tokenizer
        self.max_length = cfg.dataset.max_length
        # BiomedCLIP context window is fixed at 256; legacy PubMedBERT uses cfg value.
        self.teacher_max_length = (
            cfg.dataset.get("teacher_max_length", 512) if self._teacher_is_hf else 256
        )
        self.findings_field = cfg.dataset.get("findings_field", "findings")
        self.impression_field = cfg.dataset.get("impression_field", "impression")
        self.concat = cfg.dataset.get("concatenate_sections", True)
        self.img_transform = build_image_transform(cfg, is_train=is_train)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        findings   = item.get(self.findings_field, "")  or ""
        impression = item.get(self.impression_field, "") or ""
        text = (
            f"Findings: {findings} Impression: {impression}".strip()
            if self.concat else (findings or impression)
        )

        s_enc = self.student_tok(
            text, max_length=self.max_length,
            truncation=True, padding="max_length", return_tensors="pt",
        )
        if self._teacher_is_hf:
            t_enc = self.teacher_tok(
                text, max_length=self.teacher_max_length,
                truncation=True, padding="max_length", return_tensors="pt",
            )
            t_ids = t_enc["input_ids"].squeeze(0)
            t_mask = t_enc["attention_mask"].squeeze(0)
        else:
            # open_clip tokenizer: Callable[[List[str]], LongTensor of shape (1, L)]
            # Pads/truncates to its built-in context (256 for BiomedCLIP).
            t_ids = self.teacher_tok([text])[0]
            t_mask = (t_ids != 0).long()

        img = item.get("image")
        if img is None:
            raise ValueError(f"Image is None for sample {idx}")
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif not isinstance(img, Image.Image):
            img = Image.fromarray(img).convert("RGB")
        else:
            img = img.convert("RGB")
        pixel_values = self.img_transform(img)

        # Phase 6D-3: stable hash of the case/whitespace-normalised report.
        # MIMIC reports are heavily templated, so many studies share literally
        # identical text; the CLIP loss currently pushes those apart as
        # negatives (hard arange(B) targets, lightning_module.py:540). Equal
        # hashes let the loss treat them as a positive SET instead. blake2b is
        # used rather than Python's hash() because the latter is salted per
        # process and would differ across dataloader workers.
        norm = " ".join(text.lower().split())
        text_hash = int.from_bytes(
            hashlib.blake2b(norm.encode("utf-8"), digest_size=8).digest(), "big"
        ) % (2 ** 62)

        return {
            "input_ids":              s_enc["input_ids"].squeeze(0),
            "attention_mask":         s_enc["attention_mask"].squeeze(0),
            "pixel_values":           pixel_values,
            "teacher_input_ids":      t_ids,
            "teacher_attention_mask": t_mask,
            "text_hash":              torch.tensor(text_hash, dtype=torch.long),
        }


def load_mimic_cxr(cfg, split, tokenizer, teacher_tokenizer=None):
    """Load MIMIC-CXR for joint training.

    Two sources, selected by whether dataset.local_parquet_dir is set:
      - local_parquet_dir: the Phase-8 PhysioNet build (build_mimic_cxr_local.py),
        train.parquet/validate.parquet/test.parquet under that directory. "image"
        is a local absolute path (str), handled by MIMICJointDataset/ImageTextDataset
        without further changes — they already branch on isinstance(image, str).
      - otherwise: the itsanmolgupta/mimic-cxr-dataset HF mirror (legacy path,
        kept for the Arm-0 reproduction control and for runs that predate
        credentialing).

    When teacher_tokenizer is provided, returns MIMICJointDataset (dual tokens + image).
    Otherwise returns ImageTextDataset (image + student tokens only).
    """
    local_dir = cfg.dataset.get("local_parquet_dir", None)
    split_map = {
        "train":      cfg.dataset.get("train_split", "train"),
        "validation": cfg.dataset.get("validation_split", "validation"),
        "test":       cfg.dataset.get("test_split", "test"),
    }
    hf_split = split_map.get(split, split)

    if local_dir:
        data_files = {
            "train":      f"{local_dir}/train.parquet",
            "validation": f"{local_dir}/validate.parquet",
            "test":       f"{local_dir}/test.parquet",
        }
        # HF slice syntax (e.g. "train[:85%]") still works against a named
        # data_files key, so Phase 9E's disjoint-selection-split override
        # pattern (dataset.train_split='train[:85%]' dataset.validation_split=
        # 'train[85%:90%]') carries over unchanged; "test" should stay the
        # literal, unsliced official held-out split.
        ds = load_dataset("parquet", data_files=data_files, split=hf_split)
        print(f"Loaded local MIMIC-CXR-JPG build (split={hf_split}) from {local_dir}: "
              f"{len(ds)} samples, columns: {ds.column_names}")
    else:
        hf_repo = cfg.dataset.get("hf_repo_id", "itsanmolgupta/mimic-cxr-dataset")
        ds = load_dataset(hf_repo, split=hf_split, cache_dir=cfg.dataset.cache_dir)
        print(f"Loaded {hf_repo} (split={hf_split}): {len(ds)} samples, "
              f"columns: {ds.column_names}")

    ds = ds.filter(lambda x: x.get("image") is not None)
    print(f"After filtering None images: {len(ds)} samples")
    if len(ds) == 0:
        raise RuntimeError(
            f"All samples have image=None in "
            f"{'local build at ' + local_dir if local_dir else hf_repo}. "
            "Check dataset config."
        )

    is_train = (split == "train")
    if teacher_tokenizer is not None:
        return MIMICJointDataset(ds, tokenizer, teacher_tokenizer, cfg, is_train=is_train)
    return ImageTextDataset(ds, tokenizer, cfg, is_train=is_train)


def load_indiana_cxr(cfg, split: str, tokenizer):
    """Load CXR dataset from HuggingFace for Stage 2 CLIP alignment.

    Dataset configured via cfg.dataset.hf_repo_id.
    Default: MLforHealthcare/Indiana_University_Chest_X-ray_Collection
      - 7,430 IU-Xray images (train=6,687 / test=743)
      - PIL images embedded, fields: image / report
      - Public, no token required

    Handles two schemas:
    - PIL-based (image column): uses ImageTextDataset directly.
    - Path-based (images column): uses IUXrayPathDataset with snapshot_download.
    """
    hf_repo = cfg.dataset.get("hf_repo_id",
                               "MLforHealthcare/Indiana_University_Chest_X-ray_Collection")

    # Map logical split names to HF split names via yaml config
    split_map = {
        "train":      cfg.dataset.get("train_split", "train"),
        "validation": cfg.dataset.get("validation_split", "test"),
        "test":       cfg.dataset.get("test_split", "test"),
    }
    hf_split = split_map.get(split, split)

    ds = load_dataset(hf_repo, split=hf_split, cache_dir=cfg.dataset.cache_dir)
    print(f"Loaded {hf_repo} (split={hf_split}): {len(ds)} samples, "
          f"columns: {ds.column_names}")

    # --- PIL-based schema ---
    if "image" in ds.column_names:
        ds = ds.filter(lambda x: x.get("image") is not None)
        print(f"After filtering None images: {len(ds)} samples")
        if len(ds) == 0:
            raise RuntimeError(
                f"All samples have image=None in {hf_repo}. "
                "Check hf_repo_id in indiana_cxr.yaml."
            )
        return ImageTextDataset(ds, tokenizer, cfg)

    # --- Path-based schema (fallback for path-based repos) ---
    if "images" in ds.column_names:
        from huggingface_hub import snapshot_download
        repo_cache = os.path.join(cfg.dataset.cache_dir, "repo_snapshot")
        print(f"Downloading repo snapshot for {hf_repo} → {repo_cache} ...")
        repo_local = snapshot_download(hf_repo, local_dir=repo_cache)
        print(f"Snapshot ready: {repo_local}")
        dataset = IUXrayPathDataset(ds, repo_local, tokenizer, cfg)
        if len(dataset) == 0:
            raise RuntimeError(
                f"IUXrayPathDataset: 0 valid samples after resolving paths from "
                f"{repo_local}. Check hf_repo_id and image paths."
            )
        return dataset

    raise RuntimeError(
        f"Neither 'image' nor 'images' column found in {hf_repo}. "
        f"Columns: {ds.column_names}. Update hf_repo_id in indiana_cxr.yaml."
    )


def prepare_dataloader(cfg, split: str, tokenizer, teacher_tokenizer=None):
    name = cfg.dataset.dataset_name
    distill_cfg = cfg.get("distill", None)
    teacher_max_length = int(distill_cfg.get("teacher_max_length", 512)) if distill_cfg else 512

    if name == "pubmed":
        dataset = load_pubmed(
            cfg, split, tokenizer,
            teacher_tokenizer=teacher_tokenizer,
            teacher_max_length=teacher_max_length,
        )
    elif name == "indiana_cxr":
        dataset = load_indiana_cxr(cfg, split, tokenizer)
    elif name == "mimic_cxr":
        dataset = load_mimic_cxr(
            cfg, split, tokenizer,
            teacher_tokenizer=teacher_tokenizer,
        )
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
    write_run_metadata(cfg, cfg.output_dir, extra={"entrypoint": "train_contrastive.py"})

    # When Stage 1 distillation is active, clamp student max_length to the
    # teacher's max (PubMedBERT = 512) so both see the same text window and
    # the cosine-distillation signal is comparing aligned representations.
    distill_cfg_top = cfg.get("distill", None)
    if distill_cfg_top is not None and "teacher_max_length" in distill_cfg_top:
        t_max = int(distill_cfg_top.teacher_max_length)
        cur_max = int(cfg.dataset.max_length)
        if cur_max > t_max:
            print(
                f"[Stage1 distill] Clamping dataset.max_length {cur_max} -> {t_max} "
                f"to match teacher_max_length."
            )
            cfg.dataset.max_length = t_max
            if "max_seq_length" in cfg.dataset:
                cfg.dataset.max_seq_length = t_max

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.dataset.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # last-token pooling requires right-pad
    tokenizer.model_max_length = cfg.model.max_position_embeddings

    # Model config
    # MAMBA3_PLAN.md M2-F: every dataclass field present in the yaml is carried
    # through automatically. Do not go back to listing fields by hand -- that is
    # how norm_topology (Phase 9) and scan_impl (job 2513007) were silently lost.
    model_config = HybridConfig.from_hydra(cfg.model,
        use_gradient_checkpointing=cfg.model.get("use_gradient_checkpointing", False),
        proj_head_dropout=cfg.model.get("proj_head_dropout", 0.1),
        pooling_strategy=cfg.model.get("pooling_strategy", "mean"),
        bidirectional_encode=cfg.model.get("bidirectional_encode", False),
    )

    # Build text encoder
    text_encoder = HybridTextEncoder(model_config, embed_dim=512)

    # Optionally load weights from a prior LM checkpoint
    lm_ckpt = cfg.get("lm_checkpoint", None)
    if lm_ckpt:
        print(f"Loading LM backbone weights from: {lm_ckpt}")
        ckpt = torch.load(lm_ckpt, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        # Strip Lightning "model." prefix (e.g. full .ckpt from Stage 1)
        state = {k.replace("model.", "", 1): v for k, v in state.items()}
        # Strip additional "lm." prefix present when loading Stage 1 .ckpt into text_encoder.lm
        state = {(k[3:] if k.startswith("lm.") else k): v for k, v in state.items()}
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

    # --- Module dispatch ---
    distill_cfg = cfg.get("distill", None)
    teacher_tokenizer = None

    if contrastive_mode == "joint":
        if distill_cfg is None:
            raise ValueError(
                "contrastive_mode=joint requires a distill config. "
                "Add distill=joint_mimic to your command."
            )

        teacher_kind = distill_cfg.get("teacher", "pubmedbert")
        if teacher_kind == "biomedclip_text":
            import open_clip
            from hybrid_xmamba.training.lightning_module import (
                _load_biomedclip_text_teacher,
            )

            print("\nLoading joint teacher: BiomedCLIP text tower (512-d joint)...")
            teacher_model = _load_biomedclip_text_teacher()
            teacher_tokenizer = open_clip.get_tokenizer(
                'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
            )
            t_params = sum(p.numel() for p in teacher_model.parameters())
            print(f"Teacher params: {t_params:,} ({t_params/1e6:.0f}M), frozen.")
        else:
            teacher_name = distill_cfg.teacher_model
            teacher_dtype_str = distill_cfg.get("teacher_dtype", "bfloat16")
            teacher_dtype = torch.bfloat16 if teacher_dtype_str == "bfloat16" else torch.float16

            print(f"\nLoading joint teacher: {teacher_name} ({teacher_dtype_str})...")
            teacher_model = AutoModel.from_pretrained(
                teacher_name,
                torch_dtype=teacher_dtype,
                low_cpu_mem_usage=True,
            )
            teacher_model.eval()
            for p in teacher_model.parameters():
                p.requires_grad_(False)

            teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_name)
            t_params = sum(p.numel() for p in teacher_model.parameters())
            print(f"Teacher params: {t_params:,} ({t_params/1e6:.0f}M), frozen.")

        lightning_module = JointMultiTaskLightningModule(
            model=text_encoder,
            teacher=teacher_model,
            alpha_kd=float(distill_cfg.get("alpha_kd", 0.3)),
            alpha_kd_warmup=(
                float(distill_cfg.get("alpha_kd_warmup"))
                if distill_cfg.get("alpha_kd_warmup", None) is not None else None
            ),
            alpha_kd_post=(
                float(distill_cfg.get("alpha_kd_post"))
                if distill_cfg.get("alpha_kd_post", None) is not None else None
            ),
            beta_clip=float(distill_cfg.get("beta_clip", 1.0)),
            gamma_simcse=float(distill_cfg.get("gamma_simcse", 0.1)),
            backbone_lr=float(distill_cfg.get("backbone_lr", 1e-5)),
            head_lr=float(distill_cfg.get("head_lr", 3e-4)),
            weight_decay=cfg.model.weight_decay,
            warmup_steps=cfg.model.warmup_steps,
            max_steps=cfg.trainer.max_steps,
            gradient_clip_val=cfg.model.gradient_clip_val,
            freeze_text_encoder_steps=int(distill_cfg.get("freeze_text_encoder_steps", 500)),
            # Phase 10F: prefer the distill yaml (where the v2 recipe lives);
            # fall back to cfg.model for backward compatibility.
            vit_unfreeze_blocks=int(
                distill_cfg.get("vit_unfreeze_blocks", cfg.model.get("vit_unfreeze_blocks", 0))
            ),
            vit_lr=float(distill_cfg.get("vit_lr", cfg.model.get("vit_lr", 1e-6))),
            # Phase 6G-2: "blocks" (default) or "all" (whole image tower).
            vit_unfreeze_scope=str(distill_cfg.get("vit_unfreeze_scope", "blocks")),
            moco_queue_size=int(distill_cfg.get("moco_queue_size", 0)),
            moco_momentum=float(distill_cfg.get("moco_momentum", 0.999)),
            # Phase 10B: frequency-decoupled KD.
            freq_kd=bool(distill_cfg.get("freq_kd", False)),
            freq_kd_low_bins=int(distill_cfg.get("freq_kd_low_bins", 32)),
            freq_kd_alpha_high=float(distill_cfg.get("freq_kd_alpha_high", 0.1)),
            # Phase 6D-2 / 6D-3: KD-anchor decay + objective repair. All four
            # defaults reproduce the Phase-6B recipe exactly, so 6D-0 is a true
            # control.
            kd_decay_steps=int(distill_cfg.get("kd_decay_steps", 0)),
            alpha_kd_floor=float(distill_cfg.get("alpha_kd_floor", 0.0)),
            clip_loss_type=str(distill_cfg.get("clip_loss_type", "infonce")),
            use_multipos=bool(distill_cfg.get("use_multipos", False)),
        )
        print(
            f"JointMultiTaskLightningModule: "
            f"α_kd={distill_cfg.get('alpha_kd', 0.3)} "
            f"(warmup={distill_cfg.get('alpha_kd_warmup', distill_cfg.get('alpha_kd', 0.3))}, "
            f"post={distill_cfg.get('alpha_kd_post', distill_cfg.get('alpha_kd', 0.3))}) "
            f"β_clip={distill_cfg.get('beta_clip', 1.0)} "
            f"γ_simcse={distill_cfg.get('gamma_simcse', 0.1)} "
            f"freeze_text_encoder_steps={distill_cfg.get('freeze_text_encoder_steps', 500)}"
        )
        # Phase 6D levers echoed explicitly. This repo has twice lost days to a
        # silently-drifted override (hardcoded LRs, freq_kd default), so every
        # new knob goes in the SLURM log where the run can be audited from it.
        print(
            f"  Phase 6D/6G levers: vit_unfreeze_blocks="
            f"{distill_cfg.get('vit_unfreeze_blocks', cfg.model.get('vit_unfreeze_blocks', 0))} "
            f"vit_lr={distill_cfg.get('vit_lr', 1e-6)} "
            f"vit_scope={distill_cfg.get('vit_unfreeze_scope', 'blocks')} "
            f"kd_decay_steps={distill_cfg.get('kd_decay_steps', 0)} "
            f"alpha_kd_floor={distill_cfg.get('alpha_kd_floor', 0.0)} "
            f"clip_loss_type={distill_cfg.get('clip_loss_type', 'infonce')} "
            f"use_multipos={distill_cfg.get('use_multipos', False)}"
        )

    elif distill_cfg is not None and contrastive_mode == "simcse":
        teacher_name = distill_cfg.teacher_model
        teacher_dtype_str = distill_cfg.get("teacher_dtype", "bfloat16")
        teacher_dtype = torch.bfloat16 if teacher_dtype_str == "bfloat16" else torch.float16

        print(f"\nLoading Stage 1 teacher: {teacher_name} ({teacher_dtype_str})...")
        teacher_model = AutoModel.from_pretrained(
            teacher_name,
            torch_dtype=teacher_dtype,
            low_cpu_mem_usage=True,
        )
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)

        teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_name)
        t_params = sum(p.numel() for p in teacher_model.parameters())
        print(f"Teacher params: {t_params:,} ({t_params/1e6:.0f}M), frozen.")

        lightning_module = DistillContrastiveLightningModule(
            teacher=teacher_model,
            lambda_max=float(distill_cfg.get("lambda_max", 0.3)),
            distill_warmup=int(distill_cfg.get("warmup_steps", 500)),
            distill_ramp=int(distill_cfg.get("ramp_steps", 500)),
            model=text_encoder,
            contrastive_mode=contrastive_mode,
            image_encoder_name=image_encoder_name,
            learning_rate=cfg.model.learning_rate,
            weight_decay=cfg.model.weight_decay,
            warmup_steps=cfg.model.warmup_steps,
            max_steps=cfg.trainer.max_steps,
            gradient_clip_val=cfg.model.gradient_clip_val,
            freeze_text_encoder_steps=int(cfg.model.get("freeze_text_encoder_steps", 0)),
            vit_unfreeze_blocks=int(cfg.model.get("vit_unfreeze_blocks", 0)),
            vit_lr=float(cfg.model.get("vit_lr", 1e-6)),
        )
        print(f"Using DistillContrastiveLightningModule "
              f"(lambda_max={distill_cfg.get('lambda_max', 0.3)}, "
              f"warmup={distill_cfg.get('warmup_steps', 500)}, "
              f"ramp={distill_cfg.get('ramp_steps', 500)})")

    else:
        lightning_module = HybridContrastiveLightningModule(
            model=text_encoder,
            contrastive_mode=contrastive_mode,
            image_encoder_name=image_encoder_name,
            learning_rate=cfg.model.learning_rate,
            weight_decay=cfg.model.weight_decay,
            warmup_steps=cfg.model.warmup_steps,
            max_steps=cfg.trainer.max_steps,
            gradient_clip_val=cfg.model.gradient_clip_val,
            freeze_text_encoder_steps=int(cfg.model.get("freeze_text_encoder_steps", 0)),
            vit_unfreeze_blocks=int(cfg.model.get("vit_unfreeze_blocks", 0)),
            vit_lr=float(cfg.model.get("vit_lr", 1e-6)),
        )

    # Callbacks
    use_total_loss_monitor = (distill_cfg is not None or contrastive_mode == "joint")
    monitor_metric = "val/total_loss" if use_total_loss_monitor else "val/contrastive_loss"
    ckpt_fname = (
        "contrastive-{step:06d}-{val/total_loss:.4f}"
        if use_total_loss_monitor
        else "contrastive-{step:06d}-{val/contrastive_loss:.4f}"
    )
    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.checkpoint_dir,
            monitor=monitor_metric,
            mode="min",
            save_top_k=3,
            save_last=True,
            filename=ckpt_fname,
        ),
        LearningRateMonitor(logging_interval="step"),
        SignalCheckpointCallback(checkpoint_dir=cfg.checkpoint_dir),
    ]

    # Stage 1 SimCSE: in-training biomedical eval + anomaly detection
    if contrastive_mode == "simcse":
        callbacks.append(
            ContrastiveEvalCallback(
                tokenizer=tokenizer,
                eval_every_n_steps=500,
                align_unif_every_n_steps=1000,
            )
        )
        callbacks.append(AnomalyDetectionCallback(max_steps=200))

    # CLIP / joint: retrieval metrics (R@1/5/10 i2t and t2i)
    if contrastive_mode in ("clip", "joint"):
        callbacks.append(CLIPRetrievalCallback(eval_every_n_epochs=1, max_samples=0))

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
        # Must be threaded through: with the default (1), an int val_check_interval is
        # read as "batches WITHIN an epoch" and Lightning errors when it exceeds the
        # epoch length (MIMIC 27.5k @ bs128 = 215 batches < 250). The configs set this
        # to null, which makes val_check_interval count GLOBAL STEPS instead — correct
        # for a max_steps-driven run and safe at any batch size (bs=256 -> 107 batches).
        check_val_every_n_epoch=cfg.trainer.get("check_val_every_n_epoch", 1),
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        # gradient_clip_val omitted: clipping handled in on_before_optimizer_step
        callbacks=callbacks,
        logger=loggers,
        enable_checkpointing=True,
        enable_progress_bar=True,
        num_sanity_val_steps=2,
        default_root_dir=cfg.trainer.default_root_dir,
    )

    print("Preparing dataloaders...")
    train_dl = prepare_dataloader(cfg, "train", tokenizer, teacher_tokenizer=teacher_tokenizer)
    val_dl   = prepare_dataloader(cfg, "validation", tokenizer, teacher_tokenizer=teacher_tokenizer)

    print(f"Contrastive mode : {contrastive_mode}")
    print(f"Dataset          : {cfg.dataset.dataset_name}")
    print(f"Max steps        : {cfg.trainer.max_steps:,}")
    
    # Check if we should resume from a checkpoint
    resume_ckpt = None
    if cfg.get("resume_from_checkpoint"):
        resume_ckpt = cfg.resume_from_checkpoint
        print(f"Resuming training from: {resume_ckpt}")
    
    print("Starting contrastive training...")

    trainer.fit(
        lightning_module, 
        train_dataloaders=train_dl, 
        val_dataloaders=val_dl,
        ckpt_path=resume_ckpt
    )
    print("Done.")


if __name__ == "__main__":
    main()
