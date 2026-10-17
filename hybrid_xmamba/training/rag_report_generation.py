"""TEXTENC_PLAN.md T4: retrieval-augmented report generation.

The decoder reads  [32 image prefix tokens][retrieved report 1]...[retrieved report k][report]
and is trained with the next-token loss on the report only. Retrieved reports come from the
re-ranked index of scripts/textenc_rag_index.py (training queries never retrieve their own
patient). Padding goes only at the end of the sequence, so a causal decoder never reads a pad
token before a real one.

Nothing here changes the published path: RAGReportGenerationLightningModule defers to
ReportGenerationLightningModule._step for any batch without the RAG keys, and RAGDataset wraps
the published ImageTextDataset without modifying it.
"""
from typing import Dict, List, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from hybrid_xmamba.training.lightning_module import ReportGenerationLightningModule

CONTEXT_LEAD = "Similar case: "
CONTEXT_SEP = "\n"


def context_token_ids(tokenizer, reports: Sequence[str], per_report_tokens: int) -> List[int]:
    """Token ids of the retrieved reports, each led by CONTEXT_LEAD, cut to per_report_tokens
    (the separator included), and followed by CONTEXT_SEP."""
    sep = tokenizer.encode(CONTEXT_SEP)
    ids: List[int] = []
    for text in reports:
        body = tokenizer.encode(CONTEXT_LEAD + " ".join(str(text).split()))
        ids.extend(body[: max(per_report_tokens - len(sep), 1)] + sep)
    return ids


def build_rag_example(context_ids: Sequence[int], report_ids: Sequence[int], total_len: int,
                      pad_id: int) -> Dict[str, torch.Tensor]:
    """Concatenate context and report, right-pad to total_len; labels cover the report only."""
    seq = list(context_ids) + list(report_ids)
    if len(seq) > total_len:
        raise ValueError(f"RAG sequence of {len(seq)} tokens exceeds total_len={total_len}")
    n_pad = total_len - len(seq)
    labels = [-100] * len(context_ids) + list(report_ids) + [-100] * n_pad
    return {
        "rag_input_ids": torch.tensor(seq + [pad_id] * n_pad, dtype=torch.long),
        "rag_labels": torch.tensor(labels, dtype=torch.long),
        "rag_attention_mask": torch.tensor([1] * len(seq) + [0] * n_pad, dtype=torch.long),
        "rag_context_len": torch.tensor(len(context_ids), dtype=torch.long),
    }


class RAGDataset(Dataset):
    """Wraps the published ImageTextDataset and adds the RAG keys to every item.

    Args:
        base: ImageTextDataset (its `data` must expose a `study_id` column).
        topk_by_study: study_id -> gallery rows (best first) from the re-ranked index.
        gallery_reports: report text per gallery (train parquet) row.
        tokenizer: the decoder tokenizer (GPT-2).
        k: retrieved reports per study.
        per_report_tokens: token budget per retrieved report, separator included.
    """

    def __init__(self, base, topk_by_study: Mapping[int, Sequence[int]],
                 gallery_reports: Sequence[str], tokenizer, k: int, per_report_tokens: int):
        self.base = base
        self.topk_by_study = topk_by_study
        self.gallery_reports = gallery_reports
        self.tokenizer = tokenizer
        self.k = int(k)
        self.per_report_tokens = int(per_report_tokens)
        self.study_ids = [int(s) for s in base.data["study_id"]]
        self.total_len = self.k * self.per_report_tokens + int(base.max_length)
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        missing = sum(1 for s in set(self.study_ids) if s not in topk_by_study)
        if missing:
            raise KeyError(f"{missing} studies of this split have no entry in the RAG index")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        out = dict(self.base[idx])
        rows = list(self.topk_by_study[self.study_ids[idx]])[: self.k]
        ctx = context_token_ids(self.tokenizer, [self.gallery_reports[r] for r in rows],
                                self.per_report_tokens)
        report = out["input_ids"][out["attention_mask"].bool()].tolist()
        out.update(build_rag_example(ctx, report, self.total_len, self.pad_id))
        return out


class RAGReportGenerationLightningModule(ReportGenerationLightningModule):
    """ReportGenerationLightningModule that also accepts RAG batches (see module docstring)."""

    def _step(self, batch: Dict[str, torch.Tensor], split: str) -> torch.Tensor:
        if "rag_input_ids" not in batch:
            return super()._step(batch, split)
        if getattr(self, "aux_head", None) is not None:
            raise ValueError("the auxiliary CheXpert loss is not supported with RAG batches")
        ids = batch["rag_input_ids"]
        B, L = ids.shape
        patch_grid = batch["patch_grid"] if "patch_grid" in batch else self._patch_grid(batch["pixel_values"])
        prefix_embeds = self.prefix_mapper(patch_grid)
        k = prefix_embeds.shape[1]
        inputs_embeds = torch.cat([prefix_embeds, self.decoder.embeddings(ids)], dim=1)
        labels = torch.full((B, k + L), -100, dtype=ids.dtype, device=ids.device)
        labels[:, k:] = batch["rag_labels"]
        loss = self.decoder(inputs_embeds=inputs_embeds, labels=labels, return_dict=True).loss

        self.log(f"{split}/lm_loss", loss, prog_bar=(split == "train"),
                 on_step=(split == "train"), on_epoch=True)
        if split == "val":
            self.log("val_lm_loss_ckpt", loss, on_epoch=True)
        if split == "train":
            try:
                self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True)
            except RuntimeError:
                pass
        return loss
