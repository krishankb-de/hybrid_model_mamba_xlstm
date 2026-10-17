"""TEXTENC_PLAN.md T4-C: Stage 3 with retrieved reports in the decoder context.

Runs the published `scripts/train_report_generation.py` main() unchanged, with two names in that
module swapped before it starts:
  * ReportGenerationLightningModule -> RAGReportGenerationLightningModule (accepts RAG batches)
  * prepare_report_gen_dataloader   -> the same loader with its dataset wrapped in RAGDataset
So every Hydra argument, the decoder init, the image tower load, the optimiser and the checkpoint
callback are exactly the published code path.

RAG settings come from the environment (Hydra's strict mode would reject new keys):
  TEXTENC_RAG_DIR    directory with rag_train.pt / rag_validate.pt from textenc_rag_index.py
  TEXTENC_DATA       dataset dir with train/validate parquets (gallery = train)
  RAG_K              retrieved reports per study (default 3)
  RAG_REPORT_TOKENS  token budget per retrieved report (default 160)
"""
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SPLIT_FILES = {"train": "train", "validation": "validate", "validate": "validate", "test": "test"}


def load_topk_by_study(rag_dir: Path, data_dir: Path, split: str):
    """study_id -> gallery rows (best first), from rag_<split>.pt and the split parquet order."""
    import pandas as pd
    import torch

    name = SPLIT_FILES[split]
    res = torch.load(rag_dir / f"rag_{name}.pt")
    df = pd.read_parquet(data_dir / f"{name}.parquet", columns=["study_id"])
    topk = res["topk_rows"].tolist()
    if len(topk) != len(df):
        raise ValueError(f"rag_{name}.pt has {len(topk)} rows, {name}.parquet has {len(df)}")
    out = {}
    for sid, rows in zip(df["study_id"].tolist(), topk):
        out.setdefault(int(sid), rows)          # first image of a study decides
    return out


def gallery_reports(data_dir: Path):
    import pandas as pd
    from scripts.textenc_rag_index import report_text

    tr = pd.read_parquet(data_dir / "train.parquet", columns=["findings", "impression"])
    return [report_text(f, i) for f, i in zip(tr["findings"], tr["impression"])]


def install(trg, rag_dir: Path, data_dir: Path, k: int, per_report_tokens: int):
    """Swap the module class and dataloader factory inside train_report_generation."""
    from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
    from hybrid_xmamba.training.rag_report_generation import (RAGDataset,
                                                             RAGReportGenerationLightningModule)

    original = trg.prepare_report_gen_dataloader
    gallery = gallery_reports(data_dir)

    def prepare(cfg, split, tokenizer):
        dl = original(cfg, split, tokenizer)
        if not isinstance(dl.sampler, (RandomSampler, SequentialSampler)):
            raise ValueError("RAG training does not support a custom sampler (oversampling)")
        ds = RAGDataset(dl.dataset, load_topk_by_study(rag_dir, data_dir, split), gallery,
                        tokenizer, k=k, per_report_tokens=per_report_tokens)
        if split == "train":
            meta = {"rag_dir": str(rag_dir), "k": k, "per_report_tokens": per_report_tokens,
                    "total_len": ds.total_len, "entrypoint": "textenc_train_rag.py"}
            Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
            (Path(cfg.output_dir) / "rag_metadata.json").write_text(json.dumps(meta, indent=2))
            print(f"RAG: k={k}, {per_report_tokens} tokens per retrieved report, "
                  f"sequence {ds.total_len} tokens + prefix")
        return DataLoader(ds, batch_size=dl.batch_size,
                          shuffle=isinstance(dl.sampler, RandomSampler),
                          num_workers=dl.num_workers, pin_memory=dl.pin_memory,
                          drop_last=dl.drop_last, persistent_workers=dl.persistent_workers)

    trg.prepare_report_gen_dataloader = prepare
    trg.ReportGenerationLightningModule = RAGReportGenerationLightningModule


def main():
    import scripts.train_report_generation as trg

    rag_dir = Path(os.environ["TEXTENC_RAG_DIR"])
    data_dir = Path(os.environ["TEXTENC_DATA"])
    install(trg, rag_dir, data_dir, k=int(os.environ.get("RAG_K", 3)),
            per_report_tokens=int(os.environ.get("RAG_REPORT_TOKENS", 160)))
    trg.main()


if __name__ == "__main__":
    main()
