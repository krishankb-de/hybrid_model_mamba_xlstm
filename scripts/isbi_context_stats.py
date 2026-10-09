"""ISBI_BASELINES_PLAN.md B7-A: how long is the text a chest X-ray report could read?

Aggregate statistics only (rule R6): nothing per study leaves this script.

1. Report length in GPT-2 tokens, in the training format ("Findings: ... Impression: ...").
2. Patient history: for every test study, the earlier studies of the same patient (MIMIC-CXR
   StudyDate/StudyTime) and the GPT-2 token count of all their reports joined. That is the
   context a report generator would read if it were given the patient's prior reports.
   The split is patient-disjoint, so a test patient's priors are test studies.
3. How often reference reports compare with an earlier study (word list below).

    python scripts/isbi_context_stats.py --data /sc/home/$USER/dataset/mimic_full --output stats.json
"""
import argparse
import json
import re
from pathlib import Path

COMPARISON = re.compile(r"\b(compar\w*|prior|previous\w*|interval\w*|unchanged|stable|"
                        r"again|no change|redemonstrat\w*|persist\w*|since)\b", re.I)


def pct(values, q):
    import numpy as np
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else float("nan")


def summary(values):
    import numpy as np
    v = np.asarray(values, dtype=float)
    return {"n": int(v.size), "mean": float(v.mean()), "median": pct(values, 50),
            "p75": pct(values, 75), "p90": pct(values, 90), "p95": pct(values, 95),
            "p99": pct(values, 99), "max": float(v.max())}


def report_text(row) -> str:
    return f"Findings: {row.get('findings', '')} Impression: {row.get('impression', '')}".strip()


def main() -> int:
    import pandas as pd
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    data = Path(args.data)

    tok = AutoTokenizer.from_pretrained("gpt2")
    meta = pd.read_csv(data / "mimic-cxr-2.0.0-metadata.csv.gz",
                       usecols=["subject_id", "study_id", "StudyDate", "StudyTime"])
    when = (meta.groupby("study_id")[["subject_id", "StudyDate", "StudyTime"]].first())

    out = {}
    for split in ("train", "test"):
        df = pd.read_parquet(data / f"{split}.parquet")
        df = df.drop_duplicates("study_id")
        texts = [report_text(r) for _, r in df.iterrows()]
        lengths = [len(tok(t)["input_ids"]) for t in texts]
        out[f"{split}_report_tokens"] = summary(lengths)
        out[f"{split}_report_tokens"]["share_over_256"] = float(sum(l > 256 for l in lengths) / len(lengths))
        if split == "test":
            comp = [bool(COMPARISON.search(t)) for t in texts]
            out["test_share_reports_comparing_with_prior"] = float(sum(comp) / len(comp))

            df = df.assign(_len=lengths).join(when, on="study_id", rsuffix="_meta")
            subj = df["subject_id"] if "subject_id" in df else df["subject_id_meta"]
            df = df.assign(_subj=subj.values,
                           _t=df["StudyDate"].astype("int64") * 1_000_000 + df["StudyTime"].fillna(0).astype(float))
            n_prior, hist = [], []
            for _, g in df.sort_values("_t").groupby("_subj"):
                cum = 0
                for i, (_, r) in enumerate(g.iterrows()):
                    n_prior.append(i)
                    hist.append(cum)
                    cum += int(r["_len"])
            out["test_prior_studies"] = summary(n_prior)
            out["test_prior_studies"]["share_with_any_prior"] = float(sum(n > 0 for n in n_prior) / len(n_prior))
            out["test_history_tokens"] = summary(hist)
            # what a history-aware decoder would read: 32 prefix + history + 100 new tokens
            ctx = [32 + h + 100 for h in hist]
            out["test_context_tokens_with_history"] = summary(ctx)
            for thr in (1024, 2048, 4096, 8192, 16384):
                out["test_context_tokens_with_history"][f"share_over_{thr}"] = \
                    float(sum(c > thr for c in ctx) / len(ctx))
            out["test_studies_matched_to_metadata"] = int(df["StudyDate"].notna().sum())
    print(json.dumps(out, indent=2))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
