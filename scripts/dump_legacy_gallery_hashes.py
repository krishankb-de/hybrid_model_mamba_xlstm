#!/usr/bin/env python3
"""Dump report_hash values for the legacy MIMIC eval gallery (train[90%:] on
the itsanmolgupta/mimic-cxr-dataset HF mirror), for use as
build_mimic_cxr_local.py's `pack --exclude-hashes` input.

Part of the Phase 8D leakage guard (H100_SCALING_PLAN.md): the local
full-MIMIC build must exclude every SUBJECT that appears in the legacy
gallery this project's retrieval numbers were measured against, or those
3,063 studies leak into training. Text construction and hashing here
DELIBERATELY match build_mimic_cxr_local.py's norm_hash() and the
in-repo convention (normalize_report_text @ evaluate_cxr_retrieval.py:414,
text_hash @ train_contrastive.py:419-424) exactly, so the join is not a
second, drifted normalisation scheme.

⚠️ Read Phase 8D before trusting this join: the legacy gallery's own
section-parsing convention is unknown, so this hash join WILL under-match
some studies whose text differs by even one token from how build_mimic_cxr_
local.py's (official, vendored) parser extracted the same report. pack's
--min-match-frac gate (default 0.95) is what actually protects against that
— this script only produces the input to it.

Usage:
    python scripts/dump_legacy_gallery_hashes.py \\
        --cache-dir /sc/home/$USER/dataset/mimic_cxr_cache \\
        --out legacy_gallery_hashes.txt
"""

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset  # noqa: E402
from scripts.evaluate_cxr_retrieval import MIMIC_REPO  # noqa: E402


def norm_hash(text: str) -> str:
    """Identical to build_mimic_cxr_local.py:norm_hash -- kept as a literal
    copy rather than a shared import so this script has zero dependency on
    the build script (it may run on a different host / before `meta`)."""
    norm = " ".join((text or "").lower().split())
    return hashlib.blake2b(norm.encode("utf-8"), digest_size=8).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default="/scratch/bhushkri/mimic_cxr_cache")
    ap.add_argument("--split", default="train[90%:]",
                     help="the legacy eval gallery slice this project's numbers used")
    ap.add_argument("--out", default="legacy_gallery_hashes.txt")
    args = ap.parse_args()

    print("Loading {} split={} ...".format(MIMIC_REPO, args.split))
    ds = load_dataset(MIMIC_REPO, split=args.split, cache_dir=args.cache_dir)
    print("  {} rows, columns: {}".format(len(ds), ds.column_names))

    hashes = []
    for item in ds:
        findings = (item.get("findings") or "").strip()
        impression = (item.get("impression") or "").strip()
        text = "Findings: {} Impression: {}".format(findings, impression).strip()
        hashes.append(norm_hash(text))

    Path(args.out).write_text("\n".join(hashes) + "\n")
    print("Wrote {} hashes -> {}".format(len(hashes), args.out))


if __name__ == "__main__":
    main()
