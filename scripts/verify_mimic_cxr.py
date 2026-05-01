"""Phase 4 — MIMIC-CXR dataset verification + pre-cache on willi.

Run on willi login node (no GPU needed):

    # Step 1: verify columns only (fast, < 1 min)
    python scripts/verify_mimic_cxr.py

    # Step 2: also pre-cache the full train split (~30K rows)
    python scripts/verify_mimic_cxr.py --precache

    # Override cache dir or split explicitly
    python scripts/verify_mimic_cxr.py --cache-dir /scratch/bhushkri/mimic_cxr_cache --split train

Exit codes:
    0 — required columns present, everything looks good
    1 — columns missing or dataset empty (update mimic_cxr.yaml before sbatch)

Updates to make in configs/dataset/mimic_cxr.yaml after running:
    - findings_field   (if different from "findings")
    - impression_field (if different from "impression")
    - train_split / validation_split (if HF splits have different names)
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional

_REQUIRED_IMAGE_COL = "image"
_DEFAULT_FINDINGS_COL = "findings"
_DEFAULT_IMPRESSION_COL = "impression"
_TEXT_COL_CANDIDATES = [
    "findings", "impression", "report", "text", "report_text",
    "findings_text", "impression_text", "clinical_notes",
]


def _load_split(repo: str, split: str, cache_dir: str):
    from datasets import load_dataset  # type: ignore
    print(f"Loading {repo} (split='{split}') ...")
    try:
        ds = load_dataset(repo, split=split, cache_dir=cache_dir)
    except Exception as exc:
        # MIMIC-CXR requires PhysioNet credentials via HF_TOKEN.
        # If you hit AuthError, run: huggingface-cli login --token $HF_TOKEN
        raise RuntimeError(
            f"Failed to load {repo} (split='{split}').\n"
            f"Error: {exc}\n"
            "If this is an auth error: export HF_TOKEN=<your_token> and retry, "
            "or run: huggingface-cli login"
        )
    return ds


def _infer_text_cols(column_names: List[str]) -> Dict[str, Optional[str]]:
    """Map logical field names to actual column names."""
    cols_lower = {c.lower(): c for c in column_names}
    result: Dict[str, Optional[str]] = {"findings": None, "impression": None}
    for logical in ("findings", "impression"):
        for candidate in _TEXT_COL_CANDIDATES:
            if candidate in cols_lower and logical in candidate:
                result[logical] = cols_lower[candidate]
                break
        if result[logical] is None and logical in cols_lower:
            result[logical] = cols_lower[logical]
    return result


def _sample_text(ds, col: str, n: int = 3) -> None:
    for i in range(min(n, len(ds))):
        val = ds[i].get(col, "")
        snippet = (val or "")[:120].replace("\n", " ")
        print(f"    [{i}] {col}: {repr(snippet)}")


def verify_split(repo: str, split: str, cache_dir: str, precache: bool) -> bool:
    ds = _load_split(repo, split, cache_dir)

    print(f"\n{'─' * 55}")
    print(f"Split        : {split}")
    print(f"Rows         : {len(ds)}")
    print(f"Columns ({len(ds.column_names)}): {ds.column_names}")

    ok = True

    # Image column
    if _REQUIRED_IMAGE_COL in ds.column_names:
        print(f"\n  ✓ '{_REQUIRED_IMAGE_COL}' column present")
        # Spot-check: verify image is not None in first 3 rows
        none_count = sum(1 for i in range(min(10, len(ds))) if ds[i].get("image") is None)
        if none_count > 0:
            print(f"  ⚠ {none_count}/10 sampled rows have image=None (expected with MIMIC; "
                  f"check dataset completeness)")
        else:
            print(f"  ✓ First 10 rows all have non-None images")
    else:
        print(f"\n  ✗ '{_REQUIRED_IMAGE_COL}' column MISSING — cannot train CLIP objective")
        print(f"    Available columns: {ds.column_names}")
        ok = False

    # Text columns
    inferred = _infer_text_cols(ds.column_names)
    print()
    for logical, actual in inferred.items():
        if actual:
            print(f"  ✓ '{logical}' → column '{actual}'")
            _sample_text(ds, actual, n=2)
        else:
            print(f"  ✗ No column found for '{logical}' — update "
                  f"configs/dataset/mimic_cxr.yaml: {logical}_field")
            ok = False

    # Check that at least one text field is non-empty
    if inferred.get("findings") or inferred.get("impression"):
        primary = inferred.get("findings") or inferred.get("impression")
        empty = sum(1 for i in range(min(20, len(ds)))
                    if not (ds[i].get(primary) or "").strip())
        if empty > 5:
            print(f"\n  ⚠ {empty}/20 sampled '{primary}' values are empty strings — "
                  "check concatenate_sections logic in MIMICJointDataset")

    # Required config block
    print(f"\n  → Paste into configs/dataset/mimic_cxr.yaml if different from defaults:")
    findings_col   = inferred.get("findings")   or _DEFAULT_FINDINGS_COL
    impression_col = inferred.get("impression") or _DEFAULT_IMPRESSION_COL
    print(f"      findings_field: \"{findings_col}\"")
    print(f"      impression_field: \"{impression_col}\"")

    # Pre-cache: iterate all rows to force HF arrow cache to disk
    if precache:
        print(f"\n  Pre-caching all {len(ds)} rows (iterating dataset) ...")
        count = 0
        for _ in ds:
            count += 1
            if count % 5000 == 0:
                print(f"    {count}/{len(ds)} cached ...", flush=True)
        print(f"  ✓ Pre-cache complete ({count} rows, cache_dir={cache_dir})")

    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 MIMIC-CXR dataset verification")
    parser.add_argument(
        "--repo", default="itsanmolgupta/mimic-cxr-dataset",
        help="HuggingFace repo ID"
    )
    parser.add_argument(
        "--split", default="train",
        help="HF split to verify (default: train). Use 'train,validation' for both."
    )
    parser.add_argument(
        "--cache-dir", default="/scratch/bhushkri/mimic_cxr_cache",
        help="Local HF cache directory (must exist on willi scratch)"
    )
    parser.add_argument(
        "--precache", action="store_true",
        help="Iterate the full split to force HF arrow cache to disk"
    )
    args = parser.parse_args()

    splits = [s.strip() for s in args.split.split(",")]

    print("=" * 55)
    print("Phase 4: MIMIC-CXR dataset verification")
    print(f"Repo      : {args.repo}")
    print(f"Cache dir : {args.cache_dir}")
    print(f"Splits    : {splits}")
    print(f"Precache  : {args.precache}")
    print("=" * 55)

    all_ok = True
    for split in splits:
        try:
            ok = verify_split(
                repo=args.repo,
                split=split,
                cache_dir=args.cache_dir,
                precache=args.precache,
            )
            all_ok = all_ok and ok
        except Exception as exc:
            print(f"\n  FAIL [{split}]: {exc}")
            all_ok = False

    print(f"\n{'=' * 55}")
    if all_ok:
        print("ALL CHECKS PASSED — safe to proceed to Phase 5 (sbatch train_joint_mimic.sh).")
        print("Remember to update joint_training_state.json: mimic_cxr.verified=true")
    else:
        print("FAILED — fix column names in configs/dataset/mimic_cxr.yaml before sbatch.")
        sys.exit(1)
    print("=" * 55)


if __name__ == "__main__":
    main()
