"""P5-B (CHAT_UI_PLAN.md): the retrieval gallery the chat UI looks neighbours up in (section 6.5 of the plan).

    python scripts/build_retrieval_gallery.py --checkpoint-13d <13D last.ckpt> --decoder-checkpoint <report-gen last.ckpt> \\
        --data <dir with train.parquet and test.parquet> --out <gallery dir> [--decoder-config hybrid_150m_m3_rrg] [--workers N]
    python scripts/build_retrieval_gallery.py --compare-rk <gallery dir> [--wall-s N]
    python scripts/build_retrieval_gallery.py --tiny <dir>

Why it reproduces the retrieval chapter by construction. Both official splits are encoded through evaluate_cxr_retrieval's own
load_models, build_dataloader and encode_dataset, at that script's own batch size (32) and token length (256), so the vectors are
the ones its published R@k came from, and gate_rk.app is its own compute_retrieval_metrics on the test split. The wrapper
(scripts/build_retrieval_gallery_h100.sh) then runs that script UNCHANGED (R3) on the same checkpoint, and --compare-rk requires
every i2t and t2i R@k to be equal. The image transform there (_img_transform) gives the decoder's tensors for grayscale-origin
images (P2-C), so the gallery's image vectors are the report model's tower's whenever the towers are identical: the build hashes
both (tower_sha256 for the 13D tower, decoder_tower_sha256 for the report model's) and records towers_identical.

Files, in --out (section 6.5; labels.npy and label_names.json come from P5-C):
  img_emb.npy float16 and test_img_emb.npy float32      train and test image vectors (13D tower, unit length)
  txt_emb.npy float16 and txt_emb_test.npy float32      report vectors: train rows then test rows, and the test rows alone
  txt_groups, group_order, group_starts, txt_test_groups  duplicate-report groups (group_ids_from_texts), rows sorted by group,
                                                          where each group begins, and the groups among the test rows alone
  txt_split, txt_split_row, img_txt_row                 which split and row a report row is, and an image's own report row
  img_meta.parquet, test_meta.parquet                   ids, view, image path and the SHA-256 of every image file
  report_texts.txt                                      one `Findings: ... Impression: ...` per report row, as refs.txt prints it
  gate_rk.json, manifest.json                           the R@k gate, then provenance and counts (written last: the marker of a
                                                        finished build)
--compare-rk writes `reference` and `equal` into the manifest and then into gate_rk.json, whose `equal` is the verdict marker and so
comes last, and exits 1 when unequal; unequal, it also prints the reference's values of what differs and by how many studies. --tiny
writes the same layout from synthetic data, with no model and no MIMIC, for the laptop tests (the `tiny_gallery` fixture in
tests/conftest.py). A directory that holds a manifest.json is never written again, by either (R8): AlreadyBuilt.

Output is MIMIC-derived (Class R, analysis/ARCHIVE_MANIFEST.md): it stays on the cluster and is never committed. R7: stdout carries
`[gallery]` lines of known shapes (counts, booleans, hashes and norms: never a path, an id or report text, and the wrapper lets only
those shapes through) and, in --compare-rk mode, a few `RESULT {json}` or `ERROR ...` lines of numbers and key names. Everything else
the loaders print is for the wrapper to keep in a file.
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))      # app, scripts and hybrid_xmamba come from this tree, however the venv's install points

BATCH_SIZE = 32          # evaluate_cxr_retrieval.py's --batch-size default: the same batches as the published R@k
MAX_LENGTH = 256         # ... and its --max-length default
NORM_TOLERANCE = 1e-3    # every vector is unit length to this much: app/gallery.py reads a dot product as a cosine
DEFAULT_DECODER_CONFIG = "hybrid_150m_m3_rrg"
RK_KEYS = ("i2t_R@1", "i2t_R@5", "i2t_R@10", "t2i_R@1", "t2i_R@5", "t2i_R@10")
TOKENIZER = {"name": "gpt2", "max_length": MAX_LENGTH, "padding": "max_length", "truncation": True, "padding_side": "right",
             "pad_token": "eos"}
META_COLUMNS = ["study_id", "subject_id", "dicom_id", "view", "image"]
REQUIRED_COLUMNS = META_COLUMNS + ["findings", "impression"]
CHEXBERT_14 = ["Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation", "Pneumonia",
               "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding"]

TINY_IMAGES = 200        # train images (one report each), so 240 report rows with the test ones
TINY_TEST = 40
TINY_DIM = 16            # app.tiny.TINY_POOLED_DIM: a tiny engine's query meets the tiny gallery
TINY_SIDE = 320
TINY_POOL = 120          # distinct reports the rows are drawn from, with a templated head: duplicates by design


def say(message: str) -> None:
    """One [gallery] line: counts, booleans, hashes and norms, never a path, an id or report text (R7). The wrapper passes a line to
    the job log only if it has one of the shapes in its GALLERY_SHAPES, so a new line needs a new shape there and in the tests."""
    print("[gallery] " + message, flush=True)


class AlreadyBuilt(RuntimeError):
    """The output directory holds a manifest.json: the marker of a finished build, which is never written again (R8)."""


def refuse_if_built(out: Path) -> None:
    if (Path(out) / "manifest.json").exists():
        raise AlreadyBuilt("manifest.json exists in the output directory: a finished build is never overwritten")


# ── pure pieces ───────────────────────────────────────────────────────────────

def report_text(row: Any) -> str:
    """The reference string of one parquet row, as the published refs.txt holds it: run_checkpoint_inspection's f-string (no
    `or ""`, so a None prints as None, as it did there), then write_hyps_refs's whitespace collapse. `row` is a dict or a Series."""
    return " ".join("Findings: {} Impression: {}".format(row.get("findings", ""), row.get("impression", "")).strip().split())


def report_texts(frame: Any) -> List[str]:
    return [report_text(r) for r in frame[["findings", "impression"]].to_dict("records")]


def group_layout(groups: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(order, starts) for duplicate-group ids: the rows sorted by group (stable, so a group keeps its rows in file order) and the
    position in that order where each group begins. With ids 0..G-1 each present, group g is the g-th segment, which is what
    app/gallery.py's np.maximum.reduceat(sims[order], starts) relies on."""
    groups = np.asarray(groups)
    order = np.argsort(groups, kind="stable").astype(np.int64)
    if len(order) == 0:
        return order, np.zeros(0, dtype=np.int64)
    sorted_ids = groups[order]
    starts = np.flatnonzero(np.r_[True, sorted_ids[1:] != sorted_ids[:-1]]).astype(np.int64)
    return order, starts


def towers_identical(hashes: Dict[str, str], img_proj_present: bool) -> bool:
    """The gallery's image vectors are the report model's tower's only if the two towers hold the same weights and nothing is
    projected after the tower (a Phase 8+ checkpoint has no img_proj)."""
    return bool(hashes["tower_sha256"] == hashes["decoder_tower_sha256"] and not img_proj_present)


def transform_facts(mean: Sequence[float], std: Sequence[float], size: int) -> Dict[str, Any]:
    return {"source": "scripts/evaluate_cxr_retrieval.py _img_transform",
            "pipeline": "convert RGB, Resize(({0}, {0})), CenterCrop({0}), ToTensor, Normalize".format(size),
            "mean": list(mean), "std": list(std)}


def provenance(args: Any, root: Optional[Path] = None, env: Optional[Dict[str, str]] = None,
               now: Optional[datetime] = None) -> Dict[str, Any]:
    """build id, UTC time, the commit the tree was synced from (and whether it was clean), the job, and both checkpoints with
    their SHA-256. On the cluster there is no .git: git_provenance reads the .sync_stamp that `chat_remote.sh sync` leaves."""
    from app.engine import REPO_ROOT, file_sha256, git_provenance
    env = os.environ if env is None else env
    now = now or datetime.now(timezone.utc)
    restart = env.get("SLURM_RESTART_COUNT")
    prov = {"build_id": args.build_id or Path(args.out).name, "created": now.strftime("%Y-%m-%dT%H:%M:%SZ")}
    prov.update(git_provenance(Path(root) if root is not None else REPO_ROOT))          # git_sha, git_dirty, git_source
    prov.update(job_id=env.get("SLURM_JOB_ID"), restart_count=int(restart) if restart is not None and restart.strip().isdigit() else None,
                checkpoint_13d=str(args.checkpoint_13d), checkpoint_13d_sha256=file_sha256(Path(args.checkpoint_13d)),
                decoder_checkpoint=str(args.decoder_checkpoint),
                decoder_checkpoint_sha256=file_sha256(Path(args.decoder_checkpoint)), decoder_config=args.decoder_config)
    return prov


def assemble_manifest(hashes: Dict[str, str], img_proj_present: bool, counts: Dict[str, int], gate: Dict[str, Any],
                      prov: Dict[str, Any], transform: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    manifest = dict(prov)
    manifest.update(hashes)
    manifest.update(towers_identical=towers_identical(hashes, img_proj_present), img_proj_present=bool(img_proj_present),
                    counts=counts, transform=transform, tokenizer=dict(TOKENIZER), labels_status="pending", gate_rk=gate)
    if extra:
        manifest.update(extra)
    return manifest


def write_json_atomic(path: Path, obj: Any) -> None:
    """Written whole or not at all: manifest.json is the marker of a finished build, and a half-written one would stop the requeue
    that has to redo the build."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(str(tmp), str(path))


def hash_files(paths: Sequence[Any], hasher: Callable[[Path], str], workers: int) -> List[str]:
    """hasher over every path, in order, on a few threads: reading 194k image files is I/O bound and hashlib drops the GIL."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        return list(pool.map(lambda p: hasher(Path(p)), paths))


def write_embeddings(out: Path, test_img: np.ndarray, test_txt: np.ndarray, train_img: np.ndarray, train_txt: np.ndarray) -> None:
    np.save(out / "test_img_emb.npy", np.ascontiguousarray(test_img, dtype=np.float32))
    np.save(out / "txt_emb_test.npy", np.ascontiguousarray(test_txt, dtype=np.float32))
    np.save(out / "img_emb.npy", np.ascontiguousarray(train_img, dtype=np.float16))
    np.save(out / "txt_emb.npy", np.ascontiguousarray(np.concatenate([train_txt, test_txt]), dtype=np.float16))


def write_layout(out: Path, texts: List[str], n_train: int, n_test: int,
                 group_ids: Callable[[List[str]], np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """The report side of the gallery: the texts (train rows, then test rows), their duplicate groups and the maps between rows,
    splits and images. One image and one report per train study, so an image's own report row is its own row. -> (groups, starts)."""
    if len(texts) != n_train + n_test:
        raise ValueError("{} report rows for {} train and {} test studies".format(len(texts), n_train, n_test))
    (out / "report_texts.txt").write_text("\n".join(texts) + "\n", encoding="utf-8")
    groups = np.asarray(group_ids(texts), dtype=np.int64)
    order, starts = group_layout(groups)
    np.save(out / "txt_groups.npy", groups)
    np.save(out / "group_order.npy", order)
    np.save(out / "group_starts.npy", starts)
    np.save(out / "txt_test_groups.npy", np.asarray(group_ids(texts[n_train:]), dtype=np.int64))
    np.save(out / "txt_split.npy", np.r_[np.zeros(n_train, np.int8), np.ones(n_test, np.int8)])
    np.save(out / "txt_split_row.npy", np.r_[np.arange(n_train), np.arange(n_test)].astype(np.int64))
    np.save(out / "img_txt_row.npy", np.arange(n_train, dtype=np.int64))
    return groups, starts


def write_meta(out: Path, frames: Dict[str, Any], hasher: Callable[[Path], str], workers: int) -> None:
    """img_meta.parquet and test_meta.parquet: ids, view, image path and the SHA-256 of every image file (what 'identical to a
    gallery image' compares against)."""
    for split, name, key in (("train", "img_meta.parquet", "row"), ("test", "test_meta.parquet", "test_row")):
        meta = frames[split][META_COLUMNS].reset_index(drop=True)
        meta.insert(0, key, np.arange(len(meta), dtype=np.int64))
        meta["file_sha256"] = hash_files(meta["image"].tolist(), hasher, workers)
        meta.to_parquet(out / name, index=False)


def check_columns(frames: Dict[str, Any]) -> None:
    """The columns the texts and the metadata are built from. Run before any model is loaded: a column missing is found in a second,
    not after the hour of encoding. Messages carry column names and counts only."""
    for split in ("train", "test"):
        missing = [c for c in REQUIRED_COLUMNS if c not in frames[split].columns]
        if missing:
            raise RuntimeError("{}.parquet lacks the columns {}".format(split, ", ".join(missing)))


def check_rows(frames: Dict[str, Any], rows: Dict[str, int], emb: Dict[str, Tuple[np.ndarray, np.ndarray]],
               splits: Sequence[str] = ("train", "test")) -> None:
    """The parquet, the loader and the vectors must describe the same rows, or the texts and metadata would be misaligned with the
    vectors without a sound. build() asks about the test split the moment it is encoded, before the train pass of about an hour."""
    for split in splits:
        counts = (len(frames[split]), int(rows[split]), int(emb[split][0].shape[0]), int(emb[split][1].shape[0]))
        if len(set(counts)) != 1:
            raise RuntimeError("{}: the parquet, the loader, the image vectors and the text vectors disagree about the rows: {}"
                               .format(split, counts))


def norm_stats(vectors: Tuple[np.ndarray, np.ndarray]) -> Dict[str, Tuple[float, float, float]]:
    """(least, greatest, mean) norm of the image vectors and of the text vectors of one split."""
    found = {}
    for kind, array in zip(("img", "txt"), vectors):
        norms = np.linalg.norm(array, axis=1)
        found[kind] = (float(norms.min()), float(norms.max()), float(norms.mean(dtype=np.float64)))
    return found


def norms_line(split: str, stats: Dict[str, Tuple[float, float, float]]) -> str:
    return "norms: {} img min={:.6f} max={:.6f} mean={:.6f} txt min={:.6f} max={:.6f} mean={:.6f}".format(
        split, *(stats["img"] + stats["txt"]))


def check_unit_norms(split: str, stats: Dict[str, Tuple[float, float, float]], tolerance: float = NORM_TOLERANCE) -> None:
    """Every vector is unit length, to a thousandth: app/gallery.py ranks by dot product and calls it cosine, so a vector of another
    length would be ranked by its length as well. A NaN fails too, because only a number inside the bounds passes."""
    for kind, (least, greatest, _) in stats.items():
        if not (1.0 - tolerance <= least and greatest <= 1.0 + tolerance):
            raise RuntimeError("{} {} vectors are not unit length: least norm {:.6f}, greatest norm {:.6f}, bound 1 +- {}".format(
                split, kind, least, greatest, tolerance))


def isbi_cross_check(img_emb: np.ndarray, path: Optional[Path]) -> Dict[str, Any]:
    """Optional, never fatal. scripts/isbi_figure_cases.py caches the decoder tower's fp16 train embeddings
    (torch.save({"encoder", "n", "emb"})); max |img_emb - isbi| near fp16 rounding says, independently of the hashes, that the 13D
    tower and the decoder's saw the same images the same way (D4)."""
    if path is None or not Path(path).is_file():
        return {"status": "absent"}
    try:
        import torch
        saved = torch.load(str(path), map_location="cpu", weights_only=True)
        encoder, n, isbi = saved["encoder"], int(saved["n"]), saved["emb"].float().numpy()
    except Exception:       # a damaged or foreign file: say so, without a message that could carry a path
        return {"status": "unreadable"}
    if encoder != "adapted" or n != len(img_emb) or isbi.shape != img_emb.shape:
        return {"status": "mismatch"}
    worst = 0.0
    for lo in range(0, n, 20000):
        worst = max(worst, float(np.abs(img_emb[lo:lo + 20000].astype(np.float32) - isbi[lo:lo + 20000]).max()))
    return {"status": "compared", "rows": n, "max_abs_diff": worst}


def gate_line(app: Dict[str, Any]) -> str:
    return "gate_rk app:" + "".join(" {}={:.4f}".format(k, app[k]) for k in RK_KEYS) + " n={}".format(app["N"])


def isbi_line(found: Dict[str, Any]) -> str:
    if found["status"] == "compared":
        return "isbi cross-check: status=compared rows={} max_abs_diff={:.3e}".format(found["rows"], found["max_abs_diff"])
    return "isbi cross-check: status={}".format(found["status"])


# ── --compare-rk ──────────────────────────────────────────────────────────────

def compare_rk(out: Path, wall_s: Optional[int] = None) -> int:
    """The gate: gate_rk.json's `app` (compute_retrieval_metrics on the build's own test vectors) against the newest
    reference_rk/phase6_mimic_*.json (scripts/evaluate_cxr_retrieval.py, unchanged, on the same checkpoint and split). Every i2t and
    t2i R@k must be equal, to every digit, and so must the number of studies. 0 equal, 1 unequal, 2 nothing to compare."""
    out = Path(out)
    try:
        gate = json.loads((out / "gate_rk.json").read_text())
        app = gate["app"]
    except (OSError, ValueError, KeyError, TypeError):
        print("ERROR gate_rk.json is missing or has no app metrics")
        return 2
    found = sorted((out / "reference_rk").glob("phase6_mimic_*.json"))      # the stamp in the name sorts by time
    if not found:
        print("ERROR no reference result found: the reference evaluation has not run")
        return 2
    try:
        reference = json.loads(found[-1].read_text())["metrics"]
    except (OSError, ValueError, KeyError, TypeError):
        print("ERROR the newest reference result is unreadable or has no metrics")
        return 2
    lacking = [k for k in RK_KEYS if k not in app or k not in reference]
    if lacking:
        print("ERROR recall keys missing: " + ", ".join(lacking))
        return 2
    differs = [k for k in RK_KEYS if app[k] != reference[k]]
    if "N" in app and "N" in reference and app["N"] != reference["N"]:
        differs.append("N")
    gate.update(reference=reference, equal=not differs, reference_file=found[-1].name)
    # The verdict is gate_rk.json's `equal`, and the wrapper reads it to tell a decided gate from one still to be decided, so it is
    # written last: a crash between the two files leaves the gate undecided, and the next attempt decides it again.
    if (out / "manifest.json").is_file():
        manifest = json.loads((out / "manifest.json").read_text())
        manifest["gate_rk"] = gate
        write_json_atomic(out / "manifest.json", manifest)
    write_json_atomic(out / "gate_rk.json", gate)
    payload = {"gate_rk_equal": not differs, "n": app.get("N")}
    payload.update({k: round(float(app[k]), 4) for k in RK_KEYS})
    if differs:
        payload["differs"] = differs
    if wall_s is not None:
        payload["wall_s"] = int(wall_s)
    print("RESULT " + json.dumps(payload))
    if differs:
        try:
            lines = reference_diagnosis(app, reference, differs)
        except (TypeError, ValueError, KeyError):     # a diagnosis that cannot be made changes nothing about the verdict
            lines = []
        for line in lines:
            print("RESULT " + json.dumps(line))
    return 0 if not differs else 1


def reference_diagnosis(app: Dict[str, Any], reference: Dict[str, Any], differs: List[str]) -> List[Dict[str, Any]]:
    """What an unequal gate leaves to read through `summary`, which shows lines of a job log and cuts them at 300 characters: the
    reference's own values of the recalls that differ (and its study count, if that differs), then the gap in studies. A flip of
    one study across a cut is a gap of 1, and a real mismatch is a gap of many. Numbers and key names only; a line of its own for each,
    so that none comes near the cut however much differs."""
    keys = [k for k in differs if k in RK_KEYS]
    shown = {"gate_rk_reference": {k: round(float(reference[k]), 4) for k in keys}}
    if "N" in differs:
        shown["reference_n"] = int(reference["N"])
    lines = [shown]
    n = app.get("N")
    if keys and n:
        lines.append({"delta_studies": {k: int(round((float(reference[k]) - float(app[k])) * n)) for k in keys}})
    return lines


# ── the real build ────────────────────────────────────────────────────────────

def real_deps() -> SimpleNamespace:
    """build()'s heavy collaborators: the retrieval chapter's own loaders (R3: imported, never edited), the thesis's checkpoint
    loader for the report model, and the engine's hashes. Imported here, not at the top, so that --compare-rk, which needs none of
    it, imports neither torch nor datasets nor transformers, and --tiny, which needs torch only for the vocabulary list in
    app/tiny.py and builds no model, imports none of datasets, transformers, the reference script and the engine (both are tested
    in a fresh interpreter). A test passes fakes behind the same signatures instead."""
    import torch
    from transformers import AutoTokenizer
    from app.engine import file_sha256, tensor_sha256
    from scripts import evaluate_cxr_retrieval as ref
    from scripts.evaluate_report_generation import load_report_generation_module

    def make_tokenizer():
        tok = AutoTokenizer.from_pretrained("gpt2")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "right"
        return tok

    return SimpleNamespace(
        device=lambda: "cuda" if torch.cuda.is_available() else "cpu",
        load_models=ref.load_models, load_decoder=load_report_generation_module, make_tokenizer=make_tokenizer,
        build_dataloader=ref.build_dataloader, encode_dataset=ref.encode_dataset,
        group_ids_from_texts=ref.group_ids_from_texts, compute_retrieval_metrics=ref.compute_retrieval_metrics,
        tensor_sha256=tensor_sha256, file_sha256=file_sha256, provenance=provenance,
        transform=transform_facts(ref.IMAGE_MEAN, ref.IMAGE_STD, ref.IMAGE_SIZE))


def build(args: Any, deps: Optional[SimpleNamespace] = None) -> Dict[str, Any]:
    """The test split first (2,663 rows: the gate), then the train split (191,462), through the reference loaders, each checked the
    moment it is encoded (its rows against the parquet's, its vectors for unit length) so that a wrong test split is found before
    the pass of about an hour over the train split; then the report texts, their groups, the metadata and the manifest, which is
    written last. Refuses a directory that already holds a manifest.json (AlreadyBuilt). Returns the manifest."""
    import gc
    import pandas as pd
    out = Path(args.out)
    refuse_if_built(out)                                  # first of all: before the imports of real_deps(), and before any file
    deps = real_deps() if deps is None else deps
    out.mkdir(parents=True, exist_ok=True)
    frames = {s: pd.read_parquet(Path(args.data) / "{}.parquet".format(s)) for s in ("train", "test")}
    check_columns(frames)
    device = deps.device()
    say("device={}".format(device))

    text_enc, img_proj, tower13d = deps.load_models(args.checkpoint_13d, device)
    module = deps.load_decoder(args.decoder_checkpoint, args.decoder_config, device=device)
    hashes = {"tower_sha256": deps.tensor_sha256(tower13d), "decoder_tower_sha256": deps.tensor_sha256(module.image_encoder)}
    del module
    gc.collect()
    say("towers_identical={} img_proj={}".format(towers_identical(hashes, img_proj is not None), img_proj is not None))
    say("tower_sha256={tower_sha256} decoder_tower_sha256={decoder_tower_sha256}".format(**hashes))

    tok = deps.make_tokenizer()
    emb, rows, gate = {}, {}, None
    for split in ("test", "train"):
        loader, n = deps.build_dataloader("mimic", "", tok, max_length=MAX_LENGTH, batch_size=args.batch_size,
                                          num_workers=args.workers, local_parquet_dir=args.data, mimic_split=split)
        emb[split] = deps.encode_dataset(loader, text_enc, img_proj, tower13d, device)        # (img, txt), float32, unit length
        rows[split] = n
        say("{}: {} rows".format(split, n))
        check_rows(frames, rows, emb, (split,))
        stats = norm_stats(emb[split])
        say(norms_line(split, stats))                       # printed before the check, so that a failure leaves its numbers in the log
        check_unit_norms(split, stats)
        if split == "test":
            gate = {"app": deps.compute_retrieval_metrics(emb["test"][0], emb["test"][1])}
            say(gate_line(gate["app"]))

    n_train, n_test = len(frames["train"]), len(frames["test"])
    write_embeddings(out, emb["test"][0], emb["test"][1], emb["train"][0], emb["train"][1])

    texts = report_texts(frames["train"]) + report_texts(frames["test"])
    _, starts = write_layout(out, texts, n_train, n_test, deps.group_ids_from_texts)
    say("texts: {} rows, {} groups".format(len(texts), len(starts)))
    write_meta(out, frames, deps.file_sha256, args.workers)
    say("hashed {} image files".format(n_train + n_test))
    isbi = isbi_cross_check(np.ascontiguousarray(emb["train"][0], dtype=np.float16),
                            Path(args.isbi_cache) if args.isbi_cache else None)
    say(isbi_line(isbi))

    counts = {"images": n_train, "report_rows": len(texts), "report_groups": int(len(starts)), "test": n_test}
    manifest = assemble_manifest(hashes, img_proj is not None, counts, gate, deps.provenance(args), deps.transform,
                                 extra={"isbi_cross_check": isbi})
    write_json_atomic(out / "gate_rk.json", gate)
    write_json_atomic(out / "manifest.json", manifest)
    say("manifest written, labels_status=pending")
    return manifest


# ── the synthetic gallery (--tiny): the same layout, no model, no data ────────

def _normalize(text: Optional[str]) -> str:
    """evaluate_cxr_retrieval.normalize_report_text, which --tiny does not import (3-4 s of datasets and transformers);
    tests/test_build_retrieval_gallery.py keeps the copy equal."""
    return " ".join((text or "").lower().split())


def _group_ids(texts: Sequence[Optional[str]]) -> np.ndarray:
    """evaluate_cxr_retrieval.group_ids_from_texts: ids by first appearance of the normalised text."""
    lookup = {}     # type: Dict[str, int]
    ids = np.empty(len(texts), dtype=np.int64)
    for i, text in enumerate(texts):
        key = _normalize(text)
        if key not in lookup:
            lookup[key] = len(lookup)
        ids[i] = lookup[key]
    return ids


def _recall_metrics(img_embs: np.ndarray, txt_embs: np.ndarray) -> Dict[str, float]:
    """evaluate_cxr_retrieval.compute_retrieval_metrics without groups (the authoritative protocol), kept equal by the tests."""
    sim = img_embs @ txt_embs.T
    n = sim.shape[0]

    def recall(mat: np.ndarray, k: int) -> float:
        k_eff = min(k, n)
        top_k = np.argpartition(-mat, kth=k_eff - 1, axis=1)[:, :k_eff]
        return float((top_k == np.arange(n)[:, None]).any(axis=1).mean())

    i2t = {k: recall(sim, k) for k in (1, 5, 10)}
    t2i = {k: recall(sim.T, k) for k in (1, 5, 10)}
    return {"i2t_R@1": i2t[1], "i2t_R@5": i2t[5], "i2t_R@10": i2t[10], "t2i_R@1": t2i[1], "t2i_R@5": t2i[5], "t2i_R@10": t2i[10],
            "mean_R@10": (i2t[10] + t2i[10]) / 2, "N": n}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(str(path), "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _tiny_image(rng: np.random.Generator, side: int = TINY_SIDE):
    """A blocky gray picture: different for every seed, a few KB as a JPEG."""
    from PIL import Image
    cells = 16
    blocks = np.kron(rng.random((cells, cells)), np.ones((side // cells, side // cells)))
    ramp = (np.arange(side)[None, :] + np.arange(side)[:, None]) / (2.0 * side)
    return Image.fromarray(((0.6 * blocks + 0.4 * ramp) * 255.0).astype(np.uint8))


def _tiny_sentence(rng: np.random.Generator, words: List[str], low: int, high: int) -> str:
    return " ".join(str(w) for w in rng.choice(words, size=int(rng.integers(low, high))))


def tiny_frames(images_dir: Path, seed: int = 0, n_images: int = TINY_IMAGES, n_test: int = TINY_TEST) -> Tuple[Dict[str, Any], np.ndarray]:
    """The synthetic dataset: {"train", "test"} frames with the columns of the real parquet files (image, findings, impression,
    study_id, subject_id, dicom_id, view) and the gray JPEGs they name, written to images_dir; and, per report row, the index of the
    report it was drawn from. The rows are drawn from a pool with a templated head, then bent on purpose: four train studies share
    one report, two test studies share one, a test study repeats a train report, and one report appears in two spellings."""
    import pandas as pd
    from app.tiny import TINY_VOCAB
    assert n_images >= 8 and n_test >= 3, "the deliberate duplicates need at least 8 train and 3 test studies"
    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng((seed, 1))
    n = n_images + n_test
    words = [w for w in TINY_VOCAB if w not in ("Findings:", "Impression:")]
    pool, seen = [], set()
    while len(pool) < TINY_POOL:
        pair = ("{}.".format(_tiny_sentence(rng, words, 5, 11)), "{}.".format(_tiny_sentence(rng, words, 3, 8)))
        if pair not in seen:
            seen.add(pair)
            pool.append(pair)
    weights = 1.0 / (np.arange(TINY_POOL) + 4.0)
    pick = rng.choice(TINY_POOL, size=n, p=weights / weights.sum())
    pick[1] = pick[2] = pick[3] = pick[0]                  # four train studies, one report
    pick[5] = pick[4]                                      # (spelt differently below)
    pick[n_images + 1] = pick[n_images]                    # two test studies, one report
    pick[n_images + 2] = pick[0]                           # a test study whose report a train study repeats
    findings = [pool[i][0] for i in pick]
    impression = [pool[i][1] for i in pick]
    findings[5], impression[5] = findings[4].upper(), impression[4].upper()        # one group, two spellings
    findings[7] = findings[7].replace(" ", "  ", 1) + "\n"                          # whitespace that report_text collapses
    frames = {}
    for split, rows in (("train", range(n_images)), ("test", range(n_images, n))):
        records = []
        for j, i in enumerate(rows):
            path = images_dir / "{}_{:03d}.jpg".format(split, j)
            _tiny_image(np.random.default_rng((seed, 1000 + i))).save(str(path), "JPEG", quality=90)
            ids = np.random.default_rng((seed, 2000 + i)).integers(0, 2 ** 32, size=5)
            records.append({"image": str(path.resolve()), "findings": findings[i], "impression": impression[i],
                            "study_id": (50000000 if split == "train" else 56000000) + j,
                            "subject_id": 10000000 + j // 2 if split == "train" else 19000000 + j,
                            "dicom_id": "-".join("{:08x}".format(int(v)) for v in ids), "view": ("PA", "AP", "LATERAL")[j % 3]})
        frames[split] = pd.DataFrame(records, columns=["image", "findings", "impression", "study_id", "subject_id", "dicom_id", "view"])
    return frames, pick


def tiny_vectors(pick: np.ndarray, seed: int, n_images: int, n_test: int,
                 dim: int = TINY_DIM) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(train image vectors, test image vectors, report vectors for every row, labels for every row), float32 unit vectors and
    uint8 labels. A report row's vector and labels are those of the report it was drawn from, so duplicates share both, as they
    would in a real build; an image's vector is its report's with noise on it."""
    rng = np.random.default_rng((seed, 77))
    pool_vec = _unit(rng.standard_normal((TINY_POOL, dim)))
    pool_labels = (rng.random((TINY_POOL, len(CHEXBERT_14))) < 0.3).astype(np.uint8)
    txt = pool_vec[pick]
    img = _unit(txt[:n_images] + 0.35 * rng.standard_normal((n_images, dim)))
    test_img = _unit(txt[n_images:] + 0.35 * rng.standard_normal((n_test, dim)))
    return img.astype(np.float32), test_img.astype(np.float32), txt.astype(np.float32), pool_labels[pick]


def build_tiny(out: Path, seed: int = 0, n_images: int = TINY_IMAGES, n_test: int = TINY_TEST, dim: int = TINY_DIM,
               with_labels: bool = True) -> Dict[str, Any]:
    """A synthetic gallery in the layout of section 6.5, written by the same writers as the real build: n_images random unit
    dim-d image vectors (and as many train reports), n_test test studies, duplicate report groups, gray JPEGs under out/images/.
    With_labels adds random per-group labels and label_names.json (P5-C's files) and sets labels_status to done. Returns the
    manifest. No model and no MIMIC; of the heavy imports only torch, which app/tiny.py brings in for its vocabulary list (not
    datasets, transformers, the reference script or the engine). Refuses a directory that already holds a manifest.json."""
    from app.imaging import CLIP_MEAN, CLIP_STD
    out = Path(out)
    refuse_if_built(out)
    out.mkdir(parents=True, exist_ok=True)
    frames, pick = tiny_frames(out / "images", seed, n_images, n_test)
    img, test_img, txt, labels = tiny_vectors(pick, seed, n_images, n_test, dim)
    write_embeddings(out, test_img, txt[n_images:], img, txt[:n_images])
    texts = report_texts(frames["train"]) + report_texts(frames["test"])
    _, starts = write_layout(out, texts, n_images, n_test, _group_ids)
    write_meta(out, frames, _sha256_file, workers=1)
    gate = {"app": _recall_metrics(test_img, txt[n_images:])}
    tower = hashlib.sha256(b"tiny gallery: no tower").hexdigest()
    prov = {"build_id": "tiny", "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "git_sha": None,
            "git_dirty": None, "git_source": None, "job_id": None, "restart_count": None, "checkpoint_13d": None,
            "checkpoint_13d_sha256": None, "decoder_checkpoint": None, "decoder_checkpoint_sha256": None, "decoder_config": None}
    counts = {"images": n_images, "report_rows": len(texts), "report_groups": int(len(starts)), "test": n_test}
    manifest = assemble_manifest({"tower_sha256": tower, "decoder_tower_sha256": tower}, False, counts, gate, prov,
                                 transform_facts(CLIP_MEAN, CLIP_STD, 224), extra={"synthetic": True})
    if with_labels:
        np.save(out / "labels.npy", labels)
        (out / "label_names.json").write_text(json.dumps(CHEXBERT_14))
        manifest["labels_status"] = "done"
    write_json_atomic(out / "gate_rk.json", gate)
    write_json_atomic(out / "manifest.json", manifest)
    return manifest


# ── command line ──────────────────────────────────────────────────────────────

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tiny", metavar="OUT", help="write a synthetic gallery (no model, no data) into OUT and stop")
    p.add_argument("--compare-rk", metavar="OUT", dest="compare_rk",
                   help="compare OUT/gate_rk.json with the newest OUT/reference_rk/phase6_mimic_*.json; exit 1 when unequal")
    p.add_argument("--wall-s", type=int, default=None, help="with --compare-rk: the job's wall time in seconds, for the RESULT line")
    p.add_argument("--checkpoint-13d", help="the retrieval checkpoint: the 13D image tower and text encoder")
    p.add_argument("--decoder-checkpoint", help="the report-generation checkpoint whose tower is compared with the 13D one")
    p.add_argument("--decoder-config", default=DEFAULT_DECODER_CONFIG, help="its model config (default: %(default)s)")
    p.add_argument("--data", help="the directory with train.parquet and test.parquet (the official split)")
    p.add_argument("--out", help="the gallery directory")
    p.add_argument("--build-id", default=None, help="recorded in the manifest (default: the name of --out)")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="default: the reference script's, so the batches are too")
    p.add_argument("--workers", type=int, default=4, help="data loader processes and file-hash threads")
    p.add_argument("--isbi-cache", default=None, help="the ISBI figure job's isbi_gallery_adapted.pt, for the optional cross-check")
    args = p.parse_args(argv)
    if args.tiny and args.compare_rk:
        p.error("--tiny and --compare-rk are separate modes")
    if not args.tiny and not args.compare_rk:
        missing = [flag for flag, value in (("--checkpoint-13d", args.checkpoint_13d), ("--decoder-checkpoint", args.decoder_checkpoint),
                                            ("--data", args.data), ("--out", args.out)) if not value]
        if missing:
            p.error("the build needs " + ", ".join(missing))
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.compare_rk:
        return compare_rk(Path(args.compare_rk), args.wall_s)
    try:
        if args.tiny:
            counts = build_tiny(Path(args.tiny))["counts"]
            say("tiny: {images} images, {report_rows} report rows, {report_groups} groups, {test} test".format(**counts))
        else:
            build(args)
    except AlreadyBuilt as err:      # a plain line instead of a traceback: the directory is a finished build, and stays one
        print("ERROR " + str(err))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
