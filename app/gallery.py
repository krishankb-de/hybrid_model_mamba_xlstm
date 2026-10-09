"""The retrieval gallery of the chat app (CHAT_UI_PLAN.md P5-D): the files scripts/build_retrieval_gallery.py writes (section 6.5 of the plan),
loaded once and queried with the report model's image vector.

    Gallery.open(dir, expect_tower_sha256)   loads, or raises GalleryMismatch
    image_neighbors(q, k)     the k train images most like the query (at most 12), each with its study and its own report's labels
    report_matches(q, k)      the k best report groups (at most 10): duplicate reports are one group, scored by their best member
    own_report_rank(q, row)   where test study `row`'s own report ranks among the test reports: the retrieval chapter's protocol (D5)
    find_identical(sha)       whether an uploaded file is, byte for byte, a train or a test image
    test_study, list_test_studies, image_path   the test-split picker and the neighbour thumbnails

open() refuses a gallery it cannot vouch for: a tower other than the engine's, a build whose R@k gate was not decided equal (manifest
gate_rk.equal is not true: it failed, or never ran), a missing file, or arrays that disagree with manifest["counts"] or with the layout the
queries read off them. Embeddings are held as float32 RAM copies (about 0.8 GB at full size; `nbytes`), read-only, and every query vector is
normalised here so that a similarity is a cosine whatever the caller passes.

R1 and R7. This module returns data: report texts, study ids, rows, image paths. What may leave the cluster in public mode is app/redact.py's
decision, applied by the pipeline. It logs nothing, and an error message holds counts and file basenames only, never a path, an id or a text.
"""
import json
import operator
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.labels import CHEXBERT_14

MAX_K_IMAGES = 12        # Options.k_images (le=12) and Options.k_reports (le=10), section 2: a query asks for no more than that
MAX_K_REPORTS = 10
IMAGE_URL = "/v1/gallery/images/{}"
PROTOCOL = "i2t, official test split, strict pairing (compute_retrieval_metrics, groups=None)"
COUNT_KEYS = ("images", "report_rows", "report_groups", "test")
META_COLUMNS = ("study_id", "view", "image", "file_sha256")       # all the queries read: subject and DICOM ids are never loaded
LABEL_FILES = ("labels.npy", "label_names.json")


class GalleryMismatch(RuntimeError):
    """A gallery the app must not serve. The message holds counts and file basenames only (R7)."""


def _topk(sims: np.ndarray, k: int) -> np.ndarray:
    k = min(k, sims.shape[0])
    idx = np.argpartition(-sims, k - 1)[:k]
    return idx[np.argsort(-sims[idx], kind="stable")]


def _clamp(k: Any, high: int) -> int:
    return max(0, min(int(k), high))


def _index(value: Any, size: int, what: str) -> int:
    """`value` as a row number in 0..size-1: IndexError (numbers only) outside it, and no wrap-around of a negative one; TypeError for
    anything that is not an integer (a float row is a bug, not a row)."""
    try:
        row = operator.index(value)
    except TypeError:
        raise TypeError("{} must be an integer".format(what)) from None
    if not 0 <= row < size:
        raise IndexError("{} {} is outside 0..{}".format(what, row, size - 1))
    return row


def _normalised(sha: Any) -> str:
    return str(sha).strip().lower()


# ── open(): the checks, cheapest first ────────────────────────────────────────────────────────────────────────────────────────────────

def _read_manifest(root: Path) -> Dict[str, Any]:
    path = root / "manifest.json"
    if not path.is_file():
        raise GalleryMismatch("manifest.json is missing")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):           # a JSONDecodeError and a UnicodeDecodeError are ValueErrors
        raise GalleryMismatch("manifest.json is not readable JSON") from None
    if not isinstance(manifest, dict):
        raise GalleryMismatch("manifest.json is not a JSON object")
    return manifest


def _read_counts(manifest: Dict[str, Any]) -> Dict[str, int]:
    """The four sizes every array is checked against. One image and one report per train study, so the report table is the train reports
    then the test reports (the layout test_study's report_texts[n_train + row] and the split maps rely on)."""
    raw = manifest.get("counts")
    if not isinstance(raw, dict):
        raise GalleryMismatch("manifest.json has no counts")
    counts = {}
    for key in COUNT_KEYS:
        value = raw.get(key)
        if type(value) is not int or value < 1:
            raise GalleryMismatch("manifest.json: counts.{} is not a positive integer".format(key))
        counts[key] = value
    if counts["report_rows"] != counts["images"] + counts["test"]:
        raise GalleryMismatch("manifest.json: {} report rows are not {} train + {} test".format(
            counts["report_rows"], counts["images"], counts["test"]))
    return counts


def _map(root: Path, name: str, kinds: str, rows: int, ndim: int) -> np.ndarray:
    """The array in one file, memory-mapped so that its header is checked (kind of number, dimensions, rows) before anything big is read.
    Pickles are refused. A file that cannot be read is named, not explained: the reason can hold a path."""
    try:
        array = np.load(str(root / name), mmap_mode="r", allow_pickle=False)
    except Exception:
        raise GalleryMismatch("{} is unreadable".format(name)) from None
    if array.dtype.kind not in kinds:
        raise GalleryMismatch("{} does not hold {} numbers".format(name, "floating-point" if kinds == "f" else "integer"))
    if array.ndim != ndim:
        raise GalleryMismatch("{} has {} dimensions, expected {}".format(name, array.ndim, ndim))
    if array.shape[0] != rows:
        raise GalleryMismatch("{} has {} rows, the manifest's counts say {}".format(name, array.shape[0], rows))
    return array


def _ram(mapped: np.ndarray, dtype: Any) -> np.ndarray:
    """A read-only RAM copy of a mapped array, in `dtype`."""
    array = np.array(mapped, dtype=dtype)
    array.setflags(write=False)
    return array


def _label_names(root: Path) -> List[str]:
    try:
        names = json.loads((root / "label_names.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise GalleryMismatch("label_names.json is not readable JSON") from None
    if names != CHEXBERT_14:
        raise GalleryMismatch("label_names.json does not hold the {} CheXbert names in the app's order".format(len(CHEXBERT_14)))
    return list(names)


def _check_layout(counts: Dict[str, int], a: Dict[str, np.ndarray]) -> None:
    """What the queries read off the arrays. report_matches takes group g to be the g-th segment of group_order and every row of that segment
    to carry the id g (np.maximum.reduceat over group_starts); test_study takes the report of test row r to be report row n_train + r; an
    image's own report is a row of the report table. Wrong in any of them and a query would answer, quietly, about the wrong rows."""
    n, groups, images, test = counts["report_rows"], counts["report_groups"], counts["images"], counts["test"]
    order, starts = a["group_order"], a["group_starts"]
    if int(order.min()) < 0 or int(order.max()) >= n or int(np.unique(order).size) != n:
        raise GalleryMismatch("group_order.npy is not a permutation of the {} report rows".format(n))
    if int(starts[-1]) >= n or not bool((np.diff(starts) > 0).all()):      # np.repeat below would raise on a negative size; a first start other than 0 fails the next test
        raise GalleryMismatch("group_starts.npy does not cut the {} report rows into {} ordered groups".format(n, groups))
    if not np.array_equal(a["txt_groups"][order], np.repeat(np.arange(groups), np.diff(np.append(starts, n)))):
        raise GalleryMismatch("txt_groups.npy, group_order.npy and group_starts.npy disagree about which of the {} report rows form a group".format(n))
    split, split_row = a["txt_split"], a["txt_split_row"]
    if not ((split[:images] == 0).all() and (split[images:] == 1).all() and np.array_equal(split_row[:images], np.arange(images))
            and np.array_equal(split_row[images:], np.arange(test))):
        raise GalleryMismatch("txt_split.npy and txt_split_row.npy do not put the {} train reports before the {} test reports".format(images, test))
    own = a["img_txt_row"]
    if int(own.min()) < 0 or int(own.max()) >= n:
        raise GalleryMismatch("img_txt_row.npy points outside the {} report rows".format(n))


def _read_texts(root: Path, rows: int) -> List[str]:
    """One report per line, as the builder writes them (whitespace already collapsed, so a text holds no line break). Bytes, not text mode:
    no newline translation."""
    try:
        raw = (root / "report_texts.txt").read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError):
        raise GalleryMismatch("report_texts.txt is not readable UTF-8 text") from None
    if not raw.endswith("\n"):
        raise GalleryMismatch("report_texts.txt does not end in a newline: a truncated file")
    texts = raw[:-1].split("\n")
    if len(texts) != rows:
        raise GalleryMismatch("report_texts.txt has {} reports, the manifest's counts say {}".format(len(texts), rows))
    return texts


def _read_meta(root: Path, name: str, key: str, rows: int) -> Dict[str, List[Any]]:
    """The columns the queries read of a meta parquet, as plain lists: one entry per row, the rows numbered 0.. in order."""
    try:
        frame = pd.read_parquet(str(root / name), columns=[key] + list(META_COLUMNS))
    except Exception:
        raise GalleryMismatch("{} is unreadable or lacks a column the queries need".format(name)) from None
    if len(frame) != rows:
        raise GalleryMismatch("{} has {} rows, the manifest's counts say {}".format(name, len(frame), rows))
    if not np.array_equal(frame[key].to_numpy(), np.arange(rows)):
        raise GalleryMismatch("{} does not number its {} rows 0.. in order".format(name, rows))
    return {column: frame[column].tolist() for column in META_COLUMNS}


def _first_rows(shas: List[Any]) -> Dict[str, int]:
    """file hash -> row. A file that is in the gallery twice is its first row."""
    rows = {}     # type: Dict[str, int]
    for row, sha in enumerate(shas):
        rows.setdefault(_normalised(sha), row)
    return rows


class Gallery:
    """An opened gallery: build it with Gallery.open. Read-only after that, so the one pipeline worker and the request threads (the picker, the
    thumbnails) may use it side by side."""

    def __init__(self, manifest: Dict[str, Any], counts: Dict[str, int], arrays: Dict[str, np.ndarray], labels: Optional[np.ndarray],
                 label_names: Optional[List[str]], report_texts: List[str], img_meta: Dict[str, List[Any]], test_meta: Dict[str, List[Any]]):
        self.manifest = manifest
        self.build_id = str(manifest.get("build_id") or "")
        self.n_train = counts["images"]
        self.img_emb, self.txt_emb, self.txt_emb_test = arrays["img_emb"], arrays["txt_emb"], arrays["txt_emb_test"]
        self.dim = int(self.img_emb.shape[1])
        self.txt_groups, self.group_order, self.group_starts = arrays["txt_groups"], arrays["group_order"], arrays["group_starts"]
        self.txt_test_groups, self.txt_split, self.txt_split_row = arrays["txt_test_groups"], arrays["txt_split"], arrays["txt_split_row"]
        self.img_txt_row = arrays["img_txt_row"]
        self.labels, self.label_names = labels, label_names
        self.report_texts = report_texts
        self._img_study, self._img_path = img_meta["study_id"], img_meta["image"]
        self._test_study, self._test_view, self._test_path = test_meta["study_id"], test_meta["view"], test_meta["image"]
        self._test_study_text = [str(s) for s in self._test_study]
        self._sha_train, self._sha_test = _first_rows(img_meta["file_sha256"]), _first_rows(test_meta["file_sha256"])

    @classmethod
    def open(cls, root: Path, expect_tower_sha256: Optional[str] = None) -> "Gallery":
        """Load the gallery in `root`, or raise GalleryMismatch. The refusals come cheapest first, so that a wrong gallery is turned away before
        its 0.8 GB are read: manifest.json is missing, unreadable or not an object; expect_tower_sha256 is given and is not the manifest's
        tower_sha256 (the vectors are not the engine's tower's); gate_rk.equal is not true (the R@k gate failed or never ran); a count is not a
        positive integer, or the counts do not add up; a file is missing; an array holds the wrong kind of number, has the wrong rows for the
        counts, or has a width that differs from the other embeddings'; the group, split and report-row layout the queries read does not hold;
        the texts or the meta files disagree with the counts. labels is None unless labels_status is "done"."""
        root = Path(root)
        manifest = _read_manifest(root)
        if expect_tower_sha256 is not None and manifest.get("tower_sha256") != expect_tower_sha256:
            raise GalleryMismatch("manifest.json: tower_sha256 is not the engine's: the gallery was built with another image tower")
        gate = manifest.get("gate_rk")
        if not (isinstance(gate, dict) and gate.get("equal") is True):
            raise GalleryMismatch("manifest.json: gate_rk.equal is not true: the build's R@k check failed or never ran")
        counts = _read_counts(manifest)
        labelled = manifest.get("labels_status") == "done"
        floats = {"img_emb.npy": counts["images"], "txt_emb.npy": counts["report_rows"], "txt_emb_test.npy": counts["test"]}
        ints = {"txt_groups.npy": counts["report_rows"], "group_order.npy": counts["report_rows"], "group_starts.npy": counts["report_groups"],
                "txt_test_groups.npy": counts["test"], "txt_split.npy": counts["report_rows"], "txt_split_row.npy": counts["report_rows"],
                "img_txt_row.npy": counts["images"]}
        wanted = list(floats) + list(ints) + ["img_meta.parquet", "test_meta.parquet", "report_texts.txt"] + (list(LABEL_FILES) if labelled else [])
        missing = [name for name in wanted if not (root / name).is_file()]
        if missing:
            raise GalleryMismatch("missing: " + ", ".join(missing))

        mapped = {name: _map(root, name, "f", rows, 2) for name, rows in floats.items()}
        mapped.update({name: _map(root, name, "iu", rows, 1) for name, rows in ints.items()})
        widths = {name: int(array.shape[1]) for name, array in mapped.items() if array.ndim == 2}
        if len(set(widths.values())) != 1:
            raise GalleryMismatch("the embedding files disagree about the dimension: " + ", ".join(
                "{} has {}".format(name, width) for name, width in widths.items()))
        if min(widths.values()) < 1:
            raise GalleryMismatch("the embedding files have no columns")
        label_names = labels = None
        if labelled:
            label_names = _label_names(root)
            mapped_labels = _map(root, "labels.npy", "iu", counts["report_rows"], 2)
            if mapped_labels.shape[1] != len(label_names):
                raise GalleryMismatch("labels.npy has {} columns, expected {}".format(mapped_labels.shape[1], len(label_names)))
            if int(mapped_labels.min()) < 0 or int(mapped_labels.max()) > 1:
                raise GalleryMismatch("labels.npy holds a value that is not 0 or 1")
            labels = _ram(mapped_labels, np.uint8)

        arrays = {name[:-4]: _ram(array, np.float32 if name in floats else np.int64) for name, array in mapped.items()}
        mapped.clear()
        _check_layout(counts, arrays)
        texts = _read_texts(root, counts["report_rows"])
        img_meta = _read_meta(root, "img_meta.parquet", "row", counts["images"])
        test_meta = _read_meta(root, "test_meta.parquet", "test_row", counts["test"])
        return cls(manifest, counts, arrays, labels, label_names, texts, img_meta, test_meta)

    @property
    def nbytes(self) -> int:
        """Bytes held in numpy arrays: the embeddings (float32 copies, nearly all of it), the group and split maps and the labels. Not the
        report texts or the meta lists, which are Python objects."""
        return sum(value.nbytes for value in vars(self).values() if isinstance(value, np.ndarray))

    # ── queries ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

    def _query(self, query: Any) -> np.ndarray:
        """The query as a unit float32 vector of the gallery's dimension, or ValueError. The message holds numbers only: a query is the
        fingerprint of an image."""
        try:
            with np.errstate(over="ignore", invalid="ignore"):          # a float64 beyond float32 becomes inf here, and is refused below
                q = np.asarray(query, dtype=np.float32)
        except (TypeError, ValueError):
            raise ValueError("the query is not a vector of numbers") from None
        if q.ndim != 1 or q.shape[0] != self.dim:
            raise ValueError("the query has shape {}, the gallery's vectors have {} dimensions".format(q.shape, self.dim))
        bad = int((~np.isfinite(q)).sum())
        if bad:
            raise ValueError("the query has {} values that are not finite".format(bad))
        wide = q.astype(np.float64)
        norm = float(np.linalg.norm(wide))
        if norm == 0.0:
            raise ValueError("the query has zero length")
        return (wide / norm).astype(np.float32)

    def _labels(self, txt_row: int) -> Optional[Dict[str, int]]:
        if self.labels is None:
            return None
        return {name: int(value) for name, value in zip(self.label_names, self.labels[txt_row])}

    def image_neighbors(self, query: np.ndarray, k: int) -> List[Dict[str, Any]]:
        """The k train images most similar to the query, best first (k is clamped to 0..12). Each: rank, similarity (the cosine), gallery_row,
        study_id, txt_row (the report of that study), image_url, and labels (those of that report, or None until they are built)."""
        q = self._query(query)
        k = _clamp(k, MAX_K_IMAGES)
        if k == 0:
            return []
        sims = self.img_emb @ q
        out = []
        for rank, row in enumerate(_topk(sims, k), start=1):
            row = int(row)
            txt_row = int(self.img_txt_row[row])
            out.append({"rank": rank, "similarity": float(sims[row]), "gallery_row": row, "study_id": self._img_study[row], "txt_row": txt_row,
                        "image_url": IMAGE_URL.format(row), "labels": self._labels(txt_row)})
        return out

    def report_matches(self, query: np.ndarray, k: int) -> List[Dict[str, Any]]:
        """The k best report groups, best first (k is clamped to 0..10). A group is the reports that read the same (group_ids_from_texts), so
        a templated report that thousands of studies share is one match, scored by its best member: similarity, the group and its size, and
        that member's row, text and labels."""
        q = self._query(query)
        k = _clamp(k, MAX_K_REPORTS)
        if k == 0:
            return []
        sims = self.txt_emb @ q
        group_max = np.maximum.reduceat(sims[self.group_order], self.group_starts)   # best member per group
        out = []
        for rank, g in enumerate(_topk(group_max, k), start=1):
            end = self.group_starts[g + 1] if g + 1 < len(self.group_starts) else len(self.group_order)
            members = self.group_order[self.group_starts[g]:end]
            best = int(members[np.argmax(sims[members])])
            out.append({"rank": rank, "similarity": float(group_max[g]), "group": int(g),
                        "group_size": int(len(members)), "txt_row": best,
                        "report": self.report_texts[best], "labels": self._labels(best)})
        return out

    def own_report_rank(self, query: np.ndarray, test_row: int) -> Dict[str, Any]:
        """The chapter's protocol (D5): strict pairing inside the test reports. rank counts the reports strictly more similar than test row
        `test_row`'s own, so a copy of the same report that scores exactly the same does not push it down; rank_dedup counts against the best
        copy of it (every report that reads the same), and is never worse."""
        row = _index(test_row, self.txt_emb_test.shape[0], "test_row")
        sims = self.txt_emb_test @ self._query(query)
        own = sims[row]
        same = self.txt_test_groups == self.txt_test_groups[row]
        rank = 1 + int((sims > own).sum())
        return {"rank": rank, "of": int(sims.shape[0]), "rank_dedup": 1 + int((sims > sims[same].max()).sum()),
                "hit_at_10": rank <= 10, "protocol": PROTOCOL}

    # ── identical files, test studies, thumbnails ─────────────────────────────────────────────────────────────────────────────────────

    def find_identical(self, file_sha256: str) -> Optional[Dict[str, Any]]:
        """{"split": "train"|"test", "row"} if the file with this SHA-256 is one of the gallery's image files (the first such row; train
        before test), else None."""
        key = _normalised(file_sha256)
        if key in self._sha_train:
            return {"split": "train", "row": self._sha_train[key]}
        if key in self._sha_test:
            return {"split": "test", "row": self._sha_test[key]}
        return None

    def image_path(self, gallery_row: int) -> Path:
        """The image file of a train gallery row. IndexError outside the gallery."""
        return Path(self._img_path[_index(gallery_row, len(self._img_path), "gallery_row")])

    def test_study(self, test_row: int) -> Dict[str, Any]:
        """A test-split study: its image file (Path), study_id, and reference, the study's own report as the dumps write it (the report table
        holds the train reports first, so it is row n_train + test_row). IndexError outside the split."""
        row = _index(test_row, len(self._test_path), "test_row")
        return {"image": Path(self._test_path[row]), "study_id": self._test_study[row], "reference": self.report_texts[self.n_train + row]}

    def list_test_studies(self, query: str = "", limit: int = 50) -> List[Dict[str, Any]]:
        """The test studies whose study id starts with `query` (surrounding space ignored), in test-row order, at most `limit`:
        test_row, study_id and view for each."""
        prefix = "" if query is None else str(query).strip()
        limit = max(0, int(limit))
        out = []
        for row, study in enumerate(self._test_study_text):
            if len(out) >= limit:
                break
            if study.startswith(prefix):
                out.append({"test_row": row, "study_id": self._test_study[row], "view": self._test_view[row]})
        return out
