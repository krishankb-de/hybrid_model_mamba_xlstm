"""The retrieval gallery of the chat app (CHAT_UI_PLAN.md P5-D): the files scripts/build_retrieval_gallery.py writes (section 6.5 of the plan),
loaded once and queried with the report model's image vector.

    Gallery.open(dir, expect_tower_sha256)   loads, or raises GalleryMismatch
    image_neighbors(q, k)     the k train images most like the query (at most 12), each with its study and its own report's labels
    report_matches(q, k)      the k best report groups (at most 10): duplicate reports are one group, scored by their best member
    own_report_rank(q, row)   where test study `row`'s own report ranks among the test reports, by the retrieval chapter's strict pairing; a
                              report tied with it counts in its favour, which compute_retrieval_metrics(groups=None) does not do (D5)
    find_identical(sha)       whether an uploaded file is, byte for byte, a train or a test image
    test_study, list_test_studies, image_path   the test-split picker and the neighbour thumbnails
    facts()                   the five fields of the retrieve stage's `gallery` detail: all of the manifest that may travel

open() refuses a gallery it cannot vouch for: a tower other than the engine's, a build whose R@k gate was not decided equal (manifest
gate_rk.equal is not true: it failed, or never ran), an image projection after the tower (the vectors would sit in another space than the
engine's pooled one), a missing file, or arrays that disagree with manifest["counts"] or with the layout the queries read off them. Embeddings
are held as float32 RAM copies (about 0.8 GB at full size; `nbytes`), read-only, and every query vector is normalised here so that a
similarity is a cosine whatever the caller passes.

R1 and R7. This module returns data: report texts, study ids, rows, image paths. What may leave the cluster in public mode is app/redact.py's
decision, applied by the pipeline. The manifest is private (it holds checkpoint paths, which carry the username, the commit and the job):
facts() is what to send. The module logs nothing, and an error message holds counts and file basenames only, never a path, an id or a text.
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
# What own_report_rank's number is, said for whoever stores or shows it. The chapter's function does not count a tie the way rank does: its
# argpartition decides among equal similarities by their position in the array, not by anything about the reports (for k=1 the first one
# wins), so under ties its recall and the recall of these ranks are not the same number.
PROTOCOL = ("i2t, official test split, strict pairing: a tied copy counts in this report's favour, "
            "while compute_retrieval_metrics(groups=None) breaks such ties by array position")
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
    # A start at n is a group with no rows, which np.maximum.reduceat cannot take (>=, not >: the ids and the counts can still line up); a falling
    # start would make np.repeat raise below; a first start other than 0 fails the next test.
    if int(starts[-1]) >= n or not bool((np.diff(starts) > 0).all()):
        raise GalleryMismatch("group_starts.npy does not cut the {} report rows into {} ordered groups".format(n, groups))
    if not np.array_equal(a["txt_groups"][order], np.repeat(np.arange(groups), np.diff(np.append(starts, n)))):
        raise GalleryMismatch("txt_groups.npy, group_order.npy and group_starts.npy disagree about which of the {} report rows form a group".format(n))
    split, split_row = a["txt_split"], a["txt_split_row"]
    if not ((split[:images] == 0).all() and (split[images:] == 1).all() and np.array_equal(split_row[:images], np.arange(images))
            and np.array_equal(split_row[images:], np.arange(test))):
        raise GalleryMismatch("txt_split.npy and txt_split_row.npy do not put the {} train reports before the {} test reports".format(images, test))
    own = a["img_txt_row"]
    if int(own.min()) < 0 or int(own.max()) >= n:         # >= : a report row n is one past the last, and no other check would see it
        raise GalleryMismatch("img_txt_row.npy points outside the {} report rows".format(n))


def _read_texts(root: Path, rows: int) -> List[str]:
    """One report per line, as the builder writes them (whitespace already collapsed, so a text holds no line break). Read line by line, in
    binary mode (no newline translation): the file, some 150 MB at full size, is never held whole beside its list of strings."""
    texts = []     # type: List[str]
    last = b"\n"
    try:
        with open(str(root / "report_texts.txt"), "rb") as handle:
            for line in handle:
                texts.append(line[:-1].decode("utf-8"))
                last = line
    except (OSError, UnicodeDecodeError):
        raise GalleryMismatch("report_texts.txt is not readable UTF-8 text") from None
    if not last.endswith(b"\n"):
        raise GalleryMismatch("report_texts.txt does not end in a newline: a truncated file")
    if len(texts) != rows:
        raise GalleryMismatch("report_texts.txt has {} reports, the manifest's counts say {}".format(len(texts), rows))
    return texts


def _json_values(series: Any) -> List[Any]:
    """A column as a list of plain values for JSON, in which a missing one (None, NaN or NA) is None: NaN is not valid JSON, and the browser's
    JSON.parse refuses it in an SSE data line. MIMIC leaves some views blank."""
    values = series.tolist()
    if series.isna().any():
        values = [None if pd.isna(value) else value for value in values]
    return values


def _read_meta(root: Path, name: str, key: str, rows: int) -> Dict[str, List[Any]]:
    """The columns the queries read of a meta parquet, as plain lists: one entry per row, the rows numbered 0.. in order. study_id and view
    are returned as they are, so they come out JSON-clean (a missing one is None); the path and the hash are only looked up."""
    try:
        frame = pd.read_parquet(str(root / name), columns=[key] + list(META_COLUMNS))
    except Exception:
        raise GalleryMismatch("{} is unreadable or lacks a column the queries need".format(name)) from None
    if len(frame) != rows:
        raise GalleryMismatch("{} has {} rows, the manifest's counts say {}".format(name, len(frame), rows))
    if not np.array_equal(frame[key].to_numpy(), np.arange(rows)):
        raise GalleryMismatch("{} does not number its {} rows 0.. in order".format(name, rows))
    return {"study_id": _json_values(frame["study_id"]), "view": _json_values(frame["view"]),
            "image": frame["image"].tolist(), "file_sha256": frame["file_sha256"].tolist()}


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
        self._manifest = manifest             # private: it holds checkpoint paths, the commit and the job. What may travel is facts()
        self._counts = dict(counts)
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
        self._test_study_text = ["" if s is None else str(s) for s in self._test_study]      # a missing id starts with nothing
        self._sha_train, self._sha_test = _first_rows(img_meta["file_sha256"]), _first_rows(test_meta["file_sha256"])

    @classmethod
    def open(cls, root: Path, expect_tower_sha256: Optional[str] = None) -> "Gallery":
        """Load the gallery in `root`, or raise GalleryMismatch. The refusals come cheapest first, so that a wrong gallery is turned away before
        its 0.8 GB are read: manifest.json is missing, unreadable or not an object; expect_tower_sha256 is given and is not the manifest's
        tower_sha256 (the vectors are not the engine's tower's); gate_rk.equal is not true (the R@k gate failed or never ran); img_proj_present is
        not false (a projection after the tower: matching tower hashes do not rule it out, and the engine's pooled vector has none); a count is
        not a positive integer, or the counts do not add up; a file is missing; an array holds the wrong kind of number, has the wrong rows for
        the counts, or has a width that differs from the other embeddings'; the group, split and report-row layout the queries read does not
        hold (a group that starts at row n, a report row n, and so on); the texts or the meta files disagree with the counts. labels is None
        unless labels_status is "done". manifest["towers_identical"] is not enforced: it describes the build's decoder checkpoint, not these
        vectors, and facts() passes it on."""
        root = Path(root)
        manifest = _read_manifest(root)
        if expect_tower_sha256 is not None and manifest.get("tower_sha256") != expect_tower_sha256:
            raise GalleryMismatch("manifest.json: tower_sha256 is not the engine's: the gallery was built with another image tower")
        gate = manifest.get("gate_rk")
        if not (isinstance(gate, dict) and gate.get("equal") is True):
            raise GalleryMismatch("manifest.json: gate_rk.equal is not true: the build's R@k check failed or never ran")
        if manifest.get("img_proj_present") is not False:
            raise GalleryMismatch("manifest.json: img_proj_present is not false: the vectors may sit behind a projection the engine's query lacks")
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
        """Bytes in the numpy arrays only: the embeddings (float32 copies, nearly all of it), the group and split maps and the labels. It is not
        the memory the gallery takes. The report texts, the meta lists and the hash maps are Python objects and are not counted, and the
        half-precision files stay mapped until open() returns. On a synthetic full-size gallery the process grew by 1.2 times this once open()
        returned and by 1.5 times at the peak inside it; a review run saw about 2 times, and real report texts are longer than synthetic ones.
        Plan a node's memory on the process, not on this number."""
        return sum(value.nbytes for value in vars(self).values() if isinstance(value, np.ndarray))

    def facts(self) -> Dict[str, Any]:
        """The five fields of the retrieve stage's `gallery` detail (section 6.3) and nothing else, as a fresh dict: build_id, images,
        report_rows, report_groups and towers_identical (true only if the manifest says true). The manifest also holds checkpoint paths, which
        carry the username, the commit and the job, and none of that may travel (R1): send this, never the manifest."""
        return {"build_id": self.build_id, "images": self._counts["images"], "report_rows": self._counts["report_rows"],
                "report_groups": self._counts["report_groups"], "towers_identical": self._manifest.get("towers_identical") is True}

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
        """Strict pairing inside the test reports (D5), where the chapter's function and this one part ways on ties. rank counts the reports
        strictly more similar than test row `test_row`'s own, so a copy that scores exactly the same does not push it down: ties are decided in
        its favour, the best rank it can have. compute_retrieval_metrics(groups=None) decides them by position in the array, so under ties its
        recall is not the recall of these ranks. n_tied is how many other reports score exactly the same (the copies of it, in practice), so
        rank + n_tied is the worst rank it can have. rank_dedup counts against the best copy of it (every report that reads the same), which
        ties cannot move, and is never worse than rank."""
        row = _index(test_row, self.txt_emb_test.shape[0], "test_row")
        sims = self.txt_emb_test @ self._query(query)
        own = sims[row]
        same = self.txt_test_groups == self.txt_test_groups[row]
        rank = 1 + int((sims > own).sum())
        return {"rank": rank, "of": int(sims.shape[0]), "rank_dedup": 1 + int((sims > sims[same].max()).sum()),
                "n_tied": int((sims == own).sum()) - 1, "hit_at_10": rank <= 10, "protocol": PROTOCOL}

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
