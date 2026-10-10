"""P5-F (CHAT_UI_PLAN.md): the gates of the live path, on the real gallery and the real labeller, on the CPU.

    python scripts/chat_retrieval_gates.py --checkpoint <report model last.ckpt> --gallery <gallery dir> --data <dataset dir> \\
        --published-labels <chexbert_labels.json> --published-hyps <hyps.txt> --published-refs <refs.txt> \\
        --labeler-url http://127.0.0.1:<port> --out <results dir> [--model-config hybrid_150m_m3_rrg] [--threads 8]

Three checks, each through the code a turn runs (app.engine, app.gallery, app.labels), none of it re-implemented here:

  1. Self-retrieval. 50 train images, spread evenly over the split, go through Engine.preprocess and Engine.encode from their file BYTES
     (the path of an upload), and Gallery.image_neighbors must put each of them first. A miss is recorded with its row, the row found
     instead, the gap between the first two similarities, and where the image's own row stands among the k=12 nearest (own_rank, 0 when
     it is not among them) and how far behind the first it is (own_gap), so that a duplicate image (a tie: own_rank=2 own_gap=0.000000)
     reads differently from a wrong embedding (own_rank=0). Gated: all 50.
  2. Live own-rank. For the first 50 test studies, Gallery.own_report_rank is asked with the live vector and with the build's own
     embedding of the same image (test_img_emb.npy, the H100's): one formula on both sides, so the two ranks are comparable, for `rank`
     and for `rank_dedup`. Counted and recorded, not gated: the live vector is the CPU's, and a CPU and a GPU can swap two reports whose
     similarities are a rounding error apart (D6). The prediction is at least 48 of 50.
  3. The labeller. The first 50 hyps and the first 50 refs of the published dump go through the labeller service (LabelerClient, one call of
     50 each; its limit is 64) and must come back equal to the dump's y_pred and y_true, row for row. Gated: 100 of 100, and a label order
     that is CHEXBERT_14's: the dump's own `label_names` (a dump without them is read in CHEXBERT_14 order, with a note, and one that names the
     same labels in another order is reordered by name, as scripts/label_gallery_reports.py reads it). The client itself refuses a service
     that names another order, which is the check of CHEXBERT_14 against the live service.

The gallery is opened against the ENGINE's tower hash, so that check 1 cannot compare across embedding spaces: a mismatch is
`ERROR gallery tower mismatch`. The published files are read first, since a wrong dump is known in a second and the engine takes a minute.

Writes <out>/gates.json once the output directory can be made, whatever happens after: numbers, indices and a few fixed words, with whatever
was measured before a refusal (and `error`, `passed`). Exit 0 when all three gates hold, 1 otherwise (a refusal too). --engine tiny is for
the laptop tests (tests/test_chat_retrieval_gates.py: random-init stand-ins, no checkpoint); the job never passes it.

R7. What is printed here could be MIMIC-derived (a report, an id, a path), so none of that is: every line is a `[gates]` line, a `RESULT {json}`
line, an `ERROR <code> [name=number ...]` line or a `=== note ... ===` line of a fixed shape, with counts, booleans, row numbers and
similarities. No path, no id, no report text, no exception message (an unexpected exception prints its class name; its traceback, and the
words of a refusal by the gallery or the labeller client, which hold counts and file names, go to stderr). The wrapper lets only lines of those
shapes into the job log (its GATES_SHAPES, tested against everything printed here) and keeps the rest in gates.log.
"""
import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))      # app comes from this tree, however the venv's install points

from app.engine import build_engine  # noqa: E402
from app.gallery import MAX_K_IMAGES, Gallery, GalleryMismatch  # noqa: E402
from app.labels import CHEXBERT_14, LabelerClient, LabelerUnavailable  # noqa: E402

N_SELF = 50                 # train images that must find themselves
N_RANK = 50                 # test studies whose live own-rank is compared with the build's
N_LABELLED = 50             # hyps, and refs, the labeller is checked on: one call each (LabelerClient's limit is 64 texts)
LABELER_TIMEOUT_S = 120
K_NEIGHBOURS = MAX_K_IMAGES  # how far down the neighbours an image's own row is looked for when it is not first: all the gallery gives (12)
MAX_MISS_LINES = 20         # miss lines in the job log; gates.json has every miss
N_LABELS = len(CHEXBERT_14)
DEFAULT_MODEL_CONFIG = "hybrid_150m_m3_rrg"

# Every reason this script stops with, as the text of an `ERROR <code> [name=number ...]` line. The wrapper's allowlist names exactly these.
ERROR_CODES = ("gallery tower mismatch", "gallery refused", "data rows disagree", "published unreadable", "published shape",
               "labeller order mismatch", "labeller unavailable")


# ── what is printed (R7) ──────────────────────────────────────────────────────

def say(message: str) -> None:
    """One [gates] line: counts, booleans, row numbers and similarities, never a path, an id or report text. The wrapper passes a line to the
    job log only if it has one of the shapes in its GATES_SHAPES, so a new line needs a new shape there and in the tests."""
    print("[gates] " + message, flush=True)


def note(message: str) -> None:
    print("=== note: " + message + " ===", flush=True)


def emit_result(payload: Dict[str, Any]) -> None:
    print("RESULT " + json.dumps(payload, separators=(",", ":")), flush=True)


class Refused(Exception):
    """A reason to stop that the job log may show: a code from ERROR_CODES and whole numbers, never a path, a name, an id or text."""

    def __init__(self, code: str, **numbers: int):
        assert code in ERROR_CODES, code
        super().__init__(code)
        self.code, self.numbers = code, numbers

    def line(self) -> str:
        return "ERROR " + self.code + "".join(" {}={}".format(key, int(value)) for key, value in self.numbers.items())


def exception_name(exc: BaseException) -> str:
    """The class name of an exception in the one shape an `ERROR failed` line takes (letters, digits and underscores, at most 60)."""
    name = re.sub(r"[^A-Za-z0-9_]", "_", type(exc).__name__)[:60]
    return name if re.match(r"[A-Za-z_]", name) else "_" + name[:59]


def write_json_atomic(path: Path, obj: Any) -> None:
    """Written whole or not at all."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(str(tmp), str(path))


# ── the pieces ────────────────────────────────────────────────────────────────

def sample_rows(n_rows: int, n: int) -> np.ndarray:
    """`n` rows spread evenly from the first to the last of `n_rows` (the brief's np.linspace(0, len(train) - 1, 50).astype(int))."""
    return np.linspace(0, n_rows - 1, n).astype(int)


def read_lines(path: Path) -> List[str]:
    """A report file the way the scoring job reads it (score_chexbert_standalone.py): splitlines. The reports have had their whitespace
    collapsed to single spaces, so no line break can hide inside one."""
    return path.read_text(encoding="utf-8").splitlines()


def read_published(path: Path) -> Dict[str, Any]:
    """The y_pred and y_true of the published chexbert_labels.json as (n, 14) matrices in CHEXBERT_14 order, whether the names are right
    (`names_ok`) and where the order came from (`names_source`). The file comes in two formats, with a `label_names` key (a different order
    of the same names is put right by name; any other names are `foreign`, which fails the names gate) and without (CHEXBERT_14 order is
    assumed, and a note says so: the comparison with the live labeller then carries the whole check)."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        y_pred, y_true = np.array(payload["y_pred"], dtype=np.int64), np.array(payload["y_true"], dtype=np.int64)
        has_names, names = "label_names" in payload, payload.get("label_names")
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        raise Refused("published unreadable") from None
    if any(rows.ndim != 2 or rows.shape[1] != N_LABELS for rows in (y_pred, y_true)):
        raise Refused("published unreadable")
    source, names_ok = "assumed", True
    if not has_names:
        note("chexbert_labels.json has no label_names key: CHEXBERT_14 order assumed")
    elif names == CHEXBERT_14:
        source = "dump"
    elif isinstance(names, list) and len(names) == N_LABELS and all(isinstance(n, str) for n in names) and sorted(names) == sorted(CHEXBERT_14):
        order = [names.index(n) for n in CHEXBERT_14]
        y_pred, y_true = y_pred[:, order], y_true[:, order]
        source = "reordered"
        note("chexbert_labels.json label_names reordered to the CHEXBERT_14 order")
    else:
        source, names_ok = "foreign", False
    return {"y_pred": y_pred, "y_true": y_true, "names_source": source, "names_ok": names_ok}


def make_engine(args: argparse.Namespace) -> Any:
    if args.engine == "tiny":
        return build_engine("tiny")
    return build_engine("real", checkpoint=args.checkpoint, model_config=args.model_config, device="cpu", threads=args.threads)


def open_gallery(root: Path, engine: Any) -> Gallery:
    """The gallery, opened against the engine's own tower hash. GalleryMismatch is `gallery tower mismatch` when the manifest's tower is not the
    engine's (open() checks that first, so a differing hash is always that refusal) and `gallery refused` for any other reason; its words
    (counts and file names) go to stderr."""
    tower = engine.tower_sha256()
    try:
        return Gallery.open(root, expect_tower_sha256=tower)
    except GalleryMismatch as exc:
        print("gallery: {}".format(exc), file=sys.stderr, flush=True)
        try:
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            manifest = None
        if isinstance(manifest, dict) and manifest.get("tower_sha256") != tower:
            raise Refused("gallery tower mismatch") from None
        raise Refused("gallery refused") from None


def miss_record(row: int, top: List[Dict[str, Any]]) -> Dict[str, Any]:
    """A self-retrieval miss, in numbers only. `got` is the row found first and `gap` its similarity lead over the second. `own_rank` is where the
    image's own row stands among the k neighbours asked for (1 is first; 0: it is not among them) and `own_gap` how far its similarity is behind the
    first one's; with own_rank 0 that is the gap to the k-th neighbour, which the own row is at least as far behind as. A duplicate image reads
    own_rank=2 own_gap=0.000000: a tie that the search broke the other way, not a wrong embedding. k is named, so that 0 can be read."""
    own = next((n for n in top if n["gallery_row"] == row), None)
    behind = top[-1] if own is None else own
    return {"row": row, "got": int(top[0]["gallery_row"]), "gap": float(top[0]["similarity"] - top[1]["similarity"]),
            "own_rank": 0 if own is None else int(own["rank"]), "own_gap": float(top[0]["similarity"] - behind["similarity"]), "k": len(top)}


def live_vector(engine: Any, image_path: Any) -> np.ndarray:
    """The vector an upload of this file gets: Engine.preprocess from the file's bytes, then Engine.encode. Encoded.pooled is the query (D4)."""
    _, prepared = engine.preprocess(Path(image_path).read_bytes())
    _, encoded = engine.encode(prepared)
    return encoded.pooled.numpy()


def label_with(labeller: Any, texts: List[str], expected: np.ndarray) -> List[int]:
    """The rows (within `texts`) where the labeller's 14 labels are not the published ones. LabelerUnavailable is a refusal: the client raises
    it for a service that is down, that answers something that is not 14 zeros and ones per text, or that names another label order."""
    try:
        got = labeller.label(texts)
    except LabelerUnavailable as exc:
        print("labeller: {}".format(exc), file=sys.stderr, flush=True)
        raise Refused("labeller order mismatch" if str(exc).startswith("label order mismatch") else "labeller unavailable") from None
    return [i for i, (a, b) in enumerate(zip(got, expected.tolist())) if a != b] + list(range(len(got), len(texts)))


def failed_gates(result: Dict[str, Any]) -> List[str]:
    """The `ERROR gate` lines of the gates that did not hold: self_retrieval == 50, labeller_equal == 100 and label_names_ok."""
    failed = []
    if result["self_retrieval"] != N_SELF:
        failed.append("ERROR gate self_retrieval={} expected={}".format(result["self_retrieval"], N_SELF))
    if result["labeller_equal"] != 2 * N_LABELLED:
        failed.append("ERROR gate labeller_equal={} expected={}".format(result["labeller_equal"], 2 * N_LABELLED))
    if not result["label_names_ok"]:
        failed.append("ERROR gate label_names_ok=false")
    return failed


# ── the job ───────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace, labeller: Any = None, result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The three checks. `result` is filled as each part completes, so that a refusal leaves what was measured before it; `labeller` is
    anything with label(texts) (the tests' RuleLabeler), by default the service at args.labeler_url. -> result."""
    result = {} if result is None else result
    gallery_dir, data_dir = Path(args.gallery), Path(args.data)

    # The cheap inputs first: a dump that cannot be read or does not line up is known before the engine is built.
    published = read_published(Path(args.published_labels))
    hyps, refs = read_lines(Path(args.published_hyps)), read_lines(Path(args.published_refs))
    counts = {"hyps": len(hyps), "y_pred": len(published["y_pred"]), "refs": len(refs), "y_true": len(published["y_true"])}
    if counts["hyps"] != counts["y_pred"] or counts["refs"] != counts["y_true"]:
        raise Refused("published shape", **counts)
    result.update(label_names_ok=published["names_ok"], label_names_source=published["names_source"])

    engine = make_engine(args)
    card = engine.card()
    say("engine device={} threads={}".format(card["device"], card["threads"]))
    result["engine"] = {"device": str(card["device"]), "threads": int(card["threads"])}
    gallery = open_gallery(gallery_dir, engine)
    facts = gallery.facts()
    labels_status = "pending" if gallery.labels is None else "done"
    # open() refuses a manifest whose img_proj_present is not false, so reaching this line is the report on it.
    say("gallery images={} report_rows={} report_groups={} towers_identical={} img_proj_present=false labels_status={}".format(
        facts["images"], facts["report_rows"], facts["report_groups"], "true" if facts["towers_identical"] else "false", labels_status))
    result["gallery"] = {"images": facts["images"], "report_rows": facts["report_rows"], "report_groups": facts["report_groups"],
                         "towers_identical": facts["towers_identical"], "labels_status": labels_status}

    # Only the image column: the other columns hold the reports.
    train = pd.read_parquet(data_dir / "train.parquet", columns=["image"])
    test = pd.read_parquet(data_dir / "test.parquet", columns=["image"])
    test_img = np.load(str(gallery_dir / "test_img_emb.npy"))
    # The rows of the dataset must be the gallery's: row r of train.parquet is gallery row r. (gallery_test is the rows of the file this job
    # reads itself; the gallery's own report vectors for the test split must agree with it.)
    sizes = {"train": len(train), "gallery_train": facts["images"], "test": len(test), "gallery_test": int(test_img.shape[0])}
    if sizes["train"] != sizes["gallery_train"] or sizes["test"] != sizes["gallery_test"] or gallery.txt_emb_test.shape[0] != sizes["gallery_test"]:
        raise Refused("data rows disagree", **sizes)
    seconds = result["seconds"] = {}

    # 1. self-retrieval through the live upload path
    started = time.perf_counter()
    hits = 0
    misses: List[Dict[str, Any]] = []
    for row in sample_rows(len(train), N_SELF):
        row = int(row)
        top = gallery.image_neighbors(live_vector(engine, train["image"].iloc[row]), K_NEIGHBOURS)
        if top[0]["gallery_row"] == row:
            hits += 1
        else:
            misses.append(miss_record(row, top))
    seconds["self_retrieval"] = round(time.perf_counter() - started, 2)
    say("self_retrieval hits={} of={} misses={}".format(hits, N_SELF, len(misses)))
    for miss in misses[:MAX_MISS_LINES]:
        say("miss row={row} got={got} gap={gap:.6f} own_rank={own_rank} own_gap={own_gap:.6f} k={k}".format(**miss))
    result.update(self_retrieval=hits, self_retrieval_of=N_SELF, misses=misses)

    # 2. the live own-rank against the build's, one formula on both sides
    started = time.perf_counter()
    equal = dedup_equal = max_diff = 0
    differs: List[Dict[str, Any]] = []
    for t in range(N_RANK):
        live = gallery.own_report_rank(live_vector(engine, test["image"].iloc[t]), t)
        built = gallery.own_report_rank(test_img[t], t)
        same, same_dedup = live["rank"] == built["rank"], live["rank_dedup"] == built["rank_dedup"]
        equal, dedup_equal = equal + int(same), dedup_equal + int(same_dedup)
        max_diff = max(max_diff, abs(int(live["rank"]) - int(built["rank"])))
        if not (same and same_dedup):
            differs.append({"test_row": t, "rank": [int(live["rank"]), int(built["rank"])],
                            "rank_dedup": [int(live["rank_dedup"]), int(built["rank_dedup"])]})
    seconds["own_rank"] = round(time.perf_counter() - started, 2)
    say("own_rank equal={} dedup_equal={} of={} max_diff={}".format(equal, dedup_equal, N_RANK, max_diff))
    result.update(own_rank_equal=equal, own_rank_dedup_equal=dedup_equal, own_rank_of=N_RANK, own_rank_max_diff=max_diff, own_rank_differs=differs)

    # 3. the labeller service against the published labels
    started = time.perf_counter()
    labeller = LabelerClient(args.labeler_url, timeout=LABELER_TIMEOUT_S) if labeller is None else labeller
    hyps_differ = label_with(labeller, hyps[:N_LABELLED], published["y_pred"])
    refs_differ = label_with(labeller, refs[:N_LABELLED], published["y_true"])
    seconds["labeller"] = round(time.perf_counter() - started, 2)
    pred_equal, true_equal = len(hyps[:N_LABELLED]) - len(hyps_differ), len(refs[:N_LABELLED]) - len(refs_differ)
    say("labeller pred_equal={} true_equal={} of={}".format(pred_equal, true_equal, N_LABELLED))
    result.update(labeller_equal=pred_equal + true_equal, labeller_of=2 * N_LABELLED, label_mismatch_rows={"hyps": hyps_differ, "refs": refs_differ})
    return result


def finish(args: argparse.Namespace, result: Dict[str, Any], code: int) -> int:
    """gates.json, then the exit code. A directory that cannot be made, or a file that cannot be written, is a failure of its own."""
    result["passed"] = code == 0 and "error" not in result
    try:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        write_json_atomic(out / "gates.json", result)
    except OSError as exc:
        traceback.print_exc()
        print("ERROR failed " + exception_name(exc), flush=True)
        return 1
    return code


def main(argv: Optional[Sequence[str]] = None, labeller: Any = None) -> int:
    args = parse_args(argv)
    result: Dict[str, Any] = {}
    try:
        run(args, labeller=labeller, result=result)
    except Refused as refusal:
        print(refusal.line(), flush=True)
        result["error"] = refusal.code
        return finish(args, result, 1)
    except Exception as exc:       # noqa: BLE001  any other failure: its class name for the job log, its traceback for the raw one
        traceback.print_exc()
        print("ERROR failed " + exception_name(exc), flush=True)
        result["error"] = "failed " + exception_name(exc)
        return finish(args, result, 1)
    failed = failed_gates(result)
    code = finish(args, result, 1 if failed else 0)
    if code == 0 or failed:
        emit_result({"self_retrieval": result["self_retrieval"], "own_rank_equal": result["own_rank_equal"],
                     "own_rank_dedup_equal": result["own_rank_dedup_equal"], "labeller_equal": result["labeller_equal"],
                     "label_names_ok": result["label_names_ok"]})
        for line in failed:
            print(line, flush=True)
    return code


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True, help="the report model's checkpoint (its image tower is the engine's)")
    p.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG, help="its model config (default: %(default)s)")
    p.add_argument("--gallery", required=True, help="the gallery directory (CHAT_HOME/gallery/<build id>)")
    p.add_argument("--data", required=True, help="the dataset directory with train.parquet and test.parquet")
    p.add_argument("--published-labels", required=True, help="the published dump's chexbert_labels.json")
    p.add_argument("--published-hyps", required=True, help="the published dump's hyps.txt")
    p.add_argument("--published-refs", required=True, help="the published dump's refs.txt")
    p.add_argument("--labeler-url", required=True, help="the labeller service, for example http://127.0.0.1:8001")
    p.add_argument("--out", required=True, help="the directory gates.json is written to")
    p.add_argument("--threads", type=int, default=8, help="CPU threads of the engine (default: %(default)s)")
    p.add_argument("--engine", choices=("real", "tiny"), default="real", help="tiny: random-init stand-ins, for the laptop tests only")
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
