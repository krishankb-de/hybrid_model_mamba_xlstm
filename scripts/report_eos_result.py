"""P9-G3 (CHAT_UI_PLAN.md): the one RESULT line the EOS training job prints when training is over.

    python scripts/report_eos_result.py --out-dir <run dir> --published <published run_metadata.json> --wall-s <seconds>

    RESULT {"train":"done","steps":N,"wall_s":S,"val_loss":x,"published_val_loss":y,"ckpt_exists":true}

steps is the number of optimizer steps the run's newest TensorBoard version covers: its largest step stamp plus one. Lightning
stamps a step with the number completed BEFORE it (validation runs before the step is counted), so a finished 12000-step run
ends at stamp 11999; tests/test_report_eos_job.py measures that against the installed Lightning. The wrapper compares steps
with MAX_STEPS before it writes DONE. val_loss is the last val/lm_loss that version logged, and published_val_loss the same for
the published run (read from the event files under its logs/ directory, the sibling of its run_metadata.json).
TensorBoardLogger starts a new version_N for every attempt, so a requeued run is read from its last attempt. Both runs
validate on the same split, but the EOS run's loss counts one more token per report, so the two are close, not equal.

The reader is tensorboard's (requirements.txt lists it), or tbparse's when that is the one installed. With neither
importable the losses and steps are null, the checkpoint basename is added, and a === line says so. A run with no events
gets a === line too. R7: stdout is numbers, flags and one basename, plus === lines without paths.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

VAL_TAG = "val/lm_loss"
CKPT_NAME = "last.ckpt"
_VERSION = re.compile(r"^version_(\d+)$")

Scalars = Dict[str, List[Tuple[int, float]]]


def event_dirs(log_dir: Path) -> List[Path]:
    """Directories under log_dir that hold TensorBoard event files, newest first: the highest version_N, then the rest by
    modification time."""
    found = {p.parent for p in Path(log_dir).rglob("events.out.tfevents.*")}

    def age(directory: Path):
        match = _VERSION.match(directory.name)
        return (int(match.group(1)) if match else -1, directory.stat().st_mtime)

    return sorted(found, key=age, reverse=True)


def _read_with_tensorboard(version_dir: Path) -> Scalars:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    acc = EventAccumulator(str(version_dir), size_guidance={"scalars": 0})    # 0 keeps every point
    acc.Reload()
    return {tag: [(int(e.step), float(e.value)) for e in acc.Scalars(tag)] for tag in acc.Tags().get("scalars", [])}


def _read_with_tbparse(version_dir: Path) -> Scalars:
    from tbparse import SummaryReader
    frame = SummaryReader(str(version_dir)).scalars                            # long format: step, tag, value
    found: Scalars = {}
    for tag, step, value in zip(frame["tag"], frame["step"], frame["value"]):
        found.setdefault(str(tag), []).append((int(step), float(value)))
    return found


def read_scalars(version_dir: Path) -> Scalars:
    """tag -> [(step, value)] for one version directory: through tensorboard, or through tbparse when only that is
    installed. Raises ImportError when neither can be imported."""
    try:
        return _read_with_tensorboard(version_dir)
    except ImportError:
        return _read_with_tbparse(version_dir)


def run_summary(log_dir: Path, read: Callable[[Path], Scalars] = read_scalars) -> Optional[Dict[str, Any]]:
    """{"steps", "val_loss"} from the newest version directory under log_dir that logged any scalar, or None. steps counts
    optimizer steps: the largest step stamp plus one, stamps being zero-based (see the module docstring)."""
    for version_dir in event_dirs(log_dir):
        scalars = read(version_dir)
        steps = [step for points in scalars.values() for step, _ in points]
        if not steps:
            continue
        val = sorted(scalars.get(VAL_TAG, []), key=lambda point: point[0])
        return {"steps": max(steps) + 1, "val_loss": val[-1][1] if val else None}
    return None


def _rounded(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(value, 4)


def build_result(out_dir: Path, published_meta: Path, wall_s: int,
                 read: Callable[[Path], Scalars] = read_scalars) -> Tuple[Dict[str, Any], List[str]]:
    """(the RESULT payload, the === notes that explain a null in it)."""
    out_dir = Path(out_dir)
    result: Dict[str, Any] = {
        "train": "done", "steps": None, "wall_s": int(wall_s), "val_loss": None, "published_val_loss": None,
        "ckpt_exists": (out_dir / "checkpoints" / CKPT_NAME).is_file(),
    }
    try:
        own = run_summary(out_dir / "logs", read)
        published = run_summary(Path(published_meta).parent / "logs", read)
    except ImportError:
        result["ckpt"] = CKPT_NAME
        return result, ["=== tensorboard is not importable here: steps and both validation losses are null ==="]
    notes = []
    if own is None or own["val_loss"] is None:
        notes.append("=== no validation loss was logged for this run ===")
    if own is not None:
        result["steps"], result["val_loss"] = own["steps"], _rounded(own["val_loss"])
    if published is None or published["val_loss"] is None:
        notes.append("=== no validation loss in the published run's events: published_val_loss is null ===")
    if published is not None:
        result["published_val_loss"] = _rounded(published["val_loss"])
    return result, notes


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", required=True, help="the EOS run's output directory")
    parser.add_argument("--published", required=True, help="the published run's run_metadata.json")
    parser.add_argument("--wall-s", type=int, required=True, help="wall time of the training call, in seconds")
    args = parser.parse_args(argv)
    result, notes = build_result(Path(args.out_dir), Path(args.published), args.wall_s)
    print("RESULT " + json.dumps(result, separators=(",", ":")))
    for note in notes:
        print(note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
