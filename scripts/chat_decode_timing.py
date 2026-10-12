"""CHAT_UI_PLAN.md P8-F (part 1): where does a live turn's time go? A timing job. It finds the cause and fixes nothing.

    python scripts/chat_decode_timing.py --home <CHAT_HOME> --gallery <gallery dir> --job <id> [--engine real|tiny] [--threads 8] [--rows 0-4]
                                         [--local-dir <dir>]

The P1-D decision predicted about 4 s per turn on CPU, from job 2590184: 3.05 s per report for a cached beam-3 decode of 100 tokens on gx14
with 8 CPUs. The live server (job 2638789, the same node, the same 8 CPUs) takes 7.4 to 8.1 s for the same decode in its `generate` stage, a
gap of about 45 ms per step. This job decodes the same five test rows (0 to 4, the picker's own rows) seven ways, each adding one part of the
live path to the one before, so that the first arm that is slower says where the gap comes from:

  A        decoder.beam_search_cached(..., on_step=None) in the main thread: the P1-C path, decode only.
  B        A with a no-op on_step.
  C        engine.generate(enc, opts, no-op on_snapshot, cancel) in the main thread: B plus the engine's per-step text decode.
  D        C inside a ThreadPoolExecutor(max_workers=1) worker, as the pipeline runs it.
  E        the pipeline's whole turn without HTTP: a real Store on a fresh directory on the NFS home (<home>/timing/<job>/e), the real Worker,
           and the turn's events drained on an asyncio loop the way the SSE bridge drains them (app.server's own sse, _push and _frames).
  E-local  E with the store on node-local disk (--local-dir, ${TMPDIR:-/tmp} in the wrapper), so that E against E-local is the NFS.
  F        E through HTTP: app.server.create_app on the real engine, uvicorn on a free loopback port inside this process, turns sent with httpx
           and the stream read with scripts/chat_realturn_probe.py's own reader (read_into).
  A-end    A again, after F, on the same rows and options with no warm-up turn before it. The seven arms run in a fixed order, so a neighbour
           that came or went on the node while they ran would look like a difference between arms: A-end against A shows it. It is not one of
           the seven the threshold looks at; its ratio to A is on its line and, as `a_end_ratio`, in the summary.

Reading it: A to B is the decoder building its callback's argument at every step (the best beam's ids as a list); B to C the engine's own work
(the text decode of the best beam at every step, the repair of the finished report); C to D the worker thread; D to E the pipeline's work at
every step (the session check, the redaction, the SQLite write, the hand-over to the loop); E to E-local the NFS (if the environment line says
the local directory is on a local disk, and the home on nfs); E to F the HTTP and the server's own loop. F's client is a thread of this process,
where the live one is on the laptop, so F reads a little high by the client's own wake-ups. E, E-local and F report the turn's total_ms too, but
the job runs no labeller: the label stage and the CheXbert half of the score stage do not run, so that total is not the live turn's. If every
arm, A included, is slow, the code path is not the cause: the environment line says what the node gave this job, and A-end and the load readings
say whether that changed while it ran.

The engine is built as the server builds it (app.server._engines, which is what app/cli.py reaches through create_app) on the CPU, with the
published protocol (app.pipeline.PUBLISHED_PROTOCOL: beam 3, 100 tokens, no stop on repeat, no compile) and cached decode, the server's default.
The rows' images come from the gallery's test_meta (Gallery.test_study), which is the picker's source. One warm-up turn on row 0, which no arm
counts, comes before every arm. An arm that runs in a new context, a worker thread, a new store or a new server, starts with one more
uncounted turn on row 0, so that its rows are steady state, as the live server's are long after its first turn.

Sanity check. Every arm's report for a row must be the same text, A-end's too: they are compared in memory and only `same_report_all_arms` is
printed. The arms are measured with time.perf_counter. An arm that fails is named by its exception's class, or by a fixed code (`turn_deadline`
for a turn of E, E-local or F that has not ended after TURN_DEADLINE_S, so that a hung turn costs five minutes and not the job's hour; the load
drill's own code, such as `session_create_http_403`, for a session the server would not make), and the others go on; the job then exits 1.

Output. stdout carries only `RESULT {json}` lines, each under the 300 characters `chat_remote.sh summary` keeps: the environment (node, CPU
model, the size of the CPU affinity and how many physical cores it covers, the node's load average once the engines are loaded, the filesystem
of the local store's directory and of the NFS store's, each a word of the closed list FS_WORDS, torch's thread count in the main thread and in
a worker thread), one line per arm as it finishes (the rows' ms, their mean, the ratio to A's mean, and for E, E-local and F the whole turn's
ms and its mean), and the summary: the arm of A to F whose mean first exceeds 1.5 times A's mean, or `none`, `a_end_ratio`,
`same_report_all_arms`, and the node's load average before the engines were loaded and after the last arm. Exit status 0 when every arm ran and
every report was the same, 1 otherwise (a failed arm, a different report, a setup that could not finish: its RESULT line names the class), 2
for an argument it refuses, which is one `ERROR ...` line on stderr before anything is loaded. Everything else the process prints, a library's
included, goes to stderr, and an exception is named by its class only, in every place.

R7 (DUA). The job decodes MIMIC test images, and the reports, the study ids and the paths that hold them must not leave the cluster: none is
printed, logged or written. Reports are held in memory, compared and dropped. The mount table, read for the two filesystem words, names the
home's server and its export: only the word of the closed list is kept. The only files written are the stores of E and F under
<home>/timing/<job>, which stay in place (R8), and the node-local store of E-local, a directory this script made and removes. Never port 8000
(the dev server's) or 8010 (the controller's tunnel): the server of arm F binds port 0, which is an ephemeral port, and check_port refuses
both all the same, and the client's URL goes through the load drill's own check, which refuses 8000.
"""
import argparse
import asyncio
import contextlib
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, TextIO, Tuple

import httpx
import torch
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))      # app comes from this tree, however the venv's install points

from app.engine import Encoded, Engine, Generated  # noqa: E402
from app.gallery import Gallery  # noqa: E402
from app.pipeline import PUBLISHED_MODEL, PUBLISHED_PROTOCOL, Pipeline, TurnJob, Worker  # noqa: E402
from app.schemas import Options  # noqa: E402
from app.server import _chat_home, _engines, _frames, _push, create_app, sse  # noqa: E402
from app.store import Store  # noqa: E402
from scripts import chat_load_drill as drill  # noqa: E402
from scripts import chat_realturn_probe as probe  # noqa: E402

MODEL = PUBLISHED_MODEL                          # hybrid_150m_m3_rrg: the model the 2.5 times gap was seen on
ARMS = ("A", "B", "C", "D", "E", "E-local", "F")   # the ladder: each adds one part of the live path to the one before
A_END = "A-end"                                  # A again after the ladder: what drifted on the node while it ran; not one of the seven
ALL_ARMS = ARMS + (A_END,)
TURN_ARMS = ("E", "E-local", "F")                # the arms that run the whole turn: they also report its total
OVER = 1.5                                       # an arm is over when its mean is more than this many times A's
TURN_DEADLINE_S = 300.0                          # a live turn takes 8 s: one that is still going after five minutes fails its arm, not the job's hour
DEFAULT_ROWS = "0-4"
MAX_ROWS = 6                                     # five are measured; six, with the widest values below, keep every RESULT line under 300 characters
MAX_ROW = 9999                                   # the test split has 2,663 studies
DEFAULT_THREADS = 8
NODE_CHARS, NODE_MAX = "A-Za-z0-9._-", 16        # gx14, ga03, gx17v1: the cluster's nodes
CPU_CHARS, CPU_MAX = "A-Za-z0-9 ().@+_-", 48     # the longest Intel string of those nodes is 44 characters
FS_WORDS = ("nfs", "tmpfs", "xfs", "ext4", "lustre", "gpfs", "beegfs", "other")   # the closed list: all that is printed of a mount table
FS_KINDS = {"nfs": "nfs", "nfs4": "nfs", "tmpfs": "tmpfs", "xfs": "xfs", "ext4": "ext4", "lustre": "lustre", "gpfs": "gpfs", "beegfs": "beegfs"}
CODE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,59}")   # an arm's error as the wrapper's allowlist takes it: a class name or a fixed code
JOB_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")      # the job id goes into a directory name: only a plain one is used
ROWS_REFUSAL = "must be test rows such as 0-4 or 1,3,5: whole numbers from 0 to {}, none twice, at most {} of them".format(MAX_ROW, MAX_ROWS)


class Refused(Exception):
    """An argument the script will not run with. Its text is fixed: it never contains what was given."""


@dataclass
class ArmResult:
    """What one arm came to. The numbers are printed; `reports` are MIMIC-derived text, held for the sanity check and never printed."""
    arm: str
    ms: List[float] = field(default_factory=list)                # per row: the decode or generate ms
    total_ms: List[float] = field(default_factory=list)          # per row, E / E-local / F: the whole turn as the consumer or the client saw it
    reports: List[Optional[str]] = field(default_factory=list)   # per row, in memory only
    tokens: List[int] = field(default_factory=list)              # per row: how many tokens were decoded
    error: Optional[str] = None                                  # the class of the exception that ended the arm, or a fixed word


# ---- the arguments -------------------------------------------------------------------------------------------------------------

class _Parser(argparse.ArgumentParser):
    """argparse for a script that does not echo what it is given: a refusal is one fixed sentence (whatever argparse quoted is cut off), an
    argument it does not know is not named, and there is no abbreviation of an option."""

    def error(self, message: str) -> Any:
        raise Refused(re.sub(r"'.*$", "", message).rstrip(" :"))

    def parse_args(self, args: Optional[Sequence[str]] = None, namespace: Any = None) -> argparse.Namespace:
        parsed, extras = self.parse_known_args(args, namespace)
        if extras:
            raise Refused("unrecognized arguments")
        return parsed


def job_id(text: str) -> str:
    if JOB_ID.fullmatch(text) is None:
        raise argparse.ArgumentTypeError("must be 1 to 64 characters: letters, digits, dash and underscore")
    return text


def parse_rows(text: str) -> List[int]:
    """The probe's own parser of "0-4", "3" or "1,3,5", with this script's smaller limit and its own fixed sentence."""
    try:
        rows = probe.parse_rows(text)
    except argparse.ArgumentTypeError:
        raise argparse.ArgumentTypeError(ROWS_REFUSAL) from None
    if len(rows) > MAX_ROWS or max(rows) > MAX_ROW:
        raise argparse.ArgumentTypeError(ROWS_REFUSAL)
    return rows


def build_parser() -> argparse.ArgumentParser:
    p = _Parser(prog="chat_decode_timing.py", allow_abbrev=False, add_help=False,
                description="Decode the same test rows seven ways (A to F, each adding one part of the live path to the one before), then A "
                            "again (A-end), and print numbers only: RESULT lines for the environment, each arm and the summary. Exit 0 when "
                            "every arm ran and every report was the same, 1 when not, 2 for an argument it refuses.")
    p.add_argument("--home", metavar="DIR", help="CHAT_HOME: the stores of E and F are made under its timing/<job> (required)")
    p.add_argument("--gallery", metavar="DIR", help="a gallery build directory: the test rows come from it, as the picker's do (required)")
    p.add_argument("--job", type=job_id, default="local", metavar="ID", help="names <home>/timing/<job>, which must not exist (default local)")
    p.add_argument("--engine", choices=("real", "tiny"), default="real", help="real: the published model on the CPU; tiny: the laptop's stand-in")
    p.add_argument("--threads", type=drill._whole(1, 512), default=DEFAULT_THREADS, help="CPU threads of the real engine (default 8)")
    p.add_argument("--rows", type=parse_rows, default=DEFAULT_ROWS, metavar="ROWS",
                   help="the test rows, such as 0-4 or 1,3,5 (default 0-4; at most {})".format(MAX_ROWS))
    p.add_argument("--local-dir", metavar="DIR", help="where E-local's node-local store is made (default: the system's temporary directory)")
    p.add_argument("-h", "--help", action="store_true", help="show this help")
    return p


def check_arguments(args: argparse.Namespace) -> Dict[str, Optional[Path]]:
    """What is wrong with the arguments, found before anything is loaded or made. -> the home, the gallery, the local directory and the job's
    timing directory, as paths. Refused, with a fixed sentence, when one will not do."""
    if args.home is None:   # not `required=True`: --help needs no arguments
        raise Refused("--home is required")
    if args.gallery is None:
        raise Refused("--gallery is required")
    home = Path(args.home).expanduser()
    if not home.is_dir():
        raise Refused("the home directory does not exist")
    try:
        home = _chat_home(str(home))
    except RuntimeError:   # its text names the path
        raise Refused("the home directory must be outside this repository and any checkout of it") from None
    gallery = Path(args.gallery).expanduser()
    if not gallery.is_dir():
        raise Refused("the gallery directory does not exist")
    local = None
    if args.local_dir is not None:
        local = Path(args.local_dir).expanduser()
        if not local.is_dir():
            raise Refused("the local directory does not exist")
        try:   # E-local's store holds report text: not inside a checkout either
            _chat_home(str(local))
        except RuntimeError:
            raise Refused("the local directory must be outside this repository and any checkout of it") from None
    timing_dir = home / "timing" / args.job
    if timing_dir.exists():
        raise Refused("the timing directory of this job exists already: it is never reused")
    return {"home": home, "gallery": gallery, "local": local, "timing": timing_dir}


# ---- the pieces of a line ------------------------------------------------------------------------------------------------------

def plain(text: str, limit: int, keep: str = NODE_CHARS) -> str:
    """`text` with every character outside `keep` (a regular-expression character class body) turned into `_`, cut to `limit`: what is
    printed of the environment is made of plain words, whatever the node or /proc/cpuinfo says."""
    return re.sub("[^{}]".format(keep), "_", text)[:limit]


def cpu_model(path: str = "/proc/cpuinfo") -> Optional[str]:
    """The first `model name` of /proc/cpuinfo, made plain; None where there is none (an ARM node, a laptop)."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("model name"):
            name = line.partition(":")[2].strip()
            return plain(name, CPU_MAX, CPU_CHARS) if name else None
    return None


def raw_mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def mean_ms(result: ArmResult) -> Optional[float]:
    mean = raw_mean(result.ms)
    return None if mean is None else round(mean, 1)


def _rounded(values: Sequence[float]) -> List[float]:
    return [round(v, 1) for v in values]


def _ratio(mean: Optional[float], base: Optional[float]) -> Optional[float]:
    return None if mean is None or base is None or base <= 0 else round(mean / base, 2)


def arm_payload(result: ArmResult, base: Optional[float], same: bool) -> Dict[str, Any]:
    """One arm's line. `base` is A's mean. An arm that failed is its class and the rows it finished, and has no mean and no ratio."""
    if result.error is not None:
        return {"arm": result.arm, "error": result.error, "rows_done": len(result.ms), "ms": _rounded(result.ms)}
    mean = raw_mean(result.ms)
    payload: Dict[str, Any] = {"arm": result.arm, "ms": _rounded(result.ms), "mean": None if mean is None else round(mean, 1),
                               "ratio": _ratio(mean, base)}
    if result.total_ms:
        payload["total_ms"] = _rounded(result.total_ms)
        payload["mean_total_ms"] = round(raw_mean(result.total_ms), 1)
    payload["same_report"] = same
    return payload


def first_over(results: Dict[str, ArmResult], factor: float = OVER) -> str:
    """The first arm, in ladder order, whose mean is more than `factor` times A's; `none` when no arm is; `unknown` when A has no mean to
    measure against (it failed, or it never ran). An arm that failed has no mean and is passed over."""
    base_result = results.get("A")
    base = None if base_result is None or base_result.error is not None else raw_mean(base_result.ms)
    if base is None or base <= 0:
        return "unknown"
    for arm in ARMS[1:]:
        result = results.get(arm)
        if result is None or result.error is not None:
            continue
        mean = raw_mean(result.ms)
        if mean is not None and mean > factor * base:
            return arm
    return "none"


def same_reports(results: Dict[str, ArmResult]) -> Dict[str, bool]:
    """For each arm, whether its report of every row is A's, row by row: compared here, in memory, and never printed. An arm that failed, is
    short, or has a missing report is not the same; with no report of A's own to compare with, none is."""
    base = results.get("A")
    reference = None if base is None or base.error is not None else base.reports
    ok = bool(reference) and None not in reference
    return {arm: bool(ok and results[arm].error is None and len(results[arm].reports) == len(reference)
                      and all(mine is not None and mine == theirs for mine, theirs in zip(results[arm].reports, reference)))
            for arm in ALL_ARMS if arm in results}


def _load(values: Optional[Sequence[float]]) -> Optional[List[float]]:
    return None if values is None else [round(v, 2) for v in values]


def _fs_word(word: Optional[str]) -> Optional[str]:
    """A filesystem word as it is printed: one of FS_WORDS, or None. Whatever else it was given, it is `other`: no text of a mount table or a
    path can be in the environment line."""
    return None if word is None else word if word in FS_WORDS else "other"


def env_payload(node: str, cpu: Optional[str], affinity: Optional[int], cores: Optional[int], load: Optional[Sequence[float]], threads_main: int,
                threads_worker: int, engine: str, rows: Sequence[int], fs_local: Optional[str] = None,
                fs_home: Optional[str] = None) -> Dict[str, Any]:
    return {"env": {"node": plain(node, NODE_MAX), "cpu": None if cpu is None else plain(cpu, CPU_MAX, CPU_CHARS), "affinity": affinity,
                    "cores": cores, "load": _load(load), "fs_local": _fs_word(fs_local), "fs_home": _fs_word(fs_home),
                    "threads_main": threads_main, "threads_worker": threads_worker, "engine": engine, "rows": list(rows)}}


def a_end_ratio(results: Dict[str, ArmResult]) -> Optional[float]:
    """A-end's mean against the first A's, to two places: how far the node drifted while the ladder ran. None where either has no mean (it
    failed, or never ran)."""
    first, last = results.get("A"), results.get(A_END)
    if first is None or last is None or first.error is not None or last.error is not None:
        return None
    return _ratio(raw_mean(last.ms), raw_mean(first.ms))


def summary_payload(results: Dict[str, ArmResult], same_all: bool, load_before: Optional[Sequence[float]] = None,
                    load_after: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    """The last line: the arm of A to F that is first over 1.5 times A, A-end's ratio to A, the sanity check, A's token counts, the arms that
    failed, and the node's load average before the engines were loaded and after the last arm."""
    base = results.get("A")
    return {"summary": {"first_over_1_5x": first_over(results), "a_end_ratio": a_end_ratio(results), "same_report_all_arms": same_all,
                        "tokens": [] if base is None else list(base.tokens),
                        "arms_failed": [arm for arm in ALL_ARMS if arm in results and results[arm].error is not None],
                        "load_before": _load(load_before), "load_after": _load(load_after)}}


def fixed_code(text: str, fallback: str) -> str:
    """`text`, when it is a word of letters, digits and underscores (a class name, or a code such as the drill's session_create_http_403, which
    holds a status or a class and never a part of an answer), else `fallback`: what an arm's error line may carry is a word and no free text."""
    return text if CODE.fullmatch(text) else fallback


def result_line(payload: Dict[str, Any]) -> str:
    """The one line this script prints for a result. allow_nan=False: a number that is no JSON is an error here, never a printed line."""
    return "RESULT " + json.dumps(payload, separators=(",", ":"), allow_nan=False)


def affinity_size() -> Optional[int]:
    """How many CPUs this process may run on (the job's cpuset); None where the system has no such call (a Mac)."""
    getter = getattr(os, "sched_getaffinity", None)
    return None if getter is None else len(getter(0))


def core_count(cpus: Optional[Iterable[int]] = None, root: str = "/sys/devices/system/cpu") -> Optional[int]:
    """How many physical cores the logical CPUs `cpus` (by default this process's affinity) sit on, from sysfs: the distinct (package, core)
    pairs. Eight CPUs can be eight cores or four cores of two hardware threads each, which no thread count shows and a decode that is bound by
    compute feels. None where it cannot be told: no affinity call, no sysfs, a file that is not a number."""
    if cpus is None:
        getter = getattr(os, "sched_getaffinity", None)
        cpus = None if getter is None else getter(0)
    if not cpus:
        return None
    cores = set()
    try:
        for cpu in cpus:
            topology = Path(root) / "cpu{}".format(cpu) / "topology"
            cores.add((int((topology / "physical_package_id").read_text()), int((topology / "core_id").read_text())))
    except (OSError, ValueError):
        return None
    return len(cores)


def load_average() -> Optional[List[float]]:
    """The node's load average over 1, 5 and 15 minutes, to two places: what else is running on the node, which a job in its own cpuset does
    not see and which shares its caches and its memory bandwidth. None where there is none."""
    getter = getattr(os, "getloadavg", None)
    if getter is None:
        return None
    try:
        return [round(v, 2) for v in getter()]
    except OSError:
        return None


def fs_word(path: Any, table: str = "/proc/self/mountinfo") -> Optional[str]:
    """What the directory `path` sits on, as one word of FS_WORDS, from the kernel's mount table: the type of the longest mount point that holds
    it (a mount over another of the same place hides it: the later line), nfs4 as nfs, a type that is not in the list as `other`. E against
    E-local is the NFS only if E-local's store is on a local disk, and the line says whether it is. None where it cannot be told (no table: a
    Mac). The table names the home's server and its export, so nothing of it but the word is kept (R7)."""
    try:
        where = os.path.realpath(str(path))
        text = Path(table).read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return None
    best, kind = -1, None
    for line in text.splitlines():
        head, _, tail = line.partition(" - ")
        fields, rest = head.split(" "), tail.split(" ")
        if len(fields) < 5 or not rest[0]:
            continue
        mount = os.path.normpath(re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), fields[4]))   # a space is written \040
        if (where == mount or where.startswith(mount.rstrip("/") + "/")) and len(mount) >= best:
            best, kind = len(mount), rest[0]
    return None if kind is None else FS_KINDS.get(kind, "other")


def node_name() -> str:
    return platform.node() or "unknown"


# ---- the world: the engine as the server builds it, the picker's rows, the published protocol -----------------------------------

class SetupFailed(Exception):
    """The job could not be set up: `step` is engine, gallery or rows, `error` the class of what went wrong (never its text)."""

    def __init__(self, step: str, error: str):
        super().__init__(step)
        self.step, self.error = step, error


@dataclass
class World:
    """What every arm decodes with: one engine, its gallery, the rows and each row's image already preprocessed and encoded as an upload's is."""
    kind: str                       # real or tiny
    name: str                       # the engine's name in the server's registry: options.model
    engine: Engine
    gallery: Gallery
    rows: List[int]
    encoded: List[Encoded]
    opts: Options                   # the published protocol on the cached decode
    home: Path
    timing: Path                    # <home>/timing/<job>: the stores of E and F
    local: Optional[Path]           # where E-local's store is made
    threads: int


def encode_rows(engine: Engine, gallery: Gallery, rows: Sequence[int]) -> List[Encoded]:
    """Each row's test image, from where the picker's list keeps it (Gallery.test_study), through Engine.preprocess from its file bytes and
    Engine.encode: the path an upload takes, and the first two stages of a turn."""
    encoded = []
    for row in rows:
        _, prepared = engine.preprocess(Path(gallery.test_study(row)["image"]).read_bytes())
        encoded.append(engine.encode(prepared)[1])
    return encoded


def load_world(args: argparse.Namespace, paths: Dict[str, Optional[Path]]) -> World:
    """The engine on the CPU, built as the server builds its engines (app.server._engines: the published model from the repo root, the
    thread count given), the gallery opened against that engine's tower, and the rows. SetupFailed names the step that failed."""
    try:
        engines = _engines(args.engine, (MODEL,), 0.0, "", "cpu", args.threads, paths["home"])
        name, engine = next(iter(engines.items()))
    except Exception as exc:
        raise SetupFailed("engine", type(exc).__name__) from None
    try:
        gallery = Gallery.open(paths["gallery"], expect_tower_sha256=engine.tower_sha256())
    except Exception as exc:
        raise SetupFailed("gallery", type(exc).__name__) from None
    try:
        encoded = encode_rows(engine, gallery, args.rows)
    except Exception as exc:
        raise SetupFailed("rows", type(exc).__name__) from None
    return World(kind=args.engine, name=name, engine=engine, gallery=gallery, rows=list(args.rows), encoded=encoded,
                 opts=Options(cached_decode=True, **PUBLISHED_PROTOCOL), home=paths["home"], timing=paths["timing"], local=paths["local"],
                 threads=args.threads)


def local_base(world: World) -> Path:
    """The directory E-local makes its store in: the one it was given, else the system's temporary directory."""
    return world.local if world.local is not None else Path(tempfile.gettempdir())


def environment(world: World) -> Dict[str, Any]:
    """The environment line: the node, the CPU, the size of the CPU affinity and how many physical cores it covers, the node's load average
    (read once the engines are loaded), the filesystem of E-local's directory and of the one E's and F's stores are made under (one word of
    FS_WORDS each), and torch's thread count in the main thread and in a worker thread of the kind the pipeline decodes in."""
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="turn") as pool:
        threads_worker = pool.submit(torch.get_num_threads).result()
    return env_payload(node_name(), cpu_model(), affinity_size(), core_count(), load_average(), torch.get_num_threads(), threads_worker, world.kind,
                       world.rows, fs_word(local_base(world)), fs_word(world.timing))


# ---- arms A to D: the decode alone, and the decode through the engine, in the main thread and in a worker -----------------------

def _no_snapshot(step: int, text: str) -> None:
    """The engine's on_snapshot when nothing is to be sent: the pipeline's, minus the store, the redaction and the event."""


def _no_step(step: int, ids: List[int]) -> None:
    """The decoder's on_step when nothing is to be observed."""


def decode_ids(world: World, enc: Encoded, on_step: Optional[Callable[[int, List[int]], None]]) -> Tuple[float, List[int]]:
    """decoder.beam_search_cached as Engine.generate calls it and without the engine's callback, timed. -> (ms, the token ids)."""
    opts = world.opts
    empty = torch.zeros((1, 0), dtype=torch.long, device=world.engine.device)   # no BOS, as in training
    started = time.perf_counter()
    with torch.no_grad():
        out = world.engine.decoder.beam_search_cached(empty, prefix_embeds=enc.prefix, beam_size=opts.beam_size,
                                                      max_new_tokens=opts.max_new_tokens, on_step=on_step)
    ms = (time.perf_counter() - started) * 1000.0
    return ms, out[0].tolist()


def time_generate(world: World, enc: Encoded) -> Tuple[float, Generated]:
    """Engine.generate with a no-op snapshot and a cancel that is never set, timed. -> (ms, what it generated)."""
    started = time.perf_counter()
    _, generated = world.engine.generate(enc, world.opts, _no_snapshot, threading.Event())
    return (time.perf_counter() - started) * 1000.0, generated


def _note_generated(result: ArmResult, ms: float, generated: Generated) -> None:
    result.ms.append(ms)
    result.reports.append(generated.report)
    result.tokens.append(len(generated.token_ids))


def warm_up(world: World) -> None:
    """The one turn that no arm counts, before every arm: a whole generate on row 0 in the main thread."""
    time_generate(world, world.encoded[0])


def _decode_arm(world: World, result: ArmResult, on_step: Optional[Callable[[int, List[int]], None]]) -> None:
    for enc in world.encoded:
        ms, ids = decode_ids(world, enc, on_step)
        result.ms.append(ms)
        result.reports.append(world.engine._decode_text(ids))
        result.tokens.append(len(ids))


def arm_a(world: World, result: ArmResult) -> None:
    """A: the cached beam search with on_step=None, in the main thread. The P1-C path: decode only."""
    _decode_arm(world, result, None)


def arm_b(world: World, result: ArmResult) -> None:
    """B: A with an on_step that does nothing."""
    _decode_arm(world, result, _no_step)


def arm_c(world: World, result: ArmResult) -> None:
    """C: Engine.generate in the main thread: B plus the engine's text decode at every step."""
    for enc in world.encoded:
        _note_generated(result, *time_generate(world, enc))


def arm_d(world: World, result: ArmResult) -> None:
    """D: C in a ThreadPoolExecutor(max_workers=1) worker, as the pipeline runs it. A worker thread is new, so the first turn in it is one
    that is not counted."""
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="turn") as pool:
        pool.submit(time_generate, world, world.encoded[0]).result()
        for enc in world.encoded:
            _note_generated(result, *pool.submit(time_generate, world, enc).result())


# ---- the turns of E, E-local and F: what is read of a stream, and when a turn has not ended well -----------------------------------

class TurnFailed(Exception):
    """A turn that did not end well. `code` is a fixed word or a number built here (http_422, stream_model_error, final_aborted), never a text
    the server sent."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def turn_problem(facts: probe.Facts) -> Optional[str]:
    """What is wrong with a turn that was read down to its facts, as a code that holds no text; None when it went as it should: the POST
    answered 200, no exception on the client's side, no error event, a message_stop with status done, a seq with no gap, the generate stage's ms."""
    if facts.error is not None:
        return facts.error
    if facts.status is not None and facts.status != 200:
        return "http_{}".format(facts.status)
    if facts.stream_error is not None:
        return "stream_" + facts.stream_error
    if facts.final != "done":
        return "final_{}".format(facts.final)
    if facts.seq_contiguous is not True:
        return "seq_gap"
    if "generate" not in facts.stages_ms:
        return "no_generate_stage"
    return None


def final_report(chunks: Sequence[str]) -> Optional[str]:
    """The report of a turn's message_stop, from the frames of its stream: found with the probe's own frame splitter, held in memory for the
    sanity check, never printed."""
    report = None
    for event, text in probe.frames(iter(chunks)):
        if event == "message_stop":
            try:
                data = json.loads(text)
            except (ValueError, RecursionError):
                continue
            value = data.get("report") if isinstance(data, dict) else None
            report = value if isinstance(value, str) else None
    return report


def read_turn(chunks: Sequence[str], started: float, clock: Callable[[], float] = time.perf_counter) -> Tuple[probe.Facts, Optional[str]]:
    """A turn's frames, read by the probe's reader (read_into) down to numbers, booleans and fixed words, and the report apart: -> (the facts,
    the report). `started` and `clock` are the probe's: the milliseconds it takes from a stream that is read as it arrives."""
    facts, handles = probe.Facts(), probe.Handles()
    probe.read_into(facts, handles, iter(chunks), started, clock)
    return facts, final_report(chunks)


def count_turn(result: ArmResult, position: int, facts: probe.Facts, report: Optional[str], total_ms: float) -> None:
    """The turn at `position` of an arm's turns: the first (0) is the warm-up and is only checked, each after it is a row. TurnFailed when a
    turn did not end well. The row's ms is the generate stage's stage_end ms, as it was sent."""
    problem = turn_problem(facts)
    if problem is None and report is None:
        problem = "no_report"
    if problem is not None:
        raise TurnFailed(problem)
    if position:
        result.ms.append(facts.stages_ms["generate"])
        result.total_ms.append(total_ms)
        result.reports.append(report)
        result.tokens.append(facts.tokens or 0)


# ---- arms E and E-local: the pipeline's whole turn, a real store, the real worker, the events drained as the SSE bridge drains them ---

QUEUE_CAP = 4   # the server's own default (create_app's queue_cap)


def turn_job(world: World, store: Store, session_id: str, row: int) -> TurnJob:
    """The turn the picker starts for test row `row`, as POST /v1/sessions/<id>/messages starts it in private mode (app/server.py post_message):
    the turn is recorded in the store with the study's image sha256 and its row, and the job carries the sha256 and the options, not the
    image: the pipeline reads the study itself, where the dataset keeps it."""
    sha256 = hashlib.sha256(Path(world.gallery.test_study(row)["image"]).read_bytes()).hexdigest()
    opts = world.opts.model_copy(update={"model": world.name, "test_row": row})
    user_message_id, message_id = store.start_turn(session_id, "", "private", opts.model_dump(), sha256, None, row)
    return TurnJob(session_id, user_message_id, message_id, "", None, None, opts, mode="private", previous_sha256=sha256)


async def drain_turn(worker: Worker, job: TurnJob, deadline_s: Optional[float] = None) -> Tuple[List[str], float]:
    """Queue a turn on the worker and drain its frames the way the server's SSE bridge does, with its own functions: the worker's thread hands
    each event to this loop as a frame (sse, _push: loop.call_soon_threadsafe) and the loop reads them off a queue (_frames).
    -> (the frames, the ms from the submission to the end of the stream). A turn that has not ended after `deadline_s` (TURN_DEADLINE_S) is
    TurnFailed("turn_deadline"): the arm ends there, and the shutdown of its worker stops the turn at its next step."""
    limit = TURN_DEADLINE_S if deadline_s is None else deadline_s
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()
    if not worker.reserve():
        raise TurnFailed("queue_full")
    started = time.perf_counter()
    worker.submit(job, lambda event, data: _push(loop, queue, sse(event, data)), lambda: _push(loop, queue, None))

    async def drained() -> List[str]:
        return [frame async for frame in _frames(queue)]

    try:
        frames = await asyncio.wait_for(drained(), timeout=limit)
    except asyncio.TimeoutError:
        raise TurnFailed("turn_deadline") from None
    return frames, (time.perf_counter() - started) * 1000.0


async def _pipeline_turns(world: World, result: ArmResult, store: Store, worker: Worker) -> None:
    """One session, the warm-up turn on row 0 (which is not counted: it is the first turn on this thread and this store) and then each row."""
    session_id = store.create_session("private")["id"]
    for position, row in enumerate([world.rows[0]] + world.rows):
        started = time.perf_counter()
        frames, total_ms = await drain_turn(worker, turn_job(world, store, session_id, row))
        facts, report = read_turn(frames, started)
        count_turn(result, position, facts, report, total_ms)


def _pipeline_arm(world: World, result: ArmResult, home: Path) -> None:
    """The pipeline as the server has it, in a store at `home`, with no labeller: the engine, the gallery (retrieval runs, as it does live),
    a Store, and the one-thread Worker that runs the turns."""
    store = Store(home)
    try:
        worker = Worker(Pipeline({world.name: world.engine}, world.name, store, "private", gallery=world.gallery), QUEUE_CAP)
        try:
            asyncio.run(_pipeline_turns(world, result, store, worker))
        finally:
            worker.shutdown()
    finally:
        store.close()


def arm_e(world: World, result: ArmResult) -> None:
    """E: the pipeline's whole turn without HTTP, over a store on a fresh directory of the NFS home, which is never reused and stays (R8)."""
    home = world.timing / "e"
    home.mkdir(parents=True)
    _pipeline_arm(world, result, home)


def arm_e_local(world: World, result: ArmResult) -> None:
    """E-local: E with the store on node-local disk. The directory is made here (mkdtemp: mode 0700, and the reports of five studies go into
    it) and removed when the arm ends, whatever the arm came to: it is this job's own."""
    base = local_base(world)
    home = Path(tempfile.mkdtemp(prefix="chat_decode_timing_", dir=str(base)))
    try:
        _pipeline_arm(world, result, home)
    finally:
        shutil.rmtree(str(home), ignore_errors=True)


# ---- arm F: E through HTTP ---------------------------------------------------------------------------------------------------

NOT_THIS_JOBS_PORTS = (8000, 8010)   # the dev server's, and the controller's tunnel
START_WAIT_S, STOP_WAIT_S = 120.0, 60.0


class PortRefused(Exception):
    """The server of arm F is on a port this job never uses."""


class ServerNotStarted(Exception):
    """The server of arm F did not come up."""


def check_port(port: int) -> int:
    """`port`, unless it is 8000 or 8010. The server binds port 0, an ephemeral port, so it never is: this holds that to be so."""
    if port in NOT_THIS_JOBS_PORTS:
        raise PortRefused("port {}".format(port))
    return port


def http_turn(client: httpx.Client, session_id: str, options: Dict[str, Any], deadline_s: Optional[float] = None,
              clock: Callable[[], float] = time.perf_counter) -> Tuple[probe.Facts, Optional[str], float]:
    """One turn over HTTP, sent as the page sends it (a form of the note and the options; the test row is one of the options, as the picker
    sets it) and read as it arrives by the probe's read_into. -> (the facts, the report, the ms from the request to the end of the stream).
    The report is kept in memory for the sanity check only. A failure is the facts' `error`, the class of the exception. A stream that is still
    going `deadline_s` (TURN_DEADLINE_S) after the request is let go, by the probe's own `watched`, after the chunk in hand: the server pings
    every 15 s, so that is within 15 s of the deadline, and the facts' error is `turn_deadline`. A stream that says nothing at all ends with
    the client's own read timeout, 120 s, as a ReadTimeout."""
    limit = TURN_DEADLINE_S if deadline_s is None else deadline_s
    facts, handles = probe.Facts(), probe.Handles()
    kept: List[str] = []

    def teed(chunks: Any) -> Any:
        for chunk in chunks:
            kept.append(chunk)
            yield chunk

    started = clock()
    try:
        with client.stream("POST", "/v1/sessions/{}/messages".format(session_id),
                           data={"text": "", "options": json.dumps(options)}) as response:
            facts.status = response.status_code
            if response.status_code == 200:
                probe.read_into(facts, handles, teed(probe.watched(response.iter_text(), started + limit, clock)), started, clock)
            else:
                response.read()
    except probe.Timeout:
        facts.error = "turn_deadline"
    except Exception as exc:   # the class, never the text: an httpx error can name a url
        facts.error = type(exc).__name__
    total_ms = (clock() - started) * 1000.0
    return facts, final_report(kept), total_ms


def _http_turns(world: World, result: ArmResult, base: str) -> None:
    """One session on the server at `base`: the warm-up turn on row 0, which is not counted, and each row."""
    with httpx.Client(base_url=base, timeout=drill.CLIENT_TIMEOUT, trust_env=False, follow_redirects=False) as client:
        session_id = drill.create_session(client)
        for position, row in enumerate([world.rows[0]] + world.rows):
            facts, report, total_ms = http_turn(client, session_id, probe.turn_options(model=world.name, test_row=row))
            count_turn(result, position, facts, report, total_ms)


def arm_f(world: World, result: ArmResult) -> None:
    """F: E through HTTP. app.server.create_app on the engine kind this job runs (the real engine builds its own copy, as a server does), over a
    store on a fresh directory of the NFS home that stays (R8); uvicorn serves it from a thread of this process on a free loopback port, with
    its worker thread and its event loop as in `python -m app.server`; the turns go over that port with httpx, the stream read by the probe's
    reader. No labeller: the stages after generate skip, as in E. The server is stopped, and its worker and store released, whatever happened."""
    home = world.timing / "f"
    home.mkdir(parents=True)
    app = create_app(engine=world.kind, models=(MODEL,), home=str(home), gallery=world.gallery, device="cpu", threads=world.threads)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_config=None, log_level="warning", access_log=False))
    thread = threading.Thread(target=server.run, name="timing-server", daemon=True)
    try:
        thread.start()
        deadline = time.monotonic() + START_WAIT_S
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not server.started:
            raise ServerNotStarted()
        port = check_port(server.servers[0].sockets[0].getsockname()[1])
        _http_turns(world, result, drill.check_url("http://127.0.0.1:{}".format(port)))
    finally:
        server.should_exit = True
        if thread.is_alive():
            thread.join(STOP_WAIT_S)
        app.state.worker.shutdown()   # a server that stopped ran these in its lifespan: both are safe to repeat, and one that never started did not
        app.state.store.close()


ARM_FUNCTIONS: Dict[str, Callable[[World, ArmResult], None]] = {"A": arm_a, "B": arm_b, "C": arm_c, "D": arm_d, "E": arm_e,
                                                                "E-local": arm_e_local, "F": arm_f, A_END: arm_a}   # A-end is A again


def run_arm(name: str, world: World) -> ArmResult:
    """One arm, over the rows. An exception ends it and is named by its class (or, for a turn that did not end well, by its fixed code, and for
    a session the server would not make by the load drill's own code): its text could hold a report or a path. What the arm had finished
    before it is kept."""
    result = ArmResult(name)
    try:
        ARM_FUNCTIONS[name](world, result)
    except TurnFailed as exc:
        result.error = exc.code
    except drill.DrillError as exc:   # its text is a code of its own (session_create_http_403): kept when it is a word, never free text
        result.error = fixed_code(str(exc), "DrillError")
    except Exception as exc:
        result.error = type(exc).__name__
    return result


# ---- the job -------------------------------------------------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None, stdout: Optional[TextIO] = None, stderr: Optional[TextIO] = None) -> int:
    """Run the job; -> the exit status. `stdout` and `stderr` are where its two kinds of line go (the process's own by default)."""
    stdout = sys.stdout if stdout is None else stdout
    stderr = sys.stderr if stderr is None else stderr
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.help:
            drill._write(stdout, parser.format_help().rstrip("\n"))
            return 0
        paths = check_arguments(args)
    except Refused as exc:
        drill._write(stderr, "ERROR " + str(exc))
        return 2
    root = logging.getLogger()
    quiet = logging.NullHandler()   # the pipeline logs the traceback of a failed turn, and a traceback can hold report text and paths
    root.addHandler(quiet)
    try:
        with contextlib.redirect_stdout(stderr), contextlib.redirect_stderr(stderr):   # whatever a library prints is stderr's, never a RESULT line's
            try:
                return run_job(args, paths, stdout)
            except Exception as exc:   # the class, never the text
                drill._write(stdout, result_line({"fatal": type(exc).__name__, "step": "run"}))
                return 1
    finally:
        root.removeHandler(quiet)


def run_job(args: argparse.Namespace, paths: Dict[str, Optional[Path]], stdout: TextIO) -> int:
    """The job: the load average, set up (the engines are loaded), the environment, the warm-up, the seven arms and A-end (each printed as it
    finishes), the load average again, the summary. -> the exit status."""
    def say(payload: Dict[str, Any]) -> None:
        drill._write(stdout, result_line(payload))

    load_before = load_average()   # before the engines load: what else was on the node when this job began
    try:
        world = load_world(args, paths)
    except SetupFailed as exc:
        say({"fatal": exc.error, "step": exc.step})
        return 1
    try:
        paths["timing"].mkdir(parents=True)   # after the setup, so that a job that cannot begin leaves nothing; it must not exist (R8)
    except OSError as exc:
        say({"fatal": type(exc).__name__, "step": "timing"})
        return 1
    say(environment(world))
    try:
        warm_up(world)
    except Exception as exc:
        say({"fatal": type(exc).__name__, "step": "warmup"})
        return 1
    results: Dict[str, ArmResult] = {}
    for arm in ALL_ARMS:   # the seven, then A-end: its ratio is to the first A's mean
        results[arm] = run_arm(arm, world)
        base = results["A"]
        say(arm_payload(results[arm], None if base.error is not None else raw_mean(base.ms), same_reports(results).get(arm, False)))
    load_after = load_average()   # after the last decode: with the first reading and the one in the environment line, what the node did meanwhile
    same = same_reports(results)
    same_all = bool(same) and all(same.values())
    say(summary_payload(results, same_all, load_before, load_after))
    return 0 if same_all and all(r.error is None for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
