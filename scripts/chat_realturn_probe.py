"""CHAT_UI_PLAN.md P7-D: the real-turn gate, run from the laptop against the chat server through the SSH tunnel. It prints numbers only.

    LOCAL_PORT=8010 bash app/tunnel/tunnel.sh                  # another terminal: the tunnel, on a port that is not 8000
    CHAT_TOKEN=... venv/bin/python scripts/chat_realturn_probe.py http://localhost:8010
    venv/bin/python scripts/chat_realturn_probe.py http://localhost:8010 --token-file ~/chat.token --rows 0-4

The plan has a person open the page, pick test rows 0 to 4 and compare the reports by eye. A private server streams MIMIC-derived text (the
report, the matched reports, the sources of the labels), and under rule R7 no agent may read any of it, so the numeric half of that gate is
measured here, and the visual half stays with the user. The probe runs three passes against a private server, each in a session of its own,
one turn after another, on test-split rows picked by number (--rows, default 0-4; the server's `test_row` option, as the page's picker sets it):

  1  The gate: the published model, named (hybrid_150m_m3_rrg), under the published protocol (beam 3, 100 tokens, stop_on_repeat off). Per row:
     first-byte ms (the request's start to the stream's first byte) and total ms, each stage's ms, live_equals_published (the server's own
     comparison with the published line), the stop reason and the tokens, how many neighbours and matching reports came (4 and 3 were asked),
     the own-report rank, how many of the 14 labels agree with each neighbour's, and the status and byte length of GET
     /v1/messages/<id>/image?variant=thumb and variant=model_input and of GET /v1/gallery/images/<row> for each neighbour.
  2  The stop measurement: the same rows with stop_on_repeat on, at 100 and at 200 tokens. Per row and budget: whether the stop fired, the tokens
     and the ms, and whether the report holds a complete sentence (worked out here, in memory); at 100 tokens also what it saved against pass 1.
  3  The EOS model (hybrid_150m_m3_rrg_eos) on the same rows at --eos-tokens (default 200, the budget its registry note was measured at), and on
     one picture drawn here. Per turn: the stop reason (`eos` is expected), the tokens and the words, whether the last sentence is terminated,
     the neighbour and matching-report counts, and the four timings pass 1 records (first byte, total, the server's own, each stage's ms); the
     pass's line has the slowest first byte, turn and generate stage of its turns, the rows and the picture alike (the default model's turn time).

Output. Stdout is one `RESULT {json}` line per pass (each says whether the pass `ran`: every turn ended done and its session was deleted; that is not a
verdict on what it measured) and then `RESULT {"gate": {...}}`: published_equal (every pass 1 row), eos_stops (every EOS turn
stopped by `eos`), features_ok (every pass 1 row has its counts as asked, its own-report rank, one label agreement per neighbour and every image and
gallery request answered 200 with a body, and every EOS turn its counts), first_byte_ms_max and turn_ms_max (the slowest of pass 1, for the reader to
set against the targets), eos_turn_ms_max (the slowest EOS turn, a number that is not judged), sessions_deleted and ok. Exit status 0 when the gate
holds, which also wants every turn of all three passes to have ended in a message_stop with status done and no gap in its seq, no session left
behind and nothing else gone wrong (`error` names it); 1 when it does not hold; 2 for an argument it refuses, which is one `ERROR ...` line on
stderr, before anything is sent. A test row number is printed as it was given: it is this project's row number, not a MIMIC id. On a CPU server a
turn takes about 4 s, and the default rows are 5 + 10 + 6 turns: 2 to 3 minutes.

R7 (DUA). The stream carries MIMIC-derived text. This script reads every frame in memory, keeps of it only stage names it knows, milliseconds, skip
reasons from a fixed list, counts (neighbours, matches, label agreement n/14, the own-report rank), `live_equals_published`, the stop reason, the status
word of message_stop, token and word counts and HTTP status codes, and drops the rest as it reads it. A number is kept only as a plain int (a time
may be a float) from 0 to below a trillion: a bool, a NaN, an infinity or an integer of 400 digits is dropped as a string is. Images are fetched
for their status and byte length only and dropped as they are counted. Nothing of a frame, or of any answer, is printed, logged or written to a file,
a failure included: an exception is named by its class, never by its text. The only writes are the RESULT lines, the refusals and the help text,
through the drill's one function, and what they hold is numbers, booleans and fixed words.

Transport. As scripts/chat_load_drill.py (P8-C), whose helpers it imports: httpx only, one client that goes straight to the URL it was given (no proxy
named in the environment is used, no redirect is followed). The token comes from the environment variable CHAT_TOKEN or from --token-file (a regular
file of mode 0600 that you own), never from an argument, is sent as a bearer to that URL and nowhere else, is never printed, and is not sent over plain
http to a host that is not loopback. It sends one fixed X-Client-Id, `chat-realturn-probe`. Port 8000 is the local dev server's: a URL with that port is
refused, and the tunnel's own port is whatever LOCAL_PORT it was started with (the controller's is 8010). One session is made per pass and deleted when
the pass ends, also when something went wrong or Ctrl-C was pressed: the server hides a deleted session at once and removes its uploaded pictures.
"""
import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, TextIO, Tuple

import httpx

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:   # `python scripts/chat_realturn_probe.py` puts scripts/ on the path, not the repo's root
    sys.path.insert(0, REPO_ROOT)
from scripts import chat_load_drill as drill  # noqa: E402

CLIENT_ID = "chat-realturn-probe"    # one fixed id, on every request: the server's pattern is visible ASCII, 1 to 128 characters

# What the app holds, restated: the probe imports nothing of the app (it starts with the server's code absent), and a test holds each equal.
PUBLISHED_MODEL = "hybrid_150m_m3_rrg"
EOS_MODEL = "hybrid_150m_m3_rrg_eos"
PUBLISHED_PROTOCOL = {"decode": "beam", "beam_size": 3, "max_new_tokens": 100, "stop_on_repeat": False, "compile": False}
STAGES = ("preprocess", "encode", "retrieve", "generate", "label", "score")
SKIP_REASONS = ("gallery_unavailable", "k_zero", "label_off", "labeler_unavailable", "no_reference")   # app/pipeline.py's, as written there
STOPPED = ("budget", "repeat", "eos")                                                                   # generate.detail.stopped
ERROR_KINDS = ("validation_error", "model_error", "overloaded_error", "server_restart", "store_failed")  # error.error.type, the kinds the app sends
REFUSAL_KINDS = ("invalid_request_error", "authentication_error", "permission_error", "not_found_error", "validation_error",
                 "overloaded_error", "internal_error", "unavailable_error", "rate_limited")             # error.type of a refusal: app/server.py ERROR_KINDS
N_LABELS = 14                        # CheXbert-14: the `of` of a label agreement
PASS_NAMES = {1: "gate", 2: "stop", 3: "eos"}   # the three passes, in the order they run

K_IMAGES, K_REPORTS = 4, 3           # the page's retrieval defaults, sent as numbers: "the counts as asked" is what the probe asked
STOP_BUDGETS = (100, 200)            # pass 2: the token budgets the stop is measured at
DEFAULT_ROWS, MAX_ROWS, MAX_ROW = "0-4", 32, 9999999
DEFAULT_EOS_TOKENS = 200             # the budget the EOS model's registry note was measured at
DEFAULT_TIMEOUT_S = 1800.0
ROWS = re.compile(r"([0-9]{1,7})(?:-([0-9]{1,7}))?")
ROWS_REFUSAL = "must be test rows such as 0-4 or 1,3,5: whole numbers from 0, none twice, at most {} of them".format(MAX_ROWS)
ID = re.compile(r"[A-Za-z0-9_-]{1,64}")   # a message id goes into a path: only a plain one is used
ROW_LIMIT = 2 ** 31                       # a gallery row goes into a path as a number: only a whole number below this is used
NUMBER_LIMIT = 10 ** 12                   # a count or a time that a frame holds is kept only below this (a trillion ms is 31 years); more is dropped

# The sentence rule of scripts/repair_generations.py (the app's own: app/engine.py reads it through split_sentences), restated so that the
# probe stays free of the app and the model stack; a test holds each of the three equal to the original.
ABBREVIATIONS = frozenset(["dr", "drs", "mr", "mrs", "ms", "prof", "vs", "etc", "e.g", "i.e", "a.m", "p.m", "approx", "fig", "cf", "st", "jr", "sr",
                           "inc", "ltd", "al", "am", "pm"])
TERMINALS = ".?!"
TRAILING = ".?!\"')]}:;,"


# ---- the arguments -------------------------------------------------------------------------------------------------------------

def parse_rows(typed: str) -> List[int]:
    """Test rows from "0-4", "3", "0,2,5" or "5,1-3,9": whole numbers, ranges up and lists of them, in the order given, none twice and at
    most MAX_ROWS. ArgumentTypeError, with a fixed sentence that never holds what was given, when it will not do."""
    rows: List[int] = []
    for part in typed.split(","):
        found = ROWS.fullmatch(part.strip())
        if found is None:
            raise argparse.ArgumentTypeError(ROWS_REFUSAL)
        low = int(found.group(1))
        high = low if found.group(2) is None else int(found.group(2))
        if high < low or high > MAX_ROW or len(rows) + high - low + 1 > MAX_ROWS:   # judged before the range is expanded: 0-9999999 is no list
            raise argparse.ArgumentTypeError(ROWS_REFUSAL)
        rows.extend(range(low, high + 1))
    if len(set(rows)) != len(rows):
        raise argparse.ArgumentTypeError(ROWS_REFUSAL)
    return rows


def build_parser() -> argparse.ArgumentParser:
    p = drill._Parser(prog="chat_realturn_probe.py", allow_abbrev=False, add_help=False,
                      description="Run real test-split turns on a private chat server and print numbers only: the gate (the published model), "
                                  "the stop measurement and the EOS model. One RESULT line per pass, then RESULT {\"gate\": ...}; exit 0 "
                                  "when the gate holds, 1 when it does not, 2 for an argument it refuses.")
    p.add_argument("url", metavar="URL", nargs="?", help="the server, through the tunnel: http://localhost:PORT (not port 8000)")
    p.add_argument("--token-file", metavar="PATH", help="the token: a file of mode 0600 that you own; or set CHAT_TOKEN. Never an argument")
    p.add_argument("--rows", type=parse_rows, default=DEFAULT_ROWS, metavar="ROWS",
                   help="the test-split rows to run, such as 0-4 or 1,3,5 (default 0-4; at most {})".format(MAX_ROWS))
    p.add_argument("--eos-tokens", type=drill._whole(drill.MIN_NEW_TOKENS, drill.MAX_NEW_TOKENS), default=DEFAULT_EOS_TOKENS,
                   help="the EOS model's token budget, 16 to 200 (default 200, where its note was measured)")
    p.add_argument("--timeout", type=drill._seconds, default=DEFAULT_TIMEOUT_S, metavar="SECONDS",
                   help="how long the whole probe may take (default 1800)")
    p.add_argument("-h", "--help", action="store_true", help="show this help")
    return p


# ---- the options each pass sends -----------------------------------------------------------------------------------------------

def turn_options(**changes: Any) -> Dict[str, Any]:
    """The published protocol and the page's retrieval counts, with `changes` on top."""
    return dict(PUBLISHED_PROTOCOL, k_images=K_IMAGES, k_reports=K_REPORTS, **changes)


def gate_options(row: int) -> Dict[str, Any]:
    """Pass 1: the published model, named, under the published protocol, on a test-split row."""
    return turn_options(model=PUBLISHED_MODEL, test_row=row)


def stop_options(row: int, budget: int) -> Dict[str, Any]:
    """Pass 2: pass 1's turn with the stop on, at `budget` tokens."""
    return turn_options(model=PUBLISHED_MODEL, test_row=row, stop_on_repeat=True, max_new_tokens=budget)


def eos_options(budget: int, row: Optional[int] = None) -> Dict[str, Any]:
    """Pass 3: the EOS model, at `budget` tokens, on a test-split row, or with no row for the picture the probe draws."""
    options = turn_options(model=EOS_MODEL, max_new_tokens=budget)
    if row is not None:
        options["test_row"] = row
    return options


# ---- one turn: what is kept of its stream --------------------------------------------------------------------------------------

@dataclass
class Facts:
    """What one turn came to: numbers, booleans and fixed words only, so that all of it may be printed (R7). Nothing of a frame's text has a
    field here. The words are: `final` (message_stop's status), `stopped` (the stop reason), a skip reason, `stream_error` (an error event's
    kind) and `refusal`, the `error.type` of a refused turn; and `error`, the class of an exception that ended the turn, or "timeout"."""
    status: Optional[int] = None                 # the HTTP status of the POST, None when no answer came
    refusal: Optional[str] = None                # error.type of a status other than 200: one of REFUSAL_KINDS, else unknown
    error: Optional[str] = None
    final: Optional[str] = None                  # done, error, aborted or unknown; None when the stream had no message_stop
    stream_error: Optional[str] = None
    seq_contiguous: Optional[bool] = None        # every frame's seq is 1, 2, 3 ... with no gap; None when there was no frame
    first_byte_ms: Optional[float] = None        # request start to the first byte of the stream
    total_ms: Optional[float] = None             # request start to its last byte
    server_ms: Optional[float] = None            # message_stop.total_ms
    stages_ms: Dict[str, float] = field(default_factory=dict)
    skipped: Dict[str, str] = field(default_factory=dict)
    eos_trained: Optional[bool] = None
    stopped: Optional[str] = None
    tokens: Optional[int] = None
    neighbors: Optional[int] = None
    matches: Optional[int] = None
    own_rank: Optional[int] = None
    own_rank_dedup: Optional[int] = None
    own_of: Optional[int] = None
    agreement: List[int] = field(default_factory=list)   # per image neighbour: how many of the 14 labels agree with the report's
    agreement_pending: Optional[bool] = None
    live_equals_published: Optional[bool] = None
    words: Optional[int] = None
    complete_sentence: Optional[bool] = None
    last_sentence_terminated: Optional[bool] = None


@dataclass
class Handles:
    """What the requests that follow a turn need of it: the id of its image's message and the gallery rows of its image neighbours. They
    are opaque ids and row numbers, never printed: nothing builds a line from them."""
    message_id: Optional[str] = None
    neighbor_rows: List[int] = field(default_factory=list)


def is_sentence_end(token: str) -> bool:
    """Does this whitespace-delimited token end a sentence? (scripts/repair_generations.py is_sentence_end.)"""
    if not token or token[-1] not in TERMINALS:
        return False
    if token[-1] in "?!":
        return True
    word = token.rstrip(TRAILING)
    if not word:
        return False                      # a bare "." or "..." on its own
    if word.isdigit():
        return False                      # a list marker: "1."
    if len(word) == 1 and word.isalpha():
        return False                      # an initial: "J."
    return word.lower() not in ABBREVIATIONS


def sentence_facts(report: str) -> Tuple[int, bool, bool]:
    """-> (the words of a report, whether it holds a complete sentence, whether its last sentence is terminated), computed here and dropped."""
    tokens = report.split()
    return len(tokens), any(is_sentence_end(t) for t in tokens), bool(tokens) and is_sentence_end(tokens[-1])


def frames(chunks: Iterable[str]) -> Iterator[Tuple[str, str]]:
    """The frames of an SSE stream as (event name, data text); a keep-alive comment and a frame without data are none."""
    buffer = ""
    for chunk in chunks:
        buffer = (buffer + chunk).replace("\r\n", "\n")
        while "\n\n" in buffer:
            frame, buffer = buffer.split("\n\n", 1)
            event, data = "message", []
            for line in frame.split("\n"):
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data.append(line[5:][1:] if line[5:].startswith(" ") else line[5:])
            if data:
                yield event, "\n".join(data)


def _count(value: Any) -> Optional[int]:
    """The value when it is a whole number from 0 to below NUMBER_LIMIT (a bool is none), else None."""
    return value if type(value) is int and 0 <= value < NUMBER_LIMIT else None


def _flag(value: Any) -> Optional[bool]:
    return value if type(value) is bool else None


def _ms(value: Any) -> Optional[float]:
    """The value as milliseconds to a tenth, when it is a plain int or float from 0 to below NUMBER_LIMIT, else None. JSON allows an integer of
    any length, so the bound is the first thing asked: a huge int is compared and dropped, never converted to a float (math.isfinite of one
    raises OverflowError), and NaN and an infinity fail it as well."""
    if type(value) in (int, float) and 0 <= value < NUMBER_LIMIT:
        return round(float(value), 1)
    return None


def _word(value: Any, fixed: Sequence[str], other: str = "unknown") -> str:
    """The value when it is one of the fixed words, else `other`: what the server sent is never kept, only a word chosen here."""
    return value if isinstance(value, str) and value in fixed else other


def _since(started: float, clock: Callable[[], float]) -> float:
    return round((clock() - started) * 1000.0, 1)


def _retrieve(facts: Facts, handles: Handles, detail: Dict[str, Any]) -> None:
    neighbors = detail.get("image_neighbors")
    if isinstance(neighbors, list):
        facts.neighbors = len(neighbors)
        for neighbor in neighbors:
            row = neighbor.get("gallery_row") if isinstance(neighbor, dict) else None
            if type(row) is int and 0 <= row < ROW_LIMIT:
                handles.neighbor_rows.append(row)
    reports = detail.get("report_matches")
    if isinstance(reports, list):
        facts.matches = len(reports)
    own = detail.get("true_report_rank")
    if isinstance(own, dict):
        facts.own_rank, facts.own_rank_dedup, facts.own_of = _count(own.get("rank")), _count(own.get("rank_dedup")), _count(own.get("of"))


def _label(facts: Facts, detail: Dict[str, Any]) -> None:
    agreement = detail.get("neighbor_agreement")
    for entry in agreement if isinstance(agreement, list) else []:
        agree, of = (entry.get("agree"), entry.get("of")) if isinstance(entry, dict) else (None, None)
        if type(agree) is int and type(of) is int and of == N_LABELS and 0 <= agree <= N_LABELS:
            facts.agreement.append(agree)
    facts.agreement_pending = detail.get("neighbor_agreement_pending") is True


def _stage_end(facts: Facts, handles: Handles, data: Dict[str, Any]) -> None:
    stage = data.get("stage")
    if not isinstance(stage, str) or stage not in STAGES:   # a stage this probe does not know is not read
        return
    if "skipped" in data:
        facts.skipped[stage] = _word(data["skipped"], SKIP_REASONS, "other")
        return
    ms = _ms(data.get("ms"))
    if ms is not None:
        facts.stages_ms[stage] = ms
    detail = data.get("detail")
    if not isinstance(detail, dict):
        return
    if stage == "retrieve":
        _retrieve(facts, handles, detail)
    elif stage == "generate":
        facts.tokens = _count(detail.get("tokens"))
        if "stopped" in detail:
            facts.stopped = _word(detail["stopped"], STOPPED)
    elif stage == "label":
        _label(facts, detail)
    elif stage == "score":
        published = detail.get("published")
        if isinstance(published, dict):
            facts.live_equals_published = _flag(published.get("live_equals_published"))


def _note(facts: Facts, handles: Handles, event: str, data: Dict[str, Any]) -> None:
    """One frame's data, read down to its facts. Whatever else it holds is dropped when this returns."""
    if event == "message_start":
        for key in ("user_message_id", "message_id"):   # either names the turn's image
            value = data.get(key)
            if isinstance(value, str) and ID.fullmatch(value):
                handles.message_id = value
                break
        model = data.get("model")
        if isinstance(model, dict):
            facts.eos_trained = _flag(model.get("eos_trained"))
    elif event == "stage_end":
        _stage_end(facts, handles, data)
    elif event == "error":
        error = data.get("error")
        facts.stream_error = _word(error.get("type") if isinstance(error, dict) else None, ERROR_KINDS)
    elif event == "message_stop":
        facts.final = _word(data.get("status"), drill.STATUSES)
        facts.server_ms = _ms(data.get("total_ms"))
        report = data.get("report")
        if isinstance(report, str):
            facts.words, facts.complete_sentence, facts.last_sentence_terminated = sentence_facts(report)


def read_into(facts: Facts, handles: Handles, chunks: Iterable[str], started: float,
              clock: Callable[[], float] = time.perf_counter) -> None:
    """Read a turn's stream into `facts` and `handles`, frame by frame, as it arrives; `started` is the clock's reading when the request was
    sent. The one place a frame is looked at: every other function of this script sees the facts."""
    seqs: List[Optional[int]] = []

    def timed() -> Iterator[str]:
        for chunk in chunks:
            if chunk and facts.first_byte_ms is None:
                facts.first_byte_ms = _since(started, clock)
            yield chunk

    for event, text in frames(timed()):
        try:
            data = json.loads(text)
        except (ValueError, RecursionError):
            data = None
        seqs.append(_count(data.get("seq")) if isinstance(data, dict) else None)
        if isinstance(data, dict):
            _note(facts, handles, event, data)
    facts.total_ms = _since(started, clock)
    facts.seq_contiguous = seqs == list(range(1, len(seqs) + 1)) if seqs else None


# ---- the requests: a turn, and the image requests that follow it ---------------------------------------------------------------

class Timeout(Exception):
    """The deadline passed while a stream was still open."""


@dataclass
class Fetched:
    """One image request: its status and the length of its body (counted as it arrives and dropped), or the class of what failed."""
    status: Optional[int] = None
    size: Optional[int] = None
    error: Optional[str] = None


@dataclass
class Features:
    """The image requests that follow a turn: its picture as the page shows it (thumbnail, model input) and one gallery image per neighbour."""
    thumb: Fetched
    model_input: Fetched
    gallery: List[Fetched]


@dataclass
class Spec:
    """One turn to run: the test row (None for the picture the probe draws), the token budget it is given, its options and its picture."""
    row: Optional[int]
    budget: Optional[int]
    options: Dict[str, Any]
    png: Optional[bytes] = None


@dataclass
class Turn:
    row: Optional[int]
    facts: Facts
    features: Optional[Features] = None   # pass 1: the image requests that followed the turn
    budget: Optional[int] = None


@dataclass
class PassRun:
    number: int
    name: str
    turns: List[Turn] = field(default_factory=list)
    expected: int = 0                      # how many turns the pass is to run
    session_deleted: Optional[bool] = None
    error: Optional[str] = None            # the pass ended early: a code, a class name, "timeout" or "interrupted"


def watched(chunks: Iterable[str], deadline: float, clock: Callable[[], float] = time.perf_counter) -> Iterator[str]:
    """`chunks`, until the deadline passes: then Timeout, after the chunk in hand. The server pings every 15 s, so a stream that says nothing
    is let go within that."""
    for chunk in chunks:
        yield chunk
        if clock() >= deadline:
            raise Timeout


def run_turn(client: httpx.Client, session_id: str, spec: Spec, deadline: float,
             clock: Callable[[], float] = time.perf_counter) -> Tuple[Facts, Handles]:
    """One turn, on the session: its stream is read down to its facts as it arrives. It never raises (an interrupt apart): what went wrong is
    the facts' `error`, a class name, or `refusal`, a fixed word. `deadline` is the clock's reading after which the stream is let go."""
    facts, handles = Facts(), Handles()
    files = None if spec.png is None else {"image": ("probe.png", spec.png, "image/png")}
    started = clock()
    try:
        with client.stream("POST", "/v1/sessions/{}/messages".format(session_id), files=files,
                           data={"text": "", "options": json.dumps(spec.options)}) as response:
            facts.status = response.status_code
            if response.status_code == 200:
                read_into(facts, handles, watched(response.iter_text(), deadline, clock), started, clock)
            else:
                response.read()
                facts.refusal = _word(drill.error_type(drill._json_or_none(response)), REFUSAL_KINDS)
    except Timeout:
        facts.error = "timeout"
    except Exception as exc:   # the class, never the text: an httpx error can name a url, and nothing here may name a frame
        facts.error = type(exc).__name__
    return facts, handles


def fetch_size(client: httpx.Client, path: str) -> Fetched:
    """GET a path for its status and the length of its body, which is counted as it arrives and dropped: no image is kept or written."""
    try:
        with client.stream("GET", path) as response:
            return Fetched(response.status_code, sum(len(chunk) for chunk in response.iter_bytes()))
    except Exception as exc:
        return Fetched(error=type(exc).__name__)


def fetch_features(client: httpx.Client, handles: Handles) -> Features:
    """The image requests the page makes for a turn: its thumbnail and its model input (by the id of the turn's message) and the gallery image of
    each of its neighbours (by the neighbour's row). The id and the rows were checked when they were read: a plain id, whole numbers."""
    def variant(name: str) -> Fetched:
        if handles.message_id is None:
            return Fetched(error="no_id")
        return fetch_size(client, "/v1/messages/{}/image?variant={}".format(handles.message_id, name))

    return Features(variant("thumb"), variant("model_input"),
                    [fetch_size(client, "/v1/gallery/images/{}".format(row)) for row in handles.neighbor_rows])


def pass_specs(number: int, rows: Sequence[int], eos_tokens: int) -> List[Spec]:
    """The turns of a pass, in order. Pass 1: each row on the published model. Pass 2: each row at each stop budget. Pass 3: each row on the
    EOS model, then the picture the probe draws."""
    if number == 1:
        return [Spec(row, None, gate_options(row)) for row in rows]
    if number == 2:
        return [Spec(row, budget, stop_options(row, budget)) for row in rows for budget in STOP_BUDGETS]
    return ([Spec(row, eos_tokens, eos_options(eos_tokens, row)) for row in rows]
            + [Spec(None, eos_tokens, eos_options(eos_tokens), drill.synthetic_png())])


def run_pass(client: httpx.Client, run: PassRun, args: argparse.Namespace, deadline: float,
             clock: Callable[[], float] = time.perf_counter) -> None:
    """Fill `run`: one session, the pass's turns one after another, the session deleted whatever happens (a failure, a timeout, Ctrl-C). A turn
    is filed before the requests that follow it are made, so that an interrupt leaves what is known."""
    specs = pass_specs(run.number, args.rows, args.eos_tokens)
    run.expected = len(specs)
    session_id: Optional[str] = None
    try:
        session_id = drill.create_session(client)
        for spec in specs:
            if clock() >= deadline:
                run.error = "timeout"
                break
            facts, handles = run_turn(client, session_id, spec, deadline, clock)
            turn = Turn(spec.row, facts, None, spec.budget)
            run.turns.append(turn)
            if facts.error == "timeout":
                run.error = "timeout"
                break
            if run.number == 1 and facts.status == 200:
                turn.features = fetch_features(client, handles)
    except drill.DrillError as exc:
        run.error = str(exc)
    except KeyboardInterrupt:
        run.error = "interrupted"
    finally:
        if session_id is not None:
            run.session_deleted = drill.delete_session(client, session_id)


# ---- the judging ---------------------------------------------------------------------------------------------------------------

def turn_ok(f: Facts) -> bool:
    """The turn was answered, its stream ran to a message_stop with status done, with no gap in its seq, and nothing failed."""
    return f.status == 200 and f.error is None and f.final == "done" and f.seq_contiguous is True


def counts_ok(f: Facts) -> bool:
    """As many neighbours and matching reports as the probe asked for."""
    return f.neighbors == K_IMAGES and f.matches == K_REPORTS


def _served(fetched: Fetched) -> bool:
    return fetched.status == 200 and (fetched.size or 0) > 0


def row_ok(turn: Turn) -> bool:
    """A pass 1 row has all the features the page shows for it: the counts as asked, the own-report rank, one label agreement per neighbour,
    and its thumbnail, its model input and each neighbour's gallery image served (200, a body)."""
    f, features = turn.facts, turn.features
    return (turn_ok(f) and counts_ok(f) and f.own_rank is not None and f.own_of is not None and 1 <= f.own_rank <= f.own_of
            and len(f.agreement) == f.neighbors and features is not None and _served(features.thumb) and _served(features.model_input)
            and len(features.gallery) == f.neighbors and all(_served(x) for x in features.gallery))


def _all_turns(run: Optional[PassRun]) -> bool:
    return run is not None and run.expected > 0 and len(run.turns) == run.expected


def _ran(run: Optional[PassRun]) -> bool:
    """The pass ran every turn it was to run, none failed, and it did not end early."""
    return run is not None and run.error is None and _all_turns(run) and all(turn_ok(t.facts) for t in run.turns)


def _published_equal(run: Optional[PassRun]) -> bool:
    return run is not None and _all_turns(run) and all(t.facts.live_equals_published is True for t in run.turns)


def _features_ok(run: Optional[PassRun]) -> bool:
    return run is not None and _all_turns(run) and all(row_ok(t) for t in run.turns)


def _eos_stops(run: Optional[PassRun]) -> bool:
    return run is not None and _all_turns(run) and all(t.facts.stopped == "eos" for t in run.turns)


def _eos_counts_ok(run: Optional[PassRun]) -> bool:
    return run is not None and _all_turns(run) and all(counts_ok(t.facts) for t in run.turns)


def _highest(values: List[Optional[float]]) -> Optional[float]:
    got = [v for v in values if v is not None]
    return max(got) if got else None


def gate_payload(runs: Sequence[PassRun], error: Optional[str]) -> Dict[str, Any]:
    """The gate: published_equal (every pass 1 row's live report equals its published line), eos_stops (every EOS turn stopped by `eos`),
    features_ok (every pass 1 row has its features, and every EOS turn its neighbours and matching reports), the two timing maxima (of pass 1),
    eos_turn_ms_max (the slowest turn of pass 3, the default model's turn time: a number, not judged) and whether every session was deleted.
    `ok`, the exit status, also wants all three passes to have run every turn without a failure and no error of the run itself (`error`: a
    code, a class name, "timeout" or "interrupted")."""
    by_number = {run.number: run for run in runs}
    one, three = by_number.get(1), by_number.get(3)
    facts = [t.facts for t in one.turns] if one is not None else []
    eos_facts = [t.facts for t in three.turns] if three is not None else []
    problem = error or next((run.error for run in runs if run.error), None)
    gate: Dict[str, Any] = {
        "published_equal": _published_equal(one), "eos_stops": _eos_stops(three), "features_ok": _features_ok(one) and _eos_counts_ok(three),
        "first_byte_ms_max": _highest([f.first_byte_ms for f in facts]), "turn_ms_max": _highest([f.total_ms for f in facts]),
        "eos_turn_ms_max": _highest([f.total_ms for f in eos_facts]),
        "sessions_deleted": bool(runs) and all(run.session_deleted is True for run in runs)}
    gate["ok"] = (gate["published_equal"] and gate["eos_stops"] and gate["features_ok"] and gate["sessions_deleted"] and problem is None
                  and all(_ran(by_number.get(n)) for n in PASS_NAMES))
    if problem is not None:
        gate["error"] = problem
    return gate


# ---- the lines -----------------------------------------------------------------------------------------------------------------

def _pair(fetched: Fetched) -> List[Optional[int]]:
    return [fetched.status, fetched.size]


def _trouble(f: Facts) -> Dict[str, Any]:
    """The keys of a row that are there only when something went wrong with its turn: a class name or a fixed word."""
    return {key: value for key, value in (("error", f.error), ("refusal", f.refusal), ("stream_error", f.stream_error)) if value is not None}


def _timings(f: Facts) -> Dict[str, Any]:
    """A turn's times as pass 1 and pass 3 both print them: to the first byte, to the end of the stream, the server's own, and each stage's (ms)."""
    return {"first_byte_ms": f.first_byte_ms, "total_ms": f.total_ms, "server_ms": f.server_ms, "stages_ms": dict(f.stages_ms)}


def _gate_row(turn: Turn) -> Dict[str, Any]:
    f, features = turn.facts, turn.features
    row: Dict[str, Any] = {
        "row": turn.row, "status": f.status, "final": f.final, "seq_contiguous": f.seq_contiguous, **_timings(f), "skipped": dict(f.skipped),
        "live_equals_published": f.live_equals_published, "stopped": f.stopped, "tokens": f.tokens, "neighbors": f.neighbors,
        "matches": f.matches, "own_rank": f.own_rank, "own_rank_dedup": f.own_rank_dedup, "own_of": f.own_of, "agreement": list(f.agreement),
        "images": {"thumb": _pair(features.thumb) if features else [None, None],
                   "model_input": _pair(features.model_input) if features else [None, None]},
        "gallery": [_pair(x) for x in features.gallery] if features else [], "features_ok": row_ok(turn)}
    if f.agreement_pending:
        row["agreement_pending"] = True
    row.update(_trouble(f))
    return row


def _gap(before: Optional[float], after: Optional[float]) -> Optional[float]:
    return round(before - after, 1) if before is not None and after is not None else None


def _saved(base: Optional[Facts], f: Facts) -> Dict[str, Any]:
    """What a stop turn saved against pass 1's turn of the same row: tokens, generating ms (the generate stage's) and whole-turn ms."""
    tokens = base.tokens - f.tokens if base is not None and base.tokens is not None and f.tokens is not None else None
    return {"tokens_saved": tokens, "ms_saved": _gap(base.stages_ms.get("generate") if base else None, f.stages_ms.get("generate")),
            "turn_ms_saved": _gap(base.total_ms if base else None, f.total_ms)}


def _stop_entry(turn: Optional[Turn], base: Optional[Facts], with_savings: bool) -> Optional[Dict[str, Any]]:
    if turn is None:
        return None
    f = turn.facts
    entry: Dict[str, Any] = {"status": f.status, "final": f.final, "stopped": f.stopped, "fired": None if f.stopped is None else f.stopped == "repeat",
                             "tokens": f.tokens, "ms": f.stages_ms.get("generate"), "turn_ms": f.total_ms,
                             "complete_sentence": f.complete_sentence}
    if with_savings:
        entry.update(_saved(base, f))
    entry.update(_trouble(f))
    return entry


def _total_and_mean(values: List[Optional[float]]) -> Optional[Dict[str, Any]]:
    got = [v for v in values if v is not None]
    if not got:
        return None
    total = sum(got)
    return {"total": round(total, 1) if isinstance(total, float) else total, "mean": round(total / len(got), 1)}


def _gate_line(run: PassRun) -> Dict[str, Any]:
    facts = [t.facts for t in run.turns]
    return {"max_new_tokens": PUBLISHED_PROTOCOL["max_new_tokens"], "rows": [_gate_row(t) for t in run.turns],
            "published_equal": _published_equal(run), "features_ok": _features_ok(run),
            "first_byte_ms_max": _highest([f.first_byte_ms for f in facts]), "turn_ms_max": _highest([f.total_ms for f in facts])}


def _stop_line(run: PassRun, baseline: Optional[PassRun]) -> Dict[str, Any]:
    base = {t.row: t.facts for t in baseline.turns} if baseline is not None else {}
    rows = list(dict.fromkeys(t.row for t in run.turns))
    entries = []
    for row in rows:
        turns = {t.budget: t for t in run.turns if t.row == row}
        entries.append({"row": row, **{"b{}".format(budget): _stop_entry(turns.get(budget), base.get(row), budget == STOP_BUDGETS[0])
                                       for budget in STOP_BUDGETS}})
    first = [e["b{}".format(STOP_BUDGETS[0])] for e in entries]
    return {"budgets": list(STOP_BUDGETS), "rows": entries,
            "fired": {str(budget): sum(1 for e in entries if (e["b{}".format(budget)] or {}).get("fired") is True) for budget in STOP_BUDGETS},
            "of": len(rows),
            "tokens_saved_100": _total_and_mean([e and e["tokens_saved"] for e in first]),
            "ms_saved_100": _total_and_mean([e and e["ms_saved"] for e in first]),
            "turn_ms_saved_100": _total_and_mean([e and e["turn_ms_saved"] for e in first]),
            "complete_sentence_all": bool(run.turns) and all(t.facts.complete_sentence is True for t in run.turns)}


def _eos_entry(turn: Turn) -> Dict[str, Any]:
    f = turn.facts
    entry: Dict[str, Any] = {"status": f.status, "final": f.final, **_timings(f), "eos_trained": f.eos_trained, "stopped": f.stopped,
                             "tokens": f.tokens, "words": f.words, "last_sentence_terminated": f.last_sentence_terminated,
                             "neighbors": f.neighbors, "matches": f.matches, "skipped": dict(f.skipped)}
    entry.update(_trouble(f))
    return entry


def _eos_line(run: PassRun) -> Dict[str, Any]:
    picked = [t for t in run.turns if t.row is not None]
    drawn = next((t for t in run.turns if t.row is None), None)
    facts = [t.facts for t in run.turns]
    return {"max_new_tokens": run.turns[0].budget if run.turns else None, "rows": [dict({"row": t.row}, **_eos_entry(t)) for t in picked],
            "upload": _eos_entry(drawn) if drawn is not None else None, "eos_stops": _eos_stops(run), "counts_ok": _eos_counts_ok(run),
            "first_byte_ms_max": _highest([f.first_byte_ms for f in facts]), "turn_ms_max": _highest([f.total_ms for f in facts]),
            "generate_ms_max": _highest([f.stages_ms.get("generate") for f in facts])}


def pass_payload(run: PassRun, baseline: Optional[PassRun] = None) -> Dict[str, Any]:
    """A pass's RESULT line, from its turns' facts: numbers, booleans and fixed words. Pass 2 takes pass 1 as `baseline`, for what the stop saved."""
    line: Dict[str, Any] = {"pass": run.number, "name": run.name, "model": EOS_MODEL if run.number == 3 else PUBLISHED_MODEL}
    line.update(_gate_line(run) if run.number == 1 else _stop_line(run, baseline) if run.number == 2 else _eos_line(run))
    line["session_deleted"] = run.session_deleted
    line["ran"] = _ran(run) and run.session_deleted is True   # the pass ran: not whether what it measured is as it should be
    if run.error is not None:
        line["error"] = run.error
    return line


def result_line(payload: Dict[str, Any]) -> str:
    """The one line this script prints for a result."""
    return "RESULT " + json.dumps(payload, separators=(",", ":"))


def main(argv: Optional[Sequence[str]] = None, stdout: Optional[TextIO] = None, stderr: Optional[TextIO] = None) -> int:
    """Run the probe; -> the exit status. `stdout` and `stderr` are where its two kinds of line go (the process's own by default)."""
    stdout = sys.stdout if stdout is None else stdout
    stderr = sys.stderr if stderr is None else stderr
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.help:
            drill._write(stdout, parser.format_help().rstrip("\n"))
            return 0
        if args.url is None:
            raise drill.Refused(drill.URL_MISSING)
        base, token = drill.check_url(args.url), drill.find_token(args.token_file)
        if token is not None and drill.sends_token_in_clear(base):   # before there is a client to send it with
            raise drill.Refused(drill.CLEARTEXT_REFUSAL)
    except drill.Refused as exc:
        drill._write(stderr, "ERROR " + str(exc))
        return 2
    headers = {"X-Client-Id": CLIENT_ID}
    if token is not None:
        headers["Authorization"] = "Bearer " + token
    deadline = time.perf_counter() + args.timeout
    runs: List[PassRun] = []
    error: Optional[str] = None
    try:
        # trust_env=False: the connection goes straight to the URL that was given. Left on, httpx sends even loopback requests to the proxy
        # named in HTTP_PROXY or ALL_PROXY, and the token and the stream would pass through it. No redirect is followed, for the same reason.
        with httpx.Client(base_url=base, headers=headers, timeout=drill.CLIENT_TIMEOUT, trust_env=False, follow_redirects=False) as client:
            for number, name in PASS_NAMES.items():
                run = PassRun(number, name)
                runs.append(run)
                try:
                    run_pass(client, run, args, deadline)
                finally:   # its line is written whether it finished, failed or was interrupted: what was learnt is not lost
                    drill._write(stdout, result_line(pass_payload(run, runs[0] if number == 2 else None)))
                if run.error is not None:   # no session, a timeout, Ctrl-C: the later passes would meet the same
                    break
    except KeyboardInterrupt:
        error = "interrupted"
    except Exception as exc:   # the class, never the text
        error = "internal_" + type(exc).__name__
    gate = gate_payload(runs, error)
    drill._write(stdout, result_line({"gate": gate}))
    return 0 if gate["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
