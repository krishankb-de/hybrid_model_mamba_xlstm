"""CHAT_UI_PLAN.md P7-E: the requeue drill, run from the laptop against the chat server through the SSH tunnel. It prints numbers only.

    LOCAL_PORT=<port> bash app/tunnel/tunnel.sh              # another terminal: the tunnel, on a port that is not the dev server's
    CHAT_TOKEN=... venv/bin/python scripts/chat_requeue_drill.py http://localhost:PORT --job JOBID
    venv/bin/python scripts/chat_requeue_drill.py http://localhost:PORT --job JOBID --token-file ~/chat.token --wait 1200

The plan's box: `scontrol requeue <job>` while a turn is running stores the turn as `error` with `server_restart`, the job comes back, and the
tunnel re-attaches without anyone's hand. The drill does it once, on the live server, and records what it saw. It interrupts that server for a few
minutes, so it is run once the user has said so. In order:

  1  Before. It reads /healthz and keeps the server's `started_at` in memory (it is compared and never printed), and asks the scheduler for the job's
     node. A server that does not answer, or a job that has no node (it is not running), ends the drill here: no session is made, nothing is requeued.
  2  The turn. It makes one session and sends one turn on the published model, at 200 tokens, which is the server's cap, with a picture that it draws
     itself (no MIMIC image). The published model always decodes to its budget, so the turn runs for seconds (about 16 s on the live server today)
     and is far from its end when the requeue comes. The model is named as the server's registry names it, `hybrid_150m_m3_rrg`: `m3` is only the
     serving command's alias for it (`--models m3`), and the EOS-trained model, which may stop early, is not the one asked for. The stream is read on
     a thread of its own, counting event names only, until three `content_block_delta` events have arrived: under beam search the server sends one
     for every decode step.
  3  The requeue. It runs once. `requeue_s` is how long the command took.
  4  The cut. The reader goes on to the end of the stream and says how it ended, as one word: `error_event` (an `error` event came, and then the
     stream closed: the server's own shutdown on SLURM's SIGTERM), `reset` (the connection broke, as when a node is lost), `closed` (the stream
     ended with no error event) or `timeout` (still open when the drill gave up on it). `error_event` says whether an `error` event came before the
     end, and `stream_error` its `error.type`, a word of the probe's ERROR_KINDS.
  5  The return. /healthz is read at once and then every 5 s until it answers ok with a start time other than the one from step 1. `back_s` is the
     time from the start of the requeue command to that answer: the wait in the queue, the engines loading and the tunnel's re-attach are all in it.
     `first_refused_s` is the time to the first read that failed. A server that is not back after --wait seconds (default 1800) is `not_back`, and
     the drill gives up there.
  6  The stored turn. GET /v1/messages/<id>: its `status`, the name of its last event, the `error.type` of its `error` event and how many events it
     has. The expected answer is `error`, `message_stop`, `server_restart`. After SIGTERM the stopping server stores it; after SIGKILL, or a lost
     node, the next start's recovery does. The node is asked for again (`node_after`).
  7  Clean up. The session is deleted, whatever happened.

Output. Stdout is one line, `RESULT {json}`: drill (`requeue`), node_before, node_after (a node's name such as gx14, `other` for any other text, null
for none), deltas_at_requeue, requeue_s, first_refused_s and back_s (seconds, to a tenth; null when it never happened), stream_end, error_event,
stream_error, turn_status, turn_error, turn_last_event, events (how many events the stored turn has), session_deleted, ok, and `error`, a fixed code,
when the drill itself went wrong: `healthz_<class>`, `healthz_http_<status>` or `healthz_bad_answer` (the first read), `node_failed` (the cluster did
not answer), `job_not_running`, the load drill's `session_create_*` codes, `turn_http_<status>` (the turn was refused), `no_deltas` (the stream ended
or ran out of time before three deltas), `requeue_failed`, `not_back`, `stored_turn_http_<status>`, `stored_turn_<class>`, `stored_turn_bad_answer`,
`stored_turn_no_id`, `interrupted` and `internal_<class>`. `ok` is true only when the server came back and the stored turn is an `error` whose kind is
`server_restart`, and nothing else went wrong; whether the session could be deleted is on the line and is no part of it. Exit status 0 when ok is
true, 1 when it is not, 2 for an argument it refuses, which is one `ERROR ...` line on stderr, before anything is sent. Nothing else is printed:
stderr is empty on every run that was not refused.

The cluster. The drill starts nothing of its own. The cluster is an object with two methods, node(job) and requeue(job), and main() takes it as an
argument. The real one, SshCluster, is the only place in this script that names ssh or starts a process, and it builds exactly two commands, each one
ssh call with no shell on this side and nothing on stdin, under a time limit:

    ssh -o BatchMode=yes hpi-hpc squeue -j JOB -h -o %N
    ssh -o BatchMode=yes hpi-hpc scontrol requeue JOB

JOB is digits only, at most 12 of them, checked when the arguments are read and again where the command is made. The tests hand the drill a cluster
of their own, and none of them runs ssh.

D22 is read elsewhere. Whether the SQLite file opened cleanly on the new node is in the serve job's log, which `scripts/chat_remote.sh summary` shows:
the new run's `[server] sqlite journal_mode=...` and `[server] recovered N running turn(s)` lines. The stored turn here being readable says the same.

R7 (DUA). The live server's turns carry MIMIC-derived text: the generated report and every snapshot of it, the matched reports, the study ids. This
script keeps of the stream only event names, a count, and the `error.type` of an `error` event as a word of a fixed list; a delta's data, which holds
a snapshot, is never parsed. Of the stored turn it keeps a status, the name of its last event, an error's kind, each as a word of a closed list, and a
count. Of /healthz it compares one string and keeps nothing; of the scheduler it keeps a node's name, only when it is a plain name. The ids it uses
(the session's, the message's) are opaque and are never printed. Nothing of a frame, or of any answer, is printed, logged or written to a file, a
failure included: an exception is named by its class, never by its text, and a cluster that fails is a fixed code. The only writes are the RESULT
line, the refusals and the help text, through the load drill's one function, and what they hold is numbers, booleans and fixed words.

Transport. As scripts/chat_load_drill.py and scripts/chat_realturn_probe.py, whose helpers it imports: httpx only, one client that goes straight to
the URL it was given (no proxy named in the environment is used, no redirect is followed, no connection is kept alive). The token comes from the
environment variable CHAT_TOKEN or from --token-file (a regular file of mode 0600 that you own), never from an argument, is sent as a bearer to that
URL and nowhere else, is never printed, and is not sent over plain http to a host that is not loopback. The URL itself must be a loopback address,
the tunnel's own end: the drill requeues a job, and what it talks to is the tunnel. It sends one fixed X-Client-Id, `chat-requeue-drill`. The dev
server's port is refused. One session is made and deleted when the drill ends, also when something went wrong or Ctrl-C was pressed: the server
hides a deleted session at once and stops its turn.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, TextIO, Tuple

import httpx

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:   # `python scripts/chat_requeue_drill.py` puts scripts/ on the path, not the repo's root
    sys.path.insert(0, REPO_ROOT)
from scripts import chat_load_drill as drill  # noqa: E402
from scripts import chat_realturn_probe as probe  # noqa: E402

CLIENT_ID = "chat-requeue-drill"     # one fixed id, on every request: the server's pattern is visible ASCII, 1 to 128 characters
CLUSTER_HOST = "hpi-hpc"             # the login node's ssh alias (scripts/chat_cluster.env.example)

# What the app holds, restated: the script imports nothing of the app (it starts with the server's code absent), and a test holds each equal.
MODEL = probe.PUBLISHED_MODEL        # the registry's name of the model that `--models m3` serves: it always decodes to its budget
MAX_NEW_TOKENS = 200                 # the server's cap
MIN_DELTAS = 3                       # the requeue is sent once this many content_block_delta events have arrived
DELTA = "content_block_delta"
EVENT_NAMES = ("message_start", "stage_start", "stage_end", "content_block_start", "content_block_delta", "content_block_stop", "warning",
               "error", "message_stop")
STORED_STATUSES = ("running",) + drill.STATUSES   # the status of a stored message

DEFAULT_WAIT_S = 1800.0              # how long the server may take to come back
POLL_S = 5.0                         # between two reads of /healthz
HEALTHZ_TIMEOUT_S = 10.0             # one read of /healthz: a dead tunnel must not hold a poll for the client's two minutes
TURN_START_S = 300.0                 # from the POST to the third delta: the queue, the stages before generate, three steps
STREAM_S = 900.0                     # how long the reader keeps a stream open, from its POST
FINISH_S = 5.0                       # how long the drill waits for the stream to end once it is over with the server: it ended long before, or it hangs
SSH_TIMEOUT_S = 60.0                 # one command on the cluster: ssh gives up connecting after its own ten seconds, a scheduler call is quick
JOB = re.compile(r"[0-9]{1,12}")     # a job id goes into the cluster's commands: only digits are taken
NODE = re.compile(r"[a-z]{1,8}[0-9]{1,4}")   # a node's name as this cluster writes it (gx14, cn01): the one shape that is printed

# What a refusal says is fixed text: it never holds a value that was given (it could be a secret put in the wrong place).
JOB_REFUSAL = "must be the SLURM job id of the serve job: a whole number of at most 12 digits"
JOB_MISSING = "the SLURM job id of the serve job is required: --job JOBID"
LOOPBACK_REFUSAL = "the drill goes through the tunnel: the URL must be a loopback address (localhost, 127.0.0.1 or ::1)"


# ---- the arguments -------------------------------------------------------------------------------------------------------------

def _job(text: str) -> str:
    if JOB.fullmatch(text) is None:
        raise argparse.ArgumentTypeError(JOB_REFUSAL)
    return text


def build_parser() -> argparse.ArgumentParser:
    p = drill._Parser(prog="chat_requeue_drill.py", allow_abbrev=False, add_help=False,
                      description="Requeue the chat server's SLURM job while a turn is running on it, and print numbers only: one RESULT line. "
                                  "Exit 0 when the interrupted turn is stored as an error with server_restart and the server came back, 1 when "
                                  "not, 2 for an argument it refuses.")
    p.add_argument("url", metavar="URL", nargs="?", help="the server, through the tunnel: http://localhost:PORT (a loopback address, and not the "
                                                         "dev server's port)")
    p.add_argument("--job", type=_job, metavar="JOBID", help="the SLURM job id of the serve job: digits only")
    p.add_argument("--token-file", metavar="PATH", help="the token: a file of mode 0600 that you own; or set CHAT_TOKEN. Never an argument")
    p.add_argument("--wait", type=drill._seconds, default=DEFAULT_WAIT_S, metavar="SECONDS",
                   help="how long the server may take to come back (default 1800)")
    p.add_argument("-h", "--help", action="store_true", help="show this help")
    return p


def turn_options() -> Dict[str, Any]:
    """The published protocol and the page's retrieval counts, on the published model, at the server's cap of 200 tokens: the turn is not stopped
    by a repeat or by an end of report, so it runs for its whole budget."""
    return probe.turn_options(model=MODEL, max_new_tokens=MAX_NEW_TOKENS)


# ---- the cluster ---------------------------------------------------------------------------------------------------------------

class ClusterError(Exception):
    """The cluster could not be asked. Its text is a fixed word (timeout, not_run, failed, bad_job), never a part of what ssh or the scheduler said."""


class SshCluster:
    """The real cluster, and the one place in this script that names ssh or starts a process: two commands, each a plain argument list (no shell on
    this side), with nothing on stdin, run under a time limit. A test hands the drill a cluster of its own with the same two methods."""

    def node(self, job: str) -> Optional[str]:
        """The node the job runs on, as the scheduler prints it; None when it prints none (the job is not running)."""
        job = self._checked(job)
        return self._run(["ssh", "-o", "BatchMode=yes", CLUSTER_HOST, "squeue", "-j", job, "-h", "-o", "%N"]).strip() or None

    def requeue(self, job: str) -> None:
        job = self._checked(job)
        self._run(["ssh", "-o", "BatchMode=yes", CLUSTER_HOST, "scontrol", "requeue", job])

    @staticmethod
    def _checked(job: Any) -> str:
        """The parser took only digits; asked again here, where the id is put into a command, so that no other caller can pass anything else."""
        if not isinstance(job, str) or JOB.fullmatch(job) is None:
            raise ClusterError("bad_job")
        return job

    @staticmethod
    def _run(argv: List[str]) -> str:
        try:
            done = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True, text=True, errors="replace", timeout=SSH_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            raise ClusterError("timeout") from None
        except OSError:
            raise ClusterError("not_run") from None
        if done.returncode != 0:
            raise ClusterError("failed")
        return done.stdout


def node_word(value: Any) -> Optional[str]:
    """A node as the scheduler wrote it: kept when it is a plain name (gx14), `other` for any other text, None for none. Only these reach the line."""
    text = value.strip() if isinstance(value, str) else ""
    if not text:
        return None
    return text if NODE.fullmatch(text) else "other"


# ---- /healthz: is the server back ----------------------------------------------------------------------------------------------

BACK, OLD, REFUSED = "back", "old", "refused"   # what one poll of /healthz came to


def read_health(client: httpx.Client) -> Tuple[Optional[str], Optional[str]]:
    """GET /healthz -> (its started_at, None) when it answers 200 with status ok and a started_at; else (None, a fixed code)."""
    try:
        response = client.get("/healthz", timeout=HEALTHZ_TIMEOUT_S)
    except Exception as exc:   # the class, never the text
        return None, "healthz_" + type(exc).__name__
    if response.status_code != 200:
        return None, "healthz_http_{}".format(response.status_code)
    body = drill._json_or_none(response)
    started = body.get("started_at") if isinstance(body, dict) and body.get("status") == "ok" else None
    return (started, None) if isinstance(started, str) else (None, "healthz_bad_answer")


def ask_health(client: httpx.Client, before: str) -> str:
    """One poll: BACK when /healthz answers ok with a start time other than `before` (the server's start before the requeue), OLD when it answers
    with that one (the old server, still stopping), REFUSED when it answers anything else or nothing."""
    started, _ = read_health(client)
    if started is None:
        return REFUSED
    return OLD if started == before else BACK


def wait_for_return(ask: Callable[[], str], t0: float, wait_s: float, poll_s: float = POLL_S, clock: Callable[[], float] = time.perf_counter,
                    sleep: Callable[[float], None] = time.sleep) -> Tuple[Optional[float], Optional[float]]:
    """Poll with `ask` (the first poll at once, then one every `poll_s` seconds) until it says BACK, or until `wait_s` seconds have passed since
    `t0`, the clock's reading when the requeue was started. -> (first_refused_s, back_s): the seconds from t0 to the answer that first said
    REFUSED, and to the one that said BACK, each None when it never happened. A poll is timed by when its answer came."""
    first_refused: Optional[float] = None
    while True:
        answer = ask()
        elapsed = clock() - t0
        if answer == BACK:
            return first_refused, round(elapsed, 1)
        if answer == REFUSED and first_refused is None:
            first_refused = round(elapsed, 1)
        if elapsed >= wait_s:
            return first_refused, None
        sleep(min(poll_s, wait_s - elapsed))


# ---- the turn's stream: event names only ---------------------------------------------------------------------------------------

@dataclass
class Stream:
    """What the turn's stream came to: numbers, booleans and fixed words only (R7). Nothing of a frame's text has a field here; the message id is an
    opaque id of the stored turn, never printed."""
    status: Optional[int] = None         # the HTTP status of the POST, None when no answer came
    message_id: Optional[str] = None
    deltas: int = 0                      # content_block_delta events counted
    error_event: bool = False            # an `error` event arrived
    error_kind: Optional[str] = None     # its error.type: a word of the probe's ERROR_KINDS, or unknown
    end: Optional[str] = None            # how the stream ended: reset, closed, error_event or timeout; None while it is open or when it never opened


def error_kind(text: str) -> str:
    """The kind of an `error` event, from its data: a word of the probe's ERROR_KINDS, else `unknown`. Its message is not looked at."""
    try:
        data = json.loads(text)
    except (ValueError, RecursionError):
        return "unknown"
    error = data.get("error") if isinstance(data, dict) else None
    return probe._word(error.get("type") if isinstance(error, dict) else None, probe.ERROR_KINDS)


def count_frames(stream: Stream, chunks: Iterable[str], on_enough: Callable[[], None]) -> None:
    """Read a turn's stream frame by frame and keep the names: how many content_block_delta events came (`on_enough` is called as the MIN_DELTAS-th
    does), and whether an `error` event came, with its kind. A delta's data, which holds a snapshot of the report, is never parsed; the data of an
    `error` event is, for its kind alone."""
    for event, text in probe.frames(chunks):
        if event == DELTA:
            stream.deltas += 1
            if stream.deltas == MIN_DELTAS:
                on_enough()
        elif event == "error":
            stream.error_event = True
            stream.error_kind = error_kind(text)


class TurnReader:
    """The turn's POST and its stream, read on a thread of its own: the main thread waits for the third delta (`enough`), sends the requeue, and
    goes on to poll /healthz at once, while this thread reads the stream to its end, whatever that is. It never raises: what went wrong is the
    stream's `end`, a fixed word."""

    def __init__(self, client: httpx.Client, session_id: str, deadline_s: float = STREAM_S, clock: Callable[[], float] = time.perf_counter):
        self.client, self.session_id, self.deadline_s, self.clock = client, session_id, deadline_s, clock
        self.stream = Stream()
        self.enough = threading.Event()     # set at the third delta, and when the stream is over without it
        self.thread = threading.Thread(target=self._run, name="turn", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def finish(self, wait_s: float) -> None:
        """Give the thread `wait_s` seconds to end; a stream that is still open then is given up on, and ended as a timeout."""
        self.thread.join(wait_s)
        if self.thread.is_alive() and self.stream.end is None:
            self.stream.end = "timeout"

    def _run(self) -> None:
        try:
            self._read()
        except Exception:   # nothing may leave this thread: an exception that escaped it would be printed, with its text, on stderr
            self.stream.end = self.stream.end or "reset"
        finally:
            self.enough.set()

    def _read(self) -> None:
        stream = self.stream
        deadline = self.clock() + self.deadline_s
        try:
            with self.client.stream("POST", "/v1/sessions/{}/messages".format(self.session_id),
                                    files={"image": ("drill.png", drill.synthetic_png(), "image/png")},
                                    data={"text": "", "options": json.dumps(turn_options())}) as response:
                stream.status = response.status_code
                if response.status_code != 200:
                    return
                found = response.headers.get("X-Message-Id")
                stream.message_id = found if isinstance(found, str) and probe.ID.fullmatch(found) else None
                count_frames(stream, probe.watched(response.iter_text(), deadline, self.clock), self.enough.set)
            stream.end = "error_event" if stream.error_event else "closed"
        except (probe.Timeout, httpx.TimeoutException):
            stream.end = "timeout"
        except Exception:   # the connection broke, was closed in the middle of the body, or was never made
            stream.end = "reset"


# ---- the stored turn -----------------------------------------------------------------------------------------------------------

@dataclass
class Stored:
    """What GET /v1/messages/<id> says of the interrupted turn once the server is back: words of closed lists, and a count."""
    http: Optional[int] = None
    status: Optional[str] = None         # running, done, error, aborted or unknown
    error: Optional[str] = None          # the kind of its `error` event (the probe's ERROR_KINDS, or unknown); None when it has none
    last_event: Optional[str] = None     # the name of its last event: one of EVENT_NAMES, or other
    events: Optional[int] = None


def read_stored(client: httpx.Client, message_id: Optional[str]) -> Tuple[Stored, Optional[str]]:
    """GET the stored turn and read it down to a Stored. -> (it, None), or (what is known, a fixed code) when it could not be read. The events,
    which carry the report and the retrieval, are looked at for their names and for the kind of an error; nothing else of them is kept."""
    stored = Stored()
    if message_id is None:
        return stored, "stored_turn_no_id"
    try:
        response = client.get("/v1/messages/{}".format(message_id))
    except httpx.HTTPError as exc:   # the class, never the text
        return stored, "stored_turn_" + type(exc).__name__
    stored.http = response.status_code
    if response.status_code != 200:
        return stored, "stored_turn_http_{}".format(response.status_code)
    body = drill._json_or_none(response)
    if not isinstance(body, dict):
        return stored, "stored_turn_bad_answer"
    stored.status = probe._word(body.get("status"), STORED_STATUSES)
    events = body.get("events")
    if isinstance(events, list):
        stored.events = len(events)
        for event in events:
            if isinstance(event, dict) and event.get("event") == "error":
                data = event.get("data")
                inner = data.get("error") if isinstance(data, dict) else None
                stored.error = probe._word(inner.get("type") if isinstance(inner, dict) else None, probe.ERROR_KINDS)
        if events:
            last = events[-1]
            stored.last_event = probe._word(last.get("event") if isinstance(last, dict) else None, EVENT_NAMES, "other")
    return stored, None


# ---- the drill -----------------------------------------------------------------------------------------------------------------

@dataclass
class Drill:
    """What the drill found: numbers, booleans and fixed words only, so that all of it may be printed (R7). `error` is a fixed code or a class name,
    and is None when nothing went wrong with the drill itself."""
    node_before: Optional[str] = None
    node_after: Optional[str] = None
    deltas: Optional[int] = None               # content_block_delta events counted when the requeue was sent
    requeue_s: Optional[float] = None          # how long the requeue command took
    first_refused_s: Optional[float] = None    # from the start of the requeue command to the first /healthz that failed
    back_s: Optional[float] = None             # from the start of the requeue command to the /healthz that said ok with a new start time
    stream: Stream = field(default_factory=Stream)
    stored: Stored = field(default_factory=Stored)
    session_deleted: Optional[bool] = None
    error: Optional[str] = None


def ask_node(cluster: Any, job: str) -> Tuple[bool, Optional[str]]:
    """-> (whether the cluster answered, the job's node as a word: its name, `other`, or None when the scheduler named none)."""
    try:
        return True, node_word(cluster.node(job))
    except Exception:   # whatever it says, the text is not used
        return False, None


def run_drill(client: httpx.Client, cluster: Any, job: str, wait_s: float, poll_s: float = POLL_S,
              clock: Callable[[], float] = time.perf_counter, sleep: Callable[[float], None] = time.sleep) -> Drill:
    """The drill: read /healthz and ask for the node; make a session and send one turn on a thread of its own; once it has streamed MIN_DELTAS
    deltas, requeue the job and poll /healthz until the server answers ok with a start time it did not have; read the stored turn; delete the
    session. A session that was made is deleted whatever happens (a failure, a timeout, Ctrl-C). It does not raise (an interrupt is turned into the
    code `interrupted`): what went wrong is `error`, a fixed code or the class of an exception."""
    run = Drill()
    session_id: Optional[str] = None
    reader: Optional[TurnReader] = None
    try:
        before, run.error = read_health(client)
        if before is None:
            return run
        answered, run.node_before = ask_node(cluster, job)
        if not answered or run.node_before is None:
            run.error = "node_failed" if not answered else "job_not_running"
            return run
        session_id = drill.create_session(client)
        reader = TurnReader(client, session_id)
        run.stream = reader.stream
        reader.start()
        reader.enough.wait(TURN_START_S)
        if reader.stream.deltas < MIN_DELTAS:
            status = reader.stream.status
            run.error = "turn_http_{}".format(status) if status not in (None, 200) else "no_deltas"
            return run
        run.deltas = reader.stream.deltas
        t0 = clock()
        try:
            cluster.requeue(job)
        except Exception:   # whatever it says, the text is not used
            run.error = "requeue_failed"
            return run
        run.requeue_s = round(clock() - t0, 1)
        run.first_refused_s, run.back_s = wait_for_return(lambda: ask_health(client, before), t0, wait_s, poll_s, clock, sleep)
        reader.finish(FINISH_S)
        if run.back_s is None:
            run.error = "not_back"
        else:
            run.stored, run.error = read_stored(client, reader.stream.message_id)
        run.node_after = ask_node(cluster, job)[1]
    except KeyboardInterrupt:
        run.error = "interrupted"
    except drill.DrillError as exc:   # its text is a code, written in the drill
        run.error = str(exc)
    except Exception as exc:   # the class, never the text
        run.error = "internal_" + type(exc).__name__
    finally:
        if session_id is not None:
            run.session_deleted = drill.delete_session(client, session_id)
            if reader is not None and reader.stream.end is None:   # the deletion stops a turn that is still running, and with it its stream
                reader.finish(FINISH_S)
    return run


def payload(run: Drill) -> Dict[str, Any]:
    """The RESULT line's content. `ok` holds when the server came back and the interrupted turn is stored as an error of kind server_restart, and
    nothing else went wrong with the drill. Whether the session could be deleted is on the line, and is no part of it."""
    s, t = run.stream, run.stored
    ok = run.error is None and run.back_s is not None and t.status == "error" and t.error == "server_restart"
    out: Dict[str, Any] = {
        "drill": "requeue", "node_before": run.node_before, "node_after": run.node_after, "deltas_at_requeue": run.deltas,
        "requeue_s": run.requeue_s, "first_refused_s": run.first_refused_s, "back_s": run.back_s, "stream_end": s.end, "error_event": s.error_event,
        "stream_error": s.error_kind, "turn_status": t.status, "turn_error": t.error, "turn_last_event": t.last_event, "events": t.events,
        "session_deleted": run.session_deleted, "ok": ok}
    if run.error is not None:
        out["error"] = run.error
    return out


def main(argv: Optional[Sequence[str]] = None, stdout: Optional[TextIO] = None, stderr: Optional[TextIO] = None,
         cluster: Any = None, poll_s: float = POLL_S) -> int:
    """Run the drill; -> the exit status. `stdout` and `stderr` are where its two kinds of line go (the process's own by default), `cluster` the
    object that answers for the cluster (the real one, SshCluster, by default) and `poll_s` the seconds between two reads of /healthz."""
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
        if args.job is None:
            raise drill.Refused(JOB_MISSING)
        base, token = drill.check_url(args.url), drill.find_token(args.token_file)
        if token is not None and drill.sends_token_in_clear(base):   # before there is a client to send it with
            raise drill.Refused(drill.CLEARTEXT_REFUSAL)
        if not drill.is_loopback(httpx.URL(base).host):
            raise drill.Refused(LOOPBACK_REFUSAL)
    except drill.Refused as exc:
        drill._write(stderr, "ERROR " + str(exc))
        return 2
    headers = {"X-Client-Id": CLIENT_ID}
    if token is not None:
        headers["Authorization"] = "Bearer " + token
    run = Drill()
    try:
        # trust_env=False: the connection goes straight to the URL that was given. Left on, httpx sends even loopback requests to the proxy
        # named in HTTP_PROXY or ALL_PROXY, and the token and the stream would pass through it. No redirect is followed, for the same reason.
        # No connection is kept alive: a poll never reuses one that the old server left in the pool.
        with httpx.Client(base_url=base, headers=headers, timeout=drill.CLIENT_TIMEOUT, trust_env=False, follow_redirects=False,
                          limits=httpx.Limits(max_keepalive_connections=0)) as client:
            run = run_drill(client, SshCluster() if cluster is None else cluster, args.job, args.wait, poll_s)
    except KeyboardInterrupt:
        run.error = "interrupted"
    except Exception as exc:   # the class, never the text
        run.error = "internal_" + type(exc).__name__
    result = payload(run)
    drill._write(stdout, probe.result_line(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
