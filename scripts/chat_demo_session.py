"""CHAT_UI_PLAN.md P9-C: the supervisor demo's session, run from the laptop against the chat server through the SSH tunnel. It prints numbers only.

    LOCAL_PORT=<port> bash app/tunnel/tunnel.sh              # another terminal: the tunnel, on a port that is not the dev server's
    CHAT_TOKEN=... venv/bin/python scripts/chat_demo_session.py http://localhost:PORT --rows A,B,C,D,E
    venv/bin/python scripts/chat_demo_session.py http://localhost:PORT --rows A,B,C,D,E --token-file ~/chat.token

The plan's box: five test studies (two normal, two with support devices, one with a pneumothorax, picked from the published labels by
scripts/chat_demo_cases.py) run through the live server in private mode, and the session's export saved on the cluster. This script does the run.
In order:

  1  One session, titled "Supervisor demo".
  2  One test-split turn per row of --rows, one after another, in the order given, on the server's default model (the EOS-trained model on the
     live server: no model is named), under the published protocol (beam 3, no stop on repeat) and the page's retrieval counts (4 images, 3
     reports), at --max-new-tokens (default 200, the budget the EOS model's registry note was measured at). Each turn's stream is read down to its
     facts by the probe's reader (scripts/chat_realturn_probe.py run_turn), as the probe reads it.
  3  The saves. POST /v1/sessions/<id>/export/save, for `format=json` and then `format=md`: the server writes each export under exports/ in its
     CHAT_HOME (owner-only, a new file each time) and answers the file's name and size. The export never comes here: it holds the reports, the
     matched reports and the study ids, and under rules R1 and R7 it stays on the cluster. This script asks for the saves even when a turn did not
     go as it should, so that the session is on the cluster as it stands, and does not ask when the run itself broke off (a time limit, Ctrl-C).
  4  The session is kept. It is not deleted, so that the user can open it in the page. (The session id is a random id of the server's, not
     MIMIC-derived.)

Output. Stdout is one line, `RESULT {json}`: demo (`done` when ok, else `failed`), rows (as given), turns (how many ran), statuses (one word per
turn: `done`, `error`, `aborted` or `unknown` for the status of its message_stop, `refused` for a turn the server answered with a status other than
200, `timeout`, `failed` for one that no answer came to, `incomplete` for a stream with no message_stop), tokens (the generate stage's token count
per turn, null where there is none), stopped (per turn, the generate stage's stop word: `eos`, `budget` or `repeat`, the probe's closed list, or
`unknown` for any other word the server sent, null where the stage said none), eos_trained (once: whether the model that ran the turns, the
server's default, was trained to end its reports, as the turns' own message_start says: true or false when they agree, null when none says or they
differ), session (its id, null when none was made), saved (the file names the server gave, json then md, only for a save that worked) and ok. Of
the two, `budget` with `eos_trained` true is a report that was cut at the token budget, where `eos` is a report the model ended itself, and `budget`
with false is how a model that never stops ends every report. A turn that did not go as it should adds `errors`: per turn the word that says why,
the `error.type` of its error event
(`model_error`, `store_failed` ...) or of its refusal (`validation_error`, `permission_error` ...) from the probe's closed lists (`unknown` for
anything else), `seq_gap` for a done turn whose seq has a gap, the class name of a failure or `timeout`, and null for a turn that was fine. The
driver's own trouble is `error`, a fixed code: the load drill's `session_create_*` codes, `timeout`, `interrupted`, `internal_<class>`, and for the
first save that failed `save_http_<status>`, `save_bad_answer` (the answer was not a file name of the server's own for this session and format, with a
size) or `save_<class>`. `ok` is true only when every turn ended `done` with a seq that has no gap, both saves worked, and nothing else went wrong.
Exit status 0 when ok, 1 when not, 2 for an argument it refuses, which is one `ERROR ...` line on stderr, before anything is sent. Nothing else is
printed: stderr is empty on every run that was not refused.

R7 (DUA). The live server's turns carry MIMIC-derived text: the generated report and every snapshot of it, the matched reports, the study ids, the
reference. The turn is read by the probe's reader, which keeps only numbers, booleans and words chosen from fixed lists, and drops the rest as it
reads it. Of a save's answer this script keeps one string, and only when it matches the shape of a file name that the server's own save makes for this
session and format (a name that does not is dropped, and says `save_bad_answer`), and one number under a bound; nothing else of it. Nothing of a
frame, or of any answer, is printed, logged or written to a file, a failure included: an exception is named by its class, never by its text. The only
writes are the RESULT line, the refusals and the help text, through the load drill's one function, and what they hold is numbers, booleans, fixed
words and the names of files the server wrote.

Transport. As scripts/chat_load_drill.py and scripts/chat_realturn_probe.py, whose helpers it imports: httpx only, one client that goes straight to
the URL it was given (no proxy named in the environment is used, no redirect is followed). The token comes from the environment variable CHAT_TOKEN
or from --token-file (a regular file of mode 0600 that you own), never from an argument, is sent as a bearer to that URL and nowhere else, is never
printed, and is not sent over plain http to a host that is not loopback. The URL itself must be a loopback address, the tunnel's own end: this script
asks a server to write private text to its disk, and what it talks to is the tunnel. The dev server's port is refused. It sends one fixed X-Client-Id,
`chat-demo-session`.
"""
import argparse
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, TextIO, Tuple

import httpx

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:   # `python scripts/chat_demo_session.py` puts scripts/ on the path, not the repo's root
    sys.path.insert(0, REPO_ROOT)
from scripts import chat_load_drill as drill  # noqa: E402
from scripts import chat_realturn_probe as probe  # noqa: E402

CLIENT_ID = "chat-demo-session"      # one fixed id, on every request: the server's pattern is visible ASCII, 1 to 128 characters
TITLE = "Supervisor demo"             # the session's title
SAVE_FORMATS = ("json", "md")        # the exports asked for, in this order
SAVE_PATH = "/v1/sessions/{}/export/save"
DEFAULT_MAX_NEW_TOKENS = probe.DEFAULT_EOS_TOKENS   # 200: the budget at which the EOS model's registry note was measured
DEFAULT_TIMEOUT_S = probe.DEFAULT_TIMEOUT_S
# What the server names a saved export (Store.save_export): session-<id>-<UTC time>[-<n>].<json|md>. Anything else is not taken for a file name.
SAVED_NAME = re.compile(r"session-[A-Za-z0-9_-]{1,64}-[0-9]{8}T[0-9]{6}Z(-[0-9]{1,4})?\.(json|md)")

# What a refusal says is fixed text: it never holds a value that was given (it could be a secret put in the wrong place).
ROWS_MISSING = "the test rows are required: --rows A,B,C,D,E (row numbers of the test split)"
LOOPBACK_REFUSAL = "the demo goes through the tunnel: the URL must be a loopback address (localhost, 127.0.0.1 or ::1)"


# ---- the arguments -------------------------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = drill._Parser(prog="chat_demo_session.py", allow_abbrev=False, add_help=False,
                      description="Run the supervisor demo on the chat server: one session titled \"{}\", one test-split turn per row on the server's "
                                  "default model, and the session's export saved as JSON and Markdown on the server (it never comes here). Print "
                                  "numbers only: one RESULT line. Exit 0 when every turn is done and both saves worked, 1 when not, 2 for an "
                                  "argument it refuses.".format(TITLE))
    p.add_argument("url", metavar="URL", nargs="?", help="the server, through the tunnel: http://localhost:PORT (a loopback address, and not the "
                                                         "dev server's port)")
    p.add_argument("--rows", type=probe.parse_rows, metavar="ROWS",
                   help="the test-split rows to run, one turn each, in this order: such as 12,345,901,1500,2001 (at most {})".format(probe.MAX_ROWS))
    p.add_argument("--token-file", metavar="PATH", help="the token: a file of mode 0600 that you own; or set CHAT_TOKEN. Never an argument")
    p.add_argument("--max-new-tokens", type=drill._whole(drill.MIN_NEW_TOKENS, drill.MAX_NEW_TOKENS), default=DEFAULT_MAX_NEW_TOKENS,
                   help="the token budget of each turn, 16 to 200 (default 200, where the EOS model's note was measured)")
    p.add_argument("--timeout", type=drill._seconds, default=DEFAULT_TIMEOUT_S, metavar="SECONDS",
                   help="how long the whole run may take (default 1800)")
    p.add_argument("-h", "--help", action="store_true", help="show this help")
    return p


def turn_options(row: int, max_new_tokens: int) -> Dict[str, Any]:
    """One turn's options: the probe's published protocol and the page's retrieval counts, on test-split row `row` at `max_new_tokens`, and no
    `model` key, so that the server's default model runs it."""
    return probe.turn_options(test_row=row, max_new_tokens=max_new_tokens)


# ---- one turn, as one word and one error word ----------------------------------------------------------------------------------

def status_word(f: probe.Facts) -> str:
    """The word that says how a turn ended: the status of its message_stop (`done`, `error`, `aborted`, or `unknown` for a word that is none of
    those) when the stream reached one; else `refused` (the server answered with a status other than 200), `timeout`, `failed` or `incomplete`.
    Every word is chosen here: nothing the server sent is passed on."""
    if f.error is not None:
        return "timeout" if f.error == "timeout" else "failed"
    if f.status != 200:
        return "refused" if f.status is not None else "failed"
    return f.final if f.final is not None else "incomplete"


def error_word(f: probe.Facts) -> Optional[str]:
    """What is wrong with a turn, as one word, or None for a turn that is fine: the class of a failure or `timeout`, the kind of a refusal, the
    kind of an error event (the last two are words of the probe's closed lists, `unknown` for any other), `seq_gap` for a done turn whose events
    have a gap."""
    word = f.error or f.refusal or f.stream_error
    if word is None and status_word(f) == "done" and f.seq_contiguous is not True:
        word = "seq_gap"
    return word


# ---- the saves -----------------------------------------------------------------------------------------------------------------

def save_export(client: httpx.Client, session_id: str, fmt: str) -> Tuple[Optional[str], Optional[str]]:
    """Ask the server to save the session's export as `fmt` on its own disk. -> (the file's name, None) when it did, else (None, a fixed code):
    `save_http_<status>`, `save_<class>` for a request that failed, or `save_bad_answer` when the answer is not the name of a file of this session
    in this format, with a size from 1 to below the probe's bound. The export itself is never in the answer, and nothing else of it is kept."""
    try:
        response = client.post(SAVE_PATH.format(session_id), params={"format": fmt})
    except Exception as exc:   # the class, never the text: an httpx error can name a url
        return None, "save_" + type(exc).__name__
    if response.status_code != 200:
        return None, "save_http_{}".format(response.status_code)
    body = drill._json_or_none(response)
    name = body.get("saved") if isinstance(body, dict) else None
    size = body.get("bytes") if isinstance(body, dict) else None
    named = (isinstance(name, str) and SAVED_NAME.fullmatch(name) is not None and name.startswith("session-{}-".format(session_id))
             and name.endswith("." + fmt))
    if not (named and type(size) is int and 0 < size < probe.NUMBER_LIMIT):
        return None, "save_bad_answer"
    return name, None


# ---- the run -------------------------------------------------------------------------------------------------------------------

@dataclass
class Demo:
    """What a run came to: the session, each turn's facts (numbers, booleans and fixed words: R7), the names of the files saved and `error`, the first
    thing that went wrong with the driver itself, a fixed code."""
    session_id: Optional[str] = None
    turns: List[probe.Facts] = field(default_factory=list)
    saved: List[str] = field(default_factory=list)
    error: Optional[str] = None


def run_demo(client: httpx.Client, rows: Sequence[int], max_new_tokens: int, deadline: float,
             clock: Callable[[], float] = time.perf_counter) -> Demo:
    """The session, one turn per row, then the saves: see the module's docstring. It never raises (an exception is named by its class). A turn that
    went wrong does not stop the run; a time limit, Ctrl-C or a bug does, and then nothing is saved. The session is never deleted."""
    demo = Demo()
    try:
        demo.session_id = drill.create_session(client, TITLE)
        for row in rows:
            if clock() >= deadline:
                demo.error = "timeout"
                break
            facts, _ = probe.run_turn(client, demo.session_id, probe.Spec(row, max_new_tokens, turn_options(row, max_new_tokens)), deadline, clock)
            demo.turns.append(facts)
            if facts.error == "timeout":
                demo.error = "timeout"
                break
        if demo.error is None:
            for fmt in SAVE_FORMATS:
                name, code = save_export(client, demo.session_id, fmt)
                if name is not None:
                    demo.saved.append(name)
                elif demo.error is None:
                    demo.error = code
    except drill.DrillError as exc:
        demo.error = str(exc)
    except KeyboardInterrupt:
        demo.error = "interrupted"
    except Exception as exc:   # the class, never the text
        demo.error = "internal_" + type(exc).__name__
    return demo


def is_ok(rows: Sequence[int], demo: Demo) -> bool:
    """Every row ran and ended `done` with no gap in its seq, both exports were saved, and the driver itself had no trouble."""
    return (demo.error is None and demo.session_id is not None and len(demo.turns) == len(rows) and all(probe.turn_ok(f) for f in demo.turns)
            and len(demo.saved) == len(SAVE_FORMATS))


def eos_trained(turns: Sequence[probe.Facts]) -> Optional[bool]:
    """Whether the model that ran the turns was trained to end its reports, as their message_start cards say (the probe reads the flag with `_flag`:
    a bool, or None for anything else). The default model ran them all, so they agree: true or false then; None when no turn says, and when they
    differ, which is not guessed."""
    said = {f.eos_trained for f in turns if f.eos_trained is not None}
    return said.pop() if len(said) == 1 else None


def payload(rows: Sequence[int], demo: Demo) -> Dict[str, Any]:
    """The RESULT line's content: numbers, booleans, fixed words and the names of the files the server wrote. `stopped` is the probe's own word per
    turn (one of its closed list, `unknown` for any other, None for none), and `eos_trained` a boolean or None: nothing else of a turn's stream."""
    ok = is_ok(rows, demo)
    out: Dict[str, Any] = {"demo": "done" if ok else "failed", "rows": list(rows), "turns": len(demo.turns),
                           "statuses": [status_word(f) for f in demo.turns], "tokens": [f.tokens for f in demo.turns],
                           "stopped": [f.stopped for f in demo.turns], "eos_trained": eos_trained(demo.turns), "session": demo.session_id,
                           "saved": list(demo.saved), "ok": ok}
    errors = [error_word(f) for f in demo.turns]
    if any(e is not None for e in errors):
        out["errors"] = errors
    if demo.error is not None:
        out["error"] = demo.error
    return out


def main(argv: Optional[Sequence[str]] = None, stdout: Optional[TextIO] = None, stderr: Optional[TextIO] = None) -> int:
    """Run the demo; -> the exit status. `stdout` and `stderr` are where its two kinds of line go (the process's own by default)."""
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
        if args.rows is None:
            raise drill.Refused(ROWS_MISSING)
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
    deadline = time.perf_counter() + args.timeout
    demo = Demo()
    try:
        # trust_env=False: the connection goes straight to the URL that was given. Left on, httpx sends even loopback requests to the proxy
        # named in HTTP_PROXY or ALL_PROXY, and the token and the stream would pass through it. No redirect is followed, for the same reason.
        with httpx.Client(base_url=base, headers=headers, timeout=drill.CLIENT_TIMEOUT, trust_env=False, follow_redirects=False) as client:
            demo = run_demo(client, args.rows, args.max_new_tokens, deadline)
    except KeyboardInterrupt:
        demo.error = demo.error or "interrupted"
    except Exception as exc:   # the class, never the text
        demo.error = demo.error or "internal_" + type(exc).__name__
    result = payload(args.rows, demo)
    drill._write(stdout, probe.result_line(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
