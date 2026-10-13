"""CHAT_UI_PLAN.md P8-C: the load drill, run from the laptop against the chat server through the SSH tunnel.

    LOCAL_PORT=8010 bash app/tunnel/tunnel.sh                  # another terminal: the tunnel, on a port that is not 8000
    CHAT_TOKEN=... venv/bin/python scripts/chat_load_drill.py http://localhost:8010
    venv/bin/python scripts/chat_load_drill.py http://localhost:8010 --token-file ~/chat.token

It opens one session and sends six turns at once on it, each with a picture of its own that the script draws itself. The server takes at most
four turns at a time (its queue cap), so the drill expects four accepted, each streamed to a `message_stop` with status `done` and a `seq` of
1, 2, 3 ... with no gap, and two refused with 429 `overloaded_error` before their stream opens. --turns and --queue-cap change the numbers:
min(turns, queue cap) turns are expected to be accepted and the rest refused. Then it deletes the session, also when something went wrong or
Ctrl-C was pressed: the server hides a deleted session at once and removes its uploaded images (its stored rows stay until the retention
sweep, P8-B).

Output. Stdout is one line, `RESULT {json}`: requested, accepted, refused_429 and refused_429_types (the `error.type` of each refusal), other_status
and failed (the HTTP codes and the exception class names of anything else), done (accepted turns that ended in message_stop with status done),
seq_contiguous (true when every accepted turn's seq is 1, 2, 3 ... with no gap) and seq_contiguous_turns (one true or false per accepted turn),
session_deleted (null when no session was made), wall_s, ok, and error when the drill itself failed. Exit status 0 when the drill met its
expectation (ok: the right number accepted and refused, every refusal the queue's own, every accepted turn done and contiguous, the session
deleted), 1 when it did not, 2 for an argument it refuses, which is one `ERROR ...` line on stderr, before anything is sent.

R7 (DUA). A private server streams MIMIC-derived text: the report, matched reports, study ids and image urls. From a frame this script keeps its
event name, its seq and, for message_stop, one of the three status words, and drops the rest as it reads it. Nothing of a frame, or of any
answer, is printed, logged or written to a file, a failure included: an exception is named by its class, never by its text. The only writes
are the RESULT line, the refusals and the help text, through one function, and what they hold is numbers, booleans and fixed words.

The token. A server in public mode, or bound off loopback, wants `Authorization: Bearer <token>`. It comes from the environment variable
CHAT_TOKEN, or from --token-file (a regular file of mode 0600 that you own, read as the server's own command line reads one, app/cli.py), and
never from an argument, where it would show in `ps` and in the shell history: an argument that could be one is refused, unnamed. --token-file
wins when both are given. The token is sent as a bearer, to the URL that was given and nowhere else (no proxy named in the environment is
used, no redirect is followed), and never printed. It is not sent over plain http to a host that is not loopback (localhost, 127.0.0.0/8,
::1): that is refused, in one line, before anything is sent. https to any host, and http to the tunnel's loopback end, are fine.

The client id. In public mode a client id scopes sessions and counts turns (6 per client in 10 minutes, app/hardening.py). The drill sends one
fixed id, `chat-load-drill`, on every request, as the page sends its own. It starts 6 turns and is given 2 back when they are refused, so a
second drill on a public server within 10 minutes of the first finds only 2 left and is refused by the rate limit (`rate_limited`): wait.

Port 8000 is the local dev server's, and a URL with that port is refused. The tunnel's own port is whatever LOCAL_PORT it was started with.
"""
import argparse
import ipaddress
import json
import os
import re
import stat
import struct
import sys
import threading
import time
import zlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, TextIO, Tuple

import httpx

CLIENT_ID = "chat-load-drill"        # one fixed id, on every request: the server's pattern is visible ASCII, 1 to 128 characters
TOKEN_ENV = "CHAT_TOKEN"
DEV_PORT = 8000                      # the user's local dev server: never contacted
DEFAULT_TURNS, DEFAULT_QUEUE_CAP, DEFAULT_MAX_NEW_TOKENS, DEFAULT_TIMEOUT_S = 6, 4, 60, 900.0
MAX_TURNS, MAX_TIMEOUT_S = 32, 86400.0
MIN_NEW_TOKENS, MAX_NEW_TOKENS = 16, 200   # the server's own bounds for max_new_tokens
MAX_TOKEN_FILE_BYTES = 4096          # as app/cli.py
TOKEN = re.compile(r"[\x21-\x7e]+")  # visible ASCII, no space: it travels in an Authorization header
HOST = re.compile(r"[A-Za-z0-9._:-]+")
SESSION_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")   # an id goes into a path: only a plain one is used
ERROR_TYPE = re.compile(r"[a-z_]{1,40}")
STATUSES = ("done", "error", "aborted")   # the words of message_stop.status; anything else is counted as "unknown"
QUEUE_REFUSAL = "overloaded_error"   # the error type of the queue's 429 (the public-mode rate limit's is rate_limited)
CLIENT_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=60.0, pool=60.0)   # the server pings every 15 s, so a read this long is a dead stream

# What a refusal says is fixed text: it never holds a value that was given (it could be a secret put in the wrong place).
ARGV_REFUSAL = "unrecognized arguments: a token never goes on the command line, set CHAT_TOKEN or use --token-file"
URL_MISSING = "the URL of the server is required, for example http://localhost:8010"
DEV_REFUSAL = "port 8000 is the local dev server: give the tunnel's own port"
URL_REFUSAL = "the URL must be http://host[:port] or https://host[:port], with a port from 1 to 65535 and no login, query or fragment"
CLEARTEXT_REFUSAL = "a token is not sent over plain http to a host that is not loopback: use https, or the tunnel's loopback URL"


class Refused(Exception):
    """An argument the script will not run with. Its text is fixed: it never contains what was given."""


class DrillError(Exception):
    """The drill cannot go on. Its text is a short code (a status code or an exception class), never a part of an answer."""


@dataclass
class Outcome:
    """What one turn came to. Numbers and short fixed words only: nothing of a frame is kept (R7)."""
    status: Optional[int] = None                  # the HTTP status of the POST, None when no answer came
    refusal: Optional[str] = None                 # error.type of a 429
    seqs: List[Optional[int]] = field(default_factory=list)   # the seq of each frame, None where a frame had none
    last_event: Optional[str] = None
    final: Optional[str] = None                   # message_stop's status: done, error, aborted or unknown
    error: Optional[str] = None                   # the class of an exception that ended the turn


def _write(stream: TextIO, line: str) -> None:
    """The one place this script writes: a whole line, in one call."""
    stream.write(line + "\n")
    stream.flush()


# ---- the arguments -------------------------------------------------------------------------------------------------------------

def _whole(low: int, high: int) -> Any:
    def parse(text: str) -> int:
        message = "must be a whole number from {} to {}".format(low, high)
        try:
            value = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(message) from None
        if not low <= value <= high:
            raise argparse.ArgumentTypeError(message)
        return value
    return parse


def _seconds(text: str) -> float:
    message = "must be a number of seconds above 0 and at most {:g}".format(MAX_TIMEOUT_S)
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(message) from None
    if not 0 < value <= MAX_TIMEOUT_S:   # nan fails both
        raise argparse.ArgumentTypeError(message)
    return value


class _Parser(argparse.ArgumentParser):
    """argparse for a script that handles a secret: no abbreviations (`--token abc` must not become `--token-file abc`), and a refusal never
    quotes what it was given (argparse's own do: `invalid int value: 'abc'`, `unrecognized arguments: --token abc`)."""

    def error(self, message: str) -> Any:
        raise Refused(re.sub(r"'.*$", "", message).rstrip(" :"))   # whatever argparse quoted, and what follows, is dropped

    def parse_args(self, args: Optional[Sequence[str]] = None, namespace: Any = None) -> argparse.Namespace:
        parsed, extras = self.parse_known_args(args, namespace)
        if extras:
            raise Refused(ARGV_REFUSAL)
        return parsed


def build_parser() -> argparse.ArgumentParser:
    p = _Parser(prog="chat_load_drill.py", allow_abbrev=False, add_help=False,
                description="Send turns at the chat server all at once; expect the queue's cap accepted and the rest refused with 429.")
    p.add_argument("url", metavar="URL", nargs="?", help="the server, through the tunnel: http://localhost:PORT (not port 8000)")
    p.add_argument("--token-file", metavar="PATH", help="the token: a file of mode 0600 that you own; or set CHAT_TOKEN. Never an argument")
    p.add_argument("--turns", type=_whole(1, MAX_TURNS), default=DEFAULT_TURNS, help="turns sent at once (default 6)")
    p.add_argument("--queue-cap", type=_whole(1, MAX_TURNS), default=DEFAULT_QUEUE_CAP, help="the server's queue cap (default 4)")
    p.add_argument("--max-new-tokens", type=_whole(MIN_NEW_TOKENS, MAX_NEW_TOKENS), default=DEFAULT_MAX_NEW_TOKENS,
                   help="the decoding budget of each turn, 16 to 200 (default 60): long enough that the queue stays full while the last arrive")
    p.add_argument("--timeout", type=_seconds, default=DEFAULT_TIMEOUT_S, metavar="SECONDS",
                   help="how long all the turns together may take (default 900)")
    p.add_argument("-h", "--help", action="store_true", help="show this help")
    return p


def check_url(raw: str) -> str:
    """The server's origin, `scheme://host[:port]`, from what was typed (a missing scheme is http); Refused when it is port 8000 or not plain.

    httpx's own parse decides the port, so a spelling httpx would connect to port 8000 with (08000, +8000, 8_000, full-width digits) is
    refused as 8000. A path is dropped. A login, a query or a fragment is refused: they are where a token would be put."""
    text = raw.strip()
    if "://" not in text:
        text = "http://" + text
    # Every part of the URL that is read is read inside the try: httpx raises InvalidURL for a port that is no number or a character a URL cannot
    # hold, but a byte of the path that is no UTF-8 is a UnicodeEncodeError when the URL is made, and a host name that is none ("http://xn--") an
    # idna.IDNAError (a UnicodeError) when .host is read. Each was a traceback and exit 1, and is one ERROR line and exit 2 (P8 follow-up, from the P8-C review).
    try:
        url = httpx.URL(text)
        port, host, scheme = url.port, url.host, url.scheme
        plain = not (url.userinfo or url.query or url.fragment)
    except (httpx.InvalidURL, UnicodeError):
        raise Refused(URL_REFUSAL) from None
    if port == DEV_PORT:
        raise Refused(DEV_REFUSAL)
    if scheme not in ("http", "https") or HOST.fullmatch(host) is None:
        raise Refused(URL_REFUSAL)
    if port is not None and not 1 <= port <= 65535:
        raise Refused(URL_REFUSAL)
    if not plain:
        raise Refused(URL_REFUSAL)
    shown = "[{}]".format(host) if ":" in host else host
    return "{}://{}{}".format(scheme, shown, "" if port is None else ":{}".format(port))


def is_loopback(host: str) -> bool:
    """localhost, an address in 127.0.0.0/8, or ::1 (an IPv4-mapped one too). This decides where a token may travel readable, so what it cannot
    show to be loopback is not: a name that happens to resolve there, 127.1 and the integer form of 127.0.0.1 are all "not"."""
    if host.lower() == "localhost":
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return (getattr(ip, "ipv4_mapped", None) or ip).is_loopback


def sends_token_in_clear(base: str) -> bool:
    """True when `base` (what check_url returns) is plain http to a host that is not loopback: a bearer token sent there crosses the network
    readable. https to any host, and http to loopback (the tunnel's own end), are not."""
    url = httpx.URL(base)
    return url.scheme == "http" and not is_loopback(url.host)


def read_token(path: str) -> str:
    """The token in `path`, without its surrounding whitespace, as app/cli.py's read_token reads one: a regular file, mode 0600 exactly, the
    user's own, 4096 bytes at most, visible ASCII without spaces. It is opened without blocking and judged by what was opened. Refused, with
    a fixed sentence that names neither the file nor anything in it, when it will not do."""
    try:
        fd = os.open(os.path.expanduser(path), os.O_RDONLY | os.O_NONBLOCK)
    except OSError:
        raise Refused("the token file cannot be read") from None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise Refused("the token file is not a regular file")
        if info.st_uid != os.getuid():
            raise Refused("the token file is not owned by you")
        if stat.S_IMODE(info.st_mode) != 0o600:
            raise Refused("the token file must be mode 0600 (chmod 600)")
        data = os.read(fd, MAX_TOKEN_FILE_BYTES + 1)
    except OSError:
        raise Refused("the token file cannot be read") from None
    finally:
        os.close(fd)
    if len(data) > MAX_TOKEN_FILE_BYTES:
        raise Refused("the token file is too large")
    try:
        text = data.decode("ascii").strip()
    except UnicodeDecodeError:
        raise Refused("the token must be visible ASCII without spaces") from None
    if not text:
        raise Refused("the token file is empty")
    if TOKEN.fullmatch(text) is None:
        raise Refused("the token must be visible ASCII without spaces")
    return text


def find_token(token_file: Optional[str]) -> Optional[str]:
    """The token to send, or None. It comes from --token-file, which wins, or from CHAT_TOKEN (empty is unset), and from nowhere else."""
    if token_file is not None:
        return read_token(token_file)
    value = os.environ.get(TOKEN_ENV, "").strip()
    if not value:
        return None
    if TOKEN.fullmatch(value) is None:
        raise Refused(TOKEN_ENV + " must be visible ASCII without spaces")
    return value


# ---- one turn ------------------------------------------------------------------------------------------------------------------

def synthetic_png(size: int = 256, variant: int = 0) -> bytes:
    """A size x size 8-bit grayscale PNG of a diagonal gradient, drawn here with zlib and nothing else: no MIMIC image, and no file, is ever
    read or sent. `variant` (0 to 31) brightens it by 8 per step, so that each turn of a drill has a picture of its own, as six people's
    would; a drill is about the queue. The same bytes for the same arguments."""
    top = 2 * (size - 1)
    rows = b"".join(b"\x00" + bytes(((x + y) * 255 // top + 8 * variant) % 256 for x in range(size)) for y in range(size))

    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def _shape(text: str) -> Tuple[Optional[int], str]:
    """(seq, status) of one frame's data; (None, "unknown") when it is not a JSON object, and seq None when that is not an integer. The
    parsed frame is dropped on return."""
    try:
        data = json.loads(text)
    except (ValueError, RecursionError):
        return None, "unknown"
    if not isinstance(data, dict):
        return None, "unknown"
    seq, status = data.get("seq"), data.get("status")
    return (seq if type(seq) is int else None), (status if status in STATUSES else "unknown")


def read_stream(chunks: Iterable[str]) -> Tuple[List[Optional[int]], Optional[str], Optional[str]]:
    """A turn's stream, read for its shape only: -> (the seq of every frame in order, the event name of the last frame, the status word of its
    message_stop). A keep-alive comment is no frame. Of each frame only the event name, the seq and the status are looked at, and none of
    the rest is kept (R7)."""
    seqs: List[Optional[int]] = []
    last_event: Optional[str] = None
    final: Optional[str] = None
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
            if not data:
                continue
            seq, status = _shape("\n".join(data))
            seqs.append(seq)
            last_event = event
            if event == "message_stop":
                final = status
    return seqs, last_event, final


def error_type(body: Any) -> str:
    """The `error.type` of a server's error answer (its parsed JSON), when that is a short lower-case word; else "unknown". Nothing else of the
    answer is read."""
    inner = body.get("error") if isinstance(body, dict) else None
    kind = inner.get("type") if isinstance(inner, dict) else None
    return kind if isinstance(kind, str) and ERROR_TYPE.fullmatch(kind) else "unknown"


def _json_or_none(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return None


def _turn(client: httpx.Client, path: str, png: bytes, options: str, gate: threading.Barrier, outcome: Outcome) -> None:
    """One turn, on its own thread, once all of them are ready: fills `outcome`. It never raises."""
    try:
        try:
            gate.wait(30)
        except threading.BrokenBarrierError:
            pass   # a sibling never came: go on, the count of what was accepted says what happened
        with client.stream("POST", path, files={"image": ("drill.png", png, "image/png")},
                           data={"text": "", "options": options}) as response:
            outcome.status = response.status_code
            if response.status_code == 200:
                outcome.seqs, outcome.last_event, outcome.final = read_stream(response.iter_text())
            elif response.status_code == 429:
                response.read()
                outcome.refusal = error_type(_json_or_none(response))
    except Exception as exc:   # the class, never the text: an httpx error can name a url, and nothing here may name a frame
        outcome.error = type(exc).__name__


def run_turns(client: httpx.Client, session_id: str, args: argparse.Namespace, outcomes: List[Outcome]) -> bool:
    """Send args.turns turns on the session at once; their outcomes are appended to `outcomes` before any is sent, so that an interrupt
    still leaves what is known. -> True when args.timeout ran out with a stream still open."""
    path = "/v1/sessions/{}/messages".format(session_id)
    options = json.dumps({"max_new_tokens": args.max_new_tokens})
    gate = threading.Barrier(args.turns)
    outcomes.extend(Outcome() for _ in range(args.turns))
    threads = [threading.Thread(target=_turn, args=(client, path, synthetic_png(variant=n), options, gate, outcome), daemon=True)
               for n, outcome in enumerate(outcomes)]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + args.timeout
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))
    return any(thread.is_alive() for thread in threads)


# ---- the session and the result -------------------------------------------------------------------------------------------------

def create_session(client: httpx.Client, title: Optional[str] = None) -> str:
    """-> the new session's id. DrillError, with a code that holds no part of the answer, when there is none. `title` (P9-C: the demo's session is
    named) is the session's title; without one the body is empty and the server names the session by its first turn, as it always did."""
    try:
        response = client.post("/v1/sessions", json={} if title is None else {"title": title})
    except httpx.HTTPError as exc:
        raise DrillError("session_create_" + type(exc).__name__) from None
    if response.status_code != 200:
        raise DrillError("session_create_http_{}".format(response.status_code))
    body = _json_or_none(response)
    session_id = body.get("id") if isinstance(body, dict) else None
    if not isinstance(session_id, str) or SESSION_ID.fullmatch(session_id) is None:
        raise DrillError("session_create_bad_answer")
    return session_id


def delete_session(client: httpx.Client, session_id: str) -> bool:
    """True when the server answered 204. Whatever else happens, False: the session may still be there."""
    try:
        return client.delete("/v1/sessions/{}".format(session_id)).status_code == 204
    except Exception:
        return False


def result_payload(outcomes: Sequence[Outcome], turns: int, queue_cap: int, session_deleted: Optional[bool], error: Optional[str],
                   wall_s: float) -> Dict[str, Any]:
    """The RESULT line's content, from the turns' outcomes. `ok` holds when min(turns, queue_cap) turns were accepted and the rest were
    refused by the queue (429 overloaded_error), every accepted turn ended done with a seq of 1, 2, 3 ... and no gap, the session was deleted,
    and nothing else went wrong."""
    accepted = [o for o in outcomes if o.status == 200]
    refused = [o for o in outcomes if o.status == 429]
    types = Counter(o.refusal or "unknown" for o in refused)
    other = Counter(str(o.status) for o in outcomes if o.status not in (None, 200, 429))
    failed = Counter(o.error for o in outcomes if o.error)
    done = sum(1 for o in accepted if o.last_event == "message_stop" and o.final == "done")
    contiguous = [bool(o.seqs) and o.seqs == list(range(1, len(o.seqs) + 1)) for o in accepted]
    expected = min(turns, queue_cap)
    ok = (error is None and session_deleted is True and len(outcomes) == turns and len(accepted) == expected
          and len(refused) == turns - expected and set(types) <= {QUEUE_REFUSAL} and done == len(accepted) and all(contiguous))
    payload: Dict[str, Any] = {
        "requested": turns, "accepted": len(accepted), "refused_429": len(refused), "refused_429_types": dict(sorted(types.items())),
        "other_status": dict(sorted(other.items())), "failed": dict(sorted(failed.items())), "done": done,
        "seq_contiguous": bool(contiguous) and all(contiguous), "seq_contiguous_turns": contiguous,
        "session_deleted": session_deleted, "wall_s": round(wall_s, 2), "ok": ok}
    if error is not None:
        payload["error"] = error
    return payload


def main(argv: Optional[Sequence[str]] = None, stdout: Optional[TextIO] = None, stderr: Optional[TextIO] = None) -> int:
    """Run the drill; -> the exit status. `stdout` and `stderr` are where its two kinds of line go (the process's own by default)."""
    stdout = sys.stdout if stdout is None else stdout
    stderr = sys.stderr if stderr is None else stderr
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.help:
            _write(stdout, parser.format_help().rstrip("\n"))
            return 0
        if args.url is None:
            raise Refused(URL_MISSING)
        base, token = check_url(args.url), find_token(args.token_file)
        if token is not None and sends_token_in_clear(base):   # before there is a client to send it with
            raise Refused(CLEARTEXT_REFUSAL)
    except Refused as exc:
        _write(stderr, "ERROR " + str(exc))
        return 2
    headers = {"X-Client-Id": CLIENT_ID}
    if token is not None:
        headers["Authorization"] = "Bearer " + token
    outcomes: List[Outcome] = []
    error: Optional[str] = None
    deleted: Optional[bool] = None
    started = time.monotonic()
    try:
        # trust_env=False: the connection goes straight to the URL that was given. Left on, httpx sends even loopback requests to the proxy
        # named in HTTP_PROXY or ALL_PROXY, and the token and the stream would pass through it. No redirect is followed, for the same reason.
        with httpx.Client(base_url=base, headers=headers, timeout=CLIENT_TIMEOUT, trust_env=False, follow_redirects=False) as client:
            session_id = None
            try:
                session_id = create_session(client)
                if run_turns(client, session_id, args, outcomes):
                    error = "timeout"
            finally:   # the session is deleted whatever happened above: a failure, a timeout, Ctrl-C
                if session_id is not None:
                    deleted = delete_session(client, session_id)
    except KeyboardInterrupt:
        error = "interrupted"
    except DrillError as exc:
        error = str(exc)
    except Exception as exc:   # the class, never the text
        error = "internal_" + type(exc).__name__
    payload = result_payload(outcomes, args.turns, args.queue_cap, deleted, error, time.monotonic() - started)
    _write(stdout, "RESULT " + json.dumps(payload, separators=(",", ":")))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
