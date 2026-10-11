"""Request hardening for the chat app (CHAT_UI_PLAN.md P8-A): request ids, the request log and the public-mode turn rate limit.

Pure ASGI middlewares, wired by app/server.py. From the outside in: RequestLog, then the CORS and token guard of server.py, then
TurnRateLimit (public mode only), then the routes.

**Request ids.** Every response carries X-Request-Id, errors included. A caller's own id is kept when it is 1 to 64 characters of
letters, digits, "." "_" and "-", and anything else is replaced by a new one (a uuid4 in hex), so a header can never put a newline or
markup into a log. The id is also in request.state.request_id, which the error handlers read.

**The request log.** One JSON line per request, written when the response has ended (`ts` is that moment, and `ms` of a streamed turn is the
whole stream):

    {"ts": ..., "request_id": ..., "method": ..., "path": ..., "status": ..., "ms": ..., "client": ...}

`path` is the path without its query string: a query can hold a test-split study id (the picker's `q`), and the server log, which lives on the
cluster, must not carry one (R1, DUA). `client` is the X-Client-Id when it is well formed, else "-". No other header is logged, and no body,
token or Authorization value, ever. `status` is the one the client got; 499 is a request that ended before any response (the client went away).
The lines go through the `app.request` logger to stderr and never to stdout, which P7-B's job log keeps for its `[server]`, `RESULT` and
`ERROR` lines. uvicorn's own access log (it prints the query) is off in app/cli.py.

**The turn rate limit.** In public mode a client (its X-Client-Id) may start 6 turns per 10 minutes: a token bucket of capacity 6 that refills
6 per 600 s. Only turn creation, POST /v1/sessions/{id}/messages, is limited, and the check runs before the body is parsed, so a refused request
costs one dictionary lookup: 429 `rate_limited` with Retry-After in whole seconds. A token is taken at the check and given back unless the turn is
accepted (a 200), so a refusal of any other kind costs the client nothing, and so does a note that runs no model (P8 follow-up): text with no
image that is no command is answered with a warning and a 200, and the route, which alone knows that from the body, marks the request
(FREE_TURN in its state) so that the token is given back at the end of the response. The check comes first, before the body is read, so a
client with no token left is refused a note too, until one is back. Private mode has no limit.

The limit is by client id, and a client id is whatever the caller sends: **a client that mints a new X-Client-Id for every turn escapes the
limit.** It is a courtesy between honest visitors of a demo, not a defence. What bounds the load is the queue cap (4 accepted turns at once,
app/server.py), which holds whoever asks, and the token that every /v1 route needs in public mode.
"""
import json
import logging
import math
import re
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Tuple

from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse

from app.schemas import error_body

REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")   # fullmatch: "$" would let a trailing newline through
REQUEST_ID_HEADER = "X-Request-Id"
REQUEST_LOG = logging.getLogger("app.request")
MAX_LOGGED_PATH = 256     # a path is the caller's text: one line of the log must stay short
MAX_LOGGED_METHOD = 16
NO_RESPONSE = 499         # the status logged for a request that ended before any response started (nginx's convention)
RATE_CAPACITY = 6         # turns a public-mode client can start at once ...
RATE_WINDOW_S = 600.0     # ... and in every 10 minutes: the bucket refills RATE_CAPACITY tokens per RATE_WINDOW_S
RATE_LIMITED = "rate_limited"   # the error type of its 429 (the queue's own 429 is "overloaded_error")
FREE_TURN = "free_turn"   # the key of the request's state (request.state.free_turn) that a route sets for a turn that ran no model: its token is given back
RATE_LIMITED_MSG = "Too many turns from this client: at most {} per {} minutes. Try again in {} s."   # capacity, window in minutes, seconds
TURN_PATH = re.compile(r"/v1/sessions/[^/]+/messages")   # the one route that is limited: POST creates a turn
MAX_TRACKED = 10_000      # clients in the limiter's table before it forgets those whose bucket has refilled


def new_request_id() -> str:
    return uuid.uuid4().hex


class _StderrHandler(logging.Handler):
    """The request log's one handler. It writes to whatever sys.stderr is when the line is emitted, and a whole line in one call (as
    app/cli.py's emit does), so a line is neither lost with a stream that was swapped nor split by another writer."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            sys.stderr.write(self.format(record) + "\n")
            sys.stderr.flush()
        except Exception:
            self.handleError(record)


def configure_request_log() -> logging.Logger:
    """Point the request log at stderr, once: its own handler, no propagation (a root handler would write each line a second time) and not
    disabled (a dictConfig elsewhere in the process may have done that). Safe to call again, and create_app does."""
    REQUEST_LOG.setLevel(logging.INFO)
    REQUEST_LOG.propagate = False
    REQUEST_LOG.disabled = False
    if not any(isinstance(h, _StderrHandler) for h in REQUEST_LOG.handlers):
        handler = _StderrHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        REQUEST_LOG.addHandler(handler)
    return REQUEST_LOG


class RequestLog:
    """Outermost middleware: gives the request its id, puts it on the response (every http.response.start, whoever sent it) and logs the
    request's line once it has ended. `client_id` is the pattern an X-Client-Id must fully match to be logged (server.CLIENT_ID)."""

    def __init__(self, app: Callable, client_id: "re.Pattern"):
        self.app, self.client_id = app, client_id

    async def __call__(self, scope: Dict[str, Any], receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = time.perf_counter()
        headers = Headers(scope=scope)
        given = headers.get(REQUEST_ID_HEADER, "")
        request_id = given if REQUEST_ID.fullmatch(given) else new_request_id()
        scope.setdefault("state", {})["request_id"] = request_id
        client = headers.get("x-client-id", "")
        status = 0

        async def send_with_id(message: Dict[str, Any]) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message.setdefault("headers", [])
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        except Exception:
            status = status or 500   # no response yet: the server's error middleware answers 500
            raise
        except BaseException:        # cancelled: the client went away
            status = status or NO_RESPONSE
            raise
        else:
            status = status or 500   # returned without a response: the server answers that with a 500 too
        finally:
            REQUEST_LOG.info(request_line(request_id, scope.get("method", ""), scope.get("path", ""), status,
                                          (time.perf_counter() - started) * 1000.0,
                                          client if self.client_id.fullmatch(client) else "-"))


def request_line(request_id: str, method: str, path: str, status: int, ms: float, client: str) -> str:
    """The log's line: the seven fields in a fixed order, compact JSON, ASCII only (so no character of a path can start another line)."""
    return json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "request_id": request_id,
                       "method": method[:MAX_LOGGED_METHOD], "path": path[:MAX_LOGGED_PATH], "status": status, "ms": round(ms, 1),
                       "client": client}, separators=(",", ":"))


# ---- the turn rate limit ------------------------------------------------------------------------------------------------------------

class TurnLimiter:
    """A token bucket per client: `capacity` tokens, refilled `capacity` per `window_s` (so one token per window_s / capacity), none beyond the
    capacity. take() takes a token or says how long until there is one; refund() gives one back. A client whose bucket is full is not in
    the table, which is also how a client never seen starts. `clock` is a monotonic clock in seconds (a test passes its own)."""

    def __init__(self, capacity: int = RATE_CAPACITY, window_s: float = RATE_WINDOW_S, clock: Callable[[], float] = time.monotonic,
                 max_tracked: int = MAX_TRACKED):
        if capacity < 1 or window_s <= 0:
            raise ValueError("a rate limit needs a capacity of at least 1 and a window above 0 s, got {} per {} s".format(capacity, window_s))
        self.capacity, self.window_s, self.clock, self.max_tracked = capacity, float(window_s), clock, max_tracked
        self._buckets: Dict[str, Tuple[float, float]] = {}   # client -> (tokens, the time they were counted at)
        self._lock = threading.Lock()

    def _refilled(self, client: str, now: float) -> float:
        tokens, counted = self._buckets.get(client, (float(self.capacity), now))
        return min(float(self.capacity), tokens + (now - counted) * self.capacity / self.window_s)

    def take(self, client: str) -> float:
        """0.0 when a token was taken; else the seconds (above 0) until the next token is there, and nothing was taken."""
        with self._lock:
            now = self.clock()
            tokens = self._refilled(client, now)
            if tokens >= 1.0:
                self._buckets[client] = (tokens - 1.0, now)
                if len(self._buckets) > self.max_tracked:
                    self._forget_refilled(now)
                return 0.0
            self._buckets[client] = (tokens, now)
            return (1.0 - tokens) * self.window_s / self.capacity

    def refund(self, client: str) -> None:
        """Give back a token taken for a turn that was not accepted. A client that is not in the table is at capacity already."""
        with self._lock:
            if client not in self._buckets:
                return
            now = self.clock()
            tokens = min(float(self.capacity), self._refilled(client, now) + 1.0)
            if tokens >= self.capacity:
                del self._buckets[client]
            else:
                self._buckets[client] = (tokens, now)

    def tracked(self) -> int:
        with self._lock:
            return len(self._buckets)

    def _forget_refilled(self, now: float) -> None:
        """Drop every client whose bucket is back at capacity (the lock is held): minted ids cannot grow the table for ever."""
        for client in [c for c in self._buckets if self._refilled(c, now) >= self.capacity]:
            del self._buckets[client]


class TurnRateLimit:
    """Public mode only: limits POST /v1/sessions/{id}/messages per X-Client-Id with a TurnLimiter, before the body is parsed. It sits inside the
    token guard, so a request that has no token never reaches it. A request with no well-formed client id is passed on: the route refuses it
    (400) as it always has. The token is taken at the check and given back unless the response is a 200, so only an accepted turn costs one, and
    a burst started together is still admitted `capacity` at a time. A 200 that the route marked FREE_TURN (a note that ran no model) gives its
    token back as well, once its response has ended."""

    def __init__(self, app: Callable, limiter: TurnLimiter, client_id: "re.Pattern"):
        self.app, self.limiter, self.client_id = app, limiter, client_id

    async def __call__(self, scope: Dict[str, Any], receive: Callable, send: Callable) -> None:
        if scope["type"] != "http" or scope["method"] != "POST" or TURN_PATH.fullmatch(scope["path"]) is None:
            await self.app(scope, receive, send)
            return
        client = Headers(scope=scope).get("x-client-id", "")
        if self.client_id.fullmatch(client) is None:
            await self.app(scope, receive, send)
            return
        wait = self.limiter.take(client)
        if wait > 0:
            seconds = max(1, math.ceil(wait - 1e-9))
            refusal = JSONResponse(error_body(RATE_LIMITED, RATE_LIMITED_MSG.format(
                self.limiter.capacity, int(round(self.limiter.window_s / 60.0)), seconds)),
                status_code=429, headers={"Retry-After": str(seconds)})
            await refusal(scope, receive, send)
            return
        accepted = False

        async def send_watching(message: Dict[str, Any]) -> None:
            nonlocal accepted
            if message["type"] == "http.response.start":
                accepted = message["status"] == 200
            await send(message)

        try:
            await self.app(scope, receive, send_watching)
        finally:
            free = bool((scope.get("state") or {}).get(FREE_TURN))   # the route says so after it has read the body: request.state is this dict
            if free or not accepted:
                self.limiter.refund(client)
