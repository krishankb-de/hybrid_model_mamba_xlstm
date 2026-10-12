"""CHAT_UI_PLAN.md P6-F fix 1: what the three checks (scripts/chat_ui_layout_check.py, chat_ui_browser_check.py, chat_ui_features_check.py) do with
a --url, as cases the three test files share, so that each of them tests its own main() the same way. Not a test file.

A --url is an app that someone else started. The three checks send it uploads and turns (the browser and features checks) or load its page
(the layout check), and the two that write evidence screenshot what it shows, so each refuses, before it sends anything: the dev server's port
and the tunnel's (scripts/chat_ui_cdp.REFUSED_PORTS), a host that is not this machine's loopback, and an app whose own /v1/models is not the
tiny engine's without a checkpoint. Every refusal is one fixed line and the status 2, and nothing of what the server said is in it.
"""
import contextlib
import copy
import json
import signal
import socket
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Iterator, Optional, Tuple

from scripts import chat_ui_cdp

MARKER = "SECRET-MARKER-8731"   # text a server could send: it must not come back in a refusal


def _live_models(gallery: bool) -> Dict[str, Any]:
    from fastapi.testclient import TestClient
    from app.server import create_app
    with tempfile.TemporaryDirectory() as home:
        with TestClient(create_app(engine="tiny", home=home, tiny_gallery=gallery)) as client:
            return client.get("/v1/models").json()


# The /v1/models answers of the live tiny app, create_app(engine="tiny") without the gallery and with it, read when the test modules are collected
# and not when a test first asks. The app prints its start-up to stdout and makes its temporary home with tempfile; built inside a test, as
# the first test to ask in the files' alphabetical order was (browser check, features check, layout check), the build landed in the output that
# the test reads with capsys, and in the tempfile.mkdtemp that the test had made to fail: four cases failed in the suite and passed alone.
_TINY = {False: _live_models(False), True: _live_models(True)}  # type: Dict[bool, Dict[str, Any]]


def tiny_models(gallery: bool = False) -> Dict[str, Any]:
    """The /v1/models answer of the live tiny app, with or without the synthetic gallery: a copy, to be changed."""
    return copy.deepcopy(_TINY[gallery])


def _real_model(models: Dict[str, Any]) -> Any:
    from app.server import MODELS_EXAMPLE   # what a server that serves a real checkpoint says, as the route documents it
    return copy.deepcopy(MODELS_EXAMPLE)


def _edited(edit: Any) -> Any:
    def apply(models: Dict[str, Any]) -> Any:
        edit(models)
        return models
    return apply


# (why it is not the tiny app, what turns the live tiny answer into the answer of that server): each must be refused.
NOT_TINY = [
    ("a real model", _real_model),
    ("the tiny model with a checkpoint", _edited(lambda m: m["models"][0].update(checkpoint="/sc/home/x/outputs/run/checkpoints/last.ckpt"))),
    ("the tiny model with a checkpoint file name (public mode)", _edited(lambda m: m["models"][0].update(checkpoint="last.ckpt"))),
    ("the tiny model with a checkpoint hash", _edited(lambda m: m["models"][0].update(checkpoint_sha256="ab" * 32))),
    ("the tiny model with a training run", _edited(lambda m: m["models"][0].update(train_experiment="h100_report_gen"))),
    ("a model named tiny that says another thing of itself", _edited(lambda m: m["models"][0].update(drift_note="measured on the cluster"))),
    ("a second model beside the tiny one", _edited(lambda m: m["models"].append(dict(m["models"][0], name="hybrid_150m_m3_rrg")))),
    ("the tiny model listed and another as the default", _edited(lambda m: m.update(default_model="hybrid_150m_m3_rrg"))),
    ("no default model", _edited(lambda m: m.pop("default_model"))),
    ("no models", _edited(lambda m: m.update(models=[]))),
    ("models that are not a list", _edited(lambda m: m.update(models={"tiny": m["models"][0]}))),
    ("a card that is not an object", _edited(lambda m: m.update(models=["tiny"]))),
    ("a card with no checkpoint field", _edited(lambda m: m["models"][0].pop("checkpoint"))),
    ("a card with no checkpoint hash field", _edited(lambda m: m["models"][0].pop("checkpoint_sha256"))),
    ("a card with no training run field", _edited(lambda m: m["models"][0].pop("train_experiment"))),
    ("a card with no note", _edited(lambda m: m["models"][0].pop("drift_note"))),
    ("a card with another name", _edited(lambda m: m["models"][0].update(name="Tiny"))),
    ("an answer that is a list", lambda m: [m["models"][0]]),
    ("an answer that is a string", lambda m: "tiny"),
    ("an answer that is a number", lambda m: 3),
    ("an answer that is null", lambda m: None),
]
NOT_TINY_WHY = [why for why, _ in NOT_TINY]


def not_tiny_answer(why: str) -> Any:
    """The /v1/models answer of the server that `why` describes: the live tiny one, changed (it is built here, not when the tests are collected)."""
    return dict(NOT_TINY)[why](tiny_models(True))


class StubApp:
    """What a loopback server was asked: every request it got, as (method, path)."""

    def __init__(self, models: Any, status: int, body: Optional[bytes], location: Optional[str]) -> None:
        self.models, self.status, self.body, self.location = models, status, body, location
        self.requests = []  # type: List[Tuple[str, str]]
        self.url = ""


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:
        pass

    def _answer(self) -> None:
        stub = self.server.stub  # type: ignore[attr-defined]
        stub.requests.append((self.command, self.path))
        if self.command == "GET" and self.path == "/v1/models":
            if stub.location:
                self.send_response(302)
                self.send_header("Location", stub.location)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            body = stub.body if stub.body is not None else json.dumps(stub.models).encode("utf-8")
            self.send_response(stub.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    do_GET = do_POST = do_DELETE = do_PUT = do_HEAD = _answer


@contextlib.contextmanager
def stub_app(models: Any = None, status: int = 200, body: Optional[bytes] = None, location: Optional[str] = None) -> Iterator[StubApp]:
    """A loopback server on a port of the OS's choosing that answers GET /v1/models with `models` as JSON (or `body` as it is, with `status`, or a
    redirect to `location`), 404 to anything else, and records what it was asked."""
    stub = StubApp(models, status, body, location)
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    server.stub = stub  # type: ignore[attr-defined]
    stub.url = "http://127.0.0.1:{}/".format(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield stub
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@contextlib.contextmanager
def keeping_signals() -> Iterator[None]:
    """A main() that gets as far as its SIGTERM handler leaves it installed: put back what the test found."""
    before = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGALRM)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, before[0])
        signal.signal(signal.SIGALRM, before[1])
        signal.alarm(0)


def _forbid(monkeypatch: Any, module: Any, connections: bool) -> None:
    """Whatever a refused --url must not do is made to fail the test: look for Chrome, start the app or the browser, install the SIGTERM
    handler, send the warm-up turn, make the work directory, and (connections) open any connection at all."""
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the check started or sent something for a refused --url")

    for name in ("find_chrome", "App", "Browser", "exit_on_sigterm", "warm_up", "get_json"):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, forbidden)
    if hasattr(module, "tempfile"):
        monkeypatch.setattr(module.tempfile, "mkdtemp", forbidden)
    if connections:
        monkeypatch.setattr(socket.socket, "connect", forbidden)   # not even a connection: a refused port might be the dev server's


def refuses_a_port(module: Any, monkeypatch: Any, capsys: Any, port: int, args: Tuple[str, ...] = ()) -> None:
    """The dev server's port or the tunnel's, on any spelling of this machine: one line, status 2, nothing started, no connection made."""
    _forbid(monkeypatch, module, connections=True)
    for url in ("http://127.0.0.1:{}/", "http://127.0.0.1:{}", "http://localhost:{}/", "http://[::1]:{}/chat"):
        assert module.main(["--url", url.format(port)] + list(args)) == 2, url
        out = capsys.readouterr().out
        assert out.startswith("ERROR --url refused: ") and str(port) in out and out.count("\n") == 1, out
        assert out == "ERROR --url refused: " + chat_ui_cdp.url_problem(url.format(port)) + "\n"


def refuses_a_host(module: Any, monkeypatch: Any, capsys: Any, args: Tuple[str, ...] = ()) -> None:
    """An address that is not this machine's loopback, or not http, or has no port: one line, status 2, nothing started, no connection made."""
    _forbid(monkeypatch, module, connections=True)
    for url in ("http://example.com:5000/", "http://192.168.1.5:51234/", "http://0.0.0.0:51234/", "https://127.0.0.1:51234/", "http://127.0.0.1/",
                "127.0.0.1:51234", "ftp://127.0.0.1:51234/"):
        assert module.main(["--url", url] + list(args)) == 2, url
        out = capsys.readouterr().out
        assert out == "ERROR --url refused: " + chat_ui_cdp.url_problem(url) + "\n", out


def refuses_a_server_that_is_not_the_tiny_app(module: Any, monkeypatch: Any, capsys: Any, args: Tuple[str, ...] = ()) -> None:
    """A loopback app whose /v1/models is not the tiny engine's without a checkpoint, or cannot be read: one fixed line, status 2, no browser, no
    app, no handler, no warm-up turn, nothing of what the server said in the line, and nothing asked of it but GET /v1/models."""
    _forbid(monkeypatch, module, connections=False)
    for why in NOT_TINY_WHY:
        answer = not_tiny_answer(why)
        if isinstance(answer, dict):
            for card in answer.get("models", []) if isinstance(answer.get("models"), list) else []:
                if isinstance(card, dict):
                    card["marker"] = MARKER   # something to find in the refusal if it were echoed
            answer["marker"] = MARKER
        with stub_app(answer) as stub:
            assert module.main(["--url", stub.url] + list(args)) == 2, why
            out = capsys.readouterr().out
            assert out == "ERROR --url refused: " + chat_ui_cdp.NOT_THE_TINY_APP + "\n", (why, out)
            assert stub.requests == [("GET", "/v1/models")], (why, stub.requests)
    for why, kwargs in (("an error status", {"status": 500, "models": {}}), ("something that is not JSON", {"body": MARKER.encode()}),
                        ("JSON that is cut off", {"body": b'{"default_model": "tiny", "models": [' + MARKER.encode()})):
        with stub_app(**kwargs) as stub:
            assert module.main(["--url", stub.url] + list(args)) == 2, why
            out = capsys.readouterr().out
            assert out == "ERROR --url refused: " + chat_ui_cdp.MODELS_UNREADABLE + "\n", (why, out)
            assert MARKER not in out and stub.requests == [("GET", "/v1/models")], (why, stub.requests)
    with stub_app(tiny_models(True)) as elsewhere, stub_app(location=elsewhere.url + "v1/models") as stub:   # an app that sends the page's reader to another
        assert module.main(["--url", stub.url] + list(args)) == 2
        assert capsys.readouterr().out == "ERROR --url refused: " + chat_ui_cdp.MODELS_UNREADABLE + "\n"
        assert stub.requests == [("GET", "/v1/models")] and elsewhere.requests == [], "the redirect was followed: {}".format(elsewhere.requests)
    unused = chat_ui_cdp.free_port()   # nothing listens there
    assert module.main(["--url", "http://127.0.0.1:{}/".format(unused)] + list(args)) == 2
    assert capsys.readouterr().out == "ERROR --url refused: " + chat_ui_cdp.MODELS_UNREADABLE + "\n"


def goes_on_with_the_tiny_app(module: Any, monkeypatch: Any, capsys: Any, args: Tuple[str, ...] = ()) -> None:
    """The tiny app, with the gallery and without it, is not refused: main() goes on to look for Chrome (there is none, here), and has asked the
    app nothing but GET /v1/models."""
    monkeypatch.setattr(module, "find_chrome", lambda: None)
    for gallery in (False, True):
        with stub_app(tiny_models(gallery)) as stub, keeping_signals():
            assert module.main(["--url", stub.url] + list(args)) == 2
            assert capsys.readouterr().out == "ERROR no Chrome found: set $CHROME to its executable\n"
            assert stub.requests == [("GET", "/v1/models")], stub.requests
