"""CHAT_UI_PLAN.md P5-A: CheXbert-14 names, the labeller client, the rule labeller, the agreement count and the labeller service.

CPU only, offline, synthetic text only (R7). The client runs against a loopback HTTP server started inside the test; the service against a
fake f1chexbert injected into sys.modules, because the real one would download CheXbert."""
import ast
import http.server
import importlib
import json
import logging
import socket
import sys
import threading
import time
import traceback
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from app.labels import CHEXBERT_14, LabelerClient, LabelerUnavailable, RuleLabeler, label_agreement
from tests.app_helpers import start_live_server, wait_until

REPO_ROOT = Path(__file__).resolve().parent.parent
NO_FINDING = [0] * 13 + [1]
MARKER = "SYNTHETIC-7f3a9c"      # stands for report text: it must never come back out in an error message


# ── the plan's tests ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_agreement_counts_and_names_the_differing_positives():
    gen = [0] * 14
    nb = [0] * 14
    gen[CHEXBERT_14.index("Cardiomegaly")] = 1
    nb[CHEXBERT_14.index("Edema")] = 1
    a = label_agreement(gen, nb)
    assert (a["agree"], a["of"]) == (12, 14)
    assert a["generated_only"] == ["Cardiomegaly"] and a["neighbor_only"] == ["Edema"]


def test_rule_labeler_sets_no_finding_only_when_nothing_else_fires():
    rows = RuleLabeler().label(["Findings: lungs clear.", "Small left pleural effusion."])
    assert rows[0][CHEXBERT_14.index("No Finding")] == 1
    assert rows[1][CHEXBERT_14.index("Pleural Effusion")] == 1 and rows[1][CHEXBERT_14.index("No Finding")] == 0


def test_labeler_service_normalises_whitespace_and_reports_names(monkeypatch):
    seen = []

    class FakeF1:
        target_names = CHEXBERT_14

        def get_label(self, text):
            seen.append(text)
            return [0] * 13 + [1]

    monkeypatch.setitem(sys.modules, "f1chexbert", types.SimpleNamespace(F1CheXbert=FakeF1))
    import importlib
    import app.labeler as svc
    importlib.reload(svc)
    from fastapi.testclient import TestClient
    body = TestClient(svc.app).post("/label", json={"texts": ["a  b\nc"]}).json()
    assert body["label_names"] == CHEXBERT_14 and body["labels"] == [[0] * 13 + [1]] and seen == ["a b c"]


# ── names ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_chexbert_14_is_the_repos_published_label_order():
    """scripts/train_report_generation.py's CHEXPERT_14_LABELS is, by its own comment, the single source of truth for the label order;
    chexbert_labels.json's label_names and the per-label tables follow it. Read by AST: the script's heavy imports stay out of here."""
    tree = ast.parse((REPO_ROOT / "scripts" / "train_report_generation.py").read_text())
    published = [ast.literal_eval(n.value) for n in ast.walk(tree)
                 if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "CHEXPERT_14_LABELS" for t in n.targets)]
    assert len(published) == 1
    assert CHEXBERT_14 == published[0]
    assert len(set(CHEXBERT_14)) == 14 and CHEXBERT_14[-1] == "No Finding"


# ── label_agreement ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_agreement_names_the_positives_both_sides_share():
    gen = [0] * 14
    nb = [0] * 14
    for name in ("Edema", "Support Devices"):
        gen[CHEXBERT_14.index(name)] = 1
    for name in ("Edema", "Pleural Effusion"):
        nb[CHEXBERT_14.index(name)] = 1
    a = label_agreement(gen, nb)
    assert a == {"agree": 12, "of": 14, "both_positive": ["Edema"],
                 "neighbor_only": ["Pleural Effusion"], "generated_only": ["Support Devices"]}
    same = label_agreement(NO_FINDING, NO_FINDING)
    assert (same["agree"], same["of"], same["both_positive"]) == (14, 14, ["No Finding"])
    assert same["neighbor_only"] == [] and same["generated_only"] == []


@pytest.mark.parametrize("generated, neighbor", [([0] * 13, [0] * 14), ([0] * 14, [0] * 15), ([], []), ([0] * 14, [])])
def test_agreement_refuses_rows_that_are_not_14_labels_wide(generated, neighbor):
    with pytest.raises(ValueError, match="14 labels on each side"):
        label_agreement(generated, neighbor)


# ── RuleLabeler ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_rule_labeler_rows_are_14_binary_labels_in_chexbert_order():
    labeller = RuleLabeler()
    rows = labeller.label(["Mild CARDIOMEGALY with pulmonary edema.", "No acute process."])
    assert all(len(r) == 14 and set(r) <= {0, 1} for r in rows)
    first = dict(zip(CHEXBERT_14, rows[0]))
    assert first["Cardiomegaly"] == 1 and first["Edema"] == 1 and first["No Finding"] == 0
    assert sum(first.values()) == 2                              # nothing else fired
    assert rows[1] == NO_FINDING
    assert labeller.label([]) == [] and labeller.healthy() is True


def test_every_rule_names_a_chexbert_label():
    """RULES.get(name, ()) would let a misspelt key go on silently labelling nothing."""
    assert set(RuleLabeler.RULES) <= set(CHEXBERT_14)
    assert "No Finding" not in RuleLabeler.RULES                 # derived: set only when no other label fired


# ── LabelerClient, against a loopback server ─────────────────────────────────────────────────────────────────────────────────────────

def as_bytes(obj):
    return json.dumps(obj).encode()


class FakeLabeler:
    """A loopback HTTP server standing in for app.labeler. `reply(method, path, body)` returns (status, payload) or (status, payload,
    declared Content-Length), or raw bytes to be written as they are (not HTTP at all). Every request is recorded in `requests` as
    (method, path, content type, parsed JSON body). `hang` makes it take requests and never answer them."""

    def __init__(self):
        self.requests = []
        self.reply = self._good
        self.hang = False
        self.release = threading.Event()
        fake = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def _serve(self, body):
                fake.requests.append((self.command, self.path, self.headers.get("Content-Type"), body))
                if fake.hang:
                    fake.release.wait(10)
                    return
                out = fake.reply(self.command, self.path, body)
                if isinstance(out, bytes):
                    self.wfile.write(out)
                    return
                status, payload, *declared = out
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(declared[0] if declared else len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                self._serve(None)

            def do_POST(self):
                self._serve(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))

            def log_message(self, *args):                        # keep pytest's output clean
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = "http://127.0.0.1:{}".format(self.server.server_address[1])
        # the default 0.5 s poll is how long shutdown() would wait, once per test
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()

    @staticmethod
    def _good(method, path, body):
        if (method, path) == ("GET", "/healthz"):
            return 200, b'{"status": "ok"}'
        if (method, path) == ("POST", "/label"):
            return 200, as_bytes({"label_names": CHEXBERT_14, "labels": [NO_FINDING] * len(body["texts"])})
        return 404, b"{}"

    def close(self):
        self.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


@pytest.fixture
def fake_labeler():
    fake = FakeLabeler()
    yield fake
    fake.close()


def closed_port_url():
    """A loopback URL nothing listens on: the port is free again as soon as the probe socket closes."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return "http://127.0.0.1:{}".format(probe.getsockname()[1])


def test_client_posts_the_texts_and_returns_the_rows_of_a_good_response(fake_labeler):
    texts = ["Synthetic report one.", "Synthetic report two."]
    rows = LabelerClient(fake_labeler.url + "/").label(texts)              # a trailing slash on the url is dropped
    assert rows == [NO_FINDING, NO_FINDING]
    assert fake_labeler.requests == [("POST", "/label", "application/json", {"texts": texts})]


def test_client_sends_no_request_for_no_texts(fake_labeler):
    assert LabelerClient(fake_labeler.url).label([]) == []
    assert fake_labeler.requests == []


def good_reply(n_rows=2):
    return {"label_names": CHEXBERT_14, "labels": [NO_FINDING] * n_rows}


def with_row(row, n_rows=2):
    return dict(good_reply(n_rows), labels=[NO_FINDING] * (n_rows - 1) + [row])


# Each of these is a reply the client must not hand on as labels. Where the case echoes text, the text is MARKER.
UNTRUSTWORTHY = [
    ("not-json", 200, ("<html>echo " + MARKER + "</html>").encode()),
    ("json-array", 200, as_bytes([MARKER])),
    ("empty-body", 200, b""),
    ("http-500-echoing-the-request", 500, ("echo " + MARKER).encode()),
    ("http-422-echoing-the-request", 422, as_bytes({"detail": [{"loc": ["body"], "input": [MARKER]}]})),
    ("no-names", 200, as_bytes({"labels": [NO_FINDING] * 2})),
    ("names-reordered", 200, as_bytes(dict(good_reply(), label_names=CHEXBERT_14[1::-1] + CHEXBERT_14[2:]))),
    ("names-reversed", 200, as_bytes(dict(good_reply(), label_names=CHEXBERT_14[::-1]))),
    ("names-are-text", 200, as_bytes(dict(good_reply(), label_names=[MARKER] * 14))),
    ("names-not-a-list", 200, as_bytes(dict(good_reply(), label_names=MARKER))),
    ("no-rows", 200, as_bytes({"label_names": CHEXBERT_14})),
    ("rows-not-a-list", 200, as_bytes(dict(good_reply(), labels=MARKER))),
    ("too-few-rows", 200, as_bytes(good_reply(1))),
    ("too-many-rows", 200, as_bytes(good_reply(3))),
    ("row-not-a-list", 200, as_bytes(with_row(MARKER))),
    ("row-too-short", 200, as_bytes(with_row(NO_FINDING[:13]))),
    ("row-too-long", 200, as_bytes(with_row(NO_FINDING + [0]))),
    ("value-2", 200, as_bytes(with_row([2] + [0] * 13))),
    ("value-negative", 200, as_bytes(with_row([-1] + [0] * 13))),
    ("value-text", 200, as_bytes(with_row([MARKER] * 14))),
    ("value-null", 200, as_bytes(with_row([None] + [0] * 13))),
    ("value-float", 200, as_bytes(with_row([1.0] + [0] * 13))),
    ("value-bool", 200, as_bytes(with_row([True] + [0] * 13))),
]


@pytest.mark.parametrize("status, payload", [c[1:] for c in UNTRUSTWORTHY], ids=[c[0] for c in UNTRUSTWORTHY])
def test_client_refuses_a_reply_it_cannot_trust_and_never_echoes_text(fake_labeler, status, payload):
    fake_labeler.reply = lambda method, path, body: (status, payload)
    with pytest.raises(LabelerUnavailable) as err:
        LabelerClient(fake_labeler.url).label(["Synthetic report " + MARKER, "Synthetic second report."])
    message = str(err.value)
    assert MARKER not in message                                      # R7: not the text it sent, nor the text it was sent back
    assert not any(name in message for name in CHEXBERT_14)           # nor the label names
    assert "Synthetic" not in message


def test_client_names_counts_not_values_in_its_shape_errors(fake_labeler):
    def refused(reply):
        fake_labeler.reply = lambda method, path, body: (200, as_bytes(reply))
        with pytest.raises(LabelerUnavailable) as err:
            LabelerClient(fake_labeler.url).label(["Synthetic one.", "Synthetic two."])
        return str(err.value)

    assert refused(dict(good_reply(), label_names=CHEXBERT_14[::-1])) == \
        "label order mismatch: the service reported 14 label names, expected 14"
    assert refused({"labels": []}) == "label order mismatch: the service reported no label names, expected 14"
    assert refused(good_reply(1)) == "the service returned 1 label rows for 2 texts"
    assert refused(dict(good_reply(), labels=None)) == "the service returned no label rows for 2 texts"
    assert refused(with_row(NO_FINDING[:13])) == "label row 1 has 13 entries, expected 14"
    assert refused(with_row("x")) == "label row 1 has no entries, expected 14"
    assert refused(with_row([2] + [0] * 13)) == "label row 1 holds a value that is not 0 or 1"
    assert refused([MARKER]) == "labeller reply is not a JSON object"


def test_client_reports_http_status_and_exception_names_never_messages(fake_labeler):
    fake_labeler.reply = lambda method, path, body: (500, ("echo " + MARKER).encode())
    with pytest.raises(LabelerUnavailable) as err:
        LabelerClient(fake_labeler.url).label(["Synthetic report."])
    assert str(err.value) == "labeller request failed: HTTP 500"
    fake_labeler.reply = lambda method, path, body: (200, b"<html>" + MARKER.encode() + b"</html>")
    with pytest.raises(LabelerUnavailable) as err:
        LabelerClient(fake_labeler.url).label(["Synthetic report."])
    assert str(err.value) == "labeller request failed: JSONDecodeError"


def test_client_a_peers_own_bytes_reach_neither_the_message_nor_the_traceback(fake_labeler):
    """http.client's BadStatusLine carries the first line the peer sent; the chained traceback would print it (`from None`)."""
    fake_labeler.reply = lambda method, path, body: ("echo " + MARKER + "\r\n\r\n").encode()
    with pytest.raises(LabelerUnavailable) as err:
        LabelerClient(fake_labeler.url).label(["Synthetic report."])
    printed = "".join(traceback.format_exception(type(err.value), err.value, err.value.__traceback__))
    assert "BadStatusLine" in str(err.value)
    assert MARKER not in str(err.value) and MARKER not in printed


def test_client_a_reply_that_never_comes_is_unavailable(fake_labeler):
    fake_labeler.hang = True
    with pytest.raises(LabelerUnavailable, match="labeller request failed: (TimeoutError|timeout)"):    # `timeout` before 3.10
        LabelerClient(fake_labeler.url, timeout=0.3).label(["Synthetic report."])
    assert len(wait_until(lambda: fake_labeler.requests)) == 1        # it was asked, and gave up waiting


def test_client_a_refused_connection_is_unavailable():
    with pytest.raises(LabelerUnavailable, match="labeller request failed"):
        LabelerClient(closed_port_url(), timeout=2.0).label(["Synthetic report."])


def test_client_a_reply_cut_off_mid_body_is_unavailable(fake_labeler):
    """http.client.IncompleteRead is an HTTPException, not an OSError: without its own clause it would escape as itself."""
    payload = as_bytes(good_reply())
    fake_labeler.reply = lambda method, path, body: (200, payload[:20], len(payload))
    with pytest.raises(LabelerUnavailable, match="labeller request failed: IncompleteRead"):
        LabelerClient(fake_labeler.url).label(["Synthetic one.", "Synthetic two."])


def test_client_a_reply_that_is_not_http_is_unavailable(fake_labeler):
    fake_labeler.reply = lambda method, path, body: b"this is not http at all\r\n\r\n"
    with pytest.raises(LabelerUnavailable, match="labeller request failed: BadStatusLine"):
        LabelerClient(fake_labeler.url).label(["Synthetic report."])


def test_client_a_malformed_url_is_unavailable_not_a_crash():
    client = LabelerClient("no-scheme-here")
    with pytest.raises(LabelerUnavailable, match="labeller request failed: ValueError"):
        client.label(["Synthetic report."])
    assert client.healthy() is False


def test_healthy_is_true_only_for_a_200_from_healthz(fake_labeler):
    client = LabelerClient(fake_labeler.url)
    assert client.healthy() is True
    assert fake_labeler.requests == [("GET", "/healthz", None, None)]
    fake_labeler.reply = lambda method, path, body: (503, b'{"status": "unavailable"}')
    assert client.healthy() is False
    fake_labeler.reply = lambda method, path, body: (204, b"")      # a success that is not a 200
    assert client.healthy() is False


def test_healthy_is_false_when_nothing_answers_properly(fake_labeler):
    assert LabelerClient(closed_port_url()).healthy() is False
    fake_labeler.reply = lambda method, path, body: b"this is not http at all\r\n\r\n"      # BadStatusLine: an HTTPException
    assert LabelerClient(fake_labeler.url).healthy() is False
    assert LabelerClient("no-scheme-here").healthy() is False                              # ValueError: unknown url type


# ── the service (app/labeler.py), against a fake f1chexbert ──────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _forget_the_service_module():
    """Every test imports app.labeler afresh, and none leaves it (with its fake model) behind in sys.modules."""
    yield
    sys.modules.pop("app.labeler", None)
    if hasattr(sys.modules["app"], "labeler"):
        delattr(sys.modules["app"], "labeler")


def fake_f1(label=NO_FINDING, names=CHEXBERT_14, init=None):
    """A stand-in for f1chexbert.F1CheXbert, and the list of texts it was asked to label. `init` runs when it is constructed."""
    seen = []

    class FakeF1:
        target_names = names

        def __init__(self):
            if init:
                init()

        def get_label(self, text):
            seen.append(text)
            return label

    return FakeF1, seen


def raising(exc):
    def raiser():
        raise exc
    return raiser


@pytest.fixture
def load_service(monkeypatch):
    """load(FakeF1) imports app.labeler afresh with f1chexbert replaced by it (None: f1chexbert cannot be imported at all)."""
    def load(fake_cls):
        monkeypatch.setitem(sys.modules, "f1chexbert",
                            None if fake_cls is None else types.SimpleNamespace(F1CheXbert=fake_cls))
        sys.modules.pop("app.labeler", None)
        svc = importlib.import_module("app.labeler")
        from fastapi.testclient import TestClient
        return svc, TestClient(svc.app)
    return load


def test_service_loads_the_model_on_first_use_not_at_import_and_once(load_service):
    constructed = []
    Fake, seen = fake_f1(init=lambda: constructed.append(1))
    svc, client = load_service(Fake)
    assert constructed == []
    assert client.get("/healthz").json() == {"status": "ok"}
    client.get("/healthz")
    assert client.post("/label", json={"texts": ["a"]}).status_code == 200
    assert constructed == [1]


def test_service_labels_come_back_as_plain_ints(load_service):
    """get_label can hand back numpy ints depending on the f1chexbert version, and json refuses them (job 2525606: a scoring run died
    writing its output); the service coerces."""
    np = pytest.importorskip("numpy")
    Fake, seen = fake_f1(label=[np.int64(v) for v in NO_FINDING])
    svc, client = load_service(Fake)
    r = client.post("/label", json={"texts": ["a", "b"]})
    assert r.status_code == 200
    assert r.json() == {"label_names": CHEXBERT_14, "labels": [NO_FINDING, NO_FINDING]}


def test_service_answers_an_empty_request_with_the_names_and_no_rows(load_service):
    Fake, seen = fake_f1()
    svc, client = load_service(Fake)
    assert client.post("/label", json={"texts": []}).json() == {"label_names": CHEXBERT_14, "labels": []}
    assert seen == []


def test_service_accepts_64_texts_and_refuses_65_without_echoing_them(load_service):
    Fake, seen = fake_f1()
    svc, client = load_service(Fake)
    assert client.post("/label", json={"texts": ["x"] * 64}).status_code == 200
    assert len(seen) == 64
    r = client.post("/label", json={"texts": [MARKER] * 65})
    assert r.status_code == 422
    assert MARKER not in r.text
    assert len(seen) == 64                                           # nothing was labelled for the refused request


def test_service_accepts_20000_characters_and_refuses_20001(load_service):
    Fake, seen = fake_f1()
    svc, client = load_service(Fake)
    assert client.post("/label", json={"texts": ["x" * 20000]}).status_code == 200
    r = client.post("/label", json={"texts": ["ok", MARKER + "x" * (20001 - len(MARKER))]})
    assert r.status_code == 422
    assert r.json() == {"detail": [{"loc": ["body", "texts", 1], "type": "string_too_long"}]}    # where and why, not what
    assert MARKER not in r.text and "xxxx" not in r.text
    assert len(seen) == 1                                            # only the accepted request was labelled


@pytest.mark.parametrize("body", [{"texts": MARKER}, {"texts": [1, 2]}, {"texts": [[MARKER]]}, {"other": [MARKER]}, {}, [MARKER]])
def test_service_malformed_requests_get_a_422_that_echoes_nothing(load_service, body):
    svc, client = load_service(fake_f1()[0])
    r = client.post("/label", json=body)
    assert r.status_code == 422
    assert MARKER not in r.text
    assert all(set(e) == {"loc", "type"} for e in r.json()["detail"])


def test_service_a_body_that_is_not_json_gets_a_422_that_echoes_nothing(load_service):
    svc, client = load_service(fake_f1()[0])
    r = client.post("/label", content=('{"texts": [' + MARKER).encode(), headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    assert MARKER not in r.text


@pytest.mark.parametrize("exc", [RuntimeError("cannot load weights for " + MARKER + " from /secret/dir"),
                                 ImportError("No module named f1chexbert " + MARKER),
                                 OSError(2, "no such file", "/secret/dir/" + MARKER)])
def test_healthz_is_503_with_one_fixed_message_whatever_failed(load_service, exc):
    svc, client = load_service(fake_f1(init=raising(exc))[0])
    health = client.get("/healthz")
    label = client.post("/label", json={"texts": ["a"]})
    assert health.status_code == 503 and label.status_code == 503      # /label is not a 500 either
    assert health.json() == label.json() == {"status": "unavailable", "message": "CheXbert failed to load"}
    assert MARKER not in health.text and "/secret/dir" not in health.text


class Collect(logging.Handler):
    """Attached to the logger itself: caplog listens on the root logger, and a uvicorn.Config built earlier in the session
    (the end-to-end tests below, test_app_api) leaves `uvicorn` not propagating to it."""

    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def test_the_cause_of_a_failed_load_is_logged_for_the_operator_not_returned(load_service):
    """A 503 alone would leave whoever reads the job log with nothing to go on."""
    svc, client = load_service(fake_f1(init=raising(OSError("chexbert.pth is not in the cache")))[0])
    collected = Collect()
    svc.log.addHandler(collected)
    try:
        r = client.get("/healthz")
    finally:
        svc.log.removeHandler(collected)
    assert r.status_code == 503 and "chexbert.pth" not in r.text
    assert collected.messages == ["CheXbert failed to load: OSError: chexbert.pth is not in the cache"]


def test_healthz_is_503_when_f1chexbert_cannot_be_imported_and_the_service_still_imports(load_service):
    svc, client = load_service(None)
    assert client.get("/healthz").status_code == 503


def test_a_failed_load_is_retried_not_remembered(load_service):
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("weights not on disk yet")

    svc, client = load_service(fake_f1(init=flaky)[0])
    assert client.get("/healthz").status_code == 503
    assert client.get("/healthz").status_code == 200
    assert len(attempts) == 2


def test_concurrent_first_calls_load_the_model_once(load_service):
    """/healthz can be polled while the model loads (slow on a CPU); every poll that started its own load would stack another."""
    constructed = []
    svc, client = load_service(fake_f1(init=lambda: (constructed.append(1), time.sleep(0.2)))[0])
    with ThreadPoolExecutor(max_workers=4) as pool:
        models = list(pool.map(lambda _: svc._get(), range(4)))
    assert len(constructed) == 1
    assert all(m is models[0] for m in models)


# ── the two halves together: LabelerClient against app.labeler served by uvicorn ──────────────────────────────────────────────────────

@pytest.fixture
def serve(load_service):
    """serve(FakeF1) runs app.labeler under uvicorn on a free loopback port and returns its url; every server is stopped afterwards."""
    stops = []

    def start(fake_cls):
        svc, _ = load_service(fake_cls)
        url, stop = start_live_server(svc.app)
        stops.append(stop)
        return url

    yield start
    for stop in stops:
        stop()


def test_the_client_accepts_what_the_service_sends_and_meets_its_limits(serve):
    """Neither side's own tests would notice the other drifting: a renamed key, a label name changed, ints sent as something else."""
    Fake, seen = fake_f1()
    client = LabelerClient(serve(Fake), timeout=5.0)
    assert client.healthy() is True
    assert client.label(["Synthetic  report\none.", "Synthetic report two."]) == [NO_FINDING, NO_FINDING]
    assert seen == ["Synthetic report one.", "Synthetic report two."]
    assert len(client.label(["x"] * 64)) == 64
    with pytest.raises(LabelerUnavailable, match="HTTP 422"):
        client.label(["x"] * 65)
    with pytest.raises(LabelerUnavailable, match="HTTP 422"):
        client.label(["x" * 20001])


def test_the_client_sees_a_service_whose_model_will_not_load_as_unavailable(serve):
    Fake, seen = fake_f1(init=raising(RuntimeError("cannot load weights for " + MARKER)))
    client = LabelerClient(serve(Fake), timeout=5.0)
    assert client.healthy() is False
    with pytest.raises(LabelerUnavailable) as err:
        client.label(["Synthetic report."])
    assert str(err.value) == "labeller request failed: HTTP 503"
    assert seen == []
