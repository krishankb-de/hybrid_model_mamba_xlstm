"""CHAT_UI_PLAN.md P3-E: the chat server's OpenAPI reference (/docs), on the tiny engine."""
import json

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from app.imaging import MAX_UPLOAD_BYTES
from app.schemas import Options
from app.server import ERROR_KINDS, create_app
from tests.app_helpers import iter_sse, png_bytes

MESSAGES = "/v1/sessions/{session_id}/messages"   # the streaming turn
# The Options dict test_curl_walkthrough_options_validate pins: what the README's curl walkthrough sends (P8-E).
WALKTHROUGH_OPTIONS = {"decode": "beam", "beam_size": 3, "max_new_tokens": 100, "cached_decode": True, "compile": False,
                       "k_images": 4, "k_reports": 3, "label": True, "reference": None, "display_repair": False}


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(engine="tiny", home=str(tmp_path))) as c:
        yield c


def _operations(spec):
    return [(method.upper(), path, op) for path, ops in spec["paths"].items() for method, op in ops.items()]


def _lead(message):
    """A message up to its first ": ": what this app wrote. Pydantic words the field detail after it, and that wording
    is its own."""
    return message.split(": ")[0]


def _body_schema(spec, path=MESSAGES):
    """The component schema of a route's multipart form."""
    ref = spec["paths"][path]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]["$ref"]
    return spec["components"]["schemas"][ref.rsplit("/", 1)[1]]


def test_openapi_documents_every_v1_route(client):
    spec = client.get("/openapi.json").json()
    routes = {(m.upper(), p) for p, ops in spec["paths"].items() for m in ops}
    for want in [("POST", "/v1/sessions"), ("GET", "/v1/sessions"), ("POST", "/v1/sessions/{session_id}/messages"),
                 ("GET", "/v1/messages/{message_id}"), ("POST", "/v1/messages/{message_id}/cancel")]:
        assert want in routes
    for path, ops in spec["paths"].items():
        for op in ops.values():
            assert op.get("summary"), path


def test_curl_walkthrough_options_validate():
    from app.schemas import Options
    Options(**{"decode": "beam", "beam_size": 3, "max_new_tokens": 100, "cached_decode": True, "compile": False,
               "k_images": 4, "k_reports": 3, "label": True, "reference": None, "display_repair": False})


# ---- beyond the brief: what makes /docs a reference rather than a list of function names ----------------------------

def test_every_route_has_its_own_summary_and_a_plain_description(client):
    for route in client.app.routes:
        if isinstance(route, APIRoute) and route.include_in_schema:
            assert route.summary, route.path   # without one FastAPI derives it from the function name ("Post Message")
    for method, path, op in _operations(client.get("/openapi.json").json()):
        description = op.get("description", "")
        assert description and "ruling" not in description, (method, path)   # no docstring's plan jargon either


def test_the_options_example_is_the_walkthrough_json_and_validates_as_options(client):
    # FastAPI drops Form(openapi_examples=...) from a multi-field form; a field's `examples` reaches its schema.
    example = _body_schema(client.get("/openapi.json").json())["properties"]["options"]["examples"][0]
    assert isinstance(example, str)   # a form field's value is a string
    parsed = json.loads(example)
    options = Options(**parsed)       # the example cannot drift from the schema
    assert (options.decode, options.beam_size, options.max_new_tokens) == ("beam", 3, 100)   # the published protocol
    assert parsed == WALKTHROUGH_OPTIONS and example == json.dumps(WALKTHROUGH_OPTIONS)


def test_form_fields_and_the_client_id_header_are_described(client):
    spec = client.get("/openapi.json").json()
    fields = _body_schema(spec)["properties"]
    for name in ("image", "text", "options"):
        assert fields[name].get("description"), name
    assert all(key in fields["options"]["description"] for key in Options.model_fields)   # every option is named
    headers = [p for p in spec["paths"][MESSAGES]["post"]["parameters"] if p["in"] == "header"]
    assert [p["name"] for p in headers] == ["x-client-id"] and headers[0].get("description")


def test_the_streaming_route_documents_its_event_stream(client):
    op = client.get("/openapi.json").json()["paths"][MESSAGES]["post"]
    for needle in ("text/event-stream", "message_start", "message_stop", "X-Message-Id"):
        assert needle in op["description"], needle
    ok = op["responses"]["200"]
    assert list(ok["content"]) == ["text/event-stream"] and "X-Message-Id" in ok["headers"]   # never JSON
    sid = client.post("/v1/sessions", json={}).json()["id"]   # and the route does what its description says
    r = client.post(MESSAGES.format(session_id=sid), files={"image": ("x.png", png_bytes(), "image/png")},
                    data={"text": "", "options": json.dumps({"max_new_tokens": 16})})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    frames = list(iter_sse([r.text]))
    assert frames[0]["event"] == "message_start" and frames[-1]["event"] == "message_stop"
    assert r.headers["x-message-id"] == frames[0]["data"]["message_id"]


def test_the_api_description_names_the_access_rules_and_every_error_kind(client):
    description = client.get("/openapi.json").json()["info"].get("description", "")
    for needle in ("Authorization: Bearer", "X-Client-Id", "401", "loopback"):
        assert needle in description, needle
    for status, kind in ERROR_KINDS.items():
        assert "{} {}".format(status, kind) in description, status


REFUSED_BY_THE_ROUTE = {   # what each route declares by itself; 401, 403 and the client id's 400 are its `default`
    ("POST", "/v1/sessions"): {422}, ("GET", "/v1/sessions"): {422},
    ("GET", "/v1/sessions/{session_id}"): {404}, ("DELETE", "/v1/sessions/{session_id}"): {404},
    ("POST", MESSAGES): {400, 403, 404, 413, 422, 429, 500},
    ("GET", "/v1/messages/{message_id}"): {404, 422}, ("POST", "/v1/messages/{message_id}/cancel"): {404},
    ("GET", "/v1/sessions/{session_id}/export"): {404, 422},
    ("GET", "/v1/models"): set(), ("GET", "/healthz"): set(),
}


def test_refusals_are_declared_as_the_error_envelope_and_named_in_the_description(client):
    ops = {(method, path): op for method, path, op in _operations(client.get("/openapi.json").json())}
    assert set(ops) == set(REFUSED_BY_THE_ROUTE)   # a new route needs its refusals declared here
    for key, op in ops.items():
        responses = op["responses"]
        declared = {int(status) for status in responses if status != "default" and int(status) >= 400}
        assert declared == REFUSED_BY_THE_ROUTE[key], key
        for status in declared:   # the description names each one, and the example is that status's envelope
            assert str(status) in op["description"], (key, status)
            example = responses[str(status)]["content"]["application/json"]["example"]
            assert example["type"] == "error" and example["error"]["type"] == ERROR_KINDS[status], (key, status)
            assert set(example["error"]) == {"type", "message"} and example["error"]["message"], (key, status)
        if key != ("GET", "/healthz"):   # every /v1 route can be refused by the guard or the client id
            assert responses["default"]["content"]["application/json"]["example"]["type"] == "error", key
            assert "HTTPValidationError" not in json.dumps(responses), key   # FastAPI's 422 is not this API's body


def test_every_refusal_the_server_really_sends_is_documented(client, tmp_path, monkeypatch):
    spec = client.get("/openapi.json").json()
    sid = client.post("/v1/sessions", json={}).json()["id"]
    turn, image = MESSAGES.format(session_id=sid), {"image": ("x.png", png_bytes(), "image/png")}

    def check(r, method, path, status):   # the status is declared for this route, and its example is the reply
        assert r.status_code == status, (method, path, r.text)
        declared = spec["paths"][path][method.lower()]["responses"][str(status)]
        example = declared["content"]["application/json"]["example"]
        assert r.json()["type"] == example["type"] and r.json()["error"]["type"] == example["error"]["type"]
        assert _lead(r.json()["error"]["message"]) == _lead(example["error"]["message"]), (method, path, status)

    for method, path, url, kwargs, status in [
        ("GET", "/v1/sessions/{session_id}", "/v1/sessions/s_missing", {}, 404),
        ("DELETE", "/v1/sessions/{session_id}", "/v1/sessions/s_missing", {}, 404),
        ("GET", "/v1/sessions/{session_id}/export", "/v1/sessions/s_missing/export", {}, 404),
        ("GET", "/v1/messages/{message_id}", "/v1/messages/m_missing", {}, 404),
        ("POST", "/v1/messages/{message_id}/cancel", "/v1/messages/m_missing/cancel", {}, 404),
        ("POST", MESSAGES, "/v1/sessions/s_missing/messages", {"files": image}, 404),
        ("POST", "/v1/sessions", "/v1/sessions", {"json": {"nope": 1}}, 422),
        ("GET", "/v1/sessions", "/v1/sessions?limit=0", {}, 422),
        ("GET", "/v1/messages/{message_id}", "/v1/messages/m_missing?after=-1", {}, 422),
        ("GET", "/v1/sessions/{session_id}/export", "/v1/sessions/{}/export?format=xml".format(sid), {}, 422),
        ("POST", MESSAGES, turn, {"files": image, "data": {"options": "[1, 2]"}}, 400),
        ("POST", MESSAGES, turn, {"files": image, "data": {"options": '{"beam_size": 9}'}}, 422),
        ("POST", MESSAGES, turn, {"files": {"image": ("x.png", bytes(MAX_UPLOAD_BYTES + 1), "image/png")}}, 413),
    ]:
        check(client.request(method, url, **kwargs), method, path, status)
    with monkeypatch.context() as m:   # 429: the queue is full
        m.setattr(client.app.state.worker, "reserve", lambda: False)
        check(client.post(turn, files=image), "POST", MESSAGES, 429)

    def full_disk(*args):
        raise OSError(28, "No space left on device")

    with monkeypatch.context() as m:   # 500: the image cannot be stored
        m.setattr(client.app.state.store, "save_upload", full_disk)
        check(client.post(turn, files=image), "POST", MESSAGES, 500)
    public = {"Authorization": "Bearer t", "X-Client-Id": "a"}
    with TestClient(create_app(engine="tiny", home=str(tmp_path / "public"), mode="public", token="t")) as c:
        psid = c.post("/v1/sessions", json={}, headers=public).json()["id"]
        r = c.post(MESSAGES.format(session_id=psid), data={"options": json.dumps({"test_row": 0})}, headers=public)
        check(r, "POST", MESSAGES, 403)   # a test-split study in public mode


def test_the_default_refusal_is_what_the_guard_and_the_client_id_answer(tmp_path):
    with TestClient(create_app(engine="tiny", home=str(tmp_path / "open"))) as c:
        default = c.get("/openapi.json").json()["paths"]["/v1/models"]["get"]["responses"]["default"]
        off_loopback = c.get("http://10.1.2.3:8000/v1/sessions")   # no token: only loopback is served
    with TestClient(create_app(engine="tiny", home=str(tmp_path / "token"), token="t")) as c:
        no_token = c.get("/v1/sessions")
    with TestClient(create_app(engine="tiny", home=str(tmp_path / "public"), mode="public", token="t")) as c:
        no_client_id = c.get("/v1/sessions", headers={"Authorization": "Bearer t"})
    for status, r in [(401, no_token), (400, no_client_id), (403, off_loopback)]:
        assert r.status_code == status and r.json()["error"]["type"] == ERROR_KINDS[status]
        assert str(status) in default["description"], status
    assert no_token.json() == default["content"]["application/json"]["example"]   # the example is the real 401
