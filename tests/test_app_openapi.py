"""CHAT_UI_PLAN.md P3-E: the chat server's OpenAPI reference (/docs), on the tiny engine."""
import json
import re
from typing import Annotated

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from app import server
from app.commands import COMMAND_HELP
from app.imaging import MAX_UPLOAD_BYTES
from app.schemas import Options
from app.server import ERROR_KINDS, create_app
from app.store import FINAL_STATUSES
from tests.app_helpers import iter_sse, png_bytes

MESSAGES = "/v1/sessions/{session_id}/messages"   # the streaming turn
STREAM_EVENTS = ["message_start", "stage_start", "stage_end", "content_block_start", "content_block_delta",
                 "content_block_stop", "warning", "error", "message_stop"]   # every event name of the stream
# The Options dict test_curl_walkthrough_options_validate pins: what the README's curl walkthrough sends (P8-E).
WALKTHROUGH_OPTIONS = {"decode": "beam", "beam_size": 3, "max_new_tokens": 100, "cached_decode": True, "compile": False,
                       "k_images": 4, "k_reports": 3, "label": True, "reference": None, "display_repair": False,
                       "stop_on_repeat": False}


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


def test_the_models_route_documents_its_features_and_its_example_has_the_real_keys(client):
    op = client.get("/openapi.json").json()["paths"]["/v1/models"]["get"]
    assert "`features`" in op["description"] and "retrieval" in op["description"] and "labels" in op["description"]
    example = op["responses"]["200"]["content"]["application/json"]["example"]
    live = client.get("/v1/models").json()
    assert set(example) == set(live)                                  # the example cannot drift from the answer
    assert set(example["features"]) == set(live["features"]) == {"retrieval", "labels"}
    assert set(example["models"][0]) <= set(live["models"][0])        # and the card it shows is made of real card fields


def test_curl_walkthrough_options_validate():
    from app.schemas import Options
    Options(**{"decode": "beam", "beam_size": 3, "max_new_tokens": 100, "cached_decode": True, "compile": False,
               "k_images": 4, "k_reports": 3, "label": True, "reference": None, "display_repair": False,
               "stop_on_repeat": False})


# ---- beyond the brief: what makes /docs a reference rather than a list of function names ----------------------------

def test_every_route_has_its_own_summary_and_a_plain_description(client):
    spec = client.get("/openapi.json").json()
    for route in client.app.routes:
        if isinstance(route, APIRoute) and route.include_in_schema:
            assert route.summary, route.path   # without one FastAPI derives it from the function name ("Post Message")
    for method, path, op in _operations(spec):
        description = op.get("description", "")
        assert description and len(re.split(r"(?<=[.!?])\s+", description.strip())) <= 2, (method, path)   # 1 or 2
    text = json.dumps(spec)   # and no jargon of the plan, whose files a reader of /docs cannot open
    assert "ruling" not in text and "CHAT_UI_PLAN" not in text


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


def test_the_options_description_says_what_stop_on_repeat_does_and_that_leaving_it_out_decodes_the_whole_budget(client):   # P4-G
    text = _body_schema(client.get("/openapi.json").json())["properties"]["options"]["description"]
    about = text.split("`stop_on_repeat`", 1)[1].split(".")[0]   # its own sentence
    assert "first sentence" in about and "word for word" in about, about   # it stops at the first verbatim repeat, not at a likeness
    assert "`stopped: repeat`" in about, about                              # and the generate stage says so
    assert "`truncated_mid_sentence` is false" in about and "raw `report`" in about, about   # a repeat stop is no cut-off, whatever the raw text ends in
    assert "max_new_tokens" in about and "published protocol" in about, about   # left out, the whole budget is decoded


def test_the_streaming_route_documents_its_event_stream(client, monkeypatch):
    op = client.get("/openapi.json").json()["paths"][MESSAGES]["post"]
    for needle in ("text/event-stream", "message_start", "message_stop", "X-Message-Id"):
        assert needle in op["description"], needle
    ok = op["responses"]["200"]
    assert list(ok["content"]) == ["text/event-stream"] and "X-Message-Id" in ok["headers"]   # never JSON
    for word in STREAM_EVENTS + list(FINAL_STATUSES):   # the 200 names every event and every status message_stop has
        assert "`{}`".format(word) in ok["description"], word
    assert "`error` event followed by a `message_stop` with status `error`" in ok["description"]   # after it opened
    sid = client.post("/v1/sessions", json={}).json()["id"]   # and the stream really is what the 200 describes
    turn, options = MESSAGES.format(session_id=sid), json.dumps({"max_new_tokens": 16})
    image = {"image": ("x.png", png_bytes(), "image/png")}
    seen, statuses = set(), set()

    def run(**kwargs):
        r = client.post(turn, **kwargs)
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        frames = list(iter_sse([r.text]))
        assert frames[0]["event"] == "message_start" and frames[-1]["event"] == "message_stop"
        assert r.headers["x-message-id"] == frames[0]["data"]["message_id"]
        seen.update(f["event"] for f in frames)
        statuses.add(frames[-1]["data"]["status"])
        return frames

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    run(files=image, data={"options": options})                    # a turn
    run(data={"text": "is this pneumonia?", "options": options})   # a question: a warning and no model
    monkeypatch.setattr(client.app.state.engines["tiny"], "generate", boom)
    failed = run(files=image, data={"options": options})           # a failure once the 200 has opened
    assert [f["event"] for f in failed[-2:]] == ["error", "message_stop"] and failed[-1]["data"]["status"] == "error"
    assert seen == set(STREAM_EVENTS) and statuses == {"done", "error"}


def test_the_text_and_image_docs_match_what_a_text_only_turn_really_does(client):
    fields = _body_schema(client.get("/openapi.json").json())["properties"]
    text_doc, image_doc = fields["text"]["description"], fields["image"]["description"]
    sid = client.post("/v1/sessions", json={}).json()["id"]
    turn, options = MESSAGES.format(session_id=sid), json.dumps({"max_new_tokens": 16})
    for text in ("beam 2", "is this pneumonia?", ""):   # no image in this session yet: whatever the text, a 422
        refused = client.post(turn, data={"text": text, "options": options})
        assert refused.status_code == 422 and refused.json()["error"]["message"] == "Attach an X-ray first.", text
    assert "422" in text_doc and "no image yet" in text_doc and "no image yet" in image_doc
    image = {"image": ("x.png", png_bytes(), "image/png")}
    client.post(turn, files=image, data={"options": options})   # and now it has one
    question =list(iter_sse([client.post(turn, data={"text": "is this pneumonia?", "options": options}).text]))
    assert [f["event"] for f in question] == ["message_start", "warning", "message_stop"]   # the command list, no model
    assert question[1]["data"]["code"] == "not_a_command" and COMMAND_HELP in question[1]["data"]["message"]
    assert COMMAND_HELP in text_doc and "command list" in text_doc and "no model runs" in text_doc
    for text in ("beam 2", ""):   # a command, or no text at all, runs the session's latest image again
        rerun = list(iter_sse([client.post(turn, data={"text": text, "options": options}).text]))
        assert rerun[0]["data"]["image"]["source"] == "previous", text
        assert "generate" in [f["data"]["stage"] for f in rerun if f["event"] == "stage_end"], text


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
        assert responses["default"]["content"]["application/json"]["example"]["type"] == "error", key   # every route
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


ROUTE_BODIES = {("POST", "/v1/sessions"): {"json": {}}, ("POST", MESSAGES): {"data": {"options": "{}"}}}   # to parse


def test_each_routes_default_names_exactly_the_guard_refusals_it_really_gets(tmp_path):
    # Which of 401 (token), 400 (client id) and 403 (loopback) a route can answer differs: /v1/models takes no client id
    # and /healthz no token. So every route is asked in each of the three settings that refuse before a route runs.
    def ask_all(c, base="", headers=None):
        urls = {key: base + key[1].replace("{session_id}", "s_x").replace("{message_id}", "m_x")
                for key in REFUSED_BY_THE_ROUTE}
        return {key: c.request(key[0], url, headers=headers, **ROUTE_BODIES.get(key, {})) for key, url in urls.items()}

    with TestClient(create_app(engine="tiny", home=str(tmp_path / "open"))) as c:
        spec = c.get("/openapi.json").json()
        replies = {403: ask_all(c, "http://10.1.2.3:8000")}   # no token: only loopback is served
    with TestClient(create_app(engine="tiny", home=str(tmp_path / "token"), token="t")) as c:
        replies[401] = ask_all(c)   # a token, and none sent
    with TestClient(create_app(engine="tiny", home=str(tmp_path / "public"), mode="public", token="t")) as c:
        replies[400] = ask_all(c, headers={"Authorization": "Bearer t"})   # public mode, and no client id
    for key in REFUSED_BY_THE_ROUTE:
        default = spec["paths"][key[1]][key[0].lower()]["responses"]["default"]
        given = []
        for status, answers in replies.items():
            really = answers[key].status_code == status
            assert really == (str(status) in default["description"]), (key, status)   # named if and only if it is given
            if really:
                assert answers[key].json()["error"]["type"] == ERROR_KINDS[status], (key, status)
                given.append(answers[key].json())
        assert default["content"]["application/json"]["example"] in given, key   # and the example is one of its replies


def test_option_ranges_cope_with_a_field_that_has_only_one_bound(monkeypatch):
    class OneBound(BaseModel):
        both: Annotated[int, Field(ge=1, le=5)] = 2
        most: Annotated[int, Field(le=7)] = 3   # no minimum: this was a KeyError at import
        least: Annotated[int, Field(ge=4)] = 4
        free: int = 0

    monkeypatch.setattr(server, "Options", OneBound)
    assert server._option_ranges() == "both 1-5, most up to 7, least from 4"
