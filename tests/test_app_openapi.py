"""CHAT_UI_PLAN.md P3-E: the chat server's OpenAPI reference (/docs), on the tiny engine."""
import json
import re
import shlex
import subprocess
from pathlib import Path
from typing import Annotated

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from app import hardening, server
from app.commands import COMMAND_HELP
from app.gallery import Gallery
from app.imaging import MAX_UPLOAD_BYTES
from app.schemas import Options
from app.server import ERROR_KINDS, create_app
from app.store import FINAL_STATUSES
from tests.app_helpers import decide_gate, iter_sse, png_bytes

MESSAGES = "/v1/sessions/{session_id}/messages"   # the streaming turn
IMAGE = "/v1/messages/{message_id}/image"          # a turn's image (P6-B)
GALLERY_IMAGE = "/v1/gallery/images/{row}"         # a similar X-ray's image (P6-C)
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
                 ("GET", "/v1/messages/{message_id}"), ("POST", "/v1/messages/{message_id}/cancel"),
                 ("POST", "/v1/retrieve"), ("POST", "/v1/label"), ("GET", "/v1/test-studies"),   # P5-E
                 ("GET", "/v1/messages/{message_id}/image"),   # P6-B
                 ("GET", "/v1/gallery/images/{row}")]:         # P6-C
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
    assert "`eos_trained`" in op["description"] and "stopped: eos" in op["description"]   # P9-G2: the field and what it changes
    assert example["models"][0]["eos_trained"] is False and live["models"][0]["eos_trained"] is False


def test_the_health_route_documents_its_answer_and_its_example_has_the_real_keys(client):
    op = client.get("/openapi.json").json()["paths"]["/healthz"]["get"]
    assert "`code_version`" in op["description"] and "`started_at`" in op["description"]   # P4-H A2: what tells a stale server
    assert "`retention_days`" in op["description"] and "RETENTION_DAYS" in op["description"]   # P8-B: how long a session is kept
    example = op["responses"]["200"]["content"]["application/json"]["example"]
    live = client.get("/healthz").json()
    assert set(example) == set(live)                                  # the example cannot drift from the answer
    assert re.fullmatch(r"[0-9a-f]{7}", example["code_version"]) and example["started_at"].endswith("+00:00")
    assert example["retention_days"] == live["retention_days"] == 45 and isinstance(example["retention_days"], int)


def test_curl_walkthrough_options_validate():
    from app.schemas import Options
    Options(**{"decode": "beam", "beam_size": 3, "max_new_tokens": 100, "cached_decode": True, "compile": False,
               "k_images": 4, "k_reports": 3, "label": True, "reference": None, "display_repair": False,
               "stop_on_repeat": False})


REPO = Path(__file__).resolve().parent.parent
README = REPO / "app" / "README.md"   # the runbook (P8-E); its section 9 is the curl walkthrough


def _readme_walkthrough_calls(readme=None):
    """The curl calls of the README's section 9, as written: method, path template ($S as {session_id}), query, headers,
    -F fields, -d body, the key its python3 line reads the session id from, and the status a trailing `# NNN` promises."""
    section = (readme or README).read_text(encoding="utf-8").split("## 9.", 1)[1]
    block = re.search(r"```bash\n(.*?)```", section, re.S).group(1)
    calls = []
    for line in block.splitlines():
        if "curl " not in line:
            continue
        promised = re.search(r"\s#\s*(\d{3})\s*$", line)
        id_key = re.search(r"json\.load\(sys\.stdin\)\[\"(\w+)\"\]", line)
        argv = shlex.split(line[line.index("curl "):].split(" | ", 1)[0], comments=True)   # the S=$(...) line pipes into python3
        method, url, headers, forms, body, args = None, None, {}, [], None, iter(argv[1:])
        for arg in args:
            if arg == "-X":
                method = next(args)
            elif arg == "-H":
                name, _, value = next(args).partition(":")
                headers[name.strip()] = value.strip()
            elif arg == "-F":
                forms.append(next(args))
            elif arg == "-d":
                body = next(args)
            elif arg in ("-o", "-w"):
                next(args)
            elif arg.startswith("$B/"):
                url = arg
        path, _, query = url[len("$B"):].partition("?")
        calls.append({"method": method or ("POST" if forms or body is not None else "GET"),
                      "path": path.replace("$S", "{session_id}"), "query": query, "headers": headers, "forms": forms,
                      "body": body, "id_key": id_key.group(1) if id_key else None,
                      "status": int(promised.group(1)) if promised else 200})
    return calls


def _readme_walkthrough_options():
    """The options the README's walkthrough sends: the JSON of its one `-F 'options=...'` field."""
    found = [f.partition("=")[2] for call in _readme_walkthrough_calls() for f in call["forms"] if f.startswith("options=")]
    assert len(found) == 1, found
    return json.loads(found[0])


def test_the_readme_walkthrough_sends_the_published_protocol():
    assert _readme_walkthrough_options() == WALKTHROUGH_OPTIONS   # the dict test_curl_walkthrough_options_validate pins


def _walkthrough_problems(client, readme=None):
    """What is wrong with the README's walkthrough on this server: [] when every call is a real route with real form fields
    and query names, and the calls, run in the README's order with its own headers, bodies and session-id key on the tiny
    engine, answer what the README says they answer."""
    calls = _readme_walkthrough_calls(readme)
    if not calls or (calls[0]["method"], calls[0]["path"]) != ("POST", "/v1/sessions") or not calls[0]["id_key"]:
        return ["the walkthrough does not start by creating the session the other calls use, and reading its id"]
    routes = {(method, route.path) for route in client.app.routes if isinstance(route, APIRoute) for method in route.methods}
    spec, sid, problems = client.get("/openapi.json").json(), None, []
    for call in calls:
        where = "{} {}".format(call["method"], call["path"])
        if (call["method"], call["path"]) not in routes:
            problems.append("no route " + where)
            continue
        params = {p["name"] for p in spec["paths"][call["path"]][call["method"].lower()].get("parameters", [])
                  if p["in"] == "query"}
        for name in [pair.partition("=")[0] for pair in call["query"].split("&") if pair]:
            if name not in params:
                problems.append("no query parameter {} on {}".format(name, where))
        files, data = {}, {}
        for field in call["forms"]:
            name, _, value = field.partition("=")
            if name not in _body_schema(spec, call["path"])["properties"]:
                problems.append("no form field {} on {}".format(name, where))
            elif value.startswith("@"):
                files[name] = (value[1:], png_bytes(), "image/png")
            else:
                data[name] = value
        url = call["path"].format(session_id=sid) + ("?" + call["query"] if call["query"] else "")
        extra = {} if call["body"] is None else {"content": call["body"]}
        r = client.request(call["method"], url, headers=call["headers"], files=files or None, data=data or None, **extra)
        if r.status_code != call["status"]:
            problems.append("{} answered {}, the README says {}".format(where, r.status_code, call["status"]))
            continue
        if sid is None:
            sid = r.json().get(call["id_key"])
            if sid is None:
                return problems + ["the session's answer has no {!r}".format(call["id_key"])]
        if files:
            frames = list(iter_sse([r.text]))
            if frames[-1]["event"] != "message_stop" or frames[-1]["data"]["status"] != "done":
                problems.append("the turn did not end done")
    return problems


def test_the_readme_walkthrough_runs_as_written(client):
    assert _walkthrough_problems(client) == []


@pytest.mark.parametrize("old, new", [
    ('/messages"', '/message"'),                                    # a route the server does not have
    ("-X DELETE", "-X PUT"),                                        # a method it does not have
    ("-F 'image=", "-F 'img="),                                     # a form field it does not take
    ("format=md", "format=pdf"),                                    # a request it refuses (422)
    ("'%{http_code}\\n'   # 204", "'%{http_code}\\n'   # 200"),  # a status it does not answer
    ("-H 'Content-Type: application/json' ", ""),                   # a JSON body sent without its content type (422)
    ("application/json", "text/plain"),                             # ... or with the wrong one
    ('json.load(sys.stdin)["id"]', 'json.load(sys.stdin)["session_id"]'),   # a session-id key the answer does not have
    ("format=md", "fmt=md"),                                        # a query name the route does not take
])
def test_a_drifted_walkthrough_is_caught(client, tmp_path, old, new):
    good = README.read_text(encoding="utf-8")
    assert good.count(old) == 1, old
    drifted = tmp_path / "README.md"
    drifted.write_text(good.replace(old, new), encoding="utf-8")
    assert _walkthrough_problems(client, drifted), old


def test_the_readme_is_allow_listed_past_the_md_rule():
    def git(*args):
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)

    if git("rev-parse", "--git-dir").returncode != 0:
        pytest.skip("not a git checkout")
    assert git("check-ignore", "-q", "--no-index", "app/README.md").returncode == 1   # 1: no rule ignores it, tracked or not


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


def test_the_options_description_says_a_model_trained_to_end_its_reports_stops_there_whatever_stop_on_repeat_says(client):   # P9-G2
    text = _body_schema(client.get("/openapi.json").json())["properties"]["options"]["description"]
    about = text.split("`eos_trained: true`", 1)[1].split(".")[0]
    assert "`stopped: eos`" in about and "stop_on_repeat" in about, about
    assert "already ended" in text and "the stop is `eos` too" in text      # a repeat stop after a report ended keeps that report


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


def test_the_image_route_documents_that_public_mode_refuses_the_original(client):   # P6 fix 1
    op = client.get("/openapi.json").json()["paths"][IMAGE]["get"]
    assert "`variant=original` in public mode" in op["description"] and "403" in op["description"]
    assert "original" in op["responses"]["403"]["description"]


def test_the_api_description_names_the_access_rules_and_every_error_kind(client):
    description = client.get("/openapi.json").json()["info"].get("description", "")
    for needle in ("Authorization: Bearer", "X-Client-Id", "401", "loopback"):
        assert needle in description, needle
    for status, kind in ERROR_KINDS.items():
        assert "{} {}".format(status, kind) in description, status


def test_the_api_description_states_the_size_cap_the_rate_limit_and_the_request_id_from_the_values_the_server_uses(client):   # P8-A
    description = " ".join(client.get("/openapi.json").json()["info"]["description"].split())
    mb = server.MAX_REQUEST_BYTES // (1024 * 1024)
    assert "{} MB".format(mb) in description and "413" in description and "before it is read" in description
    assert "`X-Request-Id`" in description and "errors included" in description and "1 to 64" in description
    assert "{} turns per {} minutes".format(hardening.RATE_CAPACITY, int(hardening.RATE_WINDOW_S // 60)) in description   # "6 turns per 10 minutes"
    assert "`{}`".format(hardening.RATE_LIMITED) in description and "`Retry-After`" in description and "public mode" in description
    assert "one JSON line per request" in description   # and what the log never holds
    assert "no query string" in description and "no body" in description and "no header but `X-Client-Id`" in description
    assert "a note that runs no model" in description and "costs the client nothing" in description   # P8 follow-up (3): such a turn is free


def test_the_api_description_says_a_422_never_repeats_a_value_that_was_sent_and_it_is_true(client):   # P8 follow-up (6)
    """A 422 names the key that was wrong ("retrieval_k: Extra inputs are not permitted") and never the value it held, so the sentence is about
    values: "never repeats what was sent" was not true of a key name."""
    description = " ".join(client.get("/openapi.json").json()["info"]["description"].split())
    assert "a 422 never repeats a value that was sent" in description
    assert "never repeats what was sent" not in description
    assert "a 500 never says what went wrong inside" in description
    sid = client.post("/v1/sessions", json={}).json()["id"]
    for r in (client.post("/v1/sessions", json={"nope": "VALUE-THAT-MUST-NOT-COME-BACK"}),
              client.post(MESSAGES.format(session_id=sid), data={"options": json.dumps({"retrieval_k": "VALUE-THAT-MUST-NOT-COME-BACK"})}),
              client.get("/v1/sessions?limit=VALUE-THAT-MUST-NOT-COME-BACK")):
        assert r.status_code == 422 and "VALUE-THAT-MUST-NOT-COME-BACK" not in r.text
    assert "nope" in client.post("/v1/sessions", json={"nope": "x"}).json()["error"]["message"]   # the key is named


UPLOAD_ROUTES = {("POST", MESSAGES), ("POST", "/v1/retrieve")}   # the routes that take an image


def test_the_413_example_names_the_request_on_a_route_that_takes_no_image_and_keeps_the_image_message_on_the_upload_routes(client):   # P8 follow-up (6)
    spec = client.get("/openapi.json").json()
    request_msg = "The request is larger than {} MB.".format(server.MAX_REQUEST_BYTES // (1024 * 1024))
    assert server.TOO_LARGE_REQUEST_MSG == request_msg
    seen = set()
    for method, path, op in _operations(spec):
        message = op["responses"]["413"]["content"]["application/json"]["example"]["error"]["message"]
        seen.add((method, path))
        if (method, path) in UPLOAD_ROUTES:
            assert message == "Image is larger than {} MB.".format(MAX_UPLOAD_BYTES // (1024 * 1024)), (method, path)
        else:
            assert message == request_msg and "image" not in message.lower(), (method, path)
    assert UPLOAD_ROUTES <= seen and len(seen - UPLOAD_ROUTES) >= 10   # and every other route has the one example
    for method, path in sorted(UPLOAD_ROUTES):   # the status line is the same everywhere: the image limit, and the whole request's
        description = spec["paths"][path][method.lower()]["responses"]["413"]["description"]
        assert "upload limit" in description and "{} MB".format(server.MAX_REQUEST_BYTES // (1024 * 1024)) in description


def test_the_turn_route_documents_both_429s_and_the_retry_after_header_of_the_rate_limit(client):   # P8-A
    op = client.get("/openapi.json").json()["paths"][MESSAGES]["post"]
    refusal = op["responses"]["429"]
    assert "overloaded_error" in refusal["description"] and hardening.RATE_LIMITED in refusal["description"]   # the queue's, and the rate's
    assert refusal["headers"]["Retry-After"]["schema"] == {"type": "string"} and "whole seconds" in refusal["headers"]["Retry-After"]["description"].lower()
    assert "429" in op["description"]


SIZE_CAP = {413}   # P8-A: the cap on a request's body is answered before routing, so every route declares it (REFUSED_BY_THE_ROUTE adds its own 413s)

REFUSED_BY_THE_ROUTE = {   # what each route declares by itself; 401, 403 and the client id's 400 are its `default`
    ("POST", "/v1/sessions"): {422}, ("GET", "/v1/sessions"): {422},
    ("GET", "/v1/sessions/{session_id}"): {404}, ("DELETE", "/v1/sessions/{session_id}"): {404},
    ("POST", MESSAGES): {400, 403, 404, 413, 422, 429, 500},
    ("GET", "/v1/messages/{message_id}"): {404, 422}, ("POST", "/v1/messages/{message_id}/cancel"): {404},
    ("GET", "/v1/sessions/{session_id}/export"): {404, 422},
    ("GET", "/v1/models"): set(), ("GET", "/healthz"): set(),
    ("POST", "/v1/retrieve"): {413, 422, 429, 500, 503}, ("POST", "/v1/label"): {422, 503},   # P5-E
    ("GET", "/v1/test-studies"): {403, 422, 503},
    ("GET", "/v1/messages/{message_id}/image"): {403, 404, 422, 500, 503},   # P6-B
    ("GET", "/v1/gallery/images/{row}"): {403, 404, 422, 500, 503},          # P6-C
}


def test_refusals_are_declared_as_the_error_envelope_and_named_in_the_description(client):
    ops = {(method, path): op for method, path, op in _operations(client.get("/openapi.json").json())}
    assert set(ops) == set(REFUSED_BY_THE_ROUTE)   # a new route needs its refusals declared here
    for key, op in ops.items():
        responses = op["responses"]
        declared = {int(status) for status in responses if status != "default" and int(status) >= 400}
        assert declared == REFUSED_BY_THE_ROUTE[key] | SIZE_CAP, key   # P8-A: and 413 on every route, not on two
        for status in declared:   # the description names each one the route answers itself, and the example is that status's envelope
            if status in REFUSED_BY_THE_ROUTE[key]:
                assert str(status) in op["description"], (key, status)   # (the cap is named by the API's own description: the test above)
            example = responses[str(status)]["content"]["application/json"]["example"]
            assert example["type"] == "error" and example["error"]["type"] == ERROR_KINDS[status], (key, status)
            assert set(example["error"]) == {"type", "message"} and example["error"]["message"], (key, status)
        assert responses["default"]["content"]["application/json"]["example"]["type"] == "error", key   # every route
        assert "HTTPValidationError" not in json.dumps(responses), key   # FastAPI's 422 is not this API's body


def test_every_refusal_the_server_really_sends_is_documented(client, tmp_path, monkeypatch, tiny_gallery):
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
        # P8-A: the cap on the whole request, answered before routing: any route can say it, and each declares it
        ("GET", "/healthz", "/healthz", {"content": bytes(server.MAX_REQUEST_BYTES + 1)}, 413),
        ("GET", "/v1/models", "/v1/models", {"content": bytes(server.MAX_REQUEST_BYTES + 1)}, 413),
        ("POST", "/v1/sessions", "/v1/sessions", {"content": bytes(server.MAX_REQUEST_BYTES + 1)}, 413),
        ("DELETE", "/v1/sessions/{session_id}", "/v1/sessions/s_missing", {"content": bytes(server.MAX_REQUEST_BYTES + 1)}, 413),
        ("POST", "/v1/label", "/v1/label", {"content": bytes(server.MAX_REQUEST_BYTES + 1)}, 413),
        ("GET", GALLERY_IMAGE, "/v1/gallery/images/0", {"content": bytes(server.MAX_REQUEST_BYTES + 1)}, 413),
        # P5-E: this server has no gallery and no labeller
        ("POST", "/v1/retrieve", "/v1/retrieve", {"files": image}, 503),
        ("POST", "/v1/retrieve", "/v1/retrieve", {"files": image, "data": {"k_images": "13"}}, 422),
        ("POST", "/v1/label", "/v1/label", {"json": {"text": "Findings: clear."}}, 503),
        ("POST", "/v1/label", "/v1/label", {"json": {"text": ""}}, 422),
        ("GET", "/v1/test-studies", "/v1/test-studies", {}, 503),
        ("GET", "/v1/test-studies", "/v1/test-studies?limit=0", {}, 422),
        # P6-B: a turn's image
        ("GET", IMAGE, "/v1/messages/m_missing/image?variant=thumb", {}, 404),
        ("GET", IMAGE, "/v1/messages/m_missing/image?variant=huge", {}, 422),
        # P6-C: a similar X-ray's image, on a server with no gallery
        ("GET", GALLERY_IMAGE, "/v1/gallery/images/0", {}, 503),
        ("GET", GALLERY_IMAGE, "/v1/gallery/images/-1", {}, 422),
    ]:
        check(client.request(method, url, **kwargs), method, path, status)
    planted, _ = client.app.state.store.start_turn(sid, "", "private", {}, "ab" * 32, None, 0)   # a test study, on a server with no gallery
    check(client.get("/v1/messages/{}/image?variant=thumb".format(planted)), "GET", IMAGE, 503)
    with TestClient(create_app(engine="tiny", home=str(tmp_path / "gallery_home"),
                               gallery=Gallery.open(decide_gate(tiny_gallery), None))) as g:
        gsid = g.post("/v1/sessions", json={}).json()["id"]
        studied = g.post(MESSAGES.format(session_id=gsid), data={"options": json.dumps({"test_row": 1, "max_new_tokens": 16})})
        assert studied.status_code == 200
        g.app.state.gallery.test_study(1)["image"].unlink()   # gone since the turn ran
        check(g.get("/v1/messages/{}/image?variant=original".format(studied.headers["x-message-id"])), "GET", IMAGE, 500)
        check(g.get("/v1/gallery/images/{}".format(10 ** 6)), "GET", GALLERY_IMAGE, 404)   # P6-C: outside the gallery
        g.app.state.gallery.image_path(2).unlink()
        check(g.get("/v1/gallery/images/2"), "GET", GALLERY_IMAGE, 500)                    # and a file that cannot be read
        check(g.post("/v1/retrieve", files={"image": ("x.png", bytes(MAX_UPLOAD_BYTES + 1), "image/png")}),
              "POST", "/v1/retrieve", 413)
        with monkeypatch.context() as m:
            m.setattr(g.app.state.worker, "reserve", lambda: False)
            check(g.post("/v1/retrieve", files=image), "POST", "/v1/retrieve", 429)

        def broken(*args):
            raise RuntimeError("boom")

        with monkeypatch.context() as m:   # P5-E fix 1: a retrieval that fails is a 500 in the envelope
            m.setattr(g.app.state.worker.pipeline, "retrieve_upload", broken)
            check(g.post("/v1/retrieve", files=image), "POST", "/v1/retrieve", 500)
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
        check(c.get("/v1/test-studies", headers=public), "GET", "/v1/test-studies", 403)   # and the picker's list
        planted, _ = c.app.state.store.start_turn(psid, "", "public", {}, "ab" * 32, None)
        c.app.state.store._con.execute("UPDATE messages SET test_row = 0 WHERE id = ?", (planted,))   # never stored in public: planted
        check(c.get("/v1/messages/{}/image?variant=thumb".format(planted), headers=public), "GET", IMAGE, 403)   # and its image
        check(c.get("/v1/gallery/images/0", headers=public), "GET", GALLERY_IMAGE, 403)   # P6-C: a gallery image in public mode


ROUTE_BODIES = {("POST", "/v1/sessions"): {"json": {}}, ("POST", MESSAGES): {"data": {"options": "{}"}}}   # to parse


def test_each_routes_default_names_exactly_the_guard_refusals_it_really_gets(tmp_path):
    # Which of 401 (token), 400 (client id) and 403 (loopback) a route can answer differs: /v1/models takes no client id
    # and /healthz no token. So every route is asked in each of the three settings that refuse before a route runs.
    def ask_all(c, base="", headers=None):
        urls = {key: base + key[1].replace("{session_id}", "s_x").replace("{message_id}", "m_x").replace("{row}", "0")
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
