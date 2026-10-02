"""CHAT_UI_PLAN.md P3-D: the streaming API on the tiny engine."""
import asyncio
import http.client
import io
import json
import shutil
import socket
import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import server
from app.commands import NOT_A_QA_BOT
from app.engine import REPO_ROOT, build_engine
from app.imaging import FORMATS_MSG, MAX_UPLOAD_BYTES, TOO_LARGE_MSG, TOO_SMALL_MSG, UNREADABLE_MSG, UploadError
from app.pipeline import REFERENCE_IGNORED_PUBLIC, Pipeline, TurnJob, Worker
from app.redact import PUBLIC_ERROR_MESSAGE
from app.schemas import Options
from app.server import create_app
from app.store import Store
from tests.app_helpers import iter_sse, png_bytes, wait_until

STAGES = ["preprocess", "encode", "retrieve", "generate", "label", "score"]


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(engine="tiny", home=str(tmp_path))) as c:
        yield c


@pytest.fixture
def live(tmp_path):
    """A real uvicorn on an ephemeral port: TestClient buffers whole responses, so streaming and
    disconnect behaviour are tested over real sockets."""
    import uvicorn
    app = create_app(engine="tiny", home=str(tmp_path), tiny_step_delay_s=0.02, queue_cap=2)
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.02)
    yield "http://127.0.0.1:{}".format(port)
    server.should_exit = True
    t.join(5)


def _turn(c, sid, options=None, image=True, text=""):
    files = {"image": ("x.png", png_bytes(), "image/png")} if image else None
    data = {"text": text, "options": json.dumps(options or {"max_new_tokens": 16})}
    with c.stream("POST", "/v1/sessions/{}/messages".format(sid), files=files, data=data) as r:
        assert r.status_code == 200, r.read()
        return list(iter_sse(r.iter_text()))


def test_turn_streams_contiguous_events_in_contract_order(client):
    sid = client.post("/v1/sessions", json={}).json()["id"]
    frames = _turn(client, sid)
    assert [f["data"]["seq"] for f in frames] == list(range(1, len(frames) + 1))
    assert frames[0]["event"] == "message_start" and frames[-1]["event"] == "message_stop"
    assert [f["data"]["stage"] for f in frames if f["event"] == "stage_end"] == STAGES
    assert frames[-1]["data"]["status"] == "done"
    assert frames[-1]["data"]["disclaimer"] == "Research prototype; not for clinical use."


def test_stored_events_replay_exactly_what_was_sent(client):
    sid = client.post("/v1/sessions", json={}).json()["id"]
    frames = _turn(client, sid)
    mid = frames[0]["data"]["message_id"]
    stored = client.get("/v1/messages/{}".format(mid)).json()["events"]
    assert [(e["event"], e["data"]) for e in stored] == [(f["event"], f["data"]) for f in frames]
    assert [e["seq"] for e in client.get("/v1/messages/{}?after=5".format(mid)).json()["events"]][0] == 6


def test_dropped_stream_does_not_cancel_the_turn(live):   # Review Focus 4
    sid = httpx.post(live + "/v1/sessions", json={}).json()["id"]
    with httpx.stream("POST", live + "/v1/sessions/{}/messages".format(sid),
                      files={"image": ("x.png", png_bytes(), "image/png")},
                      data={"text": "", "options": json.dumps({"max_new_tokens": 60})}, timeout=30) as r:
        frames = []
        for frame in iter_sse(r.iter_text()):
            frames.append(frame)
            if frame["event"] == "content_block_delta":
                break                                       # the tunnel drops here
    mid = frames[0]["data"]["message_id"]
    for _ in range(400):
        msg = httpx.get(live + "/v1/messages/{}".format(mid)).json()
        if msg["status"] != "running":
            break
        time.sleep(0.05)
    assert msg["status"] == "done"
    assert [e["seq"] for e in msg["events"]] == list(range(1, len(msg["events"]) + 1))


def test_cancel_endpoint_aborts_the_turn(live):
    sid = httpx.post(live + "/v1/sessions", json={}).json()["id"]
    frames, cancel_status = [], []
    with httpx.stream("POST", live + "/v1/sessions/{}/messages".format(sid),
                      files={"image": ("x.png", png_bytes(), "image/png")},
                      data={"text": "", "options": json.dumps({"max_new_tokens": 150})}, timeout=30) as r:
        for frame in iter_sse(r.iter_text()):
            frames.append(frame)
            if frame["event"] == "content_block_delta" and not cancel_status:
                mid = frames[0]["data"]["message_id"]
                cancel_status.append(httpx.post(live + "/v1/messages/{}/cancel".format(mid)).status_code)
    assert cancel_status == [200]
    assert frames[-1]["event"] == "message_stop" and frames[-1]["data"]["status"] == "aborted"
    assert len([f for f in frames if f["event"] == "content_block_delta"]) < 150
    mid = frames[0]["data"]["message_id"]
    assert httpx.get(live + "/v1/messages/{}".format(mid)).json()["status"] == "aborted"


def test_overload_is_refused_before_the_stream_opens(live):   # the live fixture has queue_cap=2
    sid = httpx.post(live + "/v1/sessions", json={}).json()["id"]
    url = live + "/v1/sessions/{}/messages".format(sid)
    body = {"text": "", "options": json.dumps({"max_new_tokens": 150})}
    opened = []

    def hold():
        with httpx.stream("POST", url, files={"image": ("x.png", png_bytes(), "image/png")},
                          data=body, timeout=60) as r:
            opened.append(r.status_code)
            for _ in r.iter_text():
                pass

    threads = [threading.Thread(target=hold) for _ in range(2)]
    for t in threads:
        t.start()
    deadline = time.time() + 10
    while len(opened) < 2 and time.time() < deadline:
        time.sleep(0.02)
    r = httpx.post(url, files={"image": ("x.png", png_bytes(), "image/png")}, data=body, timeout=30)
    assert r.status_code == 429 and r.json()["error"]["type"] == "overloaded_error"
    for t in threads:
        t.join(60)
    assert opened == [200, 200]


def test_refuses_non_loopback_bind_without_token(tmp_path):   # Review Focus 2, R6
    with pytest.raises(RuntimeError, match="token"):
        create_app(engine="tiny", home=str(tmp_path), host="0.0.0.0", token=None)
    with pytest.raises(RuntimeError, match="token"):
        create_app(engine="tiny", home=str(tmp_path), mode="public", token=None)


def test_token_guards_every_v1_route_but_not_health(tmp_path):
    with TestClient(create_app(engine="tiny", home=str(tmp_path), token="t0k")) as c:
        assert c.get("/healthz").status_code == 200
        assert c.get("/v1/sessions").status_code == 401
        assert c.post("/v1/sessions", json={}).status_code == 401
        assert c.post("/v1/sessions", json={}, headers={"Authorization": "Bearer t0k"}).status_code == 200


def test_public_client_cannot_read_another_clients_session(tmp_path):   # Review Focus 3
    with TestClient(create_app(engine="tiny", home=str(tmp_path), mode="public", token="t")) as c:
        auth = {"Authorization": "Bearer t"}
        sid = c.post("/v1/sessions", json={}, headers=dict(auth, **{"X-Client-Id": "a"})).json()["id"]
        assert c.get("/v1/sessions/{}".format(sid), headers=dict(auth, **{"X-Client-Id": "b"})).status_code == 404
        assert c.get("/v1/sessions", headers=dict(auth, **{"X-Client-Id": "b"})).json()["sessions"] == []


def test_text_only_turn_reuses_the_previous_image_and_applies_the_command(client):
    sid = client.post("/v1/sessions", json={}).json()["id"]
    _turn(client, sid)
    frames = _turn(client, sid, image=False, text="beam 2")
    start = frames[0]["data"]
    assert start["image"]["source"] == "previous" and start["options"]["beam_size"] == 2


def test_question_without_image_gets_the_fixed_answer(client):
    sid = client.post("/v1/sessions", json={}).json()["id"]
    _turn(client, sid)
    frames = _turn(client, sid, image=False, text="is this pneumonia?")
    assert any(f["event"] == "warning" and f["data"]["code"] == "not_a_command" for f in frames)


def test_first_turn_without_image_is_a_422(client):
    sid = client.post("/v1/sessions", json={}).json()["id"]
    r = client.post("/v1/sessions/{}/messages".format(sid), data={"text": "beam 2", "options": "{}"})
    assert r.status_code == 422 and "Attach an X-ray" in r.json()["error"]["message"]


def test_bad_options_are_a_422_before_streaming(client):
    sid = client.post("/v1/sessions", json={}).json()["id"]
    r = client.post("/v1/sessions/{}/messages".format(sid), files={"image": ("x.png", png_bytes(), "image/png")},
                    data={"text": "", "options": json.dumps({"retrieval_k": 3})})
    assert r.status_code == 422 and r.json()["error"]["type"] == "validation_error"


def test_export_and_delete(client):
    sid = client.post("/v1/sessions", json={}).json()["id"]
    _turn(client, sid)
    js = client.get("/v1/sessions/{}/export?format=json".format(sid))
    assert js.headers["content-type"].startswith("application/json")
    assert js.json()["messages"][1]["events"][-1]["event"] == "message_stop"   # [0] user, [1] assistant
    md = client.get("/v1/sessions/{}/export?format=md".format(sid))
    assert md.headers["content-type"].startswith("text/markdown") and "## Turn 1" in md.text
    assert client.delete("/v1/sessions/{}".format(sid)).status_code == 204
    assert client.get("/v1/sessions/{}".format(sid)).status_code == 404


# ---- beyond the brief: the behaviours it lists and the controller's rulings (task-P3-D-rulings.md) -------------------

PUBLIC = {"Authorization": "Bearer t", "X-Client-Id": "a"}
URL_VARIANTS = ("original", "thumb", "model_input")


def _post(c, sid, options=None, text="", image=True, headers=None, files=None):
    """A turn POST read whole (TestClient buffers it): a refusal's JSON, or the whole stream."""
    if files is None and image:
        files = {"image": ("x.png", png_bytes(), "image/png")}
    data = {"text": text, "options": json.dumps({"max_new_tokens": 16} if options is None else options)}
    return c.post("/v1/sessions/{}/messages".format(sid), files=files, data=data, headers=headers)


def _public_turn(c, sid, options=None, text="", headers=PUBLIC):
    r = _post(c, sid, options=options, text=text, headers=headers)
    assert r.status_code == 200, r.text
    return list(iter_sse([r.text]))


def _stage_ends(frames):
    return {f["data"]["stage"]: f["data"] for f in frames if f["event"] == "stage_end"}


def _new_session(base, headers=None):
    return httpx.post(base + "/v1/sessions", json={}, headers=headers).json()["id"]


def _fake_real_engines(monkeypatch, cacheless=()):
    """Ruling 1's real path without checkpoints (none exist on the laptop): build_engine("real", ...) is patched to
    return tiny engines named after their model, with the checkpoint path on their card."""
    calls = []

    def fake(kind, **kw):
        calls.append((kind, kw))
        engine = build_engine("tiny")
        engine.name = kw["model_config"]
        engine._card.update(name=engine.name, checkpoint=kw["checkpoint"])
        if engine.name in cacheless:   # as the 13D model: the uncached beam search only
            engine.decoder.supports_cached_decode = lambda: False
            engine._card["cached_decode_available"] = False
        return engine

    monkeypatch.setattr(server, "build_engine", fake)
    return calls


# ---- R6/D9, D23, D8: bind refusal, token, client scoping ------------------------------------------------------------

def test_loopback_binds_need_no_token_but_any_other_host_does(tmp_path):
    for i, host in enumerate(["127.0.0.1", "127.0.0.2", "::1", "localhost"]):
        create_app(engine="tiny", home=str(tmp_path / str(i)), host=host).state.store.close()
    with pytest.raises(RuntimeError, match="token"):
        create_app(engine="tiny", home=str(tmp_path / "x"), host="gx08", token="")
    assert not (tmp_path / "x").exists()   # refused before anything was opened or built
    with pytest.raises(ValueError, match="mode"):
        create_app(engine="tiny", home=str(tmp_path / "y"), mode="demo")


def test_chat_home_must_live_outside_the_repository():   # DUA: the session database never sits in the repo tree
    home = REPO_ROOT / "chat_home_guard_test"
    try:
        with pytest.raises(RuntimeError, match="outside the repository"):
            create_app(engine="tiny", home=str(home))
        assert not home.exists()   # refused before anything was created
    finally:
        shutil.rmtree(home, ignore_errors=True)   # only a broken guard leaves it behind


def test_the_token_guards_every_v1_path_and_its_errors_use_the_envelope(tmp_path):
    with TestClient(create_app(engine="tiny", home=str(tmp_path), token="t0k")) as c:
        for method, path in [("GET", "/v1/models"), ("GET", "/v1/sessions/s_x"), ("DELETE", "/v1/sessions/s_x"),
                             ("POST", "/v1/sessions/s_x/messages"), ("GET", "/v1/messages/m_x"),
                             ("POST", "/v1/messages/m_x/cancel"), ("GET", "/v1/sessions/s_x/export"),
                             ("GET", "/v1/not-a-route")]:
            for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": "Basic t0k"}):
                r = c.request(method, path, headers=headers)
                assert r.status_code == 401, (method, path, headers)
                assert r.json()["error"]["type"] == "authentication_error"
                assert r.headers["www-authenticate"] == "Bearer"
        assert c.get("/v1/models", headers={"Authorization": "bearer t0k"}).status_code == 200


def test_index_and_static_files_stay_open_with_a_token(tmp_path, monkeypatch):
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<!doctype html><title>chat</title><p>the page</p>")
    (static / "app.js").write_text("export const x = 1;\n")
    monkeypatch.setattr(server, "STATIC_DIR", static)
    with TestClient(create_app(engine="tiny", home=str(tmp_path / "home"), token="t0k")) as c:
        page = c.get("/")
        assert page.status_code == 200 and "the page" in page.text
        assert c.get("/static/app.js").text == "export const x = 1;\n"
        assert c.get("/static/missing.js").status_code == 404
        assert c.get("/v1/models").status_code == 401


def test_without_a_built_page_the_index_is_a_placeholder_and_static_is_404(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "STATIC_DIR", tmp_path / "not_built_yet")   # app/static/ arrives with P4
    with TestClient(create_app(engine="tiny", home=str(tmp_path / "home"))) as c:
        page = c.get("/")
        assert page.status_code == 200 and page.headers["content-type"].startswith("text/html")
        assert "Research prototype; not for clinical use." in page.text
        r = c.get("/static/app.js")
        assert r.status_code == 404 and r.json()["error"]["type"] == "not_found_error"


def test_public_mode_needs_a_client_id_on_every_scoped_route(tmp_path):
    with TestClient(create_app(engine="tiny", home=str(tmp_path), mode="public", token="t")) as c:
        auth = {"Authorization": "Bearer t"}
        for method, path in [("GET", "/v1/sessions"), ("POST", "/v1/sessions"), ("GET", "/v1/sessions/s_x"),
                             ("DELETE", "/v1/sessions/s_x"), ("GET", "/v1/messages/m_x"),
                             ("POST", "/v1/messages/m_x/cancel"), ("GET", "/v1/sessions/s_x/export")]:
            for headers in (auth, dict(auth, **{"X-Client-Id": ""}), dict(auth, **{"X-Client-Id": "x" * 129})):
                r = c.request(method, path, headers=headers)
                assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error", (method, path)
        assert c.get("/healthz").json()["mode"] == "public"
        assert c.get("/v1/models", headers=auth).status_code == 200   # no session data: no client id needed


def test_public_scoping_covers_messages_turns_cancels_exports_and_deletes(tmp_path):
    with TestClient(create_app(engine="tiny", home=str(tmp_path), mode="public", token="t")) as c:
        sid = c.post("/v1/sessions", json={}, headers=PUBLIC).json()["id"]
        mid = _public_turn(c, sid)[0]["data"]["message_id"]
        b = dict(PUBLIC, **{"X-Client-Id": "b"})
        assert c.get("/v1/messages/{}".format(mid), headers=b).status_code == 404
        assert c.post("/v1/messages/{}/cancel".format(mid), headers=b).status_code == 404
        assert c.get("/v1/sessions/{}/export?format=json".format(sid), headers=b).status_code == 404
        assert _post(c, sid, headers=b).status_code == 404
        assert c.delete("/v1/sessions/{}".format(sid), headers=b).status_code == 404
        assert c.get("/v1/messages/{}".format(mid), headers=PUBLIC).json()["status"] == "done"   # still a's
        assert [s["id"] for s in c.get("/v1/sessions", headers=PUBLIC).json()["sessions"]] == [sid]


def test_cors_preflight_is_answered_ahead_of_the_token(tmp_path):
    origin = "http://127.0.0.1:5173"
    with TestClient(create_app(engine="tiny", home=str(tmp_path), token="t", cors_origins=(origin,))) as c:
        r = c.options("/v1/sessions", headers={"Origin": origin, "Access-Control-Request-Method": "POST",
                                               "Access-Control-Request-Headers": "authorization,x-client-id"})
        assert r.status_code == 200 and r.headers["access-control-allow-origin"] == origin
        r = c.post("/v1/sessions", json={}, headers={"Origin": origin, "Authorization": "Bearer t"})
        assert r.status_code == 200 and r.headers["access-control-allow-origin"] == origin
        assert "x-message-id" in r.headers["access-control-expose-headers"].lower()


# ---- ruling 1: engines ---------------------------------------------------------------------------------------------

def test_real_engines_are_one_cpu_engine_per_model_and_the_first_is_the_default(tmp_path, monkeypatch):
    calls = _fake_real_engines(monkeypatch)
    models = ("hybrid_150m_m3_rrg", "hybrid_150m_v2_rrg")
    app = create_app(engine="real", home=str(tmp_path), models=models, drift_note="CPU vs GPU: 0/20 differ")
    assert server.MODEL_CHECKPOINTS == {
        "hybrid_150m_m3_rrg": "outputs/h100_report_gen_m3_tower13d_s42/checkpoints/last.ckpt",
        "hybrid_150m_v2_rrg": "outputs/h100_report_gen_full_ext_4gpu_tower13d/checkpoints/last.ckpt"}
    assert calls == [("real", {"checkpoint": str(REPO_ROOT / server.MODEL_CHECKPOINTS[m]), "model_config": m,
                               "device": "cpu", "drift_note": "CPU vs GPU: 0/20 differ"}) for m in models]
    with TestClient(app) as c:
        listed = c.get("/v1/models").json()
        assert listed["default_model"] == "hybrid_150m_m3_rrg"
        assert [m["name"] for m in listed["models"]] == list(models)
        assert listed["models"][0]["checkpoint"] == str(REPO_ROOT / server.MODEL_CHECKPOINTS[models[0]])   # private
        sid = c.post("/v1/sessions", json={}).json()["id"]
        start = _turn(c, sid)[0]["data"]
        assert start["model"]["name"] == start["options"]["model"] == "hybrid_150m_m3_rrg"   # model None: the default
        start = _turn(c, sid, options={"max_new_tokens": 16, "model": "hybrid_150m_v2_rrg"})[0]["data"]
        assert start["model"]["name"] == start["options"]["model"] == "hybrid_150m_v2_rrg"


def test_the_tiny_engine_ignores_models_and_an_unknown_model_is_a_422(tmp_path):
    with TestClient(create_app(engine="tiny", home=str(tmp_path), models=("hybrid_150m_v2_rrg",))) as c:
        listed = c.get("/v1/models").json()
        assert listed["default_model"] == "tiny" and [m["name"] for m in listed["models"]] == ["tiny"]
        sid = c.post("/v1/sessions", json={}).json()["id"]
        r = _post(c, sid, options={"model": "hybrid_150m_v2_rrg"})
        assert r.status_code == 422 and r.json()["error"]["type"] == "validation_error"
        assert _turn(c, sid, options={"max_new_tokens": 16, "model": "tiny"})[-1]["data"]["status"] == "done"


def test_unknown_engine_kinds_and_models_are_refused_at_start(tmp_path):
    with pytest.raises(ValueError, match="engine"):
        create_app(engine="mock", home=str(tmp_path / "a"))
    with pytest.raises(ValueError, match="models"):
        create_app(engine="real", home=str(tmp_path / "b"), models=("hybrid_70m",))


# ---- rulings 2 and 3: error kinds and the pre-stream checks, in order ----------------------------------------------

def test_malformed_options_are_a_400_and_invalid_ones_a_readable_422(client):
    sid = client.post("/v1/sessions", json={}).json()["id"]
    url = "/v1/sessions/{}/messages".format(sid)
    for raw in ("{not json", "[1, 2]", "3"):
        r = client.post(url, files={"image": ("x.png", png_bytes(), "image/png")}, data={"text": "", "options": raw})
        assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error"
    r = _post(client, sid, options={"beam_size": 9, "retrieval_k": 3})
    assert r.status_code == 422 and r.json()["error"]["type"] == "validation_error"
    message = r.json()["error"]["message"]
    assert "beam_size" in message and "retrieval_k" in message and "9" not in message.replace("beam_size", "")
    r = _post(client, sid, options={}, text="beam 20")   # a command is validated like the drawer's options
    assert r.status_code == 422 and "beam_size" in r.json()["error"]["message"]
    assert client.get("/v1/sessions/{}".format(sid)).json()["messages"] == []   # nothing was stored


def test_unknown_sessions_and_messages_are_404s_in_the_envelope(client):
    for path in ("/v1/sessions/s_missing", "/v1/messages/m_missing", "/v1/sessions/s_missing/export?format=md",
                 "/v1/no/such/route"):
        r = client.get(path)
        assert r.status_code == 404 and r.json()["error"]["type"] == "not_found_error", path
    assert client.delete("/v1/sessions/s_missing").status_code == 404
    assert client.post("/v1/messages/m_missing/cancel").status_code == 404
    assert _post(client, "s_missing").status_code == 404


def test_cached_decode_on_a_model_without_a_cache_is_a_422_in_the_engines_own_words(tmp_path, monkeypatch):
    _fake_real_engines(monkeypatch, cacheless=("hybrid_150m_v2_rrg",))
    app = create_app(engine="real", home=str(tmp_path), models=("hybrid_150m_m3_rrg", "hybrid_150m_v2_rrg"))
    with pytest.raises(ValueError) as raised:   # what the engine itself raises at generate (P2-D)
        app.state.engines["hybrid_150m_v2_rrg"].generate(None, Options(), lambda step, text: None, threading.Event())
    with TestClient(app) as c:
        assert [m["cached_decode_available"] for m in c.get("/v1/models").json()["models"]] == [True, False]
        sid = c.post("/v1/sessions", json={}).json()["id"]
        r = _post(c, sid, options={"model": "hybrid_150m_v2_rrg"})   # cached_decode defaults to true
        assert r.status_code == 422
        assert r.json()["error"] == {"type": "validation_error", "message": str(raised.value)}
        frames = _turn(c, sid, options={"max_new_tokens": 16, "model": "hybrid_150m_v2_rrg", "cached_decode": False})
        assert frames[-1]["data"]["status"] == "done"
        assert _stage_ends(frames)["generate"]["detail"]["cached_decode"] is False


def test_compile_is_refused_unless_the_server_allows_it(tmp_path):
    with TestClient(create_app(engine="tiny", home=str(tmp_path / "a"))) as c:
        assert c.get("/v1/models").json()["allow_compile"] is False
        sid = c.post("/v1/sessions", json={}).json()["id"]
        r = _post(c, sid, options={"compile": True})
        assert r.status_code == 422 and "compile" in r.json()["error"]["message"]
    with TestClient(create_app(engine="tiny", home=str(tmp_path / "b"), allow_compile=True)) as c:
        assert c.get("/v1/models").json()["allow_compile"] is True
        sid = c.post("/v1/sessions", json={}).json()["id"]
        assert _turn(c, sid, options={"max_new_tokens": 16, "compile": True})[-1]["data"]["status"] == "done"


def test_a_test_row_is_refused_in_public_mode_and_without_a_gallery(tmp_path):
    with TestClient(create_app(engine="tiny", home=str(tmp_path / "a"))) as c:
        sid = c.post("/v1/sessions", json={}).json()["id"]
        r = _post(c, sid, options={"test_row": 0}, image=False)
        assert r.status_code == 422 and "gallery" in r.json()["error"]["message"]
    with TestClient(create_app(engine="tiny", home=str(tmp_path / "b"), mode="public", token="t")) as c:
        sid = c.post("/v1/sessions", json={}, headers=PUBLIC).json()["id"]
        r = _post(c, sid, options={"test_row": 0}, image=False, headers=PUBLIC)
        assert r.status_code == 422 and "public" in r.json()["error"]["message"]


def test_pre_stream_checks_refuse_in_the_ruled_order(tmp_path, monkeypatch):
    """auth, client id, session, options, model, cached_decode, compile, test_row, image, previous image, overload."""
    with TestClient(create_app(engine="tiny", home=str(tmp_path), mode="public", token="t")) as c:
        sid = c.post("/v1/sessions", json={}, headers=PUBLIC).json()["id"]
        bad_options = {"retrieval_k": 3}
        gif = {"image": ("x.gif", b"GIF89a" + bytes(64), "image/gif")}

        def code(headers=PUBLIC, session=sid, options=None, files=None, image=False):
            return _post(c, session, options=bad_options if options is None else options, headers=headers,
                         files=files, image=image).status_code

        assert code(headers={}) == 401
        assert code(headers={"Authorization": "Bearer t"}) == 400
        assert code(session="s_not_this_clients") == 404
        assert code() == 422                                                   # options
        monkeypatch.setattr(c.app.state.worker, "reserve", lambda: False)    # as if queue_cap turns were in flight
        assert code(options={"model": "nope"}, files=gif) == 422              # model, before the image
        assert code(options={"compile": True}, files=gif) == 422              # compile, before the image
        assert code(options={"test_row": 1}, files=gif) == 422                # test_row, before the image
        r = _post(c, sid, options={}, files=gif, headers=PUBLIC)               # the image, before overload
        assert r.status_code == 422 and r.json()["error"]["message"] == FORMATS_MSG
        r = _post(c, sid, options={}, image=False, headers=PUBLIC)             # no image anywhere, before overload
        assert r.status_code == 422 and r.json()["error"]["message"] == "Attach an X-ray first."
        r = _post(c, sid, options={}, headers=PUBLIC)                          # a valid turn: overload is last
        assert r.status_code == 429 and r.json()["error"]["type"] == "overloaded_error"
        assert c.get("/v1/sessions/{}".format(sid), headers=PUBLIC).json()["messages"] == []


# ---- ruling 4: upload size and the upload check before the stream ---------------------------------------------------

def test_an_image_over_the_limit_is_a_413_and_an_unusable_one_a_422_before_streaming(client):
    sid = client.post("/v1/sessions", json={}).json()["id"]
    r = _post(client, sid, files={"image": ("x.png", bytes(MAX_UPLOAD_BYTES + 1), "image/png")})
    assert r.status_code == 413 and r.json()["error"] == {"type": "request_too_large", "message": TOO_LARGE_MSG}
    r = _post(client, sid, files={"image": ("x.png", png_bytes(32, 32), "image/png")})
    assert r.status_code == 422 and r.json()["error"] == {"type": "validation_error", "message": TOO_SMALL_MSG}
    r = _post(client, sid, files={"image": ("x.dcm", bytes(128) + b"DICM" + bytes(64), "application/dicom")})
    assert r.status_code == 422 and "DICOM" in r.json()["error"]["message"]
    assert client.get("/v1/sessions/{}".format(sid)).json()["messages"] == []


def test_a_body_without_content_length_is_capped_as_it_arrives(client):
    sid = client.post("/v1/sessions", json={}).json()["id"]

    def body():   # a generator body goes out chunked, with no Content-Length to refuse it by
        yield b'--zzz\r\nContent-Disposition: form-data; name="image"; filename="x.png"\r\n\r\n'
        yield bytes(server.MAX_REQUEST_BYTES)

    r = client.post("/v1/sessions/{}/messages".format(sid), content=body(),
                    headers={"Content-Type": "multipart/form-data; boundary=zzz"})
    assert "content-length" not in r.request.headers
    assert r.status_code == 413 and r.json()["error"]["type"] == "request_too_large"


def test_a_content_length_over_the_cap_is_refused_before_the_body_is_read(live):
    sid = _new_session(live)
    host, port = live[len("http://"):].split(":")
    with socket.create_connection((host, int(port)), timeout=10) as sock:
        # Headers only: no body byte is ever sent, so the answer can only come from the Content-Length header.
        sock.sendall("POST /v1/sessions/{}/messages HTTP/1.1\r\nHost: {}\r\nContent-Type: multipart/form-data; "
                     "boundary=zzz\r\nContent-Length: {}\r\n\r\n".format(sid, host, server.MAX_REQUEST_BYTES + 1)
                     .encode())
        reply = http.client.HTTPResponse(sock)
        reply.begin()
        body = json.loads(reply.read())
    assert reply.status == 413 and body["error"] == {"type": "request_too_large", "message": TOO_LARGE_MSG}
    assert server.MAX_REQUEST_BYTES == MAX_UPLOAD_BYTES + 1024 * 1024


# ---- ruling 5: start-up -------------------------------------------------------------------------------------------

def test_start_up_prints_the_journal_mode_and_closes_turns_a_restart_left_running(tmp_path, capsys):
    store = Store(tmp_path)
    session = store.create_session("private")
    _, mid = store.start_turn(session["id"], "x", "private", {})
    store.close()
    app = create_app(engine="tiny", home=str(tmp_path))
    out = capsys.readouterr().out
    assert "[server] sqlite journal_mode=wal\n" in out and "[server] recovered 1 running turn(s)\n" in out
    with TestClient(app) as c:
        msg = c.get("/v1/messages/{}".format(mid)).json()
        assert msg["status"] == "error" and msg["events"][0]["data"]["error"]["type"] == "server_restart"


# ---- ruling 6 and the fixed skips: retrieve, label and score until P5-E ----------------------------------------------

def test_the_last_three_stages_end_skipped_with_their_fixed_reasons(tmp_path):
    with TestClient(create_app(engine="tiny", home=str(tmp_path), labeler_url="http://127.0.0.1:9",
                               gallery_dir=str(tmp_path / "gallery"))) as c:
        assert c.app.state.labeler_url == "http://127.0.0.1:9"
        sid = c.post("/v1/sessions", json={}).json()["id"]
        frames = _turn(c, sid)
        skipped = {stage: data.get("skipped") for stage, data in _stage_ends(frames).items()}
        assert skipped == {"preprocess": None, "encode": None, "retrieve": "gallery_unavailable", "generate": None,
                           "label": "labeler_unavailable", "score": "no_reference"}
        starts = [f["data"] for f in frames if f["event"] == "stage_start"]
        assert [d["stage"] for d in starts] == ["preprocess", "encode", "generate"]   # a skipped stage has no start
        assert all(d == {"stage": d["stage"], "index": STAGES.index(d["stage"]), "seq": d["seq"]} for d in starts)
        frames = _turn(c, sid, options={"max_new_tokens": 16, "label": False})
        assert _stage_ends(frames)["label"]["skipped"] == "label_off"
        frames = _turn(c, sid, options={"max_new_tokens": 16, "reference": "Findings: clear."})   # P5-E scores it
        assert _stage_ends(frames)["score"]["skipped"] == "no_reference"
        assert not [f for f in frames if f["event"] == "warning"]


# ---- ruling 7: every event is redacted before it is stored and sent -------------------------------------------------

def test_a_public_turn_is_redacted_before_it_is_stored_and_sent(tmp_path):
    with TestClient(create_app(engine="tiny", home=str(tmp_path), mode="public", token="t")) as c:
        sid = c.post("/v1/sessions", json={}, headers=PUBLIC).json()["id"]
        frames = _public_turn(c, sid, options={"max_new_tokens": 16, "reference": "Findings: my own secret line."})
        assert [f["data"]["seq"] for f in frames] == list(range(1, len(frames) + 1))   # a dropped event takes no seq
        assert [f["data"]["stage"] for f in frames if f["event"] == "stage_end"] == STAGES[:-1]   # no score in public
        start = frames[0]["data"]
        assert start["mode"] == "public" and "reference" not in start["options"] and "test_row" not in start["options"]
        assert sorted(start["image"]["urls"]) == ["model_input", "thumb"]
        warnings = [(f["data"]["code"], f["data"]["message"]) for f in frames if f["event"] == "warning"]
        assert warnings == [("reference_ignored_public", REFERENCE_IGNORED_PUBLIC)]
        assert frames[-1]["data"]["status"] == "done"
        stored = c.get("/v1/messages/{}".format(start["message_id"]), headers=PUBLIC).json()
        assert [(e["event"], e["data"]) for e in stored["events"]] == [(f["event"], f["data"]) for f in frames]
        everything = [json.dumps(frames), json.dumps(stored), c.get("/v1/sessions/{}".format(sid), headers=PUBLIC).text]
        everything += [c.get("/v1/sessions/{}/export?format={}".format(sid, f), headers=PUBLIC).text
                       for f in ("json", "md")]
        assert not [text for text in everything if "my own secret line" in text]


def test_public_cards_and_provenance_carry_the_checkpoint_file_name_only(tmp_path, monkeypatch):
    _fake_real_engines(monkeypatch)
    with TestClient(create_app(engine="real", home=str(tmp_path), mode="public", token="t")) as c:
        assert c.get("/v1/models", headers=PUBLIC).json()["models"][0]["checkpoint"] == "last.ckpt"
        sid = c.post("/v1/sessions", json={}, headers=PUBLIC).json()["id"]
        frames = _public_turn(c, sid)
        assert frames[0]["data"]["model"]["checkpoint"] == "last.ckpt"
        assistant = c.get("/v1/sessions/{}".format(sid), headers=PUBLIC).json()["messages"][1]
        assert assistant["provenance"]["checkpoint"] == "last.ckpt"   # ruling 13: finish_turn stores redact_card
        md = c.get("/v1/sessions/{}/export?format=md".format(sid), headers=PUBLIC).text
        assert str(REPO_ROOT) not in md and str(REPO_ROOT) not in json.dumps(frames)


def test_a_turn_runs_in_its_own_sessions_mode(client):
    # A CHAT_HOME used in both modes: a private server lists the public sessions too, and must redact their turns.
    session = client.app.state.store.create_session("public", "someone")
    frames = _turn(client, session["id"], options={"max_new_tokens": 16, "reference": "Findings: typed."})
    assert frames[0]["data"]["mode"] == "public" and "original" not in frames[0]["data"]["image"]["urls"]
    assert [f["data"]["code"] for f in frames if f["event"] == "warning"] == ["reference_ignored_public"]
    assert "score" not in _stage_ends(frames)


# ---- ruling 8: the worker never dies --------------------------------------------------------------------------------

def _fail_once(monkeypatch, engine, method, exc):
    real, calls = getattr(engine, method), []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise exc
        return real(*args, **kwargs)

    monkeypatch.setattr(engine, method, flaky)


@pytest.mark.parametrize("exc_type", [RuntimeError, ValueError, KeyError])
def test_an_engine_failure_is_a_model_error_without_its_text_and_the_worker_lives_on(client, monkeypatch, caplog,
                                                                                     exc_type):
    _fail_once(monkeypatch, client.app.state.engines["tiny"], "generate",
               exc_type("Findings: secret text from /sc/home/someone/outputs"))
    sid = client.post("/v1/sessions", json={}).json()["id"]
    frames = _turn(client, sid)
    errors = [f["data"] for f in frames if f["event"] == "error"]
    assert len(errors) == 1 and set(errors[0]) == {"type", "error", "seq"} and errors[0]["type"] == "error"
    assert errors[0]["error"] == {"type": "model_error", "message": "Internal error ({})".format(exc_type.__name__)}
    stop = frames[-1]
    assert stop["event"] == "message_stop" and stop["data"]["status"] == "error" and stop["data"]["report"] is None
    stored = client.get("/v1/messages/{}".format(frames[0]["data"]["message_id"])).json()
    assert stored["status"] == "error" and "secret" not in json.dumps(stored) and "secret" not in json.dumps(frames)
    assert any(r.exc_info for r in caplog.records if r.name == "app.pipeline")   # the traceback: server-side only
    assert _turn(client, sid)[-1]["data"]["status"] == "done"
    assert client.get("/healthz").json()["turns_in_flight"] == 0


def test_an_upload_error_inside_the_turn_is_a_validation_error_event(client, monkeypatch):
    _fail_once(monkeypatch, client.app.state.engines["tiny"], "preprocess", UploadError(UNREADABLE_MSG))
    sid = client.post("/v1/sessions", json={}).json()["id"]
    frames = _turn(client, sid)
    assert [f["data"]["error"] for f in frames if f["event"] == "error"] == [
        {"type": "validation_error", "message": UNREADABLE_MSG}]
    assert frames[-1]["event"] == "message_stop" and frames[-1]["data"]["status"] == "error"
    assert _turn(client, sid)[-1]["data"]["status"] == "done"


def test_public_engine_failure_is_a_bare_error_body_with_the_public_message(tmp_path, monkeypatch):   # ruling 13
    app = create_app(engine="tiny", home=str(tmp_path), mode="public", token="t")
    _fail_once(monkeypatch, app.state.engines["tiny"], "generate", RuntimeError("Findings: secret at /sc/home/u/x"))
    with TestClient(app) as c:
        sid = c.post("/v1/sessions", json={}, headers=PUBLIC).json()["id"]
        frames = _public_turn(c, sid)
        errors = [f["data"] for f in frames if f["event"] == "error"]
        assert len(errors) == 1 and set(errors[0]) == {"type", "error", "seq"} and errors[0]["type"] == "error"
        assert errors[0]["error"] == {"type": "model_error", "message": PUBLIC_ERROR_MESSAGE}
        assert frames[-1]["event"] == "message_stop" and frames[-1]["data"]["status"] == "error"
        assert "secret" not in json.dumps(frames)
        assert _public_turn(c, sid)[-1]["data"]["status"] == "done"   # the worker serves the next turn


def test_a_session_deleted_before_its_turn_runs_ends_that_turn_quietly(live, caplog):
    sid_a, sid_b = _new_session(live), _new_session(live)
    a_generating, b_opened, frames = threading.Event(), [], {"a": [], "b": []}

    def run(name, sid, tokens):
        with httpx.stream("POST", live + "/v1/sessions/{}/messages".format(sid), timeout=60,
                          files={"image": ("x.png", png_bytes(), "image/png")},
                          data={"text": "", "options": json.dumps({"max_new_tokens": tokens})}) as r:
            if name == "b":
                b_opened.append(r.status_code)
            for frame in iter_sse(r.iter_text()):
                frames[name].append(frame)
                if frame["event"] == "content_block_delta":
                    a_generating.set()

    a = threading.Thread(target=run, args=("a", sid_a, 150))
    a.start()
    assert a_generating.wait(10)                  # A holds the one worker, so B can only wait in the queue
    b = threading.Thread(target=run, args=("b", sid_b, 16))
    b.start()
    assert wait_until(lambda: b_opened) == [200]  # B is accepted
    assert httpx.delete(live + "/v1/sessions/{}".format(sid_b)).status_code == 204
    httpx.post(live + "/v1/messages/{}/cancel".format(frames["a"][0]["data"]["message_id"]))
    a.join(30)
    b.join(30)
    assert [f["event"] for f in frames["b"]] == ["message_start", "stage_start", "message_stop"]
    assert frames["b"][-1]["data"]["status"] == "aborted"   # its upload could not be stored: no error event
    assert any("deleted" in r.getMessage() for r in caplog.records if r.name == "app.pipeline")
    assert httpx.get(live + "/healthz").json()["turns_in_flight"] == 0
    sid_c = _new_session(live)
    with httpx.stream("POST", live + "/v1/sessions/{}/messages".format(sid_c), timeout=30,
                      files={"image": ("x.png", png_bytes(), "image/png")},
                      data={"text": "", "options": json.dumps({"max_new_tokens": 16})}) as r:
        assert list(iter_sse(r.iter_text()))[-1]["data"]["status"] == "done"


def test_a_text_only_turn_whose_session_was_deleted_ends_quietly(client, caplog):
    sid = client.post("/v1/sessions", json={}).json()["id"]
    sha = _turn(client, sid)[0]["data"]["image"]["sha256"]
    store, pipeline = client.app.state.store, client.app.state.worker.pipeline
    uid, mid = store.start_turn(sid, "greedy", "private", {}, sha, "x.png")   # accepted, then deleted in the queue
    assert store.delete_session(sid, None)
    sent = []
    job = TurnJob(sid, uid, mid, "greedy", None, "x.png", Options(decode="greedy"), mode="private",
                  previous_sha256=sha)
    pipeline.run(job, lambda event, data: sent.append((event, data)), threading.Event())
    assert [e for e, _ in sent] == ["message_start", "stage_start", "message_stop"]
    assert sent[-1][1]["status"] == "aborted" and sent[0][1]["image"]["source"] == "previous"
    assert [e["event"] for e in store.events_after(mid)] == ["message_start", "stage_start", "message_stop"]
    assert any("deleted" in r.getMessage() for r in caplog.records if r.name == "app.pipeline")


def test_the_runner_never_raises_even_when_the_store_is_gone(tmp_path, caplog):
    store = Store(tmp_path)
    pipeline = Pipeline({"tiny": build_engine("tiny")}, "tiny", store, "private")
    session = store.create_session("private")
    uid, mid = store.start_turn(session["id"], "", "private", {})
    store.close()   # every store call now fails
    sent = []
    job = TurnJob(session["id"], uid, mid, "", png_bytes(), "x.png", Options(max_new_tokens=16), mode="private")
    pipeline.run(job, lambda event, data: sent.append(event), threading.Event())   # returns: the worker lives on
    assert sent == [] and any(r.exc_info for r in caplog.records if r.name == "app.pipeline")


# ---- rulings 9 and 10: the overload count and cancel ----------------------------------------------------------------

def test_the_worker_admits_at_most_cap_unfinished_turns():
    worker = Worker(pipeline=None, cap=2)
    try:
        assert [worker.reserve() for _ in range(3)] == [True, True, False]
        worker.release()
        assert worker.reserve() is True and worker.in_flight == 2
    finally:
        worker.shutdown()


def test_turns_run_one_at_a_time_on_the_single_worker(live):
    sid = _new_session(live)
    done = []

    def one():
        with httpx.stream("POST", live + "/v1/sessions/{}/messages".format(sid), timeout=60,
                          files={"image": ("x.png", png_bytes(), "image/png")},
                          data={"text": "", "options": json.dumps({"max_new_tokens": 16})}) as r:
            done.append(list(iter_sse(r.iter_text()))[-1]["data"]["status"])

    threads = [threading.Thread(target=one) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert done == ["done", "done"]
    turns = [m["events"] for m in httpx.get(live + "/v1/sessions/{}/export?format=json".format(sid)).json()["messages"]
             if m["role"] == "assistant"]
    first, second = sorted(turns, key=lambda events: events[0]["ts"])
    assert first[-1]["ts"] <= second[0]["ts"]   # the second turn started after the first had stopped


def test_cancel_is_scoped_and_idempotent(tmp_path):
    with TestClient(create_app(engine="tiny", home=str(tmp_path), mode="public", token="t")) as c:
        sid = c.post("/v1/sessions", json={}, headers=PUBLIC).json()["id"]
        frames = _public_turn(c, sid)
        mid, uid = frames[0]["data"]["message_id"], frames[0]["data"]["user_message_id"]
        other = dict(PUBLIC, **{"X-Client-Id": "b"})
        assert c.post("/v1/messages/{}/cancel".format(mid), headers=other).status_code == 404
        for _ in range(2):
            r = c.post("/v1/messages/{}/cancel".format(mid), headers=PUBLIC)
            assert r.status_code == 200 and r.json() == {"id": mid, "status": "done", "cancel_requested": False}
        assert c.post("/v1/messages/{}/cancel".format(uid), headers=PUBLIC).json()["cancel_requested"] is False


def test_a_queued_turn_can_be_cancelled_by_the_id_its_response_header_carries(live):
    sid = _new_session(live)
    a_generating, ids, frames = threading.Event(), {}, {"a": [], "b": []}

    def run(name, tokens):
        with httpx.stream("POST", live + "/v1/sessions/{}/messages".format(sid), timeout=60,
                          files={"image": ("x.png", png_bytes(), "image/png")},
                          data={"text": "", "options": json.dumps({"max_new_tokens": tokens})}) as r:
            ids[name] = r.headers["X-Message-Id"]   # known before the turn reaches the worker
            for frame in iter_sse(r.iter_text()):
                frames[name].append(frame)
                if frame["event"] == "content_block_delta":
                    a_generating.set()

    a = threading.Thread(target=run, args=("a", 150))
    a.start()
    assert a_generating.wait(10)
    b = threading.Thread(target=run, args=("b", 16))
    b.start()
    b_id = wait_until(lambda: ids.get("b"))
    r = httpx.post(live + "/v1/messages/{}/cancel".format(b_id))
    assert r.json() == {"id": b_id, "status": "running", "cancel_requested": True}
    httpx.post(live + "/v1/messages/{}/cancel".format(ids["a"]))
    a.join(30)
    b.join(30)
    assert frames["a"][0]["data"]["message_id"] == ids["a"] and frames["b"][0]["data"]["message_id"] == b_id
    assert [f["event"] for f in frames["b"]] == ["message_start", "message_stop"]   # no stage ran
    assert frames["b"][-1]["data"]["status"] == "aborted"
    assert httpx.get(live + "/v1/messages/{}".format(b_id)).json()["status"] == "aborted"


# ---- ruling 11: reading a message; ruling 13: message_stop's fields -------------------------------------------------

def test_message_read_returns_the_stored_message_and_its_events_after_a_seq(client):
    sid = client.post("/v1/sessions", json={}).json()["id"]
    frames = _turn(client, sid)
    mid = frames[0]["data"]["message_id"]
    msg = client.get("/v1/messages/{}".format(mid)).json()
    stop = frames[-1]["data"]
    assert {k: msg[k] for k in ("id", "session_id", "role", "status", "report", "display_report")} == {
        "id": mid, "session_id": sid, "role": "assistant", "status": "done", "report": stop["report"],
        "display_report": stop["display_report"]}
    assert msg["report"] and msg["total_ms"] == stop["total_ms"]
    assert client.get("/v1/messages/{}?after={}".format(mid, len(frames))).json()["events"] == []
    r = client.get("/v1/messages/{}?after=-1".format(mid))
    assert r.status_code == 422 and r.json()["error"]["type"] == "validation_error"


def test_a_turn_is_finished_in_the_store_before_its_message_stop_is_sent(client, monkeypatch):
    # A client that acts on message_stop (reads the message, sends the next turn) must find the final status.
    store, sent, stops_sent_at_finish = client.app.state.store, [], []
    push, finish = server._push, store.finish_turn

    def recording_push(loop, queue, item):
        sent.append(item)
        push(loop, queue, item)

    def recording_finish(*args, **kwargs):
        stops_sent_at_finish.append([f for f in sent if f and f.startswith("event: message_stop")])
        return finish(*args, **kwargs)

    monkeypatch.setattr(server, "_push", recording_push)
    monkeypatch.setattr(store, "finish_turn", recording_finish)
    frames = _turn(client, client.post("/v1/sessions", json={}).json()["id"])
    assert stops_sent_at_finish == [[]] and frames[-1]["event"] == "message_stop"
    assert sent[-1] is None   # the end-of-turn sentinel comes last


def test_message_stop_carries_exactly_the_contract_fields(client):
    sid = client.post("/v1/sessions", json={}).json()["id"]
    stop = _turn(client, sid)[-1]["data"]
    assert set(stop) == {"message_id", "status", "total_ms", "report", "display_report", "truncated_mid_sentence",
                         "disclaimer", "seq"}
    assert stop["report"] and stop["total_ms"] > 0 and isinstance(stop["truncated_mid_sentence"], bool)


# ---- the turn: message_start, uploads, text-only turns, commands ----------------------------------------------------

def test_message_start_names_the_turn_and_points_at_the_user_messages_image(client):
    sid = client.post("/v1/sessions", json={}).json()["id"]
    frames = _turn(client, sid, options={"max_new_tokens": 16, "beam_size": 2})
    start = frames[0]["data"]
    user, assistant = client.get("/v1/sessions/{}".format(sid)).json()["messages"]
    assert (start["message_id"], start["user_message_id"], start["session_id"]) == (assistant["id"], user["id"], sid)
    assert start["mode"] == "private" and start["model"]["name"] == "tiny"
    assert start["options"] == dict(Options(max_new_tokens=16, beam_size=2).model_dump(), model="tiny")
    assert start["image"] == {"sha256": user["image_sha256"], "filename": "x.png", "source": "upload",
                              "urls": {v: "/v1/messages/{}/image?variant={}".format(user["id"], v)
                                       for v in URL_VARIANTS}}
    pre = _stage_ends(frames)["preprocess"]["detail"]
    assert pre["source"] == "upload" and pre["image_sha256"] == user["image_sha256"] and pre["input_px"] == [320, 320]
    assert assistant["options"] == start["options"]


def test_preprocess_stores_the_upload_once_per_session_and_hash(client):
    sid = client.post("/v1/sessions", json={}).json()["id"]
    sha = _turn(client, sid)[0]["data"]["image"]["sha256"]
    store = client.app.state.store
    assert store.upload_path(sid, sha, "original").read_bytes() == png_bytes()   # the bytes as uploaded
    assert Image.open(store.upload_path(sid, sha, "thumb")).format == "JPEG"
    model_input = Image.open(store.upload_path(sid, sha, "model_input"))
    assert model_input.format == "PNG" and model_input.size == (224, 224)
    stamps = [store.upload_path(sid, sha, v).stat().st_mtime_ns for v in URL_VARIANTS]
    _turn(client, sid)   # the same bytes again: the first copy stands
    assert [store.upload_path(sid, sha, v).stat().st_mtime_ns for v in URL_VARIANTS] == stamps


def test_a_text_only_turn_reruns_the_stored_original_under_its_own_message(client):
    sid = client.post("/v1/sessions", json={}).json()["id"]
    sha = _turn(client, sid)[0]["data"]["image"]["sha256"]
    frames = _turn(client, sid, image=False, text="tokens 20")
    start = frames[0]["data"]
    assert start["image"] == {"sha256": sha, "filename": "x.png", "source": "previous",
                              "urls": {v: "/v1/messages/{}/image?variant={}".format(start["user_message_id"], v)
                                       for v in URL_VARIANTS}}
    ends = _stage_ends(frames)
    assert ends["preprocess"]["detail"]["image_sha256"] == sha and ends["preprocess"]["detail"]["source"] == "previous"
    assert ends["generate"]["detail"]["tokens"] == 20 and frames[-1]["data"]["status"] == "done"
    user = client.get("/v1/sessions/{}".format(sid)).json()["messages"][2]
    assert (user["role"], user["text"], user["image_sha256"], user["image_filename"]) == (
        "user", "tokens 20", sha, "x.png")


def test_a_question_is_answered_without_running_the_model(client):
    sid = client.post("/v1/sessions", json={}).json()["id"]
    _turn(client, sid)
    frames = _turn(client, sid, image=False, text="is this pneumonia?")
    assert [f["event"] for f in frames] == ["message_start", "warning", "message_stop"]
    assert frames[0]["data"]["image"] is None
    assert frames[1]["data"] == {"code": "not_a_command", "message": NOT_A_QA_BOT, "seq": 2}
    assert frames[-1]["data"]["status"] == "done" and frames[-1]["data"]["report"] is None
    user, assistant = client.get("/v1/sessions/{}".format(sid)).json()["messages"][2:]
    assert user["text"] == "is this pneumonia?" and user["image_sha256"] is None   # no image was used
    assert assistant["status"] == "done" and assistant["report"] is None and assistant["provenance"] is None
    assert _turn(client, sid, image=False, text="greedy")[0]["data"]["image"]["source"] == "previous"


def test_text_with_an_image_is_a_note_and_a_command_in_it_applies(client):
    sid = client.post("/v1/sessions", json={}).json()["id"]
    frames = _turn(client, sid, text="greedy")
    assert frames[0]["data"]["options"]["decode"] == "greedy" and frames[0]["data"]["image"]["source"] == "upload"
    assert _stage_ends(frames)["generate"]["detail"]["beam_size"] == 1
    frames = _turn(client, sid, text="Cough for three weeks.")
    assert not [f for f in frames if f["event"] == "warning"] and frames[-1]["data"]["status"] == "done"
    assert frames[0]["data"]["options"]["decode"] == "beam"
    user = client.get("/v1/sessions/{}".format(sid)).json()["messages"][2]
    assert user["text"] == "Cough for three weeks." and user["image_filename"] == "x.png"


def test_a_previous_upload_that_is_gone_from_disk_is_no_image(client, tmp_path):
    sid = client.post("/v1/sessions", json={}).json()["id"]
    _turn(client, sid)
    shutil.rmtree(tmp_path / "uploads" / sid)
    r = client.post("/v1/sessions/{}/messages".format(sid), data={"text": "greedy", "options": "{}"})
    assert r.status_code == 422 and r.json()["error"]["message"] == "Attach an X-ray first."


def test_sessions_are_created_listed_and_titled(client):
    a = client.post("/v1/sessions", json={"title": "  Night   shift  "}).json()
    b = client.post("/v1/sessions").json()   # the body is optional
    assert a["title"] == "Night shift" and b["title"] == "" and a["mode"] == "private"
    assert client.post("/v1/sessions", json={"mode": "public"}).status_code == 422
    listed = client.get("/v1/sessions").json()
    assert [s["id"] for s in listed["sessions"]] == [b["id"], a["id"]] and listed["next_cursor"] is None
    page = client.get("/v1/sessions?limit=1").json()
    assert [s["id"] for s in page["sessions"]] == [b["id"]] and page["next_cursor"] == b["id"]
    _turn(client, b["id"])
    assert client.get("/v1/sessions/{}".format(b["id"])).json()["title"] == "x.png"


def test_markdown_export_has_a_heading_per_turn_and_exports_download(client):
    sid = client.post("/v1/sessions", json={}).json()["id"]
    _turn(client, sid)
    _turn(client, sid, image=False, text="greedy")
    md = client.get("/v1/sessions/{}/export?format=md".format(sid))
    assert md.text.startswith("# Session {}\n".format(sid))
    assert [line[:10] for line in md.text.splitlines() if line.startswith("## Turn")] == ["## Turn 1 ", "## Turn 2 "]
    assert md.headers["content-disposition"] == 'attachment; filename="session-{}.md"'.format(sid)
    r = client.get("/v1/sessions/{}/export?format=pdf".format(sid))
    assert r.status_code == 422 and r.json()["error"]["type"] == "validation_error"


def test_health_reports_mode_and_load_without_a_token(tmp_path):
    with TestClient(create_app(engine="tiny", home=str(tmp_path), token="t", queue_cap=3)) as c:
        assert c.get("/healthz").json() == {"status": "ok", "mode": "private", "default_model": "tiny",
                                            "turns_in_flight": 0, "queue_cap": 3}


# ---- the SSE bridge -----------------------------------------------------------------------------------------------

def test_sse_frames_are_one_compact_json_line():
    frame = server.sse("warning", {"code": "x", "message": "é"})
    assert frame == 'event: warning\ndata: {"code":"x","message":"é"}\n\n'


def test_the_bridge_pings_while_idle_and_ends_on_the_sentinel(monkeypatch):
    monkeypatch.setattr(server, "PING_S", 0.01)

    async def drain():   # every await is bounded: a broken bridge fails here instead of hanging the suite
        queue = asyncio.Queue()
        frames = server._frames(queue)

        async def rest():
            return [f async for f in frames]

        idle = await asyncio.wait_for(frames.__anext__(), 5)   # nothing queued yet: a keep-alive, never stored
        queue.put_nowait("event: x\ndata: {}\n\n")
        frame = await asyncio.wait_for(frames.__anext__(), 5)
        queue.put_nowait(None)                                  # the worker finished the turn
        return idle, frame, await asyncio.wait_for(rest(), 5)

    assert asyncio.run(drain()) == (": ping\n\n", "event: x\ndata: {}\n\n", [])


def test_a_frame_for_a_closed_loop_is_dropped_quietly():
    loop = asyncio.new_event_loop()
    queue = asyncio.Queue()
    loop.close()
    server._push(loop, queue, "event: x\ndata: {}\n\n")   # no RuntimeError: the event is stored already
    assert queue.empty()
