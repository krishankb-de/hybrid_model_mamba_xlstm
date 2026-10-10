"""CHAT_UI_PLAN.md P6-B: the images of a turn, as the chat page loads them back (GET /v1/messages/{id}/image).

The tiny engine, and the synthetic tiny gallery where a test needs a test-split study (scripts/build_retrieval_gallery.py --tiny, its gate
decided as the cluster decides it): synthetic data only (R7), so an assertion may print whatever it compares.
"""
import io
import json
import logging
import shutil

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageChops

from app.gallery import Gallery
from app.imaging import load_upload, model_input_image
from app.labels import RuleLabeler
from app.pipeline import URL_VARIANTS, image_urls
from app.server import NO_GALLERY_MSG, PUBLIC_TEST_SPLIT_MSG, TEST_IMAGE_MSG, create_app
from tests.app_helpers import decide_gate, iter_sse, png_bytes

PUBLIC = {"Authorization": "Bearer t", "X-Client-Id": "a"}
OTHER = {"Authorization": "Bearer t", "X-Client-Id": "b"}
UPLOAD_CACHE = "private, max-age=3600"   # a user's own upload
NO_STORE = "no-store"                    # anything MIMIC-derived
TYPES = {"original": "image/png", "thumb": "image/jpeg", "model_input": "image/png"}   # for a PNG upload


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(engine="tiny", home=str(tmp_path / "home"))) as c:
        yield c


@pytest.fixture
def gallery(tiny_gallery):
    return Gallery.open(decide_gate(tiny_gallery), None)


@pytest.fixture
def gclient(tmp_path, gallery):
    with TestClient(create_app(engine="tiny", home=str(tmp_path / "ghome"), gallery=gallery, labeler=RuleLabeler())) as c:
        yield c


def _turn(c, image=None, options=None, text="", headers=None, sid=None):
    """One turn, read to its end; -> (its frames, its session id). image None sends png_bytes(); b"" sends none."""
    sid = sid or c.post("/v1/sessions", json={}, headers=headers).json()["id"]
    files = None if image == b"" else {"image": ("x.png", png_bytes() if image is None else image, "image/png")}
    data = {"text": text, "options": json.dumps({"max_new_tokens": 16} if options is None else options)}
    r = c.post("/v1/sessions/{}/messages".format(sid), files=files, data=data, headers=headers)
    assert r.status_code == 200, r.text
    return list(iter_sse([r.text])), sid


def _image(c, message_id, variant, headers=None):
    return c.get("/v1/messages/{}/image".format(message_id), params={"variant": variant}, headers=headers)


def _error(r, status, kind, message=None):
    assert r.status_code == status, (r.status_code, r.text)
    body = r.json()
    assert body["type"] == "error" and body["error"]["type"] == kind, body
    if message is not None:
        assert body["error"]["message"] == message, body


# ---- an upload -----------------------------------------------------------------------------------------------------------------

def test_each_variant_of_an_upload_is_served_by_the_turns_user_and_assistant_message_id(client):
    upload = png_bytes(400, 300)
    frames, sid = _turn(client, upload)
    start = frames[0]["data"]
    store = client.app.state.store
    for message_id in (start["user_message_id"], start["message_id"]):   # either message of the turn names its image
        for variant in URL_VARIANTS:
            r = _image(client, message_id, variant)
            assert r.status_code == 200, (message_id, variant, r.text)
            assert r.headers["content-type"] == TYPES[variant], variant
            assert r.headers["cache-control"] == UPLOAD_CACHE, variant
            assert r.content == store.upload_path(sid, start["image"]["sha256"], variant).read_bytes(), variant
        assert _image(client, message_id, "original").content == upload   # the bytes as uploaded
        thumb = Image.open(io.BytesIO(_image(client, message_id, "thumb").content))
        assert thumb.format == "JPEG" and thumb.size == (400, 300)        # under 512 px: the thumbnail keeps the size
        model_input = Image.open(io.BytesIO(_image(client, message_id, "model_input").content))
        assert model_input.format == "PNG" and model_input.size == (224, 224)   # what the tower saw


def test_a_large_uploads_thumbnail_is_512_px_on_its_long_side(client):
    frames, _ = _turn(client, png_bytes(1200, 800))
    thumb = Image.open(io.BytesIO(_image(client, frames[0]["data"]["user_message_id"], "thumb").content))
    assert thumb.size == (512, 341)


def test_the_urls_a_turn_announces_are_the_ones_that_serve_it(client):
    frames, _ = _turn(client)
    image = frames[0]["data"]["image"]
    assert image["urls"] == image_urls(frames[0]["data"]["user_message_id"])
    for variant, url in image["urls"].items():
        r = client.get(url)
        assert r.status_code == 200 and r.headers["content-type"] == TYPES[variant], (variant, r.text)


def test_after_a_reload_the_user_message_carries_the_three_urls_and_they_still_resolve(client):
    frames, sid = _turn(client)
    served = {v: _image(client, frames[0]["data"]["user_message_id"], v).content for v in URL_VARIANTS}
    _turn(client, b"", text="is it pneumonia?", sid=sid)   # a question: no model runs, no image is used
    user, assistant, question, answer = client.get("/v1/sessions/{}".format(sid)).json()["messages"]
    assert user["image_urls"] == image_urls(user["id"]) == frames[0]["data"]["image"]["urls"]
    for variant, url in user["image_urls"].items():
        r = client.get(url)
        assert r.status_code == 200 and r.content == served[variant], variant
    assert question["image_urls"] is None                        # a turn with no image has none to show
    assert "image_urls" not in assistant and "image_urls" not in answer   # an assistant message's image is its user message's


def test_a_text_only_rerun_serves_the_image_it_ran_again(client):
    first, sid = _turn(client, png_bytes(256, 256))
    again, _ = _turn(client, b"", text="greedy", sid=sid)
    start = again[0]["data"]
    assert start["image"]["source"] == "previous"
    for message_id in (start["user_message_id"], start["message_id"]):
        assert _image(client, message_id, "original").content == png_bytes(256, 256)
    user = client.get("/v1/sessions/{}".format(sid)).json()["messages"][2]
    assert user["image_urls"] == image_urls(user["id"])


def test_an_unknown_or_missing_variant_is_a_422_and_an_unknown_message_a_404(client):
    frames, _ = _turn(client)
    uid = frames[0]["data"]["user_message_id"]
    for params in ({"variant": "huge"}, {"variant": ""}, {"variant": "../original"}, {}):
        r = client.get("/v1/messages/{}/image".format(uid), params=params)
        _error(r, 422, "validation_error")
        assert r.json()["error"]["message"].startswith("Invalid request: variant: "), params
    _error(_image(client, "m_missing", "thumb"), 404, "not_found_error", "Message not found.")


def test_a_question_has_no_image_to_serve(client):
    frames, sid = _turn(client)
    question, _ = _turn(client, b"", text="is it pneumonia?", sid=sid)
    for message_id in (question[0]["data"]["user_message_id"], question[0]["data"]["message_id"]):
        _error(_image(client, message_id, "thumb"), 404, "not_found_error", "This turn has no image.")


def test_after_delete_the_image_is_a_404(client):
    frames, sid = _turn(client)
    start = frames[0]["data"]
    assert _image(client, start["user_message_id"], "thumb").status_code == 200
    assert client.delete("/v1/sessions/{}".format(sid)).status_code == 204
    for message_id in (start["user_message_id"], start["message_id"]):
        for variant in URL_VARIANTS:
            _error(_image(client, message_id, variant), 404, "not_found_error", "Message not found.")


def test_an_upload_gone_from_disk_is_a_404(client, tmp_path):
    frames, sid = _turn(client)
    shutil.rmtree(tmp_path / "home" / "uploads" / sid)
    _error(_image(client, frames[0]["data"]["user_message_id"], "thumb"), 404, "not_found_error", "This turn's image is no longer stored.")


def test_public_mode_serves_a_clients_own_upload_and_hides_everyone_elses(tmp_path):
    with TestClient(create_app(engine="tiny", home=str(tmp_path), mode="public", token="t")) as c:
        frames, sid = _turn(c, headers=PUBLIC)
        start = frames[0]["data"]
        assert sorted(start["image"]["urls"]) == ["model_input", "thumb"]   # app/redact.py: no original in public mode
        for variant in ("thumb", "model_input"):
            r = _image(c, start["user_message_id"], variant, PUBLIC)
            assert r.status_code == 200 and r.headers["cache-control"] == UPLOAD_CACHE, variant   # a user's own upload, in both modes
            _error(_image(c, start["user_message_id"], variant, OTHER), 404, "not_found_error", "Message not found.")
        _error(_image(c, start["user_message_id"], "thumb", {"Authorization": "Bearer t"}), 400, "invalid_request_error")
        user = c.get("/v1/sessions/{}".format(sid), headers=PUBLIC).json()["messages"][0]
        assert user["image_urls"] == start["image"]["urls"]   # the reload's URLs go through the same public policy


# ---- a test-split study (private mode only) ------------------------------------------------------------------------------------------

def test_a_test_split_turns_image_is_read_from_the_dataset_and_sent_with_no_store(gclient, gallery):
    row = 3
    frames, sid = _turn(gclient, b"", {"max_new_tokens": 16, "test_row": row})
    start = frames[0]["data"]
    assert start["image"]["source"] == "test_split"
    data = gallery.test_study(row)["image"].read_bytes()
    expected_input = model_input_image(load_upload(data)[0])
    for message_id in (start["user_message_id"], start["message_id"]):
        for variant in URL_VARIANTS:
            r = _image(gclient, message_id, variant)
            assert r.status_code == 200 and r.headers["cache-control"] == NO_STORE, (variant, r.text)
        original, thumb = _image(gclient, message_id, "original"), _image(gclient, message_id, "thumb")
        assert original.headers["content-type"] == thumb.headers["content-type"] == "image/jpeg"
        assert original.content == thumb.content == data   # the 320 px dataset JPEG is both: never copied, never resized
        made = Image.open(io.BytesIO(_image(gclient, message_id, "model_input").content))
        assert made.format == "PNG" and made.size == (224, 224)
        assert ImageChops.difference(made.convert("RGB"), expected_input.convert("RGB")).getbbox() is None   # made in memory, as preprocess makes it
    user = gclient.get("/v1/sessions/{}".format(sid)).json()["messages"][0]
    assert user["test_row"] == row and user["image_urls"] == image_urls(user["id"])
    again, _ = _turn(gclient, b"", text="beam 2", sid=sid)   # a follow-up reruns the study: its image is the study's too
    assert _image(gclient, again[0]["data"]["user_message_id"], "original").content == data


def test_a_test_split_image_is_refused_in_public_mode_even_where_a_test_row_was_planted(tmp_path, gallery):
    # start_turn never stores a test row in a public session (R1), so the row is planted here: the image route refuses it by itself.
    home = str(tmp_path / "shared")
    with TestClient(create_app(engine="tiny", home=home, mode="public", token="t", gallery=gallery)) as c:
        store = c.app.state.store
        sid = c.post("/v1/sessions", json={}, headers=PUBLIC).json()["id"]
        uid, mid = store.start_turn(sid, "", "public", {}, "ab" * 32, None)
        store._con.execute("UPDATE messages SET test_row = 0 WHERE id = ?", (uid,))
        for message_id in (uid, mid):
            for variant in URL_VARIANTS:
                _error(_image(c, message_id, variant, PUBLIC), 403, "permission_error", PUBLIC_TEST_SPLIT_MSG)
    with TestClient(create_app(engine="tiny", home=home, gallery=gallery)) as c:   # a private server that lists that public session
        for variant in URL_VARIANTS:
            _error(_image(c, uid, variant), 403, "permission_error", PUBLIC_TEST_SPLIT_MSG)


def test_a_test_split_turn_on_a_server_without_a_gallery_is_a_503(client):
    store = client.app.state.store
    sid = client.post("/v1/sessions", json={}).json()["id"]
    uid, _ = store.start_turn(sid, "", "private", {}, "ab" * 32, None, 3)   # as a server with the gallery stored it
    _error(_image(client, uid, "thumb"), 503, "unavailable_error", NO_GALLERY_MSG)


def test_a_test_split_image_that_cannot_be_read_is_a_500_and_no_path_is_logged(gclient, gallery, caplog):
    frames, _ = _turn(gclient, b"", {"max_new_tokens": 16, "test_row": 4})
    path = gallery.test_study(4)["image"]
    path.unlink()
    with caplog.at_level(logging.DEBUG):
        for variant in URL_VARIANTS:
            _error(_image(gclient, frames[0]["data"]["user_message_id"], variant), 500, "internal_error", TEST_IMAGE_MSG)
    assert caplog.records and str(path) not in caplog.text and path.name not in caplog.text
