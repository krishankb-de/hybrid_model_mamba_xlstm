"""CHAT_UI_PLAN.md P4-H: the chat page driven like a user, in a real Chrome (Playwright), on the tiny engine.

One test per journey (P4-H brief, B3). Each checks what the user sees and what the server stored, and ends with UI.assert_clean: no
console error, no uncaught page error, and no HTTP 4xx or 5xx but the refusals the test provokes on purpose. The harness, the server
and the synthetic images are in conftest.py.

    venv/bin/python -m pytest tests/e2e -m e2e -q
"""
import json
import math
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Tuple

import pytest

pytest.importorskip("playwright.sync_api", reason="the e2e tests need `pip install -r requirements-e2e.txt`")

from app.commands import NOT_A_QA_BOT  # noqa: E402
from app.labels import CHEXBERT_14  # noqa: E402
from app.imaging import FORMATS_MSG, TOO_SMALL_MSG  # noqa: E402
from tests.e2e.conftest import DESKTOP, FAULT_TOKENS, REPO_ROOT, squeezed, stage_detail  # noqa: E402

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

BANNER = "Research prototype — not for clinical use."
EMPTY_STATE = "Attach a chest X-ray below to generate a report."
REPEAT_NOTE = "Stopped when the model began repeating itself."
STOPPED_NOTE = "Turn stopped"
SAVED = "Settings saved. They apply from your next Send."
TOKENS_RANGE = "Enter a whole number from 16 to 200."
BEAM_RANGE = "Enter a whole number from 1 to 8."
RERUN = "No new image: Send re-runs {} with these settings."
QUESTION_HINT = "This note is not a command: with no new image, Send gets no report."
INTERNAL_ERROR = "Error: Internal error (ImportError)"
INTERNAL_HINT = "The server hit an internal error. If you just updated the code, restart the server."
RESTART_ERROR = "Error: The server restarted while this turn was running."
NOT_AN_IMAGE = "Use a PNG, JPEG or WEBP image."   # the page's own refusals (app/static/composer.js), before anything is sent
TOO_BIG = "Image is larger than 20 MB."
DICOM = "DICOM is not supported. Export the image as PNG or JPEG first."
SETTLED_STAGES = {"preprocess": "done", "encode": "done", "retrieve": "skipped", "generate": "done", "label": "skipped",
                  "score": "skipped"}
STAGE_RANK = {"pending": 0, "running": 1, "done": 2, "skipped": 2, "error": 2}

BUBBLE_JS = """() => {
  const u = [...document.querySelectorAll('#conversation article.turn.user')].pop();
  if (!u) return null;
  const img = u.querySelector('img.thumb'), text = u.querySelector('.user-text');
  return { text: text ? text.textContent : '', chips: [...u.querySelectorAll('.options .chip')].map((c) => c.textContent),
           image: img ? img.getAttribute('alt') : null, hidden: u.hidden };
}"""

# The controls Tab can reach, in document order: shown, enabled, not taken out of the tab order.
FOCUSABLE_JS = """() => [...document.querySelectorAll('a[href], button, input, select, textarea, [tabindex]')].filter((e) => {
  if (e.disabled || e.getAttribute('tabindex') === '-1' || e.closest('[hidden]')) return false;
  const r = e.getBoundingClientRect(), s = getComputedStyle(e);
  return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none';
}).map((e) => e.id || e.getAttribute('data-setting') || e.getAttribute('aria-label') || e.textContent.trim().slice(0, 40))"""

# What has focus, named as FOCUSABLE_JS names it, and whether its focus ring shows.
FOCUSED_JS = """() => {
  const e = document.activeElement;
  if (!e || e === document.body) return { name: 'body', ring: false };
  const s = getComputedStyle(e);
  const ring = (s.outlineStyle !== 'none' && parseFloat(s.outlineWidth) > 0) || (s.boxShadow && s.boxShadow !== 'none');
  return { name: e.id || e.getAttribute('data-setting') || e.getAttribute('aria-label') || e.textContent.trim().slice(0, 40), ring };
}"""

GEOMETRY_JS = """() => {
  const d = document.documentElement, b = document.body;
  const over = (e) => e.scrollWidth - e.clientWidth;
  const hit = (sel) => {
    const e = document.querySelector(sel);
    e.scrollIntoView({ block: 'nearest', inline: 'nearest' });
    const r = e.getBoundingClientRect(), x = r.left + r.width / 2, y = r.top + r.height / 2, top = document.elementFromPoint(x, y);
    return { inside: r.left >= 0 && r.top >= 0 && r.right <= innerWidth && r.bottom <= innerHeight, hit: !!top && (top === e || e.contains(top)) };
  };
  return { overflow: { document: over(d), body: over(b), conversation: over(document.querySelector('#conversation')),
                       composer: over(document.querySelector('#composer')) },
           send: hit('#send'), save: document.querySelector('#drawer').hidden ? null : hit('#drawer-save') };
}"""

# A text colour and the background it is read on: the first ancestor (or itself) whose background is not transparent.
COLOURS_JS = """() => {
  const background = (e) => { for (; e; e = e.parentElement) { const c = getComputedStyle(e).backgroundColor;
    if (c && c !== 'transparent' && !/rgba\\([^)]*,\\s*0\\)$/.test(c)) return c; } return 'rgb(255, 255, 255)'; };
  const pair = (sel) => { const e = document.querySelector(sel); return e ? { color: getComputedStyle(e).color, background: background(e) } : null; };
  return { body: pair('body'), card: pair('#conversation article.card'), report: pair('#conversation article.card .report-body li'),   // a sentence (P6-H)
           note: pair('#conversation article.card .note'), provenance: pair('#conversation article.card .provenance p'),
           banner: pair('.banner .disclaimer'), prompt: pair('#prompt'), chip: pair('#chips span'), send: pair('#send'),
           hint: pair('#rerun-hint'), session: pair('#session-list .session-title'),
           notice: pair('#conversation article.card .test-notice'), badge: pair('#test-badge') };   // P9-G5b: the tiny engine's notice and badge
}"""


def _channels(colour: str) -> Tuple[float, float, float]:
    """rgb(), rgba() or color(srgb ...) -> (r, g, b) in 0..1."""
    found = re.match(r"color\(srgb ([\d.]+) ([\d.]+) ([\d.]+)", colour)
    if found:
        return tuple(float(v) for v in found.groups())   # type: ignore[return-value]
    numbers = [float(v) for v in re.findall(r"[\d.]+", colour)[:3]]
    return tuple(v / 255.0 for v in numbers)   # type: ignore[return-value]


def _luminance(colour: str) -> float:
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in _channels(colour)]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast(foreground: str, background: str) -> float:
    a, b = sorted((_luminance(foreground), _luminance(background)), reverse=True)
    return (a + 0.05) / (b + 0.05)


def bubble(ui: Any, page: Any = None) -> Dict[str, Any]:
    return (page or ui.page).evaluate(BUBBLE_JS)


def assistants(ui: Any, session_id: str) -> List[Dict[str, Any]]:
    """The stored assistant messages of a session, each with its events."""
    return [ui.server.message(m["id"]) for m in ui.server.session(session_id)["messages"] if m["role"] == "assistant"]


def tab_from_the_top(page: Any) -> List[Dict[str, Any]]:
    """Click the banner's text, as a user does to start over from the top (blur() would leave Chrome's sequential-navigation starting
    point where the focus was), then press Tab once per control FOCUSABLE_JS lists; -> what had the focus after each press."""
    page.click(".banner .disclaimer")
    seen = []
    for _ in page.evaluate(FOCUSABLE_JS):
        page.keyboard.press("Tab")
        seen.append(page.evaluate(FOCUSED_JS))
    return seen


def tab_to(page: Any, name: str, limit: int = 60) -> int:
    """Press Tab until the control named `name` has focus; -> how many presses. A control Tab never reaches is a failure."""
    for presses in range(1, limit + 1):
        page.keyboard.press("Tab")
        if page.evaluate(FOCUSED_JS)["name"] == name:
            return presses
    raise AssertionError("Tab never reached {!r} in {} presses".format(name, limit))


# ---- 1 -----------------------------------------------------------------------------------------------------------------------------

def test_first_load_shows_the_banner_the_empty_chat_and_the_server_strip(ui):
    page = ui.open()
    assert page.evaluate("location.hash") == "#/new"   # no chat yet: home is an empty one
    assert page.inner_text(".banner .disclaimer") == BANNER
    assert page.evaluate("document.querySelector('#conversation').children.length") == 0
    assert page.evaluate("getComputedStyle(document.querySelector('#conversation'), '::before').content") == '"{}"'.format(EMPTY_STATE)
    assert page.text_content("#mode-badge") == "private"
    page.wait_for_function("document.querySelector('#health').textContent.includes('turns in flight')")
    assert page.text_content("#health") == "private · 0 of 4 turns in flight"
    assert page.get_attribute("#health", "data-state") is None   # not "server restarting…"
    page.wait_for_function("!document.querySelector('#retention').hidden")   # P8-B: how long a session is kept, from /healthz, in the banner
    assert page.inner_text("#retention") == "Sessions are deleted after 45 days"
    assert page.evaluate("document.querySelector('.banner').contains(document.querySelector('#retention'))")
    composer = ui.composer()
    assert composer["chips"] == ["beam 3", "100 tok", "cached"]   # the page's defaults; no k: this server runs no retrieval
    assert (composer["preview"], composer["rerun"], composer["notice"], composer["send_disabled"]) == (None, None, None, False)
    assert page.evaluate("document.querySelector('#exports').hidden")   # nothing to export yet
    page.click("#settings")   # the model this server runs, in the drawer
    assert page.inner_text("#models-section h4") == "tiny"
    page.keyboard.press("Escape")
    assert ui.server.sessions() == []   # a first load stores nothing
    health = ui.server.get("healthz")
    assert health["status"] == "ok" and health["started_at"] and "code_version" in health
    ui.shot("empty_state_1280x900_light.png")
    ui.assert_clean()


# ---- 2 -----------------------------------------------------------------------------------------------------------------------------

def test_upload_by_file_chooser_and_drag_and_drop_and_the_image_can_be_removed_and_replaced(ui, images):
    page = ui.open()
    with page.expect_file_chooser() as chooser:
        page.click("#image-well")
    chooser.value.set_files(images["xray_a.png"])
    page.wait_for_function("!document.querySelector('#preview').hidden && document.querySelector('#preview span').textContent.includes(' px ')")
    assert re.fullmatch(r"xray_a\.png · 320×320 px · \d+ KB", ui.composer()["preview"])   # P6-A: the pixel size once the picture decodes
    assert page.get_attribute("#preview img", "src").startswith("blob:")
    assert page.get_attribute("#preview img", "alt") == "Preview of xray_a.png"
    assert page.get_attribute("#preview span", "title") == ui.composer()["preview"]   # whole, where a narrow composer cuts the line short
    page.click("#preview button[aria-label='Remove image']")   # Remove (×)
    composer = ui.composer()
    assert composer["preview"] is None and composer["focus"] == "image-well"   # the focus is not lost with the button
    assert ui.drop(images["xray_b.png"]) is True   # dropped from the desktop onto the well: taken, and the page stays
    page.wait_for_function("document.querySelector('#preview span').textContent.startsWith('xray_b.png')")
    ui.attach(images["xray_c.jpg"])   # another file replaces the attached one
    assert ui.composer()["preview"].startswith("xray_c.jpg · ") and page.locator("#preview img").count() == 1
    assert ui.drop(images["xray_a.png"], target="#conversation") is True   # dropped beside the well: the page does not navigate to it
    assert page.url.endswith("#/new") and ui.composer()["preview"].startswith("xray_c.jpg")
    ui.drop(images["xray_a.png"], target="#prompt", kind="paste")   # pasted into the note field: attached
    assert ui.composer()["preview"].startswith("xray_a.png · ")
    assert ui.drop(images["xray_b.png"], target="#conversation", kind="paste") is True   # P6-A: a paste is the page's, wherever the focus is
    page.wait_for_function("document.querySelector('#preview span').textContent.startsWith('xray_b.png')")
    assert ui.server.sessions() == []   # nothing was sent
    ui.assert_clean()


# ---- 3 -----------------------------------------------------------------------------------------------------------------------------

def test_an_image_with_no_note_streams_its_stages_in_order_and_settles_into_a_listed_chat(ui, images):
    page = ui.open()
    ui.attach(images["xray_a.png"])
    ui.record()
    page.dblclick("#send")   # an impatient double click sends once
    card = ui.wait_settled(0)
    assert ui.cards() == 1 and page.evaluate("document.querySelectorAll('#conversation article.turn.user').length") == 1
    rec = ui.recorded()
    for frame in rec["frames"]:   # no stage is ever ahead of an earlier one ...
        ranks = [STAGE_RANK[s] for s in frame["stages"]]
        assert ranks == sorted(ranks, reverse=True), frame
    for i in range(6):            # ... and none goes back
        ranks = [STAGE_RANK[f["stages"][i]] for f in rec["frames"]]
        assert ranks == sorted(ranks), (i, ranks)
    assert len(rec["texts"]) >= 2, rec["texts"]   # the report grew on screen while the turn ran (one frame per animation frame)
    assert card["status"] == "done" and card["stages"] == SETTLED_STAGES and card["spinning"] == 0
    assert card["report_notes"] == [REPEAT_NOTE] and card["notes"] == [] and card["shown"]
    assert card["provenance"].startswith("tiny · ") and re.search(r" · \d+ tok · ", card["provenance"])
    you = bubble(ui)
    assert you["image"] == "Uploaded X-ray: xray_a.png" and you["text"] == ""
    assert you["chips"] == ["beam 3", "100 tok", "cached"]   # what ran; retrieval did not, so no k
    sid = ui.session_id()
    [row] = ui.sidebar()
    assert (row["id"], row["title"], row["current"]) == (sid, "xray_a.png", True) and row["meta"].endswith("· 1 turn")
    session = ui.server.session(sid)
    assert session["title"] == "xray_a.png" and [m["role"] for m in session["messages"]] == ["user", "assistant"]
    assert session["messages"][0]["image_filename"] == "xray_a.png" and session["messages"][0]["text"] == ""
    [message] = assistants(ui, sid)
    assert message["id"] == card["id"] and message["status"] == "done"
    ended = [row["data"]["stage"] for row in message["events"] if row["event"] == "stage_end"]
    assert ended == list(SETTLED_STAGES)   # the server ran them in the contract order too
    assert len([row for row in message["events"] if row["event"] == "content_block_delta"]) >= 10   # and streamed the report
    assert stage_detail(message, "generate")["stopped"] == "repeat"
    assert squeezed(card["shown"]) == squeezed(message["display_report"])
    ui.shot("settled_1280x900_light.png")
    ui.assert_clean()


# ---- 4 -----------------------------------------------------------------------------------------------------------------------------

def test_a_note_rides_with_its_xray_a_question_gets_the_fixed_answer_and_a_command_reruns(ui, images):
    page = ui.open()
    note = "Patient 54, dry cough for three weeks."
    card = ui.turn(images["xray_a.png"], note=note)
    assert card["status"] == "done" and bubble(ui)["text"] == note
    sid = ui.session_id()
    assert ui.server.session(sid)["title"] == note and ui.server.session(sid)["messages"][0]["text"] == note

    ui.type_note("Is there pneumonia?")   # free text with no image: not a command
    assert ui.composer()["rerun"] == QUESTION_HINT
    page.click("#send")
    card = ui.wait_settled(1)
    assert card["status"] == "done" and card["notes"] == [NOT_A_QA_BOT] and card["shown"] == "" and card["timeline_hidden"]
    you = bubble(ui)
    assert you["text"] == "Is there pneumonia?" and you["chips"] == []   # no model ran, so no settings were used
    question = assistants(ui, sid)[-1]
    assert question["report"] is None and [r["data"]["code"] for r in question["events"] if r["event"] == "warning"] == ["not_a_command"]
    assert ui.server.session(sid)["messages"][-2]["image_filename"] is None

    ui.type_note("tokens 30")   # a command with no image: the last X-ray runs again, with the command on top of the settings
    assert ui.composer()["rerun"] == RERUN.format("xray_a.png")
    page.click("#send")
    card = ui.wait_settled(2)
    assert card["status"] == "done" and "30 tok" in bubble(ui)["chips"] and bubble(ui)["text"] == "tokens 30"
    rerun = assistants(ui, sid)[-1]
    assert rerun["options"]["max_new_tokens"] == 30 and stage_detail(rerun, "generate")["tokens"] <= 30
    start = rerun["events"][0]["data"]
    assert start["image"]["source"] == "previous" and start["image"]["filename"] == "xray_a.png"

    ui.expect_refusal(422, "POST", r"/messages$")   # a command the server cannot run: refused before any turn starts
    ui.type_note("beam 0")
    page.click("#send")
    page.wait_for_function("!document.querySelector('#notice').hidden")
    composer = ui.composer()
    assert composer["notice"].startswith("Invalid options: beam_size") and composer["prompt"] == "beam 0"
    assert not composer["send_disabled"] and composer["stop_hidden"] and ui.cards() == 3
    assert len(ui.server.session(sid)["messages"]) == 6 and len(ui.refusals) == 1
    ui.assert_clean()


# ---- 5 -----------------------------------------------------------------------------------------------------------------------------

def test_send_with_no_new_image_reruns_the_last_xray_with_the_budget_set_since(ui, images):
    ui.open()
    ui.turn(images["xray_a.png"])
    assert ui.composer()["rerun"] == RERUN.format("xray_a.png")
    ui.apply_settings(max_new_tokens=40, stop_on_repeat=False)   # off: the whole budget is decoded, so it is the token count
    assert ui.composer()["chips"] == ["beam 3", "40 tok", "cached", "full budget"]
    assert ui.composer()["rerun"] == RERUN.format("xray_a.png")
    card = ui.turn()
    assert card["status"] == "done" and " · 40 tok · " in card["provenance"]
    assert bubble(ui)["chips"] == ["beam 3", "40 tok", "cached", "full budget"]
    sid = ui.session_id()
    rerun = assistants(ui, sid)[-1]
    assert rerun["options"]["max_new_tokens"] == 40 and rerun["options"]["stop_on_repeat"] is False
    assert stage_detail(rerun, "generate")["tokens"] == 40 and stage_detail(rerun, "generate")["stopped"] == "budget"
    image = rerun["events"][0]["data"]["image"]   # message_start: the first turn's X-ray, run again
    assert (image["source"], image["filename"]) == ("previous", "xray_a.png")
    ui.apply_settings(max_new_tokens=24)
    card = ui.turn()
    assert " · 24 tok · " in card["provenance"] and stage_detail(assistants(ui, sid)[-1], "generate")["tokens"] == 24
    ui.assert_clean()


# ---- 6 -----------------------------------------------------------------------------------------------------------------------------

def test_the_settings_drawer_opens_validates_saves_reverts_and_its_switches_apply_to_the_next_turn(ui, images):
    page = ui.open()
    page.click("#settings")   # by click: focus on its close button; the close button gives focus back to Settings
    assert ui.drawer()["open"] and page.get_attribute("#settings", "aria-expanded") == "true" and ui.composer()["focus"] == "drawer-close"
    page.click("#drawer-close")
    assert not ui.drawer()["open"] and ui.composer()["focus"] == "settings"
    page.focus("#settings")
    page.keyboard.press("Enter")   # by keyboard: Enter, and Space
    assert ui.drawer()["open"]
    page.keyboard.press("Escape")
    page.focus("#settings")
    page.keyboard.press(" ")
    assert ui.drawer()["open"]

    page.click("[data-setting=max_new_tokens]")   # a click selects the whole value: typing replaces it ...
    page.keyboard.type("150")
    assert ui.drawer()["tokens"] == "150" and ui.drawer()["errors"] == []
    assert "150 tok" in ui.composer()["chips"]    # ... and the chips follow before any blur
    page.keyboard.press("Enter")                  # Enter saves, closes to Settings, and says so
    composer = ui.composer()
    assert not ui.drawer()["open"] and composer["focus"] == "settings" and composer["saved"] == SAVED
    assert ui.drawer()["stored"]["max_new_tokens"] == 150

    page.click("#settings")
    page.click("[data-setting=max_new_tokens]")
    page.keyboard.type("300")                     # out of range: a line under the field, and nothing taken or stored
    drawer = ui.drawer()
    assert drawer["tokens"] == "300" and drawer["errors"] == ["max_new_tokens-error: " + TOKENS_RANGE]
    assert drawer["invalid"] == ["max_new_tokens"] and drawer["stored"]["max_new_tokens"] == 150
    assert "max_new_tokens-error" in page.get_attribute("[data-setting=max_new_tokens]", "aria-describedby")
    assert "150 tok" in ui.composer()["chips"]
    ui.shot("drawer_error_1280x900_light.png")
    page.keyboard.press("Enter")                  # Enter on a wrong value saves nothing and stays open on it
    assert ui.drawer()["open"] and ui.drawer()["errors"] and ui.composer()["saved"] == ""
    assert ui.drawer()["stored"]["max_new_tokens"] == 150
    page.keyboard.press("ControlOrMeta+a")
    page.keyboard.type("120")
    page.keyboard.press("Enter")
    composer = ui.composer()
    assert not ui.drawer()["open"] and composer["saved"] == SAVED
    assert ui.drawer()["stored"]["max_new_tokens"] == 120 and "120 tok" in composer["chips"]

    page.click("#settings")                       # Esc with a wrong value puts the saved one back
    page.click("[data-setting=beam_size]")
    page.keyboard.type("9")
    assert ui.drawer()["errors"] == ["beam_size-error: " + BEAM_RANGE]
    page.keyboard.press("Escape")
    page.click("#settings")
    assert ui.drawer()["beam"] == "3" and ui.drawer()["errors"] == [] and ui.drawer()["stored"]["beam_size"] == 3
    page.click("[data-setting=max_new_tokens]")   # and so does the close button
    page.keyboard.type("8")
    assert ui.drawer()["errors"] == ["max_new_tokens-error: " + TOKENS_RANGE]
    page.click("#drawer-close")
    page.click("#settings")
    assert ui.drawer()["tokens"] == "120" and ui.drawer()["errors"] == []
    page.click("[data-setting=beam_size]")        # the Save button saves
    page.keyboard.type("5")
    page.click("#drawer-save")
    assert not ui.drawer()["open"] and ui.composer()["saved"] == SAVED and ui.drawer()["stored"]["beam_size"] == 5
    assert ui.composer()["chips"][:2] == ["beam 5", "120 tok"]

    card = ui.turn(images["xray_a.png"])           # the page's defaults: stop at the first repeat, show the repaired text
    sid = ui.session_id()
    first = assistants(ui, sid)[-1]
    assert first["options"]["stop_on_repeat"] is True and first["options"]["display_repair"] is True
    assert stage_detail(first, "generate")["stopped"] == "repeat" and card["report_notes"] == [REPEAT_NOTE]
    assert squeezed(card["shown"]) == squeezed(first["display_report"]) != squeezed(first["report"])
    ui.apply_settings(stop_on_repeat=False, display_repair=False)
    assert ui.composer()["chips"] == ["beam 5", "120 tok", "cached", "raw text", "full budget"]
    card = ui.turn()                               # both switches off: the whole budget, and the decoder's own text
    second = assistants(ui, sid)[-1]
    assert second["options"]["stop_on_repeat"] is False and second["options"]["display_repair"] is False
    assert stage_detail(second, "generate")["stopped"] == "budget" and stage_detail(second, "generate")["tokens"] == 120
    assert squeezed(card["shown"]) == squeezed(second["report"]) and REPEAT_NOTE not in card["report_notes"]
    ui.assert_clean()


# ---- 7 -----------------------------------------------------------------------------------------------------------------------------

def test_stop_ends_a_running_turn_and_the_composer_works_again(ui, images):
    page = ui.open()
    ui.apply_settings(max_new_tokens=200, stop_on_repeat=False)   # a turn long enough to stop
    ui.attach(images["xray_a.png"])
    page.click("#send")
    ui.wait_running(snapshots=4)
    assert ui.composer()["focus"] != "prompt"   # a click on Send leaves the focus to the browser: only the keyboard's is handed on
    ui.shot("streaming_1280x900_light.png")
    # The chat is listed under its title from the moment its first turn starts, not as "New chat · 0 turns" until it ends.
    page.wait_for_function("() => { const t = document.querySelector('#session-list li .session-title');"
                           " const m = document.querySelector('#session-list li .session-meta');"
                           " return t && t.textContent === 'xray_a.png' && m.textContent.endsWith('· 1 turn'); }", timeout=10000)
    assert ui.card()["status"] == "running"
    page.click("#stop")
    card = ui.wait_settled(0, timeout=20)
    assert card["status"] == "aborted" and STOPPED_NOTE in card["notes"] and card["spinning"] == 0
    assert card["stages"]["generate"] == "skipped" and "(stopped)" in card["stage_text"]["generate"]
    composer = ui.composer()
    assert not composer["send_disabled"] and composer["stop_hidden"] and composer["stop_text"] == "Stop"
    assert composer["focus"] != "prompt"         # nor does a click on Stop
    [message] = assistants(ui, ui.session_id())
    assert message["status"] == "aborted"
    assert len([r for r in message["events"] if r["event"] == "content_block_delta"]) < 200
    card = ui.turn(note="tokens 16")   # the composer is usable again: a re-run finishes
    assert card["status"] == "done" and ui.cards() == 2
    ui.assert_clean()


# ---- 8 -----------------------------------------------------------------------------------------------------------------------------

def test_show_raw_toggles_the_decoders_own_text_and_back(ui, images):
    page = ui.open()
    ui.apply_settings(max_new_tokens=80, stop_on_repeat=False)   # the tiny model loops: its raw text repeats, the display copy does not
    card = ui.turn(images["xray_a.png"])
    [message] = assistants(ui, ui.session_id())
    assert squeezed(message["report"]) != squeezed(message["display_report"])
    raw = page.locator("#conversation article.card").last.locator("button[data-action=raw]")
    assert raw.get_attribute("aria-pressed") == "false" and squeezed(card["shown"]) == squeezed(message["display_report"])
    raw.click()
    card = ui.card()
    assert raw.get_attribute("aria-pressed") == "true" and squeezed(card["raw"]) == squeezed(message["report"]) and card["shown"] == ""
    raw.click()
    card = ui.card()
    assert raw.get_attribute("aria-pressed") == "false" and card["raw"] is None
    assert squeezed(card["shown"]) == squeezed(message["display_report"])
    ui.assert_clean()


# ---- 9 -----------------------------------------------------------------------------------------------------------------------------

def test_sessions_new_chat_switching_reload_and_delete(ui, images):
    page = ui.open()
    card_a = ui.turn(images["xray_a.png"])
    sid_a = ui.session_id()
    page.click("#new-session")
    page.wait_for_function("location.hash === '#/new' && !document.querySelector('#conversation').children.length")
    card_b = ui.turn(images["xray_b.png"])
    sid_b = ui.session_id()
    assert sid_a != sid_b
    assert [(r["id"], r["title"], r["current"]) for r in ui.sidebar()] == [(sid_b, "xray_b.png", True), (sid_a, "xray_a.png", False)]

    page.click("#session-list li[data-session='{}'] a".format(sid_a))
    ui.wait_loaded()
    assert ui.session_id() == sid_a and ui.card()["shown"] == card_a["shown"] and ui.cards() == 1
    assert [r["current"] for r in ui.sidebar()] == [False, True]
    page.click("#session-list li[data-session='{}'] a".format(sid_b))
    ui.wait_loaded()
    assert ui.session_id() == sid_b and ui.card()["shown"] == card_b["shown"]

    before = ui.card()
    ui.reload()                                   # the open chat replays from the stored log, card and all
    assert ui.session_id() == sid_b and ui.card() == before and bubble(ui)["chips"] == ["beam 3", "100 tok", "cached"]

    asked = []
    page.once("dialog", lambda dialog: (asked.append(dialog.message), dialog.accept()))
    page.click("#session-list li[data-session='{}'] .session-delete".format(sid_a))
    page.wait_for_function("(id) => ![...document.querySelectorAll('#session-list li')].some((li) => li.getAttribute('data-session') === id)",
                           arg=sid_a)
    assert asked == ['Delete "xray_a.png"? This cannot be undone.']
    assert [r["id"] for r in ui.sidebar()] == [sid_b] and ui.session_id() == sid_b   # deleting another chat leaves this one open
    ui.reload()
    assert [r["id"] for r in ui.sidebar()] == [sid_b]   # and it stays gone
    assert [s["id"] for s in ui.server.sessions()] == [sid_b] and ui.server.status("v1/sessions/" + sid_a) == 404
    sent = len(ui.requests)
    page.once("dialog", lambda dialog: dialog.dismiss())   # Cancel in the dialog deletes nothing, and asks the server nothing
    page.click("#session-list li[data-session='{}'] .session-delete".format(sid_b))
    page.wait_for_timeout(300)                              # a DELETE goes out as the dialog closes: 300 ms is ample
    assert [r for r in ui.requests[sent:] if r[0] == "DELETE"] == []
    assert [r["id"] for r in ui.sidebar()] == [sid_b] and [s["id"] for s in ui.server.sessions()] == [sid_b]

    # Deleting a chat whose turn is running stops that turn: the server's one worker is free at once for the next chat's turn.
    ui.apply_settings(max_new_tokens=200, stop_on_repeat=False)
    page.click("#send")
    ui.wait_running(snapshots=3)
    page.once("dialog", lambda dialog: dialog.accept())
    page.click("#session-list li[data-session='{}'] .session-delete".format(sid_b))
    page.wait_for_function("location.hash === '#/new'")   # the last chat is gone: an empty one
    for _ in range(50):   # 5 s, for a loaded machine; left running, the 200-token turn would hold the worker for about 10 s more
        if ui.server.get("healthz")["turns_in_flight"] == 0:
            break
        page.wait_for_timeout(100)
    assert ui.server.get("healthz")["turns_in_flight"] == 0, "the deleted chat's turn still runs"
    assert ui.server.sessions() == [] and ui.sidebar() == []
    ui.assert_clean()


# ---- 10 ----------------------------------------------------------------------------------------------------------------------------

def test_both_exports_download_the_reports_and_the_turn_count(ui, images, tmp_path):
    page = ui.open()
    ui.turn(images["xray_a.png"])
    ui.turn(note="tokens 20")
    sid = ui.session_id()
    stored = assistants(ui, sid)
    assert [m["status"] for m in stored] == ["done", "done"]
    with page.expect_download() as info:
        page.click("#exports button[data-format=json]")
    assert info.value.suggested_filename == "session-{}.json".format(sid)
    info.value.save_as(str(tmp_path / "export.json"))
    doc = json.loads((tmp_path / "export.json").read_text(encoding="utf-8"))
    assert doc["session"]["id"] == sid and doc["session"]["turns"] == 2
    exported = [m for m in doc["messages"] if m["role"] == "assistant"]
    assert [m["report"] for m in exported] == [m["report"] for m in stored]
    assert [m["events"][-1]["event"] for m in exported] == ["message_stop", "message_stop"]
    with page.expect_download() as info:
        page.click("#exports button[data-format=md]")
    assert info.value.suggested_filename == "session-{}.md".format(sid)
    info.value.save_as(str(tmp_path / "export.md"))
    md = (tmp_path / "export.md").read_text(encoding="utf-8")
    assert md.startswith("# Session {}\n".format(sid))
    assert len(re.findall(r"^## Turn \d+ ", md, flags=re.M)) == 2
    for message in stored:
        assert "> " + message["report"] in md
    ui.assert_clean()


# ---- 11 ----------------------------------------------------------------------------------------------------------------------------

def _deleting(response: Any) -> bool:
    """The page's DELETE of a chat: what follows a first turn the server refused."""
    return response.request.method == "DELETE" and "/v1/sessions/" in response.url


def _refused_and_usable(ui: Any, notice: str) -> None:
    """The refusal is said in the notice, nothing is left spinning, and the composer works."""
    composer = ui.composer()
    assert composer["notice"] == notice, composer
    assert not composer["send_disabled"] and composer["stop_hidden"]
    assert ui.page.evaluate("document.querySelectorAll('#conversation .timeline > li[data-state=\"running\"]').length") == 0


def test_errors_are_refused_visibly_and_leave_the_composer_usable(ui, images):
    page = ui.open()
    ui.expect_refusal(422, "POST", r"/messages$")

    ui.attach(images["notes.png"])                # a text file renamed .png: the server reads its bytes and refuses it
    with page.expect_response(_deleting) as dropped:
        page.click("#send")
    assert dropped.value.status == 204            # the chat made for the refused first turn is deleted again
    _refused_and_usable(ui, FORMATS_MSG)
    assert ui.cards() == 0 and page.evaluate("document.querySelector('#conversation').children.length") == 0
    assert ui.composer()["preview"].startswith("notes.png")   # still attached, to remove or replace
    assert page.evaluate("location.hash") == "#/new" and ui.sidebar() == [] and ui.server.sessions() == []   # no empty chat is left behind
    page.click("#preview button")

    page.set_input_files("#file", images["scan.gif"])   # a GIF: refused by the page itself, before any request
    page.wait_for_function("(m) => document.querySelector('#notice p').textContent === m", arg=NOT_AN_IMAGE)
    _refused_and_usable(ui, NOT_AN_IMAGE)
    assert ui.composer()["preview"] is None

    ui.attach(images["tiny_32.png"])              # 32 x 32: the server refuses it
    with page.expect_response(_deleting):
        page.click("#send")
    _refused_and_usable(ui, TOO_SMALL_MSG)
    assert ui.server.sessions() == []
    page.click("#preview button")

    page.set_input_files("#file", images["huge.png"])   # over 20 MB: refused by the page, nothing uploaded
    page.wait_for_function("(m) => document.querySelector('#notice p').textContent === m", arg=TOO_BIG)
    _refused_and_usable(ui, TOO_BIG)
    assert ui.composer()["preview"] is None and len(ui.refusals) == 2

    page.set_input_files("#file", images["scan.dcm"])   # a DICOM file: refused by the page, by its name (P6-A)
    page.wait_for_function("(m) => document.querySelector('#notice p').textContent === m", arg=DICOM)
    _refused_and_usable(ui, DICOM)
    assert ui.composer()["preview"] is None and len(ui.refusals) == 2

    card = ui.turn(images["xray_a.png"], note="tokens {}".format(FAULT_TOKENS))   # an internal error inside the engine
    assert card["status"] == "error" and card["notes"] == [INTERNAL_ERROR, INTERNAL_HINT] and card["spinning"] == 0
    assert not ui.composer()["send_disabled"] and ui.composer()["stop_hidden"]
    ui.shot("error_card_1280x900_light.png")
    sid = ui.session_id()
    [failed] = assistants(ui, sid)
    assert failed["status"] == "error"
    assert [r["data"]["error"] for r in failed["events"] if r["event"] == "error"] == [
        {"type": "model_error", "message": "Internal error (ImportError)"}]
    assert "planted" not in json.dumps(failed)    # the exception's text stays in the server log

    ui.apply_settings(max_new_tokens=200, stop_on_repeat=False)   # the server dies mid-turn
    page.click("#send")
    ui.wait_running(snapshots=4)
    ui.server_down(True)
    ui.server.kill()
    page.wait_for_timeout(1500)
    assert ui.card()["status"] == "running"       # the page cannot know yet: it keeps polling
    ui.server.start()
    card = ui.wait_settled(1, timeout=40)
    ui.server_down(False)
    assert card["status"] == "error" and RESTART_ERROR in card["notes"] and card["spinning"] == 0
    assert INTERNAL_HINT not in card["notes"]
    assert not ui.composer()["send_disabled"] and ui.composer()["stop_hidden"]
    killed = assistants(ui, sid)[-1]
    assert killed["status"] == "error" and [r["data"]["error"]["type"] for r in killed["events"] if r["event"] == "error"] == ["server_restart"]
    card = ui.turn(note="tokens 16")               # and the next Send works
    assert card["status"] == "done"
    ui.assert_clean()


# ---- 12 ----------------------------------------------------------------------------------------------------------------------------

def test_keyboard_only(ui, images):
    page = ui.open()
    expected = page.evaluate(FOCUSABLE_JS)
    assert expected == ["new-session", "image-well", "prompt", "settings", "send"]
    seen = tab_from_the_top(page)                  # Tab visits every control in document order, each with its focus ring
    assert [f["name"] for f in seen] == expected and all(f["ring"] for f in seen), seen

    page.focus("#settings")                        # the drawer, by keyboard only: a long turn, to stop
    page.keyboard.press("Enter")
    assert ui.composer()["focus"] == "drawer-close"
    tab_to(page, "max_new_tokens")
    page.keyboard.type("200")                      # Tab selected the whole value: the keys replace it
    tab_to(page, "stop_on_repeat")
    page.keyboard.press("Space")
    tab_to(page, "drawer-save")
    page.keyboard.press("Enter")
    assert not ui.drawer()["open"] and ui.composer()["focus"] == "settings" and ui.drawer()["stored"]["stop_on_repeat"] is False

    page.focus("#image-well")                      # attach by keyboard
    with page.expect_file_chooser() as chooser:
        page.keyboard.press("Enter")
    chooser.value.set_files(images["xray_a.png"])
    page.wait_for_function("!document.querySelector('#preview').hidden")
    tab_to(page, "send", limit=6)
    page.keyboard.press("Enter")                   # Send by keyboard: the focus goes to the note field, not to nowhere
    page.wait_for_function("!document.querySelector('#stop').hidden && !document.querySelector('#stop').disabled")
    assert ui.composer()["focus"] == "prompt"
    ui.wait_running(snapshots=3)
    tab_to(page, "stop", limit=5)                  # Stop is a few Tabs away
    page.keyboard.press("Enter")
    card = ui.wait_settled(0)
    assert card["status"] == "aborted" and ui.composer()["focus"] == "prompt"

    expected = page.evaluate(FOCUSABLE_JS)         # with a chat and a card: the sidebar, the card's controls, the composer
    assert expected[:5] == ["new-session", "Export JSON", "Export Markdown", expected[3], "Delete chat: xray_a.png"]
    assert expected[-4:] == ["image-well", "prompt", "settings", "send"] and "Show raw report, turn 1" in expected
    seen = tab_from_the_top(page)
    assert [f["name"] for f in seen] == expected and all(f["ring"] for f in seen), seen

    page.focus("#settings")                        # Esc closes the drawer, back to Settings
    page.keyboard.press("Enter")
    page.keyboard.press("Escape")
    assert not ui.drawer()["open"] and ui.composer()["focus"] == "settings"

    page.click("#settings")                        # every control has a name, the drawer's included (Chrome's own accessibility tree)
    cdp = page.context.new_cdp_session(page)
    nodes = cdp.send("Accessibility.getFullAXTree")["nodes"]
    roles = {"button", "link", "textbox", "checkbox", "spinbutton", "combobox", "switch", "searchbox", "slider", "radio"}
    unnamed = [n for n in nodes if not n.get("ignored") and n.get("role", {}).get("value") in roles
               and not str(n.get("name", {}).get("value", "")).strip()]
    assert unnamed == [], unnamed
    page.keyboard.press("Escape")

    phone = ui.new_page(viewport=(390, 844))       # on a phone, Esc closes the sessions panel, back to its toggle
    ui.open(phone, "#/s/" + ui.session_id())
    phone.focus("#sidebar-toggle")
    phone.keyboard.press("Enter")
    assert phone.get_attribute("#sidebar-toggle", "aria-expanded") == "true"
    phone.keyboard.press("Escape")
    assert phone.get_attribute("#sidebar-toggle", "aria-expanded") == "false"
    assert phone.evaluate("document.activeElement.id") == "sidebar-toggle"
    ui.assert_clean()


# ---- 13 ----------------------------------------------------------------------------------------------------------------------------

def test_phone_tablet_and_desktop_viewports_and_the_dark_theme(ui, images):
    ui.open()
    ui.turn(images["xray_a.png"])
    sid = ui.session_id()
    for width, height in ((390, 844), (768, 1024), (1440, 900)):
        phone = width == 390
        page = ui.new_page(viewport=(width, height), **({"has_touch": True, "is_mobile": True} if phone else {}))
        ui.open(page, "#/s/" + sid)
        geometry = page.evaluate(GEOMETRY_JS)
        assert all(v <= 0 for v in geometry["overflow"].values()), (width, geometry)   # no horizontal scroll anywhere
        assert geometry["send"] == {"inside": True, "hit": True}, (width, geometry)    # Send is on screen and nothing covers it
        if phone:
            ui.shot("phone_390x844_light.png", page)
            before = ui.cards(page)
            page.tap("#send")   # a tap re-runs the X-ray, and leaves the note field alone: no soft keyboard over the card that streams
            assert page.evaluate("document.activeElement.id") != "prompt"
            assert ui.wait_settled(before, page)["status"] == "done"
        page.click("#settings")
        geometry = page.evaluate(GEOMETRY_JS)
        assert all(v <= 0 for v in geometry["overflow"].values()), (width, "drawer", geometry)
        assert geometry["save"]["hit"], (width, geometry)
        page.close()

    dark = ui.new_page(viewport=DESKTOP, scheme="dark")
    ui.open(dark, "#/s/" + sid)
    colours = dark.evaluate(COLOURS_JS)
    assert _luminance(colours["body"]["background"]) < 0.05, colours["body"]   # the dark theme is on
    for part, pair in colours.items():
        assert pair is not None, part
        ratio = contrast(pair["color"], pair["background"])
        assert ratio >= 4.5, (part, pair, round(ratio, 2))   # WCAG AA for text
    ui.shot("dark_1280x900_dark.png", dark)
    ui.assert_clean()


# ---- 14 ----------------------------------------------------------------------------------------------------------------------------

def test_two_tabs_run_turns_at_once_and_neither_disturbs_the_other(ui, images):
    context = ui.new_context()
    one, two = ui.new_page(context), ui.new_page(context)
    for page in (one, two):
        ui.open(page, "#/new")
    ui.attach(images["xray_a.png"], one)
    ui.attach(images["xray_b.png"], two)
    one.click("#send")
    two.click("#send")
    card_one, card_two = ui.wait_settled(0, one), ui.wait_settled(0, two)
    sid_one, sid_two = ui.session_id(one), ui.session_id(two)
    assert sid_one != sid_two and card_one["status"] == card_two["status"] == "done"
    assert bubble(ui, one)["image"] == "Uploaded X-ray: xray_a.png" and bubble(ui, two)["image"] == "Uploaded X-ray: xray_b.png"
    for sid, card, title in ((sid_one, card_one, "xray_a.png"), (sid_two, card_two, "xray_b.png")):
        session = ui.server.session(sid)
        assert session["title"] == title and session["turns"] == 1
        [message] = assistants(ui, sid)
        assert message["id"] == card["id"] and squeezed(message["display_report"]) == squeezed(card["shown"])
    ui.reload(one)
    assert ui.session_id(one) == sid_one and ui.card(one)["shown"] == card_one["shown"]
    assert {row["id"] for row in ui.sidebar(one)} == {sid_one, sid_two}
    ui.assert_clean()


# ---- 15 (P5-E) ---------------------------------------------------------------------------------------------------------------------

# The newest card's label area: its chips (how many, positive, marked against a reference), any note in place of them, and the scores.
LABELS_JS = """() => {
  const c = [...document.querySelectorAll('#conversation article.card')].pop();
  const chips = [...c.querySelectorAll('.labels li.chip.label')];
  const source = c.querySelector('.labels .score-source');
  return { chips: chips.length, positive: chips.filter((x) => x.classList.contains('positive')).length,
           marked: chips.filter((x) => x.hasAttribute('data-agree')).length,
           notes: [...c.querySelectorAll('.labels .note')].map((n) => n.textContent),
           scores: [...c.querySelectorAll('.labels .score dt')].map((n) => n.textContent), source: source ? source.textContent : null };
}"""
GALLERY_STAGES = {"preprocess": "done", "encode": "done", "retrieve": "done", "generate": "done", "label": "done", "score": "skipped"}


@pytest.mark.parametrize("ui", [{"tiny_gallery": True}], indirect=True, ids=["tiny_gallery"])
def test_with_a_gallery_every_stage_runs_and_the_card_settles_on_labels_and_a_test_studys_score(ui, images):
    """P5-E on a tiny_gallery server (the synthetic gallery and the keyword labeller). An upload's stored events carry its similar
    X-rays and matching reports, and one label agreement per neighbour; its card settles on the 14 label chips with no placeholder
    left behind. A test study, sent as the picker will send it (options.test_row and no image; the picker itself is P6-D), is scored
    against its reference, and once the chat is loaded again its card shows the scores and an agree mark on every chip."""
    page = ui.open()
    assert ui.server.get("v1/models")["features"] == {"retrieval": True, "labels": True}
    card = ui.turn(images["xray_a.png"])
    assert card["status"] == "done" and card["stages"] == GALLERY_STAGES and card["spinning"] == 0   # no reference: no score
    labels = page.evaluate(LABELS_JS)
    assert labels["chips"] == 14 and labels["positive"] >= 1 and labels["notes"] == [] and labels["marked"] == 0   # not "labelling…"
    assert card["labels"] == []
    assert bubble(ui)["chips"] == ["beam 3", "100 tok", "cached", "k 4/3"]   # retrieval ran, so the k it used is shown
    sid = ui.session_id()
    [upload] = assistants(ui, sid)
    retrieve, label = stage_detail(upload, "retrieve"), stage_detail(upload, "label")
    assert len(retrieve["image_neighbors"]) == 4 and len(retrieve["report_matches"]) == 3   # the drawer's k 4/3
    assert [a["rank"] for a in label["neighbor_agreement"]] == [n["rank"] for n in retrieve["image_neighbors"]] == [1, 2, 3, 4]
    assert all(a["of"] == 14 for a in label["neighbor_agreement"]) and len(label["chexbert_14"]) == 14

    picked = ui.server.message(ui.server.post_turn(sid, {"test_row": 0, "max_new_tokens": 24}))
    assert picked["status"] == "done" and picked["events"][0]["data"]["image"]["source"] == "test_split"
    score = stage_detail(picked, "score")
    assert score["reference_source"] == "test_split" and len(score["reference_chexbert_14"]) == 14
    assert stage_detail(picked, "retrieve")["true_report_rank"]["of"] == 40   # the tiny test split
    ui.reload()
    card = ui.card()
    assert card["n"] == 2 and card["status"] == "done" and set(card["stages"].values()) == {"done"} and card["labels"] == []
    labels = page.evaluate(LABELS_JS)
    assert labels["chips"] == 14 and labels["marked"] == 14 and labels["notes"] == []
    assert labels["scores"] == ["ROUGE-L", "BLEU-1", "BLEU-4", "CheXbert-14 micro F1", "CheXbert-14 exact match"]
    assert labels["source"].startswith("vs test-split reference")
    assert "test row 0" in bubble(ui)["chips"]
    ui.shot("labels_and_score_1280x900_light.png")
    ui.assert_clean()


# ---- 16 (P6-B) ---------------------------------------------------------------------------------------------------------------------

# The newest turn's pictures: the user turn's thumbnail and the card's images row, each with the server path it shows (data-src; none for
# the page's own preview), whether its source is an object URL, whether it has loaded, and its natural size.
PICTURES_JS = """() => {
  const u = [...document.querySelectorAll('#conversation article.turn.user')].pop();
  const c = [...document.querySelectorAll('#conversation article.card')].pop();
  const look = (i) => i ? { path: i.getAttribute('data-src'), blob: (i.getAttribute('src') || '').startsWith('blob:'), alt: i.getAttribute('alt'),
                            loaded: i.complete && i.naturalWidth > 0, natural: [i.naturalWidth, i.naturalHeight] } : null;
  return { bubble: u ? look(u.querySelector('img')) : null,
           tiles: c ? [...c.querySelectorAll('section.images figure')].map((f) => ({ caption: f.querySelector('figcaption').textContent,
                                                                                    img: look(f.querySelector('img')) })) : [] };
}"""
PICTURES_LOADED = ("() => { const p = (%s)(); return !!p.bubble && p.bubble.loaded && p.tiles.length === 2"
                   " && p.tiles.every((t) => t.img.loaded); }" % PICTURES_JS)


def test_your_xray_and_what_the_model_saw_are_in_the_card_and_still_there_after_a_reload(ui, images):
    """P6-B: the card's images row shows the upload's thumbnail, labelled with its pixel size, and the 224x224 image the model saw, both
    fetched with the page's auth from GET /v1/messages/{id}/image. After a reload the user turn shows the server's thumbnail in place of
    the page's own preview, and the card the same two pictures."""
    page = ui.open()
    card = ui.turn(images["xray_b.png"])   # 288 x 256
    assert card["status"] == "done"
    page.wait_for_function(PICTURES_LOADED)
    shown = page.evaluate(PICTURES_JS)
    sid = ui.session_id()
    user = ui.server.session(sid)["messages"][0]
    assert user["image_urls"] == {v: "/v1/messages/{}/image?variant={}".format(user["id"], v) for v in ("original", "thumb", "model_input")}
    assert [t["caption"] for t in shown["tiles"]] == ["Your X-ray · 288×256 px", "What the model saw (224×224)"]
    assert [t["img"]["path"] for t in shown["tiles"]] == [user["image_urls"]["thumb"], user["image_urls"]["model_input"]]
    assert all(t["img"]["blob"] for t in shown["tiles"])   # through loadImage: the token never rides an <img src>
    assert [t["img"]["natural"] for t in shown["tiles"]] == [[288, 256], [224, 224]]
    assert shown["bubble"]["path"] is None and shown["bubble"]["blob"]   # the live turn shows the page's own preview of the file
    ui.reload()
    page.wait_for_function(PICTURES_LOADED)
    again = page.evaluate(PICTURES_JS)
    assert again["bubble"]["path"] == user["image_urls"]["thumb"] and again["bubble"]["natural"] == [288, 256]   # the server's thumbnail now
    assert again["bubble"]["alt"] == "Uploaded X-ray: xray_b.png"
    assert [t["caption"] for t in again["tiles"]] == [t["caption"] for t in shown["tiles"]]
    assert [t["img"]["natural"] for t in again["tiles"]] == [[288, 256], [224, 224]]
    asked = [path for method, path in ui.requests if method == "GET" and path == "/v1/messages/{}/image".format(user["id"])]
    assert len(asked) >= 3   # the card's two before the reload, the bubble's and the card's after it; assert_clean fails any refusal
    ui.shot("images_row_1280x900_light.png")
    ui.assert_clean()


# ---- 17, 18 (P6-C) ----------------------------------------------------------------------------------------------------------------

# The newest card's grid of similar X-rays: its title and note, and for each neighbour its rank, "#rank · similarity", the bar's width, its
# picture (the server path it shows, whether it has loaded, its natural size) or a placeholder, the agreement line and the chips beside it.
NEIGHBOURS_JS = """() => {
  const c = [...document.querySelectorAll('#conversation article.card')].pop();
  const s = c && c.querySelector('section.neighbors');
  if (!s) return null;
  const note = s.querySelector('.section-note');
  return { title: s.querySelector('h3').textContent, note: note ? note.textContent : null,
           items: [...s.querySelectorAll('li.neighbor')].map((li) => {
             const img = li.querySelector('img'), fill = li.querySelector('.sim-fill'), agree = li.querySelector('.agreement');
             return { rank: li.getAttribute('data-rank'), score: li.querySelector('.neighbor-score').textContent,
                      bar: fill ? fill.style.width : null, placeholder: !!li.querySelector('.neighbor-placeholder'),
                      img: img ? { path: img.getAttribute('data-src'), loaded: img.complete && img.naturalWidth > 0,
                                   natural: [img.naturalWidth, img.naturalHeight] } : null,
                      agreement: agree ? agree.textContent : null, chips: [...li.querySelectorAll('.chip.diff')].map((x) => x.textContent) };
           }) };
}"""
NEIGHBOURS_LOADED = ("() => { const g = (%s)(); return !!g && g.items.length > 0 && g.items.every((i) => !!i.img && i.img.loaded); }"
                     % NEIGHBOURS_JS)


def _bar(similarity: float) -> str:
    """The bar's width as the page sets it: Math.round of the similarity, held to 0..1, in per cent."""
    return "{}%".format(math.floor(min(1.0, max(0.0, similarity)) * 100 + 0.5))


@pytest.mark.parametrize("ui", [{"tiny_gallery": True}], indirect=True, ids=["tiny_gallery"])
def test_similar_xrays_show_their_pictures_their_similarity_and_how_many_labels_agree(ui, images):
    """P6-C on a tiny_gallery server: the card's "Similar X-rays (13D tower)" grid shows each neighbour's stored 320 px image (fetched with
    the page's auth from GET /v1/gallery/images/{row}), "#rank · similarity" with a bar, and, once the label stage has ended, "n/14 labels
    agree" with a chip for each positive that differs. A reload shows the same grid."""
    page = ui.open()
    card = ui.turn(images["xray_a.png"])
    assert card["status"] == "done"
    page.wait_for_function(NEIGHBOURS_LOADED)
    grid = page.evaluate(NEIGHBOURS_JS)
    [message] = assistants(ui, ui.session_id())
    retrieve, label = stage_detail(message, "retrieve"), stage_detail(message, "label")
    assert grid["title"] == "Similar X-rays (13D tower)" and grid["note"] is None
    assert [i["rank"] for i in grid["items"]] == [str(n["rank"]) for n in retrieve["image_neighbors"]] == ["1", "2", "3", "4"]
    for item, n, a in zip(grid["items"], retrieve["image_neighbors"], label["neighbor_agreement"]):
        assert item["score"] == "#{} · {:.3f}".format(n["rank"], n["similarity"]) and item["bar"] == _bar(n["similarity"])
        assert item["img"]["path"] == n["image_url"] == "/v1/gallery/images/{}".format(n["gallery_row"])
        assert item["img"]["natural"] == [320, 320] and not item["placeholder"]   # the stored gallery size
        assert item["agreement"] == "{}/14 labels agree".format(a["agree"])
        assert item["chips"] == ["neighbour: " + x for x in a["neighbor_only"]] + ["report: " + x for x in a["generated_only"]]
    asked = {path for method, path in ui.requests if method == "GET" and path.startswith("/v1/gallery/images/")}
    assert asked == {n["image_url"] for n in retrieve["image_neighbors"]}   # and each was answered 200: assert_clean fails any refusal
    ui.shot("similar_xrays_1280x900_light.png")
    ui.reload()
    page.wait_for_function(NEIGHBOURS_LOADED)
    assert page.evaluate(NEIGHBOURS_JS) == grid   # replayed from the stored log
    ui.assert_clean()


@pytest.mark.parametrize("ui", [{"tiny_gallery": True, "public_token": "t"}], indirect=True, ids=["tiny_gallery_public"])
def test_in_public_mode_similar_xrays_are_placeholders_and_the_page_asks_for_no_gallery_image(ui, images):
    """P6-C and R1 on a public tiny_gallery server: the grid shows rank and similarity only, with a neutral placeholder for each picture and
    no label agreement, and the page never asks for a gallery image. The images row still shows the user's own upload, as in private mode."""
    ui.page.context.add_init_script(ui.server.setup_script())   # the token, as Settings holds it, and the shared client id
    page = ui.open()
    assert page.text_content("#mode-badge") == "public"
    card = ui.turn(images["xray_a.png"])
    assert card["status"] == "done" and card["stages"]["retrieve"] == "done" and card["stages"]["score"] == "skipped"
    grid = page.evaluate(NEIGHBOURS_JS)
    assert grid["note"] == "Public mode shows rank and similarity only." and len(grid["items"]) == 4
    for item in grid["items"]:
        assert item["placeholder"] and item["img"] is None and item["agreement"] is None and item["chips"] == [], item
        assert re.fullmatch(r"#\d+ · -?\d\.\d{3}", item["score"]) and re.fullmatch(r"\d+%", item["bar"]), item
    page.wait_for_function(PICTURES_LOADED)   # the user's own pictures: its upload is shown back to it in both modes
    assert not [path for _, path in ui.requests if path.startswith("/v1/gallery/")]   # not one request for a gallery image
    stored = ui.server.message(card["id"])
    assert all(set(n) == {"rank", "similarity"} for n in stage_detail(stored, "retrieve")["image_neighbors"])   # U2, as stored and sent
    ui.shot("similar_xrays_public_1280x900_light.png")
    ui.assert_clean()


# ---- 19 (P6-D) --------------------------------------------------------------------------------------------------------------------

# The newest card's matching reports: the title, the own-report badge (its visible text and its tooltip), and per report its rank, score,
# group size, text, whether it is clamped and whether it overflows its three lines (measured here, in the page), its Show all button when
# that is shown (P6 fix 1: exactly when the text overflows), and its label chips.
MATCHES_JS = """() => {
  const c = [...document.querySelectorAll('#conversation article.card')].pop();
  const s = c && c.querySelector('section.matches');
  if (!s) return null;
  const badge = s.querySelector('.own-rank');
  return { title: s.querySelector('h3').textContent,
           badge: badge ? { text: badge.childNodes[0].textContent, title: badge.getAttribute('title') } : null,
           items: [...s.querySelectorAll('li.match')].map((li) => {
             const text = li.querySelector('.clamp-text'), more = li.querySelector('button[data-action="show"]'), size = li.querySelector('.group-size');
             const shown = !!more && !more.closest('[hidden]') && more.getClientRects().length > 0;
             return { rank: li.getAttribute('data-rank'), score: li.querySelector('.match-score').textContent, size: size ? size.textContent : null,
                      text: text ? text.textContent : null, clamped: !!text && text.classList.contains('clamped'),
                      overflows: !!text && text.scrollHeight > text.clientHeight + 1, more: shown ? more.textContent : null,
                      chips: [...li.querySelectorAll('.chip.finding')].map((x) => x.textContent) };
           }) };
}"""
PICKER_JS = """() => {
  const p = document.querySelector('#picker'), b = document.querySelector('#pick-study');
  return { button: b ? { hidden: b.hidden, expanded: b.getAttribute('aria-expanded'), disabled: b.disabled } : null, open: !!p && !p.hidden,
           status: document.querySelector('#picker-status') ? document.querySelector('#picker-status').textContent : null,
           focus: document.activeElement ? document.activeElement.id : null,
           studies: [...document.querySelectorAll('#picker-list button')].map((x) => ({ row: x.getAttribute('data-test-row'), text: x.textContent })) };
}"""


@pytest.mark.parametrize("ui", [{"tiny_gallery": True}], indirect=True, ids=["tiny_gallery"])
def test_the_picker_runs_a_test_study_whose_card_ranks_its_own_report_and_lists_the_matching_reports(ui, images):
    """P6-D on a tiny_gallery server: the composer's "Test-split study" button opens a list from GET /v1/test-studies that narrows as a study
    id is typed; choosing one sends a turn with options.test_row and no upload. Its card shows "Own report: rank r of 40 test reports" with the
    dedup-aware rank and the protocol in the tooltip, and the matching reports (13D image and text encoders) ranked, with group sizes and
    label chips. A text-only follow-up runs the same study again."""
    page = ui.open()
    picker = page.evaluate(PICKER_JS)
    assert picker["button"] == {"hidden": False, "expanded": "false", "disabled": False} and not picker["open"]
    page.click("#pick-study")
    page.wait_for_function("document.querySelectorAll('#picker-list button').length > 0")
    picker = page.evaluate(PICKER_JS)
    assert picker["open"] and picker["button"]["expanded"] == "true" and picker["focus"] == "picker-search"
    assert len(picker["studies"]) == 40 and picker["status"] == "40 studies"   # the tiny test split, all under the page's limit of 50
    listed = ui.server.get("v1/test-studies?limit=50")["studies"]
    assert [s["row"] for s in picker["studies"]] == [str(s["test_row"]) for s in listed]
    study = listed[3]
    prefix = str(study["study_id"])
    page.fill("#picker-search", prefix)
    page.wait_for_function("(row) => { const b = [...document.querySelectorAll('#picker-list button')];"
                           " return b.length > 0 && b.every((x) => x.getAttribute('data-test-row') === row); }", arg=str(study["test_row"]))
    page.click('#picker-list button[data-test-row="{}"]'.format(study["test_row"]))
    card = ui.wait_settled(0)
    assert card["status"] == "done" and not page.evaluate(PICKER_JS)["open"]
    sid = ui.session_id()
    [message] = assistants(ui, sid)
    start = message["events"][0]["data"]
    assert start["image"]["source"] == "test_split" and start["options"]["test_row"] == study["test_row"]   # no upload: the study itself
    retrieve = stage_detail(message, "retrieve")
    rank = retrieve["true_report_rank"]
    shown = page.evaluate(MATCHES_JS)
    assert shown["title"] == "Matching reports (13D image and text encoders)"
    assert shown["badge"]["text"] == "Own report: rank {} of 40 test reports · R@10 {}".format(rank["rank"], "hit" if rank["hit_at_10"] else "miss")
    assert "Dedup-aware rank {}".format(rank["rank_dedup"]) in shown["badge"]["title"] and rank["protocol"] in shown["badge"]["title"]
    assert [i["rank"] for i in shown["items"]] == [str(m["rank"]) for m in retrieve["report_matches"]] == ["1", "2", "3"]
    for item, m in zip(shown["items"], retrieve["report_matches"]):
        assert item["score"] == "#{} · {:.3f}".format(m["rank"], m["similarity"]) and item["text"] == m["report"]
        assert item["size"] == ("×{} identical reports".format(m["group_size"]) if m["group_size"] > 1 else None)
        assert item["chips"] == [name for name in CHEXBERT_14 if m["labels"][name]]
        assert item["clamped"] and item["more"] == ("Show all" if item["overflows"] else None)   # three lines, and Show all only to reveal more
    you = bubble(ui)
    assert you["image"] == "Test-split X-ray" and "test row {}".format(study["test_row"]) in you["chips"]
    page.wait_for_function(PICTURES_LOADED)   # the study's picture in the user turn and the images row: served from the dataset, private
    assert ui.composer()["rerun"] == RERUN.format("test row {}".format(study["test_row"]))
    ui.shot("picker_turn_1280x900_light.png")
    card = ui.turn(note="beam 2")             # a text-only follow-up runs the same study again (server: _previous_image)
    assert card["status"] == "done"
    follow = assistants(ui, sid)[-1]
    assert follow["events"][0]["data"]["image"]["source"] == "test_split" and follow["options"]["test_row"] == study["test_row"]
    assert stage_detail(follow, "retrieve")["true_report_rank"] == rank
    assert page.evaluate(MATCHES_JS)["badge"] == shown["badge"]
    ui.assert_clean()


@pytest.mark.parametrize("ui", [{"tiny_gallery": True}], indirect=True, ids=["tiny_gallery"])
def test_the_picker_by_keyboard_runs_the_study_typed_and_escape_closes_the_drawer_over_it(ui, images):
    """P6 fix 1: the two keyboard defects the review's probes found, replayed in a real Chrome. A study id typed and Enter pressed at once,
    within the 250 ms pause, focuses the first study of the new list, and a second Enter runs that study, not one of the list shown before.
    With the list and the drawer both open, Escape closes the drawer and gives the focus back to Settings."""
    page = ui.open()
    target = ui.server.get("v1/test-studies?limit=50")["studies"][7]
    query = str(target["study_id"])
    first = ui.server.get("v1/test-studies?limit=50&q=" + query)["studies"][0]   # what the list for that query starts with
    page.click("#pick-study")
    page.wait_for_function("document.querySelectorAll('#picker-list button').length > 0")
    assert page.evaluate("document.activeElement.id") == "picker-search"
    page.keyboard.type(query)
    page.keyboard.press("Enter")   # within the pause
    page.wait_for_function("(row) => document.activeElement && document.activeElement.getAttribute('data-test-row') === row",
                           arg=str(first["test_row"]))
    page.keyboard.press("Enter")   # the focused study
    card = ui.wait_settled(0)
    assert card["status"] == "done"
    [message] = assistants(ui, ui.session_id())
    assert message["options"]["test_row"] == first["test_row"] and first["test_row"] != 0   # never the old list's first study
    page.click("#pick-study")
    page.wait_for_function("!document.querySelector('#picker').hidden")
    page.click("#settings")        # the drawer opens over the open list, which closes
    assert page.evaluate("[document.querySelector('#picker').hidden, document.querySelector('#drawer').hidden]") == [True, False]
    page.keyboard.press("Escape")
    assert page.evaluate("[document.querySelector('#drawer').hidden, document.activeElement.id]") == [True, "settings"]
    ui.assert_clean()


# ---- 21 (P6-E) --------------------------------------------------------------------------------------------------------------------

# The viewer as the user sees it: whether it is open, what it says it shows, its panes (footer, source, the server path it shows, loaded, zoom),
# the page behind it, and what has the focus (a card control by its data-key).
VIEWER_JS = """() => {
  const v = document.querySelector('#viewer');
  const a = document.activeElement;
  return { hidden: v.hidden, label: v.getAttribute('aria-label'), held: document.body.classList.contains('viewer-open'),
           inert: ['.banner', '.layout', '#composer'].map((s) => document.querySelector(s).hasAttribute('inert')),
           panes: [...v.querySelectorAll('.viewer-pane')].map((p) => { const img = p.querySelector('img');
             return { footer: p.querySelector('figcaption').textContent, src: img.getAttribute('src') || '', path: img.getAttribute('data-src'),
                      loaded: img.complete && img.naturalWidth > 0, scale: Number(p.querySelector('.viewer-stage').getAttribute('data-scale')) }; }),
           focus: a && a !== document.body ? (a.getAttribute('data-key') || a.getAttribute('class') || a.id) : 'body' };
}"""


@pytest.mark.parametrize("ui", [{"tiny_gallery": True}], indirect=True, ids=["tiny_gallery"])
def test_the_viewer_opens_a_similar_xray_beside_yours_zooms_in_and_escape_gives_the_focus_back(ui, images):
    """P6-E on a tiny_gallery server: a similar X-ray of the card opens the viewer side by side, the turn's own X-ray (its original) on the
    left and the match on the right, each loaded through the page's auth (object URLs, never a server path as a src) and footed with its pixel
    size; the page behind is inert. + zooms both panes (linked), the wheel zooms further at the cursor, and Esc closes the viewer, frees the
    page and puts the focus back on the button that opened it. P6-G: first the upload in the user's own message, live and again after a reload,
    opens the original in one pane, as the card's "Your X-ray" tile does, and Esc puts the focus back on that thumbnail."""
    page = ui.open()
    card = ui.turn(images["xray_smooth.png"])   # not noise: the viewer fills the page with this picture, and a screenshot of that is over the evidence limit (P6-F fix 1)
    assert card["status"] == "done"
    page.wait_for_function(NEIGHBOURS_LOADED)

    def the_upload_opens_the_original() -> None:
        thumb = page.locator("#conversation article.turn.user button.thumb-button")
        assert thumb.count() == 1 and thumb.get_attribute("aria-label") == "Open X-ray in the viewer, turn 1"
        thumb.click()
        page.wait_for_function("!document.querySelector('#viewer').hidden && document.querySelectorAll('#viewer img').length === 1"
                               " && [...document.querySelectorAll('#viewer img')].every((i) => i.complete && i.naturalWidth > 0)")
        seen = page.evaluate(VIEWER_JS)
        assert seen["label"] == "Image viewer: Your X-ray" and [p["footer"] for p in seen["panes"]] == ["Your X-ray · 320×320 px"], seen
        assert seen["panes"][0]["path"].endswith("/image?variant=original") and seen["panes"][0]["src"].startswith("blob:"), seen   # through the page's auth
        assert seen["held"] and seen["inert"] == [True, True, True]
        page.keyboard.press("Escape")
        gone = page.evaluate(VIEWER_JS)
        assert gone["hidden"] and not gone["held"] and gone["panes"] == [] and gone["focus"] == "thumb-button", gone   # back on the thumbnail

    the_upload_opens_the_original()   # live: the user turn shows the page's own preview of the file
    ui.reload()                       # replayed: the server's thumbnail in its place
    page.wait_for_function(NEIGHBOURS_LOADED)
    the_upload_opens_the_original()
    opener = page.locator("section.neighbors button[data-action='view']").first
    key = opener.get_attribute("data-key")
    assert key == "neighbor:1"
    opener.click()
    page.wait_for_function("!document.querySelector('#viewer').hidden && document.querySelectorAll('#viewer img').length === 2"
                           " && [...document.querySelectorAll('#viewer img')].every((i) => i.complete && i.naturalWidth > 0)")
    shown = page.evaluate(VIEWER_JS)
    assert shown["label"] == "Image viewer: Your X-ray beside Similar X-ray #1"
    assert [p["footer"] for p in shown["panes"]] == ["Your X-ray · 320×320 px", "Similar X-ray #1 · 320×320 px (stored gallery size)"]
    assert all(p["src"].startswith("blob:") for p in shown["panes"]), shown
    assert shown["panes"][0]["path"].endswith("/image?variant=original") and shown["panes"][1]["path"].startswith("/v1/gallery/images/")
    assert shown["held"] and shown["inert"] == [True, True, True] and [p["scale"] for p in shown["panes"]] == [1, 1]
    ui.shot("viewer_side_by_side_1280x900_light.png")
    page.keyboard.press("+")   # zoom in, both panes (linked)
    assert [p["scale"] for p in page.evaluate(VIEWER_JS)["panes"]] == [1.25, 1.25]
    stage = page.locator("#viewer .viewer-stage").first.bounding_box()
    page.mouse.move(stage["x"] + stage["width"] / 3, stage["y"] + stage["height"] / 3)
    page.mouse.wheel(0, -240)   # and further, at the cursor
    page.wait_for_function("Number(document.querySelector('#viewer .viewer-stage').getAttribute('data-scale')) > 1.25")
    page.keyboard.press("Escape")
    closed = page.evaluate(VIEWER_JS)
    assert closed["hidden"] and not closed["held"] and closed["inert"] == [False, False, False] and closed["panes"] == []
    assert closed["focus"] == key   # back on the similar X-ray that opened it
    ui.assert_clean()


# ---- P9-G5b: what the page says about the model ------------------------------------------------------------------------------------

TEST_NOTICE = "Test model with random weights: this text is not a report. Real reports come from the cluster server."
STOP_HINT_EOS = "Off: this model still ends its report where it was trained to; the token budget is the limit."
EOS_NOTE = "EOS-trained test model: a report ends at the first full stop it writes."   # tests/e2e/conftest.py LAUNCH's "eos" flag


def test_the_tiny_model_is_announced_by_a_badge_in_the_banner_and_a_notice_above_the_report_of_every_card(ui, images):
    """The laptop's dev server runs the random-weights engine, and its output was once read as a broken report. The banner says "test model"
    while it is the default model, and every card it answers says, above the report, that the text is no report: while the turn runs, once it
    is done, after a reload, and not on the answer to a question, where no model ran."""
    page = ui.open()
    page.wait_for_function("!document.querySelector('#test-badge').hidden")
    assert page.text_content("#test-badge") == "test model"
    assert page.evaluate("document.querySelector('.banner').contains(document.querySelector('#test-badge'))")
    assert page.is_visible("#test-badge") and page.evaluate("document.querySelector('.banner').getBoundingClientRect().height") < 160
    ui.attach(images["xray_a.png"])
    page.click("#send")
    ui.wait_running(page)
    assert page.inner_text("#conversation article.card .test-notice") == TEST_NOTICE   # while the report is still being written
    card = ui.wait_settled(0)
    assert card["status"] == "done" and card["notes"] == [] and card["report_notes"] == [REPEAT_NOTE]   # the notice is none of a card's notes
    notice = page.locator("#conversation article.card .test-notice")
    assert notice.count() == 1 and notice.inner_text() == TEST_NOTICE and notice.is_visible()
    assert page.evaluate("(() => { const c = document.querySelector('#conversation article.card');"
                         " return c.querySelector('.test-notice').getBoundingClientRect().bottom <= c.querySelector('.report').getBoundingClientRect().top; })()")
    assert contrast(*page.evaluate("(() => { const s = getComputedStyle(document.querySelector('.test-notice'));"
                                   " return [s.color, s.backgroundColor]; })()")) >= 4.5   # legible on its own ground, which is tinted
    ui.shot("tiny_notice_1280x900_light.png")
    ui.reload()   # the stored card has it too
    assert page.inner_text("#conversation article.card .test-notice") == TEST_NOTICE
    ui.type_note("Is there pneumonia?")   # no model runs for a question: nothing to call a report
    page.click("#send")
    answer = ui.wait_settled(1)
    assert answer["timeline_hidden"] and answer["notes"] == [NOT_A_QA_BOT]
    assert page.locator("#conversation article.card").nth(1).locator(".test-notice").count() == 0
    assert page.locator("#conversation article.card").nth(0).locator(".test-notice").count() == 1
    ui.assert_clean()


@pytest.mark.parametrize("ui", [{"eos": True}], indirect=True, ids=["eos"])
def test_a_model_trained_to_end_its_reports_words_its_stop_switch_for_it_shows_its_note_and_its_chip_says_the_report_ended(ui, images):
    """The harness's "eos" flag makes the tiny engine a model whose card says eos_trained, with the real EOS search ending its report at its
    first full stop. The drawer then words the stop switch for it and lists its evaluation note under the model's name; with the switch off the
    chips never say "full budget", not in the composer and not on the turn, whose chip says the report ended, as the generate stage does."""
    page = ui.open()
    page.click("#settings")
    assert page.inner_text("#stop-hint") == STOP_HINT_EOS
    assert page.locator("#models-section p.model-note").count() == 1 and page.inner_text("#models-section p.model-note") == EOS_NOTE
    assert "evaluation_note" not in page.inner_text("#models-section table")   # said once, as a paragraph: not again in the card's table
    ui.shot("drawer_eos_model_1280x900_light.png")
    page.click("#drawer-close")
    ui.apply_settings(stop_on_repeat=False, max_new_tokens=40)
    assert ui.composer()["chips"] == ["beam 3", "40 tok", "cached"]   # off, and for this model nothing is claimed of the next turn
    card = ui.turn(images["xray_a.png"])
    assert card["status"] == "done" and card["report_notes"] == []   # it ended where the model meant it to: no budget to blame
    assert bubble(ui)["chips"] == ["beam 3", "40 tok", "cached", "ended the report"]
    sid = ui.session_id()
    [message] = assistants(ui, sid)
    detail = stage_detail(message, "generate")
    assert detail["stopped"] == "eos" and 0 < detail["tokens"] < 40 and message["options"]["stop_on_repeat"] is False
    ui.reload()   # the replayed turn says the same, from its stored log
    assert bubble(ui)["chips"] == ["beam 3", "40 tok", "cached", "ended the report"]
    ui.assert_clean()


@pytest.mark.parametrize("ui", [{"public_token": "t"}], indirect=True, ids=["public"])
def test_in_public_mode_a_note_costs_no_turn_and_the_page_says_a_refused_turn_is_the_rate_limit_with_its_wait(ui, images):
    """P8 follow-up (3, 4) in a real browser against a real server: the six notes between an image turn and five re-runs cost nothing, so the
    sixth turn that runs the model is the last one the bucket (six per ten minutes) allows, and the seventh is refused. The notice says what
    the server says, 429 rate_limited, with the wait it asks for in Retry-After, and not the queue's "The server is busy": the note and the
    image are still in the composer, for the next Send."""
    ui.page.context.add_init_script(ui.server.setup_script())   # the token, as Settings holds it, and the shared client id
    page = ui.open()
    card = ui.turn(images["xray_a.png"])   # the first turn: one of the six
    assert card["status"] == "done"
    for n in range(6):   # a note that runs no model: answered with the command list, no stage, and no token
        answer = ui.turn(note="Is there pneumonia?")
        assert answer["status"] == "done" and answer["notes"] == [NOT_A_QA_BOT] and answer["timeline_hidden"], (n, answer)
    for n in range(5):   # five re-runs of the X-ray: each runs the model and takes a token, which makes six
        rerun = ui.turn(note="tokens 16")
        assert rerun["status"] == "done" and not rerun["timeline_hidden"], (n, rerun)
    ui.expect_refusal(429, "POST", r"/messages$")
    ui.type_note("tokens 16")
    page.click("#send")
    page.wait_for_function("!document.querySelector('#notice').hidden")
    composer = ui.composer()
    found = re.fullmatch(r"Too many turns from this client: at most 6 per 10 minutes\. Try again in (\d+) s\.", composer["notice"])
    assert found and 0 < int(found.group(1)) <= 100, composer["notice"]   # the server's message, whose wait is Retry-After's
    assert composer["prompt"] == "tokens 16" and not composer["send_disabled"] and ui.cards() == 12   # nothing was sent for it, nothing was spent
    assert "busy" not in composer["notice"]
    assert len(ui.refusals) == 1 and ui.refusals[0].startswith("HTTP 429 POST")
    ui.shot("rate_limited_1280x900_light.png")
    ui.assert_clean()


def test_a_request_of_the_page_that_the_server_never_answers_gives_up_after_15_seconds_and_says_the_server_cannot_be_reached(ui):
    """P8 follow-up (5) in a real Chrome, whose AbortSignal.timeout and .any are the browser's own, not node's: a half-open tunnel takes the
    list of chats and never answers, and fetch never fails by itself. The page's request gives up at api.js's limit (REQUEST_TIMEOUT_MS, 15 s)
    and the notice is the one every network failure has. The request is held by the test; nothing else is touched."""
    held = []
    ui.page.route("**/v1/sessions?*", lambda route: held.append(route))   # the list the start-up asks for, and nothing answers it
    ui.page.goto(ui.server.url)
    ui.page.wait_for_function("document.querySelector('#mode-badge').textContent !== ''", timeout=30000)   # the rest of the page started
    assert ui.composer()["notice"] is None, "the page does not give up before its limit"
    ui.page.wait_for_function("!document.querySelector('#notice').hidden", timeout=40000)   # at the limit: 15 s after the request
    assert ui.composer()["notice"] == "Cannot reach the server. Check the connection and try again."
    assert held, "the list was asked for, and the test held it"
    ui.page.unroute("**/v1/sessions?*")
    ui.assert_clean()


COMPOSER_BOXES_JS = """() => {
  const q = (s) => document.querySelector(s);
  const box = (el) => { const r = el.getBoundingClientRect(); return [r.left, r.top, r.right, r.bottom]; };
  const c = q('#composer');
  return { composer: box(c), scrolled: c.scrollTop, overflow: c.scrollHeight > c.clientHeight, viewport: [innerWidth, innerHeight],
           at_end: c.scrollTop + c.clientHeight >= c.scrollHeight - 1, saved: q('#saved').textContent,
           settings: box(q('#settings')), send: box(q('#send')), stop: q('#stop').hidden ? null : box(q('#stop')),
           stop_text: q('#stop').textContent, send_disabled: q('#send').disabled };
}"""


def test_a_phone_keeps_cancel_upload_on_the_row_of_send_and_settings_and_in_sight_while_an_upload_goes_up(ui, images):
    """P9-G5b (2c): at 320 x 568 Stop, reading "Cancel upload" while the image goes up, wrapped onto a row of its own, the composer's last, and
    9 px of it showed above the composer's fold. The page itself is held at that moment (its upload is not answered), and Settings, Send and Stop
    are measured on one row, inside the composer and the viewport, with the composer not scrolled. Layout check k (busy) is the sweep; this is the
    page as app.js builds it."""
    page = ui.new_page(viewport=(320, 568))
    ui.open(page)
    held = []
    page.route("**/v1/sessions/*/messages", lambda route: held.append(route))   # the turn's upload is on its way and nothing answers it
    ui.attach(images["xray_a.png"], page)
    page.click("#send")
    page.wait_for_function("document.querySelector('#stop').textContent === 'Cancel upload' && !document.querySelector('#stop').hidden")
    seen = page.evaluate(COMPOSER_BOXES_JS)
    left, top, right, bottom = seen["composer"]
    assert seen["send_disabled"] and seen["scrolled"] == 0 and not seen["overflow"], seen   # the whole composer shows: nothing to scroll to
    assert abs(seen["stop"][1] - seen["send"][1]) <= 2 and abs(seen["settings"][1] - seen["send"][1]) <= 2, seen   # one row, left to right
    assert seen["settings"][2] <= seen["send"][0] <= seen["send"][2] <= seen["stop"][0] and seen["stop"][2] <= right, seen
    assert top <= seen["stop"][1] and seen["stop"][3] <= bottom + 0.5 and seen["stop"][3] <= seen["viewport"][1], seen
    ui.shot("busy_composer_320x568_light.png", page)
    while not held:
        page.wait_for_timeout(20)
    held[0].continue_()   # let the upload through: the turn runs and ends as any turn does
    assert ui.wait_settled(0, page)["status"] == "done"
    ui.assert_clean()


UPLOAD_ON_ITS_WAY_JS = "document.querySelector('#stop').textContent === 'Cancel upload' && !document.querySelector('#stop').hidden"


@pytest.mark.parametrize("order", ["Save, then Send", "Send, then Save"])
def test_a_phone_keeps_cancel_upload_in_sight_when_a_save_comes_before_the_send_or_while_the_upload_goes_up(ui, images, order):
    """P9-G5b fix 1: the test above held only while the composer's content exactly fit. A Save and then a Send within 4 s left "Settings saved." in
    the composer and Stop ("Cancel upload") 49 px under its fold; a Save while the upload went up put the row there and Stop 55 px under. The
    page now takes the row down at Send, and scrolls a busy composer to its end whenever a row comes to it. Both orders in a real Chrome at 320 x
    568, with the upload held: Stop lies inside the composer and the viewport, whether or not it shares Send's row, and the row is where the
    order puts it."""
    page = ui.new_page(viewport=(320, 568))
    ui.open(page)
    held = []
    page.route("**/v1/sessions/*/messages", lambda route: held.append(route))   # the turn's upload is on its way and nothing answers it
    ui.attach(images["xray_a.png"], page)
    saved_first = order == "Save, then Send"

    def save() -> None:
        page.click("#settings")
        page.click("#drawer-save")
        assert ui.composer(page)["saved"] == SAVED, order

    if saved_first:
        save()
    page.click("#send")
    page.wait_for_function(UPLOAD_ON_ITS_WAY_JS)
    if not saved_first:
        save()
    seen = page.evaluate(COMPOSER_BOXES_JS)
    assert seen["send_disabled"] and seen["stop"] is not None, (order, seen)
    assert seen["saved"] == ("" if saved_first else SAVED), (order, seen)   # a Send takes the row down; a Save during the upload puts it there
    left, top, right, bottom = seen["composer"]
    assert top <= seen["stop"][1] and seen["stop"][3] <= bottom + 0.5 and seen["stop"][3] <= seen["viewport"][1], (order, seen)   # in sight, as it is
    assert seen["settings"][3] <= bottom + 0.5 and seen["send"][3] <= bottom + 0.5 and seen["send"][1] >= top, (order, seen)   # and Send and Settings with it
    if seen["overflow"]:   # more in the composer than its rows (the row after a Save during the upload): the page has scrolled it to its end
        assert seen["at_end"], (order, seen)
    while not held:
        page.wait_for_timeout(20)
    held[0].continue_()   # let the upload through: the turn runs and ends as any turn does
    assert ui.wait_settled(0, page)["status"] == "done"
    ui.assert_clean()


# ---- P6-H: the report card a person can read ---------------------------------------------------------------------------------------

# Neutral sentences made up for this test (R7), run through a server whose every report is this text (the harness's report flag) and whose keyword
# labeller (the tiny_gallery server) marks the words "nodule", "atelectasis" and "effusion": the summary has three positive labels. The text has
# a [removed] (the underscores), a sentence about an earlier study, a term said two ways, a decimal, and a repeat of a term already numbered.
CARD_REPORT = ("Findings: Moderate left pleural effusion with basilar atelectasis. The heart size is normal. "
               "Compared with the image from ___, the effusion is stable. The nodule measures 1.5 cm. "
               "Impression: Left pleural effusion. No acute abnormality in the upper zones.")
CARD_LINES = ["Moderate left pleural effusion with basilar atelectasis.", "The heart size is normal.",
              "Compared with the image from [removed], the effusion is stable.", "The nodule measures 1.5 cm.",
              "Left pleural effusion.", "No acute abnormality in the upper zones."]
CARD_TERMS = [("pleural effusion", "fluid between the lung and the chest wall"), ("basilar", "the lower parts of the lungs"),
              ("atelectasis", "part of a lung is partly collapsed or not fully inflated"), ("nodule", "a small round spot"), ("acute", "new or sudden")]
CARD_SUMMARY = ("The report mentions: a spot or mass in the lung (lung lesion); part of a lung is partly collapsed or not fully inflated (atelectasis); "
                "fluid between the lung and the chest wall (pleural effusion).")
REMOVED_NOTE = ("[removed] marks text the training reports hid for privacy, usually a date or a name. The model copies the pattern, "
                "so nothing is missing from your image.")
EARLIER_NOTE = "⚠ marks a sentence that refers to an earlier X-ray. The model never saw one; it writes the way reports are written."

# The newest card's report as it is drawn: its parts in order, the summary, the sentences (without what is only drawn on them), the marks, the notes,
# the terms with their numbers and titles, how a term is underlined, the controls in it, and what Show raw has put in its place.
REPORT_CARD_JS = """() => {
  const c = [...document.querySelectorAll('#conversation article.card')].pop();
  const r = c.querySelector('.report');
  const text = (n) => (n ? n.textContent : null);
  const bare = (n) => { const x = n.cloneNode(true); x.querySelectorAll('sup, .flag').forEach((y) => y.remove()); return x.textContent; };
  const term = r.querySelector('.term');
  const probe = document.createElement('span');
  probe.style.color = 'var(--muted)';
  document.body.append(probe);
  const muted = getComputedStyle(probe).color;
  probe.remove();
  const plain = [...r.querySelectorAll('.report-section li:not(.earlier)')][0];
  const earlier = r.querySelector('li.earlier');
  return {
    parts: [...r.children].map((n) => n.getAttribute('class')),
    summary: { title: text(r.querySelector('.summary-box h3')), text: text(r.querySelector('.summary-box p')), note: text(r.querySelector('.summary-note')) },
    headings: [...r.querySelectorAll('.report-section h3')].map(text),
    levels: [...r.querySelectorAll('h1, h2, h3, h4, h5, h6')].map((n) => n.localName),
    lines: [...r.querySelectorAll('.report-section li')].map(bare),
    paragraphs: r.querySelectorAll('.report-section p').length,
    earlier: [...r.querySelectorAll('li.earlier')].map(bare),
    flags: [...r.querySelectorAll('.flag')].map((n) => [n.textContent, n.getAttribute('role'), n.getAttribute('aria-label')]),
    removed: [...r.querySelectorAll('.removed')].map(text),
    notes: [...r.querySelectorAll('.report-key')].map(text),
    marked: [...r.querySelectorAll('.term')].map((n) => ({ text: bare(n), n: n.querySelector('sup').textContent, title: n.getAttribute('title') })),
    list: [...r.querySelectorAll('.report-terms ol > li')].map(text),
    listHeading: text(r.querySelector('.report-terms h3')),
    underline: term ? [getComputedStyle(term).textDecorationLine, getComputedStyle(term).textDecorationStyle] : null,
    muted: { earlier: earlier ? getComputedStyle(earlier).color === muted : null, plain: plain ? getComputedStyle(plain).color === muted : null },
    controls: [...r.querySelectorAll('a[href], button, input, select, textarea, [tabindex]')].map((n) => [n.localName, n.getAttribute('data-action')]),
    raw: text(r.querySelector('pre.report-raw')),
    extras: r.querySelector('.report-extras') ? r.querySelector('.report-extras').getClientRects().length > 0 : null,
  };
}"""

# The report of the newest card at the top of the conversation, for a picture of it (a card with a gallery is taller than the pane).
REPORT_TO_TOP_JS = ("() => { const pane = document.querySelector('#conversation');"
                    " const report = [...pane.querySelectorAll('article.card')].pop().querySelector('.report');"
                    " pane.scrollTop += report.getBoundingClientRect().top - pane.getBoundingClientRect().top - 8; }")   # the pane only: the page stays

# Every text colour the report card adds, with the background it is read on (the first ancestor that has one).
REPORT_COLOURS_JS = """() => {
  const background = (e) => { for (; e; e = e.parentElement) { const c = getComputedStyle(e).backgroundColor;
    if (c && c !== 'transparent' && !/rgba\\([^)]*,\\s*0\\)$/.test(c)) return c; } return 'rgb(255, 255, 255)'; };
  const pair = (sel) => { const e = document.querySelector('#conversation article.card ' + sel);
    return e ? { color: getComputedStyle(e).color, background: background(e) } : null; };
  return { summary: pair('.summary-box p'), summary_title: pair('.summary-box h3'), caption: pair('.summary-note'), sentence: pair('.report-section li'),
           earlier: pair('li.earlier'), removed: pair('.removed'), number: pair('.term sup'), note: pair('.report-key'),
           terms_heading: pair('.report-terms h3'), terms: pair('.term-list li') };
}"""


@pytest.mark.parametrize("ui", [{"tiny_gallery": True, "report": CARD_REPORT}], indirect=True, ids=["fixed-report"])
def test_the_report_card_is_a_plain_summary_one_sentence_to_a_line_the_notes_and_numbered_terms(ui, images):
    """P6-H in a real Chrome: a finished report reads as a plain summary of its labels, its sentences a line each (a [removed] for the underscores, a
    muted ⚠ before the sentence about an earlier study, each term dotted-underlined and numbered once, in order), the two notes, the Terms, then
    its buttons; nothing in it takes a Tab; Show raw puts the original text in its place, with no note or term; the replayed card is the same;
    and every text it adds is readable in both schemes."""
    page = ui.open()
    ui.apply_settings(max_new_tokens=16)
    card = ui.turn(images["xray_a.png"])
    assert card["status"] == "done" and card["report_notes"] == [] and card["labels"] == []
    assert squeezed(card["shown"]) == squeezed(CARD_REPORT).replace("___", "[removed]")   # the sentences, put back together, are the report as written
    [message] = assistants(ui, ui.session_id())
    assert message["report"] == squeezed(CARD_REPORT) and message["display_report"] == squeezed(CARD_REPORT)   # the stored texts are untouched
    shown = page.evaluate(REPORT_CARD_JS)
    assert shown["parts"] == ["report-summary", "report-body", "report-extras", "report-actions"]   # the summary, the sentences, the notes and terms, the buttons
    assert shown["summary"] == {"title": "What this means (plain language)", "text": CARD_SUMMARY,
                                "note": "A plain-language summary of the findings detected in the model's report. It is not a diagnosis."}
    assert shown["headings"] == ["Findings", "Impression"] and shown["paragraphs"] == 0
    assert shown["levels"] == ["h3", "h3", "h3", "h3"]                                         # the summary's, the two sections', the Terms': one level (fix 1, E)
    assert shown["lines"] == CARD_LINES                                                        # one sentence to a line, 1.5 cm in one piece
    assert shown["earlier"] == [CARD_LINES[2]] and shown["flags"] == [["⚠", "img", "refers to an earlier X-ray"]]
    assert shown["removed"] == ["[removed]"] and shown["notes"] == [REMOVED_NOTE, EARLIER_NOTE]
    assert [(m["text"], m["n"]) for m in shown["marked"]] == [(name, str(i)) for i, (name, _) in enumerate(CARD_TERMS, start=1)]   # once each, in order
    assert [m["title"] for m in shown["marked"]] == ["{}: {}".format(i, why) for i, (_, why) in enumerate(CARD_TERMS, start=1)]
    assert shown["listHeading"] == "Terms" and shown["list"] == ["{}: {}".format(name, why) for name, why in CARD_TERMS]
    assert shown["underline"] == ["underline", "dotted"]                                       # the CSS is in force, not only the class
    assert shown["muted"] == {"earlier": True, "plain": False}                                 # the ⚠ sentence reads quieter than the others
    assert shown["controls"] == [["button", "copy"], ["button", "raw"]]                        # nothing the card adds is a stop of Tab
    assert shown["raw"] is None and shown["extras"] is True
    page.evaluate(REPORT_TO_TOP_JS)   # the picture is of the report, not of the end of the card
    ui.shot("report_card_1280x900_light.png")

    page.click("#conversation article.card button[data-action=raw]")                          # Show raw: the original in place of all of it
    raw = page.evaluate(REPORT_CARD_JS)
    assert raw["raw"] == squeezed(CARD_REPORT) and "___" in raw["raw"] and raw["lines"] == [] and raw["notes"] == [] and raw["list"] == []
    assert raw["extras"] is False and raw["summary"]["text"] == CARD_SUMMARY                   # the summary is about the report, however it is read
    page.click("#conversation article.card button[data-action=raw]")
    assert page.evaluate(REPORT_CARD_JS) == shown                                              # and back, as it was

    ui.reload()                                                                                # the card, replayed from the stored log, is the same
    assert page.evaluate(REPORT_CARD_JS) == shown
    dark = ui.new_page(viewport=DESKTOP, scheme="dark")
    ui.open(dark, "#/s/" + ui.session_id())
    for scheme, target in (("light", page), ("dark", dark)):
        for part, pair in target.evaluate(REPORT_COLOURS_JS).items():
            assert pair is not None, (scheme, part)
            assert contrast(pair["color"], pair["background"]) >= 4.5, (scheme, part, pair, round(contrast(pair["color"], pair["background"]), 2))
    dark.evaluate(REPORT_TO_TOP_JS)
    ui.shot("report_card_1280x900_dark.png", dark)
    ui.assert_clean()


# ---- the harness itself ----------------------------------------------------------------------------------------------------------

PROBE_CONFTEST = """\
from tests.e2e.conftest import *  # noqa: F401,F403  (the harness: its fixtures and its report hook)
from tests.e2e.conftest import _sigterm_runs_teardown  # noqa: F401
"""
PROBE_TEST = """\
def test_a_problem_after_the_last_assertion(ui):
    page = ui.open()
    ui.assert_clean()
    page.evaluate("console.error('late: after the last assertion')")
"""


def test_the_harness_fails_a_test_whose_page_reports_a_problem_after_its_last_assertion(tmp_path):
    """P4-H fix 1: teardown asks the pages again for every test that passed, even one that ended with assert_clean, so a console error,
    a page error or a failed request that comes in late still fails the test. A nested run of one probe test, with this harness."""
    (tmp_path / "conftest.py").write_text(PROBE_CONFTEST)
    (tmp_path / "test_probe.py").write_text(PROBE_TEST)
    probe = subprocess.Popen([sys.executable, "-m", "pytest", str(tmp_path), "-q", "-p", "no:cacheprovider"], cwd=str(tmp_path),
                             env=dict(os.environ, PYTHONPATH=REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        out, _ = probe.communicate(timeout=240)
    finally:
        if probe.poll() is None:   # its harness turns SIGTERM into teardown, so the server and the Chrome it started end with it
            probe.terminate()
            try:
                probe.communicate(timeout=60)
            except subprocess.TimeoutExpired:
                probe.kill()
                probe.communicate()
    assert "1 passed, 1 error" in out, out[-3000:]
    assert "console.error: late: after the last assertion" in out, out[-3000:]
