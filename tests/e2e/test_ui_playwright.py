"""CHAT_UI_PLAN.md P4-H: the chat page driven like a user, in a real Chrome (Playwright), on the tiny engine.

One test per journey (P4-H brief, B3). Each checks what the user sees and what the server stored, and ends with UI.assert_clean: no
console error, no uncaught page error, and no HTTP 4xx or 5xx but the refusals the test provokes on purpose. The harness, the server
and the synthetic images are in conftest.py.

    venv/bin/python -m pytest tests/e2e -m e2e -q
"""
import json
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Tuple

import pytest

pytest.importorskip("playwright.sync_api", reason="the e2e tests need `pip install -r requirements-e2e.txt`")

from app.commands import NOT_A_QA_BOT  # noqa: E402
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
NOT_AN_IMAGE = "Choose a PNG, JPEG or WEBP image."   # the page's own refusal, before anything is sent
TOO_BIG = "The image is over the 20 MB limit."
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
  return { body: pair('body'), card: pair('#conversation article.card'), report: pair('#conversation article.card .report-body p'),
           note: pair('#conversation article.card .note'), provenance: pair('#conversation article.card .provenance p'),
           banner: pair('.banner .disclaimer'), prompt: pair('#prompt'), chip: pair('#chips span'), send: pair('#send'),
           hint: pair('#rerun-hint'), session: pair('#session-list .session-title') };
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
    page.wait_for_function("!document.querySelector('#preview').hidden")
    assert re.fullmatch(r"xray_a\.png · \d+ KB", ui.composer()["preview"])
    assert page.get_attribute("#preview img", "src").startswith("blob:")
    page.click("#preview button")   # Remove
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
