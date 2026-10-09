"""CHAT_UI_PLAN.md P4-E: the browser checklist for the chat page, scripted. A local development tool; validate.sh does not
run it.

It starts the app (tiny engine: random weights, a toy vocabulary, 20 ms between decoding steps; a temporary home; a free
loopback port), or takes --url, and drives headless Chrome over the DevTools protocol (scripts/chat_ui_cdp.py) with real
mouse and key events. The image is a synthetic PNG (tests/app_helpers.png_bytes): nothing here is MIMIC-derived. Ten checks,
one `CHECK <name> PASS|FAIL <detail>` line each:

  stream    a turn streams: all six stages settle, the report grows over many snapshots, then the card says done
  reload    Page.reload shows the identical card (its markup, text and stage states)
  sessions  a second chat switches cleanly with the first, also when a turn is left running: no leaked stream, no stale image
  exports   the JSON and Markdown exports download and open
  stop      Stop aborts a 150-token turn within a step: aborted, fewer than 150 snapshots, the card says stopped
  keyboard  Tab visits the controls in DOM order; Enter sends; Enter on Settings opens the drawer and Esc closes it, to Settings
  narrow    375 x 812: a full turn, its details open, the sidebar and the drawer, with no horizontal overflow
  a11y      the accessibility tree: stage buttons named from their text with their expanded state, label chips that say
            positive or negative, a status region that says "Report ready"
  error     a server 422 shows the dismissible notice and keeps the composer as it was
  settings  (P4-F, P4-G) the drawer is a form that ends in Save, and says what it took. A click into a field that holds a value selects
            all of it, and a budget typed with real keys replaces it (it is not appended) and shows in the composer chips before any
            blur; Enter saves, closes the drawer to Settings and says "Settings saved." for about 4 seconds; "300" and "8" are refused
            with the line "Enter a whole number from 16 to 200." under the field (never snapped to 200 or 16), the drawer stays open
            and Enter changes nothing; the Save button saves. A default turn (the stop switch on) ends by itself with stopped "repeat",
            fewer tokens than its budget and no budget note; a Send with no new image re-runs the last X-ray under a hint that says so
            and uses the budget typed since; with the switch off, at 200 tokens with the page's default (Display repair on) no
            sentence repeats on the settled card or in any frame of the stream, and Show raw still has the decoder's whole text

The tiny model never writes an end-of-report token and loops after about 20 tokens, so with the page's default (stop when the report
starts repeating, on) a tiny turn ends by itself near 21 tokens. The checks that need a long turn (stream, reload, sessions, stop) turn
that switch off first, by storing the setting before the page loads; every check starts from the page's defaults.

No check passes while the page logged a console error or threw. Chips: the tiny pipeline skips retrieval, labelling and
scoring until P5-E, so the a11y check reads its chips from a second tiny app whose home was seeded with one SYNTHETIC
labelled turn (app.store.Store, the same replay path as any stored chat). Screenshots go to docs/chat_ui/evidence/p4e/ with
checklist.json (the result and each check's measurements); --no-evidence runs without writing anything.

    venv/bin/python scripts/chat_ui_browser_check.py [--url http://127.0.0.1:8000/] [--out DIR] [--no-evidence]

Prints `RESULT {"checks": N, "failures": [...]}` last. Exits 1 on any failed check and 2 when Chrome or the app cannot be
started (or the run overruns --timeout seconds). With --url the app should pace its decoding (create_app's
tiny_step_delay_s), or the stop check has no time to press Stop. Every process it starts is killed on exit.
"""
import argparse
import datetime
import io
import json
import os
import platform
import shutil
import signal
import sys
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.chat_ui_cdp import App, Browser, exit_on_sigterm, find_chrome  # noqa: E402
from scripts.repair_generations import split_sentences  # noqa: E402
from tests.app_helpers import png_bytes  # noqa: E402

EVIDENCE_DIR = os.path.join(REPO_ROOT, "docs", "chat_ui", "evidence", "p4e")
STEP_DELAY_S = 0.02            # between two decoding steps of the tiny engine: a 100-token turn takes about 2.6 s
STREAM_TOKENS = 40             # the stream check's budget: a report that, settled, fits the 1280 x 900 page whole
DESKTOP = (1280, 900)
PHONE = (375, 812)
EVIDENCE_FILES = ["streaming_1280x900_light.png", "settled_1280x900_light.png", "settled_1280x900_dark.png", "drawer_1280x900_light.png",
                  "settled_375x812_light.png", "stopped_1280x900_light.png", "error_notice_1280x900_light.png",
                  "labelled_chips_1280x900_light.png", "settings_1280x900_light.png", "drawer_error_1280x900_light.png",
                  "checklist.json"]   # what a full run writes, and the only files it replaces
STAGES = ["preprocess", "encode", "retrieve", "generate", "label", "score"]
DETAILED = ("preprocess", "encode", "generate")   # the stages of the tiny pipeline that have a detail, so a disclosure button
SETTLED = ("done", "skipped")  # what a stage of a finished turn can be
CHEXBERT_14 = ["Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation", "Pneumonia",
               "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding"]
SYNTHETIC_POSITIVE = ["Cardiomegaly", "Edema", "Support Devices"]          # what the seeded turn's "model" found
SYNTHETIC_REFERENCE = ["Cardiomegaly", "Pleural Effusion", "Support Devices"]   # and what its "reference" says
SYNTHETIC_REPORT = "Findings: SYNTHETIC placeholder report for the browser check, not model output. Impression: none."
SYNTHETIC_SHA = "ab" * 32

# ---- the page, as scripts: each is a function that the browser calls with one JSON argument ---------------------------------------

CARD_JS = """() => {
  const cards = [...document.querySelectorAll('#conversation article.card')];
  const c = cards[cards.length - 1];
  if (!c) return null;
  const body = c.querySelector('.report-body');
  const stopped = c.querySelector('.note.stopped');
  return {
    id: c.getAttribute('data-message-id'), status: c.getAttribute('data-status'), cards: cards.length,
    stages: Object.fromEntries([...c.querySelectorAll('.timeline > li')].map((l) => [l.getAttribute('data-stage'), l.getAttribute('data-state')])),
    report: body ? body.textContent : '', stopped: stopped ? stopped.textContent : null, html: c.outerHTML, text: c.textContent,
  };
}"""

# Watches the page's own DOM, so that no render between two polls is missed: every distinct state of the newest card, and every
# thing the status region says. The recorder lives in the page and goes with it.
RECORD_JS = """() => {
  const rec = { t0: performance.now(), frames: [], said: [], texts: new Set(), shown: new Set(), busy: [] };
  window.__rec = rec;
  const last = () => { const all = document.querySelectorAll('#conversation article.card'); return all[all.length - 1] || null; };
  // the report as the card says it: each section's header and body joined back into the text the server sent
  const said = (c) => [...c.querySelectorAll('.report-body .report-section')].map((s) => {
    const h = s.querySelector('h3'), p = s.querySelector('p');
    return (h ? h.textContent + ': ' : '') + (p ? p.textContent : '');
  }).join(' ');
  let key = null;
  new MutationObserver(() => {
    const c = last();
    if (!c) return;
    const body = c.querySelector('.report-body');
    const frame = { t: Math.round(performance.now() - rec.t0), status: c.getAttribute('data-status'),
                    stages: [...c.querySelectorAll('.timeline > li')].map((l) => l.getAttribute('data-state')),
                    report: body ? body.textContent : '', shown: said(c) };
    const k = frame.status + '|' + frame.stages.join(',') + '|' + frame.report + '|' + frame.shown;
    if (k === key) return;
    key = k;
    rec.frames.push(frame);
    if (frame.status === 'running' && frame.report) rec.texts.add(frame.report);
    if (frame.status === 'running' && frame.shown) rec.shown.add(frame.shown);
  }).observe(document.querySelector('#conversation'), { subtree: true, childList: true, attributes: true, characterData: true });
  new MutationObserver(() => rec.busy.push({ t: Math.round(performance.now() - rec.t0),
                                              on: document.querySelector('#conversation').hasAttribute('aria-busy') }))
    .observe(document.querySelector('#conversation'), { attributes: true, attributeFilter: ['aria-busy'] });
  new MutationObserver(() => rec.said.push({ t: Math.round(performance.now() - rec.t0), text: document.querySelector('#status').textContent }))
    .observe(document.querySelector('#status'), { subtree: true, childList: true, characterData: true });
  return true;
}"""

PROGRESS_JS = """() => {
  const r = window.__rec, f = r.frames[r.frames.length - 1];
  return f ? { status: f.status, snapshots: r.texts.size } : null;
}"""

RECORDED_JS = """() => {
  const r = window.__rec;
  return { frames: r.frames, said: r.said, texts: [...r.texts], shown: [...r.shown], busy: r.busy };
}"""

# The tab stops of the page in the order the DOM has them, and the one that has focus, in the same words.
FOCUSABLE_JS = """() => {
  const visible = (el) => { const r = el.getBoundingClientRect(); return r.width > 0 && r.height > 0 && !el.closest('[hidden]')
    && getComputedStyle(el).visibility !== 'hidden' && !el.disabled && el.tabIndex >= 0; };
  const name = (el) => el.id ? '#' + el.id : el.getAttribute('data-stage') ? 'stage:' + el.getAttribute('data-stage')
    : el.closest('li[data-session]') && el.localName === 'a' ? 'chat: ' + el.querySelector('.session-title').textContent
    : el.getAttribute('aria-label') || el.textContent.trim().slice(0, 30) || el.localName;
  window.__stop = () => { const a = document.activeElement; return !a || a === document.body ? 'body' : name(a); };
  window.__ring = () => {   // does the control that has focus show it? an outline of 2 px or more (the page's :focus-visible rule), or a shadow
    const a = document.activeElement;
    if (!a || a === document.body) return null;
    const cs = getComputedStyle(a);
    const outlined = cs.outlineStyle !== 'none' && parseFloat(cs.outlineWidth) >= 2;
    return { shown: outlined || cs.boxShadow !== 'none', outline: cs.outlineStyle + ' ' + cs.outlineWidth };
  };
  return [...document.querySelectorAll('a[href], button, input, select, textarea, [tabindex]')].filter(visible).map(name);
}"""

OVERFLOW_JS = """(args) => {
  const doc = document.documentElement, q = (s) => document.querySelector(s);
  const out = { doc: [doc.scrollWidth, doc.clientWidth], conversation: [q('#conversation').scrollWidth, q('#conversation').clientWidth],
                composer: [q('#composer').scrollWidth, q('#composer').clientWidth], inner: window.innerWidth, wide: [] };
  for (const el of document.querySelectorAll('body *')) {
    const r = el.getBoundingClientRect();
    if (r.width > 0 && r.height > 0 && getComputedStyle(el).visibility !== 'hidden' && !el.closest('[hidden], .visually-hidden')
        && (r.right > args.width + 0.5 || r.left < -0.5)) {   // the visually hidden status region is a clipped 1 px box
      const cls = el.className && el.className.baseVal === undefined ? '.' + String(el.className).split(' ')[0] : '';
      out.wide.push((el.id ? '#' + el.id : el.localName + cls) + ' [' + Math.round(r.left) + ', ' + Math.round(r.right) + ']');
    }
  }
  out.wide = out.wide.slice(0, 8);
  return out;
}"""

MESSAGE_JS = """async (id) => {
  const res = await fetch('/v1/messages/' + id + '?after=0');
  const body = await res.json();
  return { status: body.status, events: body.events.map((e) => [e.event, e.data.status || null]) };
}"""

HEALTH_JS = """async () => (await (await fetch('/healthz')).json())"""

MODELS_JS = """async () => (await (await fetch('/v1/models')).json())"""

# The settings drawer as it reads at the moment, and the composer's chips beside it (P4-F, P4-G).
DRAWER_JS = """() => {
  const q = (s) => document.querySelector(s);
  const note = (id) => { const n = q('#' + id); return n ? { hidden: n.hidden, text: n.textContent } : null; };
  const field = (setting) => { const f = q('#drawer [data-setting="' + setting + '"]');
    return f ? { disabled: f.disabled, checked: f.checked, value: f.value, describedby: f.getAttribute('aria-describedby'),
                 min: f.getAttribute('min'), max: f.getAttribute('max') } : null; };
  const save = q('#drawer-save'), form = q('#settings-form');
  return { open: !q('#drawer').hidden, second: q('#drawer').children[1] ? q('#drawer').children[1].id : null,
           apply: note('apply-note'), running: note('running-note'), retrieval: note('retrieval-note'), labels: note('labels-note'),
           rerun: note('rerun-hint'), k_images: field('k_images'), k_reports: field('k_reports'), label: field('label'),
           repair: field('display_repair'), stop: field('stop_on_repeat'), stop_hint: note('stop-hint'),
           ranges: [...document.querySelectorAll('#drawer label')].map((l) => l.firstChild.textContent).filter((t) => t.indexOf('(') >= 0 && t.indexOf('–') >= 0),
           form: !!form && form.localName === 'form' && form.hasAttribute('novalidate'),
           save: save ? { text: save.textContent, type: save.getAttribute('type'), last: !!form && form.lastElementChild.contains(save),
                          closeFirst: q('#drawer button') === q('#drawer-close') } : null,
           saved: q('#saved').textContent, chips: [...document.querySelectorAll('#chips span')].map((s) => s.textContent) };
}"""

# One number field of the drawer while it is typed in: its text, whether it has focus, what is selected in it, the chips, the setting that
# is stored, and the line under it (shown or not, with aria-invalid) that says what it takes.
FIELD_JS = """(selector) => {
  const f = document.querySelector(selector);
  const e = document.getElementById(f.getAttribute('data-setting') + '-error');
  let stored = null;
  try {
    stored = (JSON.parse(localStorage.getItem('cxrchat.settings') || 'null') || {})[f.getAttribute('data-setting')];
  } catch (x) { stored = 'unreadable'; }
  return { value: f.value, focused: document.activeElement === f, selected: String(window.getSelection()),
           stored: stored === undefined ? null : stored, chips: [...document.querySelectorAll('#chips span')].map((s) => s.textContent),
           error: e ? { hidden: e.hidden, text: e.textContent } : null, invalid: f.getAttribute('aria-invalid'),
           describedby: f.getAttribute('aria-describedby'), drawerOpen: !document.querySelector('#drawer').hidden,
           saved: document.querySelector('#saved').textContent,
           active: document.activeElement.id || document.activeElement.getAttribute('data-setting') || document.activeElement.localName };
}"""

# The newest user turn and card: what the turn says it used (chips, provenance) and whether it carried an image of its own.
TURN_JS = """() => {
  const users = [...document.querySelectorAll('#conversation .turn.user')];
  const cards = [...document.querySelectorAll('#conversation article.card')];
  const u = users[users.length - 1], c = cards[cards.length - 1];
  const prov = c ? c.querySelector('.provenance') : null;
  return { users: users.length, cards: cards.length, id: c ? c.getAttribute('data-message-id') : null,
           chips: u ? [...u.querySelectorAll('.options .chip')].map((x) => x.textContent) : null,
           image: !!(u && u.querySelector('img')), provenance: prov ? prov.textContent : '' };
}"""

# The newest card's report as the page shows it (sections joined back into text), and the raw text when Show raw is on.
SHOWN_JS = """() => {
  const cards = [...document.querySelectorAll('#conversation article.card')];
  const c = cards[cards.length - 1];
  if (!c) return null;
  const shown = [...c.querySelectorAll('.report-body .report-section')].map((s) => {
    const h = s.querySelector('h3'), p = s.querySelector('p');
    return (h ? h.textContent + ': ' : '') + (p ? p.textContent : '');
  }).join(' ');
  const raw = c.querySelector('.report-raw');
  const toggle = c.querySelector('[data-action="raw"]');
  const prov = c.querySelector('.provenance');
  return { shown, raw: raw ? raw.textContent : null, pressed: toggle ? toggle.getAttribute('aria-pressed') : null,
           truncated: !!c.querySelector('.note.truncated'), notes: [...c.querySelectorAll('.report .note')].map((n) => n.textContent),
           repeatNote: !!c.querySelector('.report .note[data-stopped="repeat"]'), footer: prov ? prov.textContent : '' };
}"""

# A message as the server stored it: what it was asked, the image it used, every snapshot streamed, and the two reports.
LOG_JS = """async (id) => {
  const body = await (await fetch('/v1/messages/' + id + '?after=0')).json();
  const first = (name) => body.events.find((e) => e.event === name);
  const start = first('message_start'), stop = first('message_stop');
  const generate = body.events.find((e) => e.event === 'stage_end' && e.data.stage === 'generate');
  return { status: body.status, options: start ? start.data.options : null, image: start ? start.data.image : null,
           generate: generate ? generate.data.detail : null,
           snapshots: body.events.filter((e) => e.event === 'content_block_delta').map((e) => e.data.delta.text),
           skipped: Object.fromEntries(body.events.filter((e) => e.event === 'stage_end' && e.data.skipped).map((e) => [e.data.stage, e.data.skipped])),
           report: stop ? stop.data.report : null, display: stop ? stop.data.display_report : null,
           truncated: stop ? stop.data.truncated_mid_sentence : null };
}"""

# Run by Chrome before the page's own scripts on every load (Context.stop_off): the stored settings, with the stop switch off.
STOP_OFF_JS = """try { var s = JSON.parse(localStorage.getItem('cxrchat.settings') || 'null') || {}; s.stop_on_repeat = false;
  localStorage.setItem('cxrchat.settings', JSON.stringify(s)); } catch (e) { /* storage blocked: the page runs with its defaults */ }"""

# A text that is too faint on purpose: Chrome's contrast audit reports it, so that its silence about everything else is a result and
# not an audit that did not run. It is put on the page for the audit and taken off again.
CANARY_JS = """() => {
  const s = document.createElement('span');
  s.id = 'contrast-canary';
  s.textContent = 'canary';
  s.style.cssText = 'position: fixed; left: 0; bottom: 0; z-index: 1; color: #e6e6e6; background: #ffffff';
  document.body.append(s);
  return true;
}"""

# What each screenshot must show when it is taken, asked of the page itself: a file named "stopped" is not a picture of a
# running turn. Each predicate is the body of a function of no arguments that returns true or false.
SHOT_RULES = {
    "streaming": "const c = document.querySelector('#conversation article.card'); return !!c && c.getAttribute('data-status') === 'running' "
                 "&& !!c.querySelector('.report-body') && !document.querySelector('#stop').hidden",
    "settled": "const c = document.querySelector('#conversation article.card'); return !!c && c.getAttribute('data-status') === 'done' "
               "&& !!c.querySelector('.report-body') && document.querySelector('#stop').hidden",
    "drawer": "return !document.querySelector('#drawer').hidden && document.querySelector('#settings').getAttribute('aria-expanded') === 'true'",
    "stopped": "const c = document.querySelector('#conversation article.card'); return !!c && c.getAttribute('data-status') === 'aborted' "
               "&& !!c.querySelector('.note.stopped')",
    "error_notice": "const n = document.querySelector('#notice'); return !n.hidden && n.getAttribute('role') === 'alert' && n.textContent.length > 4 "
                    "&& !document.querySelector('#preview').hidden",
    "labelled_chips": "return document.querySelectorAll('#conversation li.chip.label').length === 14 "
                      "&& document.querySelectorAll('#conversation li.chip.label.positive').length > 0",
    "settings": "const d = document.querySelector('#drawer'), r = document.querySelector('#rerun-hint'); "
                "return !d.hidden && !document.querySelector('#apply-note').hidden && !r.hidden && r.textContent.indexOf('No new image') === 0 "
                "&& [...document.querySelectorAll('#chips span')].some((c) => c.textContent === '200 tok') "
                "&& document.querySelector('#drawer-save').getBoundingClientRect().bottom <= innerHeight",
    "drawer_error": "const e = document.querySelector('#max_new_tokens-error'), f = document.querySelector('#drawer input[data-setting=\"max_new_tokens\"]'); "
                    "return !document.querySelector('#drawer').hidden && !e.hidden && e.textContent.indexOf('Enter a whole number') === 0 "
                    "&& f.getAttribute('aria-invalid') === 'true' && document.activeElement === f "
                    "&& document.querySelector('#drawer-save').getBoundingClientRect().bottom <= innerHeight",
}


class Failure(Exception):
    """A check's expectation was not met."""


class Overrun(BaseException):
    """The whole run ran out of time. Not an Exception, so that no check swallows it."""


# ---- the run -------------------------------------------------------------------------------------------------------------------

class Context:
    """What the checks share: the browser, the app, the images and the screenshots. A check makes the chats it needs itself, so that
    one that fails does not take the next with it."""

    def __init__(self, browser: Browser, app: Optional[App], base: str, work: str, out_dir: Optional[str]) -> None:
        self.browser, self.app, self.base, self.work, self.out_dir = browser, app, base, work, out_dir
        self.serial = 0
        self.shots = []  # type: List[Dict[str, Any]]
        self.stop_script = None  # type: Optional[str]   # the id of the script that stores "stop_on_repeat: false" before each page loads
        self.console = []  # type: List[str]
        self.images = {}  # type: Dict[str, str]
        self.downloads = os.path.join(work, "downloads")
        os.makedirs(self.downloads)
        for name, (w, h) in {"a": (320, 320), "b": (288, 256), "c": (256, 224)}.items():
            self.images[name] = os.path.join(work, "xray_{}.png".format(name))
            with open(self.images[name], "wb") as handle:
                handle.write(png_bytes(w, h))

    # -- the stop switch: on in the page's defaults, off for a check that needs a turn that runs to its budget
    def stop_off(self) -> None:
        """From the next page load on, the stored settings say stop_on_repeat false. The tiny model loops after about 20 tokens, and with the
        page's default (the switch on) a turn ends there; a check that needs a long turn (a Stop to press, a stream to watch, a reload in
        the middle) asks for this first, and so does not depend on how long the model happens to write."""
        if self.stop_script is None:
            self.stop_script = self.browser.call("Page.addScriptToEvaluateOnNewDocument", source=STOP_OFF_JS)["identifier"]

    def stop_default(self) -> None:
        """Back to the page's defaults: the script is removed and whatever it stored is forgotten. run_check does this before every check."""
        if self.stop_script is not None:
            self.browser.call("Page.removeScriptToEvaluateOnNewDocument", identifier=self.stop_script)
            self.stop_script = None
        forget_settings(self.browser)

    # -- the page
    def open(self, hash_: str = "", viewport: Tuple[int, int] = DESKTOP, scheme: str = "light", base: Optional[str] = None) -> None:
        """A fresh load of the app, at this viewport and colour scheme. The query is unique, so that a stale page cannot pass."""
        b = self.browser
        self.serial += 1
        b.set_viewport(viewport[0], viewport[1], scheme)
        b.call("Page.navigate", url="{}?run={}{}".format(base or self.base, self.serial, hash_))
        b.wait_for("location.search === '?run=%d' && document.readyState === 'complete' && !!document.querySelector('#exports')"
                   % self.serial, what="the page to load")
        b.wait_for("document.querySelector('#mode-badge').textContent !== '' && location.hash !== ''", what="the app to start")

    def card(self) -> Optional[Dict[str, Any]]:
        return self.browser.evaluate("(%s)()" % CARD_JS)

    def wait_card(self, status: Optional[str] = None, timeout: float = 20.0, message: Optional[str] = None) -> Dict[str, Any]:
        cond = "(() => { const c = (%s)(); return c && %s ? c : null; })()" % (
            CARD_JS, "c.status === %s" % json.dumps(status) if status else "true")
        return self.browser.wait_for(cond, timeout, message or "a card" + (" that is " + status if status else ""))

    def attach(self, name: str) -> None:
        """Choose the image as the file dialog would, and wait for the composer to show it."""
        self.browser.set_files("#file", [self.images[name]])
        self.browser.wait_for("!document.querySelector('#preview').hidden && document.querySelector('#preview').textContent.indexOf('xray_%s') >= 0" % name,
                              what="the attached image in the composer")

    def send(self, note: Optional[str] = None) -> None:
        if note:
            self.browser.click("#prompt")
            self.browser.insert_text(note)
        self.browser.click("#send")

    def session_id(self) -> str:
        return self.browser.evaluate("location.hash.replace('#/s/', '')")

    def run_turn(self, name: str, note: Optional[str] = None, timeout: float = 30.0) -> Dict[str, Any]:
        """Attach, send and wait until that turn is over; -> its settled card (the newest: a turn before it does not count)."""
        before = self.browser.evaluate("document.querySelectorAll('#conversation article.card').length")
        self.attach(name)
        self.send(note)
        return self.browser.wait_for("(() => { const c = (%s)(); return c && c.cards > %d && c.status !== 'running' ? c : null; })()" % (CARD_JS, before),
                                     timeout, "the turn to end")

    def shot(self, key: str, name: str, what: str, viewport: Tuple[int, int], scheme: str = "light") -> None:
        """A screenshot, once the page itself says that it shows what `key` claims (SHOT_RULES)."""
        rule = SHOT_RULES[key]
        if not self.browser.evaluate("(() => { %s })()" % rule):
            raise Failure("screenshot {}: the page does not show {}".format(name, what))
        if self.out_dir is None:
            return
        os.makedirs(self.out_dir, exist_ok=True)
        file_name = "{}_{}x{}_{}.png".format(name, viewport[0], viewport[1], scheme)
        path = os.path.join(self.out_dir, file_name)
        size = self.browser.screenshot(path)
        if not self.browser.evaluate("(() => { %s })()" % rule):   # still so after the capture: the picture cannot be of another state
            os.remove(path)
            raise Failure("screenshot {}: the page stopped showing {} while it was being taken".format(name, what))
        self.shots.append({"file": file_name, "shows": what, "viewport": list(viewport), "scheme": scheme, "bytes": size,
                           "page_confirmed": True})

    def drain_console(self) -> None:
        """Pull what the page logged since the last call: errors and uncaught exceptions only."""
        self.browser.evaluate("0")   # an answer from the page: the events that came before it are in the record
        for params in self.browser.events("Runtime.consoleAPICalled"):
            if params.get("type") == "error":
                self.console.append("console.error: " + " ".join(str(a.get("value", a.get("description", ""))) for a in params.get("args", []))[:200])
        for params in self.browser.events("Runtime.exceptionThrown"):
            details = params.get("exceptionDetails", {})
            self.console.append("uncaught: " + str((details.get("exception") or {}).get("description") or details.get("text"))[:200])


def contrast_issues(b: Browser) -> List[str]:
    """Chrome's own contrast audit (WCAG AA) of what the page shows now: one line per text that is too faint, the canary left out."""
    b.call("Audits.enable")
    b.evaluate("(%s)()" % CANARY_JS)
    b.events("Audits.issueAdded")
    b.call("Audits.checkContrast", reportAAA=False)
    found = []  # type: List[Dict[str, Any]]
    deadline = time.time() + 8
    while time.time() < deadline and not any("contrast-canary" in i["selector"] for i in found):   # the canary says the audit is done
        time.sleep(0.1)
        b.evaluate("0")
        for event in b.events("Audits.issueAdded"):
            if event["issue"]["code"] == "LowTextContrastIssue":
                detail = event["issue"]["details"]["lowTextContrastIssueDetails"]
                found.append(dict(detail, selector=detail["violatingNodeSelector"]))
    b.evaluate("document.querySelector('#contrast-canary').remove()")
    if not any("contrast-canary" in i["selector"] for i in found):
        raise Failure("Chrome's contrast audit did not report its canary: the audit did not run")
    return ["{} {:.2f}:1 (needs {})".format(i["selector"], i["contrastRatio"], i["thresholdAA"]) for i in found if "contrast-canary" not in i["selector"]]


def one_line(text: Any, limit: int = 220) -> str:
    out = " ".join(str(text).split())
    return out if len(out) <= limit else out[:limit - 1] + "…"


def expect(condition: Any, message: str) -> None:
    if not condition:
        raise Failure(message)


# ---- the AX tree ------------------------------------------------------------------------------------------------------------------

class Ax:
    """Accessibility.getFullAXTree, indexed by node id and by the DOM node a node stands for."""

    def __init__(self, browser: Browser) -> None:
        nodes = browser.call("Accessibility.getFullAXTree")["nodes"]
        self.by_id = {n["nodeId"]: n for n in nodes}
        self.by_backend = {n["backendDOMNodeId"]: n for n in nodes if n.get("backendDOMNodeId") is not None}

    @staticmethod
    def role(node: Dict[str, Any]) -> str:
        return (node.get("role") or {}).get("value", "")

    @staticmethod
    def name(node: Dict[str, Any]) -> str:
        return (node.get("name") or {}).get("value", "")

    @staticmethod
    def prop(node: Dict[str, Any], key: str) -> Any:
        for p in node.get("properties", []):
            if p["name"] == key:
                return p["value"].get("value")
        return None

    def text(self, node: Dict[str, Any]) -> str:
        """What a screen reader reads of the node's contents: the text of its descendants that are exposed (aria-hidden and
        display: none are not), in order."""
        parts = []  # type: List[str]

        def walk(n: Dict[str, Any]) -> None:
            if self.role(n) == "StaticText" and not n.get("ignored"):
                parts.append(self.name(n))
            for child in n.get("childIds", []):
                if child in self.by_id:
                    walk(self.by_id[child])

        walk(node)
        return "".join(parts)


# ---- the checks --------------------------------------------------------------------------------------------------------------------

def check_stream(ctx: Context) -> Tuple[str, Dict[str, Any]]:
    b = ctx.browser
    ctx.stop_off()   # the report is watched growing over many snapshots: the turn runs to its 40 tokens
    ctx.open("#/new")
    ctx.attach("a")
    b.run(RECORD_JS, None)
    started = time.time()
    ctx.send("tokens %d" % STREAM_TOKENS)
    shot_taken = False
    deadline = time.time() + 40
    while True:
        progress = b.evaluate("(%s)()" % PROGRESS_JS)
        if progress is not None and progress["status"] == "running" and progress["snapshots"] >= 10 and not shot_taken:
            ctx.shot("streaming", "streaming", "a turn that is running, its report part-written, and Stop shown", DESKTOP)
            shot_taken = True
        if progress is not None and progress["status"] != "running":
            break
        if time.time() > deadline:
            raise Failure("the card was still running after 40 s")
        time.sleep(0.04)
    elapsed = time.time() - started
    expect(shot_taken, "the turn ended before 10 snapshots were seen running; no mid-turn picture could be taken")
    b.wait_for("!document.querySelector('#conversation').hasAttribute('aria-busy')", 3, "aria-busy to come off #conversation")   # a frame after the card
    rec = b.evaluate("(%s)()" % RECORDED_JS)
    final = ctx.card()
    texts = rec["texts"]
    frames = rec["frames"]
    expect(final["status"] == "done", "the card ended {}, not done".format(final["status"]))
    expect(sorted(final["stages"]) == sorted(STAGES), "the timeline shows {}".format(sorted(final["stages"])))
    unsettled = {k: v for k, v in final["stages"].items() if v not in SETTLED}
    expect(not unsettled, "stages not settled: {}".format(unsettled))
    seen_running = sorted({STAGES[i] for f in frames for i, s in enumerate(f["stages"]) if s == "running" and i < len(STAGES)})
    expect("generate" in seen_running, "the generate stage was never seen running")
    expect(len(texts) >= 3, "the report changed over {} snapshots, not 3 or more".format(len(texts)))
    busy = rec["busy"]
    expect([x["on"] for x in busy[-2:]] == [True, False], "aria-busy on #conversation went {}, not on and then off".format([x["on"] for x in busy]))
    done_at = next(f["t"] for f in frames if f["status"] == "done")
    expect(busy[-1]["t"] >= done_at, "aria-busy came off at {} ms, before the card was done at {} ms".format(busy[-1]["t"], done_at))
    expect(len(texts[-1]) > len(texts[0]), "the report did not grow: {} then {} characters".format(len(texts[0]), len(texts[-1])))
    expect(final["report"].strip() != "", "the settled card has no report")
    said = [s["text"] for s in rec["said"]]
    expect(said and said[-1] == "Report ready", "the status region ended on {!r}, not 'Report ready'".format(said[-1] if said else None))
    expect("generate running" in said, "the status region never said 'generate running': {}".format(said))
    controls = b.evaluate("({ send: !document.querySelector('#send').disabled, stop: document.querySelector('#stop').hidden })")
    expect(controls == {"send": True, "stop": True}, "after the turn Send/Stop are {}".format(controls))
    ctx.shot("settled", "settled", "a finished turn: stages, the report, provenance", DESKTOP)
    b.set_viewport(DESKTOP[0], DESKTOP[1], "dark")
    ctx.shot("settled", "settled", "the same finished turn in the dark colour scheme", DESKTOP, "dark")
    b.set_viewport(DESKTOP[0], DESKTOP[1], "light")
    states = {s: final["stages"][s] for s in STAGES}
    data = {"stages": states, "stages_seen_running": seen_running, "snapshots": len(texts), "report_chars": [len(texts[0]), len(texts[-1])],
            "frames_recorded": len(frames), "status_region": said, "aria_busy": [[x["on"], x["t"]] for x in busy[-2:]], "card_done_ms": done_at,
            "token_budget": STREAM_TOKENS, "seconds": round(elapsed, 2)}
    done = sum(1 for v in states.values() if v == "done")
    return ("{} stages settled ({} done, {} skipped); {} snapshots, report {} -> {} chars; status ended on 'Report ready'; done in {:.1f} s".format(
        len(states), done, len(states) - done, len(texts), len(texts[0]), len(texts[-1]), elapsed), data)


def first_difference(a: str, b: str) -> str:
    n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return "first difference at {}: {!r} against {!r}".format(n, a[max(0, n - 40):n + 60], b[max(0, n - 40):n + 60])


def check_reload(ctx: Context) -> Tuple[str, Dict[str, Any]]:
    b = ctx.browser
    ctx.stop_off()   # a reload in the middle of a 150-token turn needs a turn that is still running
    ctx.open("#/new")
    before = ctx.run_turn("a", "tokens 24")
    session = ctx.session_id()
    expect(before["status"] == "done", "the turn ended {}".format(before["status"]))
    b.call("Page.reload")
    b.wait_for("document.readyState === 'complete' && !!document.querySelector('#exports')", what="the page to reload")
    after = ctx.wait_card("done", message="the replayed card")
    expect(b.evaluate("location.hash") == "#/s/" + session, "the address is no longer this chat")
    expect(after["id"] == before["id"], "another message is shown: {}".format(after["id"]))
    expect(after["stages"] == before["stages"], "stage states differ: {} against {}".format(before["stages"], after["stages"]))
    expect(after["text"] == before["text"], "text differs: " + first_difference(before["text"], after["text"]))
    expect(after["html"] == before["html"], "markup differs: " + first_difference(before["html"], after["html"]))
    # A reload in the middle of a turn: the page finds the turn running, follows it to its end, and the card it ends with is the one
    # that another reload shows.
    ctx.attach("b")
    ctx.send("tokens 150")
    b.wait_for("(() => { const c = (%s)(); return c && c.cards === 2 && c.status === 'running' && c.report.length > 0 ? c : null; })()" % CARD_JS,
               20, "the second turn to start writing")
    b.call("Page.reload")
    b.wait_for("document.readyState === 'complete' && !!document.querySelector('#exports')", what="the page to reload")
    b.wait_for("(() => { const c = (%s)(); return c && c.cards === 2 && c.status === 'running' ? c : null; })()" % CARD_JS,
               10, "the turn found running after the reload")
    found = b.evaluate("({ send: !document.querySelector('#send').disabled, stop: !document.querySelector('#stop').hidden,"
                       " busy: document.querySelector('#conversation').hasAttribute('aria-busy') })")
    expect(found == {"send": False, "stop": True, "busy": True}, "the page that found a running turn has Send/Stop/aria-busy {}".format(found))
    followed = ctx.wait_card("done", timeout=30, message="the turn the reloaded page was following")
    ended = b.evaluate("({ send: !document.querySelector('#send').disabled, stop: document.querySelector('#stop').hidden })")
    expect(ended == {"send": True, "stop": True}, "when the followed turn ended Send/Stop were {}".format(ended))
    b.call("Page.reload")
    b.wait_for("document.readyState === 'complete' && !!document.querySelector('#exports')", what="the page to reload again")
    replayed = ctx.wait_card("done", message="the replayed second card")
    expect(replayed["id"] == followed["id"] and replayed["cards"] == 2, "another turn is shown after the second reload: {}".format(replayed["id"]))
    expect(replayed["html"] == followed["html"], "the followed card is not the replayed one: " + first_difference(followed["html"], replayed["html"]))
    return ("the card after Page.reload equals the live one ({} characters of markup, the same six stage states); a reload mid-turn followed "
            "the running turn to the card another reload shows".format(len(after["html"])),
            {"markup_chars": len(after["html"]), "text_chars": len(after["text"]), "stages": after["stages"],
             "mid_turn_reload": {"found_running": found, "ended": ended, "followed_equals_replayed": True,
                                 "markup_chars": len(replayed["html"])}})


def blob_images(b: Browser) -> List[str]:
    return b.evaluate("[...document.querySelectorAll('#conversation img')].map((i) => i.getAttribute('src') || '')")


def server_status(base: str, message: str) -> str:
    """A message's status as the server has it, asked from here: the page is under watch and must not be the one that asks."""
    with urllib.request.urlopen("{}v1/messages/{}?after=0".format(base, message), timeout=10) as response:
        return json.load(response)["status"]


def check_sessions(ctx: Context) -> Tuple[str, Dict[str, Any]]:
    b = ctx.browser
    b.call("Network.enable")
    ctx.stop_off()   # a chat is left while its 150-token turn runs, and the turn must still be running to be left
    ctx.open("#/new")
    card_a = ctx.run_turn("a", "tokens 24")
    id_a, msg_a = ctx.session_id(), card_a["id"]
    b.click("#new-session")
    b.wait_for("location.hash === '#/new' && document.querySelectorAll('#conversation > *').length === 0", what="an empty new chat")
    card_b = ctx.run_turn("b", "tokens 24")
    id_b, msg_b = ctx.session_id(), card_b["id"]
    expect(id_b != id_a and msg_b != msg_a, "the second turn did not make a chat of its own")
    rows = b.evaluate("[...document.querySelectorAll('#session-list li')].map((li) => li.getAttribute('data-session'))")
    expect(rows[:2] == [id_b, id_a], "the sidebar lists {} instead of [new, first]".format(rows))

    def show(session: str, message: str, filename: str) -> Dict[str, Any]:
        b.click('#session-list li[data-session="%s"] a' % session)
        card = b.wait_for("(() => { const c = (%s)(); return c && c.id === %s && c.status === 'done' ? c : null; })()" % (CARD_JS, json.dumps(message)),
                          20, "the card of {}".format(session))
        facts = b.evaluate("({ cards: document.querySelectorAll('#conversation article.card').length,"
                           " users: [...document.querySelectorAll('#conversation .turn.user')].map((u) => u.textContent),"
                           " current: [...document.querySelectorAll('#session-list [aria-current]')].map((a) => a.parentNode.getAttribute('data-session')),"
                           " hash: location.hash, preview: document.querySelector('#preview').hidden,"
                           " send: !document.querySelector('#send').disabled, stop: document.querySelector('#stop').hidden })")
        expect(card["cards"] == 1 and facts["cards"] == 1, "{} shows {} cards".format(session, facts["cards"]))
        expect(facts["hash"] == "#/s/" + session, "the address is {}".format(facts["hash"]))
        expect(facts["current"] == [session], "the sidebar marks {} as the current chat".format(facts["current"]))
        expect(len(facts["users"]) == 1 and filename in facts["users"][0], "the user turn is {!r}, not the one for {}".format(facts["users"], filename))
        expect(not [s for s in blob_images(b) if s.startswith("blob:")], "a stale preview image is still on the page")
        expect(facts["send"] and facts["stop"], "the composer is locked in {}".format(session))
        expect(facts["preview"], "an image is attached in the composer of {}".format(session))
        return card

    to_a = show(id_a, msg_a, "xray_a.png")
    expect(to_a["html"] == card_a["html"], "the first chat's card differs from the one it had: " + first_difference(card_a["html"], to_a["html"]))
    to_b = show(id_b, msg_b, "xray_b.png")
    expect(to_b["html"] == card_b["html"], "the second chat's card differs from the one it had: " + first_difference(card_b["html"], to_b["html"]))
    # Leave a chat with a turn running in it: its stream must go with the view, nothing may ask for the turn while the chat is away,
    # and the turn must still be there, whole, when the chat is opened again.
    ctx.attach("c")
    b.events("Network.requestWillBeSent")
    b.events("Network.loadingFailed")
    ctx.send("tokens 150")
    b.wait_for("(() => { const c = (%s)(); return c && c.cards === 2 && c.status === 'running' && c.report.length > 0 ? c : null; })()" % CARD_JS,
               20, "the second turn of the second chat to start writing")
    msg_b2 = ctx.card()["id"]
    b.evaluate("0")
    streams = [p["requestId"] for p in b.events("Network.requestWillBeSent", clear=False)
               if p["request"]["method"] == "POST" and p["request"]["url"].endswith("/messages")]
    expect(len(streams) == 1, "{} turn requests were sent for one turn".format(len(streams)))
    b.click('#session-list li[data-session="%s"] a' % id_a)
    b.wait_for("(() => { const c = (%s)(); return c && c.id === %s ? c : null; })()" % (CARD_JS, json.dumps(msg_a)), 20, "the first chat's card")
    samples = 0
    deadline = time.time() + 20
    while server_status(ctx.base, msg_b2) == "running":
        if time.time() > deadline:
            raise Failure("the turn left running in the second chat never ended")
        seen = b.evaluate("({ ids: [...document.querySelectorAll('#conversation article.card')].map((c) => c.getAttribute('data-message-id')),"
                          " send: !document.querySelector('#send').disabled, stop: document.querySelector('#stop').hidden })")
        samples += 1
        expect(seen["ids"] == [msg_a], "the first chat shows {} while the second one's turn runs".format(seen["ids"]))
        expect(seen["send"] and seen["stop"], "the first chat's composer was locked by the other chat's turn")
        time.sleep(0.08)
    expect(samples >= 1, "the turn ended before the other chat could be watched: nothing was sampled")   # else every assertion above is vacuous
    time.sleep(0.4)   # and a little longer: nothing may still arrive
    after = b.evaluate("({ ids: [...document.querySelectorAll('#conversation article.card')].map((c) => c.getAttribute('data-message-id')),"
                       " img: [...document.querySelectorAll('#conversation img')].length })")
    expect(after["ids"] == [msg_a] and after["img"] == 0, "the first chat changed after the other turn ended: {}".format(after))
    b.evaluate("0")
    dropped = [p for p in b.events("Network.loadingFailed", clear=False) if p["requestId"] == streams[0]]
    expect(dropped and dropped[0].get("canceled") is True, "the stream of the turn was not dropped when its chat was left: {}".format(dropped))
    asked = [p["request"]["url"] for p in b.events("Network.requestWillBeSent") if p["request"]["method"] == "GET" and msg_b2 in p["request"]["url"]]
    expect(not asked, "the page asked for the turn of the chat it had left: {}".format(asked))
    b.click('#session-list li[data-session="%s"] a' % id_b)
    cards = b.wait_for("(() => { const all = [...document.querySelectorAll('#conversation article.card')];"
                       " const done = all.every((c) => c.getAttribute('data-status') === 'done');"
                       " return all.length === 2 && done ? all.map((c) => c.getAttribute('data-message-id')) : null; })()",
                       20, "both turns of the second chat, done")
    expect(cards == [msg_b, msg_b2], "the second chat shows {} instead of [{}, {}]".format(cards, msg_b, msg_b2))
    report2 = b.evaluate("document.querySelectorAll('#conversation article.card')[1].querySelector('.report-body').textContent")
    expect(len(report2) > 100, "the turn that finished while the chat was left has a {}-character report".format(len(report2)))
    expect(not [s for s in blob_images(b) if s.startswith("blob:")], "a stale preview image is on the second chat")
    return ("2 chats switched 4 times, each with its own card and user turn; leaving a running turn cancelled its stream (request aborted), "
            "nothing asked for it ({} samples), and it replays done".format(samples),
            {"sessions": [id_a, id_b], "messages": {"a": msg_a, "b": msg_b, "b2": msg_b2}, "samples_while_away": samples,
             "stream_request_cancelled": True, "report_chars_of_turn_finished_while_away": len(report2)})


def wait_download(directory: str, name: str, timeout: float = 15.0) -> str:
    path = os.path.join(directory, name)
    deadline = time.time() + timeout
    last = -1
    while time.time() < deadline:
        if os.path.exists(path):
            size = os.path.getsize(path)
            if size > 0 and size == last:
                return path
            last = size
        time.sleep(0.15)
    raise Failure("{} was not downloaded within {:g} s (the folder has {})".format(name, timeout, sorted(os.listdir(directory))))


def check_exports(ctx: Context) -> Tuple[str, Dict[str, Any]]:
    b = ctx.browser
    b.call("Browser.setDownloadBehavior", behavior="allow", downloadPath=ctx.downloads, eventsEnabled=True)
    ctx.open("#/new")
    ctx.run_turn("a", "tokens 24")
    session = ctx.session_id()
    ctx.run_turn("b", "tokens 24")   # a second turn in the same chat
    ctx.open("#/s/" + session)       # exported from a page that replays it
    ctx.wait_card("done", message="the chat to export")
    expect(b.evaluate("!document.querySelector('#exports').hidden"), "the export buttons are not shown for an open chat")
    b.click('#exports button[data-format="json"]')
    path = wait_download(ctx.downloads, "session-{}.json".format(session))
    with open(path, encoding="utf-8") as handle:
        doc = json.load(handle)
    expect(doc["session"]["id"] == session, "the JSON is for session {}".format(doc["session"]["id"]))
    roles = [m["role"] for m in doc["messages"]]
    expect(roles == ["user", "assistant", "user", "assistant"], "the JSON has messages {}".format(roles))
    last_events = [m["events"][-1] for m in doc["messages"] if m["role"] == "assistant"]
    expect(all(e["event"] == "message_stop" and e["data"]["status"] == "done" for e in last_events), "an exported log does not end in message_stop done")
    b.click('#exports button[data-format="md"]')
    md_path = wait_download(ctx.downloads, "session-{}.md".format(session))
    with open(md_path, encoding="utf-8") as handle:
        md = handle.read()
    expect("## Turn 1" in md and "## Turn 2" in md, "the Markdown has no '## Turn 1' and '## Turn 2'")
    expect("Research prototype; not for clinical use." in md, "the Markdown carries no disclaimer")
    expect(b.evaluate("document.querySelector('#notice').hidden"), "the page showed a notice during the exports")
    return ("JSON parses ({} bytes, {} messages, logs end in message_stop done); Markdown ({} bytes) has '## Turn 1' and '## Turn 2'".format(
        os.path.getsize(path), len(doc["messages"]), os.path.getsize(md_path)),
            {"json": {"file": os.path.basename(path), "bytes": os.path.getsize(path), "messages": len(doc["messages"]),
                      "events_per_turn": [len(m["events"]) for m in doc["messages"] if m["role"] == "assistant"]},
             "markdown": {"file": os.path.basename(md_path), "bytes": os.path.getsize(md_path),
                          "headings": [line for line in md.splitlines() if line.startswith("## ")]}})


def snapshot_count(b: Browser, message: str) -> Tuple[int, str]:
    log = b.evaluate("(%s)(%s)" % (MESSAGE_JS, json.dumps(message)))
    return sum(1 for e in log["events"] if e[0] == "content_block_delta"), log["status"]


def check_stop(ctx: Context) -> Tuple[str, Dict[str, Any]]:
    b = ctx.browser
    ctx.stop_off()   # Stop is pressed in a turn that is writing: with the switch on the tiny turn ends by itself in about 30 tokens and the check flakes
    ctx.open("#/new")
    ctx.attach("a")
    ctx.send("tokens 150")
    card = b.wait_for("(() => { const c = (%s)(); return c && c.status === 'running' && c.report.length > 0 ? c : null; })()" % CARD_JS, 20,
                      "the first snapshot of a 150-token turn")
    message = card["id"]
    b.wait_for("!document.querySelector('#stop').hidden && !document.querySelector('#stop').disabled", 5, "Stop to be usable")
    at_press, _ = snapshot_count(b, message)
    pressed = time.time()
    b.click("#stop")
    settled = b.wait_for("(() => { const c = (%s)(); return c && c.status !== 'running' ? c : null; })()" % CARD_JS, 10, "the stopped card")
    seconds = time.time() - pressed
    expect(settled["status"] == "aborted", "the card ended {}, not aborted".format(settled["status"]))
    expect(settled["stopped"] and "stopped" in settled["stopped"].lower(), "the card's note is {!r}, not 'stopped'".format(settled["stopped"]))
    total, status = snapshot_count(b, message)
    expect(status == "aborted", "the server stored the turn as {}".format(status))
    expect(total < 150, "all {} snapshots were produced".format(total))
    expect(seconds < 2.0, "the card took {:.1f} s to say aborted after Stop was pressed".format(seconds))
    expect(total - at_press <= 20, "{} more snapshots (steps of 20 ms) were produced after Stop was pressed".format(total - at_press))
    time.sleep(0.5)
    later_total, _ = snapshot_count(b, message)
    expect(later_total == total, "the turn kept producing after it was stopped: {} then {}".format(total, later_total))
    health = b.evaluate("(%s)()" % HEALTH_JS)
    expect(health["turns_in_flight"] == 0, "the server still counts {} turns in flight".format(health["turns_in_flight"]))
    controls = b.evaluate("({ send: !document.querySelector('#send').disabled, stop: document.querySelector('#stop').hidden,"
                          " label: document.querySelector('#stop').textContent, said: document.querySelector('#status').textContent })")
    expect(controls["send"] and controls["stop"] and controls["label"] == "Stop", "after the stop the controls are {}".format(controls))
    expect(controls["said"] == "Turn stopped", "the status region says {!r}".format(controls["said"]))
    skipped = sorted(k for k, v in settled["stages"].items() if v == "skipped")
    ctx.shot("stopped", "stopped", "a turn that was stopped part-way: 'Turn stopped', the report so far", DESKTOP)
    return ("stopped after {} of 150 snapshots ({} more by the time the card said aborted, {:.2f} s after the press); no more afterwards; note {!r}".format(
        total, total - at_press, seconds, settled["stopped"]),
            {"budget": 150, "snapshots_when_pressed": at_press, "snapshots_total": total, "seconds_to_aborted": round(seconds, 2),
             "note": settled["stopped"], "server_status": status, "stages": settled["stages"], "skipped": skipped})


def check_keyboard(ctx: Context) -> Tuple[str, Dict[str, Any]]:
    b = ctx.browser
    ctx.open("#/new")
    b.set_files("#file", [ctx.images["a"]])   # the file dialog is the one thing a script cannot press a key in
    b.wait_for("!document.querySelector('#preview').hidden", what="the attached image")
    stops = []  # type: List[str]
    on_page = b.evaluate("(%s)()" % FOCUSABLE_JS)   # the tab stops of this page, in DOM order: the walk to the note field cannot need more
    for _ in range(len(on_page) + 1):   # from the top of the page, Tab until the note field has focus
        b.key("Tab", "Tab", 9)
        stops.append(b.evaluate("window.__stop()"))
        if stops[-1] == "#prompt":
            break
    expect(stops and stops[-1] == "#prompt", "Tab never reached the note field: {}".format(stops))
    expect("#image-well" in stops and stops.index("#image-well") < stops.index("#prompt"), "Tab skipped the image well: {}".format(stops))
    b.type_text("tokens 20")
    expect(b.evaluate("document.querySelector('#prompt').value") == "tokens 20", "typing did not reach the note field")
    b.key("Enter", "Enter", 13, text="\r")
    card = b.wait_for("(() => { const c = (%s)(); return c && c.status !== 'running' ? c : null; })()" % CARD_JS, 30, "the turn that Enter sent")
    expect(card["status"] == "done", "Enter sent a turn that ended {}".format(card["status"]))
    expect(b.evaluate("document.activeElement.id") == "prompt", "focus is on {}, not back in the note field".format(b.evaluate("document.activeElement.id")))
    # A page with a card on it: the whole tab order, from the top, against the DOM order.
    b.call("Page.reload")
    b.wait_for("document.readyState === 'complete' && !!document.querySelector('#exports')", what="the page to reload")
    ctx.wait_card("done", message="the replayed card")
    expected = b.evaluate("(%s)()" % FOCUSABLE_JS)
    walked = []  # type: List[str]
    rings = []  # type: List[Any]
    for _ in range(len(expected) + 2):
        b.key("Tab", "Tab", 9)
        walked.append(b.evaluate("window.__stop()"))
        rings.append(b.evaluate("window.__ring()"))
    bare = [w for w, r in zip(walked, rings) if r is not None and not r["shown"]]
    expect(not bare, "these tab stops show no focus ring: {}".format(bare))
    expect(walked[:len(expected)] == expected, "Tab order differs from DOM order: {} against {}".format(walked[:len(expected)], expected))
    expect(walked[len(expected)] == "body", "after the last control Tab stays in the page: {}".format(walked[len(expected)]))
    def at(name: str) -> int:
        expect(name in expected, "{} is not a tab stop (stops: {})".format(name, expected))
        return expected.index(name)
    chats = [i for i, name in enumerate(expected) if name.startswith("chat: ")]
    expect(chats, "no chat link is a tab stop: {}".format(expected))
    sidebar = [at("#new-session"), at("Export JSON"), at("Export Markdown")] + chats
    stages = [at("stage:preprocess"), at("stage:encode"), at("stage:generate")]
    composer = [at("#image-well"), at("#prompt"), at("#settings"), at("#send")]
    expect(sidebar == sorted(sidebar) and max(sidebar) < min(stages) and max(stages) < min(composer) and composer == sorted(composer),
           "the order is not sidebar (with its chats), then the card's stage buttons, then the composer: {}".format(expected))
    expect(any(s.startswith("Delete chat") for s in expected) and any(s.startswith("Copy report") for s in expected),
           "the chat's Delete or the card's Copy is not a tab stop: {}".format(expected))
    # A stage's details open and close from the keyboard, with the focus staying on the button.
    for _ in range(len(expected) + 1):
        if b.evaluate("window.__stop()") == "stage:encode":
            break
        b.key("Tab", "Tab", 9)
    expect(b.evaluate("window.__stop()") == "stage:encode", "Tab never came round to the encode button")
    toggled = []
    for label, press in (("Enter", lambda: b.key("Enter", "Enter", 13, text="\r")), ("Space", lambda: b.key(" ", "Space", 32, text=" "))):
        press()
        toggled.append(b.evaluate("({ expanded: document.activeElement.getAttribute('aria-expanded'),"
                                  " open: document.activeElement.closest('li').classList.contains('open'), focus: window.__stop() })"))
    expect(toggled == [{"expanded": "true", "open": True, "focus": "stage:encode"}, {"expanded": "false", "open": False, "focus": "stage:encode"}],
           "Enter then Space on the encode button gave {}".format(toggled))
    # Settings by keyboard: Enter opens the drawer and moves focus into it; Esc closes it and gives focus back to Settings.
    for _ in range(len(expected) + 1):
        if b.evaluate("window.__stop()") == "#settings":
            break
        b.key("Tab", "Tab", 9)
    expect(b.evaluate("window.__stop()") == "#settings", "Tab never came back round to Settings")
    b.key("Enter", "Enter", 13, text="\r")
    opened = b.evaluate("({ hidden: document.querySelector('#drawer').hidden, expanded: document.querySelector('#settings').getAttribute('aria-expanded'),"
                        " focus: window.__stop() })")
    expect(opened["hidden"] is False and opened["expanded"] == "true", "Enter on Settings left the drawer {}".format(opened))
    expect(b.evaluate("document.querySelector('#drawer').contains(document.activeElement)"), "focus did not move into the drawer: {}".format(opened["focus"]))
    ctx.shot("drawer", "drawer", "the settings drawer open, focus inside it", DESKTOP)
    b.key("Escape", "Escape", 27)
    closed = b.evaluate("({ hidden: document.querySelector('#drawer').hidden, expanded: document.querySelector('#settings').getAttribute('aria-expanded'),"
                        " focus: window.__stop() })")
    expect(closed == {"hidden": True, "expanded": "false", "focus": "#settings"}, "Esc left {}".format(closed))
    return ("Tab visits {} controls in DOM order, each with a focus ring; Enter sends; Enter and Space toggle a stage; "
            "Enter opens the drawer and Esc closes it to Settings".format(len(expected)),
            {"tab_stops": expected, "chat_links_among_them": len(chats), "every_stop_shows_a_focus_ring": True, "stops_to_note_field": stops,
             "after_enter": {"status": card["status"], "focus": "prompt"}, "stage_toggle": {"Enter": toggled[0], "Space": toggled[1]},
             "drawer": {"opened": opened, "closed": closed}})


def check_narrow(ctx: Context) -> Tuple[str, Dict[str, Any]]:
    b = ctx.browser
    width = PHONE[0]
    ctx.open("#/new", PHONE)
    measured = {}  # type: Dict[str, Any]

    def measure(state: str) -> None:
        m = b.run(OVERFLOW_JS, {"width": width})
        measured[state] = m
        expect(m["inner"] == width, "the viewport is {} px wide, not {}".format(m["inner"], width))
        for name, (sw, cw) in (("document", m["doc"]), ("#conversation", m["conversation"]), ("#composer", m["composer"])):
            expect(sw <= cw, "{} overflows horizontally in the {} state: scrollWidth {} > clientWidth {}".format(name, state, sw, cw))
        expect(not m["wide"], "in the {} state these reach past the {} px: {}".format(state, width, m["wide"]))

    measure("empty chat")
    card = ctx.run_turn("a", "tokens 24")
    expect(card["status"] == "done", "the turn ended {}".format(card["status"]))
    measure("settled card")
    b.evaluate("document.querySelector('#conversation').scrollTop = 0")   # the page follows a turn to its end: from the top, the stage pills show
    ctx.shot("settled", "settled", "a finished turn on a phone-width page, from its top: the image, the stage pills, the report", PHONE)
    for stage in ("preprocess", "encode", "generate"):
        b.click('#conversation li[data-stage="%s"] > button' % stage)
    expect(b.evaluate("document.querySelectorAll('#conversation li.open').length") == 3, "the three stage details did not open")
    measure("stage details open")
    b.click("#sidebar-toggle")
    expect(b.evaluate("document.body.classList.contains('sidebar-open')"), "the sidebar toggle did not open the sidebar")
    measure("sidebar open")
    b.key("Escape", "Escape", 27)
    expect(not b.evaluate("document.body.classList.contains('sidebar-open')"), "Esc did not close the sidebar")
    b.click("#settings")
    expect(b.evaluate("!document.querySelector('#drawer').hidden"), "Settings did not open the drawer")
    measure("drawer open")
    reach = b.evaluate("(() => { const r = document.querySelector('#send').getBoundingClientRect();"
                       " return r.bottom <= innerHeight && r.right <= innerWidth && r.left >= 0; })()")
    expect(reach, "Send is outside the viewport")
    return ("no horizontal overflow at {} px in 5 states (empty chat, settled card, stage details open, sidebar open, drawer open)".format(width),
            {"viewport": list(PHONE), "overflow": {state: {"document": m["doc"], "conversation": m["conversation"], "composer": m["composer"]}
                                                  for state, m in measured.items()}})


def seed_labelled_home(home: str) -> None:
    """One SYNTHETIC finished turn in a fresh CHAT_HOME, with the label and score stages the tiny pipeline does not run yet."""
    from app.schemas import DISCLAIMER
    from app.store import Store
    options = {"model": "tiny", "decode": "beam", "beam_size": 3, "max_new_tokens": 100, "cached_decode": True, "compile": False, "k_images": 4,
               "k_reports": 3, "label": True, "reference": None, "display_repair": False, "test_row": None}
    card = {"name": "tiny", "checkpoint": None, "checkpoint_sha256": None, "prefix_k": 4, "scan_impl": "legacy", "tfla_impl": "exact",
            "device": "cpu", "cached_decode_available": True,
            "drift_note": "SYNTHETIC turn seeded by scripts/chat_ui_browser_check.py: not model output"}
    generated = {n: 1 if n in SYNTHETIC_POSITIVE else 0 for n in CHEXBERT_14}
    reference = {n: 1 if n in SYNTHETIC_REFERENCE else 0 for n in CHEXBERT_14}
    store = Store(Path(home))
    try:
        session = store.create_session("private")
        user_id, message_id = store.start_turn(session["id"], "", "private", options, image_sha256=SYNTHETIC_SHA, image_filename="synthetic.png")
        events = [
            ("message_start", {"message_id": message_id, "user_message_id": user_id, "session_id": session["id"], "mode": "private", "model": card,
                               "options": options, "image": {"sha256": SYNTHETIC_SHA, "filename": "synthetic.png", "source": "upload", "urls": {}}}),
            ("stage_start", {"stage": "preprocess", "index": 0}),
            ("stage_end", {"stage": "preprocess", "ms": 1.0, "detail": {"format": "PNG", "resized_to": [224, 224]}}),
            ("stage_start", {"stage": "encode", "index": 1}),
            ("stage_end", {"stage": "encode", "ms": 2.0, "detail": {"patch_grid": [197, 32], "pooled_dim": 16, "prefix_tokens": 4}}),
            ("stage_end", {"stage": "retrieve", "skipped": "gallery_unavailable"}),
            ("stage_start", {"stage": "generate", "index": 3}),
            ("content_block_start", {"index": 0, "content_block": {"type": "report", "text": ""}}),
            ("content_block_delta", {"index": 0, "delta": {"type": "beam_snapshot", "step": 0, "text": SYNTHETIC_REPORT}}),
            ("content_block_stop", {"index": 0}),
            ("stage_end", {"stage": "generate", "ms": 3.0, "detail": {"decode": "beam", "beam_size": 3, "tokens": 20, "device": "cpu"}}),
            ("stage_start", {"stage": "label", "index": 4}),
            ("stage_end", {"stage": "label", "ms": 1.0, "detail": {"chexbert_14": generated, "positives": SYNTHETIC_POSITIVE}}),
            ("stage_start", {"stage": "score", "index": 5}),
            ("stage_end", {"stage": "score", "ms": 1.0, "detail": {"rouge_l": 0.25, "bleu_1": 0.5, "bleu_4": 0.125, "chexbert_14_micro_f1": 0.75,
                                                                  "exact_match_14": False, "reference_source": "user",
                                                                  "reference_chexbert_14": reference}}),
            ("message_stop", {"message_id": message_id, "status": "done", "total_ms": 8.0, "report": SYNTHETIC_REPORT,
                              "display_report": SYNTHETIC_REPORT, "truncated_mid_sentence": False, "disclaimer": DISCLAIMER}),
        ]
        for event, data in events:
            store.append_event(message_id, event, data)
        store.finish_turn(message_id, "done", report=SYNTHETIC_REPORT, display_report=SYNTHETIC_REPORT, provenance=card, total_ms=8.0)
    finally:
        store.close()


def check_a11y(ctx: Context) -> Tuple[str, Dict[str, Any]]:
    b = ctx.browser
    b.call("Accessibility.enable")
    # -- a live turn: the stage buttons and the status region
    ctx.open("#/new")
    b.run(RECORD_JS, None)
    ctx.run_turn("a", "tokens 24")
    said = [s["text"] for s in b.evaluate("(%s)()" % RECORDED_JS)["said"]]
    expect(said and said[-1] == "Report ready", "the status region ended on {!r}".format(said[-1] if said else None))
    ax = Ax(b)
    exposed = [n for n in ax.by_id.values() if not n.get("ignored") and Ax.prop(n, "focusable") is True and Ax.role(n) != "RootWebArea"]
    unnamed = [(Ax.role(n), n.get("backendDOMNodeId")) for n in exposed if not Ax.name(n).strip()]
    expect(not unnamed, "controls a user can focus have no accessible name: {}".format(unnamed))
    buttons = []
    ids = b.backend_ids('#conversation li[data-stage] > button')
    expect(len(ids) == len(DETAILED), "{} stage buttons on the page, not {}".format(len(ids), len(DETAILED)))   # zip would stop at the shorter
    for stage, backend in zip(DETAILED, ids):
        node = ax.by_backend.get(backend)
        expect(node is not None, "the {} button is not in the accessibility tree".format(stage))
        visible = b.evaluate("document.querySelector('#conversation li[data-stage=\"%s\"] > button').textContent" % stage)
        expect(visible.strip(), "the {} button has no visible text".format(stage))   # an empty text is the start of every name
        name = Ax.name(node)
        expect(Ax.role(node) == "button", "the {} control is a {}, not a button".format(stage, Ax.role(node)))
        expect(name.startswith(visible) and len(name) > len(visible),
               "the {} button is named {!r}, which does not start with its text {!r} and add to it".format(stage, name, visible))
        expect(name[:1].isalpha(), "the {} button is named {!r}: a glyph comes before its label".format(stage, name))
        expect(Ax.prop(node, "expanded") is False, "the {} button's expanded state is {!r}, not false".format(stage, Ax.prop(node, "expanded")))
        buttons.append({"stage": stage, "visible": visible, "name": name, "expanded": False})
    expect(len(buttons) == len(DETAILED), "{} stage buttons were checked, not {}".format(len(buttons), len(DETAILED)))
    b.click('#conversation li[data-stage="encode"] > button')
    ax2 = Ax(b)
    opened = ax2.by_backend[b.backend_ids('#conversation li[data-stage="encode"] > button')[0]]
    expect(Ax.prop(opened, "expanded") is True, "after a click the encode button's expanded state is {!r}".format(Ax.prop(opened, "expanded")))
    buttons[1]["expanded_after_click"] = True
    items = {}  # type: Dict[str, str]
    for stage in STAGES:
        li = ax2.by_backend[b.backend_ids('#conversation li[data-stage="%s"]' % stage)[0]]
        items[stage] = ax2.text(li)
    glyphed = {stage: spoken[:12] for stage, spoken in items.items() if not spoken[:1].isalpha()}
    expect(not glyphed, "stage items read a glyph before their label (the ::before shape is part of what a screen reader hears): {}".format(glyphed))
    plain = []
    for stage in ("retrieve", "label", "score"):
        expect(stage in items[stage] and "skipped" in items[stage], "the skipped {} stage reads as {!r}".format(stage, items[stage]))
        plain.append(items[stage])
    status = [n for n in ax2.by_id.values() if Ax.role(n) == "status"]
    expect(len(status) >= 1, "no status region in the accessibility tree")
    region = next((n for n in status if ax2.text(n) == "Report ready"), None)
    expect(region is not None, "no status region reads 'Report ready' (they read {})".format([ax2.text(n) for n in status]))
    expect(Ax.prop(region, "live") == "polite", "the status region is live={!r}, not polite".format(Ax.prop(region, "live")))
    faint = {}  # type: Dict[str, List[str]]
    faint["settled card, light"] = contrast_issues(b)
    b.click("#settings")
    faint["drawer open, light"] = contrast_issues(b)
    b.set_viewport(DESKTOP[0], DESKTOP[1], "dark")
    faint["drawer open, dark"] = contrast_issues(b)
    b.key("Escape", "Escape", 27)
    faint["settled card, dark"] = contrast_issues(b)
    b.set_viewport(DESKTOP[0], DESKTOP[1], "light")
    expect(not any(faint.values()), "text below WCAG AA contrast: {}".format({k: v[:3] for k, v in faint.items() if v}))
    # -- a labelled turn, on a second app whose home was seeded: the chips
    home = tempfile.mkdtemp(prefix="chat_ui_seed_")
    seeded = None
    try:
        seed_labelled_home(home)
        seeded = App(home=home, static_dir=ctx.app.static_dir if ctx.app is not None else None)   # the page under test, over the seeded chat
        ctx.open("", DESKTOP, "light", base=seeded.url)
        ctx.wait_card("done", message="the seeded turn")
        ax3 = Ax(b)
        chips = []
        for backend, label, value, agree in zip(
                b.backend_ids("#conversation li.chip.label"),
                *[b.evaluate("[...document.querySelectorAll('#conversation li.chip.label')].map((c) => c.getAttribute(%s))" % json.dumps(a))
                  for a in ("data-label", "data-value", "data-agree")]):
            node = ax3.by_backend.get(backend)
            expect(node is not None, "the {} chip is not in the accessibility tree".format(label))
            spoken = ax3.text(node)
            state = "positive" if value == "1" else "negative"
            expect(spoken.startswith(label) and (": " + state) in spoken, "the {} chip reads {!r}, not '{}: {}'".format(label, spoken, label, state))
            if agree == "true":
                expect("matches reference" in spoken, "the {} chip reads {!r}, without 'matches reference'".format(label, spoken))
            elif agree == "false":
                expect("differs from reference" in spoken, "the {} chip reads {!r}, without 'differs from reference'".format(label, spoken))
            expect("✓" not in spoken and "✗" not in spoken, "the {} chip reads its mark glyph aloud: {!r}".format(label, spoken))
            chips.append(spoken)
        expect(len(chips) == 14, "{} chips, not 14".format(len(chips)))
        expect(sum(": positive" in c for c in chips) == len(SYNTHETIC_POSITIVE) and sum(": negative" in c for c in chips) == 14 - len(SYNTHETIC_POSITIVE),
               "the chips say {} positive and {} negative".format(sum(": positive" in c for c in chips), sum(": negative" in c for c in chips)))
        lists = [n for n in ax3.by_id.values() if Ax.role(n) == "list" and Ax.name(n).startswith("CheXbert-14")]
        expect(len(lists) == 1, "the chip list has no accessible name that starts with 'CheXbert-14'")
        b.evaluate("document.querySelector('#conversation .labels').scrollIntoView({ block: 'center' })")
        ctx.shot("labelled_chips", "labelled_chips", "the 14 label chips of a SYNTHETIC labelled turn, filled for positive, with agreement marks", DESKTOP)
    finally:
        if seeded is not None:
            seeded.stop()
        shutil.rmtree(home, ignore_errors=True)
    return ("stage buttons: their text, then state, with expanded, no glyph first; status region live=polite reads 'Report ready'; 14 chips read "
            "'<name>: positive|negative'; every control named; no text under AA contrast in 4 states",
            {"contrast_issues_under_aa": faint, "focusable_controls_named": len(exposed), "stage_buttons": buttons, "skipped_stages": plain,
             "stage_items_as_read": items,
             "status_region": {"live": "polite", "text": "Report ready", "history": said}, "chips_on_seeded_synthetic_turn": chips})


def check_error(ctx: Context) -> Tuple[str, Dict[str, Any]]:
    b = ctx.browser
    b.call("Network.enable")
    ctx.open("#/new")
    ctx.attach("a")
    b.click("#prompt")
    b.insert_text("tokens 500")   # the drawer refuses a budget over 200 (with a line under the field), so the server's bounds are reached through the note's command
    b.events("Network.responseReceived")
    b.click("#send")
    b.wait_for("!document.querySelector('#notice').hidden", 15, "the notice")
    b.evaluate("0")
    statuses = [(p["response"]["url"].split("?")[0].rsplit("/", 3)[-3:], p["response"]["status"]) for p in b.events("Network.responseReceived")
                if p["response"]["url"].endswith("/messages")]
    expect(statuses and statuses[-1][1] == 422, "the turn request was answered {}, not 422".format(statuses))
    notice = b.evaluate("({ text: document.querySelector('#notice p').textContent, role: document.querySelector('#notice').getAttribute('role'),"
                        " retry: !document.querySelector('#notice button:not([aria-label])').hidden, prompt: document.querySelector('#prompt').value,"
                        " preview: !document.querySelector('#preview').hidden, file: document.querySelector('#preview span').textContent,"
                        " turns: document.querySelectorAll('#conversation > *').length, send: !document.querySelector('#send').disabled,"
                        " stop: document.querySelector('#stop').hidden, busy: document.querySelector('#conversation').hasAttribute('aria-busy') })")
    expect("max_new_tokens" in notice["text"], "the notice says {!r}, not the server's reason".format(notice["text"]))
    expect(notice["role"] == "alert", "the notice has role {!r}".format(notice["role"]))
    expect(notice["prompt"] == "tokens 500" and notice["preview"] and "xray_a.png" in notice["file"], "the composer was not kept: {}".format(notice))
    expect(notice["turns"] == 0, "the refused turn is still on the page ({} nodes)".format(notice["turns"]))
    expect(notice["send"] and notice["stop"], "the composer is locked after the refusal")
    ctx.shot("error_notice", "error_notice", "the dismissible notice above the composer, with the refused note and the image still in it", DESKTOP)
    b.click('#notice button[aria-label="Dismiss"]')
    dismissed = b.evaluate("({ hidden: document.querySelector('#notice').hidden, prompt: document.querySelector('#prompt').value,"
                           " preview: !document.querySelector('#preview').hidden, focus: document.activeElement.id })")
    expect(dismissed["hidden"] and dismissed["prompt"] == "tokens 500" and dismissed["preview"], "Dismiss changed the composer: {}".format(dismissed))
    expect(dismissed["focus"] == "prompt", "focus is on {!r} after Dismiss, not the note field".format(dismissed["focus"]))
    # and the same composer sends once the note is fixed
    b.evaluate("(() => { const p = document.querySelector('#prompt'); p.value = ''; p.dispatchEvent(new Event('input', { bubbles: true })); })()")
    b.click("#send")
    card = b.wait_for("(() => { const c = (%s)(); return c && c.status !== 'running' ? c : null; })()" % CARD_JS, 30, "the retried turn")
    expect(card["status"] == "done", "the retried turn ended {}".format(card["status"]))
    return ("a server {} showed the alert '{}', kept the note and the image, dismissed to the note field; the same composer then sent".format(
        statuses[-1][1], one_line(notice["text"], 60)),
            {"http_status": statuses[-1][1], "notice": notice["text"], "composer_kept": True, "dismissed": dismissed})


# ---- settings (P4-F, P4-G) ------------------------------------------------------------------------------------------------------

BUDGET_FIELD = '#drawer input[data-setting="max_new_tokens"]'
STOP_SWITCH = '#drawer input[data-setting="stop_on_repeat"]'
SAVE_BUTTON = "#drawer-save"
APPLY_NOTE = "Changes apply from your next Send."
RUNNING_NOTE = "The running turn keeps the settings it started with."
RETRIEVAL_NOTE = "This server has no retrieval gallery, so similar X-rays and matching reports are skipped."
LABELS_NOTE = "This server has no CheXbert labeller, so labels are skipped."
STORAGE_TEXT = "Browser storage is unavailable, so these settings last until the page closes."
SAVED_TEXT = "Settings saved. They apply from your next Send."
BUDGET_ERROR = "Enter a whole number from 16 to 200."
REPEAT_NOTE = "Stopped when the model began repeating itself."
STOP_LABEL = "Stop when the report starts repeating"
STOP_HINT = "Off: the published protocol, which always decodes the whole token budget."
FIELD_LABELS = ["Beam size (1–8)", "Token budget (16–200)", "Similar images (0–12)", "Matching reports (0–10)"]
# What is typed, with real keys: the first Send's budget is reached by 120 and Enter, then a refused 300 and 8, then 150 and the Save button.
# 200 is the longest budget the server allows, and the re-run's.
ENTER_BUDGET, FIRST_BUDGET, SECOND_BUDGET = 120, 150, 200
SAVED_SECONDS = (3.0, 6.5)   # "Settings saved." stays for about 4 s, by the clock of the page
RERUN_HINT = "No new image: Send re-runs xray_a.png with these settings."


def budget_note(tokens: int) -> str:
    """The card's note for a turn that ran to its token budget with the display repair on and a sentence cut off."""
    return "Reached the {}-token budget; the unfinished last sentence is hidden (Show raw shows it).".format(tokens)


def repeated(text: str) -> int:
    """How many sentences of the text repeat an earlier one (the repair's key: lower case, whitespace normalised)."""
    keys = [s.lower() for s in split_sentences(text)]
    return len(keys) - len(set(keys))


def budget_chip(chips: List[str]) -> Optional[str]:
    return next((c for c in chips if c.endswith(" tok")), None)


def errors_shown(states: List[Dict[str, Any]]) -> List[bool]:
    """Whether the line under the field was showing after each key."""
    return [not (state["error"] or {}).get("hidden", True) for state in states]


def click_and_type(b: Browser, selector: str, digits: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """What a user does: a real click into a number field of the drawer, then the digits one key at a time over whatever it holds. The
    field is given up first (a click on the drawer's title), as it is when a user comes back to it: a click in a field that has focus puts
    the caret where it lands, like any other, and only the focus a click gives selects the whole value.
    -> the field as the click left it (its text, whether it has focus, what is selected), and the field after each digit: its text,
    focus, the composer chips, the stored setting, and whether the line under it is showing."""
    b.click("#drawer h2")
    b.click(selector)
    clicked = b.run(FIELD_JS, selector)
    states = []  # type: List[Dict[str, Any]]
    for ch in digits:
        b.type_text(ch)
        states.append(b.run(FIELD_JS, selector))
    return clicked, states


def type_budget(b: Browser, digits: str, replaces: str, chips: List[str], stored: List[Optional[int]], errors: List[bool]) -> Dict[str, Any]:
    """Click into the budget field, which holds `replaces`, and type the digits. The click selects the whole value, so the field reads the
    digits and not the digits after the old value (100 and 150 typed as one text is 100150, which is out of range); and after every key
    the field says what was typed and keeps focus, the chips and the stored setting move only for a whole number inside the bounds, and
    the line under the field shows while the text is not one. Nothing is clamped: a text that is not right changes nothing."""
    clicked, states = click_and_type(b, BUDGET_FIELD, digits)
    expect(clicked["focused"] and clicked["value"] == replaces and clicked["selected"] == replaces,
           "the click left the field {} (it held {!r}): its whole value should be selected, so that typing replaces it".format(clicked, replaces))
    typed = [digits[:i + 1] for i in range(len(digits))]
    expect([t["value"] for t in states] == typed,
           "the field read {} while {} was typed over {!r}: a value that is appended reads {!r}".format([t["value"] for t in states], digits, replaces, replaces + digits))
    expect(all(t["focused"] for t in states), "the field lost focus while it was typed in: {}".format([t["focused"] for t in states]))
    expect([budget_chip(t["chips"]) for t in states] == chips,
           "the chips read {}, not {}, as {} was typed".format([budget_chip(t["chips"]) for t in states], chips, typed))
    expect([t["stored"] for t in states] == stored,
           "the stored setting went {}, not {}, as {} was typed".format([t["stored"] for t in states], stored, typed))
    expect(errors_shown(states) == errors, "the line under the field showed {}, not {}, as {} was typed".format(errors_shown(states), errors, typed))
    expect(all((t["invalid"] == "true") == e and (t["describedby"] == "max_new_tokens-error") == e for t, e in zip(states, errors)),
           "aria-invalid / aria-describedby do not follow the line: {}".format([(t["invalid"], t["describedby"]) for t in states]))
    return {"replaces": replaces, "selected_by_the_click": clicked["selected"], "states": states}


def press_enter(b: Browser) -> None:
    b.key("Enter", "Enter", 13, text="\r")


def finished_card(b: Browser, cards: int, timeout: float = 60.0) -> Dict[str, Any]:
    """The newest card once it is the `cards`-th and its turn is over."""
    return b.wait_for("(() => { const c = (%s)(); return c && c.cards === %d && c.status !== 'running' ? c : null; })()" % (CARD_JS, cards),
                      timeout, "turn {} to end".format(cards))


def forget_settings(b: Browser) -> None:
    try:
        b.evaluate("localStorage.removeItem('cxrchat.settings')")
    except RuntimeError:   # no page of the app is open (yet): nothing of it is stored
        pass


def check_drawer_before_anything_changes(drawer: Dict[str, Any], features: Dict[str, bool]) -> None:
    """The drawer as a fresh page shows it: when changes apply, the page's defaults, the range in every number field's label, the stop switch
    on with its hint, a form that ends in Save, and a stage the server does not run switched off with its reason (one it does run is left
    alone), as /v1/models says."""
    retrieval, labelling = features["retrieval"], features["labels"]
    expect(drawer["second"] == "apply-note" and drawer["apply"] == {"hidden": False, "text": APPLY_NOTE},
           "the first line under the drawer's title is {} / {}, not {!r}".format(drawer["second"], drawer["apply"], APPLY_NOTE))
    expect(drawer["running"] == {"hidden": True, "text": RUNNING_NOTE}, "the running-turn line is {} before any turn".format(drawer["running"]))
    expect(drawer["repair"]["checked"] and not drawer["repair"]["disabled"], "Display repair is not on by default: {}".format(drawer["repair"]))
    expect(not {"raw text", "repair on", "full budget"} & set(drawer["chips"]),
           "the chips say something about Display repair or the stop switch while they are on: {}".format(drawer["chips"]))
    expect(drawer["ranges"] == FIELD_LABELS, "the number fields are labelled {}, not {}".format(drawer["ranges"], FIELD_LABELS))
    for name, bounds in (("k_images", (0, 12)), ("k_reports", (0, 10))):
        expect((drawer[name]["min"], drawer[name]["max"]) == tuple(str(b) for b in bounds), "{} takes {}".format(name, (drawer[name]["min"], drawer[name]["max"])))
    stop = drawer["stop"]
    expect(stop is not None and stop["checked"] and not stop["disabled"] and stop["describedby"] == "stop-hint",
           "the stop switch is not on, enabled and described by its hint: {}".format(stop))
    expect(drawer["stop_hint"] == {"hidden": False, "text": STOP_HINT}, "the stop switch's hint is {}".format(drawer["stop_hint"]))
    expect(drawer["form"], "the drawer's controls are not in a form with novalidate")
    expect(drawer["save"] == {"text": "Save", "type": "submit", "last": True, "closeFirst": True},
           "the drawer's Save button is {}: it should be a submit button that ends the form, with the close button first".format(drawer["save"]))
    expect(drawer["saved"] == "", "the page says {!r} before anything was saved".format(drawer["saved"]))
    for name in ("k_images", "k_reports"):
        expect(drawer[name]["disabled"] == (not retrieval) and drawer[name]["describedby"] == (None if retrieval else "retrieval-note"),
               "{} is {} on a server whose retrieval is {}".format(name, drawer[name], retrieval))
    expect(drawer["retrieval"] == {"hidden": retrieval, "text": RETRIEVAL_NOTE},
           "the retrieval note is {} on a server whose retrieval is {}".format(drawer["retrieval"], retrieval))
    label = drawer["label"]
    expect(label["disabled"] == (not labelling) and label["describedby"] == (None if labelling else "labels-note") and (labelling or not label["checked"]),
           "CheXbert labels is {} on a server whose labelling is {}".format(label, labelling))
    expect(drawer["labels"] == {"hidden": labelling, "text": LABELS_NOTE},
           "the labels note is {} on a server whose labelling is {}".format(drawer["labels"], labelling))
    expect(bool([c for c in drawer["chips"] if c.startswith("k ")]) == retrieval,
           "the composer chips {} on a server whose retrieval is {}: k belongs to them only when the stage runs".format(drawer["chips"], retrieval))
    expect(drawer["rerun"]["hidden"], "the re-run hint is shown in an empty chat: {}".format(drawer["rerun"]))


def check_the_enter_that_saves(b: Browser) -> Dict[str, Any]:
    """Enter has just been pressed in the budget field, holding a whole number inside the bounds: the form submitted, which is Save. The
    drawer is closed and focus is back on Settings, the setting is stored, and "Settings saved." is on the page for about 4 seconds."""
    appeared = b.wait_for("document.querySelector('#saved').textContent !== '' ? 1 : 0", 5, "the confirmation after Enter") and time.time()
    after = b.run(FIELD_JS, BUDGET_FIELD)
    expect(not after["drawerOpen"] and after["active"] == "settings", "after Enter the drawer is {} and focus is on {!r}".format(
        "open" if after["drawerOpen"] else "closed", after["active"]))
    expect(after["saved"] == SAVED_TEXT, "the page says {!r}, not {!r}".format(after["saved"], SAVED_TEXT))
    expect(after["stored"] == ENTER_BUDGET and budget_chip(after["chips"]) == "{} tok".format(ENTER_BUDGET),
           "after Enter the stored budget is {} and the chips {}".format(after["stored"], after["chips"]))
    region = b.evaluate("(() => { const r = document.querySelector('#saved'), cs = getComputedStyle(r), box = r.getBoundingClientRect();"
                        " return { role: r.getAttribute('role'), live: r.getAttribute('aria-live'), visible: box.width > 1 && box.height > 1 && cs.visibility !== 'hidden',"
                        " near: !!r.previousElementSibling && r.previousElementSibling.id === 'chips' }; })()")
    expect(region == {"role": "status", "live": "polite", "visible": True, "near": True},
           "the confirmation is {}: it should be a polite status region, in sight, right after the chips".format(region))
    b.wait_for("document.querySelector('#saved').textContent === ''", 8, "the confirmation to go")
    lasted = time.time() - appeared
    expect(SAVED_SECONDS[0] <= lasted <= SAVED_SECONDS[1], "the confirmation stayed {:.1f} s, not about 4".format(lasted))
    return {"drawer_open": False, "focus": after["active"], "message": after["saved"], "stored": after["stored"], "chips": after["chips"],
            "region": region, "seconds_on_page": round(lasted, 1)}


def check_a_refused_enter(b: Browser, before: Dict[str, Any], typed: str) -> Dict[str, Any]:
    """Enter pressed in the budget field holding a text that is not a whole number inside the bounds: the drawer stays open on that field,
    the line under it says what it takes, the text is as typed, and nothing changed (not the chips, not the stored setting, no
    confirmation): Enter did not snap it to 200 or to 16."""
    after = b.run(FIELD_JS, BUDGET_FIELD)
    expect(after["drawerOpen"] and after["focused"], "after Enter on {!r} the drawer is {} and focus is on {!r}".format(
        typed, "open" if after["drawerOpen"] else "closed", after["active"]))
    expect(after["value"] == typed, "Enter changed the field from {!r} to {!r}".format(typed, after["value"]))
    expect(after["error"] == {"hidden": False, "text": BUDGET_ERROR} and after["invalid"] == "true",
           "the line under the field is {} (aria-invalid {!r}), not {!r}".format(after["error"], after["invalid"], BUDGET_ERROR))
    expect(after["chips"] == before["chips"] and after["stored"] == before["stored"],
           "Enter changed the chips {} -> {} or the stored setting {} -> {}".format(before["chips"], after["chips"], before["stored"], after["stored"]))
    expect(after["saved"] == "", "the page says {!r} although the form was refused".format(after["saved"]))
    return {"typed": typed, "drawer_open": True, "field_kept_as_typed": True, "error": after["error"]["text"], "chips": after["chips"],
            "stored": after["stored"], "saved_message": after["saved"]}


def check_repeat_stop(log: Dict[str, Any], page: Dict[str, Any], budget: int) -> Dict[str, Any]:
    """A turn that ran with the page's default (the stop switch on): it ended by itself, on a repeat, before its token budget; the server
    stored a snapshot for every step it took; the card has the quiet note and no budget note, and its footer says the tokens decoded."""
    options, generate = log["options"], log["generate"]
    expect(options["stop_on_repeat"] is True and options["max_new_tokens"] == budget, "the server ran with {}".format(options))
    expect(generate["stopped"] == "repeat" and 0 < generate["tokens"] < budget,
           "the turn stopped {!r} after {} tokens of {}: it should have stopped on a repeat before its budget".format(generate["stopped"], generate["tokens"], budget))
    expect(len(log["snapshots"]) == generate["tokens"], "{} snapshots for {} tokens".format(len(log["snapshots"]), generate["tokens"]))
    expect(log["truncated"] is False and not page["truncated"], "the turn that stopped on a repeat is flagged as cut off by the budget")
    expect(page["notes"] == [REPEAT_NOTE] and page["repeatNote"], "the card's notes are {}, not [{!r}]".format(page["notes"], REPEAT_NOTE))
    expect("{} tok".format(generate["tokens"]) in page["footer"].split(" · "),
           "the card's footer is {!r}, without {} tok (what was decoded, which is fewer than the budget {})".format(page["footer"], generate["tokens"], budget))
    expect(repeated(page["shown"]) == 0 and repeated(log["report"]) >= 1,
           "the card repeats {} sentence(s) and the decoder's text {}".format(repeated(page["shown"]), repeated(log["report"])))
    expect(log["status"] == "done", "the stopped turn is stored as {}".format(log["status"]))
    return {"budget": budget, "tokens": generate["tokens"], "stopped": generate["stopped"], "snapshots": len(log["snapshots"]),
            "card_notes": page["notes"], "footer": page["footer"], "truncated_mid_sentence": log["truncated"],
            "raw_report_repeats": repeated(log["report"]), "card_repeats": repeated(page["shown"])}


def check_clean_display(b: Browser, log: Dict[str, Any], rec: Dict[str, Any]) -> Dict[str, Any]:
    """The 200-token re-run, run with the page's default for the display (Display repair on) and the stop switch off: it ran to its budget
    (stopped "budget", all 200 tokens), and no sentence repeats on the card, in any snapshot the server stored or in any frame of the card
    that was seen while it streamed; the card says what its budget hid, and Show raw is the decoder's whole text. -> what was measured."""
    page = b.evaluate("(%s)()" % SHOWN_JS)
    shown, report = page["shown"], log["report"]
    expect(log["generate"]["stopped"] == "budget" and log["generate"]["tokens"] == SECOND_BUDGET,
           "with the switch off the turn stopped {!r} after {} tokens, not at its budget of {}".format(log["generate"]["stopped"], log["generate"]["tokens"], SECOND_BUDGET))
    expect(page["notes"] == ([budget_note(SECOND_BUDGET)] if log["truncated"] else []) and not page["repeatNote"],
           "the card's notes are {} for a turn that reached its budget with truncated_mid_sentence {}".format(page["notes"], log["truncated"]))
    expect("{} tok".format(SECOND_BUDGET) in page["footer"].split(" · "), "the card's footer is {!r}, without {} tok".format(page["footer"], SECOND_BUDGET))
    expect(page["pressed"] == "false" and page["raw"] is None, "the card starts on the raw text: {}".format(page["pressed"]))
    expect(" ".join(shown.split()) == " ".join(log["display"].split()), "the card shows {!r}, not the display report {!r}".format(shown, log["display"]))
    expect(repeated(shown) == 0, "the settled card repeats {} sentence(s): {!r}".format(repeated(shown), one_line(shown, 300)))
    expect(repeated(report) >= 5, "the decoder's own report repeats only {} sentence(s): the case is not the user's".format(repeated(report)))
    stored_bad = [t for t in log["snapshots"] if repeated(t)]
    expect(len(log["snapshots"]) == SECOND_BUDGET and not stored_bad,
           "{} stored snapshots, {} of them repeat a sentence".format(len(log["snapshots"]), len(stored_bad)))
    frames_bad = [t for t in rec["shown"] if repeated(t)]
    expect(len(rec["shown"]) >= 10 and not frames_bad, "{} frames of the stream were seen; {} repeat a sentence".format(len(rec["shown"]), len(frames_bad)))
    expect(len(shown.split()) * 2 < len(report.split()),
           "the clean report ({} words) is not much shorter than the raw one ({})".format(len(shown.split()), len(report.split())))
    b.click('#conversation > article.card:last-child [data-action="raw"]')
    raw = b.wait_for("(() => { const r = (%s)(); return r && r.raw !== null ? r : null; })()" % SHOWN_JS, 5, "Show raw to show the raw text")
    expect(raw["pressed"] == "true" and raw["raw"] == report, "Show raw shows {!r}, not the decoder's report".format(one_line(raw["raw"], 200)))
    expect(repeated(raw["raw"]) >= 5, "Show raw has only {} repeated sentence(s)".format(repeated(raw["raw"])))
    return {"words": {"card": len(shown.split()), "raw_report": len(report.split())},
            "sentences": {"card": len(split_sentences(shown)), "raw_report": len(split_sentences(report))},
            "repeated_sentences": {"card": repeated(shown), "raw_report": repeated(report), "worst_stream_frame": max(repeated(t) for t in rec["shown"]),
                                   "worst_stored_snapshot": max(repeated(t) for t in log["snapshots"])},
            "stream_frames_seen": len(rec["shown"]), "stored_snapshots": len(log["snapshots"]), "stopped": log["generate"]["stopped"],
            "tokens": log["generate"]["tokens"], "card_notes": page["notes"],
            "truncated_mid_sentence": log["truncated"], "show_raw_equals_decoder_report": True}


def check_settings(ctx: Context) -> Tuple[str, Dict[str, Any]]:
    b = ctx.browser
    ctx.open("#/new")
    forget_settings(b)   # the page's defaults, whatever an earlier check left in this browser
    try:
        return settings_walk(ctx)
    finally:
        forget_settings(b)


def settings_walk(ctx: Context) -> Tuple[str, Dict[str, Any]]:
    b = ctx.browser
    ctx.open("#/new")   # a fresh load, which reads the defaults
    b.click("#settings")
    b.wait_for("!document.querySelector('#drawer').hidden", what="the drawer")
    features = b.evaluate("(%s)()" % MODELS_JS)["features"]
    expect(set(features) == {"retrieval", "labels"} and all(isinstance(v, bool) for v in features.values()), "/v1/models says features {}".format(features))
    drawer = b.evaluate("(%s)()" % DRAWER_JS)
    check_drawer_before_anything_changes(drawer, features)
    ctx.attach("a")
    # -- the user's own flow, with real keys. A click into the budget selects "100", so 1, 2, 0 typed over it read 120 (the old page left the
    # caret after the 100, and 150 typed there was 100150, which Enter silently made 200). 1 and 12 are below the bound of 16: not applied,
    # and the line under the field says so. Enter in the field is Save.
    entered = type_budget(b, str(ENTER_BUDGET), "100", ["100 tok", "100 tok", "{} tok".format(ENTER_BUDGET)], [None, None, ENTER_BUDGET], [True, True, False])
    press_enter(b)
    saved_by_enter = check_the_enter_that_saves(b)
    # -- a number that is too big, typed key by key over the 120 that the field holds. The 30 on the way is a whole number inside the bounds
    # and applies like any other (the chips follow the keys); 300 does not, and changes nothing. Enter then saves nothing and says why.
    b.click("#settings")
    b.wait_for("!document.querySelector('#drawer').hidden", what="the drawer, opened again")
    refused_300 = type_budget(b, "300", str(ENTER_BUDGET), ["{} tok".format(ENTER_BUDGET), "30 tok", "30 tok"], [ENTER_BUDGET, 30, 30], [True, False, True])
    before = b.run(FIELD_JS, BUDGET_FIELD)
    press_enter(b)
    enter_on_300 = check_a_refused_enter(b, before, "300")
    ctx.shot("drawer_error", "drawer_error", "the drawer with 300 typed in the token budget: the line under the field, the red border, the chips unchanged, Save in sight", DESKTOP)
    # -- 8 has no whole number on the way to it: the setting the field had stays exactly as it was
    refused_8 = type_budget(b, "8", "300", ["30 tok"], [30], [True])
    before = b.run(FIELD_JS, BUDGET_FIELD)
    press_enter(b)
    enter_on_8 = check_a_refused_enter(b, before, "8")
    # -- put right, and saved with the Save button this time
    fixed = type_budget(b, str(FIRST_BUDGET), "8", ["30 tok", "30 tok", "{} tok".format(FIRST_BUDGET)], [30, 30, FIRST_BUDGET], [True, True, False])
    b.click(SAVE_BUTTON)
    b.wait_for("document.querySelector('#drawer').hidden", 5, "the drawer to close on Save")
    by_button = b.run(FIELD_JS, BUDGET_FIELD)
    expect(by_button["saved"] == SAVED_TEXT and by_button["active"] == "settings" and by_button["stored"] == FIRST_BUDGET and not by_button["drawerOpen"],
           "after the Save button: {}".format({k: by_button[k] for k in ("saved", "active", "stored", "drawerOpen")}))
    # -- Send: the page's default, the stop switch on. The drawer says that the running turn keeps its settings; the turn used the typed
    # budget (the chips say what was asked) and ended by itself, on a repeat, before it (the footer says what was decoded)
    ctx.send()
    b.wait_for("!document.querySelector('#running-note').hidden", 10, "the drawer to say that a turn is running")
    first = finished_card(b, 1)
    expect(first["status"] == "done", "the first turn ended {}".format(first["status"]))
    one = b.evaluate("(%s)()" % TURN_JS)
    expect("{} tok".format(FIRST_BUDGET) in one["chips"] and "full budget" not in one["chips"],
           "the new turn's chips are {}: they should say {} tok, and not full budget".format(one["chips"], FIRST_BUDGET))
    stopped = check_repeat_stop(b.evaluate("(%s)(%s)" % (LOG_JS, json.dumps(one["id"]))), b.evaluate("(%s)()" % SHOWN_JS), FIRST_BUDGET)
    after = b.evaluate("(%s)()" % DRAWER_JS)
    expect(after["running"]["hidden"], "the drawer still says a turn is running after it ended")
    expect(after["rerun"] == {"hidden": False, "text": RERUN_HINT}, "after the turn the re-run hint is {}, not {!r}".format(after["rerun"], RERUN_HINT))
    # -- the hint belongs to Send with no image: an attached one hides it, removing that image brings it back
    ctx.attach("b")
    expect(b.evaluate("(%s)()" % DRAWER_JS)["rerun"]["hidden"], "the re-run hint stays while an image is attached")
    b.click("#preview button")
    b.wait_for("document.querySelector('#preview').hidden", what="the attached image to go")
    expect(b.evaluate("(%s)()" % DRAWER_JS)["rerun"] == {"hidden": False, "text": RERUN_HINT},
           "the re-run hint did not come back with the image removed")
    # -- no new image, the longest budget, and the stop switch off (a click on the switch): in force at once, and used by the re-run
    b.click("#settings")
    b.wait_for("!document.querySelector('#drawer').hidden", what="the drawer, opened for the second budget")
    retyped = type_budget(b, str(SECOND_BUDGET), str(FIRST_BUDGET), ["{} tok".format(FIRST_BUDGET), "20 tok", "{} tok".format(SECOND_BUDGET)],
                          [FIRST_BUDGET, 20, SECOND_BUDGET], [True, False, False])
    b.click(STOP_SWITCH)
    b.wait_for("[...document.querySelectorAll('#chips span')].some((c) => c.textContent === 'full budget')", 5, "the chips to say full budget")
    switched = b.evaluate("(%s)()" % DRAWER_JS)
    expect(not switched["stop"]["checked"] and "full budget" in switched["chips"], "after the click the stop switch is {} and the chips {}".format(switched["stop"], switched["chips"]))
    ctx.shot("settings", "settings", "the drawer with its ranges, notes and Save, the re-run hint under the image well, and the chips at the typed budget", DESKTOP)
    b.click(SAVE_BUTTON)
    b.wait_for("document.querySelector('#drawer').hidden", 5, "the drawer to close on Save")
    b.run(RECORD_JS, None)
    ctx.send()
    b.wait_for("!document.querySelector('#running-note').hidden", 10, "the drawer to say that the second turn is running")
    second = finished_card(b, 2)
    expect(second["status"] == "done", "the re-run ended {}".format(second["status"]))
    two = b.evaluate("(%s)()" % TURN_JS)
    expect(two["users"] == 2 and two["cards"] == 2 and not two["image"], "the re-run is not a turn without an image of its own: {}".format(two))
    expect("{} tok".format(SECOND_BUDGET) in two["chips"] and "full budget" in two["chips"],
           "the re-run's chips {} do not say {} tok and full budget".format(two["chips"], SECOND_BUDGET))
    log = b.evaluate("(%s)(%s)" % (LOG_JS, json.dumps(two["id"])))
    expect(log["image"]["source"] == "previous" and log["image"]["filename"] == "xray_a.png", "the re-run used {}, not the last X-ray".format(log["image"]))
    expect(log["options"]["max_new_tokens"] == SECOND_BUDGET and log["options"]["display_repair"] is True and log["options"]["stop_on_repeat"] is False,
           "the server ran with {}".format(log["options"]))
    skips = {"score": "no_reference"}
    skips.update({} if features["retrieval"] else {"retrieve": "gallery_unavailable"})
    skips.update({} if features["labels"] else {"label": "labeler_unavailable"})
    expect(log["skipped"] == skips, "the stages the server skipped are {}, not {}: the drawer and the pipeline disagree".format(log["skipped"], skips))
    clean = check_clean_display(b, log, b.evaluate("(%s)()" % RECORDED_JS))
    return ("a click into the budget selected it and 1-2-0 typed over it read 120 (not 100120); Enter saved it, closed the drawer to Settings and the page said "
            "'Settings saved.' for {:.1f} s; 300 and 8 were refused with the line under the field (300 on its way applied 30, 8 changed nothing; Enter "
            "changed nothing); the Save button saved 150; a default turn stopped on a repeat after {} of its {} tokens with the quiet note and no budget "
            "note; with the switch off, at 200 tokens the card has {} words and {} repeated sentences (raw: {} words, {} repeated), none in {} frames of "
            "the stream; Show raw is the decoder's text".format(
                saved_by_enter["seconds_on_page"], stopped["tokens"], FIRST_BUDGET, clean["words"]["card"], clean["repeated_sentences"]["card"],
                clean["words"]["raw_report"], clean["repeated_sentences"]["raw_report"], clean["stream_frames_seen"]),
            {"drawer_before": {"server_features": features, "apply_note": drawer["apply"]["text"], "retrieval_note": drawer["retrieval"],
                               "labels_note": drawer["labels"], "k_fields_disabled": drawer["k_images"]["disabled"] and drawer["k_reports"]["disabled"],
                               "labels_disabled": drawer["label"]["disabled"], "chips": drawer["chips"],
                               "display_repair_checked": drawer["repair"]["checked"], "number_field_labels": drawer["ranges"],
                               "stop_switch": {"checked": drawer["stop"]["checked"], "hint": drawer["stop_hint"]["text"]}, "save_button": drawer["save"]},
             "enter_saves": {"typed": entered, "after_enter": saved_by_enter},
             "refused_300": {"typed": refused_300, "after_enter": enter_on_300},
             "refused_8": {"typed": refused_8, "after_enter": enter_on_8},
             "save_button": {"typed": fixed, "after_click": {k: by_button[k] for k in ("saved", "active", "stored", "drawerOpen")}},
             "first_send_stop_switch_on": {"chips_of_turn": one["chips"], "rerun_hint": after["rerun"]["text"], "repeat_stop": stopped},
             "rerun_stop_switch_off": {"typed": retyped, "chips_of_turn": two["chips"], "footer": two["provenance"], "image": log["image"],
                                       "options": log["options"], "skipped_stages": log["skipped"]},
             "clean_display_at_200_tokens": clean})


CHECKS = [("stream", check_stream), ("reload", check_reload), ("sessions", check_sessions), ("exports", check_exports),
          ("stop", check_stop), ("keyboard", check_keyboard), ("narrow", check_narrow), ("a11y", check_a11y), ("error", check_error),
          ("settings", check_settings)]


def run_check(ctx: Context, name: str, fn: Callable[[Context], Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
    mark = len(ctx.console)
    started = time.time()
    status, detail, data = "PASS", "", {}  # type: str, str, Dict[str, Any]
    try:
        ctx.stop_default()   # each check starts from the page's defaults; one that needs a long turn asks for the switch off itself
        detail, data = fn(ctx)
    except Failure as exc:
        status, detail = "FAIL", str(exc)
    except Exception as exc:   # a CDP error, a timeout waiting for the page: a failed check, not a crashed run
        status, detail = "FAIL", "{}: {}".format(type(exc).__name__, exc)
    try:
        ctx.drain_console()
    except Exception as exc:
        ctx.console.append("could not read the console: {}".format(exc))
    logged = ctx.console[mark:]
    if status == "PASS" and logged:
        status, detail = "FAIL", "the page logged {} error(s): {}".format(len(logged), logged[0])
    print("CHECK {} {} {}".format(name, status, one_line(detail)), flush=True)
    return {"name": name, "status": status, "detail": one_line(detail, 400), "seconds": round(time.time() - started, 1), "data": data,
            "console_errors": logged}


def warm_up(base: str) -> float:
    """One throwaway turn over HTTP, so that the first turn in the browser does not wait for the engine's first-use imports (which
    would look like 3 s of silence to the page). Its chat is deleted. -> seconds."""
    started = time.time()
    boundary = uuid.uuid4().hex

    def part(name: str, value: str) -> bytes:
        return ("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n" % (boundary, name, value)).encode()

    session = json.load(urllib.request.urlopen(urllib.request.Request(
        base + "v1/sessions", data=b"{}", headers={"Content-Type": "application/json"}, method="POST"), timeout=30))["id"]
    body = io.BytesIO()
    body.write(part("text", "") + part("options", json.dumps({"max_new_tokens": 16})))
    body.write(("--%s\r\nContent-Disposition: form-data; name=\"image\"; filename=\"warm.png\"\r\nContent-Type: image/png\r\n\r\n" % boundary).encode())
    body.write(png_bytes(128, 128) + b"\r\n--%s--\r\n" % boundary.encode())
    request = urllib.request.Request(base + "v1/sessions/%s/messages" % session, data=body.getvalue(), method="POST",
                                     headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
    with urllib.request.urlopen(request, timeout=120) as response:
        response.read()
    urllib.request.urlopen(urllib.request.Request(base + "v1/sessions/" + session, method="DELETE"), timeout=30).read()
    return time.time() - started


def run_all(browser: Browser, app: Optional[App], base: str, work: str, out_dir: Optional[str]) -> Tuple[List[Dict[str, Any]], Context]:
    """Every check in order, on this browser and this app; -> (one result per check, the context with the screenshots)."""
    browser.call("Runtime.enable")
    browser.call("Emulation.setFocusEmulationEnabled", enabled=True)   # the page behaves as the focused one, so key events move focus
    ctx = Context(browser, app, base, work, out_dir)
    if out_dir is not None:   # a picture of an earlier run must not pass for this run's
        for name in EVIDENCE_FILES:
            if os.path.exists(os.path.join(out_dir, name)):
                os.remove(os.path.join(out_dir, name))
    return [run_check(ctx, name, fn) for name, fn in CHECKS], ctx


def environment(browser: Browser, app: Optional[App], warm: Optional[float]) -> Dict[str, Any]:
    version = browser.call("Browser.getVersion")["product"]
    return {"date": datetime.date.today().isoformat(), "engine": "tiny (random weights, toy vocabulary; no checkpoint, no data)",
            "tiny_step_delay_s": STEP_DELAY_S if app is not None else None, "image": "synthetic PNG (tests/app_helpers.png_bytes)",
            "browser": version + " (headless)", "python": platform.python_version(), "os": platform.system(),
            "warm_up_turn_s": None if warm is None else round(warm, 2), "viewports": {"desktop": list(DESKTOP), "phone": list(PHONE)},
            "notes": ["The tiny pipeline skips retrieve, label and score until P5-E, so the a11y check reads its label chips from a second tiny app "
                      "whose home was seeded with one SYNTHETIC labelled turn (labelled_chips_*.png); the other checks run the live pipeline.",
                      "The settings drawer refuses a token budget outside the server's 16-200 with a line under the field, so the 422 of the error check "
                      "is reached through the note's command ('tokens 500'), which the server applies on top of the drawer's options.",
                      "Display repair and Stop when the report starts repeating are on by default in the page (the server's own defaults stay off). "
                      "The tiny model never writes an end-of-report token and loops after about 20 tokens, so a default turn ends by itself near 21 tokens "
                      "(the settings check measures it); the checks that need a long turn (stream, reload, sessions, stop) store the switch off before the "
                      "page loads, and every check starts from the page's defaults.",
                      "Display repair on means the reports in every picture are the repaired display copy; the settings check runs 200 tokens, with the stop "
                      "switch off, to show that none of its sentences repeats on the card or in any frame of the stream, while Show raw keeps the decoder's "
                      "whole text (the tiny model repeats one sentence about a dozen times).",
                      "The tiny server runs no retrieval gallery and no labeller, so its /v1/models says features retrieval and labels are false: the "
                      "drawer disables k_images, k_reports and CheXbert labels with a note, and the composer chips leave out k (settings_*.png).",
                      "Every screenshot was confirmed by the page itself (a DOM rule per picture, true before and after the capture)."]}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="The browser checklist for the chat page, in headless Chrome.")
    parser.add_argument("--url", help="check this running app instead of starting one (its page URL, e.g. http://127.0.0.1:8000/)")
    parser.add_argument("--out", default=EVIDENCE_DIR, help="where screenshots and checklist.json go (default: docs/chat_ui/evidence/p4e)")
    parser.add_argument("--no-evidence", action="store_true", help="run the checks and write no screenshots and no checklist.json")
    parser.add_argument("--timeout", type=int, default=420, help="seconds the whole run may take (default 420)")
    args = parser.parse_args(argv)
    exit_on_sigterm()   # a SIGTERM runs the finally below, which stops Chrome and the app
    chrome = find_chrome()
    if chrome is None:
        print("ERROR no Chrome found: set $CHROME to its executable")
        return 2

    def overrun(signum: int, frame: Any) -> None:
        raise Overrun("the browser check ran past its {} s limit".format(args.timeout))

    signal.signal(signal.SIGALRM, overrun)
    signal.alarm(args.timeout)
    out_dir = None if args.no_evidence else os.path.abspath(args.out)
    work = tempfile.mkdtemp(prefix="chat_ui_browser_check_")
    app = None  # type: Optional[App]
    browser = None  # type: Optional[Browser]
    results = []  # type: List[Dict[str, Any]]
    ctx = None  # type: Optional[Context]
    warm = None  # type: Optional[float]
    try:
        if args.url:
            base = args.url if args.url.endswith("/") else args.url + "/"
        else:
            app = App(step_delay_s=STEP_DELAY_S)
            base = app.url
        warm = warm_up(base)
        browser = Browser(chrome, label="chat_ui_browser_chrome")
        results, ctx = run_all(browser, app, base, work, out_dir)
        env = environment(browser, app, warm)
    except Overrun as exc:
        print("ERROR {}".format(exc))
        return 2
    except Exception as exc:   # the app or Chrome would not start, a request failed: not a failed check, a run that could not be made
        print("ERROR {}: {}".format(type(exc).__name__, exc))
        return 2
    finally:
        signal.alarm(0)
        if browser is not None:
            browser.close()
        if app is not None:
            app.stop()
        shutil.rmtree(work, ignore_errors=True)
    failures = [r["name"] for r in results if r["status"] != "PASS"]
    result = {"checks": len(results), "failures": failures}
    if out_dir is not None and ctx is not None:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "checklist.json"), "w", encoding="utf-8") as handle:
            json.dump({"result": result, "environment": env, "checks": results, "screenshots": ctx.shots}, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    print("RESULT " + json.dumps(result))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
