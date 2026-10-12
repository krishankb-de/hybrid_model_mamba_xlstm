"""CHAT_UI_PLAN.md P4-A fix round 1, P4-E: a geometry check for the chat page. A local development tool; validate.sh does
not run it.

It starts the app (tiny engine, temporary home, free loopback port), or takes --url, and drives headless Chrome over the
DevTools protocol (scripts/chat_ui_cdp.py) with exact viewports (Emulation.setDeviceMetricsOverride) and an emulated
colour scheme. Each case is one viewport in one scheme with the drawer closed or open (its `hidden` attribute removed in the
page); six chips and a 140-character unbroken word are injected, and Stop is shown and hidden. Checks:

  a  no horizontal overflow on the document, #conversation or #composer (scrollWidth <= clientWidth)
  b  the centre of #settings, #send and (shown) #stop hits the button itself, not an overlay
  c  #composer lies fully inside the viewport
  d  #conversation is at least 120 px tall on a viewport at least 568 px tall
  e  Settings, Send and Stop follow each other in DOM order, row by row and left to right
  f  the focus ring of #sidebar-toggle (shown at 800 px and below) lies inside the viewport
  g  #stop and #drawer are rendered, with a size, when the case shows them, and are not rendered when it hides them; an
     open drawer lies inside the viewport's width
  h  (P4-G, one more case for every viewport in every scheme) the drawer's Save button is rendered when the drawer is open and not
     when it is closed; in the open drawer it lies inside the drawer's width, is at least 40 px tall, is in sight without scrolling
     (it sticks to the bottom of the drawer) and is not covered once it is scrolled to
  i  (P6-B..D, one more case for every viewport in every scheme) the card's own sections (SECTIONS), drawn by render.js in the page from a
     synthetic finished turn with every picture a generated PNG (nothing is fetched): each section is there, lies inside the card's width
     and does not overflow it, every picture has loaded with a size and lies inside the card, every clamped report shows at most three
     lines and offers Show all exactly when it overflows them (render.js settleClamps, measured as the page measures it: P6 fix 1), and
     nothing overflows the page horizontally, with the drawer closed and open. The clamp is proven at both ends with the drawer closed: at
     375 px and narrower a ~170-character report has Show all, and at 1024 px and wider no report that short has one. P9-G5b: the card
     is the tiny engine's, as every card of this app is, so it carries the notice above its report (NOTICE), held to the same: there, inside
     the card's width and not wider inside than it is
  j  (P6 fix 1, one more case for every viewport in every scheme) the test-split picker open, its list full: the page does not overflow
     horizontally, the button and the list are rendered inside the viewport's width, and the transcript's last card header, scrolled
     into view, is not covered (a point just inside its top hits the card, not the composer, the picker or the banner)
  k  (P6-A, COMPOSER_CASES in every scheme, two cases each) the composer's own rows where its column is narrowest: 801 to 834 px wide and
     568 to 650 px tall with the drawer open, and 320 x 568, with six chips and a 103-character file name. One case shows the re-run hint,
     the other the image preview (P6-A's, made by the page from a pasted file), as the page shows one or the other: the row is rendered
     inside the composer, the other is not, nothing overflows horizontally, the report keeps its 120 px, and Send lies inside the
     composer's box and the viewport with nothing scrolled (never below the composer's fold), its centre hitting it. P6 fix 2 adds a third
     case at the drawer-open sizes (COMPOSER_BUSY_CASES): an upload on its way, the preview still shown, Send off and Stop reading "Cancel
     upload", its longest label; Stop is held to what Send is. P9-G5b adds that case at 320 x 568 (both drawer states), where Stop must also
     share Send's row: on a row of its own it lay 9 px above the composer's fold (the re-review of P6 fix 2). P9-G5b fix 1 (the review of that
     fix, which held only while the composer's content exactly fit): the busy state is no longer set by the check. The image is attached as
     above, the turn's upload is held (HOLD_UPLOAD_JS: the page's fetch never answers the POST of a turn) and Send is clicked, so that the
     page's own send() makes the chat, switches Send off, shows Stop reading "Cancel upload" and scrolls the composer to its end; each busy
     case has a page of its own, and the composer is measured where the page left it, scrolled. Stop must be in sight and, where the composer
     does not scroll, on Send's row (one that scrolls has a classic scrollbar, three buttons do not fit in what is left, and Stop wraps: the
     page's scroll is what keeps it in sight). Three more busy cases at 320 x 568, the drawer closed (COMPOSER_STRESS_CASES), put more in the
     composer than its rows: a Save, then a Send within 4 s, with seven chips (the "Settings saved" row must be gone, the page takes it down at
     Send); a Save while the upload goes up (the row is there, and Stop is still in sight); and eight chips, three rows of them
  l  (P6-E, two more cases for every viewport in every scheme) the image viewer open, with one picture and with two: neither the page nor
     the viewer overflows horizontally, the viewer lies inside the viewport's width, each pane's picture has loaded and is drawn whole
     inside its stage, each stage is at least 40 px each way, each footer gives its picture's pixel size, two panes are side by side from
     700 px and stacked below, and every control of the viewer (its buttons, sliders, its link switch and each stage) can be reached: once
     scrolled into view (the viewer scrolls inside itself where the page is short), its centre hits it. Where it scrolls with its panes
     stacked, each stage takes at most 60% of its height: a stage takes a touch for its picture, and the rest is where a finger scrolls the
     viewer. P6 fix 1 adds a landscape phone (568x320) to the viewer's cases

On the short viewports, where the page scrolls instead (667x375, 320x256), c and d give way to: the report keeps its
natural height (no scroller of its own, at least 120 px), the composer is not capped, the banner stays at the top of
the viewport while the page scrolls, and each button is hit-tested after it is scrolled into view.

The tall-composer cases (one per width and scheme, 568 px tall) put a composer taller than the viewport on the page and pin
the report's 120 px floor: #conversation keeps its 120 px, the composer stays inside the viewport and scrolls inside itself,
and Settings and Send are still reachable by scrolling it.

    venv/bin/python scripts/chat_ui_layout_check.py [--url http://127.0.0.1:PORT/]

Prints one `FAIL ...` line per failure, then `RESULT {"cases": N, "failures": [...]}`. Exits 1 on any failure and 2 when
Chrome or the app cannot be started, or when --url is refused. It writes no file (no screenshot, no checklist), so a --url needs no
--no-evidence here; it is refused all the same unless it is a tiny app on this machine's loopback (scripts/chat_ui_cdp.py url_problem
and tiny_only_problem: the dev server's port and the tunnel's, any other host, and any app whose /v1/models is not the tiny engine's),
before Chrome or anything else is started. Chrome is $CHROME, else /Applications/Google Chrome.app/..., else one on PATH.
Standard library only.
"""
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app.labels import CHEXBERT_14  # noqa: E402  (standard library only, like this script)
from scripts.chat_ui_cdp import App, Browser, exit_on_sigterm, find_chrome, tiny_only_problem, url_problem  # noqa: E402

WIDTHS = [320, 375, 800, 801, 820, 834, 900, 925, 1024, 1280]
HEIGHTS = [568, 900]
SHORT_VIEWPORTS = [(667, 375), (320, 256)]   # below 480 px tall the page scrolls instead of pinning the composer
TALL_WIDTHS = [375, 801, 1280]               # the tall-composer cases: one column, the first two-column width, a wide page
TALL_HEIGHT = 568
SCHEMES = ["light", "dark"]
MIN_REPORT_PX = 120
MIN_TALL_PX = 568
MIN_TOUCH_PX = 40   # a control's height: the page's buttons are 40 px at the least
CHIPS = ["beam 3", "100 tok", "cached", "k 4/3", "label on", "repair off"]
LONG_WORD = "x" * 140
REPORT_TEXT = ("The lungs are clear. There is no focal consolidation, pleural effusion or pneumothorax. The "
               "cardiomediastinal silhouette is within normal limits. No acute osseous abnormality is seen. ") * 3
# The card's own sections (check i): P6-B's images row, P6-C's similar X-rays, P6-D's published lines and matching reports.
SECTIONS = ["section.images", "section.neighbors", "section.published", "section.matches"]
NOTICE = "p.test-notice"   # P9-G5b: the tiny engine's notice above the report (render.js renderTestNotice): the synthetic turn's card is the tiny engine's
CARD_PARTS = SECTIONS + [NOTICE]   # what check i measures in the card
# P6 fix 1: the clamp's proof. Three lines of a matching report hold about 130 characters at 375 px and about 280 on a wide page.
MID_REPORT = ("Findings: The heart is normal in size. The lungs are clear, with no focal consolidation, pleural effusion or pneumothorax. "
              "Impression: No acute cardiopulmonary process.")   # 168 characters: over three lines at phone width, two lines on a wide page
SHORT_REPORT = "Findings: No acute process."
PROOF_CHARS = (150, 190)   # what check i takes for the ~170-character report
PICKER_STUDIES = 60        # check j's list: more studies than the list shows, so that it scrolls inside its own box
# Check k (P6-A, carried from P4): the composer's rows at the widths where its column is narrowest. From 801 px the page has its sidebar and,
# open, the drawer beside the conversation, so the composer is about 170 px wide there; at 320 px it is the phone's width.
COMPOSER_CASES = [(801, 568, True), (801, 650, True), (820, 568, True), (820, 650, True), (834, 568, True), (320, 568, False), (320, 568, True)]
COMPOSER_STATES = ["hint", "preview"]   # the page shows the re-run hint only with no image attached, and the preview only with one
# P6 fix 2: the composer busy with an upload (Stop beside Send, the preview still up), at the drawer-open sizes where its column is narrowest.
# P9-G5b: and at 320 x 568, the phone, where the rows stacked above them left Stop ("Cancel upload") 9 px above the composer's fold (the re-review
# of P6 fix 2): with the drawer closed and open, which is the same composer, as the page's drawer covers the conversation only below 800 px.
COMPOSER_BUSY_CASES = [case for case in COMPOSER_CASES if (case[2] and case[0] > 800) or case[:2] == (320, 568)]
# P9-G5b fix 1 (the review of the fix above, which held only while the composer's content exactly fit): one more row in the composer (the "Settings
# saved" row for 4 s after a Save, a third row of chips) put Stop ("Cancel upload") below the fold again, by 49 to 105 px at 320 x 568, as the classic
# scrollbar that the overflow adds narrows the composer by 15 px and Stop wraps onto a row of its own. These busy cases put more in the composer than
# its rows, at the phone with its drawer closed (the Save they make goes through the page's own drawer, which has to be shut when they start):
#   busy-after-save  Save, then Send within 4 s, seven chips: the page takes the saved row down at Send (its sentence is about the next Send)
#   busy-save        Save while the upload is on its way: the row is there, and Stop is still in sight (the page scrolls the composer to its end)
#   busy-8-chips     eight chips, three rows of them (the page's "compiled" chip needs --allow-compile, which the layout must take all the same)
COMPOSER_STRESS_CASES = [(320, 568, False)]
STRESS_STATES = ["busy-after-save", "busy-save", "busy-8-chips"]
BUSY_STATES = ["busy"] + STRESS_STATES   # "busy" is the P6 fix 2 / P9-G5b case: six chips, the upload on its way
SEVEN_CHIPS = ["beam 3", "100 tok", "cached", "k 4/3", "labels off", "raw text", "full budget"]   # the page's own words: all it shows with a gallery, but one
EIGHT_CHIPS = SEVEN_CHIPS + ["compiled"]   # and with --allow-compile
BUSY_CHIPS = {"busy": CHIPS, "busy-after-save": SEVEN_CHIPS, "busy-save": CHIPS, "busy-8-chips": EIGHT_CHIPS}   # injected before Send (a Save draws the page's own)
BUSY_STOP_LABEL = "Cancel upload"   # what Stop reads while an image goes up: the longest of its labels
PHONE_MAX_PX = 800                  # the page is one column up to here (styles.css): the composer is as wide as the page, and Stop shares Send's row
ROW_SLACK_PX = 2                    # two buttons whose tops are this close lie in the same row
LONG_FILE = "chest_xray_" + "a1b2c3d4" * 11 + ".png"   # 103 characters with no space: the widest name a row must take
VIEWER_STATES = ["single", "pair"]   # check l (P6-E): the viewer with one picture, and with the turn's X-ray beside a similar one
MIN_STAGE_PX = 40                    # a pane's picture area, each way: something to look at
STACK_BELOW_PX = 700                 # below this width the two panes are stacked
VIEWER_TOUCH_VIEWPORTS = [(568, 320)]   # check l only, as well (P6 fix 1): a landscape phone, where the stacked viewer scrolls by touch
MAX_TOUCH_STAGE_SHARE = 0.6          # what a stage may take of a stacked viewer that scrolls: the rest is where a finger scrolls it


def section_events() -> List[Dict[str, Any]]:
    """A finished private turn with every section a card can show, as the server streams it: synthetic, nothing MIMIC-derived. Each picture
    names a path on the page's own origin, which INJECT_SECTIONS_JS answers with a generated PNG, so nothing is fetched. The lists are as long
    as the options allow (12 similar X-rays, 10 matching reports) and one report holds a 140-character word, the widest a card must take."""
    def labels(*positives: str) -> Dict[str, int]:
        return {name: int(name in positives) for name in CHEXBERT_14}

    image = {"sha256": "ab" * 32, "filename": "layout.png", "source": "upload",
             "urls": {v: "/v1/messages/u_layout/image?variant={}".format(v) for v in ("original", "thumb", "model_input")}}
    neighbours = [{"rank": r, "similarity": round(0.95 - 0.03 * r, 3), "gallery_row": 100 + r, "study_id": 50000000 + r, "txt_row": 200 + r,
                   "image_url": "/v1/gallery/images/{}".format(100 + r), "labels": labels("Edema" if r % 2 else "Cardiomegaly")}
                  for r in range(1, 13)]
    texts = {2: LONG_WORD + " " + REPORT_TEXT, 3: MID_REPORT, 4: SHORT_REPORT}   # the rest are REPORT_TEXT, over three lines at any width
    matches = [{"rank": r, "similarity": round(0.6 - 0.02 * r, 3), "group": 300 + r, "group_size": 37 if r == 1 else 1, "txt_row": 400 + r,
                "report": texts.get(r, REPORT_TEXT), "labels": labels("Pleural Effusion")} for r in range(1, 11)]
    agreement = [{"rank": n["rank"], "agree": 12, "of": 14, "both_positive": [], "neighbor_only": ["Edema"], "generated_only": ["Cardiomegaly"]}
                 for n in neighbours]
    steps = [
        ("message_start", {"message_id": "m_layout", "user_message_id": "u_layout", "session_id": "s_layout", "mode": "private",
                           "model": {"name": "tiny", "device": "cpu"}, "options": {"k_images": 12, "k_reports": 10, "test_row": 7}, "image": image}),
        ("stage_end", {"stage": "preprocess", "ms": 2.0, "detail": {"format": "PNG", "input_px": [2544, 3056], "source": "upload"}}),
        ("stage_end", {"stage": "encode", "ms": 5.0, "detail": {"pooled_dim": 16}}),
        ("stage_end", {"stage": "retrieve", "ms": 3.0, "detail": {
            "image_neighbors": neighbours, "report_matches": matches,
            "true_report_rank": {"rank": 3, "of": 2663, "rank_dedup": 2, "n_tied": 1, "hit_at_10": True, "protocol": "i2t, official test split"},
            "gallery": {"build_id": "layout", "images": 200, "report_rows": 240, "report_groups": 78, "towers_identical": True}}}),
        ("content_block_delta", {"index": 0, "delta": {"type": "beam_snapshot", "step": 0, "text": REPORT_TEXT}}),
        ("stage_end", {"stage": "generate", "ms": 9.0, "detail": {"decode": "beam", "beam_size": 3, "tokens": 100, "stopped": "budget"}}),
        ("stage_end", {"stage": "label", "ms": 2.0, "detail": {"chexbert_14": labels("Cardiomegaly"), "positives": ["Cardiomegaly"],
                                                              "neighbor_agreement": agreement}}),
        ("stage_end", {"stage": "score", "ms": 1.0, "detail": {
            "rouge_l": 0.2, "bleu_1": 0.3, "bleu_4": 0.1, "reference_source": "test_split",
            "published": {"model_report": REPORT_TEXT, "floor_report": LONG_WORD + " " + REPORT_TEXT, "live_equals_published": False}}}),
        ("message_stop", {"message_id": "m_layout", "status": "done", "total_ms": 30.0, "report": REPORT_TEXT, "display_report": REPORT_TEXT,
                          "truncated_mid_sentence": False}),
    ]
    return [{"event": event, "data": dict(data, seq=i)} for i, (event, data) in enumerate(steps, start=1)]

INJECT_JS = """(args) => {
  const q = (s) => document.querySelector(s);
  q('#chips').replaceChildren(...args.chips.map((t) => { const s = document.createElement('span'); s.textContent = t; return s; }));
  const card = document.createElement('article');
  card.className = 'card';
  for (const t of [args.word, args.text]) { const p = document.createElement('p'); p.textContent = t; card.append(p); }
  q('#conversation').replaceChildren(card);
  return true;
}"""

TALL_JS = """(args) => {
  const filler = document.createElement('div');
  filler.id = 'tall-filler';
  filler.style.cssText = 'flex:1 0 100%;height:' + args.px + 'px';
  document.querySelector('#composer').prepend(filler);
  return true;
}"""

MEASURE_TALL_JS = """() => {
  const q = (s) => document.querySelector(s);
  const doc = document.documentElement;
  const box = (el) => { const r = el.getBoundingClientRect(); return [r.left, r.top, r.right, r.bottom]; };
  window.scrollTo(0, 0);
  const out = { doc: [doc.scrollWidth, doc.clientWidth], conversation: { ch: q('#conversation').clientHeight },
                composer: { box: box(q('#composer')), sh: q('#composer').scrollHeight, ch: q('#composer').clientHeight }, hits: {} };
  for (const s of ['#settings', '#send']) {   // the controls sit below the filler: reachable by scrolling the composer itself
    const el = q(s), c = q('#composer');
    const cb = box(c), eb = box(el);
    c.scrollTop += (eb[1] + eb[3]) / 2 - (cb[1] + cb[3]) / 2;   // its own scroll, not the page's: the page does not scroll here
    const b = box(el), x = (b[0] + b[2]) / 2, y = (b[1] + b[3]) / 2, top = document.elementFromPoint(x, y);
    out.hits[s] = !!top && (top === el || el.contains(top));
  }
  return out;
}"""

MEASURE_JS = """(args) => {
  const q = (s) => document.querySelector(s);
  const doc = document.documentElement;
  const shown = (el) => el.getClientRects().length > 0;
  const drawn = (el) => { const r = el.getBoundingClientRect(); return shown(el) && getComputedStyle(el).visibility !== 'hidden' && r.width > 0 && r.height > 0; };
  const box = (el) => { const r = el.getBoundingClientRect(); return [r.left, r.top, r.right, r.bottom]; };
  const name = (e) => !e ? null : (e.id ? '#' + e.id : e.tagName.toLowerCase());
  window.scrollTo(0, 0);
  const out = { doc: [doc.scrollWidth, doc.clientWidth], boxes: {}, hits: {}, rendered: {}, drawer: null };
  for (const s of ['#stop', '#drawer']) out.rendered[s] = drawn(q(s));
  if (out.rendered['#drawer']) out.drawer = box(q('#drawer'));
  for (const s of ['#conversation', '#composer']) {
    const el = q(s);
    out[s] = { sw: el.scrollWidth, cw: el.clientWidth, sh: el.scrollHeight, ch: el.clientHeight };
  }
  for (const s of ['#composer', '#settings', '#send', '#stop']) out.boxes[s] = shown(q(s)) ? box(q(s)) : null;
  for (const s of ['#settings', '#send', '#stop']) {
    const el = q(s);
    if (!shown(el)) { out.hits[s] = null; continue; }
    if (args.scroll) el.scrollIntoView({ block: 'center', inline: 'center' });
    const b = box(el), x = (b[0] + b[2]) / 2, y = (b[1] + b[3]) / 2, top = document.elementFromPoint(x, y);
    out.hits[s] = { ok: !!top && (top === el || el.contains(top)), at: [Math.round(x), Math.round(y)], got: name(top) };
  }
  const toggle = q('#sidebar-toggle');
  if (shown(toggle)) {
    window.scrollTo(0, 0);
    toggle.focus();
    const cs = getComputedStyle(toggle), b = box(toggle);
    out.toggle = { box: b, ring: cs.outlineStyle === 'none' ? null : parseFloat(cs.outlineWidth) + parseFloat(cs.outlineOffset) };
    toggle.blur();
  }
  if (args.scroll) {   // the page scrolls here: does the banner stay at the top of the viewport?
    const room = doc.scrollHeight - innerHeight;
    window.scrollTo(0, Math.min(150, Math.max(room, 0)));
    out.banner = { scrolled: window.scrollY, top: q('.banner').getBoundingClientRect().top };
    window.scrollTo(0, 0);
  }
  return out;
}"""


MEASURE_SAVE_JS = """(args) => {
  const q = (s) => document.querySelector(s);
  const box = (el) => { const r = el.getBoundingClientRect(); return [r.left, r.top, r.right, r.bottom]; };
  const drawn = (el) => { const r = el.getBoundingClientRect(); return el.getClientRects().length > 0 && getComputedStyle(el).visibility !== 'hidden' && r.width > 0 && r.height > 0; };
  const name = (e) => !e ? null : (e.id ? '#' + e.id : e.tagName.toLowerCase());
  const save = q('#drawer-save');
  window.scrollTo(0, 0);
  if (!save) return { present: false };
  const out = { present: true, rendered: drawn(save), drawer: drawn(q('#drawer')) ? box(q('#drawer')) : null, box: null, in_view: false, hit: null };
  if (!out.rendered) return out;
  const inside = (b) => b[0] >= -0.5 && b[2] <= innerWidth + 0.5 && b[1] >= -0.5 && b[3] <= innerHeight + 0.5;
  out.in_view = inside(box(save));   // where it is, with nothing scrolled: a bar that sticks to the bottom of the drawer is in sight
  save.scrollIntoView({ block: 'nearest', inline: 'nearest' });   // and where the user can scroll it to
  const b = box(save), x = (b[0] + b[2]) / 2, y = (b[1] + b[3]) / 2, top = document.elementFromPoint(x, y);
  out.box = b;
  out.hit = { ok: !!top && (top === save || save.contains(top)) && inside(b), at: [Math.round(x), Math.round(y)], got: name(top) };
  return out;
}"""


# Check i: a real card in place of the injected one, drawn by the page's own render.js from the synthetic turn (args.events), with its
# pictures answered by a generated PNG. It resolves once every picture has its source and has decoded.
INJECT_SECTIONS_JS = """async (args) => {
  const render = await import('/static/render.js');
  const state = await import('/static/state.js');
  const canvas = document.createElement('canvas');
  canvas.width = 96;
  canvas.height = 80;
  const g = canvas.getContext('2d');
  g.fillStyle = '#777777';
  g.fillRect(0, 0, 96, 80);
  g.fillStyle = '#dddddd';
  g.fillRect(24, 16, 48, 48);
  const png = canvas.toDataURL('image/png');
  const card = render.renderAssistantCard(state.replay(args.events), { loadImage: async () => png, labelNames: args.names, turn: 1, ui: new Map() });
  document.querySelector('#conversation').replaceChildren(card);
  for (let i = 0; i < 20 && [...card.querySelectorAll('img')].some((img) => !img.hasAttribute('src')); i++) await new Promise((r) => setTimeout(r, 10));
  await Promise.all([...card.querySelectorAll('img')].map((img) => img.decode().catch(() => null)));
  window.__layoutRender = render;   // MEASURE_SECTIONS_JS measures the clamps again for the drawer's state, as the page does on a resize
  render.settleClamps(card);
  return true;
}"""

MEASURE_SECTIONS_JS = """(args) => {
  const q = (s) => document.querySelector(s);
  const doc = document.documentElement;
  const box = (el) => { const r = el.getBoundingClientRect(); return [r.left, r.top, r.right, r.bottom]; };
  const card = q('#conversation article.card');
  if (card && window.__layoutRender) window.__layoutRender.settleClamps(card);
  window.scrollTo(0, 0);
  const out = { doc: [doc.scrollWidth, doc.clientWidth], conversation: [q('#conversation').scrollWidth, q('#conversation').clientWidth],
                card: card ? box(card) : null, sections: {} };
  for (const s of args.sections) {
    const el = card ? card.querySelector(s) : null;
    out.sections[s] = !el ? null : { box: box(el), sw: el.scrollWidth, cw: el.clientWidth,
      pictures: [...el.querySelectorAll('img')].map((i) => ({ box: box(i), loaded: i.complete && i.naturalWidth > 0 })),
      clamped: [...el.querySelectorAll('.clamp-text.clamped:not(.open)')].map((t) => {
        const row = t.nextElementSibling, toggle = row && row.classList.contains('clamp-actions') ? row : null;
        return { h: t.clientHeight, sh: t.scrollHeight, line: parseFloat(getComputedStyle(t).lineHeight) || 0, chars: t.textContent.length,
                 toggle: !!toggle && !toggle.hidden && toggle.getClientRects().length > 0 };
      }) };
  }
  return out;
}"""


def check_sections(m: Dict[str, Any], width: int, sections: List[str], drawer_open: bool = False) -> List[str]:
    """The failures in one measurement of the card's own sections (check i): each is there, inside the card's width and not wider inside
    than it is; each picture has loaded, has a size and lies inside the card; each clamped report shows at most three lines and offers
    Show all exactly when it overflows them; neither the page nor the conversation overflows horizontally. With the drawer closed, the
    clamp's proof: at 375 px and narrower a ~170-character report has Show all, at 1024 px and wider no report that short has one."""
    failures = []
    for name, (sw, cw) in (("documentElement", m["doc"]), ("#conversation", m["conversation"])):
        if sw > cw:
            failures.append("{} overflows horizontally: scrollWidth {} > clientWidth {}".format(name, sw, cw))
    card = m["card"]
    if card is None:
        return failures + ["no card was drawn"]
    if card[0] < -0.5 or card[2] > width + 0.5:
        failures.append("the card leaves the viewport's width: left {:.0f}, right {:.0f}, viewport {}".format(card[0], card[2], width))
    for selector in sections:
        section = m["sections"].get(selector)
        if section is None:
            failures.append("{} is not in the card".format(selector))
            continue
        left, _, right, _ = section["box"]
        if left < card[0] - 0.5 or right > card[2] + 0.5:
            failures.append("{} leaves the card: left {:.0f}, right {:.0f}, card {:.0f} to {:.0f}".format(selector, left, right, card[0], card[2]))
        if section["sw"] > section["cw"]:
            failures.append("{} overflows inside: scrollWidth {} > clientWidth {}".format(selector, section["sw"], section["cw"]))
        if not section["pictures"] and selector in PICTURED:
            failures.append("{} shows no picture".format(selector))
        clamped = section.get("clamped", [])
        if selector in CLAMPED and not any(text["sh"] > text["h"] + 1 for text in clamped):
            failures.append("{} clamps no report that overflows its three lines".format(selector))
        for n, text in enumerate(clamped, start=1):
            over = text["sh"] > text["h"] + 1
            if text["line"] > 0 and text["h"] > 3 * text["line"] + 2:
                failures.append("{} clamped report {} is {} px tall, over three lines of {:g} px".format(selector, n, text["h"], text["line"]))
            if text.get("toggle") and not over:
                failures.append("{} clamped report {} has a Show all that reveals nothing: scrollHeight {} <= clientHeight {}".format(
                    selector, n, text["sh"], text["h"]))
            if over and not text.get("toggle"):
                failures.append("{} clamped report {} overflows its three lines with no Show all".format(selector, n))
        if selector == "section.matches" and not drawer_open:
            mid = [text for text in clamped if PROOF_CHARS[0] <= text.get("chars", 0) <= PROOF_CHARS[1]]
            if width <= 375:
                if not mid:
                    failures.append("section.matches has no ~170-character report to prove the clamp with")
                if any(not text.get("toggle") for text in mid):
                    failures.append("section.matches: a ~170-character report has no Show all at {} px".format(width))
            if width >= 1024 and any(text.get("toggle") for text in clamped if text.get("chars", 0) <= PROOF_CHARS[1]):
                failures.append("section.matches: a short report has Show all at {} px".format(width))
        for n, picture in enumerate(section["pictures"], start=1):
            p_left, p_top, p_right, p_bottom = picture["box"]
            if not picture["loaded"]:
                failures.append("{} picture {} has not loaded".format(selector, n))
            if p_right - p_left < 1 or p_bottom - p_top < 1:
                failures.append("{} picture {} has no size".format(selector, n))
            if p_left < card[0] - 0.5 or p_right > card[2] + 0.5:
                failures.append("{} picture {} leaves the card: left {:.0f}, right {:.0f}".format(selector, n, p_left, p_right))
    return failures


PICTURED = {"section.images", "section.neighbors"}   # the sections that must show pictures in the synthetic turn
CLAMPED = {"section.published", "section.matches"}   # and those whose long reports must overflow their three lines, with Show all


# Check j (P6 fix 1): the test-split picker open with a full list, as app.js builds it: its button shown, its panel shown, PICKER_STUDIES study
# buttons in its list. The tiny app this check starts runs no gallery, so the page itself never offers the picker: the case opens it.
INJECT_PICKER_JS = """(args) => {
  const q = (s) => document.querySelector(s);
  const pick = q('#pick-study'), picker = q('#picker'), list = q('#picker-list'), status = q('#picker-status');
  if (!pick || !picker || !list) return false;
  pick.hidden = false;
  pick.setAttribute('aria-expanded', 'true');
  list.replaceChildren(...Array.from({ length: args.studies }, (_, row) => {
    const li = document.createElement('li'), b = document.createElement('button');
    b.type = 'button';
    b.className = 'study';
    b.textContent = 'study ' + (50000000 + row * 7) + ' · PA · test row ' + row;
    li.append(b);
    return li;
  }));
  if (status) status.textContent = args.studies + ' studies';
  picker.hidden = false;
  return true;
}"""

# The last card's header (its first part) scrolled to the top of the pane that scrolls it, and, where the page scrolls instead (short
# viewports), to just under the sticky banner; then what a point just inside its top hits. (A header can be taller than the pane, which keeps
# its 120 px floor while the picker fills the composer: centring it would put its top under the banner.) And the page, the conversation and the
# composer for horizontal overflow, and the picker's button and panel.
MEASURE_PICKER_JS = """() => {
  const q = (s) => document.querySelector(s);
  const doc = document.documentElement;
  const box = (el) => { const r = el.getBoundingClientRect(); return [r.left, r.top, r.right, r.bottom]; };
  const drawn = (el) => !!el && el.getClientRects().length > 0 && getComputedStyle(el).visibility !== 'hidden';
  const name = (e) => !e ? null : (e.id ? '#' + e.id : e.tagName.toLowerCase());
  const cards = document.querySelectorAll('#conversation article.card');
  const card = cards[cards.length - 1];
  const head = card ? (card.firstElementChild || card) : null;
  const out = { doc: [doc.scrollWidth, doc.clientWidth], conversation: [q('#conversation').scrollWidth, q('#conversation').clientWidth],
                composer: [q('#composer').scrollWidth, q('#composer').clientWidth],
                picker: drawn(q('#picker')) ? box(q('#picker')) : null, pick: drawn(q('#pick-study')) ? box(q('#pick-study')) : null,
                head: null, hit: null };
  if (!head) return out;
  head.scrollIntoView({ block: 'start', inline: 'nearest' });
  const under = q('.banner').getBoundingClientRect().bottom;
  if (head.getBoundingClientRect().top < under) window.scrollBy(0, head.getBoundingClientRect().top - under - 8);
  const b = box(head), x = (b[0] + b[2]) / 2, y = b[1] + Math.min(10, (b[3] - b[1]) / 2), top = document.elementFromPoint(x, y);
  out.head = b;
  out.hit = { ok: !!top && card.contains(top), got: name(top) };
  const pick = q('#pick-study');   // the button sits in the composer's last row: with the list open the composer may scroll to it
  if (drawn(pick)) {
    pick.scrollIntoView({ block: 'nearest', inline: 'nearest' });
    const pb = box(pick), at = document.elementFromPoint((pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2);
    out.pickHit = { ok: !!at && (at === pick || pick.contains(at)), got: name(at) };
  }
  return out;
}"""


# Check k: the composer as the page leaves it with the re-run hint shown (no image attached) or with an image attached (the preview, built by
# composer.js from a pasted file, as a paste anywhere on the page does). args: chips, name, state ('hint' or 'preview'), drawer. A busy composer
# (P6 fix 2) starts from 'preview' and goes busy by the page's own Send (P9-G5b fix 1: HOLD_UPLOAD_JS, then a click): nothing of the busy state
# is written here, so that what is measured is what the user sees.
INJECT_COMPOSER_JS = """async (args) => {
  const q = (s) => document.querySelector(s);
  const hint = q('#rerun-hint'), preview = q('#preview');
  if (!hint || !preview || !q('#stop') || !q('#send')) return false;
  q('#drawer').hidden = !args.drawer;
  q('#chips').replaceChildren(...args.chips.map((t) => { const s = document.createElement('span'); s.textContent = t; return s; }));
  const remove = preview.querySelector('button');
  if (remove) remove.click();
  if (args.state === 'hint') {
    hint.textContent = 'No new image: Send re-runs ' + args.name + ' with these settings.';
    hint.hidden = false;
    return true;
  }
  hint.hidden = true;
  const canvas = document.createElement('canvas');   // an X-ray's own size: its preview line says "2544×3056 px", the widest a size gets
  canvas.width = 2544;
  canvas.height = 3056;
  canvas.getContext('2d').fillRect(0, 0, 2544, 3056);
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/png'));
  const data = new DataTransfer();
  data.items.add(new File([blob], args.name, { type: 'image/png' }));
  document.dispatchEvent(new ClipboardEvent('paste', { bubbles: true, cancelable: true, clipboardData: data }));
  const img = preview.querySelector('img');
  if (!img || preview.hidden) return false;
  await img.decode().catch(() => null);
  for (let i = 0; i < 50 && !/ px /.test(preview.textContent); i++) await new Promise((r) => setTimeout(r, 10));   // its line has the pixel size
  return true;
}"""

# The chips as the check sets them (the page draws its own again after a Save and at start-up): args.chips, as text.
SET_CHIPS_JS = """(args) => {
  const q = (s) => document.querySelector(s);
  q('#chips').replaceChildren(...args.chips.map((t) => { const s = document.createElement('span'); s.textContent = t; return s; }));
  return q('#chips').children.length;
}"""

# The turn's upload, held: the page's own fetch never answers the POST of a turn (/v1/sessions/<id>/messages) and ends it only when the page aborts it
# (Cancel upload still works), and passes every other request through, so the page makes the chat for real and then waits, as it does on a slow
# tunnel. The held request ends with the page: each case that uses it has a page of its own.
HOLD_UPLOAD_JS = """() => {
  const original = window.fetch.bind(window);
  window.__heldUploads = 0;
  window.fetch = (url, init) => {
    if (init && init.method === 'POST' && /\\/v1\\/sessions\\/[^/]+\\/messages$/.test(String(url))) {
      window.__heldUploads += 1;
      return new Promise((resolve, reject) => {
        if (init.signal) init.signal.addEventListener('abort', () => reject(init.signal.reason), { once: true });
      });
    }
    return original(url, init);
  };
  return true;
}"""

# args.keep: a busy composer is measured where the page left it (it scrolls it to its end, so that Stop is in sight); the others as they are drawn,
# nothing scrolled. The page's own words come back with it: what Stop reads, whether Send is off, and the saved row (shown when it has text and a height).
MEASURE_COMPOSER_JS = """(args) => {
  const q = (s) => document.querySelector(s);
  const doc = document.documentElement, composer = q('#composer'), send = q('#send'), stop = q('#stop'), saved = q('#saved');
  const box = (el) => { const r = el.getBoundingClientRect(); return [r.left, r.top, r.right, r.bottom]; };
  const shown = (el) => !!el && !el.hidden && el.getClientRects().length > 0;
  const name = (e) => !e ? null : (e.id ? '#' + e.id : e.tagName.toLowerCase());
  window.scrollTo(0, 0);
  if (!args.keep) composer.scrollTop = 0;   // as it is drawn: nothing scrolled
  const out = { doc: [doc.scrollWidth, doc.clientWidth],
                composer: { box: box(composer), sw: composer.scrollWidth, cw: composer.clientWidth, sh: composer.scrollHeight, ch: composer.clientHeight,
                            st: composer.scrollTop },
                conversation: q('#conversation').clientHeight, rows: {}, send: shown(send) ? box(send) : null, hit: null,
                stop: shown(stop) ? box(stop) : null, stop_hit: null, stop_text: stop.textContent, send_disabled: send.disabled,
                saved: { text: saved ? saved.textContent : '', shown: !!saved && saved.textContent.trim() !== '' && saved.getBoundingClientRect().height > 5 } };
  for (const s of ['#rerun-hint', '#preview']) out.rows[s] = shown(q(s)) ? box(q(s)) : null;
  const hit = (el, b) => { const top = document.elementFromPoint((b[0] + b[2]) / 2, (b[1] + b[3]) / 2);
                           return { ok: !!top && (top === el || el.contains(top)), got: name(top) }; };
  if (out.send) out.hit = hit(send, out.send);
  if (out.stop) out.stop_hit = hit(stop, out.stop);
  return out;
}"""


def _in_sight(name: str, label: str, b: List[float], hit: Optional[Dict[str, Any]], composer: List[float], width: int,
              height: int) -> List[str]:
    """A button of the composer inside its box (its fold: the composer scrolls inside, and a button under its bottom is out of sight until it
    is scrolled) and inside the viewport, its centre hitting it (check k)."""
    failures = []
    if b[1] < composer[1] - 0.5 or b[3] > composer[3] + 0.5:
        failures.append("{} is below the composer's fold: {} {:.0f} to {:.0f}, composer {:.0f} to {:.0f} (it is out of sight until the "
                        "composer is scrolled)".format(name, label, b[1], b[3], composer[1], composer[3]))
    if b[0] < -0.5 or b[1] < -0.5 or b[2] > width + 0.5 or b[3] > height + 0.5:
        failures.append("{} is not inside the {}x{} viewport: {}".format(name, width, height, [round(v) for v in b]))
    if not (hit and hit["ok"]):
        failures.append("{} is covered: its centre hits {}".format(name, (hit or {}).get("got") or "nothing"))
    return failures


def check_composer(m: Dict[str, Any], width: int, height: int, state: str) -> List[str]:
    """The failures in one measurement of the composer's rows (check k): the row the case shows is rendered inside the composer and the other
    is not; nothing overflows horizontally; the report keeps its floor; and Send lies inside the composer's box (its fold: the composer
    scrolls inside, and a Send under its bottom is out of sight until it is scrolled) and inside the viewport, its centre hitting it. A busy
    composer (P6 fix 2; every state of BUSY_STATES) shows the preview, and its Stop is held to the same, where the page left the composer (it
    scrolls it to its end, P9-G5b fix 1). On a phone (800 px and narrower: one column, the composer as wide as the page) Stop also shares
    Send's row (P9-G5b), unless the composer scrolls: the longest label, "Cancel upload", on a row of its own was the composer's last, which at
    320 x 568 lay under its fold, and a composer that scrolls has a classic scrollbar, which leaves the three buttons no room for one row. The
    busy case must be the upload (Stop reads "Cancel upload", Send is off): it was entered by the page's own Send. The saved row is gone after a
    Save and a Send ("busy-after-save": its sentence is about the next Send) and there where a Save came during the upload ("busy-save")."""
    failures = []
    for name, (sw, cw) in (("documentElement", m["doc"]), ("#composer", (m["composer"]["sw"], m["composer"]["cw"]))):
        if sw > cw:
            failures.append("{} overflows horizontally: scrollWidth {} > clientWidth {}".format(name, sw, cw))
    row, other = ("#rerun-hint", "#preview") if state == "hint" else ("#preview", "#rerun-hint")
    composer = m["composer"]["box"]
    shown = m["rows"].get(row)
    if shown is None:
        failures.append("{} is not rendered, but the case shows it".format(row))
    elif shown[0] < composer[0] - 0.5 or shown[2] > composer[2] + 0.5:
        failures.append("{} leaves the composer's width: left {:.0f}, right {:.0f}, composer {:.0f} to {:.0f}".format(
            row, shown[0], shown[2], composer[0], composer[2]))
    if m["rows"].get(other) is not None:
        failures.append("{} is rendered with {}, which the page never shows together".format(other, row))
    if m["conversation"] < MIN_REPORT_PX:
        failures.append("the report area is {} px tall, under {}".format(m["conversation"], MIN_REPORT_PX))
    if state.startswith("busy"):
        scrolls = m["composer"].get("sh", 0) > m["composer"].get("ch", 0) + 1
        if m.get("stop") is None:
            failures.append("#stop is not rendered, but the case shows it")
        else:
            failures += _in_sight("#stop", "Stop", m["stop"], m.get("stop_hit"), composer, width, height)
            if width <= PHONE_MAX_PX and not scrolls and m["send"] is not None and abs(m["stop"][1] - m["send"][1]) > ROW_SLACK_PX:
                failures.append("#stop is not on Send's row at {} px: Stop starts at {:.0f}, Send at {:.0f}".format(width, m["stop"][1], m["send"][1]))
        if m.get("stop_text") != BUSY_STOP_LABEL:
            failures.append("#stop reads {!r}, not {!r}: the case is not measuring the upload".format(m.get("stop_text"), BUSY_STOP_LABEL))
        if not m.get("send_disabled"):
            failures.append("#send is on while the upload goes up")
        saved = m.get("saved") or {}
        if state == "busy-after-save" and saved.get("shown"):
            failures.append("the saved row is still showing after Send ({!r}): its sentence is about the next Send, which this was".format(saved.get("text")))
        if state == "busy-save" and not saved.get("shown"):
            failures.append("the saved row is not showing: the case is not measuring a Save while the upload goes up")
    send = m["send"]
    if send is None:
        return failures + ["#send is not rendered"]
    return failures + _in_sight("#send", "Send", send, m["hit"], composer, width, height)


def _press_send(browser: Browser) -> None:
    """Send, as a click on its centre, with the turn's upload held (HOLD_UPLOAD_JS): the page makes its chat, goes busy and waits for an answer that
    never comes. Returns once Stop reads what it reads while an image goes up. TimeoutError or RuntimeError when the page does not get there."""
    browser.run(HOLD_UPLOAD_JS, None)
    browser.click("#send")
    browser.wait_for("window.__heldUploads === 1 && !document.querySelector('#stop').hidden && document.querySelector('#stop').textContent === {}".format(
        json.dumps(BUSY_STOP_LABEL)), what="the upload to be on its way, Stop reading {!r}".format(BUSY_STOP_LABEL))


def _save_settings(browser: Browser) -> None:
    """Settings, then Save in the drawer, as clicks: the page closes the drawer and says "Settings saved." beside the chips, which it draws
    again from its own settings. Returns once the row is there."""
    browser.click("#settings")
    browser.wait_for("!document.querySelector('#drawer').hidden", what="the drawer to open")
    browser.click("#drawer-save")
    browser.wait_for("document.querySelector('#saved').textContent.trim() !== ''", what="the saved row")


def enter_busy(browser: Browser, state: str, drawer: bool) -> Optional[str]:
    """The page in the busy composer of one case, by what a user does (P9-G5b fix 1): the image attached as in the idle cases, then, for
    "busy" and "busy-8-chips", Send with the upload held; for "busy-after-save" Save, the seven chips again (the page drew its own), Send;
    for "busy-save" Send, then Save while the upload goes up. -> None, or why the page did not get there (a failure of that case, in words)."""
    chips = BUSY_CHIPS[state]
    if not browser.run(INJECT_COMPOSER_JS, {"chips": chips, "name": LONG_FILE, "state": "preview", "drawer": drawer}):
        return "the image was not attached"
    try:
        if state == "busy-after-save":
            _save_settings(browser)
            browser.run(SET_CHIPS_JS, {"chips": chips})   # a Save draws the page's own chips: the seven are put back, as a server with a gallery shows them
        _press_send(browser)
        if state == "busy-save":
            _save_settings(browser)
    except (RuntimeError, TimeoutError) as exc:
        return str(exc)
    return None


# Check l (P6-E): the page's own viewer.js opened over the page with generated pictures (data: URLs, so nothing is fetched): a 512x640 one, and
# a 320x320 one beside it. It resolves once each picture has decoded and the viewer has drawn it.
INJECT_VIEWER_JS = """async (args) => {
  const viewer = await import('/static/viewer.js');
  if (window.__layoutViewer) window.__layoutViewer.close({ restore: false });
  const picture = (w, h, shade) => {
    const c = document.createElement('canvas');
    c.width = w;
    c.height = h;
    const g = c.getContext('2d');
    g.fillStyle = shade;
    g.fillRect(0, 0, w, h);
    g.fillStyle = '#dddddd';
    g.fillRect(w / 4, h / 4, w / 2, h / 2);
    return c.toDataURL('image/png');
  };
  const left = { url: picture(512, 640, '#555555'), label: 'Your X-ray', note: '' };
  const right = { url: picture(320, 320, '#333333'), label: 'Similar X-ray #1', note: '(stored gallery size)' };
  window.__layoutViewer = viewer.openViewer(args.pair ? { left, right } : { left }, { root: document.querySelector('#viewer'), document, window });
  const imgs = [...document.querySelectorAll('#viewer img')];
  for (let i = 0; i < 50 && imgs.some((img) => !img.complete || !img.naturalWidth); i++) await new Promise((r) => setTimeout(r, 10));
  await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
  return imgs.length === (args.pair ? 2 : 1) && imgs.every((img) => img.naturalWidth > 0);
}"""

MEASURE_VIEWER_JS = """() => {
  const q = (s) => document.querySelector(s);
  const doc = document.documentElement, viewer = q('#viewer');
  const box = (el) => { const r = el.getBoundingClientRect(); return [r.left, r.top, r.right, r.bottom]; };
  const name = (e) => !e ? null : e.id ? '#' + e.id : e.getAttribute('data-viewer') ? '[' + e.getAttribute('data-viewer') + ']'
    : (e.getAttribute('class') || e.tagName.toLowerCase());
  if (viewer.hidden) return { doc: [doc.scrollWidth, doc.clientWidth], viewer: null, panes: [], controls: [] };
  viewer.scrollTop = 0;
  const panes = [...viewer.querySelectorAll('.viewer-pane')].map((p) => {
    const img = p.querySelector('img');
    return { box: box(p), stage: box(p.querySelector('.viewer-stage')), img: { box: box(img), loaded: img.complete && img.naturalWidth > 0 },
             footer: p.querySelector('figcaption').textContent };
  });
  const controls = [...viewer.querySelectorAll('button, input, [tabindex]')].map((c) => {
    c.scrollIntoView({ block: 'nearest', inline: 'nearest' });
    const b = box(c), x = (b[0] + b[2]) / 2, y = (b[1] + b[3]) / 2, top = document.elementFromPoint(x, y);
    return { name: name(c), ok: !!top && (top === c || c.contains(top)), got: name(top) };
  });
  viewer.scrollTop = 0;
  return { doc: [doc.scrollWidth, doc.clientWidth], viewer: { box: box(viewer), sw: viewer.scrollWidth, cw: viewer.clientWidth, ch: viewer.clientHeight },
           panes, controls };
}"""


def check_viewer(m: Dict[str, Any], width: int, height: int, pair: bool, scroll: bool = False) -> List[str]:
    """The failures in one measurement of the open viewer (check l): no horizontal overflow, the viewer inside the viewport's width, a pane per
    picture, each picture loaded and whole inside its stage, each stage big enough to look at and inside the viewer (and the viewport, where
    the page does not scroll: there the viewer scrolls inside itself, and a pane below its fold is fine if it can be scrolled to; stacked, a
    stage then takes at most 60% of its height, P6 fix 1), each footer with its pixel size, two panes side by side from 700 px and stacked
    below, and every control reachable."""
    failures = []
    if m["doc"][0] > m["doc"][1]:
        failures.append("documentElement overflows horizontally: scrollWidth {} > clientWidth {}".format(*m["doc"]))
    viewer = m["viewer"]
    if viewer is None:
        return failures + ["the viewer is not open"]
    if viewer["sw"] > viewer["cw"]:
        failures.append("#viewer overflows horizontally: scrollWidth {} > clientWidth {}".format(viewer["sw"], viewer["cw"]))
    vbox = viewer["box"]
    if vbox[0] < -0.5 or vbox[2] > width + 0.5:
        failures.append("#viewer leaves the viewport's width: left {:.0f}, right {:.0f}, viewport {}".format(vbox[0], vbox[2], width))
    panes = m["panes"]
    if len(panes) != (2 if pair else 1):
        failures.append("{} pane(s), not {}".format(len(panes), 2 if pair else 1))
    for n, pane in enumerate(panes, start=1):
        stage, img = pane["stage"], pane["img"]
        w, h = stage[2] - stage[0], stage[3] - stage[1]
        if w < MIN_STAGE_PX - 0.5 or h < MIN_STAGE_PX - 0.5:
            failures.append("pane {} is {:.0f}x{:.0f} px, under {} px each way".format(n, w, h, MIN_STAGE_PX))
        if stage[0] < vbox[0] - 0.5 or stage[2] > vbox[2] + 0.5:
            failures.append("pane {} leaves the viewer's width".format(n))
        if not scroll and (stage[1] < -0.5 or stage[3] > height + 0.5):
            failures.append("pane {} leaves the viewport: top {:.0f}, bottom {:.0f}, viewport {}".format(n, stage[1], stage[3], height))
        if scroll and width < STACK_BELOW_PX and viewer.get("ch") and h > MAX_TOUCH_STAGE_SHARE * viewer["ch"] + 0.5:   # P6 fix 1 (8)
            failures.append("pane {} takes {:.0f} of the {:.0f} px viewer ({:.0%}), over {:.0%}: little of it is left for a touch to scroll "
                            "it".format(n, h, viewer["ch"], h / viewer["ch"], MAX_TOUCH_STAGE_SHARE))
        if not img["loaded"]:
            failures.append("pane {}'s picture has not loaded".format(n))
        elif img["box"][0] < stage[0] - 1 or img["box"][1] < stage[1] - 1 or img["box"][2] > stage[2] + 1 or img["box"][3] > stage[3] + 1:
            failures.append("pane {}'s picture is not whole inside its stage at the fit: {} in {}".format(
                n, [round(v) for v in img["box"]], [round(v) for v in stage]))
        if " px" not in pane["footer"]:
            failures.append("pane {}'s footer gives no pixel size: {!r}".format(n, pane["footer"]))
    if pair and len(panes) == 2:
        a, b = panes[0]["box"], panes[1]["box"]
        if width >= STACK_BELOW_PX and not (b[0] >= a[2] - 1 and abs(b[1] - a[1]) <= 1):
            failures.append("the two panes are not side by side at {} px".format(width))
        if width < STACK_BELOW_PX and not (b[1] >= a[3] - 1 and abs(b[0] - a[0]) <= 1):
            failures.append("the two panes are not stacked below 700 px ({} px)".format(width))
    for control in m["controls"]:
        if not control["ok"]:
            failures.append("{} cannot be reached: its centre hits {}".format(control["name"], control.get("got") or "nothing"))
    return failures


def check_picker(m: Dict[str, Any], width: int, height: int, scroll: bool = False) -> List[str]:
    """The failures in a page whose test-split picker is open (check j): no horizontal overflow; the button and the list rendered inside the
    viewport's width, the list inside its height too where the composer is pinned rather than scrolled with the page, and the button reachable
    by scrolling (the composer, or the page); and the last card's header, scrolled into view, not covered by the composer, the picker or
    anything else."""
    failures = []
    for name, (sw, cw) in (("documentElement", m["doc"]), ("#conversation", m["conversation"]), ("#composer", m["composer"])):
        if sw > cw:
            failures.append("{} overflows horizontally: scrollWidth {} > clientWidth {}".format(name, sw, cw))
    for name, key in (("#picker", "picker"), ("#pick-study", "pick")):
        b = m[key]
        if b is None:
            failures.append("{} is not rendered, but the case opens the picker".format(name))
        elif b[0] < -0.5 or b[2] > width + 0.5:
            failures.append("{} leaves the viewport's width: left {:.0f}, right {:.0f}, viewport {}".format(name, b[0], b[2], width))
        elif key == "picker" and not scroll and (b[1] < -0.5 or b[3] > height + 0.5):
            failures.append("{} is not inside the {}x{} viewport: top {:.0f}, bottom {:.0f}".format(name, width, height, b[1], b[3]))
    reach = m.get("pickHit")
    if m["pick"] is not None and not (reach and reach["ok"]):
        failures.append("#pick-study cannot be reached by scrolling: its centre hits {}".format((reach or {}).get("got") or "nothing"))
    if m["head"] is None:
        return failures + ["no card was drawn"]
    if not m["hit"]["ok"]:
        failures.append("the last card's header is covered by {}".format(m["hit"]["got"] or "nothing (it is outside the viewport)"))
    return failures


def check_save(m: Dict[str, Any], width: int, height: int, scroll: bool, drawer_open: bool) -> List[str]:
    """The failures in one measurement of the Save button: with the drawer closed it is not rendered; with it open it is rendered, inside the
    drawer's width and the viewport's, a touch target tall, in sight where it stands (not where the page scrolls: the page is then the one
    that scrolls to it) and not covered once it is scrolled to."""
    if not m["present"]:
        return ["no #drawer-save in the page: the drawer is not a form that ends in Save"]
    if not drawer_open:
        return ["#drawer-save is rendered, but the case hides the drawer"] if m["rendered"] else []
    if not m["rendered"]:
        return ["#drawer-save is not rendered, but the case opens the drawer"]
    failures = []
    left, top, right, bottom = m["box"]
    drawer = m["drawer"]
    if drawer is not None and (left < drawer[0] - 0.5 or right > drawer[2] + 0.5):
        failures.append("#drawer-save leaves the drawer: left {:.0f}, right {:.0f}, drawer {:.0f} to {:.0f}".format(left, right, drawer[0], drawer[2]))
    if left < -0.5 or right > width + 0.5:
        failures.append("#drawer-save leaves the viewport's width: left {:.0f}, right {:.0f}, viewport {}".format(left, right, width))
    if bottom - top < MIN_TOUCH_PX - 0.5:
        failures.append("#drawer-save is {:.0f} px tall, under {}".format(bottom - top, MIN_TOUCH_PX))
    if m["hit"] is None or not m["hit"]["ok"]:
        hit = m["hit"] or {"at": (), "got": None}
        failures.append("#drawer-save is covered or outside the viewport once scrolled to: its centre {} hits {}".format(
            tuple(hit["at"]), hit["got"] or "nothing"))
    if not scroll and not m["in_view"]:
        failures.append("#drawer-save is not in sight without scrolling (the bar should stick to the bottom of the drawer)")
    return failures


def check_state(m: Dict[str, Any], width: int, height: int, scroll: bool, drawer_open: bool = False,
                stop_shown: bool = False) -> List[str]:
    """The failures in one measurement of one state: what is wrong, in words. drawer_open and stop_shown say what the case
    shows: #drawer and #stop must be rendered, with a size, exactly then."""
    failures = []
    for selector, claimed in (("#stop", stop_shown), ("#drawer", drawer_open)):
        if claimed and not m["rendered"][selector]:
            failures.append("{} is not rendered, but the case shows it".format(selector))
        elif not claimed and m["rendered"][selector]:
            failures.append("{} is rendered, but the case hides it".format(selector))
    if drawer_open and m["drawer"] is not None and (m["drawer"][0] < -0.5 or m["drawer"][2] > width + 0.5):
        failures.append("#drawer leaves the viewport's width: left {:.0f}, right {:.0f}, viewport {}".format(
            m["drawer"][0], m["drawer"][2], width))
    for name, (sw, cw) in (("documentElement", m["doc"]), ("#conversation", (m["#conversation"]["sw"],
                                                                          m["#conversation"]["cw"])),
                           ("#composer", (m["#composer"]["sw"], m["#composer"]["cw"]))):
        if sw > cw:
            failures.append("{} overflows horizontally: scrollWidth {} > clientWidth {}".format(name, sw, cw))
    for selector in ("#settings", "#send", "#stop"):
        hit = m["hits"][selector]
        if hit is None and selector != "#stop":
            failures.append("{} is not rendered".format(selector))
        elif hit is not None and not hit["ok"]:
            failures.append("{} is covered: its centre {} hits {}".format(
                selector, tuple(hit["at"]), hit["got"] or "nothing (outside the viewport)"))
    present = [s for s in ("#settings", "#send", "#stop") if m["boxes"][s] is not None]
    for first, second in zip(present, present[1:]):
        a, b = m["boxes"][first], m["boxes"][second]
        if not (b[1] >= a[3] - 1 or b[0] >= a[2] - 1):   # a row below, or the same row and further right
            failures.append("{} is not after {} in visual order".format(second, first))
    toggle = m.get("toggle")
    if toggle is not None:
        ring = toggle["ring"]
        left, top = toggle["box"][0], toggle["box"][1]
        if ring is None:
            failures.append("#sidebar-toggle shows no focus ring")
        elif left - ring < -0.5 or top - ring < -0.5:
            failures.append("#sidebar-toggle focus ring leaves the viewport (box at {:.0f},{:.0f}, ring {:g}px)".format(
                left, top, ring))
    composer = m["boxes"]["#composer"]
    conversation = m["#conversation"]
    if scroll:   # short viewport: the page scrolls, the report keeps its natural height
        if conversation["sh"] > conversation["ch"] + 1:
            failures.append("the report is clipped: #conversation scrollHeight {} > clientHeight {}".format(
                conversation["sh"], conversation["ch"]))
        if conversation["ch"] < MIN_REPORT_PX:
            failures.append("the report area is {} px tall, under {}".format(conversation["ch"], MIN_REPORT_PX))
        if m["#composer"]["sh"] > m["#composer"]["ch"] + 1:
            failures.append("#composer is capped: scrollHeight {} > clientHeight {}".format(
                m["#composer"]["sh"], m["#composer"]["ch"]))
        banner = m["banner"]
        if abs(banner["top"]) > 0.5:
            failures.append("the banner scrolls away: its top is {:.0f} px after scrolling {} px".format(
                banner["top"], banner["scrolled"]))
        return failures
    if composer is None or composer[0] < -0.5 or composer[1] < -0.5 or composer[2] > width + 0.5 \
            or composer[3] > height + 0.5:
        failures.append("#composer is not inside the {}x{} viewport: {}".format(
            width, height, None if composer is None else [round(v) for v in composer]))
    if height >= MIN_TALL_PX and conversation["ch"] < MIN_REPORT_PX:
        failures.append("the report area is {} px tall, under {}".format(conversation["ch"], MIN_REPORT_PX))
    return failures


def check_tall(m: Dict[str, Any], height: int) -> List[str]:
    """The failures in a page whose composer is taller than the viewport: the report keeps its floor, the composer stays on
    the page and scrolls inside itself, and its buttons can be scrolled to."""
    failures = []
    if m["doc"][0] > m["doc"][1]:
        failures.append("documentElement overflows horizontally: scrollWidth {} > clientWidth {}".format(*m["doc"]))
    if m["conversation"]["ch"] < MIN_REPORT_PX:
        failures.append("the report area is {} px tall, under the {} px floor".format(m["conversation"]["ch"], MIN_REPORT_PX))
    composer = m["composer"]
    if composer["box"][3] > height + 0.5:
        failures.append("#composer runs past the bottom of the viewport: {:.0f} > {}".format(composer["box"][3], height))
    if composer["sh"] <= composer["ch"] + 1:
        failures.append("#composer does not scroll inside itself: scrollHeight {} <= clientHeight {}".format(
            composer["sh"], composer["ch"]))
    for selector, reachable in sorted(m["hits"].items()):
        if not reachable:
            failures.append("{} cannot be reached by scrolling the composer".format(selector))
    return failures


def run_checks(browser: Browser, url: str) -> Tuple[int, List[str]]:
    """Every viewport in every scheme, drawer closed and open, Stop hidden and shown. A case is a viewport in a scheme
    with the drawer closed or open, or (one case each) its Save button and its card sections in both drawer states, its picker open, and
    (P6-E) the viewer open with one picture and with two, or (P6-A) one state of the composer's rows at a composer case's viewport.
    -> (cases, failures); a failure seen in both schemes is reported once."""
    cases = 0
    seen = {}   # type: Dict[Tuple[str, str, str], List[str]]   # (viewport, state, failure) -> schemes
    viewports = [(w, h, False) for h in HEIGHTS for w in WIDTHS] + [(w, h, True) for w, h in SHORT_VIEWPORTS]
    serial = 0
    for width, height, scroll in viewports:
        for scheme in SCHEMES:
            serial += 1
            browser.open("{}?case={}".format(url, serial), width, height, scheme)
            browser.key("Tab", "Tab", 9)   # keyboard modality, so :focus-visible matches when the toggle is focused
            browser.run(INJECT_JS, {"chips": CHIPS, "word": LONG_WORD, "text": REPORT_TEXT})
            for drawer in ("closed", "open"):
                cases += 1
                browser.evaluate("document.querySelector('#drawer').hidden = {}".format(
                    "false" if drawer == "open" else "true"))
                for stop in ("hidden", "shown"):
                    browser.evaluate("document.querySelector('#stop').hidden = {}".format(
                        "false" if stop == "shown" else "true"))
                    measured = browser.run(MEASURE_JS, {"scroll": scroll})
                    for failure in check_state(measured, width, height, scroll, drawer == "open", stop == "shown"):
                        seen.setdefault(("{}x{}".format(width, height), "drawer={} stop={}".format(drawer, stop), failure), []).append(scheme)
            cases += 1   # the Save button, with the drawer closed and then open
            browser.evaluate("document.querySelector('#stop').hidden = true")
            for drawer in ("closed", "open"):
                browser.evaluate("document.querySelector('#drawer').hidden = {}".format("false" if drawer == "open" else "true"))
                for failure in check_save(browser.run(MEASURE_SAVE_JS, {"scroll": scroll}), width, height, scroll, drawer == "open"):
                    seen.setdefault(("{}x{}".format(width, height), "drawer={} save".format(drawer), failure), []).append(scheme)
            cases += 1   # the card's own sections (P6-B..D), with the drawer closed and then open
            browser.evaluate("document.querySelector('#drawer').hidden = true")
            browser.run(INJECT_SECTIONS_JS, {"events": section_events(), "names": CHEXBERT_14})
            for drawer in ("closed", "open"):
                browser.evaluate("document.querySelector('#drawer').hidden = {}".format("false" if drawer == "open" else "true"))
                for failure in check_sections(browser.run(MEASURE_SECTIONS_JS, {"sections": CARD_PARTS}), width, CARD_PARTS, drawer == "open"):
                    seen.setdefault(("{}x{}".format(width, height), "drawer={} card sections".format(drawer), failure), []).append(scheme)
            cases += 1   # the test-split picker open, with the drawer closed (P6 fix 1)
            browser.evaluate("document.querySelector('#drawer').hidden = true")
            browser.run(INJECT_PICKER_JS, {"studies": PICKER_STUDIES})
            for failure in check_picker(browser.run(MEASURE_PICKER_JS, None), width, height, scroll):
                seen.setdefault(("{}x{}".format(width, height), "picker open", failure), []).append(scheme)
            for state in VIEWER_STATES:   # the viewer open over it all (P6-E), with one picture and with two
                cases += 1
                pair = state == "pair"
                where = ("{}x{}".format(width, height), "viewer {}".format(state))
                if not browser.run(INJECT_VIEWER_JS, {"pair": pair}):
                    seen.setdefault(where + ("the viewer did not open with its pictures",), []).append(scheme)
                    continue
                for failure in check_viewer(browser.run(MEASURE_VIEWER_JS, None), width, height, pair, scroll):
                    seen.setdefault(where + (failure,), []).append(scheme)
    for width, height, drawer in COMPOSER_CASES:   # the composer's rows where its column is narrowest (P6-A)
        busy_states = ((["busy"] if (width, height, drawer) in COMPOSER_BUSY_CASES else [])   # P6 fix 2
                       + (STRESS_STATES if (width, height, drawer) in COMPOSER_STRESS_CASES else []))   # P9-G5b fix 1
        for scheme in SCHEMES:
            serial += 1
            browser.open("{}?case={}".format(url, serial), width, height, scheme)
            for state in COMPOSER_STATES:
                cases += 1
                where = ("{}x{}".format(width, height), "drawer={} composer {}".format("open" if drawer else "closed", state))
                if not browser.run(INJECT_COMPOSER_JS, {"chips": CHIPS, "name": LONG_FILE, "state": state, "drawer": drawer}):
                    seen.setdefault(where + ("the page did not show the {}".format(state),), []).append(scheme)
                    continue
                for failure in check_composer(browser.run(MEASURE_COMPOSER_JS, {"keep": False}), width, height, state):
                    seen.setdefault(where + (failure,), []).append(scheme)
            for state in busy_states:   # a busy page is busy for good (an upload held, a chat made): each busy case has a page of its own
                serial += 1
                browser.open("{}?case={}".format(url, serial), width, height, scheme)
                cases += 1
                where = ("{}x{}".format(width, height), "drawer={} composer {}".format("open" if drawer else "closed", state))
                reason = enter_busy(browser, state, drawer)
                if reason is not None:
                    seen.setdefault(where + ("the page did not get to the busy composer: {}".format(reason),), []).append(scheme)
                    continue
                for failure in check_composer(browser.run(MEASURE_COMPOSER_JS, {"keep": True}), width, height, state):
                    seen.setdefault(where + (failure,), []).append(scheme)
    for width, height in VIEWER_TOUCH_VIEWPORTS:   # the viewer on a landscape phone, where it is stacked and scrolls by touch (P6 fix 1)
        for scheme in SCHEMES:
            serial += 1
            browser.open("{}?case={}".format(url, serial), width, height, scheme)
            for state in VIEWER_STATES:
                cases += 1
                pair = state == "pair"
                where = ("{}x{}".format(width, height), "viewer {}".format(state))
                if not browser.run(INJECT_VIEWER_JS, {"pair": pair}):
                    seen.setdefault(where + ("the viewer did not open with its pictures",), []).append(scheme)
                    continue
                for failure in check_viewer(browser.run(MEASURE_VIEWER_JS, None), width, height, pair, scroll=True):
                    seen.setdefault(where + (failure,), []).append(scheme)
    for width in TALL_WIDTHS:   # a composer taller than the viewport: the report's floor
        for scheme in SCHEMES:
            serial += 1
            cases += 1
            browser.open("{}?case={}".format(url, serial), width, TALL_HEIGHT, scheme)
            browser.run(INJECT_JS, {"chips": CHIPS, "word": LONG_WORD, "text": REPORT_TEXT})
            browser.run(TALL_JS, {"px": TALL_HEIGHT})
            for failure in check_tall(browser.run(MEASURE_TALL_JS, None), TALL_HEIGHT):
                seen.setdefault(("{}x{}".format(width, TALL_HEIGHT), "tall composer", failure), []).append(scheme)
    failures = ["{} {} {}: {}".format(viewport, "+".join(schemes), state, failure) for (viewport, state, failure), schemes in seen.items()]
    return cases, failures


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Geometry check for the chat page, in headless Chrome.")
    parser.add_argument("--url", help="check this running tiny app instead of starting one (its page URL, e.g. http://127.0.0.1:PORT/)")
    args = parser.parse_args(argv)
    if args.url:   # before anything is started or sent: the address, then the app's own answer about itself
        problem = url_problem(args.url) or tiny_only_problem(args.url)
        if problem:
            print("ERROR --url refused: " + problem)
            return 2
    exit_on_sigterm()   # a SIGTERM runs the finally below, which stops Chrome and the app
    chrome = find_chrome()
    if chrome is None:
        print("ERROR no Chrome found: set $CHROME to its executable")
        return 2
    app = None  # type: Optional[App]
    browser = None  # type: Optional[Browser]
    try:
        if args.url:
            url = args.url if args.url.endswith("/") else args.url + "/"
        else:
            app = App()
            url = app.url
        browser = Browser(chrome, label="chat_ui_layout_chrome")
        cases, failures = run_checks(browser, url)
    except (OSError, RuntimeError) as exc:   # ConnectionError is an OSError
        print("ERROR {}".format(exc))
        return 2
    finally:
        try:   # the app is stopped even when an interrupt (Ctrl-C, a SIGTERM's SystemExit) cuts the closing of the browser short
            if browser is not None:
                browser.close()
        finally:
            if app is not None:
                app.stop()
    for failure in failures:
        print("FAIL " + failure)
    print("RESULT " + json.dumps({"cases": cases, "failures": failures}))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
