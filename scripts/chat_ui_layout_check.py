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
     375 px and narrower a ~170-character report has Show all, and at 1024 px and wider no report that short has one
  j  (P6 fix 1, one more case for every viewport in every scheme) the test-split picker open, its list full: the page does not overflow
     horizontally, the button and the list are rendered inside the viewport's width, and the transcript's last card header, scrolled
     into view, is not covered (a point just inside its top hits the card, not the composer, the picker or the banner)

On the short viewports, where the page scrolls instead (667x375, 320x256), c and d give way to: the report keeps its
natural height (no scroller of its own, at least 120 px), the composer is not capped, the banner stays at the top of
the viewport while the page scrolls, and each button is hit-tested after it is scrolled into view.

The tall-composer cases (one per width and scheme, 568 px tall) put a composer taller than the viewport on the page and pin
the report's 120 px floor: #conversation keeps its 120 px, the composer stays inside the viewport and scrolls inside itself,
and Settings and Send are still reachable by scrolling it.

    venv/bin/python scripts/chat_ui_layout_check.py [--url http://127.0.0.1:8000/]

Prints one `FAIL ...` line per failure, then `RESULT {"cases": N, "failures": [...]}`. Exits 1 on any failure and 2 when
Chrome or the app cannot be started. Chrome is $CHROME, else /Applications/Google Chrome.app/..., else one on PATH.
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
from scripts.chat_ui_cdp import App, Browser, exit_on_sigterm, find_chrome  # noqa: E402

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
# P6 fix 1: the clamp's proof. Three lines of a matching report hold about 130 characters at 375 px and about 280 on a wide page.
MID_REPORT = ("Findings: The heart is normal in size. The lungs are clear, with no focal consolidation, pleural effusion or pneumothorax. "
              "Impression: No acute cardiopulmonary process.")   # 168 characters: over three lines at phone width, two lines on a wide page
SHORT_REPORT = "Findings: No acute process."
PROOF_CHARS = (150, 190)   # what check i takes for the ~170-character report
PICKER_STUDIES = 60        # check j's list: more studies than the list shows, so that it scrolls inside its own box


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
    with the drawer closed or open, or (one case each) its Save button and its card sections in both drawer states, and its picker open.
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
                for failure in check_sections(browser.run(MEASURE_SECTIONS_JS, {"sections": SECTIONS}), width, SECTIONS, drawer == "open"):
                    seen.setdefault(("{}x{}".format(width, height), "drawer={} card sections".format(drawer), failure), []).append(scheme)
            cases += 1   # the test-split picker open, with the drawer closed (P6 fix 1)
            browser.evaluate("document.querySelector('#drawer').hidden = true")
            browser.run(INJECT_PICKER_JS, {"studies": PICKER_STUDIES})
            for failure in check_picker(browser.run(MEASURE_PICKER_JS, None), width, height, scroll):
                seen.setdefault(("{}x{}".format(width, height), "picker open", failure), []).append(scheme)
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
    parser.add_argument("--url", help="check this running app instead of starting one (its page URL, e.g. "
                                      "http://127.0.0.1:8000/)")
    args = parser.parse_args(argv)
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
        if browser is not None:
            browser.close()
        if app is not None:
            app.stop()
    for failure in failures:
        print("FAIL " + failure)
    print("RESULT " + json.dumps({"cases": cases, "failures": failures}))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
