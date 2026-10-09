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
    with the drawer closed or open. -> (cases, failures); a failure seen in both schemes is reported once."""
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
