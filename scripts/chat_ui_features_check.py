"""CHAT_UI_PLAN.md P6-F: the five image features of the chat page, checked end to end in real Chrome on the tiny stack, with screenshots.
A local development tool; validate.sh does not run it.

It starts the app as the dev server is started, create_app(engine="tiny", tiny_gallery=True): random weights, a toy vocabulary, the
synthetic gallery (200 train images, 40 test studies, 14-label vectors) and the keyword labeller, on a free loopback port and in a
temporary home, or takes --url. (The serving command, `python -m app.server --engine tiny --gallery DIR --labeler rule`, cannot serve
the tiny gallery: its manifest names a tower that is not the tiny engine's, and the server refuses a gallery it cannot match to its
engine, which is why this script starts the app the way the dev server does.) It then drives headless Chrome over the DevTools protocol
(scripts/chat_ui_cdp.py) with real mouse and key events. Five items, one `CHECK <name> PASS|FAIL <detail>` line each:

  attach   click, drag-drop and paste each attach an image with a preview (its picture, name, pixel size and size on disk), a
           file dropped beside the well neither navigates nor attaches, and a refused file shows its message: the page's own (a GIF, a
           DICOM file by its name, a file over 20 MB: nothing is sent) and the server's (a 32 px picture, 422: the chat made for it
           is deleted again, and the image stays attached)
  reload   a turn's card shows the X-ray and the 224x224 image the model saw, each loaded, with the sizes they have; the same after a
           reload of the page, and after leaving the chat and reopening it from the sidebar (the user turn then shows the server's
           thumbnail in place of the page's own preview)
  similar  the similar-X-rays grid of a card: for each neighbour its stored picture, "#rank . similarity" and a bar as long as the
           similarity, and "n/14 labels agree" with a chip for each positive only one side has; all of it equal to what the server stored
           for the turn, and the n recounted here from the 14 labels of the neighbour and of the report
  matches  the matching-reports list of a card (rank, similarity, how many reports read the same, the text clamped to three lines, and
           the positive labels; Show all is held to its rule on every report: it is offered exactly when the text overflows the three
           lines) and, for a test study picked in the composer's list, the badge that says where its own report ranks among the test
           reports; an upload has no such badge. The reports of the tiny gallery are 10 to 19 words (168 characters at most), so none
           overflows its three lines at the width checked (a 1280 px page) and Show all is never offered here: the item clicks Show all
           and Show less only on a report that overflows, which none does, and its facts say so (show_all). The click path (Show all
           opens the whole text, Show less closes it, the focus comes back) is covered by tests/frontend/render.test.mjs, not by a real
           Chrome
  viewer   the viewer opens from every picture of a card (the two of the images row and each similar X-ray, which opens beside the
           turn's own X-ray) and from the upload in the user's own message (P6-G), which opens what the card's "Your X-ray" tile
           opens: the same picture, name and footer. Each closes to the control that opened it, and on a side-by-side pair zoom
           (wheel, + and -, the buttons, 1:1), pan (drag, the arrow keys), brightness and contrast (a click on the slider, the arrow
           keys on it) work, by mouse and by keyboard, each measured on what is drawn (the picture's box and the screenshot's pixels)
           and not on the page's own bookkeeping

Input that is real: a click on the well (the file chooser Chrome would open is intercepted and answered with a file, which a headless Chrome
cannot do for itself), a file dropped from disk (Input.dispatchDragEvent), the mouse (clicks, drags and the wheel) and the keyboard.
Input that is not: the paste. A headless Chrome has no clipboard of its own to paste from, and a real Cmd+V would read the user's, so the
page's paste handler is given a ClipboardEvent made here that holds a File (or text). The check says so in its checklist.

Every picture of the page that is not a screenshot is synthetic: the phantoms drawn here (phantom_png) and the tiny gallery's gray JPEGs.
Nothing is MIMIC-derived. No check passes while the page logged a console error or threw. Screenshots, each confirmed by the page itself
(a DOM rule per picture, true before and after the capture), go to docs/chat_ui/evidence/p6f/ with checklist.json (the result, the
environment and, for each item, its name, its status and the facts it measured); --no-evidence runs without writing anything.

    venv/bin/python scripts/chat_ui_features_check.py [--url http://127.0.0.1:PORT/ --no-evidence] [--out DIR] [--no-evidence] [--timeout S]

Prints `RESULT {"checks": 5, "failures": [...]}` last. Exits 1 on any failed item and 2 when Chrome or the app cannot be started, when --url
is refused, or when the run overruns --timeout. A --url is an app that someone else started, so it is taken only with --no-evidence (the
pictures and the checklist are evidence of the tiny stack that the check starts itself), and only when it is a tiny app on this machine's
loopback with the gallery: scripts/chat_ui_cdp.py refuses (url_problem, tiny_only_problem) the dev server's port and the tunnel's, any other
host, and any app whose /v1/models is not the tiny engine with no checkpoint, with one fixed line and no word of what the app said, before
Chrome starts or anything is sent. Every process it starts is killed on exit.
"""
import argparse
import base64
import datetime
import io
import json
import math
import os
import platform
import re
import shutil
import signal
import sys
import tempfile
import time
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from app.imaging import TOO_SMALL_MSG  # noqa: E402
from app.labels import CHEXBERT_14  # noqa: E402
from scripts.chat_ui_browser_check import CARD_JS, Failure, Overrun, expect, run_check, warm_up  # noqa: E402
from scripts.chat_ui_cdp import (App, Browser, exit_on_sigterm, find_chrome, tiny_only_problem, url_needs_no_evidence,  # noqa: E402
                                 url_problem)

EVIDENCE_DIR = os.path.join(REPO_ROOT, "docs", "chat_ui", "evidence", "p6f")
STEP_DELAY_S = 0.02            # between two decoding steps of the tiny engine
DESKTOP = (1280, 900)
TALL = (1280, 2300)            # tall enough for a whole card above the composer: the images row, the similar X-rays, the matching reports
MAX_SHOT_BYTES = 400 * 1024    # the limit of the committed pictures (tests/e2e/conftest.py MAX_SHOT_BYTES)

# What the page says, as the page says it (the tests read the page's sources for every one of them).
NOT_AN_IMAGE = "Use a PNG, JPEG or WEBP image."
DICOM = "DICOM is not supported. Export the image as PNG or JPEG first."
TOO_BIG = "Image is larger than 20 MB."
TOO_SMALL = TOO_SMALL_MSG      # the server's own refusal of a picture under 64 px a side, shown by the page as it is
TITLES = {"neighbors": "Similar X-rays (13D tower)", "matches": "Matching reports (13D image and text encoders)"}
MODEL_SAW = "What the model saw (224×224)"
# viewer.js: one press of + or -, one arrow key, the bounds of the zoom.
ZOOM_STEP = 1.25
PAN_STEP = 48
MIN_SCALE = 0.25
MAX_SCALE = 16
# What the script drives in the page: each must still be in the page's sources (tests/test_chat_ui_features_check.py).
PAGE_HOOKS = ("image-well", "preview", "preview-name", "dragging", "pick-study", "picker-list", "new-session", "session-list",
              "picture-button", "thumb-button", "data-key", "image-tile", "neighbor-score", "sim-bar", "agreement", "own-rank", "clamp-text",
              "match-score", "viewer-stage", "viewer-pane", "viewer-open", "data-viewer", "data-scale")
USER_THUMB = "user:thumb"   # the thumbnail in the user's own message (P6-G): no data-key names it, so the viewer item does

EVIDENCE_FILES = ["attach_preview_1280x900_light.png", "attach_refused_1280x900_light.png", "reopened_images_1280x900_light.png",
                  "similar_xrays_1280x900_light.png", "matches_1280x900_light.png", "own_rank_1280x900_light.png",
                  "card_1280x2300_light.png", "card_1280x2300_dark.png", "viewer_side_by_side_1280x900_light.png",
                  "viewer_adjusted_1280x900_light.png", "viewer_side_by_side_1280x900_dark.png",
                  "checklist.json"]   # what a full run writes, and the only files it replaces


# ---- pure parts: arguments, the checklist -----------------------------------------------------------------------------------------
# (what a --url may be, and the refusal of one that is not, are scripts/chat_ui_cdp.py's: url_problem, tiny_only_problem, url_needs_no_evidence)

def missing_features(models: Dict[str, Any]) -> Optional[str]:
    """What the app lacks for these features, said for /v1/models, or None: a private-mode app (the test-split picker and the neighbours'
    pictures are private), with a gallery and a labeller."""
    features = models.get("features") if isinstance(models.get("features"), dict) else {}
    lacking = []
    if models.get("mode") != "private":
        lacking.append("it is not in private mode (the test-split picker and the gallery pictures are private)")
    if not features.get("retrieval"):
        lacking.append("it has no retrieval gallery")
    if not features.get("labels"):
        lacking.append("it has no labeller")
    return "the app cannot show these features: " + "; ".join(lacking) + " (start it with tiny_gallery=True)" if lacking else None


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="The five image features of the chat page, checked in headless Chrome on the tiny stack.")
    parser.add_argument("--url", help="check this running tiny app with a gallery instead of starting one (http://127.0.0.1:PORT/); needs --no-evidence")
    parser.add_argument("--out", default=EVIDENCE_DIR, help="where screenshots and checklist.json go (default: docs/chat_ui/evidence/p6f)")
    parser.add_argument("--no-evidence", action="store_true", help="run the checks and write no screenshots and no checklist.json")
    parser.add_argument("--timeout", type=int, default=420, help="seconds the whole run may take (default 420)")
    args = parser.parse_args(argv)
    url_needs_no_evidence(parser, args)   # an app that someone else started is not what the evidence is of
    return args


def clear_evidence(out_dir: str) -> None:
    """Removes the files a full run writes (EVIDENCE_FILES), so that a picture of an earlier run cannot pass for this run's, and nothing else."""
    for name in EVIDENCE_FILES:
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            os.remove(path)


def write_checklist(out_dir: str, result: Dict[str, Any], environment: Dict[str, Any], checks: List[Dict[str, Any]],
                    screenshots: List[Dict[str, Any]]) -> str:
    """checklist.json: the result, the environment, one entry per item (its name, status and measured facts) and the screenshots; -> its path."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "checklist.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"result": result, "environment": environment, "checks": checks, "screenshots": screenshots}, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return path


# ---- pure parts: what the page is expected to show, as it builds it (render.js) ------------------------------------------------------

def score_text(rank: int, similarity: float) -> str:
    """"#3 · 0.913": render.js scoreText."""
    return "#{} · {:.3f}".format(rank, similarity)


def bar_width(similarity: float) -> str:
    """The bar's width as render.js simBar sets it: the similarity held to 0..1, in per cent, rounded half up."""
    return "{}%".format(int(math.floor(min(1.0, max(0.0, similarity)) * 100 + 0.5)))


def agreement_text(agree: int, of: int) -> str:
    return "{}/{} labels agree".format(agree, of)


def own_rank_text(rank: Dict[str, Any]) -> str:
    """The badge of a test study, render.js ownRankBadge: "Own report: rank 3 of 2,663 test reports · R@10 hit"."""
    hit = " · R@10 hit" if rank.get("hit_at_10") is True else " · R@10 miss" if rank.get("hit_at_10") is False else ""
    return "Own report: rank {:,} of {:,} test reports{}".format(rank["rank"], rank["of"], hit)


def positives(labels: Optional[Dict[str, Any]]) -> List[str]:
    """The names of the labels that are on, in the order the chips list them."""
    return [name for name in CHEXBERT_14 if isinstance(labels, dict) and labels.get(name)]


def show_all_facts(pages: List[Dict[str, Any]], clicked: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """What the matches item states of Show all, from the lists it read (MATCHES_JS answers: the upload's first, where a click is made, then the
    picked study's): the rule it held every report to (the button is offered exactly when its text overflows the three lines), how many reports
    that was, how many overflowed and how many had the button, the width of each list, and whether Show all and Show less were clicked. They are
    clicked only on a report of the upload's list that overflows; where none does, nothing was offered to click, and the facts say so."""
    items = [item for page in pages for item in page["items"]]
    return {"rule": "a report's Show all is offered exactly when its text overflows its three lines (held to it on every report read)",
            "reports_checked": len(items), "reports_that_overflow": sum(1 for i in items if i["overflows"]),
            "reports_with_the_button": sum(1 for i in items if i["more"] == "Show all"), "list_width_px": [page["width"] for page in pages],
            "clicked_show_all_and_show_less": clicked is not None, "clicked": clicked,
            "not_clicked_because": None if clicked is not None else
            "no report of the upload's list overflows its three lines at this width, so no Show all was offered there to click"}


def wheel_factor(delta_y: float, delta_mode: int = 0) -> float:
    """viewer.js wheelFactor: the zoom of one turn of the wheel, never more than double or half."""
    pixels = delta_y * (16 if delta_mode == 1 else 400 if delta_mode == 2 else 1)
    return min(2.0, max(0.5, math.exp(-pixels * 0.002)))


def expected_viewer(key: str) -> Dict[str, Any]:
    """What the viewer must say when it is opened from this picture: its name, the footer of each pane (the tiny gallery's pictures are 320 px, the
    model's input 224 px) and the variant of the first pane's picture. The keys are the data-keys of the card's pictures (image:thumb,
    image:model_input, neighbor:<rank>) and USER_THUMB, the upload in the user's own message, which opens exactly what the card's "Your X-ray" tile
    opens (P6-G). A key it does not know is a ValueError, not the neighbour's words."""
    if key == USER_THUMB or key == "image:thumb":
        return {"label": "Image viewer: Your X-ray", "footers": ["Your X-ray · 320×320 px"], "variant": "original"}
    if key == "image:model_input":
        return {"label": "Image viewer: What the model saw", "footers": ["What the model saw · 224×224 px"], "variant": "model_input"}
    found = re.fullmatch(r"neighbor:(\d+)", key) if isinstance(key, str) else None
    if not found:
        raise ValueError("no picture of the card is called {!r}".format(key))
    rank = found.group(1)
    return {"label": "Image viewer: Your X-ray beside Similar X-ray #{}".format(rank), "variant": "original",
            "footers": ["Your X-ray · 320×320 px", "Similar X-ray #{} · 320×320 px (stored gallery size)".format(rank)]}


def variant_of(path: Any) -> Optional[str]:
    """The variant ("original", "thumb", "model_input") of a picture the server serves for a message, from the path the page shows
    (/v1/messages/<id>/image?variant=<variant>); None for anything else: a blob, a gallery path, a file of this machine. A fact names the variant
    and never a path that is only this machine's."""
    found = re.fullmatch(r"/v1/messages/[\w-]+/image\?variant=(\w+)", path) if isinstance(path, str) else None
    return found.group(1) if found else None


def stages_of(events: Any) -> Dict[str, Any]:
    """A stored turn (GET /v1/messages/{id}) as {"start": its message_start data, "stop": its message_stop data, "stages": {stage: stage_end data}}."""
    turn = {"start": None, "stop": None, "stages": {}}  # type: Dict[str, Any]
    for row in events if isinstance(events, list) else []:
        data = row.get("data") if isinstance(row, dict) else None
        if not isinstance(data, dict):
            continue
        if row.get("event") == "message_start":
            turn["start"] = data
        elif row.get("event") == "message_stop":
            turn["stop"] = data
        elif row.get("event") == "stage_end" and isinstance(data.get("stage"), str):
            turn["stages"][data["stage"]] = data
    return turn


def luma_stats(png: bytes) -> Tuple[float, float]:
    """The mean and the standard deviation of a picture's luma (0..255): what brightness and contrast change."""
    pixels = np.asarray(Image.open(io.BytesIO(png)).convert("L"), dtype=float)
    return float(pixels.mean()), float(pixels.std())


def phantom_png(w: int, h: int) -> bytes:
    """A chest-like phantom, 8-bit gray: a soft rounded torso, two darker lung fields with ribs across them, a heart shadow, clavicles, a spine
    with its ribbing, the domes of the diaphragm and a little grain (the same grain for a size every time). It has structure at every scale a
    viewer shows, and no pixel of it is from a patient."""
    y, x = np.mgrid[0:h, 0:w].astype(float)
    u, v = (x - w / 2) / (w / 2), (y - h / 2) / (h / 2)

    def blob(cx: float, cy: float, rx: float, ry: float, power: float = 2.0) -> Any:
        return np.exp(-((np.abs(u - cx) / rx) ** power + (np.abs(v - cy) / ry) ** power))

    body = blob(0.0, 0.05, 0.86, 0.98, 4.0)
    lungs = np.minimum(blob(-0.40, -0.10, 0.30, 0.56) + blob(0.40, -0.10, 0.30, 0.56), 1.0)
    heart = blob(0.14, 0.26, 0.22, 0.26)
    spine = blob(0.0, 0.0, 0.05, 1.2) * (0.75 + 0.25 * np.cos(v * 38))
    ribs = (0.5 + 0.5 * np.cos(v * 26 - np.abs(u) * 11)) ** 8 * lungs                      # down and out from the spine
    clavicles = np.exp(-(((v + 0.60) + 0.28 * np.abs(u)) / 0.035) ** 2) * (np.abs(u) < 0.72)   # up and out from the breastbone
    diaphragm = 1.0 / (1.0 + np.exp(-(v - (0.66 - 0.30 * u ** 2)) / 0.035))
    pixels = 20 + 125 * body - 80 * lungs * body + 50 * heart * body + 45 * spine * body + 38 * ribs * body \
        + 35 * clavicles * body + 35 * diaphragm * body
    pixels = pixels + np.random.default_rng(w * 1000 + h).normal(0.0, 3.0, pixels.shape)
    buffer = io.BytesIO()
    Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8)).save(buffer, "PNG")
    return buffer.getvalue()


def close_to(a: float, b: float, tolerance: float) -> bool:
    return abs(a - b) <= tolerance


# ---- the page, as scripts: each is a function that the browser calls ---------------------------------------------------------------------

# The composer's preview of an attached file (composer.js): the picture, the name, the sizes, the button that removes it, and whether the well
# shows that something is being dragged over it.
PREVIEW_JS = """() => {
  const p = document.querySelector('#preview'), w = document.querySelector('#image-well');
  const img = p.querySelector(':scope > img'), meta = p.querySelector(':scope > span'), x = p.querySelector(':scope > button');
  const name = meta && meta.querySelector('.preview-name'), size = meta && meta.querySelector('.preview-size');
  const a = document.activeElement;
  return { shown: !p.hidden, name: name ? name.textContent : null, sizes: size ? size.textContent : null,
           title: meta ? meta.getAttribute('title') : null, alt: img ? img.getAttribute('alt') : null,
           blob: !!img && (img.getAttribute('src') || '').startsWith('blob:'), loaded: !!img && img.complete && img.naturalWidth > 0,
           natural: img ? [img.naturalWidth, img.naturalHeight] : null, remove: x ? x.getAttribute('aria-label') : null,
           dragging: w.classList.contains('dragging'), href: location.href, hash: location.hash,
           focus: a && a !== document.body ? (a.id || a.localName) : 'body' };
}"""

NOTICE_JS = """() => {
  const n = document.querySelector('#notice'), p = n.querySelector('p');
  return { shown: !n.hidden, role: n.getAttribute('role'), text: p ? p.textContent : '' };
}"""

# A file dropped on (or pasted into) an element is made here for paste: a DataTransfer that holds a File (an image) or text.
PASTE_JS = """async ({ b64, name, type, text, target }) => {
  const data = new DataTransfer();
  if (b64) {
    const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    data.items.add(new File([bytes], name, { type }));
  } else {
    data.setData('text/plain', text);
  }
  const event = new ClipboardEvent('paste', { bubbles: true, cancelable: true, clipboardData: data });
  document.querySelector(target).dispatchEvent(event);
  return { taken: event.defaultPrevented };
}"""

# What a card's pictures are: the user turn's thumbnail and the images row's two, each with its server path (data-src; none for the page's own
# preview), whether its source is an object URL and has loaded, its natural size and the box it is drawn in, and what its pixels are (the mean and
# the spread of their luma, read back from a canvas: a picture that is drawn and flat would not be an X-ray).
PICTURES_JS = """() => {
  const u = [...document.querySelectorAll('#conversation article.turn.user')].pop();
  const c = [...document.querySelectorAll('#conversation article.card')].pop();
  const stats = (i) => {
    try {
      const k = document.createElement('canvas');
      k.width = i.naturalWidth; k.height = i.naturalHeight;
      const g = k.getContext('2d');
      g.drawImage(i, 0, 0);
      const d = g.getImageData(0, 0, k.width, k.height).data;
      let s = 0, q = 0;
      const n = d.length / 4;
      for (let j = 0; j < d.length; j += 4) { const y = 0.299 * d[j] + 0.587 * d[j + 1] + 0.114 * d[j + 2]; s += y; q += y * y; }
      const m = s / n;
      return { mean: m, std: Math.sqrt(Math.max(0, q / n - m * m)) };
    } catch (e) { return null; }
  };
  const look = (i) => { if (!i) return null; const r = i.getBoundingClientRect();
    return { path: i.getAttribute('data-src'), blob: (i.getAttribute('src') || '').startsWith('blob:'), alt: i.getAttribute('alt'),
             loaded: i.complete && i.naturalWidth > 0, natural: [i.naturalWidth, i.naturalHeight], box: [r.left, r.top, r.width, r.height],
             pixels: i.complete && i.naturalWidth > 0 ? stats(i) : null }; };
  const card = c ? c.getBoundingClientRect() : null;
  return { bubble: u ? look(u.querySelector('img.thumb')) : null,
           card: card ? [card.left, card.top, card.width, card.height] : null,
           tiles: c ? [...c.querySelectorAll('section.images figure.image-tile')].map((f) => ({ caption: f.querySelector('figcaption').textContent,
                                                                                               img: look(f.querySelector('img')) })) : [] };
}"""

# The newest card's grid of similar X-rays, as the user reads it.
NEIGHBOURS_JS = """() => {
  const c = [...document.querySelectorAll('#conversation article.card')].pop();
  const s = c && c.querySelector('section.neighbors');
  if (!s) return null;
  const note = s.querySelector('.section-note');
  return { title: s.querySelector('h3').textContent, note: note ? note.textContent : null,
           items: [...s.querySelectorAll('li.neighbor')].map((li) => {
             const img = li.querySelector('img'), fill = li.querySelector('.sim-fill'), agree = li.querySelector('.agreement');
             const bar = li.querySelector('.sim-bar'), score = li.querySelector('.neighbor-score');
             const box = (e) => { if (!e) return null; const r = e.getBoundingClientRect(); return [r.left, r.top, r.width, r.height]; };
             return { rank: li.getAttribute('data-rank'), score: score ? score.textContent : null, scoreBox: box(score), bar: fill ? fill.style.width : null,
                      barBox: box(bar), placeholder: !!li.querySelector('.neighbor-placeholder'),
                      img: img ? { path: img.getAttribute('data-src'), loaded: img.complete && img.naturalWidth > 0,
                                   natural: [img.naturalWidth, img.naturalHeight], box: box(img) } : null,
                      agreement: agree ? agree.textContent : null, agreementBox: box(agree),
                      chips: [...li.querySelectorAll('.chip.diff')].map((x) => x.textContent) };
           }) };
}"""

# The newest card's matching reports: the title, the width of the list, the own-report badge (its text and its tooltip), and per report its rank and
# score, how many reports read the same, its text, whether the text is clamped and overflows its three lines, its Show all (when shown) and its label chips.
MATCHES_JS = """() => {
  const c = [...document.querySelectorAll('#conversation article.card')].pop();
  const s = c && c.querySelector('section.matches');
  if (!s) return null;
  const badge = s.querySelector('.own-rank');
  return { title: s.querySelector('h3').textContent, width: Math.round(s.getBoundingClientRect().width),
           badge: badge ? { text: badge.childNodes[0].textContent, title: badge.getAttribute('title'), shown: badge.getClientRects().length > 0 } : null,
           items: [...s.querySelectorAll('li.match')].map((li) => {
             const text = li.querySelector('.clamp-text'), more = li.querySelector('button[data-action="show"]'), size = li.querySelector('.group-size');
             const shown = !!more && !more.closest('[hidden]') && more.getClientRects().length > 0;
             return { rank: li.getAttribute('data-rank'), score: li.querySelector('.match-score').textContent, size: size ? size.textContent : null,
                      text: text ? text.textContent : null, clamped: !!text && text.classList.contains('clamped'), open: !!text && text.classList.contains('open'),
                      overflows: !!text && text.scrollHeight > text.clientHeight + 1, more: shown ? more.textContent : null,
                      expanded: more ? more.getAttribute('aria-expanded') : null,
                      chips: [...li.querySelectorAll('.chip.finding')].map((x) => x.textContent) };
           }) };
}"""

PICKER_JS = """() => {
  const p = document.querySelector('#picker'), b = document.querySelector('#pick-study');
  return { button: b ? { hidden: b.hidden, expanded: b.getAttribute('aria-expanded'), disabled: b.disabled } : null, open: !!p && !p.hidden,
           status: document.querySelector('#picker-status') ? document.querySelector('#picker-status').textContent : null,
           studies: [...document.querySelectorAll('#picker-list button')].map((x) => ({ row: x.getAttribute('data-test-row'), text: x.textContent })) };
}"""

# The newest user turn: its picture's alt text and the chips of the settings it used.
BUBBLE_JS = """() => {
  const u = [...document.querySelectorAll('#conversation article.turn.user')].pop();
  if (!u) return null;
  const img = u.querySelector('img.thumb');
  return { image: img ? img.getAttribute('alt') : null, chips: [...u.querySelectorAll('.options .chip')].map((c) => c.textContent),
           thumbIsOpener: !!(img && img.closest('button')) };
}"""

# The pictures of the newest card that open the viewer, in the order of the page: each by its data-key (image:thumb, image:model_input,
# neighbor:<rank>).
OPENERS_JS = """() => {
  const c = [...document.querySelectorAll('#conversation article.card')].pop();
  return c ? [...c.querySelectorAll('button.picture-button')].map((b) => ({ key: b.getAttribute('data-key'), label: b.getAttribute('aria-label') })) : [];
}"""

# The viewer as the user sees it: whether it is open, what it says it shows, its panes (footer, source, loaded, the box the picture is drawn in,
# the box of its stage, the zoom the page says and the filter on the picture), the sliders, the page behind it, and what has the focus.
VIEWER_JS = """() => {
  const v = document.querySelector('#viewer'), a = document.activeElement;
  const slider = (n) => { const i = v.querySelector('input[data-viewer="' + n + '"]'); return i ? { value: Number(i.value), text: i.getAttribute('aria-valuetext') } : null; };
  const link = v.querySelector('input[data-viewer="link"]');
  return { hidden: v.hidden, label: v.getAttribute('aria-label'), held: document.body.classList.contains('viewer-open'),
           inert: ['.banner', '.layout', '#composer'].map((s) => document.querySelector(s).hasAttribute('inert')),
           panes: [...v.querySelectorAll('.viewer-pane')].map((p) => {
             const img = p.querySelector('img'), st = p.querySelector('.viewer-stage'), r = img.getBoundingClientRect(), s = st.getBoundingClientRect();
             return { side: p.getAttribute('data-side'), footer: p.querySelector('figcaption').textContent, alt: img.getAttribute('alt'),
                      blob: (img.getAttribute('src') || '').startsWith('blob:'), path: img.getAttribute('data-src'),
                      loaded: img.complete && img.naturalWidth > 0, natural: [img.naturalWidth, img.naturalHeight],
                      scale: Number(st.getAttribute('data-scale')), box: [r.left, r.top, r.width, r.height], stage: [s.left, s.top, s.width, s.height],
                      filter: img.style.filter }; }),
           brightness: slider('brightness'), contrast: slider('contrast'), linked: link ? link.checked : null,
           focus: a && a !== document.body ? (a.getAttribute('data-key') || a.getAttribute('data-viewer') || a.getAttribute('class') || a.id || a.localName) : 'body' };
}"""
# A pane is loaded once its picture has decoded and the page has had its load event (its footer then gives the pixel size): a picture can be complete a task
# before its footer says so.
VIEWER_LOADED = ("(() => { const v = document.querySelector('#viewer'); const p = [...v.querySelectorAll('.viewer-pane')];"
                 " return !v.hidden && p.length > 0 && p.every((x) => { const i = x.querySelector('img');"
                 " return i.complete && i.naturalWidth > 0 && x.querySelector('figcaption').textContent.indexOf(' px') > 0; }); })()")

# What each screenshot must show when it is taken, asked of the page itself: a file named "viewer_adjusted" is not a picture of a viewer at its fit.
# Each is the body of a function of no arguments that returns true or false.
SHOT_RULES = {
    "attach_preview": "const p = document.querySelector('#preview'), i = p.querySelector('img'); return !p.hidden && !!i && i.complete && i.naturalWidth > 0 "
                      "&& p.textContent.indexOf('.png') > 0 && !document.querySelector('#image-well').classList.contains('dragging')",
    "attach_refused": "const n = document.querySelector('#notice'); return !n.hidden && n.getAttribute('role') === 'alert' "
                      "&& n.textContent.indexOf('Use a PNG, JPEG or WEBP image.') >= 0 && !document.querySelector('#preview').hidden",
    "reopened_images": "const c = [...document.querySelectorAll('#conversation article.card')].pop(); if (!c) return false; "
                       "const i = [...c.querySelectorAll('section.images img')]; const r = c.querySelector('section.images').getBoundingClientRect(); "
                       "return c.getAttribute('data-status') === 'done' && i.length === 2 && i.every((x) => x.complete && x.naturalWidth > 0) "
                       "&& r.top >= 0 && r.bottom <= innerHeight",
    "similar_xrays": "const s = [...document.querySelectorAll('#conversation article.card')].pop().querySelector('section.neighbors'); "
                     "const r = s.getBoundingClientRect(); const i = [...s.querySelectorAll('img')]; "
                     "return i.length === 4 && i.every((x) => x.complete && x.naturalWidth > 0) && r.top >= 0 && r.bottom <= innerHeight "
                     "&& [...s.querySelectorAll('.agreement')].every((a) => /^[0-9]+\\/14 labels agree$/.test(a.textContent))",
    "matches": "const s = [...document.querySelectorAll('#conversation article.card')].pop().querySelector('section.matches'); const r = s.getBoundingClientRect(); "
               "return s.querySelectorAll('li.match').length === 3 && !s.querySelector('.own-rank') && r.top >= 0 && r.bottom <= innerHeight",
    "own_rank": "const s = [...document.querySelectorAll('#conversation article.card')].pop().querySelector('section.matches'); const b = s.querySelector('.own-rank'); "
                "const r = s.getBoundingClientRect(); return !!b && b.getClientRects().length > 0 && b.textContent.indexOf('Own report: rank') === 0 "
                "&& s.querySelectorAll('li.match').length === 3 && r.top >= 0 && r.bottom <= innerHeight",
    "card": "const c = [...document.querySelectorAll('#conversation article.card')].pop(); const r = c.getBoundingClientRect(); "
            "const k = document.querySelector('#composer').getBoundingClientRect(); "
            "return c.getAttribute('data-status') === 'done' && r.top >= 0 && r.bottom <= k.top && !!c.querySelector('section.matches .own-rank') "
            "&& c.querySelectorAll('section.neighbors img, section.images img').length === 6 "
            "&& [...c.querySelectorAll('img')].every((x) => x.complete && x.naturalWidth > 0)",
    "viewer_side_by_side": "const v = document.querySelector('#viewer'); const p = [...v.querySelectorAll('.viewer-pane')]; "
                           "return !v.hidden && p.length === 2 && p.every((x) => { const i = x.querySelector('img'); return i.complete && i.naturalWidth > 0; }) "
                           "&& p.every((x) => x.querySelector('.viewer-stage').getAttribute('data-scale') === '1') "
                           "&& v.querySelector('input[data-viewer=\"brightness\"]').value === '100' && v.querySelector('input[data-viewer=\"contrast\"]').value === '100'",
    "viewer_adjusted": "const v = document.querySelector('#viewer'); const p = [...v.querySelectorAll('.viewer-pane')]; "
                       "return !v.hidden && p.length === 2 && p.every((x) => Number(x.querySelector('.viewer-stage').getAttribute('data-scale')) > 1) "
                       "&& p.every((x) => x.querySelector('img').style.filter.indexOf('brightness(100%) contrast(100%)') < 0)",
}


# ---- the run -------------------------------------------------------------------------------------------------------------------------

class Context:
    """What the items share: the browser, the app, the pictures and the screenshots. An item makes the chats it needs itself, so that one that
    fails does not take the next with it."""

    def __init__(self, browser: Browser, base: str, work: str, out_dir: Optional[str]) -> None:
        self.browser, self.base, self.work, self.out_dir = browser, base, work, out_dir
        self.serial = 0
        self.shots = []  # type: List[Dict[str, Any]]
        self.console = []  # type: List[str]
        self.gallery = None  # type: Optional[Dict[str, Any]]   # the retrieve stage's own account of the gallery it searched, as the first turn gave it
        self.images = {}  # type: Dict[str, str]
        self.sizes = {"a": (320, 320), "b": (288, 256), "c": (256, 224)}
        for name, (w, h) in self.sizes.items():
            self.images[name] = self._write("xray_{}.png".format(name), phantom_png(w, h))
        gif = io.BytesIO()
        Image.new("L", (128, 128), 128).save(gif, "GIF")
        small = io.BytesIO()
        Image.fromarray(np.full((32, 32), 90, dtype=np.uint8)).save(small, "PNG")
        self.images["gif"] = self._write("scan.gif", gif.getvalue())
        self.images["dcm"] = self._write("scan.dcm", b"\0" * 128 + b"DICM" + b"\0" * 64)   # a DICOM preamble and magic: the page refuses it by its name
        self.images["huge"] = self._write("huge.png", phantom_png(64, 64) + b"\0" * (21 * 1024 * 1024))   # over the 20 MB the page takes
        self.images["small"] = self._write("small.png", small.getvalue())   # 32 px: the page takes it, the server refuses it

    def _write(self, name: str, data: bytes) -> str:
        path = os.path.join(self.work, name)
        with open(path, "wb") as handle:
            handle.write(data)
        return path

    # -- the page
    def open(self, hash_: str = "", viewport: Tuple[int, int] = DESKTOP, scheme: str = "light") -> None:
        """A fresh load of the app, at this viewport and colour scheme. The query is unique, so that a stale page cannot pass."""
        b = self.browser
        self.serial += 1
        b.set_viewport(viewport[0], viewport[1], scheme)
        b.call("Page.navigate", url="{}?run={}{}".format(self.base, self.serial, hash_))
        b.wait_for("location.search === '?run=%d' && document.readyState === 'complete' && !!document.querySelector('#exports')" % self.serial,
                   what="the page to load")
        b.wait_for("document.querySelector('#mode-badge').textContent !== '' && location.hash !== ''", what="the app to start")
        # The app has started once it knows its server's features: the badge and the hash come with the page, the models a request later, and until then
        # the test-split button is hidden (the check runs on a server that has the gallery: main refuses any other).
        b.wait_for("!document.querySelector('#pick-study').hidden", what="the app to know its server's features")

    def reload(self) -> None:
        self.browser.call("Page.reload")
        self.browser.wait_for("document.readyState === 'complete' && !!document.querySelector('#exports')", what="the page to reload")

    def card(self) -> Optional[Dict[str, Any]]:
        return self.browser.evaluate("(%s)()" % CARD_JS)

    def stuck(self, what: str, timeout: float) -> Failure:
        """What a wait that ran out says: not only that it did, but where the newest card and the server's view of its turn were."""
        try:
            card = self.card()
            seen = "no card" if card is None else "the newest card is {} with {} picture(s) not loaded, stages {}".format(card["status"], card["pending"], card["stages"])
            page = self.browser.evaluate("({ send_disabled: document.querySelector('#send').disabled, stop_hidden: document.querySelector('#stop').hidden,"
                                         " notice: document.querySelector('#notice').hidden ? null : document.querySelector('#notice').textContent,"
                                         " hash: location.hash, turns: document.querySelectorAll('#conversation > *').length,"
                                         " preview: document.querySelector('#preview').hidden ? null : document.querySelector('#preview').textContent.slice(0, 40) })")
            sent = {p["requestId"]: "{} {}".format(p["request"]["method"], p["request"]["url"].split("?")[0].rsplit(":", 1)[-1]) for p in
                    self.browser.events("Network.requestWillBeSent", clear=False)}
            failed = [(sent.get(p["requestId"]), p.get("errorText"), p.get("canceled")) for p in self.browser.events("Network.loadingFailed", clear=False)][-4:]
            refused = [(sent.get(p["requestId"]), p["response"]["status"]) for p in self.browser.events("Network.responseReceived", clear=False)
                       if p["response"]["status"] >= 400][-4:]
            seen += "; the page: {}; the last requests that failed: {}; answered with an error: {}; the last 6 requests: {}".format(
                page, failed, refused, list(sent.values())[-6:])
            server = ""
            if card is not None and card["id"]:
                with urllib.request.urlopen("{}v1/messages/{}?after=0".format(self.base, card["id"]), timeout=10) as response:
                    stored = json.load(response)
                server = "; the server has the turn {} after {} event(s), the last {}".format(stored["status"], len(stored["events"]), stored["events"][-1]["event"] if stored["events"] else None)
        except Exception as exc:   # the page or the app is gone: that is the answer
            seen, server = "the page or the app could not be asked ({})".format(type(exc).__name__), ""
        return Failure("waited {:g} s for {}: {}{}".format(timeout, what, seen, server))

    def wait_card(self, status: Optional[str] = "done", cards: Optional[int] = None, timeout: float = 30.0, message: Optional[str] = None) -> Dict[str, Any]:
        """The newest card, once it has this status, its pictures have loaded (they arrive after the card) and, given, there are this many cards."""
        cond = "(() => { const c = (%s)(); return c && c.pending === 0 && %s%s ? c : null; })()" % (
            CARD_JS, "c.status === %s" % json.dumps(status) if status else "true", " && c.cards === %d" % cards if cards else "")
        what = message or "a card" + (" that is " + status if status else "")
        try:
            return self.browser.wait_for(cond, timeout, what)
        except TimeoutError:
            raise self.stuck(what, timeout)

    def session_id(self) -> str:
        return self.browser.evaluate("location.hash.replace('#/s/', '')")

    def attach(self, name: str) -> None:
        """Choose the file as the page's own input takes it (DOM.setFileInputFiles), and wait for the composer to show it."""
        self.browser.set_files("#file", [self.images[name]])
        self.browser.wait_for("!document.querySelector('#preview').hidden && document.querySelector('#preview').textContent.indexOf('%s') >= 0"
                              % os.path.basename(self.images[name]), what="the attached image in the composer")

    def run_turn(self, name: str, note: Optional[str] = None, timeout: float = 40.0) -> Dict[str, Any]:
        """Attach, send and wait until that turn is over; -> its settled card (the newest: a turn before it does not count)."""
        b = self.browser
        before = b.evaluate("document.querySelectorAll('#conversation article.card').length")
        self.attach(name)
        if note:
            b.click("#prompt")
            b.insert_text(note)
        b.click("#send")
        try:
            return b.wait_for("(() => { const c = (%s)(); return c && c.cards > %d && c.status !== 'running' && c.pending === 0 ? c : null; })()"
                              % (CARD_JS, before), timeout, "the turn to end")
        except TimeoutError:
            raise self.stuck("the turn to end", timeout)

    # -- input a user has
    def choose(self, name: str) -> Dict[str, Any]:
        """What a click on the well does: the page asks for a file (input.click()), the dialog Chrome would open is intercepted, as a headless
        Chrome cannot show one, and answered with the file, as the user's choice would be. -> the chooser's facts (its mode)."""
        b = self.browser
        b.call("Page.setInterceptFileChooserDialog", enabled=True)
        try:
            b.events("Page.fileChooserOpened")
            b.click("#image-well")
            deadline = time.time() + 10
            opened = []  # type: List[Dict[str, Any]]
            while not opened:
                b.evaluate("0")   # an answer from the page: the events that came before it are in the record
                opened = b.events("Page.fileChooserOpened")
                if not opened and time.time() > deadline:
                    raise Failure("a click on the well opened no file chooser")
                time.sleep(0.03)
            b.call("DOM.getDocument", depth=0)
            b.call("DOM.setFileInputFiles", files=[self.images[name]], backendNodeId=opened[0]["backendNodeId"])
        finally:
            b.call("Page.setInterceptFileChooserDialog", enabled=False)
        return {"mode": opened[0].get("mode")}

    def drop(self, name: str, target: str = "#image-well") -> Dict[str, Any]:
        """A file dragged in from the desktop and dropped on the target (Input.dispatchDragEvent): the drag enters and moves over it, then the
        file drops. -> what the page showed while the file was over it, and where the page was after."""
        b = self.browser
        x, y = b.point(target)
        data = {"items": [], "files": [self.images[name]], "dragOperationsMask": 1}
        b.call("Input.dispatchDragEvent", type="dragEnter", x=x, y=y, data=data)
        b.call("Input.dispatchDragEvent", type="dragOver", x=x, y=y, data=data)
        over = b.evaluate("document.querySelector('#image-well').classList.contains('dragging')")
        b.call("Input.dispatchDragEvent", type="drop", x=x, y=y, data=data)
        b.evaluate("0")
        return {"well_shows_the_drag": over}

    def paste(self, name: Optional[str] = None, as_name: str = "image.png", text: str = "", target: str = "#conversation") -> bool:
        """A paste into the page (a ClipboardEvent made here: see the module's docstring) of a picture or of text; -> whether the page took it."""
        b64 = ""
        if name:
            with open(self.images[name], "rb") as handle:
                b64 = base64.b64encode(handle.read()).decode("ascii")
        return bool(self.browser.run(PASTE_JS, {"b64": b64, "name": as_name, "type": "image/png", "text": text, "target": target})["taken"])

    def sessions(self) -> int:
        with urllib.request.urlopen(self.base + "v1/sessions", timeout=10) as response:
            body = json.load(response)
        return len(body["sessions"] if isinstance(body, dict) else body)

    def stored(self, message: str) -> Dict[str, Any]:
        """A turn as the server stored it, asked from here: the page under watch is not the one that asks."""
        with urllib.request.urlopen("{}v1/messages/{}?after=0".format(self.base, message), timeout=10) as response:
            return stages_of(json.load(response)["events"])

    def posts(self) -> List[Dict[str, Any]]:
        """The requests that carried a turn (POST .../messages) since the last call."""
        self.browser.evaluate("0")
        return [p for p in self.browser.events("Network.requestWillBeSent") if p["request"]["method"] == "POST" and p["request"]["url"].endswith("/messages")]

    def preview(self, name: str, natural: Tuple[int, int]) -> Dict[str, Any]:
        """The composer's preview of this file, once its picture has decoded, held to what composer.js builds."""
        b = self.browser
        b.wait_for("(() => { const p = document.querySelector('#preview'); const i = p.querySelector(':scope > img');"
                   " return !p.hidden && !!i && i.complete && i.naturalWidth > 0 && p.textContent.indexOf(' px') > 0; })()", 10, "the preview of " + name)
        found = b.evaluate("(%s)()" % PREVIEW_JS)
        base = os.path.basename(self.images[name]) if name in self.images else name
        expect(found["shown"] and found["name"] == base, "the preview names {!r}, not {!r}".format(found["name"], base))
        expect(found["loaded"] and found["blob"] and found["natural"] == list(natural),
               "the preview's picture is {} (loaded {}, blob {}), not {}".format(found["natural"], found["loaded"], found["blob"], list(natural)))
        expect(re.fullmatch(r"{}×{} px · \d+ KB".format(natural[0], natural[1]), found["sizes"] or ""), "the preview's sizes read {!r}".format(found["sizes"]))
        expect(found["alt"] == "Preview of " + base and found["remove"] == "Remove image" and found["title"] == "{} · {}".format(base, found["sizes"]),
               "the preview's alt text, its Remove button or its title are {!r}, {!r}, {!r}".format(found["alt"], found["remove"], found["title"]))
        return found

    def wait_notice(self, message: str) -> Dict[str, Any]:
        self.browser.wait_for("(() => { const n = (%s)(); return n.shown && n.text === %s ? n : null; })()" % (NOTICE_JS, json.dumps(message)), 10,
                              "the notice {!r}".format(message))
        notice = self.browser.evaluate("(%s)()" % NOTICE_JS)
        expect(notice["role"] == "alert", "the notice has the role {!r}, not alert".format(notice["role"]))
        return notice

    def dismiss_notice(self) -> None:
        self.browser.click('#notice button[aria-label="Dismiss"]')
        self.browser.wait_for("document.querySelector('#notice').hidden", 5, "the notice to go")

    def shot(self, key: str, name: str, what: str, viewport: Tuple[int, int], scheme: str = "light") -> None:
        """A screenshot, once the page itself says that it shows what `key` claims (SHOT_RULES), and still says so after the capture."""
        rule = SHOT_RULES[key]
        if not self.browser.evaluate("(() => { %s })()" % rule):
            raise Failure("screenshot {}: the page does not show {}".format(name, what))
        if self.out_dir is None:
            return
        os.makedirs(self.out_dir, exist_ok=True)
        file_name = "{}_{}x{}_{}.png".format(name, viewport[0], viewport[1], scheme)
        path = os.path.join(self.out_dir, file_name)
        size = self.browser.screenshot(path)
        if not self.browser.evaluate("(() => { %s })()" % rule):
            os.remove(path)
            raise Failure("screenshot {}: the page stopped showing {} while it was being taken".format(name, what))
        if size >= MAX_SHOT_BYTES:
            os.remove(path)
            raise Failure("screenshot {} is {} bytes, over the {} of a committed picture".format(file_name, size, MAX_SHOT_BYTES))
        self.shots.append({"file": file_name, "shows": what, "viewport": list(viewport), "scheme": scheme, "bytes": size, "page_confirmed": True})

    def clip_stats(self, box: Sequence[float]) -> Tuple[float, float]:
        """The luma mean and spread of what is drawn in this box of the page (left, top, width, height), from a screenshot of exactly it."""
        shot = self.browser.call("Page.captureScreenshot", format="png", clip={"x": box[0], "y": box[1], "width": box[2], "height": box[3], "scale": 1})
        return luma_stats(base64.b64decode(shot["data"]))


# ---- the items ------------------------------------------------------------------------------------------------------------------------

def check_attach(ctx: Context) -> Tuple[str, Dict[str, Any]]:
    b = ctx.browser
    ctx.open("#/new")
    b.call("Network.enable")
    well = b.evaluate("document.querySelector('#image-well').textContent.trim()")
    expect(well.startswith("Drop, paste or click to attach an X-ray"), "the image well says {!r}".format(well))
    sessions_before = ctx.sessions()
    ctx.posts()   # what was sent before this item does not count
    # -- a click on the well: the chooser opens, a file is chosen, the preview shows it; Remove clears it and the focus goes back to the well
    chooser = ctx.choose("a")
    expect(chooser["mode"] == "selectSingle", "the chooser the click opened is {!r}, not a single-file one".format(chooser["mode"]))
    by_click = ctx.preview("a", ctx.sizes["a"])
    ctx.shot("attach_preview", "attach_preview", "the composer with an X-ray attached by a click on the well: its picture, name, pixel size and size on disk", DESKTOP)
    b.click("#preview button")
    removed = b.evaluate("(%s)()" % PREVIEW_JS)
    expect(not removed["shown"] and removed["focus"] == "image-well", "after Remove the preview is {} and the focus is on {!r}".format(
        "shown" if removed["shown"] else "gone", removed["focus"]))
    # -- a file dragged in from the desktop: the well shows the drag while the file is over it, and the drop attaches it
    dragged = ctx.drop("b")
    expect(dragged["well_shows_the_drag"], "the well did not show a file dragged over it (its class dragging)")
    by_drop = ctx.preview("b", ctx.sizes["b"])
    expect(not by_drop["dragging"], "the well still shows a drag after the drop")
    # -- a file dropped beside the well must not take the page away from the chat, and does not attach
    ctx.drop("c", target="#conversation")
    after_beside = b.evaluate("(%s)()" % PREVIEW_JS)
    expect(after_beside["href"] == by_drop["href"] and after_beside["name"] == "xray_b.png",
           "a file dropped beside the well left the page at {} showing {!r}".format(after_beside["href"], after_beside["name"]))
    # -- a paste: the picture of one is attached wherever the focus is; the text of one is the browser's
    text_taken = ctx.paste(text="a note, not a picture")
    expect(text_taken is False, "the page took a paste of text, which is the browser's")
    expect(b.evaluate("(%s)()" % PREVIEW_JS)["name"] == "xray_b.png", "a paste of text changed the attached image")
    expect(ctx.paste("c") is True, "the page did not take a pasted picture")
    by_paste = ctx.preview("image.png", ctx.sizes["c"])
    # -- refused files: the page's own (nothing is sent, what was attached stays), each shown in the alert
    refused = {}  # type: Dict[str, Any]
    for kind, name, message in (("gif", "gif", NOT_AN_IMAGE), ("dicom", "dcm", DICOM), ("over_20_mb", "huge", TOO_BIG)):
        ctx.choose(name)
        notice = ctx.wait_notice(message)
        kept = b.evaluate("(%s)()" % PREVIEW_JS)
        expect(kept["shown"] and kept["name"] == "image.png", "after the refused {} the preview is {!r}, not the picture that was attached".format(kind, kept["name"]))
        refused[kind] = {"message": notice["text"], "role": notice["role"], "attached_image_kept": kept["name"]}
        if kind == "gif":
            ctx.shot("attach_refused", "attach_refused", "a refused file: the alert with the page's message, and the earlier attachment still in the composer", DESKTOP)
        ctx.dismiss_notice()
    expect(ctx.posts() == [], "a refused file was sent to the server")
    # -- a refused file the page cannot know to refuse: the server's 422 is shown as it is, the image stays attached, and the chat the first turn made is gone
    b.click("#preview button")
    ctx.attach("small")
    ctx.preview("small", (32, 32))
    b.events("Network.responseReceived")
    b.click("#send")
    server_notice = ctx.wait_notice(TOO_SMALL)
    b.evaluate("0")
    answered = [p["response"]["status"] for p in b.events("Network.responseReceived") if p["response"]["url"].endswith("/messages")]
    expect(answered == [422], "the turn request was answered {}, not 422".format(answered))
    expect(len(ctx.posts()) == 1, "the refused turn was not the one request that carried a turn")
    kept = b.evaluate("(%s)()" % PREVIEW_JS)
    expect(kept["shown"] and kept["name"] == "small.png", "after the server's refusal the preview is {!r}".format(kept["name"]))
    deadline = time.time() + 8
    while ctx.sessions() != sessions_before:
        expect(time.time() < deadline, "the chat made for the refused turn is still listed ({} chats, {} before)".format(ctx.sessions(), sessions_before))
        time.sleep(0.1)
    expect(b.evaluate("location.hash") == "#/new" and b.evaluate("document.querySelector('#conversation').children.length") == 0,
           "the refused first turn left a chat on the page")
    return ("click, drop and paste each attached a picture with its preview (name, pixel size, size on disk; Remove puts the focus back on the well); the "
            "drag showed on the well; a drop beside it and a paste of text did nothing; the page refused a GIF, a DICOM name and 21 MB without a request, "
            "and showed the server's 422 for a 32 px picture and deleted its chat",
            {"well_text": well, "click": {"chooser": chooser, "preview": {k: by_click[k] for k in ("name", "sizes", "natural", "alt", "remove", "title")},
                                          "after_remove": {"shown": removed["shown"], "focus": removed["focus"]}},
             "drop": {"drag_shown_on_the_well": dragged["well_shows_the_drag"], "preview": {k: by_drop[k] for k in ("name", "sizes", "natural")},
                      "beside_the_well": {"navigated": after_beside["href"] != by_drop["href"], "attached": after_beside["name"] != "xray_b.png"}},
             "paste": {"text_taken": text_taken, "picture_taken": True, "mechanism": "a ClipboardEvent made in the page (headless Chrome has no clipboard)",
                       "preview": {k: by_paste[k] for k in ("name", "sizes", "natural")}},
             "refused_by_the_page": refused, "turn_requests_after_them": 0,
             "refused_by_the_server": {"message": server_notice["text"], "http_status": answered[0], "image_kept": kept["name"],
                                       "chats_left_behind": ctx.sessions() - sessions_before}})


def verify_pictures(ctx: Context, found: Dict[str, Any], name: str, bubble: str) -> Dict[str, Any]:
    """The pictures of the newest turn, held to what render.js draws: the X-ray (captioned with its pixel size) and the 224x224 image the model saw,
    each loaded from the server through the page's own loader, inside the card, with a picture's pixels; and the user turn's thumbnail, which is the
    page's own preview of the file while the turn is live (bubble "preview") and the server's thumbnail once the chat has been replayed ("server")."""
    w, h = ctx.sizes[name]
    base = "xray_{}.png".format(name)
    tiles = found["tiles"]
    expect(len(tiles) == 2, "the images row has {} pictures, not 2".format(len(tiles)))
    captions = [t["caption"] for t in tiles]
    expect(captions == ["Your X-ray · {}×{} px".format(w, h), MODEL_SAW], "the images row is captioned {}".format(captions))
    for tile in tiles:
        img = tile["img"]
        expect(img is not None and img["loaded"] and img["blob"], "the picture {!r} has not loaded through the page's loader: {}".format(tile["caption"], img))
    expect([t["img"]["natural"] for t in tiles] == [[w, h], [224, 224]], "the pictures are {} px, not {}".format([t["img"]["natural"] for t in tiles], [[w, h], [224, 224]]))
    paths = [t["img"]["path"] for t in tiles]
    thumb, model = (re.fullmatch(r"/v1/messages/([\w-]+)/image\?variant=" + v, p or "") for v, p in (("thumb", paths[0]), ("model_input", paths[1])))
    expect(thumb and model and thumb.group(1) == model.group(1), "the pictures come from {}, not the thumb and model input of one message".format(paths))
    card = found["card"]
    for tile in tiles:
        x, y, bw, bh = tile["img"]["box"]
        expect(bw >= 100 and bh >= 100 and x >= card[0] - 1 and x + bw <= card[0] + card[2] + 1 and y >= card[1] - 1 and y + bh <= card[1] + card[3] + 1,
               "the picture {!r} is drawn in {} outside its card {} or too small".format(tile["caption"], [round(v) for v in tile["img"]["box"]], [round(v) for v in card]))
    pixels = [t["img"]["pixels"] for t in tiles]
    expect(all(p is not None and p["std"] > 8 for p in pixels), "a picture of the images row is flat: {}".format(pixels))
    with open(ctx.images[name], "rb") as handle:
        sent = luma_stats(handle.read())
    expect(close_to(pixels[1]["mean"], sent[0], 8) and close_to(pixels[0]["mean"], sent[0], 8),
           "the pictures' brightness {} is not the X-ray's {:.1f}: the model input is not the X-ray, resized".format([round(p["mean"], 1) for p in pixels], sent[0]))
    thumb_img = found["bubble"]
    expect(thumb_img is not None and thumb_img["loaded"] and thumb_img["natural"] == [w, h], "the user turn's picture is {}".format(thumb_img))
    if bubble == "preview":
        expect(thumb_img["path"] is None and thumb_img["blob"], "the live user turn shows {} and not the page's own preview".format(thumb_img["path"]))
    else:
        expect(thumb_img["path"] == paths[0] and thumb_img["blob"] and thumb_img["alt"] == "Uploaded X-ray: " + base,
               "the replayed user turn shows {!r} (alt {!r}), not the server's thumbnail {!r}".format(thumb_img["path"], thumb_img["alt"], paths[0]))
    return {"captions": captions, "natural": [t["img"]["natural"] for t in tiles], "paths": paths, "user_turn_picture": bubble,
            "luma_mean_std": [[round(p["mean"], 1), round(p["std"], 1)] for p in pixels], "x_ray_luma_mean": round(sent[0], 1)}


def check_reload(ctx: Context) -> Tuple[str, Dict[str, Any]]:
    b = ctx.browser
    ctx.open("#/new")
    card = ctx.run_turn("b")
    expect(card["status"] == "done", "the turn ended {}".format(card["status"]))
    session = ctx.session_id()
    live = verify_pictures(ctx, b.evaluate("(%s)()" % PICTURES_JS), "b", "preview")
    ctx.reload()
    expect(b.evaluate("location.hash") == "#/s/" + session, "the address is no longer this chat after the reload")
    ctx.wait_card("done", cards=1, message="the replayed card")
    after_reload = verify_pictures(ctx, b.evaluate("(%s)()" % PICTURES_JS), "b", "server")
    b.click("#new-session")
    b.wait_for("location.hash === '#/new' && document.querySelectorAll('#conversation > *').length === 0", what="an empty new chat")
    b.click('#session-list li[data-session="%s"] a' % session)
    ctx.wait_card("done", cards=1, message="the card of the reopened chat")
    expect(b.evaluate("location.hash") == "#/s/" + session, "the reopened chat is not {}".format(session))
    reopened = verify_pictures(ctx, b.evaluate("(%s)()" % PICTURES_JS), "b", "server")
    b.evaluate("document.querySelector('#conversation article.card section.images').scrollIntoView({ block: 'center' })")
    ctx.shot("reopened_images", "reopened_images", "the images row of a reopened chat: the X-ray with its size, and the 224x224 image the model saw", DESKTOP)
    return ("the X-ray ({}x{} px) and the 224x224 model input were shown, loaded, inside the card, live, after a reload and after reopening the chat "
            "from the sidebar; the model input's brightness is the X-ray's".format(*ctx.sizes["b"]),
            {"x_ray_px": list(ctx.sizes["b"]), "live": live, "after_reload": after_reload, "after_reopening": reopened})


def check_similar(ctx: Context) -> Tuple[str, Dict[str, Any]]:
    b = ctx.browser
    ctx.open("#/new")
    card = ctx.run_turn("a")
    expect(card["status"] == "done", "the turn ended {}".format(card["status"]))
    turn = ctx.stored(card["id"])
    retrieve, label = turn["stages"]["retrieve"].get("detail") or {}, turn["stages"]["label"].get("detail") or {}
    ctx.gallery = retrieve.get("gallery") if isinstance(retrieve.get("gallery"), dict) else ctx.gallery   # what environment() says of the gallery
    neighbours, agreement = retrieve.get("image_neighbors") or [], label.get("neighbor_agreement") or []
    k = turn["start"]["options"]["k_images"]
    expect(len(neighbours) == k and len(agreement) == k and k > 0, "the server stored {} neighbours and {} agreements for k_images {}".format(len(neighbours), len(agreement), k))
    b.evaluate("document.querySelector('#conversation article.card section.neighbors').scrollIntoView({ block: 'center' })")
    grid = b.evaluate("(%s)()" % NEIGHBOURS_JS)
    expect(grid is not None and grid["title"] == TITLES["neighbors"] and grid["note"] is None, "the grid is titled {!r}".format(grid and grid["title"]))
    expect([int(i["rank"]) for i in grid["items"]] == [n["rank"] for n in neighbours] == list(range(1, k + 1)),
           "the grid lists ranks {} for the server's {}".format([i["rank"] for i in grid["items"]], [n["rank"] for n in neighbours]))
    similarities = [n["similarity"] for n in neighbours]
    expect(similarities == sorted(similarities, reverse=True), "the similarities are not in falling order: {}".format(similarities))
    generated = label["chexbert_14"]
    shown = []  # type: List[Dict[str, Any]]
    for item, n, a in zip(grid["items"], neighbours, agreement):
        expect(item["score"] == score_text(n["rank"], n["similarity"]), "neighbour {} reads {!r}, not {!r}".format(n["rank"], item["score"], score_text(n["rank"], n["similarity"])))
        expect(item["bar"] == bar_width(n["similarity"]), "neighbour {}'s bar is {} wide, not {}".format(n["rank"], item["bar"], bar_width(n["similarity"])))
        expect(item["img"] is not None and not item["placeholder"] and item["img"]["loaded"] and item["img"]["natural"] == [320, 320]
               and item["img"]["path"] == n["image_url"] == "/v1/gallery/images/{}".format(n["gallery_row"]),
               "neighbour {}'s picture is {} (the stored gallery size is 320 px)".format(n["rank"], item["img"]))
        for part in ("scoreBox", "barBox", "agreementBox"):
            expect(item[part] is not None and item[part][2] > 0 and item[part][3] > 0, "neighbour {}'s {} is not drawn".format(n["rank"], part))
        expect(a["rank"] == n["rank"] and a["of"] == 14, "the stored agreement {} is not for neighbour {}".format(a, n["rank"]))
        recounted = sum(1 for name in CHEXBERT_14 if bool(n["labels"][name]) == bool(generated[name]))
        expect(recounted == a["agree"], "neighbour {} agrees on {} of the 14 labels by a recount here, and the server says {}".format(n["rank"], recounted, a["agree"]))
        expect(item["agreement"] == agreement_text(recounted, 14) and re.fullmatch(r"\d+/14 labels agree", item["agreement"] or ""),
               "neighbour {} reads {!r}, not {!r}".format(n["rank"], item["agreement"], agreement_text(recounted, 14)))
        differing = ["neighbour: " + x for x in positives(n["labels"]) if not generated[x]] + ["report: " + x for x in positives(generated) if not n["labels"][x]]
        expect(sorted(item["chips"]) == sorted(differing), "neighbour {}'s chips are {}, not {}".format(n["rank"], item["chips"], differing))
        shown.append({"rank": n["rank"], "score": item["score"], "bar": item["bar"], "agreement": item["agreement"], "chips": item["chips"],
                      "picture_px": item["img"]["natural"], "gallery_path": item["img"]["path"]})
    ctx.shot("similar_xrays", "similar_xrays", "the grid of similar X-rays: each stored picture, its score and bar, and how many of the 14 labels agree", DESKTOP)
    return ("{} similar X-rays, each with its stored 320 px picture, '#rank . similarity' and a bar of that length, and 'n/14 labels agree' (n recounted here from "
            "the labels, as the server says) with the chips of the labels that differ".format(k),
            {"k_images": k, "neighbours": shown, "agreement_recounted_here": True, "similarity_falling": True})


def check_matches(ctx: Context) -> Tuple[str, Dict[str, Any]]:
    b = ctx.browser
    ctx.open("#/new")
    card = ctx.run_turn("a")
    expect(card["status"] == "done", "the turn ended {}".format(card["status"]))
    turn = ctx.stored(card["id"])
    retrieve = turn["stages"]["retrieve"].get("detail") or {}
    reports = retrieve.get("report_matches") or []
    k = turn["start"]["options"]["k_reports"]
    expect(len(reports) == k and k > 0, "the server stored {} matching reports for k_reports {}".format(len(reports), k))
    expect("true_report_rank" not in retrieve, "an upload has a rank of its own report: {}".format(retrieve.get("true_report_rank")))
    b.evaluate("document.querySelector('#conversation article.card section.matches').scrollIntoView({ block: 'center' })")
    shown = b.evaluate("(%s)()" % MATCHES_JS)
    expect(shown is not None and shown["title"] == TITLES["matches"], "the list is titled {!r}".format(shown and shown["title"]))
    expect(shown["badge"] is None, "an upload shows the own-rank badge {}".format(shown["badge"]))

    def verify_items(page: Dict[str, Any], stored: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        expect([int(i["rank"]) for i in page["items"]] == [m["rank"] for m in stored], "the page lists ranks {}".format([i["rank"] for i in page["items"]]))
        out = []
        for item, m in zip(page["items"], stored):
            expect(item["score"] == score_text(m["rank"], m["similarity"]), "report {} reads {!r}, not {!r}".format(m["rank"], item["score"], score_text(m["rank"], m["similarity"])))
            expect(item["text"] == m["report"] and item["clamped"], "report {}'s text is not the stored one, clamped".format(m["rank"]))
            expect(item["size"] == ("×{:,} identical reports".format(m["group_size"]) if m["group_size"] > 1 else None), "report {}'s group size reads {!r}".format(m["rank"], item["size"]))
            expect(item["chips"] == positives(m["labels"]), "report {}'s label chips are {}, not {}".format(m["rank"], item["chips"], positives(m["labels"])))
            expect((item["more"] == "Show all") == item["overflows"], "report {}: Show all is {!r} where the text {} its three lines".format(
                m["rank"], item["more"], "overflows" if item["overflows"] else "fits"))
            out.append({"rank": m["rank"], "score": item["score"], "group": item["size"], "label_chips": item["chips"], "words": len(m["report"].split()),
                        "show_all": item["more"]})
        return out

    upload_items = verify_items(shown, reports)
    first_long = next((i for i, item in enumerate(shown["items"]) if item["overflows"]), None)
    show_all = None
    if first_long is not None:   # Show all opens the whole text, and Show less shuts it again
        item_selector = "section.matches li.match:nth-child({})".format(first_long + 1)
        b.click(item_selector + ' button[data-action="show"]')
        opened = b.evaluate("(%s)()" % MATCHES_JS)["items"][first_long]
        expect(opened["open"] and opened["expanded"] == "true" and not opened["overflows"] and opened["more"] == "Show less",
               "Show all left report {} as {}".format(first_long + 1, opened))
        b.click(item_selector + ' button[data-action="show"]')
        closed = b.evaluate("(%s)()" % MATCHES_JS)["items"][first_long]
        expect(not closed["open"] and closed["expanded"] == "false" and closed["overflows"], "Show less left report {} as {}".format(first_long + 1, closed))
        show_all = {"report": first_long + 1, "opened_whole": True, "closed_again": True}
    b.evaluate("document.querySelector('#conversation article.card section.matches').scrollIntoView({ block: 'center' })")
    ctx.shot("matches", "matches", "the matching reports of an upload: score, bar, group size, the clamped text, the positive labels, and no badge", DESKTOP)
    # -- a test study picked in the composer: the card ranks its own report
    ctx.open("#/new")
    before = b.evaluate("(%s)()" % PICKER_JS)
    expect(before["button"] == {"hidden": False, "expanded": "false", "disabled": False} and not before["open"], "the picker's button is {}".format(before["button"]))
    b.click("#pick-study")
    b.wait_for("document.querySelectorAll('#picker-list button').length > 0", 15, "the list of test studies")
    listed = b.evaluate("(%s)()" % PICKER_JS)
    with urllib.request.urlopen(ctx.base + "v1/test-studies?limit=50", timeout=10) as response:
        served = json.load(response)["studies"]
    expect(listed["open"] and [s["row"] for s in listed["studies"]] == [str(s["test_row"]) for s in served] and len(served) > 3,
           "the picker lists {} studies for the server's {}".format(len(listed["studies"]), len(served)))
    study = served[3]
    b.click('#picker-list button[data-test-row="{}"]'.format(study["test_row"]))
    picked_card = ctx.wait_card("done", cards=1, timeout=40, message="the card of the picked study")
    expect(picked_card["status"] == "done", "the picked study's turn ended {}".format(picked_card["status"]))
    picked = ctx.stored(picked_card["id"])
    expect(picked["start"]["image"]["source"] == "test_split" and picked["start"]["options"]["test_row"] == study["test_row"],
           "the picked turn ran {} and not test row {}".format(picked["start"]["image"], study["test_row"]))
    rank = (picked["stages"]["retrieve"].get("detail") or {}).get("true_report_rank")
    expect(isinstance(rank, dict) and isinstance(rank.get("rank"), int), "the server stored no rank for the picked study's own report")
    b.evaluate("document.querySelector('#conversation article.card section.matches').scrollIntoView({ block: 'center' })")
    page = b.evaluate("(%s)()" % MATCHES_JS)
    expect(page["title"] == TITLES["matches"] and page["badge"] is not None and page["badge"]["shown"], "the picked study's card shows no badge")
    expect(page["badge"]["text"] == own_rank_text(rank), "the badge reads {!r}, not {!r}".format(page["badge"]["text"], own_rank_text(rank)))
    tip = page["badge"]["title"] or ""
    expect(("Dedup-aware rank {:,}".format(rank["rank_dedup"]) in tip) and ("Protocol: {}.".format(rank["protocol"]) in tip),
           "the badge's tooltip is {!r}: it should give the dedup-aware rank {} and the protocol {}".format(tip, rank.get("rank_dedup"), rank.get("protocol")))
    picked_items = verify_items(page, (picked["stages"]["retrieve"].get("detail") or {}).get("report_matches") or [])
    bubble = b.evaluate("(%s)()" % BUBBLE_JS)
    expect(bubble["image"] == "Test-split X-ray" and "test row {}".format(study["test_row"]) in bubble["chips"],
           "the user turn of the picked study is {}".format(bubble))
    ctx.shot("own_rank", "own_rank", "a picked test study: the badge with where its own report ranks, above its matching reports", DESKTOP)
    # -- the whole card, light and dark, on a page tall enough for it
    b.set_viewport(TALL[0], TALL[1], "light")
    b.evaluate("document.querySelector('#conversation').scrollTop = 0")
    ctx.shot("card", "card", "the whole card of the picked study: stages, report, labels, images, similar X-rays, matching reports with the own-rank badge, provenance", TALL)
    b.set_viewport(TALL[0], TALL[1], "dark")
    ctx.shot("card", "card", "the same card in the dark colour scheme", TALL, "dark")
    b.set_viewport(DESKTOP[0], DESKTOP[1], "light")
    return ("{} matching reports (rank, similarity, group size, the stored text clamped to three lines, the positive labels) and no badge for an upload; a test "
            "study picked in the composer shows '{}'".format(k, page["badge"]["text"]),
            {"k_reports": k, "upload": {"reports": upload_items, "own_rank_badge": None},
             "show_all": show_all_facts([shown, page], show_all),
             "picked_study": {"test_row": study["test_row"], "badge": page["badge"]["text"], "tooltip": tip, "stored_rank": {x: rank[x] for x in ("rank", "of", "rank_dedup", "hit_at_10") if x in rank},
                              "user_turn": bubble, "reports": picked_items, "studies_listed": len(served)}})


def visible_part(box: Sequence[float], stage: Sequence[float]) -> List[float]:
    """The part of a picture's box (left, top, width, height) that lies in its stage's: what the viewer shows of it."""
    x0, y0 = max(box[0], stage[0]), max(box[1], stage[1])
    x1, y1 = min(box[0] + box[2], stage[0] + stage[2]), min(box[1] + box[3], stage[1] + stage[3])
    return [x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)]


def check_viewer(ctx: Context) -> Tuple[str, Dict[str, Any]]:
    b = ctx.browser
    ctx.open("#/new")
    card = ctx.run_turn("a")
    expect(card["status"] == "done", "the turn ended {}".format(card["status"]))
    turn = ctx.stored(card["id"])
    neighbours = (turn["stages"]["retrieve"].get("detail") or {}).get("image_neighbors") or []
    openers = b.evaluate("(%s)()" % OPENERS_JS)
    keys = [o["key"] for o in openers]
    expected = ["image:thumb", "image:model_input"] + ["neighbor:{}".format(n["rank"]) for n in neighbours]
    expect(keys == expected and len(neighbours) > 0, "the card's pictures that open the viewer are {}, not {}".format(keys, expected))
    bubble = b.evaluate("(%s)()" % BUBBLE_JS)
    expect(bubble is not None and bubble["thumbIsOpener"] is True, "the upload in the user's own message is not a control that opens the viewer: {}".format(bubble))
    pictures = [USER_THUMB] + keys   # the upload in the user's own message (P6-G), then the card's pictures

    def selector_of(key: str) -> str:
        """The control a picture is opened from: the thumbnail in the user's own message, or a picture button of the card by its data-key."""
        if key == USER_THUMB:
            return "#conversation article.turn.user button.thumb-button"
        return '#conversation article.card button.picture-button[data-key="{}"]'.format(key)

    def focus_of(key: str) -> str:
        """What VIEWER_JS names the control that has the focus once the viewer has closed to it: its data-key, else its class."""
        return "thumb-button" if key == USER_THUMB else key

    def state() -> Dict[str, Any]:
        return b.evaluate("(%s)()" % VIEWER_JS)

    def settle() -> None:
        b.evaluate("0")
        time.sleep(0.12)   # a frame of the page: the transform and the filter it sets are drawn

    def mouse(kind: str, x: float, y: float, **more: Any) -> None:
        b.call("Input.dispatchMouseEvent", type=kind, x=x, y=y, **more)

    def rect_of(selector: str) -> List[float]:
        return b.evaluate("(() => { const r = document.querySelector(%s).getBoundingClientRect(); return [r.left, r.top, r.width, r.height]; })()" % json.dumps(selector))

    def open_viewer(key: str, how: str = "click") -> Dict[str, Any]:
        selector = selector_of(key)
        if how == "click":
            b.click(selector)   # a real click on the picture
        else:   # the keyboard: the picture's button has the focus, where Tab would leave it, and Enter presses it
            b.evaluate("document.querySelector({}).focus()".format(json.dumps(selector)))
            b.key("Enter", "Enter", 13, text="\r")
        b.wait_for(VIEWER_LOADED, 15, "the viewer to open from " + key)
        return state()

    def close_viewer(key: str, how: str) -> None:
        if how == "Esc":
            b.key("Escape", "Escape", 27)
        else:
            b.click('#viewer button[data-viewer="close"]')
        b.wait_for("document.querySelector('#viewer').hidden", 5, "the viewer to close")
        closed = state()
        expect(closed["held"] is False and closed["inert"] == [False, False, False] and closed["panes"] == [],
               "after {} the viewer left the page held ({}) or inert ({}) or its panes in place".format(how, closed["held"], closed["inert"]))
        expect(closed["focus"] == focus_of(key), "after {} the focus is on {!r}, not on {} that opened the viewer".format(how, closed["focus"], focus_of(key)))

    def verify_open(v: Dict[str, Any], key: str) -> Dict[str, Any]:
        side = key.startswith("neighbor:")
        rank = key.split(":")[1] if side else None
        want = expected_viewer(key)
        label, footers = want["label"], want["footers"]
        expect(v["label"] == label, "opened from {} the viewer says {!r}, not {!r}".format(key, v["label"], label))
        expect([p["footer"] for p in v["panes"]] == footers, "opened from {} the footers read {}, not {}".format(key, [p["footer"] for p in v["panes"]], footers))
        expect(v["held"] and v["inert"] == [True, True, True], "opened from {} the page behind is not inert ({}, {})".format(key, v["held"], v["inert"]))
        for pane in v["panes"]:
            expect(pane["loaded"] and pane["blob"] and pane["scale"] == 1, "a pane of the viewer opened from {} is {}".format(key, pane))
            x, y, w, h = pane["box"]
            sx, sy, sw, sh = pane["stage"]
            expect(sw >= 40 and sh >= 40 and x >= sx - 1 and y >= sy - 1 and x + w <= sx + sw + 1 and y + h <= sy + sh + 1
                   and (close_to(w, sw, 2) or close_to(h, sh, 2)), "the picture is drawn at {} in a stage {}: not whole and fitted".format(
                       [round(v) for v in pane["box"]], [round(v) for v in pane["stage"]]))
        paths = [p["path"] for p in v["panes"]]
        expect(variant_of(paths[0]) == want["variant"], "opened from {} the first pane shows {!r}, not the {} variant of the turn's picture".format(key, paths[0], want["variant"]))
        if side:
            expect(paths[1] == "/v1/gallery/images/{}".format(neighbours[int(rank) - 1]["gallery_row"]), "the neighbour's pane shows {!r}".format(paths[1]))
            left, right = v["panes"][0]["stage"], v["panes"][1]["stage"]
            expect(left[0] + left[2] <= right[0] + 1 and close_to(left[1], right[1], 1), "the panes are not side by side: {} and {}".format(left, right))
        expect(v["focus"] == "viewer-stage", "the keys are not on the picture: the focus is on {!r}".format(v["focus"]))
        return {"label": v["label"], "footers": [p["footer"] for p in v["panes"]], "first_picture_variant": variant_of(paths[0]), "panes": len(v["panes"]),
                "page_behind_inert": True}

    # -- the viewer opens from the upload in the user's own message and from every picture of the card, closes to the picture that opened it, by Esc
    # and by its close button in turn; what each says is held to expected_viewer, which is the card's "Your X-ray" tile's for the upload
    opened = {}  # type: Dict[str, Any]
    seen = {}  # type: Dict[str, Dict[str, Any]]   # what the viewer said when it had opened from each
    for index, key in enumerate(pictures):
        how_open, how_close = ("click", "Esc") if index % 2 == 0 else ("Enter", "its close button")   # the mouse and the keyboard in turn, both ways
        seen[key] = open_viewer(key, how_open)
        opened[key] = dict(verify_open(seen[key], key), opened_by="a click on the picture" if how_open == "click" else "Enter on the focused picture",
                           closed_by=how_close)
        close_viewer(key, how_close)
    # the upload in the user's own message opens what the card's "Your X-ray" tile opens: the same picture (its path), name and footer; and the other way
    # of opening and of closing it (Enter on it, its close button) opens it and gives the focus back to it as well
    upload, tile = seen[USER_THUMB], seen["image:thumb"]
    same = [(p["path"], p["alt"], p["footer"]) for p in upload["panes"]] == [(p["path"], p["alt"], p["footer"]) for p in tile["panes"]]
    expect(same and upload["label"] == tile["label"], "the upload in the user's own message opens {!r} {} where the card's tile opens {!r} {}".format(
        upload["label"], [(p["alt"], p["footer"]) for p in upload["panes"]], tile["label"], [(p["alt"], p["footer"]) for p in tile["panes"]]))
    verify_open(open_viewer(USER_THUMB, "Enter"), USER_THUMB)
    close_viewer(USER_THUMB, "its close button")
    opened[USER_THUMB]["same_picture_as_the_card_tile"] = {"tile": "image:thumb", "label": upload["label"], "variant": variant_of(upload["panes"][0]["path"])}
    opened[USER_THUMB]["and_also"] = "opened by Enter on the focused thumbnail and closed by its close button: the focus came back to it too"
    # -- a similar X-ray beside the turn's own: the controls, by mouse and by keyboard, measured on what is drawn
    key = "neighbor:1"
    v0 = open_viewer(key)
    verify_open(v0, key)
    ctx.shot("viewer_side_by_side", "viewer_side_by_side", "the viewer on a similar X-ray beside the turn's own, at its fit", DESKTOP)
    left0, right0 = v0["panes"]
    facts = {}  # type: Dict[str, Any]
    # zoom by the wheel, about the cursor, both panes together (linked)
    cx, cy = left0["stage"][0] + left0["stage"][2] * 0.6, left0["stage"][1] + left0["stage"][3] * 0.4
    under = lambda box: ((cx - box[0]) / box[2], (cy - box[1]) / box[3])
    u0 = under(left0["box"])
    mouse("mouseMoved", cx, cy)
    mouse("mouseWheel", cx, cy, deltaX=0, deltaY=-240)
    settle()
    left1, right1 = state()["panes"]
    factor = left1["box"][2] / left0["box"][2]
    expect(close_to(factor, wheel_factor(-240), 0.01 * factor), "a turn of the wheel zoomed the picture by {:.3f}, not {:.3f}".format(factor, wheel_factor(-240)))
    expect(close_to(right1["box"][2] / right0["box"][2], factor, 0.01 * factor), "the linked pane zoomed by {:.3f}, not {:.3f}".format(right1["box"][2] / right0["box"][2], factor))
    u1 = under(left1["box"])
    expect(close_to(u1[0], u0[0], 0.01) and close_to(u1[1], u0[1], 0.01), "the point under the cursor moved from {} to {}: it zooms about the page's centre, not the cursor".format(
        [round(x, 3) for x in u0], [round(x, 3) for x in u1]))
    expect(close_to(left1["scale"], factor, 0.01 * factor), "the page says zoom {} where the picture is drawn {:.3f} times larger".format(left1["scale"], factor))
    facts["wheel_zoom"] = {"delta_y": -240, "factor_drawn": round(factor, 4), "factor_expected": round(wheel_factor(-240), 4), "linked_pane_factor": round(right1["box"][2] / right0["box"][2], 4),
                           "point_under_cursor_before_after": [[round(x, 3) for x in u0], [round(x, 3) for x in u1]]}
    # pan by dragging
    box_before = left1["box"]
    mouse("mousePressed", cx, cy, button="left", clickCount=1)
    for step in range(1, 7):
        mouse("mouseMoved", cx + 10 * step, cy + 6 * step, button="left", buttons=1)
    mouse("mouseReleased", cx + 60, cy + 36, button="left", clickCount=1)
    settle()
    left2, right2 = state()["panes"]
    moved = (left2["box"][0] - box_before[0], left2["box"][1] - box_before[1])
    expect(close_to(moved[0], 60, 1.5) and close_to(moved[1], 36, 1.5), "dragging by (60, 36) moved the picture by {}".format([round(x, 1) for x in moved]))
    expect(close_to(right2["box"][0] - right1["box"][0], 60, 1.5), "the linked pane did not follow the drag: it moved {}".format(round(right2["box"][0] - right1["box"][0], 1)))
    facts["drag_pan"] = {"dragged": [60, 36], "picture_moved": [round(x, 1) for x in moved]}
    # pan and zoom by the keys, on the picture that has the focus
    expect(state()["focus"] == "viewer-stage", "the keys are not on a picture: the focus is on {!r}".format(state()["focus"]))
    box_before = left2["box"]
    b.key("ArrowLeft", "ArrowLeft", 37)
    settle()
    left3 = state()["panes"][0]
    expect(close_to(left3["box"][0] - box_before[0], PAN_STEP, 1.5), "ArrowLeft moved the picture by {}, not {}".format(round(left3["box"][0] - box_before[0], 1), PAN_STEP))
    b.key("ArrowDown", "ArrowDown", 40)
    settle()
    left4 = state()["panes"][0]
    expect(close_to(left4["box"][1] - left3["box"][1], -PAN_STEP, 1.5), "ArrowDown moved the picture by {}, not {}".format(round(left4["box"][1] - left3["box"][1], 1), -PAN_STEP))
    b.key("+", "Equal", 187, text="+", modifiers=8)
    settle()
    left5 = state()["panes"][0]
    expect(close_to(left5["box"][2] / left4["box"][2], ZOOM_STEP, 0.01 * ZOOM_STEP), "+ zoomed by {:.3f}, not {}".format(left5["box"][2] / left4["box"][2], ZOOM_STEP))
    b.key("-", "Minus", 189, text="-")
    settle()
    left6 = state()["panes"][0]
    expect(close_to(left6["box"][2], left4["box"][2], 1.5), "- did not undo +: the picture is {:.1f} wide, was {:.1f}".format(left6["box"][2], left4["box"][2]))
    facts["keys"] = {"arrow_left_moved": round(left3["box"][0] - box_before[0], 1), "arrow_down_moved": round(left4["box"][1] - left3["box"][1], 1),
                     "plus_factor": round(left5["box"][2] / left4["box"][2], 4), "minus_undid_plus": True}
    # reset by the key: the fit, and both sliders at 100
    b.key("0", "Digit0", 48, text="0")
    settle()
    fit = state()
    expect(all(close_to(p["box"][i], q["box"][i], 1.5) for p, q in zip(fit["panes"], v0["panes"]) for i in range(4)) and all(p["scale"] == 1 for p in fit["panes"]),
           "0 did not bring the pictures back to their fit: {} against {}".format([p["box"] for p in fit["panes"]], [p["box"] for p in v0["panes"]]))
    # the buttons, by mouse: + (a step), 1:1 (a picture pixel to a CSS pixel), Fit
    b.click('#viewer button[data-viewer="in"]')
    settle()
    stepped = state()["panes"]
    expect(all(close_to(p["box"][2] / q["box"][2], ZOOM_STEP, 0.01 * ZOOM_STEP) for p, q in zip(stepped, fit["panes"])), "the + button did not zoom both panes by {}".format(ZOOM_STEP))
    b.click('#viewer button[data-viewer="actual"]')
    settle()
    actual = state()["panes"]
    expect(all(close_to(p["box"][2], p["natural"][0], 1.5) and close_to(p["box"][3], p["natural"][1], 1.5) for p in actual),
           "1:1 draws the pictures at {}, not at their {}".format([[round(x) for x in p["box"][2:]] for p in actual], [p["natural"] for p in actual]))
    b.click('#viewer button[data-viewer="fit"]')
    settle()
    expect(all(close_to(p["box"][2], q["box"][2], 1.5) for p, q in zip(state()["panes"], fit["panes"])), "Fit did not bring the pictures back to their fit")
    facts["buttons"] = {"in_factor": ZOOM_STEP, "one_to_one_px": [p["natural"] for p in actual], "fit_restored": True}
    # link off: the wheel then zooms the pane under it and no other
    b.click('#viewer input[data-viewer="link"]')
    expect(state()["linked"] is False, "the Link panes box is still on after a click")
    before_unlinked = state()["panes"]
    mouse("mouseMoved", cx, cy)
    mouse("mouseWheel", cx, cy, deltaX=0, deltaY=-240)
    settle()
    unlinked = state()["panes"]
    expect(unlinked[0]["box"][2] > before_unlinked[0]["box"][2] * 1.3 and close_to(unlinked[1]["box"][2], before_unlinked[1]["box"][2], 1.5),
           "with the panes unlinked the wheel zoomed {} and {}".format(round(unlinked[0]["box"][2] / before_unlinked[0]["box"][2], 3), round(unlinked[1]["box"][2] / before_unlinked[1]["box"][2], 3)))
    b.click('#viewer input[data-viewer="link"]')
    b.key("0", "Digit0", 48, text="0")
    settle()
    facts["link_panes"] = {"unlinked_zoom_left_only": True}

    def luma_of_left() -> Tuple[float, float]:
        pane = state()["panes"][0]
        return ctx.clip_stats(visible_part(pane["box"], pane["stage"]))

    def click_slider(name: str, fraction: float) -> None:
        x, y, w, h = rect_of('#viewer input[data-viewer="{}"]'.format(name))
        mouse("mouseMoved", x + w * fraction, y + h / 2)
        mouse("mousePressed", x + w * fraction, y + h / 2, button="left", clickCount=1)
        mouse("mouseReleased", x + w * fraction, y + h / 2, button="left", clickCount=1)
        settle()

    def focus_slider(name: str) -> int:
        """Tab to the slider, as a keyboard user does (the viewer keeps the focus inside itself), from a picture: a click on the picture (no drag)
        puts the focus there, so that the Tab presses it takes are the walk of a keyboard user and not a slider that already had the focus from the
        mouse. -> the Tab presses it took."""
        x, y = left0["stage"][0] + left0["stage"][2] / 2, left0["stage"][1] + left0["stage"][3] / 2
        mouse("mousePressed", x, y, button="left", clickCount=1)
        mouse("mouseReleased", x, y, button="left", clickCount=1)
        expect(b.evaluate("document.activeElement.getAttribute('data-viewer')") != name, "the {} slider already has the focus after a click on the picture".format(name))
        for presses in range(1, 25):
            b.key("Tab", "Tab", 9)
            if b.evaluate("document.activeElement.getAttribute('data-viewer')") == name:
                return presses
        raise Failure("Tab did not reach the {} slider in 24 presses".format(name))

    def keys_to(name: str, target: int) -> Dict[str, Any]:
        """Arrow keys on the focused slider until it is within a step of the target; -> the presses, and the state."""
        presses = 0
        while presses < 80:
            now = state()[name]["value"]
            if abs(now - target) <= 2.5:
                break
            b.key("ArrowRight" if now < target else "ArrowLeft", "ArrowRight" if now < target else "ArrowLeft", 39 if now < target else 37)
            presses += 1
        settle()
        return {"presses": presses, "value": state()[name]["value"]}

    def filters_agree(name: str, value: int) -> None:
        now = state()
        expect(all("{}({}%)".format(name, value) in p["filter"] for p in now["panes"]) and now[name]["text"] == "{}%".format(value),
               "the {} slider is at {} but the pictures' filters are {} and its text {!r}".format(name, value, [p["filter"] for p in now["panes"]], now[name]["text"]))

    mean0, std0 = luma_of_left()
    # brightness: the slider clicked, then the keys on it
    click_slider("brightness", 0.75)
    bright = state()["brightness"]["value"]
    expect(abs(bright - 225) <= 35, "a click at three quarters of the brightness slider set it to {}".format(bright))
    filters_agree("brightness", bright)
    mean_up, _ = luma_of_left()
    expect(mean_up > mean0 + 15, "brightness {} did not brighten the picture: mean luma {:.1f} then {:.1f}".format(bright, mean0, mean_up))
    tabs = focus_slider("brightness")
    pressed = keys_to("brightness", 60)
    filters_agree("brightness", pressed["value"])
    mean_down, _ = luma_of_left()
    expect(mean_down < mean0 - 15, "brightness {} did not darken the picture: mean luma {:.1f} then {:.1f}".format(pressed["value"], mean0, mean_down))
    facts["brightness"] = {"click_at_75_percent": bright, "mean_luma": {"at_100": round(mean0, 1), "by_click": round(mean_up, 1), "by_keys": round(mean_down, 1)},
                           "keys": {"tab_presses_to_reach_it": tabs, "arrow_presses": pressed["presses"], "value": pressed["value"]}}
    b.key("0", "Digit0", 48, text="0")
    settle()
    expect(state()["brightness"]["value"] == 100 and state()["contrast"]["value"] == 100, "0 did not put both sliders back to 100")
    # contrast: the same, on the spread of the luma
    click_slider("contrast", 0.8)
    contrast = state()["contrast"]["value"]
    expect(abs(contrast - 240) <= 35, "a click at four fifths of the contrast slider set it to {}".format(contrast))
    filters_agree("contrast", contrast)
    _, std_up = luma_of_left()
    expect(std_up > std0 + 5, "contrast {} did not widen the picture's range: luma spread {:.1f} then {:.1f}".format(contrast, std0, std_up))
    tabs = focus_slider("contrast")
    pressed = keys_to("contrast", 50)
    filters_agree("contrast", pressed["value"])
    _, std_down = luma_of_left()
    expect(std_down < std0 - 8, "contrast {} did not narrow the picture's range: luma spread {:.1f} then {:.1f}".format(pressed["value"], std0, std_down))
    facts["contrast"] = {"click_at_80_percent": contrast, "luma_spread": {"at_100": round(std0, 1), "by_click": round(std_up, 1), "by_keys": round(std_down, 1)},
                         "keys": {"tab_presses_to_reach_it": tabs, "arrow_presses": pressed["presses"], "value": pressed["value"]}}
    b.key("0", "Digit0", 48, text="0")
    settle()
    # one picture of the viewer as it is used: zoomed, panned, a little brighter and with more contrast
    mouse("mouseMoved", cx, cy)
    mouse("mouseWheel", cx, cy, deltaX=0, deltaY=-240)
    mouse("mousePressed", cx, cy, button="left", clickCount=1)
    for step in range(1, 5):
        mouse("mouseMoved", cx - 12 * step, cy + 8 * step, button="left", buttons=1)
    mouse("mouseReleased", cx - 48, cy + 32, button="left", clickCount=1)
    focus_slider("brightness")
    keys_to("brightness", 130)
    focus_slider("contrast")
    keys_to("contrast", 125)
    settle()
    ctx.shot("viewer_adjusted", "viewer_adjusted", "the viewer zoomed, panned, brighter and with more contrast", DESKTOP)
    b.key("0", "Digit0", 48, text="0")
    settle()
    b.set_viewport(DESKTOP[0], DESKTOP[1], "dark")
    ctx.shot("viewer_side_by_side", "viewer_side_by_side", "the same viewer in the dark colour scheme", DESKTOP, "dark")
    b.set_viewport(DESKTOP[0], DESKTOP[1], "light")
    close_viewer(key, "Esc")
    return ("the viewer opened from all {} pictures (the upload in the user's own message, which opens what the card's X-ray tile opens; the card's X-ray, "
            "the model input and {} similar X-rays: each closed to the picture that opened it) and, "
            "on a similar X-ray beside the turn's own, zoom (wheel about the cursor, + and -, the buttons, 1:1), pan (drag, arrows), brightness and contrast "
            "(a click, then the keys) all worked, measured on the pictures drawn and on their pixels".format(len(pictures), len(neighbours)),
            {"opens_from": pictures, "user_turn_thumbnail_is_an_opener": bubble["thumbIsOpener"], "each": opened, "side_by_side_controls": facts})


CHECKS = [("attach", "click, drag-drop and paste each attach with a preview, and a refused file shows its message", check_attach),
          ("reload", "after a reload, and after reopening the session, the X-ray and the 224×224 model input are both shown", check_reload),
          ("similar", "the similar-X-rays grid shows scores and 'n/14 labels agree'", check_similar),
          ("matches", "the matching-reports list and, for a picked test study, the own-rank badge", check_matches),
          ("viewer", "the viewer opens from every image (the user's own upload included), side by side for a neighbour, with zoom, pan, brightness and contrast "
                     "working by mouse and by keyboard", check_viewer)]


def run_all(browser: Browser, base: str, work: str, out_dir: Optional[str]) -> Tuple[List[Dict[str, Any]], Context]:
    """Every item in order, on this browser and this app; -> (one result per item, the context with the screenshots)."""
    browser.call("Runtime.enable")
    browser.call("Emulation.setFocusEmulationEnabled", enabled=True)   # the page behaves as the focused one, so key events move focus
    ctx = Context(browser, base, work, out_dir)
    if out_dir is not None:
        clear_evidence(out_dir)
    return [run_check(ctx, name, fn, claim=claim, keep=1500, failure_chars=1500)   # a failure says all of what it knows
            for name, claim, fn in CHECKS], ctx


def get_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=15) as response:
        return json.load(response)


CARD_FACTS = ("name", "checkpoint", "checkpoint_sha256", "train_experiment", "drift_note", "device", "prefix_k", "eos_trained",
              "cached_decode_available")   # what the checklist keeps of a model's card: which engine it is, and whether it has a checkpoint


def environment(browser: Browser, own_app: bool, models: Dict[str, Any], gallery: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The checklist's environment. What it says of the app is what the app said of itself, read and not written here: the mode, the default model
    and the features of its /v1/models, each model's card (CARD_FACTS: name, checkpoint, hash, run, the note it gives of itself, device...), and
    the retrieve stage's own account of the gallery it searched (build_id, images, report_rows, report_groups, towers_identical). The words
    "synthetic" and "no checkpoint" are in it only when the app's own answer says them."""
    cards = [card for card in models["models"] if isinstance(card, dict)] if isinstance(models.get("models"), list) else []
    return {"date": datetime.date.today().isoformat(), "browser": browser.call("Browser.getVersion")["product"] + " (headless)", "python": platform.python_version(),
            "os": platform.system(),
            "server": {"started_by_the_check": own_app, "mode": models.get("mode"), "default_model": models.get("default_model"),
                       "features": models.get("features")},
            "models": [{key: card.get(key) for key in CARD_FACTS} for card in cards], "gallery": gallery,
            "uploaded_pictures": "phantoms drawn by the check (scripts/chat_ui_features_check.py phantom_png); nothing is from a patient or from MIMIC",
            "viewports": {"desktop": list(DESKTOP), "tall (the whole card)": list(TALL)},
            "notes": ["The serving command (python -m app.server --engine tiny --gallery DIR --labeler rule) cannot serve the tiny gallery: its manifest names a "
                      "tower whose hash is not the tiny engine's, and the server refuses a gallery it cannot match to its engine ('the gallery was built with "
                      "another image tower', exit 1). The check starts the app as the laptop's dev server does, create_app(tiny_gallery=True).",
                      "Real input: the click on the well (Chrome's file chooser is intercepted and answered with a file: a headless Chrome has no dialog), the file "
                      "dropped from disk (Input.dispatchDragEvent), the mouse and the keyboard of the viewer. Not real: the paste, which is a ClipboardEvent made "
                      "in the page that holds a File or text, because a headless Chrome has no clipboard of its own and a real Cmd+V would read the user's.",
                      "Brightness and contrast are measured on pixels: a screenshot of the left picture's box, its luma mean (brightness) and spread (contrast). "
                      "Zoom and pan are measured on the picture's drawn box, and compared with the page's own data-scale.",
                      "Every screenshot was confirmed by the page itself (a DOM rule per picture, true before and after the capture)."]}


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.url:   # before anything is started or sent (the warm-up turn is a POST): the address, then the app's own answer about itself
        problem = url_problem(args.url) or tiny_only_problem(args.url)
        if problem:
            print("ERROR --url refused: " + problem)
            return 2
    exit_on_sigterm()   # a SIGTERM runs the finally below, which stops Chrome and the app
    chrome = find_chrome()
    if chrome is None:
        print("ERROR no Chrome found: set $CHROME to its executable")
        return 2

    def overrun(signum: int, frame: Any) -> None:
        raise Overrun("the features check ran past its {} s limit".format(args.timeout))

    signal.signal(signal.SIGALRM, overrun)
    signal.alarm(args.timeout)
    out_dir = None if args.no_evidence else os.path.abspath(args.out)
    work = tempfile.mkdtemp(prefix="chat_ui_features_check_")
    app = None  # type: Optional[App]
    browser = None  # type: Optional[Browser]
    results = []  # type: List[Dict[str, Any]]
    ctx = None  # type: Optional[Context]
    env = {}  # type: Dict[str, Any]
    try:
        if args.url:
            base = args.url if args.url.endswith("/") else args.url + "/"
        else:
            app = App(step_delay_s=STEP_DELAY_S, tiny_gallery=True)
            base = app.url
            problem = url_problem(base)   # a port the OS gave: never a refused one, but this check does not take its chance
            if problem:
                raise RuntimeError("the app came up where this check does not go: " + problem)
        models = get_json(base + "v1/models")
        lacking = missing_features(models)
        if lacking:
            print("ERROR " + lacking)
            return 2
        warm_up(base)
        browser = Browser(chrome, label="chat_ui_features_chrome")
        results, ctx = run_all(browser, base, work, out_dir)
        env = environment(browser, app is not None, models, ctx.gallery if ctx is not None else None)
    except Overrun as exc:
        print("ERROR {}".format(exc))
        return 2
    except Exception as exc:   # the app or Chrome would not start, a request failed: not a failed item, a run that could not be made
        print("ERROR {}: {}".format(type(exc).__name__, exc))
        return 2
    finally:
        signal.alarm(0)
        try:   # each step in the finally of the one before it: an interrupt in one must not leave the next running
            if browser is not None:
                browser.close()
        finally:
            try:
                if app is not None:
                    app.stop()
            finally:
                shutil.rmtree(work, ignore_errors=True)
    failures = [r["name"] for r in results if r["status"] != "PASS"]
    result = {"checks": len(results), "failures": failures}
    if out_dir is not None and ctx is not None:
        write_checklist(out_dir, result, env, results, ctx.shots)
    print("RESULT " + json.dumps(result))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
