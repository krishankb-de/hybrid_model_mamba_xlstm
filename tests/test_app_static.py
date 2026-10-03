"""CHAT_UI_PLAN.md P4-A: the page shell (app/static/index.html and styles.css) and how the server serves it."""
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import server
from app.server import create_app

STATIC = Path(__file__).resolve().parents[1] / "app" / "static"
BANNER = "Research prototype — not for clinical use."
TOKENS = ["--bg", "--fg", "--muted", "--accent", "--card", "--border", "--ok", "--warn", "--bad"]
IDS = ["mode-badge", "health", "sidebar", "new-session", "session-list", "conversation", "drawer", "composer",
       "image-well", "preview", "prompt", "chips", "settings", "send", "stop", "file", "viewer", "sidebar-toggle"]
HIDDEN_AT_START = ["drawer", "preview", "stop", "file", "viewer"]   # the scripts show these by clearing `hidden`
DARK = r"@media\s*\(\s*prefers-color-scheme:\s*dark\s*\)"
NARROW = r"@media\s*\(\s*max-width:\s*800px\s*\)"


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(engine="tiny", home=str(tmp_path))) as c:
        yield c


def test_index_is_served_and_has_the_banner(client):
    r = client.get("/")
    assert r.status_code == 200 and "Research prototype — not for clinical use." in r.text


def test_page_makes_no_external_requests():
    import re
    from pathlib import Path
    static = Path("app/static")
    for f in static.glob("*"):
        text = f.read_text()
        assert not re.search(r"https?://", text), f
        assert "EventSource" not in text, f


# ---- beyond the brief: what P4-B..D and the 375 px target depend on -------------------------------------------------

class _Tags(HTMLParser):
    """index.html as (tag, attributes) pairs in document order; a valueless attribute such as `hidden` maps to ''."""

    def __init__(self, text):
        super().__init__()
        self.tags = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, {k: v or "" for k, v in attrs}))


def _html():
    return (STATIC / "index.html").read_text(encoding="utf-8")


def _tags():
    return _Tags(_html()).tags


def _by_id():
    return {a["id"]: (t, a) for t, a in _tags() if "id" in a}


def _rules(css):
    """A stylesheet's top-level (prelude, body) pairs, braces balanced, comments dropped. An at-rule's body is itself a
    stylesheet, so _rules(body) reads the rules inside a @media block."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    rules, depth, start, head_from, head = [], 0, 0, 0, ""
    for i, ch in enumerate(css):
        if ch == "{":
            if depth == 0:
                head, start = css[head_from:i].strip(), i + 1
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                rules.append((head, css[start:i]))
                head_from = i + 1
    return rules


def _stylesheet():
    css = re.sub(r"/\*.*?\*/", "", (STATIC / "styles.css").read_text(encoding="utf-8"), flags=re.S)
    assert css.count("{") == css.count("}")   # _rules would silently drop what follows an unbalanced brace
    return _rules(css)


def _media(query):
    """The rules inside every @media block whose query matches the pattern."""
    return [rule for head, body in _stylesheet() if re.fullmatch(query, head) for rule in _rules(body)]


def _declared(rules):
    """{custom property: value} over the :root rules among (selector, declarations) pairs."""
    return {k: v.strip() for head, body in rules if head == ":root" for k, v in re.findall(r"(--[\w-]+)\s*:\s*([^;]+);", body)}


def test_the_page_is_what_the_server_serves(client):
    assert server.STATIC_DIR == STATIC
    page, css = client.get("/"), client.get("/static/styles.css")
    assert page.status_code == 200 and page.headers["content-type"].startswith("text/html")
    assert page.content == (STATIC / "index.html").read_bytes()   # the file, not the placeholder
    assert css.status_code == 200 and css.headers["content-type"].startswith("text/css")
    assert css.content == (STATIC / "styles.css").read_bytes()


def test_static_files_are_self_contained():
    files = [p for p in STATIC.rglob("*") if p.is_file()]
    assert {"index.html", "styles.css"} <= {p.name for p in files}   # the brief's scan passes on an empty directory
    forbidden = [r"https?://", r"EventSource", r"<svg", r"@import", r"@font-face",    # no CDN, no web font, no SVG icon
                 r"""(?:src|href)\s*=\s*["']\s*//""", r"""url\(\s*["']?\s*//"""]   # nor a protocol-relative URL
    for p in files:
        assert p.suffix != ".svg", p
        if p.suffix in (".html", ".css", ".js", ".mjs"):
            text = p.read_text(encoding="utf-8")
            for pattern in forbidden:
                assert not re.search(pattern, text), (p.name, pattern)


def test_every_element_id_the_scripts_use_is_there_exactly_once():
    ids = Counter(a["id"] for _, a in _tags() if "id" in a)
    assert {i: ids[i] for i in IDS if ids[i] != 1} == {}
    assert max(ids.values()) == 1
    tag_of = {"sidebar": "nav", "conversation": "main", "drawer": "aside", "composer": "form", "session-list": "ol",
              "prompt": "textarea", "file": "input", "send": "button", "stop": "button", "settings": "button",
              "new-session": "button", "sidebar-toggle": "button"}
    by_id = _by_id()
    assert {i: by_id[i][0] for i in tag_of} == tag_of


def test_initial_state_and_aria_the_scripts_rely_on():
    by_id = _by_id()
    for ident in HIDDEN_AT_START:
        assert "hidden" in by_id[ident][1], ident
    assert by_id["file"][1]["type"] == "file" and by_id["send"][1]["type"] == "submit"
    assert by_id["conversation"][1]["aria-live"] == "polite" and by_id["health"][1]["aria-live"] == "polite"
    assert by_id["settings"][1]["aria-controls"] == "drawer"
    assert by_id["viewer"][1]["role"] == "dialog" and by_id["viewer"][1]["aria-modal"] == "true"


def test_banner_is_one_string_with_the_sidebar_toggle_first():
    html = _html()
    banner = re.search(r'<header class="banner" role="note">(.*?)</header>', html, re.S).group(1)
    assert banner.lstrip().startswith('<button id="sidebar-toggle"')
    assert BANNER in banner   # one unbroken string, em dash included
    assert _by_id()["sidebar-toggle"][1] == {"id": "sidebar-toggle", "type": "button", "aria-controls": "sidebar",
                                              "aria-expanded": "false", "aria-label": "Sessions"}
    assert re.search(r'id="sidebar-toggle"[^>]*>☰</button>', html)   # a Unicode glyph, never an SVG


def test_page_declares_a_viewport_and_loads_only_its_own_files():
    tags = _tags()
    viewport = [a["content"] for t, a in tags if t == "meta" and a.get("name") == "viewport"]
    assert viewport and "width=device-width" in viewport[0] and "initial-scale=1" in viewport[0]
    assert ("html", {"lang": "en"}) in tags
    loaded = [a.get("href") or a.get("src") for t, a in tags if t in ("link", "script")]
    assert {"/static/styles.css", "/static/app.js"} <= set(loaded)
    assert all(url.startswith("/static/") for url in loaded)
    assert [a["type"] for t, a in tags if t == "script"] == ["module"]


def test_colour_tokens_are_on_root_and_redefined_for_dark():
    light, dark = _declared(_stylesheet()), _declared(_media(DARK))
    assert set(TOKENS) <= set(light), sorted(set(TOKENS) - set(light))
    assert set(TOKENS) <= set(dark), sorted(set(TOKENS) - set(dark))
    assert all(light[t] != dark[t] for t in ("--bg", "--fg", "--card", "--border"))   # a second palette, not a copy


def _luminance(colour):
    r, g, b = [int(colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    r, g, b = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(a, b):
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_text_and_control_colours_are_readable_in_both_palettes(scheme):
    tokens = _declared(_stylesheet() if scheme == "light" else _media(DARK))
    pairs = [("--fg", "--bg", 7), ("--fg", "--card", 7), ("--muted", "--bg", 4.5), ("--muted", "--card", 4.5),
             ("--accent", "--bg", 4.5), ("--accent", "--card", 4.5), ("--on-accent", "--accent", 4.5),
             ("--ok", "--bg", 4.5), ("--ok", "--card", 4.5), ("--warn", "--bg", 4.5), ("--warn", "--card", 4.5),
             ("--bad", "--bg", 4.5), ("--bad", "--card", 4.5), ("--control", "--bg", 3), ("--control", "--card", 3)]
    for fore, back, least in pairs:   # WCAG 2: 7 for body text, 4.5 for text, 3 for the border of a control
        for name in (fore, back):
            assert re.fullmatch(r"#[0-9a-f]{6}", tokens[name]), (scheme, name)   # #rrggbb keeps this check honest
        assert _contrast(tokens[fore], tokens[back]) >= least, (scheme, fore, back)


def test_body_sets_an_explicit_background_and_colour():
    body = " ".join(b for h, b in _stylesheet() if h == "body")
    assert re.search(r"(?<![\w-])background(-color)?:\s*var\(--bg\)", body)
    assert re.search(r"(?<![\w-])color:\s*var\(--fg\)", body)


def test_the_hidden_attribute_beats_display_rules():
    hidden = [b for h, b in _stylesheet() if h == "[hidden]"]   # else `#drawer { display: ... }` would show it
    assert hidden and re.search(r"display:\s*none\s*!important", hidden[0])


def test_layout_collapses_below_800px_to_one_column_with_a_16px_gutter():
    css, narrow = _stylesheet(), _media(NARROW)
    assert narrow, "no @media (max-width: 800px)"
    tokens = _declared(css)
    assert (tokens["--sidebar-w"], tokens["--drawer-w"]) == ("260px", "320px")
    assert re.search(r"--gutter:\s*16px", " ".join(b for h, b in narrow))
    assert any("sidebar-open" in h for h, b in narrow)   # P4-D sets body.sidebar-open to show the off-canvas sidebar
    assert any(h == "#sidebar-toggle" and re.search(r"display:\s*(inline-)?flex", b) for h, b in narrow)
    assert any(h == "#sidebar-toggle" and re.search(r"display:\s*none", b) for h, b in css)   # gone at 801 px and up


def test_long_words_wrap_focus_is_visible_and_the_banner_sticks():
    css = _stylesheet()
    assert re.search(r"overflow-wrap:\s*(anywhere|break-word)", " ".join(b for h, b in css))
    assert any(":focus-visible" in h for h, b in css)
    assert any(h == ".banner" and re.search(r"position:\s*sticky", b) for h, b in css)
