"""CHAT_UI_PLAN.md P4-E: the parts of the browser checklist (scripts/chat_ui_browser_check.py) that are plain Python: the labelled turn
its a11y check runs (P5-E: live, on a tiny app with the tiny gallery), the accessibility-tree reader, and the consistency of its own
lists. No Chrome and no app process.

The run itself needs Chrome and is a local tool (venv/bin/python scripts/chat_ui_browser_check.py); validate.sh does not run it.
Synthetic data only: the tiny engine, the tiny gallery, and a reference written for the check.
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from app.commands import parse_command
from app.labels import CHEXBERT_14
from app.server import create_app
from scripts import chat_ui_browser_check as check
from scripts import chat_ui_cdp
from tests import chat_ui_url_cases as url_cases
from tests.app_helpers import iter_sse, png_bytes

SCRIPT = Path(check.__file__)


# ---- the labelled turn of the a11y check: live, on a tiny app with the tiny gallery (P5-E) ------------------------------------------

@pytest.fixture
def labelled(tmp_path):
    with TestClient(create_app(engine="tiny", home=str(tmp_path), tiny_gallery=True)) as client:
        yield client


def test_the_note_of_the_labelled_turn_is_a_reference_command():
    assert parse_command("reference: " + check.LIVE_REFERENCE) == {"reference": check.LIVE_REFERENCE}
    assert "SYNTHETIC" not in check.LIVE_REFERENCE and len(check.LIVE_REFERENCE) < 200   # written for the check: no MIMIC text


def test_the_labelled_turn_runs_every_stage_and_scores_its_report_against_that_reference(labelled):
    sid = labelled.post("/v1/sessions", json={}).json()["id"]
    r = labelled.post("/v1/sessions/{}/messages".format(sid), files={"image": ("xray_a.png", png_bytes(320, 320), "image/png")},
                      data={"text": "reference: " + check.LIVE_REFERENCE, "options": json.dumps({"max_new_tokens": 24})})
    frames = list(iter_sse([r.text]))
    ends = {f["data"]["stage"]: f["data"] for f in frames if f["event"] == "stage_end"}
    assert [stage for stage in check.STAGES if "skipped" in ends[stage]] == []   # every stage ran, so every chip has its reference
    assert list(ends["label"]["detail"]["chexbert_14"]) == CHEXBERT_14
    score = ends["score"]["detail"]
    assert score["reference_source"] == "user" and list(score["reference_chexbert_14"]) == CHEXBERT_14
    assert sorted(n for n, v in score["reference_chexbert_14"].items() if v) == ["Cardiomegaly", "Pleural Effusion", "Support Devices"]


# ---- the a11y item on a server with a gallery: the stage buttons are counted on the page, and expected from the server's features ----------

def test_the_stages_with_a_button_are_those_the_server_runs():
    assert check.detailed_stages({"retrieval": False, "labels": False}) == ["preprocess", "encode", "generate"]   # the app of the checks
    assert check.detailed_stages({"retrieval": True, "labels": True}) == ["preprocess", "encode", "retrieve", "generate", "label"]   # a tiny_gallery server
    assert check.detailed_stages({"retrieval": True, "labels": False}) == ["preprocess", "encode", "retrieve", "generate"]
    assert check.detailed_stages({"retrieval": False, "labels": True}) == ["preprocess", "encode", "generate", "label"]
    assert check.detailed_stages({"retrieval": True, "labels": True}, reference=True) == check.STAGES   # a reference makes score run too
    assert check.detailed_stages({}) == ["preprocess", "encode", "generate"]   # a server that says nothing of its features runs none of them


@pytest.mark.parametrize("tiny_gallery", [False, True], ids=["no-gallery-server", "gallery-server"])
def test_the_stages_that_have_a_detail_in_a_live_turn_are_the_stages_the_check_expects_buttons_for(tmp_path, tiny_gallery):
    """Both servers of the a11y item, live: a turn's stages with a detail table (render.js draws a button for each) are exactly what
    detailed_stages says for that server's /v1/models features, and the others are skipped (the item reads them as such)."""
    with TestClient(create_app(engine="tiny", home=str(tmp_path), tiny_gallery=tiny_gallery)) as client:
        features = client.get("/v1/models").json()["features"]
        sid = client.post("/v1/sessions", json={}).json()["id"]
        r = client.post("/v1/sessions/{}/messages".format(sid), files={"image": ("xray_a.png", png_bytes(320, 320), "image/png")},
                        data={"text": "", "options": json.dumps({"max_new_tokens": 24})})
    ends = {f["data"]["stage"]: f["data"] for f in iter_sse([r.text]) if f["event"] == "stage_end"}
    assert features == {"retrieval": tiny_gallery, "labels": tiny_gallery}
    detailed = [stage for stage in check.STAGES if ends[stage].get("detail")]
    assert detailed == check.detailed_stages(features)
    assert [stage for stage in check.STAGES if "skipped" in ends[stage]] == [s for s in check.STAGES if s not in detailed]


class FakeDom:
    """The DOM end of the a11y item: DOM.describeNode answers with the flat attribute list of the element a backend id names."""

    def __init__(self, attributes: Dict[int, List[str]]) -> None:
        self.attributes = attributes

    def call(self, method: str, **params: Any) -> Dict[str, Any]:
        assert method == "DOM.describeNode"
        return {"node": {"attributes": self.attributes.get(params["backendNodeId"], [])}}


def test_a_focus_target_that_no_key_reaches_is_no_control_that_needs_a_name():
    """A server with a gallery shows matching reports, whose text is a <p tabindex="-1" class="clamp-text">: the page focuses it after Show
    less (render.js clampedText) and no Tab stops on it. Chrome's tree calls it focusable and a paragraph has no name of its own, so the item
    that wants a name of everything focusable failed on such a server, on three of them (one per matching report), before it reached the
    stage buttons. A focusable thing that a key does reach, with no name, is still one."""
    target = ["class", "clamp-text clamped", "tabindex", "-1", "data-action", "clamp"]
    assert check.focus_target_only(target, "paragraph") is True
    assert check.focus_target_only(["tabindex", " -2 "], "generic") is True
    for attributes in (["tabindex", "0"], ["id", "image-well", "role", "button", "tabindex", "0"], ["class", "x"], [], ["tabindex", "-1x"], ["tabindex", ""]):
        assert check.focus_target_only(attributes, "paragraph") is False, attributes
    nodes = [_node("1", "paragraph", backend=10, focusable=True),            # a focus target: tabindex -1
             _node("2", "paragraph", backend=11, focusable=True),            # a Tab stop with no name
             _node("3", "button", backend=12, focusable=True),               # a control with no name
             _node("4", "button", "Send", backend=13, focusable=True)]       # named
    dom = FakeDom({10: ["class", "clamp-text", "tabindex", "-1"], 11: ["tabindex", "0"], 12: [], 13: []})
    unnamed, left_out = check.unnamed_controls(dom, nodes)
    assert unnamed == [("paragraph", 11), ("button", 12)] and left_out == [("paragraph", 10)]


def test_only_a_role_that_is_no_control_is_a_focus_target_that_needs_no_name():
    """P6-F fix 1. The exemption was any unnamed node with a negative tabindex, whatever it was, so an unnamed <button tabindex="-1"> (a roving
    control that a key reaches by an arrow) passed 'every control has a name'. It is now only for a role that is not a control: a control, and a
    role that nobody has said is not one (the list is closed), is reported."""
    target = ["class", "clamp-text", "tabindex", "-1"]
    for role in ("paragraph", "generic", "heading", "group", "region", "main", "article", "list", "listitem", "figure", "StaticText"):
        assert check.focus_target_only(target, role) is True, role
    for role in ("button", "link", "textbox", "searchbox", "checkbox", "radio", "combobox", "listbox", "option", "slider", "spinbutton", "switch", "tab",
                 "menuitem", "menuitemcheckbox", "treeitem", "gridcell", "", "somethingnew"):
        assert check.focus_target_only(target, role) is False, role
    nodes = [_node("1", "button", backend=20, focusable=True), _node("2", "paragraph", backend=21, focusable=True)]
    unnamed, left_out = check.unnamed_controls(FakeDom({20: ["tabindex", "-1"], 21: ["tabindex", "-1"]}), nodes)
    assert unnamed == [("button", 20)] and left_out == [("paragraph", 21)]   # the roving button is a control without a name


@pytest.mark.parametrize("value", ["0", "-0", "-00", "+1", "1", "-", "- 1", "-1x", "-1.5", "", "−1"])   # the last is a Unicode minus
def test_only_a_whole_number_below_zero_is_a_tabindex_that_no_key_reaches(value):
    """'-0' is zero, which is a Tab stop (an unnamed element with it is one a keyboard user lands on), and so is anything that is not a whole number
    below zero. The regex was -\\d+, which let '-0' through."""
    assert check.focus_target_only(["tabindex", value], "paragraph") is False, value
    assert check.focus_target_only(["tabindex", " -1 "], "paragraph") is True


def test_the_contrast_audit_reports_what_a_user_can_see_and_not_text_made_for_a_screen_reader():
    """Chrome's audit names a node by its tag and classes. A .visually-hidden span (text for a screen reader, drawn as one clipped pixel) has no
    pixels of its own, and the audit reads it against the page's ground and not against the chip it sits in: the ": positive" of a label chip
    on a server with a gallery came out at 1.00:1 in light and 1.16:1 in dark, while the chip's own text, which is seen, was fine."""
    assert check.screen_reader_only("span.visually-hidden") is True
    assert check.screen_reader_only("p.note.visually-hidden") is True
    for selector in ("li.chip.label.positive", "span.visually-hidden-not", "span.not-visually-hidden", "span#contrast-canary", ""):
        assert check.screen_reader_only(selector) is False, selector
    found = [{"selector": "span.visually-hidden", "contrastRatio": 1.0, "thresholdAA": 4.5},
             {"selector": "li.chip.label.positive", "contrastRatio": 3.0, "thresholdAA": 4.5},
             {"selector": "span#contrast-canary", "contrastRatio": 1.05, "thresholdAA": 4.5}]
    assert check.faint_texts(found) == ["li.chip.label.positive 3.00:1 (needs 4.5)"]   # the canary is the audit's own, the hidden span is not seen
    assert check.faint_texts([found[0], found[2]]) == []


def test_the_a11y_item_counts_the_stage_buttons_the_page_has_and_no_longer_a_fixed_three():
    import inspect
    source = inspect.getsource(check.check_a11y)
    assert "detailed_stages(" in source and "DETAILED" not in source
    assert not hasattr(check, "DETAILED")


# ---- --only: some checks, by name (the reload step is shown stable by running it on its own, many times) -------------------------------------

def test_select_checks_takes_names_and_keeps_the_order_of_the_checklist():
    assert [name for name, _ in check.select_checks(None)] == [name for name, _ in check.CHECKS]
    assert [name for name, _ in check.select_checks("reload")] == ["reload"]
    assert [name for name, _ in check.select_checks("stop, reload")] == ["reload", "stop"]   # the order of the checklist, not of the request
    with pytest.raises(ValueError, match="no check named reloads"):
        check.select_checks("reloads")
    with pytest.raises(ValueError, match="no check named"):
        check.select_checks("")   # nothing named is not "everything"


def test_only_runs_no_evidence_because_a_part_of_the_checklist_would_replace_all_of_it(capsys):
    assert check.parse_args(["--only", "reload", "--no-evidence"]).only == "reload"
    assert check.parse_args([]).only is None
    with pytest.raises(SystemExit) as refused:
        check.parse_args(["--only", "reload"])
    assert refused.value.code == 2 and "--no-evidence" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        check.parse_args(["--only", "nonesuch", "--no-evidence"])


# ---- a --url: the app is someone else's, so nothing of this run is evidence, and it is the tiny app or it is refused (P6-F fix 1) ---------

def test_a_url_needs_no_evidence_because_the_pictures_and_the_checklist_would_be_evidence_of_an_app_the_check_did_not_start(capsys):
    with pytest.raises(SystemExit) as refused:
        check.parse_args(["--url", "http://127.0.0.1:51234/"])
    assert refused.value.code == 2
    assert chat_ui_cdp.URL_NEEDS_NO_EVIDENCE in capsys.readouterr().err
    assert check.parse_args(["--url", "http://127.0.0.1:51234/", "--no-evidence"]).url == "http://127.0.0.1:51234/"
    assert check.parse_args([]).url is None


@pytest.mark.parametrize("port", [8000, 8010])
def test_the_browser_check_refuses_the_dev_servers_and_the_tunnels_port_before_it_sends_or_starts_anything(monkeypatch, capsys, port):
    url_cases.refuses_a_port(check, monkeypatch, capsys, port, ("--no-evidence",))


def test_the_browser_check_refuses_an_address_that_is_not_this_machines_loopback_before_it_sends_or_starts_anything(monkeypatch, capsys):
    url_cases.refuses_a_host(check, monkeypatch, capsys, ("--no-evidence",))


def test_the_browser_check_refuses_an_app_that_is_not_the_tiny_one_with_a_fixed_line_before_its_warm_up_turn(monkeypatch, capsys):
    url_cases.refuses_a_server_that_is_not_the_tiny_app(check, monkeypatch, capsys, ("--no-evidence",))


def test_the_browser_check_goes_on_with_the_tiny_app(monkeypatch, capsys):
    url_cases.goes_on_with_the_tiny_app(check, monkeypatch, capsys, ("--no-evidence",))


# ---- the reload step holds its turn instead of racing it ------------------------------------------------------------------------------

def test_the_reload_step_holds_the_turn_before_it_sends_it_and_always_releases_it():
    """The step reloads in the middle of a turn, and the turn used to be a race (a 150-token turn of the 20 ms engine is over in 3 s, and a
    reload on a loaded machine can take longer than that). The app the check owns holds its turns (App.hold): held before the turn is
    sent, released in a finally, so that a step that fails while it holds one does not leave every later check waiting on it."""
    import ast
    import inspect
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(check.check_reload)))

    def calls(node: Any, name: str) -> List[Any]:
        return [c for c in ast.walk(node) if isinstance(c, ast.Call) and ast.unparse(c.func) == name]

    [hold] = calls(tree, "ctx.app.hold")
    [release] = calls(tree, "ctx.app.release")
    in_finally = [t for t in ast.walk(tree) if isinstance(t, ast.Try) and any(release in ast.walk(s) for s in t.finalbody)]
    assert in_finally, "ctx.app.release() is not in a finally"
    sends = [c for c in calls(tree, "ctx.send") if "tokens 150" in ast.unparse(c)]
    assert len(sends) == 1 and hold.lineno < sends[0].lineno, "the turn is sent before it is held"
    assert hold.lineno < in_finally[0].lineno, "the hold is not made before the try that releases it"


# ---- run_check and drain_console: the one implementation of the browser check and the features check (P6-F fix 1) ----------------------------

class _Page:
    """The browser end of a check's run: evaluate() answers, and the DevTools events the page raised are given out once, by method."""

    def __init__(self, events: Any = ()) -> None:
        self.recorded = list(events)   # (method, params)
        self.asked = []  # type: List[str]

    def evaluate(self, expression: str) -> Any:
        self.asked.append(expression)
        return None

    def events(self, method: str) -> List[Dict[str, Any]]:
        taken = [params for m, params in self.recorded if m == method]
        self.recorded = [(m, params) for m, params in self.recorded if m != method]
        return taken


class _Ctx:
    def __init__(self, events: Any = ()) -> None:
        self.browser = _Page(events)
        self.console = []  # type: List[str]


def _console(kind: str, text: str) -> Any:
    return ("Runtime.consoleAPICalled", {"type": kind, "args": [{"value": text}]})


def test_the_console_is_read_once_after_an_answer_from_the_page_and_only_errors_and_exceptions_are_kept():
    ctx = _Ctx([_console("log", "fine"), _console("error", "bad " + "x" * 300), _console("warning", "meh"),
                ("Runtime.exceptionThrown", {"exceptionDetails": {"exception": {"description": "TypeError: no"}, "text": "Uncaught"}}),
                ("Runtime.exceptionThrown", {"exceptionDetails": {"text": "Uncaught (in promise)"}})])
    check.drain_console(ctx.browser, ctx.console)
    assert ctx.browser.asked == ["0"]   # an answer from the page first: the events that came before it are in the record
    assert ctx.console == ["console.error: bad " + "x" * 196, "uncaught: TypeError: no", "uncaught: Uncaught (in promise)"]   # 200 characters of what was logged
    check.drain_console(ctx.browser, ctx.console)
    assert len(ctx.console) == 3   # given out once


def test_a_check_that_passes_is_one_entry_and_one_line(capsys):
    entry = check.run_check(_Ctx(), "stream", lambda ctx: ("six stages", {"snapshots": 30}))
    assert list(entry) == ["name", "status", "detail", "seconds", "data", "console_errors"]
    assert (entry["name"], entry["status"], entry["detail"], entry["data"], entry["console_errors"]) == ("stream", "PASS", "six stages", {"snapshots": 30}, [])
    assert capsys.readouterr().out == "CHECK stream PASS six stages\n"


def test_a_claim_is_kept_in_the_entry_after_the_name_when_one_is_given():
    entry = check.run_check(_Ctx(), "attach", lambda ctx: ("ok", {}), claim="click, drop and paste")
    assert list(entry) == ["name", "claim", "status", "detail", "seconds", "data", "console_errors"] and entry["claim"] == "click, drop and paste"


def test_a_failed_expectation_and_any_other_exception_are_a_failed_check_and_not_a_crashed_run(capsys):
    def fails(ctx: Any) -> Any:
        raise check.Failure("the card is not done")

    def crashes(ctx: Any) -> Any:
        raise TimeoutError("waited 10 s for the turn")

    assert [e["detail"] for e in (check.run_check(_Ctx(), "a", fails), check.run_check(_Ctx(), "b", crashes))] == [
        "the card is not done", "TimeoutError: waited 10 s for the turn"]
    assert capsys.readouterr().out == "CHECK a FAIL the card is not done\nCHECK b FAIL TimeoutError: waited 10 s for the turn\n"


def test_a_page_that_logged_an_error_fails_a_check_that_passed_and_only_the_checks_own_errors_count(capsys):
    ctx = _Ctx()
    ctx.console.append("console.error: from an earlier check")   # not this check's
    ctx.browser.recorded.append(_console("error", "boom"))
    entry = check.run_check(ctx, "reload", lambda c: ("ok", {}))
    assert entry["status"] == "FAIL" and entry["detail"] == "the page logged 1 error(s): console.error: boom"
    assert entry["console_errors"] == ["console.error: boom"]


def test_a_console_that_cannot_be_read_is_an_error_of_the_check_too(capsys):
    class Mute(_Page):
        def evaluate(self, expression: str) -> Any:
            raise RuntimeError("the page is gone")

    ctx = _Ctx()
    ctx.browser = Mute()
    entry = check.run_check(ctx, "a", lambda c: ("ok", {}))
    assert entry["status"] == "FAIL" and entry["console_errors"] == ["could not read the console: the page is gone"]


def test_what_runs_before_a_check_is_inside_its_guard_and_comes_first(capsys):
    order = []  # type: List[str]
    entry = check.run_check(_Ctx(), "a", lambda c: order.append("check") or ("ok", {}), before=lambda c: order.append("before"))
    assert order == ["before", "check"] and entry["status"] == "PASS"

    def breaks(c: Any) -> None:
        raise RuntimeError("could not reset")

    entry = check.run_check(_Ctx(), "b", lambda c: order.append("never") or ("ok", {}), before=breaks)
    assert order == ["before", "check"] and entry["status"] == "FAIL" and entry["detail"] == "RuntimeError: could not reset"


def test_how_much_of_a_detail_is_printed_and_kept_is_the_checks_to_say(capsys):
    long = "word " * 400
    entry = check.run_check(_Ctx(), "a", lambda c: (long, {}))
    printed = capsys.readouterr().out
    assert len(printed.split(" ", 3)[3].strip()) == 220 and len(entry["detail"]) == 400   # the browser check's own limits
    entry = check.run_check(_Ctx(), "b", lambda c: (long, {}), keep=1500, failure_chars=1500)
    assert len(capsys.readouterr().out.split(" ", 3)[3].strip()) == 220 and len(entry["detail"]) == 1500   # a pass says 220 whatever
    check.run_check(_Ctx(), "c", lambda c: (_ for _ in ()).throw(check.Failure(long)), failure_chars=1500)
    assert len(capsys.readouterr().out.split(" ", 3)[3].strip()) == 1500   # and a failure says all of what it knows


# ---- the keyboard item walks the tab order of a page that is the browser's active one ----------------------------------------------------

def test_the_keyboard_item_brings_the_page_to_the_front_before_it_presses_a_tab():
    """The item asserts that after the last control Tab leaves the page for the browser's own controls (the page's active element is then the
    body). Chrome does that only when its window is the active one. When it is not, Tab wraps to the first control instead and the item says
    'after the last control Tab stays in the page: #new-session'. The item passed on this machine in earlier runs and then failed every run,
    with the same page and the same Chrome flags, so the state of the window is the machine's and not the page's; why it changed is not known.
    Measured on a page of three buttons: without Page.bringToFront the focus after eight Tabs is a b c a b c body a, with it a b c body a b c a.
    The item brings the page to the front before its first Tab, so that it measures the page and not the window."""
    import ast
    import inspect
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(check.check_keyboard)))

    def calls(name: str, first_arg: str) -> List[Any]:
        return [c for c in ast.walk(tree) if isinstance(c, ast.Call) and ast.unparse(c.func) == name and c.args and ast.unparse(c.args[0]) == first_arg]

    fronts = calls("b.call", "'Page.bringToFront'")
    tabs = calls("b.key", "'Tab'")
    assert tabs, "the pin looks at the Tab presses of the item"
    assert len(fronts) == 1, "the item does not bring the page to the front exactly once: {}".format(len(fronts))
    assert fronts[0].lineno < min(t.lineno for t in tabs), "the first Tab is pressed before the page is brought to the front"


# ---- the accessibility-tree reader ----------------------------------------------------------------------------------------------------

class FakeTree:
    def __init__(self, nodes: List[Dict[str, Any]]) -> None:
        self.nodes = nodes

    def call(self, method: str, **params: Any) -> Dict[str, Any]:
        assert method == "Accessibility.getFullAXTree"
        return {"nodes": self.nodes}


def _node(node_id: str, role: str, name: str = "", children: Any = (), ignored: bool = False, backend: Any = None, **props: Any) -> Dict[str, Any]:
    node = {"nodeId": node_id, "ignored": ignored, "role": {"type": "role", "value": role}, "name": {"type": "computedString", "value": name},
            "properties": [{"name": k, "value": {"type": "x", "value": v}} for k, v in props.items()], "childIds": list(children)}
    if backend is not None:
        node["backendDOMNodeId"] = backend
    return node


def test_the_ax_reader_reads_the_text_a_screen_reader_would_and_not_what_is_hidden():
    tree = check.Ax(FakeTree([
        _node("1", "listitem", children=["2", "3", "4"], backend=10),
        _node("2", "StaticText", "Edema"),
        _node("3", "StaticText", ": positive"),                     # visually hidden text: exposed
        _node("4", "generic", children=["5"], ignored=True),         # a wrapper that is ignored: what is in it still counts
        _node("5", "StaticText", ", differs from reference"),
        _node("6", "generic", children=["7"], ignored=True, backend=11),
        _node("7", "StaticText", "✗", ignored=True),                 # aria-hidden: not exposed
        _node("8", "button", "encode · 1 ms, done, turn 1", backend=12, expanded=False),
    ]))
    assert tree.text(tree.by_id["1"]) == "Edema: positive, differs from reference"
    assert tree.text(tree.by_id["6"]) == ""
    button = tree.by_backend[12]
    assert (check.Ax.role(button), check.Ax.name(button), check.Ax.prop(button, "expanded")) == ("button", "encode · 1 ms, done, turn 1", False)
    assert check.Ax.prop(button, "live") is None
    assert tree.by_backend[10] is tree.by_id["1"]


# ---- the lists the script keeps about itself ---------------------------------------------------------------------------------------------

def test_the_checks_are_the_ten_in_order():
    assert [name for name, _ in check.CHECKS] == ["stream", "reload", "sessions", "exports", "stop", "keyboard", "narrow", "a11y", "error",
                                                  "settings"]   # P4-F added the tenth


def test_repeated_counts_the_sentences_that_repeat_an_earlier_one_by_the_repairs_key():
    assert check.repeated("The heart is normal. The lungs are clear.") == 0
    assert check.repeated("The heart is normal. the HEART is   normal. The lungs are clear. The heart is normal.") == 2   # case and whitespace
    assert check.repeated("Findings: no effusion. Findings: no effusion. Findings: no") == 1   # a fragment is not a sentence yet
    assert check.repeated("") == 0


def test_budget_chip_finds_the_token_chip_of_the_composer():
    assert check.budget_chip(["beam 3", "150 tok", "cached"]) == "150 tok"
    assert check.budget_chip(["beam 3", "cached"]) is None


def test_the_settings_check_types_the_budgets_the_brief_names_and_reads_the_notes_the_page_has():
    assert (check.FIRST_BUDGET, check.SECOND_BUDGET) == (150, 200)   # 1 -> 15 -> 150 is the half-typed value of the brief; 200 is the longest run
    source = (Path(__file__).resolve().parents[1] / "app" / "static" / "app.js").read_text()
    for note in (check.APPLY_NOTE, check.RUNNING_NOTE, check.RETRIEVAL_NOTE, check.LABELS_NOTE):
        assert note in source, note   # the check asks the page for exactly the words the page has
    assert check.BUDGET_FIELD == '#drawer input[data-setting="max_new_tokens"]' and 'data-setting' in source


def test_the_words_the_drawer_check_looks_for_are_the_words_of_the_page():   # P4-G
    app = (Path(__file__).resolve().parents[1] / "app" / "static" / "app.js").read_text()
    render = (Path(__file__).resolve().parents[1] / "app" / "static" / "render.js").read_text()
    for words in (check.SAVED_TEXT, check.STOP_LABEL, check.STOP_HINT, check.STORAGE_TEXT):
        assert words in app, words
    head, tail = "Reached the ", "; the unfinished last sentence is hidden (Show raw shows it)."
    assert check.REPEAT_NOTE in render and check.budget_note(123) == head + "123-token budget" + tail   # the card builds it from its pieces
    assert head in render and tail in render and "-token budget" in render
    assert check.BUDGET_ERROR == "Enter a whole number from 16 to 200."   # the server's bounds, in the page's words: from {low} to {high}
    assert "Enter a whole number from ${BOUNDS[key][0]} to ${BOUNDS[key][1]}." in app
    assert check.FIELD_LABELS == ["Beam size (1–8)", "Token budget (16–200)", "Similar images (0–12)", "Matching reports (0–10)"]
    assert check.STOP_SWITCH == '#drawer input[data-setting="stop_on_repeat"]' and check.SAVE_BUTTON == "#drawer-save"
    assert "id: 'drawer-save'" in app and "'stop_on_repeat'" in app


def test_budget_note_is_the_card_note_of_a_budget_stop_with_repair_on():
    assert check.budget_note(200) == "Reached the 200-token budget; the unfinished last sentence is hidden (Show raw shows it)."


def test_the_drawer_check_reads_a_typed_field_as_the_page_should_have_it():
    # "300" typed key by key over 120: 3 is not a number the setting can take, 30 is (it applies), 300 is not (it goes back to the 120)
    states = [{"value": v, "focused": True, "stored": s, "chips": ["beam 3", c], "error": {"hidden": h, "text": check.BUDGET_ERROR}, "invalid": i}
              for v, s, c, h, i in (("3", 120, "120 tok", False, "true"), ("30", 30, "30 tok", True, None), ("300", 120, "120 tok", False, "true"))]
    assert [check.budget_chip(t["chips"]) for t in states] == ["120 tok", "30 tok", "120 tok"]
    assert check.errors_shown(states) == [True, False, True]   # the line is shown while the text is not a whole number inside the bounds
    assert check.errors_shown([{"error": {"hidden": True}}]) == [False]


def test_a_wrong_text_leaves_the_setting_where_it_was_before_the_typing_began():   # P4-G fix 1
    states = [{"stored": 120, "chips": ["beam 3", "120 tok"]}, {"stored": 30, "chips": ["beam 3", "30 tok"]}, {"stored": 120, "chips": ["beam 3", "120 tok"]}]
    assert check.settled_where_it_began(states, 120) is True
    assert check.settled_where_it_began(states[:2], 120) is False   # the 30 on the way is not where it began
    assert check.settled_where_it_began([dict(states[-1], stored=30)], 120) is False   # chips back, the stored copy left at 30
    assert check.settled_where_it_began([dict(states[-1], chips=["beam 3", "30 tok"])], 120) is False   # stored back, the chips left at 30
    assert check.settled_where_it_began([], 120) is False   # nothing typed is nothing shown


def test_the_drawer_error_picture_asks_for_the_chips_it_captions():
    rule = check.SHOT_RULES["drawer_error"]
    assert "'{} tok'".format(check.ENTER_BUDGET) in rule and "'30 tok'" in rule   # at the value the field began with, and not at the 30 on the way


def test_the_settled_pictures_ask_for_a_turn_run_with_the_pages_own_settings():
    rule = check.SHOT_RULES["settled"]
    assert 'data-stopped="repeat"' in rule and "'full budget'" in rule   # the note that says why it stopped, and no chip that says the switch is off
    assert "#chips span" in rule and ".turn.user .options .chip" in rule   # in the composer and in the user's own turn


def test_the_stream_recorder_counts_a_snapshot_only_while_the_report_is_still_being_written():   # P6-H fix 1 (F)
    """The snapshots the stream check counts are the texts the card showed while its report grew. When the block closes the card draws the
    report as a list of sentences, another string for the same report: counted, it would be one more 'snapshot' and the last of them, which
    is none, and `len(texts) >= 3` would be satisfied by one fewer. So the recorder adds to its texts only a frame whose `.report` is still
    `.provisional` (the card's own word for a report that is being written, render.js renderReport)."""
    adds = [line for line in check.RECORD_JS.splitlines() if "rec.texts.add" in line]
    assert len(adds) == 1 and ".report.provisional" in adds[0], adds
    render = (Path(__file__).resolve().parents[1] / "app" / "static" / "render.js").read_text(encoding="utf-8")
    assert "'report provisional'" in render   # the class the recorder reads is the page's own


def test_the_stop_off_script_writes_the_one_setting_and_nothing_else():
    assert "stop_on_repeat" in check.STOP_OFF_JS and "cxrchat.settings" in check.STOP_OFF_JS
    assert "try" in check.STOP_OFF_JS and "catch" in check.STOP_OFF_JS   # a browser that blocks storage must not break the page
    assert "http" not in check.STOP_OFF_JS


def test_every_screenshot_the_script_takes_has_a_rule_and_a_file_it_declares():
    source = SCRIPT.read_text()
    calls = re.findall(r'ctx\.shot\("(\w+)", "(\w+)", [^\n]*?(DESKTOP|PHONE)(?:, "(\w+)")?\)', source)
    assert calls, "no ctx.shot call found: the pattern is stale"
    files = set()
    for key, name, viewport, scheme in calls:
        assert key in check.SHOT_RULES, "{} has no page rule".format(key)
        width, height = {"DESKTOP": check.DESKTOP, "PHONE": check.PHONE}[viewport]
        files.add("{}_{}x{}_{}.png".format(name, width, height, scheme or "light"))
    assert files == set(check.EVIDENCE_FILES) - {"checklist.json"}, sorted(files ^ set(check.EVIDENCE_FILES))
    assert set(check.SHOT_RULES) == {key for key, *_ in calls}, "a rule that no screenshot uses"
    assert len([f for f in check.EVIDENCE_FILES if f.endswith(".png")]) <= 10   # the brief's limit on the committed pictures (P4-G: at most 10)


def test_each_page_rule_is_a_function_body_that_returns_and_reads_the_page_only():
    for key, rule in check.SHOT_RULES.items():
        assert "return " in rule, key
        assert "http" not in rule and "innerHTML" not in rule and "fetch" not in rule, key   # a rule looks at the page; it changes and asks nothing
        assert rule.count("(") == rule.count(")") and rule.count("'") % 2 == 0, key


def test_the_script_names_no_absolute_path_of_this_machine():
    assert "/Users/" not in SCRIPT.read_text()


def test_one_line_squeezes_whitespace_and_clips_long_text():
    assert check.one_line("a\n  b\tc") == "a b c"
    clipped = check.one_line("x" * 500, 50)
    assert len(clipped) == 50 and clipped.endswith("…")


def test_first_difference_says_where_two_strings_part_and_expect_raises_failure():
    assert "first difference at 3" in check.first_difference("abcdef", "abcXef")
    assert "first difference at 2" in check.first_difference("ab", "abcd")   # one is a prefix of the other
    with pytest.raises(check.Failure, match="nope"):
        check.expect(0, "nope")
    check.expect(1, "never raised")


def test_evidence_file_names_are_the_ones_the_plan_asks_for():
    json.dumps(check.EVIDENCE_FILES)
    assert any(f.startswith("streaming_") for f in check.EVIDENCE_FILES)
    assert {"settled_1280x900_light.png", "settled_1280x900_dark.png", "drawer_1280x900_light.png", "settled_375x812_light.png",
            "stopped_1280x900_light.png", "error_notice_1280x900_light.png", "drawer_error_1280x900_light.png", "checklist.json"} <= set(check.EVIDENCE_FILES)


# ---- P6 fix 1: which pictures belong to the chat on the page (the sessions check's stray-image test) -----------------------------------

def test_a_chats_own_pictures_and_its_similar_xrays_are_not_strays_and_anything_else_is():
    ids = ["m_u1", "m_a1"]
    shown = ["/v1/messages/m_u1/image?variant=thumb", "/v1/messages/m_u1/image?variant=model_input", "/v1/gallery/images/101",
             "/v1/gallery/images/7", "/v1/gallery/images/1011", "/v1/messages/m_u1x/image?variant=thumb",
             "/v1/messages/m_other/image?variant=thumb", "src blob:http://127.0.0.1:1/abc", "src "]
    assert check.strays(shown, ids, [101, 102]) == [
        "/v1/gallery/images/7",                        # a gallery picture this chat never retrieved
        "/v1/gallery/images/1011",                     # not row 101 by prefix
        "/v1/messages/m_u1x/image?variant=thumb",      # not message m_u1 by prefix
        "/v1/messages/m_other/image?variant=thumb",    # another chat's
        "src blob:http://127.0.0.1:1/abc", "src "]     # the page's own preview of a file, or no source at all
    assert check.strays(shown[:2], ids, []) == []
    assert check.strays(["/v1/gallery/images/101"], ids, []) == ["/v1/gallery/images/101"]   # a chat with no retrieval owns no gallery row


def test_the_gallery_rows_a_chat_may_show_are_those_its_retrieve_stages_named():
    def message(*details):
        return {"events": [{"event": "message_start", "data": {"seq": 1}}]
                + [{"event": "stage_end", "data": {"stage": "retrieve", "detail": d, "seq": i + 2}} for i, d in enumerate(details)]}
    private = message({"image_neighbors": [{"rank": 1, "gallery_row": 101}, {"rank": 2, "gallery_row": 7}], "report_matches": []})
    public = message({"image_neighbors": [{"rank": 1, "similarity": 0.9}]})                # rank and similarity only: no row
    skipped = {"events": [{"event": "stage_end", "data": {"stage": "retrieve", "skipped": "gallery_unavailable", "seq": 2}}]}
    odd = message({"image_neighbors": [{"gallery_row": "5"}, {"gallery_row": True}, None, "x"]}, None)
    assert check.retrieved_rows([private, public, skipped, odd, {}, None, {"events": [None, "x", 3]}]) == {101, 7}
