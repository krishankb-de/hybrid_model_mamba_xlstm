"""CHAT_UI_PLAN.md P4-E: the parts of the browser checklist (scripts/chat_ui_browser_check.py) that are plain Python: the synthetic
labelled turn it seeds, the accessibility-tree reader, and the consistency of its own lists. No Chrome and no app process.

The run itself needs Chrome and is a local tool (venv/bin/python scripts/chat_ui_browser_check.py); validate.sh does not run it.
Synthetic data only: the seeded turn says so in its report, its model card and its file name.
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from app.server import create_app
from scripts import chat_ui_browser_check as check

SCRIPT = Path(check.__file__)


# ---- the seeded turn: the same replay path as any stored chat ------------------------------------------------------------------------

@pytest.fixture
def seeded(tmp_path):
    check.seed_labelled_home(str(tmp_path))
    with TestClient(create_app(engine="tiny", home=str(tmp_path))) as client:
        yield client


def test_the_seeded_turn_is_one_finished_chat_the_server_lists_and_replays(seeded):
    sessions = seeded.get("/v1/sessions").json()["sessions"]
    assert len(sessions) == 1 and sessions[0]["title"] == "synthetic.png" and sessions[0]["turns"] == 1
    session = seeded.get("/v1/sessions/" + sessions[0]["id"]).json()
    assert [m["role"] for m in session["messages"]] == ["user", "assistant"]
    assistant = session["messages"][1]
    assert assistant["status"] == "done"
    log = seeded.get("/v1/messages/{}?after=0".format(assistant["id"])).json()
    names = [e["event"] for e in log["events"]]
    assert [e["seq"] for e in log["events"]] == list(range(1, len(names) + 1))   # 1, 2, ... with no gap
    assert names[0] == "message_start" and names[-1] == "message_stop"
    assert [e["data"]["stage"] for e in log["events"] if e["event"] == "stage_end"] == check.STAGES   # every stage ends once, in order
    assert log["events"][-1]["data"]["status"] == "done"


def test_the_seeded_turn_carries_the_labels_the_chips_are_checked_against(seeded):
    sessions = seeded.get("/v1/sessions").json()["sessions"]
    message = seeded.get("/v1/sessions/" + sessions[0]["id"]).json()["messages"][1]["id"]
    events = {(e["event"], e["data"].get("stage")): e["data"] for e in seeded.get("/v1/messages/{}?after=0".format(message)).json()["events"]}
    labels = events[("stage_end", "label")]["detail"]["chexbert_14"]
    reference = events[("stage_end", "score")]["detail"]["reference_chexbert_14"]
    assert list(labels) == check.CHEXBERT_14 and list(reference) == check.CHEXBERT_14   # the 14 names, in the labeller's order
    assert sorted(n for n, v in labels.items() if v) == sorted(check.SYNTHETIC_POSITIVE)
    assert sorted(n for n, v in reference.items() if v) == sorted(check.SYNTHETIC_REFERENCE)
    assert sorted(set(check.SYNTHETIC_POSITIVE) ^ set(check.SYNTHETIC_REFERENCE)) == ["Edema", "Pleural Effusion"]   # the two chips that disagree


def test_nothing_in_the_seeded_turn_looks_like_model_output_or_mimic(seeded):
    sessions = seeded.get("/v1/sessions").json()["sessions"]
    export = seeded.get("/v1/sessions/{}/export?format=json".format(sessions[0]["id"])).text
    assert "SYNTHETIC" in export
    for forbidden in ("study_id", "subject_id", "gallery_row", "mimic"):
        assert forbidden not in export.lower(), forbidden
    assert check.SYNTHETIC_REPORT.startswith("Findings: SYNTHETIC")


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
    states = [{"value": v, "focused": True, "stored": s, "chips": ["beam 3", c], "error": {"hidden": h, "text": check.BUDGET_ERROR}, "invalid": i}
              for v, s, c, h, i in (("3", 120, "120 tok", False, "true"), ("30", 30, "30 tok", True, None), ("300", 30, "30 tok", False, "true"))]
    assert [check.budget_chip(t["chips"]) for t in states] == ["120 tok", "30 tok", "30 tok"]
    assert check.errors_shown(states) == [True, False, True]   # the line is shown while the text is not a whole number inside the bounds
    assert check.errors_shown([{"error": {"hidden": True}}]) == [False]


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
