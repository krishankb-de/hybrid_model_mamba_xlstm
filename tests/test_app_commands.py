"""CHAT_UI_PLAN.md P3-D: text commands (spec §4)."""
import json
import re
from pathlib import Path

from app.commands import _RULES, COMMAND_HELP, NOT_A_QA_BOT, WHITESPACE, collapse, is_question, parse_command

COMMANDS_FIXTURE = Path(__file__).resolve().parent / "frontend" / "fixtures" / "commands.json"
WHITESPACE_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "whitespace.json"   # P6-A (f): the page's node test reads it too


def _cases():
    return json.loads(COMMANDS_FIXTURE.read_text(encoding="utf-8"))


def _uncovered(patterns, cases):
    """The patterns that no command of the fixture is an example of: a rule added without a case."""
    commands = [" ".join(c["note"].split()) for c in cases if c["command"]]
    return [p for p in patterns if not any(re.fullmatch(p, note, flags=re.IGNORECASE | re.ASCII) for note in commands)]


def test_the_pages_list_of_commands_is_the_servers():
    """P4-H: the page tells a command from a note (app/static/app.js isCommand) for the hint under the image well. Its node test reads
    the same notes, so the two cannot drift apart without one of them failing."""
    cases = _cases()
    assert len(cases) >= 20 and {c["command"] for c in cases} == {True, False}
    for case in cases:
        assert (parse_command(case["note"]) is not None) is case["command"], case


def test_every_server_command_has_an_example_in_the_shared_fixture():
    """P4-H fix 1: a rule added to _RULES without a case in commands.json fails here, so the page's copy cannot fall behind unseen."""
    assert _uncovered([pattern for pattern, _ in _RULES], _cases()) == []
    assert _uncovered([pattern for pattern, _ in _RULES] + [r"stop (on|off)"], _cases()) == [r"stop (on|off)"]   # the guard bites


def test_commands_read_ascii_digits_and_letters_only_as_the_page_does():
    """P4-H fix 1: Python's \\d also takes other scripts' digits and its case folding takes the long s for an s; JavaScript's does
    neither. The server reads ASCII only, so the page and the server tell a command the same way."""
    assert parse_command("beam ５") is None and parse_command("tokens ٣٠") is None
    assert parse_command("tokenſ 30") is None
    assert parse_command("BEAM 5") == {"decode": "beam", "beam_size": 5}


def test_commands_map_to_options():
    assert parse_command("beam 5") == {"decode": "beam", "beam_size": 5}
    assert parse_command("  GREEDY ") == {"decode": "greedy"}
    assert parse_command("tokens 150") == {"max_new_tokens": 150}
    assert parse_command("retrieve 6") == {"k_images": 6, "k_reports": 6}
    assert parse_command("reference: Findings: clear lungs.") == {"reference": "Findings: clear lungs."}
    assert parse_command("repair on") == {"display_repair": True}
    assert parse_command("what does this mean?") is None


def test_whole_text_must_be_one_command():
    assert parse_command("repair OFF") == {"display_repair": False}
    assert parse_command("Beam   2") == {"decode": "beam", "beam_size": 2}       # inner whitespace collapses
    assert parse_command("beam 5 please") is None
    assert parse_command("use beam 5") is None
    assert parse_command("beam five") is None
    assert parse_command("") is None and parse_command("   ") is None and parse_command(None) is None


def test_reference_text_is_kept_whitespace_collapsed():
    assert parse_command("reference:\nFindings:  no\teffusion.") == {"reference": "Findings: no effusion."}
    assert parse_command("reference:") is None


def test_the_fixed_answer_names_every_command():
    assert NOT_A_QA_BOT.endswith(COMMAND_HELP)
    for word in ("beam N", "greedy", "tokens N", "retrieve N", "reference:", "repair on|off"):
        assert word in COMMAND_HELP


# ---- P6-A (f): one definition of whitespace, the page's and the server's -------------------------------------------------------------

def _whitespace():
    return json.loads(WHITESPACE_FIXTURE.read_text(encoding="utf-8"))


def test_a_note_is_split_on_the_one_whitespace_the_page_splits_it_on():
    """Python's str.split() took U+001C..U+001F and U+0085 for whitespace and JavaScript's \\s did not, and JavaScript took U+FEFF and
    Python did not: "beam<U+001C>5" was a command to the server and a question to the page's hint, "<U+FEFF>" a question to the server and no
    note at all to the page. Both now split on the same characters, listed in tests/fixtures/whitespace.json, which the node test reads too."""
    data = _whitespace()
    for ch in data["whitespace"]:
        assert collapse("a" + ch + "b" + ch) == "a b", repr(ch)
    for ch in data["not_whitespace"]:
        assert collapse("a" + ch + "b") == "a" + ch + "b", repr(ch)
    assert len(data["notes"]) >= 15 and {c["runs"] for c in data["notes"]} == {True, False}
    for case in data["notes"]:
        assert collapse(case["note"]) == case["collapsed"], case
        assert (parse_command(case["note"]) is not None) is case["command"], case
        assert is_question(case["note"]) is (not case["runs"]), case   # what post_message does with it when no image comes with it


def test_the_shared_whitespace_is_pythons_and_javascripts_together_and_no_more():
    python = {chr(c) for c in range(0x110000) if chr(c).isspace()}
    javascript = set("\t\n\x0b\x0c\r \xa0\u1680\u2028\u2029\u202f\u205f\u3000\ufeff") | {chr(c) for c in range(0x2000, 0x200b)}   # what JavaScript's \s matches
    shared = set(_whitespace()["whitespace"])
    assert shared == python | javascript
    assert shared - javascript == {"\x1c", "\x1d", "\x1e", "\x1f", "\x85"} and shared - python == {"\ufeff"}   # the drift this closes
    everything = "".join(chr(c) for c in range(0x110000))
    assert set(re.findall("[" + WHITESPACE + "]", everything)) == shared   # the class the server splits on is that list exactly
