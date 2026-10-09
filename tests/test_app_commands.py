"""CHAT_UI_PLAN.md P3-D: text commands (spec §4)."""
import json
from pathlib import Path

from app.commands import COMMAND_HELP, NOT_A_QA_BOT, parse_command

COMMANDS_FIXTURE = Path(__file__).resolve().parent / "frontend" / "fixtures" / "commands.json"


def test_the_pages_list_of_commands_is_the_servers():
    """P4-H: the page tells a command from a note (app/static/app.js isCommand) for the hint under the image well. Its node test reads
    the same notes, so the two cannot drift apart without one of them failing."""
    cases = json.loads(COMMANDS_FIXTURE.read_text(encoding="utf-8"))
    assert len(cases) >= 20 and {c["command"] for c in cases} == {True, False}
    for case in cases:
        assert (parse_command(case["note"]) is not None) is case["command"], case


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
