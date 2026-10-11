"""Text-only turns (spec §4): a small command set; anything else gets one fixed answer."""
import re
from typing import Any, Dict, Optional

COMMAND_HELP = "Commands: beam N, greedy, tokens N, retrieve N, reference: <text>, repair on|off."
NOT_A_QA_BOT = ("I generate chest X-ray reports and can't answer questions. Attach an X-ray, "
                "or send a command. " + COMMAND_HELP)
# The whitespace a note is split on (P6-A f), the same on both sides: Python's str.split() whitespace and JavaScript's \s together, so the
# server and the page (app/static/app.js SPACE) tell a command, and an empty note, alike. Python's alone took U+001C..U+001F and U+0085,
# JavaScript's alone U+FEFF. tests/fixtures/whitespace.json lists it, and both test suites read that file. A regex class body.
WHITESPACE = "\t\n\x0b\x0c\r\x1c-\x1f \x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
_SPACE = re.compile("[" + WHITESPACE + "]+")
_RULES = [
    (r"beam (\d+)", lambda m: {"decode": "beam", "beam_size": int(m.group(1))}),
    (r"greedy", lambda m: {"decode": "greedy"}),
    (r"tokens (\d+)", lambda m: {"max_new_tokens": int(m.group(1))}),
    (r"retrieve (\d+)", lambda m: {"k_images": int(m.group(1)), "k_reports": int(m.group(1))}),
    (r"reference:\s*(.+)", lambda m: {"reference": m.group(1)}),
    (r"repair (on|off)", lambda m: {"display_repair": m.group(1).lower() == "on"}),
]


def collapse(text: Optional[str]) -> str:
    """The note's words joined by single spaces, split on WHITESPACE as the page splits them."""
    return " ".join(word for word in _SPACE.split(text or "") if word)


def parse_command(text: str) -> Optional[Dict[str, Any]]:
    """Option overrides for a text-only turn, or None if the text is not a command. ASCII only (re.ASCII): Unicode's \\d would also
    take other scripts' digits and its case folding the long s for an s, which the page's copy of these rules (app/static/app.js
    isCommand, JavaScript) does not (P4-H)."""
    t = collapse(text)
    for pattern, build in _RULES:
        m = re.fullmatch(pattern, t, flags=re.IGNORECASE | re.ASCII)
        if m:
            return build(m)
    return None


def is_question(text: Optional[str]) -> bool:
    """A note that says something and is no command: sent with no image, it gets the fixed answer and runs no model (post_message)."""
    return bool(collapse(text)) and parse_command(text) is None
