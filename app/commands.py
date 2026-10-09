"""Text-only turns (spec §4): a small command set; anything else gets one fixed answer."""
import re
from typing import Any, Dict, Optional

COMMAND_HELP = "Commands: beam N, greedy, tokens N, retrieve N, reference: <text>, repair on|off."
NOT_A_QA_BOT = ("I generate chest X-ray reports and can't answer questions. Attach an X-ray, "
                "or send a command. " + COMMAND_HELP)
_RULES = [
    (r"beam (\d+)", lambda m: {"decode": "beam", "beam_size": int(m.group(1))}),
    (r"greedy", lambda m: {"decode": "greedy"}),
    (r"tokens (\d+)", lambda m: {"max_new_tokens": int(m.group(1))}),
    (r"retrieve (\d+)", lambda m: {"k_images": int(m.group(1)), "k_reports": int(m.group(1))}),
    (r"reference:\s*(.+)", lambda m: {"reference": m.group(1)}),
    (r"repair (on|off)", lambda m: {"display_repair": m.group(1).lower() == "on"}),
]


def parse_command(text: str) -> Optional[Dict[str, Any]]:
    """Option overrides for a text-only turn, or None if the text is not a command. ASCII only (re.ASCII): Unicode's \\d would also
    take other scripts' digits and its case folding the long s for an s, which the page's copy of these rules (app/static/app.js
    isCommand, JavaScript) does not (P4-H)."""
    t = " ".join((text or "").split())
    for pattern, build in _RULES:
        m = re.fullmatch(pattern, t, flags=re.IGNORECASE | re.ASCII)
        if m:
            return build(m)
    return None
