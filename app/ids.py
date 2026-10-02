"""Sortable id generation for messages and sessions.

IDs are generated as <prefix>_<12 hex ms timestamp><10 hex suffix>.
Within the same millisecond, a counter in the suffix (4 hex) ensures monotonicity;
the remaining 6 hex are random.
"""
import secrets
import threading
import time
from typing import Dict, Tuple

_id_lock = threading.Lock()
_id_state: Dict[str, Tuple[int, int]] = {}  # prefix -> (last_ms, counter)


def new_id(prefix: str) -> str:
    """Generate a sortable id with the given prefix.

    Format: <prefix>_<12 hex ms time><10 hex suffix>
    Suffix: 4 hex counter + 6 hex random.
    Ids are unique and sorted by creation time within the same prefix.
    """
    global _id_state

    with _id_lock:
        now_ms = int(time.time() * 1000)
        last_ms, counter = _id_state.get(prefix, (now_ms, 0))

        if now_ms > last_ms:
            # New millisecond: reset counter
            counter = 0
        else:
            # Same millisecond: increment counter to ensure ordering
            counter += 1

        _id_state[prefix] = (now_ms, counter)

    # Format: 12 hex for ms, 4 hex for counter, 6 hex for random
    random_part = secrets.token_hex(3)  # 6 hex chars
    suffix = f"{counter:04x}{random_part}"
    return f"{prefix}_{now_ms:012x}{suffix}"
