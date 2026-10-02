"""Sortable id generation for messages and sessions."""
import secrets
import threading
import time
from typing import Dict, Tuple

_id_lock = threading.Lock()
_id_state: Dict[str, Tuple[int, int]] = {}


def new_id(prefix: str) -> str:
    """Generate a sortable id with the given prefix. Format: <prefix>_<12 hex ms><4 hex counter><6 hex random>."""
    with _id_lock:
        now_ms = int(time.time() * 1000)
        last_ms, counter = _id_state.get(prefix, (0, 0))

        if now_ms > last_ms:
            counter = 0
        else:
            now_ms = last_ms
            counter += 1
            if counter > 0xFFFF:
                now_ms, counter = now_ms + 1, 0

        _id_state[prefix] = (now_ms, counter)

    random_part = secrets.token_hex(3)
    suffix = "{:04x}{}".format(counter, random_part)
    return "{}_{}{}".format(prefix, "{:012x}".format(now_ms), suffix)
