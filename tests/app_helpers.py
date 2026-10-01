"""Shared helpers for the chat-app tests (CHAT_UI_PLAN.md §8)."""
import io
import json
from typing import Dict, Iterable, Iterator

import numpy as np
from PIL import Image


def png_bytes(w: int = 320, h: int = 320, mode: str = "L") -> bytes:
    arr = (np.random.default_rng(0).random((h, w)) * 255).astype(np.uint8)
    img = Image.fromarray(arr).convert(mode)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def jpeg_bytes(w: int = 320, h: int = 320) -> bytes:
    buf = io.BytesIO()
    Image.open(io.BytesIO(png_bytes(w, h))).convert("RGB").save(buf, "JPEG", quality=92)
    return buf.getvalue()


def iter_sse(chunks: Iterable[str]) -> Iterator[Dict]:
    """The Python twin of app/static/api.js parseSSE: yields {"event", "data"}; skips comments."""
    buf = ""
    for chunk in chunks:
        buf = (buf + chunk).replace("\r\n", "\n")
        while "\n\n" in buf:
            frame, buf = buf.split("\n\n", 1)
            event, data = "message", []
            for line in frame.split("\n"):
                if line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data.append(line[5:][1:] if line[5:].startswith(" ") else line[5:])
            if data:
                yield {"event": event, "data": json.loads("\n".join(data))}
