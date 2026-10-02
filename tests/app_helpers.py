"""Shared helpers for the chat-app tests (CHAT_UI_PLAN.md §8)."""
import io
import json
import time
from typing import Any, Callable, Dict, Iterable, Iterator

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


def start_live_server(app):
    """A real uvicorn on an ephemeral port (TestClient buffers whole responses). -> (base_url, stop)."""
    import socket
    import threading
    import time
    import uvicorn
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.02)

    def stop():
        server.should_exit = True
        t.join(5)

    return "http://127.0.0.1:{}".format(port), stop


def wait_until(predicate: Callable[[], Any], timeout: float = 10.0, interval: float = 0.02) -> Any:
    """Poll until predicate() is truthy; -> that value. A bounded wait: AssertionError after `timeout` seconds."""
    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value:
            return value
        if time.monotonic() >= deadline:
            raise AssertionError("condition not met within {} s".format(timeout))
        time.sleep(interval)
