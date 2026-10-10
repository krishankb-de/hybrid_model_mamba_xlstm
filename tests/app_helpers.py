"""Shared helpers for the chat-app tests (CHAT_UI_PLAN.md §8)."""
import io
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, Iterator, Optional
from unittest import mock

import numpy as np
import torch
from PIL import Image

from app.tiny import TINY_PREFIX_K, TINY_VOCAB


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


# ---- scripted decoders (P9-G2): next-token logits that a test writes down, over the real tiny model ----------------------------
# A script is a function (n, last) -> logits of shape (vocab,): n tokens have been written so far, `last` is the latest one (None
# before the first). The decoders are driven through the same cached and uncached paths as with real weights, so a test can say
# "the report ends after 7 tokens" and check where each search stops, which no random-init model will do.

TINY_VOCAB_SIZE = len(TINY_VOCAB)
EOS_ID = TINY_VOCAB_SIZE - 1   # the last id of the tiny vocab: in reach of the scripted decoders, unlike the real engine's 50256
REPORT = ("The heart is normal. The lungs are clear. No pleural effusion. Impression: no acute disease. Findings: the heart "
          "is mildly enlarged and there is a small effusion.")   # 27 words: ends in sentences 4, 8, 11, 15 and 27


class GrowingText:
    """A tokenizer for a report written one word per step: n ids decode to the first n words of one text."""

    def __init__(self, text):
        self.words = text.split()

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(self.words[:len(ids)])


def noise_script(seed: int, eos: int, eos_logit: Callable[[int, Optional[int]], float], vocab: int = TINY_VOCAB_SIZE):
    """Seeded N(0, 2^2) logits per (n, last), no two alike, so no two candidates tie; the EOS id gets eos_logit(n, last)."""
    def script(n: int, last: Optional[int]) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed * 100003 + n * 1009 + (0 if last is None else last + 1))
        logits = torch.randn(vocab, generator=g) * 2.0
        logits[eos] = eos_logit(n, last)
        return logits
    return script


def markov_script(table: Dict[Any, Dict[int, float]], default: Callable[[int, Optional[int]], torch.Tensor],
                  vocab: int = TINY_VOCAB_SIZE):
    """Probabilities from a table {(n, last): {token: p}} (they should sum to 1); every other token is out of reach (-1e4).
    A state the table does not name is answered by `default`."""
    def script(n: int, last: Optional[int]) -> torch.Tensor:
        if (n, last) not in table:
            return default(n, last)
        logits = torch.full((vocab,), -1e4)
        for token, p in table[(n, last)].items():
            logits[token] = math.log(p)
        return logits
    return script


@contextmanager
def scripted_decoder(model, script: Callable[[int, Optional[int]], torch.Tensor], prefix_len: int = TINY_PREFIX_K):
    """Within the block, `model` (a tiny decoder) answers forward, prefill and step_logits from `script`; it keeps its own
    embeddings, cache allocation and reorder_cache, so the decoders handle it as they handle the real one.

    prefix_len counts the positions before the first generated token (image prefix, then any prompt). The token a position holds is
    read back from its embedding, which is the exact row of the embedding table, so a bit-for-bit match finds it.
    """
    table = model.embeddings.token_embedding.weight.detach()
    state = {"n": 0}

    def token_of(vec: torch.Tensor) -> int:
        return int((table == vec).all(dim=-1).nonzero()[0, 0])

    def forward(inputs_embeds=None, return_dict=True, **_):
        length = inputs_embeds.shape[1]
        n = length - prefix_len
        logits = torch.zeros(1, length, table.shape[0])
        logits[0, -1] = script(n, token_of(inputs_embeds[0, -1]) if n > 0 else None)
        return SimpleNamespace(logits=logits)

    def prefill(hidden, caches):
        state["n"] = 0
        return torch.stack([script(0, None)] * hidden.shape[0])

    def step_logits(hidden_t, caches):
        state["n"] += 1
        return torch.stack([script(state["n"], token_of(h)) for h in hidden_t])

    with mock.patch.object(model, "forward", forward), mock.patch.object(model, "prefill", prefill), \
            mock.patch.object(model, "step_logits", step_logits):
        yield model


# ---- the retrieval gallery (P5-D) ---------------------------------------------------------------------------------------------

def decide_gate(root: Path) -> Path:
    """Decide the R@k gate of a --tiny gallery, as the cluster does once the reference evaluation has run: write a reference result equal to the
    gallery's own gate_rk.json["app"] and run scripts/build_retrieval_gallery.compare_rk on it, which writes the verdict `equal: true` into
    gate_rk.json and manifest.json. A --tiny build has no reference run and so no verdict, and Gallery.open refuses a gallery without one: a test
    that wants an open gallery decides it first, Gallery.open(decide_gate(tiny_gallery), None). The tiny gallery's tower_sha256 is a fixed text hash
    that is no engine's, so open it with expect_tower_sha256=None, or write the engine's hash into manifest.json first. -> root."""
    from scripts import build_retrieval_gallery as bg
    gate = json.loads((root / "gate_rk.json").read_text())
    (root / "reference_rk").mkdir(exist_ok=True)
    (root / "reference_rk" / "phase6_mimic_20260101T000000Z.json").write_text(json.dumps({"metrics": gate["app"]}))
    assert bg.compare_rk(root) == 0
    assert json.loads((root / "manifest.json").read_text())["gate_rk"]["equal"] is True
    return root
