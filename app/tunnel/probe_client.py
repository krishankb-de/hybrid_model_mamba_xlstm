"""CHAT_UI_PLAN.md P1-B: time SSE frames arriving through the tunnel. Run on the laptop.

    venv/bin/python app/tunnel/probe_client.py http://127.0.0.1:8000/stream
Exit 0 iff at least 5 frames arrived spread out (min gap >= 0.5 s), i.e. nothing buffered them.
"""
import sys
import time
import urllib.request
from typing import List


def measure(url: str, timeout: float = 30.0) -> List[float]:
    req = urllib.request.Request(url, data=b"{}", method="POST",
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    arrivals, buf = [], b""
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        while True:
            chunk = resp.read1(65536)
            if not chunk:
                break
            buf += chunk
            while b"\n\n" in buf:
                _, buf = buf.split(b"\n\n", 1)
                arrivals.append(time.monotonic() - t0)
    return arrivals


def main() -> int:
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000/stream"
    arrivals = measure(url)
    for i, t in enumerate(arrivals):
        print("frame {} at {:.2f} s".format(i, t))
    gaps = [b - a for a, b in zip(arrivals, arrivals[1:])]
    ok = len(arrivals) >= 5 and min(gaps) >= 0.5
    print("STREAMED" if ok else "BUFFERED or SHORT: {}".format(gaps))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
