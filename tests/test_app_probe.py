"""CHAT_UI_PLAN.md P1-A: the reachability probe streams SSE frames as written (stdlib only)."""
import json
import socket
import threading
import urllib.request

from app.tunnel.probe_client import measure
from app.tunnel.probe_server import serve, sqlite_commit_ms


def _start(**kw):
    server = serve("127.0.0.1", 0, **kw)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, "http://127.0.0.1:{}".format(server.server_address[1])


def test_probe_streams_frames_as_written_not_buffered():
    server, base = _start(n_frames=4, interval_s=0.2)
    try:
        arrivals = measure(base + "/stream")
    finally:
        server.shutdown()
    assert len(arrivals) == 4
    gaps = [b - a for a, b in zip(arrivals, arrivals[1:])]
    assert min(gaps) > 0.1, gaps   # a buffering hop would deliver all four at once


def test_probe_healthz_names_the_host():
    server, base = _start()
    try:
        body = urllib.request.urlopen(base + "/healthz").read()
    finally:
        server.shutdown()
    assert json.loads(body) == {"status": "ok", "host": socket.gethostname()}


def test_sqlite_probe_reports_a_positive_commit_cost(tmp_path):
    assert sqlite_commit_ms(str(tmp_path / "probe.db"), n=20) > 0
