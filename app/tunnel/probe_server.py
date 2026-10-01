"""CHAT_UI_PLAN.md P1-A: stdlib-only streaming probe for the cluster.

Answers one question before any app code exists: can a CPU job serve HTTP that reaches the laptop
through lx01 with SSE frames arriving as written rather than buffered? It also times SQLite commits
on the filesystem the store will use (D22). Standard library only, so it runs in any interpreter.

    python3 app/tunnel/probe_server.py --endpoint-file ~/chat_sessions/probe_endpoint \
        [--bind 127.0.0.1] [--sqlite-probe ~/chat_sessions/probe.db]
"""
import argparse
import json
import socket
import sqlite3
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def make_handler(n_frames: int, interval_s: float):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            print("[probe] " + fmt % args, flush=True)

        def do_GET(self):
            if self.path != "/healthz":
                self.send_error(404)
                return
            body = json.dumps({"status": "ok", "host": socket.gethostname()}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != "/stream":
                self.send_error(404)
                return
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for i in range(n_frames):
                frame = "event: tick\ndata: {}\n\n".format(
                    json.dumps({"i": i, "server_ts": time.time()})).encode()
                self.wfile.write(b"%x\r\n" % len(frame) + frame + b"\r\n")
                self.wfile.flush()
                time.sleep(interval_s)
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

    return Handler


def serve(bind: str, port: int, n_frames: int = 5, interval_s: float = 1.0) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((bind, port), make_handler(n_frames, interval_s))


def sqlite_commit_ms(path: str, n: int = 200) -> float:
    """Mean ms per single-row commit, with the pragmas app/store.py will use (D22)."""
    con = sqlite3.connect(path)
    con.execute("PRAGMA locking_mode=EXCLUSIVE")
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("CREATE TABLE IF NOT EXISTS t (i INTEGER, s TEXT)")
    t0 = time.perf_counter()
    for i in range(n):
        con.execute("INSERT INTO t VALUES (?, ?)", (i, "x" * 300))
        con.commit()
    ms = (time.perf_counter() - t0) * 1000.0 / n
    con.close()
    return ms


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=0, help="0 picks a free port")
    ap.add_argument("--endpoint-file", required=True)
    ap.add_argument("--sqlite-probe", default=None)
    args = ap.parse_args()
    if args.sqlite_probe:
        Path(args.sqlite_probe).expanduser().parent.mkdir(parents=True, exist_ok=True)
        print("[probe] sqlite_commit_ms={:.3f}".format(sqlite_commit_ms(str(Path(args.sqlite_probe).expanduser()))),
              flush=True)
    server = serve(args.bind, args.port)
    endpoint = Path(args.endpoint_file).expanduser()
    endpoint.parent.mkdir(parents=True, exist_ok=True)
    endpoint.write_text("{}:{}\n".format(socket.gethostname(), server.server_address[1]))
    print("[probe] serving on {}:{} (bind {})".format(socket.gethostname(), server.server_address[1], args.bind),
          flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
