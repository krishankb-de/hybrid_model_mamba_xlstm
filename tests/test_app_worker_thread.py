"""CHAT_UI_PLAN.md P8-F part 3: one OpenMP pool per server process. The engines are built on the worker thread that runs the turns.

libgomp gives every thread that enters a parallel region a pool of its own, which lives as long as the thread. The server built its engines in
create_app on the main thread and ran its turns on the Worker's one thread: two pools of 8 on a node's 8 CPUs, so libgomp stopped spinning, each
of a decode step's many small parallel regions paid a futex wake-up, and the live decode took about 3 times as long as the same decode on the main
thread. create_app now makes the Worker first and builds the engines through Worker.call, a blocking call on the thread that runs every turn, and
no other thread runs a torch op: not the main thread, not the event loop, not FastAPI's thread pool.

A laptop cannot show the throttle (macOS torch uses LLVM's libomp), so these tests are structural: the threads that build an engine, hash its
tower and decode a turn (the real path through a recording build_engine, and the tiny path), what the thread that calls create_app dispatches (a
TorchDispatchMode sees its own thread's ops only), what a refused start leaves behind, the image work of the request paths, and SIGTERM while the
engines load. The speed is measured on the cluster.
"""
import io
import json
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time

import pytest
from fastapi.testclient import TestClient
from torch.utils._python_dispatch import TorchDispatchMode

from app import cli, server
from app.engine import REPO_ROOT, TinyEngine, build_engine
from app.imaging import load_upload, model_input_image, model_transform
from app.pipeline import Worker
from app.server import ModelUnavailable, create_app
from app.store import Store
from tests.app_helpers import iter_sse, jpeg_bytes, png_bytes, wait_until

MODELS = ("hybrid_150m_m3_rrg", "hybrid_150m_v2_rrg")
EOS_MODEL = "hybrid_150m_m3_rrg_eos"
NOT_MARKED = "{} is not marked as trained to end its reports in its run metadata: serve the run it came from.".format(EOS_MODEL)


class TorchOps(TorchDispatchMode):
    """The torch ops dispatched on the thread that holds the mode, by name. The mode is thread-local: an op of another thread is not seen."""

    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func))
        return func(*args, **(kwargs or {}))


def note(seen, what):
    seen.append((what, threading.current_thread(), threading.get_ident()))


def watch(engine, seen):
    """The engine notes the thread that hashes its tower and the thread that decodes its turns."""
    tower_sha256, generate = engine.tower_sha256, engine.generate

    def hashing():
        note(seen, "tower")
        return tower_sha256()

    def decoding(*args, **kwargs):
        note(seen, "generate")
        return generate(*args, **kwargs)

    engine.tower_sha256, engine.generate = hashing, decoding
    return engine


def real_builder(monkeypatch, seen):
    """server.build_engine("real", ...) without checkpoints (none exist on the laptop): a tiny engine called after its model, which notes the
    thread that builds it, hashes its tower and decodes its turns."""
    def fake(kind, **kw):
        note(seen, "build")
        engine = build_engine("tiny")
        engine.name = kw.get("name") or kw["model_config"]
        engine._card.update(name=engine.name, checkpoint=kw["checkpoint"])
        return watch(engine, seen)

    monkeypatch.setattr(server, "build_engine", fake)


def turn(client, model=None):
    """One turn on an upload, read whole; -> its frames, the last a message_stop that says done."""
    sid = client.post("/v1/sessions", json={}).json()["id"]
    options = dict({"max_new_tokens": 16}, **({} if model is None else {"model": model}))
    r = client.post("/v1/sessions/{}/messages".format(sid), files={"image": ("x.png", png_bytes(), "image/png")},
                    data={"text": "", "options": json.dumps(options)})
    assert r.status_code == 200, r.text
    frames = list(iter_sse([r.text]))
    assert frames[-1]["event"] == "message_stop" and frames[-1]["data"]["status"] == "done", frames[-1]
    return frames


def release(app):
    """An app that was built and is not served lets go of its worker's thread and of the database's lock, as app/cli.py's _release does."""
    app.state.worker.shutdown()
    app.state.store.close()


def released(home):
    """The store under `home` was closed: its EXCLUSIVE lock is gone, so another connection reads the file at once."""
    other = sqlite3.connect(str(home / "chat.db"), timeout=0)
    try:
        return other.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
    finally:
        other.close()


# ---- one thread: the engines are built, their towers hashed and their turns decoded on it -------------------------------------

def test_the_real_engines_are_built_and_hashed_on_the_thread_that_runs_their_turns_and_never_on_the_main_thread(tmp_path, monkeypatch):
    """The real path (_engines with a recording build_engine), with a gallery_dir so that the gallery's tower check hashes every engine's tower
    (the gallery is not there, so it is not opened: the check comes first). The start's thread lives on after create_app, since
    ThreadPoolExecutor keeps its thread, decodes both models' turns, and is stopped by the lifespan's worker.shutdown()."""
    main, seen = threading.get_ident(), []
    real_builder(monkeypatch, seen)
    app = create_app(engine="real", home=str(tmp_path / "home"), models=MODELS, gallery_dir=str(tmp_path / "no-gallery"))
    assert [what for what, _, _ in seen] == ["build", "build", "tower", "tower"], "both engines, then the tower check of both"
    worker = seen[0][1]
    assert worker.is_alive() and worker.name.startswith("turn"), "the thread that built the engines lives on"
    with TestClient(app) as client:
        for model in MODELS:
            turn(client, model)
    assert [what for what, _, _ in seen][4:] == ["generate", "generate"]
    assert {thread for _, thread, _ in seen} == {worker} and {ident for _, _, ident in seen} == {worker.ident}
    assert worker.ident != main and worker is not threading.main_thread()
    assert not worker.is_alive(), "the lifespan's worker.shutdown() ended it"


def test_the_tiny_engine_is_built_and_hashed_on_the_thread_that_runs_its_turns_by_the_same_path(tmp_path, monkeypatch):
    """The tiny engine goes through the same code (_engines, then build_engine("tiny")): TinyEngine's own __init__, tower hash and generate
    note their threads."""
    main, seen = threading.get_ident(), []
    init, tower_sha256, generate = TinyEngine.__init__, TinyEngine.tower_sha256, TinyEngine.generate

    def building(self, *args, **kwargs):
        note(seen, "build")
        init(self, *args, **kwargs)

    def hashing(self):
        note(seen, "tower")
        return tower_sha256(self)

    def decoding(self, *args, **kwargs):
        note(seen, "generate")
        return generate(self, *args, **kwargs)

    monkeypatch.setattr(TinyEngine, "__init__", building)
    monkeypatch.setattr(TinyEngine, "tower_sha256", hashing)
    monkeypatch.setattr(TinyEngine, "generate", decoding)
    app = create_app(engine="tiny", home=str(tmp_path / "home"), gallery_dir=str(tmp_path / "no-gallery"))
    worker = seen[0][1]
    assert worker.is_alive() and worker.name.startswith("turn")
    with TestClient(app) as client:
        turn(client)
    assert [what for what, _, _ in seen] == ["build", "tower", "generate"]
    assert {thread for _, thread, _ in seen} == {worker} and {ident for _, _, ident in seen} == {worker.ident}
    assert worker.ident != main and not worker.is_alive()


def test_create_app_runs_no_torch_op_on_the_thread_that_calls_it(tmp_path, monkeypatch):
    """The thread that calls create_app (a server's main thread) runs no torch op at start-up, for the tiny engine and for the real path, the
    tower check included. The mode sees an engine built and hashed on its own thread, so an empty list means the start ran on another one."""
    with TorchOps() as control:
        build_engine("tiny").tower_sha256()
    assert control.ops, "the mode sees what its own thread runs"
    with TorchOps() as tiny:
        app = create_app(engine="tiny", home=str(tmp_path / "tiny"), gallery_dir=str(tmp_path / "no-gallery"))
    release(app)
    real_builder(monkeypatch, [])
    with TorchOps() as real:
        app = create_app(engine="real", home=str(tmp_path / "real"), models=MODELS, gallery_dir=str(tmp_path / "no-gallery"))
    release(app)
    assert tiny.ops == [] and real.ops == []


def test_worker_call_runs_on_the_thread_of_the_turns_and_gives_back_the_result_or_the_exception_itself():
    """Worker.call is a blocking call on the worker's one thread (max_workers=1), the thread run_task and the turns use; it takes no slot."""
    worker = Worker(None, 1)
    try:
        thread = worker.call(threading.current_thread)
        assert thread is not threading.main_thread() and thread.name.startswith("turn")
        assert worker.call(divmod, 7, 2) == (3, 1) and worker.in_flight == 0, "a call is not a turn: it takes no slot"
        assert worker.reserve() and worker.run_task(threading.current_thread).result() is thread, "the thread of run_task (POST /v1/retrieve)"
        problem = ModelUnavailable("refused")

        def refuse():
            raise problem

        with pytest.raises(ModelUnavailable) as caught:
            worker.call(refuse)
        assert caught.value is problem and worker.call(threading.current_thread) is thread, "the same exception, and the thread lives on"
    finally:
        worker.shutdown()
    assert not thread.is_alive()


# ---- a refused start: the store closed, the worker's thread gone, the exception as it was raised ----------------------------

def _eos_run(home):
    """The EOS entry's checkpoint and DONE marker where the registry keeps them under CHAT_HOME (made-up bytes), so the start builds it."""
    checkpoint = home / server.HOME_MODELS[EOS_MODEL]["checkpoint"]
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"not a real checkpoint")
    (checkpoint.parent.parent / "DONE").write_text("done\n")


def _refusing(where, home, gallery_dir, monkeypatch, seen):
    """Arrange a start that is refused at `where`, the place noting its thread; -> (create_app's arguments, the exception it must raise)."""
    if where == "build":   # the loader raises: a checkpoint that is not there
        problem = FileNotFoundError("the checkpoint is not where the server keeps it")

        def fake(kind, **kw):
            note(seen, where)
            raise problem

        monkeypatch.setattr(server, "build_engine", fake)
        return {"engine": "real", "models": MODELS[:1]}, problem
    if where == "eos_check":   # an EOS entry whose run does not say eos_trained (P9-G5a fix 1)
        _eos_run(home)
        real_builder(monkeypatch, [])
        check = server._require_eos_entries_marked

        def judged(engines):
            note(seen, where)
            check(engines)

        monkeypatch.setattr(server, "_require_eos_entries_marked", judged)
        return {"engine": "real", "models": (EOS_MODEL,)}, ModelUnavailable(NOT_MARKED)
    if where == "tower_check":   # the gallery's tower check, which hashes the engine's weights
        problem = RuntimeError("the tower could not be hashed")

        def broken_tower():
            note(seen, where)
            raise problem

        monkeypatch.setattr(TinyEngine, "tower_sha256", lambda self: broken_tower())
        return {"engine": "tiny", "gallery_dir": gallery_dir}, problem
    problem = RuntimeError("the published dumps could not be read")   # "published": the part of the start that runs no torch, after it

    def broken_dumps(dirs, gallery):
        note(seen, where)
        raise problem

    monkeypatch.setattr(server, "_wire_published", broken_dumps)
    return {"engine": "tiny"}, problem


@pytest.mark.parametrize("where", ["build", "eos_check", "tower_check", "published"])
def test_a_refused_start_closes_the_store_ends_the_workers_thread_and_raises_what_was_raised(tmp_path, monkeypatch, where):
    """A refusal raised on the worker thread (the build, the EOS check, the tower check) reaches the caller as it was raised, its type and its
    message, and so does one raised after it on the calling thread. Either way the store is closed and no thread the start made is left."""
    home, seen = tmp_path / "home", []
    kwargs, expected = _refusing(where, home, str(tmp_path / "no-gallery"), monkeypatch, seen)
    before = set(threading.enumerate())
    with pytest.raises(type(expected)) as stop:
        create_app(home=str(home), **kwargs)
    assert type(stop.value) is type(expected) and str(stop.value) == str(expected)
    assert [thread for thread in threading.enumerate() if thread not in before] == [], "the worker's thread ended with the start"
    assert released(home), "the store was closed"
    (_, thread, ident), = [s for s in seen if s[0] == where]
    if where == "published":
        assert thread is threading.main_thread()
    else:
        assert thread is not threading.main_thread() and ident != threading.get_ident() and thread.name.startswith("turn"), where


def test_the_cli_that_refuses_a_gallery_that_will_not_open_ends_the_thread_that_built_the_engines(tmp_path, monkeypatch, capsys):
    """create_app tolerates a gallery that will not open and the serving command line does not: its _release runs worker.shutdown(), which
    ends the thread that built the engines."""
    monkeypatch.setattr(cli, "ServingServer", lambda *a, **k: pytest.fail("the server was started"))
    before = set(threading.enumerate())
    code = cli.main(["--engine", "tiny", "--mode", "private", "--home", str(tmp_path / "home"), "--gallery", str(tmp_path / "no-gallery")])
    assert code == 1 and "ERROR the retrieval gallery could not be opened" in capsys.readouterr().out
    assert [thread for thread in threading.enumerate() if thread not in before] == []


def test_a_sigterm_while_the_engines_load_on_the_worker_still_ends_the_process_at_once(tmp_path):
    """app/cli.py: until uvicorn serves there is no handler, so SIGTERM ends the process on the spot. The engines now load on the worker's
    thread while the main thread waits for them; the signal's default action does not wait for the load."""
    script = ("import sys, threading, time\n"
              "from app import cli, server\n"
              "def slow(kind, **kw):\n"
              "    print('[server] building on ' + threading.current_thread().name, flush=True)\n"
              "    time.sleep(120)\n"
              "server.build_engine = slow\n"
              "sys.exit(cli.main(sys.argv[1:]))\n")
    out = tmp_path / "stdout.txt"
    with open(str(out), "wb") as stdout:
        proc = subprocess.Popen([sys.executable, "-c", script, "--engine", "tiny", "--mode", "private", "--home", str(tmp_path / "home")],
                                cwd=str(REPO_ROOT), stdin=subprocess.DEVNULL, stdout=stdout, stderr=subprocess.DEVNULL,
                                env=dict(os.environ, PYTHONUNBUFFERED="1"), start_new_session=True)
        try:
            wait_until(lambda: "[server] building on " in out.read_text(), timeout=60)
            assert "[server] building on turn" in out.read_text(), "the engines load on the worker's thread"
            began = time.monotonic()
            proc.send_signal(signal.SIGTERM)
            assert proc.wait(10) == -signal.SIGTERM and time.monotonic() - began < 5
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()


# ---- the request paths: their image work runs in FastAPI's thread pool, and is PIL and numpy ------------------------------------

def test_the_image_work_of_the_request_paths_runs_no_torch_op(tmp_path):
    """FastAPI's thread pool runs _checked_upload and _save_upload for an upload (its thumbnail and its 224x224 model input), and GET
    .../image?variant=model_input makes a test study's model input in memory with model_input_image: PIL and numpy, since torchvision's Resize
    and Grayscale take the PIL path for a PIL image. torchvision checks its extension once, when it is first imported (no kernel); the first
    call below makes that check before the mode is on. The worker's preprocess, by contrast, is torch."""
    model_input_image(load_upload(png_bytes())[0])
    store = Store(tmp_path / "home")
    try:
        session = store.create_session("private")
        data = png_bytes(1024, 900)
        with TorchOps() as ops:
            img, ext, sha256 = server._checked_upload(data)
            server._save_upload(store, session["id"], sha256, data, img, ext)
            model_input_image(load_upload(jpeg_bytes())[0]).save(io.BytesIO(), "PNG")
        with TorchOps() as control:
            model_transform()(img)
    finally:
        store.close()
    assert ops.ops == [] and control.ops
