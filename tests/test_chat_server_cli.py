"""CHAT_UI_PLAN.md P7-B: the serving CLI, `python -m app.server` (app/cli.py).

In this order: the arguments and every refusal (R6: no token, a token file that is not private; and, for a serving process, a gallery that
will not open and a CUDA that is not there), the token file reader, the endpoint file, what main() hands to create_app and to uvicorn, the
filter that keeps a job log to its four line shapes, the shutdown (a stop ends a running turn as a server restart at once, the endpoint
file goes), and live runs of the real CLI on the tiny engine over real sockets (the SIGTERM exit is clean). The two serving wrappers are
rehearsed in tests/test_chat_server_job.py. Synthetic data only (R7); every server here uses a free port and a temp home.
"""
import asyncio
import datetime
import io
import json
import os
import re
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import torch
import uvicorn

from app import cli, server
from app.engine import TinyEngine, build_engine
from app.labels import RuleLabeler
from app.server import create_app
from app.store import Store
from tests.app_helpers import iter_sse, png_bytes, wait_until

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE = ["--engine", "tiny", "--mode", "private", "--home", "HOME"]
JOB_LOG_LINE = re.compile(r"^(=== |\[server\] |RESULT |ERROR)")   # what a job log may carry (R7): the test's own copy of the rule


def parsed(*extra):
    return cli.parse_args(BASE + list(extra))


def refused(capsys, argv, *needles):
    """The CLI turned the arguments down: exit 2, and one ERROR line on stdout (the stream a job log keeps) that says why."""
    with pytest.raises(SystemExit) as stop:
        cli.parse_args(argv)
    assert stop.value.code == 2
    out = capsys.readouterr().out
    assert len(out.splitlines()) == 1 and out.startswith("ERROR "), out
    for needle in needles:
        assert needle in out, (needle, out)
    return out


def make_token(tmp_path, text="s3cret-token\n", mode=0o600):
    path = tmp_path / "app_token"
    path.write_text(text)
    path.chmod(mode)
    return path


# ---- the arguments ---------------------------------------------------------------------------------------------------------

def test_the_defaults_are_a_private_loopback_server_with_the_published_model():
    args = parsed()
    assert (args.engine, args.mode, args.home) == ("tiny", "private", "HOME")
    assert (args.device, args.host, args.port, args.threads) == ("cpu", "127.0.0.1", 0, 8)
    assert args.models == ("hybrid_150m_m3_rrg",)
    assert (args.gallery, args.labeler, args.endpoint_file, args.token_file, args.token) == (None, "none", None, None, None)
    assert (args.drift_note, args.allow_compile, args.allow_no_gallery) == ("", False, False)
    assert (args.published_model, args.published_floor) == (None, None)


def test_every_flag_of_the_plan_is_accepted(tmp_path):
    token = make_token(tmp_path)
    args = cli.parse_args(["--engine", "real", "--mode", "public", "--home", "H", "--device", "cuda", "--gallery", "G", "--labeler",
                           "http://127.0.0.1:8001", "--host", "127.0.0.1", "--port", "8123", "--endpoint-file", "E", "--token-file",
                           str(token), "--models", "m3,13d", "--threads", "4", "--drift-note", "CPU vs GPU: 0/20 differ",
                           "--allow-compile", "--allow-no-gallery", "--published-model", "PM", "--published-floor", "PF"])
    assert (args.engine, args.mode, args.home, args.device, args.gallery) == ("real", "public", "H", "cuda", "G")
    assert (args.labeler, args.host, args.port, args.endpoint_file) == ("http://127.0.0.1:8001", "127.0.0.1", 8123, "E")
    assert (args.threads, args.drift_note, args.allow_compile, args.allow_no_gallery) == (4, "CPU vs GPU: 0/20 differ", True, True)
    assert args.models == ("hybrid_150m_m3_rrg", "hybrid_150m_v2_rrg") and args.token == "s3cret-token"
    assert (args.published_model, args.published_floor) == ("PM", "PF")


@pytest.mark.parametrize("missing", ["--engine", "--mode", "--home"])
def test_engine_mode_and_home_are_never_guessed(capsys, missing):
    argv = list(BASE)
    at = argv.index(missing)
    del argv[at:at + 2]
    refused(capsys, argv, "required", missing)


@pytest.mark.parametrize("flag, value", [("--engine", "mock"), ("--mode", "demo"), ("--device", "tpu")])
def test_a_choice_outside_its_list_is_refused(capsys, flag, value):
    argv = list(BASE)
    if flag in argv:
        argv[argv.index(flag) + 1] = value
    else:
        argv += [flag, value]
    refused(capsys, argv, "invalid choice")


def test_there_is_no_flag_that_takes_the_token_itself_and_no_abbreviation_reaches_the_token_file_flag(capsys):
    """The token is read from a file, so it stays out of `scontrol show job`, `ps` and shell history. argparse would otherwise take
    `--token abc` for `--token-file abc`."""
    parser = cli.build_parser()
    assert "--token" not in parser._option_string_actions and "--token-file" in parser._option_string_actions
    refused(capsys, BASE + ["--token", "abc"], "unrecognized")
    # and the refusal does not hand the secret on to a job log: the option's name is what it may show, never what follows it
    with pytest.raises(SystemExit):
        cli.parse_args(BASE + ["--token", "SECRET-token-value"])
    shown = capsys.readouterr()
    assert "SECRET" not in shown.out and "SECRET" not in shown.err
    assert shown.out == "ERROR unrecognized arguments: --token\n"


@pytest.mark.parametrize("argv", [
    BASE + ["--token", "SECRET-1"],                                   # an unknown option and its value
    BASE + ["--token=SECRET-2"],                                      # the same joined by =
    BASE + ["--token", "SECRET-3", "--bogus", "SECRET-4", "SECRET-5"],  # several, and a bare value after them
    BASE + ["SECRET-6"],                                              # a value with no option at all
    ["--engine", "SECRET-7", "--mode", "private", "--home", "H"],     # a value outside a choice list
    ["--engine", "tiny", "--mode", "SECRET-8", "--home", "H"],
    BASE + ["--port", "SECRET-9"],                                    # a value that is not a number
    BASE + ["--threads", "SECRET-10"],
    BASE + ["--device", "SECRET-11"],
    BASE + ["--labeler", "SECRET-12"],                                # refusals of this module's own
    BASE + ["--models", "SECRET-13"],
])
def test_no_refusal_ever_echoes_the_value_of_an_argument(capsys, argv):
    """A value can be a secret that was put where it does not belong (an operator who tries `--token SECRET`), and a refusal goes to the job
    log: argparse's own messages quote the offending value, so they are cut down to what was wrong, with the option's name at most."""
    with pytest.raises(SystemExit) as stop:
        cli.parse_args(argv)
    assert stop.value.code == 2
    shown = capsys.readouterr()
    assert "SECRET" not in shown.out and "SECRET" not in shown.err, shown
    assert len(shown.out.splitlines()) == 1 and shown.out.startswith("ERROR "), shown.out


def test_models_take_the_short_names_or_the_config_names_in_the_order_given():
    assert parsed("--models", "m3,13d").models == ("hybrid_150m_m3_rrg", "hybrid_150m_v2_rrg")
    assert parsed("--models", "13d").models == ("hybrid_150m_v2_rrg",)
    assert parsed("--models", "hybrid_150m_v2_rrg,m3").models == ("hybrid_150m_v2_rrg", "hybrid_150m_m3_rrg")
    assert set(cli.MODEL_ALIASES.values()) == set(server.MODEL_CHECKPOINTS), "an alias for a model the server has no checkpoint for"


@pytest.mark.parametrize("value, why", [("m4", "unknown model"), ("hybrid_70m", "unknown model"), ("", "empty"), ("m3,,13d", "empty"),
                                        (",m3", "empty"), ("m3,m3", "twice"), ("m3,hybrid_150m_m3_rrg", "twice")])
def test_an_unknown_empty_or_repeated_model_is_refused(capsys, value, why):
    refused(capsys, BASE + ["--models", value], why)


@pytest.mark.parametrize("value", ["none", "rule", "http://127.0.0.1:8001", "https://labeller.example:9000/x"])
def test_the_labeler_is_a_url_rule_or_none(value):
    assert parsed("--labeler", value).labeler == value


@pytest.mark.parametrize("value", ["", "ftp://127.0.0.1:1", "http://", "127.0.0.1:8001", "Rule", "yes"])
def test_a_labeler_that_is_none_of_the_three_is_refused(capsys, value):
    refused(capsys, BASE + ["--labeler", value], "--labeler must be")


@pytest.mark.parametrize("flag, value, why", [("--port", "-1", "--port must be"), ("--port", "65536", "--port must be"),
                                              ("--port", "x", "invalid int"), ("--threads", "0", "--threads must be"),
                                              ("--threads", "-3", "--threads must be"), ("--threads", "x", "invalid int")])
def test_a_port_or_a_thread_count_out_of_range_is_refused(capsys, flag, value, why):
    refused(capsys, BASE + [flag, value], why)


def test_the_published_dumps_come_as_a_pair(capsys):
    refused(capsys, BASE + ["--published-model", "a"], "--published-model and --published-floor go together")
    refused(capsys, BASE + ["--published-floor", "b"], "--published-model and --published-floor go together")
    assert parsed("--published-model", "a", "--published-floor", "b").published_floor == "b"


# ---- R6: no unauthenticated exposure -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("host", ["0.0.0.0", "10.1.2.3", "::", "gx08", "example.org", ""])
def test_a_bind_off_loopback_needs_a_token_file(capsys, host):
    refused(capsys, BASE + ["--host", host], "needs a token file", "R6")


@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.2", "::1", "localhost"])
def test_loopback_binds_need_no_token(host):
    assert parsed("--host", host).token is None


def test_a_bind_off_loopback_is_allowed_with_a_token_file(tmp_path):
    assert parsed("--host", "0.0.0.0", "--token-file", str(make_token(tmp_path))).token == "s3cret-token"


def test_public_mode_needs_a_token_file_even_on_loopback(capsys, tmp_path):
    argv = ["--engine", "tiny", "--mode", "public", "--home", "HOME"]
    refused(capsys, argv, "public mode needs a token file", "R6")
    assert cli.parse_args(argv + ["--token-file", str(make_token(tmp_path))]).mode == "public"


@pytest.mark.parametrize("mode", [0o644, 0o640, 0o660, 0o666, 0o700, 0o400, 0o604])
def test_a_token_file_must_be_mode_0600_exactly(capsys, tmp_path, mode):
    out = refused(capsys, BASE + ["--token-file", str(make_token(tmp_path, mode=mode))], "mode 0600")
    assert "s3cret" not in out and str(tmp_path) not in out, "the refusal names neither the token nor a path"


def test_a_token_file_must_belong_to_the_user_running_the_server(capsys, tmp_path, monkeypatch):
    path = make_token(tmp_path)
    monkeypatch.setattr(os, "getuid", lambda: os.stat(str(path)).st_uid + 1)
    refused(capsys, BASE + ["--token-file", str(path)], "not owned by you")


@pytest.mark.parametrize("text, why", [("", "is empty"), ("  \n\n", "is empty"), ("MARK-91ac\nsecond\n", "visible ASCII"),
                                       ("MARK-91ac and-more\n", "visible ASCII"), ("MARK-91ac\tx\n", "visible ASCII"),
                                       ("MARK-91ac-café\n", "visible ASCII")])
def test_an_empty_or_unusable_token_is_refused_without_echoing_it(capsys, tmp_path, text, why):
    out = refused(capsys, BASE + ["--token-file", str(make_token(tmp_path, text=text))], why)
    assert "MARK-91ac" not in out and "second" not in out


def test_a_missing_file_or_a_directory_or_a_pipe_is_no_token_file(capsys, tmp_path):
    """A pipe would block an open() for reading until someone writes to it: the reader must refuse it, not wait."""
    refused(capsys, BASE + ["--token-file", str(tmp_path / "nope")], "cannot be read")
    directory = tmp_path / "dir"
    directory.mkdir()
    refused(capsys, BASE + ["--token-file", str(directory)], "regular file")
    fifo = tmp_path / "fifo"
    os.mkfifo(str(fifo), 0o600)
    refused(capsys, BASE + ["--token-file", str(fifo)], "regular file")


def test_a_symlink_is_judged_by_the_file_behind_it(capsys, tmp_path):
    good = make_token(tmp_path)
    link = tmp_path / "link"
    link.symlink_to(good)
    assert parsed("--token-file", str(link)).token == "s3cret-token"
    good.chmod(0o644)
    refused(capsys, BASE + ["--token-file", str(link)], "mode 0600")


def test_the_token_is_the_files_text_without_its_surrounding_whitespace(tmp_path):
    assert cli.read_token(str(make_token(tmp_path, text="  abc-DEF_123.~\n\n"))) == "abc-DEF_123.~"
    with pytest.raises(cli.TokenFileError):
        cli.read_token(str(tmp_path / "missing"))
    assert issubclass(cli.TokenFileError, ValueError)


# ---- the endpoint file -------------------------------------------------------------------------------------------------------

def test_the_endpoint_file_is_one_json_line_of_five_fields_and_mode_0600(tmp_path):
    path = tmp_path / "chat" / "endpoint"                       # its directory is made when it is missing
    cli.write_endpoint(str(path), 43123, "private")
    text = path.read_text()
    assert text.endswith("\n") and len(text.splitlines()) == 1
    data = json.loads(text)
    assert list(data) == ["host", "port", "mode", "pid", "started_at"]
    assert (data["host"], data["port"], data["mode"], data["pid"]) == (socket.gethostname(), 43123, "private", os.getpid())
    started = datetime.datetime.fromisoformat(data["started_at"])
    assert started.utcoffset() == datetime.timedelta(0) and abs(time.time() - started.timestamp()) < 60
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert [p.name for p in path.parent.iterdir()] == ["endpoint"], "no temporary file is left beside it"


def test_a_new_endpoint_file_replaces_an_older_one_whole_and_private(tmp_path):
    path = tmp_path / "endpoint"
    path.write_text("gx01:1111\n")
    path.chmod(0o644)
    cli.write_endpoint(str(path), 2222, "public")
    assert json.loads(path.read_text())["port"] == 2222 and stat.S_IMODE(path.stat().st_mode) == 0o600
    assert sorted(p.name for p in tmp_path.iterdir()) == ["endpoint"]


def test_an_endpoint_file_that_cannot_be_finished_leaves_the_old_one_and_no_temporary(tmp_path, monkeypatch):
    path = tmp_path / "endpoint"
    cli.write_endpoint(str(path), 1111, "private")
    before = path.read_text()

    def refuse(*_):
        raise OSError("the disk went away")

    monkeypatch.setattr(os, "replace", refuse)
    with pytest.raises(OSError):
        cli.write_endpoint(str(path), 2222, "private")
    assert path.read_text() == before and sorted(p.name for p in tmp_path.iterdir()) == ["endpoint"]


def test_only_the_endpoint_file_this_process_wrote_is_removed(tmp_path):
    path = tmp_path / "endpoint"
    cli.write_endpoint(str(path), 1111, "private")
    cli.remove_endpoint(str(path))
    assert not path.exists()
    cli.remove_endpoint(str(path))                              # already gone: nothing to say
    path.write_text(json.dumps({"host": socket.gethostname(), "port": 9, "mode": "private", "pid": os.getpid() + 1, "started_at": "x"}))
    cli.remove_endpoint(str(path))                              # a later server's file: not ours
    assert path.exists()
    path.write_text(json.dumps({"host": "another-node", "port": 9, "mode": "private", "pid": os.getpid(), "started_at": "x"}))
    cli.remove_endpoint(str(path))                              # the same pid on another node: not ours either
    assert path.exists()
    path.write_text("not json\n")
    cli.remove_endpoint(str(path))
    assert path.read_text() == "not json\n"


# ---- the job log filter ------------------------------------------------------------------------------------------------------

def test_the_job_log_filter_passes_four_shapes_and_sends_every_other_line_aside():
    out, rest = io.StringIO(), io.StringIO()
    keep = cli.JobLogFilter(out, rest)
    keep.write("[server] sqlite journal_mode=wal\nLoaded checkpoint: /sc/home/x/last.ckpt\n=== a ===\n  prefix_k = 32\nRES")
    keep.write("ULT {}\nERROR x\n[servers] no\n[server]no\nINFO: Uvicorn running\npartial")
    keep.finish()
    assert out.getvalue() == "[server] sqlite journal_mode=wal\n=== a ===\nRESULT {}\nERROR x\n"
    assert rest.getvalue() == ("Loaded checkpoint: /sc/home/x/last.ckpt\n  prefix_k = 32\n[servers] no\n[server]no\n"
                               "INFO: Uvicorn running\npartial\n")


def test_the_job_log_filter_flushes_both_streams_and_looks_like_a_text_stream():
    class Counting(io.StringIO):
        flushed = 0

        def flush(self):
            self.flushed += 1

    out, rest = Counting(), Counting()
    keep = cli.JobLogFilter(out, rest)
    keep.flush()
    assert out.flushed == 1 and rest.flushed == 1
    assert keep.writable() and not keep.isatty()


def test_every_line_is_written_whole_in_one_call(monkeypatch):
    """print() writes the text and the newline apart, and on unbuffered output those are two writes: SLURM's SIGTERM reaches the wrapper and
    the server at once, and a line of the wrapper's could land between the halves of one of ours."""
    class Recording(io.StringIO):
        writes = []

        def write(self, text):
            self.writes.append(text)
            return super().write(text)

    out = Recording()
    monkeypatch.setattr(sys, "stdout", out)
    cli.say("serving: retrieval=on")
    cli.emit("ERROR x")
    assert out.writes == ["[server] serving: retrieval=on\n", "ERROR x\n"]


# ---- what main() hands on ----------------------------------------------------------------------------------------------------

class FakeWorker:
    def __init__(self):
        self.stopped = False

    def shutdown(self):
        self.stopped = True


class FakeStore:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def fake_app(gallery=None, labeler=None, published=None):
    return SimpleNamespace(state=SimpleNamespace(gallery=gallery, labeler=labeler, published=published, worker=FakeWorker(), store=FakeStore()))


@pytest.fixture
def serving(monkeypatch):
    """main() with no model and no socket: create_app and the server are recorders."""
    rec = SimpleNamespace(create_app=None, config=None, server_kw=None, ran=0, app=fake_app(), build=None, serve=None)

    def fake_create_app(**kw):
        rec.create_app = kw
        if rec.build is not None:
            rec.build()
        return rec.app

    class FakeServer:
        def __init__(self, config, app, endpoint_file=None, mode="private"):
            rec.config, rec.server_kw = config, {"endpoint_file": endpoint_file, "mode": mode}

        def run(self):
            rec.ran += 1
            if rec.serve is not None:       # what happens while the server is up
                rec.serve()

    monkeypatch.setattr(cli, "create_app", fake_create_app)
    monkeypatch.setattr(cli, "ServingServer", FakeServer)
    return rec


def test_main_hands_every_argument_to_create_app_as_it_reads_it(serving, tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    serving.app = fake_app(gallery=object())            # --gallery is given below, and a serving process wants it opened
    token = make_token(tmp_path)
    assert cli.main(["--engine", "real", "--mode", "public", "--home", "H", "--device", "cuda", "--gallery", "G", "--labeler",
                     "http://127.0.0.1:8001", "--host", "127.0.0.1", "--port", "8123", "--endpoint-file", "E", "--token-file", str(token),
                     "--models", "m3,13d", "--threads", "4", "--drift-note", "CPU vs GPU: 0/20 differ", "--allow-compile",
                     "--published-model", "PM", "--published-floor", "PF"]) == 0
    assert serving.create_app == {
        "engine": "real", "mode": "public", "home": "H", "host": "127.0.0.1", "token": "s3cret-token", "gallery_dir": "G",
        "labeler_url": "http://127.0.0.1:8001", "labeler": None, "models": ("hybrid_150m_m3_rrg", "hybrid_150m_v2_rrg"),
        "allow_compile": True, "drift_note": "CPU vs GPU: 0/20 differ", "device": "cuda", "threads": 4,
        "published_dirs": {"model": "PM", "floor": "PF"}}
    assert serving.ran == 1 and serving.server_kw == {"endpoint_file": "E", "mode": "public"}


def test_main_with_the_defaults_asks_for_no_gallery_labeller_or_dumps(serving):
    assert cli.main(BASE) == 0
    assert serving.create_app == {
        "engine": "tiny", "mode": "private", "home": "HOME", "host": "127.0.0.1", "token": None, "gallery_dir": None, "labeler_url": None,
        "labeler": None, "models": ("hybrid_150m_m3_rrg",), "allow_compile": False, "drift_note": "", "device": "cpu", "threads": 8,
        "published_dirs": None}
    assert serving.server_kw == {"endpoint_file": None, "mode": "private"}


def test_the_labeler_flag_picks_a_client_the_rule_labeler_or_nothing(serving):
    cli.main(BASE + ["--labeler", "rule"])
    assert isinstance(serving.create_app["labeler"], RuleLabeler) and serving.create_app["labeler_url"] is None
    cli.main(BASE + ["--labeler", "http://127.0.0.1:8001"])
    assert serving.create_app["labeler"] is None and serving.create_app["labeler_url"] == "http://127.0.0.1:8001"
    cli.main(BASE + ["--labeler", "none"])
    assert serving.create_app["labeler"] is None and serving.create_app["labeler_url"] is None


def test_uvicorn_gets_the_bind_no_access_log_and_a_graceful_period_shorter_than_slurms_kill_wait(serving):
    cli.main(BASE + ["--host", "localhost", "--port", "8123"])
    config = serving.config
    assert (config.host, config.port, config.access_log) == ("localhost", 8123, False)
    assert config.timeout_graceful_shutdown == cli.GRACEFUL_S and 0 < cli.GRACEFUL_S <= 15, "SLURM's KillWait is 30 s by default"


def test_main_returns_zero_and_leaves_stdout_to_server_lines_even_when_the_loader_prints(serving, capsys):
    def loader_prints():
        print("[server] sqlite journal_mode=wal", flush=True)
        print("Loaded checkpoint: /sc/home/someone/outputs/x/checkpoints/last.ckpt")
        print("  Missing keys: 0, Unexpected: 0")
        print("  prefix_k = 32")

    serving.build = loader_prints
    assert cli.main(BASE) == 0
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert lines and not [l for l in lines if not JOB_LOG_LINE.match(l)], lines
    assert "[server] sqlite journal_mode=wal" in lines and any(l.startswith("[server] starting:") for l in lines)
    assert "Loaded checkpoint" in captured.err and "prefix_k = 32" in captured.err, "the loader's own lines went to the server log"


def test_the_filter_stays_on_stdout_for_the_whole_life_of_the_process_and_stdout_is_restored_after(serving, capsys):
    """Not only while the app loads: a library that prints while the server is up (a progress line, a stray debug print) must not put a line
    of no known shape into the job log. The server's own [server] lines still pass."""
    def while_serving():
        print("a stray line from a library while serving")
        print("[server] a line of the server's own")
        cli.say("stopping")
        sys.stdout.write("half a line, no newline")

    serving.serve = while_serving
    real_stdout = sys.stdout
    assert cli.main(BASE) == 0
    assert sys.stdout is real_stdout, "main hands stdout back"
    shown = capsys.readouterr()
    lines = shown.out.splitlines()
    assert "[server] a line of the server's own" in lines and "[server] stopping" in lines
    assert not [l for l in lines if not JOB_LOG_LINE.match(l)], lines
    assert "a stray line from a library while serving" not in shown.out and "half a line" not in shown.out
    assert "a stray line from a library while serving" in shown.err and "half a line, no newline" in shown.err, "they went to the server log"


def test_main_says_what_it_serves_without_a_path_a_host_or_a_token(serving, capsys, tmp_path):
    token = make_token(tmp_path, "tok-5d1e-main\n")
    serving.build = None
    cli.main(["--engine", "tiny", "--mode", "public", "--home", str(tmp_path / "homedir"), "--token-file", str(token),
              "--endpoint-file", str(tmp_path / "ep"), "--host", "127.0.0.1", "--port", "8123"])
    out = capsys.readouterr().out
    assert "tok-5d1e-main" not in out and str(tmp_path) not in out and "8123" not in out and "127.0.0.1" not in out


def test_a_failure_while_the_app_loads_is_one_error_line_with_the_class_only(monkeypatch, capsys):
    def boom(**_):
        raise RuntimeError("secret detail /sc/home/someone/chat_sessions")

    monkeypatch.setattr(cli, "create_app", boom)
    assert cli.main(BASE) == 1
    captured = capsys.readouterr()
    errors = [l for l in captured.out.splitlines() if l.startswith("ERROR")]
    assert len(errors) == 1 and "RuntimeError" in errors[0] and "secret detail" not in captured.out
    assert "secret detail" in captured.err, "the traceback is in the server log, not in the job log"


def test_an_interrupt_while_the_app_loads_ends_the_start_quietly(monkeypatch, capsys):
    def interrupt(**_):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "create_app", interrupt)
    assert cli.main(BASE) == 130
    assert "ERROR interrupted while starting" in capsys.readouterr().out


def test_a_server_that_will_not_start_is_one_error_line_and_its_exit_code(serving, capsys, monkeypatch):
    class Stuck:
        def __init__(self, *a, **k):
            pass

        def run(self):
            raise SystemExit(3)

    monkeypatch.setattr(cli, "ServingServer", Stuck)
    assert cli.main(BASE) == 3
    assert "ERROR the server did not start (exit 3)" in capsys.readouterr().out


# ---- a serving process is strict about its gallery and its GPU -------------------------------------------------------------------

def test_a_gallery_that_will_not_open_stops_the_server_unless_it_is_explicitly_allowed(serving, capsys, tmp_path):
    gallery = tmp_path / "gallery" / "g13d_m3_v1"
    assert cli.main(BASE + ["--gallery", str(gallery)]) == 1                       # create_app's lenient path left gallery None
    out = capsys.readouterr().out
    assert [l for l in out.splitlines() if l.startswith("ERROR")] == ["ERROR the retrieval gallery could not be opened (see the [server] gallery line)"]
    assert str(tmp_path) not in out and serving.ran == 0
    assert serving.app.state.worker.stopped and serving.app.state.store.closed, "the half-built app is released, not abandoned"
    assert cli.main(BASE + ["--gallery", str(gallery), "--allow-no-gallery"]) == 0 and serving.ran == 1
    assert "could not be opened" not in capsys.readouterr().out


def test_an_opened_gallery_or_no_gallery_asked_for_is_not_an_error(serving):
    serving.app = fake_app(gallery=object())
    assert cli.main(BASE + ["--gallery", "G"]) == 0 and serving.ran == 1
    serving.app = fake_app()
    assert cli.main(BASE) == 0 and serving.ran == 2


def test_a_real_engine_on_cuda_needs_a_cuda_that_is_there(serving, capsys, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert cli.main(["--engine", "real", "--mode", "private", "--home", "HOME", "--device", "cuda"]) == 1
    out = capsys.readouterr().out
    assert [l for l in out.splitlines() if l.startswith("ERROR")] == ["ERROR CUDA is not available on this node"]
    assert serving.create_app is None, "refused before any model was loaded"
    assert cli.main(["--engine", "real", "--mode", "private", "--home", "HOME", "--device", "cpu"]) == 0
    assert cli.main(BASE + ["--device", "cuda"]) == 0, "the tiny engine has no device to be missing"


# ---- create_app's two new inputs ---------------------------------------------------------------------------------------------

def test_create_app_hands_the_device_and_the_threads_to_the_real_engines(tmp_path, monkeypatch):
    calls = []

    def fake(kind, **kw):
        calls.append(kw)
        engine = build_engine("tiny")
        engine.name = kw["model_config"]
        engine._card.update(name=engine.name, checkpoint=kw["checkpoint"])
        return engine

    monkeypatch.setattr(server, "build_engine", fake)
    create_app(engine="real", home=str(tmp_path / "a"), device="cuda", threads=3).state.store.close()
    assert calls[0]["device"] == "cuda" and calls[0]["threads"] == 3
    create_app(engine="real", home=str(tmp_path / "b")).state.store.close()
    assert calls[1]["device"] == "cpu" and "threads" not in calls[1], "without a thread count the engine keeps its own default"
    with pytest.raises(ValueError, match="device"):
        create_app(engine="real", home=str(tmp_path / "c"), device="tpu")
    assert not (tmp_path / "c").exists(), "refused before anything was opened"


# ---- the shutdown ------------------------------------------------------------------------------------------------------------

def test_a_stop_ends_a_running_turn_as_a_server_restart_at_once_and_clears_the_endpoint(tmp_path):
    """The turns are stopped when the shutdown begins, not after uvicorn's graceful period: with a turn that would run for 40 s, the
    stream has ended and the server is gone well inside half of GRACEFUL_S."""
    app = create_app(engine="tiny", home=str(tmp_path / "home"), tiny_step_delay_s=0.2)
    endpoint = tmp_path / "endpoint"
    config = uvicorn.Config(app, host="127.0.0.1", port=0, access_log=False, log_level="warning", timeout_graceful_shutdown=cli.GRACEFUL_S)
    serving_server = cli.ServingServer(config, app, endpoint_file=str(endpoint), mode="private")
    runner = threading.Thread(target=serving_server.run, daemon=True)
    runner.start()
    frames, saw_text = [], threading.Event()
    try:
        wait_until(endpoint.exists, timeout=30)
        assert json.loads(endpoint.read_text())["mode"] == "private"
        base = "http://127.0.0.1:{}".format(json.loads(endpoint.read_text())["port"])
        sid = httpx.post(base + "/v1/sessions", json={}).json()["id"]

        def stream():
            with httpx.stream("POST", base + "/v1/sessions/{}/messages".format(sid), files={"image": ("x.png", png_bytes(), "image/png")},
                              data={"text": "", "options": json.dumps({"max_new_tokens": 200})}, timeout=60) as response:
                for frame in iter_sse(response.iter_text()):
                    frames.append(frame)
                    if frame["event"] == "content_block_delta":
                        saw_text.set()

        reader = threading.Thread(target=stream, daemon=True)
        reader.start()
        assert saw_text.wait(30), "the turn never began to decode"
        began = time.monotonic()
        serving_server.should_exit = True
        reader.join(cli.GRACEFUL_S)
        assert not reader.is_alive() and time.monotonic() - began < cli.GRACEFUL_S / 2
        runner.join(cli.GRACEFUL_S)
        assert not runner.is_alive() and time.monotonic() - began < cli.GRACEFUL_S / 2
    finally:
        serving_server.should_exit = True
        runner.join(10)
    errors = [f["data"]["error"]["type"] for f in frames if f["event"] == "error"]
    assert errors == ["server_restart"] and frames[-1]["event"] == "message_stop" and frames[-1]["data"]["status"] == "error"
    assert not endpoint.exists(), "the endpoint file goes with the server"
    store = Store(tmp_path / "home")
    try:
        assert store.get_message(frames[0]["data"]["message_id"], None)["status"] == "error", "and the stored turn says the same"
    finally:
        store.close()


def test_the_endpoint_file_goes_as_the_shutdown_begins_not_after_the_running_turn_has_stopped(tmp_path, monkeypatch):
    """The tunnel re-reads the endpoint file to find a server: it must stop finding this one at once, not once the drain is over. The turn is
    held inside generate, and only the test frees it (the shutdown's stop is set, but this wait does not look at it), so the drain cannot be
    over while the file is looked for."""
    real = TinyEngine.generate
    entered, release = threading.Event(), threading.Event()

    def held(self, enc, opts, on_snapshot, cancel):
        entered.set()
        release.wait(60)
        return real(self, enc, opts, on_snapshot, cancel)

    monkeypatch.setattr(TinyEngine, "generate", held)
    app = create_app(engine="tiny", home=str(tmp_path / "home"))
    endpoint = tmp_path / "endpoint"
    config = uvicorn.Config(app, host="127.0.0.1", port=0, access_log=False, log_level="warning", timeout_graceful_shutdown=cli.GRACEFUL_S)
    serving_server = cli.ServingServer(config, app, endpoint_file=str(endpoint), mode="private")
    runner = threading.Thread(target=serving_server.run, daemon=True)
    runner.start()
    frames = []
    try:
        wait_until(endpoint.exists, timeout=30)
        base = "http://127.0.0.1:{}".format(json.loads(endpoint.read_text())["port"])
        sid = httpx.post(base + "/v1/sessions", json={}).json()["id"]

        def stream():
            with httpx.stream("POST", base + "/v1/sessions/{}/messages".format(sid), files={"image": ("x.png", png_bytes(), "image/png")},
                              data={"text": "", "options": json.dumps({"max_new_tokens": 16})}, timeout=60) as response:
                frames.extend(iter_sse(response.iter_text()))

        reader = threading.Thread(target=stream, daemon=True)
        reader.start()
        assert entered.wait(30), "the turn never reached generate"
        serving_server.should_exit = True
        wait_until(lambda: not endpoint.exists(), timeout=cli.GRACEFUL_S / 2)
        assert reader.is_alive() and runner.is_alive(), "the file went, and the turn it belonged to has not finished stopping"
        release.set()
        reader.join(cli.GRACEFUL_S)
        assert not reader.is_alive()
        runner.join(cli.GRACEFUL_S)
        assert not runner.is_alive()
    finally:
        release.set()
        serving_server.should_exit = True
        runner.join(10)
    assert [f["data"]["error"]["type"] for f in frames if f["event"] == "error"] == ["server_restart"]
    assert frames[-1]["event"] == "message_stop" and frames[-1]["data"]["status"] == "error"


def test_a_signal_while_the_model_loads_takes_the_default_action_and_ends_the_process_at_once(tmp_path):
    """What app/cli.py's docstring says: until uvicorn serves there is no handler, so SIGTERM ends the process on the spot, which the wrapper (it
    sent the signal) counts as a stop. A handler that waited for the load to finish would not be heard for minutes, and SLURM's KillWait is 30 s."""
    script = "import sys, time\nfrom app import cli\ncli.create_app = lambda **kw: time.sleep(120)\nsys.exit(cli.main(sys.argv[1:]))\n"
    out = tmp_path / "stdout.txt"
    with open(str(out), "wb") as stdout:
        proc = subprocess.Popen([sys.executable, "-c", script, "--engine", "tiny", "--mode", "private", "--home", str(tmp_path / "home")],
                                cwd=str(REPO_ROOT), stdin=subprocess.DEVNULL, stdout=stdout, stderr=subprocess.DEVNULL,
                                env=dict(os.environ, PYTHONUNBUFFERED="1"), start_new_session=True)
        try:
            wait_until(lambda: "[server] starting:" in out.read_text(), timeout=60)
            began = time.monotonic()
            proc.send_signal(signal.SIGTERM)
            assert proc.wait(10) == -signal.SIGTERM and time.monotonic() - began < 5
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()


def test_a_stopping_server_accepts_no_new_turn_and_says_so_before_uvicorn_starts_to_wait(monkeypatch, capsys):
    """A request that is inside post_message when the signal lands must be refused (reserve() is False: the busy 429), not queued behind
    the stop, where Worker.submit would find the pool shut down and the turn's row would stay `running` until the next start."""
    from app.pipeline import Worker
    worker = Worker(None, 4)
    serving_server = cli.ServingServer(uvicorn.Config(object(), port=0), SimpleNamespace(state=SimpleNamespace(worker=worker)))
    seen = {}

    async def uvicorn_waits(self, sockets=None):
        seen["reserved"], seen["said"] = worker.reserve(), capsys.readouterr().out

    monkeypatch.setattr(uvicorn.Server, "shutdown", uvicorn_waits)
    assert worker.reserve() is True and worker.in_flight == 1      # open for turns until the stop
    worker.release()
    asyncio.run(serving_server.shutdown())
    assert seen["reserved"] is False and worker.in_flight == 0
    assert seen["said"] == "[server] stopping\n" and capsys.readouterr().out == "[server] stopped\n"


def test_the_endpoint_file_appears_only_once_the_server_is_serving_and_names_its_real_port(tmp_path):
    app = create_app(engine="tiny", home=str(tmp_path / "home"))
    endpoint = tmp_path / "endpoint"
    config = uvicorn.Config(app, host="127.0.0.1", port=0, access_log=False, log_level="warning")
    serving_server = cli.ServingServer(config, app, endpoint_file=str(endpoint), mode="private")
    assert not endpoint.exists()
    runner = threading.Thread(target=serving_server.run, daemon=True)
    runner.start()
    try:
        wait_until(endpoint.exists, timeout=30)
        data = json.loads(endpoint.read_text())
        assert data["port"] != 0 and httpx.get("http://127.0.0.1:{}/healthz".format(data["port"])).json()["status"] == "ok"
    finally:
        serving_server.should_exit = True
        runner.join(15)
    assert not runner.is_alive() and not endpoint.exists()


# ---- the real CLI, as a child process on the tiny engine ---------------------------------------------------------------------

class LiveCli:
    """`python -m app.server` on the tiny engine with its output in files; whatever still runs when the test is over is killed."""

    def __init__(self, tmp_path, *extra):
        self.endpoint, self.out, self.err = tmp_path / "endpoint", tmp_path / "stdout.txt", tmp_path / "stderr.txt"
        argv = [sys.executable, "-m", "app.server", "--engine", "tiny", "--home", str(tmp_path / "home"), "--port", "0",
                "--endpoint-file", str(self.endpoint)] + list(extra)
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
        env.pop("CHAT_HOME", None)
        self._files = (open(str(self.out), "wb"), open(str(self.err), "wb"))
        self.proc = subprocess.Popen(argv, cwd=str(REPO_ROOT), env=env, stdin=subprocess.DEVNULL, stdout=self._files[0], stderr=self._files[1],
                                     start_new_session=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait()
        for handle in self._files:
            handle.close()

    def base(self):
        wait_until(lambda: self.endpoint.exists() or self.proc.poll() is not None, timeout=120)
        assert self.proc.poll() is None, "the CLI ended before it served: " + self.out.read_text() + self.err.read_text()
        return "http://127.0.0.1:{}".format(json.loads(self.endpoint.read_text())["port"])

    def stop(self, sig=signal.SIGTERM, timeout=30):
        self.proc.send_signal(sig)
        return self.proc.wait(timeout)


def test_a_live_private_server_answers_then_exits_cleanly_on_sigterm_and_removes_its_endpoint(tmp_path):
    with LiveCli(tmp_path, "--mode", "private", "--labeler", "rule") as live:
        base = live.base()
        health = httpx.get(base + "/healthz").json()
        assert health["status"] == "ok" and health["mode"] == "private"
        models = httpx.get(base + "/v1/models").json()
        assert models["features"] == {"retrieval": False, "labels": True}, "the rule labeller is wired, there is no gallery"
        data = json.loads(live.endpoint.read_text())
        assert data["mode"] == "private" and data["pid"] == live.proc.pid and data["host"] == socket.gethostname()
        assert stat.S_IMODE(live.endpoint.stat().st_mode) == 0o600
        live.proc.send_signal(signal.SIGTERM)
        assert live.stop(signal.SIGTERM) == 0, "a second SIGTERM (SLURM's, then the wrapper's) changes nothing; the exit is a normal one"
    assert not live.endpoint.exists()
    stdout = live.out.read_text()
    lines = stdout.splitlines()
    assert lines and not [l for l in lines if not JOB_LOG_LINE.match(l)], lines
    assert any(l.startswith("[server] serving:") and "labels=on" in l and "retrieval=off" in l for l in lines)
    assert lines[-1] == "[server] stopped"
    assert str(data["port"]) not in stdout, "the endpoint never reaches a job log (R7)"
    assert "Uvicorn running" in live.err.read_text(), "uvicorn's own log is on stderr, which the wrapper keeps in a file"


def test_a_live_public_server_needs_the_token_and_keeps_it_out_of_its_output_and_sigint_is_a_clean_stop_too(tmp_path):
    token = make_token(tmp_path, "tok-7f3a9c-live\n")
    with LiveCli(tmp_path, "--mode", "public", "--token-file", str(token)) as live:
        base = live.base()
        assert httpx.get(base + "/healthz").json()["mode"] == "public", "/healthz stays open"
        assert httpx.get(base + "/v1/models").status_code == 401
        ok = httpx.get(base + "/v1/models", headers={"Authorization": "Bearer tok-7f3a9c-live"})
        assert ok.status_code == 200 and ok.json()["mode"] == "public"
        assert json.loads(live.endpoint.read_text())["mode"] == "public"
        assert live.stop(signal.SIGINT) == 0
    assert not live.endpoint.exists()
    for text in (live.out.read_text(), live.err.read_text()):
        assert "tok-7f3a9c-live" not in text


def test_python_dash_m_app_server_leaves_a_secret_given_as_an_argument_in_neither_stream(tmp_path):
    """The operator's slip: `--token SECRET` (the token is a file, so this is no option). The refusal names the option, and the secret is in
    neither stdout, which is the job log, nor stderr, which is the server log."""
    done = subprocess.run([sys.executable, "-m", "app.server", "--engine", "tiny", "--mode", "private", "--home", str(tmp_path / "home"),
                           "--token", "SECRET-from-the-command-line"],
                          cwd=str(REPO_ROOT), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
    assert done.returncode == 2
    assert done.stdout == "ERROR unrecognized arguments: --token\n", done.stdout
    assert "SECRET" not in done.stdout and "SECRET" not in done.stderr
    assert not (tmp_path / "home").exists()


def test_python_dash_m_app_server_refuses_public_mode_without_a_token_before_loading_anything(tmp_path):
    done = subprocess.run([sys.executable, "-m", "app.server", "--engine", "tiny", "--mode", "public", "--home", str(tmp_path / "home")],
                          cwd=str(REPO_ROOT), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                          timeout=120)
    assert done.returncode == 2
    assert done.stdout.startswith("ERROR public mode needs a token file") and len(done.stdout.splitlines()) == 1
    assert not (tmp_path / "home").exists(), "nothing was created"
