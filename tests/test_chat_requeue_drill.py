"""CHAT_UI_PLAN.md P7-E, the laptop part: scripts/chat_requeue_drill.py, the requeue drill run from the laptop against the chat server through the
tunnel. The plan has `scontrol requeue <job>` sent while a turn runs: the turn is stored `error` with `server_restart`, the job comes back, and the
tunnel re-attaches by itself. The live server's turns carry MIMIC-derived text (the reports, the matched reports), and under rule R7 no agent may read
any of it, so the drill is a script that prints numbers, booleans and fixed words, and nothing else. It starts nothing of its own: the cluster is an
object handed to it, whose real form runs two ssh commands and whose test forms signal a server process or flip a fake. Here every run is against
synthetic data: a scripted fake of the server (FakeChat) that puts a SECRET in every text it sends, or a real server process on the tiny engine, so
that an assertion may print whatever it compares. No test contacts port 8000 or 8010, and none runs ssh.
"""
import io
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from app import cli
from app.schemas import Options
from app.server import MODEL_CHECKPOINTS
from scripts import chat_load_drill as drill
from scripts import chat_realturn_probe as probe
from scripts import chat_requeue_drill as requeue
from scripts.chat_ui_cdp import free_port, kill_group, stop_group
from tests.test_chat_load_drill import NoNetwork, leaves, refusal_line, write_token_file

REPO_ROOT = Path(__file__).resolve().parent.parent
SECRET = "s3cr3t-T0KEN-value"
TOKEN = "t0k-for-the-drill"
JOB = "2650001"


@pytest.fixture(autouse=True)
def no_ambient_token(monkeypatch):
    """Whoever runs the real drill has CHAT_TOKEN exported in that shell: no test may send it, to a local test server or anywhere."""
    monkeypatch.delenv("CHAT_TOKEN", raising=False)


@pytest.fixture(autouse=True)
def no_real_ssh(monkeypatch):
    """The real cluster runs ssh through the `subprocess` the script imported, and no test may start it: the script's own reference is replaced by a
    copy of the module whose run() fails the test. Every other module keeps the real subprocess (the app asks git for its provenance with it)."""
    def forbidden(*args, **kwargs):
        raise AssertionError("a test started a process through the script: that is ssh")
    monkeypatch.setattr(requeue, "subprocess", SimpleNamespace(**dict(vars(subprocess), run=forbidden)))


@pytest.fixture(autouse=True)
def never_the_ports_of_the_users_own_servers(monkeypatch):
    """8000 is the user's dev server and 8010 the tunnel to the live cluster server: no test may connect to either, whatever it says in a URL. The
    socket's own connect is guarded for the whole module, so that a test that was wrong about its fake or its server would fail instead."""
    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex

    def guard(address):
        if isinstance(address, tuple) and len(address) >= 2 and address[1] in (8000, 8010):
            raise AssertionError("a test tried to connect to a port of the user's own servers")

    def connect(self, address):
        guard(address)
        return real_connect(self, address)

    def connect_ex(self, address):
        guard(address)
        return real_connect_ex(self, address)
    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)


def test_the_guard_that_keeps_every_test_off_the_users_ports_stops_a_connection_before_it_is_made():
    for port in (8000, 8010):
        with pytest.raises(AssertionError, match="the user's own servers"):
            socket.create_connection(("127.0.0.1", port), timeout=1)
        with socket.socket() as sock, pytest.raises(AssertionError, match="the user's own servers"):
            sock.connect_ex(("127.0.0.1", port))


@pytest.fixture
def offline(monkeypatch):
    """A refusal must come before anything is opened: httpx's client and its own request functions fail the test when called."""
    monkeypatch.setattr(httpx, "Client", NoNetwork)
    for name in ("get", "post", "delete", "request", "stream"):
        monkeypatch.setattr(httpx, name, NoNetwork)


class NoCluster:
    """The cluster of a test that must not reach it (a refusal, a server that is down): any use of it fails the test."""

    def node(self, job):
        raise AssertionError("the cluster was asked for a node")

    def requeue(self, job):
        raise AssertionError("the job was requeued")


def run_main(args, cluster=None, expect_code=None, poll_s=0.05):
    """main() with its own two streams. -> (code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    code = requeue.main(list(args), out, err, cluster=NoCluster() if cluster is None else cluster, poll_s=poll_s)
    if expect_code is not None:
        assert code == expect_code, (code, out.getvalue(), err.getvalue())
    return code, out.getvalue(), err.getvalue()


def the_result(stdout):
    """The one RESULT line of a run, parsed."""
    lines = stdout.splitlines()
    assert len(lines) == 1 and lines[0].startswith("RESULT "), lines
    return json.loads(lines[0][len("RESULT "):])


# ---- what the drill asks for: the published model, 200 tokens, three deltas ---------------------------------------------------------

def test_the_drill_names_the_published_model_the_server_serves_and_the_serving_line_calls_it_m3():
    """The request names the model as the server's registry does (`options.model` is a registry name: `m3` is only the serving command's alias for
    it, `--models m3`, app/cli.py), and not the EOS model, which may stop before its budget: the published one always decodes to the cap."""
    assert requeue.MODEL == probe.PUBLISHED_MODEL == "hybrid_150m_m3_rrg" and requeue.MODEL in MODEL_CHECKPOINTS
    assert cli.MODEL_ALIASES["m3"] == requeue.MODEL and requeue.MODEL != probe.EOS_MODEL == cli.MODEL_ALIASES["m3eos"]


def test_the_turn_runs_to_the_cap_of_200_tokens_under_the_published_protocol_and_the_server_validates_its_options():
    assert (requeue.MAX_NEW_TOKENS, drill.MAX_NEW_TOKENS) == (200, 200), "the server's cap: the longest turn there is"
    options = requeue.turn_options()
    assert options == probe.turn_options(model="hybrid_150m_m3_rrg", max_new_tokens=200)
    assert options["stop_on_repeat"] is False and options["decode"] == "beam" and options["beam_size"] == 3, "no stop: it runs to the budget"
    assert Options.model_validate(options).max_new_tokens == 200


def test_the_requeue_waits_for_three_content_block_deltas_the_event_the_pipeline_sends_once_for_every_decode_step():
    """Under beam search the published model sends one content_block_delta per step, with delta.type beam_snapshot (app/pipeline.py), and the tiny
    engine goes through the same pipeline: the name is counted, never the text it carries."""
    source = (REPO_ROOT / "app" / "pipeline.py").read_text(encoding="utf-8")
    assert requeue.MIN_DELTAS == 3 and requeue.DELTA == "content_block_delta"
    assert 'self._send(turn, "content_block_delta",' in source and '"type": "beam_snapshot"' in source


def test_the_event_names_it_knows_are_those_the_app_sends():
    """The last event of the stored turn is kept as a name from this closed list, and any other is `other`: the list is the app's, nine names."""
    source = (REPO_ROOT / "app" / "pipeline.py").read_text(encoding="utf-8")
    sent = set(re.findall(r'self\._send\(turn, "([a-z_]+)"', source)) | {"message_stop"}   # _stop writes it through the store
    assert sent == set(requeue.EVENT_NAMES), (sorted(sent), sorted(requeue.EVENT_NAMES))
    assert "message_stop" in source and set(requeue.STORED_STATUSES) == {"running", "done", "error", "aborted"}


def test_the_client_id_is_one_fixed_value_the_server_takes():
    from app.server import CLIENT_ID as SERVER_CLIENT_ID
    ids = {drill.CLIENT_ID, probe.CLIENT_ID}
    assert requeue.CLIENT_ID == "chat-requeue-drill" and requeue.CLIENT_ID not in ids and SERVER_CLIENT_ID.fullmatch(requeue.CLIENT_ID)


def test_the_defaults_are_a_wait_of_30_minutes_and_a_poll_every_5_seconds():
    args = requeue.build_parser().parse_args(["http://127.0.0.1:18080", "--job", JOB])
    assert args.job == JOB and args.wait == requeue.DEFAULT_WAIT_S == 1800.0 and args.token_file is None
    assert requeue.POLL_S == 5.0


# ---- what the script refuses: it never connects -------------------------------------------------------------------------------------

@pytest.mark.parametrize("url", ["http://127.0.0.1:8000", "http://127.0.0.1:8000/", "http://localhost:8000", "http://[::1]:8000/",
                                 "127.0.0.1:8000", "localhost:8000", "https://example.org:8000", "HTTP://LOCALHOST:8000",
                                 "http://127.0.0.1:08000", "http://127.0.0.1:+8000", "http://127.0.0.1:８０００", "http://user@127.0.0.1:8000"])
def test_the_script_refuses_port_8000_in_one_line_with_exit_2_and_connects_to_nothing(url, offline):
    code, out, err = run_main([url, "--job", JOB])
    assert code == 2 and out == ""
    assert "8000" in refusal_line(err) and "dev server" in err


@pytest.mark.parametrize("url", ["http://127.0.0.1:18000", "http://127.0.0.1:8001", "http://127.0.0.1:80", "http://127.0.0.1:18080",
                                 "http://localhost", "http://127.0.0.1:8010"])
def test_only_port_8000_itself_is_refused_for_being_the_dev_server(url):
    """8010 is the tunnel's own port, the one the controller runs the drill on."""
    assert drill.check_url(url).startswith("http")


@pytest.mark.parametrize("url", ["", "   ", "http://", "ftp://127.0.0.1:2121", "http://127.0.0.1:0", "http://127.0.0.1:70000",
                                 "http://user:pass@127.0.0.1:1234", "http://tok@127.0.0.1:1234", "http://127.0.0.1:1234/?token=abc",
                                 "http://127.0.0.1:1234/#frag"])
def test_a_url_that_is_not_plain_scheme_host_and_port_is_refused_without_being_quoted(url, offline):
    code, out, err = run_main([url, "--job", JOB])
    assert code == 2 and out == ""
    line = refusal_line(err)
    assert "dev server" not in line and "127.0.0.1" not in line and "pass" not in line and "tok@" not in line and "abc" not in line


NOT_LOOPBACK = ["http://example.org:18080", "http://10.0.0.5:18080", "http://192.168.1.20", "http://[2001:db8::1]:18080",
                "http://[::ffff:10.0.0.5]:18080", "http://0.0.0.0:18080", "http://localhost.example.org:18080", "http://127.1:18080",
                "http://2130706433:18080", "http://node-07.internal", "example.org:18080", "https://example.org", "https://10.0.0.5:18080",
                "https://[2001:db8::1]:18080"]


@pytest.mark.parametrize("url", NOT_LOOPBACK)
def test_an_address_that_is_not_loopback_is_refused_in_one_line_before_anything_is_sent(url, offline):
    """The drill requeues a job: it goes to the tunnel's loopback end and nowhere else. What cannot be shown to be loopback is not (the drill's own
    test: a name that happens to resolve there, 127.1 and the integer form of 127.0.0.1 are all `not`)."""
    code, out, err = run_main([url, "--job", JOB])
    assert code == 2 and out == "" and refusal_line(err) == requeue.LOOPBACK_REFUSAL
    for piece in ("example", "10.0.0.5", "192.168", "internal", "2001", "2130706433", "127.1"):
        assert piece not in err, piece


@pytest.mark.parametrize("url", [u for u in NOT_LOOPBACK if u.startswith("http://") or "://" not in u])
def test_a_token_is_not_sent_over_plain_http_to_a_host_that_is_not_loopback_which_is_the_probes_refusal(url, offline, monkeypatch):
    monkeypatch.setenv("CHAT_TOKEN", SECRET)
    code, out, err = run_main([url, "--job", JOB])
    assert code == 2 and out == "" and refusal_line(err) == drill.CLEARTEXT_REFUSAL and SECRET not in err


def test_the_refusal_to_send_a_token_in_the_clear_holds_for_a_token_file_too(tmp_path, offline):
    path = write_token_file(tmp_path, TOKEN + "\n")
    code, out, err = run_main(["http://example.org:18080", "--job", JOB, "--token-file", str(path)])
    assert code == 2 and out == "" and refusal_line(err) == drill.CLEARTEXT_REFUSAL and TOKEN not in err


@pytest.mark.parametrize("argv", [
    ["http://127.0.0.1:18080", "--token", SECRET],                  # an option the script does not have
    ["http://127.0.0.1:18080", "--token=" + SECRET],
    ["http://127.0.0.1:18080", "--tok", SECRET],                    # an abbreviation of --token-file must not be taken for it
    ["http://127.0.0.1:18080", "--token-fil", SECRET],
    ["--token", SECRET, "http://127.0.0.1:18080"],                  # the secret in the URL's place
    ["http://127.0.0.1:18080", SECRET],                             # a second positional: URL TOKEN
    ["http://127.0.0.1:18080", "-t", SECRET],
    ["http://127.0.0.1:18080", "--bearer", SECRET, "--password=" + SECRET],
    ["http://127.0.0.1:18080", "--jo", JOB],                        # nor an abbreviation of --job
])
def test_the_script_refuses_a_token_on_the_command_line_and_never_echoes_it(argv, offline):
    code, out, err = run_main(argv)
    assert code == 2 and out == ""
    assert refusal_line(err) == drill.ARGV_REFUSAL, "one fixed sentence, which names no argument"
    assert SECRET not in err


@pytest.mark.parametrize("argv", [[], ["--wait", "60"], ["--job", JOB]])
def test_a_missing_url_is_a_usage_refusal_in_one_line(argv, offline):
    code, out, err = run_main(argv)
    assert code == 2 and out == ""
    assert refusal_line(err) == drill.URL_MISSING


def test_a_missing_job_is_a_usage_refusal_in_one_line_and_nothing_is_sent(offline):
    code, out, err = run_main(["http://127.0.0.1:18080"])
    assert code == 2 and out == "" and refusal_line(err) == requeue.JOB_MISSING and "--job" in requeue.JOB_MISSING


@pytest.mark.parametrize("job", ["", " ", "abc", "-1", "+5", "1.5", "12 34", "12;reboot", "12\n34", "$(id)", "٣٤٥", "1" * 13, "0x10", "12_3", "SECRET-job"])
def test_a_job_that_is_not_a_whole_number_is_refused_unquoted_because_it_goes_into_an_ssh_command(job, offline):
    """The job id is part of the commands the cluster runs, so only digits, at most 12 of them, are taken; and a refusal never says what it was given."""
    code, out, err = run_main(["http://127.0.0.1:18080", "--job", job])
    assert code == 2 and out == ""
    line = refusal_line(err)
    assert line.startswith("argument --job: ") and requeue.JOB_REFUSAL in line
    assert job.strip() == "" or job not in line


@pytest.mark.parametrize("job", ["1", "2650001", "123456789012", "007"])
def test_a_job_of_digits_is_taken(job):
    assert requeue.build_parser().parse_args(["http://127.0.0.1:18080", "--job", job]).job == job


@pytest.mark.parametrize("value", ["0", "-5", "soon", "nan", "86401", "inf", ""])
def test_a_wait_out_of_range_is_refused_in_one_line_before_anything_is_sent(value, offline):
    code, out, err = run_main(["http://127.0.0.1:18080", "--job", JOB, "--wait", value])
    assert code == 2 and out == ""
    line = refusal_line(err)
    assert "--wait" in line
    if not value.lstrip("-").isdigit():
        assert value == "" or value not in line, "a refusal does not quote the value it was given"


@pytest.mark.parametrize("argv", [["--help"], ["-h"], ["http://127.0.0.1:18080", "--help"]])
def test_help_goes_to_stdout_with_exit_0_needs_no_url_and_connects_to_nothing(argv, offline):
    code, out, err = run_main(argv)
    assert code == 0 and err == "" and out.startswith("usage: ")
    for word in ("--job", "--wait", "--token-file", "CHAT_TOKEN", "dev server", "RESULT", "1800", "loopback"):
        assert word in out, word


def test_a_token_file_that_will_not_do_is_refused_without_naming_the_file_or_what_is_in_it(tmp_path, offline):
    wide = write_token_file(tmp_path, SECRET, 0o644, "wide.token")
    code, out, err = run_main(["http://127.0.0.1:18080", "--job", JOB, "--token-file", str(wide)])
    assert code == 2 and out == "" and "0600" in refusal_line(err)
    assert SECRET not in err and "wide.token" not in err and str(tmp_path) not in err


def test_a_chat_token_that_is_not_visible_ascii_is_refused_without_being_quoted(monkeypatch, offline):
    for bad in ("has a space", "tokén", "tab\there", SECRET + " " + SECRET):
        monkeypatch.setenv("CHAT_TOKEN", bad)
        code, out, err = run_main(["http://127.0.0.1:18080", "--job", JOB])
        assert code == 2 and out == "" and "CHAT_TOKEN" in refusal_line(err)
        assert SECRET not in err and bad not in err


class Down:
    """In place of httpx.Client when a test needs the drill to get past its checks to its first request, which nothing answers: that is how a test
    sees it was not refused. Whatever is asked of it is a connection error, in words that must not be printed."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def _down(self, *args, **kwargs):
        raise httpx.ConnectError("could not connect to a thing with a secret in its name")

    get = post = delete = stream = request = _down


@pytest.mark.parametrize("url", ["http://localhost:18080", "http://127.0.0.1:18080", "http://127.9.9.9:18080", "http://[::1]:18080",
                                 "https://localhost:18080", "http://127.0.0.1:8010", "http://LOCALHOST", "http://[::ffff:127.0.0.1]:18080"])
def test_a_loopback_address_is_not_refused_and_a_server_that_is_down_ends_in_one_result_line_with_a_fixed_code(url, monkeypatch):
    """Past the checks the drill asks for /healthz, and nothing answers: one RESULT line, the class of the failure as the code, exit 1, and the
    cluster not touched (a drill that cannot see its server requeues nothing)."""
    monkeypatch.setenv("CHAT_TOKEN", SECRET)
    monkeypatch.setattr(httpx, "Client", Down)
    code, out, err = run_main([url, "--job", JOB])
    result = the_result(out)
    assert code == 1 and err == "" and result["error"] == "healthz_ConnectError" and result["ok"] is False
    assert SECRET not in out and "secret" not in out


# ---- the cluster: two ssh commands, and nothing else that starts a process ----------------------------------------------------------

class Ssh:
    """In place of the `subprocess` the script imported: run() writes down what it was asked and answers with a status, a stdout and a stderr (a
    stderr that holds a SECRET, as a scheduler's message might hold anything), or raises; the rest of the module is the real one."""

    def __init__(self, stdout="gx14\n", returncode=0, stderr="SECRET scheduler message", raises=None):
        self.asked, self.stdout, self.returncode, self.stderr, self.raises = [], stdout, returncode, stderr, raises
        self.TimeoutExpired, self.DEVNULL, self.CalledProcessError = subprocess.TimeoutExpired, subprocess.DEVNULL, subprocess.CalledProcessError

    def run(self, argv, **kwargs):
        self.asked.append((list(argv), kwargs))
        if self.raises is not None:
            raise self.raises
        return SimpleNamespace(returncode=self.returncode, stdout=self.stdout, stderr=self.stderr)


def with_ssh(monkeypatch, **kwargs):
    ssh = Ssh(**kwargs)
    monkeypatch.setattr(requeue, "subprocess", ssh)
    return ssh


def test_the_real_cluster_asks_for_the_jobs_node_with_exactly_one_ssh_command(monkeypatch):
    ssh = with_ssh(monkeypatch, stdout="gx14\n")
    assert requeue.SshCluster().node(JOB) == "gx14"
    (argv, kwargs), = ssh.asked
    assert argv == ["ssh", "-o", "BatchMode=yes", "hpi-hpc", "squeue", "-j", JOB, "-h", "-o", "%N"]
    assert kwargs["stdin"] == subprocess.DEVNULL and kwargs["capture_output"] is True and kwargs["timeout"] == requeue.SSH_TIMEOUT_S
    assert "shell" not in kwargs, "an argument list, never a line a shell would read"


def test_the_real_cluster_requeues_the_job_with_exactly_one_ssh_command(monkeypatch):
    ssh = with_ssh(monkeypatch, stdout="")
    assert requeue.SshCluster().requeue(JOB) is None
    (argv, kwargs), = ssh.asked
    assert argv == ["ssh", "-o", "BatchMode=yes", "hpi-hpc", "scontrol", "requeue", JOB]
    assert kwargs["stdin"] == subprocess.DEVNULL and kwargs["timeout"] == requeue.SSH_TIMEOUT_S and "shell" not in kwargs


def test_a_job_with_no_node_yet_is_none_and_the_output_is_stripped(monkeypatch):
    for text, node in (("\n", None), ("", None), ("   \n", None), ("  gx14 \n", "gx14")):
        with_ssh(monkeypatch, stdout=text)
        assert requeue.SshCluster().node(JOB) == node, repr(text)


@pytest.mark.parametrize("job", ["", "12;reboot", "12 34", "$(id)", "-1", "1" * 13, "12\n", None, 12])
def test_the_real_cluster_runs_nothing_for_a_job_that_is_not_digits_even_when_it_is_called_with_one(job, monkeypatch):
    """The parser takes only digits, and the cluster asks again: it is the one place a job id is put into a command."""
    ssh = with_ssh(monkeypatch)
    for call in (requeue.SshCluster().node, requeue.SshCluster().requeue):
        with pytest.raises(requeue.ClusterError):
            call(job)
    assert ssh.asked == []


@pytest.mark.parametrize("failure, word", [
    (dict(returncode=1), "failed"), (dict(returncode=255), "failed"), (dict(returncode=-9), "failed"),
    (dict(raises=subprocess.TimeoutExpired(["ssh", "SECRET"], 60)), "timeout"),
    (dict(raises=FileNotFoundError("SECRET: ssh")), "not_run"), (dict(raises=PermissionError("SECRET")), "not_run"),
])
def test_a_command_that_fails_is_a_fixed_word_and_nothing_the_scheduler_or_ssh_said(failure, word, monkeypatch):
    with_ssh(monkeypatch, stdout="SECRET output", stderr="SECRET error text", **failure)
    for call in (requeue.SshCluster().node, requeue.SshCluster().requeue):
        with pytest.raises(requeue.ClusterError) as caught:
            call(JOB)
        assert str(caught.value) == word and caught.value.__cause__ is None and "SECRET" not in repr(caught.value)


@pytest.mark.parametrize("value, word", [("gx14", "gx14"), ("cn01", "cn01"), (" gx14\n", "gx14"), ("", None), ("  ", None), (None, None), (14, None),
                                         ("gx[14-15]", "other"), ("gx14,gx15", "other"), ("SECRET-node", "other"), ("GX14", "other"),
                                         ("gx14.cluster.example", "other"), ("a" * 70, "other"), ("gx14 SECRET", "other"), ("1gx", "other")])
def test_a_node_is_kept_only_as_a_plain_name_of_the_form_gx14_and_any_other_text_is_the_word_other(value, word):
    assert requeue.node_word(value) == word


# ---- /healthz: what counts as the server being back ---------------------------------------------------------------------------------

class Answer:
    """A response a fake client hands out: a status and a body (a dict, or an exception instance to raise from .json())."""

    def __init__(self, status=200, body=None):
        self.status_code, self.body = status, body

    def json(self):
        if isinstance(self.body, Exception):
            raise self.body
        return self.body


class Health:
    """In place of httpx.Client for /healthz alone: it writes down what was asked and answers what it was told to, or raises."""

    def __init__(self, *answers):
        self.answers, self.asked = list(answers), []

    def get(self, path, **kwargs):
        self.asked.append((path, kwargs))
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, Exception):
            raise answer
        return answer


OK_OLD = {"status": "ok", "mode": "private", "default_model": "SECRET-model", "started_at": "SECRET-start-1"}


def test_the_servers_start_time_is_read_from_a_healthz_that_says_ok_and_nothing_else_of_it_is_kept():
    client = Health(Answer(200, dict(OK_OLD, code_version="SECRET-sha")))
    assert requeue.read_health(client) == ("SECRET-start-1", None)
    assert client.asked == [("/healthz", {"timeout": requeue.HEALTHZ_TIMEOUT_S})], "a short timeout of its own: a dead tunnel must not hold a poll"


@pytest.mark.parametrize("answer, code", [
    (httpx.ConnectError("SECRET refused"), "healthz_ConnectError"), (httpx.ReadTimeout("SECRET"), "healthz_ReadTimeout"),
    (httpx.RemoteProtocolError("SECRET"), "healthz_RemoteProtocolError"), (RuntimeError("SECRET text"), "healthz_RuntimeError"),
    (Answer(503, {"error": {"message": "SECRET"}}), "healthz_http_503"), (Answer(502, None), "healthz_http_502"),
    (Answer(200, dict(OK_OLD, status="starting")), "healthz_bad_answer"), (Answer(200, {"status": "ok"}), "healthz_bad_answer"),
    (Answer(200, dict(OK_OLD, started_at=5)), "healthz_bad_answer"), (Answer(200, ["SECRET"]), "healthz_bad_answer"),
    (Answer(200, ValueError("not JSON")), "healthz_bad_answer"), (Answer(200, None), "healthz_bad_answer"),
])
def test_a_healthz_that_does_not_answer_ok_with_a_start_time_is_a_fixed_code_from_a_status_or_a_class_name(answer, code):
    started, got = requeue.read_health(Health(answer))
    assert started is None and got == code and "SECRET" not in got


def test_a_poll_is_back_only_for_an_ok_answer_with_another_start_time_than_the_one_before_the_requeue():
    before = "SECRET-start-1"
    new = Answer(200, dict(OK_OLD, started_at="SECRET-start-2"))
    assert requeue.ask_health(Health(new), before) == requeue.BACK
    assert requeue.ask_health(Health(Answer(200, OK_OLD)), before) == requeue.OLD, "the old server, still shutting down: neither refused nor back"
    for refused in (httpx.ConnectError("x"), Answer(503, {}), Answer(200, dict(OK_OLD, status="starting", started_at="SECRET-start-2")), Answer(200, {})):
        assert requeue.ask_health(Health(refused), before) == requeue.REFUSED


class Clock:
    """A fake clock whose sleep() moves it. `cost` is what one poll takes (a request's own time)."""

    def __init__(self, cost=0.0):
        self.now, self.cost, self.slept = 1000.0, cost, []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds

    def ask(self, *answers):
        """An ask() that answers from `answers` (the last one for ever), every call costing `cost`."""
        pending = list(answers)

        def ask():
            self.now += self.cost
            return pending.pop(0) if len(pending) > 1 else pending[0]
        return ask


def wait(clock, *answers, wait_s=1800.0, poll_s=5.0):
    return requeue.wait_for_return(clock.ask(*answers), clock(), wait_s, poll_s, clock, clock.sleep)


def test_the_wait_says_when_the_server_first_refused_and_when_it_was_back_both_from_the_requeue_to_a_tenth_of_a_second():
    clock = Clock()
    first_refused, back = wait(clock, requeue.OLD, requeue.REFUSED, requeue.REFUSED, requeue.REFUSED, requeue.BACK)
    assert (first_refused, back) == (5.0, 20.0) and clock.slept == [5.0] * 4, "a poll every 5 seconds, the first one at once"


def test_a_server_that_was_back_before_it_was_ever_seen_down_has_no_first_refusal():
    first_refused, back = wait(Clock(), requeue.OLD, requeue.BACK)
    assert (first_refused, back) == (None, 5.0)
    assert wait(Clock(), requeue.BACK) == (None, 0.0), "back at the very first poll: the time is 0, not missing"


def test_the_old_server_answering_while_it_stops_is_no_refusal_and_the_first_refusal_is_the_first_only():
    first_refused, back = wait(Clock(), requeue.OLD, requeue.OLD, requeue.REFUSED, requeue.OLD, requeue.REFUSED, requeue.BACK)
    assert (first_refused, back) == (10.0, 25.0)


def test_the_time_of_a_poll_is_when_its_answer_came_so_a_slow_request_is_counted():
    clock = Clock(cost=3.0)
    first_refused, back = wait(clock, requeue.REFUSED, requeue.BACK)
    assert (first_refused, back) == (3.0, 11.0), "3 s for the first request, 5 s of sleep, 3 s for the second"


def test_a_server_that_does_not_come_back_is_given_up_on_at_the_limit_after_one_last_poll_and_never_after():
    clock = Clock()
    asked = []

    def ask():
        asked.append(clock())
        return requeue.REFUSED
    first_refused, back = requeue.wait_for_return(ask, clock(), 12.0, 5.0, clock, clock.sleep)
    assert (first_refused, back) == (0.0, None)
    assert [round(t - 1000.0, 1) for t in asked] == [0.0, 5.0, 10.0, 12.0], "the pause before the last poll is cut to what is left of the wait"
    assert sum(clock.slept) == 12.0


def test_the_wait_is_over_at_once_when_the_limit_is_already_past_after_the_first_poll():
    clock = Clock(cost=9.0)
    assert wait(clock, requeue.OLD, wait_s=2.0) == (None, None) and clock.slept == [] and clock() == 1009.0


def test_back_is_a_back_even_when_the_slow_poll_that_saw_it_ended_after_the_limit():
    clock = Clock(cost=9.0)
    assert wait(clock, requeue.BACK, wait_s=2.0) == (None, 9.0)


# ---- the turn's stream: event names, and the kind of an error event ------------------------------------------------------------------

def sse(frames):
    """(event, data) pairs as the server writes them: `event: x`, then `data:` and the JSON on one line, then a blank line."""
    return "".join("event: {}\ndata: {}\n\n".format(event, json.dumps(data, separators=(",", ":"))) for event, data in frames)


def hostile_turn(deltas=5, error=None, stop=True):
    """The frames of a turn as a private server sends them, a SECRET in every text: its report, a snapshot of it at each step, the stages' details,
    an error's message. `error` is the kind of an error event to put before the stop."""
    frames = [("message_start", {"seq": 1, "message_id": "m_SECRET-id", "options": {"reference": "SECRET-reference"}}),
              ("stage_end", {"seq": 2, "stage": "retrieve", "detail": {"report_matches": [{"report": "SECRET-matched report"}]}}),
              ("content_block_start", {"seq": 3, "index": 0, "content_block": {"type": "report", "text": ""}})]
    frames += [("content_block_delta", {"seq": 4 + i, "index": 0, "delta": {"type": "beam_snapshot", "step": i + 1, "text": "SECRET-snapshot " * (i + 1)}})
               for i in range(deltas)]
    if error is not None:
        frames.append(("error", {"seq": 100, "type": "error", "error": {"type": error, "message": "SECRET error message"}}))
    if stop:
        frames.append(("message_stop", {"seq": 101, "status": "error" if error else "done", "report": "SECRET-final report"}))
    return frames


def counted(text, size=None):
    """count_frames over `text` in chunks of `size` characters. -> (the Stream, how many times on_enough was called, and after which delta)."""
    chunks = [text] if size is None else [text[i:i + size] for i in range(0, len(text), size)]
    stream, calls = requeue.Stream(), []
    requeue.count_frames(stream, iter(chunks), lambda: calls.append(stream.deltas))
    return stream, calls


def test_a_stream_is_counted_by_the_names_of_its_frames_and_the_third_delta_says_enough_once():
    stream, calls = counted(sse(hostile_turn(deltas=5)))
    assert (stream.deltas, stream.error_event, stream.error_kind) == (5, False, None) and calls == [3], "on the third, not before and not again"
    stream, calls = counted(sse(hostile_turn(deltas=2)))
    assert (stream.deltas, calls) == (2, [])


@pytest.mark.parametrize("crlf", [False, True])
@pytest.mark.parametrize("size", [1, 7, 64, 100000])
def test_the_count_does_not_depend_on_where_the_chunks_are_cut_or_on_the_line_ends(size, crlf):
    text = sse(hostile_turn(deltas=4, error="server_restart"))
    stream, calls = counted(text.replace("\n", "\r\n") if crlf else text, size)
    assert (stream.deltas, stream.error_event, stream.error_kind, calls) == (4, True, "server_restart", [3])


def test_comments_and_frames_without_data_are_no_events_and_a_frame_that_is_cut_off_is_not_counted():
    text = ": ping\n\n" + sse(hostile_turn(deltas=3)) + ": ping\n\nevent: content_block_delta\n\nevent: content_block_delta\ndata: {\"seq\":"
    stream, calls = counted(text)
    assert (stream.deltas, calls) == (3, [3]), "the last frame never ended, and a frame with no data is no frame"


def test_a_delta_is_counted_by_its_name_alone_its_data_is_never_parsed_and_only_an_error_events_is(monkeypatch):
    """R7: the text of a snapshot is a report. The script does not parse a delta's data, however it is shaped; of all the frames of a turn it parses
    the data of the `error` events, for the kind, and drops the rest of that frame."""
    parsed, real = [], json.loads

    def spy(text, *args, **kwargs):
        parsed.append(text)
        return real(text, *args, **kwargs)
    monkeypatch.setattr(requeue, "json", SimpleNamespace(loads=spy, dumps=json.dumps))
    text = sse(hostile_turn(deltas=4, error="model_error")) + "event: content_block_delta\ndata: {not json, SECRET}\n\n"
    stream, calls = counted(text)
    assert (stream.deltas, calls) == (5, [3]) and (stream.error_event, stream.error_kind) == (True, "model_error")
    assert len(parsed) == 1 and "SECRET error message" in parsed[0], "the one frame it parsed is the error's: {} parsed".format(len(parsed))


@pytest.mark.parametrize("kind", ["validation_error", "model_error", "overloaded_error", "server_restart"])
def test_an_error_event_keeps_its_kind_when_it_is_one_the_server_has_and_never_its_message(kind):
    stream, _ = counted(sse(hostile_turn(deltas=1, error=kind)))
    assert (stream.error_event, stream.error_kind) == (True, kind)
    assert set(probe.ERROR_KINDS) == {"validation_error", "model_error", "overloaded_error", "server_restart"}


@pytest.mark.parametrize("data", [{"error": {"type": "SECRET-kind", "message": "SECRET"}}, {"error": {"type": 7}}, {"error": "SECRET"}, {"type": "error"},
                                  ["SECRET"], "SECRET", 5, None, {"error": {"type": ["server_restart"]}}, {"error": {"type": "Server_Restart"}}])
def test_an_error_event_of_a_kind_the_app_does_not_have_or_of_a_shape_it_does_not_have_is_unknown(data):
    stream, _ = counted(sse([("error", data)]))
    assert (stream.error_event, stream.error_kind) == (True, "unknown")


def test_an_error_event_whose_data_is_not_json_is_still_an_error_event_of_kind_unknown():
    stream, _ = counted("event: error\ndata: {not json SECRET\n\nevent: error\ndata: " + "[" * 100000 + "\n\n")
    assert (stream.error_event, stream.error_kind) == (True, "unknown")


# ---- the reader: the POST, the message id, how the stream ended ----------------------------------------------------------------------

class Streaming:
    """A response a fake client hands out as a context manager: a status, headers, and a body that is read in chunks. A chunk that is an Event holds
    the stream open until it is set. `raises` is raised when the body is read to its end, as a connection that breaks is."""

    def __init__(self, chunks=(), status=200, headers=None, raises=None):
        self.status_code, self.chunks, self.raises = status, list(chunks), raises
        given = {"X-Message-Id": "m_fake0001"} if headers is None else headers
        self.headers = httpx.Headers({k: v for k, v in given.items() if v is not None})   # as httpx's own: case-insensitive

    def iter_text(self):
        for chunk in self.chunks:
            if isinstance(chunk, threading.Event):
                assert chunk.wait(20), "the test never let the stream go"
                continue
            yield chunk
        if self.raises is not None:
            raise self.raises

    def read(self):
        return b""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class Streams:
    """In place of httpx.Client for the turn's POST alone: it writes down each request and hands out its stream, or raises it."""

    def __init__(self, answer):
        self.answer, self.sent = answer, []

    def stream(self, method, path, **kwargs):
        self.sent.append((method, path, kwargs))
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def read_turn(answer, deadline_s=600.0, clock=time.perf_counter):
    """A TurnReader over the fake, run to its end. -> (the reader, its client)."""
    client = Streams(answer)
    reader = requeue.TurnReader(client, "s_fake", deadline_s, clock)
    reader.start()
    assert reader.thread.join(20) is None and not reader.thread.is_alive()
    return reader, client


def test_the_turn_is_sent_as_the_form_the_page_sends_with_the_published_options_and_a_picture_the_script_draws():
    reader, client = read_turn(Streaming([sse(hostile_turn(deltas=3))]))
    (method, path, kwargs), = client.sent
    assert (method, path) == ("POST", "/v1/sessions/s_fake/messages")
    assert kwargs["files"] == {"image": ("drill.png", drill.synthetic_png(), "image/png")}, "no MIMIC image: the one the drill draws"
    assert kwargs["data"]["text"] == "" and json.loads(kwargs["data"]["options"]) == requeue.turn_options() == probe.turn_options(
        model="hybrid_150m_m3_rrg", max_new_tokens=200)


def test_a_stream_that_ends_with_no_error_event_is_closed_and_one_with_an_error_event_is_error_event():
    closed, _ = read_turn(Streaming([sse(hostile_turn(deltas=4))]))
    assert (closed.stream.status, closed.stream.end, closed.stream.deltas, closed.stream.message_id) == (200, "closed", 4, "m_fake0001")
    errored, _ = read_turn(Streaming([sse(hostile_turn(deltas=4, error="server_restart"))]))
    assert (errored.stream.end, errored.stream.error_event, errored.stream.error_kind) == ("error_event", True, "server_restart")
    assert closed.enough.is_set() and errored.enough.is_set()


@pytest.mark.parametrize("failure, end", [
    (httpx.RemoteProtocolError("SECRET peer closed connection without sending complete message body"), "reset"),
    (httpx.ReadError("SECRET"), "reset"), (httpx.ConnectError("SECRET"), "reset"), (httpx.WriteError("SECRET"), "reset"),
    (httpx.ReadTimeout("SECRET"), "timeout"), (httpx.ConnectTimeout("SECRET"), "timeout"), (httpx.PoolTimeout("SECRET"), "timeout"),
    (probe.Timeout(), "timeout"), (RuntimeError("SECRET text"), "reset"), (UnicodeDecodeError("utf-8", b"\xff", 0, 1, "SECRET"), "reset"),
])
def test_a_stream_that_breaks_is_reset_and_one_that_runs_out_of_time_is_timeout_whatever_the_failures_text(failure, end):
    reader, _ = read_turn(Streaming([sse(hostile_turn(deltas=4))], raises=failure))
    assert reader.stream.end == end and reader.stream.deltas == 4 and reader.enough.is_set()
    assert "SECRET" not in repr(reader.stream)


def test_a_connection_that_is_never_made_is_a_reset_with_no_status_and_the_wait_for_deltas_is_released():
    reader, _ = read_turn(httpx.ConnectError("SECRET refused"))
    assert (reader.stream.status, reader.stream.end, reader.stream.message_id) == (None, "reset", None) and reader.enough.is_set()


@pytest.mark.parametrize("status", [401, 404, 422, 429, 500, 503])
def test_a_turn_the_server_refuses_has_its_status_and_no_end_and_nothing_of_its_answer(status):
    reader, _ = read_turn(Streaming([sse(hostile_turn(error="overloaded_error"))], status=status))
    assert (reader.stream.status, reader.stream.end, reader.stream.deltas, reader.stream.error_event) == (status, None, 0, False)
    assert reader.enough.is_set()


@pytest.mark.parametrize("header", [{}, {"X-Message-Id": ""}, {"X-Message-Id": "../x"}, {"X-Message-Id": "m a"}, {"X-Message-Id": "m?x=1"},
                                    {"X-Message-Id": "x" * 65}, {"X-Message-Id": None}])
def test_the_message_id_is_taken_from_the_header_only_when_a_path_can_hold_it(header):
    reader, _ = read_turn(Streaming([sse(hostile_turn(deltas=1))], headers=header))
    assert reader.stream.message_id is None and reader.stream.end == "closed"


def test_the_message_id_may_hold_a_word_a_sentinel_looks_for_because_it_is_never_printed():
    reader, _ = read_turn(Streaming([sse(hostile_turn(deltas=1))], headers={"X-Message-Id": "m_SECRET-message-id"}))
    assert reader.stream.message_id == "m_SECRET-message-id"


def test_the_third_delta_releases_the_wait_while_the_stream_is_still_open_and_the_end_comes_later():
    hold = threading.Event()
    client = Streams(Streaming([sse(hostile_turn(deltas=3, stop=False)), hold, sse([("error", {"error": {"type": "server_restart"}})])]))
    reader = requeue.TurnReader(client, "s_fake")
    reader.start()
    try:
        assert reader.enough.wait(10) and reader.stream.deltas == 3
        assert reader.stream.end is None and reader.thread.is_alive(), "still reading: the requeue is sent now"
    finally:
        hold.set()
    reader.finish(10)
    assert (reader.stream.end, reader.stream.error_kind) == ("error_event", "server_restart")


def test_a_stream_still_open_when_the_reader_is_given_up_on_is_a_timeout():
    hold = threading.Event()
    reader = requeue.TurnReader(Streams(Streaming([sse(hostile_turn(deltas=3, stop=False)), hold])), "s_fake")
    reader.start()
    try:
        assert reader.enough.wait(10)
        reader.finish(0.05)
        assert reader.stream.end == "timeout" and reader.thread.is_alive()
    finally:
        hold.set()
        reader.thread.join(10)


def test_the_stream_is_let_go_at_its_deadline_by_the_probes_own_watch():
    ticks = iter([0.0, 5.0, 10.0, 20.0, 30.0])
    reader, _ = read_turn(Streaming([sse(hostile_turn(deltas=1)), sse(hostile_turn(deltas=1)), sse(hostile_turn(deltas=1)), sse(hostile_turn(deltas=1))]),
                          deadline_s=15.0, clock=lambda: next(ticks))
    assert reader.stream.end == "timeout" and reader.stream.deltas == 3, "the chunk in hand when the deadline passed is read, then the stream is let go"


def test_a_failure_inside_the_reader_never_leaves_its_thread_where_a_traceback_would_be_printed_and_the_wait_is_released(monkeypatch):
    seen = []
    monkeypatch.setattr(threading, "excepthook", lambda args: seen.append(args))
    monkeypatch.setattr(requeue, "count_frames", lambda *args, **kwargs: (_ for _ in ()).throw(KeyError("SECRET")))
    reader, _ = read_turn(Streaming([sse(hostile_turn(deltas=1))]))
    assert seen == [] and reader.stream.end == "reset" and reader.enough.is_set()


# ---- the stored turn: its status, its last event, its error's kind, how many events --------------------------------------------------

def stored_message(status="error", events=None, **more):
    if events is None:
        events = [{"seq": 1, "event": "message_start", "data": {"model": {"name": "SECRET"}}},
                  {"seq": 2, "event": "content_block_delta", "data": {"delta": {"text": "SECRET snapshot"}}},
                  {"seq": 3, "event": "error", "data": {"type": "error", "error": {"type": "server_restart", "message": "SECRET restart text"}}},
                  {"seq": 4, "event": "message_stop", "data": {"status": "error", "report": None}}]
    return dict({"id": "m_fake", "role": "assistant", "status": status, "report": "SECRET report", "display_report": "SECRET", "events": events}, **more)


def read_stored_from(answer, message_id="m_fake0001"):
    client = Health(answer)
    return requeue.read_stored(client, message_id), client


def test_the_stored_turn_is_read_down_to_its_status_its_error_kind_its_last_event_and_its_event_count():
    (stored, code), client = read_stored_from(Answer(200, stored_message()))
    assert client.asked == [("/v1/messages/m_fake0001", {})]
    assert code is None and stored == requeue.Stored(http=200, status="error", error="server_restart", last_event="message_stop", events=4)
    assert "SECRET" not in repr(stored)


def test_a_stored_turn_that_did_not_end_in_an_error_has_no_error_kind():
    done = stored_message("done", events=[{"seq": 1, "event": "message_start", "data": {}}, {"seq": 2, "event": "message_stop", "data": {"status": "done"}}])
    (stored, code), _ = read_stored_from(Answer(200, done))
    assert code is None and stored == requeue.Stored(http=200, status="done", error=None, last_event="message_stop", events=2)


@pytest.mark.parametrize("status, word", [("running", "running"), ("done", "done"), ("error", "error"), ("aborted", "aborted"),
                                          ("SECRET-status", "unknown"), (None, "unknown"), (5, "unknown"), (["error"], "unknown")])
def test_the_stored_status_is_a_word_of_the_apps_four_or_unknown(status, word):
    (stored, _), _ = read_stored_from(Answer(200, stored_message(status)))
    assert stored.status == word


@pytest.mark.parametrize("kind, word", [("validation_error", "validation_error"), ("model_error", "model_error"), ("overloaded_error", "overloaded_error"),
                                        ("server_restart", "server_restart"), ("SECRET-kind", "unknown"), (None, "unknown"), (3, "unknown")])
def test_the_stored_errors_kind_is_a_word_of_the_apps_kinds_or_unknown(kind, word):
    events = [{"seq": 1, "event": "error", "data": {"error": {"type": kind, "message": "SECRET"}}}]
    (stored, _), _ = read_stored_from(Answer(200, stored_message(events=events)))
    assert stored.error == word and stored.last_event == "error"


def test_an_error_event_with_no_error_in_its_data_is_of_kind_unknown_and_the_last_of_two_error_events_counts():
    events = [{"seq": 1, "event": "error", "data": {"type": "error"}}, {"seq": 2, "event": "error", "data": "SECRET"}]
    assert read_stored_from(Answer(200, stored_message(events=events)))[0][0].error == "unknown"
    events = [{"seq": 1, "event": "error", "data": {"error": {"type": "model_error"}}}, {"seq": 2, "event": "error", "data": {"error": {"type": "server_restart"}}}]
    assert read_stored_from(Answer(200, stored_message(events=events)))[0][0].error == "server_restart"


@pytest.mark.parametrize("name, word", [(n, n) for n in requeue.EVENT_NAMES] + [("SECRET-event", "other"), (None, "other"), (7, "other")])
def test_the_last_events_name_is_from_the_apps_nine_or_other(name, word):
    (stored, _), _ = read_stored_from(Answer(200, stored_message(events=[{"seq": 1, "event": name, "data": {}}])))
    assert stored.last_event == word and stored.events == 1


@pytest.mark.parametrize("events", [None, "SECRET", {"seq": 1}, 5])
def test_a_stored_turn_with_no_list_of_events_has_no_count_and_no_last_event(events):
    (stored, code), _ = read_stored_from(Answer(200, dict(stored_message(), events=events)))
    assert code is None and (stored.status, stored.events, stored.last_event, stored.error) == ("error", None, None, None)


def test_a_stored_turn_with_no_events_at_all_has_a_count_of_zero_and_no_last_event():
    (stored, _), _ = read_stored_from(Answer(200, stored_message(events=[])))
    assert (stored.events, stored.last_event, stored.error) == (0, None, None)


@pytest.mark.parametrize("answer, code", [
    (Answer(404, {"error": {"message": "SECRET Message not found."}}), "stored_turn_http_404"), (Answer(500, None), "stored_turn_http_500"),
    (httpx.ConnectError("SECRET"), "stored_turn_ConnectError"), (httpx.ReadTimeout("SECRET"), "stored_turn_ReadTimeout"),
    (Answer(200, ValueError("SECRET not JSON")), "stored_turn_bad_answer"), (Answer(200, ["SECRET"]), "stored_turn_bad_answer"),
    (Answer(200, None), "stored_turn_bad_answer"),
])
def test_a_stored_turn_that_cannot_be_read_is_a_fixed_code_from_a_status_or_a_class_name(answer, code):
    (stored, got), _ = read_stored_from(answer)
    assert got == code and "SECRET" not in got and stored.status is None and stored.events is None
    assert stored.http == (answer.status_code if isinstance(answer, Answer) else None)


def test_with_no_message_id_there_is_nothing_to_read_and_nothing_is_asked():
    (stored, code), client = read_stored_from(Answer(200, stored_message()), message_id=None)
    assert code == "stored_turn_no_id" and client.asked == [] and stored == requeue.Stored()


# ---- a scripted server and a scripted cluster: the whole drill, in this process ------------------------------------------------------

class FakeChat:
    """The chat server as far as the drill can see it, on a free loopback port of this process (http.server): the routes it uses, the turn's frames as
    the app writes them (chunked, one event at a time, stored before they are sent), and a store that outlives a restart. A SECRET is in every text it
    sends: the session's title, the report, each snapshot of it, the matched reports, the error's message, the stored events, a header on every answer
    and the start times themselves. requeued() is what a requeue does to it: `graceful` is SIGTERM (the open turn ends as an error with server_restart,
    stored and sent, then the server goes), `hard` is SIGKILL (it goes at once, every connection cut, the turn left `running` for the next start's
    recovery). After `down_s` seconds it starts again, with another start time, or never."""

    SESSION, MESSAGE = "s_SECRET-session-id", "m_SECRET-message-id"

    def __init__(self, deltas=None, step_s=0.02, token=None):
        self.deltas, self.step_s, self.token = deltas, step_s, token
        self.generation, self.started_at, self.port, self.httpd = 0, "", None, None
        self.store, self.deleted, self.requests, self.requeues = {}, False, [], []
        self.deltas_sent, self.streams = 0, 0
        self.ending = None            # "graceful": the open turn ends as a server restart
        self.silenced = 0             # the streams of this generation and the ones before are left open and silent: a tunnel whose far end is gone
        self.closed = False
        self.refuse_turn = None       # (status, error kind): the turn's POST is answered with that
        self.delete_status = 204
        self.message_status = None    # the status GET /v1/messages/<id> answers with instead of the message
        self.odd_words = False        # the stored turn and its error carry words the app does not have
        self.open = set()
        self.start()

    @property
    def base(self):
        return "http://127.0.0.1:{}".format(self.port)

    # -- life
    def start(self):
        self.generation += 1
        self.started_at = "SECRET-start-{}".format(self.generation)
        for record in self.store.values():       # Store.recover_after_restart: a turn left running ends as an error, with server_restart
            if record["status"] == "running":
                events = record["events"]
                for event, data in (("error", {"type": "error", "error": {"type": "server_restart", "message": "SECRET restarted while running"}}),
                                    ("message_stop", {"status": "error", "report": None})):
                    events.append({"seq": len(events) + 1, "event": event, "data": dict(data, seq=len(events) + 1)})
                record["status"] = "error"
        self.httpd = ThreadingHTTPServer(("127.0.0.1", self.port or 0), self._handler())
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        assert self.port not in (8000, 8010), "a test server never takes the ports of the user's own servers"
        threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()

    def stop(self, cut=True):
        httpd, self.httpd = self.httpd, None
        if httpd is None:
            return
        httpd.shutdown()
        httpd.server_close()
        if cut:
            for sock in list(self.open):
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        else:
            self.silenced = self.generation

    def close(self):
        self.closed = True
        self.stop(cut=True)
        for sock in list(self.open):
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def requeued(self, how, down_s):
        """What the cluster's requeue does to the server, in the background: scontrol returns at once."""
        self.requeues.append((how, self.deltas_sent))

        def go():
            if how == "graceful":
                self.ending = "graceful"
                deadline = time.monotonic() + 5
                while self.streams and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.ending = None
            self.stop(cut=(how != "silent"))
            if down_s is not None and not self.closed:
                time.sleep(down_s)
                self.start()
        threading.Thread(target=go, daemon=True).start()

    # -- the routes
    def _handler(self):
        chat = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):   # the default writes every request to stderr
                pass

            def setup(self):
                super().setup()
                chat.open.add(self.connection)

            def finish(self):
                chat.open.discard(self.connection)
                try:
                    super().finish()
                except OSError:
                    pass

            def do_GET(self):
                chat.handle(self, "GET")

            def do_POST(self):
                chat.handle(self, "POST")

            def do_DELETE(self):
                chat.handle(self, "DELETE")
        return Handler

    def _json(self, h, status, body=None):
        data = b"" if body is None else json.dumps(body).encode()
        h.send_response(status)
        h.send_header("Content-Type", "application/json")
        h.send_header("Content-Length", str(len(data)))
        h.send_header("X-Secret", "SECRET-header")
        h.send_header("Connection", "close")
        h.end_headers()
        h.wfile.write(data)
        h.close_connection = True

    def _error(self, h, status, kind):
        self._json(h, status, {"type": "error", "error": {"type": kind, "message": "SECRET error message"}})

    def handle(self, h, method):
        path = h.path.split("?")[0]
        self.requests.append((method, path, {name.lower(): value for name, value in h.headers.items()}))
        length = int(h.headers.get("Content-Length") or 0)
        if length:
            h.rfile.read(length)
        if path != "/healthz" and self.token and h.headers.get("Authorization") != "Bearer " + self.token:
            return self._error(h, 401, "authentication_error")
        if method == "GET" and path == "/healthz":
            return self._json(h, 200, {"status": "ok", "mode": "private", "default_model": "SECRET-model", "turns_in_flight": 0, "queue_cap": 4,
                                       "retention_days": 45, "code_version": "SECRET-sha", "started_at": self.started_at})
        if method == "POST" and path == "/v1/sessions":
            return self._json(h, 200, {"id": self.SESSION, "title": "SECRET-title", "mode": "private", "created_at": "SECRET-time"})
        if method == "POST" and path == "/v1/sessions/{}/messages".format(self.SESSION):
            if self.refuse_turn is not None:
                return self._error(h, *self.refuse_turn)
            return self._stream(h)
        if method == "GET" and path == "/v1/messages/{}".format(self.MESSAGE):
            record = self.store.get(self.MESSAGE)
            if self.message_status is not None or record is None:
                return self._error(h, self.message_status or 404, "not_found_error")
            return self._json(h, 200, {"id": self.MESSAGE, "role": "assistant", "status": "SECRET-status" if self.odd_words else record["status"],
                                       "report": "SECRET-report", "display_report": "SECRET-report", "events": self._stored_events(record)})
        if method == "DELETE" and path == "/v1/sessions/{}".format(self.SESSION):
            self.deleted = self.delete_status == 204
            return self._json(h, self.delete_status) if self.delete_status == 204 else self._error(h, self.delete_status, "not_found_error")
        return self._error(h, 404, "not_found_error")

    def _stored_events(self, record):
        if not self.odd_words:
            return list(record["events"])
        return [dict(e, event="SECRET-event" if e["event"] == "message_stop" else e["event"],
                     data=dict(e["data"], error={"type": "SECRET-kind", "message": "SECRET"}) if e["event"] == "error" else e["data"])
                for e in record["events"]]

    def _stream(self, h):
        events = []
        record = {"status": "running", "events": events}
        self.store[self.MESSAGE] = record
        h.send_response(200)
        for name, value in (("Content-Type", "text/event-stream"), ("Cache-Control", "no-cache"), ("Transfer-Encoding", "chunked"),
                            ("X-Message-Id", self.MESSAGE), ("X-Secret", "SECRET-header"), ("Connection", "close")):
            h.send_header(name, value)
        h.end_headers()
        h.close_connection = True

        def emit(event, data):
            data = dict(data, seq=len(events) + 1)
            events.append({"seq": data["seq"], "event": event, "data": data})   # stored first, then sent: as the app does
            frame = "event: {}\ndata: {}\n\n".format(event, json.dumps(data, separators=(",", ":"))).encode()
            h.wfile.write("{:x}\r\n".format(len(frame)).encode() + frame + b"\r\n")
            h.wfile.flush()

        def finish(status, error=None, report=None):
            if error is not None:
                emit("error", {"type": "error", "error": {"type": error, "message": "SECRET error message"}})
            emit("message_stop", {"status": status, "report": report, "display_report": report})
            record["status"] = status
            h.wfile.write(b"0\r\n\r\n")
            h.wfile.flush()

        generation = self.generation
        self.streams += 1
        try:
            emit("message_start", {"message_id": self.MESSAGE, "model": {"name": "SECRET-model"}, "options": {"reference": "SECRET-reference"}})
            emit("stage_end", {"stage": "retrieve", "ms": 1.0, "detail": {"report_matches": [{"report": "SECRET-matched report"}]}})
            emit("content_block_start", {"index": 0, "content_block": {"type": "report", "text": ""}})
            sent = 0
            while not self.closed:
                if generation <= self.silenced:   # its server is gone and its connection is not: nothing more is said on it, and it is not closed
                    time.sleep(0.02)
                    continue
                if self.ending == "graceful":
                    return finish("error", "server_restart")
                if self.deleted:
                    return finish("aborted")
                if self.deltas is not None and sent >= self.deltas:
                    return finish("done", report="SECRET-final report")
                sent += 1
                self.deltas_sent += 1
                emit("content_block_delta", {"index": 0, "delta": {"type": "beam_snapshot", "step": sent, "text": "SECRET-snapshot " * sent}})
                time.sleep(self.step_s)
        except OSError:   # the client went, or the server was stopped under the stream
            pass
        finally:
            self.streams -= 1


class FakeCluster:
    """The cluster as the drill sees it: node() names the node the job runs on, which is another one once the job has come back (and none while it
    waits in the queue), and requeue() is what scontrol does: it returns at once and the server goes down and comes back by itself (FakeChat.requeued).
    `how` is graceful or hard (None: the requeue changes nothing), `delay_s` how long the call takes, `fail` an exception it raises."""

    def __init__(self, chat, how="graceful", down_s=0.3, delay_s=0.0, fail=None, node_fail=None, no_node=False):
        self.chat, self.how, self.down_s, self.delay_s, self.fail, self.node_fail, self.no_node = chat, how, down_s, delay_s, fail, node_fail, no_node
        self.calls, self.requeued, self.deltas_at_requeue = [], False, None

    def node(self, job):
        self.calls.append(("node", job))
        if self.node_fail is not None:
            raise self.node_fail
        if self.no_node:
            return ""
        if not self.requeued or self.how is None:
            return "gx14"
        return "gx15" if self.chat.generation > 1 else None

    def requeue(self, job):
        self.calls.append(("requeue", job))
        self.deltas_at_requeue = self.chat.deltas_sent
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.fail is not None:
            raise self.fail
        self.requeued = True
        if self.how is not None:
            self.chat.requeued(self.how, self.down_s)


@pytest.fixture
def chat():
    server = FakeChat()
    yield server
    server.close()


def flow(chat, cluster, *extra, wait="30", poll_s=0.05):
    """main() against the fake, with its own two streams. -> (code, stdout, stderr)."""
    return run_main([chat.base, "--job", JOB, "--wait", wait] + list(extra), cluster=cluster, poll_s=poll_s)


def requests_of(chat):
    return [(method, path) for method, path, _ in chat.requests]


def test_with_no_cluster_given_the_real_one_is_used_and_it_asks_for_the_node_with_its_one_ssh_command_before_anything_else(chat, monkeypatch):
    """main() is handed no cluster, as the controller runs it: SshCluster. Its ssh is the recorder of this module here; the job has no node, so the drill
    ends before it makes a session, and the one command that was run is the squeue."""
    ssh = with_ssh(monkeypatch, stdout="\n")
    out, err = io.StringIO(), io.StringIO()
    code = requeue.main([chat.base, "--job", JOB], out, err, poll_s=0.05)
    result = the_result(out.getvalue())
    assert code == 1 and err.getvalue() == "" and result["error"] == "job_not_running"
    assert [argv for argv, _ in ssh.asked] == [["ssh", "-o", "BatchMode=yes", "hpi-hpc", "squeue", "-j", JOB, "-h", "-o", "%N"]]
    assert requests_of(chat) == [("GET", "/healthz")]


def test_a_graceful_requeue_is_run_to_its_end_and_the_line_says_what_happened_in_numbers_and_fixed_words(chat):
    """SIGTERM: the open turn ends as an error with server_restart, which the stream carries, then the server is gone for a moment and comes back
    on a node of its own. The line has the node before and after, the deltas counted when the requeue was sent, the times, how the stream ended, and
    the stored turn: its status, its error's kind, its last event and how many events it has."""
    cluster = FakeCluster(chat, "graceful")
    code, out, err = flow(chat, cluster)
    result = the_result(out)
    assert code == 0 and err == "", (code, out, err)
    assert list(result) == ["drill", "node_before", "node_after", "deltas_at_requeue", "requeue_s", "first_refused_s", "back_s", "stream_end",
                            "error_event", "stream_error", "turn_status", "turn_error", "turn_last_event", "events", "session_deleted", "ok"]
    assert (result["drill"], result["node_before"], result["node_after"]) == ("requeue", "gx14", "gx15")
    assert result["deltas_at_requeue"] >= 3 and cluster.deltas_at_requeue >= 3, "the requeue went out once three deltas had arrived"
    assert 0 <= result["requeue_s"] <= result["back_s"] and result["first_refused_s"] is not None and 0 < result["first_refused_s"] <= result["back_s"]
    assert (result["stream_end"], result["error_event"], result["stream_error"]) == ("error_event", True, "server_restart")
    assert (result["turn_status"], result["turn_error"], result["turn_last_event"]) == ("error", "server_restart", "message_stop")
    assert result["events"] == len(chat.store[chat.MESSAGE]["events"]) >= 7
    assert (result["session_deleted"], result["ok"]) == (True, True) and "error" not in result


def test_a_hard_requeue_is_a_reset_stream_and_the_next_starts_recovery_stores_the_turn_as_the_restart_it_was(chat):
    """SIGKILL: the connection is cut with no frame after the last delta, so the stream has no error event; the turn is left running in the store, and
    the start of the next server closes it as an error with server_restart. The stored turn is the same as after a graceful requeue."""
    code, out, err = flow(chat, FakeCluster(chat, "hard"))
    result = the_result(out)
    assert code == 0 and err == "" and result["ok"] is True
    assert (result["stream_end"], result["error_event"], result["stream_error"]) == ("reset", False, None)
    assert (result["turn_status"], result["turn_error"], result["turn_last_event"]) == ("error", "server_restart", "message_stop")
    assert chat.generation == 2


def test_the_cluster_is_asked_for_the_node_then_the_job_is_requeued_once_then_the_node_is_asked_again_and_the_server_is_asked_in_order(chat):
    cluster = FakeCluster(chat)
    flow(chat, cluster, wait="30")
    assert cluster.calls == [("node", JOB), ("requeue", JOB), ("node", JOB)], "one requeue, however long the wait"
    paths = requests_of(chat)
    assert paths[:3] == [("GET", "/healthz"), ("POST", "/v1/sessions"), ("POST", "/v1/sessions/{}/messages".format(chat.SESSION))]
    assert paths[-2:] == [("GET", "/v1/messages/" + chat.MESSAGE), ("DELETE", "/v1/sessions/" + chat.SESSION)], "the stored turn is read, then the session is deleted"
    assert paths.count(("POST", "/v1/sessions/{}/messages".format(chat.SESSION))) == 1, "one turn"
    assert {method for method, _, _ in chat.requests} == {"GET", "POST", "DELETE"}


def test_every_request_carries_the_drills_client_id_and_with_a_token_a_bearer_and_the_token_never_reaches_the_line(chat, monkeypatch):
    chat.token = TOKEN
    monkeypatch.setenv("CHAT_TOKEN", TOKEN)
    code, out, err = flow(chat, FakeCluster(chat))
    assert code == 0 and err == "" and TOKEN not in out
    assert len(chat.requests) >= 6 and all(headers.get("x-client-id") == "chat-requeue-drill" for _, _, headers in chat.requests)
    assert all(headers.get("authorization") == "Bearer " + TOKEN for _, _, headers in chat.requests), "to that URL and nowhere else, on every request"


def test_a_token_file_wins_over_chat_token_and_a_server_that_wants_a_token_the_drill_was_not_given_ends_it_before_any_requeue(chat, monkeypatch, tmp_path):
    chat.token = TOKEN
    monkeypatch.setenv("CHAT_TOKEN", "the-wrong-token")
    path = write_token_file(tmp_path, TOKEN + "\n")
    assert flow(chat, FakeCluster(chat), "--token-file", str(path))[0] == 0
    chat2 = FakeChat(token=TOKEN)
    try:
        for token in (None, "the-wrong-token"):
            if token:
                monkeypatch.setenv("CHAT_TOKEN", token)
            else:
                monkeypatch.delenv("CHAT_TOKEN")
            cluster = FakeCluster(chat2)
            code, out, err = flow(chat2, cluster)
            result = the_result(out)
            assert code == 1 and err == "" and result["error"] == "session_create_http_401" and result["ok"] is False
            assert result["session_deleted"] is None and ("requeue", JOB) not in cluster.calls and "the-wrong-token" not in out
    finally:
        chat2.close()


def test_the_drill_connects_straight_to_the_url_whatever_proxy_the_environment_names(chat, monkeypatch):
    """httpx trusts HTTP_PROXY and ALL_PROXY by default, loopback included, and the token and the stream would pass through that proxy."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        dead = "http://127.0.0.1:{}".format(s.getsockname()[1])   # a free port, and nothing listens on it once the socket is closed
    for name in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(name, dead)
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    code, out, err = flow(chat, FakeCluster(chat))
    assert code == 0 and err == "" and the_result(out)["ok"] is True


def test_no_line_the_drill_prints_holds_a_word_of_what_a_hostile_server_says_in_either_arm(chat, tmp_path, monkeypatch):
    """The sentinel. Every text the fake sends carries a SECRET (the title, the snapshots, the report, the error's message, a header on every answer,
    the start times, the ids); stdout is exactly one RESULT line, stderr is empty, nothing is written to the directory it ran in, and every string in
    the line is a word of a closed list (the node names the cluster gave are the one open word, and they pass the pattern)."""
    workdir = tmp_path / "cwd"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    words = {"requeue", "gx14", "gx15", "error_event", "reset", "closed", "timeout", "error", "done", "running", "aborted", "unknown", "server_restart",
             "model_error", "validation_error", "overloaded_error", "other", *requeue.EVENT_NAMES}
    for how in ("graceful", "hard"):
        chat2 = FakeChat()
        try:
            code, out, err = flow(chat2, FakeCluster(chat2, how))
        finally:
            chat2.close()
        assert code == 0 and err == "" and out.count("\n") == 1 and out.endswith("\n") and out.startswith("RESULT {")
        assert "SECRET" not in out and "secret" not in out.lower() and "s3cr3t" not in out
        assert {s for s in leaves(the_result(out))} <= words | set(requeue.EVENT_NAMES), (how, sorted(set(leaves(the_result(out))) - words))
    assert list(workdir.iterdir()) == [], "the drill wrote no file"


def test_words_the_app_does_not_have_become_unknown_and_the_hostile_text_stays_out_of_the_line(chat):
    """The stored turn's status, its last event and its error's kind are all SECRET strings: unknown, other, unknown. That turn is not the restart the
    drill looks for, so it does not hold; and nothing of it is printed."""
    chat.odd_words = True
    code, out, err = flow(chat, FakeCluster(chat))
    result = the_result(out)
    assert code == 1 and err == "" and "SECRET" not in out
    assert (result["turn_status"], result["turn_error"], result["turn_last_event"], result["ok"]) == ("unknown", "unknown", "other", False)


def test_a_server_that_never_comes_back_is_given_up_on_at_the_wait_with_not_back_and_exit_1(chat):
    cluster = FakeCluster(chat, "graceful", down_s=None)
    began = time.monotonic()
    code, out, err = flow(chat, cluster, wait="2")
    result = the_result(out)
    assert time.monotonic() - began < 20, "the wait is the limit, and the drill does not run on"
    assert code == 1 and err == "" and result["error"] == "not_back" and result["ok"] is False
    assert result["back_s"] is None and result["first_refused_s"] is not None and result["first_refused_s"] < 10
    assert result["turn_status"] is None and result["events"] is None and result["stream_end"] == "error_event", "the stream had ended: the old server said why"
    assert result["session_deleted"] is False, "the server is not there to delete it"
    assert result["node_before"] == "gx14" and result["node_after"] is None, "the job waits in the queue: the scheduler names no node"
    assert cluster.calls.count(("requeue", JOB)) == 1


def test_a_requeue_that_changes_nothing_is_not_back_either_and_has_no_first_refusal_because_the_old_server_answers_all_along(chat):
    chat.deltas = 8     # the turn ends by itself, a done
    code, out, err = flow(chat, FakeCluster(chat, how=None), wait="1.5")
    result = the_result(out)
    assert code == 1 and result["error"] == "not_back" and result["first_refused_s"] is None and result["back_s"] is None
    assert result["session_deleted"] is True and result["stream_end"] == "closed" and result["node_after"] == "gx14"


def test_a_requeue_that_fails_is_a_fixed_code_nothing_of_the_exceptions_text_is_printed_and_the_session_is_still_deleted(chat):
    cluster = FakeCluster(chat, fail=RuntimeError("SECRET: Permission denied (publickey) for job 123"))
    code, out, err = flow(chat, cluster)
    result = the_result(out)
    assert code == 1 and err == "" and "SECRET" not in out and "Permission" not in out
    assert result["error"] == "requeue_failed" and result["ok"] is False
    assert (result["requeue_s"], result["first_refused_s"], result["back_s"], result["turn_status"], result["node_after"]) == (None,) * 5
    assert result["deltas_at_requeue"] >= 3 and result["session_deleted"] is True and chat.deleted, "the turn that was started is stopped by the deletion"
    assert cluster.calls == [("node", JOB), ("requeue", JOB)], "no second try"


@pytest.mark.parametrize("failure", [RuntimeError("SECRET"), requeue.ClusterError("failed"), OSError("SECRET ssh"), subprocess.TimeoutExpired("ssh", 60)])
def test_whatever_the_cluster_raises_the_code_is_the_same_fixed_one(chat, failure):
    code, out, err = flow(chat, FakeCluster(chat, fail=failure))
    assert the_result(out)["error"] == "requeue_failed" and err == "" and "SECRET" not in out


def test_a_cluster_that_cannot_name_the_node_ends_the_drill_before_a_session_or_a_turn_is_made(chat):
    cluster = FakeCluster(chat, node_fail=requeue.ClusterError("failed"))
    code, out, err = flow(chat, cluster)
    result = the_result(out)
    assert code == 1 and err == "" and result["error"] == "node_failed" and result["ok"] is False and result["session_deleted"] is None
    assert requests_of(chat) == [("GET", "/healthz")] and cluster.calls == [("node", JOB)]


def test_a_job_that_is_not_running_has_no_node_and_is_not_requeued(chat):
    """The scheduler prints no node for a job that is pending or gone: nothing is requeued, and nothing is sent to the server but the first /healthz."""
    cluster = FakeCluster(chat, no_node=True)
    code, out, err = flow(chat, cluster)
    result = the_result(out)
    assert code == 1 and result["error"] == "job_not_running" and result["node_before"] is None and requests_of(chat) == [("GET", "/healthz")]
    assert cluster.calls == [("node", JOB)]


def test_a_server_that_is_not_up_when_the_drill_starts_ends_it_before_the_cluster_is_asked_anything(chat):
    chat.stop()
    code, out, err = flow(chat, NoCluster())
    result = the_result(out)
    assert code == 1 and err == "" and result["error"] == "healthz_ConnectError" and result["ok"] is False and result["session_deleted"] is None


@pytest.mark.parametrize("status, kind", [(422, "validation_error"), (429, "overloaded_error"), (500, "internal_error"), (503, "unavailable_error")])
def test_a_turn_the_server_refuses_is_the_status_as_a_code_and_nothing_is_requeued(chat, status, kind):
    chat.refuse_turn = (status, kind)
    cluster = FakeCluster(chat)
    code, out, err = flow(chat, cluster)
    result = the_result(out)
    assert code == 1 and err == "" and result["error"] == "turn_http_{}".format(status) and result["ok"] is False and "SECRET" not in out
    assert ("requeue", JOB) not in cluster.calls and result["deltas_at_requeue"] is None and result["stream_end"] is None
    assert result["session_deleted"] is True


def test_a_turn_that_ends_before_its_third_delta_is_not_interrupted_there_is_nothing_to_interrupt(chat):
    chat.deltas = 2
    cluster = FakeCluster(chat)
    code, out, err = flow(chat, cluster)
    result = the_result(out)
    assert code == 1 and result["error"] == "no_deltas" and result["stream_end"] == "closed" and result["deltas_at_requeue"] is None
    assert ("requeue", JOB) not in cluster.calls and result["session_deleted"] is True


def test_a_turn_that_finishes_before_the_requeue_lands_is_stored_done_and_the_drill_does_not_hold(chat):
    """The requeue takes long to reach the scheduler: the turn ended on its own, `done`, before the server went down. The drill ran, the server came
    back, and the answer is that the turn was not interrupted."""
    chat.deltas = 6
    code, out, err = flow(chat, FakeCluster(chat, "hard", delay_s=1.0))
    result = the_result(out)
    assert code == 1 and err == "" and result["ok"] is False and "error" not in result
    assert (result["stream_end"], result["turn_status"], result["turn_error"], result["turn_last_event"]) == ("closed", "done", None, "message_stop")
    assert result["back_s"] is not None and result["requeue_s"] >= 1.0


def test_a_stored_turn_that_cannot_be_read_after_the_server_is_back_is_a_fixed_code_and_does_not_hold(chat):
    chat.message_status = 500
    code, out, err = flow(chat, FakeCluster(chat))
    result = the_result(out)
    assert code == 1 and result["error"] == "stored_turn_http_500" and result["back_s"] is not None and result["turn_status"] is None
    assert result["ok"] is False and result["session_deleted"] is True


def test_a_session_that_cannot_be_deleted_is_reported_and_is_no_part_of_ok(chat):
    """`ok` is the drill's own question: the turn stored as error with server_restart, and the server back. The session that is left behind is on the
    line, as `session_deleted`, for the person who runs it."""
    chat.delete_status = 404
    code, out, err = flow(chat, FakeCluster(chat))
    result = the_result(out)
    assert code == 0 and err == "" and result["session_deleted"] is False and result["ok"] is True


def test_an_interrupt_while_the_requeue_is_sent_still_prints_the_line_deletes_the_session_and_exits_1(chat):
    code, out, err = flow(chat, FakeCluster(chat, fail=KeyboardInterrupt()))
    result = the_result(out)
    assert code == 1 and err == "" and result["error"] == "interrupted" and result["ok"] is False and result["session_deleted"] is True


def test_a_failure_inside_the_drill_is_named_by_its_class_and_never_by_its_text_and_the_session_is_still_deleted(chat, monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("SECRET Findings: report text in an exception")
    monkeypatch.setattr(requeue, "wait_for_return", broken)
    code, out, err = flow(chat, FakeCluster(chat, how=None))   # the server stays up, so the session can be deleted
    result = the_result(out)
    assert code == 1 and err == "" and result["error"] == "internal_ValueError" and "SECRET" not in out and result["session_deleted"] is True


def test_a_stream_that_goes_silent_when_the_server_goes_down_does_not_hold_up_the_polling_or_the_drill(chat, monkeypatch):
    """A tunnel whose far end is gone keeps the page's connection open and says nothing. The reader waits on the stream, the drill does not: it polls
    at once, sees the refusals and the return, and gives the stream a moment at the end, then calls it a timeout."""
    monkeypatch.setattr(requeue, "FINISH_S", 0.3)
    began = time.monotonic()
    code, out, err = flow(chat, FakeCluster(chat, "silent", down_s=0.4))
    result = the_result(out)
    assert code == 0 and err == "" and result["ok"] is True and result["stream_end"] == "timeout" and result["error_event"] is False
    assert result["first_refused_s"] is not None and result["back_s"] < 15 and time.monotonic() - began < 30
    assert (result["turn_status"], result["turn_error"]) == ("error", "server_restart"), "the next start's recovery closed the turn"


# ---- the real server: a process on the tiny engine, signalled, and started again on the same home and port ---------------------------

# The serving command line's own entry (app/cli.py main, which is what `python -m app.server` runs) with one change: the engine of `--models m3` is the
# tiny one, paced by a step delay so that a turn lasts seconds, and named as the registry names the published model (the swap of build_engine that
# tests/test_app_retrieval_stages.py makes in process). The options of create_app cannot pass through the command line, and cli.py takes none for this.
LAUNCH = """\
import sys
from app import cli, server
from app.engine import build_engine

home, port, step = sys.argv[1], sys.argv[2], float(sys.argv[3])


def named_like_the_registry(kind, **kw):
    engine = build_engine("tiny", step_delay_s=step)
    engine.name = kw.get("name") or kw["model_config"]
    engine._card.update(name=engine.name, checkpoint=kw["checkpoint"])
    return engine


server.build_engine = named_like_the_registry
sys.exit(cli.main(["--engine", "real", "--models", "m3", "--mode", "private", "--home", home, "--port", port]))
"""
STEP_S = 0.05    # between two decoding steps: 200 tokens take about ten seconds, so the turn is far from its end when the signal comes


class TinyServerProcess:
    """The serving command line on the tiny engine in a process of its own, in a session of its own (a kill reaches all of it), on a given home and
    port, with its output in a log file. One home and one port, many lives: start() brings up the next one."""

    def __init__(self, home, port, log):
        self.home, self.port, self.log = home, port, log
        self.proc = None

    @property
    def base(self):
        return "http://127.0.0.1:{}".format(self.port)

    def start(self, timeout=90.0):
        with open(str(self.log), "ab") as out:
            self.proc = subprocess.Popen([sys.executable, "-c", LAUNCH, str(self.home), str(self.port), repr(STEP_S)], cwd=str(REPO_ROOT),
                                         env=dict(os.environ, PYTHONPATH=str(REPO_ROOT), PYTHONUNBUFFERED="1"), stdin=subprocess.DEVNULL,
                                         stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.time() + timeout
        while time.time() < deadline and self.proc.poll() is None:
            try:
                if httpx.get(self.base + "/healthz", timeout=2).status_code == 200:
                    return
            except httpx.HTTPError:
                time.sleep(0.1)
        kill_group(self.proc)
        raise RuntimeError("the tiny server did not start:\n" + self.text()[-3000:])

    def text(self):
        return Path(self.log).read_text(errors="replace") if Path(self.log).exists() else ""


class ProcessCluster:
    """The fake cluster of the two arms. node() answers gx14, and gx15 once the job is on its way back (the new node). requeue() signals the server
    process as SLURM does on a requeue (SIGTERM; SIGKILL for the hard arm) and returns at once, as scontrol does; in the background it waits a bounded
    time for the process to go, pauses a moment (the queue), and starts the next server on the same home and port. It is the only thing here that
    signals a process: the drill starts and kills nothing of its own."""

    def __init__(self, first, second, signum, pause_s=1.0, grace_s=20.0):
        self.first, self.second, self.signum, self.pause_s, self.grace_s = first, second, signum, pause_s, grace_s
        self.calls, self.node_name, self.error, self.comeback = [], "gx14", None, None

    def node(self, job):
        self.calls.append(("node", job))
        return self.node_name

    def requeue(self, job):
        self.calls.append(("requeue", job))
        os.kill(self.first.proc.pid, self.signum)
        self.comeback = threading.Thread(target=self._come_back, daemon=True)
        self.comeback.start()

    def _come_back(self):
        try:
            try:
                self.first.proc.wait(timeout=self.grace_s)
            except subprocess.TimeoutExpired:   # a server that ignores SIGTERM is killed, as SLURM kills it after KillWait
                kill_group(self.first.proc)
            time.sleep(self.pause_s)
            self.node_name = "gx15"
            self.second.start()
        except BaseException as exc:   # reported by the test, from this thread
            self.error = exc


@pytest.fixture
def servers(tmp_path):
    """Two lives of one tiny server: the first started, the second not yet (the cluster starts it). Whatever still runs at the end is stopped."""
    port = free_port()
    assert port not in (8000, 8010), "a test server never takes the ports of the user's own servers"
    home = tmp_path / "chat_home"
    first, second = TinyServerProcess(home, port, tmp_path / "first.log"), TinyServerProcess(home, port, tmp_path / "second.log")
    first.start()
    yield first, second
    for process in (first, second):
        stop_group(process.proc, 5.0)


ARMS = [("graceful", signal.SIGTERM, 0, "error_event", True, 0), ("hard", signal.SIGKILL, 1, "reset", False, -signal.SIGKILL)]


@pytest.mark.parametrize("arm, signum, recovered, stream_end, error_event, exit_status", ARMS, ids=[a[0] for a in ARMS])
def test_the_drill_against_a_real_server_that_is_signalled_mid_turn_and_started_again_on_its_home_and_port(
        arm, signum, recovered, stream_end, error_event, exit_status, servers):
    """Both arms end in a stored turn that is an error with server_restart, and a drill that holds. SIGTERM is the production path of app/cli.py: the
    running turn is stopped at its next step as an error with server_restart, stored and sent, and the process exits 0, so the new server recovers no
    turn. SIGKILL is a node that is lost: the stream is cut with no frame, the turn is left running in the database, and the next start's
    recover_after_restart is what stores the restart. Either way the new server opens the same SQLite file in WAL mode, which is the D22 line."""
    first, second = servers
    cluster = ProcessCluster(first, second, signum)
    code, out, err = run_main([first.base, "--job", JOB, "--wait", "120"], cluster=cluster, poll_s=0.2)
    result = the_result(out)
    assert code == 0 and err == "" and result["ok"] is True, (code, out, err)
    cluster.comeback.join(60)
    assert cluster.error is None, cluster.error
    assert (result["drill"], result["node_before"], result["node_after"]) == ("requeue", "gx14", "gx15")
    assert cluster.calls == [("node", JOB), ("requeue", JOB), ("node", JOB)]
    assert result["deltas_at_requeue"] >= 3 and result["requeue_s"] >= 0
    assert result["first_refused_s"] is not None and result["requeue_s"] <= result["first_refused_s"] < result["back_s"] < 60
    assert (result["stream_end"], result["error_event"], result["stream_error"]) == (stream_end, error_event, "server_restart" if error_event else None)
    assert (result["turn_status"], result["turn_error"], result["turn_last_event"]) == ("error", "server_restart", "message_stop")
    assert result["events"] >= 7 and result["session_deleted"] is True and result["ok"] is True and "error" not in result
    assert first.proc.returncode == exit_status, "the first server was stopped as the arm says"
    log = second.text()
    assert "[server] sqlite journal_mode=wal\n" in log, "D22: the new server opened the database in WAL mode"
    assert "[server] recovered {} running turn(s)\n".format(recovered) in log


def test_a_model_the_server_does_not_have_is_refused_by_the_real_server_and_the_drill_requeues_nothing(servers, monkeypatch):
    """`m3` is the serving command's alias, not a model name: asked for by that name the real server answers 422, before any stream, and the drill
    ends on it without a requeue. (The drill names the registry's own name, hybrid_150m_m3_rrg, which the arms above run.)"""
    first, second = servers
    monkeypatch.setattr(requeue, "MODEL", "m3")
    cluster = ProcessCluster(first, second, signal.SIGTERM)
    code, out, err = run_main([first.base, "--job", JOB, "--wait", "10"], cluster=cluster, poll_s=0.2)
    result = the_result(out)
    assert code == 1 and err == "" and result["error"] == "turn_http_422" and result["ok"] is False and ("requeue", JOB) not in cluster.calls
    assert first.proc.poll() is None, "the server was never signalled"


# ---- the script as the controller runs it: its own process, the real SshCluster, and an `ssh` that is a stub -------------------------

STUB_SSH = """#!/bin/bash
# a stand-in for ssh: it writes down its arguments, one call per line, and answers as squeue would (the node's name) or says nothing
printf '%s\\n' "$*" >> "$STUB_LOG"
case " $* " in *" squeue "*) echo gx14 ;; esac
exit 0
"""


def test_the_script_runs_as_a_script_from_any_directory_through_its_own_ssh_commands_and_prints_nothing_but_the_result_line(tmp_path):
    """`venv/bin/python scripts/chat_requeue_drill.py URL --job N`, in a process of its own, from a directory that is not the repository: the real
    SshCluster, whose ssh is a stub found first on a PATH that holds nothing else (so no real ssh can run), against the fake server, which goes down
    when the stub has been called with `scontrol requeue`. What the process printed on both streams is read for the SECRETs the fake sent, and what
    the stub was asked is the two commands, in order: the node, the requeue, the node."""
    stubs = tmp_path / "bin"
    stubs.mkdir()
    (stubs / "ssh").write_text(STUB_SSH)
    (stubs / "ssh").chmod(0o755)
    log = tmp_path / "ssh_calls.log"
    log.write_text("")
    work = tmp_path / "cwd"
    work.mkdir()
    env = {"PATH": str(stubs), "STUB_LOG": str(log), "HOME": str(tmp_path), "PYTHONDONTWRITEBYTECODE": "1"}
    script = REPO_ROOT / "scripts" / "chat_requeue_drill.py"
    chat = FakeChat()
    watching = threading.Event()

    def watch():   # the cluster's side of it: a requeue that reached the scheduler takes the server down, and it comes back
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and not watching.is_set():
            if "scontrol" in log.read_text():
                chat.requeued("graceful", 0.4)
                return
            time.sleep(0.02)
    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        done = subprocess.run([sys.executable, str(script), chat.base, "--job", JOB, "--wait", "60"], cwd=str(work), env=env, capture_output=True,
                              text=True, timeout=180)
        refused = subprocess.run([sys.executable, str(script), "http://localhost:8000", "--job", JOB], cwd=str(work), env=env, capture_output=True,
                                 text=True, timeout=60)
    finally:
        watching.set()
        watcher.join(5)
        chat.close()
    assert done.returncode == 0 and done.stderr == "", (done.stdout, done.stderr)
    result = the_result(done.stdout)
    assert result["ok"] is True and (result["node_before"], result["node_after"]) == ("gx14", "gx14") and result["turn_error"] == "server_restart"
    assert "SECRET" not in done.stdout + done.stderr and list(work.iterdir()) == []
    assert log.read_text().splitlines() == [
        "-o BatchMode=yes hpi-hpc squeue -j {} -h -o %N".format(JOB), "-o BatchMode=yes hpi-hpc scontrol requeue {}".format(JOB),
        "-o BatchMode=yes hpi-hpc squeue -j {} -h -o %N".format(JOB)], "the stub saw the two commands the drill may build, and nothing else"
    assert refused.returncode == 2 and refused.stdout == "" and refused.stderr.startswith("ERROR ") and "8000" in refused.stderr
    assert refused.stderr.count("\n") == 1 and log.read_text().count("\n") == 3, "the refusal ran no command"
