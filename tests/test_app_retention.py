"""CHAT_UI_PLAN.md P8-B: retention. How many days a session is kept (RETENTION_DAYS, else 45 private and 7 public), the thread that deletes the
older ones at start-up and every 24 hours, and the /healthz field that the page's banner reads.

The brief's test comes first, verbatim. Then the number of days (the default, the environment, the argument, and what keeps the server from
starting), both modes swept by every sweep, the thread (it starts after the restart recovery and has finished its first sweep before the first
request, it stops at once, it survives a sweep that fails, it never takes a session with a turn in flight) and what the page is told. Nothing here
waits for a day: the thread's interval is a parameter, and a test that needs several sweeps uses 20 ms. Synthetic data only (R7).
"""
import logging
import threading
import time
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from fastapi.testclient import TestClient

from app import retention, server
from app.retention import RetentionSweeper
from app.server import create_app
from app.store import Store
from tests.app_helpers import wait_until


def _age(store, session_id, days):
    """Make a session `days` days old: Store.sweep counts from its creation time."""
    stamp = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="microseconds")
    with store._lock:   # the sweep thread may be using the connection
        store._con.execute("UPDATE sessions SET created_at = ? WHERE id = ?", (stamp, session_id))


def _survivors(tmp_path):
    """The ids of the sessions left in the database at `tmp_path` (the server that used it has closed it)."""
    store = Store(tmp_path)
    try:
        return {row[0] for row in store._con.execute("SELECT id FROM sessions")}
    finally:
        store.close()


@pytest.fixture(autouse=True)
def no_retention_in_the_environment(monkeypatch):
    monkeypatch.delenv(retention.RETENTION_ENV, raising=False)


# ---- the brief ------------------------------------------------------------------------------------------------------------------------

def test_startup_sweep_removes_expired_sessions_and_health_reports_the_policy(tmp_path):
    from app.store import Store
    s = Store(tmp_path)
    old = s.create_session("public", "a")
    s._con.execute("UPDATE sessions SET created_at='2000-01-01T00:00:00+00:00' WHERE id=?", (old["id"],))
    s._con.commit()
    s.close()
    with TestClient(create_app(engine="tiny", home=str(tmp_path), mode="public", token="t")) as c:
        assert c.get("/healthz").json()["retention_days"] == 7
        h = {"Authorization": "Bearer t", "X-Client-Id": "a"}
        assert c.get("/v1/sessions/{}".format(old["id"]), headers=h).status_code == 404


# ---- ruling 6: the number of days ---------------------------------------------------------------------------------------------------------

def test_the_defaults_are_45_days_on_a_private_server_and_7_on_a_public_one():
    assert retention.DEFAULT_RETENTION_DAYS == {"private": 45, "public": 7}
    assert retention.resolve_days("private") == 45 and retention.resolve_days("public") == 7


def test_retention_days_in_the_environment_is_the_number_in_either_mode(monkeypatch):
    monkeypatch.setenv("RETENTION_DAYS", "14")
    assert retention.resolve_days("private") == 14 and retention.resolve_days("public") == 14


def test_the_argument_wins_over_the_environment(monkeypatch):
    monkeypatch.setenv("RETENTION_DAYS", "14")
    assert retention.resolve_days("private", 30) == 30 and retention.resolve_days("public", 1) == 1


@pytest.mark.parametrize("raw, days", [("1", 1), ("3650", 3650), (" 21 ", 21), ("007", 7), ("30\n", 30)])
def test_a_whole_number_of_days_from_1_to_3650_is_taken_from_the_environment(monkeypatch, raw, days):
    monkeypatch.setenv("RETENTION_DAYS", raw)
    assert retention.resolve_days("private") == days


@pytest.mark.parametrize("raw", ["", "  ", "\n"])
def test_an_environment_variable_that_is_empty_is_the_same_as_none(monkeypatch, raw):
    monkeypatch.setenv("RETENTION_DAYS", raw)
    assert retention.resolve_days("private") == 45 and retention.resolve_days("public") == 7


BAD_ENV = ["0", "00", "-1", "+5", "3651", "100000", "9" * 5000, "SECRET-9", "7.5", "1e2", "0x10", "1_0", "7 days", "٣", "one", "١٢"]


@pytest.mark.parametrize("raw", BAD_ENV)
def test_anything_but_a_whole_number_from_1_to_3650_in_the_environment_is_refused_with_one_message_that_quotes_nothing(monkeypatch, raw):
    monkeypatch.setenv("RETENTION_DAYS", raw)
    with pytest.raises(ValueError) as refused:
        retention.resolve_days("private")
    assert str(refused.value) == "RETENTION_DAYS must be a whole number of days from 1 to 3650"
    assert "SECRET" not in str(refused.value)


@pytest.mark.parametrize("bad", [0, -1, 3651, True, False, 7.5, 7.0, "7", [7], 10 ** 9])
def test_the_argument_is_refused_unless_it_is_a_whole_number_from_1_to_3650(bad):
    with pytest.raises(ValueError) as refused:
        retention.resolve_days("private", bad)
    assert str(refused.value) == "retention_days must be a whole number of days from 1 to 3650"
    assert retention.resolve_days("private", 1) == 1 and retention.resolve_days("private", 3650) == 3650


def test_a_server_with_a_bad_retention_refuses_to_start_and_opens_nothing(tmp_path, monkeypatch):
    home = tmp_path / "home"
    store = Store(home)
    session = store.create_session("private")
    _age(store, session["id"], 1000)
    store.close()
    threads = set(threading.enumerate())
    with pytest.raises(ValueError, match="retention_days must be a whole number of days from 1 to 3650"):
        create_app(engine="tiny", home=str(home), retention_days=0)   # 0 would delete every session at start-up
    monkeypatch.setenv("RETENTION_DAYS", "0")
    with pytest.raises(ValueError, match="RETENTION_DAYS must be a whole number of days from 1 to 3650"):
        create_app(engine="tiny", home=str(home))
    assert set(threading.enumerate()) == threads                     # no thread was started
    assert _survivors(home) == {session["id"]}                        # and no sweep ran: the database opened for this check is the first since


def test_the_cli_refuses_a_bad_retention_days_with_one_error_line_before_it_loads_anything(monkeypatch, capsys):
    from app import cli
    monkeypatch.setenv("RETENTION_DAYS", "0")
    with pytest.raises(SystemExit) as stop:
        cli.parse_args(["--engine", "tiny", "--mode", "private", "--home", "HOME"])
    assert stop.value.code == 2
    assert capsys.readouterr().out.splitlines() == ["ERROR RETENTION_DAYS must be a whole number of days from 1 to 3650"]
    monkeypatch.setenv("RETENTION_DAYS", "14")
    assert cli.parse_args(["--engine", "tiny", "--mode", "private", "--home", "HOME"]).mode == "private"


# ---- ruling 6: both policies at every sweep -------------------------------------------------------------------------------------------------

def _four_sessions(home):
    """One private and one public session each 50 and 30 days old, and one public one 8 and 3 days old: -> {name: id}."""
    store = Store(home)
    ids = {}
    for name, mode, days in [("private 50", "private", 50), ("private 30", "private", 30), ("public 8", "public", 8), ("public 3", "public", 3)]:
        ids[name] = store.create_session(mode, "a")["id"]
        _age(store, ids[name], days)
    store.close()
    return ids


@pytest.mark.parametrize("mode, days, kept", [
    ("private", None, ["private 30", "public 3"]),    # 45 days for private, the default 7 for public
    ("public", None, ["private 30", "public 3"]),     # 7 days for public, the default 45 for private
    ("private", 10, ["public 3"]),                    # private 10 days: both private ones go; public still 7
    ("public", 2, ["private 30"]),                    # public 2 days: both public ones go; private still 45
    ("private", 60, ["private 50", "private 30", "public 3"]),   # a longer private keep never lengthens the public one: public 8 is gone
    ("public", 30, ["private 30", "public 8", "public 3"]),      # public may be set longer than 7 on purpose: private 50 is past its 45
])
def test_every_sweep_runs_the_servers_days_for_its_mode_and_the_default_days_for_the_other(tmp_path, mode, days, kept):
    ids = _four_sessions(tmp_path)
    with TestClient(create_app(engine="tiny", home=str(tmp_path), mode=mode, token="t", retention_days=days)):
        pass
    assert _survivors(tmp_path) == {ids[name] for name in kept}


def test_a_default_that_the_environment_changes_still_leaves_the_other_modes_default_alone(tmp_path, monkeypatch):
    ids = _four_sessions(tmp_path)
    monkeypatch.setenv("RETENTION_DAYS", "100")   # a private server that keeps 100 days: the public 8-day-old one is still past 7
    with TestClient(create_app(engine="tiny", home=str(tmp_path))):
        pass
    assert _survivors(tmp_path) == {ids["private 50"], ids["private 30"], ids["public 3"]}


def test_the_policy_is_the_servers_days_for_its_mode_and_the_default_for_the_other():
    assert retention.retention_policy("private", 14) == {"private": 14, "public": 7}
    assert retention.retention_policy("public", 30) == {"public": 30, "private": 45}


# ---- ruling 7: the thread -----------------------------------------------------------------------------------------------------------------

class FakeStore:
    """Store.sweep as the thread calls it: the calls are kept, `raises` are exceptions for the first calls, `result` is what each call purges."""

    def __init__(self, raises=(), result=0, gate=None):
        self.calls, self.raises, self.result, self.gate = [], list(raises), result, gate
        self.lock = threading.Lock()

    def sweep(self, days, mode=None):
        if self.gate is not None:
            self.gate.wait(10)
        with self.lock:
            self.calls.append((days, mode))
            if self.raises:
                raise self.raises.pop(0)
            return self.result


POLICY = {"private": 45, "public": 7}


def test_a_sweep_runs_the_policy_of_each_mode_and_adds_up_what_was_purged():
    store = FakeStore(result=2)
    assert RetentionSweeper(store, POLICY).sweep_once() == 4
    assert store.calls == [(45, "private"), (7, "public")]


def test_the_thread_sweeps_when_it_starts_and_then_every_interval():
    store = FakeStore()
    sweeper = RetentionSweeper(store, POLICY, interval_s=0.02)
    assert not sweeper.running()
    sweeper.start()
    try:
        wait_until(lambda: len(store.calls) >= 6)   # three rounds of two modes
    finally:
        sweeper.stop()
    assert not sweeper.running() and store.calls[:2] == [(45, "private"), (7, "public")]
    seen = len(store.calls)
    time.sleep(0.1)
    assert len(store.calls) == seen, "it went on sweeping after stop()"


def test_stop_wakes_a_thread_that_is_waiting_for_a_day_at_once():
    store = FakeStore()
    sweeper = RetentionSweeper(store, POLICY)   # the default interval: 24 hours
    assert sweeper.interval_s == 24 * 3600 == retention.SWEEP_INTERVAL_S
    sweeper.start()
    assert sweeper.wait_first(5) is True
    started = time.monotonic()
    sweeper.stop()
    assert time.monotonic() - started < 1.0 and not sweeper.running()
    sweeper.stop()   # safe to repeat


def test_stop_before_the_first_sweep_runs_none():
    store = FakeStore()
    sweeper = RetentionSweeper(store, POLICY)
    sweeper.stop()
    sweeper.start()
    sweeper.stop()
    assert store.calls == [] and not sweeper.running()


def test_the_thread_is_a_daemon_named_for_its_job_and_start_twice_starts_one():
    sweeper = RetentionSweeper(FakeStore(), POLICY)
    sweeper.start()
    first = sweeper._thread
    sweeper.start()
    try:
        assert sweeper._thread is first and first.daemon and first.name == "retention-sweep"
        assert [t.name for t in threading.enumerate()].count("retention-sweep") == 1
    finally:
        sweeper.stop()


def test_a_sweep_that_fails_is_one_log_line_with_the_class_only_and_the_thread_goes_on(caplog):
    store = FakeStore(raises=[RuntimeError("SECRET-detail /sc/home/someone/uploads/s_1 mimic"), OSError("disk SECRET-2")])
    sweeper = RetentionSweeper(store, POLICY, interval_s=0.02)
    with caplog.at_level(logging.WARNING, logger="app.retention"):
        sweeper.start()
        try:
            wait_until(lambda: len(store.calls) >= 6)   # the first round failed on both modes, and the thread swept again
        finally:
            sweeper.stop()
    failures = [r for r in caplog.records if r.name == "app.retention"]
    assert len(failures) == 2 and [r.levelno for r in failures] == [logging.WARNING] * 2
    assert "RuntimeError" in failures[0].getMessage() and "private" in failures[0].getMessage()
    assert "OSError" in failures[1].getMessage() and "public" in failures[1].getMessage()
    for record in failures:
        assert "SECRET" not in record.getMessage() and "mimic" not in record.getMessage() and record.exc_info is None
    assert store.calls[:2] == [(45, "private"), (7, "public")]   # the failure of the first mode did not skip the second


def test_the_first_sweep_can_be_waited_for_without_waiting_for_ever():
    gate = threading.Event()
    sweeper = RetentionSweeper(FakeStore(gate=gate), POLICY)
    sweeper.start()
    try:
        assert sweeper.wait_first(0.05) is False     # the sweep is blocked: wait_first gives up at its timeout
        gate.set()
        assert sweeper.wait_first(5) is True
    finally:
        gate.set()
        sweeper.stop()


def test_the_report_gets_each_sweeps_count_and_whether_it_was_the_first_and_a_report_that_fails_ends_nothing(caplog):
    reports = []

    def report(purged, first):
        reports.append((purged, first))
        if len(reports) == 1:
            raise ValueError("SECRET-report")

    sweeper = RetentionSweeper(FakeStore(result=1), POLICY, interval_s=0.02, report=report)
    with caplog.at_level(logging.WARNING, logger="app.retention"):
        sweeper.start()
        try:
            wait_until(lambda: len(reports) >= 3)
        finally:
            sweeper.stop()
    assert reports[:3] == [(2, True), (2, False), (2, False)]
    assert "ValueError" in caplog.text and "SECRET" not in caplog.text


def test_a_session_with_a_turn_in_flight_is_never_swept_and_goes_with_the_next_sweep_after_the_turn_has_ended(tmp_path):
    store = Store(tmp_path)
    try:
        session = store.create_session("private")
        _, message_id = store.start_turn(session["id"], "x", "private", {})   # running
        _age(store, session["id"], 1000)
        sweeper = RetentionSweeper(store, retention.retention_policy("private", 45))
        assert sweeper.sweep_once() == 0 and store.get_session(session["id"], None) is not None
        store.finish_turn(message_id, "done", report="r")
        assert sweeper.sweep_once() == 1 and store.get_session(session["id"], None) is None
    finally:
        store.close()


def test_the_thread_starts_with_the_server_serving_and_has_swept_before_the_first_request(tmp_path):
    app = create_app(engine="tiny", home=str(tmp_path))
    sweeper = app.state.sweeper
    assert isinstance(sweeper, RetentionSweeper) and not sweeper.running() and sweeper.sweeps == 0   # nothing before the server serves
    with TestClient(app) as c:
        assert sweeper.running() and sweeper.sweeps >= 1 and sweeper._thread.daemon
        assert c.get("/healthz").json()["status"] == "ok"
    assert not sweeper.running()   # stopped with the server, not 24 hours later
    assert "retention-sweep" not in [t.name for t in threading.enumerate()]


def test_the_server_stops_at_once_though_the_thread_would_wait_for_a_day(tmp_path):
    app = create_app(engine="tiny", home=str(tmp_path))
    assert app.state.sweeper.interval_s == 24 * 3600
    with TestClient(app):
        assert app.state.sweeper.running()   # asleep until tomorrow
        stopping = time.monotonic()
    assert time.monotonic() - stopping < 5 and not app.state.sweeper.running()   # the Event wait ended at the stop, not a day later


def test_a_later_sweep_removes_what_has_expired_since_the_last_one(tmp_path):
    app = create_app(engine="tiny", home=str(tmp_path))
    app.state.sweeper.interval_s = 0.02
    with TestClient(app) as c:
        store = app.state.store
        session = store.create_session("private")
        _age(store, session["id"], 100)
        wait_until(lambda: c.get("/v1/sessions/{}".format(session["id"])).status_code == 404)


def test_the_first_sweep_runs_after_the_recovery_of_turns_a_restart_left_running_and_before_the_first_request(tmp_path):
    store = Store(tmp_path)
    session = store.create_session("private")
    store.start_turn(session["id"], "x", "private", {})   # a turn the last server died in
    _age(store, session["id"], 100)
    store.close()
    with TestClient(create_app(engine="tiny", home=str(tmp_path))) as c:
        # recovered first (so the turn is over and the session is idle), then swept: with the order the other way round it would be skipped
        assert c.get("/v1/sessions/{}".format(session["id"])).status_code == 404


def test_a_first_sweep_that_takes_long_does_not_hold_the_server_back_for_ever(tmp_path, monkeypatch):
    app = create_app(engine="tiny", home=str(tmp_path))
    release = threading.Event()
    real = app.state.store.sweep

    def slow(days, mode=None):
        release.wait(20)
        return real(days, mode)

    monkeypatch.setattr(app.state.store, "sweep", slow)
    monkeypatch.setattr(retention, "FIRST_SWEEP_WAIT_S", 0.2)
    started = time.monotonic()
    try:
        with TestClient(app) as c:
            assert time.monotonic() - started < 10 and c.get("/healthz").status_code == 200   # serving, the sweep still running
            assert app.state.sweeper.running()
            release.set()
            wait_until(lambda: app.state.sweeper.sweeps >= 1)
    finally:
        release.set()


# ---- ruling 8: what the page is told ----------------------------------------------------------------------------------------------------------

def test_healthz_says_the_days_of_the_servers_own_mode(tmp_path, monkeypatch):
    with TestClient(create_app(engine="tiny", home=str(tmp_path / "private"))) as c:
        assert c.get("/healthz").json()["retention_days"] == 45
    with TestClient(create_app(engine="tiny", home=str(tmp_path / "public"), mode="public", token="t")) as c:
        assert c.get("/healthz").json()["retention_days"] == 7
        assert c.app.state.retention_days == 7
    monkeypatch.setenv("RETENTION_DAYS", "21")
    with TestClient(create_app(engine="tiny", home=str(tmp_path / "env"), mode="public", token="t")) as c:
        assert c.get("/healthz").json()["retention_days"] == 21
    with TestClient(create_app(engine="tiny", home=str(tmp_path / "arg"), retention_days=5)) as c:
        assert c.get("/healthz").json()["retention_days"] == 5   # the argument over the environment


def test_the_example_in_the_reference_has_the_field_and_the_description_names_it(tmp_path):
    with TestClient(create_app(engine="tiny", home=str(tmp_path))) as c:
        op = c.get("/openapi.json").json()["paths"]["/healthz"]["get"]
        example = op["responses"]["200"]["content"]["application/json"]["example"]
        assert set(example) == set(c.get("/healthz").json()) and isinstance(example["retention_days"], int)
    assert "`retention_days`" in op["description"] and "RETENTION_DAYS" in op["description"]
    assert server.HEALTHZ_EXAMPLE["mode"] == "private" and server.HEALTHZ_EXAMPLE["retention_days"] == 45   # P7-C's tunnel tests read this dict


def test_start_up_prints_one_counts_only_server_line_for_the_first_sweep(tmp_path, capsys):
    store = Store(tmp_path)
    for days in (50, 60):
        _age(store, store.create_session("private")["id"], days)
    store.close()
    with TestClient(create_app(engine="tiny", home=str(tmp_path), retention_days=14)):
        out = capsys.readouterr().out
    assert [l for l in out.splitlines() if "retention" in l] == ["[server] retention: purged 2 session(s) (private 14 d, public 7 d)"]


def test_a_live_server_reads_retention_days_from_its_environment_and_says_so_in_one_server_line(tmp_path, monkeypatch):
    from tests.test_chat_server_cli import JOB_LOG_LINE, LiveCli
    monkeypatch.setenv("RETENTION_DAYS", "14")
    store = Store(tmp_path / "home")
    old, young = store.create_session("private"), store.create_session("private")
    _age(store, old["id"], 20)
    store.close()
    with LiveCli(tmp_path, "--mode", "private") as live:
        base = live.base()   # the endpoint file appears once the server serves, which is after the first sweep
        assert httpx.get(base + "/healthz").json()["retention_days"] == 14
        assert httpx.get(base + "/v1/sessions/" + old["id"]).status_code == 404
        assert httpx.get(base + "/v1/sessions/" + young["id"]).status_code == 200
        assert live.stop() == 0
    out = live.out.read_text()
    assert "[server] retention: purged 1 session(s) (private 14 d, public 7 d)" in out
    assert out.splitlines() and all(JOB_LOG_LINE.match(line) for line in out.splitlines())
