"""CHAT_UI_PLAN.md P8-B: retention. How many days a session is kept (RETENTION_DAYS, else 45 private and 7 public), the thread that deletes the
older ones at start-up and every 24 hours, and the /healthz field that the page's banner reads.

The brief's test comes first, verbatim. Then the number of days (the default, the environment, the argument, and what keeps the server from
starting), both modes swept by every sweep, the thread (it starts after the restart recovery and has finished its first sweep before the first
request, it stops at once, it survives a sweep that fails, it never takes a session with a turn in flight) and what the page is told. Nothing here
waits for a day: the thread's interval is a parameter, and a test that needs several sweeps uses 20 ms. Synthetic data only (R7).

The last sections are the P9-C follow-up (user, 2026-10-14): the exports that Store.save_export wrote go after the private days too, through
Store.sweep_exports (names, times, links, failures), the sweeper's export_days and the server's line, which now carries both counts.
"""
import logging
import os
import secrets
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

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
    assert [l for l in out.splitlines() if "retention" in l] == ["[server] retention: purged 2 session(s), 0 saved file(s) (private 14 d, public 7 d)"]


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
    assert "[server] retention: purged 1 session(s), 0 saved file(s) (private 14 d, public 7 d)" in out
    assert out.splitlines() and all(JOB_LOG_LINE.match(line) for line in out.splitlines())


# ---- the saved exports follow the private days (P9-C follow-up; user, 2026-10-14: "delete them after 45 days") -----------------------------------
# Store.save_export writes session-<id>-<UTC time>[-n].json|md under <home>/exports/. Store.sweep_exports removes the ones whose time, the one in
# the name, is more than the days ago, and the hidden temporary files that a save that died left. Synthetic names and bodies only (R7).

SID = "s_0198a4f1c2b3d4e5f60718"   # a session id as new_id("s") makes it: s_ and 22 hex digits


def _export_name(days, ext="json", suffix="", now=None):
    """The name Store.save_export gives a file it saved `days` days before `now`: session-<id>-<UTC time to the second>[-n].<ext>."""
    when = (now or datetime.now(timezone.utc)) - timedelta(days=days)
    return "session-{}-{}{}.{}".format(SID, when.strftime("%Y%m%dT%H%M%SZ"), suffix, ext)


def _put(home, name, age_days=0.0, mtime=None):
    """A file `name` under exports/ of `home` (the directory made 0700, as the save makes it), its own time `age_days` days old, or `mtime` (seconds
    since the epoch) when that is given: -> its path. It is built, time and all, beside exports/ and put in with one rename, so that it is never in
    exports/ without its time: a sweep that runs at any moment, as the thread of a test with a short interval does, finds none or the whole file."""
    directory = Path(home) / "exports"
    directory.mkdir(mode=0o700, exist_ok=True)
    staged = Path(home) / ".put-{}".format(secrets.token_hex(4))   # beside exports/, so on its filesystem: the rename below is one step
    staged.write_text("body of " + name)
    then = time.time() - age_days * 86400 if mtime is None else mtime
    os.utime(str(staged), (then, then))
    path = directory / name
    os.replace(str(staged), str(path))
    return path


@pytest.fixture
def export_store(tmp_path):
    store = Store(tmp_path)
    try:
        yield store
    finally:
        store.close()


def test_the_helper_that_puts_a_file_into_exports_makes_it_appear_whole_a_sweep_between_its_steps_finds_none_to_take(export_store, tmp_path, monkeypatch):
    """The test with a thread that sweeps every 20 ms puts an old-stamped file into a live exports/. Written there and given its time afterwards, the
    file could be swept in between, and os.utime then fails on a file that is gone. _put builds it, time and all, beside exports/ and puts it in with
    one rename. Here a sweep runs at the moment between the steps, which is when the helper's first form was caught."""
    real_utime = os.utime
    swept = []

    def sweep_then_utime(path, *args, **kwargs):
        swept.append(export_store.sweep_exports(45))   # a sweep, between the file's write and its time
        return real_utime(path, *args, **kwargs)

    with monkeypatch.context() as patched:
        patched.setattr(os, "utime", sweep_then_utime)
        put = _put(tmp_path, _export_name(60))
    assert swept == [0], "no file was in exports/ for that sweep to find"
    assert put.is_file() and put.read_text() == "body of " + put.name
    assert export_store.sweep_exports(45) == 1 and not put.exists(), "and it is there, whole, for the next sweep"


def test_an_export_stamped_46_days_ago_is_removed_and_one_stamped_44_days_ago_is_kept(export_store, tmp_path):
    old = [_put(tmp_path, _export_name(46, ext)) for ext in ("json", "md")]
    young = [_put(tmp_path, _export_name(44, ext)) for ext in ("json", "md")]
    assert export_store.sweep_exports(45) == 2
    assert not any(p.exists() for p in old) and all(p.is_file() for p in young)


def test_an_export_exactly_as_old_as_the_days_stays_until_it_is_older_the_rule_of_the_session_sweep(export_store, tmp_path):
    """Store.sweep purges a session whose creation is before now - days (`created_at < cutoff`) and keeps one that is exactly that old. The export
    sweep has the rule: strictly before. The time of the name is to the second, and the clock is given, so that a name can be exactly on the line."""
    now = datetime(2026, 10, 14, 12, 0, 0, tzinfo=timezone.utc)
    on_the_line = _put(tmp_path, _export_name(45, "json", now=now))                                # 20260830T120000Z
    a_second_over = _put(tmp_path, _export_name(45, "md", now=now - timedelta(seconds=1)))          # 20260830T115959Z
    assert on_the_line.name.endswith("20260830T120000Z.json") and a_second_over.name.endswith("20260830T115959Z.md")
    assert export_store.sweep_exports(45, now=now) == 1
    assert on_the_line.is_file() and not a_second_over.exists()
    assert export_store.sweep_exports(45, now=now + timedelta(seconds=1)) == 1 and not on_the_line.exists()


def test_a_leftover_temporary_file_older_than_a_day_is_removed_and_a_fresh_one_is_kept(export_store, tmp_path):
    old = _put(tmp_path, ".session-{}.0a1b2c3d.tmp".format(SID), age_days=1.5)
    fresh = _put(tmp_path, ".session-{}.4e5f6a7b.tmp".format(SID), age_days=0.5)
    assert export_store.sweep_exports(45) == 1
    assert not old.exists() and fresh.is_file()


def test_a_temporary_file_is_a_day_old_when_it_is_older_than_a_day_exactly_a_day_stays(export_store, tmp_path):
    now = datetime(2026, 10, 14, 12, 0, 0, tzinfo=timezone.utc)
    on_the_line = _put(tmp_path, ".session-{}.0a1b2c3d.tmp".format(SID), mtime=now.timestamp() - 86400)
    a_second_over = _put(tmp_path, ".session-{}.4e5f6a7b.tmp".format(SID), mtime=now.timestamp() - 86401)
    assert export_store.sweep_exports(45, now=now) == 1
    assert on_the_line.is_file() and not a_second_over.exists()


def test_the_time_in_the_name_decides_not_the_time_of_the_file(export_store, tmp_path):
    copied_long_ago = _put(tmp_path, _export_name(44, "json"), age_days=400)   # a restored copy: its own time is old, its name is 44 days young
    written_now = _put(tmp_path, _export_name(46, "md"), age_days=0)           # its own time is now, its name 46 days old
    assert export_store.sweep_exports(45) == 1
    assert copied_long_ago.is_file() and not written_now.exists()


def test_a_name_stamped_in_the_future_is_kept_until_its_stamp_is_the_days_behind(export_store, tmp_path):
    """The time in the name decides, so a stamp that is ahead of the clock (a clock that ran ahead when the file was saved, a copy brought from
    elsewhere) keeps the file until the stamp is more than the days behind, whatever the file's own time says."""
    now = datetime(2026, 10, 14, 12, 0, 0, tzinfo=timezone.utc)
    ahead = _put(tmp_path, _export_name(-10, "json", now=now), mtime=now.timestamp() - 400 * 86400)   # stamped 10 days ahead; its own time is 400 days old
    assert ahead.name.endswith("20261024T120000Z.json")
    assert export_store.sweep_exports(45, now=now) == 0 and ahead.is_file()
    on_the_line = now + timedelta(days=55)   # the stamp plus the 45 days, to the second
    assert export_store.sweep_exports(45, now=on_the_line) == 0 and ahead.is_file()
    assert export_store.sweep_exports(45, now=on_the_line + timedelta(seconds=1)) == 1 and not ahead.exists()


def test_a_name_of_the_right_shape_with_a_time_that_is_no_time_goes_by_the_time_of_the_file(export_store, tmp_path):
    """The name is the save route's by shape, so it is ours, but its time cannot be read (month 99, month 13): the one fall back is the file's own time."""
    old = _put(tmp_path, "session-{}-20269999T999999Z.json".format(SID), age_days=60)
    young = _put(tmp_path, "session-{}-20261301T000000Z.md".format(SID), age_days=1)
    assert export_store.sweep_exports(45) == 1
    assert not old.exists() and young.is_file()


def test_a_home_with_no_exports_directory_is_fine_and_none_is_made(export_store, tmp_path):
    assert not (tmp_path / "exports").exists()
    assert export_store.sweep_exports(45) == 0
    assert not (tmp_path / "exports").exists()


def test_only_the_names_the_save_makes_are_touched_nothing_else_in_the_directory_or_the_home(export_store, tmp_path):
    old_time = datetime.now(timezone.utc) - timedelta(days=100)
    old = old_time.strftime("%Y%m%dT%H%M%SZ")
    older = (old_time - timedelta(days=5)).strftime("%Y%m%dT%H%M%SZ")   # another time, so that no two names differ in case alone (the laptop's disk ignores case)
    others = [   # old by the time in the name and by their own, and none of them is a name Store.save_export makes
        "notes.txt", ".hidden",
        "session-{}-{}.txt".format(SID, old),             # another extension
        "session-{}-{}.json.bak".format(SID, old),        # something after
        "session-{}-{}.JSON".format(SID, old),
        "session-{}-{}-x.json".format(SID, old),          # not a number after the time
        "xsession-{}-{}.json".format(SID, old),           # something before
        "session-{}.json".format(SID),                    # the download's own name: it has no time
        "session-s_x-{}.json".format(old),                # not a session id of the store's
        "session-{}-{}.json".format(SID.upper(), older),  # capitals in the id
        ".session-{}.tmp".format(SID),                    # a temporary name without its token
        ".session-{}.0a1b2c3z.tmp".format(SID),           # a token that is not hex
        "session-{}.0a1b2c3d.tmp".format(SID),            # not hidden
    ]
    files = {name: _put(tmp_path, name, age_days=100) for name in others}
    exports = tmp_path / "exports"
    folder = exports / _export_name(100, "md")                         # a directory named as an old export, with an old export inside it
    folder.mkdir()
    inner = folder / _export_name(100, "json")
    inner.write_text("inside")
    temp_folder = exports / ".session-{}.0b1c2d3e.tmp".format(SID)     # and one named as an old temporary file
    temp_folder.mkdir()
    (temp_folder / "x").write_text("x")
    for path in (folder, temp_folder):
        os.utime(str(path), (old_time.timestamp(), old_time.timestamp()))
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "target.txt"
    target.write_text("the target of two links")
    linked_dir = outside / "linked"
    linked_dir.mkdir()
    (linked_dir / "inside.txt").write_text("in a directory a link points to")
    links = {   # a link, named as an old export or an old temporary file, to a file, to a directory and to nothing
        exports / _export_name(101, "json"): target,
        exports / ".session-{}.1a2b3c4d.tmp".format(SID): target,
        exports / _export_name(102, "md"): linked_dir,
        exports / _export_name(103, "json"): outside / "gone",
    }
    for link, destination in links.items():
        link.symlink_to(destination, target_is_directory=destination == linked_dir)
    elsewhere = [tmp_path / _export_name(100), tmp_path / "uploads" / _export_name(100), outside / _export_name(100, "md")]   # old, but not in exports/
    for path in elsewhere:
        path.write_text("not in exports/")
    removable = _put(tmp_path, _export_name(46, "json"))   # so that the same sweep is seen to remove what it may
    before = sorted(p.name for p in exports.iterdir())
    assert export_store.sweep_exports(45) == 1 and not removable.exists()
    assert sorted(p.name for p in exports.iterdir()) == [n for n in before if n != removable.name], "nothing else left the directory"
    for name, path in files.items():
        assert path.read_text() == "body of " + name, name
    assert inner.read_text() == "inside" and (temp_folder / "x").read_text() == "x"
    for link, destination in links.items():
        assert link.is_symlink() and os.readlink(str(link)) == str(destination), link.name
    assert target.read_text() == "the target of two links" and (linked_dir / "inside.txt").read_text() == "in a directory a link points to"
    assert all(path.read_text() == "not in exports/" for path in elsewhere)


def test_an_exports_directory_that_is_a_link_is_not_followed(export_store, tmp_path, caplog):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    old = elsewhere / _export_name(60)
    old.write_text("an old export in the directory the link points to")
    (tmp_path / "exports").symlink_to(elsewhere, target_is_directory=True)
    with caplog.at_level(logging.DEBUG):
        assert export_store.sweep_exports(45) == 0
    assert old.is_file() and (tmp_path / "exports").is_symlink()
    lines = [r.getMessage() for r in caplog.records if r.name == "app.store"]
    assert len(lines) == 1 and str(tmp_path) not in lines[0] and "elsewhere" not in lines[0], lines


def test_an_exports_directory_swapped_for_a_link_after_the_check_is_not_followed_the_sweep_goes_on_in_the_one_it_opened(export_store, tmp_path, monkeypatch):
    """The check that exports/ is no link and the listing used to be two looks at the path, so an exports/ swapped for a link between them was listed
    through the link and the files of its target were removed. The directory is opened once, and the listing, the look at each entry and every removal
    go through that one handle. The swap is made at the moment the listing is asked for."""
    exports = tmp_path / "exports"
    own = _put(tmp_path, _export_name(60, "json"))   # an old export of the directory itself
    target = tmp_path / "elsewhere"
    target.mkdir()
    victims = [target / _export_name(61, "json"), target / _export_name(62, "md")]
    for victim in victims:
        victim.write_text("an old export of some other directory")
    moved = tmp_path / "exports-moved"
    real_scandir = os.scandir
    swapped = []

    def swap_then_list(where, *args, **kwargs):
        if not swapped:   # between the check and the listing: exports/ becomes a link to the directory that holds the victims
            os.rename(str(exports), str(moved))
            os.symlink(str(target), str(exports), target_is_directory=True)
            swapped.append(True)
        return real_scandir(where, *args, **kwargs)

    with monkeypatch.context() as patched:
        patched.setattr(os, "scandir", swap_then_list)
        removed = export_store.sweep_exports(45)
    assert swapped == [True] and exports.is_symlink()
    assert all(victim.is_file() for victim in victims), "what the link points to is never touched"
    assert removed == 1 and not (moved / own.name).exists(), "the sweep went on in the directory it had checked"


def test_a_file_that_cannot_be_removed_is_skipped_with_its_class_logged_and_the_others_go(export_store, tmp_path, monkeypatch, caplog):
    paths = [_put(tmp_path, _export_name(60 + n, ext)) for n, ext in enumerate(("json", "md", "json"))]
    stuck = paths[1]
    real_unlink = os.unlink

    def flaky(path, *args, **kwargs):
        if os.path.basename(str(path)) == stuck.name:
            raise OSError(13, "Permission denied: SECRET /sc/home/someone/chat_sessions/exports/" + stuck.name, str(path))
        return real_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patched:
        patched.setattr(os, "unlink", flaky)
        with caplog.at_level(logging.DEBUG):
            assert export_store.sweep_exports(45) == 2   # the one it could not remove is not counted
    assert stuck.is_file() and not paths[0].exists() and not paths[2].exists()
    lines = [r for r in caplog.records if r.name == "app.store"]
    assert len(lines) == 1 and lines[0].levelno == logging.WARNING and lines[0].exc_info is None, [r.getMessage() for r in lines]
    assert "PermissionError" in lines[0].getMessage()
    for text in (lines[0].getMessage(), caplog.text):
        assert "SECRET" not in text and stuck.name not in text and "session-" not in text and str(tmp_path) not in text and "/sc/home" not in text, text
    assert export_store.sweep_exports(45) == 1 and not stuck.exists(), "the next sweep tries it again"


@pytest.mark.parametrize("owner, name", [(os, "open"), (os, "scandir")], ids=["the open", "the listing"])
def test_a_directory_that_cannot_be_opened_or_listed_is_one_log_line_with_the_class_and_no_removals(export_store, tmp_path, monkeypatch, caplog, owner, name):
    kept = _put(tmp_path, _export_name(60))

    def refuse(*args, **kwargs):
        raise PermissionError(13, "Permission denied: SECRET", str(tmp_path / "exports"))

    with monkeypatch.context() as patched:
        patched.setattr(owner, name, refuse)
        with caplog.at_level(logging.DEBUG):
            assert export_store.sweep_exports(45) == 0
    lines = [r.getMessage() for r in caplog.records if r.name == "app.store"]
    assert len(lines) == 1 and "PermissionError" in lines[0] and "SECRET" not in caplog.text and str(tmp_path) not in caplog.text, lines
    assert kept.is_file()


def test_a_negative_number_of_days_is_refused_as_for_the_session_sweep(export_store):
    with pytest.raises(ValueError):
        export_store.sweep_exports(-1)


def test_the_sweep_takes_the_names_save_export_really_makes_and_the_temporary_name_it_really_writes_first(export_store, tmp_path, monkeypatch):
    from app import store as store_module
    session_id = export_store.create_session("private", None, "x")["id"]
    old_stamp = (datetime.now(timezone.utc) - timedelta(days=46)).strftime("%Y%m%dT%H%M%SZ")
    seen = []
    real_fsync = os.fsync

    def watching(fd):
        seen.extend(n for n in os.listdir(str(tmp_path / "exports")) if n.startswith("."))   # the hidden temporary file, while its body is written
        return real_fsync(fd)

    with monkeypatch.context() as patched:
        patched.setattr(store_module, "_stamp", lambda: old_stamp)
        patched.setattr(os, "fsync", watching)
        made = [export_store.save_export(session_id, fmt, None)[0] for fmt in ("json", "md", "json")]   # the third is a second save of the second
    assert made[2].endswith("{}-2.json".format(old_stamp)) and len(set(made)) == 3 and len(seen) == 3
    new = export_store.save_export(session_id, "json", None)[0]                  # saved now, with the clock's own time
    _put(tmp_path, seen[0], age_days=2)                                          # what a crash after the write would have left
    assert export_store.sweep_exports(45) == 4, "the three of 46 days ago and the temporary name"
    assert sorted(os.listdir(str(tmp_path / "exports"))) == [new]


# ---- the thread: the sessions first, then the exports, counted apart ----------------------------------------------------------------------------

class FakeExportStore(FakeStore):
    """FakeStore that has the export sweep too: `order` is every call it got, `export_raises` are exceptions for the first export sweeps, `exports` is
    what each of them removes."""

    def __init__(self, exports=0, export_raises=(), **kwargs):
        super().__init__(**kwargs)
        self.export_calls, self.exports, self.export_raises, self.order = [], exports, list(export_raises), []

    def sweep(self, days, mode=None):
        self.order.append("sessions " + str(mode))
        return super().sweep(days, mode)

    def sweep_exports(self, days):
        self.order.append("exports")
        with self.lock:
            self.export_calls.append(days)
            if self.export_raises:
                raise self.export_raises.pop(0)
            return self.exports


def test_a_sweeper_with_export_days_sweeps_the_exports_after_the_sessions_and_counts_them_apart():
    store = FakeExportStore(exports=3, result=2)
    sweeper = RetentionSweeper(store, POLICY, export_days=45)
    assert sweeper.export_days == 45 and sweeper.exports_purged == 0
    assert sweeper.sweep_once() == 4, "what it returns is the sessions, as before"
    assert sweeper.exports_purged == 3 and store.export_calls == [45]
    assert store.order == ["sessions private", "sessions public", "exports"]


def test_a_sweeper_without_export_days_never_asks_the_store_for_the_exports():
    store = FakeExportStore(exports=3, result=2)
    sweeper = RetentionSweeper(store, POLICY)
    assert sweeper.export_days is None and sweeper.sweep_once() == 4
    assert sweeper.exports_purged == 0 and store.export_calls == []
    assert RetentionSweeper(FakeStore(result=1), POLICY).sweep_once() == 2, "a store with no export sweep is as it was"


def test_exports_purged_is_what_the_last_sweep_removed_and_not_a_total():
    counts = iter([3, 0, 2])

    class Counting(FakeExportStore):
        def sweep_exports(self, days):
            return next(counts)

    sweeper = RetentionSweeper(Counting(), POLICY, export_days=45)
    seen = []
    for _ in range(3):
        sweeper.sweep_once()
        seen.append(sweeper.exports_purged)
    assert seen == [3, 0, 2]


def test_an_export_sweep_that_fails_is_one_log_line_with_the_class_only_and_the_session_sweeps_are_untouched(caplog):
    store = FakeExportStore(export_raises=[RuntimeError("SECRET-detail /sc/home/someone/chat_sessions/exports/" + _export_name(60) + " mimic")], result=1)
    sweeper = RetentionSweeper(store, POLICY, export_days=45)
    with caplog.at_level(logging.WARNING, logger="app.retention"):
        assert sweeper.sweep_once() == 2 and sweeper.exports_purged == 0
        assert sweeper.sweep_once() == 2, "the next sweep tries the exports again"
    assert store.calls == [(45, "private"), (7, "public")] * 2 and store.export_calls == [45, 45]
    failures = [r for r in caplog.records if r.name == "app.retention"]
    assert len(failures) == 1 and failures[0].levelno == logging.WARNING and failures[0].exc_info is None
    assert "RuntimeError" in failures[0].getMessage() and "export" in failures[0].getMessage()
    for secret in ("SECRET", "mimic", "/sc/home", "session-"):
        assert secret not in caplog.text, secret


def test_the_exports_are_swept_though_the_session_sweeps_of_both_modes_failed():
    store = FakeExportStore(exports=3, raises=[RuntimeError("one"), OSError("two")])
    sweeper = RetentionSweeper(store, POLICY, export_days=45)
    assert sweeper.sweep_once() == 0
    assert store.order == ["sessions private", "sessions public", "exports"] and sweeper.exports_purged == 3


def test_the_report_can_read_the_export_count_of_the_sweep_it_is_called_for():
    seen = []
    sweeper = RetentionSweeper(FakeExportStore(exports=3, result=1), POLICY, interval_s=0.02, export_days=45,
                               report=lambda purged, first: seen.append((purged, first, sweeper.exports_purged)))
    sweeper.start()
    try:
        wait_until(lambda: len(seen) >= 2)
    finally:
        sweeper.stop()
    assert seen[:2] == [(2, True, 3), (2, False, 3)]


# ---- the server: the days of its exports, and the line it prints -----------------------------------------------------------------------------------

def _serve_once(home, **kwargs):
    """Start a tiny server over `home` and stop it: the sweep it runs when it starts is what these tests look at."""
    with TestClient(create_app(engine="tiny", home=str(home), **kwargs)):
        pass


@pytest.mark.parametrize("kwargs, env, ages, kept", [
    ({}, None, (46, 44), (44,)),                         # a private server: 45 days
    ({"retention_days": 10}, None, (20, 5), (5,)),       # the argument moves the boundary
    ({"retention_days": 60}, None, (70, 50), (50,)),     # to a longer keep as well: the one of 50 days stays
    ({}, "14", (20, 10), (10,)),                         # RETENTION_DAYS is the days of a private server, and so of its exports
    ({"retention_days": 10}, "60", (20, 5), (5,)),       # the argument wins over the environment, as for the sessions
])
def test_a_private_server_keeps_its_exports_as_many_days_as_it_keeps_a_session(tmp_path, monkeypatch, kwargs, env, ages, kept):
    if env is not None:
        monkeypatch.setenv("RETENTION_DAYS", env)
    paths = {days: _put(tmp_path, _export_name(days)) for days in ages}
    _serve_once(tmp_path, **kwargs)
    assert {days for days, path in paths.items() if path.exists()} == set(kept)


@pytest.mark.parametrize("kwargs, env", [
    ({}, None),                         # a public server keeps a session 7 days
    ({"retention_days": 3}, None),      # a shorter keep of the sessions does not shorten the exports'
    ({"retention_days": 100}, None),    # nor does a longer one lengthen them
    ({}, "3"),                          # RETENTION_DAYS is the public server's own days
])
def test_a_public_server_still_removes_the_exports_a_private_one_wrote_after_the_private_default_of_45_days(tmp_path, monkeypatch, kwargs, env):
    """Exports are private-mode data: a public server saves none, but the home it is started over may hold the ones of a private server, and
    those are kept 45 days whatever the public days are."""
    if env is not None:
        monkeypatch.setenv("RETENTION_DAYS", env)
    young, old = _put(tmp_path, _export_name(20)), _put(tmp_path, _export_name(50))
    _serve_once(tmp_path, mode="public", token="t", **kwargs)
    assert young.is_file() and not old.exists()


def test_the_sweeper_of_a_server_is_given_the_private_days_of_the_policy_for_its_exports(tmp_path):
    for name, kwargs, days in (("private", {"retention_days": 14}, 14), ("public", {"mode": "public", "token": "t", "retention_days": 3}, 45)):
        app = create_app(engine="tiny", home=str(tmp_path / name), **kwargs)
        with TestClient(app):
            assert app.state.sweeper.export_days == days, name


def test_start_up_prints_both_counts_in_one_line_that_names_no_file_and_no_path(tmp_path, capsys):
    store = Store(tmp_path)
    for days in (50, 60):
        _age(store, store.create_session("private")["id"], days)
    store.close()
    names = [_export_name(46, "json"), _export_name(47, "md"), _export_name(60, "json", suffix="-2")]
    for name in names:
        _put(tmp_path, name)
    _put(tmp_path, ".session-{}.0a1b2c3d.tmp".format(SID), age_days=3)   # a leftover temporary file is counted with them
    kept = _put(tmp_path, _export_name(1))
    with TestClient(create_app(engine="tiny", home=str(tmp_path), retention_days=14)):
        out = capsys.readouterr().out
    lines = [l for l in out.splitlines() if "retention" in l]
    assert lines == ["[server] retention: purged 2 session(s), 4 saved file(s) (private 14 d, public 7 d)"]
    assert kept.is_file() and sorted(p.name for p in (tmp_path / "exports").iterdir()) == [kept.name]
    for secret in names + [kept.name, str(tmp_path), "session-", ".tmp", "exports/"]:
        assert secret not in lines[0], secret
    assert "export" not in lines[0], "the leak check of the timing job (tests/test_chat_decode_timing.py) takes that word for a mount path, and this line is in its log"


def test_the_first_sweep_says_its_counts_even_when_they_are_none(tmp_path, capsys):
    with TestClient(create_app(engine="tiny", home=str(tmp_path))):
        out = capsys.readouterr().out
    assert [l for l in out.splitlines() if "retention" in l] == ["[server] retention: purged 0 session(s), 0 saved file(s) (private 45 d, public 7 d)"]


def test_a_later_sweep_says_its_counts_only_when_it_removed_a_session_or_an_export(tmp_path, capsys):
    app = create_app(engine="tiny", home=str(tmp_path))
    app.state.sweeper.interval_s = 0.02
    with TestClient(app):
        sweeper, store = app.state.sweeper, app.state.store
        old = _put(tmp_path, _export_name(50))
        wait_until(lambda: not old.exists())
        base = sweeper.sweeps
        wait_until(lambda: sweeper.sweeps >= base + 3)   # sweeps that removed nothing, and so said nothing
        session = store.create_session("private")
        _age(store, session["id"], 100)
        wait_until(lambda: store.get_session(session["id"], None) is None)
        base = sweeper.sweeps
        wait_until(lambda: sweeper.sweeps >= base + 3)
    assert [l for l in capsys.readouterr().out.splitlines() if "retention" in l] == [
        "[server] retention: purged 0 session(s), 0 saved file(s) (private 45 d, public 7 d)",   # the first sweep, which always says it
        "[server] retention: purged 0 session(s), 1 saved file(s) (private 45 d, public 7 d)",   # the export
        "[server] retention: purged 1 session(s), 0 saved file(s) (private 45 d, public 7 d)",   # the session
    ]


def test_a_file_the_sweep_cannot_remove_does_not_stop_the_sessions_the_thread_or_the_server(tmp_path, monkeypatch, capsys, caplog):
    store = Store(tmp_path)
    session = store.create_session("private")
    _age(store, session["id"], 100)
    store.close()
    stuck, gone = _put(tmp_path, _export_name(60, "json")), _put(tmp_path, _export_name(61, "md"))
    real_unlink = os.unlink

    def flaky(path, *args, **kwargs):
        if os.path.basename(str(path)) == stuck.name:
            raise OSError(5, "Input/output error: SECRET", str(path))
        return real_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patched:
        patched.setattr(os, "unlink", flaky)
        with caplog.at_level(logging.WARNING):
            with TestClient(create_app(engine="tiny", home=str(tmp_path))) as c:
                assert c.get("/healthz").status_code == 200, "the server is up"
                out = capsys.readouterr().out
    assert _survivors(tmp_path) == set(), "the session was swept"
    assert stuck.is_file() and not gone.exists()
    assert [l for l in out.splitlines() if "retention" in l] == ["[server] retention: purged 1 session(s), 1 saved file(s) (private 45 d, public 7 d)"]
    assert "OSError" in caplog.text and "SECRET" not in caplog.text and stuck.name not in caplog.text and str(tmp_path) not in caplog.text
