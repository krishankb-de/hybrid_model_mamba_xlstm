"""Retention for the chat app (CHAT_UI_PLAN.md P8-B): how long a session is kept, and the thread that deletes the sessions that are older.

**The number of days.** A session is deleted RETENTION_DAYS days after it was created (Store.sweep counts from creation, so a session that is still
in use goes too). RETENTION_DAYS is the environment variable when it is set, else 45 days on a private server (user, 2026-10-01) and 7 on a public
one. create_app takes the same number as `retention_days`, which wins over the environment. Only a whole number from 1 to 3650 is accepted: anything
else keeps the server from starting, with one message, and 0 above all (it would delete every session at start-up).

**Both modes.** Every sweep runs two policies: the server's own days for the sessions of its own mode, and the default days for the other mode. So
the sessions of a past public demo are gone after 7 days whatever mode the server runs in now, and a private server started with RETENTION_DAYS=60
still keeps nothing public for longer than 7 days.

**Saved exports (P9-C follow-up, user 2026-10-14: "delete them after 45 days").** The files the save route wrote under <home>/exports/ go after the
private days too: the sweep calls Store.sweep_exports with the policy's private days, so the server's own days when it is private (RETENTION_DAYS or
retention_days included) and the default 45 on a public one, which saves no export but may be started over a home that holds some. An export is old
by the time in its name, not by its session: it can outlive the session it came from, and it goes 45 days after it was saved.

**The thread.** RetentionSweeper sweeps once when the server starts, after the recovery of the turns that a restart left running (so the session of a
turn that died with the last server can go too), and every 24 hours after that. It is a daemon thread that waits on an Event, so stop() wakes it at
once and a shutdown never sleeps through the interval. It sweeps through Store.sweep, which takes the store's own lock a session at a time and never
deletes a session that has a turn in flight (that session waits for the next sweep). A sweep that raises is logged as one line, the class of the
exception and no text of it (a database or file error can name a path or a row), and the thread goes on: the other mode is still swept, and so is the
next interval. The server starts once the first sweep is done, or FIRST_SWEEP_WAIT_S later if a backlog makes it slow.
"""
import logging
import os
import re
import threading
from typing import Callable, Dict, Mapping, Optional

log = logging.getLogger(__name__)

RETENTION_ENV = "RETENTION_DAYS"
DEFAULT_RETENTION_DAYS = {"private": 45, "public": 7}   # U6 (user, 2026-10-01) and the public demo's one week
MAX_RETENTION_DAYS = 3650                               # ten years: a typo for "days" in "hours" or "seconds" stays refused
SWEEP_INTERVAL_S = 24 * 3600.0
FIRST_SWEEP_WAIT_S = 30.0   # how long start-up waits for the first sweep; a longer one goes on in the background
STOP_WAIT_S = 10.0          # how long stop() waits for a sweep that is running
REFUSAL = "{} must be a whole number of days from 1 to {}"   # names the source and never the value it was given
WHOLE_DAYS = re.compile(r"[0-9]{1,6}")                      # ASCII digits only; int() of a longer string would raise on Python 3.11+


def resolve_days(mode: str, given: Optional[int] = None, environ: Optional[Mapping[str, str]] = None) -> int:
    """The days this server keeps the sessions of its own mode: `given` (create_app's retention_days), else RETENTION_DAYS from the environment
    when it is set to something (empty is unset), else the mode's default. ValueError, with one message, for anything but a whole number from
    1 to 3650: a bool, a float, a string, 0 and a negative number are no day counts, and nothing here quotes what was given."""
    if given is not None:
        if isinstance(given, bool) or not isinstance(given, int) or not 1 <= given <= MAX_RETENTION_DAYS:
            raise ValueError(REFUSAL.format("retention_days", MAX_RETENTION_DAYS))
        return given
    raw = (os.environ if environ is None else environ).get(RETENTION_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_RETENTION_DAYS[mode]
    raw = raw.strip()
    if WHOLE_DAYS.fullmatch(raw) is None or not 1 <= int(raw) <= MAX_RETENTION_DAYS:
        raise ValueError(REFUSAL.format(RETENTION_ENV, MAX_RETENTION_DAYS))
    return int(raw)


def retention_policy(mode: str, days: int) -> Dict[str, int]:
    """{mode: days} for each of the two modes: this server's own days for its mode, the default for the other one."""
    other = "public" if mode == "private" else "private"
    return {mode: days, other: DEFAULT_RETENTION_DAYS[other]}


class RetentionSweeper:
    """Store.sweep for each mode of `days_by_mode`, once when start() is called and then every `interval_s` seconds (a test sets a short one) on a
    daemon thread. `report(purged, first)` is called after every sweep, from that thread (create_app prints the server's counts line with it).

    With `export_days` (create_app gives the private days, whatever the server's mode) every sweep also calls Store.sweep_exports with it, after the
    sessions: the exports the save route wrote go after the same days as a private session. `exports_purged` is how many files that removed in the
    sweep that finished last (0 without `export_days`), which `report` reads: `purged` stays the sessions."""

    def __init__(self, store, days_by_mode: Dict[str, int], interval_s: float = SWEEP_INTERVAL_S,
                 report: Optional[Callable[[int, bool], None]] = None, export_days: Optional[int] = None):
        self.store, self.days_by_mode, self.interval_s, self.report = store, dict(days_by_mode), interval_s, report
        self.export_days = export_days   # None: the saved exports are not swept
        self.exports_purged = 0          # the files the last sweep removed from the exports directory
        self.sweeps = 0   # sweeps finished, failed ones included
        self._stop = threading.Event()
        self._first_done = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def sweep_once(self) -> int:
        """One sweep of every mode, then of the saved exports when `export_days` is set; -> the sessions purged (exports_purged has the files).
        It never raises: a failure is one log line (the mode or the exports, the exception's class) and the rest is swept anyway."""
        purged = 0
        for mode, days in self.days_by_mode.items():
            try:
                purged += self.store.sweep(days, mode)
            except Exception as exc:
                log.warning("retention sweep of %s sessions failed (%s); it is tried again at the next sweep", mode, type(exc).__name__)
        exports = 0
        if self.export_days is not None:
            try:
                exports = self.store.sweep_exports(self.export_days)
            except Exception as exc:
                log.warning("retention sweep of saved exports failed (%s); it is tried again at the next sweep", type(exc).__name__)
        self.exports_purged = exports
        return purged

    def start(self) -> None:
        """Start the thread, once; a second call does nothing."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="retention-sweep", daemon=True)
        self._thread.start()

    def wait_first(self, timeout: float) -> bool:
        """Wait up to `timeout` seconds for the first sweep to finish; -> whether it has."""
        return self._first_done.wait(timeout)

    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def stop(self, timeout: float = STOP_WAIT_S) -> None:
        """Wake the thread, which then ends, and wait up to `timeout` seconds for it (a sweep in progress is finished first). Safe to repeat."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def _run(self) -> None:
        first = True
        while not self._stop.is_set():
            purged = self.sweep_once()
            self.sweeps += 1
            if self.report is not None:
                try:
                    self.report(purged, first)
                except Exception as exc:   # a line that could not be printed must not end the thread
                    log.warning("retention sweep report failed (%s)", type(exc).__name__)
            first = False
            self._first_done.set()
            if self._stop.wait(self.interval_s):
                return
