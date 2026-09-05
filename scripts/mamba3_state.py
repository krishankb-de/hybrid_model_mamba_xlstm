#!/usr/bin/env python3
"""Keep MAMBA3_PLAN.md checkboxes and mamba3_state.json in sync.

The plan-of-record contract (MAMBA3_PLAN.md, "State-tracking contract") requires ticking a
checkbox AND updating the state file after every meaningful change. Doing that by hand twice
is how a plan and its state drift apart, so this is the single entry point.

    python scripts/mamba3_state.py tick M0-A [M0-B ...] [--note "..."] [--evidence k=v ...]
    python scripts/mamba3_state.py note "..."
    python scripts/mamba3_state.py phase M1_pin_the_defect [--status "..."]
    python scripts/mamba3_state.py show [M0]
    python scripts/mamba3_state.py sync          # regenerate state phases from plan checkboxes

`sync` is the documented recovery path: if mamba3_state.json is lost, the plan's checkboxes are
ground truth and this rebuilds the phase tree from them.
"""
import argparse
import datetime
import json
import pathlib
import re
import sys
from typing import Dict, List, Optional, Tuple

ROOT = pathlib.Path(__file__).resolve().parent.parent
PLAN = ROOT / "MAMBA3_PLAN.md"
STATE = ROOT / "mamba3_state.json"

CHECKBOX_RE = re.compile(r"^(- \[)( |x)(\] \*\*)(M\d+-[A-Z])(\*\*\s+)(.*)$")
PHASE_RE = re.compile(r"^### (M\d+)\s+—\s+(.*)$")


def _now() -> str:
    return datetime.datetime.now().astimezone().replace(microsecond=0).isoformat()


def _clean(s: str, limit: int = 160) -> str:
    s = re.sub(r"\*\*|`", "", s).strip()
    return s[: limit - 3] + "..." if len(s) > limit else s


def parse_plan() -> Tuple[List[str], Dict[str, dict]]:
    """Return (phase order, {phase_id: {title, checkboxes: {id: {done, desc}}}})."""
    phases: Dict[str, dict] = {}
    order: List[str] = []
    cur: Optional[str] = None
    for line in PLAN.read_text().splitlines():
        m = PHASE_RE.match(line)
        if m:
            cur = m.group(1)
            order.append(cur)
            phases[cur] = {"title": _clean(re.sub(r"\(\*\*.*?\*\*\)", "", m.group(2))), "checkboxes": {}}
            continue
        c = CHECKBOX_RE.match(line)
        if c and cur:
            phases[cur]["checkboxes"][c.group(4)] = {"done": c.group(2) == "x", "desc": _clean(c.group(6))}
    return order, phases


def load_state() -> dict:
    return json.loads(STATE.read_text())


def save_state(state: dict) -> None:
    state["last_updated"] = _now()
    STATE.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")


def set_checkboxes(ids: List[str], done: bool) -> List[str]:
    """Flip checkboxes in the plan markdown. Returns the ids actually changed."""
    text = PLAN.read_text()
    out, changed = [], []
    for line in text.splitlines(keepends=True):
        c = CHECKBOX_RE.match(line.rstrip("\n"))
        if c and c.group(4) in ids:
            if (c.group(2) == "x") != done:
                changed.append(c.group(4))
            line = f"{c.group(1)}{'x' if done else ' '}{c.group(3)}{c.group(4)}{c.group(5)}{c.group(6)}\n"
        out.append(line)
    PLAN.write_text("".join(out))
    return changed


def refresh_phases(state: dict) -> dict:
    """Rebuild state['phases'] from the plan, preserving evidence/verdict."""
    order, phases = parse_plan()
    prev = state.get("phases", {})
    merged = {}
    for pid in order:
        old = prev.get(pid, {})
        boxes = phases[pid]["checkboxes"]
        n_done = sum(1 for b in boxes.values() if b["done"])
        if n_done == 0:
            status = "pending"
        elif n_done == len(boxes):
            status = "complete"
        else:
            status = "in_progress"
        merged[pid] = {
            "title": phases[pid]["title"],
            "status": status,
            "checkboxes": boxes,
            "evidence": old.get("evidence", {}),
            "verdict": old.get("verdict"),
        }
    state["phase_order"] = order
    state["phases"] = merged
    return state


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("tick", help="mark checkbox(es) done in plan + state")
    t.add_argument("ids", nargs="+")
    t.add_argument("--note", default=None)
    t.add_argument("--evidence", nargs="*", default=[], metavar="KEY=VALUE")
    t.add_argument("--undo", action="store_true", help="untick instead")

    n = sub.add_parser("note", help="append a one-line note")
    n.add_argument("text")

    p = sub.add_parser("phase", help="set current_phase")
    p.add_argument("name")
    p.add_argument("--status", default=None)

    s = sub.add_parser("show", help="show progress")
    s.add_argument("phase", nargs="?", default=None)

    sub.add_parser("sync", help="regenerate state phases from plan checkboxes")

    args = ap.parse_args()
    state = load_state()

    if args.cmd == "tick":
        _, plan_phases = parse_plan()
        known = {cid for ph in plan_phases.values() for cid in ph["checkboxes"]}
        unknown = [i for i in args.ids if i not in known]
        if unknown:
            print(f"ERROR: unknown checkbox id(s): {', '.join(unknown)}", file=sys.stderr)
            return 1
        changed = set_checkboxes(args.ids, done=not args.undo)
        state = refresh_phases(state)
        if args.evidence:
            for item in args.evidence:
                if "=" not in item:
                    print(f"ERROR: --evidence expects KEY=VALUE, got {item!r}", file=sys.stderr)
                    return 1
                k, v = item.split("=", 1)
                pid = args.ids[0].split("-")[0]
                state["phases"][pid]["evidence"][k] = v
        verb = "unticked" if args.undo else "ticked"
        state.setdefault("notes", []).append(
            args.note or f"{datetime.date.today().isoformat()}: {verb} {', '.join(args.ids)}"
        )
        save_state(state)
        print(f"{verb}: {', '.join(args.ids)}" + (f"  (no-op for {set(args.ids) - set(changed)})" if len(changed) != len(args.ids) else ""))

    elif args.cmd == "note":
        state.setdefault("notes", []).append(f"{datetime.date.today().isoformat()}: {args.text}")
        save_state(state)
        print("note appended")

    elif args.cmd == "phase":
        state["current_phase"] = args.name
        if args.status:
            state["status"] = args.status
        state.setdefault("notes", []).append(f"{datetime.date.today().isoformat()}: current_phase -> {args.name}")
        save_state(state)
        print(f"current_phase = {args.name}")

    elif args.cmd == "sync":
        state = refresh_phases(state)
        save_state(state)
        print("state phases regenerated from MAMBA3_PLAN.md checkboxes")

    elif args.cmd == "show":
        state = refresh_phases(state)
        print(f"current_phase : {state['current_phase']}")
        print(f"status        : {state['status']}")
        print(f"last_updated  : {state['last_updated']}\n")
        for pid in state["phase_order"]:
            ph = state["phases"][pid]
            if args.phase and pid != args.phase:
                continue
            boxes = ph["checkboxes"]
            done = sum(1 for b in boxes.values() if b["done"])
            mark = {"complete": "x", "in_progress": "~", "pending": " "}[ph["status"]]
            print(f"[{mark}] {pid:<4} {done}/{len(boxes):<3} {ph['title']}")
            if args.phase:
                for cid, b in boxes.items():
                    print(f"      [{'x' if b['done'] else ' '}] {cid}  {b['desc']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
