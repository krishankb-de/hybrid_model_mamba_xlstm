"""Run-metadata snapshot for reproducibility.

Writes ``<output_dir>/run_metadata.json`` at the start of every training run
so a future debugger (or future Claude session) can reconstruct what produced
a given checkpoint without relying on branch names.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf


def _git(args: List[str]) -> str:
    try:
        out = subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return ""


def write_run_metadata(cfg: DictConfig, output_dir: str, extra: Optional[Dict[str, Any]] = None) -> Path:
    """Persist a run-metadata snapshot. Returns the path written."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git(["rev-parse", "HEAD"]),
        "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(_git(["status", "--porcelain"])),
        "python_version": sys.version.split()[0],
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "resolved_config": OmegaConf.to_container(cfg, resolve=True),
    }
    if extra:
        metadata.update(extra)
    path = out_dir / "run_metadata.json"
    with path.open("w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"[run_metadata] wrote {path}")
    return path
