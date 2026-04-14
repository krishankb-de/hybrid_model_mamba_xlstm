"""SLURM-friendly callbacks for safe preemption on 36-48h Willi walltimes.

Saves a checkpoint on SIGTERM / SIGUSR1 / on_exception so a requeued job can
resume from near the kill point instead of from the last periodic checkpoint.
"""

import os
import signal
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class SignalCheckpointCallback(Callback):
    """Writes ``<checkpoint_dir>/interrupt.ckpt`` when the process is signalled.

    Handled signals: SIGTERM, SIGUSR1 (SLURM's default preemption signal).
    The callback is a no-op after the first fire — one save is enough.
    """

    def __init__(self, checkpoint_dir: str, filename: str = "interrupt.ckpt"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.filename = filename
        self._trainer: "pl.Trainer | None" = None
        self._saved = False

    def setup(self, trainer, pl_module, stage):
        self._trainer = trainer
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # Install handlers only in the main process
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            for sig in (signal.SIGTERM, signal.SIGUSR1):
                try:
                    signal.signal(sig, self._handle)
                except (ValueError, OSError):
                    # Not in the main thread (e.g. under some launchers); skip.
                    pass

    def on_exception(self, trainer, pl_module, exception):
        # Also fire on uncaught exceptions so we still capture weights.
        self._save("exception")

    def _handle(self, signum, frame):
        self._save(f"signal_{signum}")
        # Re-raise as SystemExit so Lightning unwinds cleanly and SLURM can requeue.
        raise SystemExit(0)

    def _save(self, reason: str):
        if self._saved or self._trainer is None:
            return
        self._saved = True
        target = self.checkpoint_dir / self.filename
        try:
            self._trainer.save_checkpoint(str(target))
            print(f"[SignalCheckpointCallback] Saved {target} ({reason}).")
        except Exception as e:  # checkpointing must not mask the original exit
            print(f"[SignalCheckpointCallback] Save failed on {reason}: {e}")
