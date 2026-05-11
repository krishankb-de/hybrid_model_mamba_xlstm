"""Learning-rate schedulers for hybrid models.

Implements the Warmup-Stable-Decay (WSD) schedule from the PDF roadmap
(gap 5): linear warmup (1% of max_steps), constant stable phase (85%),
and inverse-square-root-style decay (14%) via factor = 1 - sqrt(p).

Also exposes a β2 schedule helper that linearly anneals AdamW's β2 from
0.999 → 0.974 during the decay phase, to be applied externally by the
training loop on each step.
"""

import math
from typing import Tuple

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def wsd_factor(
    step: int,
    warmup_steps: int,
    stable_steps: int,
    decay_steps: int,
    min_lr_ratio: float = 0.0,
    warmup_start_factor: float = 0.01,
) -> float:
    """Compute the WSD LR multiplier at ``step``.

    Phases:
      * step < warmup_steps                : linear from ``warmup_start_factor`` → 1.0
      * warmup_steps ≤ step < decay_start  : constant 1.0
      * decay phase (length ``decay_steps``): 1 - sqrt(p), clamped at ``min_lr_ratio``
    """
    if warmup_steps > 0 and step < warmup_steps:
        return warmup_start_factor + (1.0 - warmup_start_factor) * (step / warmup_steps)
    decay_start = warmup_steps + stable_steps
    if step < decay_start:
        return 1.0
    if decay_steps <= 0:
        return min_lr_ratio
    p = min(1.0, (step - decay_start) / decay_steps)
    return max(min_lr_ratio, 1.0 - math.sqrt(p))


class WSDScheduler(LambdaLR):
    """Warmup-Stable-Decay LR scheduler (LambdaLR-based).

    Ratios default to the plan-of-record (1% / 85% / 14%) and must sum to 1.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        max_steps: int,
        warmup_ratio: float = 0.01,
        stable_ratio: float = 0.85,
        decay_ratio: float = 0.14,
        min_lr_ratio: float = 0.0,
        warmup_start_factor: float = 0.01,
        last_epoch: int = -1,
    ):
        total = warmup_ratio + stable_ratio + decay_ratio
        assert abs(total - 1.0) < 1e-6, (
            "WSD ratios must sum to 1.0, got %r" % total
        )
        assert max_steps > 0, "max_steps must be positive"

        self.max_steps = max_steps
        self.warmup_steps = int(round(max_steps * warmup_ratio))
        self.stable_steps = int(round(max_steps * stable_ratio))
        # Absorb rounding drift into the decay phase
        self.decay_steps = max(0, max_steps - self.warmup_steps - self.stable_steps)
        self.decay_start = self.warmup_steps + self.stable_steps
        self.min_lr_ratio = min_lr_ratio
        self.warmup_start_factor = warmup_start_factor

        def _lambda(step: int) -> float:
            return wsd_factor(
                step,
                self.warmup_steps,
                self.stable_steps,
                self.decay_steps,
                min_lr_ratio=self.min_lr_ratio,
                warmup_start_factor=self.warmup_start_factor,
            )

        super().__init__(optimizer, lr_lambda=_lambda, last_epoch=last_epoch)


def beta2_for_step(
    step: int,
    decay_start: int,
    decay_steps: int,
    beta2_start: float = 0.999,
    beta2_end: float = 0.974,
) -> float:
    """Linear β2 anneal across the decay phase. Constant at ``beta2_start`` before."""
    if decay_steps <= 0 or step < decay_start:
        return beta2_start
    p = min(1.0, (step - decay_start) / decay_steps)
    return beta2_start + (beta2_end - beta2_start) * p


def apply_beta2_schedule(
    optimizer: Optimizer,
    step: int,
    decay_start: int,
    decay_steps: int,
    beta2_start: float = 0.999,
    beta2_end: float = 0.974,
) -> float:
    """Update optimizer param-group ``betas`` in place to the scheduled β2.

    Returns the β2 value that was applied (for logging).
    """
    b2 = beta2_for_step(step, decay_start, decay_steps, beta2_start, beta2_end)
    for pg in optimizer.param_groups:
        if "betas" in pg:
            b1, _ = pg["betas"]
            pg["betas"] = (b1, b2)
    return b2
