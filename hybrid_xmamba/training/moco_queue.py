"""MoCo-style dynamic queue and momentum encoder for Phase 5.

Decouples the InfoNCE negative pool from batch size.
With K=16384, every CLIP step has 16384 negatives instead of 31.
"""

import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoCoQueue(nn.Module):
    """FIFO ring-buffer of L2-normalised embeddings used as InfoNCE negatives.

    Keys are pushed one batch at a time (oldest evicted when full).
    The buffer is registered as a non-parameter buffer so it persists across
    Lightning checkpoints automatically.

    Args:
        dim: embedding dimension (512 for BiomedCLIP joint space)
        K: queue capacity (number of stored negative keys)
    """

    def __init__(self, dim: int = 512, K: int = 16384) -> None:
        super().__init__()
        self.K = K
        self.dim = dim
        # Initialise with random unit vectors so the queue is valid before warmup
        queue = F.normalize(torch.randn(dim, K), dim=0)
        self.register_buffer("queue", queue)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def enqueue(self, keys: torch.Tensor) -> None:
        """Push a batch of L2-normalised keys into the queue.

        Args:
            keys: (B, dim) — must already be L2-normalised.
        """
        B = keys.shape[0]
        ptr = int(self.queue_ptr)

        # Wrap-around: if the batch overflows the end of the buffer, split it
        if ptr + B <= self.K:
            self.queue[:, ptr: ptr + B] = keys.T
        else:
            tail = self.K - ptr
            self.queue[:, ptr:] = keys[:tail].T
            self.queue[:, : B - tail] = keys[tail:].T

        self.queue_ptr[0] = (ptr + B) % self.K

    def all_keys(self) -> torch.Tensor:
        """Return a (K, dim) copy of the current queue (detached)."""
        return self.queue.T.detach()

    @torch.no_grad()
    def reset(self) -> None:
        """Re-initialise the queue with random unit vectors and zero the pointer.

        Phase 9: called at the unfreeze step to discard pre-warmup stale keys
        (still in GPT-2 space) so that post-unfreeze InfoNCE negatives are
        gradually replaced with BCT-aligned text keys.
        """
        new_q = F.normalize(torch.randn(self.dim, self.K, device=self.queue.device,
                                         dtype=self.queue.dtype), dim=0)
        self.queue.copy_(new_q)
        self.queue_ptr.zero_()


class MomentumEncoder(nn.Module):
    """Exponential moving average (EMA) copy of a query encoder.

    Parameters are never updated by the optimiser — only by the EMA rule:
        θ_k ← m·θ_k + (1−m)·θ_q

    Args:
        query_encoder: the trainable module to track
        m: EMA momentum (0.999 is MoCo default; higher = slower target drift)
    """

    def __init__(self, query_encoder: nn.Module, m: float = 0.999) -> None:
        super().__init__()
        self.m = m
        self.encoder = copy.deepcopy(query_encoder)
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, query_encoder: nn.Module) -> None:
        """Pull the query encoder's current weights toward the EMA copy."""
        for p_k, p_q in zip(self.encoder.parameters(), query_encoder.parameters()):
            p_k.data.mul_(self.m).add_((1.0 - self.m) * p_q.data)

    @torch.no_grad()
    def copy_from(self, query_encoder: nn.Module) -> None:
        """Hard-resync momentum encoder weights from the live model.

        Phase 9: at the unfreeze step the live model has just been moved into
        BiomedCLIP-text space by the KD warmup; the EMA copy is still close
        to the original (Stage-0) weights. A hard copy avoids feeding the
        queue stale GPT-2-space keys for ~1/(1-m) ≈ 1000 EMA steps.
        Buffers (e.g. running stats) are also re-copied for completeness.
        """
        for p_k, p_q in zip(self.encoder.parameters(), query_encoder.parameters()):
            p_k.data.copy_(p_q.data)
        for b_k, b_q in zip(self.encoder.buffers(), query_encoder.buffers()):
            b_k.data.copy_(b_q.data)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward through the momentum encoder; returns L2-normalised (B, dim)."""
        with torch.no_grad():
            z = self.encoder.encode(input_ids, attention_mask=attention_mask)
        return F.normalize(z.float(), dim=-1)
