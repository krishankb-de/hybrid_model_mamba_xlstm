"""Mamba-3 mixer (MAMBA3_PLAN.md M2-C).

A new layer type rather than a rewrite of `MambaBlock`. `MambaBlock` stays byte-identical so
every existing checkpoint keeps loading and the A0 control arm stays reproducible; this block is
selected by putting `"mamba3"` in `layer_pattern`.

With every feature flag off, this is **exactly Mamba-2 SSD**: scalar-per-head `A`, multi-value
`B`/`C` shared across heads, a short depthwise convolution, and the chunked scan in
`kernels/ssd`. The Mamba-3 additions (arXiv:2603.15569) each sit behind a flag that reduces to
that baseline, so every ablation arm differs from its predecessor in exactly one variable:

    use_trapezoid   Sec 3.1  exponential-trapezoidal discretization      (M3)
    use_rope        Sec 3.2  complex state via data-dependent rotation   (M4)
    bc_bias         Sec 3.4  learnable head-wise B/C biases              (M5)
    use_conv=False  Sec 4.2  drop the short convolution                  (M5)
    mimo_rank > 1   Sec 3.3  MIMO -- plumbed, never run (decision 3)

Two deliberate departures from stock Mamba-2, both recorded so they are not mistaken for bugs:

* **No post-gate RMSNorm.** Mamba-2 normalizes before the output projection; Mamba-3 removes it
  because BCNorm stabilizes training on its own (Sec 3.4). This repo's `MambaBlock` has never had
  one either, so `use_outproj_norm` defaults False and continuity is preserved. The paper reports
  it helps *hybrid* models' length generalization (Table 4), which this model is -- so it is
  exposed as a flag and is a legitimate arm.
* **The projection is sized for every flag at once.** The `trap`, `dd_A` and rotation-angle
  slices of `in_proj` exist whether or not their flags are on. That costs ~0.1% of the block and
  buys two things worth more: parameter count is identical across arms A2..A6, so a quality
  difference cannot be a capacity difference; and turning a flag off is exactly "zero that slice",
  which makes the bit-identity controls trivial to assert.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hybrid_xmamba.kernels.ssd import ssd_chunked_scan
from hybrid_xmamba.layers.normalization import RMSNorm


def heavy_tail_activation(x: torch.Tensor) -> torch.Tensor:
    """`1 + x` for `x >= 0`, `1 / (1 - x)` otherwise -- the reference's map onto (0, inf).

    Used for the data-dependent `A`. It grows linearly rather than exponentially on the positive
    side, which the reference implementation notes "improves stability during WSD training and at
    higher learning rates" -- relevant here, since this repo's 150M Stage-0 is spike-fragile
    enough that one gradient spike at step 24749 irreversibly collapsed a healthy run.
    """
    return torch.where(x >= 0, 1.0 + x, 1.0 / (1.0 - x))


class Mamba3Block(nn.Module):
    """Mamba-3 sequence mixer. Flags all-off == Mamba-2 SSD.

    Args:
        dim: model dimension.
        d_state: SSM state size N. 128 is the reference default and this project's choice --
            8x this repo's Mamba-1 setting for +0.24% parameters, because B/C are shared across
            heads (see MAMBA3_PLAN.md Context 3).
        head_dim: SSM head dimension P; `nheads = dim * expand_factor / head_dim`.
        expand_factor: inner-dimension expansion.
        ngroups: number of B/C groups. 1 means every head shares one B and C (Mamba's
            multi-value-attention structure). Raising it leaves the parameter-matched regime.
        chunk_size: SSD chunk length.
        use_conv / conv_size: short depthwise causal convolution over the concatenated x, B, C.
        use_trapezoid, use_rope, bc_bias, mimo_rank: the Mamba-3 features listed above.
        a_mode: "static" learns one `A` per head (Mamba-2); "data_dependent" projects it per
            token (Mamba-3's default; the paper reports the two perform similarly).
        dt_limit: hard clamp on Delta. `MambaBlock` normalized Delta with an RMSNorm, which was
            an accidental stabilizer; without it softplus is unbounded above, so this replaces
            the guardrail deliberately rather than by accident.
    """

    # Declares that this mixer honours packed-document resets, so `HybridBlock` can dispatch on a
    # capability rather than on a hard-coded tuple of layer names (MAMBA3_PLAN.md M2-E).
    supports_cu_seqlens = True

    def __init__(
        self,
        dim: int,
        d_state: int = 128,
        head_dim: int = 64,
        expand_factor: int = 2,
        ngroups: int = 1,
        chunk_size: int = 64,
        use_conv: bool = True,
        conv_size: int = 4,
        use_trapezoid: bool = False,
        use_rope: bool = False,
        rope_fraction: float = 0.5,
        bc_bias: str = "none",
        mimo_rank: int = 1,
        a_mode: str = "static",
        a_floor: float = 1e-4,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        dt_init_floor: float = 1e-4,
        dt_limit: float = 1.0,
        use_outproj_norm: bool = False,
        use_hybrid_norm: bool = False,  # accepted for interface parity; BCNorm is unconditional
        **unused,
    ):
        super().__init__()
        if bc_bias not in ("none", "zero_init", "one_init"):
            raise ValueError(
                f"bc_bias must be 'none', 'zero_init' or 'one_init', got {bc_bias!r}"
            )
        if a_mode not in ("static", "data_dependent"):
            raise ValueError(f"a_mode must be 'static' or 'data_dependent', got {a_mode!r}")
        if rope_fraction not in (0.5, 1.0):
            raise ValueError(f"rope_fraction must be 0.5 or 1.0, got {rope_fraction!r}")
        if mimo_rank != 1:
            raise NotImplementedError(
                "MIMO is plumbed but deliberately not implemented (MAMBA3_PLAN.md decision 3): "
                "rank 4 costs +3.2% parameters, leaving the parameter-matched regime, and its "
                "payoff is decode arithmetic intensity that this project cannot measure yet."
            )

        self.dim = dim
        self.d_state = d_state
        self.head_dim = head_dim
        self.expand_factor = expand_factor
        self.inner_dim = dim * expand_factor
        if self.inner_dim % head_dim != 0:
            raise ValueError(f"inner_dim ({self.inner_dim}) must be divisible by head_dim")
        self.nheads = self.inner_dim // head_dim
        if self.nheads % ngroups != 0:
            raise ValueError(f"nheads ({self.nheads}) must be divisible by ngroups ({ngroups})")
        self.ngroups = ngroups
        self.chunk_size = chunk_size
        self.use_conv = use_conv
        self.conv_size = conv_size
        self.use_trapezoid = use_trapezoid
        self.use_rope = use_rope
        self.rope_fraction = rope_fraction
        self.bc_bias = bc_bias
        self.mimo_rank = mimo_rank
        self.a_mode = a_mode
        self.a_floor = a_floor
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init_floor = dt_init_floor
        self.dt_limit = dt_limit

        # Rotation angles are shared across heads and cover half of `d_state` by default, so each
        # angle drives one 2-D rotation pair (Prop. 2's block-diagonal R).
        self.n_rope_angles = int(d_state * rope_fraction) // 2 * ngroups

        self.bc_dim = d_state * ngroups
        d_in_proj = (
            2 * self.inner_dim          # z (gate) and x (SSM input)
            + 2 * self.bc_dim           # B and C
            + 3 * self.nheads           # dt, A, trapezoid lambda
            + self.n_rope_angles
        )
        self.in_proj = nn.Linear(dim, d_in_proj, bias=False)
        self._split = [
            self.inner_dim, self.inner_dim, self.bc_dim, self.bc_dim,
            self.nheads, self.nheads, self.nheads, self.n_rope_angles,
        ]

        if use_conv:
            conv_channels = self.inner_dim + 2 * self.bc_dim
            self.conv1d = nn.Conv1d(
                conv_channels, conv_channels, kernel_size=conv_size,
                groups=conv_channels, padding=conv_size - 1, bias=True,
            )
        else:
            self.conv1d = None

        # BCNorm (paper Sec 3.4, "BCNorm"/"QKNorm"). Unconditional -- the paper treats it as part
        # of the layer, and this repo's canonical config already applies the same norms to its
        # Mamba-1 B/C, so it is not a new degree of freedom relative to the baseline.
        self.B_norm = RMSNorm(d_state)
        self.C_norm = RMSNorm(d_state)
        if bc_bias == "none":
            self.B_bias = None
            self.C_bias = None
        else:
            init = 0.0 if bc_bias == "zero_init" else 1.0
            self.B_bias = nn.Parameter(torch.full((self.nheads, d_state), init))
            self.C_bias = nn.Parameter(torch.full((self.nheads, d_state), init))

        self.A_log = nn.Parameter(torch.empty(self.nheads))
        self.dt_bias = nn.Parameter(torch.empty(self.nheads))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.out_norm = RMSNorm(self.inner_dim) if use_outproj_norm else None
        self.out_proj = nn.Linear(self.inner_dim, dim, bias=False)
        self.activation = nn.SiLU()
        self.post_model_init()

    def post_model_init(self) -> None:
        """(Re-)apply the SSM initializations after the parent model's global weight pass.

        `HybridLanguageModel.__init__` ends with `self.apply(self._init_weights)`, which zeroes
        every `nn.Linear` bias and re-normal-inits every weight. `dt_bias` and `A_log` are bare
        Parameters so they survive that, but this hook is called explicitly afterwards anyway --
        the Mamba-1 block lost its dt init to exactly this ordering (MAMBA3_PLAN.md M1-F), and
        relying on "Parameters happen not to be touched" is how that recurs.
        """
        with torch.no_grad():
            # Delta ~ logU[dt_min, dt_max], inverted through softplus.
            dt = torch.exp(
                torch.rand(self.nheads) * (math.log(self.dt_max) - math.log(self.dt_min))
                + math.log(self.dt_min)
            ).clamp(min=self.dt_init_floor)
            self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
            # A ~ U[1, 16] per head, stored in log space; A = -exp(A_log) is strictly negative.
            self.A_log.copy_(torch.log(torch.rand(self.nheads) * 15.0 + 1.0))

    def _compute_a(self, a_raw: torch.Tensor) -> torch.Tensor:
        """Per-head state transition, strictly negative. Shape (nheads,) or (batch, len, nheads)."""
        if self.a_mode == "static":
            return -torch.exp(self.A_log.float())
        return torch.clamp(-heavy_tail_activation(a_raw.float()), max=-self.a_floor)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """(batch, seqlen, dim) -> (batch, seqlen, dim).

        Argument order is load-bearing: `hybrid_lm.py` passes these positionally through
        `torch.utils.checkpoint.checkpoint`.
        """
        batch, seqlen, _ = x.shape
        proj = self.in_proj(x)
        z, xs, B, C, dt_raw, a_raw, trap_raw, angles = torch.split(proj, self._split, dim=-1)

        if self.conv1d is not None:
            xbc = torch.cat([xs, B, C], dim=-1).transpose(1, 2)
            xbc = self.conv1d(xbc)[..., :seqlen].transpose(1, 2)
            xbc = self.activation(xbc)
            if cu_seqlens is not None:
                xbc = self._mask_conv_across_documents(
                    torch.cat([xs, B, C], dim=-1), xbc, cu_seqlens
                )
            xs, B, C = torch.split(xbc, [self.inner_dim, self.bc_dim, self.bc_dim], dim=-1)

        dt = F.softplus(dt_raw.float() + self.dt_bias)
        if self.dt_limit is not None:
            # Replaces the RMSNorm that used to bound Delta by accident. Without it softplus is
            # unbounded above and gamma = lambda * Delta scales the state-input without limit --
            # a direct route to the gradient spikes this project has already lost runs to.
            dt = dt.clamp(max=self.dt_limit)
        A = self._compute_a(a_raw)

        B = self.B_norm(B.view(batch, seqlen, self.ngroups, self.d_state))
        C = self.C_norm(C.view(batch, seqlen, self.ngroups, self.d_state))

        xs = xs.view(batch, seqlen, self.nheads, self.head_dim)
        y = self._scan(xs, dt, A, B, C, trap_raw, angles, cu_seqlens)

        y = y.reshape(batch, seqlen, self.inner_dim)
        if self.out_norm is not None:
            y = self.out_norm(y)
        return self.out_proj(y * self.activation(z))

    def _scan(self, xs, dt, A, B, C, trap_raw, angles, cu_seqlens):
        """SSD scan. M3 adds the trapezoidal term and M4 the rotation; both are no-ops here."""
        del trap_raw, angles  # consumed by M3 / M4
        if self.B_bias is not None:
            B = B.repeat_interleave(self.nheads // self.ngroups, dim=2) + self.B_bias
            C = C.repeat_interleave(self.nheads // self.ngroups, dim=2) + self.C_bias
        if A.dim() == 1:
            return ssd_chunked_scan(
                xs, dt, A, B, C, self.D.float(),
                chunk_size=self.chunk_size, cu_seqlens=cu_seqlens,
            )
        raise NotImplementedError("data-dependent A requires the M3 scan signature")

    def _mask_conv_across_documents(self, raw, conved, cu_seqlens):
        """Recompute the first `conv_size - 1` positions of each document with a zeroed history.

        A causal depthwise convolution of width k lets each document see the last k-1 tokens of
        the previous one. Only those k-1 positions are wrong, so the fix is a masked re-run of the
        window rather than the per-segment Python loop `MambaBlock` uses.
        """
        k = self.conv_size
        if k <= 1:
            return conved
        batch, seqlen, _ = raw.shape
        pos = torch.arange(seqlen, device=raw.device)
        windows = F.pad(raw.transpose(1, 2), (k - 1, 0)).unfold(-1, k, 1)   # (b, ch, L, k)
        offsets = torch.arange(k - 1, -1, -1, device=raw.device)            # k-1 .. 0 steps back
        src = (pos.view(-1, 1) - offsets.view(1, -1)).clamp(min=0)          # (L, k)
        same = cu_seqlens.gather(1, src.reshape(1, -1).expand(batch, -1)).view(batch, seqlen, k)
        same = (same == cu_seqlens.unsqueeze(-1)) & (
            (pos.view(1, -1, 1) - offsets.view(1, 1, -1)) >= 0
        )
        w = self.conv1d.weight.view(1, -1, 1, k)
        masked = (windows * same.unsqueeze(1) * w).sum(-1) + self.conv1d.bias.view(1, -1, 1)
        masked = self.activation(masked.transpose(1, 2))
        # Only the first k-1 positions of each document differ; splice those in.
        starts = torch.zeros_like(cu_seqlens, dtype=torch.bool)
        starts[:, 1:] = cu_seqlens[:, 1:] != cu_seqlens[:, :-1]
        starts[:, 0] = True
        near_start = torch.zeros_like(starts)
        for shift in range(k - 1):
            near_start[:, shift:] |= starts[:, : seqlen - shift] if shift else starts
        return torch.where(near_start.unsqueeze(-1), masked, conved)

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, d_state={self.d_state}, head_dim={self.head_dim}, "
            f"nheads={self.nheads}, ngroups={self.ngroups}, conv={self.use_conv}, "
            f"trapezoid={self.use_trapezoid}, rope={self.use_rope}, bc_bias={self.bc_bias}, "
            f"a_mode={self.a_mode}"
        )
