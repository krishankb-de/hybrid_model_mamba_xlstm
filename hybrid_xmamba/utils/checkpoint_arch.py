"""Recover a HybridLanguageModel's architecture from a checkpoint state dict (MAMBA3_PLAN_V2.md V1-D).

The eval loaders used to decide ``mamba`` vs ``mlstm`` per layer from ``"A_log" in k or "conv1d" in k``.
A Mamba-3 block has both, so it was labelled Mamba-1 and rebuilt with the wrong mixer; an attention
block has neither, so it was labelled mLSTM. Here every mixer family is identified by ONE parameter
that only it owns, read as the first name after ``mixer.``, and a layer that matches zero or two
families is refused rather than guessed.

What this can and cannot recover:
  * layer types, norm topology and the tensor-shaped sizes (state sizes, conv widths, expand
    factor, dt_rank, Mamba-3 head_dim) -- yes, from shapes.
  * parameter-invisible flags (``scan_impl``, ``tfla_impl``, ``dt_init_strategy``, ``prefix_k``)
    -- no. Every checkpoint trained before 2026-09 used legacy/legacy/none, and the report-gen eval
    resolves ``prefix_k`` from ``run_metadata.json`` (``evaluate_report_generation.resolve_prefix_k``).
    A loader that must evaluate a corrected-operator checkpoint has to pin those from its yaml.
"""
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# (layer type, parameter names -- the FIRST segment after "mixer." -- that only this mixer owns).
FINGERPRINTS: Sequence[Tuple[str, Tuple[str, ...]]] = (
    ("mamba3", ("dt_bias", "B_bias")),   # Mamba3Block: per-head dt_bias; B_bias when bc_bias != none
    ("mamba", ("dt_proj",)),             # MambaBlock: the Mamba-1 dt projection
    ("mlstm", ("i_gate_proj",)),
    ("slstm", ("gate_proj",)),
    ("attention", ("qkv_proj",)),
)


@dataclass
class InferredArchitecture:
    layer_pattern: List[str]
    norm_topology: str
    num_layers: int
    state_size: Optional[int] = None         # Mamba-1 d_state (A_log.shape[1])
    conv_size: Optional[int] = None          # Mamba-1 conv width
    expand_factor: Optional[int] = None      # Mamba-1 d_inner / dim
    dt_rank: Optional[int] = None            # Mamba-1 x_proj rows minus 2 * state_size
    mamba3_d_state: Optional[int] = None     # Mamba3Block B_norm width
    mamba3_head_dim: Optional[int] = None    # Mamba3Block d_inner / nheads
    mamba3_conv_size: Optional[int] = None
    mamba3_use_conv: Optional[bool] = None

    def size_kwargs(self) -> Dict[str, Any]:
        """Only the tensor-derived sizes, for callers that already set the pattern themselves."""
        out = {}
        for name in ("state_size", "conv_size", "expand_factor", "dt_rank",
                     "mamba3_d_state", "mamba3_head_dim", "mamba3_conv_size", "mamba3_use_conv"):
            v = getattr(self, name)
            if v is not None:
                out[name] = v
        return out

    def config_kwargs(self) -> Dict[str, Any]:
        out = {"layer_pattern": list(self.layer_pattern), "norm_topology": self.norm_topology,
               "num_layers": self.num_layers}
        out.update(self.size_kwargs())
        return out


def _mixer_params(state: Mapping[str, Any], prefix: str) -> Dict[int, Dict[str, Any]]:
    """{layer index: {first-segment name: tensor}} for every ``<prefix><i>.mixer.*`` key."""
    pat = re.compile(r"^" + re.escape(prefix) + r"(\d+)\.mixer\.(.+)$")
    layers: Dict[int, Dict[str, Any]] = {}
    for k, v in state.items():
        m = pat.match(k)
        if not m:
            continue
        idx, rest = int(m.group(1)), m.group(2)
        layers.setdefault(idx, {})
        # keep the FULL sub-key too, so shapes can be read below
        layers[idx][rest] = v
    return layers


def _first_segments(params: Mapping[str, Any]) -> set:
    return {k.split(".")[0] for k in params}


def infer_layer_types(state: Mapping[str, Any], prefix: str = "lm.layers.") -> List[str]:
    layers = _mixer_params(state, prefix)
    if not layers:
        raise ValueError(f"no '{prefix}<i>.mixer.*' keys found; first keys: {list(state)[:5]}")
    num_layers = max(layers) + 1
    pattern: List[str] = []
    for i in range(num_layers):
        names = _first_segments(layers.get(i, {}))
        hits = [t for t, owned in FINGERPRINTS if any(n in names for n in owned)]
        if len(hits) != 1:
            kind = "ambiguous" if hits else "no fingerprint"
            raise ValueError(
                f"layer {i}: {kind} mixer fingerprint {hits} among parameters {sorted(names)}; "
                f"known fingerprints: {[(t, o) for t, o in FINGERPRINTS]}")
        pattern.append(hits[0])
    return pattern


def infer_architecture(state: Mapping[str, Any], prefix: str = "lm.layers.") -> InferredArchitecture:
    layers = _mixer_params(state, prefix)
    pattern = infer_layer_types(state, prefix)
    names_of = {i: _first_segments(layers.get(i, {})) for i in range(len(pattern))}

    # norm topology. Mamba3Block normalises B/C UNCONDITIONALLY (BCNorm is part of the block), so
    # its B_norm says nothing about the topology -- only Mamba-1 layers can tell hybrid from
    # hybrid_bc, and only the presence of some HybridNorm parameter tells hybrid from pre_rms.
    mamba1 = [i for i, t in enumerate(pattern) if t == "mamba"]
    if any("dt_norm" in names_of[i] for i in mamba1):
        topo = "hybrid"
    elif any("B_norm" in names_of[i] for i in mamba1):
        topo = "hybrid_bc"
    elif any("v_norm" in names_of[i] for i, t in enumerate(pattern) if t == "mlstm") or \
            any("q_norm" in names_of[i] for i, t in enumerate(pattern) if t == "attention"):
        topo = "hybrid"
    else:
        topo = "pre_rms"

    arch = InferredArchitecture(layer_pattern=pattern, norm_topology=topo, num_layers=len(pattern))

    def shape(i: int, sub: str):
        v = layers.get(i, {}).get(sub)
        return tuple(v.shape) if hasattr(v, "shape") else None

    for i, t in enumerate(pattern):
        if t == "mamba" and arch.state_size is None:
            a = shape(i, "A_log"); op = shape(i, "out_proj.weight"); cv = shape(i, "conv1d.weight")
            xp = shape(i, "x_proj.weight")
            if a:
                arch.state_size = int(a[1])
            if op:
                arch.expand_factor = int(op[1] // op[0])
            if cv:
                arch.conv_size = int(cv[-1])
            if xp and a:
                arch.dt_rank = int(xp[0] - 2 * a[1])
        elif t == "mamba3" and arch.mamba3_d_state is None:
            bn = shape(i, "B_norm.weight"); op = shape(i, "out_proj.weight"); db = shape(i, "dt_bias")
            cv = shape(i, "conv1d.weight")
            if bn:
                arch.mamba3_d_state = int(bn[0])
            if op and db:
                arch.mamba3_head_dim = int(op[1] // db[0])
            arch.mamba3_use_conv = cv is not None
            if cv:
                arch.mamba3_conv_size = int(cv[-1])
    return arch
