"""Verify the Stage-0 -> Phase-6 (contrastive) checkpoint handoff.

Builds the REAL HybridTextEncoder with the 150M config and runs the exact load
logic from train_contrastive.py (:626-636), then reports missing/unexpected keys.
A clean handoff = 0 missing AND 0 unexpected on text_encoder.lm.

This is the static guard for the plan's recurring "silent fresh-load" bug
(strict=False hides a config mismatch -> backbone loads nothing -> wrong metrics).

CPU-only, ~seconds. Run in the venv on a compute/run node:
  python scripts/verify_stage0_handoff.py \
    [outputs/h100_stage0_150m_v2/checkpoints/stage0_model_only.pt] \
    [configs/model/hybrid_150m_v2.yaml]
"""
import sys
from pathlib import Path

# Make the repo root importable when run as `python scripts/verify_stage0_handoff.py`
# (Python puts scripts/ on sys.path, not the repo root; mirror train_stage0_distill.py:40).
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from omegaconf import OmegaConf
from hybrid_xmamba.models.configuration_hybrid import HybridConfig
from hybrid_xmamba.models.hybrid_lm import HybridTextEncoder

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/h100_stage0_150m_v2/checkpoints/stage0_model_only.pt"
model_yaml = sys.argv[2] if len(sys.argv) > 2 else "configs/model/hybrid_150m_v2.yaml"

m = OmegaConf.load(model_yaml)

# Mirror the HybridConfig construction in train_contrastive.py (:588-620) exactly,
# incl. the norm_topology threading at :609 (default pre_rms if absent).
cfg = HybridConfig(
    vocab_size=m.vocab_size, dim=m.dim, num_layers=m.num_layers,
    layer_pattern=list(m.layer_pattern), state_size=m.state_size, conv_size=m.conv_size,
    expand_factor=m.expand_factor, dt_rank=m.get("dt_rank", None), use_fast_path=m.use_fast_path,
    head_dim=m.head_dim, num_heads=m.num_heads, use_tfla=m.use_tfla, proj_factor=m.proj_factor,
    slstm_hidden_dim=m.slstm_hidden_dim, slstm_num_heads=m.slstm_num_heads,
    use_exponential_gate=m.use_exponential_gate, norm_type=m.norm_type,
    norm_topology=m.get("norm_topology", "pre_rms"),
    use_mlp=m.use_mlp, mlp_ratio=m.mlp_ratio,
    max_position_embeddings=m.max_position_embeddings, dropout=m.dropout,
    initializer_range=m.initializer_range, use_cache=m.use_cache,
    tie_word_embeddings=m.tie_word_embeddings,
    use_gradient_checkpointing=m.get("use_gradient_checkpointing", False),
    proj_head_dropout=m.get("proj_head_dropout", 0.1),
    pooling_strategy=m.get("pooling_strategy", "mean"),
)
print(f"config: norm_topology={cfg.norm_topology}  pooling={cfg.pooling_strategy}  "
      f"dim={cfg.dim}  layers={cfg.num_layers}  pattern={list(cfg.layer_pattern)}")

text_encoder = HybridTextEncoder(cfg, embed_dim=512)

# --- exact load path from train_contrastive.py ---
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
state = ckpt.get("state_dict", ckpt)
state = {k.replace("model.", "", 1): v for k, v in state.items()}
state = {(k[3:] if k.startswith("lm.") else k): v for k, v in state.items()}
missing, unexpected = text_encoder.lm.load_state_dict(state, strict=False)

print(f"\nckpt keys: {len(state)}   text_encoder.lm keys: {len(text_encoder.lm.state_dict())}")
print(f"MISSING   (expected by lm, absent in ckpt): {len(missing)}")
print(f"UNEXPECTED(in ckpt, not in lm):             {len(unexpected)}")
if missing:
    print("  missing sample:", list(missing)[:8])
if unexpected:
    print("  unexpected sample:", list(unexpected)[:8])

norm_keys = [k for k in state if "norm" in k.lower()]
print(f"norm-related keys in ckpt: {len(norm_keys)}  (must be >0 for hybrid topology)")

ok = (len(missing) == 0 and len(unexpected) == 0 and len(norm_keys) > 0)
print("\nRESULT:", "PASS — clean exact-match handoff, Phase 6 will load the backbone correctly"
      if ok else "CHECK — non-zero missing/unexpected or no norm keys (see above)")
sys.exit(0 if ok else 1)
