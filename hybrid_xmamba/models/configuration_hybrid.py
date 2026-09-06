"""Configuration class for Hybrid models.

Defines the configuration schema for hybrid Mamba-xLSTM architectures,
compatible with Hugging Face transformers library conventions.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal


@dataclass
class HybridConfig:
    """Configuration for Hybrid Mamba-xLSTM models.
    
    Args:
        vocab_size: Size of vocabulary
        dim: Model dimension (hidden size)
        num_layers: Number of transformer-style blocks
        layer_pattern: Repeating pattern of layer types ['mamba', 'mlstm', 'slstm']
        
        # Mamba specific
        state_size: SSM state dimension for Mamba blocks
        conv_size: Convolution kernel size for Mamba
        expand_factor: Expansion factor for Mamba inner dimension
        
        # xLSTM specific
        head_dim: Dimension per attention head for mLSTM
        num_heads: Number of heads for mLSTM (auto-computed if None)
        slstm_hidden_dim: Hidden dimension for sLSTM
        slstm_num_heads: Number of heads for sLSTM
        
        # Shared parameters
        norm_type: Type of normalization ('rms', 'layer')
        use_mlp: Whether to include MLP after each mixer
        mlp_ratio: Expansion ratio for MLP layers
        
        # Training parameters
        max_position_embeddings: Maximum sequence length
        dropout: Dropout probability
        initializer_range: Range for weight initialization
        
        # Generation parameters
        use_cache: Whether to use caching during generation
        tie_word_embeddings: Whether to tie input/output embeddings
    """
    
    # Architecture
    vocab_size: int = 50257
    dim: int = 768
    num_layers: int = 12
    layer_pattern: List[Literal["mamba", "mamba3", "mlstm", "slstm"]] = field(
        default_factory=lambda: ["mamba", "mamba", "mlstm"]
    )
    
    # Mamba parameters
    state_size: int = 16
    conv_size: int = 4
    expand_factor: int = 2
    dt_rank: Optional[int] = None  # Auto if None
    use_fast_path: bool = True
    # MAMBA3_PLAN.md M1: all four default to today's behaviour, so adding them changes nothing.
    scan_impl: str = "legacy"          # "legacy" | "exact"  (default flips to "exact" at M9-A)
    dt_init_strategy: str = "none"     # "none" | "mamba"    (reference logU[dt_min, dt_max] init)
    dt_min: float = 1e-3
    dt_max: float = 1e-1
    tfla_impl: str = "legacy"          # "legacy" | "exact"  (the mLSTM counterpart, M1-H)

    # --- Mamba-3 (MAMBA3_PLAN.md M2). Every flag defaults to the Mamba-2 reduction, so a
    # `mamba3` layer built from defaults is exactly Mamba-2 SSD and each arm moves one variable.
    mamba3_d_state: int = 128          # 8x the Mamba-1 setting for +1.4% params (B/C are shared)
    mamba3_head_dim: int = 64
    mamba3_ngroups: int = 1            # >1 leaves the parameter-matched regime -- see the plan
    mamba3_chunk_size: int = 64
    mamba3_use_conv: bool = True       # dropped as an arm at M5, not by default
    mamba3_conv_size: int = 4
    mamba3_use_trapezoid: bool = False # Sec 3.1 (M3)
    mamba3_use_rope: bool = False      # Sec 3.2 (M4)
    mamba3_rope_fraction: float = 0.5
    mamba3_bc_bias: str = "none"       # "none" | "zero_init" | "one_init"  (Sec 3.4, M5)
    mamba3_mimo_rank: int = 1          # plumbed, never run (decision 3)
    mamba3_a_mode: str = "static"      # "static" (Mamba-2) | "data_dependent" (Mamba-3)
    mamba3_a_floor: float = 1e-4
    mamba3_dt_limit: float = 1.0
    mamba3_use_outproj_norm: bool = False
    
    # mLSTM parameters
    head_dim: int = 64
    num_heads: Optional[int] = None  # Auto-computed
    use_tfla: bool = True
    proj_factor: int = 2
    mlstm_gate_soft_cap: float = 15.0
    mlstm_input_gate_bias_init: float = -10.0
    mlstm_forget_gate_bias_init: float = 0.0
    
    # sLSTM parameters
    slstm_hidden_dim: Optional[int] = None  # Defaults to dim
    slstm_num_heads: int = 4
    use_exponential_gate: bool = True
    
    # Shared parameters
    norm_type: str = "rms"
    # Phase 4 (HybridNorm topology): "pre_rms" (legacy) or "hybrid"
    # (Q/K/V + Δ/B/C pre-norm + FFN post-norm post-residual; first block stays pre-norm).
    norm_topology: str = "pre_rms"
    use_mlp: bool = True
    mlp_ratio: float = 4.0
    
    # Training parameters
    max_position_embeddings: int = 2048
    dropout: float = 0.1
    initializer_range: float = 0.02
    
    # Generation parameters
    use_cache: bool = True
    tie_word_embeddings: bool = False

    # Memory optimisation
    use_gradient_checkpointing: bool = False

    # Pooling strategy for contrastive encoder: "mean" or "attention".
    # "attention" uses a single learnable query; baselines should stay "mean"
    # to serve as a clean ablation control.
    pooling_strategy: str = "mean"

    # Phase 6E — bidirectional contrastive encoding.
    # The text tower is causal end to end (Mamba is a left-to-right selective
    # scan, mLSTM a left-to-right recurrence), so token t never sees token t+1.
    # For radiology reports built as "Findings: ... Impression: ...", the
    # Impression at the END is exactly what recontextualises the Findings at the
    # start, and a causal encoder cannot propagate that backwards.
    #
    # When True, HybridTextEncoder.encode runs a second pass over the
    # length-aware reversed sequence, maps those states back to their original
    # token positions, and averages before pooling. Costs 2x text-encode FLOPs
    # (trivial next to the ViT) and adds NO parameters, so existing checkpoints
    # and every eval path keep working.
    #
    # NOTE: this is motivated by report structure, NOT by cos_text_teacher.
    # The "causal SSM cannot exceed ~0.6 cosine with a bidirectional teacher"
    # claim is FALSIFIED — a frozen causal backbone with a 15.2M head reaches
    # 0.874-0.892 under KD-only warmup. See H100_SCALING_PLAN.md 2026-07-25.
    bidirectional_encode: bool = False

    # Contrastive encoder — projection head dropout.
    # SimCSE view diversity comes from dropout; 0.3 gives meaningful
    # positive-pair variance for a pretrained backbone (default 0.1 is too
    # weak: Stage 0 embeddings are already well-separated, collapsing
    # NT-Xent loss to ~0 and killing gradients).
    proj_head_dropout: float = 0.1
    
    # Model type identifier
    model_type: str = "hybrid_xmamba"
    
    def __post_init__(self):
        """Validate and auto-compute derived parameters."""
        # Auto-compute num_heads for mLSTM if not specified
        if self.num_heads is None:
            self.num_heads = max(1, self.dim // self.head_dim)
        
        # Auto-compute dt_rank for Mamba if not specified
        if self.dt_rank is None:
            self.dt_rank = max(1, self.dim // 16)
        
        # Set sLSTM hidden dim if not specified
        if self.slstm_hidden_dim is None:
            self.slstm_hidden_dim = self.dim
        
        # Validate layer pattern
        valid_types = {"mamba", "mamba3", "mlstm", "slstm"}
        for layer_type in self.layer_pattern:
            if layer_type not in valid_types:
                raise ValueError(
                    f"Invalid layer type '{layer_type}'. "
                    f"Must be one of {valid_types}"
                )

        # MAMBA3_PLAN.md M1-F. norm_topology was previously unvalidated, so a typo silently
        # behaved as "pre_rms" -- the same silent-drop class that cost this project a run in
        # Phase 9 (see tests/test_willi_parity.py::test_norm_topology_threaded_to_hybridconfig).
        if self.norm_topology not in ("pre_rms", "hybrid", "hybrid_bc"):
            raise ValueError(
                "norm_topology must be 'pre_rms', 'hybrid' or 'hybrid_bc', got "
                f"{self.norm_topology!r}"
            )
        if self.scan_impl not in ("legacy", "exact"):
            raise ValueError(f"scan_impl must be 'legacy' or 'exact', got {self.scan_impl!r}")
        if self.tfla_impl not in ("legacy", "exact"):
            raise ValueError(f"tfla_impl must be 'legacy' or 'exact', got {self.tfla_impl!r}")
        if self.mamba3_bc_bias not in ("none", "zero_init", "one_init"):
            raise ValueError(
                "mamba3_bc_bias must be 'none', 'zero_init' or 'one_init', got "
                f"{self.mamba3_bc_bias!r}"
            )
        if self.mamba3_a_mode not in ("static", "data_dependent"):
            raise ValueError(
                f"mamba3_a_mode must be 'static' or 'data_dependent', got {self.mamba3_a_mode!r}"
            )
        if self.dt_init_strategy not in ("none", "mamba"):
            raise ValueError(
                f"dt_init_strategy must be 'none' or 'mamba', got {self.dt_init_strategy!r}"
            )
    
    def get_layer_config(self, layer_idx: int) -> dict:
        """Get configuration for a specific layer.
        
        Args:
            layer_idx: Index of the layer
            
        Returns:
            Dictionary of configuration parameters for that layer
        """
        layer_type = self.layer_pattern[layer_idx % len(self.layer_pattern)]
        
        base_config = {
            "dim": self.dim,
            "norm_type": self.norm_type,
            "use_mlp": self.use_mlp,
            "mlp_ratio": self.mlp_ratio,
        }
        
        if layer_type == "mamba":
            base_config.update({
                "state_size": self.state_size,
                "conv_size": self.conv_size,
                "expand_factor": self.expand_factor,
                "dt_rank": self.dt_rank,
                "use_fast_path": self.use_fast_path,
                "scan_impl": self.scan_impl,
                "tfla_impl": self.tfla_impl,
                "dt_init_strategy": self.dt_init_strategy,
                "dt_min": self.dt_min,
                "dt_max": self.dt_max,
            })
        elif layer_type == "mlstm":
            base_config.update({
                "head_dim": self.head_dim,
                "num_heads": self.num_heads,
                "use_tfla": self.use_tfla,
                "proj_factor": self.proj_factor,
                "gate_soft_cap": self.mlstm_gate_soft_cap,
                "input_gate_bias_init": self.mlstm_input_gate_bias_init,
                "forget_gate_bias_init": self.mlstm_forget_gate_bias_init,
            })
        elif layer_type == "slstm":
            base_config.update({
                "hidden_dim": self.slstm_hidden_dim,
                "num_heads": self.slstm_num_heads,
                "use_exponential_gate": self.use_exponential_gate,
            })
        
        return base_config
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "HybridConfig":
        """Create config from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> "HybridConfig":
        """Load config from pretrained model (placeholder for HF integration)."""
        # This would integrate with Hugging Face Hub
        raise NotImplementedError("Pretrained model loading not yet implemented")
    
    def save_pretrained(self, save_directory: str):
        """Save config to directory (placeholder for HF integration)."""
        import json
        import os
        
        os.makedirs(save_directory, exist_ok=True)
        config_path = os.path.join(save_directory, "config.json")
        
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
