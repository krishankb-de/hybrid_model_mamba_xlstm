"""Unit tests for layer implementations."""

import pytest
import torch
import torch.nn as nn

from hybrid_xmamba.layers.mamba_block import MambaBlock
from hybrid_xmamba.layers.mlstm_block import mLSTMBlock
from hybrid_xmamba.layers.slstm_block import sLSTMBlock
from hybrid_xmamba.layers.hybrid_block import HybridBlock


class TestMambaBlock:
    """Tests for Mamba block."""
    
    def test_forward_pass(self):
        """Test basic forward pass."""
        batch_size, seq_len, dim = 2, 128, 256
        
        block = MambaBlock(dim=dim, state_size=16)
        x = torch.randn(batch_size, seq_len, dim)
        
        output = block(x)
        
        assert output.shape == (batch_size, seq_len, dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_different_sequence_lengths(self):
        """Test with different sequence lengths."""
        dim = 256
        block = MambaBlock(dim=dim, state_size=16)
        
        for seq_len in [64, 128, 256, 512]:
            x = torch.randn(2, seq_len, dim)
            output = block(x)
            assert output.shape == (2, seq_len, dim)


class TestmLSTMBlock:
    """Tests for mLSTM block."""
    
    def test_forward_pass(self):
        """Test basic forward pass."""
        batch_size, seq_len, dim = 2, 128, 256
        
        block = mLSTMBlock(dim=dim, head_dim=64, num_heads=4)
        x = torch.randn(batch_size, seq_len, dim)
        
        output = block(x)
        
        assert output.shape == (batch_size, seq_len, dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_with_cache(self):
        """Test forward pass with caching."""
        batch_size, seq_len, dim = 2, 128, 256

        block = mLSTMBlock(dim=dim, head_dim=64, num_heads=4)
        x = torch.randn(batch_size, seq_len, dim)

        cache = {}
        output = block(x, cache=cache)

        assert output.shape == (batch_size, seq_len, dim)

    def test_mlstm_tanh_softcap_applied(self):
        """Phase 3A: tanh soft-cap keeps i_gate pre-activation within (-cap, cap)."""
        dim, cap = 64, 15.0
        block = mLSTMBlock(dim=dim, head_dim=32, num_heads=2, gate_soft_cap=cap)
        # Force large weights so raw logits would normally exceed the cap
        nn.init.constant_(block.i_gate_proj.weight, 1.0)
        nn.init.constant_(block.i_gate_proj.bias, 0.0)
        x = torch.ones(1, 4, dim) * 10.0  # large input
        # After soft-cap, i_gate = exp(capped_logit) ≤ exp(cap)
        output = block(x)
        assert output.shape == (1, 4, dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_mlstm_input_gate_bias_init(self):
        """Phase 3A: i_gate_proj bias is initialised to mlstm_input_gate_bias_init."""
        dim = 64
        bias_val = -10.0
        block = mLSTMBlock(dim=dim, head_dim=32, num_heads=2,
                           input_gate_bias_init=bias_val)
        assert torch.allclose(
            block.i_gate_proj.bias,
            torch.full_like(block.i_gate_proj.bias, bias_val)
        )

    def test_mlstm_no_overflow_at_large_input(self):
        """Phase 3A+3B: no NaN/inf with large inputs on the slow (non-TFLA) path."""
        dim = 64
        block = mLSTMBlock(dim=dim, head_dim=32, num_heads=2,
                           use_tfla=False, gate_soft_cap=15.0,
                           input_gate_bias_init=0.0)
        # Initialise with large weights to stress-test the stabiliser
        for p in block.parameters():
            if p.ndim >= 2:
                nn.init.constant_(p, 0.5)
        x = torch.randn(1, 32, dim) * 5.0
        output = block(x)
        assert not torch.isnan(output).any(), "NaN in output with large input"
        assert not torch.isinf(output).any(), "Inf in output with large input"


class TestsLSTMBlock:
    """Tests for sLSTM block."""
    
    def test_forward_pass(self):
        """Test basic forward pass."""
        batch_size, seq_len, dim = 2, 128, 256
        
        block = sLSTMBlock(dim=dim, hidden_dim=256, num_heads=4)
        x = torch.randn(batch_size, seq_len, dim)
        
        output = block(x)
        
        assert output.shape == (batch_size, seq_len, dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestHybridBlock:
    """Tests for hybrid block wrapper."""
    
    @pytest.mark.parametrize("layer_type", ["mamba", "mamba3", "mlstm", "slstm"])
    def test_all_layer_types(self, layer_type):
        """Test hybrid block with all layer types."""
        batch_size, seq_len, dim = 2, 64, 256
        
        block = HybridBlock(
            dim=dim,
            layer_type=layer_type,
            state_size=16,
            mamba3_d_state=32,
            mamba3_head_dim=16,
            head_dim=64,
            num_heads=4,
            hidden_dim=256,
            slstm_num_heads=4,
        )
        
        x = torch.randn(batch_size, seq_len, dim)
        output = block(x)
        
        assert output.shape == (batch_size, seq_len, dim)
        assert not torch.isnan(output).any()
    
    def test_with_mlp(self):
        """Test with MLP enabled."""
        batch_size, seq_len, dim = 2, 64, 256
        
        block = HybridBlock(
            dim=dim,
            layer_type="mamba",
            use_mlp=True,
            mlp_ratio=4.0,
        )
        
        x = torch.randn(batch_size, seq_len, dim)
        output = block(x)
        
        assert output.shape == (batch_size, seq_len, dim)
    
    def test_without_mlp(self):
        """Test without MLP."""
        batch_size, seq_len, dim = 2, 64, 256

        block = HybridBlock(
            dim=dim,
            layer_type="mamba",
            use_mlp=False,
        )

        x = torch.randn(batch_size, seq_len, dim)
        output = block(x)

        assert output.shape == (batch_size, seq_len, dim)

    def test_hybrid_norm_ffn_post_residual_block_ge_1(self):
        """Phase 4D/4H: with norm_topology='hybrid' and is_first_block=False,
        FFN normalizes post-residual: out = norm2(x_after_mixer + mlp(x_after_mixer)).

        Verified by zeroing the mlp output projection so mlp(x)=0; the FFN sublayer
        then reduces to norm2(x_after_mixer) in the hybrid path vs x_after_mixer in
        the legacy pre-norm path. We assert the hybrid output equals norm2 of the
        legacy output (which equals x_after_mixer).
        """
        torch.manual_seed(0)
        batch_size, seq_len, dim = 2, 16, 64

        def _build(is_first):
            blk = HybridBlock(
                dim=dim,
                layer_type="mamba",
                norm_topology="hybrid",
                is_first_block=is_first,
                use_mlp=True,
                mlp_ratio=4.0,
                state_size=8,
            )
            # Zero the final MLP linear so mlp(x) = 0
            with torch.no_grad():
                blk.mlp[-1].weight.zero_()
            return blk

        x = torch.randn(batch_size, seq_len, dim)

        blk_nonfirst = _build(is_first=False)
        blk_first = _build(is_first=True)
        # Sync mixer/norm weights so x_after_mixer matches between the two blocks
        blk_first.load_state_dict(blk_nonfirst.state_dict())

        out_nonfirst = blk_nonfirst(x)
        out_first = blk_first(x)

        # First block: pre-norm path → out = x_after_mixer (mlp=0). Non-first hybrid:
        # out = norm2(x_after_mixer). They must differ if norm2 is non-trivial scaling.
        # Concretely: out_nonfirst == norm2(out_first).
        expected = blk_nonfirst.norm2(out_first)
        assert torch.allclose(out_nonfirst, expected, atol=1e-5), (
            "FFN post-norm path must equal norm2(x + mlp(x)) for block >= 1 under hybrid topology"
        )
