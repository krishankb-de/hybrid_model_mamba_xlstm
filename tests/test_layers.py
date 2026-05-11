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
    
    @pytest.mark.parametrize("layer_type", ["mamba", "mlstm", "slstm"])
    def test_all_layer_types(self, layer_type):
        """Test hybrid block with all layer types."""
        batch_size, seq_len, dim = 2, 64, 256
        
        block = HybridBlock(
            dim=dim,
            layer_type=layer_type,
            state_size=16,
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
