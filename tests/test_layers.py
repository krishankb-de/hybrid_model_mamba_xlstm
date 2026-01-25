"""Unit tests for layer implementations."""

import pytest
import torch

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
