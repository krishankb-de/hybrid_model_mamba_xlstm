"""Unit tests for kernel implementations."""

import pytest
import torch

# Mark all kernel tests as requiring CUDA
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Kernel tests require CUDA"
)


class TestTFLAKernel:
    """Tests for TFLA kernel."""
    
    def test_tfla_forward(self):
        """Test TFLA forward pass."""
        from hybrid_xmamba.kernels.tfla import apply_tfla
        
        batch_size, num_heads, seq_len, head_dim = 2, 4, 64, 32
        
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
        i_gate = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda').exp()
        f_gate = torch.sigmoid(torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda'))
        
        output = apply_tfla(q, k, v, i_gate, f_gate)
        
        assert output.shape == (batch_size, num_heads, seq_len, head_dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_tfla_backward(self):
        """Test TFLA backward pass."""
        from hybrid_xmamba.kernels.tfla import apply_tfla
        
        batch_size, num_heads, seq_len, head_dim = 2, 4, 32, 16
        
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', requires_grad=True)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', requires_grad=True)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', requires_grad=True)
        i_gate = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda').exp().requires_grad_(True)
        f_gate = torch.sigmoid(torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')).requires_grad_(True)
        
        output = apply_tfla(q, k, v, i_gate, f_gate)
        loss = output.sum()
        loss.backward()
        
        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None
        assert not torch.isnan(q.grad).any()


class TestSelectiveScanKernel:
    """Tests for selective scan kernel."""
    
    def test_scan_forward(self):
        """Test selective scan forward pass."""
        from hybrid_xmamba.kernels.selective_scan import selective_scan
        
        batch_size, seq_len, dim, state_size = 2, 64, 128, 16
        
        x = torch.randn(batch_size, seq_len, dim, device='cuda')
        dt = torch.randn(batch_size, seq_len, dim, device='cuda').softplus()
        A = -torch.randn(dim, state_size, device='cuda').exp()
        B = torch.randn(batch_size, seq_len, state_size, device='cuda')
        C = torch.randn(batch_size, seq_len, state_size, device='cuda')
        D = torch.randn(dim, device='cuda')
        
        output = selective_scan(x, dt, A, B, C, D)
        
        assert output.shape == (batch_size, seq_len, dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_scan_with_gating(self):
        """Test selective scan with gating."""
        from hybrid_xmamba.kernels.selective_scan import selective_scan
        
        batch_size, seq_len, dim, state_size = 2, 64, 128, 16
        
        x = torch.randn(batch_size, seq_len, dim, device='cuda')
        dt = torch.randn(batch_size, seq_len, dim, device='cuda').softplus()
        A = -torch.randn(dim, state_size, device='cuda').exp()
        B = torch.randn(batch_size, seq_len, state_size, device='cuda')
        C = torch.randn(batch_size, seq_len, state_size, device='cuda')
        D = torch.randn(dim, device='cuda')
        z = torch.randn(batch_size, seq_len, dim, device='cuda')
        
        output = selective_scan(x, dt, A, B, C, D, z=z)
        
        assert output.shape == (batch_size, seq_len, dim)
        assert not torch.isnan(output).any()


class TestKernelCorrectness:
    """Tests for kernel correctness against reference implementations."""
    
    def test_tfla_vs_pytorch(self):
        """Compare TFLA kernel output with PyTorch reference."""
        from hybrid_xmamba.kernels.tfla.tfla_interface import apply_tfla, tfla_forward_pytorch
        
        batch_size, num_heads, seq_len, head_dim = 1, 2, 16, 8
        
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
        i_gate = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda').exp()
        f_gate = torch.sigmoid(torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda'))
        
        # Kernel output
        output_kernel = apply_tfla(q, k, v, i_gate, f_gate)
        
        # PyTorch reference
        output_pytorch = tfla_forward_pytorch(q, k, v, i_gate, f_gate)
        
        # Should be close (allowing for numerical differences)
        assert torch.allclose(output_kernel, output_pytorch, rtol=1e-3, atol=1e-3)
