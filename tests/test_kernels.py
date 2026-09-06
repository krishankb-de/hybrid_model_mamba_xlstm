"""Unit tests for kernel implementations."""

import pytest
import torch
import torch.nn.functional as F

# Most kernel tests require CUDA — gate at class level so CPU-only tests
# (e.g. doc-boundary wrapper tests added in Phase 6F) can still run on Willi.
#
# MAMBA3_PLAN.md M1-A: the `cuda` marker used to be declared in pytest.ini but never applied to
# anything, so `-m "not cuda"` deselected nothing and these tests merely skipped. That made the
# marker a lie and hid how little CPU coverage the kernels actually had. The marker is now
# applied alongside the skipif: `-m "not cuda"` deselects, a bare run still skips cleanly.
_requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Kernel tests require CUDA"
)
_cuda_only = pytest.mark.cuda


@_cuda_only
@_requires_cuda
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


@_cuda_only
@_requires_cuda
class TestSelectiveScanKernel:
    """Tests for selective scan kernel."""
    
    def test_scan_forward(self):
        """Test selective scan forward pass."""
        from hybrid_xmamba.kernels.selective_scan import selective_scan
        
        batch_size, seq_len, dim, state_size = 2, 64, 128, 16
        
        x = torch.randn(batch_size, seq_len, dim, device='cuda')
        dt = F.softplus(torch.randn(batch_size, seq_len, dim, device='cuda'))
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
        dt = F.softplus(torch.randn(batch_size, seq_len, dim, device='cuda'))
        A = -torch.randn(dim, state_size, device='cuda').exp()
        B = torch.randn(batch_size, seq_len, state_size, device='cuda')
        C = torch.randn(batch_size, seq_len, state_size, device='cuda')
        D = torch.randn(dim, device='cuda')
        z = torch.randn(batch_size, seq_len, dim, device='cuda')
        
        output = selective_scan(x, dt, A, B, C, D, z=z)
        
        assert output.shape == (batch_size, seq_len, dim)
        assert not torch.isnan(output).any()


@_cuda_only
@_requires_cuda
class TestKernelCorrectness:
    """Tests for kernel correctness against reference implementations."""
    
    def test_tfla_vs_pytorch(self):
        """Compare TFLA kernel output with PyTorch reference."""
        from hybrid_xmamba.kernels.tfla.tfla_interface import apply_tfla, tfla_forward_parallel
        
        batch_size, num_heads, seq_len, head_dim = 1, 2, 16, 8
        
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
        i_gate = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda').exp()
        f_gate = torch.sigmoid(torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda'))
        
        # Kernel output
        output_kernel = apply_tfla(q, k, v, i_gate, f_gate)
        
        # PyTorch reference
        output_pytorch = tfla_forward_parallel(q, k, v, i_gate, f_gate)
        
        # Should be close (allowing for numerical differences)
        assert torch.allclose(output_kernel, output_pytorch, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# Phase 6F: cross-document boundary reset tests (CPU-runnable, no CUDA gate)
# ---------------------------------------------------------------------------
class TestDocBoundaryReset:
    """Verify mixer per-segment wrapper isolates docs via cu_seqlens.

    A perturbation to doc-A tokens must leave doc-B outputs bit-identical when
    cu_seqlens marks them as separate segments. This is the formal invariant
    that PDF gap 4 (doc-boundary contamination) must satisfy.
    """

    def _make_cu_seqlens(self, B: int, L: int, boundary: int) -> torch.Tensor:
        ids = torch.zeros(B, L, dtype=torch.long)
        ids[:, boundary:] = 1
        return ids

    def test_selective_scan_doc_boundary_reset(self):
        from hybrid_xmamba.layers.mamba_block import MambaBlock

        torch.manual_seed(0)
        B, L, D = 2, 16, 32
        boundary = 8
        block = MambaBlock(dim=D, state_size=8, conv_size=4, expand_factor=2).eval()
        x = torch.randn(B, L, D)
        cu = self._make_cu_seqlens(B, L, boundary)

        with torch.no_grad():
            out_ref = block(x, cu_seqlens=cu)
            # Perturb doc-A only
            x_pert = x.clone()
            x_pert[:, :boundary, :] += torch.randn(B, boundary, D) * 5.0
            out_pert = block(x_pert, cu_seqlens=cu)

        # Doc-B must be unchanged
        assert torch.allclose(out_ref[:, boundary:, :], out_pert[:, boundary:, :], atol=1e-5), (
            "Mamba doc-B output leaked from doc-A perturbation"
        )
        # Doc-A must change (sanity: perturbation actually flows somewhere)
        assert not torch.allclose(out_ref[:, :boundary, :], out_pert[:, :boundary, :], atol=1e-5)

    def test_tfla_doc_boundary_reset(self):
        from hybrid_xmamba.layers.mlstm_block import mLSTMBlock

        torch.manual_seed(0)
        B, L, D = 2, 16, 32
        boundary = 8
        block = mLSTMBlock(dim=D, head_dim=16, num_heads=2, use_tfla=True).eval()
        x = torch.randn(B, L, D)
        cu = self._make_cu_seqlens(B, L, boundary)

        with torch.no_grad():
            out_ref = block(x, cu_seqlens=cu)
            x_pert = x.clone()
            x_pert[:, :boundary, :] += torch.randn(B, boundary, D) * 5.0
            out_pert = block(x_pert, cu_seqlens=cu)

        assert torch.allclose(out_ref[:, boundary:, :], out_pert[:, boundary:, :], atol=1e-5), (
            "mLSTM doc-B output leaked from doc-A perturbation"
        )
        assert not torch.allclose(out_ref[:, :boundary, :], out_pert[:, :boundary, :], atol=1e-5)


