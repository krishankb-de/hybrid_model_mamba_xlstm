#!/usr/bin/env python
"""Test script to verify contrastive training fixes.

This script tests:
1. Dataset loading (PubMed)
2. Model initialization
3. Numerical stability of contrastive loss
4. Forward pass without NaN
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from hybrid_xmamba.models.configuration_hybrid import HybridConfig
from hybrid_xmamba.models.hybrid_lm import HybridTextEncoder
from hybrid_xmamba.training.lightning_module import HybridContrastiveLightningModule


def test_dataset_loading():
    """Test that PubMed dataset loads without errors."""
    print("=" * 80)
    print("TEST 1: Dataset Loading")
    print("=" * 80)
    
    try:
        print("Loading ccdv/pubmed-summarization dataset...")
        ds = load_dataset(
            "ccdv/pubmed-summarization",
            split="train",
            streaming=False,
        )
        print(f"✓ Dataset loaded successfully: {len(ds)} samples")
        
        # Check first sample
        sample = ds[0]
        print(f"✓ Sample fields: {list(sample.keys())}")
        print(f"✓ Sample text length: {len(sample.get('article', sample.get('abstract', '')))}")
        return True
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return False


def test_model_initialization():
    """Test that HybridTextEncoder initializes correctly."""
    print("\n" + "=" * 80)
    print("TEST 2: Model Initialization")
    print("=" * 80)
    
    try:
        config = HybridConfig(
            vocab_size=50257,
            dim=512,
            num_layers=8,
            layer_pattern=["mamba", "mamba", "mlstm"],
            state_size=16,
            conv_size=4,
            expand_factor=2,
            head_dim=64,
            num_heads=8,
            max_position_embeddings=1024,
        )
        
        print("Creating HybridTextEncoder...")
        model = HybridTextEncoder(config, embed_dim=512)
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model created: {num_params:,} parameters ({num_params/1e6:.1f}M)")
        
        # Check components
        print(f"✓ LM backbone: {model.lm.__class__.__name__}")
        print(f"✓ Projection head: {len(model.projection_head)} layers")
        print(f"✓ Logit scale: {model.logit_scale.item():.4f}")
        
        return True, model
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_forward_pass(model):
    """Test forward pass and encoding."""
    print("\n" + "=" * 80)
    print("TEST 3: Forward Pass")
    print("=" * 80)
    
    try:
        # Create dummy input
        batch_size = 4
        seq_len = 256
        input_ids = torch.randint(0, 50257, (batch_size, seq_len))
        
        print(f"Input shape: {input_ids.shape}")
        
        # Test encoding
        print("Testing encode() method...")
        model.eval()
        with torch.no_grad():
            embeddings = model.encode(input_ids)
        
        print(f"✓ Embeddings shape: {embeddings.shape}")
        print(f"✓ Embeddings dtype: {embeddings.dtype}")
        
        # Check normalization
        norms = torch.norm(embeddings, dim=-1)
        print(f"✓ Embedding norms: min={norms.min():.6f}, max={norms.max():.6f}, mean={norms.mean():.6f}")
        
        if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5):
            print("⚠ Warning: Embeddings not perfectly normalized (but close enough)")
        
        # Check for NaN/Inf
        if torch.isnan(embeddings).any():
            print("✗ NaN detected in embeddings!")
            return False
        if torch.isinf(embeddings).any():
            print("✗ Inf detected in embeddings!")
            return False
        
        print("✓ No NaN/Inf in embeddings")
        return True
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_contrastive_loss(model):
    """Test contrastive loss computation with numerical stability."""
    print("\n" + "=" * 80)
    print("TEST 4: Contrastive Loss")
    print("=" * 80)
    
    try:
        # Create Lightning module
        print("Creating HybridContrastiveLightningModule...")
        lightning_module = HybridContrastiveLightningModule(
            model=model,
            contrastive_mode="simcse",
            learning_rate=0.0003,
            warmup_steps=100,
            max_steps=1000,
        )
        
        # Create dummy batch
        batch_size = 4
        seq_len = 256
        batch = {
            "input_ids": torch.randint(0, 50257, (batch_size, seq_len))
        }
        
        print(f"Batch shape: {batch['input_ids'].shape}")
        
        # Test training step
        print("Testing training step...")
        lightning_module.train()
        loss = lightning_module.training_step(batch, 0)
        
        print(f"✓ Loss computed: {loss.item():.4f}")
        
        # Check for NaN/Inf
        if torch.isnan(loss):
            print("✗ NaN detected in loss!")
            return False
        if torch.isinf(loss):
            print("✗ Inf detected in loss!")
            return False
        
        print("✓ No NaN/Inf in loss")
        
        # Check logit scale
        logit_scale = model.logit_scale.item()
        temperature = torch.exp(torch.tensor(logit_scale)).item()
        print(f"✓ Logit scale: {logit_scale:.4f} (temperature: {temperature:.2f})")
        
        # Test backward pass
        print("Testing backward pass...")
        loss.backward()
        
        # Check gradients
        has_nan_grad = False
        has_inf_grad = False
        max_grad = 0.0
        
        for name, param in lightning_module.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"✗ NaN gradient in {name}")
                    has_nan_grad = True
                if torch.isinf(param.grad).any():
                    print(f"✗ Inf gradient in {name}")
                    has_inf_grad = True
                max_grad = max(max_grad, param.grad.abs().max().item())
        
        if has_nan_grad or has_inf_grad:
            return False
        
        print(f"✓ No NaN/Inf in gradients (max grad: {max_grad:.6f})")
        
        return True
        
    except Exception as e:
        print(f"✗ Contrastive loss test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_extreme_values(model):
    """Test numerical stability with extreme values."""
    print("\n" + "=" * 80)
    print("TEST 5: Extreme Values")
    print("=" * 80)
    
    try:
        lightning_module = HybridContrastiveLightningModule(
            model=model,
            contrastive_mode="simcse",
        )
        
        # Test with very large logit_scale
        print("Testing with large logit_scale...")
        original_scale = model.logit_scale.data.clone()
        model.logit_scale.data = torch.tensor(10.0)  # exp(10) = 22026
        
        batch = {
            "input_ids": torch.randint(0, 50257, (4, 256))
        }
        
        loss = lightning_module.training_step(batch, 0)
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"✗ Loss is NaN/Inf with large logit_scale: {loss.item()}")
            return False
        
        print(f"✓ Loss stable with large logit_scale: {loss.item():.4f}")
        
        # Restore original scale
        model.logit_scale.data = original_scale
        
        # Test with very small logit_scale
        print("Testing with small logit_scale...")
        model.logit_scale.data = torch.tensor(-2.0)  # exp(-2) = 0.135
        
        loss = lightning_module.training_step(batch, 0)
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"✗ Loss is NaN/Inf with small logit_scale: {loss.item()}")
            return False
        
        print(f"✓ Loss stable with small logit_scale: {loss.item():.4f}")
        
        # Restore original scale
        model.logit_scale.data = original_scale
        
        return True
        
    except Exception as e:
        print(f"✗ Extreme values test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("CONTRASTIVE TRAINING FIXES - VERIFICATION TESTS")
    print("=" * 80)
    
    results = {}
    
    # Test 1: Dataset loading
    results["dataset"] = test_dataset_loading()
    
    # Test 2: Model initialization
    results["model"], model = test_model_initialization()
    
    if not results["model"]:
        print("\n✗ Cannot proceed without model")
        return False
    
    # Test 3: Forward pass
    results["forward"] = test_forward_pass(model)
    
    # Test 4: Contrastive loss
    results["loss"] = test_contrastive_loss(model)
    
    # Test 5: Extreme values
    results["extreme"] = test_extreme_values(model)
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("=" * 80)
        print("\nYou can now submit the training job:")
        print("  sbatch scripts/train_contrastive_stage1.sh")
    else:
        print("✗ SOME TESTS FAILED")
        print("=" * 80)
        print("\nPlease fix the issues before submitting the training job.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
