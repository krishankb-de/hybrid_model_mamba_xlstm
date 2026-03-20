"""Verification script for contrastive training setup.

This script checks that all components are correctly installed and configured
for the two-stage contrastive training pipeline.
"""

import sys
from pathlib import Path

def check_imports():
    """Check that all required packages are installed."""
    print("=" * 80)
    print("Checking Python packages...")
    print("=" * 80)
    
    required_packages = {
        "torch": "PyTorch",
        "pytorch_lightning": "PyTorch Lightning",
        "transformers": "HuggingFace Transformers",
        "datasets": "HuggingFace Datasets",
        "hydra": "Hydra",
        "omegaconf": "OmegaConf",
        "PIL": "Pillow (for image loading)",
        "open_clip": "OpenCLIP (for BiomedCLIP)",
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - NOT FOUND")
            missing.append(package)
    
    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print("Install with: pip install open-clip-torch Pillow")
        return False
    
    print("\n✓ All required packages are installed")
    return True


def check_model_files():
    """Check that model files exist and are correct."""
    print("\n" + "=" * 80)
    print("Checking model implementation files...")
    print("=" * 80)
    
    files_to_check = [
        "hybrid_xmamba/models/hybrid_lm.py",
        "hybrid_xmamba/training/lightning_module.py",
        "scripts/train_contrastive.py",
        "configs/dataset/pubmed.yaml",
        "configs/dataset/indiana_cxr.yaml",
    ]
    
    all_exist = True
    for file_path in files_to_check:
        path = Path(file_path)
        if path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - NOT FOUND")
            all_exist = False
    
    if not all_exist:
        print("\n⚠ Some files are missing. Please check your installation.")
        return False
    
    print("\n✓ All required files exist")
    return True


def check_model_classes():
    """Check that new model classes are properly defined."""
    print("\n" + "=" * 80)
    print("Checking model classes...")
    print("=" * 80)
    
    try:
        from hybrid_xmamba.models.hybrid_lm import HybridTextEncoder
        print("✓ HybridTextEncoder class found")
        
        # Check methods
        required_methods = ["encode", "forward", "get_num_params"]
        for method in required_methods:
            if hasattr(HybridTextEncoder, method):
                print(f"  ✓ {method}() method exists")
            else:
                print(f"  ✗ {method}() method missing")
                return False
        
    except ImportError as e:
        print(f"✗ Failed to import HybridTextEncoder: {e}")
        return False
    
    try:
        from hybrid_xmamba.training.lightning_module import HybridContrastiveLightningModule
        print("✓ HybridContrastiveLightningModule class found")
        
    except ImportError as e:
        print(f"✗ Failed to import HybridContrastiveLightningModule: {e}")
        return False
    
    print("\n✓ All model classes are properly defined")
    return True


def check_config_files():
    """Check that config files have correct structure."""
    print("\n" + "=" * 80)
    print("Checking configuration files...")
    print("=" * 80)
    
    try:
        from omegaconf import OmegaConf
        
        # Check PubMed config
        pubmed_config = OmegaConf.load("configs/dataset/pubmed.yaml")
        assert pubmed_config.dataset_name == "pubmed", "PubMed config: wrong dataset_name"
        assert pubmed_config.max_length == 512, "PubMed config: wrong max_length"
        print("✓ configs/dataset/pubmed.yaml is valid")
        
        # Check Indiana CXR config
        indiana_config = OmegaConf.load("configs/dataset/indiana_cxr.yaml")
        assert indiana_config.dataset_name == "indiana_cxr", "Indiana config: wrong dataset_name"
        assert indiana_config.contrastive_mode == "clip", "Indiana config: wrong contrastive_mode"
        print("✓ configs/dataset/indiana_cxr.yaml is valid")
        
    except Exception as e:
        print(f"✗ Config validation failed: {e}")
        return False
    
    print("\n✓ Configuration files are valid")
    return True


def check_cuda():
    """Check CUDA availability."""
    print("\n" + "=" * 80)
    print("Checking CUDA setup...")
    print("=" * 80)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"✓ CUDA is available")
            print(f"  GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"    Memory: {props.total_memory / 1e9:.1f} GB")
                print(f"    Compute capability: {props.major}.{props.minor}")
        else:
            print("⚠ CUDA is not available - training will be slow on CPU")
            return False
        
    except Exception as e:
        print(f"✗ CUDA check failed: {e}")
        return False
    
    print("\n✓ CUDA is properly configured")
    return True


def test_model_instantiation():
    """Test that the model can be instantiated."""
    print("\n" + "=" * 80)
    print("Testing model instantiation...")
    print("=" * 80)
    
    try:
        import torch
        from hybrid_xmamba.models.configuration_hybrid import HybridConfig
        from hybrid_xmamba.models.hybrid_lm import HybridTextEncoder
        
        # Create a small config for testing
        config = HybridConfig(
            vocab_size=50257,
            dim=256,  # Small for testing
            num_layers=2,  # Small for testing
            layer_pattern=["mamba", "mlstm"],
            max_position_embeddings=512,
        )
        
        # Instantiate model
        model = HybridTextEncoder(config, embed_dim=512)
        print(f"✓ Model instantiated successfully")
        print(f"  Parameters: {model.get_num_params():,}")
        
        # Test encode method
        dummy_input = torch.randint(0, 50257, (2, 128))
        with torch.no_grad():
            embeddings = model.encode(dummy_input)
        
        assert embeddings.shape == (2, 512), f"Wrong embedding shape: {embeddings.shape}"
        assert torch.allclose(embeddings.norm(dim=-1), torch.ones(2), atol=1e-5), "Embeddings not normalized"
        print(f"✓ encode() method works correctly")
        print(f"  Output shape: {embeddings.shape}")
        print(f"  Embedding norms: {embeddings.norm(dim=-1)}")
        
    except Exception as e:
        print(f"✗ Model instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✓ Model instantiation test passed")
    return True


def main():
    """Run all verification checks."""
    print("\n" + "=" * 80)
    print("CONTRASTIVE TRAINING SETUP VERIFICATION")
    print("=" * 80 + "\n")
    
    checks = [
        ("Package imports", check_imports),
        ("Model files", check_model_files),
        ("Model classes", check_model_classes),
        ("Config files", check_config_files),
        ("CUDA setup", check_cuda),
        ("Model instantiation", test_model_instantiation),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n✗ {name} check failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n" + "=" * 80)
        print("✓ ALL CHECKS PASSED - Ready for contrastive training!")
        print("=" * 80)
        print("\nNext steps:")
        print("1. Review CONTRASTIVE_TRAINING_GUIDE.md")
        print("2. Run Stage 1: sbatch scripts/train_contrastive_stage1.sh")
        print("3. After Stage 1 completes, run Stage 2: sbatch scripts/train_contrastive_stage2.sh")
        return 0
    else:
        print("\n" + "=" * 80)
        print("✗ SOME CHECKS FAILED - Please fix the issues above")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
