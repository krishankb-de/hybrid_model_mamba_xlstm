"""Test script to verify evaluation setup.

Checks:
1. Required packages are installed
2. Checkpoint file exists
3. Model can be loaded
4. Basic encoding works

Usage:
    python scripts/test_eval_setup.py \
        --checkpoint outputs/stage1_pubmed_simcse/checkpoints/last.ckpt
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_packages():
    """Check if required packages are installed."""
    print("Checking required packages...")
    
    required = {
        "torch": "PyTorch",
        "transformers": "Transformers",
        "datasets": "Datasets",
        "scipy": "SciPy",
        "numpy": "NumPy",
    }
    
    missing = []
    for package, name in required.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT INSTALLED")
            missing.append(package)
    
    # Optional packages
    optional = {
        "sentence_transformers": "Sentence Transformers",
    }
    
    for package, name in optional.items():
        try:
            __import__(package)
            print(f"  ✓ {name} (optional)")
        except ImportError:
            print(f"  ⚠ {name} (optional) - not installed")
    
    if missing:
        print(f"\n✗ Missing required packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    print("\n✓ All required packages installed")
    return True


def check_checkpoint(checkpoint_path):
    """Check if checkpoint exists and can be loaded."""
    print(f"\nChecking checkpoint: {checkpoint_path}")
    
    if not Path(checkpoint_path).exists():
        print(f"  ✗ Checkpoint not found at {checkpoint_path}")
        return False
    
    print(f"  ✓ Checkpoint file exists")
    
    # Try loading
    try:
        import torch
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        
        if "state_dict" in ckpt:
            num_keys = len(ckpt["state_dict"])
            print(f"  ✓ Checkpoint loaded ({num_keys} keys in state_dict)")
        else:
            print(f"  ⚠ Checkpoint loaded but no state_dict found")
        
        return True
    
    except Exception as e:
        print(f"  ✗ Error loading checkpoint: {e}")
        return False


def test_model_loading(checkpoint_path):
    """Test loading the model and basic encoding."""
    print("\nTesting model loading and encoding...")
    
    try:
        import torch
        from transformers import AutoTokenizer
        from hybrid_xmamba.models.configuration_hybrid import HybridConfig
        from hybrid_xmamba.models.hybrid_lm import HybridTextEncoder
        
        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        raw_state_dict = ckpt.get("state_dict", ckpt)
        
        # Strip prefixes
        state_dict = {}
        for k, v in raw_state_dict.items():
            if k.startswith("model."):
                new_k = k[len("model."):]
            else:
                new_k = k
            state_dict[new_k] = v
        
        # Infer config
        dim = 768
        for k, v in state_dict.items():
            if "lm.embedding.token_embedding.weight" in k:
                dim = int(v.shape[1])
                break
        
        # Count layers
        num_layers = 0
        for k in state_dict.keys():
            if "lm.layers." in k:
                import re
                m = re.search(r"lm\.layers\.(\d+)\.", k)
                if m:
                    idx = int(m.group(1))
                    if idx + 1 > num_layers:
                        num_layers = idx + 1
        
        if num_layers == 0:
            num_layers = 12
        
        print(f"  Model config: dim={dim}, num_layers={num_layers}")
        
        # Build config
        base = ["mamba", "mamba", "mlstm"]
        layer_pattern = [base[i % len(base)] for i in range(num_layers)]
        
        config = HybridConfig(
            dim=dim,
            num_layers=num_layers,
            layer_pattern=layer_pattern,
            vocab_size=50257,
            max_position_embeddings=512,
            state_size=16,
            conv_size=4,
            expand_factor=2,
            head_dim=64,
            use_tfla=True,
            proj_factor=2,
            slstm_hidden_dim=dim,
            slstm_num_heads=4,
            norm_type="rms",
            use_mlp=True,
            mlp_ratio=4.0,
            dropout=0.0,
        )
        
        # Create model
        model = HybridTextEncoder(config, embed_dim=512)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        print(f"  ✓ Model loaded successfully")
        
        # Test encoding
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        test_text = "This is a test sentence for biomedical text encoding."
        tokens = tokenizer(test_text, return_tensors="pt", max_length=256, truncation=True)
        
        with torch.no_grad():
            embedding = model.encode(tokens["input_ids"])
        
        print(f"  ✓ Encoding works (output shape: {embedding.shape})")
        print(f"  ✓ Embedding norm: {embedding.norm(dim=-1).item():.4f} (should be ~1.0)")
        
        return True
    
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test evaluation setup")
    parser.add_argument("--checkpoint", type=str, 
                        default="outputs/stage1_pubmed_simcse/checkpoints/last.ckpt",
                        help="Path to checkpoint")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  EVALUATION SETUP TEST")
    print("=" * 60)
    
    # Run checks
    checks = [
        ("Packages", check_packages()),
        ("Checkpoint", check_checkpoint(args.checkpoint)),
        ("Model Loading", test_model_loading(args.checkpoint)),
    ]
    
    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ All checks passed! Ready to run evaluations.")
        print("\nNext steps:")
        print("  1. Run comprehensive evaluation:")
        print(f"     python scripts/evaluate_stage1_full.py --checkpoint {args.checkpoint}")
        print("  2. Or submit SLURM job:")
        print("     sbatch scripts/eval_stage1.sh")
    else:
        print("\n✗ Some checks failed. Please fix the issues above.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
