"""Quick fixes for Google Colab compatibility.

This script applies all necessary fixes to ensure the Hybrid Mamba-xLSTM
project runs smoothly on Google Colab.

Usage:
    !python colab_fixes.py
"""

import os
import sys


def check_gpu():
    """Check if GPU is available and display info."""
    try:
        import torch
    except ImportError:
        print("❌ PyTorch not installed. Run: pip install -r requirements.txt")
        return False
    
    if not torch.cuda.is_available():
        print("⚠️  WARNING: GPU not detected!")
        print("To enable GPU: Runtime -> Change runtime type -> GPU")
        return False
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"✅ GPU detected: {gpu_name}")
    print(f"✅ GPU Memory: {gpu_memory:.1f} GB")
    
    # Warn about low memory
    if gpu_memory < 14:
        print("⚠️  WARNING: Low GPU memory detected.")
        print("   Recommendation: Use smaller batch sizes or model")
    
    return True


def create_directories():
    """Create necessary directories for Colab."""
    dirs = [
        "/content/data",
        "/content/cache",
        "/content/outputs",
        "/content/checkpoints",
        "/content/logs",
        "/content/tokenizer_cache",
    ]
    
    created = 0
    for dir_path in dirs:
        try:
            os.makedirs(dir_path, exist_ok=True, mode=0o755)
            created += 1
        except Exception as e:
            print(f"⚠️  Failed to create {dir_path}: {e}")
    
    print(f"✅ Created/verified {created} directories")
    return created > 0


def set_environment_variables():
    """Set environment variables for optimal Colab performance."""
    env_vars = {
        "HF_HOME": "/content/cache",
        "TRANSFORMERS_CACHE": "/content/cache",
        "TORCH_HOME": "/content/cache",
        "TOKENIZERS_PARALLELISM": "false",  # Avoid warnings
        "OMP_NUM_THREADS": "4",  # Limit CPU threads
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
    
    print(f"✅ Set {len(env_vars)} environment variables")
    print("   - Cache directories configured")
    print("   - Parallelism optimized for Colab")


def check_disk_space():
    """Check available disk space."""
    import shutil
    
    total, used, free = shutil.disk_usage("/content")
    
    free_gb = free / 1e9
    total_gb = total / 1e9
    
    print(f"✅ Disk space: {free_gb:.1f} GB free / {total_gb:.1f} GB total")
    
    if free_gb < 10:
        print("⚠️  WARNING: Low disk space!")
        print("   Consider cleaning up or using Google Drive")
    
    return free_gb > 5


def check_drive_mount():
    """Check if Google Drive is mounted."""
    drive_path = "/content/drive/MyDrive"
    
    if os.path.exists(drive_path):
        print(f"✅ Google Drive is mounted at {drive_path}")
        return True
    else:
        print("ℹ️  Google Drive not mounted")
        print("   To mount: from google.colab import drive; drive.mount('/content/drive')")
        return False


def test_triton():
    """Test if Triton is available and working."""
    try:
        import triton
        print(f"✅ Triton version: {triton.__version__}")
        return True
    except ImportError:
        print("⚠️  Triton not installed")
        print("   Install: pip install triton>=2.1.0")
        return False


def recommend_config():
    """Recommend configuration based on detected GPU."""
    try:
        import torch
        if not torch.cuda.is_available():
            return
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print("\n" + "="*60)
        print("📋 RECOMMENDED CONFIGURATION FOR YOUR GPU")
        print("="*60)
        
        if "T4" in gpu_name:
            print("GPU: Tesla T4 (16GB)")
            print("Recommended model: hybrid_350m")
            print("Recommended config:")
            print("  - batch_size: 8")
            print("  - accumulate_grad_batches: 2")
            print("  - precision: '16-mixed'")
            print("  - max_length: 2048")
            
        elif "V100" in gpu_name:
            print("GPU: Tesla V100 (16-32GB)")
            print("Recommended model: hybrid_350m or smaller hybrid_1b")
            print("Recommended config:")
            print("  - batch_size: 16")
            print("  - accumulate_grad_batches: 1")
            print("  - precision: '16-mixed'")
            print("  - max_length: 4096")
            
        elif "A100" in gpu_name:
            print("GPU: A100 (40GB)")
            print("Recommended model: hybrid_350m or hybrid_1b")
            print("Recommended config:")
            print("  - batch_size: 32")
            print("  - accumulate_grad_batches: 1")
            print("  - precision: 'bf16-mixed'")
            print("  - max_length: 8192")
        
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"⚠️  Could not generate recommendations: {e}")


def main():
    """Run all Colab compatibility checks and fixes."""
    print("\n" + "="*60)
    print("🚀 HYBRID MAMBA-xLSTM - COLAB COMPATIBILITY CHECKER")
    print("="*60 + "\n")
    
    checks_passed = 0
    checks_total = 6
    
    # Check 1: GPU
    print("[1/6] Checking GPU availability...")
    if check_gpu():
        checks_passed += 1
    print()
    
    # Check 2: Directories
    print("[2/6] Creating/verifying directories...")
    if create_directories():
        checks_passed += 1
    print()
    
    # Check 3: Environment
    print("[3/6] Setting environment variables...")
    set_environment_variables()
    checks_passed += 1
    print()
    
    # Check 4: Disk space
    print("[4/6] Checking disk space...")
    if check_disk_space():
        checks_passed += 1
    print()
    
    # Check 5: Drive
    print("[5/6] Checking Google Drive...")
    check_drive_mount()
    checks_passed += 1
    print()
    
    # Check 6: Triton
    print("[6/6] Checking Triton installation...")
    if test_triton():
        checks_passed += 1
    print()
    
    # Summary
    print("="*60)
    print(f"CHECKS PASSED: {checks_passed}/{checks_total}")
    print("="*60)
    
    if checks_passed >= 5:
        print("\n✅ System is ready for training!")
        recommend_config()
    else:
        print("\n⚠️  Some checks failed. Please review warnings above.")
    
    # Additional recommendations
    print("\n📌 NEXT STEPS:")
    print("1. Ensure requirements are installed: pip install -r requirements.txt")
    print("2. (Optional) Mount Google Drive for persistent storage")
    print("3. Configure training: Edit configs/config.yaml")
    print("4. Start training: python scripts/train.py")
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
