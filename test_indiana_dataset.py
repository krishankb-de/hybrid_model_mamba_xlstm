#!/usr/bin/env python3
"""Quick test to check Indiana CXR dataset structure."""

from datasets import load_dataset

print("Loading dataset...")
try:
    ds = load_dataset("dz-osamu/IU-Xray", split="train")
    print(f"✓ Dataset loaded: {len(ds)} samples")
    print(f"\nDataset features: {ds.features}")
    print(f"\nFirst sample keys: {list(ds[0].keys())}")
    print(f"\nFirst 3 samples:")
    for i in range(min(3, len(ds))):
        print(f"\n--- Sample {i} ---")
        for key, value in ds[i].items():
            if isinstance(value, str):
                print(f"  {key}: {value[:100]}..." if len(value) > 100 else f"  {key}: {value}")
            elif isinstance(value, list):
                print(f"  {key}: list with {len(value)} items")
                if len(value) > 0:
                    print(f"    First item: {value[0]}")
            else:
                print(f"  {key}: {type(value).__name__}")
                
    # Check for None images
    none_count = sum(1 for item in ds if item.get("image") is None)
    print(f"\n✗ Samples with None images: {none_count}/{len(ds)}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
