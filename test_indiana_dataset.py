#!/usr/bin/env python3
"""Quick test to check Indiana CXR dataset structure."""

from datasets import load_dataset

print("Loading dataset...")
try:
    ds = load_dataset("dz-osamu/IU-Xray", split="train")
    print(f"✓ Dataset loaded: {len(ds)} samples")
    print(f"\nDataset features: {ds.features}")
    print(f"\nFirst sample keys: {list(ds[0].keys())}")
    print(f"\nFirst sample:")
    for key, value in ds[0].items():
        if isinstance(value, str):
            print(f"  {key}: {value[:100]}..." if len(value) > 100 else f"  {key}: {value}")
        else:
            print(f"  {key}: {type(value)} - {value}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
