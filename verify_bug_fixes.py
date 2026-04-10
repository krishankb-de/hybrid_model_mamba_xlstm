#!/usr/bin/env python3
"""Verify that all Stage 1 evaluation bug fixes have been applied correctly."""

import sys
from pathlib import Path

def check_file_contains(filepath, search_strings, description):
    """Check if file contains all required strings."""
    print(f"\n{'='*70}")
    print(f"Checking: {filepath}")
    print(f"Fix: {description}")
    print(f"{'='*70}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    all_found = True
    for search_str in search_strings:
        if search_str in content:
            print(f"  ✅ Found: {search_str[:60]}...")
        else:
            print(f"  ❌ MISSING: {search_str[:60]}...")
            all_found = False
    
    return all_found

def main():
    print("\n" + "="*70)
    print("STAGE 1 EVALUATION BUG FIXES VERIFICATION")
    print("="*70)
    
    all_checks_passed = True
    
    # Bug 1: Projection head loading fix
    all_checks_passed &= check_file_contains(
        "scripts/evaluate_sts.py",
        [
            'if new_k == "logit_scale":',
            'def strip_state_dict_prefixes(state_dict):',
            '# BUG FIX 1: Only skip logit_scale'
        ],
        "Bug 1 - Projection head loading (evaluate_sts.py)"
    )
    
    all_checks_passed &= check_file_contains(
        "scripts/evaluate_retrieval.py",
        [
            'if new_k == "logit_scale":',
            'def strip_state_dict_prefixes(state_dict):',
            '# BUG FIX 1: Only skip logit_scale'
        ],
        "Bug 1 - Projection head loading (evaluate_retrieval.py)"
    )
    
    # Bug 2: Stage 1 checkpoint loading in evaluate_lm.py
    all_checks_passed &= check_file_contains(
        "scripts/evaluate_lm.py",
        [
            '# BUG FIX 2: Strip inner',
            'if new_k.startswith("lm."):',
            'new_k = new_k[len("lm."):]'
        ],
        "Bug 2 - Stage 1 checkpoint loading (evaluate_lm.py)"
    )
    
    # Bug 3: torch.compile prefix handling
    all_checks_passed &= check_file_contains(
        "scripts/evaluate_sts.py",
        [
            'if k.startswith("model._orig_mod."):',
            'elif k.startswith("_orig_mod.model."):',
            'elif k.startswith("_orig_mod."):'
        ],
        "Bug 3 - torch.compile prefix handling (evaluate_sts.py)"
    )
    
    all_checks_passed &= check_file_contains(
        "scripts/evaluate_retrieval.py",
        [
            'if k.startswith("model._orig_mod."):',
            'elif k.startswith("_orig_mod.model."):',
            'elif k.startswith("_orig_mod."):'
        ],
        "Bug 3 - torch.compile prefix handling (evaluate_retrieval.py)"
    )
    
    # Bug 4: diagnose_eval_issue.py fix
    all_checks_passed &= check_file_contains(
        "diagnose_eval_issue.py",
        [
            '# BUG FIX 4: DO NOT strip lm. prefix',
            'HybridTextEncoder expects it'
        ],
        "Bug 4 - Diagnostic script fix (diagnose_eval_issue.py)"
    )
    
    # Bug 6: max_position_embeddings fix
    all_checks_passed &= check_file_contains(
        "scripts/evaluate_sts.py",
        [
            'max_position_embeddings=1024',
            'max_length=512',
            '# BUG FIX 6'
        ],
        "Bug 6 - max_position_embeddings (evaluate_sts.py)"
    )
    
    all_checks_passed &= check_file_contains(
        "scripts/evaluate_retrieval.py",
        [
            'max_position_embeddings=1024',
            'max_length=512',
            '# BUG FIX 6'
        ],
        "Bug 6 - max_position_embeddings (evaluate_retrieval.py)"
    )
    
    all_checks_passed &= check_file_contains(
        "diagnose_eval_issue.py",
        [
            'max_position_embeddings=1024',
            '# BUG FIX 6'
        ],
        "Bug 6 - max_position_embeddings (diagnose_eval_issue.py)"
    )
    
    # Bug 7: Disk space fix
    all_checks_passed &= check_file_contains(
        "scripts/evaluate_lm.py",
        [
            '# BUG FIX 7',
            'num_proc=1',
            'dataset.select(range(1000))',
            'except Exception as e:'
        ],
        "Bug 7 - Disk space fix (evaluate_lm.py)"
    )
    
    # Bug 8: Retrieval task design
    all_checks_passed &= check_file_contains(
        "scripts/evaluate_retrieval.py",
        [
            'article[:512]',
            '# BUG FIX 8'
        ],
        "Bug 8 - Retrieval task design (evaluate_retrieval.py)"
    )
    
    # Bug 9: Missing keys logging
    all_checks_passed &= check_file_contains(
        "scripts/evaluate_sts.py",
        [
            'critical_missing',
            'CRITICAL KEYS MISSING',
            'list(missing)[:10]'
        ],
        "Bug 9 - Missing keys logging (evaluate_sts.py)"
    )
    
    all_checks_passed &= check_file_contains(
        "scripts/evaluate_retrieval.py",
        [
            'critical_missing',
            'CRITICAL KEYS MISSING',
            'list(missing)[:10]'
        ],
        "Bug 9 - Missing keys logging (evaluate_retrieval.py)"
    )
    
    # Final summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    if all_checks_passed:
        print("\n✅ ALL BUG FIXES VERIFIED SUCCESSFULLY!")
        print("\nAll evaluation scripts have been fixed and are ready to use.")
        print("\nYou can now run:")
        print("  - python scripts/evaluate_sts.py --checkpoint <path>")
        print("  - python scripts/evaluate_retrieval.py --checkpoint <path>")
        print("  - python scripts/evaluate_lm.py --checkpoint <path>")
        print("  - python diagnose_eval_issue.py --checkpoint <path>")
        return 0
    else:
        print("\n❌ SOME FIXES ARE MISSING!")
        print("\nPlease review the output above to see which fixes need to be applied.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
