"""Comprehensive Stage 1 Evaluation Script.

Runs all three evaluations:
1. STS Benchmark (BIOSSES) - Semantic similarity
2. Retrieval on PubMed - R@1, R@5, R@10
3. Perplexity on PubMed test split

Usage:
    python scripts/evaluate_stage1_full.py \
        --checkpoint outputs/stage1_pubmed_simcse/checkpoints/last.ckpt \
        --output-dir outputs/eval_stage1
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_command(cmd, description):
    """Run a command and capture output."""
    print("\n" + "=" * 80)
    print(f"  {description}")
    print("=" * 80)
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nError running {description}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Stage 1 evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to Stage 1 checkpoint (last.ckpt)")
    parser.add_argument("--output-dir", type=str, default="outputs/eval_stage1",
                        help="Output directory for all results")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--num-retrieval-pairs", type=int, default=1000,
                        help="Number of pairs for retrieval evaluation")
    parser.add_argument("--skip-sts", action="store_true",
                        help="Skip STS evaluation")
    parser.add_argument("--skip-retrieval", action="store_true",
                        help="Skip retrieval evaluation")
    parser.add_argument("--skip-perplexity", action="store_true",
                        help="Skip perplexity evaluation")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Track results
    all_results = {
        "checkpoint": str(args.checkpoint),
        "timestamp": datetime.now().isoformat(),
        "evaluations": {}
    }
    
    success_count = 0
    total_count = 0
    
    # 1. STS Benchmark (BIOSSES)
    if not args.skip_sts:
        total_count += 1
        sts_cmd = [
            "python", "scripts/evaluate_sts.py",
            "--checkpoint", args.checkpoint,
            "--dataset", "biosses",
            "--batch-size", str(args.batch_size),
            "--output-dir", os.path.join(args.output_dir, "sts")
        ]
        
        if run_command(sts_cmd, "1. STS Benchmark (BIOSSES) - Semantic Similarity"):
            success_count += 1
            all_results["evaluations"]["sts"] = "completed"
            
            # Load results
            sts_results_path = Path(args.output_dir) / "sts" / "sts_biosses_results.json"
            if sts_results_path.exists():
                with open(sts_results_path) as f:
                    sts_data = json.load(f)
                    all_results["evaluations"]["sts_spearman"] = sts_data.get("spearman_correlation")
        else:
            all_results["evaluations"]["sts"] = "failed"
    
    # 2. Retrieval Evaluation
    if not args.skip_retrieval:
        total_count += 1
        retrieval_cmd = [
            "python", "scripts/evaluate_retrieval.py",
            "--checkpoint", args.checkpoint,
            "--num-pairs", str(args.num_retrieval_pairs),
            "--batch-size", str(args.batch_size),
            "--output-dir", os.path.join(args.output_dir, "retrieval")
        ]
        
        if run_command(retrieval_cmd, "2. Retrieval Evaluation - R@1, R@5, R@10"):
            success_count += 1
            all_results["evaluations"]["retrieval"] = "completed"
            
            # Load results
            retrieval_results_path = Path(args.output_dir) / "retrieval" / "retrieval_results.json"
            if retrieval_results_path.exists():
                with open(retrieval_results_path) as f:
                    retrieval_data = json.load(f)
                    all_results["evaluations"]["retrieval_metrics"] = retrieval_data.get("metrics")
        else:
            all_results["evaluations"]["retrieval"] = "failed"
    
    # 3. Perplexity Evaluation
    if not args.skip_perplexity:
        total_count += 1
        perplexity_cmd = [
            "python", "scripts/evaluate_lm.py",
            "--checkpoint", args.checkpoint,
            "--dataset", "pubmed",
            "--split", "test",
            "--batch-size", str(args.batch_size // 4),  # Smaller batch for perplexity
            "--max-length", "256",
            "--output-dir", os.path.join(args.output_dir, "perplexity")
        ]
        
        if run_command(perplexity_cmd, "3. Perplexity on PubMed Test Split"):
            success_count += 1
            all_results["evaluations"]["perplexity"] = "completed"
            
            # Load results
            perplexity_results_path = Path(args.output_dir) / "perplexity" / "results.json"
            if perplexity_results_path.exists():
                with open(perplexity_results_path) as f:
                    perplexity_data = json.load(f)
                    all_results["evaluations"]["test_perplexity"] = perplexity_data.get("test_perplexity")
                    all_results["evaluations"]["bits_per_byte"] = perplexity_data.get("bits_per_byte")
        else:
            all_results["evaluations"]["perplexity"] = "failed"
    
    # Save summary
    summary_path = Path(args.output_dir) / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 80)
    print("  EVALUATION SUMMARY")
    print("=" * 80)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Completed: {success_count}/{total_count} evaluations")
    print()
    
    if "sts_spearman" in all_results["evaluations"]:
        spearman = all_results["evaluations"]["sts_spearman"]
        status = "✓ PASS" if spearman >= 0.75 else "✗ Below target"
        print(f"  STS (BIOSSES) Spearman: {spearman:.4f} {status}")
    
    if "retrieval_metrics" in all_results["evaluations"]:
        metrics = all_results["evaluations"]["retrieval_metrics"]
        r1 = metrics.get("R@1", 0)
        status = "✓ PASS" if r1 >= 0.60 else "✗ Below target"
        print(f"  Retrieval R@1: {r1:.4f} {status}")
        print(f"  Retrieval R@5: {metrics.get('R@5', 0):.4f}")
        print(f"  Retrieval R@10: {metrics.get('R@10', 0):.4f}")
    
    if "test_perplexity" in all_results["evaluations"]:
        ppl = all_results["evaluations"]["test_perplexity"]
        bpb = all_results["evaluations"]["bits_per_byte"]
        print(f"  Perplexity: {ppl:.2f}")
        print(f"  Bits per byte: {bpb:.4f}")
    
    print()
    print(f"  Summary saved to: {summary_path}")
    print("=" * 80)
    
    return all_results


if __name__ == "__main__":
    main()
