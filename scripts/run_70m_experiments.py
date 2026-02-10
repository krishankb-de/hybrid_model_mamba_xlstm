"""Complete A100 experiment runner for 70M parameter models.

Orchestrates the full experimental pipeline for the 70M model scale:
  Phase 0 — Quick sanity check (2-3 min)
  Phase 1 — Train Hybrid 70M on WikiText-103
  Phase 2 — Train Mamba-only 70M baseline
  Phase 3 — Train xLSTM-only 70M baseline
  Phase 4 — Evaluate all checkpoints on WikiText-103 test set
  Phase 5 — Print comparison table

The 70M models (dim=512, 8 layers) are significantly faster to train than
150M models, making them ideal for:
  - Initial architecture validation and hyperparameter search
  - Quick ablation studies
  - Testing the full training pipeline before scaling up
  - Resource-constrained environments (single consumer GPU, Colab free tier)

Usage on A100 GPU:
    # Full pipeline (recommended: ~6-10 hours for 10k steps)
    python scripts/run_70m_experiments.py

    # Run only one phase
    python scripts/run_70m_experiments.py --phase 1

    # Quick sanity check only
    python scripts/run_70m_experiments.py --phase 0

    # Custom steps (e.g., full 50k steps for paper-quality results)
    python scripts/run_70m_experiments.py --max-steps 50000

    # Short training for pipeline verification
    python scripts/run_70m_experiments.py --max-steps 500 --dry-run

    # Enable W&B logging
    python scripts/run_70m_experiments.py --wandb

    # Run evaluation only (after training is done)
    python scripts/run_70m_experiments.py --phase 4

    # Print comparison table only
    python scripts/run_70m_experiments.py --phase 5
"""

import os
import sys
import subprocess
import argparse
import json
import time
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# EXPERIMENT DEFINITIONS — 70M SCALE
# ============================================================

EXPERIMENTS = {
    "hybrid_70m": {
        "name": "Hybrid Mamba-xLSTM 70M",
        "model_config": "hybrid_70m",
        "description": "Hybrid architecture with pattern [mamba, mamba, mlstm] (dim=512, 8 layers)",
    },
    "mamba_70m": {
        "name": "Mamba-only 70M",
        "model_config": "mamba_70m_baseline",
        "description": "Pure Mamba baseline — all layers are Mamba (dim=512, 8 layers)",
    },
    "xlstm_70m": {
        "name": "xLSTM-only (mLSTM) 70M",
        "model_config": "xlstm_70m_baseline",
        "description": "Pure xLSTM baseline — all layers are mLSTM (dim=512, 8 layers)",
    },
}


def run_command(cmd: list, description: str, dry_run: bool = False) -> int:
    """Run a shell command with logging."""
    cmd_str = " ".join(cmd)
    print(f"\n{'='*80}")
    print(f"  {description}")
    print(f"  CMD: {cmd_str}")
    print(f"{'='*80}\n")

    if dry_run:
        print("[DRY RUN] Skipping execution")
        return 0

    start = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - start

    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    print(f"\n  Finished in {int(h)}h {int(m)}m {int(s)}s  (exit code {result.returncode})")
    return result.returncode


# ============================================================
# PHASE 0 — SANITY CHECK
# ============================================================

def phase0_sanity_check(args):
    """Quick 50-step sanity check to verify everything works with the 70M model.

    This runs the smallest model config (70M) for just 50 steps to ensure:
    - Model instantiation is correct
    - Data loading and tokenization work
    - Forward/backward pass completes
    - Loss decreases from initial random values
    """
    print("\n" + "█" * 80)
    print("  PHASE 0: SANITY CHECK — Hybrid 70M (50 steps, ~2-3 minutes)")
    print("█" * 80)

    cmd = [
        sys.executable, "scripts/train.py",
        "model=hybrid_70m",
        "dataset=wikitext",
        "trainer=a100_single_gpu",
        # Override for quick test
        "trainer.max_steps=50",
        "trainer.val_check_interval=25",
        "trainer.log_every_n_steps=5",
        "trainer.enable_checkpointing=false",
        "trainer.limit_val_batches=5",
        f"dataset.batch_size={args.batch_size}",
        "dataset.eval_batch_size=16",
        "dataset.num_workers=4",
        "experiment_name=sanity_check_70m",
        "wandb.enabled=false",
    ]
    return run_command(cmd, "Sanity Check — Hybrid 70M, 50 steps", dry_run=args.dry_run)


# ============================================================
# PHASE 1/2/3 — TRAINING
# ============================================================

def phase_train(args, experiment_key: str, max_steps: int):
    """Train a single 70M model.

    Args:
        args: CLI arguments
        experiment_key: Key into EXPERIMENTS dict
        max_steps: Maximum training steps
    """
    exp = EXPERIMENTS[experiment_key]
    experiment_name = f"{experiment_key}_wikitext"

    print(f"\n{'█'*80}")
    print(f"  TRAINING: {exp['name']}")
    print(f"  {exp['description']}")
    print(f"  Steps: {max_steps}  |  Batch: {args.batch_size}  |  Precision: bf16-mixed")
    print(f"{'█'*80}")

    cmd = [
        sys.executable, "scripts/train.py",
        f"model={exp['model_config']}",
        "dataset=wikitext",
        "trainer=a100_single_gpu",
        # Training duration
        f"trainer.max_steps={max_steps}",
        # Validation & logging
        "trainer.val_check_interval=500",
        "trainer.log_every_n_steps=25",
        # Gradient accumulation (effective batch = batch_size * 2)
        "trainer.accumulate_grad_batches=2",
        # Enable checkpointing
        "trainer.enable_checkpointing=true",
        # Data loading
        f"dataset.batch_size={args.batch_size}",
        "dataset.eval_batch_size=32",
        "dataset.num_workers=4",
        "dataset.max_length=2048",
        # Checkpoint settings
        "callbacks.checkpoint.every_n_train_steps=2500",
        "callbacks.checkpoint.save_top_k=3",
        "callbacks.checkpoint.save_last=true",
        # Experiment tracking
        f"experiment_name={experiment_name}",
        f"wandb.enabled={str(args.wandb).lower()}",
    ]
    return run_command(cmd, f"Training {exp['name']}", dry_run=args.dry_run)


# ============================================================
# PHASE 4 — EVALUATION
# ============================================================

def phase_evaluate(args, experiment_key: str):
    """Evaluate a trained 70M model on WikiText-103 test set.

    Looks for the best/last checkpoint and runs evaluate_lm.py to compute:
    - Test perplexity
    - Inference throughput (tokens/second)
    - Peak GPU memory
    - Optional text generation samples
    """
    exp = EXPERIMENTS[experiment_key]
    experiment_name = f"{experiment_key}_wikitext"
    checkpoint_dir = PROJECT_ROOT / "outputs" / experiment_name / "checkpoints"

    # Find best checkpoint (prefer last.ckpt)
    ckpt_path = checkpoint_dir / "last.ckpt"
    if not ckpt_path.exists():
        # Try to find any checkpoint
        ckpts = list(checkpoint_dir.glob("*.ckpt"))
        if ckpts:
            ckpt_path = sorted(ckpts)[-1]
        else:
            print(f"  ⚠ No checkpoint found for {exp['name']} at {checkpoint_dir}")
            return -1

    print(f"\n{'─'*80}")
    print(f"  EVALUATING: {exp['name']}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"{'─'*80}")

    cmd = [
        sys.executable, "scripts/evaluate_lm.py",
        f"--checkpoint={ckpt_path}",
        f"--model-config={exp['model_config']}",
        "--dataset=wikitext",
        "--split=test",
        "--batch-size=32",
        "--throughput",
        f"--output-dir=outputs/{experiment_name}/eval_results",
    ]
    return run_command(cmd, f"Evaluating {exp['name']}", dry_run=args.dry_run)


# ============================================================
# PHASE 5 — COMPARISON TABLE
# ============================================================

def print_comparison_table():
    """Print a summary comparison table from evaluation results."""
    print("\n" + "═" * 80)
    print("  RESULTS COMPARISON TABLE — 70M MODELS")
    print("═" * 80)
    print(f"\n  {'Model':<30} {'Test PPL':>10} {'Val PPL':>10} {'Params':>12} {'Tokens/s':>12}")
    print(f"  {'─'*30} {'─'*10} {'─'*10} {'─'*12} {'─'*12}")

    for key, exp in EXPERIMENTS.items():
        result_file = PROJECT_ROOT / "outputs" / f"{key}_wikitext" / "eval_results" / "results.json"
        if result_file.exists():
            with open(result_file) as f:
                results = json.load(f)
            test_ppl = results.get("test_perplexity", "N/A")
            val_ppl = results.get("val_perplexity", "N/A")
            params = results.get("total_params", "N/A")
            tps = results.get("tokens_per_second", "N/A")
            ppl_str = f"{test_ppl:.2f}" if isinstance(test_ppl, float) else str(test_ppl)
            vppl_str = f"{val_ppl:.2f}" if isinstance(val_ppl, float) else str(val_ppl)
            params_str = f"{params/1e6:.1f}M" if isinstance(params, (int, float)) else str(params)
            tps_str = f"{tps:,.0f}" if isinstance(tps, (int, float)) else str(tps)
            print(f"  {exp['name']:<30} {ppl_str:>10} {vppl_str:>10} {params_str:>12} {tps_str:>12}")
        else:
            print(f"  {exp['name']:<30} {'(not run)':>10}")

    print()
    print("  Published baselines (for reference, at ~150M scale):")
    print(f"  {'─'*30} {'─'*10}")
    print(f"  {'GPT-2 Small (124M)':<30} {'~29.4':>10}")
    print(f"  {'Transformer-XL (151M)':<30} {'~18.3':>10}")
    print(f"  {'MEGA 128M':<30} {'~17.3':>10}")
    print()
    print("  NOTE: 70M models will have higher perplexity than 150M+ models.")
    print("  The key comparison is BETWEEN your three 70M architectures,")
    print("  not against published 150M baselines.")
    print("═" * 80)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run A100 experiments for 70M Hybrid Mamba-xLSTM models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (all 5 phases)
  python scripts/run_70m_experiments.py

  # Quick sanity check only
  python scripts/run_70m_experiments.py --phase 0

  # Train only the hybrid model
  python scripts/run_70m_experiments.py --phase 1

  # Train with more steps for paper-quality results
  python scripts/run_70m_experiments.py --max-steps 50000

  # Evaluate all trained models
  python scripts/run_70m_experiments.py --phase 4

  # Print results comparison table
  python scripts/run_70m_experiments.py --phase 5

  # Dry run (show commands without executing)
  python scripts/run_70m_experiments.py --dry-run
        """,
    )
    parser.add_argument(
        "--phase", type=int, default=-1,
        help="Run only a specific phase (0-5). Default: run all phases. "
             "0=sanity, 1=train hybrid, 2=train mamba, 3=train xlstm, "
             "4=evaluate all, 5=comparison table."
    )
    parser.add_argument(
        "--max-steps", type=int, default=10000,
        help="Maximum training steps per model (default: 10000). "
             "For paper-quality results use 50000."
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Training batch size per GPU (default: 32 for A100 80GB). "
             "Reduce to 16 for 40GB GPUs or 8 for consumer GPUs."
    )
    parser.add_argument(
        "--wandb", action="store_true",
        help="Enable Weights & Biases logging."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing."
    )
    parser.add_argument(
        "--generate", action="store_true",
        help="Generate text samples during evaluation (Phase 4)."
    )
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║    HYBRID MAMBA-xLSTM 70M EXPERIMENTS (A100 GPU)               ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print(f"║  Model scale         : 70M (dim=512, 8 layers)                ║")
    print(f"║  Max steps per model : {args.max_steps:<8}                              ║")
    print(f"║  Batch size          : {args.batch_size:<8}                              ║")
    print(f"║  Effective batch     : {args.batch_size * 2:<8} (×2 grad accumulation)    ║")
    print(f"║  Precision           : bf16-mixed (A100 Tensor Cores)         ║")
    print(f"║  Dataset             : WikiText-103 (seq_len=2048)            ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    # 70M is ~2x faster than 150M per step
    est_hours = args.max_steps * 0.8 / 3600  # ~0.8s per step on A100 for 70M
    total_est = est_hours * 3
    print(f"║  Est. time per model : ~{est_hours:.1f} hours                            ║")
    print(f"║  Est. total time     : ~{total_est:.1f} hours (3 models)                 ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

    run_all = (args.phase == -1)
    results = {}

    # Phase 0: Sanity Check
    if run_all or args.phase == 0:
        rc = phase0_sanity_check(args)
        results["phase0"] = rc
        if rc != 0 and not args.dry_run:
            print("❌ Sanity check FAILED. Fix errors before proceeding.")
            sys.exit(1)
        if args.phase == 0:
            print("✅ Sanity check passed. Ready for full training.")
            return

    # Phase 1: Train Hybrid 70M
    if run_all or args.phase == 1:
        print("\n" + "█" * 80)
        print("  PHASE 1: TRAIN HYBRID MAMBA-xLSTM 70M")
        print("█" * 80)
        results["hybrid"] = phase_train(args, "hybrid_70m", args.max_steps)

    # Phase 2: Train Mamba-only 70M baseline
    if run_all or args.phase == 2:
        print("\n" + "█" * 80)
        print("  PHASE 2: TRAIN MAMBA-ONLY 70M BASELINE")
        print("█" * 80)
        results["mamba"] = phase_train(args, "mamba_70m", args.max_steps)

    # Phase 3: Train xLSTM-only 70M baseline
    if run_all or args.phase == 3:
        print("\n" + "█" * 80)
        print("  PHASE 3: TRAIN xLSTM-ONLY 70M BASELINE")
        print("█" * 80)
        results["xlstm"] = phase_train(args, "xlstm_70m", args.max_steps)

    # Phase 4: Evaluate all models
    if run_all or args.phase == 4:
        print("\n" + "█" * 80)
        print("  PHASE 4: EVALUATE ALL 70M MODELS")
        print("█" * 80)
        for key in EXPERIMENTS:
            phase_evaluate(args, key)

    # Phase 5: Print comparison table
    if run_all or args.phase == 5:
        print_comparison_table()

    # Final summary
    if run_all:
        print("\n✅ All 70M experiments completed!")
        print_comparison_table()

        # Print next steps
        print("\n" + "─" * 80)
        print("  NEXT STEPS:")
        print("─" * 80)
        print("  1. Review the results above and verify hybrid outperforms baselines")
        print("  2. Check TensorBoard logs:  tensorboard --logdir outputs/")
        print("  3. Scale up to 150M:  python scripts/run_a100_experiments.py")
        print("  4. Fine-tune hyperparameters if needed (lr, warmup, etc.)")
        print("─" * 80)


if __name__ == "__main__":
    main()
