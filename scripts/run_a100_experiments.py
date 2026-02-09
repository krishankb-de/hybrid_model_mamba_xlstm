"""Complete A100 experiment runner for research paper.

Orchestrates the full experimental pipeline:
  Phase 0 — Quick sanity check (5 min)
  Phase 1 — Train Hybrid 150M on WikiText-103
  Phase 2 — Train Mamba-only 150M baseline
  Phase 3 — Train xLSTM-only 150M baseline
  Phase 4 — Evaluate all checkpoints on WikiText-103 test set
  Phase 5 — Print comparison table

Usage on Lightning AI Studio (A100):
    # Full pipeline (recommended: ~18-24 hours)
    python scripts/run_a100_experiments.py

    # Run only one phase
    python scripts/run_a100_experiments.py --phase 1

    # Quick sanity check only
    python scripts/run_a100_experiments.py --phase 0

    # Custom steps
    python scripts/run_a100_experiments.py --max-steps 5000
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
# EXPERIMENT DEFINITIONS
# ============================================================

EXPERIMENTS = {
    "hybrid_150m": {
        "name": "Hybrid Mamba-xLSTM 150M",
        "model_config": "hybrid_150m",
        "description": "Hybrid architecture with pattern [mamba, mamba, mlstm]",
    },
    "mamba_150m": {
        "name": "Mamba-only 150M",
        "model_config": "mamba_150m_baseline",
        "description": "Pure Mamba baseline (all layers are Mamba)",
    },
    "xlstm_150m": {
        "name": "xLSTM-only (mLSTM) 150M",
        "model_config": "xlstm_150m_baseline",
        "description": "Pure xLSTM baseline (all layers are mLSTM)",
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


def phase0_sanity_check(args):
    """Quick 50-step sanity check to verify everything works."""
    print("\n" + "█" * 80)
    print("  PHASE 0: SANITY CHECK (50 steps, ~5 minutes)")
    print("█" * 80)

    cmd = [
        sys.executable, "scripts/train.py",
        "model=hybrid_150m",
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
        "experiment_name=sanity_check",
        "wandb.enabled=false",
    ]
    return run_command(cmd, "Sanity Check — Hybrid 150M, 50 steps", dry_run=args.dry_run)


def phase_train(args, experiment_key: str, max_steps: int):
    """Train a single model."""
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
        # A100 optimized overrides
        f"trainer.max_steps={max_steps}",
        "trainer.val_check_interval=500",
        "trainer.log_every_n_steps=25",
        "trainer.accumulate_grad_batches=2",
        "trainer.enable_checkpointing=true",
        f"dataset.batch_size={args.batch_size}",
        "dataset.eval_batch_size=32",
        "dataset.num_workers=4",
        "dataset.max_length=2048",
        # Checkpointing
        "callbacks.checkpoint.every_n_train_steps=2500",
        "callbacks.checkpoint.save_top_k=3",
        "callbacks.checkpoint.save_last=true",
        # Experiment tracking
        f"experiment_name={experiment_name}",
        f"wandb.enabled={str(args.wandb).lower()}",
    ]
    return run_command(cmd, f"Training {exp['name']}", dry_run=args.dry_run)


def phase_evaluate(args, experiment_key: str):
    """Evaluate a trained model on WikiText-103 test set."""
    exp = EXPERIMENTS[experiment_key]
    experiment_name = f"{experiment_key}_wikitext"
    checkpoint_dir = PROJECT_ROOT / "outputs" / experiment_name / "checkpoints"

    # Find best checkpoint
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
        f"--batch-size=32",
        f"--output-dir=outputs/{experiment_name}/eval_results",
    ]
    return run_command(cmd, f"Evaluating {exp['name']}", dry_run=args.dry_run)


def print_comparison_table():
    """Print a summary comparison table from evaluation results."""
    print("\n" + "═" * 80)
    print("  RESULTS COMPARISON TABLE")
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
    print("  Published baselines (for reference):")
    print(f"  {'─'*30} {'─'*10}")
    print(f"  {'Transformer (GPT-2 117M)':<30} {'~29.4':>10}")
    print(f"  {'Transformer-XL (151M)':<30} {'~18.3':>10}")
    print(f"  {'Standard Transformer ~150M':<30} {'~24-28':>10}")
    print(f"  {'MEGA 128M':<30} {'~17.3':>10}")
    print()
    print("  NOTE: Published numbers vary by training setup. The most valid")
    print("  comparison is between YOUR three models trained identically.")
    print("═" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Run A100 experiments for Hybrid Mamba-xLSTM research paper"
    )
    parser.add_argument(
        "--phase", type=int, default=-1,
        help="Run only a specific phase (0-5). Default: run all phases."
    )
    parser.add_argument(
        "--max-steps", type=int, default=10000,
        help="Maximum training steps per model (default: 10000, ~5h on A100)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Training batch size per GPU (default: 32 for A100 80GB)"
    )
    parser.add_argument(
        "--wandb", action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing"
    )
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║     HYBRID MAMBA-xLSTM RESEARCH EXPERIMENTS (A100 GPU)         ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print(f"║  Max steps per model : {args.max_steps:<8}                              ║")
    print(f"║  Batch size          : {args.batch_size:<8}                              ║")
    print(f"║  Effective batch     : {args.batch_size * 2:<8} (×2 grad accumulation)    ║")
    print(f"║  Precision           : bf16-mixed (A100 Tensor Cores)         ║")
    print(f"║  Dataset             : WikiText-103 (seq_len=2048)            ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    est_hours = args.max_steps * 1.5 / 3600  # ~1.5s per step on A100
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

    # Phase 1: Train Hybrid
    if run_all or args.phase == 1:
        results["hybrid"] = phase_train(args, "hybrid_150m", args.max_steps)

    # Phase 2: Train Mamba-only baseline
    if run_all or args.phase == 2:
        results["mamba"] = phase_train(args, "mamba_150m", args.max_steps)

    # Phase 3: Train xLSTM-only baseline
    if run_all or args.phase == 3:
        results["xlstm"] = phase_train(args, "xlstm_150m", args.max_steps)

    # Phase 4: Evaluate all
    if run_all or args.phase == 4:
        for key in EXPERIMENTS:
            phase_evaluate(args, key)

    # Phase 5: Print comparison
    if run_all or args.phase == 5:
        print_comparison_table()

    # Final summary
    if run_all:
        print("\n✅ All experiments completed!")
        print_comparison_table()


if __name__ == "__main__":
    main()
