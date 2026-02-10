"""Master experiment runner for Kill Gate Experiment (Experiment 1).

Orchestrates training, evaluation, and analysis across all configs and seeds.

Usage:
    Quick tier (r=8, seed=42 only):
        uv run python scripts/run_experiment1.py --quick

    Full tier (r=16, seeds=[42, 123, 456]):
        uv run python scripts/run_experiment1.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Configuration
CONFIGS_QUICK = {
    "emb_only": "configs/emb_only_r8.yaml",
    "rank_only": "configs/rank_only_r8.yaml",
    "joint_single": "configs/joint_single_r8.yaml",
}

CONFIGS_FULL = {
    "emb_only": "configs/emb_only_r16.yaml",
    "rank_only": "configs/rank_only_r16.yaml",
    "joint_single": "configs/joint_single_r16.yaml",
}

SEEDS_QUICK = [42]
SEEDS_FULL = [42, 123, 456]


def _run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return True if successful."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: {description} failed with exit code {e.returncode}")
        return False


def _checkpoint_exists(config_yaml: str, seed: int) -> bool:
    """Check if a final checkpoint already exists for this run."""
    # Determine output dir from config name
    config_name = Path(config_yaml).stem  # e.g., "rank_only_r8"
    checkpoint_dir = Path("outputs") / config_name / f"seed_{seed}" / "checkpoints" / "final"
    return checkpoint_dir.exists() and any(checkpoint_dir.iterdir())


def _results_exist(config_yaml: str, seed: int) -> bool:
    """Check if evaluation results already exist for this run."""
    config_name = Path(config_yaml).stem
    results_dir = Path("outputs") / config_name / f"seed_{seed}" / "results"
    return (results_dir / "reranking_results.json").exists()


def main():
    parser = argparse.ArgumentParser(
        description="Run Kill Gate Experiment 1 (all configs and seeds)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick tier: r=8, seed=42 only (validates pipeline)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (auto|cuda|mps|cpu)",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training, only run evaluation and analysis",
    )
    args = parser.parse_args()

    tier = "quick" if args.quick else "full"
    configs = CONFIGS_QUICK if args.quick else CONFIGS_FULL
    seeds = SEEDS_QUICK if args.quick else SEEDS_FULL

    total_runs = len(configs) * len(seeds)
    print(f"{'='*60}")
    print("  UniMoE Experiment 1 — Kill Gate")
    print(f"  Tier: {tier}")
    print(f"  Configs: {list(configs.keys())}")
    print(f"  Seeds: {seeds}")
    print(f"  Total runs: {total_runs}")
    print(f"{'='*60}")

    start_time = time.time()
    completed = 0
    failed = 0

    for config_name, config_path in configs.items():
        for seed in seeds:
            run_label = f"{config_name}/seed_{seed}"
            print(f"\n{'#'*60}")
            print(f"  Run {completed + 1}/{total_runs}: {run_label}")
            print(f"{'#'*60}")

            # Training phase
            if not args.skip_training:
                if _checkpoint_exists(config_path, seed):
                    print(f"  [SKIP] Checkpoint exists for {run_label}, skipping training")
                else:
                    train_cmd = [
                        sys.executable,
                        "scripts/train.py",
                        "--config", config_path,
                        "--seed", str(seed),
                    ]
                    if args.device:
                        train_cmd.extend(["--device", args.device])

                    success = _run_command(train_cmd, f"Training {run_label}")
                    if not success:
                        print(f"  [FAIL] Training failed for {run_label}")
                        failed += 1
                        continue

            # Evaluation phase
            if _results_exist(config_path, seed):
                print(f"  [SKIP] Results exist for {run_label}, skipping evaluation")
            else:
                config_stem = Path(config_path).stem
                checkpoint_path = f"outputs/{config_stem}/seed_{seed}/checkpoints/final"

                if not Path(checkpoint_path).exists():
                    print(f"  [SKIP] No checkpoint at {checkpoint_path}, skipping evaluation")
                    failed += 1
                    continue

                eval_cmd = [
                    sys.executable,
                    "scripts/evaluate.py",
                    "--checkpoint", checkpoint_path,
                    "--config", config_path,
                    "--eval-tier", "fast" if args.quick else "full",
                ]
                success = _run_command(eval_cmd, f"Evaluating {run_label}")
                if not success:
                    print(f"  [FAIL] Evaluation failed for {run_label}")
                    failed += 1
                    continue

            completed += 1
            elapsed = time.time() - start_time
            print(f"  [DONE] {run_label} ({elapsed:.0f}s elapsed)")

    # Analysis phase
    print(f"\n{'='*60}")
    print("  Running analysis...")
    print(f"{'='*60}")

    analysis_cmd = [sys.executable, "scripts/analyze_results.py", "--output-dir", "outputs"]
    _run_command(analysis_cmd, "Analysis")

    # Summary
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print("  Experiment 1 Complete")
    print(f"  Tier: {tier}")
    print(f"  Completed: {completed}/{total_runs}")
    print(f"  Failed: {failed}/{total_runs}")
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
