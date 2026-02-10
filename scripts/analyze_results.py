"""CLI entry point for results analysis.

Usage:
    uv run python scripts/analyze_results.py --output-dir outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path

from unimoe.analysis.compare import (
    compute_interference_metrics,
    compute_significance,
    generate_comparison_table,
    generate_plots,
    load_all_results,
)


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Root output directory containing experiment results",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    print("=" * 60)
    print("UniMoE Experiment Analysis")
    print("=" * 60)

    # Load all results
    print(f"\nLoading results from: {output_dir}")
    results = load_all_results(output_dir)

    if not results:
        print("No results found. Run training and evaluation first.")
        return

    print(f"Found configs: {list(results.keys())}")
    for config_name, seeds in results.items():
        print(f"  {config_name}: seeds {list(seeds.keys())}")

    # Compute TIR
    print("\n--- Task Interference Ratio (TIR) ---")
    tir_metrics = compute_interference_metrics(results)

    if "reranking_mean" in tir_metrics:
        print(f"Reranking TIR: {tir_metrics['reranking_mean']:.4f} ± {tir_metrics['reranking_std']:.4f}")
        for seed, tir in sorted(tir_metrics["reranking_tir"].items()):
            print(f"  Seed {seed}: {tir:.4f}")

    if "embedding_mean" in tir_metrics:
        print(f"Embedding TIR: {tir_metrics['embedding_mean']:.4f} ± {tir_metrics['embedding_std']:.4f}")
        for seed, tir in sorted(tir_metrics["embedding_tir"].items()):
            print(f"  Seed {seed}: {tir:.4f}")

    print(f"\nKill Gate Verdict: {tir_metrics['verdict']}")
    if tir_metrics["verdict"] == "PASS":
        print("  -> Interference detected. Proceed with MoE-LoRA experiments.")
    elif tir_metrics["verdict"] == "FAIL":
        print("  -> No interference detected. Consider pivoting to interpretability study.")
    elif tir_metrics["verdict"] == "MARGINAL":
        print("  -> Marginal interference. Investigate with more data/seeds.")

    # Significance tests
    print("\n--- Bootstrap Significance Tests ---")
    significance = compute_significance(results)
    for comparison, result in significance.items():
        sig_str = "***" if result["significant"] else "n.s."
        print(f"  {comparison}: p={result['p_value']:.4f} {sig_str} (n={result['n_queries']})")

    # Comparison table
    print("\n--- Comparison Table ---")
    table = generate_comparison_table(results)
    print(table)

    # Generate plots
    figures_dir = Path(output_dir) / "figures"
    print(f"\nGenerating plots to: {figures_dir}")
    generate_plots(results, str(figures_dir))

    # Save analysis report
    analysis_dir = Path(output_dir) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    report_path = analysis_dir / "analysis_report.md"

    with open(report_path, "w") as f:
        f.write("# UniMoE Experiment 1 Analysis Report\n\n")
        f.write("## Kill Gate Verdict\n\n")
        f.write(f"**{tir_metrics['verdict']}**\n\n")
        if "reranking_mean" in tir_metrics:
            f.write(f"- Reranking TIR: {tir_metrics['reranking_mean']:.4f} ± {tir_metrics['reranking_std']:.4f}\n")
        if "embedding_mean" in tir_metrics:
            f.write(f"- Embedding TIR: {tir_metrics['embedding_mean']:.4f} ± {tir_metrics['embedding_std']:.4f}\n")
        f.write("\n## Comparison Table\n\n")
        f.write(table)
        f.write("\n\n## Significance Tests\n\n")
        for comp, res in significance.items():
            f.write(f"- {comp}: p={res['p_value']:.4f}\n")

    print(f"\nAnalysis report saved to: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
