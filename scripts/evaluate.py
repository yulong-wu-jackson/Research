"""CLI entry point for evaluation.

Usage:
    uv run python scripts/evaluate.py --checkpoint outputs/rank_only_r8/seed_42/checkpoints/final --config configs/rank_only_r8.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from unimoe.config import TrainingMode, load_config
from unimoe.data.templates import set_tokenizer_config
from unimoe.evaluation.reranking_eval import evaluate_reranking_suite
from unimoe.model.lora_model import UnimodelForExp1


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained UniMoE model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to PEFT adapter checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--eval-tier", type=str, default=None, help="Override eval tier (fast|full)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.eval_tier:
        config.eval.eval_tier = args.eval_tier

    device = config.model.resolve_device()
    print(f"Device: {device}")
    print(f"Eval tier: {config.eval.eval_tier}")
    print(f"Checkpoint: {args.checkpoint}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.base_model_name, trust_remote_code=True
    )
    set_tokenizer_config(tokenizer)

    # Load model from checkpoint
    print("Loading model from checkpoint...")
    model = UnimodelForExp1.load_from_checkpoint(
        config, args.checkpoint, tokenizer=tokenizer
    )

    # Results directory
    results_dir = Path(args.checkpoint).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Reranking evaluation (always run)
    print("\n--- Reranking Evaluation ---")
    rerank_results = evaluate_reranking_suite(model, tokenizer, config)
    print(f"Average nDCG@10: {rerank_results['average']['ndcg@10']:.4f}")

    for dataset_name, result in rerank_results["per_dataset"].items():
        print(f"  {dataset_name}: nDCG@10 = {result['aggregate']['ndcg@10']:.4f}")

    # Save reranking results
    with open(results_dir / "reranking_results.json", "w") as f:
        json.dump(rerank_results, f, indent=2)

    # Embedding evaluation (for EMB_ONLY and JOINT configs)
    mode = config.training.mode
    if mode in (TrainingMode.EMB_ONLY, TrainingMode.JOINT_SINGLE):
        print("\n--- Embedding Evaluation (MTEB) ---")
        from unimoe.evaluation.embedding_eval import evaluate_mteb

        mteb_results = evaluate_mteb(
            model, tokenizer, config, output_dir=str(results_dir)
        )
        with open(results_dir / "mteb_results.json", "w") as f:
            json.dump(mteb_results, f, indent=2, default=str)
        print(f"MTEB results saved to {results_dir}/mteb_results.json")

    print(f"\nAll results saved to: {results_dir}")


if __name__ == "__main__":
    main()
