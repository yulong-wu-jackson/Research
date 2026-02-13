"""Embedding evaluation via MTEB v2.

Runs MTEB benchmark using the MTEBEncoderWrapper on configured retrieval tasks.
"""

from __future__ import annotations

from pathlib import Path

import mteb

from unimoe.config import ExperimentConfig
from unimoe.evaluation.model_wrappers import MTEBEncoderWrapper
from unimoe.model.lora_model import UnimodelForExp1

# Fast tier retrieval tasks (small datasets for quick evaluation)
FAST_TIER_TASKS = ["SciFact", "NFCorpus", "FiQA2018"]


def evaluate_mteb(
    model: UnimodelForExp1,
    tokenizer,
    config: ExperimentConfig,
    output_dir: str | None = None,
) -> dict:
    """Run MTEB evaluation on configured retrieval tasks.

    For fast tier: SciFact, NFCorpus, FiQA2018.
    For full tier: all MTEB(eng, v2) retrieval tasks.

    Args:
        model: Trained UnimodelForExp1 instance.
        tokenizer: Tokenizer for encoding.
        config: Experiment configuration.
        output_dir: Directory to save results.

    Returns:
        Dict with per-task and average scores.
    """
    wrapper = MTEBEncoderWrapper(model, batch_size=config.eval.eval_batch_size)

    # Select tasks based on eval tier
    if config.eval.eval_tier == "fast":
        tasks = mteb.get_tasks(tasks=FAST_TIER_TASKS)
    else:
        # Full tier: all English retrieval tasks from MTEB v2
        try:
            tasks = mteb.get_tasks(
                task_types=["Retrieval"],
                languages=["eng"],
            )
        except Exception:
            # Fallback to fast tier tasks
            tasks = mteb.get_tasks(tasks=FAST_TIER_TASKS)

    # Run evaluation
    if output_dir:
        prediction_folder = Path(output_dir) / "mteb_predictions"
    else:
        prediction_folder = None

    result = mteb.evaluate(
        model=wrapper,
        tasks=tasks,
        encode_kwargs={"batch_size": config.eval.eval_batch_size},
        prediction_folder=prediction_folder,
    )

    # Extract scores
    task_scores = {}
    if result and hasattr(result, "task_results"):
        for task_result in result.task_results:
            task_name = task_result.task_name
            main_score = task_result.get_score()
            task_scores[task_name] = {"main_score": main_score}

    return {
        "per_task": task_scores,
        "eval_tier": config.eval.eval_tier,
    }
