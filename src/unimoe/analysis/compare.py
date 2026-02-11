"""Analysis and comparison of experiment results.

Computes Task Interference Ratio (TIR), bootstrap significance tests,
generates comparison tables and plots.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_all_results(output_dir: str) -> dict:
    """Scan output directories and load all result JSONs.

    Returns structured dict: {config_name: {seed: {results_data}}}.
    """
    output_path = Path(output_dir)
    results = {}

    for config_dir in sorted(output_path.iterdir()):
        if not config_dir.is_dir():
            continue
        config_name = config_dir.name

        for seed_dir in sorted(config_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue
            seed = int(seed_dir.name.split("_")[1])

            results_dir = seed_dir / "results"
            if not results_dir.exists():
                continue

            seed_results = {}

            # Load reranking results
            rerank_path = results_dir / "reranking_results.json"
            if rerank_path.exists():
                with open(rerank_path) as f:
                    seed_results["reranking"] = json.load(f)

            # Load MTEB results
            mteb_path = results_dir / "mteb_results.json"
            if mteb_path.exists():
                with open(mteb_path) as f:
                    seed_results["mteb"] = json.load(f)

            if seed_results:
                results.setdefault(config_name, {})[seed] = seed_results

    return results


def compute_interference_metrics(results: dict) -> dict:
    """Compute Task Interference Ratio (TIR) for both tasks.

    TIR = (Specialist_metric - Joint_metric) / Specialist_metric

    Args:
        results: Output from load_all_results().

    Returns:
        Dict with TIR values per seed, mean, std, and verdict.
    """
    metrics = {
        "reranking_tir": {},
        "embedding_tir": {},
        "verdict": "INCOMPLETE",
    }

    # Find config names (handle both "rank_only_r16" and "rank_only_r8" patterns)
    rank_only = None
    emb_only = None
    joint = None

    for config_name in results:
        if "rank_only" in config_name:
            rank_only = config_name
        elif "emb_only" in config_name:
            emb_only = config_name
        elif "joint" in config_name:
            joint = config_name

    if not joint:
        return metrics

    # Compute reranking TIR per seed
    if rank_only and joint:
        for seed in results.get(joint, {}):
            if seed not in results.get(rank_only, {}):
                continue
            specialist = _get_reranking_score(results[rank_only][seed])
            joint_score = _get_reranking_score(results[joint][seed])
            if specialist and specialist > 0:
                tir = (specialist - joint_score) / specialist
                metrics["reranking_tir"][seed] = tir

    # Compute embedding TIR per seed
    if emb_only and joint:
        for seed in results.get(joint, {}):
            if seed not in results.get(emb_only, {}):
                continue
            specialist = _get_embedding_score(results[emb_only][seed])
            joint_score = _get_embedding_score(results[joint][seed])
            if specialist and specialist > 0:
                tir = (specialist - joint_score) / specialist
                metrics["embedding_tir"][seed] = tir

    # Compute mean +/- std and verdict
    reranking_tirs = list(metrics["reranking_tir"].values())
    embedding_tirs = list(metrics["embedding_tir"].values())

    if reranking_tirs:
        metrics["reranking_mean"] = float(np.mean(reranking_tirs))
        metrics["reranking_std"] = float(np.std(reranking_tirs))
    if embedding_tirs:
        metrics["embedding_mean"] = float(np.mean(embedding_tirs))
        metrics["embedding_std"] = float(np.std(embedding_tirs))

    # Kill gate verdict: only positive TIR indicates interference
    # (joint degrades vs specialist)
    all_tirs = reranking_tirs + embedding_tirs
    if all_tirs:
        max_tir = max(all_tirs)
        if max_tir >= 0.02:
            metrics["verdict"] = "PASS"
        elif max_tir < 0.01:
            metrics["verdict"] = "FAIL"
        else:
            metrics["verdict"] = "MARGINAL"

    return metrics


def _get_reranking_score(seed_results: dict) -> float | None:
    """Extract average nDCG@10 from reranking results."""
    reranking = seed_results.get("reranking", {})
    return reranking.get("average", {}).get("ndcg@10")


def _get_embedding_score(seed_results: dict) -> float | None:
    """Extract average retrieval score from MTEB results."""
    mteb = seed_results.get("mteb", {})
    per_task = mteb.get("per_task", {})
    if not per_task:
        return None
    scores = []
    for task_name, task_data in per_task.items():
        if isinstance(task_data, dict) and "ndcg_at_10" in task_data:
            scores.append(task_data["ndcg_at_10"])
        elif isinstance(task_data, dict):
            # Try extracting main_score
            for key in ["main_score", "ndcg@10", "ndcg_at_10"]:
                if key in task_data:
                    scores.append(task_data[key])
                    break
    return float(np.mean(scores)) if scores else None


def compute_significance(results: dict) -> dict:
    """Paired bootstrap resampling for key comparisons.

    Compares specialist vs joint per-query scores using 10,000 iterations.
    """
    significance = {}

    rank_only = None
    joint = None
    for config_name in results:
        if "rank_only" in config_name:
            rank_only = config_name
        elif "joint" in config_name:
            joint = config_name

    if not (rank_only and joint):
        return significance

    for seed in results.get(joint, {}):
        if seed not in results.get(rank_only, {}):
            continue

        specialist_pq = _get_per_query_scores(results[rank_only][seed])
        joint_pq = _get_per_query_scores(results[joint][seed])

        if specialist_pq and joint_pq:
            # Align query IDs
            common_qids = set(specialist_pq.keys()) & set(joint_pq.keys())
            if len(common_qids) < 10:
                continue

            s_scores = np.array([specialist_pq[qid] for qid in common_qids])
            j_scores = np.array([joint_pq[qid] for qid in common_qids])

            p_value = _bootstrap_test(s_scores, j_scores, n_iter=10000)
            significance[f"seed_{seed}_reranking"] = {
                "p_value": p_value,
                "significant": p_value < 0.05,
                "n_queries": len(common_qids),
            }

    # Embedding significance: emb_only vs joint (task-level paired test)
    emb_only = None
    for config_name in results:
        if "emb_only" in config_name:
            emb_only = config_name

    if emb_only and joint:
        # Collect per-task scores across seeds for paired comparison.
        # Each (seed, task) pair is one observation.
        all_specialist: list[float] = []
        all_joint: list[float] = []

        for seed in results.get(joint, {}):
            if seed not in results.get(emb_only, {}):
                continue

            specialist_ts = _get_per_task_embedding_scores(results[emb_only][seed])
            joint_ts = _get_per_task_embedding_scores(results[joint][seed])

            if specialist_ts and joint_ts:
                common_tasks = set(specialist_ts.keys()) & set(joint_ts.keys())
                for task in common_tasks:
                    all_specialist.append(specialist_ts[task])
                    all_joint.append(joint_ts[task])

        if len(all_specialist) >= 3:
            s_scores = np.array(all_specialist)
            j_scores = np.array(all_joint)

            p_value = _bootstrap_test(s_scores, j_scores, n_iter=10000)
            significance["embedding_overall"] = {
                "p_value": p_value,
                "significant": p_value < 0.05,
                "n_observations": len(all_specialist),
                "test_type": "paired_bootstrap",
                "note": "Each observation is a (seed, task) pair",
            }

    return significance


def _get_per_query_scores(seed_results: dict) -> dict | None:
    """Extract per-query nDCG@10 scores from reranking results.

    Query IDs are prefixed with the dataset name to avoid collisions
    between datasets that use overlapping ID schemes (e.g. SciFact
    and FiQA2018 both use small integers).
    """
    reranking = seed_results.get("reranking", {})
    per_dataset = reranking.get("per_dataset", {})
    all_pq = {}
    for dataset_name, dataset_results in per_dataset.items():
        pq = dataset_results.get("per_query", {})
        for qid, score in pq.items():
            all_pq[f"{dataset_name}_{qid}"] = score
    return all_pq if all_pq else None


def _get_per_task_embedding_scores(seed_results: dict) -> dict | None:
    """Extract per-task nDCG@10 scores from MTEB results.

    Returns dict mapping task_name -> ndcg_at_10 score.
    Used for embedding significance testing at the task level.
    """
    mteb = seed_results.get("mteb", {})
    per_task = mteb.get("per_task", {})
    if not per_task:
        return None
    scores = {}
    for task_name, task_data in per_task.items():
        if isinstance(task_data, dict):
            for key in ["ndcg_at_10", "main_score", "ndcg@10"]:
                if key in task_data:
                    scores[task_name] = task_data[key]
                    break
    return scores if scores else None


def _bootstrap_test(
    scores_a: np.ndarray, scores_b: np.ndarray, n_iter: int = 10000
) -> float:
    """Paired bootstrap resampling test."""
    rng = np.random.default_rng(42)
    n = len(scores_a)
    observed_diff = np.mean(scores_a) - np.mean(scores_b)

    count = 0
    for _ in range(n_iter):
        indices = rng.choice(n, size=n, replace=True)
        diff = np.mean(scores_a[indices]) - np.mean(scores_b[indices])
        if diff <= 0:
            count += 1

    return count / n_iter


def generate_comparison_table(results: dict) -> str:
    """Generate a markdown comparison table of all configs across metrics."""
    lines = []
    lines.append("| Config | nDCG@10 (Reranking) | MTEB Score |")
    lines.append("|--------|---------------------|------------|")

    for config_name in sorted(results.keys()):
        seeds = results[config_name]
        reranking_scores = []
        mteb_scores = []

        for seed, seed_data in seeds.items():
            rs = _get_reranking_score(seed_data)
            if rs is not None:
                reranking_scores.append(rs)
            ms = _get_embedding_score(seed_data)
            if ms is not None:
                mteb_scores.append(ms)

        r_str = f"{np.mean(reranking_scores):.4f} ± {np.std(reranking_scores):.4f}" if reranking_scores else "N/A"
        m_str = f"{np.mean(mteb_scores):.4f} ± {np.std(mteb_scores):.4f}" if mteb_scores else "N/A"
        lines.append(f"| {config_name} | {r_str} | {m_str} |")

    return "\n".join(lines)


def generate_plots(results: dict, output_path: str) -> None:
    """Generate analysis plots and save to output directory.

    Creates:
    - Bar chart of per-dataset nDCG@10 for each config
    - Training loss curves (from JSONL logs)
    - Gradient conflict heatmap (from gradient_conflicts.jsonl)
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Per-dataset nDCG@10 bar chart
    _plot_ndcg_comparison(results, output_dir / "ndcg_comparison.png")

    # 2. Training loss curves
    _plot_loss_curves(Path(output_path).parent, output_dir / "loss_curves.png")

    # 3. Gradient conflict heatmap
    _plot_gradient_conflicts(Path(output_path).parent, output_dir / "gradient_conflicts.png")


def _plot_ndcg_comparison(results: dict, save_path: Path) -> None:
    """Bar chart comparing nDCG@10 across configs and datasets."""
    config_names = sorted(results.keys())
    datasets = set()
    for config_data in results.values():
        for seed_data in config_data.values():
            reranking = seed_data.get("reranking", {})
            per_dataset = reranking.get("per_dataset", {})
            datasets.update(per_dataset.keys())

    if not datasets:
        return

    datasets = sorted(datasets)
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(datasets))
    width = 0.8 / max(1, len(config_names))

    for i, config_name in enumerate(config_names):
        scores = []
        for ds in datasets:
            ds_scores = []
            for seed_data in results[config_name].values():
                reranking = seed_data.get("reranking", {})
                per_dataset = reranking.get("per_dataset", {})
                if ds in per_dataset:
                    ds_scores.append(per_dataset[ds]["aggregate"]["ndcg@10"])
            scores.append(np.mean(ds_scores) if ds_scores else 0)

        ax.bar(x + i * width, scores, width, label=config_name)

    ax.set_ylabel("nDCG@10")
    ax.set_title("Reranking Performance by Dataset")
    ax.set_xticks(x + width * (len(config_names) - 1) / 2)
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _plot_loss_curves(base_dir: Path, save_path: Path) -> None:
    """Plot training loss curves from JSONL logs."""
    fig, ax = plt.subplots(figsize=(10, 6))

    found_any = False
    for config_dir in sorted(base_dir.iterdir()):
        if not config_dir.is_dir():
            continue
        for seed_dir in sorted(config_dir.iterdir()):
            log_path = seed_dir / "train_log.jsonl"
            if not log_path.exists():
                continue
            found_any = True

            steps, losses = [], []
            with open(log_path) as f:
                for line in f:
                    entry = json.loads(line)
                    steps.append(entry["step"])
                    losses.append(entry["loss"])

            label = f"{config_dir.name}/seed_{seed_dir.name.split('_')[1]}"
            ax.plot(steps, losses, label=label, alpha=0.8)

    if not found_any:
        plt.close()
        return

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curves")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _plot_gradient_conflicts(base_dir: Path, save_path: Path) -> None:
    """Heatmap of gradient conflicts from gradient_conflicts.jsonl."""
    conflict_data = {}

    for config_dir in sorted(base_dir.iterdir()):
        if not config_dir.is_dir() or "joint" not in config_dir.name:
            continue
        for seed_dir in sorted(config_dir.iterdir()):
            gc_path = seed_dir / "gradient_conflicts.jsonl"
            if not gc_path.exists():
                continue

            entries = []
            with open(gc_path) as f:
                for line in f:
                    entries.append(json.loads(line))

            if entries:
                # Use last entry's per-layer conflicts
                last_entry = entries[-1]
                per_layer = last_entry.get("per_layer", {})
                label = f"{config_dir.name}/seed_{seed_dir.name.split('_')[1]}"
                conflict_data[label] = per_layer

    if not conflict_data:
        return

    # Build matrix for heatmap
    all_layers = set()
    for conflicts in conflict_data.values():
        all_layers.update(conflicts.keys())

    # Filter to a subset of representative layers for readability
    layers = sorted(all_layers)
    # Take every Nth layer for readability
    if len(layers) > 30:
        step = len(layers) // 30
        layers = layers[::step]

    configs = list(conflict_data.keys())
    matrix = np.zeros((len(configs), len(layers)))

    for i, config in enumerate(configs):
        for j, layer in enumerate(layers):
            matrix[i, j] = conflict_data[config].get(layer, 0)

    fig, ax = plt.subplots(figsize=(20, 4))
    sns.heatmap(
        matrix,
        xticklabels=[l.split(".")[-1] for l in layers],
        yticklabels=configs,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
    )
    ax.set_title("Gradient Conflict: Cosine Similarity per Layer")
    plt.xticks(rotation=90, fontsize=6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
