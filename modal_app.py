"""Modal application for running UniMoE Kill Gate experiment on cloud GPUs.

Executes the full experiment pipeline (data prep, training, evaluation, analysis)
on Modal's serverless infrastructure.  All (config, seed) combinations run in
parallel on separate A100 GPUs, reducing wall-clock time from sequential hours
to a single run's duration.

Core technique: each Modal function calls ``os.chdir("/vol")`` at startup so
every relative path (``data/``, ``outputs/``) resolves against the persistent
Modal Volume.  This means **zero changes** to existing source code.

Usage:
    # Full tier (3 configs x 3 seeds = 9 parallel GPU jobs)
    uv run modal run modal_app.py

    # Quick tier (3 configs x 1 seed = 3 parallel GPU jobs)
    uv run modal run modal_app.py --quick
"""

from __future__ import annotations

import modal

# ---------------------------------------------------------------------------
# Configuration maps — mirrors scripts/run_experiment1.py
# ---------------------------------------------------------------------------

CONFIGS_QUICK = {
    "emb_only": "emb_only_r8.yaml",
    "rank_only": "rank_only_r8.yaml",
    "joint_single": "joint_single_r8.yaml",
}
CONFIGS_FULL = {
    "emb_only": "emb_only_r16.yaml",
    "rank_only": "rank_only_r16.yaml",
    "joint_single": "joint_single_r16.yaml",
}
SEEDS_QUICK = [42]
SEEDS_FULL = [42, 123, 456]

# ---------------------------------------------------------------------------
# Modal infrastructure
# ---------------------------------------------------------------------------

app = modal.App("unimoe-experiment")

vol = modal.Volume.from_name("unimoe-vol", create_if_missing=True)
VOLUME_PATH = "/vol"

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        # Core
        "torch>=2.6.0",
        "transformers>=4.51.0,<5.0.0",
        "peft>=0.18.0",
        "datasets>=4.0.0",
        "accelerate>=1.0.0",
        "sentence-transformers>=4.0.0,<6.0.0",
        "tqdm>=4.66.0",
        "numpy>=2.1.0",
        "pandas>=2.2.0,<3.0.0",
        # Evaluation
        "mteb>=2.7.0",
        "pytrec-eval-terrier>=0.5.10",
        # beir removed — BEIR datasets loaded from HuggingFace Hub directly
        "rank-bm25>=0.2.2",
        # Analysis
        "matplotlib>=3.10.0",
        "seaborn>=0.13.0",
        "ckatorch>=1.0.2",
        "deepsig>=1.2.8",
        # Config
        "pyyaml>=6.0.2",
        "dacite>=1.9.0",
        # Tracking
        "wandb>=0.19.0",
    )
    .env({"PYTHONPATH": "/root/src", "HF_HOME": "/vol/hf_cache"})
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("configs", remote_path="/root/configs")
)


# ---------------------------------------------------------------------------
# Helper: resolve config path on the Modal container
# ---------------------------------------------------------------------------

def _config_path(yaml_name: str) -> str:
    """Return absolute path to a config YAML inside the Modal image."""
    return f"/root/configs/{yaml_name}"


def _reload_volume():
    """Reload the Modal volume, stepping out of /vol to avoid open-file errors."""
    import os
    cwd = os.getcwd()
    os.chdir("/tmp")
    vol.reload()
    os.chdir(cwd)


# ---------------------------------------------------------------------------
# Step 1: Data preparation  (CPU, 32 GB RAM)
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={VOLUME_PATH: vol},
    secrets=[modal.Secret.from_dotenv()],
    memory=32768,
    timeout=3600,
)
def prepare_data(seeds: list[int], config_name: str) -> str:
    """Download MS MARCO data and pre-cache Qwen3-0.6B model on the volume.

    Args:
        seeds: List of random seeds to prepare cached datasets for.
        config_name: YAML config filename (used to read DataConfig).

    Returns:
        Summary string of what was cached.
    """
    import os

    os.chdir(VOLUME_PATH)

    from unimoe.config import load_config
    from unimoe.data.msmarco import load_embedding_dataset, load_reranking_dataset

    config = load_config(_config_path(config_name))

    # Pre-download the base model so training doesn't re-download each time
    print("[prepare_data] Pre-downloading Qwen3-0.6B model to HF cache ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    AutoTokenizer.from_pretrained(config.model.base_model_name, trust_remote_code=True)
    AutoModelForCausalLM.from_pretrained(
        config.model.base_model_name,
        trust_remote_code=True,
        dtype="auto",
    )
    print("[prepare_data] Model cached.")

    # Prepare datasets for each seed
    for seed in seeds:
        print(f"[prepare_data] Loading embedding dataset for seed={seed} ...")
        load_embedding_dataset(config.data, seed=seed)
        print(f"[prepare_data] Loading reranking dataset for seed={seed} ...")
        load_reranking_dataset(config.data, seed=seed)

    vol.commit()
    msg = f"[prepare_data] Done. Seeds={seeds}, config={config_name}"
    print(msg)
    return msg


# ---------------------------------------------------------------------------
# Step 2: Train + Evaluate  (A100-40GB GPU, up to 3 h per run)
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={VOLUME_PATH: vol},
    secrets=[modal.Secret.from_dotenv()],
    gpu="A100-40GB",
    timeout=10800,
)
def train_and_evaluate(config_name: str, seed: int, eval_tier: str) -> dict:
    """Train one (config, seed) combination, then evaluate.

    Skips training if a checkpoint already exists; skips evaluation if results
    already exist.  This makes the function idempotent/resumable.

    Args:
        config_name: YAML config filename (e.g. ``"rank_only_r8.yaml"``).
        seed: Random seed for this run.
        eval_tier: ``"fast"`` or ``"full"`` — controls BEIR dataset count.

    Returns:
        Dict summarising training + evaluation outcomes for this run.
    """
    import gc
    import os
    import random
    from pathlib import Path

    import numpy as np
    import torch

    os.chdir(VOLUME_PATH)
    _reload_volume()

    config_path = _config_path(config_name)

    from unimoe.config import TrainingMode, load_config, save_config

    config = load_config(config_path)
    config.seed = seed
    config.model.device = "cuda"
    config.model.torch_dtype = "bfloat16"
    config.eval.eval_tier = eval_tier

    config_stem = Path(config_name).stem
    run_label = f"{config_stem}/seed_{seed}"
    output_dir = Path(config.output_dir) / f"seed_{seed}"
    checkpoint_dir = output_dir / "checkpoints" / "final"
    results_dir = output_dir / "results"

    summary: dict = {"config": config_stem, "seed": seed, "trained": False, "evaluated": False}

    # --- Reproducibility ---
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        print(f"[train] Checkpoint exists for {run_label}, skipping training.")
    else:
        print(f"[train] Starting training for {run_label} ...")
        from torch.utils.data import DataLoader
        from transformers import AutoTokenizer

        from unimoe.data.collators import EmbeddingCollator, RerankingCollator
        from unimoe.data.msmarco import load_embedding_dataset, load_reranking_dataset
        from unimoe.data.templates import set_tokenizer_config
        from unimoe.model.lora_model import UnimodelForExp1
        from unimoe.training.trainer import UnifiedTrainer

        tokenizer = AutoTokenizer.from_pretrained(
            config.model.base_model_name, trust_remote_code=True
        )
        set_tokenizer_config(tokenizer)

        model = UnimodelForExp1(config, tokenizer=tokenizer)
        model.model.print_trainable_parameters()

        emb_loader = None
        rerank_loader = None

        if config.training.mode in (TrainingMode.EMB_ONLY, TrainingMode.JOINT_SINGLE):
            emb_ds = load_embedding_dataset(config.data, seed=seed)
            emb_collator = EmbeddingCollator(tokenizer, config.data)
            emb_loader = DataLoader(
                emb_ds,
                batch_size=config.training.batch_size_embedding,
                shuffle=True,
                collate_fn=emb_collator,
                drop_last=True,
                num_workers=4,
                persistent_workers=True,
                generator=torch.Generator().manual_seed(seed),
            )

        if config.training.mode in (TrainingMode.RANK_ONLY, TrainingMode.JOINT_SINGLE):
            rerank_ds = load_reranking_dataset(config.data, seed=seed)
            rerank_collator = RerankingCollator(tokenizer, config.data)
            rerank_loader = DataLoader(
                rerank_ds,
                batch_size=config.training.batch_size_reranking,
                shuffle=True,
                collate_fn=rerank_collator,
                drop_last=True,
                num_workers=4,
                persistent_workers=True,
                generator=torch.Generator().manual_seed(seed + 1),
            )

        save_config(config, str(output_dir / "config.yaml"))

        trainer = UnifiedTrainer(
            config=config,
            model=model,
            emb_dataloader=emb_loader,
            rerank_dataloader=rerank_loader,
        )
        result = trainer.train()

        print(
            f"[train] Done {run_label}: "
            f"steps={result['total_steps']}, loss={result['final_loss']:.4f}"
        )
        summary["trained"] = True
        summary["total_steps"] = result["total_steps"]
        summary["final_loss"] = result["final_loss"]

        # Close log file handlers so volume can reload
        import logging as _logging
        _trainer_logger = _logging.getLogger("unimoe.trainer")
        for handler in _trainer_logger.handlers[:]:
            handler.close()
            _trainer_logger.removeHandler(handler)

        # Free GPU memory before evaluation
        del trainer, model, emb_loader, rerank_loader
        gc.collect()
        torch.cuda.empty_cache()

        vol.commit()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    _reload_volume()

    if (results_dir / "reranking_results.json").exists():
        print(f"[eval] Results exist for {run_label}, skipping evaluation.")
    elif not checkpoint_dir.exists():
        print(f"[eval] No checkpoint for {run_label}, skipping evaluation.")
    else:
        print(f"[eval] Starting evaluation for {run_label} ...")
        import json

        from transformers import AutoTokenizer

        from unimoe.config import TrainingMode
        from unimoe.data.templates import set_tokenizer_config
        from unimoe.evaluation.reranking_eval import evaluate_reranking_suite
        from unimoe.model.lora_model import UnimodelForExp1

        tokenizer = AutoTokenizer.from_pretrained(
            config.model.base_model_name, trust_remote_code=True
        )
        set_tokenizer_config(tokenizer)

        eval_model = UnimodelForExp1.load_from_checkpoint(
            config, str(checkpoint_dir), tokenizer=tokenizer
        )

        results_dir.mkdir(parents=True, exist_ok=True)

        # Reranking evaluation (always)
        rerank_results = evaluate_reranking_suite(eval_model, tokenizer, config)
        print(f"[eval] {run_label} avg nDCG@10 = {rerank_results['average']['ndcg@10']:.4f}")
        with open(results_dir / "reranking_results.json", "w") as f:
            json.dump(rerank_results, f, indent=2)

        # Embedding evaluation (for EMB_ONLY and JOINT)
        if config.training.mode in (TrainingMode.EMB_ONLY, TrainingMode.JOINT_SINGLE):
            print(f"[eval] Running MTEB for {run_label} ...")
            from unimoe.evaluation.embedding_eval import evaluate_mteb

            mteb_results = evaluate_mteb(eval_model, tokenizer, config, output_dir=str(results_dir))
            with open(results_dir / "mteb_results.json", "w") as f:
                json.dump(mteb_results, f, indent=2, default=str)

        summary["evaluated"] = True

        del eval_model
        gc.collect()
        torch.cuda.empty_cache()

        vol.commit()

    return summary


# ---------------------------------------------------------------------------
# Step 3: Analysis  (CPU)
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={VOLUME_PATH: vol},
    timeout=600,
)
def analyze() -> str:
    """Run cross-config analysis and return the Markdown report.

    Reads all evaluation results from the volume, computes TIR, significance
    tests, and generates plots and a summary report.

    Returns:
        The analysis report as a Markdown string.
    """
    import os

    os.chdir(VOLUME_PATH)
    _reload_volume()

    from pathlib import Path

    from unimoe.analysis.compare import (
        compute_interference_metrics,
        compute_significance,
        generate_comparison_table,
        generate_plots,
        load_all_results,
    )

    output_dir = "outputs"
    results = load_all_results(output_dir)

    if not results:
        return "No results found. Training and evaluation may not have completed."

    print(f"[analyze] Found configs: {list(results.keys())}")

    tir_metrics = compute_interference_metrics(results)
    significance = compute_significance(results)
    table = generate_comparison_table(results)

    figures_dir = Path(output_dir) / "figures"
    generate_plots(results, str(figures_dir))

    # Build report
    analysis_dir = Path(output_dir) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    lines = ["# UniMoE Experiment 1 — Analysis Report", ""]
    lines.append("## Kill Gate Verdict")
    lines.append(f"\n**{tir_metrics['verdict']}**\n")
    if "reranking_mean" in tir_metrics:
        lines.append(
            f"- Reranking TIR: {tir_metrics['reranking_mean']:.4f}"
            f" +/- {tir_metrics['reranking_std']:.4f}"
        )
    if "embedding_mean" in tir_metrics:
        lines.append(
            f"- Embedding TIR: {tir_metrics['embedding_mean']:.4f}"
            f" +/- {tir_metrics['embedding_std']:.4f}"
        )
    lines.append("\n## Comparison Table\n")
    lines.append(table)
    lines.append("\n## Significance Tests\n")
    for comp, res in significance.items():
        lines.append(f"- {comp}: p={res['p_value']:.4f}")
    lines.append("")

    report = "\n".join(lines)

    report_path = analysis_dir / "analysis_report.md"
    with open(report_path, "w") as f:
        f.write(report)

    vol.commit()

    print(f"[analyze] Report saved to {report_path}")
    return report


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(quick: bool = False):
    """Orchestrate the full experiment pipeline on Modal.

    Args:
        quick: If True, run the quick tier (r=8, seed=42 only).
                If False (default), run the full tier (r=16, seeds=[42,123,456]).
    """
    configs = CONFIGS_QUICK if quick else CONFIGS_FULL
    seeds = SEEDS_QUICK if quick else SEEDS_FULL
    tier = "quick" if quick else "full"
    eval_tier = "fast" if quick else "full"

    total_runs = len(configs) * len(seeds)
    print("=" * 60)
    print("  UniMoE Experiment 1 — Kill Gate  (Modal)")
    print(f"  Tier: {tier}")
    print(f"  Configs: {list(configs.keys())}")
    print(f"  Seeds: {seeds}")
    print(f"  Total GPU jobs: {total_runs}  (running in parallel)")
    print("=" * 60)

    # Step 1: Data preparation (runs once on CPU)
    first_config = list(configs.values())[0]
    print("\n[1/3] Preparing data ...")
    prepare_data.remote(seeds, first_config)
    print("[1/3] Data preparation complete.")

    # Step 2: Train + evaluate all (config, seed) combos in parallel
    print(f"\n[2/3] Launching {total_runs} train+eval jobs in parallel ...")
    args_list = [
        (yaml_name, seed, eval_tier)
        for yaml_name in configs.values()
        for seed in seeds
    ]
    results = list(train_and_evaluate.starmap(args_list))

    print("\n--- Per-run summaries ---")
    for r in results:
        status_parts = []
        if r["trained"]:
            status_parts.append(f"trained (steps={r.get('total_steps', '?')})")
        else:
            status_parts.append("training skipped")
        if r["evaluated"]:
            status_parts.append("evaluated")
        else:
            status_parts.append("eval skipped")
        print(f"  {r['config']}/seed_{r['seed']}: {', '.join(status_parts)}")

    # Step 3: Analysis
    print("\n[3/3] Running analysis ...")
    report = analyze.remote()

    print("\n" + "=" * 60)
    print("  ANALYSIS REPORT")
    print("=" * 60)
    print(report)
    print("=" * 60)
    print("  Experiment complete!")
    print("  Download results:  uv run modal volume get unimoe-vol outputs/ ./modal_outputs/")
    print("=" * 60)
