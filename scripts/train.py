"""CLI entry point for training.

Usage:
    uv run python scripts/train.py --config configs/rank_only_r8.yaml --seed 42
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from unimoe.config import TrainingMode, load_config, save_config
from unimoe.data.collators import EmbeddingCollator, RerankingCollator
from unimoe.data.msmarco import load_embedding_dataset, load_reranking_dataset
from unimoe.data.templates import set_tokenizer_config
from unimoe.model.lora_model import UnimodelForExp1
from unimoe.training.trainer import UnifiedTrainer


def set_seed(seed: int, device: str):
    """Set seed for reproducibility across all frameworks."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif device == "mps":
        # MPS doesn't have a separate seed API; torch.manual_seed covers it
        pass


def main():
    parser = argparse.ArgumentParser(description="Train UniMoE model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=None, help="Override seed from config")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (auto|cuda|mps|cpu)",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    if args.seed is not None:
        config.seed = args.seed
    if args.device is not None:
        config.model.device = args.device

    device = config.model.resolve_device()
    print(f"Device: {device}")
    print(f"Mode: {config.training.mode.value}")
    print(f"Seed: {config.seed}")

    # Set seed
    set_seed(config.seed, device)

    # MPS fallback env var
    if device == "mps":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.base_model_name, trust_remote_code=True
    )
    set_tokenizer_config(tokenizer)

    # Build model
    print("Loading model...")
    model = UnimodelForExp1(config, tokenizer=tokenizer)
    model.model.print_trainable_parameters()

    # Build dataloaders
    emb_loader = None
    rerank_loader = None

    if config.training.mode in (TrainingMode.EMB_ONLY, TrainingMode.JOINT_SINGLE):
        print("Loading embedding dataset...")
        emb_ds = load_embedding_dataset(config.data, seed=config.seed)
        emb_collator = EmbeddingCollator(tokenizer, config.data)
        emb_loader = DataLoader(
            emb_ds,
            batch_size=config.training.batch_size_embedding,
            shuffle=True,
            collate_fn=emb_collator,
            drop_last=True,
        )

    if config.training.mode in (TrainingMode.RANK_ONLY, TrainingMode.JOINT_SINGLE):
        print("Loading reranking dataset...")
        rerank_ds = load_reranking_dataset(config.data, seed=config.seed)
        rerank_collator = RerankingCollator(tokenizer, config.data)
        rerank_loader = DataLoader(
            rerank_ds,
            batch_size=config.training.batch_size_reranking,
            shuffle=True,
            collate_fn=rerank_collator,
            drop_last=True,
        )

    # Save frozen config copy for reproducibility
    output_dir = config.output_dir + f"/seed_{config.seed}"
    save_config(config, f"{output_dir}/config.yaml")

    # Train
    print("Starting training...")
    trainer = UnifiedTrainer(
        config=config,
        model=model,
        emb_dataloader=emb_loader,
        rerank_dataloader=rerank_loader,
    )
    result = trainer.train()

    print("\nTraining complete!")
    print(f"Total steps: {result['total_steps']}")
    print(f"Final loss: {result['final_loss']:.4f}")
    print(f"Checkpoint saved to: {output_dir}/checkpoints/final/")


if __name__ == "__main__":
    main()
