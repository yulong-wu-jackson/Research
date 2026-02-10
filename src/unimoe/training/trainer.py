"""Unified trainer supporting rank-only, emb-only, and joint alternating-batch modes.

Handles AdamW optimization, cosine LR schedule, gradient accumulation/clipping,
gradient conflict logging (joint mode), and periodic checkpointing.
"""

from __future__ import annotations

import json
import math
import time
from itertools import cycle
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from unimoe.config import ExperimentConfig, TrainingMode
from unimoe.model.lora_model import UnimodelForExp1
from unimoe.training.losses import InfoNCELoss, RerankingSFTLoss


def _cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    """Create a cosine annealing LR schedule with linear warmup."""

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class UnifiedTrainer:
    """Trainer for the unified embedding/reranking model.

    Supports three modes via TrainingMode:
    - RANK_ONLY: every step uses reranking batch + SFT loss
    - EMB_ONLY: every step uses embedding batch + InfoNCE loss
    - JOINT_SINGLE: alternating batches — odd steps embedding, even steps reranking
    """

    def __init__(
        self,
        config: ExperimentConfig,
        model: UnimodelForExp1,
        emb_dataloader: DataLoader | None = None,
        rerank_dataloader: DataLoader | None = None,
    ):
        self.config = config
        self.model = model
        self.emb_dataloader = emb_dataloader
        self.rerank_dataloader = rerank_dataloader
        self.device = config.model.resolve_device()

        tc = config.training
        self.mode = tc.mode
        self.epochs = tc.epochs
        self.grad_accum_steps = tc.grad_accum_steps
        self.max_grad_norm = tc.max_grad_norm
        self.gradient_conflict_every_n = tc.gradient_conflict_every_n_steps

        # Losses
        self.emb_loss_fn = InfoNCELoss(temperature=tc.temperature)
        self.rerank_loss_fn = RerankingSFTLoss(
            yes_token_id=model.yes_token_id,
            no_token_id=model.no_token_id,
        )
        self.reranking_loss_weight = tc.reranking_loss_weight

        # Optimizer: AdamW on trainable params only (LoRA weights)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable_params, lr=tc.lr)

        # Compute total steps for LR schedule
        self.total_steps = self._compute_total_steps()
        warmup_steps = int(tc.warmup_ratio * self.total_steps)
        self.scheduler = _cosine_schedule_with_warmup(
            self.optimizer, warmup_steps, self.total_steps
        )

        # Output paths
        self.output_dir = Path(config.output_dir) / f"seed_{config.seed}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "train_log.jsonl"
        self.gradient_conflict_path = self.output_dir / "gradient_conflicts.jsonl"
        self.checkpoint_dir = self.output_dir / "checkpoints"

    def _compute_total_steps(self) -> int:
        """Compute total training steps based on mode and data size."""
        tc = self.config.training
        if self.mode == TrainingMode.EMB_ONLY:
            if self.emb_dataloader is None:
                return 0
            steps_per_epoch = len(self.emb_dataloader) // tc.grad_accum_steps
        elif self.mode == TrainingMode.RANK_ONLY:
            if self.rerank_dataloader is None:
                return 0
            steps_per_epoch = len(self.rerank_dataloader) // tc.grad_accum_steps
        else:  # JOINT_SINGLE
            # In joint mode, we alternate between tasks
            emb_steps = len(self.emb_dataloader) if self.emb_dataloader else 0
            rerank_steps = len(self.rerank_dataloader) if self.rerank_dataloader else 0
            steps_per_epoch = max(emb_steps, rerank_steps) // tc.grad_accum_steps
        return steps_per_epoch * tc.epochs

    def train(self) -> dict:
        """Run the full training loop."""
        self.model.train()
        global_step = 0
        total_loss = 0.0
        log_every = 10

        for epoch in range(self.epochs):
            emb_iter = cycle(self.emb_dataloader) if self.emb_dataloader else None
            rerank_iter = cycle(self.rerank_dataloader) if self.rerank_dataloader else None

            # Separate iterators for gradient conflict measurement (avoid consuming training data)
            gc_emb_iter = cycle(self.emb_dataloader) if self.emb_dataloader else None
            gc_rerank_iter = cycle(self.rerank_dataloader) if self.rerank_dataloader else None

            steps_in_epoch = self._steps_in_epoch()
            pbar = tqdm(range(steps_in_epoch), desc=f"Epoch {epoch + 1}/{self.epochs}")

            for step_in_epoch in pbar:
                # Determine task for this step
                if self.mode == TrainingMode.EMB_ONLY:
                    loss, task = self._embedding_step(emb_iter), "embedding"
                elif self.mode == TrainingMode.RANK_ONLY:
                    loss, task = self._reranking_step(rerank_iter), "reranking"
                else:  # JOINT_SINGLE: alternate
                    if step_in_epoch % 2 == 0:
                        loss, task = self._embedding_step(emb_iter), "embedding"
                    else:
                        loss, task = self._reranking_step(rerank_iter), "reranking"

                # Scale loss for gradient accumulation
                scaled_loss = loss / self.grad_accum_steps
                scaled_loss.backward()

                total_loss += loss.item()

                # Gradient accumulation step
                if (step_in_epoch + 1) % self.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.max_grad_norm,
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                    # Gradient conflict logging (joint mode only)
                    # Uses separate iterators to avoid consuming training data
                    if (
                        self.mode == TrainingMode.JOINT_SINGLE
                        and self.gradient_conflict_every_n > 0
                        and global_step % self.gradient_conflict_every_n == 0
                    ):
                        conflicts = self._compute_gradient_conflicts(
                            gc_emb_iter, gc_rerank_iter
                        )
                        self._log_gradient_conflicts(global_step, conflicts)

                    # Logging
                    if global_step % log_every == 0:
                        avg_loss = total_loss / (step_in_epoch + 1)
                        lr = self.scheduler.get_last_lr()[0]
                        pbar.set_postfix(
                            loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}", task=task
                        )
                        self._log_step(global_step, avg_loss, lr, epoch, task)

            # Flush remaining accumulated gradients at epoch boundary
            if (step_in_epoch + 1) % self.grad_accum_steps != 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.max_grad_norm,
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                global_step += 1

        # Save final checkpoint
        final_dir = self.checkpoint_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_adapter(str(final_dir))

        return {"total_steps": global_step, "final_loss": total_loss / max(1, steps_in_epoch)}

    def _steps_in_epoch(self) -> int:
        """Number of micro-steps per epoch."""
        if self.mode == TrainingMode.EMB_ONLY:
            return len(self.emb_dataloader) if self.emb_dataloader else 0
        elif self.mode == TrainingMode.RANK_ONLY:
            return len(self.rerank_dataloader) if self.rerank_dataloader else 0
        else:
            emb_len = len(self.emb_dataloader) if self.emb_dataloader else 0
            rerank_len = len(self.rerank_dataloader) if self.rerank_dataloader else 0
            return max(emb_len, rerank_len)

    def _embedding_step(self, data_iter) -> torch.Tensor:
        """Single embedding training step."""
        batch = next(data_iter)
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        query_emb = self.model.encode(batch["query_input_ids"], batch["query_attention_mask"])
        pos_emb = self.model.encode(batch["pos_input_ids"], batch["pos_attention_mask"])

        neg_emb = None
        if "neg_input_ids" in batch:
            # neg_input_ids: (B, K, seq_len)
            B, K, S = batch["neg_input_ids"].shape
            neg_ids_flat = batch["neg_input_ids"].view(B * K, S)
            neg_mask_flat = batch["neg_attention_mask"].view(B * K, S)
            neg_emb_flat = self.model.encode(neg_ids_flat, neg_mask_flat)
            neg_emb = neg_emb_flat.view(B, K, -1)

        return self.emb_loss_fn(query_emb, pos_emb, neg_emb)

    def _reranking_step(self, data_iter) -> torch.Tensor:
        """Single reranking training step."""
        batch = next(data_iter)
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        logits = self.model.get_rerank_logits(batch["input_ids"], batch["attention_mask"])
        loss = self.rerank_loss_fn(logits, batch["labels"])
        return loss * self.reranking_loss_weight

    def _compute_gradient_conflicts(self, emb_iter, rerank_iter) -> dict:
        """Compute per-layer cosine similarity between embedding and reranking gradients.

        This is the core measurement for Experiment 3 (gradient conflict analysis).
        """
        self.optimizer.zero_grad()

        # Compute embedding gradients
        emb_loss = self._embedding_step(emb_iter)
        emb_loss.backward()
        emb_grads = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                emb_grads[name] = param.grad.clone()

        self.optimizer.zero_grad()

        # Compute reranking gradients
        rerank_loss = self._reranking_step(rerank_iter)
        rerank_loss.backward()

        # Compute per-layer cosine similarity
        conflicts = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and name in emb_grads:
                emb_g = emb_grads[name].flatten()
                rerank_g = param.grad.flatten()
                cos_sim = torch.nn.functional.cosine_similarity(
                    emb_g.unsqueeze(0), rerank_g.unsqueeze(0)
                ).item()
                conflicts[name] = cos_sim

        self.optimizer.zero_grad()

        return conflicts

    def _log_step(self, step: int, loss: float, lr: float, epoch: int, task: str):
        """Append a training log entry."""
        entry = {
            "step": step,
            "loss": loss,
            "lr": lr,
            "epoch": epoch,
            "task": task,
            "timestamp": time.time(),
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _log_gradient_conflicts(self, step: int, conflicts: dict):
        """Append gradient conflict measurements."""
        mean_conflict = sum(conflicts.values()) / max(1, len(conflicts))
        entry = {
            "step": step,
            "mean_cosine_similarity": mean_conflict,
            "per_layer": conflicts,
            "timestamp": time.time(),
        }
        with open(self.gradient_conflict_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
