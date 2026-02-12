"""Unified trainer supporting rank-only, emb-only, and joint alternating-batch modes.

Handles AdamW optimization, cosine LR schedule, gradient accumulation/clipping,
gradient conflict logging (joint mode), and periodic checkpointing.

Logging features:
- Sliding window loss (recent 50 micro-steps) instead of cumulative average
- Per-task loss tracking (loss_emb, loss_rerank) for joint mode
- Gradient norm logging (pre-clip and post-clip)
- GPU memory usage tracking (CUDA only)
- Epoch-level summaries
- Optional WandB integration (controlled by config.wandb_enabled)
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections import deque
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from unimoe.config import ExperimentConfig, TrainingMode
from unimoe.model.lora_model import UnimodelForExp1
from unimoe.training.losses import InfoNCELoss, RerankingSFTLoss

logger = logging.getLogger("unimoe.trainer")

LOSS_WINDOW_SIZE = 50


def _infinite_dataloader(dataloader):
    """Yield batches from a dataloader, restarting when exhausted.

    Unlike ``itertools.cycle()``, this does **not** cache yielded batches in
    memory.  Each restart re-shuffles the data (if ``shuffle=True`` on the
    DataLoader) and avoids the unbounded memory growth of ``cycle()``.
    """
    while True:
        yield from dataloader


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
        self.trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = AdamW(self.trainable_params, lr=tc.lr)

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

        # Python logging — writes to both stderr and a readable log file
        self._setup_logger()

        # WandB
        self.wandb_enabled = config.wandb_enabled
        self._wandb = None
        if self.wandb_enabled:
            self._init_wandb()

    def _setup_logger(self):
        """Configure Python logger with stderr + file handlers."""
        if logger.handlers:
            return  # Already configured (e.g. from a previous trainer instance)

        logger.setLevel(logging.INFO)
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # stderr handler (visible in terminal / Modal container logs)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)

        # File handler (human-readable log alongside the JSONL)
        fh = logging.FileHandler(self.output_dir / "train.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    def _init_wandb(self):
        """Initialize Weights & Biases run.

        Falls back gracefully if wandb is unavailable or init fails.
        """
        try:
            import wandb

            wandb.init(
                project="unimoe-kill-gate",
                name=(
                    f"{self.config.experiment_name}/seed_{self.config.seed}"
                ),
                config={
                    "mode": self.mode.value,
                    "seed": self.config.seed,
                    "lr": self.config.training.lr,
                    "epochs": self.epochs,
                    "batch_size_emb": self.config.training.batch_size_embedding,
                    "batch_size_rerank": (
                        self.config.training.batch_size_reranking
                    ),
                    "grad_accum_steps": self.grad_accum_steps,
                    "max_grad_norm": self.max_grad_norm,
                    "lora_rank": self.config.lora.rank,
                    "lora_alpha": self.config.lora.alpha,
                    "temperature": self.config.training.temperature,
                    "reranking_loss_weight": self.reranking_loss_weight,
                    "base_model": self.config.model.base_model_name,
                    "total_steps": self.total_steps,
                },
                tags=[self.mode.value, f"seed_{self.config.seed}"],
                reinit="finish_previous",
            )
            self._wandb = wandb
        except Exception as e:
            print(f"[warn] WandB init failed ({e}), continuing without WandB.")
            self.wandb_enabled = False

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
            # Sum of both loaders so each task sees the same number of
            # micro-steps as its specialist counterpart.
            emb_steps = len(self.emb_dataloader) if self.emb_dataloader else 0
            rerank_steps = (
                len(self.rerank_dataloader) if self.rerank_dataloader else 0
            )
            steps_per_epoch = (emb_steps + rerank_steps) // tc.grad_accum_steps
        return steps_per_epoch * tc.epochs

    def train(self) -> dict:
        """Run the full training loop."""
        self.model.train()
        global_step = 0
        log_every = 10

        # Sliding window loss trackers (recent N micro-steps)
        loss_window: deque[float] = deque(maxlen=LOSS_WINDOW_SIZE)
        loss_window_emb: deque[float] = deque(maxlen=LOSS_WINDOW_SIZE)
        loss_window_rerank: deque[float] = deque(maxlen=LOSS_WINDOW_SIZE)

        # Epoch-level accumulators
        epoch_total_loss = 0.0
        epoch_total_emb_loss = 0.0
        epoch_total_rerank_loss = 0.0
        epoch_emb_steps = 0
        epoch_rerank_steps = 0

        step_start_time = time.time()

        for epoch in range(self.epochs):
            emb_iter = _infinite_dataloader(self.emb_dataloader) if self.emb_dataloader else None
            rerank_iter = (
                _infinite_dataloader(self.rerank_dataloader) if self.rerank_dataloader else None
            )

            # Separate iterators for gradient conflict measurement
            gc_emb_iter = (
                _infinite_dataloader(self.emb_dataloader) if self.emb_dataloader else None
            )
            gc_rerank_iter = (
                _infinite_dataloader(self.rerank_dataloader) if self.rerank_dataloader else None
            )

            steps_in_epoch = self._steps_in_epoch()
            pbar = tqdm(
                range(steps_in_epoch), desc=f"Epoch {epoch + 1}/{self.epochs}"
            )

            # Build proportional task schedule for joint mode
            task_schedule: list[str] | None = None
            if self.mode == TrainingMode.JOINT_SINGLE:
                task_schedule = self._build_task_schedule(steps_in_epoch)

            # Reset epoch accumulators
            epoch_total_loss = 0.0
            epoch_total_emb_loss = 0.0
            epoch_total_rerank_loss = 0.0
            epoch_emb_steps = 0
            epoch_rerank_steps = 0

            for step_in_epoch in pbar:
                # Determine task for this step
                if self.mode == TrainingMode.EMB_ONLY:
                    loss, task = self._embedding_step(emb_iter), "embedding"
                elif self.mode == TrainingMode.RANK_ONLY:
                    loss, task = self._reranking_step(rerank_iter), "reranking"
                else:  # JOINT_SINGLE: proportional scheduling
                    task = task_schedule[step_in_epoch]
                    if task == "embedding":
                        loss = self._embedding_step(emb_iter)
                    else:
                        loss = self._reranking_step(rerank_iter)

                # Scale loss for gradient accumulation
                scaled_loss = loss / self.grad_accum_steps
                scaled_loss.backward()

                loss_val = loss.item()
                loss_window.append(loss_val)
                epoch_total_loss += loss_val

                # Per-task loss tracking
                if task == "embedding":
                    loss_window_emb.append(loss_val)
                    epoch_total_emb_loss += loss_val
                    epoch_emb_steps += 1
                else:
                    loss_window_rerank.append(loss_val)
                    epoch_total_rerank_loss += loss_val
                    epoch_rerank_steps += 1

                # Gradient accumulation step
                if (step_in_epoch + 1) % self.grad_accum_steps == 0:
                    # Compute gradient norm before clipping
                    grad_norm_pre = torch.nn.utils.clip_grad_norm_(
                        self.trainable_params,
                        self.max_grad_norm,
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                    # Gradient conflict logging (joint mode only)
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
                        now = time.time()
                        step_duration = (now - step_start_time) / log_every
                        step_start_time = now

                        lr = self.scheduler.get_last_lr()[0]
                        windowed_loss = (
                            sum(loss_window) / len(loss_window)
                            if loss_window
                            else 0.0
                        )

                        # Build metrics dict
                        metrics = {
                            "step": global_step,
                            "loss": windowed_loss,
                            "lr": lr,
                            "epoch": epoch,
                            "task": task,
                            "grad_norm": float(grad_norm_pre),
                            "step_time_sec": step_duration,
                            "timestamp": now,
                        }

                        # Per-task windowed losses (always populated,
                        # even in single-task mode for consistency)
                        if loss_window_emb:
                            metrics["loss_emb"] = sum(loss_window_emb) / len(
                                loss_window_emb
                            )
                        if loss_window_rerank:
                            metrics["loss_rerank"] = sum(
                                loss_window_rerank
                            ) / len(loss_window_rerank)

                        # GPU memory (CUDA only)
                        if self.device == "cuda":
                            mem = torch.cuda.memory_allocated() / (1024**2)
                            mem_peak = torch.cuda.max_memory_allocated() / (
                                1024**2
                            )
                            metrics["gpu_mem_mb"] = mem
                            metrics["gpu_mem_peak_mb"] = mem_peak

                        # tqdm progress bar
                        postfix = {
                            "loss": f"{windowed_loss:.4f}",
                            "lr": f"{lr:.2e}",
                            "task": task,
                            "gnorm": f"{float(grad_norm_pre):.2f}",
                        }
                        if self.device == "cuda":
                            postfix["mem"] = f"{metrics['gpu_mem_mb']:.0f}MB"
                        pbar.set_postfix(postfix)

                        self._log_step(metrics)

            # Flush remaining accumulated gradients at epoch boundary
            if (step_in_epoch + 1) % self.grad_accum_steps != 0:
                torch.nn.utils.clip_grad_norm_(
                    self.trainable_params,
                    self.max_grad_norm,
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                global_step += 1

            # Epoch summary
            self._log_epoch_summary(
                epoch=epoch,
                global_step=global_step,
                total_loss=epoch_total_loss,
                total_emb_loss=epoch_total_emb_loss,
                total_rerank_loss=epoch_total_rerank_loss,
                emb_steps=epoch_emb_steps,
                rerank_steps=epoch_rerank_steps,
                steps_in_epoch=steps_in_epoch,
            )

        # Save final checkpoint
        final_dir = self.checkpoint_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_adapter(str(final_dir))

        if self._wandb is not None:
            self._wandb.finish()

        final_loss = epoch_total_loss / max(1, steps_in_epoch)
        return {"total_steps": global_step, "final_loss": final_loss}

    def _steps_in_epoch(self) -> int:
        """Number of micro-steps per epoch."""
        if self.mode == TrainingMode.EMB_ONLY:
            return len(self.emb_dataloader) if self.emb_dataloader else 0
        elif self.mode == TrainingMode.RANK_ONLY:
            return len(self.rerank_dataloader) if self.rerank_dataloader else 0
        else:
            emb_len = len(self.emb_dataloader) if self.emb_dataloader else 0
            rerank_len = (
                len(self.rerank_dataloader) if self.rerank_dataloader else 0
            )
            return emb_len + rerank_len

    def _build_task_schedule(self, steps_in_epoch: int) -> list[str]:
        """Build a proportional task schedule for joint training.

        Distributes embedding and reranking steps proportionally to their
        dataloader lengths using Bresenham-style interleaving, so each task
        sees the same number of micro-steps as its specialist equivalent.
        """
        emb_len = len(self.emb_dataloader) if self.emb_dataloader else 0
        rerank_len = len(self.rerank_dataloader) if self.rerank_dataloader else 0
        total = emb_len + rerank_len

        if total == 0:
            return []

        schedule: list[str] = []
        emb_count = 0
        rerank_count = 0

        for i in range(steps_in_epoch):
            # Pick whichever task is furthest behind its ideal proportion,
            # with bounds checks to guarantee exact counts.
            emb_deficit = (emb_len * (i + 1) / total) - emb_count
            rerank_deficit = (rerank_len * (i + 1) / total) - rerank_count

            if emb_count >= emb_len:
                schedule.append("reranking")
                rerank_count += 1
            elif rerank_count >= rerank_len:
                schedule.append("embedding")
                emb_count += 1
            elif emb_deficit >= rerank_deficit:
                schedule.append("embedding")
                emb_count += 1
            else:
                schedule.append("reranking")
                rerank_count += 1

        assert emb_count == emb_len, (
            f"Schedule mismatch: expected {emb_len} embedding steps, got {emb_count}"
        )
        assert rerank_count == rerank_len, (
            f"Schedule mismatch: expected {rerank_len} reranking steps, got {rerank_count}"
        )

        return schedule

    def _embedding_step(self, data_iter) -> torch.Tensor:
        """Single embedding training step."""
        batch = next(data_iter)
        batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        query_emb = self.model.encode(
            batch["query_input_ids"], batch["query_attention_mask"]
        )
        pos_emb = self.model.encode(
            batch["pos_input_ids"], batch["pos_attention_mask"]
        )

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
        batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        logits = self.model.get_rerank_logits(
            batch["input_ids"], batch["attention_mask"]
        )
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
            if (
                param.requires_grad
                and param.grad is not None
                and name in emb_grads
            ):
                emb_g = emb_grads[name].flatten()
                rerank_g = param.grad.flatten()
                cos_sim = torch.nn.functional.cosine_similarity(
                    emb_g.unsqueeze(0), rerank_g.unsqueeze(0)
                ).item()
                conflicts[name] = cos_sim

        self.optimizer.zero_grad()

        return conflicts

    def _log_step(self, metrics: dict):
        """Append a training log entry and optionally push to WandB."""
        with open(self.log_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

        # Human-readable log line (stderr + file)
        parts = [
            f"step={metrics['step']}",
            f"loss={metrics['loss']:.4f}",
        ]
        if "loss_emb" in metrics:
            parts.append(f"loss_emb={metrics['loss_emb']:.4f}")
        if "loss_rerank" in metrics:
            parts.append(f"loss_rerank={metrics['loss_rerank']:.4f}")
        parts.append(f"lr={metrics['lr']:.2e}")
        parts.append(f"gnorm={metrics['grad_norm']:.2f}")
        if "gpu_mem_mb" in metrics:
            parts.append(f"mem={metrics['gpu_mem_mb']:.0f}MB")
        logger.info(" | ".join(parts))

        if self._wandb is not None:
            wandb_metrics = {
                "train/loss": metrics["loss"],
                "train/lr": metrics["lr"],
                "train/grad_norm": metrics["grad_norm"],
                "train/step_time_sec": metrics["step_time_sec"],
            }
            if "loss_emb" in metrics:
                wandb_metrics["train/loss_emb"] = metrics["loss_emb"]
            if "loss_rerank" in metrics:
                wandb_metrics["train/loss_rerank"] = metrics["loss_rerank"]
            if "gpu_mem_mb" in metrics:
                wandb_metrics["system/gpu_mem_mb"] = metrics["gpu_mem_mb"]
                wandb_metrics["system/gpu_mem_peak_mb"] = metrics[
                    "gpu_mem_peak_mb"
                ]
            self._wandb.log(wandb_metrics, step=metrics["step"])

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

        logger.info(
            f"gradient_conflict step={step} | mean_cos_sim={mean_conflict:.4f}"
        )

        if self._wandb is not None:
            self._wandb.log(
                {"gradient_conflict/mean_cosine_sim": mean_conflict},
                step=step,
            )

    def _log_epoch_summary(
        self,
        epoch: int,
        global_step: int,
        total_loss: float,
        total_emb_loss: float,
        total_rerank_loss: float,
        emb_steps: int,
        rerank_steps: int,
        steps_in_epoch: int,
    ):
        """Log epoch-level aggregated statistics."""
        summary = {
            "type": "epoch_summary",
            "epoch": epoch,
            "global_step": global_step,
            "epoch_loss_mean": total_loss / max(1, steps_in_epoch),
            "epoch_steps": steps_in_epoch,
            "timestamp": time.time(),
        }

        if emb_steps > 0:
            summary["epoch_loss_emb_mean"] = total_emb_loss / emb_steps
            summary["epoch_emb_steps"] = emb_steps
        if rerank_steps > 0:
            summary["epoch_loss_rerank_mean"] = total_rerank_loss / rerank_steps
            summary["epoch_rerank_steps"] = rerank_steps

        if self.device == "cuda":
            summary["gpu_mem_peak_mb"] = (
                torch.cuda.max_memory_allocated() / (1024**2)
            )

        with open(self.log_path, "a") as f:
            f.write(json.dumps(summary) + "\n")

        if self._wandb is not None:
            wandb_summary = {
                "epoch/loss_mean": summary["epoch_loss_mean"],
            }
            if "epoch_loss_emb_mean" in summary:
                wandb_summary["epoch/loss_emb_mean"] = summary[
                    "epoch_loss_emb_mean"
                ]
            if "epoch_loss_rerank_mean" in summary:
                wandb_summary["epoch/loss_rerank_mean"] = summary[
                    "epoch_loss_rerank_mean"
                ]
            self._wandb.log(wandb_summary, step=global_step)

        # Epoch summary to logger (stderr + file)
        parts = [f"Epoch {epoch + 1} done"]
        parts.append(f"loss={summary['epoch_loss_mean']:.4f}")
        if "epoch_loss_emb_mean" in summary:
            parts.append(f"loss_emb={summary['epoch_loss_emb_mean']:.4f}")
        if "epoch_loss_rerank_mean" in summary:
            parts.append(
                f"loss_rerank={summary['epoch_loss_rerank_mean']:.4f}"
            )
        if "gpu_mem_peak_mb" in summary:
            parts.append(f"peak_mem={summary['gpu_mem_peak_mb']:.0f}MB")
        logger.info(" | ".join(parts))
