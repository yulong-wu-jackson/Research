"""Tests for training step mechanics.

Verifies:
- One training step completes without error
- Loss is finite
- Gradients flow only to trainable params
"""

from __future__ import annotations

import pytest
import torch
from transformers import AutoTokenizer

from unimoe.config import (
    DataConfig,
    ExperimentConfig,
    LoRAConfig,
    ModelConfig,
    TrainingConfig,
    TrainingMode,
)
from unimoe.data.collators import RerankingCollator
from unimoe.model.lora_model import UnimodelForExp1
from unimoe.training.losses import InfoNCELoss, RerankingSFTLoss


@pytest.fixture(scope="module")
def config():
    return ExperimentConfig(
        model=ModelConfig(
            base_model_name="Qwen/Qwen3-0.6B-Base",
            torch_dtype="float32",
            device="cpu",
        ),
        lora=LoRAConfig(rank=4, alpha=4, dropout=0.0),
        training=TrainingConfig(mode=TrainingMode.RANK_ONLY),
    )


@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base", trust_remote_code=True)
    tok.padding_side = "left"
    return tok


@pytest.fixture(scope="module")
def model(config, tokenizer):
    return UnimodelForExp1(config, tokenizer=tokenizer)


class TestEmbeddingTrainingStep:
    def test_one_step_completes(self, model, tokenizer):
        loss_fn = InfoNCELoss(temperature=0.05)

        # Create mock batch
        queries = tokenizer(
            ["query one", "query two"],
            padding="max_length", max_length=32, return_tensors="pt",
        )
        positives = tokenizer(
            ["positive doc one", "positive doc two"],
            padding="max_length", max_length=32, return_tensors="pt",
        )

        query_emb = model.encode(queries["input_ids"], queries["attention_mask"])
        pos_emb = model.encode(positives["input_ids"], positives["attention_mask"])

        loss = loss_fn(query_emb, pos_emb)
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

        loss.backward()

        # Check gradients flow to LoRA params
        has_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                assert torch.isfinite(param.grad).all(), f"Non-finite grad in {name}"
        assert has_grad, "No gradients computed"

        # Check no gradients on frozen params
        for name, param in model.named_parameters():
            if not param.requires_grad:
                assert param.grad is None, f"Frozen param {name} has gradients"

        model.zero_grad()


class TestTaskSchedule:
    """Verify proportional task scheduling for joint training."""

    def _make_trainer_with_loaders(self, model, emb_len, rerank_len):
        """Create a trainer with mock dataloaders of given lengths."""
        from unittest.mock import MagicMock

        joint_config = ExperimentConfig(
            model=ModelConfig(
                base_model_name="Qwen/Qwen3-0.6B-Base",
                torch_dtype="float32",
                device="cpu",
            ),
            lora=LoRAConfig(rank=4, alpha=4, dropout=0.0),
            training=TrainingConfig(mode=TrainingMode.JOINT_SINGLE),
        )

        emb_loader = MagicMock()
        emb_loader.__len__ = MagicMock(return_value=emb_len)
        rerank_loader = MagicMock()
        rerank_loader.__len__ = MagicMock(return_value=rerank_len)

        from unimoe.training.trainer import UnifiedTrainer

        trainer = UnifiedTrainer(
            config=joint_config,
            model=model,
            emb_dataloader=emb_loader,
            rerank_dataloader=rerank_loader,
        )
        return trainer

    def test_equal_lengths(self, model):
        trainer = self._make_trainer_with_loaders(model, 100, 100)
        schedule = trainer._build_task_schedule(200)
        assert schedule.count("embedding") == 100
        assert schedule.count("reranking") == 100

    def test_2_to_1_ratio(self, model):
        """emb_len=1250, rerank_len=625 (matches r8 config batch sizes)."""
        trainer = self._make_trainer_with_loaders(model, 1250, 625)
        schedule = trainer._build_task_schedule(1875)
        assert schedule.count("embedding") == 1250
        assert schedule.count("reranking") == 625

    def test_10_to_1_ratio(self, model):
        trainer = self._make_trainer_with_loaders(model, 1000, 100)
        schedule = trainer._build_task_schedule(1100)
        assert schedule.count("embedding") == 1000
        assert schedule.count("reranking") == 100

    def test_interleaving(self, model):
        """Tasks should be interleaved, not all-emb then all-rerank."""
        trainer = self._make_trainer_with_loaders(model, 100, 100)
        schedule = trainer._build_task_schedule(200)
        # Check first 4 steps alternate
        assert schedule[0] == "embedding"
        assert schedule[1] == "reranking"
        assert schedule[2] == "embedding"
        assert schedule[3] == "reranking"

    def test_steps_in_epoch_sum(self, model):
        """_steps_in_epoch should return sum of both loader lengths."""
        trainer = self._make_trainer_with_loaders(model, 1250, 625)
        assert trainer._steps_in_epoch() == 1875


class TestRerankingTrainingStep:
    def test_one_step_completes(self, model, tokenizer):
        loss_fn = RerankingSFTLoss(
            yes_token_id=model.yes_token_id, no_token_id=model.no_token_id
        )
        config = DataConfig()
        collator = RerankingCollator(tokenizer, config)

        batch_data = [
            {"query": "What is AI?", "document": "AI is artificial intelligence.", "label": "yes"},
            {"query": "What is AI?", "document": "Python is a language.", "label": "no"},
        ]

        batch = collator(batch_data)
        logits = model.get_rerank_logits(batch["input_ids"], batch["attention_mask"])
        loss = loss_fn(logits, batch["labels"])

        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

        loss.backward()

        has_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
        assert has_grad, "No gradients computed"

        model.zero_grad()
