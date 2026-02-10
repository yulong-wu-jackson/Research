"""Tests for gradient conflict computation.

Verifies gradient conflict returns valid cosine similarity values in [-1, 1] per layer.
"""

from __future__ import annotations

import pytest
import torch.nn.functional as F
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
        training=TrainingConfig(mode=TrainingMode.JOINT_SINGLE),
    )


@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base", trust_remote_code=True)
    tok.padding_side = "left"
    return tok


@pytest.fixture(scope="module")
def model(config, tokenizer):
    return UnimodelForExp1(config, tokenizer=tokenizer)


class TestGradientConflict:
    def test_gradient_conflict_values_in_range(self, model, tokenizer):
        """Gradient conflict cosine similarities should be in [-1, 1]."""
        emb_loss_fn = InfoNCELoss(temperature=0.05)
        rerank_loss_fn = RerankingSFTLoss(
            yes_token_id=model.yes_token_id, no_token_id=model.no_token_id
        )
        data_config = DataConfig()

        # Embedding forward + backward
        queries = tokenizer(
            ["query one", "query two"],
            padding="max_length", max_length=32, return_tensors="pt",
        )
        positives = tokenizer(
            ["positive doc", "another positive"],
            padding="max_length", max_length=32, return_tensors="pt",
        )

        model.zero_grad()
        query_emb = model.encode(queries["input_ids"], queries["attention_mask"])
        pos_emb = model.encode(positives["input_ids"], positives["attention_mask"])
        emb_loss = emb_loss_fn(query_emb, pos_emb)
        emb_loss.backward()

        # Store embedding gradients
        emb_grads = {}
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                emb_grads[name] = param.grad.clone()

        # Reranking forward + backward
        model.zero_grad()
        collator = RerankingCollator(tokenizer, data_config)
        batch_data = [
            {"query": "What is AI?", "document": "AI is artificial intelligence.", "label": "yes"},
            {"query": "Best pizza?", "document": "Cats are animals.", "label": "no"},
        ]
        batch = collator(batch_data)
        logits = model.get_rerank_logits(batch["input_ids"], batch["attention_mask"])
        rerank_loss = rerank_loss_fn(logits, batch["labels"])
        rerank_loss.backward()

        # Compute per-layer cosine similarity
        conflicts = {}
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None and name in emb_grads:
                emb_g = emb_grads[name].flatten()
                rerank_g = param.grad.flatten()
                cos_sim = F.cosine_similarity(
                    emb_g.unsqueeze(0), rerank_g.unsqueeze(0)
                ).item()
                conflicts[name] = cos_sim

        # Verify all values are in valid range
        assert len(conflicts) > 0, "No gradient conflicts computed"
        for name, cos_sim in conflicts.items():
            assert -1.0 <= cos_sim <= 1.0, (
                f"Cosine similarity for {name} out of range: {cos_sim}"
            )

        model.zero_grad()

    def test_gradient_conflict_has_per_layer_values(self, model, tokenizer):
        """Should have conflict values for multiple layers."""
        emb_loss_fn = InfoNCELoss(temperature=0.05)
        rerank_loss_fn = RerankingSFTLoss(
            yes_token_id=model.yes_token_id, no_token_id=model.no_token_id
        )
        data_config = DataConfig()

        # Quick forward/backward for both tasks
        model.zero_grad()
        queries = tokenizer(["q1"], padding="max_length", max_length=16, return_tensors="pt")
        pos = tokenizer(["d1"], padding="max_length", max_length=16, return_tensors="pt")
        q_emb = model.encode(queries["input_ids"], queries["attention_mask"])
        p_emb = model.encode(pos["input_ids"], pos["attention_mask"])
        emb_loss_fn(q_emb, p_emb).backward()

        emb_grads = {
            n: p.grad.clone()
            for n, p in model.named_parameters()
            if p.requires_grad and p.grad is not None
        }

        model.zero_grad()
        collator = RerankingCollator(tokenizer, data_config)
        batch = collator([
            {"query": "q", "document": "d", "label": "yes"},
        ])
        logits = model.get_rerank_logits(batch["input_ids"], batch["attention_mask"])
        rerank_loss_fn(logits, batch["labels"]).backward()

        count = 0
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None and n in emb_grads:
                count += 1

        # Should have conflicts for many layers (LoRA on 7 modules × 28 layers)
        assert count > 10, f"Too few layers with conflicts: {count}"

        model.zero_grad()
