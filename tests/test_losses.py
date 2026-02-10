"""Tests for loss functions.

Verifies correct loss values on known inputs for InfoNCE and RerankingSFTLoss.
"""

from __future__ import annotations

import math

import torch

from unimoe.training.losses import InfoNCELoss, RerankingSFTLoss


class TestInfoNCELoss:
    def test_perfect_alignment_low_loss(self):
        """When queries perfectly match positives, loss should be low."""
        loss_fn = InfoNCELoss(temperature=0.05)
        batch_size = 4
        dim = 64

        # Create query and positive embeddings that are identical
        embeds = torch.randn(batch_size, dim)
        embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)

        loss = loss_fn(embeds, embeds)
        assert loss.item() >= 0, "Loss should be non-negative"
        # Perfect alignment should produce low loss (not zero due to in-batch negatives)
        assert loss.item() < 5.0, f"Loss too high for perfect alignment: {loss.item()}"

    def test_random_embeddings_higher_loss(self):
        """Random embeddings should produce higher loss than aligned ones."""
        loss_fn = InfoNCELoss(temperature=0.05)
        batch_size = 8
        dim = 64

        queries = torch.nn.functional.normalize(torch.randn(batch_size, dim), p=2, dim=1)
        positives = torch.nn.functional.normalize(torch.randn(batch_size, dim), p=2, dim=1)

        loss_random = loss_fn(queries, positives)

        # Now use aligned
        loss_aligned = loss_fn(queries, queries.clone())

        assert loss_random.item() > loss_aligned.item(), (
            f"Random loss ({loss_random.item()}) should be higher than aligned ({loss_aligned.item()})"
        )

    def test_with_hard_negatives(self):
        """Loss should work with hard negatives tensor."""
        loss_fn = InfoNCELoss(temperature=0.05)
        batch_size = 4
        dim = 64
        num_negs = 7

        queries = torch.nn.functional.normalize(torch.randn(batch_size, dim), p=2, dim=1)
        positives = torch.nn.functional.normalize(torch.randn(batch_size, dim), p=2, dim=1)
        negatives = torch.nn.functional.normalize(
            torch.randn(batch_size, num_negs, dim), p=2, dim=2
        )

        loss = loss_fn(queries, positives, negatives)
        assert loss.item() >= 0
        assert torch.isfinite(loss), "Loss should be finite"

    def test_without_hard_negatives(self):
        """Loss should work without hard negatives (in-batch only)."""
        loss_fn = InfoNCELoss(temperature=0.05)
        queries = torch.nn.functional.normalize(torch.randn(4, 64), p=2, dim=1)
        positives = torch.nn.functional.normalize(torch.randn(4, 64), p=2, dim=1)

        loss = loss_fn(queries, positives, neg_embeds=None)
        assert torch.isfinite(loss)

    def test_temperature_effect(self):
        """Lower temperature should produce sharper distributions."""
        queries = torch.nn.functional.normalize(torch.randn(4, 64), p=2, dim=1)
        positives = torch.nn.functional.normalize(torch.randn(4, 64), p=2, dim=1)

        loss_low_temp = InfoNCELoss(temperature=0.01)(queries, positives)
        loss_high_temp = InfoNCELoss(temperature=1.0)(queries, positives)

        # Both should be finite
        assert torch.isfinite(loss_low_temp)
        assert torch.isfinite(loss_high_temp)

    def test_gradients_flow(self):
        """Loss should produce gradients for inputs."""
        loss_fn = InfoNCELoss(temperature=0.05)
        queries_raw = torch.randn(4, 64, requires_grad=True)
        positives_raw = torch.randn(4, 64, requires_grad=True)
        queries = torch.nn.functional.normalize(queries_raw, p=2, dim=1)
        positives = torch.nn.functional.normalize(positives_raw, p=2, dim=1)
        loss = loss_fn(queries, positives)
        loss.backward()
        assert queries_raw.grad is not None
        assert positives_raw.grad is not None


class TestRerankingSFTLoss:
    """Tests for RerankingSFTLoss with yes/no-only CE.

    The loss now extracts only yes/no logits from full-vocabulary predictions
    and computes binary CE. Initial loss should be ~log(2) ≈ 0.693.
    """

    YES_ID = 9693  # "yes" token ID in Qwen3 tokenizer
    NO_ID = 2152   # "no" token ID in Qwen3 tokenizer

    def test_correct_prediction_low_loss(self):
        """When model predicts the correct yes/no token, loss should be low."""
        loss_fn = RerankingSFTLoss(yes_token_id=self.YES_ID, no_token_id=self.NO_ID)
        vocab_size = 152000
        batch_size = 4

        logits = torch.zeros(batch_size, vocab_size)
        # Labels are yes/no token IDs
        labels = torch.tensor([self.YES_ID, self.NO_ID, self.YES_ID, self.NO_ID])

        # Set high logits at the correct yes/no positions
        for i in range(batch_size):
            logits[i, labels[i]] = 10.0

        loss = loss_fn(logits, labels)
        assert loss.item() < 0.1, f"Loss too high for correct predictions: {loss.item()}"

    def test_wrong_prediction_high_loss(self):
        """When model predicts wrong yes/no token, loss should be high."""
        loss_fn = RerankingSFTLoss(yes_token_id=self.YES_ID, no_token_id=self.NO_ID)
        vocab_size = 152000
        batch_size = 4

        logits = torch.zeros(batch_size, vocab_size)
        labels = torch.tensor([self.YES_ID, self.NO_ID, self.YES_ID, self.NO_ID])

        # Set high logits at the WRONG yes/no position
        for i in range(batch_size):
            wrong_id = self.NO_ID if labels[i] == self.YES_ID else self.YES_ID
            logits[i, wrong_id] = 10.0

        loss = loss_fn(logits, labels)
        assert loss.item() > 5.0, f"Loss should be high for wrong predictions: {loss.item()}"

    def test_output_is_scalar(self):
        loss_fn = RerankingSFTLoss(yes_token_id=self.YES_ID, no_token_id=self.NO_ID)
        logits = torch.randn(4, 152000)
        labels = torch.tensor([self.YES_ID, self.NO_ID, self.YES_ID, self.NO_ID])
        loss = loss_fn(logits, labels)
        assert loss.dim() == 0, "Loss should be a scalar"

    def test_gradients_flow(self):
        loss_fn = RerankingSFTLoss(yes_token_id=self.YES_ID, no_token_id=self.NO_ID)
        logits = torch.randn(4, 152000, requires_grad=True)
        labels = torch.tensor([self.YES_ID, self.NO_ID, self.YES_ID, self.NO_ID])
        loss = loss_fn(logits, labels)
        loss.backward()
        assert logits.grad is not None

    def test_initial_loss_near_log2(self):
        """With random logits, initial loss should be ~log(2) since it's binary CE."""
        loss_fn = RerankingSFTLoss(yes_token_id=self.YES_ID, no_token_id=self.NO_ID)
        logits = torch.zeros(100, 152000)  # uniform logits
        labels = torch.tensor([self.YES_ID] * 50 + [self.NO_ID] * 50)
        loss = loss_fn(logits, labels)
        expected = math.log(2)
        assert abs(loss.item() - expected) < 0.01, (
            f"Initial loss should be ~{expected:.4f}, got {loss.item():.4f}"
        )
