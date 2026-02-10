"""Loss functions for embedding and reranking training.

InfoNCELoss: improved InfoNCE with 5-component denominator and false negative masking,
matching Qwen3-Embedding's training objective.

RerankingSFTLoss: cross-entropy on yes/no token logits for SFT-style reranking training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """Improved InfoNCE loss with hard negatives and false negative masking.

    Five-component denominator:
    1. Positive pair sim(q_i, d_i+)
    2. Explicit hard negatives sim(q_i, d_ij-)
    3. In-batch query-query pairs
    4. In-batch document-document pairs
    5. In-batch cross-pair query-document (excluding own positive)

    False negative masking: mask pairs where sim(q_i, d_j) > sim(q_i, d_i+) + margin.
    """

    def __init__(self, temperature: float = 0.05, fn_margin: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.fn_margin = fn_margin

    def forward(
        self,
        query_embeds: torch.Tensor,
        pos_embeds: torch.Tensor,
        neg_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute InfoNCE loss.

        Args:
            query_embeds: (B, D) L2-normalized query embeddings.
            pos_embeds: (B, D) L2-normalized positive passage embeddings.
            neg_embeds: (B, K, D) L2-normalized hard negative embeddings, or None.

        Returns:
            Scalar loss value.
        """
        batch_size = query_embeds.size(0)
        scale = 1.0 / self.temperature

        # 1. Positive pair similarities: (B,)
        pos_sim = (query_embeds * pos_embeds).sum(dim=1) * scale

        # Collect all negative logits for the denominator
        neg_logits_list = []

        # 2. Explicit hard negatives: (B, K)
        if neg_embeds is not None and neg_embeds.size(1) > 0:
            # (B, 1, D) x (B, K, D) -> (B, K)
            hard_neg_sim = torch.bmm(
                query_embeds.unsqueeze(1), neg_embeds.transpose(1, 2)
            ).squeeze(1) * scale
            neg_logits_list.append(hard_neg_sim)

        # 3-5. In-batch negatives
        # Query-document cross similarity in unscaled cosine space: (B, B)
        cross_sim_raw = torch.mm(query_embeds, pos_embeds.t())

        # Create mask for own positive (diagonal)
        diag_mask = torch.eye(batch_size, device=query_embeds.device, dtype=torch.bool)

        # False negative masking in unscaled similarity space:
        # mask if sim(q_i, d_j) > sim(q_i, d_i+) + margin (margin=0.1 in [-1,1] space)
        pos_sim_raw = (query_embeds * pos_embeds).sum(dim=1).unsqueeze(1)  # (B, 1)
        fn_mask = cross_sim_raw > (pos_sim_raw + self.fn_margin)

        # Apply temperature scaling after mask computation
        cross_sim = cross_sim_raw * scale

        # Combine masks: exclude own positive and false negatives
        exclude_mask = diag_mask | fn_mask
        # Set excluded positions to large negative value
        cross_sim_masked = cross_sim.masked_fill(exclude_mask, float("-inf"))
        neg_logits_list.append(cross_sim_masked)

        # Query-query similarity: (B, B) excluding self
        qq_sim = torch.mm(query_embeds, query_embeds.t()) * scale
        qq_sim_masked = qq_sim.masked_fill(diag_mask, float("-inf"))
        neg_logits_list.append(qq_sim_masked)

        # Document-document similarity: (B, B) excluding self
        dd_sim = torch.mm(pos_embeds, pos_embeds.t()) * scale
        dd_sim_masked = dd_sim.masked_fill(diag_mask, float("-inf"))
        neg_logits_list.append(dd_sim_masked)

        # Concatenate all negative logits: (B, total_neg)
        all_neg_logits = torch.cat(neg_logits_list, dim=1)

        # Full logits: positive + all negatives
        # (B, 1 + total_neg) where position 0 is the positive
        logits = torch.cat([pos_sim.unsqueeze(1), all_neg_logits], dim=1)

        # Target: the positive is always at index 0
        labels = torch.zeros(batch_size, dtype=torch.long, device=query_embeds.device)

        return F.cross_entropy(logits, labels)


class RerankingSFTLoss(nn.Module):
    """Cross-entropy loss on yes/no token logits for SFT-style reranking.

    Extracts only the yes/no logits from full-vocabulary predictions and computes
    binary cross-entropy. This matches the scoring function used at inference
    (P(yes) via softmax over [no, yes] logits) and produces loss ~log(2) initially,
    which is closer in scale to InfoNCE than full-vocabulary CE (~log(vocab_size)).
    """

    def __init__(self, yes_token_id: int, no_token_id: int):
        super().__init__()
        self.yes_token_id = yes_token_id
        self.no_token_id = no_token_id
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute reranking SFT loss over yes/no logits only.

        Args:
            logits: (B, vocab_size) logits at the prediction position.
            labels: (B,) target token IDs (yes_token_id or no_token_id).

        Returns:
            Scalar loss value.
        """
        # Extract only yes/no logits: (B, 2) where index 0=no, 1=yes
        binary_logits = logits[:, [self.no_token_id, self.yes_token_id]]

        # Map token ID labels to binary indices: yes -> 1, no -> 0
        binary_labels = (labels == self.yes_token_id).long()

        return self.loss_fn(binary_logits, binary_labels)
