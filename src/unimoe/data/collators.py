"""Batch collators for embedding and reranking tasks.

EmbeddingCollator: tokenizes query (with instruction prefix), positive, and hard negatives
separately, appends EOS, and pads to max length.

RerankingCollator: uses pre-tokenized prefix/suffix concatenation (NOT full-string
tokenization) to match BPE boundary behavior of Qwen3-Reranker's official scoring code.
"""

from __future__ import annotations

import torch
from transformers import PreTrainedTokenizer

from unimoe.config import DataConfig
from unimoe.data.templates import (
    build_reranking_token_ids,
    format_embedding_query,
)


class EmbeddingCollator:
    """Collator for embedding training batches.

    Tokenizes query (with instruction prefix), positive passage, and hard negatives
    separately. Each sequence has EOS appended (for last-token pooling).
    Left-padded for causal attention / flash_attention_2 compatibility.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, config: DataConfig):
        if tokenizer.padding_side != "left":
            raise ValueError(
                f"EmbeddingCollator requires left-padding for last-token pooling, "
                f"got padding_side='{tokenizer.padding_side}'. "
                f"Call set_tokenizer_config(tokenizer) first."
            )
        self.tokenizer = tokenizer
        self.query_max_len = config.query_max_len
        self.passage_max_len = config.passage_max_len
        self.instruction = config.instruction_prefix
        # EOS token for last-token pooling (Qwen3 uses <|endoftext|>, ID 151643)
        self.eos_token = tokenizer.eos_token

    def _tokenize(self, texts: list[str], max_len: int) -> dict[str, torch.Tensor]:
        """Tokenize a list of texts with EOS appended and left padding.

        Appends the EOS token to each text before tokenization so that
        last-token pooling at the EOS position produces correct embeddings
        (matching Qwen3-Embedding behavior).
        """
        texts_with_eos = [text + self.eos_token for text in texts]
        return self.tokenizer(
            texts_with_eos,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        queries = [
            format_embedding_query(self.instruction, sample["query"]) for sample in batch
        ]
        positives = [sample["positive"] for sample in batch]

        query_enc = self._tokenize(queries, self.query_max_len)
        pos_enc = self._tokenize(positives, self.passage_max_len)

        result = {
            "query_input_ids": query_enc["input_ids"],
            "query_attention_mask": query_enc["attention_mask"],
            "pos_input_ids": pos_enc["input_ids"],
            "pos_attention_mask": pos_enc["attention_mask"],
        }

        # Tokenize hard negatives — each sample has a list of negative passages
        if "negatives" in batch[0] and batch[0]["negatives"]:
            num_negatives = len(batch[0]["negatives"])
            neg_input_ids_list = []
            neg_attention_mask_list = []

            for neg_idx in range(num_negatives):
                neg_texts = [sample["negatives"][neg_idx] for sample in batch]
                neg_enc = self._tokenize(neg_texts, self.passage_max_len)
                neg_input_ids_list.append(neg_enc["input_ids"])
                neg_attention_mask_list.append(neg_enc["attention_mask"])

            # Stack: (num_negatives, batch_size, seq_len) -> (batch_size, num_negatives, seq_len)
            result["neg_input_ids"] = torch.stack(neg_input_ids_list, dim=1)
            result["neg_attention_mask"] = torch.stack(neg_attention_mask_list, dim=1)

        return result


class RerankingCollator:
    """Collator for reranking training batches.

    Uses pre-tokenized prefix/suffix concatenation to match the official
    Qwen3-Reranker scoring code. BPE tokenization is context-sensitive at
    boundary characters, so we pre-tokenize the template prefix and suffix
    once, tokenize only the user content per sample, then concatenate at
    the token-ID level.

    Prefix: "<|im_start|>system\\n{SYSTEM_PROMPT}<|im_end|>\\n<|im_start|>user\\n"
    User content: "<Instruct>: {instruction}\\n\\n<Query>: {query}\\n\\n<Document>: {document}"
    Suffix: "<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n"
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, config: DataConfig):
        if tokenizer.padding_side != "left":
            raise ValueError(
                f"RerankingCollator requires left-padding for causal attention, "
                f"got padding_side='{tokenizer.padding_side}'. "
                f"Call set_tokenizer_config(tokenizer) first."
            )
        self.tokenizer = tokenizer
        self.max_len = config.reranking_max_len
        self.instruction = config.instruction_prefix

        # Resolve yes/no token IDs
        self.yes_token_id = tokenizer.convert_tokens_to_ids("yes")
        self.no_token_id = tokenizer.convert_tokens_to_ids("no")

    def _build_input(self, query: str, document: str) -> list[int]:
        """Build token IDs for a single query-document pair via concatenation."""
        return build_reranking_token_ids(
            self.tokenizer, self.instruction, query, document, self.max_len
        )

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        all_ids = []
        labels = []

        for sample in batch:
            ids = self._build_input(sample["query"], sample["document"])
            all_ids.append(ids)

            label_str = sample["label"]
            if label_str == "yes" or label_str is True or label_str == 1:
                labels.append(self.yes_token_id)
            else:
                labels.append(self.no_token_id)

        # Left-pad to max length in batch
        max_seq_len = min(max(len(ids) for ids in all_ids), self.max_len)
        pad_id = self.tokenizer.pad_token_id

        input_ids = []
        attention_masks = []
        for ids in all_ids:
            ids = ids[:max_seq_len]  # ensure within bounds
            pad_len = max_seq_len - len(ids)
            input_ids.append([pad_id] * pad_len + ids)
            attention_masks.append([0] * pad_len + [1] * len(ids))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
