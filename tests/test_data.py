"""Tests for the data pipeline: templates, collators, and data loading.

Tests verify:
- Template formatting matches Qwen3-Embedding/Reranker specifications
- Collator output shapes, token ID ranges, attention mask values
- Sequence length constraints
- Label values (yes/no token IDs)
- Instruction prefix presence
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
from transformers import AutoTokenizer

from unimoe.config import DataConfig
from unimoe.data.collators import EmbeddingCollator, RerankingCollator
from unimoe.data.templates import (
    DEFAULT_INSTRUCTION,
    RERANKING_SYSTEM_PROMPT,
    format_embedding_query,
    format_reranking_input,
    set_tokenizer_config,
)

# ---------------------------------------------------------------------------
# Template tests
# ---------------------------------------------------------------------------


class TestFormatEmbeddingQuery:
    def test_basic_format(self):
        result = format_embedding_query("Retrieve relevant passages", "what is python")
        assert result == "Instruct: Retrieve relevant passages\nQuery:what is python"

    def test_no_space_after_query_colon(self):
        """Matches Qwen3-Embedding: no space after 'Query:'."""
        result = format_embedding_query("test", "hello")
        assert "Query:hello" in result
        assert "Query: hello" not in result

    def test_space_after_instruct_colon(self):
        result = format_embedding_query("test instruction", "query text")
        assert "Instruct: test instruction" in result

    def test_newline_between_instruct_and_query(self):
        result = format_embedding_query("inst", "q")
        assert "inst\nQuery:" in result


class TestFormatRerankingInput:
    def test_contains_system_prompt(self):
        result = format_reranking_input("inst", "q", "doc")
        assert RERANKING_SYSTEM_PROMPT in result

    def test_contains_im_start_end_tokens(self):
        result = format_reranking_input("inst", "q", "doc")
        assert "<|im_start|>system" in result
        assert "<|im_end|>" in result
        assert "<|im_start|>user" in result
        assert "<|im_start|>assistant" in result

    def test_contains_think_tags(self):
        result = format_reranking_input("inst", "q", "doc")
        assert "<think>\n\n</think>\n\n" in result

    def test_contains_user_content_markers(self):
        result = format_reranking_input("my_inst", "my_query", "my_doc")
        assert "<Instruct>: my_inst" in result
        assert "<Query>: my_query" in result
        assert "<Document>: my_doc" in result

    def test_double_newlines_between_sections(self):
        result = format_reranking_input("inst", "q", "doc")
        assert "<Instruct>: inst\n\n<Query>: q\n\n<Document>: doc" in result


class TestSetTokenizerConfig:
    def test_sets_padding_side_left(self):
        mock_tokenizer = MagicMock()
        set_tokenizer_config(mock_tokenizer)
        assert mock_tokenizer.padding_side == "left"


class TestDefaultInstruction:
    def test_default_instruction_content(self):
        assert "web search query" in DEFAULT_INSTRUCTION
        assert "retrieve relevant passages" in DEFAULT_INSTRUCTION


# ---------------------------------------------------------------------------
# Collator tests (with real Qwen3 tokenizer)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tokenizer():
    """Load Qwen3-0.6B tokenizer for testing."""
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base", trust_remote_code=True)
    tok.padding_side = "left"
    return tok


@pytest.fixture
def data_config():
    return DataConfig()


@pytest.fixture
def embedding_batch():
    """Sample embedding batch with 2 samples, 3 negatives each."""
    return [
        {
            "query": "What is machine learning?",
            "positive": "Machine learning is a subset of artificial intelligence.",
            "negatives": [
                "Deep learning requires neural networks.",
                "Supervised learning uses labeled data.",
                "Reinforcement learning uses rewards.",
            ],
        },
        {
            "query": "How does Python work?",
            "positive": "Python is an interpreted programming language.",
            "negatives": [
                "Java is a compiled language.",
                "C++ is a systems programming language.",
                "JavaScript runs in browsers.",
            ],
        },
    ]


@pytest.fixture
def reranking_batch():
    """Sample reranking batch with yes/no labels."""
    return [
        {"query": "What is AI?", "document": "AI is artificial intelligence.", "label": "yes"},
        {"query": "What is AI?", "document": "Python is a language.", "label": "no"},
        {"query": "Best pizza?", "document": "Margherita pizza is classic.", "label": "yes"},
        {"query": "Best pizza?", "document": "Cats are furry animals.", "label": "no"},
    ]


class TestEmbeddingCollator:
    def test_output_keys(self, tokenizer, data_config, embedding_batch):
        collator = EmbeddingCollator(tokenizer, data_config)
        output = collator(embedding_batch)

        assert "query_input_ids" in output
        assert "query_attention_mask" in output
        assert "pos_input_ids" in output
        assert "pos_attention_mask" in output
        assert "neg_input_ids" in output
        assert "neg_attention_mask" in output

    def test_query_shape(self, tokenizer, data_config, embedding_batch):
        collator = EmbeddingCollator(tokenizer, data_config)
        output = collator(embedding_batch)

        batch_size = len(embedding_batch)
        assert output["query_input_ids"].shape == (batch_size, data_config.query_max_len)
        assert output["query_attention_mask"].shape == (batch_size, data_config.query_max_len)

    def test_positive_shape(self, tokenizer, data_config, embedding_batch):
        collator = EmbeddingCollator(tokenizer, data_config)
        output = collator(embedding_batch)

        batch_size = len(embedding_batch)
        assert output["pos_input_ids"].shape == (batch_size, data_config.passage_max_len)

    def test_negative_shape(self, tokenizer, data_config, embedding_batch):
        collator = EmbeddingCollator(tokenizer, data_config)
        output = collator(embedding_batch)

        batch_size = len(embedding_batch)
        num_negs = len(embedding_batch[0]["negatives"])
        assert output["neg_input_ids"].shape == (
            batch_size,
            num_negs,
            data_config.passage_max_len,
        )

    def test_attention_mask_values(self, tokenizer, data_config, embedding_batch):
        collator = EmbeddingCollator(tokenizer, data_config)
        output = collator(embedding_batch)

        mask = output["query_attention_mask"]
        assert mask.dtype == torch.long
        assert torch.all((mask == 0) | (mask == 1))

    def test_token_id_range(self, tokenizer, data_config, embedding_batch):
        collator = EmbeddingCollator(tokenizer, data_config)
        output = collator(embedding_batch)

        vocab_size = tokenizer.vocab_size
        assert torch.all(output["query_input_ids"] >= 0)
        assert torch.all(output["query_input_ids"] < vocab_size + 1000)  # margin for special

    def test_instruction_prefix_present(self, tokenizer, data_config, embedding_batch):
        """Verify the instruction prefix is in the tokenized query."""
        collator = EmbeddingCollator(tokenizer, data_config)
        output = collator(embedding_batch)

        # Decode the first query and check instruction prefix
        decoded = tokenizer.decode(output["query_input_ids"][0], skip_special_tokens=False)
        assert "Instruct:" in decoded
        assert "Query:" in decoded

    def test_eos_token_appended(self, tokenizer, data_config, embedding_batch):
        """Verify EOS token is at the end of each tokenized sequence (for last-token pooling)."""
        collator = EmbeddingCollator(tokenizer, data_config)
        output = collator(embedding_batch)

        eos_id = tokenizer.eos_token_id
        for key in ["query_input_ids", "pos_input_ids"]:
            ids = output[key]
            mask_key = key.replace("input_ids", "attention_mask")
            mask = output[mask_key]
            for i in range(ids.shape[0]):
                # Find last non-pad position using attention mask
                content_positions = (mask[i] == 1).nonzero(as_tuple=True)[0]
                assert len(content_positions) > 0
                last_content_idx = content_positions[-1].item()
                assert ids[i][last_content_idx].item() == eos_id, (
                    f"Last content token in {key}[{i}] should be EOS ({eos_id}), "
                    f"got {ids[i][last_content_idx].item()}"
                )


class TestRerankingCollator:
    def test_output_keys(self, tokenizer, data_config, reranking_batch):
        collator = RerankingCollator(tokenizer, data_config)
        output = collator(reranking_batch)

        assert "input_ids" in output
        assert "attention_mask" in output
        assert "labels" in output

    def test_shapes(self, tokenizer, data_config, reranking_batch):
        collator = RerankingCollator(tokenizer, data_config)
        output = collator(reranking_batch)

        batch_size = len(reranking_batch)
        seq_len = output["input_ids"].shape[1]

        assert output["input_ids"].shape == (batch_size, seq_len)
        assert output["attention_mask"].shape == (batch_size, seq_len)
        assert output["labels"].shape == (batch_size,)
        assert seq_len <= data_config.reranking_max_len

    def test_label_values(self, tokenizer, data_config, reranking_batch):
        collator = RerankingCollator(tokenizer, data_config)
        output = collator(reranking_batch)

        yes_id = tokenizer.convert_tokens_to_ids("yes")
        no_id = tokenizer.convert_tokens_to_ids("no")

        for i, sample in enumerate(reranking_batch):
            if sample["label"] == "yes":
                assert output["labels"][i].item() == yes_id
            else:
                assert output["labels"][i].item() == no_id

    def test_attention_mask_values(self, tokenizer, data_config, reranking_batch):
        collator = RerankingCollator(tokenizer, data_config)
        output = collator(reranking_batch)

        mask = output["attention_mask"]
        assert torch.all((mask == 0) | (mask == 1))

    def test_left_padding(self, tokenizer, data_config, reranking_batch):
        """Verify left padding: non-padded tokens are at the right side."""
        collator = RerankingCollator(tokenizer, data_config)
        output = collator(reranking_batch)

        # For each sample, padding (0s) should be on the left
        for i in range(len(reranking_batch)):
            mask = output["attention_mask"][i]
            # Find first non-zero position
            nonzero_positions = torch.nonzero(mask, as_tuple=True)[0]
            if len(nonzero_positions) > 0:
                first_nonzero = nonzero_positions[0].item()
                # All positions before first_nonzero should be 0
                assert torch.all(mask[:first_nonzero] == 0)
                # All positions from first_nonzero onward should be 1
                assert torch.all(mask[first_nonzero:] == 1)

    def test_chat_template_tokens_present(self, tokenizer, data_config, reranking_batch):
        """Verify the chat template structure is in the tokenized output."""
        collator = RerankingCollator(tokenizer, data_config)
        output = collator(reranking_batch)

        # Decode the first sample
        decoded = tokenizer.decode(output["input_ids"][0], skip_special_tokens=False)
        assert "<|im_start|>" in decoded
        assert "<|im_end|>" in decoded
        assert "Judge whether" in decoded

    def test_yes_no_token_ids_are_valid(self, tokenizer, data_config):
        """Verify yes/no tokens are single tokens in the vocabulary."""
        collator = RerankingCollator(tokenizer, data_config)
        assert isinstance(collator.yes_token_id, int)
        assert isinstance(collator.no_token_id, int)
        assert collator.yes_token_id != collator.no_token_id
        assert collator.yes_token_id > 0
        assert collator.no_token_id > 0


class TestDataLoaderIntegration:
    """Test that collators work with PyTorch DataLoader."""

    def test_embedding_dataloader(self, tokenizer, data_config, embedding_batch):
        from torch.utils.data import DataLoader

        collator = EmbeddingCollator(tokenizer, data_config)
        loader = DataLoader(embedding_batch, batch_size=2, collate_fn=collator)

        batch = next(iter(loader))
        assert batch["query_input_ids"].shape[0] == 2
        assert batch["pos_input_ids"].shape[0] == 2

    def test_reranking_dataloader(self, tokenizer, data_config, reranking_batch):
        from torch.utils.data import DataLoader

        collator = RerankingCollator(tokenizer, data_config)
        loader = DataLoader(reranking_batch, batch_size=4, collate_fn=collator)

        batch = next(iter(loader))
        assert batch["input_ids"].shape[0] == 4
        assert batch["labels"].shape[0] == 4
