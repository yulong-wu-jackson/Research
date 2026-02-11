"""UniMoE model: Qwen3-0.6B-Base + LoRA for dual-task (embedding + reranking).

Embedding mode: last-token (EOS) pooling + L2 normalization (Qwen3-Embedding)
Reranking mode: yes/no token logit scoring (Qwen3-Reranker)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from unimoe.config import ExperimentConfig


def _find_last_content_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    """Find the last position where attention_mask == 1 for each sequence.

    Works correctly with left-padded sequences where content is right-aligned.
    Uses argmax on position indices masked by attention_mask.
    """
    seq_len = attention_mask.size(1)
    positions = torch.arange(seq_len, device=attention_mask.device).unsqueeze(0)
    return (positions * attention_mask - (1 - attention_mask)).argmax(dim=1)


class UnimodelForExp1(nn.Module):
    """Unified model for Experiment 1: shared LoRA for embedding and reranking.

    Wraps Qwen3-0.6B-Base with LoRA adapters on all linear layers.
    Supports two modes:
    - encode(): last-token (EOS) pooling -> L2-normalized embeddings (B, 1024)
    - rerank(): yes/no token logit scoring -> relevance scores (B,)
    """

    def __init__(self, config: ExperimentConfig, tokenizer=None):
        super().__init__()
        self.config = config

        device = config.model.resolve_device()
        dtype = config.model.resolve_dtype()

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model.base_model_name,
            dtype=dtype,
            trust_remote_code=True,
        )

        # Apply LoRA via PEFT
        lora_config = LoraConfig(
            r=config.lora.rank,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            target_modules=config.lora.target_modules,
            bias=config.lora.bias,
            task_type=None,  # We manage task modes manually
        )
        self.model = get_peft_model(base_model, lora_config)

        # Load tokenizer if not provided
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                config.model.base_model_name, trust_remote_code=True
            )
        self.tokenizer = tokenizer

        # Resolve yes/no token IDs for reranking
        self.yes_token_id = tokenizer.convert_tokens_to_ids("yes")
        self.no_token_id = tokenizer.convert_tokens_to_ids("no")

        if self.yes_token_id is None or self.no_token_id is None:
            raise ValueError(
                "Tokenizer does not contain 'yes' or 'no' as single tokens. "
                f"Got yes_id={self.yes_token_id}, no_id={self.no_token_id}"
            )
        unk_id = getattr(tokenizer, "unk_token_id", None)
        if unk_id is not None and (self.yes_token_id == unk_id or self.no_token_id == unk_id):
            raise ValueError(
                "'yes' or 'no' resolved to the unknown token — tokenizer mismatch"
            )

        # EOS token for pooling: use pad_token_id (151643, <|endoftext|>)
        # NOT eos_token_id which is 151645 (<|im_end|>) in some configs
        self.pool_token_id = tokenizer.pad_token_id

        # Move to device
        self.model.to(device)

    def encode(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Generate L2-normalized embeddings via last-token (EOS) pooling.

        Args:
            input_ids: (B, seq_len) token IDs with EOS appended.
            attention_mask: (B, seq_len) attention mask (1 for content, 0 for padding).

        Returns:
            (B, hidden_size) L2-normalized embedding vectors.
        """
        # Access inner transformer model (bypassing LM head)
        outputs = self.model.base_model.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state  # (B, seq_len, hidden_size)

        # Last-token pooling: find the last content position per sequence
        last_positions = _find_last_content_positions(attention_mask)
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        pooled = hidden_states[batch_indices, last_positions]  # (B, hidden_size)

        # L2 normalize
        embeddings = F.normalize(pooled, p=2, dim=1)
        return embeddings

    def rerank(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute relevance scores via yes/no token logit scoring.

        Matches Qwen3-Reranker inference:
        score = softmax(logit_no, logit_yes)[yes_index] = P(yes)

        Args:
            input_ids: (B, seq_len) token IDs for reranking.
            attention_mask: (B, seq_len) attention mask.

        Returns:
            (B,) relevance scores (probability of "yes").
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        logits = outputs.logits  # (B, seq_len, vocab_size)

        # Get logits at the last content position (next token prediction)
        last_positions = _find_last_content_positions(attention_mask)
        batch_indices = torch.arange(logits.size(0), device=logits.device)
        last_logits = logits[batch_indices, last_positions]  # (B, vocab_size)

        # Yes/no scoring matching Qwen3-Reranker
        true_vector = last_logits[:, self.yes_token_id]
        false_vector = last_logits[:, self.no_token_id]
        scores = torch.stack([false_vector, true_vector], dim=1)  # (B, 2)
        scores = F.log_softmax(scores, dim=1)
        return scores[:, 1].exp()  # P(yes)

    def get_rerank_logits(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Get raw logits at yes/no positions for training loss computation.

        Returns the full logits at the last position for cross-entropy loss.

        Args:
            input_ids: (B, seq_len)
            attention_mask: (B, seq_len)

        Returns:
            (B, vocab_size) logits at the last position.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        logits = outputs.logits
        last_positions = _find_last_content_positions(attention_mask)
        batch_indices = torch.arange(logits.size(0), device=logits.device)
        return logits[batch_indices, last_positions]  # (B, vocab_size)

    def save_adapter(self, path: str) -> None:
        """Save the PEFT adapter weights."""
        self.model.save_pretrained(path)

    @classmethod
    def load_from_checkpoint(
        cls, config: ExperimentConfig, checkpoint_path: str, tokenizer=None
    ) -> "UnimodelForExp1":
        """Load model from a saved PEFT adapter checkpoint."""
        from peft import PeftModel

        device = config.model.resolve_device()
        dtype = config.model.resolve_dtype()

        base_model = AutoModelForCausalLM.from_pretrained(
            config.model.base_model_name,
            dtype=dtype,
            trust_remote_code=True,
        )

        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance.config = config

        instance.model = PeftModel.from_pretrained(base_model, checkpoint_path)
        instance.model.to(device)

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                config.model.base_model_name, trust_remote_code=True
            )
        instance.tokenizer = tokenizer
        instance.yes_token_id = tokenizer.convert_tokens_to_ids("yes")
        instance.no_token_id = tokenizer.convert_tokens_to_ids("no")
        instance.pool_token_id = tokenizer.pad_token_id

        return instance
