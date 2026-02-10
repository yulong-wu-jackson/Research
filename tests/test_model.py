"""Tests for the UnimodelForExp1 model.

Verifies:
- Only LoRA params are trainable (base + LM head frozen)
- encode() returns L2-normalized vectors of shape (B, 1024)
- rerank() returns scalars of shape (B,) using yes/no logit scoring
- LoRA applied to all 7 module types
- Forward pass completes without error
"""

from __future__ import annotations

import pytest
import torch
from transformers import AutoTokenizer

from unimoe.config import ExperimentConfig, LoRAConfig, ModelConfig
from unimoe.model.lora_model import UnimodelForExp1


@pytest.fixture(scope="module")
def small_config():
    """Config with small LoRA rank for fast testing."""
    return ExperimentConfig(
        model=ModelConfig(
            base_model_name="Qwen/Qwen3-0.6B-Base",
            torch_dtype="float32",
            device="cpu",
        ),
        lora=LoRAConfig(rank=4, alpha=4, dropout=0.0),
    )


@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base", trust_remote_code=True)
    tok.padding_side = "left"
    return tok


@pytest.fixture(scope="module")
def model(small_config, tokenizer):
    return UnimodelForExp1(small_config, tokenizer=tokenizer)


@pytest.fixture
def sample_input(tokenizer):
    """Create a simple tokenized input batch."""
    texts = ["Hello world", "Test query for embedding"]
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=32,
        return_tensors="pt",
    )
    return encoded["input_ids"], encoded["attention_mask"]


class TestModelStructure:
    def test_base_params_frozen(self, model):
        """Base model parameters should not require gradients."""
        for name, param in model.model.named_parameters():
            if "lora_" not in name:
                assert not param.requires_grad, f"Base param {name} should be frozen"

    def test_lora_params_trainable(self, model):
        """LoRA parameters should require gradients."""
        lora_params = [
            (n, p) for n, p in model.model.named_parameters() if "lora_" in n
        ]
        assert len(lora_params) > 0, "No LoRA parameters found"
        for name, param in lora_params:
            assert param.requires_grad, f"LoRA param {name} should be trainable"

    def test_lm_head_frozen(self, model):
        """LM head should be frozen (intentional deviation from full SFT)."""
        for name, param in model.model.named_parameters():
            if "lm_head" in name:
                assert not param.requires_grad, f"LM head param {name} should be frozen"

    def test_lora_applied_to_all_modules(self, model):
        """LoRA should be applied to all 7 target module types."""
        lora_param_names = [
            n for n, _ in model.model.named_parameters() if "lora_" in n
        ]
        target_modules = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
        found_modules = set()
        for name in lora_param_names:
            for target in target_modules:
                if target in name:
                    found_modules.add(target)
        assert found_modules == target_modules, (
            f"Missing LoRA on: {target_modules - found_modules}"
        )

    def test_trainable_param_count(self, model):
        """Trainable params should be reasonable for LoRA (much less than total)."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable_params < total_params * 0.1, "LoRA should be <10% of total params"
        assert trainable_params > 0, "Must have some trainable params"


class TestEncode:
    def test_output_shape(self, model, sample_input):
        input_ids, attention_mask = sample_input
        with torch.no_grad():
            embeddings = model.encode(input_ids, attention_mask)
        assert embeddings.shape == (2, 1024), f"Expected (2, 1024), got {embeddings.shape}"

    def test_l2_normalized(self, model, sample_input):
        input_ids, attention_mask = sample_input
        with torch.no_grad():
            embeddings = model.encode(input_ids, attention_mask)
        norms = torch.norm(embeddings, p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), (
            f"Embeddings not L2-normalized: norms = {norms}"
        )

    def test_different_inputs_different_embeddings(self, model, tokenizer):
        """Different inputs should produce different embeddings.

        Note: with left-padding and LoRA B=0 initialization, short texts padded
        to the same length may produce identical embeddings. Use longer, more
        distinct inputs to ensure the content tokens differ meaningfully.
        """
        texts_a = [
            "The cat sat on the mat and stared at the wall for hours on end without blinking"
        ]
        texts_b = [
            "Quantum mechanics describes the behavior of particles at subatomic scale with waves"
        ]
        # Use tight padding to minimize padding influence
        enc_a = tokenizer(texts_a, padding=True, return_tensors="pt")
        enc_b = tokenizer(texts_b, padding=True, return_tensors="pt")
        with torch.no_grad():
            emb_a = model.encode(enc_a["input_ids"], enc_a["attention_mask"])
            emb_b = model.encode(enc_b["input_ids"], enc_b["attention_mask"])
        # Different inputs should produce different embeddings
        cos_sim = torch.nn.functional.cosine_similarity(emb_a, emb_b)
        assert cos_sim.item() < 0.999, f"Embeddings too similar: cosine_sim={cos_sim.item()}"


class TestRerank:
    def test_output_shape(self, model, sample_input):
        input_ids, attention_mask = sample_input
        with torch.no_grad():
            scores = model.rerank(input_ids, attention_mask)
        assert scores.shape == (2,), f"Expected (2,), got {scores.shape}"

    def test_scores_in_valid_range(self, model, sample_input):
        """Scores should be probabilities in [0, 1]."""
        input_ids, attention_mask = sample_input
        with torch.no_grad():
            scores = model.rerank(input_ids, attention_mask)
        assert torch.all(scores >= 0) and torch.all(scores <= 1), (
            f"Scores out of range [0,1]: {scores}"
        )

    def test_get_rerank_logits_shape(self, model, sample_input):
        input_ids, attention_mask = sample_input
        with torch.no_grad():
            logits = model.get_rerank_logits(input_ids, attention_mask)
        assert logits.shape[0] == 2
        assert logits.shape[1] > 100  # vocab_size should be large


class TestYesNoTokens:
    def test_yes_no_token_ids_valid(self, model):
        assert isinstance(model.yes_token_id, int)
        assert isinstance(model.no_token_id, int)
        assert model.yes_token_id != model.no_token_id
        assert model.yes_token_id > 0
        assert model.no_token_id > 0

    def test_yes_no_are_single_tokens(self, model, tokenizer):
        yes_tokens = tokenizer.encode("yes", add_special_tokens=False)
        no_tokens = tokenizer.encode("no", add_special_tokens=False)
        assert len(yes_tokens) == 1, f"'yes' should be 1 token, got {len(yes_tokens)}"
        assert len(no_tokens) == 1, f"'no' should be 1 token, got {len(no_tokens)}"


class TestForwardPassDevice:
    def test_forward_completes_without_error(self, model, sample_input):
        """Both encode and rerank should complete without error."""
        input_ids, attention_mask = sample_input
        with torch.no_grad():
            embeddings = model.encode(input_ids, attention_mask)
            scores = model.rerank(input_ids, attention_mask)
        assert embeddings is not None
        assert scores is not None
