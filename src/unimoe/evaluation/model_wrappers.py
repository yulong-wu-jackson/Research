"""MTEB v2 model wrappers for embedding and reranking evaluation.

MTEBEncoderWrapper: implements EncoderProtocol for MTEB v2 embedding evaluation.
MTEBCrossEncoderWrapper: implements CrossEncoderProtocol for MTEB v2 reranking evaluation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from unimoe.data.templates import (
    DEFAULT_INSTRUCTION,
    build_reranking_token_ids,
    format_embedding_query,
)
from unimoe.model.lora_model import UnimodelForExp1


class MTEBEncoderWrapper:
    """MTEB v2 EncoderProtocol wrapper for UnimodelForExp1.

    Uses the model's encode() method with instruction prefix for queries.
    """

    def __init__(self, model: UnimodelForExp1, batch_size: int = 64):
        self.model = model
        self.tokenizer = model.tokenizer
        self.batch_size = batch_size
        self.device = next(model.parameters()).device

        # MTEB v2 requires mteb_model_meta
        try:
            from mteb.models import ModelMeta

            self.mteb_model_meta = ModelMeta(
                name="custom/unimoe-exp1",
                revision=None,
                languages=["eng"],
                loader=None,
                release_date=None,
                n_parameters=None,
                memory_usage_mb=None,
                max_tokens=None,
                embed_dim=1024,
                license=None,
                open_weights=True,
                public_training_code=None,
                public_training_data=None,
                framework=["PyTorch"],
                similarity_fn_name="cosine",
                use_instructions=True,
                training_datasets=None,
            )
        except (ImportError, Exception):
            self.mteb_model_meta = None

    def encode(
        self,
        inputs: DataLoader,
        *,
        task_metadata: Any = None,
        hf_split: str = "",
        hf_subset: str = "",
        prompt_type: Any = None,
        **kwargs,
    ) -> np.ndarray:
        """Encode text inputs into embeddings.

        Adds instruction prefix for query-type prompts.
        """
        all_embeddings = []
        self.model.eval()

        with torch.no_grad():
            for batch in inputs:
                texts = batch["text"] if isinstance(batch, dict) else batch
                if isinstance(texts, str):
                    texts = [texts]

                # Add instruction prefix for queries
                # prompt_type is a PromptType enum; use .value for comparison
                is_query = prompt_type is not None and getattr(
                    prompt_type, "value", str(prompt_type)
                ) == "query"
                if is_query:
                    texts = [
                        format_embedding_query(DEFAULT_INSTRUCTION, t) for t in texts
                    ]

                # Append EOS for pooling
                eos = self.tokenizer.eos_token
                texts_with_eos = [t + eos for t in texts]

                encoded = self.tokenizer(
                    texts_with_eos,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)

                embeddings = self.model.encode(
                    encoded["input_ids"], encoded["attention_mask"]
                )
                all_embeddings.append(embeddings.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

    def similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """Cosine similarity between two sets of embeddings."""
        import torch as _torch

        e1 = _torch.from_numpy(embeddings1)
        e2 = _torch.from_numpy(embeddings2)
        return _torch.nn.functional.cosine_similarity(
            e1.unsqueeze(1), e2.unsqueeze(0), dim=2
        ).numpy()

    def similarity_pairwise(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """Pairwise cosine similarity between corresponding embeddings."""
        import torch as _torch

        e1 = _torch.from_numpy(embeddings1)
        e2 = _torch.from_numpy(embeddings2)
        return _torch.nn.functional.cosine_similarity(e1, e2, dim=1).numpy()


class MTEBCrossEncoderWrapper:
    """MTEB v2 CrossEncoderProtocol wrapper for UnimodelForExp1.

    Uses the model's rerank() method with Qwen3-Reranker chat template.
    """

    def __init__(self, model: UnimodelForExp1, batch_size: int = 64):
        self.model = model
        self.tokenizer = model.tokenizer
        self.batch_size = batch_size
        self.device = next(model.parameters()).device

        try:
            from mteb.models import ModelMeta

            self.mteb_model_meta = ModelMeta(
                name="custom/unimoe-exp1-reranker",
                revision=None,
                languages=["eng"],
                loader=None,
                release_date=None,
                n_parameters=None,
                memory_usage_mb=None,
                max_tokens=None,
                embed_dim=None,
                license=None,
                open_weights=True,
                public_training_code=None,
                public_training_data=None,
                framework=["PyTorch"],
                similarity_fn_name=None,
                use_instructions=True,
                training_datasets=None,
            )
        except (ImportError, Exception):
            self.mteb_model_meta = None

    def predict(
        self,
        inputs1: DataLoader,
        inputs2: DataLoader,
        *,
        task_metadata: Any = None,
        hf_split: str = "",
        hf_subset: str = "",
        prompt_type: Any = None,
        **kwargs,
    ) -> np.ndarray:
        """Predict relevance scores for query-document pairs.

        Formats inputs using the Qwen3-Reranker chat template and
        scores via yes/no token logits.
        """
        all_scores = []
        self.model.eval()
        pad_id = self.tokenizer.pad_token_id
        max_len = 512

        with torch.no_grad():
            for batch1, batch2 in zip(inputs1, inputs2):
                queries = batch1["text"] if isinstance(batch1, dict) else batch1
                documents = batch2["text"] if isinstance(batch2, dict) else batch2
                if isinstance(queries, str):
                    queries = [queries]
                if isinstance(documents, str):
                    documents = [documents]

                # Build token IDs via concatenation (matching training collator)
                all_ids = [
                    build_reranking_token_ids(
                        self.tokenizer, DEFAULT_INSTRUCTION, q, d, max_len
                    )
                    for q, d in zip(queries, documents)
                ]

                # Left-pad to max length in batch
                max_seq_len = min(max(len(ids) for ids in all_ids), max_len)
                input_ids = []
                attention_masks = []
                for ids in all_ids:
                    ids = ids[:max_seq_len]
                    pad_len = max_seq_len - len(ids)
                    input_ids.append([pad_id] * pad_len + ids)
                    attention_masks.append([0] * pad_len + [1] * len(ids))

                input_ids_t = torch.tensor(input_ids, dtype=torch.long).to(self.device)
                attention_mask_t = torch.tensor(attention_masks, dtype=torch.long).to(self.device)

                scores = self.model.rerank(input_ids_t, attention_mask_t)
                all_scores.append(scores.cpu().numpy())

        return np.concatenate(all_scores, axis=0)
