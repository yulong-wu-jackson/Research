# Model Architecture + Training Pipeline

**Created:** 2026-02-06
**Updated:** 2026-02-07
**Status:** Draft
**Depends on:** ticket_001 (config), ticket_002 (data pipeline)
**Blocks:** ticket_004 (evaluation needs trained checkpoints)

## Overview

Implement the core model (Qwen3-0.6B-Base + LoRA + dual task modes) and the training loop that supports rank-only, embedding-only, and joint alternating-batch modes. Includes loss functions, gradient conflict logging, the train.py entry script, and a sanity training run. The model aligns with Qwen3-Embedding (last-token pooling) and Qwen3-Reranker (yes/no token scoring) architectures.

## Context

- **Base model:** `Qwen/Qwen3-0.6B-Base` — decoder-only, 28 layers, hidden_size=1024, 16 attn heads, 8 KV heads (GQA). Same base as Qwen3-Embedding-0.6B and Qwen3-Reranker-0.6B.
- **LoRA via PEFT:** Applied to ALL linear layers (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj). B initialized to zero.
- **Embedding mode:** Last-token (EOS) pooling + L2 normalization (matches Qwen3-Embedding)
- **Reranking mode:** Yes/no token logit scoring via LM head (matches Qwen3-Reranker) — NO custom MLP head
- **Training loop:** Custom PyTorch loop (not sentence-transformers Trainer) because joint mode requires alternating between two different task batches with different loss functions
- **Hardware:** CUDA primary (bfloat16), MPS fallback (float16). Batch size 16 (reranking) / 8 (embedding), gradient accumulation 4

## Requirements

### Model (`src/unimoe/model/lora_model.py`)
- [ ] `UnimodelForExp1` class wrapping Qwen3-0.6B-Base:
  - Load base model via `AutoModelForCausalLM`, freeze all base params
  - Apply LoRA via PEFT `LoraConfig`:
    - `target_modules`: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
    - `r`: from config (default 16)
    - `lora_alpha`: from config (default = rank, i.e., 16)
    - `lora_dropout`: 0.05
    - `task_type`: None (we manage task modes manually)
    - `bias`: "none"
  - `encode(input_ids, attention_mask)` -> (B, 1024) L2-normalized embeddings via last-token (EOS) pooling
  - `rerank(input_ids, attention_mask)` -> (B,) relevance scores via yes/no token logit scoring (matching Qwen3-Reranker inference):
    ```python
    logits = model(**inputs).logits[:, -1, :]  # last position
    scores = torch.stack([logits[:, no_token_id], logits[:, yes_token_id]], dim=1)
    scores = torch.nn.functional.log_softmax(scores, dim=1)
    score = scores[:, 1].exp()  # P(yes) — matches Qwen3-Reranker scoring
    ```
    Note: During training, the SFT loss (cross-entropy on target token) is used directly on the logits. The softmax scoring is for inference/evaluation only.
  - Token IDs for "yes" and "no" resolved from tokenizer at init time
  - Access hidden states via `output_hidden_states=True` for future gradient analysis

### Losses (`src/unimoe/training/losses.py`)
- [ ] `InfoNCELoss`: contrastive loss matching Qwen3-Embedding's improved InfoNCE (paper Equation 1):
    - **Five-component denominator:** (1) positive pair, (2) explicit hard negatives, (3) in-batch query-query pairs, (4) in-batch document-document pairs, (5) in-batch cross-pair query-document
    - **False negative masking:** mask factor `m_ij = 0 if sim(q_i, d_j) > sim(q_i, d_i+) + 0.1`, preventing false negatives from dominating the loss
    - Configurable temperature (default tau=0.05, i.e., scale=20.0 — Qwen3 paper does not disclose tau; 0.05 is the sentence-transformers/Jina v3 standard)
    - Supports `CachedMultipleNegativesRankingLoss`-style gradient caching for large effective batch sizes (future enhancement)
- [ ] `RerankingSFTLoss`: cross-entropy loss on yes/no token logits — negative log probability of the correct label token. Equivalent to BCE but aligned with Qwen3-Reranker's SFT approach.

### Trainer (`src/unimoe/training/trainer.py`)
- [ ] `UnifiedTrainer` class:
  - Accepts config, model, optional embedding dataloader, optional reranking dataloader
  - Mode-aware step dispatch:
    - `RANK_ONLY`: every step uses reranking batch + SFT loss
    - `EMB_ONLY`: every step uses embedding batch + InfoNCE loss
    - `JOINT_SINGLE`: **alternating batches** — odd steps use embedding batch + InfoNCE, even steps use reranking batch + SFT loss. Each step sees exactly ONE task's loss (no combined loss per step). This is the GritLM approach.
  - AdamW optimizer on trainable params only (LoRA weights — no separate reranking head since we use the LM head)
  - Learning rate: 1e-4 (default), with cosine schedule + linear warmup (warmup_ratio=0.05)
  - **No combined loss formula** — with alternating batches, each step computes a single task's loss independently. The 50/50 alternation ratio means equal training signal from both tasks over time. (Mixed-batch approach with `L = L_emb + λ·L_rerank` is deferred to future work.)
  - Gradient accumulation, gradient clipping (max_grad_norm=1.0)
  - **Gradient conflict logging:** Every N steps (configurable), in joint mode, compute and log per-layer cosine similarity between embedding and reranking gradients. Store as `gradient_conflicts.jsonl`.
  - Periodic logging to JSONL file (loss, lr, step, epoch, task, gradient_conflict_mean)
  - Periodic checkpointing (save PEFT adapter + optimizer state)
  - Optional wandb integration (disabled by default for sanity runs)

### Entry Script (`scripts/train.py`)
- [ ] CLI: `--config <yaml>`, `--seed <int>`, `--device <auto|cuda|mps|cpu>`
  - Device auto-detection: CUDA > MPS > CPU
  - Loads config, sets seed (random, numpy, torch, cuda/mps), builds model, dataloaders, trainer
  - Saves frozen config copy to output directory for reproducibility
  - Saves final checkpoint on completion

## Design Decisions

- **`AutoModelForCausalLM` (not `AutoModel`):** Qwen3-0.6B-Base is a causal LM. We need the LM head for yes/no reranking scoring AND access to hidden states for embedding.
- **All linear layers as LoRA targets (not just attention):** "How Relevance Emerges" (arXiv 2504.08780) found that MLP up/gate projections are the MOST impactful modules for reranking. Excluding MLP modules misses critical parameters. Jina v3 and Qwen3 fine-tuning both target all linear layers. Estimated trainable params increases to ~15M but this is still highly parameter-efficient.
- **LoRA r=16, alpha=16:** r=16 is the standard in recent IR research (Jina v3 uses r=16). alpha=rank gives a scaling factor of 1.0, the most stable configuration per consensus.
- **Yes/no token scoring (no MLP head):** Matches Qwen3-Reranker exactly. Enables fair comparison with SOTA baselines. Eliminates ~1M extra parameters from MLP head. Inference score = `P(yes) = softmax(logit_no, logit_yes)[1]` (Qwen3-Reranker compatible). Training uses cross-entropy on the target token directly.
- **Last-token (EOS) pooling for embedding:** Standard for decoder-only embedding models. Qwen3-Embedding uses this exact approach — pool at the `<|endoftext|>` token (ID 151643, accessed via `tokenizer.pad_token_id`, NOT `tokenizer.eos_token_id` which is 151645/`<|im_end|>`).
- **PEFT `task_type=None`:** We manage both embedding extraction and reranking scoring manually; don't want PEFT to add its own head.
- **Temperature tau=0.05:** Standard in sentence-transformers (scale=20.0) and Jina v3 Stage 2.
- **Alternating batches (not mixed):** Each gradient step sees one task's loss. Avoids loss-scale mixing. Step counter modulo 2 selects task.
- **Gradient conflict logging:** Core data for Experiment 3 (gradient conflict analysis). ~20 lines of code to compute per-layer cosine similarity between task gradients. Essential for the research contribution.
- **Cycling shorter dataloader:** If one dataloader exhausts before the other in an epoch, cycle it (restart iteration).

## Scope

**In scope:** Model class (with yes/no scoring), losses, trainer with gradient conflict logging, train.py, sanity training run

**Out of scope:** Evaluation (ticket_004), MoE routing (future experiment), staged training (future experiment), CachedMNRL large-batch optimization (future experiment)

## Technical Notes

- Qwen3-0.6B uses GQA: q_proj is (1024, 1024), k_proj and v_proj are (1024, 512), o_proj is (1024, 1024). MLP: gate_proj/up_proj are (1024, 3072), down_proj is (3072, 1024). LoRA A/B dimensions vary per module.
- Estimated trainable params with all linear targets at r=16: ~15M (all LoRA, no separate head)
- Yes/no token IDs: resolve `tokenizer.convert_tokens_to_ids("yes")` and `tokenizer.convert_tokens_to_ids("no")` at model init. Verify these are single tokens in Qwen3 vocabulary.
- **EOS token trap:** `tokenizer.eos_token_id` returns 151645 (`<|im_end|>`), NOT the `<|endoftext|>` token (151643) needed for embedding pooling. Use `tokenizer.convert_tokens_to_ids("<|endoftext|>")` or `tokenizer.pad_token_id` (which IS 151643) for pooling. Never use `eos_token_id` for embedding.
- **`encode()` must extract from hidden states, not LM head:** Since we load via `AutoModelForCausalLM`, the forward pass returns logits through the LM head. For embedding, access `model.model(**inputs).last_hidden_state` (the inner model without LM head) or use `output_hidden_states=True` and take the last hidden state.
- **LM head frozen (intentional deviation):** The official Qwen3-Reranker uses full SFT with a trainable LM head (evidenced by vocab_size changing from 151936 to 151669). Our LoRA approach freezes the LM head — this is a deliberate parameter-efficiency choice. Document as a known deviation. Consider ablation: unfreeze the "yes"/"no" rows of `lm_head.weight` for potential quality improvement.
- CUDA uses bfloat16 (native model dtype). MPS uses float16 (bfloat16 not supported). Set `PYTORCH_ENABLE_MPS_FALLBACK=1` for MPS.
- Checkpoint format: save PEFT adapter (via `model.save_pretrained()`), optimizer state, and config snapshot. No separate head checkpoint needed.

## Acceptance Criteria

- [ ] `tests/test_model.py` passes:
  - Only LoRA params are trainable (base + LM head frozen)
  - `encode()` returns L2-normalized vectors of shape (B, 1024)
  - `rerank()` returns scalars of shape (B,) using yes/no logit difference
  - LoRA applied to all 7 module types (q/k/v/o/gate/up/down)
  - Trainable param count is ~15M
  - Forward pass completes on available device without error
- [ ] `tests/test_losses.py` passes: correct loss values on known inputs for both InfoNCE (with hard negatives) and RerankingSFTLoss
- [ ] `tests/test_training_step.py` passes: one training step completes, loss is finite, gradients flow only to trainable params
- [ ] `tests/test_gradient_conflict.py` passes: gradient conflict computation returns valid cosine similarity values in [-1, 1] per layer
- [ ] Sanity run: `uv run python scripts/train.py --config configs/rank_only_r8.yaml --seed 42` with 1000 samples, 1 epoch — loss decreases
- [ ] Checkpoint saved to `outputs/rank_only_r8/seed_42/checkpoints/final/`
- [ ] `uv run pytest tests/ -v` all green (including tests from prior tickets)
