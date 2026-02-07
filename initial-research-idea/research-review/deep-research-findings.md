# Deep Research Findings: Unified Embedding + Reranking with LoRA and MoE-LoRA

> Comprehensive literature and technical review | February 2026
> Target venue: EMNLP 2026

---

## Table of Contents

1. [Base Model Choice](#1-base-model-choice-for-embeddingreranking-research)
2. [LoRA Configuration Best Practices](#2-lora-configuration-best-practices-for-embeddingreranking)
3. [Training Strategy for Joint Embedding+Reranking](#3-training-strategy-for-joint-embeddingreranking)
4. [Contrastive Loss (InfoNCE) Best Practices](#4-contrastive-loss-infonce-best-practices-for-embedding-training)
5. [Reranking Training Best Practices](#5-reranking-training-best-practices)

---

## 1. Base Model Choice for Embedding+Reranking Research

### 1.1 What Base Models Do Top Embedding/Reranking Papers Use (2025-2026)?

| Model | Base Architecture | Type | Used By | MTEB/BEIR Performance |
|-------|------------------|------|---------|----------------------|
| **Qwen3-0.6B-Base** | Decoder-only (28 layers, 1024 hidden) | Dense Transformer | Qwen3-Embedding-0.6B, Qwen3-Reranker-0.6B, Jina Reranker v3 | MTEB #1 multilingual (8B variant) |
| **Qwen3 (4B/8B)** | Decoder-only | Dense Transformer | Qwen3-Embedding-8B, E2Rank | BEIR 54.35 nDCG@10 (E2Rank-8B) |
| **Mistral-7B** | Decoder-only | Dense Transformer | GritLM-7B, E5-Mistral, Causal2Vec | MTEB SOTA (GritLM) |
| **LLaMA-3.1-8B** | Decoder-only | Dense Transformer | "How Relevance Emerges" study | Used in LoRA reranking research |
| **ModernBERT-base** (139M) | Encoder-only (bidirectional) | Encoder | ModernBERT-embed-base, Nomic-embed | Competitive at small scale |
| **ModernBERT-large** (395M) | Encoder-only (bidirectional) | Encoder | ModernBERT-embed-large, LightOn | Good efficiency-performance tradeoff |
| **Gemma-2B/9B** | Decoder-only | Dense Transformer | BGE-Reranker-v2-gemma, EmbeddingGemma | BGE reranker family |

**Key finding:** The 2025-2026 landscape is overwhelmingly dominated by **decoder-only** models for both embedding and reranking. The top-performing open models (Qwen3-Embedding, E2Rank, GritLM, Jina Reranker v3) all use decoder-only architectures. ModernBERT is the strongest encoder-only contender but at much smaller scale.

### 1.2 Decoder-Only vs. Encoder-Only: Which Is Right?

**Arguments for decoder-only (Qwen3-0.6B-Base):**
- All SOTA embedding models in 2025 use decoder-only LLMs (Qwen3, Mistral, Gemma)
- Decoder-only models naturally support reranking via next-token prediction (P(yes|q,d))
- Qwen3-0.6B-Base has been validated by both Qwen3-Embedding and Jina Reranker v3 at the 0.6B scale
- Rich pre-training on diverse data (code, multilingual) transfers to IR tasks
- LLM2Vec and Causal2Vec demonstrate that decoder-only models can match or exceed encoder models for embeddings when properly adapted

**Arguments for encoder-only (ModernBERT):**
- Bidirectional attention is inherently better for semantic understanding of full documents
- ModernBERT supports 8K context, trained on 2T tokens, includes code
- Up to 2-4x faster inference than comparable decoder models
- More natural fit for embedding extraction (no need for special EOS pooling)
- More parameter-efficient at small scale (139M vs 600M)

**Arguments against ModernBERT for THIS project:**
- Cannot naturally perform reranking via next-token prediction; would need a classification head
- Much smaller model capacity (139M/395M vs 600M) limits MoE-LoRA routing analysis
- Less comparable to SOTA baselines (Qwen3-Embedding, Jina v3)
- The "unification" story is weaker since embedding and reranking would need fundamentally different architectures

**Verdict:** Qwen3-0.6B-Base is the correct choice. It is the same base used by both Qwen3-Embedding-0.6B AND Jina Reranker v3, providing direct comparability with two SOTA systems at the target scale.

### 1.3 What Does Qwen3-Embedding Actually Use as Its Base?

**Confirmed:** Qwen3-Embedding-0.6B is fine-tuned from **Qwen3-0.6B-Base** (the pre-trained base model, NOT the instruct variant). This is explicitly shown in the HuggingFace model tree.

Source: [Qwen3-Embedding-0.6B Model Card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)

**Architecture details of Qwen3-0.6B-Base:**
| Parameter | Value |
|-----------|-------|
| Layers | 28 |
| Hidden dimension | 1024 |
| Attention heads | 16 |
| Context length | 32K tokens |
| Embedding dimension | Up to 1024 |
| Parameters | ~600M |
| Tensor type | BF16 |

### 1.4 What Does Qwen3-Reranker Use as Its Base?

**Confirmed:** Qwen3-Reranker-0.6B is also fine-tuned from **Qwen3-0.6B-Base**. Both embedding and reranking models share the same foundation, which is ideal for unified training research.

**Reranking mechanism:** The Qwen3-Reranker uses a **generative cross-encoder** approach, computing P(yes|instruction, query, document) vs P(no|instruction, query, document) as the relevance score. This is supervised fine-tuning (SFT) with cross-entropy loss on the yes/no token probabilities.

Source: [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B), [Qwen3-Embedding Paper](https://arxiv.org/abs/2506.05176)

### 1.5 Base vs. Instruct vs. Qwen3-Embedding-0.6B as Starting Point?

| Starting Point | Pros | Cons | Recommendation |
|---------------|------|------|----------------|
| **Qwen3-0.6B-Base** | Clean slate, no instruction bias, same as official models used | No instruction following, needs full training pipeline | **Best for research** |
| **Qwen3-0.6B (Instruct)** | Instruction following, 1-5% improvement with task instructions | May have alignment artifacts that interfere with embedding training | Not recommended for research baseline |
| **Qwen3-Embedding-0.6B** | Already trained for embeddings, strong starting point | Already heavily fine-tuned, limits analysis of training dynamics | Good for reranking-only adapter, not for joint training research |

**Recommendation:** Use **Qwen3-0.6B-Base** as the starting point. This matches what Qwen used officially for both their embedding and reranking models, ensures a clean research setup, and allows full control over the training pipeline. For baselines, compare against the official Qwen3-Embedding-0.6B and Qwen3-Reranker-0.6B.

### 1.6 Other Relevant Models to Consider

**Jina Reranker v3** is particularly noteworthy because:
- It is also built on **Qwen3-0.6B** as the backbone
- Uses **LoRA fine-tuning** with **rank 16** in Stage 1 training
- Adds a lightweight **MLP projector** (1024 -> 512 -> 256) for scoring
- Achieves 61.94 nDCG@10 on BEIR (SOTA for 0.6B rerankers)
- Uses multi-objective training with InfoNCE + dispersive + dual matching + similarity losses

Source: [Jina Reranker v3 Paper](https://arxiv.org/html/2509.25085v2)

**E2Rank** is the closest competitor to the unified embedding+reranking vision:
- Uses Qwen3 (0.6B/4B/8B) as base, with full parameter training
- Unifies retrieval and reranking via cosine similarity as the single scoring function
- Combines InfoNCE loss (temperature 0.03) + RankNet loss (temperature 0.1, weight lambda=2.0)
- Achieves competitive BEIR results with 5x speedup over RankQwen3

Source: [E2Rank Paper](https://arxiv.org/html/2510.22733)

---

## 2. LoRA Configuration Best Practices for Embedding/Reranking

### 2.1 Target Modules

**Evidence from research:**

The paper "How Relevance Emerges" (arXiv 2504.08780) provides the most detailed analysis of which modules matter for reranking with LoRA:

| Configuration | LLaMA3 NDCG@10 (DL19) | Mistral NDCG@10 (DL19) | Pythia NDCG@10 (DL19) |
|--------------|----------------------|------------------------|----------------------|
| No LoRA | 0.1855 | 0.1450 | 0.1895 |
| MLP only | 0.4341 | 0.4390 | 0.6262 |
| MHA only | 0.5987 | 0.5164 | 0.7009 |
| **Both MHA + MLP** | **0.7655** | **0.6891** | **0.7569** |

Key findings:
- **MHA (attention) is more impactful** than MLP alone for reranking
- **Combining both gives the best results** across all three models
- Within MLP, **up_proj and gate_proj are substantially more impactful** than down_proj
- Up+Gate projections alone recover ~96% of full MLP performance

Source: [How Relevance Emerges](https://arxiv.org/html/2504.08780)

**Recommendation for target modules:**

```python
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention (MHA)
    "gate_proj", "up_proj", "down_proj"        # MLP (all three)
]
```

**Rationale:**
- Including all attention projections (q, k, v, o) ensures full MHA coverage
- Including all MLP projections captures the gate/up synergy that is critical for reranking
- The down_proj contributes less individually, but including it adds completeness and is standard practice
- This is the default recommendation from Unsloth, QLoRA paper, and Databricks LoRA guide

**For MoE-LoRA specifically:** Consider applying MoE routing only to MLP modules (gate_proj, up_proj, down_proj) while keeping attention modules with shared LoRA. This is motivated by:
1. MLP modules carry task-specific information (up/gate for reranking)
2. Attention modules are more "shared" across tasks (both tasks need good attention patterns)
3. Reduces number of routing parameters vs. full MoE

### 2.2 LoRA Rank

**Evidence from research:**

| Source | Finding | Recommended Rank |
|--------|---------|-----------------|
| "How Relevance Emerges" | Rank 1 sufficient for reranking on MS MARCO | r=1 is a lower bound |
| "How Relevance Emerges" | Ranks 1, 2, 8, 32 all comparable on MS MARCO | Low sensitivity to rank |
| Jina Reranker v3 | Uses rank 16 for LoRA fine-tuning | r=16 |
| Unsloth Guide | Start with r=16 or r=32 for general tasks | r=16-32 |
| QLoRA paper | r=64 for best quality, r=16 for efficiency | r=16-64 |
| SMoRA (MoE-LoRA) | 8 active out of 64 total ranks outperforms full r=64 | Total r=64, active r=8 |
| Databricks guide | r between 4 and 64, bigger for smaller models | Larger rank for 0.6B |

**Recommendation:** Start with **r=16** per expert for MoE-LoRA. Test ablation at r={4, 8, 16, 32}. For single LoRA baselines, use r=16 to match Jina v3's configuration.

**For MoE-LoRA:** With 4 experts (2 task-specific + 1 shared + 1 spare), each at r=16, the effective total rank when 2 experts are active is r=32, which provides good capacity while maintaining parameter efficiency.

### 2.3 LoRA Alpha

**Best practices:**

| Source | Recommendation |
|--------|---------------|
| Unsloth Guide | alpha = rank (e.g., alpha=16 for r=16) |
| General consensus | alpha/rank ratio = 1 to 2 |
| rsLoRA (Rank-Stabilized) | Scaling factor = alpha/sqrt(r) instead of alpha/r |
| Databricks | alpha between 16 and 64 |

**Recommendation:** Set **alpha = rank** (e.g., alpha=16 for r=16). This gives a scaling factor of 1.0 (alpha/rank = 1), which is the most commonly used and stable configuration. Consider using **rsLoRA** (rank-stabilized LoRA) if experimenting with higher ranks, as it provides better stability.

### 2.4 LoRA Dropout

**Consensus:** **dropout = 0.05** is the most commonly used value for LoRA fine-tuning in IR research. Some studies use 0.0 (no dropout) or 0.1. For small models (0.6B), 0.05 provides mild regularization without impeding learning.

### 2.5 Task Type in PEFT

**Options:**
- `TaskType.FEATURE_EXTRACTION`: Returns hidden states for embedding extraction. Wraps model in `PeftModelForFeatureExtraction`.
- `TaskType.CAUSAL_LM`: For generative next-token prediction. Wraps model in `PeftModelForCausalLM`.
- `None`: No task-specific wrapping.

**Analysis:**
- For **embedding tasks**, `FEATURE_EXTRACTION` is technically correct since you want hidden states
- For **reranking via yes/no generation**, `CAUSAL_LM` is technically correct since you need next-token logits
- For **unified training**, neither is fully appropriate since the model needs to do both

**Recommendation:** Use `task_type=None` for the unified model configuration. This avoids PEFT adding any task-specific processing, and you manually handle:
1. EOS token hidden state extraction for embeddings
2. Next-token logit extraction for reranking scores

This gives maximum flexibility for the multi-task setup. The LoRA adapters will be injected into target_modules regardless of task_type.

Source: [PEFT Documentation](https://huggingface.co/docs/peft/en/package_reference/lora), [HuggingFace PEFT Feature Extraction Example](https://github.com/huggingface/peft/blob/main/examples/feature_extraction/peft_lora_embedding_semantic_search.py)

---

## 3. Training Strategy for Joint Embedding+Reranking

### 3.1 Batch Organization: Alternating vs. Mixed vs. Task-Proportional

**What existing systems do:**

| System | Strategy | Details |
|--------|----------|---------|
| **GritLM** | Alternating batches | Embedding batch size 2048, generative batch size 256, alternating during training |
| **E2Rank** | Staged (sequential) | Stage 1: embedding only (InfoNCE), Stage 2: joint (InfoNCE + RankNet) |
| **Qwen3-Embedding/Reranker** | Separate models | Trained independently, then optionally merged via SLERP |
| **Jina Reranker v3** | Staged + domain merging | Stage 1: domain-specific LoRA, Stage 2: hard negative mining, Stage 3: model merging |
| **MTL-LoRA** | Mixed batches | Task-adaptive parameters differentiate within shared training |

**Options and tradeoffs:**

1. **Alternating batches** (GritLM-style):
   - Alternate between pure embedding batches and pure reranking batches
   - Simpler implementation, clear gradient signals per task
   - Risk: optimizer state may oscillate between tasks
   - Recommended ratio: ~4:1 embedding:reranking (embedding needs more data)

2. **Task-proportional sampling**:
   - Sample batches with probability proportional to task dataset size
   - More balanced training, avoids task starvation
   - Requires careful calibration of sampling weights

3. **Staged training** (E2Rank-style):
   - Stage 1: Contrastive pre-training (embedding only)
   - Stage 2: Joint fine-tuning (embedding + reranking)
   - Allows embedding model to stabilize before adding reranking
   - **Most well-validated approach** in recent literature

**Recommendation:** Use **staged training**:
- **Stage 1:** Train embedding task only with InfoNCE loss (establish good representations)
- **Stage 2:** Joint training with alternating batches (InfoNCE + reranking loss)
- This mirrors E2Rank's successful approach and is well-motivated: the embedding model provides the foundation that reranking refines

### 3.2 Multi-Task Gradient Optimization

**Options:**

| Method | How It Works | Pros | Cons |
|--------|-------------|------|------|
| **Naive sum** | L = L_emb + beta * L_rerank | Simplest | No conflict resolution |
| **GradNorm** | Dynamically adjust loss weights to balance gradient magnitudes | Balances training rates | Doesn't fix gradient directions |
| **PCGrad** | Project conflicting gradients onto orthogonal complement | Fixes direction conflicts | ~1.2x compute overhead |
| **Ortho-LoRA** | Orthogonal projection within LoRA subspace | LoRA-specific, low overhead | Very new (Jan 2025) |
| **GCond** | Gradient accumulation + adaptive arbitration | PCGrad improvement | Complex implementation |
| **CAGrad** | Minimize maximum task loss improvement | Pareto-optimal | More compute |

**Key finding:** PCGrad has become the **de facto standard** for gradient conflict resolution in multi-task learning. It outperforms GradNorm because it addresses gradient direction, not just magnitude.

**Recommendation for this project:**
1. Start with **naive weighted sum** as the baseline (L = L_emb + beta * L_rerank)
2. Implement **PCGrad** as a comparison to quantify gradient conflict between tasks
3. The MoE-LoRA routing itself is the primary mechanism for handling task interference; PCGrad analysis serves as a diagnostic tool to measure remaining conflict

Source: [PCGrad Paper](https://proceedings.neurips.cc/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf), [Ortho-LoRA](https://arxiv.org/abs/2601.09684)

### 3.3 Loss Weighting Strategies

**What existing systems use:**

| System | Strategy | Weights |
|--------|----------|---------|
| **E2Rank** | Fixed weight | L = L_InfoNCE + 2.0 * L_RankNet |
| **Jina v3** | Fixed weights | L = L_rank + 0.45*L_disperse + 0.85*L_dual + 0.85*L_similar |
| **GritLM** | Equal weight | L_emb + L_gen (alternating, effectively equal) |

**Options:**
1. **Fixed alpha/beta**: L = alpha * L_emb + beta * L_rerank. Simple, requires tuning. Start with alpha=1.0, beta=1.0, then grid search.
2. **Dynamic (GradNorm)**: Automatically adjust weights to balance training rates. More principled but adds complexity.
3. **Uncertainty weighting** (Kendall et al.): Learn task weights as learnable parameters based on homoscedastic uncertainty.

**Recommendation:** Start with **fixed weights** (L = L_InfoNCE + lambda * L_rerank), testing lambda in {0.5, 1.0, 2.0, 5.0}. E2Rank found lambda=2.0 optimal for RankNet loss weighting. If initial experiments show imbalanced training, consider GradNorm as a more principled approach.

### 3.4 Gradient Accumulation

**Consideration:** Embedding training benefits from large batch sizes (many in-batch negatives), while reranking may use smaller batches (with explicit hard negatives). This creates a batch size mismatch.

**Strategy:**
- **Embedding:** Use CachedMultipleNegativesRankingLoss to simulate large effective batch sizes (e.g., 4096-65536) even with small mini-batches
- **Reranking:** Use gradient accumulation to match effective batch sizes
- **Implementation:** Set mini_batch_size based on GPU memory; use gradient accumulation steps to achieve target effective batch size

### 3.5 Learning Rate Configuration

**Should embedding and reranking use different LRs?**

**Evidence:**
- E2Rank uses different LRs: Stage 1 (embedding) = 2e-5, Stage 2 (joint) = 5e-6
- LoRA+ research shows using different LRs for LoRA-A and LoRA-B matrices improves stability
- General LoRA fine-tuning consensus: 1e-4 to 2e-4 for LoRA, 10x lower than full fine-tuning

**Recommendation:**
- **Single LR for the shared model**: 1e-4 (standard for LoRA)
- **Stage 1 (embedding only)**: LR = 1e-4
- **Stage 2 (joint)**: LR = 5e-5 (lower to preserve Stage 1 learning)
- For MoE-LoRA: Use the same LR for all experts; the router handles task differentiation

### 3.6 Optimizer Choice

| Optimizer | Memory per Param | Quality | Recommendation |
|-----------|-----------------|---------|----------------|
| **AdamW (32-bit)** | 8 bytes (2 states) | Gold standard | Use if GPU memory allows |
| **AdamW 8-bit (bitsandbytes)** | 2 bytes | ~Same as 32-bit | **Recommended** for 0.6B with LoRA |
| **Paged AdamW 8-bit** | 2 bytes + CPU offload | Same as 8-bit | Use if GPU memory is tight |
| **Adafactor** | 4 bytes (factored states) | Slightly lower | Alternative if memory-constrained |
| **SGD + Momentum** | 4 bytes | Lower quality | Not recommended for LoRA |

**Recommendation:** Use **AdamW 8-bit** (`adamw_bnb_8bit`). At 0.6B scale with LoRA (only ~0.5% parameters are trainable), even 32-bit AdamW is feasible, but 8-bit provides insurance with negligible quality loss. "In practice, 8-bit quantization is accurate enough and won't make much difference in training performance."

Source: [Fine-tuning LLMs with AdamW variants](https://kaitchup.substack.com/p/fine-tuning-llms-with-32-bit-8-bit)

### 3.7 Learning Rate Schedule

**Consensus:** **Cosine schedule with warmup** is the standard for LoRA fine-tuning.

**Configuration:**
- **Warmup ratio:** 0.03-0.10 (3-10% of total steps)
- **Schedule:** Cosine decay to 0 (or near-0 min LR)
- **Epochs:** 1-3 (E2Rank uses 1 epoch per stage)

Source: [Unsloth LoRA Guide](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)

---

## 4. Contrastive Loss (InfoNCE) Best Practices for Embedding Training

### 4.1 In-Batch Negatives: How Many Are Needed?

**Key principle:** In contrastive learning, every other sample in the batch acts as a negative. So batch_size - 1 = number of in-batch negatives per anchor.

**Evidence from the field:**

| System | Effective Batch Size | In-Batch Negatives | Notes |
|--------|---------------------|-------------------|-------|
| GritLM-7B | 2048 | 2047 | Uses large batch for embedding |
| E2Rank | 512 (Stage 1), 128 (Stage 2) | 511/127 | Plus 15 explicit hard negatives |
| Jina v3 | 60 (Stage 1) | 59 | Plus 15-25 mined negatives |
| Sentence-Transformers | 32+ recommended | 31+ | CachedMNRL enables 65536 |
| General consensus | 256-1024 minimum | - | Performance scales with batch size |

**Critical insight from Sentence-Transformers:** `CachedMultipleNegativesRankingLoss` enables simulating massive batch sizes (up to 65536) with constant GPU memory, using a two-pass approach:
1. Forward pass without gradients to get all embeddings
2. Compute loss over full "virtual" batch
3. Second forward pass with gradients using cached loss gradients

**Recommendation:** Use **CachedMultipleNegativesRankingLoss** with:
- `mini_batch_size`: As large as GPU memory allows (64-128)
- Effective batch size: 1024-4096 (via gradient caching)
- Plus 5-15 explicit hard negatives per query

Source: [Sentence-Transformers Losses](https://sbert.net/docs/package_reference/sentence_transformer/losses.html)

### 4.2 Hard Negative Mining Strategy

**Types of hard negatives ranked by quality (2025 consensus):**

| Strategy | Quality | Cost | Used By |
|----------|---------|------|---------|
| **Random negatives** | Low | Free | Baseline only |
| **BM25 negatives** | Medium | Low | Traditional, simple |
| **Dense retrieval negatives** | Medium-High | Medium | Self-mining or existing model |
| **Cross-encoder mined negatives** | **Highest** | High | Jina v3, NV-Retriever |
| **Cross-system hybrid** | **Highest** | High | Jina v3 (BGE + Jina + GTE + E5) |
| **Positive-aware mining (NV-Retriever)** | Highest (avoids false negatives) | High | NV-Retriever, ICLR 2025 |

**Critical findings:**
- BM25 negatives alone can **underperform** random sampling on some metrics (ACL 2025 finding)
- ~70% of the "hardest" negatives from MS MARCO are actually **false negatives** (NV-Retriever)
- **Positive-aware mining** (TopK-PercPos: threshold = 95% of positive score) effectively removes false negatives
- **Dynamic hard negative mining** (re-mining during training) significantly outperforms static mining

**Recommendation:**
1. **For MS MARCO training data:** Use existing mined hard negatives (BM25 + cross-encoder from public datasets)
2. **Mine additional negatives:** Use Qwen3-Embedding-0.6B or BGE-M3 as the dense retriever
3. **Apply false negative filtering:** Use positive-aware filtering (NV-Retriever style): reject negatives with similarity > 95% of positive similarity
4. **Number of hard negatives per query:** 7-15 (E2Rank uses 15, Jina v3 uses 15-25)

Source: [NV-Retriever](https://arxiv.org/abs/2407.15831), [Hard Negatives, Hard Lessons (EMNLP 2025)](https://aclanthology.org/2025.findings-emnlp.481.pdf)

### 4.3 Temperature Parameter

**Temperature directly controls the sharpness of the contrastive distribution.**

| System | Temperature | Scale (1/tau) | Context |
|--------|------------|---------------|---------|
| **E2Rank (InfoNCE)** | **0.03** | 33.3 | Embedding contrastive learning |
| **E2Rank (RankNet)** | **0.1** | 10.0 | Ranking loss |
| **Jina v3 Stage 1** | **0.25** | 4.0 | Foundation training |
| **Jina v3 Stage 2** | **0.05** | 20.0 | Hard negative mining stage |
| **Sentence-Transformers default** | **0.05** | 20.0 | MultipleNegativesRankingLoss |
| **General consensus** | 0.05-0.1 | 10-20 | Most common range |

**Note:** Sentence-Transformers uses `scale` parameter (inverse of temperature). `scale=20.0` corresponds to `temperature=0.05`.

**Recommendation:** Use **temperature = 0.05** (scale = 20.0) as the starting point, consistent with Sentence-Transformers and Jina v3 Stage 2. Consider lower temperature (0.03, as in E2Rank) if using very large batch sizes.

### 4.4 Should MatryoshkaLoss Be Considered?

**What it does:** Trains the model to produce embeddings that can be truncated to smaller dimensions (e.g., 1024 -> 512 -> 256 -> 128 -> 64) without significant performance loss.

**Pros:**
- Enables flexible deployment (trade precision for speed)
- Qwen3-Embedding-0.6B already supports Matryoshka dimensions (32-1024)
- Can be combined with any contrastive loss as a wrapper

**Cons:**
- Does **not** improve training speed or memory
- Adds complexity to the training pipeline
- Not directly relevant to the core research question (interference/routing)

**Recommendation:** Include **MatryoshkaLoss as an optional wrapper** during Stage 1 embedding training, but it should not be the focus. It is a practical feature that makes the model more deployable but does not contribute to the core research contribution. The priority should be on the joint embedding+reranking training dynamics.

### 4.5 Advanced Loss Alternatives

**GISTEmbedLoss** (Guided In-sample Selection of Training Negatives):
- Uses a guide model to assist in selecting in-batch negatives
- Mitigates false negative problem
- Can be used as a drop-in replacement for MultipleNegativesRankingLoss

**CachedGISTEmbedLoss:**
- Cached version for larger effective batch sizes
- Combines benefits of both GIST and caching

**Recommendation:** Test **CachedMultipleNegativesRankingLoss** first (simpler, well-validated). If false negatives are a problem, upgrade to CachedGISTEmbedLoss.

Source: [Sentence-Transformers Loss Overview](https://sbert.net/docs/sentence_transformer/loss_overview.html)

---

## 5. Reranking Training Best Practices

### 5.1 Loss Function Comparison

| Loss Type | Method | Used By | Pros | Cons |
|-----------|--------|---------|------|------|
| **Pointwise BCE** | P(relevant\|q,d) via cross-entropy on yes/no | Qwen3-Reranker, BGE-Reranker | Simple, well-understood, compatible with LLM next-token prediction | Binary labels lose graded relevance |
| **Pointwise (Generative)** | Cross-entropy on yes/no token logits | Qwen3-Reranker | Natural for decoder models | Requires instruction template |
| **Pairwise RankNet** | P(d_i > d_j) for pairs | E2Rank (lambda=2.0) | Captures relative ordering | O(n^2) pairs, no absolute scores |
| **Pairwise MarginMSE** | Minimize margin difference vs. teacher | Distillation settings | Transfers teacher knowledge | Requires teacher scores |
| **Listwise ListMLE** | Log-likelihood of correct ranking | Academic research | Directly optimizes ranking | Complex, needs list-level data |
| **Listwise LambdaRank** | Lambda gradients for NDCG optimization | LambdaMART, FIRST | Directly optimizes NDCG | Complex implementation |
| **Multi-Objective** | InfoNCE + dispersive + dual matching | Jina v3 | Rich training signal | Many hyperparameters |

### 5.2 What Top Rerankers Use

**Qwen3-Reranker (all sizes):**
- **Loss:** Cross-entropy on yes/no token probabilities
- **Mechanism:** LLM generates P(yes|instruction, query, document) as the relevance score
- **Format:** Instruction template with query and document, model predicts "yes" or "no"
- **Training:** Supervised fine-tuning (SFT) on labeled data only (skips contrastive pre-training)

**BGE-Reranker (all variants):**
- **Loss:** Cross-entropy loss
- **Architecture:** Cross-encoder (BERT-style for v1, LLM-based for v2+)
- **Base models:** bge-m3 (for v2-m3), gemma-2b (for v2-gemma), gemma-2-9b (for v2.5-lightweight)

**Jina Reranker v3:**
- **Loss:** Multi-objective (InfoNCE + 0.45*dispersive + 0.85*dual_matching + 0.85*similarity)
- **Architecture:** Qwen3-0.6B + MLP projector (1024->512->256, ReLU)
- **Mechanism:** Cosine similarity between query and document embeddings (not yes/no generation)
- **Key innovation:** Listwise processing of 64 documents simultaneously via shared context window

**E2Rank (Alibaba):**
- **Loss:** InfoNCE (tau=0.03) + RankNet (tau=0.1, lambda=2.0)
- **Mechanism:** Cosine similarity as unified scoring for both retrieval and reranking
- **Key innovation:** Listwise prompt reformulated as pseudo-relevance feedback

Source: [Qwen3-Embedding Paper](https://arxiv.org/abs/2506.05176), [Jina v3](https://arxiv.org/html/2509.25085v2), [BGE Documentation](https://bge-model.com/tutorial/5_Reranking/5.1.html)

### 5.3 Is BCE on Binary Labels Sufficient?

**Arguments for BCE (binary):**
- Simplest approach, proven effective (Qwen3-Reranker SOTA)
- Natural fit for decoder-only models (yes/no next-token prediction)
- Sufficient for most benchmarks (BEIR, MS MARCO)
- Most labeled datasets only provide binary relevance labels anyway

**Arguments for graded relevance:**
- TREC datasets have 0-3 graded labels
- MarginMSE can capture graded relevance through teacher score distillation
- ListMLE/LambdaRank directly optimize ranking metrics with graded relevance

**Recommendation:** Use **BCE on binary labels (yes/no token cross-entropy)** as the primary reranking loss. This is what Qwen3-Reranker uses and is sufficient for EMNLP publication. If you want to strengthen results:
1. Add **RankNet pairwise loss** (as in E2Rank, with lambda=2.0 weight)
2. Or add **MarginMSE distillation** from a larger teacher reranker
3. Save listwise approaches for future work (adds significant complexity)

### 5.4 Reranking Head Design

**Options for the unified model:**

| Design | Description | Used By | Complexity |
|--------|-------------|---------|-----------|
| **No head (token logits)** | Use P(yes)/P(no) from LM head | Qwen3-Reranker | Lowest (reuse existing LM head) |
| **Linear head** | Single linear layer on [CLS]/[EOS] hidden state | Traditional cross-encoders | Low |
| **MLP head** | Multi-layer (e.g., 1024->512->256->1) | Jina v3 (for embedding scoring) | Medium |
| **Attention-based** | Cross-attention scoring mechanism | Some research models | High |

**Recommendation for this project:** Use **no additional head (yes/no token logits)** for reranking. This is the simplest approach, matches Qwen3-Reranker's design, and avoids introducing additional parameters that complicate the MoE-LoRA analysis. The reranking score is:

```
score = log P(yes | instruction, query, document) - log P(no | instruction, query, document)
```

Or equivalently:
```
score = logit_yes - logit_no
```

This keeps the architecture clean: LoRA adapters modify the transformer's internal representations, and the unchanged LM head converts those to relevance scores.

**Alternative (for ablation):** Test adding a lightweight **MLP scoring head** (1024->256->1) on the EOS hidden state, similar to Jina v3. This could provide a useful ablation: does a dedicated scoring head outperform the yes/no generation approach when combined with MoE-LoRA?

---

## Summary of Specific Recommendations

### Architecture
- **Base model:** Qwen3-0.6B-Base (same as Qwen3-Embedding and Jina v3)
- **Embedding extraction:** Last hidden state of EOS token
- **Reranking scoring:** Yes/no token logit difference from LM head
- **LoRA target modules:** q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **LoRA rank:** r=16 per expert (ablate r={4, 8, 16, 32})
- **LoRA alpha:** alpha = rank (16)
- **LoRA dropout:** 0.05
- **PEFT task_type:** None (manual handling for both tasks)

### Training Configuration
- **Optimizer:** AdamW 8-bit
- **Learning rate:** 1e-4 (Stage 1), 5e-5 (Stage 2)
- **Schedule:** Cosine with warmup (warmup_ratio=0.05)
- **Training approach:** Staged (embedding first, then joint)
- **Batch organization:** Alternating batches in Stage 2
- **Loss weighting:** L = L_InfoNCE + lambda * L_rerank, lambda in {0.5, 1.0, 2.0}

### Embedding Training
- **Loss:** CachedMultipleNegativesRankingLoss (InfoNCE)
- **Temperature:** 0.05 (scale=20.0)
- **Effective batch size:** 1024-4096 (via caching)
- **Hard negatives per query:** 7-15
- **Hard negative source:** Cross-encoder mined + positive-aware filtering
- **Optional:** MatryoshkaLoss wrapper for flexible dimensionality

### Reranking Training
- **Loss:** Cross-entropy on yes/no token probabilities (BCE equivalent)
- **Optional additional loss:** RankNet (pairwise) with lambda=2.0
- **Scoring:** logit(yes) - logit(no) from LM head
- **No additional scoring head** (ablate MLP head as comparison)

### Multi-Task Optimization
- **Primary mechanism:** MoE-LoRA routing handles task separation
- **Baseline comparison:** PCGrad for gradient conflict analysis
- **Loss weighting:** Fixed weights first, GradNorm as upgrade
- **Gradient conflict measurement:** Cosine similarity between per-task gradients per layer

---

## Key References

### Base Models and Embedding/Reranking
1. [Qwen3-Embedding Paper](https://arxiv.org/abs/2506.05176) - Qwen3 Embedding and Reranker technical report
2. [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) - Model card confirming Qwen3-0.6B-Base as parent
3. [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) - Reranker model card
4. [Qwen3-Embedding Blog](https://qwenlm.github.io/blog/qwen3-embedding/) - Official blog with training pipeline details
5. [ModernBERT Introduction](https://huggingface.co/blog/modernbert) - Encoder-only alternative
6. [ModernBERT-embed-base](https://huggingface.co/nomic-ai/modernbert-embed-base) - Encoder-based embedding model
7. [Gemini Embedding](https://arxiv.org/html/2503.07891v1) - Google's top MTEB model

### LoRA and MoE-LoRA
8. [How Relevance Emerges (arXiv 2504.08780)](https://arxiv.org/html/2504.08780) - LoRA module analysis for reranking
9. [Unsloth LoRA Hyperparameters Guide](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide) - Practical LoRA configuration guide
10. [PEFT LoRA Documentation](https://huggingface.co/docs/peft/en/package_reference/lora) - Official PEFT library reference
11. [PEFT Feature Extraction Example](https://github.com/huggingface/peft/blob/main/examples/feature_extraction/peft_lora_embedding_semantic_search.py) - FEATURE_EXTRACTION task_type example
12. [Databricks LoRA Guide](https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms) - Comprehensive LoRA best practices
13. [SMoRA (arXiv 2501.15103)](https://arxiv.org/abs/2501.15103) - Single-Ranked MoE LoRA for multi-task learning
14. [MTL-LoRA (arXiv 2410.09437)](https://arxiv.org/abs/2410.09437) - Multi-Task LoRA with shared-private decomposition
15. [MoLA (NAACL 2025)](https://aclanthology.org/2025.findings-naacl.284.pdf) - MoE LoRA with layer-wise expert allocation
16. [R-LoRA (arXiv 2502.15455)](https://arxiv.org/abs/2502.15455) - Randomized Multi-Head LoRA
17. [Ortho-LoRA (arXiv 2601.09684)](https://arxiv.org/abs/2601.09684) - Orthogonal gradient projection for multi-task LoRA

### Unified Embedding+Reranking
18. [E2Rank (arXiv 2510.22733)](https://arxiv.org/html/2510.22733) - Unified embedding+reranking model
19. [GritLM (arXiv 2402.09906)](https://arxiv.org/abs/2402.09906) - Generative Representational Instruction Tuning (ICLR 2024)
20. [Jina Reranker v3 (arXiv 2509.25085)](https://arxiv.org/html/2509.25085v2) - Last but Not Late Interaction reranker
21. [LLM2Vec (COLM 2024)](https://openreview.net/pdf?id=IW1PR7vEBf) - Transforming decoder-only to encoder
22. [Causal2Vec (arXiv 2507.23386)](https://arxiv.org/abs/2507.23386) - Decoder-only embedding without bidirectional attention

### Contrastive Learning and Loss Functions
23. [Sentence-Transformers Losses](https://sbert.net/docs/package_reference/sentence_transformer/losses.html) - Complete loss function reference
24. [Sentence-Transformers Loss Overview](https://sbert.net/docs/sentence_transformer/loss_overview.html) - Loss selection guide
25. [Sentence-Transformers Training Overview](https://sbert.net/docs/sentence_transformer/training_overview.html) - Training best practices
26. [Matryoshka Embeddings Guide](https://huggingface.co/blog/matryoshka) - MatryoshkaLoss introduction
27. [NV-Retriever (arXiv 2407.15831)](https://arxiv.org/abs/2407.15831) - Positive-aware hard negative mining (ICLR 2025)
28. [Mitigating False Negatives in MNRL](https://huggingface.co/blog/dragonkue/mitigating-false-negatives-in-retriever-training) - False negative handling
29. [Hard Negatives, Hard Lessons (EMNLP 2025)](https://aclanthology.org/2025.findings-emnlp.481.pdf) - Hard negative mining comparison

### Reranking
30. [BGE Reranker Documentation](https://bge-model.com/tutorial/5_Reranking/5.1.html) - BGE reranker training details
31. [Evolution of Reranking Models](https://arxiv.org/html/2512.16236v1) - Comprehensive survey
32. [How Good are LLM-based Rerankers? (EMNLP 2025)](https://aclanthology.org/2025.findings-emnlp.305/) - Empirical evaluation
33. [REARANK (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.125/) - Reasoning re-ranking agent

### Multi-Task Optimization
34. [PCGrad Paper (NeurIPS 2020)](https://proceedings.neurips.cc/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf) - Gradient Surgery for Multi-Task Learning
35. [GCond (arXiv 2509.07252)](https://arxiv.org/abs/2509.07252) - Gradient Conflict Resolution

### Optimizer and Training
36. [AdamW 8-bit Comparison](https://kaitchup.substack.com/p/fine-tuning-llms-with-32-bit-8-bit) - Optimizer memory comparison
37. [LoRA Practical Tips (Sebastian Raschka)](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) - Practical LoRA fine-tuning guide
38. [LoRA Training 2025 Guide](https://sanj.dev/post/lora-training-2025-ultimate-guide) - Modern LoRA training techniques

### Venue
39. [EMNLP 2026 Deadlines](https://mlciv.com/ai-deadlines/conference/?id=emnlp26) - Conference deadline tracking
