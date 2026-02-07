# Unified Embedding and Reranking via MoE-LoRA: Mitigating Task Interference through Expert Routing

> Semi-formal Research Proposal Draft

**Target Venues**: ACL, EMNLP, NAACL (NLP); SIGIR, CIKM (IR); ICLR, NeurIPS (ML)
**Status**: Novel combination — no existing work applies MoE-LoRA to unified retrieval + reranking

---

## Abstract

Modern information retrieval systems rely on a two-stage pipeline: a fast embedding-based retriever (dual-encoder) followed by an accurate but slow reranker (cross-encoder). Deploying two separate models doubles memory, eliminates cross-task knowledge sharing, and increases systems complexity. Recent unified approaches (GRITLM, E2Rank) use a single parameter set for both tasks but suffer from task interference. We propose **MoE-LoRA Unified Retriever**, a parameter-efficient architecture that attaches multiple LoRA expert adapters to a frozen base model (e.g., Qwen3-0.6B) with a learned router that dynamically selects expert combinations based on input characteristics. This design enables task-specific specialization (via dedicated embedding and reranking experts) while preserving knowledge sharing (via a shared expert and common backbone). We hypothesize that explicit expert routing can factorize the representation space to mitigate task interference, achieving competitive performance on both MTEB (embedding) and BEIR (reranking) benchmarks with ~44% memory reduction compared to two separate models.

---

## 1. Introduction

### 1.1 Problem

The standard two-stage retrieval pipeline operates as follows:

1. **Stage 1 — Retrieval (Embedding Model)**: Encodes queries and documents independently into dense vectors; retrieves top-K candidates via approximate nearest neighbor search (~10ms).
2. **Stage 2 — Reranking (Reranker Model)**: Jointly encodes query-document pairs via cross-attention; scores and reorders top-K candidates for high-quality results (~100ms).

Currently, each stage uses a separate model with independent parameters, training pipelines, and deployment infrastructure. For example, pairing Qwen3-Embedding-0.6B with Qwen3-Reranker-0.6B requires 2.4 GB VRAM total, with no knowledge transfer between the two.

### 1.2 Limitations of Existing Unified Approaches

| Approach | Mechanism | Limitation |
|----------|-----------|------------|
| **GRITLM** (ICLR 2025) | Single model; bidirectional attention for embedding, causal for generation | Same parameters for both tasks → task interference |
| **E2Rank** (Alibaba NLP, 2025) | Single embedding model with listwise reranking via pseudo-relevance feedback | Compromises on both tasks; no explicit specialization mechanism |
| **Jina Embeddings v3** (ECIR 2025) | Shared backbone + hard task-specific LoRA selection | No soft routing; adapters chosen by explicit task ID, not learned |

### 1.3 Core Research Question

> Can a shared backbone augmented with MoE-LoRA experts and learned routing achieve competitive performance on both embedding retrieval and reranking, while being more memory-efficient than two separate models and more specialized than single-parameter unified models?

### 1.4 Hypothesis

**H1**: Unified retrieval + reranking suffers from fundamental task interference that can be decomposed into distinct representational subspaces. MoE-LoRA with task-aware routing provides a principled mechanism to disentangle and share these subspaces, mitigating interference while enabling positive knowledge transfer.

---

## 2. Related Work

### 2.1 Unified Retrieval + Reranking

- **GRITLM** (Muennighoff et al., ICLR 2025): Unifies embedding and generation via instruction tuning on Mistral-7B; achieves MTEB SoTA but uses a single parameter set.
- **E2Rank** (Alibaba NLP, 2025): Extends an embedding model with listwise reranking through continued training with RankNet loss; achieves strong BEIR scores with low latency. Submitted to ICLR 2026 but withdrawn.
- **Qwen3-Embedding/Reranker** (Alibaba, 2025): Separate but complementary models (0.6B to 8B) using LoRA fine-tuning from the Qwen3 base; achieves #1 on MTEB multilingual leaderboard.

### 2.2 Task-Specific LoRA Adapters

- **Jina Embeddings v3** (ECIR 2025): 570M parameter model with 5 task-specific LoRA adapters (retrieval.query, retrieval.passage, separation, classification, text-matching); adds <3% parameters. Adapters are selected by explicit task ID.
- **BSharedRAG** (EMNLP Findings 2024): Shared backbone with LoRA modules for retrieval and generation in RAG; shows 5–13% improvement in retrieval and 23% in generation over baselines.

### 2.3 MoE-LoRA Architectures

- **MOLE** (Microsoft, ICLR 2024): Treats each layer of trained LoRAs as distinct experts with learned gating; supports hierarchical weight control.
- **RAMoLE** (OpenReview 2024): Retrieval-augmented MoE over LoRA experts with input-aware routing via attention-based weighting.
- **SMoRA** (Jan 2025): Treats each LoRA rank as an independent expert for finer-grained sharing.
- **DR-LoRA** (Jan 2026): Dynamic rank growth for MoE-adapted LLMs based on expert saliency scoring.
- **MoLA** (NAACL 2025): Layer-wise expert allocation for MoE-LoRA.

### 2.4 Gap

No existing work applies MoE-LoRA routing specifically to the unified embedding retrieval + reranking problem. Jina v3 uses hard task selection (not soft routing), and MoE-LoRA works (MOLE, RAMoLE, SMoRA) focus on general multi-task or domain composition rather than the retrieval-reranking task pair.

---

## 3. Proposed Method

### 3.1 Architecture Overview

```
┌────────────────────────────────────────────────────────────┐
│              MoE-LoRA UNIFIED MODEL                        │
├────────────────────────────────────────────────────────────┤
│                                                            │
│              ┌─────────────────────────┐                   │
│              │   Base Model (Frozen)   │                   │
│              │   e.g., Qwen3-0.6B      │                   │
│              │   (~1.2 GB)             │                   │
│              └───────────┬─────────────┘                   │
│                          │                                 │
│              ┌───────────┴───────────┐                     │
│              │    Learned Router     │                     │
│              │    (~5 MB trainable)  │                     │
│              └───────────┬───────────┘                     │
│                          │                                 │
│         ┌────────────────┼────────────────┐                │
│         ▼                ▼                ▼                │
│   ┌───────────┐   ┌───────────┐   ┌───────────┐           │
│   │ Embedding │   │ Reranking │   │  Shared   │           │
│   │  Expert   │   │  Expert   │   │  Expert   │           │
│   │ (LoRA)    │   │ (LoRA)    │   │ (LoRA)    │           │
│   │ ~50 MB    │   │ ~50 MB    │   │ ~50 MB    │           │
│   └───────────┘   └───────────┘   └───────────┘           │
│                                                            │
│   Total: ~1.35 GB  (vs. 2.4 GB for two models)            │
└────────────────────────────────────────────────────────────┘
```

**Key components**:
- **Frozen base model**: Qwen3-0.6B-Base (or similar), providing general language understanding.
- **LoRA experts**: Low-rank adapters (rank 32, ~50 MB each) specialized for embedding, reranking, and shared knowledge.
- **Learned router**: A small network that produces soft routing weights over experts conditioned on the input representation.
- **Task-specific heads**: EOS-token pooling + normalization (embedding) or [CLS]-token classification head (reranking).

### 3.2 Operational Modes

**Embedding mode** (single text input):
- Router emphasizes embedding expert + shared expert.
- Output: normalized last-token embedding for dense retrieval.

**Reranking mode** (query-document pair):
- Router emphasizes reranking expert + shared expert.
- Output: scalar relevance score from classification head.

### 3.3 Mathematical Formulation

For base model hidden states **h**, the enhanced representation is:

$$h' = h + \sum_{i=1}^{N} w_i \cdot E_i(h)$$

Where:
- $w_i = \text{Router}(h)_i$ are soft routing weights (sum to 1)
- $E_i(h) = B_i \cdot A_i \cdot h$ is the LoRA expert output
- $A_i \in \mathbb{R}^{r \times d}$, $B_i \in \mathbb{R}^{d \times r}$ are low-rank matrices

### 3.4 Router Design

We propose a **learned sequence-level router** as the primary design:

```python
class LearnedRouter(nn.Module):
    def __init__(self, hidden_size, num_experts=3):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, num_experts)
        )

    def forward(self, hidden_states):
        pooled = hidden_states[:, 0, :]
        logits = self.gate(pooled) / self.temperature
        weights = F.softmax(logits, dim=-1)
        return weights
```

Additional variants for ablation: task-explicit router and token-level router.

### 3.5 Training Procedure

**Joint multi-task training** with three loss components:

1. **Embedding loss** (InfoNCE/contrastive):
$$\mathcal{L}_{emb} = -\log \frac{\exp(\text{sim}(q, p^+) / \tau)}{\exp(\text{sim}(q, p^+) / \tau) + \sum_{p^-} \exp(\text{sim}(q, p^-) / \tau)}$$

2. **Reranking loss** (binary cross-entropy):
$$\mathcal{L}_{rank} = -[y \log(\sigma(s)) + (1-y) \log(1 - \sigma(s))]$$

3. **Router auxiliary loss** (load balancing):
$$\mathcal{L}_{aux} = N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

**Combined**: $\mathcal{L} = \mathcal{L}_{emb} + \mathcal{L}_{rank} + \lambda \cdot \mathcal{L}_{aux}$

Only LoRA experts and router parameters are trainable (~150 MB); the base model remains frozen.

---

## 4. Evaluation Plan

### 4.1 Benchmarks

| Benchmark | Purpose | Metric |
|-----------|---------|--------|
| **MTEB** | Embedding quality (retrieval, classification, clustering, STS) | Average score across tasks |
| **BEIR** | Reranking quality across diverse domains | nDCG@10 |
| **MS MARCO** | Passage retrieval and reranking | MRR@10, nDCG@10 |
| **End-to-end RAG** | Unified pipeline quality (retrieval → rerank → QA) | Answer accuracy, latency |

### 4.2 Baselines

1. **Two separate models**: Qwen3-Embedding + Qwen3-Reranker (upper bound for quality, lower bound for efficiency)
2. **GRITLM**: Unified embedding + generation (single parameter set)
3. **E2Rank**: Unified embedding + listwise reranking
4. **Jina v3-style**: Shared backbone + hard task-LoRA selection (ablation of our method without soft routing)

### 4.3 Ablation Studies

| Ablation | What it tests |
|----------|---------------|
| Router: explicit vs. learned vs. token-level | Whether learned routing adds value over hard task selection |
| Number of experts: 2 vs. 3 vs. 4 | Optimal expert count |
| LoRA rank: 16 vs. 32 vs. 64 | Capacity vs. efficiency trade-off |
| With/without shared expert | Role of general knowledge expert |
| Training data ratio (embedding : reranking) | Sensitivity to task balance |
| Single LoRA (no MoE) | Quantify task interference without expert separation |

### 4.4 Analysis

- **Routing visualization**: How do routing weights distribute across tasks, datasets, and layers?
- **Expert specialization**: What does each expert learn? (probing, attention pattern analysis)
- **Knowledge transfer**: Freeze one expert, train the other — does shared expert help?
- **Task interference quantification**: Compare single-adapter vs. multi-adapter vs. MoE routing
- **Failure case analysis**: When does the model underperform separate models?

---

## 5. Expected Contributions

1. **First MoE-LoRA architecture for unified retrieval + reranking** — demonstrating that explicit expert routing mitigates task interference in this specific dual-task setting.
2. **Empirical study of task interference** in unified IR models, with evidence that embedding and reranking occupy separable representational subspaces.
3. **Deep routing and expert analysis** — interpretability study of what each expert learns and when routing decisions change.
4. **Practical efficiency gains** — ~44% memory reduction over two-model setups with competitive quality, validated in realistic deployment scenarios.
5. **Extensibility framework** — demonstrating that new task experts (classification, QA, summarization) can be added modularly.

---

## 6. Challenges and Mitigations

| Challenge | Mitigation Strategy |
|-----------|-------------------|
| Different input formats (single text vs. pair) | Task indicator or learn from input structure (presence of separator tokens) |
| Different output heads (embedding vs. score) | Shared backbone, separate lightweight heads |
| Router collapse (always same expert) | Auxiliary load-balancing loss, expert dropout, noise injection |
| Training data imbalance | Balanced sampling, alternating batches |
| Negative transfer between tasks | Careful initialization, gradual unfreezing |
| Marginal gains over E2Rank/Jina v3 | Focus on interpretability and analysis depth if gains are small |

---

## 7. Timeline

| Phase | Activities |
|-------|-----------|
| **Month 1–2** | Literature deep-dive (MoE-LoRA + unified IR); formulate hypothesis; identify specific research gaps |
| **Month 3–4** | Implement architecture + training pipeline; reproduce baselines (Qwen3, GRITLM, E2Rank); initial small-scale experiments |
| **Month 5–6** | Full-scale training; MTEB + BEIR evaluation; ablation studies |
| **Month 7** | Routing visualization; knowledge transfer analysis; expert specialization study; draft paper |
| **Month 8** | Finalize paper; internal review; submit to target venue; prepare rebuttal materials |

---

## 8. Key References

1. Muennighoff et al. "Generative Representational Instruction Tuning" (GRITLM). ICLR 2025.
2. Alibaba NLP. "E2Rank: Your Text Embedding can Also be an Effective and Efficient Listwise Reranker." arXiv:2510.22733, 2025.
3. Qwen Team. "Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models." arXiv:2506.05176, 2025.
4. Sturua et al. "jina-embeddings-v3: Multilingual Embeddings With Task LoRA." ECIR 2025.
5. Wu et al. "Mixture of LoRA Experts" (MOLE). ICLR 2024.
6. Wang et al. "RAMoLE: Retrieval-Augmented Mixture of LoRA Experts." OpenReview, 2024.
7. SMoRA. "Each Rank Could be an Expert: Single-Ranked Mixture of Experts LoRA." arXiv:2501.15103, 2025.
8. DR-LoRA. "Dynamic Rank LoRA for Mixture-of-Experts Adaptation." arXiv:2601.04823, 2026.
9. Guan et al. "BSharedRAG: Backbone Shared Retrieval-Augmented Generation." EMNLP Findings 2024.
10. MoLA. "MoE LoRA with Layer-wise Expert Allocation." NAACL 2025.

---

*Draft created: February 2026*
