# When Does Task Interference Emerge in Unified Retrieval and Reranking? A Study via Mixture-of-LoRA-Expert Routing

> Research Proposal v2 — February 2026

---

## 1. Executive Summary

Modern information retrieval deploys separate embedding (dual-encoder) and reranking (cross-encoder) models, doubling memory cost with no parameter sharing. Recent unified approaches (GRITLM, E2Rank) claim these tasks can coexist in a single model — yet GRITLM achieves this only at 7B scale with architectural task separation (bidirectional vs. causal attention), and E2Rank was withdrawn from ICLR 2026. A fundamental question remains unanswered: **under what conditions does task interference between embedding and reranking actually emerge, and how can it be systematically factorized?**

This proposal presents **the first systematic study of retrieval-reranking task interference** across model scales, and introduces **UniMoER** (Unified MoE-LoRA for Embedding and Reranking), a parameter-efficient architecture that uses learned expert routing to factorize conflicting representational demands. Our primary contribution is analytical — we quantify when interference emerges, where it localizes in the network, and how routing decomposes it — with the architecture serving as both an intervention tool and a practical solution achieving ~44% memory reduction.

**Central hypotheses:**
1. Task interference is a **capacity-dependent phenomenon** that intensifies at smaller model scales where embedding and reranking must compete for limited representational capacity.
2. MoE-LoRA routing provides a principled mechanism to **factorize** this interference into separable subspaces while preserving beneficial cross-task transfer.

**Primary target venue:** EMNLP 2026 (ARR submission ~May-Jun 2026).

---

## 2. Problem Statement and Motivation

### 2.1 The Two-Stage Retrieval Pipeline

The dominant paradigm in neural information retrieval operates in two stages:

1. **Embedding Retrieval (Dual-Encoder):** Queries and documents are independently encoded into dense vectors; top-K candidates are retrieved via ANN search. Architecture: bi-encoder with independent encoding. Latency: ~10ms.

2. **Reranking (Cross-Encoder):** Query-document pairs are jointly encoded with full cross-attention and re-scored. Architecture: cross-encoder with joint attention. Latency: ~100ms per pair.

Deploying Qwen3-Embedding-0.6B alongside Qwen3-Reranker-0.6B requires 2.4 GB VRAM total, with independent training and no parameter sharing.

### 2.2 The Task Interference Problem

Embedding and reranking impose structurally different demands on shared representations:

| Dimension | Embedding (Dual-Encoder) | Reranking (Cross-Encoder) |
|-----------|--------------------------|---------------------------|
| **Input** | Single text | Query-document pair |
| **Attention** | Independent encoding | Joint cross-attention |
| **Objective** | Contrastive (push/pull in metric space) | Pointwise/listwise relevance scoring |
| **Representation** | Global semantic similarity | Fine-grained relevance matching |
| **Output** | Dense vector (pooled) | Scalar relevance score |

When a single model must serve both tasks with shared parameters, these conflicting objectives compete — a phenomenon we call **retrieval-reranking task interference**.

### 2.3 The GRITLM Paradox

GRITLM (ICLR 2025) claims "no performance loss" when unifying embedding and generation in a single 7B model. This appears to contradict our interference premise. However, three critical distinctions apply:

1. **Architectural separation, not true parameter sharing.** GRITLM uses bidirectional attention for embedding and causal attention for generation — different compute paths, not a shared-parameter conflict.
2. **Scale absorbs interference.** At 7B parameters, the model has sufficient capacity to accommodate both tasks. We hypothesize interference intensifies at smaller scales (0.6B-4B) where capacity is constrained.
3. **Different task pair.** Embedding + generation is structurally less conflicting than embedding + reranking. Generation uses causal decoding; reranking uses cross-encoding over concatenated query-document pairs — a fundamentally different attention pattern from embedding's independent encoding.

This motivates our core research question: **Is task interference a capacity-dependent phenomenon that becomes pronounced at practical deployment scales?**

### 2.4 Why This Matters

- **Production cost:** Two separate models double VRAM, doubling deployment cost on GPU-constrained infrastructure.
- **Edge/mobile IR:** Memory-limited devices (4-8 GB) cannot host two full models.
- **RAG systems:** A single unified model simplifies the retrieve-rerank-generate stack.
- **Scientific understanding:** No prior work has systematically studied when and where interference emerges between these two fundamental IR tasks.

---

## 3. Related Work and Landscape Analysis

### 3.1 Unified Retrieval + Reranking (2024-2026)

| Work | Venue | Approach | Limitation |
|------|-------|----------|------------|
| **GRITLM** (Muennighoff et al.) | ICLR 2025 | Bidirectional attention for embedding, causal for generation; reranks via generative permutation | Architectural task separation (not shared-parameter); 7B scale; reranking via generation, not cross-encoding |
| **E2Rank** (Liu et al.) | ICLR 2026 withdrawn | Single embedding model + listwise reranking via PRF-style query construction | No explicit task specialization; withdrawn from ICLR 2026 |
| **UR2N** (Bhat et al.) | COLING 2025 (Industry) | Unified encoder-decoder with XTR parallel layer | Industry track; encoder-decoder only; modest scale |
| **FreeRet** | ICLR 2026 under review | Training-free MLLM as unified embedder + reranker | Multimodal; no fine-tuning; no task-specific adaptation |
| **Autoregressive Ranking** (Jan 2026) | arXiv 2601.05588 | Generative docID ranking bridging dual/cross encoder gap | Theoretical focus; generative paradigm, not adapter-based |

**Gap:** No published work provides a systematic empirical study of when and why task interference emerges between embedding and reranking, or demonstrates a principled factorization mechanism.

### 3.2 Task-Specific LoRA Adapters for Retrieval

| Work | Venue | Approach | Limitation |
|------|-------|----------|------------|
| **Jina Embeddings v3** (Sturua et al.) | ECIR 2025 | 570M model with 5 task-specific LoRA adapters; hard selection by task ID | No soft routing; no MoE dynamics; no reranking-specific expert |
| **RouterRetriever** (Lee et al.) | AAAI 2025 | Domain-specific LoRA experts with similarity-based routing for retrieval | Hard routing (similarity-based, not learned); retrieval only, no reranking |
| **BSharedRAG** (Guan et al.) | EMNLP Findings 2024 | Shared backbone + LoRA for retrieval + generation in e-commerce | Hard task separation; domain-specific |

**Gap:** Jina v3 and RouterRetriever demonstrate task/domain-specific LoRAs work for retrieval with **hard** routing. No work has explored **learned soft routing** between embedding and reranking LoRA experts.

### 3.3 MoE-LoRA Architectures (2024-2026)

| Work | Venue | Key Innovation |
|------|-------|----------------|
| **MOLE** (Wu et al.) | ICLR 2024 | Layer-wise gating over pre-trained LoRA experts |
| **MoLA** (Gao et al.) | NAACL 2025 Findings | Layer-wise expert allocation; more experts in higher layers |
| **SMoRA** (Jan 2025) | arXiv | Each LoRA rank as independent expert |
| **LD-MoLE** (Zhuang et al.) | arXiv, Sep 2025 | Differentiable Sparsegen replacing TopK; layer-wise adaptive routing |
| **DynMoLE** (Li et al.) | arXiv, Apr 2025 | Tsallis entropy-based hybrid routing |
| **DR-LoRA** (Jan 2026) | arXiv | Dynamic rank growth via expert saliency |
| **MoLE-CIE** (Wang et al.) | EMNLP Findings 2025 | MoE-LoRA for continual information extraction |

**Gap:** All MoE-LoRA works target general multi-task composition. None apply MoE-LoRA routing to the specific embedding-reranking pair, where tasks have well-defined, structurally different input-output characteristics.

### 3.4 Task Interference Factorization with LoRA

| Work | Venue | Key Idea |
|------|-------|----------|
| **FVAE-LoRA** (Kumar et al.) | NeurIPS 2025 | VAE-based latent space factorization for LoRA; disentangles task-salient features |
| **LoRI** (Apr 2025) | arXiv | Frozen A matrices + task-specific masks; orthogonal subspaces reduce interference |
| **OSRM** (May 2025) | arXiv | Orthogonal subspace constraints prior to fine-tuning |
| **TC-LoRA** (Aug 2025) | arXiv | Tensorized clustering + CP decomposition for task disentanglement |

**Gap:** These works address interference in general NLP tasks (reasoning, coding, QA). None study the structurally distinct embedding-reranking task pair, where interference has a clear geometric interpretation (metric space vs. relevance space).

### 3.5 Interpretability of LoRA in Reranking

| Work | Venue | Key Finding |
|------|-------|-------------|
| **"How Relevance Emerges"** (Nijasure et al.) | arXiv 2504.08780 | LoRA rank 1 sufficient for reranking; layers 5-15 most important; MLP up/gate projections most impactful |

### 3.6 MoE in Embedding Models

| Work | Details |
|------|---------|
| **Nomic Embed V2** (Feb 2025) | First MoE embedding model; 475M total / 305M active; 8 experts with top-2 routing |

**Note:** Nomic uses MoE in the **base architecture** for efficiency, not for task routing. Our approach uses MoE at the **adapter level** for task specialization.

### 3.7 Summary of Research Gap

```
Unified Embedding + Reranking       ✅ GRITLM, E2Rank, UR2N, FreeRet
                                        → No interference analysis; no factorization

Task-Specific LoRA for IR            ✅ Jina v3, RouterRetriever, BSharedRAG
                                        → Hard routing only; no learned MoE

MoE-LoRA with Learned Routing        ✅ MOLE, MoLA, LD-MoLE, DynMoLE, ...
                                        → General multi-task; not IR-specific

Interference Factorization via LoRA  ✅ FVAE-LoRA, LoRI, OSRM, TC-LoRA
                                        → General NLP; not embedding+reranking

Systematic Study of E+R Interference  ❌ NO EXISTING WORK
+ MoE-LoRA Factorization for IR         This is our contribution.
```

---

## 4. Research Hypotheses

### H1 (Scale-Dependent Interference)
Task interference between embedding and reranking is a **capacity-dependent phenomenon**: it is measurable and significant at practical deployment scales (0.6B-4B) but diminishes at larger scales (7B+), explaining why GRITLM reports no loss at 7B.

### H2 (Gradient Conflict)
Embedding and reranking produce **conflicting gradient signals** in shared parameters, with conflict concentrated in specific layers. This gradient conflict is the mechanistic cause of task interference.

### H3 (Expert Factorization)
MoE-LoRA routing can factorize the representation space into separable task-specific subspaces, mitigating interference while preserving beneficial knowledge transfer through a shared expert.

### H4 (Soft Routing Advantage)
Learned soft routing over LoRA experts outperforms hard task-selection (Jina v3-style) and hard similarity-based routing (RouterRetriever-style) because: (a) some inputs benefit from mixed expertise, and (b) the shared expert captures task-invariant knowledge.

### H5 (Knowledge Transfer)
The shared expert enables positive transfer: training on embedding data improves reranking (and vice versa) compared to isolated task-specific LoRA training.

---

## 5. Proposed Method: UniMoER

### 5.1 Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│                 UniMoER: Unified MoE-LoRA for E+R              │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│            ┌─────────────────────────────────┐                 │
│            │   Base Model (Frozen)            │                 │
│            │   e.g., Qwen3-0.6B              │                 │
│            │   (~1.2 GB, all params frozen)   │                 │
│            └────────────────┬────────────────┘                 │
│                             │                                  │
│            ┌────────────────┴────────────────┐                 │
│            │     Per-Layer Learned Router     │                 │
│            │     (~5 MB total trainable)      │                 │
│            └────────────────┬────────────────┘                 │
│                             │                                  │
│        ┌────────────────────┼────────────────────┐             │
│        ▼                    ▼                    ▼              │
│  ┌───────────┐       ┌───────────┐       ┌───────────┐        │
│  │ Embedding │       │ Reranking │       │  Shared   │        │
│  │  Expert   │       │  Expert   │       │  Expert   │        │
│  │(LoRA, r=R)│       │(LoRA, r=R)│       │(LoRA,r=Rs)│        │
│  └───────────┘       └───────────┘       └───────────┘        │
│                                                                │
│  Task Heads:                                                   │
│  ├── Embedding: last-token pooling + L2 norm                  │
│  └── Reranking: first-token pooling → linear → sigmoid        │
│                                                                │
│  Total: ~1.35 GB (vs 2.4 GB for two separate models)          │
│  Trainable: ~155 MB (~12% of total)                            │
└────────────────────────────────────────────────────────────────┘
```

### 5.2 Components

**Frozen Base Model.** A pre-trained language model (e.g., Qwen3-0.6B-Base). All base parameters remain frozen.

**LoRA Expert Adapters.** Three low-rank adapters at each targeted layer, each with down-projection A_i (d -> r) and up-projection B_i (r -> d):
- Expert 0: Embedding specialist (rank R)
- Expert 1: Reranking specialist (rank R)
- Expert 2: Shared expert (rank R_s, potentially asymmetric)

Initialization: A with Kaiming uniform, B to zero (starting from base model behavior).

**Per-Layer Learned Router.** Lightweight network producing soft routing weights, conditioned on input representation at each layer:

$$w^{(l)} = \text{Sparsegen}(\text{MLP}^{(l)}(h^{(l)}_{\text{pool}}) / \tau)$$

Following LD-MoLE's differentiable Sparsegen for adaptive, layer-wise expert activation.

**Task-Specific Heads.**
- Embedding: Pool last non-padding token, L2-normalize.
- Reranking (pointwise): Pool first token, linear projection to scalar, sigmoid.
- Reranking (listwise, optional): Following Jina Reranker v3's late interaction pattern.

### 5.3 Forward Pass

For hidden states h at layer l:

$$h'^{(l)} = h^{(l)} + \sum_{i=1}^{N} w_i^{(l)} \cdot B_i^{(l)} A_i^{(l)} h^{(l)}$$

**Embedding mode** (single text): Independent encoding -> router emphasizes embedding + shared experts -> last-token pooling -> L2 norm -> dense vector.

**Reranking mode** (query + document pair): Joint encoding of concatenated input -> router emphasizes reranking + shared experts -> first-token pooling -> reranking head -> relevance score.

### 5.4 Training Objective

**Embedding Loss (InfoNCE):**
$$\mathcal{L}_{\text{emb}} = -\log \frac{\exp(\text{sim}(q, p^+) / \tau_e)}{\exp(\text{sim}(q, p^+) / \tau_e) + \sum_{p^-} \exp(\text{sim}(q, p^-) / \tau_e)}$$

**Reranking Loss (Binary Cross-Entropy):**
$$\mathcal{L}_{\text{rank}} = -[y \log(\sigma(s)) + (1-y) \log(1 - \sigma(s))]$$

**Router Auxiliary Loss (Load Balancing):**
$$\mathcal{L}_{\text{aux}} = N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

**Combined:**
$$\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{emb}} + \beta \cdot \mathcal{L}_{\text{rank}} + \lambda \cdot \mathcal{L}_{\text{aux}}$$

Default: alpha=1, beta=1, lambda=0.01.

**Training strategy:**
- Mixed-task batches with balanced sampling (50% embedding, 50% reranking).
- In-batch negatives for contrastive loss.
- Router noise injection during training (Gaussian noise to routing logits) to prevent collapse.

### 5.5 Router Design Variants (for Ablation)

| Variant | Description |
|---------|-------------|
| **Task-Explicit (Hard)** | Fixed weights by task indicator (Jina v3 style) |
| **Similarity-Based (Hard)** | Expert selection by query-pilot similarity (RouterRetriever style) |
| **Per-Layer Learned + Sparsegen (default)** | Differentiable per-layer routing (LD-MoLE) |
| **Per-Layer Learned + Tsallis** | Entropy-based hybrid routing (DynMoLE) |
| **Token-Level** | Per-token per-layer routing (finest granularity) |

---

## 6. Experimental Design

The experimental design follows a **gated structure**: Experiment 1 is a kill gate — if interference is not measurable, the project pivots to a pure interpretability study.

### 6.1 Track A: Interference Analysis (Primary Contribution)

#### Experiment 1: Task Interference Quantification [KILL GATE]

**Goal:** Establish that task interference exists and is measurable.

| Configuration | Description | What It Tests |
|---------------|-------------|---------------|
| **Emb-Only LoRA** | Single LoRA on embedding data only | Embedding upper bound |
| **Rank-Only LoRA** | Single LoRA on reranking data only | Reranking upper bound |
| **Joint-Single LoRA** | Single LoRA on both tasks jointly | Interference without separation |
| **Joint-MoE-LoRA (Ours)** | MoE-LoRA with 3 experts, jointly | Interference mitigation via routing |

**Kill gate criteria:** If Joint-Single LoRA shows <1% degradation vs. specialists on both MTEB and BEIR (nDCG@10), pivot to interpretability-only study. If >=2% degradation on either task, proceed with full plan.

**Statistical rigor:** 3 seeds per configuration, report mean +/- std.

#### Experiment 2: Scale-Dependent Interference

**Goal:** Demonstrate that interference is a capacity phenomenon, explaining GRITLM's result.

| Scale | Model | Interference Measurement |
|-------|-------|-------------------------|
| 0.6B | Qwen3-0.6B-Base | Primary |
| 1.5B | Qwen3-1.5B-Base | Interpolation |
| 4B | Qwen3-4B-Base | Interpolation |
| 7B | Mistral-7B-v0.3 | GRITLM-scale comparison |

For each scale: run Emb-Only, Rank-Only, and Joint-Single LoRA. Measure performance drop ratio at each scale. Plot interference magnitude vs. model scale.

**Expected finding:** Interference decreases monotonically with scale, approaching zero at 7B — providing a principled explanation for GRITLM's "no performance loss" result and motivating parameter-efficient factorization at practical scales.

#### Experiment 3: Gradient Conflict Analysis

**Goal:** Identify the mechanistic cause of interference via gradient-level measurement.

**Method:**
1. In mixed-task training, compute per-layer gradients from embedding batches (g_emb) and reranking batches (g_rank) separately.
2. Measure cosine similarity cos(g_emb, g_rank) per layer.
3. Measure gradient magnitude ratio ||g_emb|| / ||g_rank|| per layer.
4. Apply PCGrad-style projection to quantify conflict magnitude.

**Expected finding:** Gradient conflict concentrates in specific layers (likely mid-range, following "How Relevance Emerges" findings), and MoE routing resolves conflict by directing conflicting gradients to separate experts.

**Methodology reference:** "To See a World in a Spark of Neuron" (2025) for neuronal subspace decomposition.

#### Experiment 4: Routing Behavior Analysis

**Goal:** Deep interpretability of learned routing.

- **Per-task routing distributions:** How do routing weights differ for embedding vs. reranking inputs?
- **Per-layer routing patterns:** Do different layers prefer different experts? (Connecting to MoLA findings.)
- **Training dynamics:** How do routing patterns evolve during training?
- **Domain-dependent routing:** Do different BEIR domains trigger different routing patterns?
- **Expert probing:** What linguistic/semantic properties does each expert capture? (Probing classifiers following "How Relevance Emerges" methodology.)

#### Experiment 5: Representation Space Analysis

**Goal:** Provide evidence that experts factorize the representation space.

- **CKA (Centered Kernel Alignment):** Measure representation similarity per expert across tasks.
- **t-SNE/UMAP visualization:** Cluster structure with vs. without MoE routing.
- **Singular value analysis:** Compare SVD spectra of expert weight matrices.
- **STIR (Subspace Task Interference Ratio):** Measure overlap between task-specific subspaces before and after MoE routing.

### 6.2 Track B: Architecture Validation

#### Experiment 6: Soft Routing vs. Hard Selection

**Goal:** Demonstrate learned routing advantage.

| Configuration | Description |
|---------------|-------------|
| **Hard-Switch (Jina v3 style)** | Two LoRAs selected by task indicator |
| **Hard-Switch + Shared** | Two task LoRAs + shared, selected by indicator |
| **Similarity-Based (RouterRetriever style)** | Expert selection by query-pilot similarity |
| **Soft-Router (Ours)** | Three experts with learned per-layer routing |

#### Experiment 7: Knowledge Transfer Analysis

**Goal:** Demonstrate positive cross-task transfer via shared expert.

| Configuration | Description |
|---------------|-------------|
| **Freeze Rerank Expert** | Train only embedding + shared; evaluate reranking |
| **Freeze Embed Expert** | Train only reranking + shared; evaluate embedding |
| **Full Training** | All experts jointly |
| **No Shared Expert** | Only embedding + reranking experts |
| **Isolated LoRAs** | Separate task-specific LoRAs, no shared training |

#### Experiment 8: Rank Sensitivity

**Goal:** Determine minimum expert rank needed for each task.

Test r = {1, 4, 8, 16, 32} for each expert independently and jointly.

**Motivation:** "How Relevance Emerges" shows rank 1 suffices for reranking. If embedding also works at low rank, the memory savings become even more compelling. Also test asymmetric ranks (e.g., r=4 for reranking, r=16 for embedding, r=32 for shared).

### 6.3 Track C: Practical Validation

#### Experiment 9: End-to-End RAG Pipeline

**Goal:** Demonstrate practical value in retrieve -> rerank -> generate.

- **Datasets:** Natural Questions, TriviaQA, HotpotQA.
- **Metrics:** Answer accuracy (EM, F1), end-to-end latency, peak GPU memory.
- **Hardware:** Single T4 (16GB), single A100 (40GB).

#### Experiment 10: Reranking Mode Comparison (Pointwise vs. Listwise)

**Goal:** Evaluate both reranking paradigms within the unified framework.

- **Pointwise:** Cross-encoder style (concatenated query+document).
- **Listwise:** Late interaction following Jina Reranker v3's pattern.
- **Evaluate:** Both modes on BEIR and MS MARCO; compare against dedicated Jina Reranker v3.

### 6.4 Benchmarks and Metrics

| Benchmark | Purpose | Metrics |
|-----------|---------|---------|
| **MTEB** | Embedding quality across task categories | Average score, per-category |
| **BEIR** (18 domains) | Retrieval + reranking quality | nDCG@10, MAP |
| **MS MARCO Passage** | Passage retrieval and reranking | MRR@10, nDCG@10 |
| **BRIGHT** | Reasoning-intensive retrieval | nDCG@10 |
| **RAG** (NQ, TriviaQA) | End-to-end pipeline quality | Answer EM/F1, latency, memory |

### 6.5 Baselines

| Baseline | Type | Purpose |
|----------|------|---------|
| **Qwen3-Embedding-0.6B + Qwen3-Reranker-0.6B** | Two separate models | Performance upper bound |
| **Qwen3-Embedding-0.6B + LoRA for reranking** | Single base + task LoRA | Simplest unified approach |
| **Jina v3-style hard LoRA selection** | Shared backbone + hard task LoRA | Ablation: soft vs. hard routing |
| **RouterRetriever-style similarity routing** | LoRA experts + similarity-based selection | Ablation: learned vs. similarity routing |
| **Single LoRA (no MoE)** | Shared backbone + joint LoRA | Ablation: MoE routing benefit |
| **E2Rank (replicated)** | Unified embedding + listwise reranking | Direct competitor |
| **Jina Reranker v3** | Dedicated listwise reranker | Reranking-specific baseline |

### 6.6 Base Models

| Model | Parameters | Purpose |
|-------|-----------|---------|
| Qwen3-0.6B-Base | 0.6B | Primary experiments |
| Qwen3-1.5B-Base | 1.5B | Scale study |
| Qwen3-4B-Base | 4B | Scale study |
| Mistral-7B-v0.3 | 7B | GRITLM-scale comparison |

---

## 7. Expected Contributions

### 7.1 Primary: Analytical Contributions

1. **First systematic study of task interference in unified embedding + reranking.** We quantify interference magnitude, identify where it localizes in the network (per-layer gradient conflict), and demonstrate it is scale-dependent — providing a principled explanation for prior contradictory findings (GRITLM's "no loss" vs. observed degradation).

2. **Gradient-level evidence for interference.** Per-layer gradient conflict analysis reveals the mechanistic cause, not just the symptoms, of retrieval-reranking interference.

3. **Deep routing interpretability.** Layer-wise, domain-wise, and training-stage-wise analysis of how MoE routing factorizes the representation space, extending findings from MoLA and "How Relevance Emerges."

4. **Evidence for soft routing advantage.** Learned routing outperforms hard task-selection (Jina v3-style) and hard similarity-based routing (RouterRetriever-style).

### 7.2 Secondary: Architectural Contributions

5. **UniMoER architecture.** ~44% memory reduction for unified retrieval + reranking with competitive quality.

6. **Rank sensitivity analysis.** Minimum LoRA rank requirements for each task, informing future parameter-efficient IR design.

7. **Pointwise vs. listwise reranking.** Within-architecture comparison of both paradigms.

---

## 8. Risk Analysis and Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| **No measurable interference at 0.6B (<1% gap)** | **Critical** | This is the kill gate. If interference is absent, pivot to: (a) studying *why* interference doesn't emerge (also publishable), or (b) focusing purely on the parameter-efficiency angle. |
| **Performance gap MoE-LoRA vs. specialist LoRAs is marginal** | High | Lean heavily on analysis: the interference study and routing interpretability are valuable independent of the architectural fix's magnitude. |
| **Soft routing does not outperform hard selection** | High | Report as informative negative finding; focus on shared expert transfer and interpretability contributions. |
| **Reviewers perceive as "MoE-LoRA applied to another task pair"** | High | Counter with: (a) analysis-first narrative (the study, not the system), (b) scale-dependent interference finding as standalone contribution, (c) gradient conflict analysis as novel methodology. |
| **GRITLM comparison viewed as unfair (different task pair, scale)** | Medium | Acknowledge explicitly in paper. The comparison is conceptual, not apples-to-apples. Our contribution is showing *when* interference matters. |
| **E2Rank reappears at EMNLP/NeurIPS 2026** | Medium | Our approach is fundamentally different (expert routing vs. single model). Position as complementary. |
| **Router collapse** | Medium | Load-balancing auxiliary loss, expert dropout, noise injection to routing logits. |

---

## 9. Venue Strategy and Timeline

### 9.1 Target Venues

| Venue | Deadline | Status | Fit |
|-------|----------|--------|-----|
| **EMNLP 2026** | ARR ~May-Jun 2026 | **Primary target** (~4 months) | Best: values IR + hypothesis-driven study + analysis |
| **NeurIPS 2026** | ~May 15, 2026 | **Parallel target** (~3.5 months) | Good if framed as representation learning + MoE |
| **SIGIR 2026 Short Paper** | TBD (check immediately) | **Quick win** if open | 4-page preliminary interference study |
| **CIKM 2026** | ~May 2026 (est.) | **Backup** | Systems-focused contribution |

### 9.2 Timeline

| Week | Phase | Activities | Gate |
|------|-------|------------|------|
| 1-2 | Foundation | Implement base: Qwen3-0.6B + single LoRA baseline; set up MTEB/BEIR eval | |
| 3-4 | **Kill Gate** | **Experiment 1: Interference quantification at 0.6B** | **If <1% gap, pivot** |
| 5-6 | Scale Study | Experiment 2: Scale-dependent interference (0.6B, 1.5B, 4B, 7B) | |
| 7-8 | MoE-LoRA | Implement UniMoER; Experiments 6-7 (routing variants, transfer) | |
| 9-10 | Analysis | Experiments 3-5 (gradient conflict, routing behavior, CKA) | |
| 11 | Benchmarks | Experiments 9-10 (RAG pipeline, pointwise vs. listwise) | |
| 12-13 | Writing | Paper draft, visualizations, internal review | |
| 14 | Submit | ARR submission targeting EMNLP 2026 | |

---

## 10. Training Data Sources

| Dataset | Task | Size | Usage |
|---------|------|------|-------|
| **MS MARCO Passage** | Embedding + Reranking | ~8.8M passages, 500K queries | Primary |
| **NQ** | Embedding | ~100K Q-A pairs | Additional embedding |
| **MTEB Training Subsets** | Embedding (various) | Varies | Task-diverse |
| **BEIR Training Splits** | Reranking | Varies by domain | Domain-diverse |

Hard negative mining, in-batch negatives, balanced task sampling.

---

## 11. Implementation Notes

### 11.1 Framework

- **Base:** PyTorch + HuggingFace Transformers
- **PEFT:** HuggingFace PEFT with custom MoE routing layer
- **Router:** Custom Sparsegen (LD-MoLE) + Tsallis entropy variant (DynMoLE)
- **Evaluation:** MTEB library, pytrec_eval for BEIR/MS MARCO
- **Tracking:** Weights & Biases

### 11.2 Hardware Requirements

| Scope | Hardware | Notes |
|-------|----------|-------|
| 0.6B, core experiments | 1x RTX 3090/4090 (24GB) | Sufficient for kill gate + main results |
| Multi-scale study (1.5B-4B) | 1x A100 (40GB) | Scale-dependent interference |
| 7B, GRITLM comparison | 2x A100 (80GB) | For full scale study |

---

## 12. Differentiation from Closest Related Work

### 12.1 vs. GRITLM

| Dimension | GRITLM | UniMoER (Ours) |
|-----------|--------|----------------|
| Task pair | Embedding + generation | Embedding + reranking |
| Task separation | Architectural (attention pattern) | Learned routing (MoE-LoRA) |
| Scale | 7B | 0.6B-7B (scale study) |
| Interference analysis | Claims none | Systematic quantification |
| Parameter efficiency | Full model fine-tuning | Frozen base + LoRA experts (~12% trainable) |

### 12.2 vs. Jina v3 / RouterRetriever

| Dimension | Jina v3 / RouterRetriever | UniMoER (Ours) |
|-----------|--------------------------|----------------|
| Routing | Hard (task ID / similarity) | Soft (learned per-layer) |
| Tasks | Embedding subtasks / domain retrieval | Embedding + reranking |
| Knowledge sharing | Only through frozen base | Through shared expert + router |
| Analysis | None | Deep routing + representation analysis |

### 12.3 vs. FVAE-LoRA / LoRI / TC-LoRA

| Dimension | Interference Factorization Works | UniMoER (Ours) |
|-----------|--------------------------------|----------------|
| Domain | General NLP (reasoning, coding) | Information retrieval (embedding + reranking) |
| Mechanism | Orthogonal subspaces / VAE / CP decomposition | MoE routing with task-aware experts |
| Interference analysis | Task-agnostic | IR-specific with geometric interpretation |

---

## 13. Key References

### Unified Retrieval + Reranking
1. Muennighoff, N. et al. "Generative Representational Instruction Tuning" (GRITLM). **ICLR 2025**. [arXiv:2402.09906](https://arxiv.org/abs/2402.09906)
2. Liu, Q. et al. "E2Rank." arXiv:2510.22733. ICLR 2026 withdrawn. [arXiv](https://arxiv.org/abs/2510.22733)
3. Bhat, R. et al. "UR2N." **COLING 2025** (Industry). [ACL Anthology](https://aclanthology.org/2025.coling-industry.51/)
4. FreeRet. ICLR 2026 under review. [arXiv:2509.24621](https://arxiv.org/abs/2509.24621)
5. "Autoregressive Ranking: Bridging the Gap." arXiv:2601.05588, Jan 2026. [arXiv](https://arxiv.org/abs/2601.05588)

### Task-Specific LoRA for IR
6. Sturua, S. et al. "jina-embeddings-v3." **ECIR 2025**. [arXiv:2409.10173](https://arxiv.org/abs/2409.10173)
7. Lee, H. et al. "RouterRetriever." **AAAI 2025**. [arXiv:2409.02685](https://arxiv.org/abs/2409.02685)
8. Wang, F. et al. "jina-reranker-v3." arXiv:2509.25085. [arXiv](https://arxiv.org/abs/2509.25085)
9. Guan, K. et al. "BSharedRAG." **EMNLP Findings 2024**. [arXiv:2409.20075](https://arxiv.org/abs/2409.20075)

### MoE-LoRA Architectures
10. Wu, X. et al. "Mixture of LoRA Experts" (MOLE). **ICLR 2024**. [arXiv:2404.13628](https://arxiv.org/abs/2404.13628)
11. Gao, C. et al. "MoLA." **NAACL 2025 Findings**. [ACL Anthology](https://aclanthology.org/2025.findings-naacl.284/)
12. "SMoRA." arXiv:2501.15103. [arXiv](https://arxiv.org/abs/2501.15103)
13. Zhuang, Y. et al. "LD-MoLE." arXiv:2509.25684. [arXiv](https://arxiv.org/abs/2509.25684)
14. Li, D. et al. "DynMoLE." arXiv:2504.00661. [arXiv](https://arxiv.org/abs/2504.00661)
15. "DR-LoRA." arXiv:2601.04823. [arXiv](https://arxiv.org/abs/2601.04823)
16. Wang, Z. et al. "MoLE for Continual Information Extraction." **EMNLP Findings 2025**. [ACL Anthology](https://aclanthology.org/2025.findings-emnlp.718/)

### Interference Factorization
17. Kumar, S. et al. "FVAE-LoRA: Latent Space Factorization in LoRA." **NeurIPS 2025**.
18. "LoRI: Reducing Cross-Task Interference in Multi-Task LoRA." arXiv:2504.07448. [arXiv](https://arxiv.org/abs/2504.07448)
19. "OSRM: Orthogonal Subspaces for Robust Model Merging." arXiv:2505.22934. [arXiv](https://arxiv.org/abs/2505.22934)
20. "TC-LoRA: Tensorized Clustered LoRA Merging." arXiv:2508.03999. [arXiv](https://arxiv.org/abs/2508.03999)
21. "To See a World in a Spark of Neuron." arXiv:2503.05320. [arXiv](https://arxiv.org/abs/2503.05320)

### Embedding Models and Rerankers
22. Qwen Team. "Qwen3 Embedding." arXiv:2506.05176. [Blog](https://qwenlm.github.io/blog/qwen3-embedding/)
23. Nussbaum, Z. & Duderstadt, B. "Nomic Embed V2." arXiv:2502.07972. [arXiv](https://arxiv.org/abs/2502.07972)

### Interpretability
24. Nijasure, A. et al. "How Relevance Emerges." arXiv:2504.08780. [arXiv](https://arxiv.org/abs/2504.08780)

---

## 14. Paper Writing Strategy

### Recommended Structure (EMNLP 2026, 8 pages + references)

| Section | Pages | Priority |
|---------|-------|----------|
| Introduction | 1 | Problem: interference at practical scale. Gap: no systematic study. Contribution: analysis + architecture. Explicitly address GRITLM. |
| Related Work | 1 | Three streams: unified IR, task-specific LoRA, MoE-LoRA. Fourth stream: interference factorization. Position gap. |
| Method | 1.5 | Architecture, forward pass, training. Keep concise — this is the tool, not the product. |
| Interference Study (Exp 1-3) | 2 | **Lead with this.** Scale-dependent interference, gradient conflict, kill gate results. This is the paper's anchor. |
| Routing Analysis (Exp 4-5) | 1.5 | Visualization-heavy. Layer-wise patterns, CKA, expert probing. Publication-quality figures. |
| Architecture Validation (Exp 6-8) | 1 | Soft vs. hard, transfer, rank sensitivity. Table-heavy, concise. |
| Practical Results (Exp 9-10) | 0.5 | RAG pipeline, pointwise vs. listwise. Brief validation. |
| Conclusion | 0.5 | Key findings, limitations, future work. |

**Narrative arc:** "We ask when interference emerges (it does, at practical scales). We identify where it lives (gradient conflict in specific layers). We show how to factorize it (MoE routing). We verify the factorization works (routing analysis, CKA). We demonstrate it's practical (RAG pipeline)."

---

## 15. Open Questions

1. **Listwise vs. pointwise reranking:** Which mode benefits more from expert routing? Both should be tested.
2. **Asymmetric expert rank:** Should the shared expert have higher rank than specialists to capture both tasks?
3. **Multilingual generalization:** Can the findings transfer to multilingual settings (Qwen3's multilingual capabilities)?
4. **Matryoshka representation learning:** Can MRL be integrated for flexible embedding dimensions?
5. **Router architecture:** Sparsegen (LD-MoLE) vs. Tsallis entropy (DynMoLE) — which is more stable for the two-task setting?

---

*Proposal v2 refined: February 2026*
*Incorporates independent review feedback addressing GRITLM positioning, scale-dependent interference hypothesis, gradient conflict analysis, expanded baselines, and analysis-first narrative strategy.*
*Literature coverage: 24 papers spanning unified IR, MoE-LoRA, interference factorization, and LoRA interpretability (2024-2026).*
