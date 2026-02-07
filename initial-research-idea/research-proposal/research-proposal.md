# Factorizing Task Interference in Unified Retrieval and Reranking via Mixture-of-LoRA-Expert Routing

> A Refined Research Proposal — February 2026

---

## 1. Executive Summary

Modern information retrieval (IR) systems rely on a two-stage pipeline: a fast embedding retriever (dual-encoder) followed by an accurate but slow reranker (cross-encoder). Deploying two separate models doubles memory cost, prevents cross-task knowledge transfer, and complicates deployment. Recent unified approaches (GRITLM, E2Rank) use a single parameter set for both tasks but suffer from task interference — the same parameters must simultaneously optimize for independent encoding (embedding) and joint encoding (reranking), two fundamentally different representational objectives.

This proposal investigates a central hypothesis: **task interference between embedding and reranking arises from conflicting representational demands that can be decomposed into separable subspaces, and Mixture-of-LoRA-Expert (MoE-LoRA) routing provides a principled mechanism to factorize and coordinate these subspaces while preserving beneficial knowledge transfer.**

We propose **UniMoER** (Unified MoE-LoRA for Embedding and Reranking), a parameter-efficient architecture that attaches specialized LoRA expert adapters to a frozen base model with a learned router. The architecture enables task-specific specialization (via dedicated embedding and reranking experts) while preserving knowledge sharing (via a shared expert and common backbone), achieving competitive performance on both MTEB and BEIR benchmarks with approximately 44% memory reduction compared to two separate models.

**Primary target venues:** EMNLP 2026 (main), SIGIR 2026 (full paper), or NeurIPS 2026.

---

## 2. Problem Statement and Motivation

### 2.1 The Two-Stage Retrieval Pipeline

The dominant paradigm in neural information retrieval operates in two stages:

1. **Stage 1 — Embedding Retrieval (Dual-Encoder):** Queries and documents are independently encoded into dense vectors; top-K candidates are retrieved via approximate nearest neighbor (ANN) search. Latency: ~10ms. Architecture: bi-encoder with independent encoding paths.

2. **Stage 2 — Reranking (Cross-Encoder):** Query-document pairs are jointly encoded with full cross-attention; candidates are re-scored for high-quality ordering. Latency: ~100ms per pair. Architecture: cross-encoder with joint query-document attention.

Currently, each stage uses a separate model. For example, deploying Qwen3-Embedding-0.6B alongside Qwen3-Reranker-0.6B requires 2.4 GB VRAM total, with completely independent training pipelines and no parameter sharing.

### 2.2 The Task Interference Problem

The embedding task and the reranking task impose fundamentally different demands on the representation space:

| Dimension | Embedding (Dual-Encoder) | Reranking (Cross-Encoder) |
|-----------|--------------------------|---------------------------|
| **Input structure** | Single text | Query-document pair |
| **Attention pattern** | Independent encoding | Joint cross-attention |
| **Optimization objective** | Contrastive (push apart/pull together) | Pointwise/pairwise relevance scoring |
| **Representation goal** | Global semantic similarity | Fine-grained relevance matching |
| **Output** | Dense vector (pooled) | Scalar relevance score |

When a single model must serve both tasks simultaneously (as in GRITLM or E2Rank), these conflicting objectives compete for the same parameters — a phenomenon we term **retrieval-reranking task interference**. This manifests as degraded performance on one or both tasks compared to dedicated specialist models.

### 2.3 Why This Matters

- **Production cost:** Two separate models double VRAM, doubling deployment cost on GPU-constrained infrastructure.
- **Edge/mobile IR:** Memory-limited devices (4-8 GB) cannot host two full models.
- **RAG systems:** Retrieval-augmented generation pipelines require both retrieval and reranking; a single unified model simplifies the stack.
- **Knowledge waste:** Embedding and reranking are related tasks — the model that understands "what is relevant" for retrieval already has knowledge useful for reranking, but separate training prevents this transfer.

---

## 3. Related Work and Landscape Analysis

### 3.1 Unified Retrieval + Reranking (2024-2026)

| Work | Venue | Approach | Limitation |
|------|-------|----------|------------|
| **GRITLM** (Muennighoff et al.) | ICLR 2025 | Single model with bidirectional attention for embedding, causal attention for generation; can rerank via generative capabilities | Same parameters for both tasks; task interference; 7B+ scale models |
| **E2Rank** (Liu et al., Alibaba NLP) | arXiv 2510.22733; **ICLR 2026 withdrawn** (Jan 2026) | Single embedding model extended with listwise reranking via PRF-style query construction and continued RankNet training | No explicit task specialization; compromises on both tasks; withdrawal from ICLR 2026 may indicate reviewer concerns about approach validity |
| **UR2N** (Bhat et al.) | COLING 2025 (Industry) | Unified encoder-decoder with XTR parallel layer for retrieval and decoder for reranking | Industry track; limited to encoder-decoder architecture; modest scale |
| **FreeRet** (under review) | ICLR 2026 (under review) | Training-free MLLM as unified embedder + reranker; bypasses lexical alignment layers for embedding, MCQ-based reranking | Multimodal focus; no fine-tuning; limited to MLLM architectures; no task-specific adaptation |
| **Jina Reranker v3** (Wang et al.) | arXiv 2509.25085 | 0.6B listwise reranker with "last but not late" causal attention interaction; built on Qwen3-0.6B | Dedicated reranker only, not unified with embedding |

**Key observation:** E2Rank's withdrawal from ICLR 2026 (modified Jan 23, 2026) creates an opening — the unified embedding+reranking space at top venues is less saturated than it appears. Moreover, no accepted work at ICLR 2026 addresses this specific problem with expert routing.

### 3.2 Task-Specific LoRA Adapters for Retrieval

| Work | Venue | Approach | Limitation |
|------|-------|----------|------------|
| **Jina Embeddings v3** (Sturua et al.) | ECIR 2025 | 570M model with 5 task-specific LoRA adapters (retrieval.query, retrieval.passage, separation, classification, text-matching); hard selection by task ID | No soft routing; adapters chosen by explicit task ID, not learned; no MoE dynamics |
| **BSharedRAG** (Guan et al.) | EMNLP Findings 2024 | Shared backbone + LoRA modules for retrieval and generation in e-commerce RAG | Hard task separation; retrieval + generation (not reranking); domain-specific |

**Gap:** Jina v3 demonstrates that task-specific LoRAs work for retrieval, but uses **hard** adapter selection. No work has explored **soft, learned routing** between embedding and reranking LoRA experts.

### 3.3 MoE-LoRA Architectures (2024-2026)

| Work | Venue | Key Innovation | Relevance |
|------|-------|----------------|-----------|
| **MOLE** (Wu, Huang & Wei; Tsinghua/Microsoft) | ICLR 2024 | Layer-wise gating over pre-trained LoRA experts; hierarchical weight control | Foundational MoE-LoRA architecture; applied to general multi-task composition |
| **MoLA** (Gao et al.) | NAACL 2025 Findings | Layer-wise expert allocation; more experts in higher layers | Shows layer-specific expert needs; applicable to our per-layer routing design |
| **SMoRA** (Jan 2025) | arXiv 2501.15103 | Each LoRA rank treated as independent expert; rank-wise MoE | Finer-grained expert granularity; potential enhancement for our approach |
| **DR-LoRA** (Jan 2026) | arXiv 2601.04823 | Dynamic rank growth based on expert saliency scoring in MoE LLMs | Adaptive capacity allocation; relevant for our expert capacity design |
| **LD-MoLE** (Zhuang et al., Sep 2025) | arXiv 2509.25684 | Learnable dynamic routing replacing TopK with differentiable Sparsegen; adaptive token/layer-wise expert activation | Most sophisticated routing; directly applicable to our router design |
| **DynMoLE** (Li et al., Apr 2025) | arXiv 2504.00661 | Hybrid routing via Tsallis entropy; dynamic expert selection based on routing confidence | Entropy-based routing strategy; useful for routing stability |
| **TT-LoRA MoE** (SC 2025) | SC 2025 | Tensorized LoRA experts with frozen two-stage training; sparse MoE routing | Two-stage training paradigm; 0.03% of AdapterFusion parameters |

**Gap:** All MoE-LoRA works target general multi-task or domain composition. **None apply MoE-LoRA routing to the specific embedding-reranking task pair,** where the tasks have well-defined, structurally different input-output characteristics.

### 3.4 MoE in Embedding Models

| Work | Details |
|------|---------|
| **Nomic Embed Text V2** (Feb 2025) | First MoE embedding model; 475M total / 305M active parameters; 8 experts with top-2 routing; outperforms dense models of same size |

**Note:** Nomic uses MoE in the **base model architecture** for efficiency, not for task routing. Our approach uses MoE at the **adapter level** for task specialization — a fundamentally different design point.

### 3.5 Interpretability of LoRA in Reranking

| Work | Venue | Key Finding |
|------|-------|-------------|
| **"How Relevance Emerges"** (Nijasure et al., Apr 2025) | arXiv 2504.08780 | LoRA rank 1 is sufficient for reranking; layers 5-15 contribute most; MLP up/gate projections more impactful than down projection; MLP-only LoRA recovers 98% of full performance |

This interpretability work provides methodological tools we can adopt for our expert specialization analysis.

### 3.6 Summary of Research Gap

```
                     Existing Work
                     ─────────────

Unified Retrieval    ✅ GRITLM, E2Rank, UR2N, FreeRet
+ Reranking             (single parameter set, task interference)

Task-Specific LoRA   ✅ Jina v3, BSharedRAG
for IR                  (hard adapter selection, no MoE routing)

MoE-LoRA with        ✅ MOLE, MoLA, SMoRA, DR-LoRA, LD-MoLE
Learned Routing         (general multi-task, not IR-specific)

MoE-LoRA with         ❌ NO EXISTING WORK
Learned Routing
for Unified IR         This is our contribution.
(Embedding +
Reranking)
```

---

## 4. Research Hypotheses

### H1 (Task Interference)
Unified embedding + reranking models trained with a single parameter set exhibit measurable task interference: performance on each task degrades compared to dedicated specialist models, and this degradation increases with model scale mismatch.

### H2 (Expert Factorization)
MoE-LoRA routing can factorize the representation space into separable task-specific subspaces, mitigating interference while preserving beneficial knowledge transfer through a shared expert.

### H3 (Soft Routing Advantage)
Learned soft routing over LoRA experts outperforms hard task-selection (Jina v3-style) because: (a) some inputs benefit from mixed expertise (e.g., short queries that resemble both embedding and reranking inputs), and (b) the shared expert captures task-invariant knowledge that hard selection cannot leverage.

### H4 (Knowledge Transfer)
The shared expert in MoE-LoRA enables positive transfer: training on embedding data improves reranking performance (and vice versa) compared to isolated task-specific LoRA training.

---

## 5. Proposed Method: UniMoER

### 5.1 Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                    UniMoER: Unified MoE-LoRA for E+R                  │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│              ┌─────────────────────────────────┐                     │
│              │   Base Model (Frozen)            │                     │
│              │   e.g., Qwen3-0.6B              │                     │
│              │   (~1.2 GB, all params frozen)   │                     │
│              └────────────────┬────────────────┘                     │
│                               │                                      │
│              ┌────────────────┴────────────────┐                     │
│              │     Per-Layer Learned Router     │                     │
│              │     (~5 MB total trainable)      │                     │
│              └────────────────┬────────────────┘                     │
│                               │                                      │
│          ┌────────────────────┼────────────────────┐                 │
│          ▼                    ▼                    ▼                  │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐           │
│   │  Embedding  │     │  Reranking  │     │   Shared    │           │
│   │   Expert    │     │   Expert    │     │   Expert    │           │
│   │ (LoRA, r=32)│     │ (LoRA, r=32)│     │ (LoRA, r=32)│           │
│   │   ~50 MB    │     │   ~50 MB    │     │   ~50 MB    │           │
│   └─────────────┘     └─────────────┘     └─────────────┘           │
│                                                                       │
│   Task Heads:                                                         │
│   ├── Embedding: last-token pooling + L2 normalization               │
│   └── Reranking: [CLS]-token → linear → sigmoid                     │
│                                                                       │
│   Total: ~1.35 GB (vs 2.4 GB for two separate models)               │
│   Trainable: ~155 MB (~12% of total)                                 │
└──────────────────────────────────────────────────────────────────────┘
```

### 5.2 Components

**Frozen Base Model.** A pre-trained language model (e.g., Qwen3-0.6B-Base) provides general language understanding. All base parameters remain frozen throughout training.

**LoRA Expert Adapters.** Three low-rank adapters, each consisting of down-projection A_i (d -> r) and up-projection B_i (r -> d) matrices at each targeted layer:
- Expert 0: Embedding specialist
- Expert 1: Reranking specialist
- Expert 2: Shared/general knowledge

Standard LoRA initialization: A initialized with Kaiming uniform, B initialized to zero (ensuring the model starts from the base model behavior).

**Per-Layer Learned Router.** A lightweight network that produces soft routing weights over experts, conditioned on the input representation at each layer. This follows the LD-MoLE design for differentiable routing, enabling layer-wise and input-dependent expert allocation:

$$w^{(l)} = \text{Sparsegen}(\text{MLP}^{(l)}(h^{(l)}_{\text{pool}}) / \tau)$$

where h_pool is the pooled hidden state (first token or mean pooling), l indexes the layer, and Sparsegen provides a differentiable sparse alternative to TopK.

**Task-Specific Heads.** Lightweight output heads for each mode:
- Embedding mode: Pool the last non-padding token embedding, L2-normalize.
- Reranking mode: Pool the first token ([CLS]) embedding, linear projection to scalar, sigmoid activation.

### 5.3 Forward Pass

For input hidden states h at layer l from the base model:

$$h'^{(l)} = h^{(l)} + \sum_{i=1}^{N} w_i^{(l)} \cdot B_i^{(l)} A_i^{(l)} h^{(l)}$$

where w_i^(l) are the routing weights from the per-layer router (sum to 1, with sparsity).

**Embedding mode** (single text):
1. Tokenize text.
2. Forward through base model + MoE-LoRA layers.
3. Router naturally emphasizes embedding expert + shared expert.
4. Pool last-token hidden state, L2-normalize.
5. Output: dense vector for similarity search.

**Reranking mode** (query + document pair):
1. Concatenate query and document with separator tokens.
2. Forward through base model + MoE-LoRA layers.
3. Router naturally emphasizes reranking expert + shared expert.
4. Pool first-token hidden state, apply reranking head.
5. Output: scalar relevance score.

### 5.4 Training Objective

Joint multi-task training with three loss components:

**Embedding Loss (InfoNCE):**
$$\mathcal{L}_{\text{emb}} = -\log \frac{\exp(\text{sim}(q, p^+) / \tau_e)}{\exp(\text{sim}(q, p^+) / \tau_e) + \sum_{p^-} \exp(\text{sim}(q, p^-) / \tau_e)}$$

**Reranking Loss (Binary Cross-Entropy):**
$$\mathcal{L}_{\text{rank}} = -[y \log(\sigma(s)) + (1-y) \log(1 - \sigma(s))]$$

**Router Auxiliary Loss (Load Balancing):**
$$\mathcal{L}_{\text{aux}} = N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

where f_i is the fraction of tokens routed to expert i, and P_i is the average routing probability for expert i.

**Combined:**
$$\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{emb}} + \beta \cdot \mathcal{L}_{\text{rank}} + \lambda \cdot \mathcal{L}_{\text{aux}}$$

with alpha, beta, lambda as tunable hyperparameters (default: alpha=1, beta=1, lambda=0.01).

**Training strategy:**
- Mixed-task batches with balanced sampling (50% embedding, 50% reranking).
- Gradual unfreezing: first train router + experts for N steps, then optionally allow limited base model fine-tuning.
- In-batch negatives for contrastive loss (following standard embedding training practice).

### 5.5 Router Design Variants (for Ablation)

| Variant | Description | Pros | Cons |
|---------|-------------|------|------|
| **Task-Explicit** | Hard-coded weights based on task indicator | Simple, no collapse | No learned flexibility |
| **Sequence-Level Learned** | Single routing decision per input sequence | Moderate complexity | Coarse granularity |
| **Per-Layer Learned (default)** | Independent routing per layer, differentiable | Layer-specific specialization | More parameters |
| **Token-Level** | Different routing per token per layer | Fine-grained control | High overhead |

---

## 6. Experimental Design

### 6.1 Research Track A: Hypothesis-Driven Study (Scientific Contribution)

This track focuses on understanding and quantifying task interference, validating that MoE-LoRA routing provides a principled solution.

#### Experiment 1: Task Interference Quantification

**Goal:** Establish that task interference exists and is measurable.

| Configuration | Description | What It Tests |
|---------------|-------------|---------------|
| **Emb-Only LoRA** | Single LoRA trained only on embedding data | Upper bound for embedding performance |
| **Rank-Only LoRA** | Single LoRA trained only on reranking data | Upper bound for reranking performance |
| **Joint-Single LoRA** | Single LoRA trained on both tasks jointly | Baseline: interference without expert separation |
| **Joint-MoE-LoRA (Ours)** | MoE-LoRA with 3 experts trained jointly | Proposed: interference mitigation via routing |

**Metric:** Performance drop ratio = (Specialist - Unified) / Specialist, measured on MTEB and BEIR.

**Expected finding:** Joint-Single LoRA shows measurable degradation on both tasks; Joint-MoE-LoRA recovers most or all of the specialist performance.

#### Experiment 2: Soft Routing vs Hard Selection

**Goal:** Demonstrate that learned soft routing outperforms Jina v3-style hard adapter selection.

| Configuration | Description |
|---------------|-------------|
| **Hard-Switch** | Two dedicated LoRAs, selected by explicit task indicator (Jina v3 style) |
| **Hard-Switch + Shared** | Two dedicated LoRAs + shared LoRA, selected by task indicator |
| **Soft-Router (Ours)** | Three experts with learned per-layer routing |

**Expected finding:** Soft routing enables mixed expertise for ambiguous inputs and leverages the shared expert more effectively.

#### Experiment 3: Knowledge Transfer Analysis

**Goal:** Demonstrate that the shared expert enables positive cross-task transfer.

| Configuration | Description |
|---------------|-------------|
| **Freeze Rerank Expert** | Train only embedding + shared experts; evaluate reranking |
| **Freeze Embed Expert** | Train only reranking + shared experts; evaluate embedding |
| **Full Training** | Train all experts jointly |
| **No Shared Expert** | Only embedding + reranking experts (no shared) |

**Expected finding:** Freezing one task-specific expert while training the shared expert still yields reasonable performance on the frozen task, demonstrating knowledge transfer through the shared expert.

#### Experiment 4: Routing Behavior Analysis

**Goal:** Deep interpretability of routing decisions.

- **Per-task routing distributions:** How do routing weights differ for embedding vs reranking inputs?
- **Per-layer routing patterns:** Do different layers prefer different experts? (Connect to MoLA's finding that higher layers need more specialization.)
- **Training dynamics:** How do routing patterns evolve during training?
- **Dataset-dependent routing:** Do different BEIR domains (medical, legal, scientific) trigger different routing?
- **Expert probing:** What semantic/linguistic properties does each expert capture? (Use probing classifiers following methodology from "How Relevance Emerges".)

#### Experiment 5: Representation Space Analysis

**Goal:** Provide evidence that experts factorize the representation space.

- **CKA (Centered Kernel Alignment):** Measure similarity between representations produced by each expert across tasks.
- **t-SNE/UMAP visualization:** Visualize how embedding and reranking representations cluster with vs without MoE routing.
- **Singular value analysis:** Compare singular value spectra of expert weight matrices to understand what each expert captures.

### 6.2 Research Track B: Practical Systems Contribution

This track focuses on deployment efficiency and real-world applicability.

#### Experiment 6: End-to-End RAG Pipeline

**Goal:** Demonstrate practical value in a realistic retrieval-augmented generation scenario.

- **Setup:** Full retrieve -> rerank -> generate pipeline on:
  - Natural Questions (NQ)
  - TriviaQA
  - HotpotQA
- **Metrics:** Answer accuracy (EM, F1), end-to-end latency, peak GPU memory.
- **Baselines:** Two separate models, E2Rank, GRITLM.
- **Hardware targets:** Single T4 (16GB), single A100 (40GB), CPU-only inference.

#### Experiment 7: Memory-Constrained Deployment

**Goal:** Show UniMoER enables retrieval+reranking on hardware where two models cannot fit.

- **Scenario:** 4GB VRAM budget (edge device simulation).
- **Quantization:** INT8 and INT4 quantization of base model + LoRA experts.
- **Baseline:** Compare against quantized two-model setup (which may not fit in 4GB).

#### Experiment 8: Multi-Task Extensibility

**Goal:** Demonstrate that new task experts can be added without degrading existing performance.

- **Add Expert 3:** Classification expert (trained on NLI data).
- **Add Expert 4:** Question answering expert (trained on SQuAD-style data).
- **Evaluate:** Original embedding + reranking performance after adding new experts.
- **Expected finding:** New experts can be added with minimal (<0.5%) degradation on existing tasks.

### 6.3 Benchmarks and Metrics

| Benchmark | Purpose | Metrics |
|-----------|---------|---------|
| **MTEB** (Massive Text Embedding Benchmark) | Embedding quality across 7 task categories | Average score, per-category scores |
| **BEIR** (Benchmarking IR) | Reranking quality across 18 diverse domains | nDCG@10, MAP |
| **MS MARCO** | Passage retrieval and reranking | MRR@10, nDCG@10 |
| **BRIGHT** | Reasoning-intensive retrieval | nDCG@10 |
| **End-to-end RAG** (NQ, TriviaQA) | Unified pipeline quality | Answer EM/F1, latency, memory |

### 6.4 Baselines

| Baseline | Type | Why It's Important |
|----------|------|-------------------|
| Qwen3-Embedding-0.6B + Qwen3-Reranker-0.6B | Two separate models | Performance upper bound, memory lower bound |
| GRITLM-7B | Unified embedding + generation | Established unified baseline (ICLR 2025) |
| E2Rank (replicated) | Unified embedding + listwise reranking | Direct competitor; withdrawn from ICLR 2026 |
| Jina v3-style hard LoRA selection | Shared backbone + hard task LoRA | Ablation: soft routing vs hard selection |
| Single LoRA (no MoE) | Shared backbone + single joint LoRA | Ablation: quantify MoE routing benefit |
| Jina Reranker v3 | Dedicated listwise reranker | Reranking-specific baseline |

### 6.5 Base Models for Experiments

| Model | Parameters | Purpose |
|-------|-----------|---------|
| Qwen3-0.6B-Base | 0.6B | Primary experiments (resource-efficient) |
| Qwen3-4B-Base | 4B | Scale sensitivity analysis |
| Mistral-7B-v0.3 | 7B | Cross-architecture validation; GRITLM comparison |

---

## 7. Expected Contributions

### 7.1 Scientific Contributions

1. **First systematic study of task interference in unified embedding + reranking models.** We provide quantitative evidence that embedding and reranking occupy separable representational subspaces and that interference is measurable and mitigatable.

2. **MoE-LoRA expert routing as a principled factorization mechanism for IR task interference.** We demonstrate that learned routing over task-specific LoRA experts decomposes conflicting representational demands while preserving positive transfer.

3. **Deep routing and expert interpretability analysis.** We provide layer-wise, dataset-wise, and training-stage-wise analysis of routing behavior, connecting to and extending findings from MoLA (layer-wise allocation) and "How Relevance Emerges" (LoRA interpretability in reranking).

4. **Evidence for soft routing advantage over hard task selection.** We show that learned routing outperforms Jina v3-style hard adapter selection, particularly for inputs that benefit from mixed expertise.

### 7.2 Practical Contributions

5. **~44% memory reduction** for unified retrieval + reranking with competitive quality, validated in realistic deployment scenarios including edge hardware.

6. **End-to-end RAG pipeline benchmark** demonstrating practical latency and quality trade-offs on constrained hardware.

7. **Extensibility framework** showing that new task experts (classification, QA) can be added modularly without degrading existing capabilities.

---

## 8. Risk Analysis and Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Performance gap between MoE-LoRA and task-specific LoRAs is <1%** | High | If gains are marginal, pivot to interpretability: the study of task interference itself is valuable even if the architectural fix is modest. Frame as "understanding the problem" rather than "solving it." |
| **Soft routing does not meaningfully outperform hard selection** | High | Include comprehensive ablation; if confirmed, report as a negative finding (informative) and focus on other contributions (shared expert analysis, extensibility). |
| **E2Rank achieves strong results with a simpler approach** | Medium | Our method provides different advantages: explicit expert specialization, multi-task extensibility, interpretable routing. Position as complementary, not competing. |
| **Router collapse (all inputs routed to same expert)** | Medium | Standard mitigations: load-balancing auxiliary loss, expert dropout during training, noise injection to routing logits (following MoE best practices). |
| **Negative transfer between tasks** | Medium | Careful initialization (LoRA B to zero), gradual unfreezing, and monitoring per-task metrics during training. If negative transfer detected, adjust task sampling ratio. |
| **Computational cost of experiments** | Low-Medium | Start with 0.6B scale; use efficient training (mixed precision, gradient checkpointing). Full MTEB/BEIR evaluation is computationally manageable. |
| **Close competitor published before our submission** | Medium | The E2Rank withdrawal from ICLR 2026 reduces immediate competition. Our MoE-LoRA approach is fundamentally different from all existing unified methods. Monitor arXiv for new entries. |

---

## 9. Venue Strategy and Timeline

### 9.1 Target Venues (Ranked by Fit)

| Venue | Conference Dates | Key Deadline | Angle | Fit |
|-------|-----------------|--------------|-------|-----|
| **EMNLP 2026** | Oct 24-29, Budapest, Hungary | ARR submission ~Jun 2026 (est.) | NLP/IR + hypothesis-driven study + practical deployment | Best fit: values IR contributions + analysis depth |
| **SIGIR 2026** | Jul 20-24, Melbourne, Australia | Abstract ~Jan 23, 2026; Paper ~Jan 30, 2026 (**passed**; based on SIGIR 2025 pattern and ACM deadline calendar) | IR-focused + practical systems story | Missed for full papers; check Resource/Reproducibility/Short Paper tracks which may have later deadlines |
| **NeurIPS 2026** | Dec 6-12, Sydney, Australia | ~May 2026 (est.) | MoE + representation learning + broader ML insights | Good fit if framed as representation factorization study |
| **ACL 2026** | Jul 2-7, San Diego, USA | ARR commitment Mar 14, 2026 (Jan cycle) | NLP + hypothesis-driven study | Tight timeline; requires Jan 5, 2026 ARR submission (**likely passed**) |
| **CIKM 2026** | ~Nov 2026 (est.) | ~May 2026 (est.) | Applied IR + systems contribution | Backup venue; lower bar but still respected |

### 9.2 Recommended Strategy

Given that SIGIR 2026 and ACL 2026 deadlines have likely passed:

1. **Primary target: EMNLP 2026** (ARR submission ~June 2026). This gives approximately 4 months for implementation and experiments. EMNLP values IR contributions, hypothesis-driven studies, and analysis depth.

2. **Parallel target: NeurIPS 2026** (submission ~May 2026). Frame as a representation learning / MoE study with IR as the application domain.

3. **Backup: CIKM 2026** or EMNLP 2026 Findings.

### 9.3 Research Timeline

| Phase | Period | Activities |
|-------|--------|------------|
| **Phase 1: Foundation** | Feb-Mar 2026 | Implement UniMoER architecture; set up training pipeline with Qwen3-0.6B; reproduce baselines (Jina v3-style, single LoRA, E2Rank); initial small-scale experiments |
| **Phase 2: Core Experiments** | Mar-Apr 2026 | Run Experiments 1-5 (task interference, soft vs hard routing, knowledge transfer, routing analysis, representation analysis); iterate on architecture based on findings |
| **Phase 3: Practical Validation** | Apr-May 2026 | Run Experiments 6-8 (RAG pipeline, memory-constrained deployment, extensibility); full-scale MTEB + BEIR evaluation |
| **Phase 4: Analysis & Writing** | May 2026 | Deep routing visualization; write paper; internal review cycle |
| **Phase 5: Submission** | Jun 2026 | Submit to EMNLP 2026 via ARR; prepare rebuttal materials |

---

## 10. Training Data Sources

| Dataset | Task | Size | Usage |
|---------|------|------|-------|
| **MS MARCO Passage** | Embedding + Reranking | ~8.8M passages, 500K queries | Primary training data |
| **NQ (Natural Questions)** | Embedding | ~100K Q-A pairs | Additional embedding training |
| **MTEB Training Subsets** | Embedding (various) | Varies | Task-diverse embedding training |
| **BEIR Training Splits** | Reranking | Varies by domain | Domain-diverse reranking training |
| **NLI (SNLI + MultiNLI)** | Classification (extensibility) | ~1M pairs | For Experiment 8 (add classification expert) |

Training follows standard practices: hard negative mining for contrastive learning, in-batch negatives, and balanced task sampling.

---

## 11. Implementation Notes

### 11.1 Framework and Libraries

- **Base framework:** PyTorch + HuggingFace Transformers
- **PEFT:** HuggingFace PEFT library for LoRA, with custom MoE routing layer
- **Router:** Custom implementation following LD-MoLE's differentiable Sparsegen formulation
- **Evaluation:** MTEB library for embedding evaluation; pytrec_eval for BEIR/MS MARCO
- **Experiment tracking:** Weights & Biases

### 11.2 Minimum Hardware Requirements

| Experiment Scope | Hardware | Notes |
|-----------------|----------|-------|
| 0.6B model, basic experiments | 1x RTX 3090/4090 (24GB) | Sufficient for core hypothesis testing |
| 4B model, full MTEB/BEIR | 1x A100 (40GB) | Recommended for publication-quality results |
| 7B model, multi-scale analysis | 2x A100 (80GB) | For NeurIPS-level comprehensive study |

---

## 12. Differentiation from Closest Related Work

### 12.1 vs E2Rank

| Dimension | E2Rank | UniMoER (Ours) |
|-----------|--------|----------------|
| Architecture | Single embedding model, no adapters | Frozen base + MoE-LoRA experts |
| Task separation | None (single parameter set) | Explicit via task-specific experts |
| Reranking mechanism | PRF-style query construction | Cross-encoder with reranking head |
| Interpretability | Limited | Rich (routing analysis, expert probing) |
| Extensibility | Low (retraining needed) | High (add new LoRA experts) |
| ICLR 2026 status | Withdrawn (Jan 2026) | N/A |

### 12.2 vs Jina Embeddings v3

| Dimension | Jina v3 | UniMoER (Ours) |
|-----------|---------|----------------|
| Adapter selection | Hard (by task ID) | Soft (learned per-layer routing) |
| Number of adapters | 5 (fixed) | 3 (extensible) |
| Routing mechanism | None (task indicator) | Learned MoE router |
| Knowledge sharing | Only through frozen base | Through shared expert + router |
| Reranking | Via "separation" adapter | Via dedicated reranking expert |

### 12.3 vs MOLE / MoLA / LD-MoLE

| Dimension | General MoE-LoRA | UniMoER (Ours) |
|-----------|-----------------|----------------|
| Application domain | General multi-task | Specifically IR (embedding + reranking) |
| Expert design | Task-agnostic | Task-aware (embedding, reranking, shared) |
| Training objective | Varies | Joint contrastive + ranking + auxiliary |
| Analysis focus | General performance | Task interference factorization in IR |

---

## 13. Key References

### Unified Retrieval + Reranking

1. Muennighoff, N. et al. "Generative Representational Instruction Tuning" (GRITLM). **ICLR 2025**. [arXiv:2402.09906](https://arxiv.org/abs/2402.09906)
2. Liu, Q. et al. "E2Rank: Your Text Embedding can Also be an Effective and Efficient Listwise Reranker." **arXiv:2510.22733**, Oct 2025. ICLR 2026 withdrawn. [arXiv](https://arxiv.org/abs/2510.22733)
3. Bhat, R. et al. "UR2N: Unified Retriever and ReraNker." **COLING 2025** (Industry). [ACL Anthology](https://aclanthology.org/2025.coling-industry.51/)
4. FreeRet: "MLLMs as Training-Free Retrievers." **ICLR 2026 under review**. [arXiv:2509.24621](https://arxiv.org/abs/2509.24621)

### Task-Specific LoRA for IR

5. Sturua, S. et al. "jina-embeddings-v3: Multilingual Embeddings With Task LoRA." **ECIR 2025**. [arXiv:2409.10173](https://arxiv.org/abs/2409.10173)
6. Wang, F. et al. "jina-reranker-v3: Last but Not Late Interaction for Listwise Document Reranking." arXiv:2509.25085, Sep 2025. [arXiv](https://arxiv.org/abs/2509.25085)
7. Guan, K. et al. "BSharedRAG: Backbone Shared Retrieval-Augmented Generation." **EMNLP Findings 2024**. [arXiv:2409.20075](https://arxiv.org/abs/2409.20075)

### MoE-LoRA Architectures

8. Wu, X., Huang, S. & Wei, F. "Mixture of LoRA Experts" (MOLE). **ICLR 2024**. [arXiv:2404.13628](https://arxiv.org/abs/2404.13628)
9. Gao, C. et al. "MoLA: MoE LoRA with Layer-wise Expert Allocation." **NAACL 2025 Findings**. [ACL Anthology](https://aclanthology.org/2025.findings-naacl.284/)
10. "SMoRA: Each Rank Could be an Expert." arXiv:2501.15103, Jan 2025. [arXiv](https://arxiv.org/abs/2501.15103)
11. "DR-LoRA: Dynamic Rank LoRA for MoE Adaptation." arXiv:2601.04823, Jan 2026. [arXiv](https://arxiv.org/abs/2601.04823)
12. Zhuang, Y. et al. "LD-MoLE: Learnable Dynamic Routing for Mixture of LoRA Experts." arXiv:2509.25684, Sep 2025. [arXiv](https://arxiv.org/abs/2509.25684)
13. Li, D. et al. "DynMoLE: Boosting Mixture of LoRA Experts with a Hybrid Routing Mechanism." arXiv:2504.00661, Apr 2025. [arXiv](https://arxiv.org/abs/2504.00661)

### Embedding Models and Rerankers

14. Qwen Team. "Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models." arXiv:2506.05176, Jun 2025. [Blog](https://qwenlm.github.io/blog/qwen3-embedding/)
15. Nussbaum, Z. & Duderstadt, B. "Training Sparse Mixture Of Experts Text Embedding Models" (Nomic Embed V2). arXiv:2502.07972, Feb 2025. [arXiv](https://arxiv.org/abs/2502.07972)

### Interpretability

16. Nijasure, A. et al. "How Relevance Emerges: Interpreting LoRA Fine-Tuning in Reranking LLMs." arXiv:2504.08780, Apr 2025. [arXiv](https://arxiv.org/abs/2504.08780)

---

## 14. Assessment of Publication Potential

### 14.1 Strengths for Top Venues

- **Clear research gap:** No existing work combines MoE-LoRA routing with the specific embedding + reranking task pair.
- **Timely:** E2Rank's ICLR 2026 withdrawal opens the unified IR space; MoE-LoRA is a hot topic (MOLE at ICLR 2024, MoLA at NAACL 2025, LD-MoLE/DynMoLE in 2025).
- **Dual contribution:** Both scientific (task interference study) and practical (memory-efficient unified IR).
- **Rich analysis potential:** Routing visualization, expert probing, knowledge transfer quantification.
- **Practical relevance:** Memory-efficient unified retrieval+reranking is a real production need for RAG systems.

### 14.2 Risks for Top Venues

- **Novelty perception:** Reviewers may see this as "applying MoE-LoRA to yet another task pair" — the hypothesis-driven framing is essential to counter this.
- **Marginal gains:** If performance improvements over simpler baselines (single LoRA, hard selection) are small (<1%), the paper must lean heavily on analysis and interpretability.
- **Strong baselines:** E2Rank (if replicated) and Jina v3 are competitive baselines that must be matched or beaten.

### 14.3 Realistic Venue Assessment

| Venue | Likelihood | Key Requirement |
|-------|-----------|-----------------|
| **EMNLP 2026 Main** | Moderate-High | Strong results + deep analysis + hypothesis validation |
| **EMNLP 2026 Findings** | High | Competitive results + good analysis |
| **NeurIPS 2026** | Moderate | Frame as representation learning; broader ML insights |
| **SIGIR 2026** | High (if deadline permits) | IR focus + practical deployment story |
| **CIKM 2026** | High | Systems-focused contribution |

---

## 15. Open Questions for Further Investigation

1. **Should the reranking mode use cross-encoder style (concatenated query+document) or listwise style (like E2Rank/Jina Reranker v3)?** Listwise reranking has shown strong results in recent work; investigating both modes would strengthen the paper.

2. **Should we incorporate ideas from SMoRA (rank-wise experts) or DR-LoRA (dynamic rank growth)?** These could provide additional technical depth but increase implementation complexity.

3. **What is the minimum LoRA rank needed?** "How Relevance Emerges" shows rank 1 is sufficient for reranking. Is the same true for embedding? This could reduce memory further.

4. **How does this scale to multilingual settings?** Jina v3 and Qwen3-Embedding excel in multilingual tasks. Demonstrating multilingual capability would significantly strengthen the paper.

5. **Can we leverage Matryoshka Representation Learning (MRL)?** Both Jina v3 and Nomic V2 support flexible embedding dimensions. Integrating MRL would add practical value.

---

*Proposal refined: February 2026*
*Based on comprehensive literature review covering 16+ papers and analysis of 2025-2026 landscape*
