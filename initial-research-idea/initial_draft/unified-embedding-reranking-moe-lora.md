# Unified Embedding + Reranking with MoE-LoRA

> A Novel Research Direction for Top Conference Publication

**Status**: Highly novel, low competition, directly applicable
**Target Venues**: ACL, EMNLP, NAACL (NLP); SIGIR, CIKM (IR); ICLR, NeurIPS (ML)

---

## Table of Contents

1. [Problem Overview](#part-1-problem-overview)
2. [Embedding Models Explained](#part-2-embedding-models-explained)
3. [Reranking Models Explained](#part-3-reranking-models-explained)
4. [The Problem - Two Separate Models](#part-4-the-problem---two-separate-models)
5. [Current Unified Approaches](#part-5-current-unified-approaches)
6. [The Novel Idea - MoE-LoRA](#part-6-the-novel-idea--moe-lora-for-unified-retrieval--reranking)
7. [How Would It Work](#part-7-how-would-it-work)
8. [Why MoE-LoRA is Better](#part-8-why-moe-lora-is-better-than-alternatives)
9. [Technical Implementation](#part-9-technical-implementation-details)
10. [What Makes This Publishable](#part-10-what-makes-this-publishable)
11. [Challenges & Solutions](#part-11-potential-challenges--solutions)
12. [Code Skeleton](#part-12-code-skeleton-to-get-started)
13. [References](#references)

---

## Part 1: Problem Overview

In modern **information retrieval** (search engines, RAG systems, document QA), there's a standard **two-stage pipeline**:

```
User Query: "How does photosynthesis work?"
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1: RETRIEVAL (Embedding Model)                       │
│  ─────────────────────────────────────────────────          │
│  • Convert query → vector (embedding)                       │
│  • Compare with millions of document vectors                │
│  • Return top-100 candidates (fast, ~10ms)                  │
│  • Architecture: Dual-encoder (separate query/doc encoders) │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼ Top 100 documents
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2: RERANKING (Reranker Model)                        │
│  ─────────────────────────────────────────────────          │
│  • Take (query, document) pairs                             │
│  • Score each pair for relevance                            │
│  • Return top-10 reranked results (slower, ~100ms)          │
│  • Architecture: Cross-encoder (query+doc together)         │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼ Top 10 documents (high quality)
```

**The key question**: Can we unify these two stages into a single efficient model?

---

## Part 2: Embedding Models Explained

### What it does

Converts text into a dense vector (embedding) that captures semantic meaning.

### Example Code

```python
# Example with Qwen3-Embedding
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

# Convert texts to vectors
query_embedding = model.encode("How does photosynthesis work?")
# Returns: [0.023, -0.156, 0.089, ...] (1024 dimensions)

doc_embedding = model.encode("Photosynthesis is the process by which plants...")
# Returns: [0.019, -0.148, 0.092, ...]

# Similarity = cosine similarity between vectors
similarity = cosine_similarity(query_embedding, doc_embedding)
# Returns: 0.87 (high = relevant)
```

### Architecture (Dual-Encoder)

```
Query: "How does photosynthesis work?"     Document: "Plants convert sunlight..."
              │                                        │
              ▼                                        ▼
       ┌─────────────┐                         ┌─────────────┐
       │   Encoder   │                         │   Encoder   │
       │  (shared)   │                         │  (shared)   │
       └──────┬──────┘                         └──────┬──────┘
              │                                        │
              ▼                                        ▼
        Query Vector                            Document Vector
        [0.02, -0.15, ...]                     [0.01, -0.14, ...]
              │                                        │
              └────────────────┬───────────────────────┘
                               │
                               ▼
                      Cosine Similarity = 0.87
```

### Why it's fast

Query and documents are encoded **independently**. You can pre-compute all document embeddings offline, then just compare vectors at query time.

---

## Part 3: Reranking Models Explained

### What it does

Takes a (query, document) pair and outputs a **relevance score**.

### Example Code

```python
# Example with Qwen3-Reranker
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("Qwen/Qwen3-Reranker-0.6B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B")

# Score a query-document pair
query = "How does photosynthesis work?"
document = "Photosynthesis is the process by which green plants convert sunlight..."

inputs = tokenizer(query, document, return_tensors="pt")
score = model(**inputs).logits[0]  # Returns: 0.92 (relevance score)
```

### Architecture (Cross-Encoder)

```
Query + Document (concatenated)
"[CLS] How does photosynthesis work? [SEP] Plants convert sunlight... [SEP]"
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Full Transformer  │
                    │   (Cross-Attention) │
                    │   Query ↔ Document  │
                    └──────────┬──────────┘
                               │
                               ▼
                    Relevance Score: 0.92
```

### Why it's more accurate

The model sees query and document **together**, allowing deep cross-attention between them. Can understand nuanced relationships.

### Why it's slower

Must run the full model for **every** (query, document) pair. Can't pre-compute.

---

## Part 4: The Problem - Two Separate Models

Currently, systems deploy **two completely separate models**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    CURRENT APPROACH                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Embedding Model              Reranking Model                   │
│   ┌─────────────────┐         ┌─────────────────┐               │
│   │ Qwen3-Embedding │         │ Qwen3-Reranker  │               │
│   │     (1.2 GB)    │         │    (1.2 GB)     │               │
│   └─────────────────┘         └─────────────────┘               │
│                                                                  │
│   Total VRAM: 2.4 GB                                            │
│   Separate training pipelines                                    │
│   No knowledge sharing                                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Problems with Current Approach

| Problem | Description |
|---------|-------------|
| **Memory inefficient** | Two full models loaded in VRAM |
| **No knowledge sharing** | Embedding learns "semantic similarity", reranker learns "relevance" — but these are related! |
| **Separate training** | Can't benefit from joint optimization |
| **Deployment complexity** | Two inference pipelines to maintain |

---

## Part 5: Current Unified Approaches

Some researchers have tried to unify these:

### Approach A: GRITLM (ICLR 2025)

```
Single model that can do both:
- Embedding mode: Use [EOS] token embedding
- Generation mode: Standard autoregressive generation (can be used for reranking)

Problem: Same parameters for both tasks → task interference
```

**Reference**: [GRITLM Paper](https://openreview.net/forum?id=70cf215430492f7d34830a24e744b3f1)

### Approach B: E²RANK (2025)

```
Single embedding model that can also rerank:
- Embedding: Standard encode
- Reranking: Reinterpret ranking as "pseudo-relevance feedback" query

Problem: Compromises on both tasks, no specialization
```

**Reference**: [E²RANK Paper](https://openreview.net/forum?id=5Iwj0WW1vT)

### Key Insight

These approaches use a **single set of parameters** for both tasks, leading to **task interference**. What if we could have **specialized components** while still sharing a base model?

---

## Part 6: The Novel Idea — MoE-LoRA for Unified Retrieval + Reranking

### Core Insight

Use a **shared base model** with **specialized LoRA experts** for each task, dynamically routed.

### Proposed Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                    PROPOSED: MoE-LoRA UNIFIED MODEL                   │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│                    ┌─────────────────────────┐                       │
│                    │   Qwen3-0.6B-Base       │                       │
│                    │   (Frozen, 1.2 GB)      │                       │
│                    └───────────┬─────────────┘                       │
│                                │                                      │
│                    ┌───────────┴───────────┐                         │
│                    │    Learned Router      │                        │
│                    │    (~5 MB trainable)   │                        │
│                    └───────────┬───────────┘                         │
│                                │                                      │
│           ┌────────────────────┼────────────────────┐                │
│           │                    │                    │                │
│           ▼                    ▼                    ▼                │
│    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐          │
│    │  Embedding  │     │  Reranking  │     │   Shared    │          │
│    │   Expert    │     │   Expert    │     │   Expert    │          │
│    │ (LoRA ~50MB)│     │ (LoRA ~50MB)│     │ (LoRA ~50MB)│          │
│    └─────────────┘     └─────────────┘     └─────────────┘          │
│                                                                       │
│    Total: 1.2 GB base + 150 MB adapters = 1.35 GB                    │
│    vs. 2.4 GB for two separate models (44% savings!)                 │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### Why This is Novel

| Aspect | Status |
|--------|--------|
| MoE-LoRA exists | ✅ Yes (MoLoRA, MixLoRA, X-LoRA) |
| Unified retrieval+reranking exists | ✅ Yes (GRITLM, E²RANK) |
| **MoE-LoRA for unified retrieval+reranking** | ❌ **NO ONE HAS DONE THIS** |

---

## Part 7: How Would It Work

### Mode 1: Embedding (Single Text Input)

```
Input: "How does photosynthesis work?"
        │
        ▼
   ┌─────────┐
   │  Base   │ ──► Router sees: single text, no [SEP]
   │  Model  │         │
   └────┬────┘         ▼
        │       Route to: Embedding Expert (weight=0.8) + Shared (weight=0.2)
        │              │
        ▼              ▼
   ┌─────────────────────────┐
   │  Base + Weighted LoRA   │
   │  h' = h + 0.8*E_emb(h)  │
   │        + 0.2*E_shared(h)│
   └───────────┬─────────────┘
               │
               ▼
        [EOS] embedding → Query Vector
```

### Mode 2: Reranking (Query + Document Pair)

```
Input: "[CLS] How does photosynthesis work? [SEP] Plants convert... [SEP]"
        │
        ▼
   ┌─────────┐
   │  Base   │ ──► Router sees: two texts with [SEP]
   │  Model  │         │
   └────┬────┘         ▼
        │       Route to: Reranking Expert (weight=0.9) + Shared (weight=0.1)
        │              │
        ▼              ▼
   ┌─────────────────────────┐
   │  Base + Weighted LoRA   │
   │  h' = h + 0.9*E_rank(h) │
   │        + 0.1*E_shared(h)│
   └───────────┬─────────────┘
               │
               ▼
        Classification head → Relevance Score: 0.92
```

### Mathematical Formulation

For input hidden states $h$ from the base model:

$$h' = h + \sum_{i=1}^{N} w_i \cdot E_i(h)$$

Where:
- $w_i = \text{Router}(h)_i$ are routing weights (sum to 1)
- $E_i(h) = B_i \cdot A_i \cdot h$ is the LoRA expert output
- $A_i \in \mathbb{R}^{r \times d}$, $B_i \in \mathbb{R}^{d \times r}$ are low-rank matrices

---

## Part 8: Why MoE-LoRA is Better Than Alternatives

| Aspect | Separate Models | Single Model (GRITLM) | MoE-LoRA (Proposed) |
|--------|-----------------|----------------------|---------------------|
| **Memory** | 2.4 GB | 1.2 GB | 1.35 GB |
| **Task Specialization** | ✅ Full | ❌ Interference | ✅ Expert-specific |
| **Knowledge Sharing** | ❌ None | ✅ Full | ✅ Via shared expert |
| **Training** | Separate | Joint but competing | Joint with routing |
| **Flexibility** | ❌ Fixed | ❌ Fixed | ✅ Add new experts |
| **Inference Efficiency** | 2 forward passes | 1 forward pass | 1 forward pass |

### Key Advantages

1. **Best of both worlds**: Specialization (like separate models) + sharing (like unified models)
2. **Extensible**: Can add new experts for other tasks (classification, QA, summarization)
3. **Efficient**: Single base model, tiny adapters
4. **Learnable routing**: Model learns when to use which expert

---

## Part 9: Technical Implementation Details

### Router Design Options

#### Option A: Task-Explicit Router (Simpler)

```python
class TaskExplicitRouter(nn.Module):
    """Router that uses explicit task indicator"""
    def forward(self, hidden_states, task_type):
        # task_type: "embedding" or "reranking"
        if task_type == "embedding":
            return torch.tensor([0.8, 0.0, 0.2])  # [emb, rank, shared]
        else:
            return torch.tensor([0.0, 0.9, 0.1])
```

**Pros**: Simple, deterministic, no routing collapse
**Cons**: No learned flexibility, requires task indicator at inference

#### Option B: Learned Router (More interesting for research)

```python
class LearnedRouter(nn.Module):
    """Router that learns from input characteristics"""
    def __init__(self, hidden_size, num_experts=3):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts)

    def forward(self, hidden_states):
        # Use [CLS] token or mean pooling
        pooled = hidden_states[:, 0, :]  # [batch, hidden]
        logits = self.gate(pooled)  # [batch, num_experts]
        weights = F.softmax(logits, dim=-1)
        return weights  # e.g., [0.7, 0.2, 0.1]
```

**Pros**: Learns task distinction automatically, can handle ambiguous inputs
**Cons**: May suffer from routing collapse, needs auxiliary loss

#### Option C: Token-Level Router (Most sophisticated)

```python
class TokenLevelRouter(nn.Module):
    """Different routing per token (like X-LoRA)"""
    def __init__(self, hidden_size, num_experts=3):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts)

    def forward(self, hidden_states):
        # hidden_states: [batch, seq_len, hidden]
        logits = self.gate(hidden_states)  # [batch, seq_len, num_experts]
        weights = F.softmax(logits, dim=-1)
        return weights  # Different experts per token!
```

**Pros**: Fine-grained control, different parts of input use different experts
**Cons**: More complex, higher overhead

### Training Procedure

```python
def train_step(batch):
    """
    Training with mixed embedding and reranking examples
    """
    # 1. Embedding examples (contrastive loss)
    emb_queries = batch["emb_queries"]      # "What is X?"
    emb_positives = batch["emb_positives"]  # Relevant doc
    emb_negatives = batch["emb_negatives"]  # Irrelevant doc

    q_emb = model.encode(emb_queries, mode="embedding")
    p_emb = model.encode(emb_positives, mode="embedding")
    n_emb = model.encode(emb_negatives, mode="embedding")

    # InfoNCE contrastive loss
    emb_loss = contrastive_loss(q_emb, p_emb, n_emb)

    # 2. Reranking examples (classification loss)
    rank_queries = batch["rank_queries"]
    rank_docs = batch["rank_docs"]
    rank_labels = batch["rank_labels"]  # 0 or 1

    scores = model.rerank(rank_queries, rank_docs, mode="reranking")
    rank_loss = F.binary_cross_entropy(scores, rank_labels)

    # 3. Router auxiliary loss (load balancing)
    router_aux_loss = compute_load_balance_loss(model.get_routing_stats())

    # 4. Combined loss
    total_loss = emb_loss + rank_loss + 0.01 * router_aux_loss

    return total_loss
```

### Loss Functions

#### Embedding Loss (InfoNCE / Contrastive)

$$\mathcal{L}_{emb} = -\log \frac{\exp(sim(q, p^+) / \tau)}{\exp(sim(q, p^+) / \tau) + \sum_{p^-} \exp(sim(q, p^-) / \tau)}$$

#### Reranking Loss (Binary Cross-Entropy)

$$\mathcal{L}_{rank} = -[y \log(\sigma(s)) + (1-y) \log(1 - \sigma(s))]$$

#### Router Auxiliary Loss (Load Balancing)

$$\mathcal{L}_{aux} = N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

Where $f_i$ is fraction of tokens routed to expert $i$, and $P_i$ is average routing probability.

---

## Part 10: What Makes This Publishable

### Novel Contributions

1. **First MoE-LoRA architecture for unified retrieval + reranking**
   - Neither GRITLM nor E²RANK use MoE routing
   - Novel architecture design

2. **Task-aware routing mechanism**
   - Learns to distinguish embedding vs reranking automatically
   - Or: explicit task conditioning with soft expert mixing

3. **Knowledge transfer analysis**
   - Show that embedding expert helps reranking (and vice versa)
   - Analyze what the shared expert learns

4. **Efficiency gains**
   - 44% memory reduction vs separate models
   - Single inference pipeline

5. **Extensibility demonstration**
   - Can add more experts: classification, QA, summarization
   - Plug-and-play new capabilities

### Experimental Plan

```
Experiments to Run:
├── MTEB Benchmark (embedding quality)
│   ├── Compare vs Qwen3-Embedding
│   ├── Compare vs GRITLM
│   └── Compare vs E²RANK
│
├── BEIR/MS MARCO (reranking quality)
│   ├── Compare vs Qwen3-Reranker
│   ├── Compare vs cross-encoder baselines
│   └── Compare vs E²RANK
│
├── Unified Pipeline (end-to-end RAG)
│   ├── Retrieval + Reranking combined
│   ├── Latency and throughput analysis
│   └── Memory footprint comparison
│
├── Ablation Studies
│   ├── Router design: explicit vs learned vs token-level
│   ├── Number of experts: 2 vs 3 vs 4
│   ├── Expert rank: 16 vs 32 vs 64
│   ├── With/without shared expert
│   └── Training data ratio (embedding vs reranking)
│
└── Analysis
    ├── Routing visualization: when does it use which expert?
    ├── Knowledge transfer: freeze one expert, train other
    ├── Expert specialization: what does each expert learn?
    └── Failure case analysis
```

### Expected Results Table (Hypothetical)

| Method | MTEB Avg | BEIR nDCG@10 | Memory | Latency |
|--------|----------|--------------|--------|---------|
| Qwen3-Embedding | 65.2 | - | 1.2 GB | 10ms |
| Qwen3-Reranker | - | 52.1 | 1.2 GB | 100ms |
| Separate (both) | 65.2 | 52.1 | 2.4 GB | 110ms |
| GRITLM | 63.8 | 50.3 | 1.2 GB | 100ms |
| E²RANK | 64.1 | 51.2 | 1.2 GB | 95ms |
| **Ours (MoE-LoRA)** | **65.5** | **52.4** | **1.35 GB** | **105ms** |

---

## Part 11: Potential Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| **Different input formats** (single text vs pair) | Use task indicator or learn from input structure (presence of [SEP]) |
| **Different output heads** (embedding vs score) | Shared backbone, separate lightweight heads |
| **Training data imbalance** | Balanced sampling strategy, alternating batches |
| **Router collapse** (always same expert) | Auxiliary load balancing loss, expert dropout during training |
| **Negative transfer** | Careful initialization, gradual unfreezing |
| **Evaluation overhead** | Automated pipelines for MTEB + BEIR |

### Router Collapse Mitigation Strategies

1. **Auxiliary Loss**: Penalize uneven expert usage
2. **Expert Dropout**: Randomly disable experts during training
3. **Noise Injection**: Add noise to routing logits
4. **Minimum Usage Constraint**: Force minimum tokens per expert

---

## Part 12: Code Skeleton to Get Started

### Full Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Optional, Literal

class LoRAExpert(nn.Module):
    """Single LoRA expert with low-rank decomposition"""
    def __init__(self, in_features: int, out_features: int, rank: int = 32):
        super().__init__()
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        # Initialize A with Kaiming, B with zeros (standard LoRA init)
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(x))


class MoERouter(nn.Module):
    """Learned router for expert selection"""
    def __init__(self, hidden_size: int, num_experts: int, temperature: float = 1.0):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, num_experts)
        )
        self.temperature = temperature
        self.num_experts = num_experts

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: [batch, seq_len, hidden]
        Returns:
            weights: [batch, num_experts] - routing weights
            aux_loss: scalar - load balancing loss
        """
        # Pool to get sequence representation
        pooled = hidden_states[:, 0, :]  # Use [CLS] token

        # Compute routing logits and weights
        logits = self.gate(pooled) / self.temperature
        weights = F.softmax(logits, dim=-1)

        # Compute auxiliary load balancing loss
        # Encourages uniform expert usage
        mean_weights = weights.mean(dim=0)  # [num_experts]
        aux_loss = self.num_experts * (mean_weights ** 2).sum()

        return weights, aux_loss


class MoELoRAUnifiedRetriever(nn.Module):
    """
    Unified Embedding + Reranking model with MoE-LoRA

    Architecture:
    - Frozen base model (e.g., Qwen3-0.6B-Base)
    - Multiple LoRA experts (embedding, reranking, shared)
    - Learned router for dynamic expert selection
    """

    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen3-0.6B-Base",
        num_experts: int = 3,
        lora_rank: int = 32,
        router_temperature: float = 1.0
    ):
        super().__init__()

        # Load and freeze base model
        self.base_model = AutoModel.from_pretrained(base_model_name)
        for param in self.base_model.parameters():
            param.requires_grad = False

        hidden_size = self.base_model.config.hidden_size

        # Create LoRA experts
        # Expert 0: Embedding specialist
        # Expert 1: Reranking specialist
        # Expert 2: Shared/general knowledge
        self.experts = nn.ModuleList([
            LoRAExpert(hidden_size, hidden_size, lora_rank)
            for _ in range(num_experts)
        ])

        # Router
        self.router = MoERouter(hidden_size, num_experts, router_temperature)

        # Task-specific heads
        self.rerank_head = nn.Linear(hidden_size, 1)

        # Store config
        self.num_experts = num_experts
        self.hidden_size = hidden_size

    def _apply_moe_lora(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor
    ) -> torch.Tensor:
        """Apply weighted combination of LoRA experts"""
        # hidden_states: [batch, seq_len, hidden]
        # routing_weights: [batch, num_experts]

        batch_size, seq_len, hidden = hidden_states.shape

        # Compute expert outputs
        expert_outputs = torch.stack([
            expert(hidden_states) for expert in self.experts
        ], dim=1)  # [batch, num_experts, seq_len, hidden]

        # Weight and sum expert outputs
        weights = routing_weights.view(batch_size, self.num_experts, 1, 1)
        combined = (expert_outputs * weights).sum(dim=1)  # [batch, seq_len, hidden]

        # Add to original (residual connection)
        return hidden_states + combined

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mode: Literal["embedding", "reranking"] = "embedding"
    ) -> torch.Tensor:
        """
        Encode input and return embeddings or reranking scores
        """
        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_states = outputs.last_hidden_state

        # Compute routing weights
        routing_weights, aux_loss = self.router(hidden_states)

        # Store aux_loss for training
        self._last_aux_loss = aux_loss

        # Apply MoE-LoRA
        enhanced_hidden = self._apply_moe_lora(hidden_states, routing_weights)

        if mode == "embedding":
            # Return normalized [EOS] or last token embedding
            # Find last non-padding token
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(enhanced_hidden.size(0))
            embeddings = enhanced_hidden[batch_indices, seq_lengths]
            return F.normalize(embeddings, p=2, dim=-1)

        elif mode == "reranking":
            # Return relevance score from [CLS] token
            cls_hidden = enhanced_hidden[:, 0, :]
            score = self.rerank_head(cls_hidden).squeeze(-1)
            return torch.sigmoid(score)

    def get_aux_loss(self) -> torch.Tensor:
        """Return the last computed auxiliary loss"""
        return getattr(self, '_last_aux_loss', torch.tensor(0.0))


class UnifiedRetrieverTrainer:
    """Training loop for the unified model"""

    def __init__(
        self,
        model: MoELoRAUnifiedRetriever,
        tokenizer,
        learning_rate: float = 1e-4,
        aux_loss_weight: float = 0.01
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.aux_loss_weight = aux_loss_weight

        # Only train LoRA experts and router
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

    def contrastive_loss(
        self,
        query_emb: torch.Tensor,
        pos_emb: torch.Tensor,
        neg_emb: torch.Tensor,
        temperature: float = 0.05
    ) -> torch.Tensor:
        """InfoNCE contrastive loss"""
        # Positive similarity
        pos_sim = (query_emb * pos_emb).sum(dim=-1) / temperature

        # Negative similarities
        neg_sim = (query_emb * neg_emb).sum(dim=-1) / temperature

        # InfoNCE loss
        logits = torch.stack([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

        return F.cross_entropy(logits, labels)

    def train_step(self, batch: dict) -> dict:
        """Single training step with mixed tasks"""
        self.model.train()
        self.optimizer.zero_grad()

        total_loss = 0.0
        losses = {}

        # 1. Embedding task
        if "emb_queries" in batch:
            q_emb = self.model.encode(
                batch["emb_query_ids"],
                batch["emb_query_mask"],
                mode="embedding"
            )
            p_emb = self.model.encode(
                batch["emb_pos_ids"],
                batch["emb_pos_mask"],
                mode="embedding"
            )
            n_emb = self.model.encode(
                batch["emb_neg_ids"],
                batch["emb_neg_mask"],
                mode="embedding"
            )

            emb_loss = self.contrastive_loss(q_emb, p_emb, n_emb)
            total_loss += emb_loss
            losses["emb_loss"] = emb_loss.item()

        # 2. Reranking task
        if "rank_query_ids" in batch:
            scores = self.model.encode(
                batch["rank_pair_ids"],
                batch["rank_pair_mask"],
                mode="reranking"
            )
            rank_loss = F.binary_cross_entropy(scores, batch["rank_labels"].float())
            total_loss += rank_loss
            losses["rank_loss"] = rank_loss.item()

        # 3. Auxiliary loss
        aux_loss = self.model.get_aux_loss()
        total_loss += self.aux_loss_weight * aux_loss
        losses["aux_loss"] = aux_loss.item()

        # Backward pass
        total_loss.backward()
        self.optimizer.step()

        losses["total_loss"] = total_loss.item()
        return losses


# Example usage
if __name__ == "__main__":
    # Initialize model
    model = MoELoRAUnifiedRetriever(
        base_model_name="Qwen/Qwen3-0.6B-Base",
        num_experts=3,
        lora_rank=32
    )

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")

    # Print trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")

    # Example: embedding mode
    query = "How does photosynthesis work?"
    inputs = tokenizer(query, return_tensors="pt", padding=True)
    embedding = model.encode(inputs["input_ids"], inputs["attention_mask"], mode="embedding")
    print(f"Embedding shape: {embedding.shape}")

    # Example: reranking mode
    query = "How does photosynthesis work?"
    doc = "Photosynthesis is the process by which plants convert sunlight into energy."
    inputs = tokenizer(query, doc, return_tensors="pt", padding=True)
    score = model.encode(inputs["input_ids"], inputs["attention_mask"], mode="reranking")
    print(f"Relevance score: {score.item():.4f}")
```

---

## References

### Core MoE-LoRA Papers

1. **MoLoRA (ICLR 2024)**: [Pushing Mixture of Experts to the Limit](https://openreview.net/forum?id=uWvKBCYh4S)
2. **MixLoRA (2024)**: [Enhancing LLM Fine-Tuning with LoRA-based MoE](https://arxiv.org/abs/2404.15159)
3. **X-LoRA (APL ML 2024)**: [Mixture of Low-Rank Adapter Experts](https://pubs.aip.org/aip/aml/article/2/2/026119/3294581)
4. **MALoRA (NAACL 2025)**: [Mixture of Asymmetric Low-Rank Adaptation](https://arxiv.org/abs/2410.22782)

### Unified Retrieval Papers

5. **GRITLM (ICLR 2025)**: [Generative Representational Instruction Tuning](https://openreview.net/forum?id=70cf215430492f7d34830a24e744b3f1)
6. **E²RANK (2025)**: [Your Text Embedding Can Also Be an Effective Reranker](https://openreview.net/forum?id=5Iwj0WW1vT)
7. **Qwen3-Embedding (2025)**: [Advancing Text Embedding and Reranking](https://qwenlm.github.io/blog/qwen3-embedding/)

### Implementation Resources

8. **X-LoRA GitHub**: https://github.com/EricLBuehler/xlora
9. **MixLoRA GitHub**: https://github.com/TUDB-Labs/MixLoRA
10. **MOELoRA-peft**: https://github.com/Applied-Machine-Learning-Lab/MOELoRA-peft

---

## Timeline for Research

```
Month 1-2: Literature Review & Problem Formulation
├── Deep dive into MoE-LoRA papers
├── Study unified retrieval approaches
├── Identify specific research gaps
└── Formulate hypothesis and contributions

Month 3-4: Implementation & Baseline
├── Implement MoE-LoRA architecture
├── Set up training pipeline
├── Reproduce baseline results (Qwen3, GRITLM)
└── Initial experiments on small scale

Month 5-6: Main Experiments
├── Full-scale training
├── MTEB evaluation
├── BEIR evaluation
├── Ablation studies

Month 7: Analysis & Writing
├── Routing visualization
├── Knowledge transfer analysis
├── Draft paper
└── Internal review

Month 8: Submission
├── Finalize paper
├── Submit to target venue
└── Prepare rebuttal materials
```

---

## Contact & Collaboration

This research idea is open for exploration. Key skills needed:
- PyTorch / Transformers experience
- Understanding of retrieval systems
- GPU resources for training (A100 recommended)

---

*Document created: February 2026*
*Last updated: February 2026*
