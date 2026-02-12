# Future Experiments & Enhancements

**Created:** 2026-02-07
**Status:** Backlog (post-kill-gate)
**Context:** These experiments are planned for after the kill gate (ticket_005) passes. They represent the full research agenda for the UniMoER paper targeting EMNLP 2026.

---

## Tier 1: Immediate Post-Kill-Gate (Weeks 5-8)

### F1. Joint-MoE-LoRA Architecture (Experiment 1 completion)

**Goal:** Implement the core MoE-LoRA routing architecture and complete the 4-config comparison.

- Implement custom `MoELoRALayer` on top of PEFT's LoRA primitives (~200-400 lines)
- 3 experts: embedding specialist (r=8), reranking specialist (r=8), shared (r=16)
- Asymmetric rank: shared expert gets higher rank per GuiLoMo (EMNLP 2025) and "How Relevance Emerges" findings
- Default router: Per-layer Sparsegen (LD-MoLE style) — must implement from paper formula (no public code)
- Load balancing auxiliary loss (lambda=0.01)
- Router noise injection: input jitter (epsilon=0.01)
- Reference HMoRA (ICLR 2025, [GitHub](https://github.com/LiaoMengqi/HMoRA)) for implementation patterns
- Add `JOINT_MOE` to `TrainingMode` enum
- Add `RouterConfig` dataclass (router_type, num_experts, expert_ranks, aux_loss_weight, noise_epsilon)
- Compare: Emb-Only vs Rank-Only vs Joint-Single vs **Joint-MoE** (the full Experiment 1)

### F1b. Mixed-Batch Joint Training (Alternative to Alternating)

**Goal:** Compare mixed-batch vs alternating-batch joint training for interference measurement.

- Current default: **alternating batches** — odd steps = embedding, even steps = reranking (GritLM-style)
- Alternative: **mixed batches** — each step computes both `L_emb` and `L_rerank` on separate data, combines as `L = L_emb + λ·L_rerank`, single backward pass (E2Rank-style)
- Test λ in {0.5, 1.0, 2.0, 5.0} (E2Rank found λ=2.0 optimal)
- The difference is scientifically interesting: mixed batches create direct gradient interference within a single step, while alternating batches create interference across consecutive steps
- Mixed batches may show stronger interference signal (both tasks compete for gradient direction simultaneously)
- Add `JOINT_MIXED` to `TrainingMode` enum

### F2. Staged Training (Embedding-First, Then Joint)

**Goal:** Test whether pre-training the embedding task first before adding reranking improves results.

- Stage 1: Train embedding-only with InfoNCE for N epochs (LR=1e-4)
- Stage 2: Joint training with alternating batches (LR=5e-5, lower to preserve Stage 1)
- This mirrors E2Rank's successful approach and is well-motivated
- Add `JOINT_STAGED` to `TrainingMode` enum
- Add `StagedTrainingConfig` with stage1_epochs, stage2_epochs, stage1_lr, stage2_lr

### F3. Scale-Dependent Interference Study (Experiment 2)

**Goal:** Demonstrate that interference is a capacity phenomenon, explaining GRITLM's result.

- Run Emb-Only, Rank-Only, Joint-Single at: 0.6B (Qwen3), 1.5B (Qwen3), 4B (Qwen3), 7B (Mistral)
- 3 seeds per scale for primary (0.6B), 3 seeds for interpolation points
- Plot interference magnitude (TIR) vs model scale
- Expected: TIR decreases monotonically with scale → provides explanation for GRITLM's "no loss" at 7B

---

## Tier 2: Analysis Deep-Dive (Weeks 7-10)

### F4. Gradient Conflict Analysis (Experiment 3)

**Goal:** Identify the mechanistic cause of interference via gradient-level measurement.

- Per-layer gradient cosine similarity between embedding and reranking batches (infrastructure built in ticket_003)
- Gradient magnitude ratio ||g_emb|| / ||g_rank|| per layer
- PCGrad-style projection to quantify conflict magnitude (use [PCGrad-PyTorch](https://github.com/OrthoDex/PCGrad-PyTorch))
- Compare gradient conflict with vs without MoE routing
- Expected: conflict concentrates in mid-range layers (per "How Relevance Emerges" findings)

### F5. Routing Behavior Analysis (Experiment 4)

**Goal:** Deep interpretability of learned routing (MoE-LoRA only).

- Per-task routing weight distributions (how do routing weights differ for embedding vs reranking?)
- Per-layer routing patterns (do different layers prefer different experts?)
- Training dynamics visualization (how do routing patterns evolve during training?)
- Domain-dependent routing (do different BEIR domains trigger different routing patterns?)
- Expert probing: what linguistic/semantic properties does each expert capture?

### F6. Representation Space Analysis (Experiment 5)

**Goal:** Evidence that experts factorize the representation space.

- CKA (Centered Kernel Alignment) using `ckatorch` library — compare expert output representations across tasks
- t-SNE/UMAP visualization of embeddings with vs without MoE routing
- Singular value analysis of expert weight matrices
- Gradient subspace overlap: SVD of per-task gradients, measure principal angles

---

## Tier 3: Architecture Validation (Weeks 8-10)

### F7. Soft Routing vs Hard Selection (Experiment 6)

**Goal:** Demonstrate learned routing advantage over hard task-selection.

| Configuration | Description |
|---------------|-------------|
| **Hard-Switch (Jina v3 style)** | Two LoRAs selected by task indicator |
| **Hard-Switch + Shared** | Two task LoRAs + shared, selected by indicator |
| **Similarity-Based (RouterRetriever style)** | Expert selection by query-pilot similarity |
| **Soft-Router (Ours)** | Three experts with learned per-layer Sparsegen routing |
| **DynMoLE variant** | Three experts with Tsallis entropy-based routing |

### F8. Knowledge Transfer Analysis (Experiment 7)

**Goal:** Demonstrate positive cross-task transfer via shared expert.

| Configuration | Description |
|---------------|-------------|
| Freeze Rerank Expert | Train only embedding + shared; evaluate reranking |
| Freeze Embed Expert | Train only reranking + shared; evaluate embedding |
| Full Training | All experts jointly |
| No Shared Expert | Only embedding + reranking experts |
| Isolated LoRAs | Separate task-specific LoRAs, no shared training |

### F9. Rank Sensitivity & Expert Count Ablation (Experiment 8)

**Goal:** Determine minimum expert rank and optimal expert count.

- Expert rank ablation: r = {1, 4, 8, 16, 32} for each expert independently and jointly
- Asymmetric rank: test r_shared in {8, 16, 32}, r_specialist in {1, 4, 8, 16}
- Expert count ablation: 2, 3, 4, 6 experts (per GuiLoMo and MoTE findings)
- "How Relevance Emerges" shows rank 1 suffices for reranking → if confirmed, dramatic parameter savings

---

## Tier 4: Practical Validation (Week 11)

### F10. End-to-End RAG Pipeline (Experiment 9)

- Datasets: Natural Questions, TriviaQA, HotpotQA
- Metrics: Answer accuracy (EM, F1), end-to-end latency, peak GPU memory
- Hardware: Single T4 (16GB), single A100 (40GB)
- Compare: two separate models vs UniMoER unified model

### F11. Reranking Mode Comparison (Experiment 10)

- Pointwise: yes/no token scoring (current approach, Qwen3-Reranker-aligned)
- MLP head: first-token pooling -> MLP -> scalar (ablation against default)
- Listwise: late interaction following Jina Reranker v3's pattern (if time permits)

---

## Tier 5: Enhancements for Paper Strength

### F12. Bidirectional Attention for Embedding (Ablation)

**Goal:** Test whether switching to bidirectional attention for embedding mode improves results.

- GRITLM uses bidirectional attention for embedding (`is_causal=False`)
- Qwen3-Embedding uses causal attention (and achieves SOTA) → our default
- Implement attention mode toggle via custom modeling file (GRITLM approach)
- Compare: causal + last-token pool vs bidirectional + mean pool
- Reference: "Pooling and Attention for LLM-based Embeddings" (ICLR 2025) shows bidirectional is better for retrieval/STS but worse for classification/clustering
- This is an interesting ablation but NOT required for the core paper

### F13. Extended Evaluation Benchmarks

- **TREC DL19/DL20:** Deep relevance judgments on MS MARCO corpus. nDCG@10 is more reliable than MRR@10.
- **BRIGHT:** Reasoning-intensive retrieval (ICLR 2025). Tests if unified model improves reasoning retrieval.
- **AIR-Bench:** LLM-generated queries across 8 domains. Fresh benchmark, less contamination risk.
- **FollowIR:** Instruction-following IR (NAACL 2025). Low priority — all models score poorly.

### F14. 5-Seed Statistical Rigor

- Upgrade primary experiments from 3 seeds to 5 seeds (42, 123, 456, 789, 2026)
- 3 seeds is minimum acceptable; 5 is safer for EMNLP reviewers
- Use Almost Stochastic Order (ASO) from `deep-significance` package for model comparisons
- Apply Bonferroni correction for multi-dataset comparisons

### F15. GradCache for Hard Negative Embeddings

**Current state:** Hard negatives are encoded with `torch.no_grad()` in
`trainer.py:_embedding_step()`. This avoids OOM (56 seqs × 256 len ×
28 layers stores ~15 GB of backward activations on A100-40GB) but means
gradients don't flow through hard negatives — only through query,
positive, and in-batch negatives.

**Design decision rationale (2026-02-12):**
- sentence-transformers' `CachedMultipleNegativesRankingLoss` uses the
  same `no_grad` initial pass, then a second mini-batch pass with gradients
- DPR keeps full gradients but only uses 1 hard negative (not 7)
- For the kill gate experiment measuring task interference, the simpler
  approach is sufficient — absolute retrieval performance is not the goal
- References: [sentence-transformers CachedMNRL](https://github.com/UKPLab/sentence-transformers),
  [DPR](https://github.com/facebookresearch/DPR)

**Future improvement:** Implement GradCache (two-pass) to restore
gradients through hard negatives while keeping memory bounded:
1. Pass 1: encode all negatives with `no_grad`, cache embeddings
2. Compute loss and backprop to embedding level, cache embedding gradients
3. Pass 2: re-encode in mini-batches with gradients, chain cached gradients
- This enables massive virtual batch sizes (4096-65536) with constant GPU memory
- Significant improvement expected for embedding quality in final paper experiments

### F16. Matryoshka Representation Learning

- Train the model to produce embeddings that can be truncated to smaller dimensions (1024→512→256→128→64)
- Qwen3-Embedding already supports this
- Wrap InfoNCE loss with MatryoshkaLoss
- Practical value: flexible deployment (trade precision for speed)
- Not core to research contribution but makes the model more practically useful

### F17. Multi-Task Loss Optimization

- **PCGrad:** Project conflicting gradients onto orthogonal complement (measurement already built in ticket_003, optimization is an extension)
- **GradNorm:** Dynamically adjust loss weights to balance training rates
- **FAMO (NeurIPS 2023):** O(1) space/time loss-based balancing (no per-task gradients needed)
- **Auxiliary-loss-free (DeepSeek-V3 style):** Dynamic bias terms on router logits instead of load-balancing loss

### F18. Additional Related Work to Reference

Papers discovered during deep research that should be cited in the paper:
- **MoTE** (ACL 2025): Task-aware expert routing for embedding models — very close to our work
- **HMoRA** (ICLR 2025): Hierarchical MoE-LoRA — closest implementation reference
- **Causal2Vec** (arXiv 2507): Decoder-only embedding without bidirectional attention
- **NV-Retriever** (ICLR 2025): Positive-aware hard negative mining
- **GuiLoMo** (EMNLP 2025): Expert number/rank allocation
- **ASE** (arXiv 2510): Adaptive shared expert design
- **Ortho-LoRA** (Jan 2025): Orthogonal gradient projection for multi-task LoRA
- **MTL-LoRA** (AAAI 2025): Shared-private LoRA decomposition
- **REARANK** (EMNLP 2025): Reasoning re-ranking agent
- **"Distillation vs Contrastive Learning"** (arXiv 2507): Evidence that contrastive/BCE is more robust OOD

---

## Implementation Priority After Kill Gate

If kill gate **PASSES** (TIR >= 2%):
1. F1 (MoE-LoRA architecture) — core contribution
2. F4 (gradient conflict analysis) — primary analytical contribution
3. F5 (routing behavior) — interpretability contribution
4. F7 (soft vs hard routing) — key ablation
5. F9 (rank sensitivity + expert count) — ablation
6. F3 (scale study) — explaining GRITLM
7. F6 (representation space) — supporting analysis
8. F8 (knowledge transfer) — supporting analysis
9. F13 (extended benchmarks) — paper completeness
10. F14 (5 seeds) — statistical rigor

If kill gate is **MARGINAL** (TIR 1-2%):
1. F15 (CachedMNRL) — improve embedding training quality
2. F2 (staged training) — may reveal clearer interference
3. F14 (5 seeds) — determine if marginal is real
4. Then re-evaluate whether to proceed with F1

If kill gate **FAILS** (TIR < 1%):
1. Pivot to interpretability study
2. F4 (gradient analysis) — study WHY interference doesn't emerge
3. F3 (scale study) — map the interference frontier
4. F12 (bidirectional attention) — test different architectural choices
