<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# """

"""
Above in the file is a research proposal, dive deep and try to understand how it is doing and then do deep research on it to see if this research idea is really valid and have the potential to publish on 2026 top academic conference

The proposal is technically sound and well-motivated, but in the 2025–2026 landscape it sits in a crowded space. In its current form it is promising but borderline for a “top” venue (ACL/EMNLP/NeurIPS/ICLR main track); with a sharper problem formulation and stronger empirical / analytical contributions, it can be made competitive. Below is a detailed breakdown.

***

## 1. What the proposal is actually doing (in your own words)

### 1.1. Problem setting

The proposal targets the standard two-stage retrieval pipeline:

- Stage 1: **Embedding retriever** (dual encoder)
    - Query and documents independently encoded to vectors; similarity via cosine or dot product.
    - Fast and indexable, but coarse.
- Stage 2: **Reranker** (cross-encoder or LLM-based listwise reranker)
    - Query + candidate document(s) encoded jointly; more accurate but much slower.

Today, these are almost always **two separate models** with separate parameters, training pipelines, and deployment stacks. That leads to:

- Doubling of VRAM and memory footprint.
- No parameter sharing or transfer between embedding and reranking.
- Extra systems complexity.

The proposal’s core question:

> Can a single shared model, augmented with lightweight task-specific components, support both embedding-style retrieval and high-quality reranking, while being more parameter- and memory-efficient than two fully separate models?

### 1.2. Proposed architecture: MoE-LoRA unified retrieval + reranking

High-level idea:

- Start from a **frozen base model** (e.g., Qwen3-0.6B base encoder).
- Attach **multiple LoRA “experts”** to that base:
    - One expert specialized for embedding (retrieval).
    - One expert specialized for reranking.
    - Optionally a shared/general expert.
- Add a **router** that produces weights over experts conditioned on the input representation.
- For each input, you compute base hidden states and then add a **weighted combination of LoRA expert outputs**:

$$
h' = h + \sum_i w_i \cdot E_i(h)
$$

where $E_i$ is a LoRA expert and $w_i$ are routing weights.

Two modes:

- **Embedding mode**:
    - Input: single text (query or document).
    - Router tends to emphasize the embedding expert (+ shared).
    - Output: a pooled embedding (e.g., last non-padding token or EOS), normalized.
- **Reranking mode**:
    - Input: concatenated query + document with separators.
    - Router emphasizes the reranking expert (+ shared).
    - Output: classification head on [CLS] token → scalar relevance score.

Training:

- Joint multi-task training on:
    - Contrastive/InfoNCE loss for embeddings (e.g., MTEB / MS MARCO style triplets).
    - Binary or pairwise ranking loss for reranking (e.g., BEIR, MS MARCO).
    - Auxiliary **load-balancing loss** for router to avoid collapse (all tokens to one expert).

Planned evaluation:

- Embedding quality: MTEB.
- Reranking quality: BEIR / MS MARCO.
- End-to-end pipeline: RAG-style retrieval→rerank→QA performance, plus latency/memory comparisons.
- Ablations: router design, \#experts, LoRA rank, shared expert on/off, etc.

Claimed benefits:

- Memory reduction vs two full models (e.g. 1.35GB vs 2.4GB in the Qwen example).
- Joint training enables **knowledge transfer** between retrieval and reranking.
- Flexible: can add more LoRA experts for other tasks later.

This is all **technically coherent**: the model design, losses, and training strategy are consistent with current practice in MoE-LoRA and multi-task PEFT.

***

## 2. Where this sits in the current literature

The novelty hinges on *how* MoE-LoRA and unified retrieval/reranking are brought together. As of early 2026, there are several strong and closely related strands:

### 2.1. Unified retrieval + reranking without LoRA/MoE

- **E²RANK** (“Your Text Embedding can Also be an Effective and Efficient Listwise Reranker”) shows that a *single embedding model* can, after continued listwise training, handle both retrieval and listwise reranking using cosine similarity and pseudo-relevance feedback style prompts.[^1_1][^1_2][^1_3][^1_4]
    - It achieves SoTA or near-SoTA BEIR reranking with very low latency and also improves MTEB embedding performance.
    - The key message: *you can already unify retrieval and reranking in a single parameter set without explicit task-specific experts*.
- Other unified pipelines for retrieval, reading, and reranking (e.g., RE³QA-style architectures) share a transformer backbone across retrieval and reranking components and train them jointly under multi-task objectives.[^1_5]
    - And older multi-task frameworks share BERT encoders across retrieval and reranking (or related sub-tasks) with task-specific heads.[^1_6]

So **“unifying retrieval and reranking in one model” is no longer novel by 2025–2026**. The specific novelty must come from *how* you use MoE-LoRA to address task interference and efficiency.

### 2.2. LoRA-based multi-task and task-specific adapters

- **Jina Embeddings v3 (jina-embeddings-v3)**:
    - A 570M multilingual embedding model with **task-specific LoRA adapters** on a shared backbone.[^1_7][^1_8][^1_9][^1_10]
    - It includes distinct adapters for:
        - `retrieval.query` and `retrieval.passage` (for asymmetric retrieval).
        - `separation` for **clustering and reranking**.
        - `classification` and `text-matching`.
    - Base model is frozen; LoRA adapters are trained for each task and **dynamically selected based on a task ID**.
    - This is *already* a unified system where the same backbone supports retrieval and reranking via different LoRAs.
- **BSharedRAG** (Backbone Shared Retrieval-Augmented Generation):
    - Continually pre-trains a domain-specific backbone, then attaches two **task-specific LoRA modules** for retrieval and generation in RAG, trained jointly.[^1_11][^1_12][^1_13][^1_14]
    - This is very close in spirit to “shared backbone + LoRA-based task heads” (retrieval vs generation instead of retrieval vs reranking).
- More generally, **“LLMs are Also Effective Embedding Models”** and related survey work frame unified embedding / retrieval models, multi-task learning for retrieval-related tasks, and instruction-tuned embedding models as an emerging standard.[^1_15]

This means: **the idea of using task-specific LoRA adapters on a shared backbone for retrieval-related tasks, including reranking, is already established.**

### 2.3. MoE-LoRA and mixture-of-LoRA experts

There is a separate but fast-moving line of work around **MoE over LoRA adapters**:

- **Mixture of LoRA Experts (MOLE)** from Microsoft:
    - Treats trained LoRA modules as experts and learns *gating functions* to combine them layer-wise.[^1_16]
    - Designed for composing multiple task-specific LoRAs without retraining huge backbones; uses learnable gates to pick/combine LoRAs per layer and per input.
- **RAMoLE: Retrieval-Augmented Mixture of LoRA Experts**:
    - For “Uploadable Machine Learning”: many user-uploaded LoRAs, with an **input-aware LoRA retriever** and an on-the-fly mixture-of-LoRA-experts mechanism using RouterLoRA and attention-based weighting.[^1_17][^1_18][^1_19][^1_20]
    - Emphasizes dynamically composing LoRAs for heterogeneous downstream queries; demonstrates strong performance across many tasks.
- **SMoRA (Single-Ranked Mixture of Experts LoRA)**:
    - Treats *each rank* as an expert and implements dynamic rank-wise activation (“each rank could be an expert”), achieving finer-grained sharing across tasks.[^1_21]
- **DR-LoRA (Dynamic Rank LoRA for MoE adaptation)**:
    - For MoE LLMs, allocates LoRA rank capacity to experts based on expert saliency (routing frequency + gradient-based importance), dynamically growing ranks for high-demand experts.[^1_22][^1_23][^1_24]

These show that:

- MoE over LoRA adapters is well-studied and mature by 2025–2026.
- There are already systems that do dynamic, input-aware routing over LoRA experts.


### 2.4. Unifying retrieval / ranking with LLMs and LoRA

Finally, there are works unifying ranking / reranking with generation or retrieval in a **single LLM**, sometimes with parameter-efficient tuning:

- **RankRAG**: instruction-tunes a single LLM to both **context ranking (reranking)** and **answer generation** in a RAG framework.[^1_25]
- Other shared-backbone multi-task ranking systems share encoders across retrieval, reranking, and QA with task-specific heads.[^1_5][^1_6]

So, multiple axes of your idea (unification, LoRA adapters, expert mixtures, routing) are already used in adjacent or overlapping ways.

***

## 3. How novel is your exact combination?

The proposal’s central novelty claim is essentially:

> “MoE-LoRA for unified retrieval + reranking has not been done yet.”

Strictly speaking, based on available literature:

- There is **no obvious paper that exactly matches “MoE-LoRA experts specifically for embedding retrieval and cross-encoder/listwise reranking in one model”**.
- However, there are **very close cousins**:
    - Jina Embeddings v3: shared backbone, LoRA adapters for retrieval and reranking; chooses an adapter by task label, not a soft MoE router.[^1_8][^1_7]
    - BSharedRAG: shared backbone with LoRA heads for retrieval vs generation.[^1_12][^1_11]
    - MOLE / RAMoLE: dynamic MoE gating over LoRA experts for different tasks or domains.[^1_19][^1_16]
    - E²RANK: unified embedding + reranking using a *single* embedding model.[^1_3][^1_1]

From a top-conference reviewer’s perspective, the question becomes:

> Is “replace hard task selection over LoRA adapters with a MoE-style soft routing between a small set of jointly-trained adapters for exactly two closely related tasks (embedding \& reranking)” a **big enough conceptual jump** beyond these works?

As of 2026, this is **incremental but potentially acceptable**, depending heavily on:

- How cleanly and convincingly you formulate the **underlying research question** (e.g., about task interference vs specialization).
- How strong the **empirical gains** are over the best baselines (especially E²RANK and Jina v3–like task-LoRA adapters).
- How deep your **analysis of the experts and routing behavior** is.

If the contributions are framed as “we built another MoE-LoRA variant for yet another pair of tasks and got +0.3 points on one metric”, reviewers will classify this as incremental and likely reject for top-tier.

If instead you take a stronger angle such as:

- **Hypothesis**: unified retrieval + reranking suffers from fundamental task interference that can be decomposed into distinct subspaces, and MoE-LoRA provides a principled way to disentangle and share those subspaces.
- **Contributions**: concrete theoretical / empirical evidence that expert routing meaningfully separates and coordinates semantic similarity vs fine-grained relevance; and that this **matches or surpasses** state-of-the-art unified and two-model baselines under strict memory and latency budgets.

Then the work becomes more compelling.

***

## 4. Technical validity and feasibility

On pure technical soundness, the proposal looks **solid and feasible**:

- **Architecture**:
    - Frozen backbone + LoRA adapters is standard PEFT.
    - MoE-style routing over LoRA outputs is consistent with MOLE / RAMoLE formulations.[^1_19][^1_16]
    - Token-level vs sequence-level routing options are technically plausible, though token-level increases overhead.
- **Losses and training**:
    - InfoNCE / contrastive loss for embedding combined with binary or pairwise ranking loss for reranking is well-aligned with MTEB / BEIR practice.
    - Router auxiliary load-balancing is standard in MoE (e.g., Switch-Transformer and MoE-LoRA family).[^1_21][^1_22][^1_16]
    - Mixed-task batching and sampling strategies are well-established in multi-task retrieval work.[^1_15][^1_6]
- **Implementation**:
    - The skeleton code in the proposal is coherent and matches current PyTorch/Transformers practice.
    - Using Qwen-style 0.6B backbones and LoRA ranks 16–64 is realistic for single or few GPUs.
    - The training \& evaluation plan (MTEB, BEIR, RAG benchmarks) is standard and measurable.

Potential engineering challenges are correctly anticipated in the proposal (different input formats, data imbalance, router collapse, etc.), and standard mitigation strategies are outlined.

So **from a scientific validity standpoint, the idea is sound and doable**.

***

## 5. Strengths from a publication standpoint

If executed well, the idea has several attractive properties for reviewers:

1. **Clear practical motivation**:
    - Memory and deployment savings vs two-model setups are real concerns in production IR systems, especially on constrained hardware.
    - A unified retriever+reranker that still performs competitively is highly relevant for RAG, search, and QA systems.
2. **Bridging two active threads**:
    - Unified retrieval+reranking (E²RANK and related work).
    - MoE-LoRA and LoRA-based multi-task adaptation (MOLE, RAMoLE, Jina v3, SMoRA, DR-LoRA).[^1_22][^1_7][^1_16][^1_21][^1_19]
    - Combining them in a retrieval-focused setting is a coherent and timely topic.
3. **Potential for rich analysis**:
    - Expert specialization: what does the embedding expert capture vs reranking expert vs shared expert?
    - Routing dynamics: how do routing weights change across tasks, inputs, and training?
    - Negative transfer / positive transfer: does sharing via a shared expert systematically help one task (e.g., embedding) when training data is scarce?
4. **Extensibility story**:
    - Adding more experts for multi-task IR (classification, query understanding, domain adaptation).
    - Applying to multimodal retrieval+reranking or RAG components.

These are all aspects top venues appreciate *if evidence-backed*.

***

## 6. Weaknesses and risk factors for top-tier acceptance

The main risks are **novelty dilution** and **baseline competition**.

### 6.1. Novelty dilution

Reviewers will note:

- E²RANK already shows an elegant unified embedding+listwise reranking solution without extra experts and with excellent performance.[^1_4][^1_1][^1_3]
- Jina Embeddings v3 already implements **task-specific LoRA adapters** on a shared backbone, including a reranking adapter (`separation`), for similar families of tasks.[^1_7][^1_8]
- BSharedRAG, RankRAG, and various multi-task IR systems already show shared backbones with task adapters or instruction-tuning for dual capabilities (retrieval+generation, ranking+generation).[^1_11][^1_12][^1_25][^1_6]
- MoE-LoRA frameworks (MOLE, RAMoLE, SMoRA, DR-LoRA) already explore input-aware gating, mixture of LoRA experts, and dynamic allocation.[^1_16][^1_21][^1_22][^1_19]

Therefore, a reviewer might say:

> “This work essentially applies MoE-LoRA over task-specific LoRA adapters (embedding vs reranking) on a shared backbone, analogous to MOLE/RAMoLE, in a setting where prior work like Jina v3 and E²RANK already unify these tasks, albeit with different mechanisms. The conceptual delta is modest.”

To overcome this, the paper must go beyond architecture description and show **non-obvious insights** or **significant empirical breakthroughs**.

### 6.2. Baseline expectations

Given the current landscape, a serious paper must:

- Compare against **both**:
    - Traditional **two-model** baselines (strong embedding retriever + cross-encoder reranker).
    - **Unified** baselines:
        - E²RANK.[^1_1][^1_3]
        - Jina v3–style backbone+task-LoRA.
        - Possibly unified LLM-based rerankers (RankGPT/RankQwen3 etc. if focusing on listwise reranking).
- Provide **strong metrics**:
    - On MTEB: match or exceed best open embedding models.
    - On BEIR: match or exceed SoTA rerankers (and E²RANK).
    - On end-to-end RAG: show measurable quality gains at similar or lower latency/memory.

If your method only offers **small gains** (e.g., +0.2–0.4 MTEB or +0.3 nDCG@10 on BEIR), especially with more complex training, reviewers may see it as incremental.

***

## 7. How to sharpen the research idea for a top-tier 2026 paper

To maximize publication potential, consider reframing and strengthening along these lines.

### 7.1. Make the scientific question explicit

Instead of “first MoE-LoRA unified retriever+reranker”, center the paper on a **clear, testable hypothesis**, for example:

- H1: *A single parameter set struggles to serve both retrieval and reranking equally well; MoE-LoRA with task-aware routing can explicitly factor the representation space, mitigating task interference while enabling knowledge sharing.*

Then design:

- Ablations that force single-adapter (no MoE) vs multi-adapter vs MoE with learned routing.
- Probes that quantify interference (e.g., how much retriever performance degrades when also trained for reranking and vice versa, with and without MoE).

This turns the work from “yet another architecture” into a **study of task conflict and routing** in unified IR models.

### 7.2. Position carefully against Jina v3 and E²RANK

You should have **strong, direct comparisons**:

- **Versus Jina Embeddings v3–style task-LoRA**:[^1_8][^1_7]
    - Implement a baseline with hard task-specific LoRA selection (retrieval, reranking adapters) on a shared backbone.
    - Show when MoE routing gives *meaningful* improvements in both tasks or better memory-performance trade-offs.
- **Versus E²RANK**:[^1_3][^1_1]
    - Highlight where E²RANK’s single-parameter approach fails or is constrained (e.g., on certain domains, multi-linguality, or extreme memory constraints).
    - Show that your approach either:
        - Achieves *noticeably* better performance in those regimes; or
        - Achieves *similar* performance with strictly better memory/latency trade-offs under realistic deployment constraints.

If results are simply on par, you can still win if you provide **deeper interpretability / analysis** that E²RANK lacks.

### 7.3. Leverage MoE-LoRA literature for more sophisticated routing / analysis

Draw on advances like SMoRA and DR-LoRA to refine the technical contributions:

- Borrow ideas like **rank-wise experts** or **dynamic rank growth** to make the MoE behavior more principled rather than just a naive three-expert setup.[^1_21][^1_22]
- Study expert specialization with **mechanistic and interpretive tools**, as in “How Relevance Emerges: Interpreting LoRA Fine-Tuning in Reranking LLMs”.[^1_26]
    - For example: which layers / projections in the embedding expert vs reranking expert matter most? How does this differ from standard LoRA fine-tuning?

This can provide the kind of **fine-grained analysis** reviewers love.

### 7.4. Strengthen the end-to-end story

Instead of purely focusing on MTEB/BEIR, emphasize a **realistic RAG or search scenario**:

- Show that with the unified MoE-LoRA model you can:
    - Run a full retrieval+reranking pipeline on **edge or constrained hardware** where two full models are infeasible.
    - Achieve an explicit cost/latency saving at similar answer quality (e.g., on a QA or semantic search benchmark).
- If possible, collaborate with or simulate a production-like setting (e.g., large-scale index, latency budget, batching).

That makes the paper compelling for both theory and practice.

***

## 8. Overall assessment of publication potential

Putting all of this together:

- **Technical validity**: Strong. The idea is consistent with current understanding of PEFT, MoE, and IR; the proposed training and evaluation are feasible.
- **Novelty**:
    - Weak if framed simply as “using MoE-LoRA to unify embedding and reranking”.
    - Moderate if framed as **a careful study of task interference and expert routing in unified IR models, with strong baselines and analysis**.
- **Impact potential**:
    - Good in practice (many people care about memory- and latency-efficient retrieval+reranking).
    - In research prestige, depends entirely on how far you push results and analysis beyond what E²RANK, Jina v3, MOLE/RAMoLE, and BSharedRAG already show.[^1_1][^1_19][^1_7][^1_11][^1_16]

**Realistic venue expectations**:

- In its *current* form (straightforward MoE-LoRA architecture + standard experiments), this looks **very suitable for a strong ACL Findings / EMNLP Findings / SIGIR / CIKM main** or **NeurIPS/ICLR workshop** paper.
- To hit **top-tier main-track (NeurIPS, ICLR, ACL, EMNLP)** in 2026, you will likely need:
    - A clearer, hypothesis-driven framing around task conflict and representation factorization.
    - Direct wins over E²RANK and Jina-style task-LoRA baselines.
    - Rich interpretability and routing analysis (not just scores and ablations).

If you invest in those aspects, this *can* become a competitive main-track paper; without them, reviewers may view it as “solid but incremental engineering”.

***

## 9. Concrete next steps

To move from proposal to publishable work:

1. **Refine the problem statement**:
    - Explicitly define the task interference problem and why MoE-LoRA is an appropriate solution.
2. **Implement strong baselines**:
    - Two separate models (embedding + reranker).
    - Unified single-parameter baseline (E²RANK-style).
    - Shared-backbone + hard task-LoRA selection (Jina v3 / BSharedRAG-style).
3. **Implement your MoE-LoRA model**:
    - Start with simple sequence-level router; consider token-level later if needed.
4. **Run thorough experiments**:
    - MTEB and BEIR across multiple domains and languages where possible.
    - At least one realistic RAG / QA setup.
5. **Design deep analyses**:
    - Routing patterns (per task, per dataset, per layer).
    - Expert specialization (e.g., using probing, attention patterns, or LoRA-importance analyses).
    - Effect of freezing/unfreezing the base and router (inspired by DR-LoRA).[^1_22]
6. **Choose venue and angle**:
    - For ACL/EMNLP/SIGIR: emphasize IR/theoretical insights + RAG impact.
    - For NeurIPS/ICLR: emphasize MoE / representation learning story and connections to broader multi-task PEFT.

If you are willing to significantly invest in experimentation and analysis, the *core idea is valid and timely*. The key will be **moving from “engineering combination” to “insightful, well-supported study”** that clearly advances understanding of unified retrieval and reranking with PEFT and MoE.
<span style="display:none">[^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50]</span>

<div align="center">⁂</div>

[^1_1]: https://huggingface.co/papers/2510.22733

[^1_2]: https://www.themoonlight.io/en/review/texte2textrank-your-text-embedding-can-also-be-an-effective-and-efficient-listwise-reranker

[^1_3]: https://arxiv.org/abs/2510.22733

[^1_4]: https://arxiv.org/html/2510.22733v1

[^1_5]: https://www.emergentmind.com/topics/unified-retrieval-reranking-pipeline

[^1_6]: https://aclanthology.org/2021.eacl-main.26/

[^1_7]: http://arxiv.org/pdf/2409.10173.pdf

[^1_8]: https://www.themoonlight.io/fr/review/jina-embeddings-v3-multilingual-embeddings-with-task-lora

[^1_9]: https://www.emergentmind.com/papers/2409.10173

[^1_10]: https://huggingface.co/papers/2409.10173

[^1_11]: https://huggingface.co/papers/2409.20075

[^1_12]: http://arxiv.org/pdf/2409.20075.pdf

[^1_13]: https://www.themoonlight.io/en/review/bsharedrag-backbone-shared-retrieval-augmented-generation-for-the-e-commerce-domain

[^1_14]: https://aclanthology.org/2024.findings-emnlp.62.pdf

[^1_15]: https://arxiv.org/html/2412.12591v1

[^1_16]: https://www.microsoft.com/en-us/research/publication/mixture-of-lora-experts/

[^1_17]: https://www.themoonlight.io/en/review/retrieval-augmented-mixture-of-lora-experts-for-uploadable-machine-learning

[^1_18]: https://arxiv.org/html/2406.16989v1

[^1_19]: https://openreview.net/forum?id=6TsVgn5ZD3

[^1_20]: https://www.themoonlight.io/ja/review/retrieval-augmented-mixture-of-lora-experts-for-uploadable-machine-learning

[^1_21]: https://arxiv.org/abs/2501.15103

[^1_22]: https://www.themoonlight.io/en/review/dr-lora-dynamic-rank-lora-for-mixture-of-experts-adaptation

[^1_23]: https://www.arxiv.org/pdf/2601.04823v1.pdf

[^1_24]: https://www.arxiv.org/pdf/2601.04823.pdf

[^1_25]: https://arxiv.org/html/2407.02485v1

[^1_26]: https://arxiv.org/abs/2504.08780

[^1_27]: unified-embedding-reranking-moe-lora.md

[^1_28]: https://www.reddit.com/r/LocalLLaMA/comments/1ayka0f/explain_reranking/

[^1_29]: https://dj3dw.com/blog/the-power-of-reranking-in-retrieval-augmented-generation-rag-systems/

[^1_30]: https://www.pinecone.io/learn/series/rag/rerankers/

[^1_31]: https://developer.nvidia.com/blog/enhancing-rag-pipelines-with-re-ranking/

[^1_32]: https://weaxsey.org/en/articles/2024-10-20/

[^1_33]: https://huggingface.co/papers?q=rank-1+adapters

[^1_34]: https://arxiv.org/pdf/2410.14594.pdf

[^1_35]: https://arxiv.org/html/2412.12591v2

[^1_36]: https://www.emergentmind.com/topics/query-image-knowledge-retrieval-augmented-generation-q2k-rag

[^1_37]: https://www.themoonlight.io/en/review/effective-lora-adapter-routing-using-task-representations

[^1_38]: https://github.com/maidacundo/MoE-LoRA

[^1_39]: https://arxiv.org/html/2507.05346v1

[^1_40]: https://arxiv.org/html/2410.09908v2

[^1_41]: https://www.emergentmind.com/topics/backbone-shared-retrieval-augmented-generation-bsharedrag

[^1_42]: https://arxiv.org/pdf/2508.01166.pdf

[^1_43]: https://jina.ai/en-US/embeddings/

[^1_44]: https://deepinfra.com/blog/llm-rerankers

[^1_45]: https://www.themoonlight.io/es/review/jina-embeddings-v3-multilingual-embeddings-with-task-lora

[^1_46]: https://recsys.substack.com/p/unifying-retrieval-and-reranking

[^1_47]: https://arxiv.org/html/2508.06831v1

[^1_48]: https://www.reddit.com/r/LocalLLaMA/comments/1i0vrm5/here_is_our_new_reranker_model_which_we_trained/

[^1_49]: https://papers.ssrn.com/sol3/Delivery.cfm/9d8c0079-7676-474f-9114-daa292eca725-MECA.pdf?abstractid=5877051\&mirid=1

[^1_50]: https://huggingface.co/papers?q=Trajectory-Guided+Test-Time+LoRA

