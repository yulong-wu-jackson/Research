# Evaluation Pipeline + Analysis Scripts

**Created:** 2026-02-06
**Updated:** 2026-02-07
**Status:** Draft
**Depends on:** ticket_003 (needs trained checkpoints to evaluate)
**Blocks:** ticket_005 (production runs use eval + analysis)

## Overview

Implement the full evaluation pipeline (reranking on BEIR/MS MARCO dev via yes/no scoring, embedding on MTEB v2) and the analysis scripts that compute the kill gate metric (Task Interference Ratio), generate comparison tables, and produce paper-ready figures.

## Context

- **Reranking eval:** Score query-passage candidates using yes/no token logits (aligned with Qwen3-Reranker), compute nDCG@10 and MRR@10 via `pytrec_eval`
- **Embedding eval:** Wrap model in MTEB v2 `EncoderProtocol`, run `mteb.evaluate()` on MTEB(eng, v2) retrieval tasks
- **Fast tier (kill gate):** SciFact, NFCorpus, FiQA2018 — small datasets, evaluates in ~15 min
- **Full tier:** All 15 publicly available BEIR datasets + MS MARCO dev + MTEB(eng, v2)
- **Kill gate metric:** Task Interference Ratio (TIR) = `(Specialist_metric - Joint_metric) / Specialist_metric`; pass if TIR >= 2%
- **BEIR datasets** need BM25 top-100 candidate retrieval before reranking

## Requirements

### Model Wrappers (`src/unimoe/evaluation/model_wrappers.py`)
- [ ] `MTEBEncoderWrapper`: wraps `UnimodelForExp1` to implement MTEB v2 `EncoderProtocol`:
  ```python
  def encode(self, inputs: DataLoader[BatchedInput], *, task_metadata, ...) -> Array
  ```
  Uses model's `encode()` method (last-token pooling + L2 norm). Handles instruction prefix for queries based on `prompt_type`.
- [ ] `MTEBCrossEncoderWrapper`: wraps `UnimodelForExp1` to implement MTEB v2 `CrossEncoderProtocol`:
  ```python
  def predict(self, inputs1: DataLoader, inputs2: DataLoader, ...) -> Array
  ```
  Uses model's `rerank()` method (yes/no token logit scoring). Formats inputs using Qwen3-Reranker chat template.

### Reranking Evaluation (`src/unimoe/evaluation/reranking_eval.py`)
- [ ] `evaluate_msmarco_dev(model, tokenizer, config)`: load MS MARCO dev queries (6,980) + qrels, use pre-computed BM25 top-100 candidates, rerank with model using yes/no scoring, compute MRR@10 via pytrec_eval
- [ ] `evaluate_beir(model, tokenizer, config, dataset_name)`: download BEIR dataset, BM25 top-100 retrieval (via `rank_bm25` or beir's built-in), rerank with model, compute nDCG@10 via pytrec_eval
- [ ] `evaluate_reranking_suite(model, tokenizer, config)`: run all configured BEIR datasets + MS MARCO dev, return aggregated results dict with per-dataset and average scores
- [ ] Reranking scoring function matching Qwen3-Reranker inference exactly:
  ```python
  logits = model(**inputs).logits[:, -1, :]
  true_vector = logits[:, yes_token_id]
  false_vector = logits[:, no_token_id]
  scores = torch.stack([false_vector, true_vector], dim=1)
  scores = torch.nn.functional.log_softmax(scores, dim=1)
  score = scores[:, 1].exp()  # P(yes)
  ```

### Embedding Evaluation (`src/unimoe/evaluation/embedding_eval.py`)
- [ ] `evaluate_mteb(model, tokenizer, config)`: run MTEB(eng, v2) benchmark using the `MTEBEncoderWrapper`, save results to output directory. For fast tier, run only retrieval task subset (SciFact, NFCorpus, FiQA2018).

### Evaluate Script (`scripts/evaluate.py`)
- [ ] CLI: `--checkpoint <path>`, `--config <yaml>`, `--eval-tier <fast|full>`
  - Loads checkpoint (PEFT adapter)
  - Runs reranking eval (always, BM25 top-100 first stage)
  - Runs embedding eval (for EMB_ONLY and JOINT configs)
  - Saves per-dataset results as JSON in `outputs/{config}/seed_{N}/results/`
  - JSON includes: per-query scores (for bootstrap significance testing), aggregate metrics, config snapshot

### Analysis (`src/unimoe/analysis/compare.py`, `scripts/analyze_results.py`)
- [ ] `load_all_results(output_dir)`: scan output directories, load all result JSONs, return structured dict `{config_name: {seed: {dataset: {metric: value}}}}`
- [ ] `compute_interference_metrics(results)`:
  - Task Interference Ratio (TIR): `(Specialist_metric - Joint_metric) / Specialist_metric`
  - Compute TIR per seed for both tasks (reranking TIR using rank_only as specialist, embedding TIR using emb_only as specialist)
  - Report mean +/- std across seeds
  - Kill gate verdict: PASS (>= 2%), FAIL (< 1%), MARGINAL (1-2%)
  - Also report absolute score deltas alongside TIR
- [ ] `compute_significance(results)`: paired bootstrap resampling (10,000 iterations) on per-query scores for key comparisons
- [ ] `generate_comparison_table(results)`: produce markdown + LaTeX-ready table comparing all configs across all metrics. Format: bold best, underline second-best, subscript std.
- [ ] `generate_plots(results, output_path)`:
  - Bar charts of per-dataset nDCG@10 for each config
  - Training loss curves (from JSONL logs)
  - Gradient conflict heatmap per layer (from gradient_conflicts.jsonl)

### Analyze Script (`scripts/analyze_results.py`)
- [ ] CLI: `--output-dir <path>`
  - Loads all results, computes metrics, generates tables + plots
  - Prints kill gate verdict to stdout with clear formatting
  - Saves analysis report as markdown in `outputs/analysis/`

## Design Decisions

- **MTEB v2 protocol (not legacy API):** MTEB v2 (2025+) has breaking API changes. Must implement `EncoderProtocol` with `DataLoader[BatchedInput]` inputs and `task_metadata`. The old `encode(sentences)` API is deprecated.
- **Yes/no scoring for reranking eval:** Matches the model architecture (ticket_003). Score = `softmax(logit_no, logit_yes)[yes_index]`. This is how Qwen3-Reranker scores at inference time.
- **BEIR BM25 retrieval first, then rerank:** Standard practice. We rerank BM25 top-100 candidates, not the full corpus. Use `rank_bm25` (lightweight, no Elasticsearch dependency) or beir's built-in.
- **Fast tier (3 datasets) for kill gate:** SciFact (5.2K docs), NFCorpus (3.6K docs), FiQA2018 (57K docs) — small enough for quick iteration, diverse enough for signal.
- **Full tier: 15 BEIR datasets:** All publicly available datasets. CQADupStack averaged into one score (12 subforums). Non-public datasets (BioASQ, Signal-1M, TREC-NEWS, Robust04) excluded.
- **Task Interference Ratio (TIR) over "PDR":** More descriptive name. Explicitly defined as: `TIR = (Specialist - Joint) / Specialist`. Always report absolute scores alongside TIR. TIR is a novel metric — must be clearly defined in the paper.
- **Paired bootstrap for significance:** Standard in IR evaluation. 10K iterations, p < 0.05 threshold. Per-query scores stored in results JSON for post-hoc analysis.
- **Results as JSON files:** Simple, human-readable, easy to aggregate. Each eval run produces one JSON per dataset, including per-query scores.
- **Separate eval from training:** Evaluation is expensive and may need re-running with different configs. Decoupling keeps the pipeline flexible.

## Scope

**In scope:** Reranking eval (BEIR 15 datasets + MS MARCO dev), embedding eval (MTEB v2), MTEB v2 model wrappers (Encoder + CrossEncoder), analysis/comparison scripts, TIR computation, bootstrap significance, plot generation

**Out of scope:** Training (ticket_003), production experiment orchestration (ticket_005), TREC DL19/DL20 eval (future work), BRIGHT eval (future work), AIR-Bench (future work)

## Technical Notes

- BEIR dataset download is handled by the `beir` library and cached. First run may take a few minutes per dataset.
- MS MARCO dev evaluation: download pre-computed BM25 top-1000 from the official MS MARCO repository. Truncate to top-100 for reranking.
- MTEB v2 requires `mteb >= 2.7.0`. The `EncoderProtocol` and `CrossEncoderProtocol` are the new standard interfaces.
- CQADupStack has 12 subforums. Evaluate all, report the average as a single score.
- pytrec_eval metric names: `"recip_rank"` for MRR, `"ndcg_cut_10"` for nDCG@10.
- Reranking scoring batch_size: 64-128 depending on GPU memory. The 0.6B model with 512 max length should fit 128 on a 24GB GPU.
- Store per-query scores in results JSON for statistical testing: `{"per_query": {"query_id": score, ...}, "aggregate": {"ndcg@10": 0.xx}}`.

## Acceptance Criteria

- [ ] Evaluation runs on the sanity checkpoint from ticket_003 (rank_only_r8)
- [ ] Reranking eval produces non-zero MRR@10 on MS MARCO dev and nDCG@10 on BEIR fast-tier datasets
- [ ] Reranking scoring uses yes/no token logits (not MLP head)
- [ ] Embedding eval (on emb_only or joint config checkpoint) produces non-zero MTEB scores
- [ ] MTEB wrapper implements `EncoderProtocol` (MTEB v2 API)
- [ ] CrossEncoder wrapper implements `CrossEncoderProtocol` (MTEB v2 API)
- [ ] Results saved as JSON files in correct output directory structure, including per-query scores
- [ ] `scripts/analyze_results.py` loads results, computes TIR, prints verdict, generates table + at least one plot
- [ ] Analysis works with partial results (e.g., only rank_only available, joint not yet run)
- [ ] `uv run pytest tests/ -v` all green
