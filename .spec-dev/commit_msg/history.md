# Commit Message History

This file contains a history of generated commit messages for this project.

---
**[2026-02-10 00:00:00]**

feat: implement full unimoe experiment pipeline (tickets 001-005)

- Set up project with uv, Python 3.13, and all dependencies (pyproject.toml)
- Implement config system with dataclass configs and YAML loading (src/unimoe/config.py)
- Create 6 experiment YAML configs (emb_only, rank_only, joint_single × r8/r16)
- Implement data pipeline: MS MARCO download, embedding/reranking collators, templates
- Implement model: Qwen3-0.6B-Base + LoRA with dual-mode encode/rerank
- Implement training: InfoNCE + RerankingSFT losses, unified trainer with alternating steps
- Implement evaluation: MTEB/BEIR wrappers, reranking eval, embedding eval
- Implement analysis: interference metrics, significance testing, comparison tables, plots
- Add CLI scripts: train, evaluate, download_data, analyze_results, run_experiment1
- Add comprehensive test suite (7 test modules, all passing)
- Mark tickets 001-005 as complete

---
**[2026-02-11 00:00:00]**

fix(data,eval): free large lookup dicts after dataset construction and fix gitignore ignoring source code

- Change .gitignore `data/` to `/data/` so src/unimoe/data/ is tracked by git
- Add explicit del + gc.collect() in load_embedding_dataset and
  load_reranking_dataset to release corpus_lookup (~8.8M entries),
  query_lookup, and raw HF datasets before building final Dataset
- Add memory cleanup in evaluate_msmarco_dev after candidate construction
- Add memory cleanup in evaluate_beir after reranking loop to free BM25
  index, tokenized corpus, and corpus text lists (500MB-2GB per dataset)

---
**[2026-02-11 12:00:00]**

fix(training,eval,analysis): address 8 review findings affecting experiment validity

- Fix doubled output path in modal_app.py that caused missed checkpoints
- Replace even/odd task alternation with proportional Bresenham scheduling
  so joint training gives each task equal micro-steps to its specialist
- Prefix BEIR per-query IDs with dataset name to prevent silent overwrites
  in bootstrap significance tests (SciFact/FiQA2018 share integer IDs)
- Replace itertools.cycle with non-caching _infinite_dataloader generator
- Include embedding_samples, num_hard_negatives, reranking_samples in
  dataset cache keys to prevent stale data reuse
- Add paired bootstrap significance test for embedding (emb_only vs joint)
- Unify train/eval reranking tokenization via shared build_reranking_token_ids
  utility to eliminate BPE boundary mismatches
- Fix evaluate_reranking_suite docstring to match BEIR-only implementation
- Add 8 new tests (128 total, all passing)

---
**[2026-02-11 22:15:00]**

fix(analysis,data,config): fix bootstrap test, loss weight confound, and data pipeline issues

- Replace anti-conservative bootstrap with proper paired permutation test
  (sign-flip under H0) for correct Type-I error control
- Set reranking_loss_weight to 1.0 in joint configs for fair TIR comparison
- Fix loss curve parser crash on epoch summary JSONL entries
- Add DataLoader num_workers=4 and per-loader seeded generators for
  deterministic shuffling and GPU utilization
- Filter hard negatives against positive passage IDs to prevent false negatives
- Randomize reranking negative sampling with seed for cross-seed variance
- Add padding_side='left' assertion in both collators
- Add yes/no token ID validation in model init
- Add Colab notebook for running experiment behind Zscaler proxies
