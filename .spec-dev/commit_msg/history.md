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
