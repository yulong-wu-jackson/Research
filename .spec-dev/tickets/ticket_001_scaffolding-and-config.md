# Project Scaffolding + Configuration System

**Created:** 2026-02-06
**Updated:** 2026-02-07
**Status:** Draft
**Depends on:** None (first ticket)
**Blocks:** All subsequent tickets

## Overview

Set up the Python project from scratch: dependency management via `uv`, directory structure for the `unimoe` package, and a dataclass-based configuration system with YAML serialization for reproducible experiments.

## Context

- **Starting state:** Zero implementation code. Repo contains only research proposals and `CLAUDE.md`.
- **Package manager:** `uv` (per CLAUDE.md)
- **Python:** 3.13
- **Primary hardware target:** CUDA (cloud GPU, e.g., A100/T4) for training; MPS (M3 Pro, 36GB) for development and debugging only
- **Design principle:** CUDA-first, MPS-compatible. Use `float16` on MPS (bfloat16 not supported), `bfloat16` on CUDA.

## Requirements

- [ ] Create `pyproject.toml` with all project dependencies (pinned to latest stable versions verified for Python 3.13 compatibility):
  - **Core:**
    - `torch>=2.6.0` — Python 3.13 + torch.compile support added in 2.6; CUDA 12.x compatible (latest: 2.10.0, Jan 2026)
    - `transformers>=4.51.0,<5.0.0` — v5.0 has breaking changes (TF/JAX removed, reported gibberish bugs); pin to 4.x stable line (latest: 4.57.x). v5.x can be revisited once mature.
    - `peft>=0.18.0` — latest stable 0.18.1 (Jan 2026); requires Python >=3.10; Python 3.13 officially supported
    - `datasets>=4.0.0` — HuggingFace datasets v4; Arrow-based; explicitly supports Python 3.13+3.14 (latest: 4.5.0)
    - `accelerate>=1.0.0` — stable 1.x line (latest: 1.12.0); requires Python >=3.10
    - `sentence-transformers>=4.0.0,<6.0.0` — v5.2.2 is latest (Jan 2026) with Qwen3 support, CrossEncoder multi-processing, multilingual NanoBEIR; requires Python >=3.10. Note: major version jumped from 3.x to 4.x to 5.x rapidly — pin lower bound conservatively.
    - `tqdm>=4.66.0`
    - `numpy>=2.1.0` — NumPy 1.x is EOL (Sep 2025); 2.1+ required for Python 3.13 support. Has breaking namespace/ABI changes from 1.x. (latest: 2.4.2)
    - `pandas>=2.2.0,<3.0.0` — stable 2.x line (latest: 2.2.3). pandas 3.0 has breaking changes (str dtype default, removed deprecations); 2.2.x is battle-tested and sufficient for our data analysis needs.
  - **Evaluation:**
    - `mteb>=2.7.0` — MTEB v2 with EncoderProtocol/CrossEncoderProtocol support
    - `pytrec-eval-terrier>=0.5.10` — the Terrier fork has pre-built Python 3.13 wheels (the original `pytrec-eval` does NOT support 3.13)
    - `beir>=2.2.0` — latest stable (Jun 2025); uses pytrec_eval internally. **Note:** does NOT officially list Python 3.13 in classifiers (lists up to 3.12). Likely works since its deps support 3.13, but needs manual verification.
    - `rank-bm25>=0.2.2` — lightweight BM25 without Elasticsearch dependency. **Note:** unmaintained since ~2022; pure Python + numpy only, so works on 3.13. Consider `bm25s` as actively-maintained alternative if issues arise.
  - **Analysis:**
    - `matplotlib>=3.10.0` — Python 3.13 binary wheels included
    - `seaborn>=0.13.0`
    - `ckatorch>=1.0.2` — released Dec 2025; explicitly supports Python 3.10-3.13
    - `deepsig>=1.2.8` — ASO statistical testing; pure Python so likely works on 3.13. **Note:** last PyPI release Oct 2023, no official Python 3.13 declaration. Low maintenance but functional. (339 GitHub stars, GPL-3.0)
  - **Config:**
    - `pyyaml>=6.0.2` — Python 3.13 compatible
    - `dacite>=1.9.0` — pure Python; latest 1.9.2 (Python >=3.7)
  - **Tracking:**
    - `wandb>=0.19.0` — latest 0.24.2 (Feb 2026); Python >=3.8
  - **Dev:**
    - `pytest>=8.3.0` — latest 9.0.2; full Python 3.13 support
    - `pytest-cov>=6.0.0` — latest 7.0.0 (Sep 2025)
    - `ruff>=0.9.0` — latest 0.15.0 (Feb 2026); Rust binary, no Python version dep
- [ ] Create `.python-version` pinning Python 3.13
- [ ] Create full directory structure: `src/unimoe/` with subpackages `data/`, `model/`, `training/`, `evaluation/`, `analysis/`; plus `configs/`, `scripts/`, `tests/`
- [ ] Update `.gitignore` to exclude `outputs/`, `data/`, `__pycache__/`, `.venv/`, `wandb/`, `*.egg-info/`, `.cache/`
- [ ] Run `uv sync` and verify all deps install correctly
- [ ] Implement `src/unimoe/config.py`:
  - `ModelConfig`: base_model_name, torch_dtype, device
  - `LoRAConfig`: rank, alpha, dropout, target_modules (list), task_type
  - `DataConfig`: dataset_name, embedding_samples, reranking_samples, query_max_len, passage_max_len, reranking_max_len, num_hard_negatives, instruction_prefix
  - `TrainingConfig`: mode (TrainingMode enum), lr, warmup_ratio, epochs, batch_size_embedding, batch_size_reranking, grad_accum_steps, max_grad_norm, optimizer, temperature, reranking_loss_weight
  - `EvalConfig`: eval_tier (fast/full), beir_datasets (list), eval_batch_size
  - `ExperimentConfig`: combines all above + seed, output_dir, wandb_enabled
  - `TrainingMode` enum: `RANK_ONLY`, `EMB_ONLY`, `JOINT_SINGLE`
  - `ScoringMode` enum: `YES_NO_LOGITS`, `MLP_HEAD`
  - YAML loading helper using `pyyaml` + `dacite`
  - Device auto-detection: CUDA > MPS > CPU with appropriate dtype selection
- [ ] Create 6 YAML config files in `configs/`:
  - `emb_only_r16.yaml` — embedding specialist baseline
  - `rank_only_r16.yaml` — reranking specialist baseline
  - `joint_single_r16.yaml` — joint training (interference measurement)
  - `emb_only_r8.yaml` — fast sanity check
  - `rank_only_r8.yaml` — fast sanity check
  - `joint_single_r8.yaml` — fast sanity check

## Design Decisions

- **Python 3.13 (not 3.11 or 3.12):** All core dependencies (PyTorch 2.6+, transformers 4.51+, PEFT, sentence-transformers, mteb, ckatorch) now support Python 3.13. PyTorch 2.6 added torch.compile support for 3.13. The free-threaded build (3.13t) is NOT supported — use standard CPython 3.13.
- **`transformers<5.0.0`:** v5.0 removed TensorFlow/JAX, renamed parameters, and has reported bugs (gibberish output, import errors). The 4.x line (4.57.x) is battle-tested and stable.
- **`pytrec-eval-terrier` (not `pytrec-eval`):** The original pytrec-eval package does NOT have Python 3.13 wheels. The Terrier team fork (maintained by University of Glasgow) ships pre-built wheels for Python 3.13 on Linux/macOS/Windows.
- **NumPy 2.x (not 1.x):** NumPy 1.x reached end-of-life in September 2025. NumPy 2.x has namespace changes (e.g., `np.string_` removed) but all our dependencies support it.
- **`uv` over pip/conda:** Mandated by CLAUDE.md
- **Dataclasses + dacite over Pydantic/Hydra:** Lightweight, no heavy dependency. dacite handles YAML-to-dataclass conversion cleanly. Sufficient for the kill gate experiment scope. (Hydra may be adopted later for large-scale ablations.)
- **`TrainingMode` enum:** Makes mode switching explicit and type-safe across the codebase
- **`ScoringMode` enum:** Supports both yes/no token logits (Qwen3-Reranker-aligned, default) and MLP head (ablation)
- **6 configs (r=8 and r=16):** r=8 configs exist solely for fast sanity checking (~30 min). r=16 are the production configs — standard in recent IR research (Jina v3 uses r=16). r=32 was excessive for 0.6B scale.
- **CUDA-first design:** MPS has silent kernel bugs, no bfloat16, limited ecosystem. Production training targets CUDA; MPS used only for development.
- **3 configs (not 2):** Must include `emb_only` alongside `rank_only` and `joint_single` to establish both task upper bounds for PDR measurement.

## Scope

**In scope:** Project init, deps, directory layout, config dataclasses, YAML configs, gitignore, device auto-detection

**Out of scope:** Any data loading, model code, training logic, or evaluation

## Acceptance Criteria

- [ ] `uv sync` completes without errors
- [ ] `uv run python -c "import torch; print(torch.cuda.is_available() or torch.backends.mps.is_available())"` prints `True`
- [ ] `uv run python -c "from unimoe.config import ExperimentConfig, TrainingMode, ScoringMode"` imports without error
- [ ] `tests/test_config.py` passes: loads each of the 6 YAML configs, verifies field types, defaults, round-trip serialization, and device auto-detection logic
- [ ] `uv run pytest tests/test_config.py -v` all green
