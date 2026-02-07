# Production Runs + Kill Gate Decision

**Created:** 2026-02-06
**Updated:** 2026-02-07
**Status:** Draft
**Depends on:** ticket_004 (evaluation and analysis pipeline must work end-to-end)
**Blocks:** None (this is the final ticket; outcome determines project direction)

## Overview

Implement the master experiment runner script and execute the full kill gate experiment: train Emb-Only, Rank-Only, and Joint-Single LoRA configs across 3 seeds, evaluate all checkpoints, and compute the Task Interference Ratio (TIR) to make the go/no-go decision for the research project.

## Context

- **Quick tier:** r=8, seed=42 only, all 3 configs — validates end-to-end pipeline works before committing to multi-day runs
- **Full tier:** r=16, seeds=[42, 123, 456], all 3 configs — production kill gate experiment
- **Kill gate:** TIR_reranking >= 2% OR TIR_embedding >= 2% means interference exists (proceed). TIR < 1% on both means no interference (pivot). Between 1-2% is marginal (needs investigation).
- All infrastructure (data, model, training, eval, analysis) must be working from prior tickets
- **3 configs are required:** Emb-Only (embedding upper bound), Rank-Only (reranking upper bound), Joint-Single (interference measurement). ALL three are needed to compute TIR for both tasks.

## Requirements

### Experiment Runner (`scripts/run_experiment1.py`)
- [ ] CLI: `--quick` (r=8, seed=42 only) vs default full tier (r=16, seeds=[42, 123, 456])
  - `--device` flag (default: auto — CUDA > MPS > CPU)
  - Sequentially runs: train -> evaluate -> (next config/seed) -> ... -> analyze
  - Runs all 3 configs: `emb_only`, `rank_only`, `joint_single`
  - For each config+seed: invoke `scripts/train.py`, then `scripts/evaluate.py`
  - After all runs complete: invoke `scripts/analyze_results.py`
  - Logs overall progress and timing to stdout
  - Handles graceful interruption (can resume by skipping completed runs based on existing checkpoints)

### Execution Plan
- [ ] Run quick tier first: `uv run python scripts/run_experiment1.py --quick`
  - 3 configs x 1 seed = 3 training runs
  - Validates full pipeline end-to-end
  - Produces preliminary TIR estimate (may not be statistically meaningful at r=8)
- [ ] Run full tier: `uv run python scripts/run_experiment1.py`
  - 3 configs x 3 seeds = 9 training runs
  - Each run: training + evaluation
  - Total: depends on hardware (estimate ~1-2 days on A100, ~3-5 days on M3 Pro)

### Kill Gate Analysis
- [ ] After full tier: `analyze_results.py` produces:
  - Per-seed TIR values for BOTH tasks (reranking TIR and embedding TIR)
  - Mean TIR +/- std across seeds for each task
  - Absolute score comparisons (Specialist vs. Joint per metric)
  - Clear verdict: PASS (>= 2%), FAIL (< 1%), or MARGINAL (1-2%)
  - Per-dataset breakdown (which BEIR domains show most interference?)
  - Gradient conflict summary (from gradient_conflicts.jsonl logged during joint training)
  - Comparison table (markdown + LaTeX-ready for the paper)
  - Bar chart figure saved to `outputs/figures/`
  - Bootstrap significance test results for key comparisons

## Design Decisions

- **3 configs (not 2):** The previous version was missing Emb-Only, which is essential. Without an embedding specialist, you cannot measure embedding-side interference (TIR_embedding). All 3 configs are listed in the proposal's Experiment 1.
- **Sequential execution (not parallel):** Single GPU. Runs must be sequential. The runner loops through configs and seeds.
- **Resume capability:** Check for existing `checkpoints/final/` directory before starting a run. If exists, skip training and go to evaluation. This prevents re-running completed experiments after interruption.
- **Quick tier before full tier:** Catches pipeline bugs early without wasting days of compute.
- **3 seeds (42, 123, 456):** Minimum acceptable for NLP/IR experiments. Sufficient for the kill gate decision. If interference is clear (>5% TIR), 3 seeds provides adequate confidence. (5 seeds recommended for the full paper — see future experiments doc.)
- **TIR on both tasks:** Interference might be asymmetric — reranking could degrade while embedding remains stable, or vice versa. Measuring both tasks is essential.
- **r=16 for production (not r=32):** Standard in recent IR research. r=32 is excessive for 0.6B; r=16 provides sufficient capacity while keeping experiments affordable.

## Scope

**In scope:** Experiment runner script, quick tier execution, full tier execution, kill gate verdict, gradient conflict summary

**Out of scope:** Joint-MoE-LoRA config (comes after kill gate passes), staged training experiments, scale study (Experiment 2), paper writing, 5-seed runs

## Acceptance Criteria

- [ ] `run_experiment1.py --quick` completes end-to-end without error
- [ ] Quick tier produces results for all 3 configs (emb_only, rank_only, joint_single)
- [ ] Quick tier produces preliminary TIR estimates for both tasks
- [ ] `run_experiment1.py` (full tier) completes all 9 runs
- [ ] Final analysis output includes:
  - Mean TIR +/- std for both embedding and reranking tasks with clear pass/fail verdict
  - Absolute score comparison table (all configs x all metrics)
  - Per-dataset nDCG@10 comparison table
  - Gradient conflict heatmap from joint training
  - Bootstrap significance test results
  - At least two figures saved to `outputs/figures/`
- [ ] Results are reproducible: same seed produces same metrics (within floating-point tolerance)
- [ ] All outputs are self-contained in `outputs/` with frozen config copies per run

## Post-Completion: Kill Gate Decision

After this ticket completes, the research direction is decided:

| TIR Result | Decision | Next Action |
|-----------|----------|-------------|
| **>= 2% on either task** | PASS | Proceed: implement Joint-MoE-LoRA (Experiment 1 full), then scale study (Experiment 2). See `future-experiments.md`. |
| **1-2% on both tasks** | MARGINAL | Investigate: try more training data, longer training, different LoRA rank, staged training, or add more BEIR domains. See `future-experiments.md` for options. |
| **< 1% on both tasks** | FAIL | Pivot: reframe as interpretability study of why interference doesn't emerge. Still publishable — see proposal Section 8 risk mitigation. |
