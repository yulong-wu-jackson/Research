"""Tests for evaluation pipeline and analysis scripts.

Tests verify:
- Model wrappers implement correct interfaces
- Reranking scoring uses yes/no token logits
- Analysis functions compute TIR correctly
- Comparison table and plots generate without error
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from transformers import AutoTokenizer

from unimoe.analysis.compare import (
    compute_interference_metrics,
    compute_significance,
    generate_comparison_table,
    generate_plots,
    load_all_results,
)
from unimoe.config import ExperimentConfig, LoRAConfig, ModelConfig
from unimoe.evaluation.model_wrappers import MTEBCrossEncoderWrapper, MTEBEncoderWrapper
from unimoe.model.lora_model import UnimodelForExp1


@pytest.fixture(scope="module")
def small_config():
    return ExperimentConfig(
        model=ModelConfig(
            base_model_name="Qwen/Qwen3-0.6B-Base",
            torch_dtype="float32",
            device="cpu",
        ),
        lora=LoRAConfig(rank=4, alpha=4, dropout=0.0),
    )


@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base", trust_remote_code=True)
    tok.padding_side = "left"
    return tok


@pytest.fixture(scope="module")
def model(small_config, tokenizer):
    return UnimodelForExp1(small_config, tokenizer=tokenizer)


class TestMTEBEncoderWrapper:
    def test_encode_returns_numpy(self, model):
        wrapper = MTEBEncoderWrapper(model, batch_size=2)
        # Simulate a DataLoader-like input
        inputs = [{"text": ["Hello world", "Test query"]}]
        result = wrapper.encode(inputs)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 1024)

    def test_encode_l2_normalized(self, model):
        wrapper = MTEBEncoderWrapper(model, batch_size=2)
        inputs = [{"text": ["Sample text for embedding"]}]
        result = wrapper.encode(inputs)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_has_model_meta(self, model):
        wrapper = MTEBEncoderWrapper(model, batch_size=2)
        assert wrapper.mteb_model_meta is not None

    def test_instruction_prefix_applied_for_queries(self, model):
        """Verify instruction prefix is added when prompt_type is 'query'."""
        from mteb.types._encoder_io import PromptType

        wrapper = MTEBEncoderWrapper(model, batch_size=2)
        inputs = [{"text": ["What is machine learning?"]}]

        # Encode with query prompt_type (should add instruction prefix)
        result_query = wrapper.encode(inputs, prompt_type=PromptType.query)
        # Encode without prompt_type (no instruction prefix)
        result_doc = wrapper.encode(inputs, prompt_type=PromptType.document)

        # Results should differ because query has instruction prefix
        assert not np.allclose(result_query, result_doc, atol=1e-3), (
            "Query and document embeddings should differ due to instruction prefix"
        )


class TestMTEBCrossEncoderWrapper:
    def test_predict_returns_numpy(self, model):
        wrapper = MTEBCrossEncoderWrapper(model, batch_size=2)
        inputs1 = [{"text": ["What is AI?", "Best pizza?"]}]
        inputs2 = [{"text": ["AI is intelligence", "Margherita pizza"]}]
        result = wrapper.predict(inputs1, inputs2)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)

    def test_predict_scores_in_range(self, model):
        wrapper = MTEBCrossEncoderWrapper(model, batch_size=2)
        inputs1 = [{"text": ["query"]}]
        inputs2 = [{"text": ["document"]}]
        result = wrapper.predict(inputs1, inputs2)
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_has_model_meta(self, model):
        wrapper = MTEBCrossEncoderWrapper(model, batch_size=2)
        assert wrapper.mteb_model_meta is not None


class TestAnalysisCompare:
    @pytest.fixture
    def mock_results_dir(self, tmp_path):
        """Create a mock output directory with result files."""
        # rank_only_r8/seed_42
        ronly = tmp_path / "rank_only_r8" / "seed_42" / "results"
        ronly.mkdir(parents=True)
        with open(ronly / "reranking_results.json", "w") as f:
            json.dump({
                "per_dataset": {
                    "SciFact": {
                        "aggregate": {"ndcg@10": 0.65},
                        "per_query": {"q1": 0.7, "q2": 0.6, "q3": 0.65},
                    }
                },
                "average": {"ndcg@10": 0.65},
            }, f)

        # emb_only_r8/seed_42
        eonly = tmp_path / "emb_only_r8" / "seed_42" / "results"
        eonly.mkdir(parents=True)
        with open(eonly / "reranking_results.json", "w") as f:
            json.dump({
                "per_dataset": {},
                "average": {"ndcg@10": 0.0},
            }, f)

        # joint_single_r8/seed_42
        joint = tmp_path / "joint_single_r8" / "seed_42" / "results"
        joint.mkdir(parents=True)
        with open(joint / "reranking_results.json", "w") as f:
            json.dump({
                "per_dataset": {
                    "SciFact": {
                        "aggregate": {"ndcg@10": 0.60},
                        "per_query": {"q1": 0.65, "q2": 0.55, "q3": 0.60},
                    }
                },
                "average": {"ndcg@10": 0.60},
            }, f)

        return tmp_path

    def test_load_all_results(self, mock_results_dir):
        results = load_all_results(str(mock_results_dir))
        assert "rank_only_r8" in results
        assert "joint_single_r8" in results
        assert 42 in results["rank_only_r8"]

    def test_compute_interference_metrics(self, mock_results_dir):
        results = load_all_results(str(mock_results_dir))
        metrics = compute_interference_metrics(results)

        # TIR = (0.65 - 0.60) / 0.65 ≈ 0.0769
        assert "reranking_tir" in metrics
        assert 42 in metrics["reranking_tir"]
        tir = metrics["reranking_tir"][42]
        assert abs(tir - 0.0769) < 0.01, f"Expected ~0.077, got {tir}"

    def test_kill_gate_verdict_pass(self, mock_results_dir):
        results = load_all_results(str(mock_results_dir))
        metrics = compute_interference_metrics(results)
        # TIR ~7.7% > 2% → PASS
        assert metrics["verdict"] == "PASS"

    def test_generate_comparison_table(self, mock_results_dir):
        results = load_all_results(str(mock_results_dir))
        table = generate_comparison_table(results)
        assert "rank_only_r8" in table
        assert "joint_single_r8" in table
        assert "nDCG@10" in table

    def test_compute_significance(self, mock_results_dir):
        results = load_all_results(str(mock_results_dir))
        sig = compute_significance(results)
        # Should have at least one comparison (rank_only vs joint)
        # With only 3 queries, it might skip (< 10 threshold)
        # This is expected behavior
        assert isinstance(sig, dict)

    def test_generate_plots_no_error(self, mock_results_dir):
        results = load_all_results(str(mock_results_dir))
        figures_dir = str(mock_results_dir / "figures")
        generate_plots(results, figures_dir)
        # Should create the figures directory even if some plots are empty
        assert Path(figures_dir).exists()

    def test_analysis_with_partial_results(self, tmp_path):
        """Analysis should work with only rank_only available (no joint)."""
        ronly = tmp_path / "rank_only_r8" / "seed_42" / "results"
        ronly.mkdir(parents=True)
        with open(ronly / "reranking_results.json", "w") as f:
            json.dump({"per_dataset": {}, "average": {"ndcg@10": 0.5}}, f)

        results = load_all_results(str(tmp_path))
        metrics = compute_interference_metrics(results)
        assert metrics["verdict"] == "INCOMPLETE"
        table = generate_comparison_table(results)
        assert "rank_only_r8" in table

    def test_query_id_collision_prevention(self, tmp_path):
        """Query IDs from different datasets should not overwrite each other."""
        # Create mock data where SciFact and FiQA2018 share query IDs (both integers)
        shared_pq = {str(i): 0.5 + i * 0.01 for i in range(20)}

        for config_name in ["rank_only_r8", "joint_single_r8"]:
            rd = tmp_path / config_name / "seed_42" / "results"
            rd.mkdir(parents=True)
            with open(rd / "reranking_results.json", "w") as f:
                json.dump({
                    "per_dataset": {
                        "SciFact": {
                            "aggregate": {"ndcg@10": 0.6},
                            "per_query": shared_pq,
                        },
                        "FiQA2018": {
                            "aggregate": {"ndcg@10": 0.55},
                            "per_query": shared_pq,
                        },
                    },
                    "average": {"ndcg@10": 0.575},
                }, f)

        results = load_all_results(str(tmp_path))
        sig = compute_significance(results)

        # With proper prefixing, we should have 40 queries (20 per dataset)
        for key, val in sig.items():
            if "reranking" in key:
                assert val["n_queries"] == 40, (
                    f"Expected 40 queries (20 per dataset), got {val['n_queries']}"
                )

    def test_embedding_significance_computed(self, tmp_path):
        """Significance should include embedding comparison when data is available."""
        for config_name in ["emb_only_r8", "joint_single_r8"]:
            for seed in [42, 123, 456]:
                rd = tmp_path / config_name / f"seed_{seed}" / "results"
                rd.mkdir(parents=True)
                with open(rd / "reranking_results.json", "w") as f:
                    json.dump({"per_dataset": {}, "average": {"ndcg@10": 0.5}}, f)
                with open(rd / "mteb_results.json", "w") as f:
                    json.dump({
                        "per_task": {
                            "SciFact": {"ndcg_at_10": 0.6 + seed * 0.001},
                            "NFCorpus": {"ndcg_at_10": 0.55 + seed * 0.001},
                            "FiQA2018": {"ndcg_at_10": 0.5 + seed * 0.001},
                        },
                        "eval_tier": "fast",
                    }, f)

        # Also add rank_only for completeness (avoids early return)
        for seed in [42, 123, 456]:
            rd = tmp_path / "rank_only_r8" / f"seed_{seed}" / "results"
            rd.mkdir(parents=True)
            with open(rd / "reranking_results.json", "w") as f:
                json.dump({"per_dataset": {}, "average": {"ndcg@10": 0.55}}, f)

        results = load_all_results(str(tmp_path))
        sig = compute_significance(results)
        assert "embedding_overall" in sig
        assert sig["embedding_overall"]["n_observations"] == 9  # 3 seeds x 3 tasks
