"""Tests for the configuration system."""

from pathlib import Path

import dacite
import pytest
import torch
import yaml

from unimoe.config import (
    ExperimentConfig,
    LoRAConfig,
    ModelConfig,
    ScoringMode,
    TrainingConfig,
    TrainingMode,
    auto_detect_device,
    dtype_for_device,
    load_config,
    resolve_torch_dtype,
    save_config,
)

CONFIGS_DIR = Path(__file__).parent.parent / "configs"

# All 6 YAML config files
CONFIG_FILES = [
    "emb_only_r16.yaml",
    "rank_only_r16.yaml",
    "joint_single_r16.yaml",
    "emb_only_r8.yaml",
    "rank_only_r8.yaml",
    "joint_single_r8.yaml",
]


class TestTrainingMode:
    def test_values(self):
        assert TrainingMode.RANK_ONLY.value == "rank_only"
        assert TrainingMode.EMB_ONLY.value == "emb_only"
        assert TrainingMode.JOINT_SINGLE.value == "joint_single"

    def test_all_modes_present(self):
        assert len(TrainingMode) == 3


class TestScoringMode:
    def test_values(self):
        assert ScoringMode.YES_NO_LOGITS.value == "yes_no_logits"

    def test_all_modes_present(self):
        assert len(ScoringMode) == 1


class TestDeviceAutoDetection:
    def test_auto_detect_returns_valid_device(self):
        device = auto_detect_device()
        assert device in ("cuda", "mps", "cpu")

    def test_dtype_for_cuda(self):
        assert dtype_for_device("cuda") == "bfloat16"

    def test_dtype_for_mps(self):
        assert dtype_for_device("mps") == "float16"

    def test_dtype_for_cpu(self):
        assert dtype_for_device("cpu") == "float32"

    def test_resolve_torch_dtype_bfloat16(self):
        assert resolve_torch_dtype("bfloat16") is torch.bfloat16

    def test_resolve_torch_dtype_float16(self):
        assert resolve_torch_dtype("float16") is torch.float16

    def test_resolve_torch_dtype_float32(self):
        assert resolve_torch_dtype("float32") is torch.float32

    def test_resolve_torch_dtype_invalid(self):
        with pytest.raises(ValueError, match="Unsupported dtype"):
            resolve_torch_dtype("int8")


class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig()
        assert cfg.base_model_name == "Qwen/Qwen3-0.6B-Base"
        assert cfg.torch_dtype == "auto"
        assert cfg.device == "auto"

    def test_resolve_device_auto(self):
        cfg = ModelConfig(device="auto")
        device = cfg.resolve_device()
        assert device in ("cuda", "mps", "cpu")

    def test_resolve_device_explicit(self):
        cfg = ModelConfig(device="cpu")
        assert cfg.resolve_device() == "cpu"

    def test_resolve_dtype_auto(self):
        cfg = ModelConfig(torch_dtype="auto")
        dtype = cfg.resolve_dtype()
        assert dtype in (torch.bfloat16, torch.float16, torch.float32)

    def test_resolve_dtype_explicit(self):
        cfg = ModelConfig(torch_dtype="float16")
        assert cfg.resolve_dtype() is torch.float16


class TestLoRAConfig:
    def test_defaults(self):
        cfg = LoRAConfig()
        assert cfg.rank == 16
        assert cfg.alpha == 16
        assert cfg.dropout == 0.05
        assert cfg.bias == "none"
        assert cfg.task_type is None
        assert len(cfg.target_modules) == 7
        assert "q_proj" in cfg.target_modules
        assert "gate_proj" in cfg.target_modules
        assert "down_proj" in cfg.target_modules

    def test_all_linear_targets(self):
        cfg = LoRAConfig()
        expected = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
        assert set(cfg.target_modules) == expected


class TestTrainingConfig:
    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.mode == TrainingMode.JOINT_SINGLE
        assert cfg.lr == 1e-4
        assert cfg.warmup_ratio == 0.05
        assert cfg.epochs == 1
        assert cfg.batch_size_embedding == 8
        assert cfg.batch_size_reranking == 16
        assert cfg.grad_accum_steps == 4
        assert cfg.max_grad_norm == 1.0
        assert cfg.optimizer == "adamw"
        assert cfg.temperature == 0.05
        assert cfg.scoring_mode == ScoringMode.YES_NO_LOGITS


class TestExperimentConfig:
    def test_defaults(self):
        cfg = ExperimentConfig()
        assert isinstance(cfg.model, ModelConfig)
        assert isinstance(cfg.lora, LoRAConfig)
        assert isinstance(cfg.training, TrainingConfig)
        assert cfg.seed == 42
        assert cfg.wandb_enabled is False


class TestLoadConfig:
    @pytest.mark.parametrize("config_file", CONFIG_FILES)
    def test_load_all_yaml_configs(self, config_file):
        path = CONFIGS_DIR / config_file
        cfg = load_config(path)
        assert isinstance(cfg, ExperimentConfig)
        assert isinstance(cfg.training.mode, TrainingMode)
        assert isinstance(cfg.training.scoring_mode, ScoringMode)

    @pytest.mark.parametrize("config_file", CONFIG_FILES)
    def test_config_field_types(self, config_file):
        path = CONFIGS_DIR / config_file
        cfg = load_config(path)
        assert isinstance(cfg.seed, int)
        assert isinstance(cfg.output_dir, str)
        assert isinstance(cfg.lora.rank, int)
        assert isinstance(cfg.lora.alpha, int)
        assert isinstance(cfg.lora.dropout, float)
        assert isinstance(cfg.lora.target_modules, list)
        assert isinstance(cfg.training.lr, float)
        assert isinstance(cfg.training.temperature, float)
        assert isinstance(cfg.eval.beir_datasets, list)

    def test_emb_only_mode(self):
        cfg = load_config(CONFIGS_DIR / "emb_only_r16.yaml")
        assert cfg.training.mode == TrainingMode.EMB_ONLY

    def test_rank_only_mode(self):
        cfg = load_config(CONFIGS_DIR / "rank_only_r16.yaml")
        assert cfg.training.mode == TrainingMode.RANK_ONLY

    def test_joint_single_mode(self):
        cfg = load_config(CONFIGS_DIR / "joint_single_r16.yaml")
        assert cfg.training.mode == TrainingMode.JOINT_SINGLE

    def test_r8_configs_have_rank_8(self):
        for name in ["emb_only_r8.yaml", "rank_only_r8.yaml", "joint_single_r8.yaml"]:
            cfg = load_config(CONFIGS_DIR / name)
            assert cfg.lora.rank == 8
            assert cfg.lora.alpha == 8

    def test_r16_configs_have_rank_16(self):
        for name in ["emb_only_r16.yaml", "rank_only_r16.yaml", "joint_single_r16.yaml"]:
            cfg = load_config(CONFIGS_DIR / name)
            assert cfg.lora.rank == 16
            assert cfg.lora.alpha == 16


class TestRoundTripSerialization:
    @pytest.mark.parametrize("config_file", CONFIG_FILES)
    def test_save_and_reload(self, config_file, tmp_path):
        original = load_config(CONFIGS_DIR / config_file)
        save_path = tmp_path / "round_trip.yaml"
        save_config(original, save_path)
        reloaded = load_config(save_path)

        assert reloaded.seed == original.seed
        assert reloaded.output_dir == original.output_dir
        assert reloaded.wandb_enabled == original.wandb_enabled
        assert reloaded.training.mode == original.training.mode
        assert reloaded.training.scoring_mode == original.training.scoring_mode
        assert reloaded.lora.rank == original.lora.rank
        assert reloaded.lora.alpha == original.lora.alpha
        assert reloaded.lora.target_modules == original.lora.target_modules
        assert reloaded.model.base_model_name == original.model.base_model_name
        assert reloaded.eval.beir_datasets == original.eval.beir_datasets
        assert reloaded.data.num_hard_negatives == original.data.num_hard_negatives

    def test_saved_yaml_is_valid(self, tmp_path):
        cfg = ExperimentConfig()
        save_path = tmp_path / "test.yaml"
        save_config(cfg, save_path)

        with open(save_path) as f:
            raw = yaml.safe_load(f)
        assert isinstance(raw, dict)
        assert "model" in raw
        assert "training" in raw
        assert raw["training"]["mode"] == "joint_single"


class TestConfigEdgeCases:
    def test_load_empty_yaml(self, tmp_path):
        """An empty YAML file should produce default config."""
        p = tmp_path / "empty.yaml"
        p.write_text("")
        cfg = load_config(p)
        assert isinstance(cfg, ExperimentConfig)
        assert cfg.seed == 42

    def test_load_partial_yaml(self, tmp_path):
        """A YAML with only some fields should fill defaults for the rest."""
        p = tmp_path / "partial.yaml"
        p.write_text("seed: 99\n")
        cfg = load_config(p)
        assert cfg.seed == 99
        assert cfg.lora.rank == 16  # default

    def test_strict_mode_rejects_unknown_keys(self, tmp_path):
        """Unknown keys should be rejected by dacite strict mode."""
        p = tmp_path / "bad.yaml"
        p.write_text("seed: 42\nunknown_field: true\n")
        with pytest.raises(dacite.UnexpectedDataError):
            load_config(p)

    def test_device_auto_detection_consistency(self):
        """Device auto-detection should be deterministic."""
        d1 = auto_detect_device()
        d2 = auto_detect_device()
        assert d1 == d2
