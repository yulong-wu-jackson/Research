"""Configuration system for UniMoE experiments.

Dataclass-based configuration with YAML serialization via pyyaml + dacite.
Supports device auto-detection (CUDA > MPS > CPU) with appropriate dtype selection.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import dacite
import torch
import yaml


class TrainingMode(Enum):
    """Training mode for the unified model."""

    RANK_ONLY = "rank_only"
    EMB_ONLY = "emb_only"
    JOINT_SINGLE = "joint_single"


class ScoringMode(Enum):
    """Reranking scoring mode."""

    YES_NO_LOGITS = "yes_no_logits"


def auto_detect_device() -> str:
    """Auto-detect best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def dtype_for_device(device: str) -> str:
    """Select appropriate dtype string for the device.

    CUDA supports bfloat16 natively. MPS does not support bfloat16,
    so we use float16 instead. CPU falls back to float32.
    """
    if device == "cuda":
        return "bfloat16"
    if device == "mps":
        return "float16"
    return "float32"


def resolve_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert a dtype string to a torch.dtype."""
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_str}. Choose from {list(mapping.keys())}")
    return mapping[dtype_str]


@dataclass
class ModelConfig:
    """Configuration for the base model and LoRA."""

    base_model_name: str = "Qwen/Qwen3-0.6B-Base"
    torch_dtype: str = "auto"
    device: str = "auto"

    def resolve_device(self) -> str:
        """Resolve 'auto' device to actual device string."""
        if self.device == "auto":
            return auto_detect_device()
        return self.device

    def resolve_dtype(self) -> torch.dtype:
        """Resolve 'auto' dtype based on device."""
        if self.torch_dtype == "auto":
            return resolve_torch_dtype(dtype_for_device(self.resolve_device()))
        return resolve_torch_dtype(self.torch_dtype)


@dataclass
class LoRAConfig:
    """Configuration for LoRA adapters."""

    rank: int = 16
    alpha: int = 16
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    task_type: Optional[str] = None
    bias: str = "none"


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    dataset_name: str = "sentence-transformers/msmarco"
    embedding_samples: Optional[int] = None
    reranking_samples: Optional[int] = None
    query_max_len: int = 128
    passage_max_len: int = 256
    reranking_max_len: int = 512
    num_hard_negatives: int = 7
    instruction_prefix: str = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )


@dataclass
class TrainingConfig:
    """Configuration for the training loop."""

    mode: TrainingMode = TrainingMode.JOINT_SINGLE
    lr: float = 1e-4
    warmup_ratio: float = 0.05
    epochs: int = 1
    batch_size_embedding: int = 8
    batch_size_reranking: int = 16
    grad_accum_steps: int = 4
    max_grad_norm: float = 1.0
    optimizer: str = "adamw"
    temperature: float = 0.05
    reranking_loss_weight: float = 1.0
    scoring_mode: ScoringMode = ScoringMode.YES_NO_LOGITS
    gradient_conflict_every_n_steps: int = 100


@dataclass
class EvalConfig:
    """Configuration for evaluation."""

    eval_tier: str = "fast"
    beir_datasets: list[str] = field(
        default_factory=lambda: ["SciFact", "NFCorpus", "FiQA2018"]
    )
    eval_batch_size: int = 64


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration combining all sub-configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    seed: int = 42
    output_dir: str = "outputs"
    wandb_enabled: bool = False
    experiment_name: str = "default"


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an ExperimentConfig from a YAML file.

    Uses pyyaml for parsing and dacite for dataclass conversion,
    supporting nested dataclasses and enum types.
    """
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raw = {}

    return dacite.from_dict(
        data_class=ExperimentConfig,
        data=raw,
        config=dacite.Config(
            cast=[TrainingMode, ScoringMode],
            strict=True,
        ),
    )


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    """Save an ExperimentConfig to a YAML file for reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = _config_to_dict(config)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def _config_to_dict(obj: object) -> object:
    """Recursively convert a dataclass to a plain dict, handling enums."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        result = {}
        for f in dataclasses.fields(obj):
            value = getattr(obj, f.name)
            result[f.name] = _config_to_dict(value)
        return result
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, list):
        return [_config_to_dict(item) for item in obj]
    return obj
