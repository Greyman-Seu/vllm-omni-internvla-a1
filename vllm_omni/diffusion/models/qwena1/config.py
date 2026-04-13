from __future__ import annotations

import json
from dataclasses import dataclass, field, fields as dataclass_fields
from pathlib import Path
from typing import Any

from .constants import DEFAULT_QWEN3_VL_MODEL


@dataclass
class QwenA1Config:
    """Standalone-compatible QwenA1 config with a few fake-smoke defaults."""

    type: str = "qwena1"
    qwen3_vl_variant: str = "qwen3_vl_28l"
    action_expert_variant: str = "qwen3_28l"
    dtype: str = "bfloat16"
    device: str = "cuda"
    chunk_size: int = 50
    n_action_steps: int = 50
    max_state_dim: int = 32
    max_action_dim: int = 32
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0
    image_resolution: tuple[int, int] = (224, 224)
    empty_cameras: int = 0
    gradient_checkpointing: bool = False
    compile_model: bool = False
    compile_mode: str = "max-autotune"
    enable_regional_compile: bool = False
    regional_compile_dynamic: bool = True
    enable_suffix_static_context_optimization: bool = False
    attn_implementation: str = "eager"
    tokenizer_max_length: int = 48
    freeze_vision_encoder: bool = False
    train_expert_only: bool = False
    train_vlm_only: bool = False
    scale_factor: int = 8
    lambda_gen: float = 0.01
    input_features: dict[str, Any] = field(default_factory=dict)
    output_features: dict[str, Any] = field(default_factory=dict)

    # Fake/smoke-only knobs used by the lightweight pipeline path.
    vocab_size: int = 256
    hidden_size: int = 128
    intermediate_size: int = 256
    num_attention_heads: int = 4
    num_hidden_layers: int = 2
    num_cameras: int = 3
    image_history: int = 2
    pixel_feature_dim: int = 48

    @classmethod
    def from_pretrained(cls, checkpoint_dir: str | Path) -> "QwenA1Config":
        checkpoint_dir = Path(checkpoint_dir)
        with open(checkpoint_dir / "config.json", "r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls.from_model_config(raw)

    @classmethod
    def from_model_config(cls, model_config: dict[str, Any] | None) -> "QwenA1Config":
        if not model_config:
            return cls()

        raw = dict(model_config)
        raw["image_resolution"] = tuple(raw.get("image_resolution", [224, 224]))
        allowed = {item.name for item in dataclass_fields(cls)}
        filtered = {key: value for key, value in raw.items() if key in allowed}
        return cls(**filtered)


@dataclass
class QwenA1TrainMetadata:
    action_mode: str = "delta"
    processor_model_name: str = DEFAULT_QWEN3_VL_MODEL

    @classmethod
    def from_pretrained(cls, checkpoint_dir: str | Path) -> "QwenA1TrainMetadata":
        checkpoint_dir = Path(checkpoint_dir)
        path = checkpoint_dir / "train_config.json"
        if not path.exists():
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        dataset_cfg = raw.get("dataset", {})
        processor_model_name = DEFAULT_QWEN3_VL_MODEL
        for transform in dataset_cfg.get("data_transforms", {}).get("inputs", []):
            if transform.get("type") == "qwena1_processor":
                processor_model_name = transform.get(
                    "pretrained_model_name_or_path",
                    processor_model_name,
                )
                break

        return cls(
            action_mode=dataset_cfg.get("action_mode", "delta"),
            processor_model_name=processor_model_name,
        )
