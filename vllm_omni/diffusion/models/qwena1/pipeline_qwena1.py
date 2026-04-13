# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import json
import os
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.compile import regionally_compile
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import (
    DiffusionPipelineProfilerMixin,
    wrap_methods_by_paths,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest

from .config import QwenA1Config
from .constants import OBS_PREFIX
from .model import StandaloneQwenA1Policy

logger = init_logger(__name__)


def get_qwena1_post_process_func(od_config: OmniDiffusionConfig):
    del od_config

    def post_process_func(x):
        return x

    return post_process_func


class QwenA1Pipeline(nn.Module, DiffusionPipelineProfilerMixin):
    """QwenA1 pipeline wrapper for the standalone policy implementation."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        del prefix
        self.od_config = od_config
        self.model_dir = od_config.model
        self.config = self._build_config(od_config)
        self.weights_sources: list[object] = []

        self._policy: StandaloneQwenA1Policy | None = None
        self._policy_has_loaded_weights = False
        self._policy_optimizations_applied = False
        self._policy_profiler_targets_wrapped = False
        self.setup_diffusion_pipeline_profiler(
            profiler_targets=["run_batch_sample_actions"],
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler,
        )

    def _build_config(self, od_config: OmniDiffusionConfig) -> QwenA1Config:
        config_dict = self._load_config_dict(od_config)
        config = QwenA1Config.from_model_config(config_dict)

        custom_args = od_config.custom_pipeline_args or {}
        device = custom_args.get("device")
        if isinstance(device, str):
            config.device = device

        dtype = custom_args.get("dtype")
        if isinstance(dtype, str):
            config.dtype = dtype
        elif od_config.dtype is not None:
            config.dtype = str(od_config.dtype).split(".")[-1]

        attn_implementation = custom_args.get("attn_implementation")
        if isinstance(attn_implementation, str):
            config.attn_implementation = attn_implementation

        enable_regional_compile = custom_args.get("enable_regional_compile")
        if isinstance(enable_regional_compile, bool):
            config.enable_regional_compile = enable_regional_compile

        regional_compile_dynamic = custom_args.get("regional_compile_dynamic")
        if isinstance(regional_compile_dynamic, bool):
            config.regional_compile_dynamic = regional_compile_dynamic

        return config

    def _load_config_dict(self, od_config: OmniDiffusionConfig) -> dict[str, Any]:
        if od_config.model_config:
            return dict(od_config.model_config)

        model_path = od_config.model
        if not model_path:
            return {}

        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            logger.info("QwenA1Pipeline config.json not found under %s; using defaults.", model_path)
            return {}

        with open(config_path, encoding="utf-8") as f:
            return json.load(f)

    def has_real_checkpoint(self) -> bool:
        return bool(self.model_dir) and os.path.exists(os.path.join(self.model_dir, "model.safetensors"))

    def runtime_mode(self) -> str:
        if self._policy is not None:
            return "real_checkpoint_loaded" if self._policy_has_loaded_weights else "real_unloaded_policy"
        if self.has_real_checkpoint():
            return "real_checkpoint_available"
        return "no_checkpoint_policy"

    def _setup_policy_profiler_targets(self) -> None:
        if not self.od_config.enable_diffusion_pipeline_profiler or self._policy_profiler_targets_wrapped:
            return
        if self._policy is None:
            return

        wrap_methods_by_paths(
            self,
            [
                "_policy.model.sample_actions",
                "_policy.model.embed_prefix",
                "_policy.model.embed_middle",
                "_policy.model.denoise_step",
            ],
        )
        self._policy_profiler_targets_wrapped = True

    def _apply_policy_optimizations(self, policy: StandaloneQwenA1Policy) -> None:
        policy.model.set_attention_implementation(self.config.attn_implementation)
        self._setup_policy_profiler_targets()

        if not self.config.enable_regional_compile or self._policy_optimizations_applied:
            self._policy_optimizations_applied = True
            return

        compile_targets = [
            "qwen3_vl_with_expert.und_expert.visual",
            "qwen3_vl_with_expert.und_expert.language_model",
            "qwen3_vl_with_expert.gen_expert",
            "qwen3_vl_with_expert.act_expert",
        ]

        for path in compile_targets:
            current = policy.model
            for part in path.split("."):
                current = getattr(current, part, None)
                if current is None:
                    break
            if current is None:
                continue
            try:
                regionally_compile(current, dynamic=self.config.regional_compile_dynamic)
                logger.info("QwenA1Pipeline regional compile applied to %s", path)
            except Exception as exc:
                logger.warning("QwenA1Pipeline regional compile failed for %s: %s", path, exc)

        self._policy_optimizations_applied = True

    def get_or_create_policy(
        self,
        *,
        load_weights: bool = True,
        strict: bool = False,
    ) -> StandaloneQwenA1Policy:
        if self._policy is not None and (not load_weights or self._policy_has_loaded_weights):
            return self._policy

        if load_weights and self.has_real_checkpoint():
            logger.info("Loading QwenA1 standalone weights from %s", self.model_dir)
            policy = StandaloneQwenA1Policy.from_pretrained(self.model_dir, config=self.config, strict=strict)
            self._policy_has_loaded_weights = True
        else:
            logger.info("Initializing QwenA1 standalone policy without checkpoint weights.")
            policy = StandaloneQwenA1Policy(self.config)
            self._policy_has_loaded_weights = False

        policy.to(self.config.device)
        policy.to(getattr(torch, self.config.dtype))
        policy.eval()
        if self._policy_has_loaded_weights or not self.has_real_checkpoint():
            self._apply_policy_optimizations(policy)
        self._policy = policy
        return policy

    @torch.no_grad()
    def run_batch_sample_actions(
        self,
        batch_inputs: dict[str, torch.Tensor],
        *,
        noise: torch.Tensor | None = None,
        decode_image: bool = False,
        load_weights: bool = True,
        strict: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        policy = self.get_or_create_policy(load_weights=load_weights, strict=strict)
        logger.debug("QwenA1Pipeline run_batch_sample_actions mode=%s", self.runtime_mode())
        pixel_values = batch_inputs[f"{OBS_PREFIX}pixel_values"]
        image_grid_thw = batch_inputs[f"{OBS_PREFIX}image_grid_thw"]
        lang_tokens = batch_inputs[f"{OBS_PREFIX}input_ids"]
        lang_masks = batch_inputs[f"{OBS_PREFIX}attention_mask"]
        state = policy.prepare_state(batch_inputs)
        images, img_masks = policy._preprocess_images(batch_inputs)
        return policy.model.sample_actions(
            images,
            img_masks,
            pixel_values,
            image_grid_thw,
            lang_tokens,
            lang_masks,
            state,
            noise=noise,
            decode_image=decode_image,
        )

    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        return DiffusionOutput(
            error=(
                "QwenA1Pipeline expects direct calls to `run_batch_sample_actions(...)` "
                "from repo-side batch inputs. Fake inputs should be constructed in the "
                "entry scripts, not inside the pipeline."
            ),
            post_process_func=get_qwena1_post_process_func(self.od_config),
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        target = self.get_or_create_policy(load_weights=False)

        state = target.state_dict()
        allowed = set(state.keys())
        shapes = {key: tuple(value.shape) for key, value in state.items()}

        def _normalize_name(name: str) -> str:
            for prefix in ("module.", "pipeline.", "policy."):
                if name.startswith(prefix):
                    name = name[len(prefix) :]
            return name

        def _iter_candidate_names(name: str) -> Iterable[str]:
            normalized = _normalize_name(name)
            yield normalized
            if not normalized.startswith("model."):
                yield f"model.{normalized}"
            if normalized.startswith("model."):
                yield normalized[len("model.") :]

        def _filtered_weights():
            total = 0
            kept = 0
            shape_mismatch = 0
            for name, tensor in weights:
                total += 1
                for candidate in _iter_candidate_names(name):
                    if candidate in allowed:
                        if tuple(tensor.shape) == shapes.get(candidate):
                            kept += 1
                            yield candidate, tensor
                            break
                        shape_mismatch += 1
            logger.info_once(
                "QwenA1Pipeline weight filter kept %d/%d tensors (shape mismatches seen: %d)",
                kept,
                total,
                shape_mismatch,
            )

        loader = AutoWeightsLoader(target)
        loaded = loader.load_weights(_filtered_weights())
        if target is self._policy:
            self._policy_has_loaded_weights = True
            self._apply_policy_optimizations(self._policy)
        return loaded
