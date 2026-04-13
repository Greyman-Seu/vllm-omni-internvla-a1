#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from typing import Any

import torch

from qwena1_infer_common import make_fake_batch_inputs, make_shared_noise, summarize_outputs
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.qwena1.pipeline_qwena1 import QwenA1Pipeline
from vllm_omni.diffusion.registry import initialize_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run QwenA1 with fake inputs in this repository."
    )
    parser.add_argument("--mode", choices=["direct", "registry", "both"], default="both")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--weight-seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--enable-regional-compile", action="store_true")
    parser.add_argument("--decode-image", action="store_true")
    return parser.parse_args()


def build_od_config(args: argparse.Namespace) -> OmniDiffusionConfig:
    return OmniDiffusionConfig(
        model=None,
        model_class_name="QwenA1Pipeline",
        dtype=getattr(torch, args.dtype),
        model_config={
            "device": args.device,
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation or "eager",
            "enable_regional_compile": args.enable_regional_compile,
        },
    )


def summarize(label: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": label,
        "mode": result["mode"],
        "summary": result["summary"],
    }


def run_one_path(pipeline: QwenA1Pipeline, args: argparse.Namespace) -> dict[str, Any]:
    policy = pipeline.get_or_create_policy(load_weights=False)
    batch_inputs = make_fake_batch_inputs(
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        dtype=getattr(torch, args.dtype),
        vision_start_token_id=policy.model.qwen3_vl_with_expert.und_expert.config.vision_start_token_id,
        image_token_id=policy.model.qwen3_vl_with_expert.und_expert.config.image_token_id,
        vision_end_token_id=policy.model.qwen3_vl_with_expert.und_expert.config.vision_end_token_id,
        tokenizer_max_length=pipeline.config.tokenizer_max_length,
        num_cameras=pipeline.config.num_cameras,
        image_history=pipeline.config.image_history,
        image_resolution=pipeline.config.image_resolution,
        max_state_dim=pipeline.config.max_state_dim,
        visual_patch_size=policy.model.qwen3_vl_with_expert.und_expert.visual.config.patch_size,
        visual_temporal_patch_size=policy.model.qwen3_vl_with_expert.und_expert.visual.config.temporal_patch_size,
        visual_in_channels=policy.model.qwen3_vl_with_expert.und_expert.visual.config.in_channels,
        visual_spatial_merge_size=policy.model.qwen3_vl_with_expert.und_expert.visual.config.spatial_merge_size,
    )
    noise = make_shared_noise(
        args.seed,
        0,
        (args.batch_size, pipeline.config.chunk_size, pipeline.config.max_action_dim),
        args.device,
    ).to(dtype=getattr(torch, args.dtype))
    actions, decoded = pipeline.run_batch_sample_actions(
        batch_inputs,
        noise=noise,
        decode_image=args.decode_image,
        load_weights=False,
    )
    return {
        "mode": pipeline.runtime_mode(),
        "summary": summarize_outputs(mode=pipeline.runtime_mode(), actions=actions, decoded=decoded),
    }


def run_direct(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.weight_seed)
    pipeline = QwenA1Pipeline(od_config=build_od_config(args))
    return run_one_path(pipeline, args)


def run_registry(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.weight_seed)
    pipeline = initialize_model(build_od_config(args))
    if not isinstance(pipeline, QwenA1Pipeline):
        raise TypeError(f"Expected QwenA1Pipeline, got {type(pipeline)!r}")
    return run_one_path(pipeline, args)


def main() -> None:
    args = parse_args()
    outputs: list[dict[str, Any]] = []
    direct_result: dict[str, Any] | None = None
    registry_result: dict[str, Any] | None = None

    if args.mode in {"direct", "both"}:
        direct_result = run_direct(args)
        outputs.append(summarize("direct", direct_result))

    if args.mode in {"registry", "both"}:
        registry_result = run_registry(args)
        outputs.append(summarize("registry", registry_result))

    if direct_result is not None and registry_result is not None:
        outputs.append(
            {
                "path": "comparison",
                "match": direct_result["summary"]["sha256"] == registry_result["summary"]["sha256"],
                "direct_sha256": direct_result["summary"]["sha256"],
                "registry_sha256": registry_result["summary"]["sha256"],
            }
        )

    print(
        json.dumps(
            {
                "mode": args.mode,
                "device": args.device,
                "dtype": args.dtype,
                "attn_implementation": args.attn_implementation,
                "enable_regional_compile": args.enable_regional_compile,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "weight_seed": args.weight_seed,
                "results": outputs,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
