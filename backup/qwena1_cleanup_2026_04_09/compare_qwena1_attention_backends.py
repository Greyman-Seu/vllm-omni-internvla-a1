#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from qwena1_infer_common import (
    DEFAULT_DATASET_DIR,
    DEFAULT_MODEL_DIR,
    make_shared_noise,
    tensor_dtype,
    tensor_sha256,
)
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.qwena1 import (
    A2DOpenLoopDataset,
    QwenA1Config,
    QwenA1TrainMetadata,
    collate_open_loop_samples,
)
from vllm_omni.diffusion.models.qwena1.constants import OBS_STATE
from vllm_omni.diffusion.models.qwena1.dataset import unnormalize_vector
from vllm_omni.diffusion.models.qwena1.pipeline_qwena1 import QwenA1Pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two integrated QwenA1 attention backend variants.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--indices", nargs="*", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--strict-load", action="store_true")
    parser.add_argument("--baseline-attn", default="eager")
    parser.add_argument("--candidate-attn", default="sdpa")
    parser.add_argument("--baseline-regional-compile", action="store_true")
    parser.add_argument("--candidate-regional-compile", action="store_true")
    parser.add_argument("--output-json", default="compare_qwena1_attention_backends.json")
    return parser.parse_args()


def build_od_config(
    *,
    model_dir: Path,
    dtype_name: str,
    device: str,
    attn_implementation: str,
    enable_regional_compile: bool,
) -> OmniDiffusionConfig:
    return OmniDiffusionConfig(
        model=str(model_dir.resolve()),
        model_class_name="QwenA1Pipeline",
        dtype=tensor_dtype(dtype_name),
        custom_pipeline_args={
            "device": device,
            "dtype": dtype_name,
            "attn_implementation": attn_implementation,
            "enable_regional_compile": enable_regional_compile,
        },
    )


def build_dataset(
    *,
    model_dir: Path,
    dataset_dir: str,
    device: str,
    dtype_name: str,
    compile_model: bool,
) -> tuple[A2DOpenLoopDataset, QwenA1Config, QwenA1TrainMetadata]:
    config = QwenA1Config.from_pretrained(model_dir)
    config.device = device
    config.dtype = dtype_name
    config.compile_model = compile_model

    train_meta = QwenA1TrainMetadata.from_pretrained(model_dir)
    with open(model_dir / "stats.json", "r", encoding="utf-8") as f:
        train_stats = json.load(f)["a2d"]

    dataset = A2DOpenLoopDataset(
        dataset_dir,
        config=config,
        train_stats=train_stats,
        processor_model_name=train_meta.processor_model_name,
    )
    return dataset, config, train_meta


def select_indices(dataset: A2DOpenLoopDataset, requested: list[int] | None, num_samples: int) -> list[int]:
    if requested:
        return requested
    indices: list[int] = []
    for _, episode_indices in dataset.episode_start_indices(max_episodes=num_samples):
        if episode_indices:
            indices.append(episode_indices[0])
    return indices[:num_samples]


def run_pipeline_sample(
    *,
    pipeline: QwenA1Pipeline,
    batch_inputs: dict[str, torch.Tensor],
    noise: torch.Tensor,
    strict_load: bool,
    physical_action_dim: int,
) -> torch.Tensor:
    with torch.no_grad():
        pred, _ = pipeline.run_batch_sample_actions(
            batch_inputs,
            noise=noise,
            decode_image=False,
            load_weights=True,
            strict=strict_load,
        )
    return pred[:, :, :physical_action_dim].to(torch.float32).cpu()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    dataset, config, train_meta = build_dataset(
        model_dir=model_dir,
        dataset_dir=args.dataset_dir,
        device=args.device,
        dtype_name=args.dtype,
        compile_model=args.compile_model,
    )
    indices = select_indices(dataset, args.indices, args.num_samples)
    dtype = tensor_dtype(args.dtype)

    baseline_pipeline = QwenA1Pipeline(
        od_config=build_od_config(
            model_dir=model_dir,
            dtype_name=args.dtype,
            device=args.device,
            attn_implementation=args.baseline_attn,
            enable_regional_compile=args.baseline_regional_compile,
        )
    )
    candidate_pipeline = QwenA1Pipeline(
        od_config=build_od_config(
            model_dir=model_dir,
            dtype_name=args.dtype,
            device=args.device,
            attn_implementation=args.candidate_attn,
            enable_regional_compile=args.candidate_regional_compile,
        )
    )

    results: list[dict[str, Any]] = []
    for index in indices:
        sample = dataset.get_sample(index)
        batch_inputs, meta = collate_open_loop_samples([sample], device=args.device, dtype=dtype)
        shared_noise = make_shared_noise(
            args.seed,
            index,
            (
                batch_inputs[OBS_STATE].shape[0],
                config.chunk_size,
                config.max_action_dim,
            ),
            args.device,
        )

        baseline_pred = run_pipeline_sample(
            pipeline=baseline_pipeline,
            batch_inputs=batch_inputs,
            noise=shared_noise,
            strict_load=args.strict_load,
            physical_action_dim=dataset.physical_action_dim,
        )
        candidate_pred = run_pipeline_sample(
            pipeline=candidate_pipeline,
            batch_inputs=batch_inputs,
            noise=shared_noise,
            strict_load=args.strict_load,
            physical_action_dim=dataset.physical_action_dim,
        )

        raw_diff = (candidate_pred - baseline_pred).abs()
        baseline_phys = unnormalize_vector(baseline_pred, dataset.action_stats)
        candidate_phys = unnormalize_vector(candidate_pred, dataset.action_stats)
        if train_meta.action_mode == "delta":
            baseline_phys[:, :, :14] += meta["state_raw"][:, None, :14]
            candidate_phys[:, :, :14] += meta["state_raw"][:, None, :14]
        phys_diff = (candidate_phys - baseline_phys).abs()

        result = {
            "index": index,
            "episode_index": sample.episode_index,
            "task": sample.task,
            "baseline_attn": args.baseline_attn,
            "baseline_regional_compile": args.baseline_regional_compile,
            "candidate_attn": args.candidate_attn,
            "candidate_regional_compile": args.candidate_regional_compile,
            "match": tensor_sha256(baseline_pred) == tensor_sha256(candidate_pred),
            "baseline_sha256": tensor_sha256(baseline_pred),
            "candidate_sha256": tensor_sha256(candidate_pred),
            "raw_mean_abs_diff": float(raw_diff.mean().item()),
            "raw_max_abs_diff": float(raw_diff.max().item()),
            "physical_mean_abs_diff": float(phys_diff.mean().item()),
            "physical_max_abs_diff": float(phys_diff.max().item()),
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))

    output = {
        "model_dir": str(model_dir.resolve()),
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "device": args.device,
        "dtype": args.dtype,
        "seed": args.seed,
        "indices": indices,
        "baseline_attn": args.baseline_attn,
        "baseline_regional_compile": args.baseline_regional_compile,
        "candidate_attn": args.candidate_attn,
        "candidate_regional_compile": args.candidate_regional_compile,
        "results": results,
    }
    output_path = Path(args.output_json)
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
