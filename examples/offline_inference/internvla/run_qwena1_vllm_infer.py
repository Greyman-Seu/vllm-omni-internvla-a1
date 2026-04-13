#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from qwena1_eval_common import run_open_loop_evaluation
from qwena1_infer_common import (
    DEFAULT_DATASET_DIR,
    DEFAULT_MODEL_DIR,
    make_shared_noise,
    select_indices,
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
from vllm_omni.diffusion.models.qwena1.constants import OBS_PREFIX, OBS_STATE
from vllm_omni.diffusion.models.qwena1.pipeline_qwena1 import QwenA1Pipeline
from vllm_omni.diffusion.registry import initialize_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run vLLM QwenA1 inference on a few samples.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--num-episodes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--enable-regional-compile", action="store_true")
    parser.add_argument("--mode", choices=["direct", "registry", "both"], default="both")
    parser.add_argument("--strict-load", action="store_true")
    parser.add_argument("--output-dir", default="outputs/qwena1/vllm_infer")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def build_od_config(args: argparse.Namespace) -> OmniDiffusionConfig:
    return OmniDiffusionConfig(
        model=str(Path(args.model_dir).resolve()),
        model_class_name="QwenA1Pipeline",
        dtype=tensor_dtype(args.dtype),
        custom_pipeline_args={
            "device": args.device,
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
            "enable_regional_compile": args.enable_regional_compile,
        },
    )


def build_dataset(args: argparse.Namespace) -> tuple[A2DOpenLoopDataset, QwenA1Config, QwenA1TrainMetadata]:
    model_dir = Path(args.model_dir)
    config = QwenA1Config.from_pretrained(model_dir)
    config.device = args.device
    config.dtype = args.dtype
    config.compile_model = args.compile_model
    if args.attn_implementation:
        config.attn_implementation = args.attn_implementation
    if args.enable_regional_compile:
        config.enable_regional_compile = True

    train_meta = QwenA1TrainMetadata.from_pretrained(model_dir)
    with open(model_dir / "stats.json", "r", encoding="utf-8") as f:
        train_stats = json.load(f)["a2d"]

    dataset = A2DOpenLoopDataset(
        args.dataset_dir,
        config=config,
        train_stats=train_stats,
        processor_model_name=train_meta.processor_model_name,
    )
    return dataset, config, train_meta


def run_one_path(
    *,
    label: str,
    pipeline: QwenA1Pipeline,
    dataset: A2DOpenLoopDataset,
    config: QwenA1Config,
    args: argparse.Namespace,
    indices: list[int],
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for index in indices:
        sample = dataset.get_sample(index)
        batch_inputs, _ = collate_open_loop_samples([sample], device=args.device, dtype=tensor_dtype(args.dtype))
        noise = make_shared_noise(
            args.seed,
            index,
            (
                batch_inputs[OBS_STATE].shape[0],
                config.chunk_size,
                config.max_action_dim,
            ),
            args.device,
        )
        with torch.no_grad():
            pred, _ = pipeline.run_batch_sample_actions(
                batch_inputs,
                noise=noise,
                decode_image=False,
                load_weights=True,
                strict=args.strict_load,
            )
        pred = pred[:, :, :dataset.physical_action_dim].to(torch.float32).cpu()
        results.append(
            {
                "path": label,
                "index": index,
                "episode_index": sample.episode_index,
                "task": sample.task,
                "seed": args.seed,
                "shape": list(pred.shape),
                "mean": float(pred.mean().item()),
                "std": float(pred.std().item()),
                "action_sha256": tensor_sha256(pred),
                "first_action_prefix": pred[0, 0, :8].tolist(),
            }
        )
    return results


def run_pipeline_sample_actions(
    pipeline: QwenA1Pipeline,
    batch_inputs: dict[str, torch.Tensor],
    noise: torch.Tensor,
    *,
    strict_load: bool,
) -> torch.Tensor:
    pred, _ = pipeline.run_batch_sample_actions(
        batch_inputs,
        noise=noise,
        decode_image=False,
        load_weights=True,
        strict=strict_load,
    )
    return pred


def main() -> None:
    args = parse_args()
    dataset, config, train_meta = build_dataset(args)
    indices = select_indices(dataset, args.num_samples)
    od_config = build_od_config(args)

    results: list[dict[str, object]] = []
    direct_results: list[dict[str, object]] = []
    registry_results: list[dict[str, object]] = []
    eval_summaries: dict[str, object] = {}

    if args.mode in {"direct", "both"}:
        direct_pipeline = QwenA1Pipeline(od_config=od_config)
        direct_results = run_one_path(
            label="direct",
            pipeline=direct_pipeline,
            dataset=dataset,
            config=config,
            args=args,
            indices=indices,
        )
        results.extend(direct_results)
        if args.num_episodes > 0:
            eval_summaries["direct"] = run_open_loop_evaluation(
                mode="vllm_direct",
                policy=direct_pipeline,
                config=config,
                dataset=dataset,
                train_meta=train_meta,
                collate_samples=collate_open_loop_samples,
                run_sample_actions=lambda policy, batch_inputs, noise: run_pipeline_sample_actions(
                    policy,
                    batch_inputs,
                    noise,
                    strict_load=args.strict_load,
                ),
                make_shared_noise=make_shared_noise,
                num_episodes=args.num_episodes,
                seed=args.seed,
                device=args.device,
                dtype=tensor_dtype(args.dtype),
                output_dir=Path(args.output_dir) / "direct",
                skip_plots=args.skip_plots,
            )

    if args.mode in {"registry", "both"}:
        registry_pipeline = initialize_model(od_config)
        if not isinstance(registry_pipeline, QwenA1Pipeline):
            raise TypeError(f"Expected QwenA1Pipeline, got {type(registry_pipeline)!r}")
        registry_results = run_one_path(
            label="registry",
            pipeline=registry_pipeline,
            dataset=dataset,
            config=config,
            args=args,
            indices=indices,
        )
        results.extend(registry_results)
        if args.num_episodes > 0:
            eval_summaries["registry"] = run_open_loop_evaluation(
                mode="vllm_registry",
                policy=registry_pipeline,
                config=config,
                dataset=dataset,
                train_meta=train_meta,
                collate_samples=collate_open_loop_samples,
                run_sample_actions=lambda policy, batch_inputs, noise: run_pipeline_sample_actions(
                    policy,
                    batch_inputs,
                    noise,
                    strict_load=args.strict_load,
                ),
                make_shared_noise=make_shared_noise,
                num_episodes=args.num_episodes,
                seed=args.seed,
                device=args.device,
                dtype=tensor_dtype(args.dtype),
                output_dir=Path(args.output_dir) / "registry",
                skip_plots=args.skip_plots,
            )

    comparisons: list[dict[str, object]] = []
    if direct_results and registry_results:
        for direct_item, registry_item in zip(direct_results, registry_results, strict=True):
            comparisons.append(
                {
                    "index": direct_item["index"],
                    "match": direct_item["action_sha256"] == registry_item["action_sha256"],
                    "direct_sha256": direct_item["action_sha256"],
                    "registry_sha256": registry_item["action_sha256"],
                }
            )

    print(
        json.dumps(
            {
                "mode": args.mode,
                "model_dir": str(Path(args.model_dir).resolve()),
                "dataset_dir": str(Path(args.dataset_dir).resolve()),
                "device": args.device,
                "dtype": args.dtype,
                "attn_implementation": args.attn_implementation,
                "enable_regional_compile": args.enable_regional_compile,
                "seed": args.seed,
                "indices": indices,
                "results": results,
                "comparisons": comparisons,
                "output_dir": str(Path(args.output_dir)),
                "eval_summaries": eval_summaries,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
