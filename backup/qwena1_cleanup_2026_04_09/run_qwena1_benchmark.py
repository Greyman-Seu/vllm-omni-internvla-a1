#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

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
from vllm_omni.diffusion.models.qwena1.constants import OBS_STATE
from vllm_omni.diffusion.models.qwena1.pipeline_qwena1 import QwenA1Pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark QwenA1 integrated inference variants.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--benchmark-iters", type=int, default=5)
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--strict-load", action="store_true")
    parser.add_argument(
        "--variants",
        nargs="*",
        default=["eager", "sdpa", "eager_regional", "sdpa_regional"],
        choices=["eager", "sdpa", "eager_regional", "sdpa_regional"],
    )
    parser.add_argument("--output-json", default="qwena1_benchmark.json")
    return parser.parse_args()


def build_dataset(args: argparse.Namespace) -> tuple[A2DOpenLoopDataset, QwenA1Config]:
    model_dir = Path(args.model_dir)
    config = QwenA1Config.from_pretrained(model_dir)
    config.device = args.device
    config.dtype = args.dtype
    config.compile_model = args.compile_model

    train_meta = QwenA1TrainMetadata.from_pretrained(model_dir)
    with open(model_dir / "stats.json", "r", encoding="utf-8") as f:
        train_stats = json.load(f)["a2d"]

    dataset = A2DOpenLoopDataset(
        args.dataset_dir,
        config=config,
        train_stats=train_stats,
        processor_model_name=train_meta.processor_model_name,
    )
    return dataset, config


def build_od_config(
    *,
    model_dir: str,
    dtype_name: str,
    device: str,
    attn_implementation: str,
    enable_regional_compile: bool,
) -> OmniDiffusionConfig:
    return OmniDiffusionConfig(
        model=str(Path(model_dir).resolve()),
        model_class_name="QwenA1Pipeline",
        dtype=tensor_dtype(dtype_name),
        custom_pipeline_args={
            "device": device,
            "dtype": dtype_name,
            "attn_implementation": attn_implementation,
            "enable_regional_compile": enable_regional_compile,
        },
    )


def synchronize_if_needed(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def summarize_latencies(latencies_ms: list[float]) -> dict[str, float]:
    ordered = sorted(latencies_ms)
    p50 = ordered[len(ordered) // 2]
    p90 = ordered[min(len(ordered) - 1, max(0, int(len(ordered) * 0.9) - 1))]
    return {
        "mean_ms": float(statistics.mean(latencies_ms)),
        "stdev_ms": float(statistics.pstdev(latencies_ms)) if len(latencies_ms) > 1 else 0.0,
        "min_ms": float(min(latencies_ms)),
        "max_ms": float(max(latencies_ms)),
        "p50_ms": float(p50),
        "p90_ms": float(p90),
    }


def decode_variant(variant: str) -> tuple[str, bool]:
    if variant == "eager":
        return "eager", False
    if variant == "sdpa":
        return "sdpa", False
    if variant == "eager_regional":
        return "eager", True
    if variant == "sdpa_regional":
        return "sdpa", True
    raise ValueError(f"Unsupported variant: {variant}")


def run_variant(
    *,
    variant: str,
    pipeline: QwenA1Pipeline,
    dataset: A2DOpenLoopDataset,
    config: QwenA1Config,
    args: argparse.Namespace,
    indices: list[int],
) -> dict[str, Any]:
    cold_latencies_ms: list[float] = []
    steady_latencies_ms: list[float] = []
    sample_summaries: list[dict[str, Any]] = []

    for index in indices:
        sample = dataset.get_sample(index)
        batch_inputs, _ = collate_open_loop_samples(
            [sample],
            device=args.device,
            dtype=tensor_dtype(args.dtype),
        )
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

        for iteration in range(args.warmup_iters + args.benchmark_iters):
            synchronize_if_needed(args.device)
            start = time.perf_counter()
            with torch.no_grad():
                pred, _ = pipeline.run_batch_sample_actions(
                    batch_inputs,
                    noise=noise,
                    decode_image=False,
                    load_weights=True,
                    strict=args.strict_load,
                )
            synchronize_if_needed(args.device)
            duration_ms = (time.perf_counter() - start) * 1000.0

            if iteration == 0:
                cold_latencies_ms.append(duration_ms)
            elif iteration >= args.warmup_iters:
                steady_latencies_ms.append(duration_ms)

        pred = pred[:, :, : dataset.physical_action_dim].to(torch.float32).cpu()
        sample_summaries.append(
            {
                "index": index,
                "episode_index": sample.episode_index,
                "task": sample.task,
                "action_sha256": tensor_sha256(pred),
            }
        )

    result: dict[str, Any] = {
        "variant": variant,
        "num_samples": len(indices),
        "warmup_iters": args.warmup_iters,
        "benchmark_iters": args.benchmark_iters,
        "cold_start": summarize_latencies(cold_latencies_ms),
        "steady_state": summarize_latencies(steady_latencies_ms) if steady_latencies_ms else None,
        "samples": sample_summaries,
    }
    return result


def main() -> None:
    args = parse_args()
    dataset, config = build_dataset(args)
    indices = select_indices(dataset, args.num_samples)

    results: list[dict[str, Any]] = []
    for variant in args.variants:
        attn_implementation, enable_regional_compile = decode_variant(variant)
        od_config = build_od_config(
            model_dir=args.model_dir,
            dtype_name=args.dtype,
            device=args.device,
            attn_implementation=attn_implementation,
            enable_regional_compile=enable_regional_compile,
        )
        pipeline = QwenA1Pipeline(od_config=od_config)
        result = run_variant(
            variant=variant,
            pipeline=pipeline,
            dataset=dataset,
            config=config,
            args=args,
            indices=indices,
        )
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))

    output = {
        "model_dir": str(Path(args.model_dir).resolve()),
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "device": args.device,
        "dtype": args.dtype,
        "seed": args.seed,
        "indices": indices,
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
