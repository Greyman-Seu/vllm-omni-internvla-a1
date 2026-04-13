#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

from qwena1_eval_common import plot_prediction_series, summarize_metric_list, summarize_prediction_metrics
from qwena1_infer_common import (
    DEFAULT_DATASET_DIR,
    DEFAULT_MODEL_DIR,
    DEFAULT_STANDALONE_ROOT,
    make_shared_noise,
    tensor_dtype,
)
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.qwena1 import (
    A2DOpenLoopDataset,
    QwenA1Config,
    QwenA1TrainMetadata,
    collate_open_loop_samples,
)
from vllm_omni.diffusion.models.qwena1.constants import OBS_PREFIX, OBS_STATE
from vllm_omni.diffusion.models.qwena1.dataset import unnormalize_vector
from vllm_omni.diffusion.models.qwena1.pipeline_qwena1 import QwenA1Pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare vLLM and standalone QwenA1 against GT and plot curves.")
    parser.add_argument("--standalone-root", default=DEFAULT_STANDALONE_ROOT)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--strict-load", action="store_true")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--enable-regional-compile", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--output-dir", default="outputs/qwena1/vllm_standalone_gt")
    return parser.parse_args()


def load_standalone_policy(standalone_root: Path, checkpoint_dir: Path, config: QwenA1Config, strict: bool):
    if str(standalone_root.resolve()) not in sys.path:
        sys.path.insert(0, str(standalone_root.resolve()))

    from qwena1_standalone import StandaloneQwenA1Policy

    return StandaloneQwenA1Policy.from_pretrained(checkpoint_dir, config=config, strict=strict)


def run_standalone_sample_actions(policy, batch_inputs: dict[str, torch.Tensor], noise: torch.Tensor):
    pixel_values = batch_inputs[f"{OBS_PREFIX}pixel_values"]
    image_grid_thw = batch_inputs[f"{OBS_PREFIX}image_grid_thw"]
    lang_tokens = batch_inputs[f"{OBS_PREFIX}input_ids"]
    lang_masks = batch_inputs[f"{OBS_PREFIX}attention_mask"]
    state = policy.prepare_state(batch_inputs)
    images, img_masks = policy._preprocess_images(batch_inputs)
    pred, _ = policy.model.sample_actions(
        images,
        img_masks,
        pixel_values,
        image_grid_thw,
        lang_tokens,
        lang_masks,
        state,
        noise=noise,
        decode_image=False,
    )
    return pred


def run_vllm_sample_actions(
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


def _to_physical_actions(
    pred: torch.Tensor,
    *,
    dataset: A2DOpenLoopDataset,
    train_meta: QwenA1TrainMetadata,
    meta: dict[str, Any],
) -> torch.Tensor:
    pred = pred[:, :, : dataset.physical_action_dim].to(torch.float32).cpu()
    pred_phys = unnormalize_vector(pred, dataset.action_stats)
    if train_meta.action_mode == "delta":
        pred_phys[:, :, :14] += meta["state_raw"][:, None, :14]
    return pred_phys


def main() -> None:
    args = parse_args()
    standalone_root = Path(args.standalone_root)
    if not standalone_root.exists():
        raise FileNotFoundError(f"standalone root not found: {standalone_root}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    dataset, config, train_meta = build_dataset(args)
    dtype = tensor_dtype(args.dtype)

    vllm_pipeline = QwenA1Pipeline(od_config=build_od_config(args))
    standalone_policy = load_standalone_policy(standalone_root, Path(args.model_dir), config, args.strict_load)
    standalone_policy.to(args.device)
    standalone_policy.to(dtype)
    standalone_policy.eval()

    mse_vllm_values: list[float] = []
    mae_vllm_values: list[float] = []
    mse_standalone_values: list[float] = []
    mae_standalone_values: list[float] = []
    mse_gap_values: list[float] = []
    mae_gap_values: list[float] = []
    episodes: list[dict[str, Any]] = []

    episode_specs = dataset.episode_start_indices(max_episodes=args.num_episodes)
    for episode_index, indices in episode_specs:
        print(f"[vllm_vs_standalone_gt] episode: {episode_index}")
        vllm_chunks = []
        standalone_chunks = []
        gt_chunks = []
        visited_indices = []
        task = None

        for index in indices:
            sample = dataset.get_sample(index)
            task = sample.task
            batch_inputs, meta = collate_open_loop_samples([sample], device=args.device, dtype=dtype)
            noise = make_shared_noise(
                args.seed,
                index,
                (batch_inputs[OBS_STATE].shape[0], config.chunk_size, config.max_action_dim),
                args.device,
            )

            with torch.no_grad():
                vllm_pred = run_vllm_sample_actions(
                    vllm_pipeline,
                    batch_inputs,
                    noise,
                    strict_load=args.strict_load,
                )
                standalone_pred = run_standalone_sample_actions(
                    standalone_policy,
                    batch_inputs,
                    noise,
                )

            vllm_phys = _to_physical_actions(
                vllm_pred,
                dataset=dataset,
                train_meta=train_meta,
                meta=meta,
            )
            standalone_phys = _to_physical_actions(
                standalone_pred,
                dataset=dataset,
                train_meta=train_meta,
                meta=meta,
            )
            gt_phys = meta["action_raw"][:, :, : dataset.physical_action_dim].to(torch.float32).cpu()

            vllm_chunks.append(vllm_phys[0])
            standalone_chunks.append(standalone_phys[0])
            gt_chunks.append(gt_phys[0])
            visited_indices.append(index)

        vllm_tensor = torch.cat(vllm_chunks, dim=0)
        standalone_tensor = torch.cat(standalone_chunks, dim=0)
        gt_tensor = torch.cat(gt_chunks, dim=0)

        vllm_metrics = summarize_prediction_metrics(gt_tensor=gt_tensor, pred_tensor=vllm_tensor)
        standalone_metrics = summarize_prediction_metrics(gt_tensor=gt_tensor, pred_tensor=standalone_tensor)
        gap_metrics = summarize_prediction_metrics(gt_tensor=standalone_tensor, pred_tensor=vllm_tensor)

        mse_vllm_values.append(vllm_metrics["mse"])
        mae_vllm_values.append(vllm_metrics["mae"])
        mse_standalone_values.append(standalone_metrics["mse"])
        mae_standalone_values.append(standalone_metrics["mae"])
        mse_gap_values.append(gap_metrics["mse"])
        mae_gap_values.append(gap_metrics["mae"])

        if not args.skip_plots:
            plot_prediction_series(
                series={
                    "Ground Truth": gt_tensor.numpy(),
                    "Standalone": standalone_tensor.numpy(),
                    "VLLM": vllm_tensor.numpy(),
                },
                output_path=plots_dir / f"vllm_standalone_gt_ep{episode_index}.jpg",
                title=f"Episode {episode_index}: GT vs Standalone vs VLLM",
            )

        episode_summary = {
            "episode_id": int(episode_index),
            "task": task,
            "visited_indices": visited_indices,
            "num_pred_steps": int(gt_tensor.shape[0]),
            "vllm_vs_gt": vllm_metrics,
            "standalone_vs_gt": standalone_metrics,
            "vllm_vs_standalone": gap_metrics,
        }
        episodes.append(episode_summary)
        print(json.dumps(episode_summary, ensure_ascii=False))

    summary = {
        "standalone_root": str(standalone_root.resolve()),
        "model_dir": str(Path(args.model_dir).resolve()),
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "device": args.device,
        "dtype": args.dtype,
        "seed": args.seed,
        "num_episodes": len(episodes),
        "attn_implementation": args.attn_implementation,
        "enable_regional_compile": args.enable_regional_compile,
        "average_vllm_mse": summarize_metric_list(mse_vllm_values),
        "average_vllm_mae": summarize_metric_list(mae_vllm_values),
        "average_standalone_mse": summarize_metric_list(mse_standalone_values),
        "average_standalone_mae": summarize_metric_list(mae_standalone_values),
        "average_vllm_vs_standalone_mse": summarize_metric_list(mse_gap_values),
        "average_vllm_vs_standalone_mae": summarize_metric_list(mae_gap_values),
        "episodes": episodes,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
