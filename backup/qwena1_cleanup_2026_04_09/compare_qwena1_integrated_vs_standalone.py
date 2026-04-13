#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

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
from vllm_omni.diffusion.models.qwena1.model import make_att_2d_masks
from vllm_omni.diffusion.models.qwena1.pipeline_qwena1 import QwenA1Pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare integrated repo QwenA1 against the current standalone copy under internvla-a1."
    )
    parser.add_argument("--standalone-root", default=DEFAULT_STANDALONE_ROOT)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--indices", nargs="*", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--enable-regional-compile", action="store_true")
    parser.add_argument("--strict-load", action="store_true")
    parser.add_argument("--debug-compare", action="store_true")
    parser.add_argument("--debug-threshold", type=float, default=1e-6)
    parser.add_argument("--output-json", default="compare_qwena1_integrated_vs_standalone.json")
    return parser.parse_args()


def select_indices(dataset: A2DOpenLoopDataset, requested: list[int] | None, num_samples: int) -> list[int]:
    if requested:
        return requested
    indices: list[int] = []
    for _, episode_indices in dataset.episode_start_indices(max_episodes=num_samples):
        if episode_indices:
            indices.append(episode_indices[0])
    return indices[:num_samples]


def load_standalone_policy(standalone_root: Path, checkpoint_dir: Path, config: QwenA1Config, strict: bool):
    if str(standalone_root.resolve()) not in sys.path:
        sys.path.insert(0, str(standalone_root.resolve()))

    from qwena1_standalone import StandaloneQwenA1Policy as LegacyStandaloneQwenA1Policy

    return LegacyStandaloneQwenA1Policy.from_pretrained(checkpoint_dir, config=config, strict=strict)


def run_standalone_sample_actions(policy, batch_inputs: dict[str, torch.Tensor], *, noise: torch.Tensor):
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
        decode_image=False,
    )


def _prepare_attention_mask_4d(model, att_2d_masks: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    try:
        return model._prepare_attention_masks_4d(att_2d_masks, dtype=dtype)
    except TypeError:
        return model._prepare_attention_masks_4d(att_2d_masks)


def _extract_first_cache_tensors(past_key_values: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if past_key_values is None:
        return None, None

    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        if len(past_key_values.key_cache) == 0:
            return None, None
        return past_key_values.key_cache[0], past_key_values.value_cache[0]

    if isinstance(past_key_values, (list, tuple)) and len(past_key_values) > 0:
        first = past_key_values[0]
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            return first[0], first[1]

    return None, None


def _tensor_diff(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    a_cpu = a.detach().to(torch.float32).cpu()
    b_cpu = b.detach().to(torch.float32).cpu()
    diff = (a_cpu - b_cpu).abs()
    return {
        "shape": list(a_cpu.shape),
        "mean_abs_diff": float(diff.mean().item()),
        "max_abs_diff": float(diff.max().item()),
    }


def build_debug_trace(policy, batch_inputs: dict[str, torch.Tensor], *, noise: torch.Tensor) -> dict[str, torch.Tensor]:
    model = policy.model
    pixel_values = batch_inputs[f"{OBS_PREFIX}pixel_values"]
    image_grid_thw = batch_inputs[f"{OBS_PREFIX}image_grid_thw"]
    lang_tokens = batch_inputs[f"{OBS_PREFIX}input_ids"]
    lang_masks = batch_inputs[f"{OBS_PREFIX}attention_mask"]
    state = policy.prepare_state(batch_inputs)
    images, img_masks = policy._preprocess_images(batch_inputs)

    trace: dict[str, torch.Tensor] = {}

    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(pixel_values, image_grid_thw, lang_tokens, lang_masks)
    trace["prefix_embs"] = prefix_embs
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids, _ = model.get_position_ids(lang_tokens, image_grid_thw, prefix_pad_masks)
    prefix_att_2d_masks_4d = _prepare_attention_mask_4d(model, prefix_att_2d_masks, dtype=prefix_embs.dtype)
    _, past_key_values = model.qwen3_vl_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None, None],
        use_cache=True,
    )
    cache_key0, cache_value0 = _extract_first_cache_tensors(past_key_values)
    if cache_key0 is not None:
        trace["prefix_cache_key0"] = cache_key0
    if cache_value0 is not None:
        trace["prefix_cache_value0"] = cache_value0

    max_prefix_position_ids = prefix_position_ids.max(dim=-1, keepdim=True).values
    middle_embs, middle_pad_masks, middle_att_masks = model.embed_middle(images[:, :, :2], img_masks)
    trace["middle_embs"] = middle_embs
    middle_len = middle_pad_masks.shape[1]
    prefix_len = prefix_pad_masks.shape[1]
    prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(state.shape[0], middle_len, prefix_len)
    middle_att_2d_masks = make_att_2d_masks(middle_pad_masks, middle_att_masks)
    full_att_2d_masks = torch.cat([prefix_pad_2d_masks, middle_att_2d_masks], dim=2)
    middle_position_ids = (
        torch.arange(1, middle_len + 1, device=max_prefix_position_ids.device)
        .repeat(3, 1, 1)
        .to(max_prefix_position_ids)
        + max_prefix_position_ids
    )
    full_att_2d_masks_4d = _prepare_attention_mask_4d(model, full_att_2d_masks, dtype=middle_embs.dtype)
    (_, middle_out, _), past_key_values = model.qwen3_vl_with_expert.forward(
        attention_mask=full_att_2d_masks_4d,
        position_ids=middle_position_ids,
        past_key_values=past_key_values,
        inputs_embeds=[None, middle_embs, None],
        use_cache=True,
    )
    trace["middle_out"] = middle_out

    max_position_ids = middle_position_ids.max(dim=-1, keepdim=True).values
    curr_pad_masks = torch.cat([prefix_pad_masks, middle_pad_masks], dim=1)
    timestep = torch.tensor(1.0, dtype=torch.float32, device=state.device).expand(state.shape[0]).to(state.dtype)

    suffix_embs, suffix_pad_masks, suffix_att_masks = model.embed_suffix(state, noise.to(state.dtype), timestep)
    trace["suffix_embs_t1"] = suffix_embs
    suffix_len = suffix_pad_masks.shape[1]
    prefix_len = curr_pad_masks.shape[1]
    prefix_pad_2d_masks = curr_pad_masks[:, None, :].expand(state.shape[0], suffix_len, prefix_len)
    suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
    suffix_full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
    suffix_position_ids = (
        torch.arange(1, suffix_len + 1, device=max_position_ids.device)
        .repeat(3, 1, 1)
        .to(max_position_ids)
        + max_position_ids
    )
    trace["suffix_mask_t1"] = _prepare_attention_mask_4d(model, suffix_full_att_2d_masks, dtype=suffix_embs.dtype)
    trace["suffix_position_ids_t1"] = suffix_position_ids
    first_v_t = model.denoise_step(
        state,
        curr_pad_masks,
        past_key_values,
        max_position_ids,
        noise.to(state.dtype),
        timestep,
    )
    trace["first_v_t"] = first_v_t

    final_pred, _ = model.sample_actions(
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
    trace["final_pred"] = final_pred
    return trace


def compare_debug_traces(integrated_trace: dict[str, torch.Tensor], standalone_trace: dict[str, torch.Tensor], *, threshold: float) -> dict[str, Any]:
    stage_order = [
        "prefix_embs",
        "prefix_cache_key0",
        "prefix_cache_value0",
        "middle_embs",
        "middle_out",
        "suffix_embs_t1",
        "suffix_mask_t1",
        "suffix_position_ids_t1",
        "first_v_t",
        "final_pred",
    ]
    stage_diffs: dict[str, Any] = {}
    first_divergent_stage: str | None = None
    for stage_name in stage_order:
        integrated_tensor = integrated_trace.get(stage_name)
        standalone_tensor = standalone_trace.get(stage_name)
        if integrated_tensor is None or standalone_tensor is None:
            continue
        diff = _tensor_diff(integrated_tensor, standalone_tensor)
        stage_diffs[stage_name] = diff
        if first_divergent_stage is None and diff["max_abs_diff"] > threshold:
            first_divergent_stage = stage_name

    return {
        "debug_threshold": threshold,
        "first_divergent_stage": first_divergent_stage,
        "stage_diffs": stage_diffs,
    }


def main() -> None:
    args = parse_args()
    standalone_root = Path(args.standalone_root)
    if not standalone_root.exists():
        raise FileNotFoundError(f"standalone root not found: {standalone_root}")

    checkpoint_dir = Path(args.model_dir)
    dtype = tensor_dtype(args.dtype)

    config = QwenA1Config.from_pretrained(checkpoint_dir)
    config.device = args.device
    config.dtype = args.dtype
    config.compile_model = args.compile_model
    if args.attn_implementation:
        config.attn_implementation = args.attn_implementation
    if args.enable_regional_compile:
        config.enable_regional_compile = True

    train_meta = QwenA1TrainMetadata.from_pretrained(checkpoint_dir)
    with open(checkpoint_dir / "stats.json", "r", encoding="utf-8") as f:
        train_stats = json.load(f)["a2d"]

    dataset = A2DOpenLoopDataset(
        args.dataset_dir,
        config=config,
        train_stats=train_stats,
        processor_model_name=train_meta.processor_model_name,
    )
    indices = select_indices(dataset, args.indices, args.num_samples)

    integrated_pipeline = QwenA1Pipeline(
        od_config=OmniDiffusionConfig(
            model=str(checkpoint_dir.resolve()),
            model_class_name="QwenA1Pipeline",
            dtype=dtype,
            custom_pipeline_args={
                "device": args.device,
                "dtype": args.dtype,
                "attn_implementation": args.attn_implementation,
                "enable_regional_compile": args.enable_regional_compile,
            },
        )
    )
    standalone_policy = load_standalone_policy(standalone_root, checkpoint_dir, config, args.strict_load)
    standalone_policy.to(args.device)
    standalone_policy.to(dtype)
    standalone_policy.eval()
    integrated_policy = integrated_pipeline.get_or_create_policy(load_weights=True, strict=args.strict_load)

    results = []
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
        with torch.no_grad():
            integrated_pred, _ = integrated_pipeline.run_batch_sample_actions(
                batch_inputs,
                noise=shared_noise,
                decode_image=False,
                load_weights=True,
                strict=args.strict_load,
            )
            standalone_pred, _ = run_standalone_sample_actions(
                standalone_policy,
                batch_inputs,
                noise=shared_noise,
            )

        integrated_pred = integrated_pred[:, :, :dataset.physical_action_dim].to(torch.float32).cpu()
        standalone_pred = standalone_pred[:, :, :dataset.physical_action_dim].to(torch.float32).cpu()
        raw_diff = (integrated_pred - standalone_pred).abs()

        integrated_phys = unnormalize_vector(integrated_pred, dataset.action_stats)
        standalone_phys = unnormalize_vector(standalone_pred, dataset.action_stats)
        if train_meta.action_mode == "delta":
            integrated_phys[:, :, :14] += meta["state_raw"][:, None, :14]
            standalone_phys[:, :, :14] += meta["state_raw"][:, None, :14]
        phys_diff = (integrated_phys - standalone_phys).abs()

        result = {
            "index": index,
            "episode_index": sample.episode_index,
            "task": sample.task,
            "raw_mean_abs_diff": float(raw_diff.mean().item()),
            "raw_max_abs_diff": float(raw_diff.max().item()),
            "physical_mean_abs_diff": float(phys_diff.mean().item()),
            "physical_max_abs_diff": float(phys_diff.max().item()),
        }
        if args.debug_compare:
            integrated_trace = build_debug_trace(integrated_policy, batch_inputs, noise=shared_noise)
            standalone_trace = build_debug_trace(standalone_policy, batch_inputs, noise=shared_noise)
            result["debug_compare"] = compare_debug_traces(
                integrated_trace,
                standalone_trace,
                threshold=args.debug_threshold,
            )
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))

    output = {
        "standalone_root": str(standalone_root.resolve()),
        "model_dir": str(checkpoint_dir.resolve()),
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "device": args.device,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "enable_regional_compile": args.enable_regional_compile,
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
