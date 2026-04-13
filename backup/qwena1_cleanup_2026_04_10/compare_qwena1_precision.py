#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration

from qwena1_infer_common import (
    DEFAULT_DATASET_DIR,
    DEFAULT_MODEL_DIR,
    DEFAULT_REPO_ROOT,
    make_shared_noise,
    tensor_dtype,
    workspace_root,
)

ROOT = workspace_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwena1_standalone import A2DOpenLoopDataset, QwenA1Config, QwenA1TrainMetadata, StandaloneQwenA1Policy, collate_open_loop_samples
from qwena1_standalone.constants import OBS_PREFIX, OBS_STATE
from qwena1_standalone.dataset import unnormalize_vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compare standalone QwenA1 inference against the original lerobot repo implementation.')
    parser.add_argument('--repo-root', default=DEFAULT_REPO_ROOT)
    parser.add_argument('--model-dir', default=DEFAULT_MODEL_DIR)
    parser.add_argument('--dataset-dir', default=DEFAULT_DATASET_DIR)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', choices=['bfloat16', 'float32'], default='bfloat16')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--indices', nargs='*', type=int, default=None)
    parser.add_argument('--num-samples', type=int, default=3)
    parser.add_argument('--compile-model', action='store_true')
    parser.add_argument('--output-json', default='compare_qwena1_precision.json')
    return parser.parse_args()


def select_indices(dataset: A2DOpenLoopDataset, requested: list[int] | None, num_samples: int) -> list[int]:
    if requested:
        return requested
    indices: list[int] = []
    for _, episode_indices in dataset.episode_start_indices(max_episodes=num_samples):
        if episode_indices:
            indices.append(episode_indices[0])
    return indices[:num_samples]


def run_repo_sample_actions(policy, batch_inputs: dict[str, torch.Tensor], *, noise: torch.Tensor, decode_image: bool = False):
    pixel_values = batch_inputs[f'{OBS_PREFIX}pixel_values']
    image_grid_thw = batch_inputs[f'{OBS_PREFIX}image_grid_thw']
    lang_tokens = batch_inputs[f'{OBS_PREFIX}input_ids']
    lang_masks = batch_inputs[f'{OBS_PREFIX}attention_mask']
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


def run_standalone_sample_actions(policy: StandaloneQwenA1Policy, batch_inputs: dict[str, torch.Tensor], *, noise: torch.Tensor, decode_image: bool = False):
    pixel_values = batch_inputs[f'{OBS_PREFIX}pixel_values']
    image_grid_thw = batch_inputs[f'{OBS_PREFIX}image_grid_thw']
    lang_tokens = batch_inputs[f'{OBS_PREFIX}input_ids']
    lang_masks = batch_inputs[f'{OBS_PREFIX}attention_mask']
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


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    src_root = repo_root / 'src'
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.qwena1 import QwenA1Config as RepoQwenA1Config
    from lerobot.policies.qwena1 import QwenA1Policy as RepoQwenA1Policy

    checkpoint_dir = Path(args.model_dir)
    dtype = tensor_dtype(args.dtype)

    repo_config = PreTrainedConfig.from_pretrained(checkpoint_dir)
    assert isinstance(repo_config, RepoQwenA1Config)
    repo_config.device = args.device
    repo_config.dtype = args.dtype
    repo_config.compile_model = args.compile_model

    standalone_config = QwenA1Config.from_pretrained(checkpoint_dir)
    standalone_config.device = args.device
    standalone_config.dtype = args.dtype
    standalone_config.compile_model = args.compile_model

    train_meta = QwenA1TrainMetadata.from_pretrained(checkpoint_dir)
    with open(checkpoint_dir / 'stats.json', 'r', encoding='utf-8') as f:
        train_stats = json.load(f)['a2d']

    dataset = A2DOpenLoopDataset(
        args.dataset_dir,
        config=standalone_config,
        train_stats=train_stats,
        processor_model_name=train_meta.processor_model_name,
    )
    indices = select_indices(dataset, args.indices, args.num_samples)

    original_qwen3_vl_from_pretrained = Qwen3VLForConditionalGeneration.from_pretrained

    @classmethod
    def _config_only_qwen3_vl_loader(cls, pretrained_model_name_or_path=None, *model_args, config=None, **model_kwargs):
        if config is None:
            raise ValueError('config must be provided for offline Qwen3-VL initialization')
        return cls(config)

    Qwen3VLForConditionalGeneration.from_pretrained = _config_only_qwen3_vl_loader
    try:
        repo_policy = RepoQwenA1Policy.from_pretrained(checkpoint_dir, config=repo_config)
        repo_policy.to(args.device)
        repo_policy.to(dtype)
        repo_policy.eval()

        standalone_policy = StandaloneQwenA1Policy.from_pretrained(checkpoint_dir, config=standalone_config)
    finally:
        Qwen3VLForConditionalGeneration.from_pretrained = original_qwen3_vl_from_pretrained

    standalone_policy.to(args.device)
    standalone_policy.to(dtype)
    standalone_policy.eval()

    results = []
    for index in indices:
        sample = dataset.get_sample(index)
        batch_inputs, meta = collate_open_loop_samples([sample], device=args.device, dtype=dtype)
        shared_noise = make_shared_noise(
            args.seed,
            index,
            (
                batch_inputs[OBS_STATE].shape[0],
                standalone_config.chunk_size,
                standalone_config.max_action_dim,
            ),
            args.device,
        )
        with torch.no_grad():
            repo_pred, _ = run_repo_sample_actions(repo_policy, batch_inputs, noise=shared_noise, decode_image=False)
            standalone_pred, _ = run_standalone_sample_actions(
                standalone_policy,
                batch_inputs,
                noise=shared_noise,
                decode_image=False,
            )

        repo_pred = repo_pred[:, :, :dataset.physical_action_dim].to(torch.float32).cpu()
        standalone_pred = standalone_pred[:, :, :dataset.physical_action_dim].to(torch.float32).cpu()
        raw_diff = (repo_pred - standalone_pred).abs()

        repo_phys = unnormalize_vector(repo_pred, dataset.action_stats)
        standalone_phys = unnormalize_vector(standalone_pred, dataset.action_stats)
        if train_meta.action_mode == 'delta':
            repo_phys[:, :, :14] += meta['state_raw'][:, None, :14]
            standalone_phys[:, :, :14] += meta['state_raw'][:, None, :14]
        phys_diff = (repo_phys - standalone_phys).abs()

        result = {
            'index': index,
            'episode_index': sample.episode_index,
            'task': sample.task,
            'raw_mean_abs_diff': float(raw_diff.mean().item()),
            'raw_max_abs_diff': float(raw_diff.max().item()),
            'physical_mean_abs_diff': float(phys_diff.mean().item()),
            'physical_max_abs_diff': float(phys_diff.max().item()),
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))

    output = {
        'workspace_root': str(ROOT),
        'repo_root': str(repo_root),
        'model_dir': str(checkpoint_dir),
        'dataset_dir': str(Path(args.dataset_dir)),
        'device': args.device,
        'dtype': args.dtype,
        'seed': args.seed,
        'indices': indices,
        'results': results,
    }
    output_path = Path(args.output_json)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'Wrote {output_path}')


if __name__ == '__main__':
    main()
