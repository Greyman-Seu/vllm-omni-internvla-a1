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
    select_indices,
    tensor_dtype,
    tensor_sha256,
    workspace_root,
)

ROOT = workspace_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwena1_standalone import A2DOpenLoopDataset, QwenA1TrainMetadata, collate_open_loop_samples
from qwena1_standalone.constants import OBS_PREFIX


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run original repo QwenA1 inference on a few samples.')
    parser.add_argument('--repo-root', default=DEFAULT_REPO_ROOT)
    parser.add_argument('--model-dir', default=DEFAULT_MODEL_DIR)
    parser.add_argument('--dataset-dir', default=DEFAULT_DATASET_DIR)
    parser.add_argument('--num-samples', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--dtype', choices=['bfloat16', 'float32'], default='bfloat16')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--compile-model', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    src_root = repo_root / 'src'
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.qwena1 import QwenA1Config as RepoQwenA1Config
    from lerobot.policies.qwena1 import QwenA1Policy as RepoQwenA1Policy

    model_dir = Path(args.model_dir)
    dataset_dir = Path(args.dataset_dir)
    dtype = tensor_dtype(args.dtype)

    repo_config = PreTrainedConfig.from_pretrained(model_dir)
    assert isinstance(repo_config, RepoQwenA1Config)
    repo_config.device = args.device
    repo_config.dtype = args.dtype
    repo_config.compile_model = args.compile_model

    train_meta = QwenA1TrainMetadata.from_pretrained(model_dir)
    with open(model_dir / 'stats.json', 'r', encoding='utf-8') as f:
        train_stats = json.load(f)['a2d']

    dataset = A2DOpenLoopDataset(
        dataset_dir,
        config=repo_config,
        train_stats=train_stats,
        processor_model_name=train_meta.processor_model_name,
    )
    indices = select_indices(dataset, args.num_samples)

    original_loader = Qwen3VLForConditionalGeneration.from_pretrained

    @classmethod
    def _config_only_loader(cls, pretrained_model_name_or_path=None, *model_args, config=None, **model_kwargs):
        if config is None:
            raise ValueError('config must be provided for offline Qwen3-VL initialization')
        return cls(config)

    Qwen3VLForConditionalGeneration.from_pretrained = _config_only_loader
    try:
        policy = RepoQwenA1Policy.from_pretrained(model_dir, config=repo_config)
    finally:
        Qwen3VLForConditionalGeneration.from_pretrained = original_loader

    policy.to(args.device)
    policy.to(dtype)
    policy.eval()

    with torch.no_grad():
        for index in indices:
            sample = dataset.get_sample(index)
            batch_inputs, _ = collate_open_loop_samples([sample], device=args.device, dtype=dtype)
            pixel_values = batch_inputs[f'{OBS_PREFIX}pixel_values']
            image_grid_thw = batch_inputs[f'{OBS_PREFIX}image_grid_thw']
            lang_tokens = batch_inputs[f'{OBS_PREFIX}input_ids']
            lang_masks = batch_inputs[f'{OBS_PREFIX}attention_mask']
            state = policy.prepare_state(batch_inputs)
            images, img_masks = policy._preprocess_images(batch_inputs)
            noise = make_shared_noise(
                args.seed,
                index,
                (batch_inputs[f'{OBS_PREFIX}attention_mask'].shape[0], repo_config.chunk_size, repo_config.max_action_dim),
                args.device,
            )
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
            pred = pred[:, :, :dataset.physical_action_dim].to(torch.float32).cpu()
            result = {
                'index': index,
                'episode_index': sample.episode_index,
                'task': sample.task,
                'seed': args.seed,
                'shape': list(pred.shape),
                'mean': float(pred.mean().item()),
                'std': float(pred.std().item()),
                'action_sha256': tensor_sha256(pred),
                'first_action_prefix': pred[0, 0, :8].tolist(),
            }
            print(json.dumps(result, ensure_ascii=False))

    print(json.dumps({
        'mode': 'repo_qwena1',
        'workspace_root': str(ROOT),
        'repo_root': str(repo_root),
        'model_dir': str(model_dir),
        'dataset_dir': str(dataset_dir),
        'num_samples': len(indices),
        'seed': args.seed,
        'dtype': args.dtype,
        'device': args.device,
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
