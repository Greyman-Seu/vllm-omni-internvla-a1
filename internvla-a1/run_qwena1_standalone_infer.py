#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from qwena1_infer_common import (
    DEFAULT_DATASET_DIR,
    DEFAULT_MODEL_DIR,
    make_shared_noise,
    select_indices,
    tensor_dtype,
    tensor_sha256,
    workspace_root,
)

ROOT = workspace_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwena1_standalone import (
    A2DOpenLoopDataset,
    QwenA1Config,
    QwenA1TrainMetadata,
    StandaloneQwenA1Policy,
    collate_open_loop_samples,
)
from qwena1_standalone.constants import OBS_PREFIX


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run standalone QwenA1 inference on a few samples.')
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
    model_dir = Path(args.model_dir)
    dataset_dir = Path(args.dataset_dir)
    dtype = tensor_dtype(args.dtype)

    config = QwenA1Config.from_pretrained(model_dir)
    config.device = args.device
    config.dtype = args.dtype
    config.compile_model = args.compile_model

    train_meta = QwenA1TrainMetadata.from_pretrained(model_dir)
    with open(model_dir / 'stats.json', 'r', encoding='utf-8') as f:
        train_stats = json.load(f)['a2d']

    dataset = A2DOpenLoopDataset(
        dataset_dir,
        config=config,
        train_stats=train_stats,
        processor_model_name=train_meta.processor_model_name,
    )
    indices = select_indices(dataset, args.num_samples)

    policy = StandaloneQwenA1Policy.from_pretrained(model_dir, config=config)
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
                (batch_inputs[f'{OBS_PREFIX}attention_mask'].shape[0], config.chunk_size, config.max_action_dim),
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
        'mode': 'standalone_qwena1',
        'workspace_root': str(ROOT),
        'model_dir': str(model_dir),
        'dataset_dir': str(dataset_dir),
        'num_samples': len(indices),
        'seed': args.seed,
        'dtype': args.dtype,
        'device': args.device,
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
