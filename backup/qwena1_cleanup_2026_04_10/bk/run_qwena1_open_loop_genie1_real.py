#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
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

from qwena1_standalone import (
    A2DOpenLoopDataset,
    QwenA1Config,
    QwenA1TrainMetadata,
    StandaloneQwenA1Policy,
    collate_open_loop_samples,
)
from qwena1_standalone.constants import OBS_PREFIX
from qwena1_standalone.dataset import unnormalize_vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run qwena1 open-loop evaluation against GT on the local Genie1 Place_Markpen dataset.')
    parser.add_argument('--repo-root', default=DEFAULT_REPO_ROOT)
    parser.add_argument('--model-dir', default=DEFAULT_MODEL_DIR)
    parser.add_argument('--dataset-dir', default=DEFAULT_DATASET_DIR)
    parser.add_argument('--mode', choices=['repo', 'standalone', 'both'], default='both')
    parser.add_argument('--num-episodes', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', choices=['bfloat16', 'float32'], default='bfloat16')
    parser.add_argument('--compile-model', action='store_true')
    parser.add_argument('--output-dir', default='outputs/qwena1/open_loop_genie1_real')
    parser.add_argument('--skip-plots', action='store_true')
    return parser.parse_args()


def load_repo_policy(model_dir: Path, repo_root: Path, device: str, dtype_name: str, compile_model: bool):
    src_root = repo_root / 'src'
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.qwena1 import QwenA1Config as RepoQwenA1Config
    from lerobot.policies.qwena1 import QwenA1Policy as RepoQwenA1Policy

    repo_config = PreTrainedConfig.from_pretrained(model_dir)
    assert isinstance(repo_config, RepoQwenA1Config)
    repo_config.device = device
    repo_config.dtype = dtype_name
    repo_config.compile_model = compile_model

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
    return policy, repo_config


def load_standalone_policy(model_dir: Path, device: str, dtype_name: str, compile_model: bool):
    config = QwenA1Config.from_pretrained(model_dir)
    config.device = device
    config.dtype = dtype_name
    config.compile_model = compile_model
    policy = StandaloneQwenA1Policy.from_pretrained(model_dir, config=config)
    return policy, config


def run_repo_sample_actions(policy, batch_inputs: dict[str, torch.Tensor], noise: torch.Tensor):
    pixel_values = batch_inputs[f'{OBS_PREFIX}pixel_values']
    image_grid_thw = batch_inputs[f'{OBS_PREFIX}image_grid_thw']
    lang_tokens = batch_inputs[f'{OBS_PREFIX}input_ids']
    lang_masks = batch_inputs[f'{OBS_PREFIX}attention_mask']
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


def run_standalone_sample_actions(policy: StandaloneQwenA1Policy, batch_inputs: dict[str, torch.Tensor], noise: torch.Tensor):
    pixel_values = batch_inputs[f'{OBS_PREFIX}pixel_values']
    image_grid_thw = batch_inputs[f'{OBS_PREFIX}image_grid_thw']
    lang_tokens = batch_inputs[f'{OBS_PREFIX}input_ids']
    lang_masks = batch_inputs[f'{OBS_PREFIX}attention_mask']
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


def plot_gt_vs_pred(gt: np.ndarray, pred: np.ndarray, output_path: Path, title: str) -> None:
    fig, axs = plt.subplots(8, 2, figsize=(16, 12))
    axs = axs.ravel()
    x_values = np.arange(gt.shape[0])
    for dim in range(gt.shape[1]):
        axs[dim].plot(x_values, gt[:, dim], label='Ground Truth', color='blue', linewidth=1.5)
        axs[dim].plot(x_values, pred[:, dim], label='Predicted', color='red', linestyle='--', linewidth=1.5)
        axs[dim].set_title(f'Dimension {dim + 1}')
        axs[dim].set_xlabel('Time Step / Sample Index')
        axs[dim].set_ylabel(f'Value Dim {dim + 1}')
        axs[dim].legend(loc='upper right')
        axs[dim].grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.savefig(output_path)
    plt.close(fig)


def evaluate_mode(
    *,
    mode: str,
    policy,
    config,
    dataset: A2DOpenLoopDataset,
    train_meta: QwenA1TrainMetadata,
    num_episodes: int,
    seed: int,
    device: str,
    dtype: torch.dtype,
    output_dir: Path,
    skip_plots: bool,
):
    policy.to(device)
    policy.to(dtype)
    policy.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    metric_mse = []
    mse_joint = []
    mse_gripper = []
    per_episode = []

    episode_specs = dataset.episode_start_indices(max_episodes=num_episodes)
    for episode_index, indices in episode_specs:
        print(f'[{mode}] episode: {episode_index}')
        pred_chunks = []
        gt_chunks = []
        visited_indices = []
        task = None

        for index in indices:
            sample = dataset.get_sample(index)
            task = sample.task
            batch_inputs, meta = collate_open_loop_samples([sample], device=device, dtype=dtype)
            noise = make_shared_noise(
                seed,
                index,
                (batch_inputs[f'{OBS_PREFIX}attention_mask'].shape[0], config.chunk_size, config.max_action_dim),
                device,
            )
            with torch.no_grad():
                if mode == 'repo':
                    pred = run_repo_sample_actions(policy, batch_inputs, noise)
                else:
                    pred = run_standalone_sample_actions(policy, batch_inputs, noise)

            pred = pred[:, :, :dataset.physical_action_dim].to(torch.float32).cpu()
            pred_phys = unnormalize_vector(pred, dataset.action_stats)
            if train_meta.action_mode == 'delta':
                pred_phys[:, :, :14] += meta['state_raw'][:, None, :14]

            gt_phys = meta['action_raw'][:, :, :dataset.physical_action_dim].to(torch.float32).cpu()
            pred_chunks.append(pred_phys[0])
            gt_chunks.append(gt_phys[0])
            visited_indices.append(index)

        pred_tensor = torch.cat(pred_chunks, dim=0)
        gt_tensor = torch.cat(gt_chunks, dim=0)

        mse_all = float(F.mse_loss(gt_tensor, pred_tensor, reduction='mean').item())
        mse_j = float(F.mse_loss(gt_tensor[:, :14], pred_tensor[:, :14], reduction='mean').item())
        mse_g = float(F.mse_loss(gt_tensor[:, 14:], pred_tensor[:, 14:], reduction='mean').item())
        metric_mse.append(mse_all)
        mse_joint.append(mse_j)
        mse_gripper.append(mse_g)

        if not skip_plots:
            plot_gt_vs_pred(
                gt_tensor.numpy(),
                pred_tensor.numpy(),
                plots_dir / f'{mode}_open_loop_ep{episode_index}.jpg',
                f'{mode} Ground Truth vs Prediction',
            )

        episode_log = {
            'episode_id': int(episode_index),
            'task': task,
            'visited_indices': visited_indices,
            'num_pred_steps': int(pred_tensor.shape[0]),
            'mse': mse_all,
            'mse_joint': mse_j,
            'mse_gripper': mse_g,
        }
        per_episode.append(episode_log)
        print(json.dumps({'mode': mode, **episode_log}, ensure_ascii=False))

    summary = {
        'mode': mode,
        'num_episodes': len(per_episode),
        'MSE': metric_mse,
        'Average MSE': float(np.mean(metric_mse)) if metric_mse else None,
        'MSE on joints': mse_joint,
        'Average MSE on joints': float(np.mean(mse_joint)) if mse_joint else None,
        'MSE on gripper': mse_gripper,
        'Average MSE on gripper': float(np.mean(mse_gripper)) if mse_gripper else None,
        'episodes': per_episode,
    }
    with open(output_dir / 'log.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    dataset_dir = Path(args.dataset_dir)
    repo_root = Path(args.repo_root)
    dtype = tensor_dtype(args.dtype)

    train_meta = QwenA1TrainMetadata.from_pretrained(model_dir)
    with open(model_dir / 'stats.json', 'r', encoding='utf-8') as f:
        train_stats = json.load(f)['a2d']

    base_config = QwenA1Config.from_pretrained(model_dir)
    base_config.device = args.device
    base_config.dtype = args.dtype
    base_config.compile_model = args.compile_model

    dataset = A2DOpenLoopDataset(
        dataset_dir,
        config=base_config,
        train_stats=train_stats,
        processor_model_name=train_meta.processor_model_name,
    )

    modes = ['repo', 'standalone'] if args.mode == 'both' else [args.mode]
    summaries = {}
    for mode in modes:
        if mode == 'repo':
            policy, config = load_repo_policy(model_dir, repo_root, args.device, args.dtype, args.compile_model)
        else:
            policy, config = load_standalone_policy(model_dir, args.device, args.dtype, args.compile_model)
        summaries[mode] = evaluate_mode(
            mode=mode,
            policy=policy,
            config=config,
            dataset=dataset,
            train_meta=train_meta,
            num_episodes=args.num_episodes,
            seed=args.seed,
            device=args.device,
            dtype=dtype,
            output_dir=Path(args.output_dir) / mode,
            skip_plots=args.skip_plots,
        )

    output = {
        'model_dir': str(model_dir),
        'dataset_dir': str(dataset_dir),
        'seed': args.seed,
        'dtype': args.dtype,
        'device': args.device,
        'mode': args.mode,
        'summaries': summaries,
    }
    with open(Path(args.output_dir) / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(json.dumps(output, ensure_ascii=False))


if __name__ == '__main__':
    main()
