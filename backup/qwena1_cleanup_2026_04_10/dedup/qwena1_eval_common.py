from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from qwena1_standalone.constants import OBS_PREFIX
from qwena1_standalone.dataset import unnormalize_vector


def plot_gt_vs_pred(gt: np.ndarray, pred: np.ndarray, output_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

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


def run_open_loop_evaluation(
    *,
    mode: str,
    policy,
    config,
    dataset,
    train_meta,
    collate_samples: Callable[..., tuple[dict[str, torch.Tensor], dict[str, Any]]],
    run_sample_actions: Callable[[Any, dict[str, torch.Tensor], torch.Tensor], torch.Tensor],
    make_shared_noise: Callable[[int, int, tuple[int, ...], str], torch.Tensor],
    num_episodes: int,
    seed: int,
    device: str,
    dtype: torch.dtype,
    output_dir: str | Path,
    skip_plots: bool,
) -> dict[str, Any]:
    policy.to(device)
    policy.to(dtype)
    policy.eval()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    metric_mse = []
    mse_joint = []
    mse_gripper = []
    per_episode = []

    plotting_available = True
    if not skip_plots:
        try:
            import matplotlib.pyplot as _plt  # noqa: F401
        except ModuleNotFoundError:
            plotting_available = False
            print('[warning] matplotlib is not installed; plots will be skipped.')

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
            batch_inputs, meta = collate_samples([sample], device=device, dtype=dtype)
            noise = make_shared_noise(
                seed,
                index,
                (batch_inputs[f'{OBS_PREFIX}attention_mask'].shape[0], config.chunk_size, config.max_action_dim),
                device,
            )
            with torch.no_grad():
                pred = run_sample_actions(policy, batch_inputs, noise)

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

        if not skip_plots and plotting_available:
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
