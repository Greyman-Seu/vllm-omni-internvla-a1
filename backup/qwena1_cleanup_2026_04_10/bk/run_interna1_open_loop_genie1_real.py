#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path
from pprint import pp

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import HF_HOME


DEFAULT_REPO_ROOT = '/mnt/petrelfs/zhuyangkun/workspace/lerobot_lab'
DEFAULT_CHECKPOINT = 'Jia-Zeng/InternVLA-A1-3B-FineTuned-Place_Markpen'
DEFAULT_DATASET_REPO = 'InternRobotics/InternData-A1'
DEFAULT_DATASET_FILE = 'real_lerobotv30/genie1/Genie1-Place_Markpen.tar.gz'
DEFAULT_DATASET_ROOT = str(Path(HF_HOME) / 'real_lerobotv30' / 'genie1' / 'Genie1-Place_Markpen')
DEFAULT_OUTPUT_DIR = 'outputs/interna1/open_loop_genie1_real'


def tensor_dtype(name: str) -> torch.dtype:
    if name == 'bfloat16':
        return torch.bfloat16
    if name == 'float32':
        return torch.float32
    raise ValueError(f'Unsupported dtype: {name}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run InternA1 open-loop evaluation for the Genie1 Place_Markpen case.')
    parser.add_argument('--repo-root', default=DEFAULT_REPO_ROOT)
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--dataset-root', default=DEFAULT_DATASET_ROOT)
    parser.add_argument('--dataset-repo', default=DEFAULT_DATASET_REPO)
    parser.add_argument('--dataset-file', default=DEFAULT_DATASET_FILE)
    parser.add_argument('--num-episodes', type=int, default=3)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', choices=['float32', 'bfloat16'], default='float32')
    parser.add_argument('--compile-model', action='store_true')
    parser.add_argument('--compile-mode', default='reduce-overhead')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--skip-plots', action='store_true')
    parser.add_argument('--force-download-dataset', action='store_true')
    return parser.parse_args()


def ensure_repo_imports(repo_root: Path) -> None:
    src_root = repo_root / 'src'
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


def _is_within_directory(directory: Path, target: Path) -> bool:
    directory = directory.resolve()
    target = target.resolve()
    return str(target).startswith(str(directory))


def ensure_dataset_ready(dataset_root: Path, dataset_repo: str, dataset_file: str, force_download: bool) -> Path:
    if dataset_root.exists() and not force_download:
        return dataset_root

    dataset_root.parent.mkdir(parents=True, exist_ok=True)
    archive_path = Path(
        hf_hub_download(
            repo_id=dataset_repo,
            filename=dataset_file,
            repo_type='dataset',
        )
    )
    with tarfile.open(archive_path, 'r:gz') as tar:
        for member in tar.getmembers():
            target = dataset_root.parent / member.name
            if not _is_within_directory(dataset_root.parent, target):
                raise RuntimeError(f'Unsafe tar member path: {member.name}')
        tar.extractall(path=dataset_root.parent)
    if not dataset_root.exists():
        raise FileNotFoundError(f'Expected extracted dataset at {dataset_root}, but it does not exist after extraction.')
    return dataset_root


def load_policy(checkpoint: str, device: str, dtype: torch.dtype, compile_model: bool, compile_mode: str):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.interna1 import InternA1Config, InternA1Policy

    config = PreTrainedConfig.from_pretrained(checkpoint)
    assert isinstance(config, InternA1Config), f'Expected InternA1Config, got {type(config)}'
    config.compile_model = compile_model
    config.compile_mode = compile_mode
    config.device = device
    config.dtype = 'bfloat16' if dtype == torch.bfloat16 else 'float32'

    policy = InternA1Policy.from_pretrained(
        config=config,
        pretrained_name_or_path=checkpoint,
    )
    policy.to(device)
    policy.to(dtype)
    policy.eval()
    return policy, config


def load_dataset(checkpoint: str, dataset_root: Path):
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_dataset
    from lerobot.datasets.utils import cast_stats_to_numpy, load_json
    from lerobot.transforms.core import UnNormalizeTransformFn
    from lerobot.utils.constants import ACTION, OBS_STATE

    cfg = TrainPipelineConfig.from_pretrained(checkpoint)
    cfg.dataset.repo_id = str(dataset_root)
    cfg.dataset.use_external_stats = False
    action_mode = cfg.dataset.action_mode
    dataset, _ = make_dataset(cfg)

    if Path(checkpoint).is_dir():
        stats_file = Path(checkpoint) / 'stats.json'
        if not stats_file.exists():
            raise FileNotFoundError(f'stats.json not found in {checkpoint}')
    else:
        stats_file = Path(hf_hub_download(repo_id=str(checkpoint), filename='stats.json'))

    stats = cast_stats_to_numpy(load_json(stats_file)[dataset.meta.robot_type])
    dataset.meta.stats.update(stats)

    stat_keys = ['min', 'max', 'mean', 'std']
    action_stat = {
        stat_key: np.concatenate([
            dataset.meta.stats['actions.joint.position'][stat_key],
            dataset.meta.stats['actions.effector.position'][stat_key],
        ], axis=-1)
        for stat_key in stat_keys
    }
    state_stat = {
        stat_key: np.concatenate([
            dataset.meta.stats['observation.states.joint.position'][stat_key],
            dataset.meta.stats['observation.states.effector.position'][stat_key],
        ], axis=-1)
        for stat_key in stat_keys
    }

    act_unnorm_fn = UnNormalizeTransformFn(
        selected_keys=[ACTION],
        mode='mean_std',
        norm_stats={ACTION: action_stat},
    )
    state_unnorm_fn = UnNormalizeTransformFn(
        selected_keys=[OBS_STATE],
        mode='mean_std',
        norm_stats={OBS_STATE: state_stat},
    )
    return cfg, dataset, action_mode, act_unnorm_fn, state_unnorm_fn


def move_sample_to_device(sample: dict[str, torch.Tensor], device: str, dtype: torch.dtype) -> dict[str, object]:
    inputs: dict[str, object] = {}
    for key, value in sample.items():
        if key == 'task':
            inputs[key] = [value]
        elif isinstance(value, torch.Tensor):
            if value.dtype in (torch.int64, torch.bool):
                inputs[key] = value[None].to(device)
            else:
                inputs[key] = value[None].to(device=device, dtype=dtype)
        else:
            inputs[key] = value
    return inputs


def save_episode_plot(action_gt: np.ndarray, action_pred: np.ndarray, output_path: Path) -> None:
    fig, axs = plt.subplots(8, 2, figsize=(16, 12))
    axs = axs.ravel()
    num_dimensions = action_gt.shape[1]
    x_values = np.arange(action_gt.shape[0])
    for dim in range(num_dimensions):
        axs[dim].plot(x_values, action_gt[:, dim], label='Ground Truth', color='blue', linewidth=1.5)
        axs[dim].plot(x_values, action_pred[:, dim], label='Predicted', color='red', linestyle='--', linewidth=1.5)
        axs[dim].set_title(f'Dimension {dim + 1}')
        axs[dim].set_xlabel('Time Step / Sample Index')
        axs[dim].set_ylabel(f'Value Dim {dim + 1}')
        axs[dim].legend(loc='upper right')
        axs[dim].grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.suptitle('Ground Truth vs Prediction', fontsize=16, y=1.02)
    plt.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    ensure_repo_imports(repo_root)

    dataset_root = ensure_dataset_ready(
        Path(args.dataset_root),
        dataset_repo=args.dataset_repo,
        dataset_file=args.dataset_file,
        force_download=args.force_download_dataset,
    )

    device = args.device
    dtype = tensor_dtype(args.dtype)

    from lerobot.utils.constants import ACTION, OBS_STATE

    policy, config = load_policy(
        checkpoint=args.checkpoint,
        device=device,
        dtype=dtype,
        compile_model=args.compile_model,
        compile_mode=args.compile_mode,
    )
    cfg, dataset, action_mode, act_unnorm_fn, state_unnorm_fn = load_dataset(
        checkpoint=args.checkpoint,
        dataset_root=dataset_root,
    )

    from_ids = np.asarray(dataset.meta.episodes['dataset_from_index']).tolist()
    to_ids = np.asarray(dataset.meta.episodes['dataset_to_index']).tolist()
    total_num_episodes = dataset.num_episodes

    output_dir = Path(args.output_dir) / Path(cfg.dataset.repo_id).name
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True, parents=True)

    metric_mse = []
    mse_joint = []
    mse_gripper = []
    per_episode = []

    for ep_id in range(min(total_num_episodes, args.num_episodes)):
        print(f'episode: {ep_id}')
        print(f'from_idx: {from_ids[ep_id]}, to_idx: {to_ids[ep_id]}')
        action_gt_list = []
        action_pred_list = []
        state_list = []
        visited_indices = []

        for idx in range(from_ids[ep_id], to_ids[ep_id], config.chunk_size):
            print(f'compute sample {idx}')
            sample = dataset[idx]
            inputs = move_sample_to_device(sample, device=device, dtype=dtype)
            with torch.no_grad():
                action_pred, _ = policy.predict_action_chunk(inputs, decode_image=False)
                action_pred = action_pred[0, :, :16]
                action_gt = inputs[ACTION][0, :, :16]
                action_gt_list.append(action_gt)
                action_pred_list.append(action_pred.clone())
                state_list.append(inputs[OBS_STATE].clone().repeat(config.chunk_size, 1)[:, :16])
                visited_indices.append(idx)

        action_gt_tensor = torch.cat(action_gt_list, dim=0)
        action_gt_tensor = act_unnorm_fn({ACTION: action_gt_tensor})[ACTION]
        action_pred_tensor = torch.cat(action_pred_list, dim=0)
        action_pred_tensor = act_unnorm_fn({ACTION: action_pred_tensor})[ACTION]

        if action_mode == 'delta':
            state_tensor = torch.cat(state_list, dim=0)
            state_tensor = state_unnorm_fn({OBS_STATE: state_tensor})[OBS_STATE]
            action_pred_tensor[:, :14] += state_tensor[:, :14]
            action_gt_tensor[:, :14] += state_tensor[:, :14]

        action_gt_tensor = action_gt_tensor.to(torch.float32)
        action_pred_tensor = action_pred_tensor.to(torch.float32)
        mse_all = float(F.mse_loss(action_gt_tensor, action_pred_tensor, reduction='mean').detach().cpu().numpy())
        mse_j = float(F.mse_loss(action_gt_tensor[:, :14], action_pred_tensor[:, :14], reduction='mean').detach().cpu().numpy())
        mse_g = float(F.mse_loss(action_gt_tensor[:, 14:], action_pred_tensor[:, 14:], reduction='mean').detach().cpu().numpy())
        metric_mse.append(mse_all)
        mse_joint.append(mse_j)
        mse_gripper.append(mse_g)

        action_gt_numpy = action_gt_tensor.detach().cpu().numpy()
        action_pred_numpy = action_pred_tensor.detach().cpu().numpy()
        if not args.skip_plots:
            save_episode_plot(action_gt_numpy, action_pred_numpy, plots_dir / f'interna1_open_loop_ep{ep_id}.jpg')

        episode_log = {
            'episode_id': ep_id,
            'from_idx': from_ids[ep_id],
            'to_idx': to_ids[ep_id],
            'visited_indices': visited_indices,
            'mse': mse_all,
            'mse_joint': mse_j,
            'mse_gripper': mse_g,
            'num_pred_steps': int(action_pred_tensor.shape[0]),
            'task': sample.get('task'),
        }
        per_episode.append(episode_log)
        print(json.dumps(episode_log, ensure_ascii=False))

    log = {
        'checkpoint': args.checkpoint,
        'dataset_root': str(dataset_root),
        'output_dir': str(output_dir),
        'dtype': args.dtype,
        'device': args.device,
        'num_episodes': len(per_episode),
        'MSE': metric_mse,
        'Average MSE': float(np.mean(metric_mse)) if metric_mse else None,
        'MSE on joints': mse_joint,
        'Average MSE on joints': float(np.mean(mse_joint)) if mse_joint else None,
        'MSE on gripper': mse_gripper,
        'Average MSE on gripper': float(np.mean(mse_gripper)) if mse_gripper else None,
        'episodes': per_episode,
    }
    output_dir.mkdir(exist_ok=True, parents=True)
    with open(output_dir / 'log.json', 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    pp(log)


if __name__ == '__main__':
    main()
