from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

_WORKSPACE_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _WORKSPACE_ROOT.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vllm_omni.diffusion.models.qwena1.constants import OBS_PREFIX
from vllm_omni.diffusion.models.qwena1.dataset import unnormalize_vector


def plot_prediction_series(
    *,
    series: dict[str, np.ndarray],
    output_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    first_series = next(iter(series.values()))
    num_dims = int(first_series.shape[1])
    num_cols = 2
    num_rows = math.ceil(num_dims / num_cols)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(16, max(6, 3 * num_rows)))
    axs = np.array(axs).reshape(-1)
    x_values = np.arange(first_series.shape[0])
    styles = [
        ("Ground Truth", "blue", "-", 1.5),
        ("Standalone", "red", "--", 1.4),
        ("VLLM", "green", "-.", 1.4),
        ("Predicted", "red", "--", 1.4),
    ]
    style_map = {name: (color, linestyle, linewidth) for name, color, linestyle, linewidth in styles}

    for dim in range(num_dims):
        ax = axs[dim]
        for name, values in series.items():
            color, linestyle, linewidth = style_map.get(name, (None, "-", 1.2))
            ax.plot(
                x_values,
                values[:, dim],
                label=name,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
            )
        ax.set_title(f"Dimension {dim + 1}")
        ax.set_xlabel("Time Step / Sample Index")
        ax.set_ylabel(f"Value Dim {dim + 1}")
        ax.legend(loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.7)

    for dim in range(num_dims, len(axs)):
        axs[dim].axis("off")

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig)


def summarize_prediction_metrics(
    *,
    gt_tensor: torch.Tensor,
    pred_tensor: torch.Tensor,
    joint_dims: int = 14,
) -> dict[str, float]:
    return {
        "mse": float(F.mse_loss(gt_tensor, pred_tensor, reduction="mean").item()),
        "mae": float(F.l1_loss(gt_tensor, pred_tensor, reduction="mean").item()),
        "mse_joint": float(F.mse_loss(gt_tensor[:, :joint_dims], pred_tensor[:, :joint_dims], reduction="mean").item()),
        "mae_joint": float(F.l1_loss(gt_tensor[:, :joint_dims], pred_tensor[:, :joint_dims], reduction="mean").item()),
        "mse_gripper": float(F.mse_loss(gt_tensor[:, joint_dims:], pred_tensor[:, joint_dims:], reduction="mean").item()),
        "mae_gripper": float(F.l1_loss(gt_tensor[:, joint_dims:], pred_tensor[:, joint_dims:], reduction="mean").item()),
    }


def summarize_metric_list(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


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
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    mse_values: list[float] = []
    mae_values: list[float] = []
    mse_joint_values: list[float] = []
    mae_joint_values: list[float] = []
    mse_gripper_values: list[float] = []
    mae_gripper_values: list[float] = []
    per_episode: list[dict[str, Any]] = []

    plotting_available = True
    if not skip_plots:
        try:
            import matplotlib.pyplot as _plt  # noqa: F401
        except ModuleNotFoundError:
            plotting_available = False
            print("[warning] matplotlib is not installed; plots will be skipped.")

    episode_specs = dataset.episode_start_indices(max_episodes=num_episodes)
    for episode_index, indices in episode_specs:
        print(f"[{mode}] episode: {episode_index}")
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
                (batch_inputs[f"{OBS_PREFIX}attention_mask"].shape[0], config.chunk_size, config.max_action_dim),
                device,
            )
            with torch.no_grad():
                pred = run_sample_actions(policy, batch_inputs, noise)

            pred = pred[:, :, : dataset.physical_action_dim].to(torch.float32).cpu()
            pred_phys = unnormalize_vector(pred, dataset.action_stats)
            if train_meta.action_mode == "delta":
                pred_phys[:, :, :14] += meta["state_raw"][:, None, :14]

            gt_phys = meta["action_raw"][:, :, : dataset.physical_action_dim].to(torch.float32).cpu()
            pred_chunks.append(pred_phys[0])
            gt_chunks.append(gt_phys[0])
            visited_indices.append(index)

        pred_tensor = torch.cat(pred_chunks, dim=0)
        gt_tensor = torch.cat(gt_chunks, dim=0)
        metrics = summarize_prediction_metrics(gt_tensor=gt_tensor, pred_tensor=pred_tensor)

        mse_values.append(metrics["mse"])
        mae_values.append(metrics["mae"])
        mse_joint_values.append(metrics["mse_joint"])
        mae_joint_values.append(metrics["mae_joint"])
        mse_gripper_values.append(metrics["mse_gripper"])
        mae_gripper_values.append(metrics["mae_gripper"])

        if not skip_plots and plotting_available:
            plot_prediction_series(
                series={
                    "Ground Truth": gt_tensor.numpy(),
                    "Predicted": pred_tensor.numpy(),
                },
                output_path=plots_dir / f"{mode}_open_loop_ep{episode_index}.jpg",
                title=f"{mode} Ground Truth vs Prediction",
            )

        episode_log = {
            "episode_id": int(episode_index),
            "task": task,
            "visited_indices": visited_indices,
            "num_pred_steps": int(pred_tensor.shape[0]),
            **metrics,
        }
        per_episode.append(episode_log)
        print(json.dumps({"mode": mode, **episode_log}, ensure_ascii=False))

    summary = {
        "mode": mode,
        "num_episodes": len(per_episode),
        "mse": mse_values,
        "mae": mae_values,
        "average_mse": summarize_metric_list(mse_values),
        "average_mae": summarize_metric_list(mae_values),
        "mse_joint": mse_joint_values,
        "mae_joint": mae_joint_values,
        "average_mse_joint": summarize_metric_list(mse_joint_values),
        "average_mae_joint": summarize_metric_list(mae_joint_values),
        "mse_gripper": mse_gripper_values,
        "mae_gripper": mae_gripper_values,
        "average_mse_gripper": summarize_metric_list(mse_gripper_values),
        "average_mae_gripper": summarize_metric_list(mae_gripper_values),
        "episodes": per_episode,
    }
    with open(output_dir / "log.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary
