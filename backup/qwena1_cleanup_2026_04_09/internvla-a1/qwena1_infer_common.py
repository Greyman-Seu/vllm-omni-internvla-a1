from __future__ import annotations

import hashlib
from pathlib import Path

import torch

DEFAULT_REPO_ROOT = '/home/zhuyangkun/vllm-omni-internvla-a1/internvla-a1'
DEFAULT_MODEL_DIR = (
    '/home/zhuyangkun/data/vllm_a1/'
    '2026_02_13_10_22_12-qwena1-a2d_real_A2D_Put_the_pen_from_the_table_into_the_pen_holder-delta-scratch-060000/'
    'pretrained_model'
)
DEFAULT_DATASET_DIR = '/home/zhuyangkun/data/vllm_a1/pick_marker_pen_inference_rollouts_v30'


def tensor_dtype(name: str) -> torch.dtype:
    return torch.bfloat16 if name == 'bfloat16' else torch.float32


def select_indices(dataset, num_samples: int) -> list[int]:
    indices: list[int] = []
    for _, episode_indices in dataset.episode_start_indices(max_episodes=num_samples):
        if episode_indices:
            indices.append(episode_indices[0])
    return indices[:num_samples]


def make_shared_noise(seed: int, sample_index: int, shape: tuple[int, ...], device: str) -> torch.Tensor:
    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed + sample_index)
    noise = torch.randn(shape, generator=generator, dtype=torch.float32)
    return noise.to(device=device)


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().contiguous().cpu().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def workspace_root() -> Path:
    return Path(__file__).resolve().parent
