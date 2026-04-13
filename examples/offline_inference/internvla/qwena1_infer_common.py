from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import torch

_WORKSPACE_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _WORKSPACE_ROOT.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vllm_omni.diffusion.models.qwena1.constants import OBS_IMAGES, OBS_PREFIX, OBS_STATE


def _pick_default_path(env_name: str, candidates: list[str | Path]) -> str:
    env_value = os.getenv(env_name)
    if env_value:
        return env_value

    normalized = [str(Path(candidate).expanduser()) for candidate in candidates]
    for candidate in normalized:
        if Path(candidate).exists():
            return candidate
    return normalized[0]


DEFAULT_STANDALONE_ROOT = _pick_default_path(
    "QWENA1_STANDALONE_ROOT",
    [
        _WORKSPACE_ROOT / "standalone",
    ],
)

DEFAULT_MODEL_DIR = _pick_default_path(
    "QWENA1_MODEL_DIR",
    [
        "/home/zhuyangkun/data/vllm_a1/new_data/InternVLA-A1-3B-ft-pen",
    ],
)
DEFAULT_DATASET_DIR = _pick_default_path(
    "QWENA1_DATASET_DIR",
    [
        "/home/zhuyangkun/data/vllm_a1/new_data/Genie1-Place_Markpen",
    ],
)


def tensor_dtype(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bfloat16" else torch.float32


def select_indices(dataset, num_samples: int) -> list[int]:
    indices: list[int] = []
    for _, episode_indices in dataset.episode_start_indices(max_episodes=num_samples):
        if episode_indices:
            indices.append(episode_indices[0])
    return indices[:num_samples]


def make_shared_noise(seed: int, sample_index: int, shape: tuple[int, ...], device: str) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + sample_index)
    noise = torch.randn(shape, generator=generator, dtype=torch.float32)
    return noise.to(device=device)


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().contiguous().cpu().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def summarize_outputs(
    *,
    mode: str,
    actions: torch.Tensor,
    decoded: torch.Tensor | None,
) -> dict[str, object]:
    actions_cpu = actions.detach().to(torch.float32).cpu().contiguous()
    summary: dict[str, object] = {
        "mode": mode,
        "shape": list(actions_cpu.shape),
        "mean": float(actions_cpu.mean().item()),
        "std": float(actions_cpu.std().item()),
        "sha256": tensor_sha256(actions_cpu),
    }
    if decoded is not None:
        decoded_cpu = decoded.detach().to(torch.float32).cpu().contiguous()
        summary["decoded_shape"] = list(decoded_cpu.shape)
        summary["decoded_sha256"] = tensor_sha256(decoded_cpu)
    return summary


def make_fake_batch_inputs(
    *,
    batch_size: int,
    seed: int,
    device: str,
    dtype: torch.dtype,
    vision_start_token_id: int,
    image_token_id: int,
    vision_end_token_id: int,
    tokenizer_max_length: int,
    num_cameras: int,
    image_history: int,
    image_resolution: tuple[int, int],
    max_state_dim: int,
    visual_patch_size: int,
    visual_temporal_patch_size: int,
    visual_in_channels: int,
    visual_spatial_merge_size: int,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    height, width = image_resolution
    grid_t = 1
    grid_h = visual_spatial_merge_size
    grid_w = visual_spatial_merge_size
    patches_per_image = grid_t * grid_h * grid_w
    image_token_count = patches_per_image // (visual_spatial_merge_size**2)
    flattened_patch_dim = visual_in_channels * visual_temporal_patch_size * visual_patch_size * visual_patch_size

    visual_prefix_tokens: list[int] = []
    for _ in range(num_cameras):
        visual_prefix_tokens.extend(
            [vision_start_token_id]
            + [image_token_id] * image_token_count
            + [vision_end_token_id]
        )

    text_token_id = 0
    while text_token_id in {vision_start_token_id, image_token_id, vision_end_token_id}:
        text_token_id += 1

    lang_tokens = torch.full(
        (batch_size, tokenizer_max_length + len(visual_prefix_tokens)),
        fill_value=text_token_id,
        dtype=torch.long,
    )
    lang_tokens[:, : len(visual_prefix_tokens)] = torch.tensor(
        visual_prefix_tokens,
        dtype=torch.long,
    )
    lang_masks = torch.ones((batch_size, lang_tokens.shape[1]), dtype=torch.bool)
    pixel_values = torch.randn(
        batch_size,
        num_cameras * patches_per_image,
        flattened_patch_dim,
        generator=generator,
        dtype=torch.float32,
    )
    image_grid_thw = torch.tensor(
        [[[grid_t, grid_h, grid_w]]] * (batch_size * num_cameras),
        dtype=torch.long,
    ).view(batch_size, num_cameras, 3)
    images = torch.randn(
        batch_size,
        num_cameras,
        image_history,
        3,
        height,
        width,
        generator=generator,
        dtype=torch.float32,
    )
    img_masks = torch.zeros(
        batch_size,
        num_cameras,
        image_history,
        dtype=torch.bool,
    )
    state = torch.randn(
        batch_size,
        max_state_dim,
        generator=generator,
        dtype=torch.float32,
    )

    batch_inputs: dict[str, torch.Tensor] = {
        f"{OBS_PREFIX}pixel_values": pixel_values.to(device=device, dtype=dtype),
        f"{OBS_PREFIX}image_grid_thw": image_grid_thw.to(device=device),
        f"{OBS_PREFIX}input_ids": lang_tokens.to(device=device),
        f"{OBS_PREFIX}attention_mask": lang_masks.to(device=device),
        OBS_STATE: state.to(device=device, dtype=dtype),
    }
    for image_idx in range(num_cameras):
        batch_inputs[f"{OBS_IMAGES}.image{image_idx}"] = images[:, image_idx].to(
            device=device,
            dtype=dtype,
        )
        batch_inputs[f"{OBS_IMAGES}.image{image_idx}_mask"] = img_masks[:, image_idx].to(
            device=device
        )
    return batch_inputs


def workspace_root() -> Path:
    return _WORKSPACE_ROOT
