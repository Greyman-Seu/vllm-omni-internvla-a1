from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import torchvision
from transformers import Qwen3VLProcessor

from .config import QwenA1Config
from .constants import DEFAULT_QWEN3_VL_MODEL, OBS_IMAGES, OBS_PREFIX, OBS_STATE


def _clamp_index(index: int, start: int, end: int) -> int:
    return max(start, min(end - 1, index))


def _load_parquet_rows(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _stack_stats(stats: dict[str, Any], keys: list[str]) -> dict[str, torch.Tensor]:
    result = {}
    for stat_name in ('mean', 'std'):
        values = []
        for key in keys:
            values.extend(stats[key][stat_name])
        result[stat_name] = torch.tensor(values, dtype=torch.float32)
    return result


def normalize_vector(values: torch.Tensor, stats: dict[str, torch.Tensor]) -> torch.Tensor:
    denom = torch.where(stats['std'] == 0, torch.ones_like(stats['std']), stats['std'])
    return (values - stats['mean']) / denom


def unnormalize_vector(values: torch.Tensor, stats: dict[str, torch.Tensor]) -> torch.Tensor:
    return values * stats['std'] + stats['mean']


def resize_with_pad(images: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    target_h, target_w = size
    if images.ndim != 4:
        raise ValueError(f'Expected [T, C, H, W], got {tuple(images.shape)}')
    _, _, src_h, src_w = images.shape
    scale = min(target_h / src_h, target_w / src_w)
    resized_h = max(1, int(round(src_h * scale)))
    resized_w = max(1, int(round(src_w * scale)))
    resized = F.interpolate(images, size=(resized_h, resized_w), mode='bilinear', align_corners=False)
    pad_h = target_h - resized_h
    pad_w = target_w - resized_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom))


class TorchvisionVideoReaderCache:
    def __init__(self, backend: str = 'pyav') -> None:
        self.backend = backend
        self._readers: dict[str, Any] = {}
        torchvision.set_video_backend(backend)

    def get(self, path: str) -> Any:
        reader = self._readers.get(path)
        if reader is None:
            reader = torchvision.io.VideoReader(path, 'video')
            self._readers[path] = reader
        return reader

    def decode_frames(self, path: str, timestamps: list[float], tolerance_s: float = 1e-4) -> torch.Tensor:
        reader = self.get(path)
        first_ts = min(timestamps)
        last_ts = max(timestamps)
        reader.seek(first_ts, keyframes_only=self.backend == 'pyav')

        loaded_frames: list[torch.Tensor] = []
        loaded_ts: list[float] = []
        for frame in reader:
            current_ts = float(frame['pts'])
            loaded_frames.append(frame['data'])
            loaded_ts.append(current_ts)
            if current_ts >= last_ts:
                break

        query_ts = torch.tensor(timestamps, dtype=torch.float32)
        loaded_ts_tensor = torch.tensor(loaded_ts, dtype=torch.float32)
        distances = torch.cdist(query_ts[:, None], loaded_ts_tensor[:, None], p=1)
        min_dist, argmin = distances.min(dim=1)
        if not torch.all(min_dist < tolerance_s):
            raise RuntimeError(
                f'Video timestamps are outside tolerance: query={query_ts.tolist()} '
                f'loaded={loaded_ts_tensor.tolist()} path={path}'
            )
        return torch.stack([loaded_frames[i] for i in argmin]).float() / 255.0


class Qwen3VLInputBuilder:
    def __init__(
        self,
        processor_model_name: str = DEFAULT_QWEN3_VL_MODEL,
        max_length: int = 48,
        spatial_merge_size: int = 2,
    ) -> None:
        self.processor = Qwen3VLProcessor.from_pretrained(processor_model_name)
        self.max_length = max_length
        self.spatial_merge_size = spatial_merge_size

    def build(self, camera_images: list[torch.Tensor], task: str) -> dict[str, torch.Tensor]:
        input_ids: list[int] = []
        attention_mask: list[int] = []
        pixel_values: list[torch.Tensor] = []
        image_grid_thw: list[torch.Tensor] = []

        for image_history in camera_images:
            current_image = image_history[1]
            image_inputs = self.processor.image_processor(current_image, do_rescale=False)
            pixel_values.append(image_inputs.pixel_values)
            image_grid_thw.append(image_inputs.image_grid_thw)
            num_img_token = int(torch.prod(image_inputs.image_grid_thw[0]).item()) // (self.spatial_merge_size ** 2)
            input_ids.extend(
                [self.processor.vision_start_token_id]
                + [self.processor.image_token_id] * num_img_token
                + [self.processor.vision_end_token_id]
            )
            attention_mask.extend([1] * (num_img_token + 2))

        text_inputs = self.processor.tokenizer(
            task,
            max_length=self.max_length,
            padding='max_length',
            padding_side='right',
            truncation=True,
        )
        input_ids.extend(text_inputs.input_ids)
        attention_mask.extend(text_inputs.attention_mask)

        return {
            f'{OBS_PREFIX}pixel_values': torch.cat(pixel_values, dim=0),
            f'{OBS_PREFIX}image_grid_thw': torch.cat(image_grid_thw, dim=0),
            f'{OBS_PREFIX}input_ids': torch.tensor(input_ids, dtype=torch.long),
            f'{OBS_PREFIX}attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        }


@dataclass
class A2DOpenLoopSample:
    index: int
    episode_index: int
    task: str
    state_raw: torch.Tensor
    action_raw: torch.Tensor
    inputs: dict[str, torch.Tensor]


class A2DOpenLoopDataset:
    image_keys = [
        'observation.images.head',
        'observation.images.hand_left',
        'observation.images.hand_right',
    ]
    state_keys = [
        'observation.states.joint.position',
        'observation.states.effector.position',
    ]
    action_keys = [
        'actions.joint.position',
        'actions.effector.position',
    ]

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        config: QwenA1Config,
        train_stats: dict[str, Any],
        processor_model_name: str = DEFAULT_QWEN3_VL_MODEL,
        image_offsets: tuple[int, int] = (-15, 0),
        tolerance_s: float = 1e-4,
    ) -> None:
        self.root = Path(dataset_root)
        self.config = config
        self.info = _load_json(self.root / 'meta' / 'info.json')
        self.dataset_stats = _load_json(self.root / 'meta' / 'stats.json')
        self.data_rows = _load_parquet_rows(self.root / 'data' / 'chunk-000' / 'file-000.parquet')
        self.episode_rows = _load_parquet_rows(self.root / 'meta' / 'episodes' / 'chunk-000' / 'file-000.parquet')
        self.task_rows = _load_parquet_rows(self.root / 'meta' / 'tasks.parquet')

        self.state_stats = _stack_stats(train_stats, self.state_keys)
        self.action_stats = _stack_stats(train_stats, self.action_keys)
        self.image_offsets = image_offsets
        self.tolerance_s = tolerance_s
        self.video_reader = TorchvisionVideoReaderCache(backend='pyav')
        self.processor = Qwen3VLInputBuilder(
            processor_model_name=processor_model_name,
            max_length=config.tokenizer_max_length,
        )

    @property
    def num_episodes(self) -> int:
        return len(self.episode_rows)

    @property
    def physical_action_dim(self) -> int:
        return 16

    def episode_start_indices(self, max_episodes: int | None = None) -> list[tuple[int, list[int]]]:
        rows = self.episode_rows if max_episodes is None else self.episode_rows[:max_episodes]
        result = []
        for ep in rows:
            start = int(ep['dataset_from_index'])
            end = int(ep['dataset_to_index'])
            result.append((int(ep['episode_index']), list(range(start, end, self.config.chunk_size))))
        return result

    def _task_text(self, task_index: int) -> str:
        return self.task_rows[task_index]['__index_level_0__']

    def _episode_for_index(self, row: dict[str, Any]) -> dict[str, Any]:
        return self.episode_rows[int(row['episode_index'])]

    def _state_vector(self, row: dict[str, Any]) -> torch.Tensor:
        return torch.tensor(row[self.state_keys[0]] + row[self.state_keys[1]], dtype=torch.float32)

    def _action_vector(self, row: dict[str, Any]) -> torch.Tensor:
        return torch.tensor(row[self.action_keys[0]] + row[self.action_keys[1]], dtype=torch.float32)

    def _query_rows(self, idx: int, deltas: list[int]) -> list[dict[str, Any]]:
        row = self.data_rows[idx]
        episode = self._episode_for_index(row)
        start = int(episode['dataset_from_index'])
        end = int(episode['dataset_to_index'])
        return [self.data_rows[_clamp_index(idx + delta, start, end)] for delta in deltas]

    def _decode_camera_history(self, episode: dict[str, Any], camera_key: str, rows: list[dict[str, Any]]) -> torch.Tensor:
        timestamps = [float(r['timestamp']) for r in rows]
        shifted = [float(episode[f'videos/{camera_key}/from_timestamp']) + ts for ts in timestamps]
        chunk_idx = int(episode[f'videos/{camera_key}/chunk_index'])
        file_idx = int(episode[f'videos/{camera_key}/file_index'])
        path = self.root / self.info['video_path'].format(
            video_key=camera_key,
            chunk_index=chunk_idx,
            file_index=file_idx,
        )
        frames = self.video_reader.decode_frames(str(path), shifted, tolerance_s=self.tolerance_s)
        return resize_with_pad(frames, self.config.image_resolution)

    def get_sample(self, idx: int) -> A2DOpenLoopSample:
        row = self.data_rows[idx]
        episode = self._episode_for_index(row)
        image_rows = self._query_rows(idx, list(self.image_offsets))
        action_rows = self._query_rows(idx, list(range(self.config.chunk_size)))
        camera_images = [
            self._decode_camera_history(episode, camera_key, image_rows)
            for camera_key in self.image_keys
        ]
        state_raw = self._state_vector(row)
        state_norm = normalize_vector(state_raw, self.state_stats)
        action_raw = torch.stack([self._action_vector(action_row) for action_row in action_rows], dim=0)
        task = self._task_text(int(row['task_index']))
        qwen_inputs = self.processor.build(camera_images, task)
        inputs = {
            OBS_STATE: state_norm,
            f'{OBS_IMAGES}.image0': camera_images[0],
            f'{OBS_IMAGES}.image1': camera_images[1],
            f'{OBS_IMAGES}.image2': camera_images[2],
            f'{OBS_IMAGES}.image0_mask': torch.tensor(True),
            f'{OBS_IMAGES}.image1_mask': torch.tensor(True),
            f'{OBS_IMAGES}.image2_mask': torch.tensor(True),
            **qwen_inputs,
        }
        return A2DOpenLoopSample(
            index=idx,
            episode_index=int(row['episode_index']),
            task=task,
            state_raw=state_raw,
            action_raw=action_raw,
            inputs=inputs,
        )


def collate_open_loop_samples(
    samples: list[A2DOpenLoopSample],
    *,
    device: str,
    dtype: torch.dtype,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    first = samples[0]
    batch_inputs: dict[str, torch.Tensor] = {}
    for key in first.inputs:
        values = [sample.inputs[key] for sample in samples]
        tensor = torch.stack(values, dim=0)
        if tensor.dtype in (torch.int64, torch.bool):
            batch_inputs[key] = tensor.to(device=device)
        else:
            batch_inputs[key] = tensor.to(device=device, dtype=dtype)

    metadata = {
        'indices': [sample.index for sample in samples],
        'episode_indices': [sample.episode_index for sample in samples],
        'tasks': [sample.task for sample in samples],
        'state_raw': torch.stack([sample.state_raw for sample in samples], dim=0),
        'action_raw': torch.stack([sample.action_raw for sample in samples], dim=0),
    }
    return batch_inputs, metadata
