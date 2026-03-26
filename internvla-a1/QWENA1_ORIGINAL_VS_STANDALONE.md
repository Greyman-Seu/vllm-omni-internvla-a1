# QwenA1 Original Vs Standalone

## Purpose

This directory separates the migrated standalone QwenA1 inference path from the original `lerobot_lab` training/repo path.

## Path Split

- Original repo path: `/mnt/petrelfs/zhuyangkun/workspace/lerobot_lab`
- Standalone path: `/mnt/petrelfs/zhuyangkun/workspace/vllm-omni`

## What Stays In Original Repo Path

- `run_qwena1_repo_infer.py` and `compare_qwena1_precision.py` still import the original `lerobot.policies.qwena1` implementation from `lerobot_lab/src`.
- Original repo-side `Qwen3VLForConditionalGeneration` still comes from `transformers.models.qwen3_vl`, matching the active environment behavior.
- This is the reference path for parity checking.

## What Moves To Standalone Path

- `qwena1_standalone/` is copied into this directory and can run without importing any file from `lerobot_lab`.
- The patched `modeling_qwen3_vl.py` is vendored inside `qwena1_standalone/`.
- Unmodified public components such as `Qwen3VLProcessor` still come from official `transformers`.

## Intentional Differences

- Repo path uses the original `lerobot` policy classes.
- Standalone path uses `qwena1_standalone.StandaloneQwenA1Policy`.
- Repo path keeps the environment's active `transformers.models.qwen3_vl` behavior.
- Standalone path avoids importing `transformers.models.qwen3_vl.modeling_qwen3_vl` from site-packages and instead imports the vendored patched file in this directory.

## Kept Aligned For Accuracy Checks

- Same checkpoint config loading.
- Same dataset sampling logic.
- Same image/state preprocessing path.
- Same deterministic noise generation: `seed + sample_index`.
- Same action truncation to physical action dimension.
- Same precision comparison metrics.

## Entrypoints

- `run_qwena1_standalone_infer.py`: standalone-only path in `vllm-omni`.
- `run_qwena1_repo_infer.py`: original repo path from `lerobot_lab` plus local dataset helpers.
- `compare_qwena1_precision.py`: compares original repo path against standalone path.
