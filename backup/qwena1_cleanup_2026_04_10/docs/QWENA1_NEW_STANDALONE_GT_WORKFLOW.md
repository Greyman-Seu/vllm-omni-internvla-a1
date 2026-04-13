# QwenA1 New Standalone / GT Workflow

## What Changed

- The standalone root is:
  - `standalone`
- Default model and dataset paths now prefer the new PetrelFS locations:
  - model: `/mnt/petrelfs/zhuyangkun/vllm_a1/data/InternVLA-A1-3B-ft-pen`
  - dataset: `/mnt/petrelfs/zhuyangkun/vllm_a1/data/Genie1-Place_Markpen`
  - and fall back to the older `/home/zhuyangkun/data/vllm_a1/...` paths
- Integrated inference now supports open-loop GT evaluation and plot generation.
- Added a combined runner for:
  - `standalone vs GT`
  - `integrated vs GT`
  - `integrated vs standalone`
  - per-episode plots with all three curves

## Main Entrypoints

- Standalone wrapper:
  - `run_qwena1_standalone_infer.sh`
- Integrated wrapper:
  - `run_qwena1_integrated_infer.sh`
- Standalone + integrated + GT comparison:
  - `run_qwena1_compare_integrated_standalone_gt.sh`

## Recommended Commands

### 1. Standalone on new ckpt/data with GT plots

```bash
bash run_qwena1_standalone_infer.sh \
  --num-episodes 2 \
  --output-dir outputs/qwena1/standalone_infer_new
```

### 2. Integrated on new ckpt/data with GT plots

Use the benchmark-preferred integrated attention backend:

```bash
bash run_qwena1_integrated_infer.sh \
  --mode direct \
  --attn-implementation sdpa \
  --num-episodes 2 \
  --output-dir outputs/qwena1/integrated_infer_new
```

If parity with standalone is more important than deployment config, use:

```bash
bash run_qwena1_integrated_infer.sh \
  --mode direct \
  --attn-implementation eager \
  --num-episodes 2 \
  --output-dir outputs/qwena1/integrated_infer_new_eager
```

### 3. Standalone + integrated + GT unified comparison

For parity checking:

```bash
bash run_qwena1_compare_integrated_standalone_gt.sh \
  --num-episodes 2 \
  --attn-implementation eager \
  --output-dir outputs/qwena1/integrated_standalone_gt_eager
```

For integrated default deployment config:

```bash
bash run_qwena1_compare_integrated_standalone_gt.sh \
  --num-episodes 2 \
  --attn-implementation sdpa \
  --output-dir outputs/qwena1/integrated_standalone_gt_sdpa
```

## Outputs

### Integrated GT evaluation

- JSON summary:
  - `outputs/qwena1/integrated_infer.../<mode>/log.json`
- Plots:
  - `outputs/qwena1/integrated_infer.../<mode>/plots/*.jpg`

### Standalone + integrated + GT comparison

- JSON summary:
  - `outputs/qwena1/integrated_standalone_gt.../summary.json`
- Plots:
  - `outputs/qwena1/integrated_standalone_gt.../plots/*.jpg`

Each plot contains:

- `Ground Truth`
- `Standalone`
- `Integrated`

## Notes

- `eager` is the right mode for strict parity checks against standalone.
- `sdpa` is the better integrated default according to the current benchmark numbers.
- `regional_compile` is not enabled by default here because the previous benchmark showed no stable gain and visible recompilation overhead.
- This update was statically validated with:
  - `python -m py_compile`
  - `bash -n`
- Full model execution was not run as part of this patch.
