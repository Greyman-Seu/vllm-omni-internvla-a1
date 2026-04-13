# QwenA1 真机验证流程

这个文档对应仓内当前实现，目标是验证三件事：

- 仓内 `QwenA1Pipeline` 的直接实例化和 registry 实例化结果一致
- 仓内 integrated 结果与旧 `internvla-a1` standalone 结果对齐
- attention backend 和 `regional_compile` 在真实 CUDA 环境下可用

## 一键脚本

直接执行：

```bash
chmod +x ./run_qwena1_real_validation.sh
./run_qwena1_real_validation.sh
```

如果默认路径不是你的环境，可以先覆盖环境变量：

```bash
MODEL_DIR=/path/to/pretrained_model \
DATASET_DIR=/path/to/dataset \
STANDALONE_ROOT=/path/to/internvla-a1 \
OUTPUT_DIR=/path/to/output \
DEVICE=cuda \
DTYPE=bfloat16 \
NUM_SAMPLES=2 \
SEED=1234 \
./run_qwena1_real_validation.sh
```

输出会写到 `OUTPUT_DIR`，默认目录是：

```bash
/home/zhuyangkun/data/vllm_a1/qwena1_validation_outputs
```

当前默认数据目录是：

```bash
/home/zhuyangkun/data/vllm_a1/pick_marker_pen_inference_rollouts_v30
```

## 分步执行

### 1. fake-input smoke

这个阶段不依赖真实 checkpoint 加载逻辑，只验证：

- pipeline 能初始化
- fake input 能跑真实 `QwenA1` 主路径
- direct 和 registry 两条链路输出一致

命令：

```bash
./run_qwena1_fake_infer.sh \
  --mode both \
  --device cuda \
  --dtype bfloat16 \
  --batch-size 1 \
  --seed 1234
```

判定：

- 输出里 `comparison.match` 应该为 `true`
- `results[*].summary.mode` 应该是 `no_checkpoint_policy`

### 2. integrated direct vs registry

这个阶段验证真实 checkpoint 下，仓内两种实例化方式结果一致。

命令：

```bash
./run_qwena1_integrated_infer.sh \
  --mode both \
  --model-dir /path/to/pretrained_model \
  --dataset-dir /path/to/dataset \
  --device cuda \
  --dtype bfloat16 \
  --num-samples 2 \
  --seed 1234
```

判定：

- `comparisons[*].match` 都应该为 `true`
- direct 和 registry 的 `action_sha256` 应完全一致

### 3. integrated vs standalone 对齐

这个阶段是最关键的功能对齐验证。建议先用最保守配置：

- `attn_implementation=eager`
- 不开 `regional_compile`

命令：

```bash
./compare_qwena1_integrated_vs_standalone.sh \
  --standalone-root /path/to/internvla-a1 \
  --model-dir /path/to/pretrained_model \
  --dataset-dir /path/to/dataset \
  --device cuda \
  --dtype bfloat16 \
  --num-samples 3 \
  --seed 1234 \
  --attn-implementation eager \
  --output-json compare_eager.json
```

判定：

- `raw_mean_abs_diff` 应非常接近 `0`
- `raw_max_abs_diff` 应非常小
- `physical_mean_abs_diff` 应非常接近 `0`
- `physical_max_abs_diff` 应非常小

如果这里不对齐，后面的加速验证先不要继续。

### 4. attention backend 验证

在 `eager` 对齐通过后，再逐步验证：

```bash
./run_qwena1_integrated_infer.sh \
  --mode both \
  --model-dir /path/to/pretrained_model \
  --dataset-dir /path/to/dataset \
  --device cuda \
  --dtype bfloat16 \
  --num-samples 2 \
  --seed 1234 \
  --attn-implementation sdpa
```

```bash
./run_qwena1_integrated_infer.sh \
  --mode both \
  --model-dir /path/to/pretrained_model \
  --dataset-dir /path/to/dataset \
  --device cuda \
  --dtype bfloat16 \
  --num-samples 2 \
  --seed 1234 \
  --attn-implementation flash_attention_2
```

判定：

- direct / registry 仍然一致
- 没有报 attention backend 不支持
- 如果要进一步看数值对齐，再用 compare 脚本把对应 backend 单独跑一遍

### 5. regional compile 验证

最后再打开 `regional_compile`：

```bash
./compare_qwena1_integrated_vs_standalone.sh \
  --standalone-root /path/to/internvla-a1 \
  --model-dir /path/to/pretrained_model \
  --dataset-dir /path/to/dataset \
  --device cuda \
  --dtype bfloat16 \
  --num-samples 3 \
  --seed 1234 \
  --attn-implementation eager \
  --enable-regional-compile \
  --output-json compare_eager_regional_compile.json
```

判定：

- 仍然能完成推理
- 和 standalone 的 diff 不应明显变坏
- 如果要看性能收益，需要你在真机上配合 profiler 或外部计时

## 推荐执行顺序

建议严格按这个顺序：

1. fake-input smoke
2. integrated direct vs registry
3. integrated vs standalone eager 对齐
4. `sdpa`
5. `flash_attention_2`
6. `regional_compile`

不要一上来同时开 `flash_attention_2 + regional_compile`，否则一旦有 diff 或崩溃，不容易定位来源。

## 结果文件说明

一键脚本默认会生成这些文件：

- `fake_input_smoke.json`
- `integrated_direct_vs_registry.json`
- `compare_eager.json`
- `compare_eager.stdout`
- `integrated_sdpa.json`
- `integrated_flash_attention_2.json`
- `compare_eager_regional_compile.json`
- `compare_eager_regional_compile.stdout`

## 当前边界

当前仓库这边已经完成：

- 代码接入
- fake-input 脚本
- integrated 推理脚本
- integrated-vs-standalone 对齐脚本
- 真机验证一键脚本

当前还没有完成的只有真实 CUDA 环境下的实际执行和结果采样，这一步需要在目标机器上完成。
