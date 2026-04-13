# QwenA1 修改说明

这次改动已经从“最小 fake 骨架”升级为“仓内自持的 standalone 迁移版”。目标是让后续删除 `internvla-a1` 目录后，当前仓库依然保留完整的 QwenA1 运行链，同时保留一套与旧 standalone 同步对齐的验证脚本。

## 1. 迁入 standalone 核心实现

已将 standalone 的核心模块 vendoring 到 [vllm_omni/diffusion/models/qwena1](/home/zhuyangkun/vllm-omni-internvla-a1/vllm_omni/diffusion/models/qwena1)：

- [config.py](/home/zhuyangkun/vllm-omni-internvla-a1/vllm_omni/diffusion/models/qwena1/config.py)
- [constants.py](/home/zhuyangkun/vllm-omni-internvla-a1/vllm_omni/diffusion/models/qwena1/constants.py)
- [cosmos.py](/home/zhuyangkun/vllm-omni-internvla-a1/vllm_omni/diffusion/models/qwena1/cosmos.py)
- [dataset.py](/home/zhuyangkun/vllm-omni-internvla-a1/vllm_omni/diffusion/models/qwena1/dataset.py)
- [modeling_qwen3_vl.py](/home/zhuyangkun/vllm-omni-internvla-a1/vllm_omni/diffusion/models/qwena1/modeling_qwen3_vl.py)
- [model.py](/home/zhuyangkun/vllm-omni-internvla-a1/vllm_omni/diffusion/models/qwena1/model.py)

其中：

- `config.py` 保留了 standalone 的配置字段与 `from_pretrained(...)` 逻辑。
- `model.py` 保留了 standalone 的真实 `QwenA1` / `StandaloneQwenA1Policy` 实现。
- `dataset.py` 保留了数据集读取、视频解码、Qwen3-VL processor 构造和 batch collate 逻辑。
- `modeling_qwen3_vl.py` 一并迁入，避免后续依赖外部 standalone 目录。

当前不再保留独立 fake 模型实现。仓内的 smoke 路径统一改成“真实 `QwenA1` 模型 + fake input”，避免再维护一套偏离主链路的假实现。

## 2. 重做 `QwenA1Pipeline`

[pipeline_qwena1.py](/home/zhuyangkun/vllm-omni-internvla-a1/vllm_omni/diffusion/models/qwena1/pipeline_qwena1.py#L34) 现在收敛成仓内统一风格的 wrapper，职责只保留：

- 真实 standalone 路径：
  当 `od_config.model` 指向包含 `model.safetensors` 的 checkpoint 目录时，pipeline 会走 `StandaloneQwenA1Policy`。

- 真实 standalone 路径的加载、包装和优化注入
- `run_batch_sample_actions(...)` 这类仓内 batch 入口
- profiler / attention backend / regional compile / load_weights 这些仓库级能力

fake input 不再在 pipeline 内部构造，统一下放到入口脚本。

当前 pipeline 主要新增/保留了这些能力：

- `get_or_create_policy(...)`
  懒加载真实 standalone policy，并统一做 `device` / `dtype` 对齐。

- `run_batch_sample_actions(...)`
  直接对 batch tensor 调用真实 standalone 的 `sample_actions(...)`，这是仓内脚本现在使用的主入口。

- `forward(req)`
  会明确返回错误说明，提示改用 `run_batch_sample_actions(...)` 或对齐脚本。这是因为当前 QwenA1 是策略模型，不是标准文本 prompt 扩散请求路径。

## 3. 注册到 diffusion registry

修改了 [vllm_omni/diffusion/registry.py](/home/zhuyangkun/vllm-omni-internvla-a1/vllm_omni/diffusion/registry.py#L90)：

- 在 `_DIFFUSION_MODELS` 中新增 `QwenA1Pipeline`
- 在 `_DIFFUSION_POST_PROCESS_FUNCS` 中新增 `get_qwena1_post_process_func`

这样可以通过 `initialize_model(od_config)` 直接实例化 `QwenA1Pipeline`，不需要深层 import。

## 4. 新增 integrated 推理脚本

新增了仓内 integrated 版推理入口：

- [run_qwena1_integrated_infer.py](/home/zhuyangkun/vllm-omni-internvla-a1/run_qwena1_integrated_infer.py#L1)
- [run_qwena1_integrated_infer.sh](/home/zhuyangkun/vllm-omni-internvla-a1/run_qwena1_integrated_infer.sh#L1)

这个脚本直接使用仓内 `QwenA1Pipeline`，支持：

- `--mode direct`
  直接实例化 `QwenA1Pipeline`

- `--mode registry`
  通过 `initialize_model(...)` 走 diffusion registry

- `--mode both`
  同时跑 direct 和 registry 两条链路，并比较输出 `sha256`

另外已经补了两个和加速相关的开关：

- `--attn-implementation`
  可显式指定 attention 实现，例如 `eager`、`sdpa` 或环境支持时的 `flash_attention_2`

- `--enable-regional-compile`
  开启仓库风格的 `regional_compile`，只编译重复 block，不整图编译整个策略逻辑

这个脚本的作用是验证仓内集成链路本身，而不是对齐外部 standalone。

## 5. 新增 integrated-vs-standalone 对齐脚本

新增了：

- [compare_qwena1_integrated_vs_standalone.py](/home/zhuyangkun/vllm-omni-internvla-a1/compare_qwena1_integrated_vs_standalone.py#L1)
- [compare_qwena1_integrated_vs_standalone.sh](/home/zhuyangkun/vllm-omni-internvla-a1/compare_qwena1_integrated_vs_standalone.sh#L1)

这个脚本会：

- 使用仓内 vendored 的 dataset / config / pipeline 构造输入
- 同时加载：
  - 仓内 `QwenA1Pipeline`
  - 旧 `internvla-a1/qwena1_standalone` 的 `StandaloneQwenA1Policy`
- 在相同输入、相同 noise 下比较两边输出
- 输出 raw action 与 physical action 的 mean/max absolute diff

同样支持：

- `--attn-implementation`
- `--enable-regional-compile`

这套脚本的作用是：

- 在删除旧 standalone 目录前，先确认当前仓内迁移版与旧实现对齐
- 后续如果 integrated 版本继续演化，可以用这个脚本做回归比较

## 6. 补充通用脚本辅助

新增 [qwena1_infer_common.py](/home/zhuyangkun/vllm-omni-internvla-a1/qwena1_infer_common.py)，统一放了一些推理脚本共用的方法：

- `tensor_dtype(...)`
- `select_indices(...)`
- `make_shared_noise(...)`
- `tensor_sha256(...)`
- 默认 `model_dir` / `dataset_dir`

## 7. 加速相关保留点

这次迁移时，刻意保留了 standalone 里对加速友好的写法，而不是简化掉：

- [model.py](/home/zhuyangkun/vllm-omni-internvla-a1/vllm_omni/diffusion/models/qwena1/model.py) 中保留了 `torch.compile(self.sample_actions, mode=config.compile_mode)` 入口
- 保留了 prefix / middle / suffix 分段执行与 KV cache 复用逻辑
- 保留了 `use_cache=True` 的前缀/中段路径
- 保留了独立的 `modeling_qwen3_vl.py`，后续如果要接更激进的 attention/缓存优化，可以直接在仓内演进

在此基础上，这一轮又补了三项更贴近本仓库的优化入口：

- `attn_implementation`
  不再在 prefix / middle / suffix 路径里硬编码回 `eager`，允许按配置切换 attention 实现

- `regional_compile`
  给 Qwen3-VL text/vision 宿主模块声明了 `_repeated_blocks`，并在 pipeline 里按仓库现有方式对重复 block 做局部编译

- `load_weights(...)`
  pipeline 现在提供了更像仓库其他模型的 `AutoWeightsLoader` 风格入口，便于后续进一步替换掉直接 `from_pretrained` 的路径

也就是说，这次不是简单“把脚本搬过来”，而是把后续做模型加速需要继续改的主体也一并迁入了仓内。

## 8. 当前验证状态

已完成：

- 新增和修改的 Python 文件全部通过 `py_compile`
- registry 注册关系已静态检查通过
- integrated 推理脚本、对齐脚本、fake 脚本都已落地

未完成：

- 真实前向验证

原因是当前机器可见的 `python` 环境里没有 `torch`，因此无法在这台机器上实际执行推理或比较脚本。当前阻塞是运行环境依赖，不是代码语法或导入层面的错误。

## 9. 当前推荐使用方式

如果你要跑仓内集成版本：

```bash
./run_qwena1_integrated_infer.sh --mode both --device cpu --dtype float32
```

如果要显式测试加速相关开关：

```bash
./run_qwena1_integrated_infer.sh \
  --mode both \
  --device cuda \
  --dtype bfloat16 \
  --attn-implementation flash_attention_2 \
  --enable-regional-compile
```

如果你要在删除旧 standalone 前做一次结果对齐：

```bash
./compare_qwena1_integrated_vs_standalone.sh --device cpu --dtype float32
```

如果只想做 fake input smoke：

```bash
./run_qwena1_fake_infer.sh --mode both --device cpu --dtype float32
```
