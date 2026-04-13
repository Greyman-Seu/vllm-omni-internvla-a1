# QwenA1 Acceleration TODO

目标：在保持当前仓库统一风格的前提下，继续把 `QwenA1` 做成标准的仓内模型实现，而不是只保留 standalone 兼容层。

原则：

- 优先复用本仓库已有机制，不额外发明一套新框架
- 优先对齐本仓库现有 diffusion/omni 模型的共性做法
- `Bagel` 只作为局部参考样本，不作为唯一风格模板
- 先做低风险高收益项，再做侵入式优化
- 每一步都保证 integrated 版本和 standalone 基线可对齐

## P0 已完成

- [x] 将 standalone 核心模块迁入 [vllm_omni/diffusion/models/qwena1](/home/zhuyangkun/vllm-omni-internvla-a1/vllm_omni/diffusion/models/qwena1)
- [x] 注册 `QwenA1Pipeline` 到 [vllm_omni/diffusion/registry.py](/home/zhuyangkun/vllm-omni-internvla-a1/vllm_omni/diffusion/registry.py)
- [x] 新增 integrated 推理脚本和 integrated-vs-standalone 对齐脚本
- [x] 去掉推理路径里对 attention 的硬编码 `eager`，改成配置化入口
- [x] 给 Qwen3-VL text/vision 宿主模块补 `_repeated_blocks`
- [x] 在 pipeline 中接入 `regional_compile`
- [x] 提供 `AutoWeightsLoader` 风格的 `load_weights(...)` 入口

## P1 立即执行

- [x] 细化 profiler target，不只记录 pipeline 包装层
  参考 [vllm_omni/diffusion/models/bagel/pipeline_bagel.py](/home/zhuyangkun/vllm-omni-internvla-a1/vllm_omni/diffusion/models/bagel/pipeline_bagel.py#L275)
  当前 `QwenA1Pipeline` 已经补到真实 policy 路径，覆盖这些 target：
  - `run_batch_sample_actions`
  - `_policy.model.sample_actions`
  - `_policy.model.embed_prefix`
  - `_policy.model.embed_middle`
  - `_policy.model.denoise_step`
  当前状态：代码已落地，真实耗时占比还需要 CUDA 环境下采样验证。

- [x] 统一 attention backend 配置入口
  当前已经在 [vllm_omni/diffusion/models/qwena1/config.py](/home/zhuyangkun/vllm-omni-internvla-a1/vllm_omni/diffusion/models/qwena1/config.py) 保留字段，并在 [vllm_omni/diffusion/models/qwena1/pipeline_qwena1.py](/home/zhuyangkun/vllm-omni-internvla-a1/vllm_omni/diffusion/models/qwena1/pipeline_qwena1.py) 统一从 `custom_pipeline_args` 注入，再由 `QwenA1.set_attention_implementation(...)` 下发到 text/vision/gen/act 各组件。
  当前状态：代码已落地，`eager/sdpa/flash_attention_2` 的真实默认值选择仍需要可用 CUDA 环境实测。

- [x] 接入 `regional_compile` 并完成静态组织
  参考 [vllm_omni/diffusion/compile.py](/home/zhuyangkun/vllm-omni-internvla-a1/vllm_omni/diffusion/compile.py#L11)
  当前已经给 Qwen3-VL text/vision 宿主模块补 `_repeated_blocks`，并在 pipeline 中对 visual、language_model、gen_expert、act_expert 做了局部 compile 接入。
  当前状态：静态接入已完成，但收益和稳定性仍需要真实环境压测。

## P1.5 向仓库统一风格做工程收敛

- [x] 把真实 checkpoint 加载进一步向仓库统一加载风格收敛
  可参考 [vllm_omni/diffusion/models/bagel/pipeline_bagel.py](/home/zhuyangkun/vllm-omni-internvla-a1/vllm_omni/diffusion/models/bagel/pipeline_bagel.py#L648) 的 `load_weights()` 设计
  当前 `QwenA1Pipeline.load_weights(...)` 已补：
  - key 归一化规则
  - shape mismatch 过滤
  - 顶层前缀兼容
  - `AutoWeightsLoader` 风格入口
  当前状态：已经可以在仓库统一加载路径上工作，component 映射这类更深层兼容暂未展开。

- [x] 在 pipeline 内明确“真实路径”和“fake 路径”的状态
  参考仓库现有 pipeline 的组织方式，避免 runtime 模式隐式分叉太多
  当前已经增加统一 runtime mode 标记和日志，区分：
  - `real_checkpoint_available`
  - `real_checkpoint_loaded`
  - `real_unloaded_policy`
  - `no_checkpoint_policy`
  当前状态：代码已落地，fake input 已移到入口脚本里构造，pipeline 内部不再分叉 fake 逻辑。

## P2 高收益但需要真实环境验证

- [x] 优化 `embed_middle` 的 Cosmos 路径
  代码位置：[vllm_omni/diffusion/models/qwena1/model.py](/home/zhuyangkun/vllm-omni-internvla-a1/vllm_omni/diffusion/models/qwena1/model.py#L378)
  当前已经先做了低风险优化，因为这里很可能是大头之一：
  - 插值到 `256x256`
  - Cosmos encode
  - conv 投影
  已完成内容：
  - 输入本身已经是 `256x256` 时跳过重复 `interpolate`
  当前状态：更激进的 cache / autocast 方案仍待真实 profiling 决定。

- [x] 优化 suffix loop 的 `denoise_step`
  代码位置：[vllm_omni/diffusion/models/qwena1/model.py](/home/zhuyangkun/vllm-omni-internvla-a1/vllm_omni/diffusion/models/qwena1/model.py#L520)
  已完成内容：
  - 引入 `SuffixStaticContext`
  - 预计算 suffix attention mask
  - 预计算 suffix position ids
  - 复用 state 投影结果
  当前状态：已完成一轮低风险静态优化，进一步 caching/近似复用要看真实瓶颈再定。

- [ ] 补 `torch.autocast` 策略
  `Bagel` 在生成主路径上明确用了 autocast，见 [pipeline_bagel.py](/home/zhuyangkun/vllm-omni-internvla-a1/vllm_omni/diffusion/models/bagel/pipeline_bagel.py#L616)
  `QwenA1` 现在主要靠 `to(dtype)`，还没有明确的 autocast 边界
  建议：
  - 在 integrated 推理路径上补可控 autocast
  - 保持 `action_out_proj` 和需要 FP32 的层继续留在 FP32
  目标：减小显存和 matmul 成本，同时不破坏数值对齐

## P3 第二阶段优化，先不默认做

- [ ] TeaCache / cache-dit 风格的 QwenA1 专用缓存
  不直接照搬 DiT 方案
  只考虑针对 suffix `act_expert` 反复调用的路径设计专用 extractor
  目标：在多步推理时减少重复 block 计算

- [ ] Layer-wise offload
  参考 [vllm_omni/diffusion/offloader/layerwise_backend.py](/home/zhuyangkun/vllm-omni-internvla-a1/vllm_omni/diffusion/offloader/layerwise_backend.py#L17)
  这更偏“降显存”而不是“提吞吐”
  适合在模型太大、显存不够时接入

- [ ] Sequence Parallel
  当前不要做
  原因：
  - `QwenA1` 还没有 `_sp_plan`
  - 改造成本高
  - 只有多卡长序列场景才明显值

- [ ] 组件量化
  当前不要做
  原因：
  - 当前主体还是 HF 模块
  - 不像 `Bagel` 那样已经深度接入 vLLM 的并行线性层

## 推荐执行顺序

1. profiler 细化
2. attention backend 实测
3. regional compile 实测
4. `load_weights(...)` 收敛到 Bagel 风格
5. Cosmos / suffix hotspot 优化
6. 再决定要不要做 cache/offload

## 验证命令

先验证 integrated 链路：

```bash
./run_qwena1_integrated_infer.sh \
  --mode both \
  --device cuda \
  --dtype bfloat16 \
  --attn-implementation flash_attention_2 \
  --enable-regional-compile
```

再验证和 standalone 的对齐：

```bash
./compare_qwena1_integrated_vs_standalone.sh \
  --device cuda \
  --dtype bfloat16 \
  --attn-implementation eager
```

对齐通过后，再逐步打开：

1. `--attn-implementation sdpa`
2. `--attn-implementation flash_attention_2`
3. `--enable-regional-compile`

每次只开一个变量，保留 diff 结果。
