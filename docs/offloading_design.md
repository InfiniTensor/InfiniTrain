# HybridDeviceOptimizer CPU Offloading 设计

## 目标与边界

`HybridDeviceOptimizer`（下文简称 HDO）是 `Optimizer` 的包装器，用于把当前 rank 本地拥有的一部分 optimizer parameter、gradient staging buffer 和 optimizer state 放到 CPU，并在设备参数与 CPU inner parameter 之间执行 D2H/H2D 同步。

它不参与 DP shard 的划分或通信。与 `DistributedOptimizer` 组合时，分层如下：

```text
DistributedOptimizer
└── local shard OptimizerCreator
    └── HybridDeviceOptimizer
        ├── gpu_optimizer_: 未 offload 的 local shard
        └── cpu_optimizers_: 已 offload 的 local shard
```

`DistributedOptimizer` 仍负责 reduce-scatter、local shard 归属、parameter all-gather 和 full parameter 同步；HDO 只看到 `shard_params_`，因此不会重新解释全局 DP 分片语义。

设计参考 Megatron Core 的 `hybrid_optimizer.py`，但使用 InfiniTrain 的 `Tensor`、`OptimizerCreator`、`DeviceGuardImpl`、`Stream` 和 `Event` 抽象，不依赖 CUDA 类型判断。

## 构造接口

HDO 构造参数与 Megatron 保持对应关系：

```cpp
HybridDeviceOptimizer(
    params,
    offload_fraction,
    cpu_optimizer_creator,
    gpu_optimizer_creator,
    param_update_in_fp32,
    pin_cpu_grads,
    pin_cpu_params,
    overlap_cpu_optimizer_d2h_h2d);
```

`OptimizerCreator` 对应 Megatron 的 `cpu_optimizer_cls` / `gpu_optimizer_cls`。creator 封装学习率、betas 和 eps 等构造参数，HDO 只负责提供每个子 optimizer 应更新的 inner parameters。

Llama3 的 CPU offload 集成固定使用 `param_update_in_fp32=true`，不额外暴露
`use_precision_aware_optimizer` CLI。InfiniTrain 当前尚未提供 Megatron/TransformerEngine 那种可分别配置
master parameter、main grad 和 optimizer state dtype 的完整 precision-aware optimizer；在只有 HDO FP32
inner parameter 语义的阶段，单独暴露该名称会造成误导。

只有实际存在对应分区时才要求 creator 非空：

- `offload_fraction=0` 不要求 CPU creator，行为等价于纯设备 optimizer。
- `offload_fraction=1` 不要求 GPU creator。
- overlap 开启时，每个 CPU inner parameter 单独构造一个 CPU optimizer。
- overlap 关闭时，全部 CPU inner parameters 共享一个 CPU optimizer。

参数按输入顺序和 `NumElements()` 累计选择，直到达到全部非 CPU 参数元素数乘以 `offload_fraction`。
阈值保持为浮点数参与比较，不提前截断为整数，因此任意正数 fraction 都会选择至少第一个完整 parameter。
为避免拆分单个 Tensor，实际 offload 元素数允许超过阈值一个 parameter 的大小，这与 Megatron 的策略一致。

## Original 与 Inner Parameter

HDO 以 original parameter 在 `params_` 中的稳定索引维护映射：

```text
original parameter index
├── original parameter：模型或 distopt local shard 实际引用
├── inner parameter：子 optimizer 实际更新的 Tensor
├── offloaded index：若位于 CPU，指向 OffloadedParam
└── uses_master_param：checkpoint 是否需要保存 FP32 master
```

`ParamMapping` 只描述 original parameter 与 inner parameter 的稳定映射，按 original parameter index
存放在 vector 中：

| 字段 | 含义 | 与 Megatron 的对应关系 |
| --- | --- | --- |
| `inner_param` | 子 optimizer 实际更新的 parameter | 对应 `param_to_inner_param[original]`；对于 CPU copy 同时对应 `gpu_params_map_cpu_copy[original]`，对于 FP32 master 同时对应 `param_to_fp32_param[original]` |
| `offloaded_index` | 指向 `offloaded_params_` 的索引；`size_t(-1)` 表示该 parameter 未 offload | 提供 original parameter 到 CPU buffers 的定位；反向关系由 `CpuOptimizerMetadata::original_indices` 承载 |
| `uses_master_param` | `inner_param` 是需要独立 checkpoint 的 FP32 master parameter | 语义上对应 original parameter 是否存在于 `param_to_fp32_param`；保存为 `hybrid.master_param.<original-index>` |

`OffloadedParam` 不是第二套 parameter 映射，而是单个 offloaded parameter 在 D2H/CPU step/H2D
流水中需要保留的 buffers：

| 字段 | 含义 | 设计原因 |
| --- | --- | --- |
| `cpu_grad` | CPU optimizer 消费的可复用 grad buffer | 对应 Megatron 的 `cpu_copy_map_grad` |
| `cpu_grad_staging` | original grad dtype 的 D2H 中转 buffer | original grad 与 CPU inner parameter dtype 不同时，先 D2H 再在 CPU 上 cast；InfiniTrain 当前的 `Tensor::CopyFrom()` 要求源和目标 dtype 一致 |
| `h2d_staging` | H2D copy 使用的 CPU cast buffer | 异步 copy 完成前保持源 Tensor 存活，避免临时 cast 结果提前释放 |

`cpu_optimizers_` 与 Megatron 一样直接保存 CPU optimizer 对象，即
`std::vector<std::shared_ptr<Optimizer>>`。每个 optimizer 负责的 original parameter indices 以及
`d2h_done`、`h2d_ready`、`h2d_done` 保存在以 `Optimizer *` 为 key 的
`cpu_optimizer_metadata_` 中。vector 持有 optimizer 生命周期，map 中的指针只用于非 owning 查找；
post-step hook 可直接用回调传入的 optimizer 定位 metadata，不需要捕获 vector 下标。
每个 CPU optimizer 的全部 grad D2H 提交后只记录一个 `d2h_done`；post-step hook 在其全部
parameter H2D 前只等待一次 `h2d_ready`，copy 完后只记录一次 `h2d_done`。这与 Megatron
按 CPU optimizer 执行 copy-back 的粒度一致。

Megatron 构造初期最显眼的是 `gpu_params_map_cpu_copy`、`cpu_copys_map_gpu_param` 和
`param_to_fp32_param` 三个 dict，但后续流程还会维护 `param_to_inner_param`、
`inner_param_to_orig_param`、`cpu_copy_map_grad` 以及每个 CPU optimizer 的 event map。InfiniTrain 使用
original parameter index 和两个结构体承载这些关系，避免重复的 Tensor-keyed hash map，但子
optimizer 选参、每参数 CPU step 以及 post-step copy-back 的语义与 Megatron 一致。
`cpu_grad_staging` 和 `h2d_staging` 是由 InfiniTrain 当前跨 dtype copy 接口和异步源 buffer 生命周期
引入的实现字段，不改变 optimizer update 语义。

inner parameter 的选择规则如下：

| 分区 | 参数 dtype | param_update_in_fp32 | inner parameter |
| --- | --- | --- | --- |
| 未 offload | FP32 | 任意 | original parameter |
| 未 offload | BF16/FP16 | true | 同设备 FP32 master copy |
| 已 offload | FP32 | false | CPU FP32 copy |
| 已 offload | 任意 | true | CPU FP32 master copy |
| 已 offload | BF16/FP16 | false | 当前不支持，`LOG(FATAL)` |

创建 CPU copy、FP32 master 和 copy-back staging 时使用 `Detach()->To()`，只转换 parameter data，
不递归复制 original/inner parameter 上已有的 grad。HDO 的 grad buffer 由 step 路径独立管理；这与
Megatron 使用 `detach().clone()` 和 `param.data.copy_()` 的数据语义一致。

未 offload 的 mixed-precision parameter 会把 original grad cast 到设备端 FP32 master grad，GPU optimizer 更新 FP32 master 后，由其 post-step hook copy 回 original dtype。该路径允许 mixed-precision partial offload，不再要求所有参数 100% offload。

`gpu_params` 和 `cpu_params` 只在构造子 optimizer 时作为局部变量存在。类内只保留后续 step、state 和 load 所需的 original index、inner parameter、子 optimizer 及 offload buffer。

## Post-Step Hook 契约

InfiniTrain 在 optimizer 层新增 `OptimizerPostStepHook`，其签名为 `void(Optimizer&)`。它不是 autograd
function hook；`Optimizer::Step()` 是统一入口：

```text
Optimizer::Step()
├── virtual StepImpl()
└── 按注册顺序执行实例 post-step hooks
```

具体 optimizer 只实现 `StepImpl()`，因此任意 `OptimizerCreator` 返回的子类都会由基类统一触发 hooks，不需要各
optimizer 手工调用。一个 optimizer 可以注册多个 hooks，注册顺序即执行顺序。

`RegisterPostStepHook()` 返回 `std::shared_ptr<HookHandle>`。这里对齐 PyTorch optimizer post-step hook 的关键
生命周期语义：hook 保存在 optimizer 实例中，只有显式调用 `HookHandle::Remove()` 才会注销；handle 析构不会注销 hook。
handle 必须在 optimizer 生命周期内使用，optimizer 析构后不得再调用 `Remove()`。

HDO 注册两类 hook：

- CPU optimizer hook：CPU step 完成后，把对应 CPU inner parameters H2D copy 回 original parameters。
- GPU optimizer hook：仅在存在设备端 FP32 master 时，把 master parameters cast/copy 回 mixed-precision original parameters。

hook 由子 optimizer 持有，HDO 注册的 lambda 捕获 HDO 指针和 original parameter indices。子 optimizer 由 HDO
独占生命周期，hook 只会在 HDO 的 `StepImpl()` 中触发，因此不会晚于 HDO 生命周期执行。HDO 不需要保存
handle，因为 handle 析构不会移除已注册的 hook。

## Step 与 Overlap

overlap 模式的执行顺序为：

```text
current device stream       D2H stream          CPU              H2D stream
        │                       │                 │                    │
cast GPU master grads           │                 │                    │
record grads-ready event ──────>wait              │                    │
        │                       ├─ copy grad[0] ──>│                    │
GPU optimizer step             ├─ event[0]        │                    │
        │                       ├─ copy grad[1] ──>│                    │
        │                       └─ event[1]        │                    │
        │                           host wait[0] ─>CPU optimizer[0]    │
        │                                             post hook ─────>H2D param[0]
        │<──────────────────────────────────────────── wait event[0]──┤
        │                           host wait[1] ─>CPU optimizer[1]    │
        │                                             post hook ─────>H2D param[1]
        │<──────────────────────────────────────────── wait event[1]──┘
next forward
```

具体保证如下：

- D2H stream 在拷贝前等待 current stream 的 grads-ready event。
- 每个 CPU optimizer 有一个 D2H done event，在其全部 parameter 的 grad copy 提交后记录。
- GPU optimizer 在 current stream 上更新未 offload parameters，可与 D2H copy 和 CPU step 重叠。
- CPU optimizer post-step hook 在 H2D stream 上发起 copy。
- current stream 等待每个 H2D done event，下一轮 forward 必然看到更新后的参数。
- `DistributedOptimizer` 随后的 all-gather 会在 current stream 记录 ready event，因此也会传递 H2D 依赖到通信 stream。
- 析构前同步 D2H/H2D streams，再销毁 events、streams 和 host buffers。

overlap 关闭时，D2H/H2D stream 直接引用构造时的 current stream；overlap 开启时才创建独立
stream。两种配置共用同一套 copy 和 event 依赖路径，与 Megatron 的 stream 选择语义一致。

所有 stream/event/memcpy 操作通过 original parameter 对应后端的 `DeviceGuardImpl` 完成。当前 overlap 只支持一个非 CPU device；多设备输入会在对应分支 `LOG(FATAL)`，代码中保留了 `// TODO(zbl): ...`。

`DeviceGuardImpl::ExchangeStream()` 与 RAII `StreamGuard` 提供线程局部的 current-stream 上下文。HDO 在 D2H/H2D
scope 内切换 current stream 后直接调用 `Tensor::CopyFrom()`；copy direction、裸指针和后端 `MemcpyAsync()` 均由
Tensor/runtime 层处理，HDO 只编排 stream/event 依赖。CUDA 后端的 `GetStream()` 返回当前线程在对应 device
上的 stream，`StreamGuard` 析构时恢复原 stream。

PyTorch `dst_stream.wait_stream(src_stream)` 在后端也是在 source stream 上 record event，再让
destination stream wait event。InfiniTrain 直接使用 `DeviceGuardImpl::EventRecord()` 和
`DeviceGuardImpl::StreamWaitEvent()` 表达同一语义，无需再增加 `WaitStream()` 封装。

## Gradient 与 ZeroGrad

每个 offloaded parameter 有一个可复用的 CPU grad buffer。original grad dtype 与 CPU inner dtype 不同时，另有一个 original dtype 的 CPU staging buffer，D2H 完成后再在 CPU cast 到 inner dtype。

`ZeroGrad(set_to_none)` 会：

- 清理 original parameters 的 grad。
- 清理 CPU inner parameters 的 grad 引用。
- 清理 GPU FP32 master parameters 的 grad。
- `set_to_none=false` 时清零复用的 CPU grad buffers。
- 不修改 inner parameter data 或 optimizer state。

InfiniTrain 当前没有 `decoupled_grad` / `main_grad` 字段，因此梯度来源为 `Tensor::grad()`。后续引入对应框架能力时，应按 Megatron 顺序优先读取 main/decoupled grad；代码不应在 HDO 内另造一套梯度所有权。

## StateDict 与恢复

HDO 不向 `Optimizer` 基类增加 per-parameter state 接口，而是把各子 optimizer 的既有 `StateDict()` 视为不透明
整体并增加命名空间：

```text
hybrid.device.<child-state-key>
hybrid.cpu.<cpu-optimizer-index>.<child-state-key>
hybrid.master_param.<original-index>   # 仅 FP32 master 路径
hybrid.layout                           # overlap 模式和每个 original parameter 的分区
```

这使 HDO 可以包装任意实现了现有 `StateDict()` / `LoadStateDict()` 的 optimizer，不要求基类或具体 optimizer
理解 HDO 的 original/inner parameter 映射。设备状态使用 `device` 前缀以保持后端无关；代码内部仍沿用与
Megatron 对应的 `gpu_optimizer_` 命名。

HDO 的 `StateDict()` / `LoadStateDict()` 支持下述同布局恢复流程。rebase 到当前 master 后，
框架的 sharded checkpoint planner 仅识别 `adam.*`，尚不接受 `hybrid.*`；因此完整的
`Checkpoint::Save/Load` 暂时不能保存或恢复 HDO optimizer state。
Llama3 在初始化模型前检查：开启 `optimizer_cpu_offload` 且指定 `save` 时必须设置
`save_optimizer_state=false`；指定 `load` 时必须设置 `load_optimizer_state=false`。
否则对应分支直接 `LOG(FATAL)`，并保留 `// TODO(zbl): ...`。直接调用底层 checkpoint API
传入 HDO 仍会被 planner 拒绝，不能以绕过 Llama3 检查的方式获得完整状态恢复。
Llama3 通过 named creator 适配器把 model/local shard parameters 交给 HDO，普通 Adam 仍保留
master 的 named-parameter 路径。HDO 接入新版 checkpoint planner 留待后续实现。

支持范围如下：

| 操作 | 当前行为 |
| --- | --- |
| Llama3 保存模型，`save_optimizer_state=false` | 支持；同时保存 training metadata 和 scheduler state，不保存 Adam state 或 FP32 master |
| Llama3 加载模型，`load_optimizer_state=false` | 支持；加载后调用 HDO `ReloadModelParams()`，从模型刷新 CPU copies 和设备 FP32 masters；Adam state 从头初始化 |
| Llama3 保存或加载 HDO optimizer state | 暂不支持，初始化模型前报错 |
| 直接调用 HDO `StateDict()` / `LoadStateDict()` | 支持同布局的内存状态恢复；不等价于已接通 sharded checkpoint planner |

`ReloadModelParams()` 不修改 Adam 的动量和步数。Llama3 在构造子 optimizer 的 creator 中保留 HDO
引用，所以即使外层是 `DistributedOptimizer`，加载模型后也能刷新对应的 local shard copies。
模型恢复仍沿用现有的训练步数、数据进度和 scheduler 恢复规则。对于 BF16 模型，只加载模型时
FP32 master 从 BF16 权重重建，无法恢复未保存的 FP32 尾数，因此不承诺等价于完整 optimizer resume。

直接调用 HDO 状态接口的恢复顺序如下：

1. 调用方先将模型状态写回 original parameters。
2. HDO 比较 `hybrid.layout` 与当前 CPU/device 分区及 overlap 模式。
3. 布局一致时，调用 `ReloadModelParams()` 刷新 CPU copies 和设备 FP32 masters。
4. 如果 checkpoint 含 `master_param`，再恢复完整 FP32 master，避免从 BF16/FP16 模型参数重建造成精度丢失。
5. 去掉 `hybrid.device` / `hybrid.cpu.<index>` 前缀后，分别调用对应子 optimizer 的 `LoadStateDict()`。

由于 InfiniTrain 当前 optimizer state 不是 PyTorch 那样以 parameter 为 key，HDO 无法在不知道具体 optimizer
状态结构的前提下安全重排 state。当前 checkpoint 必须以相同的 offload partition 和 overlap 布局恢复；不一致时
对应分支会 `LOG(FATAL)`。未来若框架原生提供 parameter-keyed optimizer state，再实现跨布局恢复。

## 当前未支持能力

以下差异来自 InfiniTrain 当前框架能力，暂不在 HDO 内重复实现：

- **param_groups**：没有 per-group lr、weight decay、betas、eps 和动态属性同步；当前只有全局 learning rate。
- **AdamW/fused optimizer**：llama3 当前使用 InfiniTrain Adam creator；其他子 optimizer 若需支持状态恢复，必须实现现有 `StateDict()` / `LoadStateDict()` 接口，不要求新增 per-parameter state API。
- **pinned host allocator**：`pin_cpu_params=true` 或 `pin_cpu_grads=true` 且实际发生 offload 时会 `LOG(FATAL)`。`offload_fraction=0` 时 pin 参数不产生作用。
- **真正的 pageable-memory overlap 性能保证**：当前测试使用 pageable CPU buffers。stream/event 顺序正确，但异步传输是否与计算重叠由后端决定；Megatron 的典型性能依赖 pinned memory。
- **runtime capability query**：`DeviceGuardImpl` 尚无 stream/event/non-blocking-copy capability API，暂时无法自动降级。增加该 API 后应在 HDO 构造时选择 overlap 或同步路径。
- **已有 CPU original parameters**：当前 HDO 面向设备上的 model/local-shard params，混合 CPU/device original inputs 会 `LOG(FATAL)`。
- **多非 CPU device overlap**：需要每个 device 独立 stream/event 资源。
- **decoupled/main grad**：Tensor 尚无对应字段。
- **dummy_step**：InfiniTrain Adam 在构造时立即分配 state，不需要 Megatron 用于初始化 lazy state 的 `dummy_step`；未来接入 lazy optimizer 时再增加统一接口。
- **optimizer pre-step/global hook**：当前只实现 HDO 所需的实例级 post-step hook；PyTorch 的 pre-step hook 和全局 optimizer hook 留待后续扩展。
- **跨 offload/overlap 布局恢复**：现有 optimizer state 缺少 parameter identity；当前通过 `hybrid.layout` 检查并拒绝不一致布局。

## Llama3 验证

以下为 rebase 前的历史训练验证记录，未作为当前 master 上的训练回归结果；原临时日志路径不保证仍可用。

验证环境：

- GPU：NVIDIA A100-SXM4-80GB
- 模型：llama3 1B FP32 checkpoint
- 数据：tinyshakespeare
- micro batch：4
- 单卡 total batch：256
- DP=2 total batch：512
- optimizer：InfiniTrain Adam，lr `1e-5`
- 每个场景：10 training steps

loss 以同拓扑 naive Adam 为 baseline，逐 step 比较 10 个值：

| 场景 | 10-step 最大 loss 差 | baseline peak used | offload peak used | 节省显存 |
| --- | ---: | ---: | ---: | ---: |
| 单卡，100% offload，同步 copy | 2e-6 | 23270 MB | 11837 MB | 11433 MB |
| 单卡，100% offload，D2H/H2D overlap | 2e-6 | 23270 MB | 11837 MB | 11433 MB |
| 单卡，50% offload，D2H/H2D overlap | 2e-6 | 23270 MB | 17426 MB | 5844 MB |
| ZeRO-1，DP=2，100% offload，同步 copy | 1e-6 | 19533 MB | 13817 MB | 5716 MB |
| ZeRO-1，DP=2，100% offload，D2H/H2D overlap | 1e-6 | 19533 MB | 13817 MB | 5716 MB |

每组日志均包含完整 `steps=10/10`。测试日志位于容器 `/tmp/infinitrain_hdo_refactor/`。

## 自动化测试

HDO CUDA 测试覆盖：

- FP32 full offload 同步与 overlap 对齐 naive Adam。
- overlap 时每参数 CPU optimizer，以及对应的 `hybrid.cpu.<index>`、master 和 layout 状态 key。
- `offload_fraction=0` 等价设备 optimizer，且不要求 CPU creator。
- BF16 partial offload：CPU FP32 master 与 GPU FP32 master 更新一致。
- 真实的 model-only `Checkpoint::Save/Load` 后刷新 HDO：FP32/BF16、0%/50%/100% offload、overlap 开关共 12 种组合，后续三步参数与加载后新建的设备 FP32 master optimizer 对齐。
- `ReloadModelParams()` 刷新 CPU/device master parameters，同时保留已有 Adam 动量和步数。
- 直接调用 `StateDict()` / `LoadStateDict()`，验证同布局的参数副本刷新与 mixed-precision master 恢复。
- 多个 post-step hooks 均由 `Optimizer::Step()` 统一触发。

布局不一致时的 `LOG(FATAL)` 分支尚无专门的拒绝测试。上述 HDO 状态接口测试不经过 sharded
optimizer checkpoint planner；通用 checkpoint 测试通过也不代表 HDO 完整 optimizer checkpoint
已经受支持。DistOpt 组合的模型恢复仍需要独立的多卡端到端验证。
