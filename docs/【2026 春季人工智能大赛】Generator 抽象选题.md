# 中期报告要求

- 比赛期：2026/05/18 - 2026/07/12 （56 天）
  - 比赛期第一天，即 2026/05/18 的 12:00 公布赛题
  - 每个参赛小组需在特定的截止日期前提交中期报告。中期报告需包含：
    - 小组名称与所有成员
    - 所选的赛题
    - 截至撰写时各个赛题的进度与完成情况
    - 中期报告没有字数要求，简明扼要、表述清晰即可
    - 提交要求与方式：
      - 报告文件命名要求：<小组名称>_<赛道名称>_中期报告.pdf
      - 提交方式：InfiniTensor 网站上提交，提交流程后续补充

# 【2026 春季人工智能大赛】Generator 抽象选题

## 一、题目背景

在深度学习训练与推理框架中，随机数生成器（Generator）是支撑参数初始化、随机采样、Dropout、噪声注入等功能的重要基础设施。当前框架中相关能力尚不完善，缺少统一的 Generator 抽象、后端实现以及全局随机种子管理机制，导致随机相关算子的行为与主流框架存在差距，也难以满足后续功能扩展需求。

当前已完成 Generator 基础设施、初始化算子与随机张量创建算子的首轮接入，进度如下：

- [x] 已建立统一的 Generator 抽象，支持随机数状态、种子设置、状态获取与恢复；
- [x] 已实现 CPU 与 CUDA 后端 Generator，并保持统一的调用接口；
- [x] 已建立按设备维护的默认 Generator；未显式传入 generator 时，初始化算子使用目标 Tensor 所在设备的默认随机源；
- [x] 已提供全局 `manual_seed(uint64_t)` 入口，重置 CPU 及所有 CUDA 默认 Generator；
- [x] 已接入 `Rand`、`Randn`，支持显式/默认 Generator、CPU/CUDA 及 `FLOAT16`、`BFLOAT16`、`FLOAT32`、`FLOAT64`；
- [ ] Dropout 与 CUDA Graph RNG 语义尚未实现。

为补齐这一基础能力，本题目要求参赛者参考主流深度学习框架的设计思路，完成一套可扩展的随机数生成器基础设施，实现 CPU 与 CUDA 后端的 Generator，并支持基于 Generator 的随机数生成与种子控制能力。

## 二、题目目标

参赛者需要围绕框架内的随机数生成体系，完成以下几个方面的工作：

- [x] 1. 设计并实现统一的 Generator 抽象接口，支持随机数状态管理、种子设置、状态获取与恢复等基础能力。
- [x] 2. 分别实现 CPU 与 CUDA 后端对应的 GeneratorImpl，使不同设备上的随机数生成具备统一的调用方式。
- [x] 3. 建立全局随机种子控制机制，支持通过统一入口固定随机种子；已接入的初始化算子和随机张量创建算子可稳定复现。
- [x] 4. 改造初始化类随机算子和随机张量创建算子，使其在未显式传入 generator 参数时，自动使用目标设备的默认全局 Generator。
- [x] 5. 已完成 CPU/CUDA 初始化、`Rand/Randn`、多 dtype 分布和参数校验的基本行为、可复现性和接口语义冒烟验证；仓库内正式测试仍待补充。

## 三、任务拆解

### 1. Generator 抽象与状态管理设计

在框架中设计统一的 Generator 抽象层，用于屏蔽不同设备后端的随机数实现差异。该抽象应支持：

- [x] 设置随机种子，如 `ManualSeed()` / `SetCurrentSeed()`；
- [x] 获取当前种子或初始种子，如 `Seed()` / `InitialSeed()`；
- [x] 获取当前随机数状态，如 `GetState()`；
- [x] 恢复随机数状态，如 `SetState(state)`；
- [x] 查询所属设备类型；
- [x] 提供必要的线程安全或状态访问保护机制。

实现时需注意区分 seed 与 state：seed 只用于初始化随机序列，state 表示随机序列当前推进到的位置。随机算子每次使用 Generator 后，应推进其内部状态，避免重复生成相同随机序列。

推荐设计思路：

- [x] 提供用户侧可持有的 `Generator` 句柄类；
- [x] 底层以 `GeneratorImpl` 作为多态实现基类；
- [x] CPU / CUDA 分别派生对应实现；
- [x] 公共接口不暴露 `std::mt19937`、curand、Philox 等后端细节；
- [x] 为后续其他平台 Generator 接入预留扩展空间。

### 2. CPU / CUDA 后端 Generator 实现

基于现有设备后端，分别实现：

- [x] `CPUGeneratorImpl`
- [x] `CUDAGeneratorImpl`

两者应保持统一接口语义，但内部状态组织方式可以不同，且**不要求 CPU 与 CUDA 在相同 seed 下生成逐元素一致的随机结果**。

实现时需重点关注：

- [x] CPU 与 CUDA 随机数状态如何组织与保存；CPU 保存 mt19937 状态与正态缓存，CUDA 保存 Philox seed 与 subsequence；
- [x] CUDA 侧按设备维度维护独立默认 Generator；
- [x] `GetState()` / `SetState()` 的状态序列化、反序列化与格式校验；CPU/CUDA 均校验 CPU `UINT8` state，CPU 额外校验 footer 与 mt19937 反序列化；
- [x] 多次随机调用后，随机序列连续推进；CPU 推进 mt19937，CUDA 为每次 kernel 保留 Philox subsequence；
- [x] 同一 seed、同一设备、同一调用顺序下结果可复现；
- [x] 初始化算子和 `Rand/Randn` 已支持显式 Generator 和默认 Generator 两条调用路径。

CUDA 后端建议在状态中考虑 seed、offset、counter 等信息，避免不同 kernel 调用之间随机序列重叠。此部分需要保证功能正确、语义清晰、状态可管理、接口一致，为后续随机算子和分布式扩展提供基础。

### 3. 默认 Generator 与统一随机种子入口

建立框架级默认 Generator 管理机制，支持两种使用方式：

- 用户显式传入某个 Generator；
- 用户不传入 Generator，系统自动使用当前设备上的默认 Generator。

该机制应包括：

- [x] CPU 默认全局 Generator；
- [x] 各 CUDA 设备对应的默认 Generator；
- [x] CPU/CUDA 均提供获取默认 Generator 的后端入口；
- [x] 统一的全局随机种子设置入口：`manual_seed(uint64_t)`；
- [x] 设置全局 seed 时同步重置 CPU 与所有 CUDA 默认 Generator 的状态。

要求：

- [x] 多次获取同一设备默认 Generator 时，应返回同一随机状态来源；（static 局部变量，单例模式）
- [x] 不同 CUDA 设备的默认 Generator 相互独立；
- [x] 用户显式传入 Generator 时，初始化算子使用用户指定 Generator；
- [x] 未传入 Generator 时，初始化算子使用目标设备默认 Generator；
- [x] 设置统一 seed 后，已接入的参数初始化和 `Rand/Randn` 随机采样可稳定复现；
- [ ] Dropout 等尚未接入的随机算子仍待完成。

这一部分是本题的核心验收点。随机数系统最终服务于训练可复现性，因此必须通过统一入口稳定控制随机行为。

### 4. 随机相关算子接入改造

改造框架中已有的随机相关使用，使其接入 Generator 机制。建议至少覆盖两类场景：

- [x] 初始化类随机算子：`Uniform`、`Normal`、`KaimingUniform` 已接入 Generator，并按 Tensor device 分发 CPU/CUDA kernel；
- [x] 通用随机张量创建算子：`Rand`、`Randn` 已接入 Generator，支持 CPU/CUDA 与四种浮点 dtype；Dropout 仍待接入；

改造后的算子应满足：

- [x] 已接入初始化算子和 `Rand/Randn` 支持显式传入 Generator；
- [x] 已接入初始化算子和 `Rand/Randn` 未传入 Generator 时，自动使用目标设备默认 Generator；
- [x] 已接入初始化算子和 `Rand/Randn` 的随机结果受统一 `manual_seed` 入口控制；
- [x] 已接入初始化算子和 `Rand/Randn` 多次调用会推进 Generator 状态；
- [x] 已接入初始化算子和 `Rand/Randn` 在同一 seed、同一调用顺序下可复现。

不要求一次性改造所有随机算子，但应至少完成一个初始化类算子和一个训练期随机算子，以证明 Generator 机制能够贯通框架层与算子层。

当前已完成初始化类算子与通用随机张量创建算子接入；Dropout 等训练期随机算子仍是后续工作。

### 5. 测试与对齐验证

为验证实现结果，需要补充系统化测试。测试重点应放在"语义是否正确"和"是否可复现"，而不是与 PyTorch 逐元素数值一致。

当前已在 `/tmp` 完成 CPU/CUDA 构建，以及 Generator 状态恢复、初始化、`Rand/Randn`、四种浮点 dtype 分布、参数校验和显式/默认 Generator 路径的冒烟验证；按项目约定尚未新增仓库内测试，因此以下正式测试清单仍未完成。

建议至少覆盖以下内容：

#### （1）接口一致性测试

验证 CPU / CUDA Generator 是否支持统一接口，包括：

- [ ] seed 设置与获取；
- [ ] state 获取与恢复；
- [ ] 设备类型查询；
- [ ] 默认 Generator 获取；
- [ ] 显式 Generator 与默认 Generator 两条调用路径。

#### （2）种子可复现测试

验证统一随机种子入口是否生效，包括：

- [ ] 同一 seed 下，多次运行参数初始化结果一致；
- [ ] 同一 seed 下，多次运行 Dropout mask 一致；
- [ ] 不同 seed 下结果应发生变化；
- [ ] 同一 seed、同一调用顺序下结果应一致。

#### （3）状态恢复测试

验证 `GetState()` / `SetState()` 是否真正恢复随机序列，包括：

- [ ] 保存 state；
- [ ] 继续生成一段随机结果；
- [ ] 恢复 state；
- [ ] 再次生成随机结果；
- [ ] 验证恢复后的结果与原序列对齐。

CPU 后端必须覆盖该测试；CUDA 后端若已实现，也应覆盖。

#### （4）默认 Generator 行为测试

验证随机算子是否正确使用默认 Generator，包括：

- [ ] 不传 Generator 时，是否使用当前设备默认 Generator；
- [ ] CPU tensor 是否使用 CPU 默认 Generator；
- [ ] CUDA tensor 是否使用对应 CUDA 设备默认 Generator；
- [ ] 显式传入 Generator 时，是否不会误用默认 Generator。

#### （5）主流框架语义对齐验证

可选取典型场景与 PyTorch 进行语义对比，包括：

- [ ] 手动设置 seed 后结果可复现；
- [ ] Dropout 等随机运算在同 seed 下可复现；
- [ ] 随机张量生成接口支持显式 Generator 与默认 Generator；
- [ ] state 保存与恢复语义一致。

需要明确：本项目不要求底层随机算法与 PyTorch 逐 bit 一致，也不要求 CPU 与 CUDA 随机结果彼此一致。验收重点是接口语义一致、可复现逻辑一致、默认 Generator 行为一致。

## 四、提交报告要求

除代码外，参赛者应提交以下内容作为技术报告：

### 1. 功能正确性验证

需要提供上述 Generator 基础设施功能验证结果

### 2. 对齐性与行为分析报告

需要说明本项目实现与主流框架的对齐情况。建议从以下角度展开：

- [ ] 参考了哪些主流框架设计；
- [ ] 当前实现与 PyTorch 在接口语义上的对应关系；
- [ ] 哪些行为做到基本一致；
- [ ] 哪些行为暂未完全对齐，以及原因分析；
- [ ] 当前实现范围与后续可扩展方向。

此部分重点不是要求一切细节完全一致，而是要求留档明确设计思路，能够清晰说明：

- [ ] 自己实现了什么；
- [ ] 为什么这样设计；
- [ ] 与主流框架相比的设计异同及原因；
- [ ] 后续演进路径是什么。

### 3. 测试与可复现性说明

需要提供完整测试脚本与说明，保证 reviewer 可在相同环境下复现主要结果。

## 五、验收要求

### 验收要求

项目需满足以下基本要求：

- [ ] 代码以 PR 形式提交，结构清晰，具备基本可 review 性；
- [x] 完成统一 Generator 抽象设计；
- [x] 实现 CPU 与 CUDA Generator；
- [x] 建立 CPU 与各 CUDA device 的默认 Generator 管理机制；
- [x] 提供统一的全局随机种子设置入口；
- [x] 初始化算子支持默认 Generator 与显式 Generator 两种使用方式；
- [x] 支持 Generator 状态的获取与恢复（get_state / set_state）；
- [ ] 至少改造一类初始化算子和一类框架内随机数生成调用，使其接入 Generator；
- [ ] 提供测试或运行日志，验证随机行为具备基本可复现性（同 seed 一致、不同 seed 不同、状态恢复有效）。

### 加分项

在满足验收要求的基础上，具备以下内容可作为加分项：

- [ ] 设计清晰，接口与实现分层合理，便于后续扩展
- [ ] 测试覆盖充分，包含 seed、state、默认/显式 generator、跨设备等关键场景；
- [ ] 调用处改造完全，接口风格统一；
- [ ] 与 PyTorch 的接口语义和行为分析完整，报告质量高
- [ ] PR 经过完整 review 流程，达到可合入标准。

## 参考链接

- https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#generator
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/core/Generator.h
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/CPUGeneratorImpl.cpp
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/CUDAGeneratorImpl.cpp
- https://docs.pytorch.org/docs/stable/notes/randomness.html#pytorch-random-number-generator
