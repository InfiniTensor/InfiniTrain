# InfiniTrain Linear 提速路线图（arch-conditional，含 DeepGEMM SM90 可行性结论）

> **文档定位**：本文件是**设计 / 路线图**，不是实现记录。遵循「本阶段不改动代码」的约束——每个 Tier 都是后续可**独立执行**的单元，落地前需另行确认。
>
> **关联文档**：本文是 [`deepgemm_sm90_bf16_gemm_design.md`](./deepgemm_sm90_bf16_gemm_design.md) 与 [`deepgemm_sm100_bf16_gemm_design.md`](./deepgemm_sm100_bf16_gemm_design.md) 的**应用侧续篇**——前两篇讲「DeepGEMM 的 kernel 怎么工作」，本篇回答「能否 / 如何用它改进 InfiniTrain 的 Linear，以及在真实硬件上最划算的提速路径是什么」。
>
> **行号约定**：所有源码坐标以撰写时的仓库状态为准，末尾附[源码坐标索引](#八源码坐标索引)便于快速跳转。

---

## 一、摘要（结论先行）

- **直接替换不可行（sm_120 / RTX 5090）**：SM90 kernel 的核心指令 `wgmma.mma_async` 是 sm_90a 专属；DeepGEMM host 侧 `bf16_gemm_nt` 对 `arch_major == 12` 直接 `DG_HOST_UNREACHABLE`；整条路径还被 `#if DG_TENSORMAP_COMPATIBLE` 包裹。**三重硬阻断**。
- **即使在 sm_90a（H100 / H800）也是低 ROI**：cuBLAS 的 BF16 NT GEMM 在 Hopper 上本就会调用 wgmma kernel；DeepGEMM 的增量优势主要在 **FP8 / grouped / MoE**，标准稠密 Linear 收益有限。
- **架构冲突（与硬件无关）**：DeepGEMM 是 `torch + pybind11 + Python` 的运行期 JIT 扩展；InfiniTrain 是**零 torch 依赖**的纯 C++/CUDA 框架。接入需抽离 device 头 + JIT runtime 并重接裸指针（约 700 行胶水 + 维护 fork + 运行期 nvcc + cache 目录 + 首调延迟）。
- **推荐做法**：走**分层提速路线**（Tier 0–2 两架构通用、零 / 极少 device 代码即可拿到大部分收益），并在 Dispatcher 的 `Gemm` 注册点引入 **arch-conditional 后端抽象**（Tier 3），把「sm_90a 可选 DeepGEMM、sm_120 走 cuBLASLt / CUTLASS」统一收敛到 L1 kernel 层，**框架层保持零 `#ifdef`**。

---

## 二、可行性结论：DeepGEMM SM90 能否替换 Linear

### 2.1 关键事实（均已核对源码）

| 维度 | 事实 | 出处 |
| --- | --- | --- |
| 目标硬件 | 构建架构 `75;80;90;120`；部署目标 RTX 5090 = **sm_120**（`arch_major = 12`） | [`CMakeLists.txt:112`](../CMakeLists.txt)；DeepGEMM `device_runtime->get_arch_major()`（[`gemm.hpp:430`](../third_party/DeepGEMM/csrc/apis/gemm.hpp) 调用） |
| SM90 路径门禁 | `if (arch_major==9) sm90_bf16_gemm(); else if (==10) sm100_...; else DG_HOST_UNREACHABLE("Unsupported architecture")` | [`gemm.hpp:430-437`](../third_party/DeepGEMM/csrc/apis/gemm.hpp) |
| 编译期门禁 | 整个 `bf16_gemm_nt` 及其 arch 分派都在 `#if DG_TENSORMAP_COMPATIBLE` 之内 | [`gemm.hpp:403`](../third_party/DeepGEMM/csrc/apis/gemm.hpp) |
| 指令级阻断 | `wgmma.mma_async.sync.aligned.m64nNk16` **仅 sm_90a**；kernel 需 `--gpu-architecture=sm_90a` 编译 | SM90 详设 §2.7 / §13.2 |
| 依赖冲突 | DeepGEMM = `find_package(Torch REQUIRED)` + `pybind11_add_module(_C)` + 「real compilation is done via JIT」；InfiniTrain 顶层 CUDA 库只链 `glog / CUDA::cudart / CUDA::cublas`，无 torch / DeepGEMM | [`DeepGEMM/CMakeLists.txt:1,17,18,28`](../third_party/DeepGEMM/CMakeLists.txt)；[`CMakeLists.txt:133-140`](../CMakeLists.txt) |
| Linear 现状 | 全部经 `Dispatcher::Call("Gemm")` → `cublasGemmEx(..., CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT)` | [`linear.cu:113-133`](../infini_train/src/kernels/cuda/linear.cu)；[`gemm.cu:45-75`](../infini_train/src/kernels/cuda/common/gemm.cu) |
| 布局天然匹配 | `Linear::Forward` 恒传 `transpose=true`；weight=`[out,in]=[N,K]` K-major、input=`[M,K]` K-major、output=`[M,N]` N-major | [`autograd/linear.cc:16`](../infini_train/src/autograd/linear.cc)；[`linear.cu:56-61`](../infini_train/src/kernels/cuda/linear.cu) |

### 2.2 结论

- **sm_120：不可行。** 即使绕过 host 门禁，`wgmma` 也无法为 sm_120 汇编 / 执行。DeepGEMM 自己在非 sm90/sm100 场景是**回退到 cuBLASLt**（`smxx_cublaslt.hpp`，`CUBLAS_COMPUTE_32F_FAST_TF32`）——这恰好反证了正解方向：**稠密 Linear 的正解是 cuBLASLt / CUTLASS，而非 DeepGEMM 的 wgmma kernel。**
- **sm_90a：技术可接、性价比低。** 布局上 InfiniTrain Linear 是 DeepGEMM 的**最佳路径**（`transpose=true` 恰好是原生 `bf16_gemm_nt` 的 K-major NT），但相对 cuBLAS-BF16 的增量收益小，还要背 torch/JIT 依赖。仅在 **FP8 / grouped-MoE** 等特殊形态才值得（见[附录四](#四deepgemm-sm90-条件性集成设计仅-sm_90a附录)）。

---

## 三、推荐路线：分层提速（Tier 0–4）

> 设计原则：**收益从高到低、改动从小到大**排布；Tier 0–2 两架构通用且几乎不碰 device 代码，Tier 3 提供可移植的路由骨架，Tier 4 是 sm_120 的兜底。

### Tier 0 — BF16 端到端打通（两架构，最高 ROI，device 代码 ≈ 0）

**现状（已核对）**：
- `linear.cu` 已支持 BF16：`ToCudaDataType` 映射 `kBFLOAT16 → CUDA_R_16BF`（[`gemm.cu:33-34`](../infini_train/src/kernels/cuda/common/gemm.cu)）；`BiasCopyKernel` 已对 FP32/BF16 双分派（[`linear.cu:75`](../infini_train/src/kernels/cuda/linear.cu)）；`LinearBackwardBias` 的 BF16 指针**已正确用** `static_cast<const nv_bfloat16*>`（[`linear.cu:316-318`](../infini_train/src/kernels/cuda/linear.cu)）。
  > ⚠️ **记忆订正**：历史记录中「`LinearBackwardBias` BF16 cast bug」在当前代码**已修复**，Tier 0 **无需再动此处**。
- **autocast 命中 Linear 已确认**：`kOpCastPolicyMap` 中 `{"Linear", CastPolicy::kLowerPrecision}` 存在且大小写正确（[`autocast.h:47`](../infini_train/include/autocast.h)）；autograd 边界在 [`function.cc:181-183`](../infini_train/src/autograd/function.cc) 对每个输入调用 `Autocast(type_, t)`，`type_="LinearFunction"` 经 `GetBaseOpName` 剥后缀得 `"Linear"`（[`autocast.h:13-19,92`](../infini_train/include/autocast.h)）→ 命中 → cast 到 `autocast_dtype`。**原「待核对缺口 #2」核查通过。**
- **BF16 路径端到端可达**：示例程序用**双参** `AutocastGuard(device.type(), dtype)`，`dtype` 来自 `--dtype` 标志（GPT-2 仅接受 `float32/bfloat16`，[`gpt2/main.cc:91,263-270,472`](../example/gpt2/main.cc)）。故 `--dtype=bfloat16` 即可让 Linear 走 BF16 → cuBLAS BF16。

**待补缺口 / 风险点**：
1. **backward 三处 dtype promotion FIXME hack**：`output_dtype = (compute_dtype==kBFLOAT16) ? kFLOAT32 : compute_dtype`（[`linear.cu:172`](../infini_train/src/kernels/cuda/linear.cu) / `244` / `296`）——即「BF16 计算、FP32 输出」。需确认这是**数值稳定性权宜**还是可在 autocast/autograd 修正后去除。
2. **CUDA 默认 autocast dtype 是 FP16，而非 BF16（潜在坑，本会话新发现）**：`kDeviceDefaultDtype = {kBFLOAT16(CPU), kFLOAT16(CUDA)}`（[`autocast.h:74-77`](../infini_train/include/autocast.h)）。
   - 单参 `AutocastGuard(device_type)`（[`autocast.h:163-164`](../infini_train/include/autocast.h)）在 CUDA 上会默认 **FP16**。
   - 但 Linear 的 bias 路径**只覆盖 FP32/BF16、不含 FP16**：`BiasCopyKernel` 分派表是 `DispatchCudaFunc<kFLOAT32, kBFLOAT16>`（[`linear.cu:75`](../infini_train/src/kernels/cuda/linear.cu)），`LinearBackwardBias` 的 `switch` 也只有 `kFLOAT32/kBFLOAT16` 两个 `DISPATCH_CASE`（[`linear.cu:308-321`](../infini_train/src/kernels/cuda/linear.cu)）。
   - **结论**：CUDA 的 FP16 默认值与 Linear 实际 dtype 覆盖**不一致**——一旦有人用单参 guard 触发 FP16 autocast，带 bias 的 Linear 会在分派处失配。**Tier 0 应显式统一走 BF16**（示例已用双参 guard 规避，但默认值本身是漂移隐患）。
3. **策略表维护漂移的旁证**：`{"Layernorm", kFP32}`（[`autocast.h:70`](../infini_train/include/autocast.h)，小写 n）与 `LayerNorm` 模块 `kType="LayerNormFunction"`→`GetBaseOpName`→`"LayerNorm"`（大写 N）**大小写不匹配**，导致 BF16 下 LayerNorm **静默走 BF16 而非提升到 FP32**（已确认缺陷）。这与 Linear 无关，但说明 `kOpCastPolicyMap` 的**双份手工维护**（数组 `kLowerPrecisionOps`/`kFP32Ops` + map）存在漂移风险——建议 Tier 0 顺带由数组生成 map 或加启动期一致性 `CHECK`。

**改动点**：`autocast` 策略表校验 / 生成化 + CUDA 默认 dtype 与 Linear 覆盖对齐 + `linear.cu` backward 输出 dtype 逻辑复核；**无新增 kernel**。

**预期收益**：sm_120 上 cuBLAS BF16 走 Blackwell tensor core，端到端约 **1.3–1.5×**；sm_90a 上自动走 wgmma。

### Tier 1 — TF32 快路径（两架构，FP32 存储模式）

**现状**：`gemm.cu` 硬编码 `const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;`（[`gemm.cu:63-65`](../infini_train/src/kernels/cuda/common/gemm.cu)，注释明说「always use CUBLAS_COMPUTE_32F」）；`GemmParams`（[`gemm.h:21-46`](../infini_train/src/kernels/common/gemm.h)）**无 compute 精度字段**；全仓库 InfiniTrain 侧无 TF32。

**改动点**：`GemmParams` 增 `compute_type`（或 `allow_tf32` bool）→ `Gemm()` 据此在 `CUBLAS_COMPUTE_32F` 与 `CUBLAS_COMPUTE_32F_FAST_TF32` 间分支；由 Linear/autocast 或全局开关驱动。**约 20 行。**

**预期收益**：需要 FP32 存储但想要 tensor core 加速时约 **1.3×**；精度损失介于 FP32 与 BF16 之间。

### Tier 2 — cuBLASLt + bias epilogue 融合（两架构）

**现状**：bias **未融合**——forward 用独立 `BiasCopyKernel` + `beta=1`（[`linear.cu:68-82`](../infini_train/src/kernels/cuda/linear.cu)），backward-bias 用 `ReduceColumnsKernel`（[`linear.cu:138-156,305-321`](../infini_train/src/kernels/cuda/linear.cu)）。相比 epilogue 融合多一次全量读写。

**改动点**：`Gemm()` 增 cuBLASLt 分支，用 `CUBLASLT_EPILOGUE_BIAS` 把 bias 并入 GEMM epilogue；可参考 DeepGEMM 的 `smxx_cublaslt.hpp`（arch-agnostic）。**约 300 行**（含 desc / heuristic 管理）。

**预期收益**：带 bias 的 Linear（如 GPT-2）省一次带宽往返，约 **+3–5%**。

### Tier 3 — Dispatcher 层 arch-conditional 后端抽象（可移植核心）

**目标**：把「按架构选 GEMM 后端」收敛到 **L1 kernel 层**（`Gemm` 注册点，[`gemm.cu:87`](../infini_train/src/kernels/cuda/common/gemm.cu)），框架层**零 `#ifdef`**，符合 L0–L7 架构约束。

**设计**：在 `Gemm()`（或新增 `GemmBackend` 选择器）内按**运行期** compute capability 路由：
- `sm_120` → cuBLASLt（Tier 2）或 CUTLASS Sm120（Tier 4）
- `sm_90a` → cuBLAS BF16（默认）；特殊形态可选 DeepGEMM `bf16_gemm_nt`（[附录四](#四deepgemm-sm90-条件性集成设计仅-sm_90a附录)）

**arch 获取**：复用 InfiniTrain 自己的 DeviceGuard / compute-capability 查询，**不引入** DeepGEMM 的 `device_runtime`。

**改动量**：约 50 行；优先级 **P1（与 Tier 0/1 并行）**——它本身不提速，但让后续后端可插拔。

### Tier 4 — CUTLASS Sm120 collective builder（sm_120 专用，兜底）

**触发条件**：Tier 0–2 触顶且需要定制 shape / 融合 epilogue。

**改动点**：新增 CUTLASS 3.x Sm120 collective 的 GEMM 后端，注册进 Tier 3 抽象。**约 500 行。** CUTLASS 已提供经验证的 Sm120 builder，无需自研 kernel。

---

## 四、DeepGEMM SM90 条件性集成设计（仅 sm_90a，附录）

> 若未来部署 H100/H800 且**确需** DeepGEMM（例如 FP8 / grouped / MoE），集成设计如下。**默认不推荐**用于标准稠密 Linear。

- **布局映射（天然匹配）**：InfiniTrain Linear 恒 `transpose=true`（[`autograd/linear.cc:16,29`](../infini_train/src/autograd/linear.cc)），三个 GEMM 对应 DeepGEMM：
  - **forward**：`output[M,N] = input[M,K] @ weight[N,K]^T` → `bf16_gemm_nt(input, weight, output)`（A/B 均 K-major、D N-major，命中最佳路径，[`gemm.hpp:404-438`](../third_party/DeepGEMM/csrc/apis/gemm.hpp)）
  - **dgrad**（`LinearBackwardInput`）：`bf16_gemm_nn` / `bf16_gemm_tn`（按 transpose 组合，内部 transpose 适配，[`gemm.hpp:440-454`](../third_party/DeepGEMM/csrc/apis/gemm.hpp)）
  - **wgrad**（`LinearBackwardWeight`）：`bf16_gemm_tn`（reduce over bs）
- **bias**：DeepGEMM **无 bias 融合** → 仍需保留独立 kernel（此处无收益；融合要走 Tier 2 cuBLASLt）。
- **dtype**：a/b 必须 BF16；d 可 BF16 或 FP32（[`gemm.hpp:421-423`](../third_party/DeepGEMM/csrc/apis/gemm.hpp)）→ 与现有 backward「BF16 计算、FP32 输出」的 promotion hack **兼容**。
- **接入方式（二选一）**：
  1. 抽离 `deep_gemm/include` device 头 + `csrc/jit` runtime，重接到 InfiniTrain 裸指针 + 自建 `CUtensorMap`（避免 torch），**维护内部 fork**。
  2. 直接依赖 libtorch/ATen（与 InfiniTrain 零 torch 架构**冲突，不推荐**）。
- **成本 / 风险**：约 700 行胶水 + 运行期 nvcc / JIT cache + 首调延迟 + fork 维护；且 H100 上相对 cuBLAS-BF16 收益有限。**建议仅在 FP8 / grouped / MoE 时启用。**

---

## 五、优先级与预期收益

| Tier | 适用硬件 | 改动量 | 预期收益 | 主要风险 | 优先级 |
| --- | --- | --- | --- | --- | --- |
| **0** BF16 端到端 | 两者 | ≈0（+校验 / hack 清理） | **1.3–1.5×** 端到端 | 收敛性（需 `compare_loss` 验证）；CUDA 默认 FP16 与 Linear 覆盖不一致 | **P0** |
| **1** TF32 快路径 | 两者 | ~20 行 | ~1.3×（FP32 模式） | 精度损失 | **P1** |
| **2** cuBLASLt + bias | 两者 | ~300 行 | +3–5%（带 bias） | 复杂度 | **P2** |
| **3** 后端抽象 | 两者 | ~50 行 | 使能可移植路由 | 抽象设计 | **P1（与 0/1 并行）** |
| **4** CUTLASS Sm120 | sm_120 | ~500 行 | shape 相关 | 集成工作量 | **P3（兜底）** |
| ✗ DeepGEMM SM90 | 仅 sm_90a | ~700 行 + fork | 低（cuBLAS 已用 wgmma） | 硬件 / 依赖 / JIT | **不推荐（除 FP8/MoE）** |

> **注**：理论 GEMM 提速会被 host 侧瓶颈（CrossEntropy CPU 归约、StackForward HtoD）稀释，**端到端收益低于 device-time 估计**。

---

## 六、验证方法

- **数值**：每 Tier 后用 [`scripts/compare_loss.py`](../scripts/compare_loss.py) 对齐 loss 曲线（dtype 改动尤其需要）。
- **性能**：用 **NVTX 标注 + nsys** 测 GEMM 段 device-time（**不要用 `PROFILE_MODE`**，其逐算子同步会使吞吐虚降约 2×）。
- **后端确认**：nsys 里核对 cuBLAS 是否切到 **tensor-core 变体**（BF16 / TF32 kernel 名），而非 FP32 SIMT。

---

## 七、假设与边界

- 假设部署目标为 **RTX 5090（sm_120）**；若实际为 H100/H800，Tier 0–3 同样适用，[附录四](#四deepgemm-sm90-条件性集成设计仅-sm_90a附录)的 DeepGEMM 集成才进入可选范围。
- 撰写时 shell **无 GPU**（`nvidia-smi` 无设备），硬件结论基于**构建配置**（[`CMakeLists.txt:112`](../CMakeLists.txt)）与既有实测记忆，未在本文成文时在线复测 compute capability。
- **本阶段不产出代码改动**；任一 Tier 的落地为后续独立任务。

---

## 八、源码坐标索引

| 主题 | 文件 : 行 | 说明 |
| --- | --- | --- |
| 构建架构 | [`CMakeLists.txt:112`](../CMakeLists.txt) | `75;80;90;120` |
| InfiniTrain CUDA 库依赖 | [`CMakeLists.txt:133-140`](../CMakeLists.txt) | 只链 glog / cudart / cublas，无 torch |
| DeepGEMM 依赖 & JIT | [`DeepGEMM/CMakeLists.txt:1,17,18,28`](../third_party/DeepGEMM/CMakeLists.txt) | JIT / pybind11 / Torch / `_C` |
| DeepGEMM arch 门禁 | [`gemm.hpp:403,430-437`](../third_party/DeepGEMM/csrc/apis/gemm.hpp) | `#if DG_TENSORMAP_COMPATIBLE` + arch_major 分派 |
| `bf16_gemm_nt/nn/tn` | [`gemm.hpp:404-454`](../third_party/DeepGEMM/csrc/apis/gemm.hpp) | NT 最佳路径 + transpose 适配 |
| Linear transpose 恒 true | [`autograd/linear.cc:16,29`](../infini_train/src/autograd/linear.cc) | weight=`[N,K]` PyTorch 约定 |
| LinearForward / bias | [`linear.cu:56-133`](../infini_train/src/kernels/cuda/linear.cu) | BiasCopyKernel + Gemm |
| backward promotion hack | [`linear.cu:172,244,296`](../infini_train/src/kernels/cuda/linear.cu) | BF16→FP32 输出 |
| LinearBackwardBias（已修） | [`linear.cu:308-321`](../infini_train/src/kernels/cuda/linear.cu) | BF16 用 `nv_bfloat16*` |
| Gemm compute_type 硬编码 | [`gemm.cu:63-69`](../infini_train/src/kernels/cuda/common/gemm.cu) | `CUBLAS_COMPUTE_32F`（Tier 1 改动点） |
| `ToCudaDataType` | [`gemm.cu:29-41`](../infini_train/src/kernels/cuda/common/gemm.cu) | FP32 / BF16 / FP16 |
| `GemmParams`（无 compute 字段） | [`gemm.h:21-46`](../infini_train/src/kernels/common/gemm.h) | Tier 1 需扩字段 |
| autocast 策略表 | [`autocast.h:45-77`](../infini_train/include/autocast.h) | Linear@47 / Layernorm@70 / 默认 dtype@74-77 |
| autocast 边界 | [`function.cc:181-183`](../infini_train/src/autograd/function.cc) | autograd 边界按 op 名 cast |
| 示例 autocast 用法 | [`gpt2/main.cc:91,263-270,472`](../example/gpt2/main.cc) | `--dtype` → 双参 `AutocastGuard` |
