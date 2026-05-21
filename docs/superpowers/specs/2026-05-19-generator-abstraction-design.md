# Generator 抽象设计 (Spec)

> 任务：2026 春季人工智能大赛 — Generator 抽象选题。
> 目标：为 InfiniTrain 引入统一的随机数生成器抽象，对齐 PyTorch `c10::GeneratorImpl` 设计，提供 CPU/CUDA 后端、默认 Generator 池、全局 seed 入口，并改造现有随机算子接入。
>
> **状态：spec frozen（2026-05-21）** — 多轮 review 闭合后可按 §2.4 分期实施。

## 0. 修订记录

| 日期 | 说明 |
|------|------|
| 2026-05-19 | 初稿 |
| 2026-05-20 | **对照代码库评审修订**：修正 `nn::function` 命名与文件路径；补全 `InitialSeed` 语义、`default_generator` 存储、CUDA stream 绑定、Dispatcher 多返回值类型；修正验证命令与测试 CMake；补充 macOS 链接、迁移行为说明、多 GPU 测试写法；明确 LoRA dropout 不在本任务接线范围。 |
| 2026-05-21 | **第二轮 review 闭合**：§4.6 `manual_seed` 窄锁域 + `ResetAllSeeds`；§5.0 `Module::Train()` 不跳过 `__pp_*`（含 `TransformerModel` 说明）；修正引用笔误（`Device::DeviceType`、`fill.cu`/`elementwise.cu`、tuple 模式）；补拷贝语义、`FromImpl` 友元、`Reseed`/`Seed` 分后端、`Create()` CPU index 校验；CUDA 随机 kernel 统一 stream 绑定；`test_ops_dropout` 增加 `Eval()` 用例。**标为 frozen。** |

## 1. 背景与现状

### 1.1 当前 RNG 现状

- **CPU**：`std::mt19937` 在 `infini_train/src/nn/init.cc` 共出现 6 处（line 35 文件作用域 `static std::mt19937 gen(kRandomSeed=42)`、line 39/116/131 三处函数签名、line 46/138 两处 OMP `local_gen`）；`std::mt19937` 还泄漏在公共头 `include/tensor.h:151`、`include/nn/init.h:18,45,48`。
- **CUDA**：完全空白，无 curand / Philox / curandState 任何使用。所有"CUDA 随机"实际是 CPU 生成 → `cudaMemcpy` 到 device。
- **Dropout**：不存在（`lora_config.h:15` 注释 "not implemented yet"）。
- **OMP 路径双 bug**：
  1. *默认分支*（`generator == nullopt`）每次调用都用 `kRandomSeed + omp_get_thread_num()` **重新构造** `mt19937`，状态根本不推进；
  2. *显式分支*（`generator` 有值）OMP 多线程**共享同一个 `generator.value()` 引用**做非原子 `dis(generator.value())` 推进，存在数据竞争。
- **公开 ABI**：`std::mt19937` 在公共头出现于 4 处签名（`tensor.h:151` + `init.h:18,45,48`）；下游 `init::Normal/Uniform/KaimingUniform` 调用点共 11 处（`nn/modules/linear.cc:35,39`、`nn/modules/sparse.cc:25`、`nn/lora/lora_linear.cc:60,62`、`nn/lora/lora_parallel_linear.cc:102,104,298,300`、`nn/parallel/tensor_parallel.cc:235,239`），**全部不传 generator**——新签名 `std::optional<Generator> = std::nullopt` 与旧默认行为兼容，调用点无需改动。
- **死代码**：`example/gpt2/checkpoint_loader.cc:31-34` 与 `example/llama3/checkpoint_loader.cc:29-32` 各自声明 `static std::mt19937 gen{kRandomSeed}` 但全文件无引用——本任务直接删除（含 `kRandomSeed` 与 `<random>` include），不替换。
- **测试覆盖**：随机算子无任何测试。
- **现有 TODO**：`infini_train/src/nn/init.cc:24-34` 已有完整 FIXME 注释列出本任务全部需求。

### 1.2 可复用的现有设施

- **Dispatcher 模式**（`include/dispatcher.h:50-82`）：单例 + `(DeviceType, name)` → 函数指针，`REGISTER_KERNEL` 静态注册。注意 `KernelFunction::Call<RetT, ArgsT...>` 用 `reinterpret_cast<RetT(*)(ArgsT...)>` 类型擦除调用，**注册函数签名必须与调用 `Call<>` 的实参类型逐一对齐**。
- **后端注册表**（`include/core/runtime/device_guard.h:199-215`）：`DeviceGuardImplRegistry` 用 `std::unordered_map<DeviceType, std::unique_ptr<DeviceGuardImpl>>`，每 `DeviceType` 一个 impl，`INFINI_TRAIN_REGISTER_DEVICE_GUARD_IMPL` 宏静态注册。**Generator 工厂注册表镜像此模式**。
- **per-GPU 懒初始化缓存**（`src/core/runtime/cuda/cuda_guard_impl.cc:14-22`）：`std::array<std::unique_ptr<...>, kMaxGpus=8>` + `std::array<std::once_flag, kMaxGpus>` 在 CudaGuardImpl 内部缓存 stream/blas handle。**Generator 默认池镜像此模式**（每 CUDA 设备一个默认 Generator）。
- **DeviceGuard RAII**（`include/core/runtime/device_guard.h:161-186`、`src/core/runtime/device_guard.cc`）：CUDA 设备切换已封装。
- **测试基类**（`tests/common/test_utils.h:48`）：`InfiniTrainTest : TestWithParam<Device::DeviceType>` + `INFINI_TRAIN_REGISTER_TEST` 宏。**USE_CUDA=ON 时实例化 CPU + CUDA，USE_CUDA=OFF 时仅 CPU**——任何"必须跨设备"的测试都需要 `ONLY_CUDA()` 守护或改写为单设备可验证。
- **静态构造的链接保障**：根 `CMakeLists.txt:158-179` 的 `link_infini_train_exe()` 用 `-Wl,--whole-archive`（Linux）包住 `infini_train` / `infini_train_cpu_kernels` / `infini_train_cuda_kernels` 三个静态库，所有 `REGISTER_KERNEL` / `INFINI_TRAIN_REGISTER_*` 静态构造保证执行。**新注册器（含 CUDA `.cu`）依赖此标志生效**。
- **CMake 自动收集**：主库 `SRC` 为 `GLOB_RECURSE infini_train/src/*.cc` 并 **排除** `kernels/cpu/`；CUDA 库为 `GLOB_RECURSE infini_train/src/*.cu`（**递归**，`src/core/generator/cuda/*.cu` 与 `src/kernels/cuda/*.cu` 均进入 `infini_train_cuda_kernels`）。新建 `.cc/.cu` 无需改根 CMake，仅测试目录需新建 `CMakeLists.txt`。
- **测试基类设备索引**：`InfiniTrainTest::GetDevice()` 固定 `Device(GetParam(), 0)`（`test_utils.h:50`）。**多 GPU 独立性**（不同 `device.index()`）须在测试内显式构造 `Device(kCUDA, 1)` 等，并 `REQUIRE_MIN_DEVICES(2)`（修订）。

## 2. 设计目标

### 2.1 必达目标

1. 统一 `Generator` 抽象：`ManualSeed/SetCurrentSeed`、`Seed/InitialSeed`、`GetState/SetState`（含格式校验）、`device()`。
2. CPU `CPUGeneratorImpl` 与 CUDA `CUDAGeneratorImpl` 双后端，状态结构可不同。
3. 默认 Generator 池：CPU 一个、每 CUDA 设备一个，懒初始化，`shared_ptr` 共享状态。
4. 全局 seed 入口 `infini_train::manual_seed(uint64_t)`：同步影响所有默认 Generator（CPU + 已初始化 CUDA + 待懒初始化 CUDA 的种子记忆）。
5. 算子改造：现有 `Normal/Uniform/KaimingUniform/Tensor::Uniform` 全部改 `Generator` 签名；新建 `Dropout/Rand/Randn` 完整链路（kernel + autograd + module）。
6. 测试：seed 复现、state 恢复、跨格式拒绝、默认/显式 Generator、Dropout 行为正确性。
7. 对齐报告：与 PyTorch 接口语义对照表、行为差异声明。

### 2.2 显式非目标

- **不要求** CPU 与 CUDA 在相同 seed 下生成逐元素一致的随机结果。
- **不要求** 与 PyTorch 数值 bit 一致（仅语义对齐）。
- **不强制** OMP 并行优化，先正确性优先（单线程 + 锁），并行作为后续可扩展项写入报告。

### 2.3 相对旧代码的行为变化（修订）

实现本 spec 后，下列变化**预期发生**，须在报告 §8 中说明：

| 变化 | 原因 |
|------|------|
| 默认随机序列与当前 `init.cc` 不同 | 引擎由 `mt19937` 改为 `mt19937_64`；去掉 OMP 错误路径后状态正常推进 |
| `ManualSeed(42)` 结果 ≠ 旧版 `kRandomSeed=42` 静态 `gen` | 算法与引擎均变；以新 Generator + kernel 路径为准 |
| CUDA tensor 随机在 device 上生成 | 不再 CPU fill + `MemcpyAsync` H2D |

### 2.4 建议实施分期（修订，非强制顺序）

1. **Phase 1**：Registry + CPU `CPUGeneratorImpl` + `UniformRandom`/`NormalRandom`（CPU）+ `init.cc` 改造 + 基础测试。
2. **Phase 2**：CUDA Philox + CUDA 随机 kernel（含 **stream 绑定**）+ `manual_seed` 全路径。
3. **Phase 3**：`nn::function::Dropout/Rand/Randn` + autograd + `nn::Dropout` 模块 + Dropout 测试。
4. **Phase 4**：对齐报告 `docs/generator-design.md` + 可选 `scripts/check_reproducibility.sh`。

## 3. 架构

### 3.1 类层次

```
infini_train::Generator                       // 用户句柄（值类型，PImpl）
    └── core::GeneratorImpl                   // 多态基类（虚接口）
          ├── core::CPUGeneratorImpl          // CPU 后端实现 (mt19937_64)
          └── core::CUDAGeneratorImpl         // CUDA 后端实现 (Philox seed+offset)

infini_train::core::GeneratorImplRegistry     // 单例注册表
    ├── factories_                             // DeviceType → factory 函数
    ├── default_cpu_                           // CPU 默认 Generator (shared_ptr)
    └── default_cuda_[kMaxGpus=8]              // 每 CUDA 设备默认 (lazy)
```

- `GeneratorImplRegistry::factories_` 镜像 `DeviceGuardImplRegistry` 的 `unordered_map<DeviceType, ...>` 模式（§1.2）。
- `default_cuda_[kMaxGpus]` + `std::array<once_flag, kMaxGpus>` 镜像 `cuda_guard_impl.cc:14-22` 的 per-GPU stream/handle 缓存模式。

**不**复用 `DeviceGuardImpl`（职责不同：DeviceGuard 管设备切换/流，Generator 管 RNG 状态，生命周期不应耦合）。

### 3.2 文件布局

#### 新建文件


| 文件                                                                 | 作用                                                               |
| ------------------------------------------------------------------ | ---------------------------------------------------------------- |
| `infini_train/include/generator.h`                                 | 公共 `Generator` 句柄 + `default_generator()` / `manual_seed()` 自由函数 |
| `infini_train/include/core/generator/generator_impl.h`             | `GeneratorImpl` 抽象基类 + `GeneratorImplRegistry`                   |
| `infini_train/include/autograd/dropout.h`                          | Dropout autograd Function 声明                                     |
| `infini_train/src/core/generator/generator_impl.cc`                | Registry 实现、默认池、`manual_seed`、`ResolveGenerator`                 |
| `infini_train/include/core/generator/cpu_generator_impl.h`         | `CPUGeneratorImpl` 声明（kernel 端 include 用）                        |
| `infini_train/src/core/generator/cpu/cpu_generator_impl.cc`        | CPU 实现 + `INFINI_TRAIN_REGISTER_GENERATOR_IMPL(kCPU, ...)`       |
| `infini_train/include/core/generator/cuda_generator_impl.h`        | `CUDAGeneratorImpl` 声明（`#ifdef USE_CUDA`，kernel 端 include 用）     |
| `infini_train/src/core/generator/cuda/cuda_generator_impl.cu`      | CUDA Generator 实现 + `INFINI_TRAIN_REGISTER_GENERATOR_IMPL`（编译进 `infini_train_cuda_kernels`，与 kernel 目录分离可接受） |
| `infini_train/src/core/generator/cuda/philox_engine.cuh`           | Philox4_32 device 端引擎（被 `kernels/cuda/*random*.cu` include；License 见 §11）   |
| `infini_train/include/nn/modules/dropout.h`                        | `nn::Dropout` 模块                                                 |
| `infini_train/src/nn/modules/dropout.cc`                           | Dropout 模块实现                                                     |
| `infini_train/src/autograd/dropout.cc`                             | Dropout autograd Function 实现                                     |
| `infini_train/src/kernels/cpu/{uniform_random,normal_random,dropout}.cc`  | CPU 随机 kernel（**扁平命名，与现有 `cpu/` 目录约定一致**）              |
| `infini_train/src/kernels/cuda/{uniform_random,normal_random,dropout}.cu` | CUDA 随机 kernel（**扁平命名，与现有 `cuda/` 目录约定一致**）              |
| `tests/generator/CMakeLists.txt`                                   | 测试套 CMake（见 §7.1）|
| `tests/generator/test_*.cc`                                        | 见 §7 测试矩阵                                                        |
| `scripts/check_reproducibility.sh`（可选）                          | 见 §9.1；赛题不强制，用于本地/CI 快速复现检查                                              |


#### 修改文件


| 文件                                                               | 改动                                                                                |
| ---------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `infini_train/include/tensor.h:151`                              | `Tensor::Uniform` 签名 `std::optional<std::mt19937>` → `std::optional<Generator>`；删除 `<random>`，增加 `#include "infini_train/include/generator.h"` |
| `infini_train/src/tensor.cc:471`                                 | 转发到改造后的 `nn::init::Uniform`                                                       |
| `infini_train/include/nn/init.h:18,45,48`                        | 三个 init 函数签名换 Generator；删除 `<random>` include                                     |
| `infini_train/include/nn/modules/module.h`                       | **修订（§5.0）**：新增 `protected: bool training_ = true;`、`virtual void Train(bool mode = true)`、`void Eval()`、`bool IsTraining() const` |
| `infini_train/src/nn/modules/module.cc`                          | **修订（§5.0）**：实现 `Module::Train(bool)`，遍历 `modules_` 递归（仅跳过 `nullptr`，对齐 `NamedModules:113`；**不**跳过 `__pp_*`，详见 §5.0） |
| `infini_train/include/nn/functional.h` + `src/nn/functional.cc`   | **修订**：在命名空间 `nn::function` 中新增 `Rand` / `Randn` / `Dropout`（与现有 `Sigmoid`、`Softmax` 一致） |
| `infini_train/src/nn/init.cc`                                    | 删除文件作用域 `gen` / `kRandomSeed`，删除 OMP 双 bug 路径，改走 dispatcher（CPU/CUDA 均通过 `UniformRandom`/`NormalRandom` kernel） |
| `example/gpt2/checkpoint_loader.cc:31-34`                        | **删除** `kRandomSeed` 常量、`std::mt19937 gen` 死声明、TODO 注释、`<random>` include（全文件无引用） |
| `example/llama3/checkpoint_loader.cc:29-32`                      | 同上                                                                                |
| `tests/CMakeLists.txt`                                           | 增加 `add_subdirectory(generator)`                                                  |

> **注**：`nn/modules/{linear,sparse}.cc` / `nn/lora/{lora_linear,lora_parallel_linear}.cc` / `nn/parallel/tensor_parallel.cc` 共 11 处下游调用全部不传 generator，新签名 `std::optional<Generator> = std::nullopt` 默认行为兼容，**无需改动**。
>
> **注**：主库与 CUDA 库分轨收集（见 §1.2）：`src/core/generator/**/*.cc` → `infini_train`；`src/core/generator/**/*.cu` 与 `src/kernels/cuda/*.cu` → `infini_train_cuda_kernels`。新建源文件无需改根 CMake。


### 3.3 头文件依赖原则

- `generator.h` 仅 include `device.h` + 标准库 — **不暴露** `mt19937`/`curand`/`Philox`。
- `cuda_generator_impl.h` 用 `#ifdef USE_CUDA` 守护，公共 header 不 include。
- 算子 kernel 通过 `static_cast<CPUGeneratorImpl*>(impl.get())` 拿后端实现（PyTorch `intrusive_ptr<GeneratorImpl>` 风格）。

### 3.4 编译与链接条件

- `USE_CUDA=OFF`：CUDA 文件全部不编译，Registry 只有 CPU 注册器，`Generator(Device::kCUDA)` 在 `Create()` 路径抛 `std::runtime_error`（消息含 `USE_CUDA` 提示）。
- `USE_CUDA=ON`：CUDA 部分通过 `INFINI_TRAIN_REGISTER_GENERATOR_IMPL` 静态注册，与 CPU 路径并存。
- **静态注册链接（修订）**：
  - **Linux**：`link_infini_train_exe()` 对三个静态库使用 `-Wl,--whole-archive` … `-Wl,--no-whole-archive`（`CMakeLists.txt:158-177`）。`REGISTER_KERNEL` 与 `INFINI_TRAIN_REGISTER_GENERATOR_IMPL` 均依赖此行为。
  - **macOS**：GNU `whole-archive` 不可用；本地验证需在 `link_infini_train_exe` 增加 `APPLE` 分支，对三个 `.a` 使用 `-Wl,-force_load,<path/to/lib.a>`（或等价方案）。赛题评测环境以 **Linux** 为准；macOS 仅开发机注意事项。
- 后续若重构 CMake 移除 whole-archive / force_load，须改为显式 `EnsureGeneratorKernelsRegistered()` 等强引用入口。

## 4. 接口设计

### 4.1 用户侧句柄 `infini_train::Generator`

```cpp
// infini_train/include/generator.h
namespace infini_train {

class Generator {
public:
    // 默认构造一个新独立 Generator（不与默认池共享）
    explicit Generator(Device device = Device(Device::DeviceType::kCPU, 0));

    // 拷贝/赋值共享 impl_ 状态（与 PyTorch Generator 一致）；勿对 default_generator() 的拷贝做 ManualSeed
    Generator(const Generator&) = default;
    Generator& operator=(const Generator&) = default;

    // seed 控制
    void     ManualSeed(uint64_t seed);
    uint64_t Seed();              // random_device 取新种子并设置；返回该种子
    uint64_t InitialSeed() const;

    // state 控制（含格式校验）
    std::vector<uint8_t> GetState() const;
    void                 SetState(const std::vector<uint8_t>& state);

    Device device() const;

    // 算子内部访问（非公开 API 但 header-visible）
    const std::shared_ptr<core::GeneratorImpl>& impl() const { return impl_; }

private:
    explicit Generator(std::shared_ptr<core::GeneratorImpl> impl);  // FromImpl 专用
    static Generator FromImpl(std::shared_ptr<core::GeneratorImpl> impl);

    friend Generator& default_generator(Device device);

    std::shared_ptr<core::GeneratorImpl> impl_;
};

// 默认 Generator 入口（返回持久存储的句柄，非临时对象）
Generator& default_generator(Device device = Device(Device::DeviceType::kCPU, 0));

// 全局 seed 入口（即 GeneratorImplRegistry::ResetAllSeeds，见 §4.6）
void manual_seed(uint64_t seed);          // 同步 CPU + 所有 CUDA 默认（已/未初始化）
void manual_seed_cuda(uint64_t seed);     // 仅 CUDA 默认（PyTorch 兼容）

}  // namespace infini_train
```

**拷贝语义（修订）**：`Generator` 为 PImpl 句柄，拷贝/赋值**浅共享** `impl_`。对 `default_generator(device)` 返回的句柄做拷贝后再 `ManualSeed`，会**直接改变默认池** RNG 状态。应使用 `manual_seed()` 或显式 `Generator(device)` 独立实例。

**`default_generator` 存储（修订）**：**禁止**对默认池调用 `Generator(device)` 公开构造（那会 `Create()` 出独立 impl）。应通过 **`FromImpl`** 绑定 Registry 默认池的 `shared_ptr`：

```cpp
// generator_impl.cc — FromImpl 在 generator.h 中为 private，friend default_generator

// generator_impl.cc
Generator& default_generator(Device device) {
    auto impl = GeneratorImplRegistry::Instance().Default(device);
    if (device.IsCPU()) {
        std::call_once(default_cpu_handle_once_, [&] {
            default_cpu_handle_ = Generator::FromImpl(impl);
        });
        CHECK(default_cpu_handle_.impl().get() == impl.get());
        return default_cpu_handle_;
    }
    const int idx = device.index();
    CHECK(0 <= idx && idx < kMaxGpus);
    std::call_once(default_cuda_handle_once_[idx], [&] {
        default_cuda_handles_[idx] = Generator::FromImpl(impl);
    });
    return default_cuda_handles_[idx];
}
```

`std::once_flag` + 成员 `Generator default_cpu_handle_` / `std::array<Generator, kMaxGpus> default_cuda_handles_` 放在 `generator_impl.cc` 匿名命名空间或 `GeneratorImplRegistry` 内。**多次调用 `default_generator(dev)` 须返回同一对象地址，且 `impl()` 与 `Registry::Default(dev)` 返回的 `shared_ptr` 相同。**

### 4.2 抽象基类 `core::GeneratorImpl`

```cpp
namespace infini_train::core {

class GeneratorImpl {
public:
    explicit GeneratorImpl(Device device) : device_(device) {}
    virtual ~GeneratorImpl() = default;

    virtual void     SetCurrentSeed(uint64_t seed) = 0;   // 同时更新 initial_seed_（见 §4.3）
    virtual uint64_t CurrentSeed() const = 0;
    virtual uint64_t InitialSeed() const = 0;             // 修订：不受 Seed() 的 random 重设影响
    virtual std::vector<uint8_t> GetState() const = 0;
    virtual void     SetState(const std::vector<uint8_t>& state) = 0;
    virtual uint32_t StateMagic() const = 0;   // 用于 SetState 校验

    Device      device() const { return device_; }
    std::mutex& mutex() { return mutex_; }

    // 默认实现：s = RandomSeed64(); Reseed(s); return CurrentSeed(); 不改 initial_seed_
    virtual uint64_t Seed();

protected:
    virtual void Reseed(uint64_t seed) = 0;   // CPU 重置 mt19937_64；CUDA 设 seed_ 且 offset_=0

    Device device_;
    mutable std::mutex mutex_;
};

class GeneratorImplRegistry {
public:
    static GeneratorImplRegistry& Instance();
    void Register(Device::DeviceType type, GeneratorImplFactory factory);
    // Create：CPU 须 device.index()==0，否则 CHECK 失败；CUDA 使用 device.index()
    std::shared_ptr<GeneratorImpl> Create(Device device) const;
    std::shared_ptr<GeneratorImpl> Default(Device device);          // 懒初始化 + 缓存
    void ResetAllSeeds(uint64_t seed);                              // manual_seed() 的实现体（§4.6）
};

#define INFINI_TRAIN_REGISTER_GENERATOR_IMPL(device_type, FactoryFn) \
    namespace { struct GenReg_##device_type { GenReg_##device_type() { \
        GeneratorImplRegistry::Instance().Register( \
            Device::DeviceType::device_type, FactoryFn); \
    }} _gen_reg_##device_type; }

}  // namespace infini_train::core
```

### 4.3 CPU 实现 `CPUGeneratorImpl`

```cpp
class CPUGeneratorImpl : public GeneratorImpl {
public:
    explicit CPUGeneratorImpl(int8_t /*device_index*/);  // index 必须为 0；与 Device::index() 类型一致

    void     SetCurrentSeed(uint64_t seed) override;  // seed_=initial_seed_=seed，并重置 engine_
    uint64_t CurrentSeed() const override { return seed_; }
    uint64_t InitialSeed() const override { return initial_seed_; }
    std::vector<uint8_t> GetState() const override;
    void     SetState(const std::vector<uint8_t>&) override;
    uint32_t StateMagic() const override { return kMagic; }

    void Reseed(uint64_t seed) override;            // seed_=s，重置 engine_；不改 initial_seed_

    std::mt19937_64& engine();   // 算子用，调用前需持 mutex()

private:
    static constexpr uint32_t kMagic   = 0x47555043;   // 'CPUG'
    static constexpr uint32_t kVersion = 1;
    uint64_t initial_seed_{0};   // ManualSeed / SetCurrentSeed 写入
    uint64_t seed_{0};           // 当前逻辑种子（Seed() 可单独更新）
    std::mt19937_64 engine_;
};
```

`GeneratorImpl::Seed()` 默认实现（`generator_impl.cc`）：`uint64_t s = RandomSeed64(); Reseed(s); return CurrentSeed();` — **不修改** `initial_seed_`。CPU/CUDA 各自在 `Reseed` 中更新 `seed_`（及 engine/offset）。

### 4.4 CUDA 实现 `CUDAGeneratorImpl`（Philox 风格）

```cpp
class CUDAGeneratorImpl : public GeneratorImpl {
public:
    explicit CUDAGeneratorImpl(int8_t device_index);   // 与 Device::index() 类型一致

    void     SetCurrentSeed(uint64_t seed) override;  // seed_=initial_seed_=seed; offset_=0
    uint64_t CurrentSeed() const override { return seed_; }
    uint64_t InitialSeed() const override { return initial_seed_; }
    std::vector<uint8_t> GetState() const override;
    void     SetState(const std::vector<uint8_t>&) override;
    uint32_t StateMagic() const override { return kMagic; }

    void Reseed(uint64_t seed) override { seed_ = seed; offset_ = 0; }  // Seed() 经基类默认实现调用；不改 initial_seed_

    struct PhiloxState { uint64_t seed; uint64_t offset; };
    // 推进 offset：消耗 num_elements 个随机数，每 4 个共享一个 Philox 4-tuple
    PhiloxState NextPhiloxState(uint64_t num_elements);

private:
    static constexpr uint32_t kMagic   = 0x47445543;   // 'CUDG'
    static constexpr uint32_t kVersion = 1;
    uint64_t initial_seed_{0};
    uint64_t seed_;
    uint64_t offset_;
};
```

**Philox 精髓**：state 仅 16 字节（seed+offset），kernel 端每个线程基于 `(seed, thread_id, captured_offset)` 自构 Philox 引擎，无需 device buffer 持久化。多次调用之间 host 端推进 offset，序列不重叠。

### 4.5 状态序列化格式

所有后端共用统一头部：

```
+---------+---------+----------------+----------------+
| magic   | version | payload_size   |    payload     |
| u32     | u32     | u64            |    bytes       |
+---------+---------+----------------+----------------+
```

**字节序假定**：InfiniTrain 仅支持 x86_64 / aarch64（均为 little-endian）。直接 `std::memcpy` 读写整型字段，不做 endian swap。若未来引入 BE 平台再补 swap helper。

`SetState` 校验顺序：
1. 总长度 ≥ header 长度（4+4+8=16 字节），否则抛 `std::runtime_error`；
2. `magic == this->StateMagic()`，不匹配抛；
3. `version` 与本实现支持的 version 匹配，否则抛；
4. `payload_size` 与剩余字节数一致，否则抛；
5. payload 解析失败（如 mt19937_64 state 大小不对）抛。

- **CPU payload**：`mt19937_64` 序列化 blob + `seed_`（u64） + `initial_seed_`（u64）。`mt19937_64` 通过 `operator<<` / `operator>>` 写入 stringstream 再转 bytes。
- **CUDA payload**：`seed_`（u64） + `offset_`（u64） + `initial_seed_`（u64）。

### 4.6 默认 Generator + manual_seed 行为

懒初始化（镜像 `cuda_guard_impl.cc:14-22` 的 per-GPU `array+once_flag` 模式）：

```cpp
std::shared_ptr<GeneratorImpl> GeneratorImplRegistry::Default(Device device) {
    if (device.IsCPU()) {
        std::call_once(default_cpu_once_, [&]{
            default_cpu_ = factories_.at(Device::DeviceType::kCPU)(0);
            default_cpu_->SetCurrentSeed(LastUserSeedOrRandom());
            cpu_initialized_.store(true, std::memory_order_release);
        });
        return default_cpu_;
    }
    const int8_t idx = device.index();
    std::call_once(default_cuda_once_[idx], [&]{
        default_cuda_[idx] = factories_.at(Device::DeviceType::kCUDA)(idx);
        default_cuda_[idx]->SetCurrentSeed(LastUserSeedOrRandom());
        cuda_initialized_[idx].store(true, std::memory_order_release);
    });
    return default_cuda_[idx];
}
```

`LastUserSeedOrRandom()` 是 Registry 私有方法：返回 `last_user_seed_`（`std::optional<uint64_t>`，由 `manual_seed` 写入）；若为空则用 `std::random_device`（两次 32-bit 拼成 64-bit）现取。

> **锁域（修订）**：本方法仅在读取 `last_user_seed_` 时持 `registry_mutex_`，**释放后**再调用 `std::random_device`。原因：本方法会在 `Default()` 的 `call_once` lambda 内被调用，而 `ResetAllSeeds()` 需要在 step 1 持 `registry_mutex_` 写入 `last_user_seed_` 之后释放、再调 `Default()`；若本方法在锁内调 `random_device` 会延长持锁时间，且若误在持锁状态调用本方法会造成嵌套取锁死锁。

```cpp
uint64_t GeneratorImplRegistry::LastUserSeedOrRandom() {
    {
        std::lock_guard<std::mutex> lk(registry_mutex_);
        if (last_user_seed_) return *last_user_seed_;
    }
    return RandomDeviceSeed64();   // 锁外，避免 random_device 阻塞带锁
}
```

Registry 字段一览：

```cpp
class GeneratorImplRegistry {
private:
    std::unordered_map<Device::DeviceType, GeneratorImplFactory> factories_;
    std::shared_ptr<GeneratorImpl> default_cpu_;
    std::array<std::shared_ptr<GeneratorImpl>, /*kMaxGpus=*/8> default_cuda_;
    std::once_flag default_cpu_once_;
    std::array<std::once_flag, /*kMaxGpus=*/8> default_cuda_once_;
    std::atomic<bool> cpu_initialized_{false};
    std::array<std::atomic<bool>, /*kMaxGpus=*/8> cuda_initialized_{};   // 用于 manual_seed 判定
    std::optional<uint64_t> last_user_seed_;     // manual_seed 最近一次设置
    std::mutex registry_mutex_;                  // 仅保护 last_user_seed_；禁止在持有本锁时调用 Default()
};
```

> **为什么需要 `cuda_initialized_`**：`std::once_flag` 用过后无法外部查询其状态；`manual_seed` 不能再 `call_once`（否则可能回调初始化），也不能 race 检查 `default_cuda_[idx]` 指针。`atomic<bool>` 在 `Default()` 路径 `release` 写、在 `manual_seed` 路径 `acquire` 读，配合 `registry_mutex_` 保证不漏不重。

`manual_seed(s)` 行为（修订：窄锁 + `call_once` 串行化）：

**锁域设计**：`registry_mutex_` 只保护 `last_user_seed_` 一个字段；初始化路径靠 `call_once` 自身串行化。**禁止**在持 `registry_mutex_` 的同时调用 `Default()`，否则 `Default()` 内 `call_once` lambda 调 `LastUserSeedOrRandom()` 会嵌套取锁死锁。本设计**不**使用 `std::recursive_mutex`——递归锁掩盖了"锁域未分清"的设计缺陷。

```cpp
void GeneratorImplRegistry::ResetAllSeeds(uint64_t seed) {
    // 1. 极窄锁：仅写入 last_user_seed_，立即释放
    {
        std::lock_guard<std::mutex> lk(registry_mutex_);
        last_user_seed_ = seed;
    }
    // 2. 锁外强制初始化 CPU 默认 — call_once 自带串行化
    auto cpu_impl = Default(Device(Device::DeviceType::kCPU, 0));
    {
        std::lock_guard<std::mutex> lk(cpu_impl->mutex());
        cpu_impl->SetCurrentSeed(seed);
    }
    // 3. 锁外遍历已初始化的 CUDA（acquire 配对 Default 中的 release）
    //    未初始化的不强制激活；下次首访经 LastUserSeedOrRandom() 自动取到 s
    for (int i = 0; i < kMaxGpus; ++i) {
        if (!cuda_initialized_[i].load(std::memory_order_acquire)) continue;
        auto& impl = default_cuda_[i];
        std::lock_guard<std::mutex> lk(impl->mutex());
        impl->SetCurrentSeed(seed);
    }
    // 4. 显式构造的 Generator 不受影响
}
```

**并发语义**：

- **`manual_seed(s)` 与未初始化 CUDA 首访竞争**：若首访先入 `call_once` lambda，它读到的 `last_user_seed_` 已是 `s`（step 1 完成于 step 2 之前）→ 用 s 初始化；step 3 后续 `SetCurrentSeed(s)` 幂等。若首访后入 lambda 同理。
- **两个 `manual_seed` 并发**：`last_user_seed_` 在 `registry_mutex_` 下"最后写入胜出"；后续各 Generator 的 `SetCurrentSeed` 顺序由各自 `mutex()` 决定 —— 此为用户已有的 UB 区域，不强行定义。
- **`acquire`/`release` 配对**：读 `cuda_initialized_[i]==true` 时，`default_cuda_[i]` 指针写入对当前线程可见；step 3 拿到的 `impl` 一定是 lambda 内构造完成的对象。

### 4.7 算子接入：`ResolveGenerator`

```cpp
// 显式 → 直接返回该 Generator 内的 impl；nullopt → 返回设备默认 Generator 的 impl
// 返回 shared_ptr 而非引用，避免对 optional 临时对象的悬空引用
std::shared_ptr<core::GeneratorImpl> ResolveGenerator(
    const std::optional<Generator>& gen, Device device);
```

算子层典型用法：

```cpp
auto impl = ResolveGenerator(generator, t->GetDevice());
Dispatcher::Instance().Call<void>(
    {t->GetDevice().type(), "UniformRandom"}, t, low, high, impl.get());
```

> 注意 1：`Device::type()` / `Device::index()` 都是小写（见 `include/device.h:28-29`），返回 `Device::DeviceType` / `int8_t`。
>
> 注意 2：kernel 收到的是 `core::GeneratorImpl*` 裸指针，仅在 kernel 调用栈内有效；shared_ptr 由调用层（init.cc / dropout.cc 等）持有，保证整个调用周期内 impl 存活。
>
> 注意 3：Dispatcher 用 `reinterpret_cast` 类型擦除调用，**注册函数签名必须与 `Call<>` 的实参类型逐一对齐**。
>
> 注意 4（修订）：单 tensor 输出的随机 kernel 签名为 `void Kernel(std::shared_ptr<Tensor>, ..., core::GeneratorImpl*)`。`DropoutForward` 返回 `std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>`（2 元组；沿用 `Call<std::tuple<...>>` 模式，参考 `normalization.cc`、`outer.cc`。LayerNorm 为 3 元组，勿混用措辞）。

## 5. 算子改造

### 5.0 Module training 模式支持（修订）

`nn::Dropout` 模块需要根据 `training`/`eval` 状态切换行为，但当前 `infini_train::nn::Module` 基类**没有** `training` 标记（grep `include/nn/`、`src/nn/modules/module.cc` 均无 `training_` / `Train()` / `Eval()` / `is_training` 命中）。本任务一并补齐，作为 Dropout 的前置基础设施。

**作用域**：仅新增 `training_` 字段 + `Train/Eval/IsTraining` 三个方法；**不**触动 hook、parameter、buffer、序列化等任何其他机制。例子里 `example/llama3/main.cc:329`、`example/gpt2/main.cc:352` 已有 `// model->Train();` 占位注释——本节落地后即可启用。

**接口（`include/nn/modules/module.h`）**：

```cpp
class Module {
public:
    // 递归设置自身与所有子 module 的 training 状态
    virtual void Train(bool mode = true);
    void         Eval() { Train(false); }
    bool         IsTraining() const { return training_; }
protected:
    bool training_ = true;
    // ... 现有成员
};
```

**实现（`src/nn/modules/module.cc`）**：

```cpp
void Module::Train(bool mode) {
    training_ = mode;
    for (auto& [name, sub] : modules_) {
        if (!sub) continue;   // 对齐 NamedModules:113
        sub->Train(mode);     // 不跳过 __pp_*：StateDict 与 Train 语义不同（详见下文）
    }
}
```

> **为什么不照搬 `StateDict()` 的 `__pp_*` 跳过**（关键，避免静默 bug）：
> `StateDict()` 跳过 `__pp_*` 是因为 `TransformerModel`（`transformer.cc:228-238, 241-247`）把 `__pp_chunk_*` 内部的 layer `shared_ptr` 又**别名暴露**到 `kTransformerModelName` 下的 `ModuleList`/`ModuleDict`，**同一参数会被发两遍**。这是序列化特有的去重需求。
>
> `Train()` 完全相反：它写一个 `bool`，**幂等**，对同一 Module 多次调用无副作用。`__pp_chunk_*` 是 PP 的真实可调用子模块（`transformer.cc:269` Forward 经它驱动 layers），里面的 Dropout 必须接收到 eval 状态。当前代码里跳过虽然碰巧仍可达（因别名存储），但这是脆弱的隐性不变量——**未来若 PP 实现不再做别名存储，`model->Eval()` 会静默漏掉 chunk 内 Dropout，eval 仍走 dropout**。
>
> **规则**：禁止把 `StateDict()` 的 `__pp_*` 跳过照搬到 `Train()`、`To()`、`Apply()` 等幂等遍历。每种遍历的跳过规则要按自身语义定，不要互相照搬。

**影响面核查（落地前已经过代码库验证，结论附下）**：

| 路径 | 是否受影响 | 证据 |
|------|-----------|------|
| DDP / DataParallel / PipelineParallel | **不受影响**：均通过 `modules_["module"] = wrapped` 包装，递归 Train 自动透传，无需覆写 | `distributed_data_parallel.cc:43`、`data_parallel.cc:71`、`pipeline_parallel.cc:83` |
| `TransformerModel` 内置 PP（`__pp_*`） | **不受影响（前提：Train 不跳 __pp）**：`__pp_chunk_*` 是真实可调用子模块（`transformer.cc:269` Forward 经它驱动 layers），递归必须进入。当前实现里 `__pp_*` 子树与 `kTransformerModelName` 共享 layer `shared_ptr`，会被 Train 触达两次但幂等无副作用 | `transformer.cc:228-238, 241-247, 266-274` |
| Tensor Parallel / LoRA | **不受影响**：继承 `CloneableModule` / `Linear`，无自有 `modules_` 子结构 | `tensor_parallel.h:21,50,79,118`、`lora_linear.h:25` |
| `Module` 复制 / `Replicate` | **语义正确**：`Module(const Module&) = default` 自动按值拷贝 trivial `bool training_`，replica 继承父状态 | `module.h:44`、`module.cc:264`、`parallel_functional.cc:111-138` |
| `StateDict()` 序列化 | **不受影响**：仅遍历 `parameters_`/`buffers_`，不反射字段；`training_` 不会被持久化（与 PyTorch 一致） | `module.cc:137-148` |
| ABI | 全量重编译即可。仓库内无外部稳定 ABI 承诺；新增 virtual 追加 vtable 末尾不破坏既有偏移 | 全仓 grep 无动态库 export 证据 |

### 5.1 公共签名

> 与现有 codebase 对齐：所有 tensor 入参/出参一律 `std::shared_ptr<Tensor>`，不用值类型。

```cpp
// init.h
std::shared_ptr<Tensor> Normal(const std::shared_ptr<Tensor>&, float mean, float std,
                               std::optional<Generator> = std::nullopt);
std::shared_ptr<Tensor> Uniform(const std::shared_ptr<Tensor>&, float low, float high,
                                std::optional<Generator> = std::nullopt);
std::shared_ptr<Tensor> KaimingUniform(const std::shared_ptr<Tensor>&, float a,
                                       KaimingMode = KaimingMode::kFanIn,
                                       NonLinearityType = NonLinearityType::kLeakyReLU,
                                       std::optional<Generator> = std::nullopt);

// tensor.h
std::shared_ptr<Tensor> Tensor::Uniform(float from = 0.0f, float to = 1.0f,
                                        std::optional<Generator> = std::nullopt);

// infini_train/include/nn/functional.h — 命名空间 nn::function（修订，勿写成 functional 命名空间）
namespace infini_train::nn::function {
std::shared_ptr<Tensor> Rand(const std::vector<int64_t>& size, DataType dtype, Device device,
                             std::optional<Generator> generator = std::nullopt);
std::shared_ptr<Tensor> Randn(const std::vector<int64_t>& size, DataType dtype, Device device,
                              std::optional<Generator> generator = std::nullopt);
std::shared_ptr<Tensor> Dropout(const std::shared_ptr<Tensor>& input, float p, bool training = true,
                                std::optional<Generator> generator = std::nullopt);
}
```

实现放在 `infini_train/src/nn/functional.cc`，内部 `ResolveGenerator` + `Dispatcher` / `autograd::Dropout`。

### 5.2 Dispatcher key


| Key                                                | CPU 实现                     | CUDA 实现                     |
| -------------------------------------------------- | -------------------------- | --------------------------- |
| `(*, "UniformRandom")`                             | `cpu/uniform_random.cc`    | `cuda/uniform_random.cu`    |
| `(*, "NormalRandom")`                              | `cpu/normal_random.cc`     | `cuda/normal_random.cu`     |
| `(*, "DropoutForward")` | `cpu/dropout.cc` → `tuple<output, mask>` | `cuda/dropout.cu` → 同上 |
| `(*, "DropoutBackward")` | `cpu/dropout.cc` → `grad_input` | `cuda/dropout.cu` → 同上 |

**DropoutForward 注册签名（修订）**：

```cpp
std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
DropoutForward(const std::shared_ptr<Tensor>& input, float p, core::GeneratorImpl* impl);
```

`Rand`/`Randn` 复用 Uniform/Normal kernel：wrapper 层先 `std::make_shared<Tensor>(size, dtype, device)` 再调对应 kernel。首期 dtype 仅 **FP32**（§10）。

**CUDA stream（修订）**：`cuda/uniform_random.cu`、`cuda/normal_random.cu`、`cuda/dropout.cu` 中所有 kernel launch **均须**绑定 `GetStream(device)`（与 `fill.cu`、`elementwise.cu` 一致），见 §5.4。

### 5.3 CPU kernel 模板

```cpp
void UniformRandom(std::shared_ptr<Tensor> t, float lo, float hi,
                   core::GeneratorImpl* impl) {
    auto* cpu_impl = static_cast<core::CPUGeneratorImpl*>(impl);
    std::lock_guard<std::mutex> lock(cpu_impl->mutex());
    auto& eng = cpu_impl->engine();
    std::uniform_real_distribution<float> dist(lo, hi);
    auto* data = static_cast<float*>(t->DataPtr());
    for (size_t i = 0; i < t->NumElements(); ++i) data[i] = dist(eng);
}
REGISTER_KERNEL(Device::DeviceType::kCPU, UniformRandom, UniformRandom);
```

**关键修复**：单线程 + 锁，正确性优先。修复了现有 `init.cc` OMP 路径"每线程重新构造 mt19937"的状态不推进 bug。

### 5.4 CUDA kernel 模板（Dropout 为例）

```cuda
__global__ void DropoutKernel(const float* in, float* out, uint8_t* mask,
                              size_t n, float p, uint64_t seed, uint64_t offset) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    Philox4x32 eng(seed, tid, offset);
    float scale = 1.0f / (1.0f - p);
    for (size_t i = tid; i < n; i += blockDim.x * gridDim.x) {
        float r = eng.uniform();
        bool keep = r >= p;
        mask[i] = keep;
        out[i] = keep ? in[i] * scale : 0.0f;
    }
}

void DropoutForward(...) {
    auto device = input->GetDevice();
    core::DeviceGuard guard(device);
    auto* cuda_impl = static_cast<core::CUDAGeneratorImpl*>(impl);
    std::lock_guard<std::mutex> lock(cuda_impl->mutex());
    auto ph = cuda_impl->NextPhiloxState(input->NumElements());
    // 修订：必须绑定当前设备的 CUDA stream（与 fill.cu、elementwise.cu 一致）
    const cudaStream_t stream = dynamic_cast<core::cuda::CudaStream*>(
        core::GetDeviceGuardImpl(device.type())->GetStream(device))->cuda_stream();
    DropoutKernel<<<num_blocks, threads_per_block, 0, stream>>>(..., ph.seed, ph.offset);
}
```

**Offset 推进规则**：每次调用 host 端把 offset 推进 `ceil(num_elements / 4)`（Philox 一次产 4 个 32-bit），下一次调用用新 offset，序列不重叠。

**同步策略（修订）**：kernel 启动后不强制 `cudaDeviceSynchronize`；与现有算子一致，依赖同一 stream 上后续 op 或用户侧同步。`init.cc` 改造后不再使用 host buffer + H2D 填随机数。

### 5.5 Dropout 完整链路

新增三层：

1. **kernel**（`{cpu,cuda}/dropout.{cc,cu}`）：forward 生成 mask（dtype `kUINT8`，与 input 同 device）+ scale 后输出；backward 用 mask 反传梯度。Dispatcher：`DropoutForward` → `std::tuple<output, mask>`；`DropoutBackward` → `grad_input`。
2. **autograd Function**（`autograd/dropout.{h,cc}`）：派生 `class Dropout : public Function`，多返回值参考 `autograd/normalization.cc`、`autograd/outer.cc`（`Call<std::tuple<...>>`）：
   - **派生类成员** `float p_`、`bool training_`、`std::optional<Generator> generator_`（`training_` 供 `nn::function::Dropout` 直调；`nn::Dropout` 模块由 `IsTraining()` 门控，eval 时不构造本 Function）。
   - `Forward(inputs)`：`auto [out, mask] = Dispatcher::Call<std::tuple<...>>({device, "DropoutForward"}, input, p_, impl.get())`；`saved_tensors_ = {mask}`；返回 `{out}`。
   - `SetupContext`：可空实现（mask 已在 Forward 保存）。
   - `Backward(grad_outputs)`：调 `DropoutBackward`，返回 `{grad_input}`。
3. **module**（`nn::Dropout`）：
   - `Dropout(float p = 0.5f, Device device = Device())`；可选 `SetGenerator(std::optional<Generator>)` 或 ctor 传入 generator（修订）。
   - `Forward` 读 `IsTraining()`（见 §5.0）：`true` → `std::make_shared<autograd::Dropout>(p_, true, generator_)->Apply(inputs)`；`false` → 直接返回 `input`。**不**在 Dropout 自身存 `training_` 字段，统一由 Module 基类管理，`model->Eval()` 一键切换。
4. **LoRA（修订，非本任务）**：`nn::lora::LoRAConfig::dropout` 字段仍存在，**不在此任务接入** `nn::Dropout`；报告可列为后续工作。

## 6. 与 PyTorch 对齐


| 行为                          | PyTorch             | InfiniTrain (本设计)         |
| --------------------------- | ------------------- | ------------------------- |
| `torch.manual_seed(s)` 影响范围 | CPU + 所有 CUDA 默认    | CPU + 已/未初始化 CUDA 默认      |
| 显式 `Generator()`            | 不受 `manual_seed` 影响 | 一致                        |
| 默认 Generator 多次获取           | 同一份                 | 一致（shared_ptr）            |
| 不同 CUDA 设备默认                | 各自独立（共用同一 seed）     | 一致                        |
| 未 `manual_seed` 时           | `random_device`     | 一致                        |
| state 格式校验                  | tensor 类型 + size 检查 | magic + version + size 检查 |
| `InitialSeed()` vs `Seed()`   | `Seed()` 不改变 `initial_seed()` | 一致（§4.3，修订） |
| 同 seed 数值一致                 | bit 一致              | **不要求**（声明）               |
| 相对旧 InfiniTrain `init.cc` 同 seed | N/A | **不要求**（§2.3，修订） |
| CPU vs CUDA 同 seed 一致       | **不要求**             | **不要求**（与 PyTorch 一致）     |


## 7. 测试矩阵

### 7.1 测试 CMake（修订）

```cmake
# tests/generator/CMakeLists.txt
file(GLOB GENERATOR_TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cc)

infini_train_add_test_suite(test_generator
  SOURCES ${GENERATOR_TEST_SOURCES}
)
```

`infini_train_add_test_suite` 会生成 `test_generator_cpu` / `test_generator_cuda`（`USE_CUDA=ON` 时），ctest 标签默认为 `cpu` / `cuda`（见 `cmake/test_macros.cmake`）。**不要依赖不存在的 `generator` 标签**；验证命令见 §9。

### 7.2 用例矩阵

所有测试用 `InfiniTrainTest : TestWithParam<Device::DeviceType>` + `INFINI_TRAIN_REGISTER_TEST` 宏。**USE_CUDA=ON 时实例化 CPU + CUDA，USE_CUDA=OFF 时仅 CPU**——纯 CPU 编译流水线必须能跑通所有用例（CUDA 专属用例用 `ONLY_CUDA()` 守护）。


| 测试文件                                        | 内容                                                                                                 |
| ------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `tests/generator/test_generator_basic.cc`   | 句柄构造、`device()` 查询、显式 vs 默认获取、`default_generator()` 多次返回共享状态                                       |
| `tests/generator/test_seed.cc`              | `ManualSeed(123)` → 取 N → `ManualSeed(123)` → 序列一致；不同 seed 不同；**显式包含状态推进 trap test**（连续两次取 N 必须不同） |
| `tests/generator/test_state.cc`             | `s = GetState(); 取 N; SetState(s); 再取 N` 与第一次相同                                                    |
| `tests/generator/test_state_validation.cc`  | **纯 CPU 端格式校验**（不依赖 CUDA 编译）：手工构造伪造 buffer 喂给 CPU `SetState`——错误 magic 抛、错误 version 抛、payload_size 与剩余字节不一致抛、总长度 < header 抛。`USE_CUDA=ON` 时另加用例：`CPUGeneratorImpl::GetState()` 喂给 `CUDAGeneratorImpl::SetState()` 因 magic 不匹配抛 |
| `tests/generator/test_default.cc`           | `manual_seed(s)` 后 CPU/CUDA 默认都用 `s`；未初始化 CUDA 在 `manual_seed` 后首次访问也用 `s`。**多 GPU**：`REQUIRE_MIN_DEVICES(2)`，在 `ONLY_CUDA()` 块内用 `Device(kCUDA,0)` 与 `Device(kCUDA,1)` 分别 `default_generator` + `Uniform`，验证序列独立（修订；不能仅靠 `GetDevice()` 的 index 0） |
| `tests/generator/test_initial_seed.cc`（修订，可选合并入 `test_seed.cc`） | `ManualSeed(1)` → `InitialSeed()==1` → `Seed()` 得到新值 → `InitialSeed()` 仍为 `1` |
| `tests/generator/test_dispatch.cc`          | 不传 Generator 时 CPU tensor 走 CPU 默认、CUDA tensor 走 CUDA 默认；显式传入时不误用默认                                |
| `tests/generator/test_ops_uniform.cc`       | 同 seed 同形状 → 结果完全一致；不同 seed → 不同                                                                   |
| `tests/generator/test_ops_normal.cc`        | 同上，正态分布                                                                                            |
| `tests/generator/test_ops_dropout.cc`       | 同 seed → mask 一致；mask 大致符合 1-p 概率；scale 正确；backward 梯度正确；**`nn::Dropout` + `model->Eval()` 后输出与输入逐元素相等**（training 门控） |
| `tests/generator/test_ops_kaiming.cc`       | 同 seed → 权重一致；fan_in 计算正确                                                                          |
| `tests/generator/test_pytorch_alignment.cc` | 文档型：复现"先 manual_seed → 算子 A → 算子 B"序列，验证语义对齐（不要求 bit 一致）                                           |


## 8. 报告内容

`docs/generator-design.md`（区别于本 spec，作为提交报告）：

1. **设计参考**：PyTorch `c10::GeneratorImpl` / `CPUGeneratorImpl` / `CUDAGeneratorImpl` / Philox4_32 / `at::detail::DefaultGenerator` 对应关系。
2. **接口对照表**：本实现 vs PyTorch 接口逐一列出（一致 / 缺失 / 命名差异）。
3. **行为对齐情况**：详尽列出每条对齐行为及验证方法。
4. **未对齐项及原因**：底层算法不同 → 同 PyTorch bit 不一致；CPU 与 CUDA 同 seed 不一致；**相对旧 `init.cc` 默认序列变化**（§2.3）。
5. **可扩展方向**：分布式 RNG、graph capture 友好的 Philox 状态、其他后端接入路径、OMP 并行 RNG、LoRA `dropout` 接线。

## 9. 验证清单（提供给 reviewer）

```bash
# 构建（赛题/CI 以 Linux 为准）
mkdir -p build && cmake -B build -DUSE_CUDA=ON -DUSE_NCCL=ON && cmake --build build -j

# 跑 Generator 测试套件（修订：按可执行名过滤，而非 -L generator）
cd build && ctest -V -R '^test_generator_'

# 仅 CPU 构建时
# cmake -B build -DUSE_CUDA=OFF && cmake --build build -j
# cd build && ctest -V -R '^test_generator_cpu'
```

### 9.1 可选：`scripts/check_reproducibility.sh`（修订）

仓库初稿**尚无**此脚本。若交付，建议最小行为：

1. 在 `build/` 下连续两次运行 `ctest -R '^test_generator_'`（或单一 gtest 可执行文件两次 `--gtest_filter=SameSeed*`）；
2. 将第二次 XML/日志与第一次 diff，或运行两次小 benchmark 写 checkpoint 再 `diff`；
3. 退出码非 0 表示不可复现。

赛题验收以 §7 单测通过为主，脚本为加分项。

## 10. 不在范围

- OMP 并行 RNG 优化（写入报告"后续可扩展方向"）。
- Bernoulli / randint 等其他随机算子（非选题强制）。
- ZeRO / 分布式 RNG（与本任务正交）。
- BF16 等非 FP32 dtype 下的 random（先做 FP32，dtype 扩展作为后续）。

## 11. 风险与缓解


| 风险                                 | 缓解                                                                                                                                        |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Philox 自实现引入 bug                   | 直接复制 PyTorch `at::Philox4_32` 的 device-side 引擎到 `philox_engine.cuh`（公开 source 可直接对照）。**License**：PyTorch 为 BSD-3，复制时必须在 `philox_engine.cuh` 文件顶部保留原文件版权声明 + LICENSE 引用，并在仓库 `LICENSE` 或 `THIRD_PARTY_NOTICES` 标注来源。测试覆盖：同 (seed, offset, tid) 序列必须可复现，offset 推进后序列必须不重叠 |
| OMP 串行化导致 Uniform 大 tensor init 变慢 | 性能不在本次验收，但报告内说明并给基准数据                                                                                                                     |
| 全局 `manual_seed` 与多线程并发的竞态         | `registry_mutex_` **仅**保护 `last_user_seed_`；初始化路径靠 `call_once` 串行化，`cuda_initialized_` 用 `std::atomic<bool>` 单调标记。锁域设计与并发语义见 §4.6 |
| `manual_seed` 与 `Default()` 嵌套取锁死锁 | §4.6 `ResetAllSeeds` 释放 `registry_mutex_` 后再调用 `Default()`；`LastUserSeedOrRandom()` 锁内仅读 `last_user_seed_`，锁外取 `random_device`。**禁止**改用 `recursive_mutex` 掩盖问题 |
| `std::shared_ptr` 引入开销             | Generator 句柄拷贝廉价，与 PyTorch `intrusive_ptr` 等价                                                                                             |
| 默认 Generator 在静态析构时的释放顺序           | Registry 自身是 Meyers singleton，析构先于全局变量；测试覆盖                                                                                               |
| Static initializer 在 `.cu` 静态库不被链接 | Linux：`--whole-archive`（§3.4）；macOS：`force_load` 或显式注册；重构 CMake 时须保留其一 |
| macOS 本地链接失败 | 赛题 CI 为 Linux；开发文档注明 `link_infini_train_exe` 的 `APPLE` 分支（§3.4） |
| `default_generator` 误用公开构造 | 仅 `FromImpl(Default())` 绑定默认池；代码评审 + `test_generator_basic` 验证多次取引用地址相同 |
| CUDA 随机 kernel 未绑 stream | 所有 `cuda/*random*.cu` 与 `dropout.cu` 均取 `GetStream(device)`（§5.2、§5.4）；否则与 Memcpy/其他 op 竞态 |


