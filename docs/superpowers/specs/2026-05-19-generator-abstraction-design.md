# Generator 抽象设计 (Spec)

> 任务：2026 春季人工智能大赛 — Generator 抽象选题。
> 目标：为 InfiniTrain 引入统一的随机数生成器抽象，对齐 PyTorch `c10::GeneratorImpl` 设计，提供 CPU/CUDA 后端、默认 Generator 池、全局 seed 入口，并改造现有随机算子接入。

## 1. 背景与现状

### 1.1 当前 RNG 现状

- **CPU**：仅 5 处使用 `std::mt19937`，集中于 `infini_train/src/nn/init.cc`，共用文件作用域 `static std::mt19937 gen(kRandomSeed=42)`；`std::mt19937` 还泄漏在公共头 `include/tensor.h:151`、`include/nn/init.h:18,45,48`。
- **CUDA**：完全空白，无 curand / Philox / curandState 任何使用。所有"CUDA 随机"实际是 CPU 生成 → `cudaMemcpy` 到 device。
- **Dropout**：不存在（`lora_config.h:15` 注释 "not implemented yet"）。
- **状态推进 bug**：OMP 路径每次调用都用 `kRandomSeed + thread_num` **重新构造** `mt19937`，状态根本不推进。
- **公开 ABI**：`std::mt19937` 在公共头出现于 4 处签名；6 处下游调用点（`nn/modules/{linear,sparse}.cc`、`nn/lora/`*、`nn/parallel/tensor_parallel.cc`、`example/{gpt2,llama3}/checkpoint_loader.cc`）。
- **测试覆盖**：随机算子无任何测试。
- **现有 TODO**：`infini_train/src/nn/init.cc:24-34` 已有完整 FIXME 注释列出本任务全部需求。

### 1.2 可复用的现有设施

- **Dispatcher 模式**（`include/dispatcher.h:50-82`）：单例 + `(DeviceType, name)` → 函数指针，`REGISTER_KERNEL` 静态注册。
- **DeviceGuardImpl + Registry 模式**（`include/core/runtime/device_guard.h:199-215`、`src/core/runtime/cuda/cuda_guard_impl.cc:14-22`）：每 `DeviceType` 一个虚接口实现，`std::array<unique_ptr<...>, kMaxGpus=8>` + `std::once_flag` 懒初始化缓存，`INFINI_TRAIN_REGISTER_DEVICE_GUARD_IMPL` 宏静态注册。**Generator 设施完全镜像此模式。**
- **DeviceGuard RAII**（`include/core/runtime/device_guard.h:161-186`、`src/core/runtime/device_guard.cc`）：CUDA 设备切换已封装。
- **测试基类**（`tests/common/test_utils.h:48`）：`InfiniTrainTest : TestWithParam<DeviceType>` + `INFINI_TRAIN_REGISTER_TEST` 宏自动为 CPU + CUDA 实例化。

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

镜像 `DeviceGuardImpl` 模式，**不**复用 `DeviceGuardImpl`（职责不同：DeviceGuard 管设备切换/流，Generator 管 RNG 状态，生命周期不应耦合）。

### 3.2 文件布局

#### 新建文件


| 文件                                                                 | 作用                                                               |
| ------------------------------------------------------------------ | ---------------------------------------------------------------- |
| `infini_train/include/generator.h`                                 | 公共 `Generator` 句柄 + `default_generator()` / `manual_seed()` 自由函数 |
| `infini_train/include/core/generator/generator_impl.h`             | `GeneratorImpl` 抽象基类 + `GeneratorImplRegistry`                   |
| `infini_train/src/core/generator/generator_impl.cc`                | Registry 实现、默认池、`manual_seed`、`ResolveGenerator`                 |
| `infini_train/include/core/generator/cpu_generator_impl.h`         | `CPUGeneratorImpl` 声明                                            |
| `infini_train/src/core/generator/cpu/cpu_generator_impl.cc`        | CPU 实现 + `INFINI_TRAIN_REGISTER_GENERATOR_IMPL(kCPU, ...)`       |
| `infini_train/include/core/generator/cuda_generator_impl.h`        | `CUDAGeneratorImpl` 声明（`#ifdef USE_CUDA`）                        |
| `infini_train/src/core/generator/cuda/cuda_generator_impl.cu`      | CUDA 实现 + 注册器                                                    |
| `infini_train/src/core/generator/cuda/philox_engine.cuh`           | Philox4_32 device 端引擎（被 CUDA 随机 kernel include）                  |
| `infini_train/include/nn/modules/dropout.h`                        | `nn::Dropout` 模块                                                 |
| `infini_train/src/nn/modules/dropout.cc`                           | Dropout 模块实现                                                     |
| `infini_train/src/autograd/dropout.cc`                             | Dropout autograd Function                                        |
| `infini_train/src/kernels/cpu/random/{uniform,normal,dropout}.cc`  | CPU 随机 kernel                                                    |
| `infini_train/src/kernels/cuda/random/{uniform,normal,dropout}.cu` | CUDA 随机 kernel                                                   |
| `tests/generator/test_*.cc`                                        | 见 §7 测试矩阵                                                        |


#### 修改文件


| 文件                                                               | 改动                                                                                |
| ---------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `infini_train/include/tensor.h:151`                              | `Tensor::Uniform` 签名 `std::optional<std::mt19937>` → `std::optional<Generator>`   |
| `infini_train/src/tensor.cc:471`                                 | 转发到改造后的 `nn::init::Uniform`                                                       |
| `infini_train/include/nn/init.h:18,45,48`                        | 三个 init 函数签名换 Generator                                                           |
| `infini_train/src/nn/init.cc`                                    | 删除文件作用域 `gen` / `kRandomSeed`，删除 OMP 重复 seed bug，改走 dispatcher                    |
| `infini_train/src/nn/modules/{linear,sparse}.cc`                 | 6 处下游：传 `std::nullopt`（走默认）                                                       |
| `infini_train/src/nn/lora/{lora_linear,lora_parallel_linear}.cc` | 同上                                                                                |
| `infini_train/src/nn/parallel/tensor_parallel.cc`                | 同上                                                                                |
| `example/gpt2/checkpoint_loader.cc:31-34`                        | 删 `std::mt19937 gen{42}` 和 TODO 注释，改用 `Generator(Device::kCPU); g.ManualSeed(42)` |
| `example/llama3/checkpoint_loader.cc:29-32`                      | 同上                                                                                |
| `infini_train/CMakeLists.txt` / `tests/CMakeLists.txt`           | 新增编译目标                                                                            |


### 3.3 头文件依赖原则

- `generator.h` 仅 include `device.h` + 标准库 — **不暴露** `mt19937`/`curand`/`Philox`。
- `cuda_generator_impl.h` 用 `#ifdef USE_CUDA` 守护，公共 header 不 include。
- 算子 kernel 通过 `static_cast<CPUGeneratorImpl*>(impl.get())` 拿后端实现（PyTorch `intrusive_ptr<GeneratorImpl>` 风格）。

### 3.4 编译条件

- `USE_CUDA=OFF`：CUDA 文件全部不编译，Registry 只有 CPU 注册器，`Generator(Device::kCUDA)` 抛 `std::runtime_error`。
- `USE_CUDA=ON`：CUDA 部分通过 `INFINI_TRAIN_REGISTER_GENERATOR_IMPL` 静态注册，与 CPU 路径并存。

## 4. 接口设计

### 4.1 用户侧句柄 `infini_train::Generator`

```cpp
// infini_train/include/generator.h
namespace infini_train {

class Generator {
public:
    // 默认构造一个新独立 Generator（不与默认池共享）
    explicit Generator(Device device = Device(Device::DeviceType::kCPU, 0));

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
    std::shared_ptr<core::GeneratorImpl> impl_;
};

// 默认 Generator 入口
Generator& default_generator(Device device = Device(Device::DeviceType::kCPU, 0));

// 全局 seed 入口
void manual_seed(uint64_t seed);          // 同步 CPU + 所有 CUDA 默认（已/未初始化）
void manual_seed_cuda(uint64_t seed);     // 仅 CUDA 默认（PyTorch 兼容）

}  // namespace infini_train
```

### 4.2 抽象基类 `core::GeneratorImpl`

```cpp
namespace infini_train::core {

class GeneratorImpl {
public:
    explicit GeneratorImpl(Device device) : device_(device) {}
    virtual ~GeneratorImpl() = default;

    virtual void     SetCurrentSeed(uint64_t seed) = 0;
    virtual uint64_t CurrentSeed() const = 0;
    virtual std::vector<uint8_t> GetState() const = 0;
    virtual void     SetState(const std::vector<uint8_t>& state) = 0;
    virtual uint32_t StateMagic() const = 0;   // 用于 SetState 校验

    Device      device() const { return device_; }
    std::mutex& mutex() { return mutex_; }

    uint64_t Seed();   // 默认实现：random_device 取 64bit、调 SetCurrentSeed

protected:
    Device device_;
    mutable std::mutex mutex_;
};

class GeneratorImplRegistry {
public:
    static GeneratorImplRegistry& Instance();
    void Register(Device::DeviceType type, GeneratorImplFactory factory);
    std::shared_ptr<GeneratorImpl> Create(Device device) const;     // 显式构造
    std::shared_ptr<GeneratorImpl> Default(Device device);          // 懒初始化 + 缓存
    void ResetAllSeeds(uint64_t seed);                              // 全局 manual_seed
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
    explicit CPUGeneratorImpl(int /*device_index*/);  // index 必须为 0

    void     SetCurrentSeed(uint64_t seed) override;
    uint64_t CurrentSeed() const override { return seed_; }
    std::vector<uint8_t> GetState() const override;
    void     SetState(const std::vector<uint8_t>&) override;
    uint32_t StateMagic() const override { return kMagic; }

    std::mt19937_64& engine();   // 算子用，调用前需持 mutex()

private:
    static constexpr uint32_t kMagic   = 0x47555043;   // 'CPUG'
    static constexpr uint32_t kVersion = 1;
    uint64_t        seed_;
    std::mt19937_64 engine_;
};
```

### 4.4 CUDA 实现 `CUDAGeneratorImpl`（Philox 风格）

```cpp
class CUDAGeneratorImpl : public GeneratorImpl {
public:
    explicit CUDAGeneratorImpl(int device_index);

    void     SetCurrentSeed(uint64_t seed) override;
    uint64_t CurrentSeed() const override { return seed_; }
    std::vector<uint8_t> GetState() const override;
    void     SetState(const std::vector<uint8_t>&) override;
    uint32_t StateMagic() const override { return kMagic; }

    struct PhiloxState { uint64_t seed; uint64_t offset; };
    // 推进 offset：消耗 num_elements 个随机数，每 4 个共享一个 Philox 4-tuple
    PhiloxState NextPhiloxState(uint64_t num_elements);

private:
    static constexpr uint32_t kMagic   = 0x47445543;   // 'CUDG'
    static constexpr uint32_t kVersion = 1;
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
| u32 LE  | u32 LE  | u64 LE         |    bytes       |
+---------+---------+----------------+----------------+
```

`SetState` 先校验 `magic == this->StateMagic()`，不匹配抛 `std::runtime_error`，version 不匹配同样抛。

- **CPU payload**：`mt19937_64` 内部 `state[]`（312×u64） + 当前位置（u64） + `seed_`（u64）。
- **CUDA payload**：`seed_`（u64） + `offset_`（u64）。

### 4.6 默认 Generator + manual_seed 行为

懒初始化（镜像 `cuda_guard_impl.cc:14-22` 模式）：

```cpp
std::shared_ptr<GeneratorImpl> GeneratorImplRegistry::Default(Device device) {
    if (device.IsCPU()) {
        std::call_once(default_cpu_once_, [&]{
            default_cpu_ = factories_[kCPU](0);
            default_cpu_->SetCurrentSeed(LastUserSeedOrRandom());
        });
        return default_cpu_;
    }
    int idx = device.Index();
    std::call_once(default_cuda_once_[idx], [&]{
        default_cuda_[idx] = factories_[kCUDA](idx);
        default_cuda_[idx]->SetCurrentSeed(LastUserSeedOrRandom());
    });
    return default_cuda_[idx];
}
```

`LastUserSeedOrRandom()` 是 Registry 私有方法：返回 `last_user_seed_`（`std::optional<uint64_t>`，由 `manual_seed` 写入）；若为空则用 `std::random_device` 现取。Registry 字段一览：

```cpp
class GeneratorImplRegistry {
private:
    std::array<GeneratorImplFactory, /*kCount=*/2> factories_;
    std::shared_ptr<GeneratorImpl> default_cpu_;
    std::array<std::shared_ptr<GeneratorImpl>, /*kMaxGpus=*/8> default_cuda_;
    std::once_flag default_cpu_once_;
    std::array<std::once_flag, 8> default_cuda_once_;
    std::optional<uint64_t> last_user_seed_;   // manual_seed 最近一次设置
    std::mutex registry_mutex_;
};
```

`manual_seed(s)` 行为：

1. 记录 `last_user_seed_ = s`；
2. 强制初始化 CPU 默认（如未初始化），并 `SetCurrentSeed(s)`；
3. 对**已初始化**的 CUDA 默认 `SetCurrentSeed(s)`；未初始化的不强制初始化（避免无意激活所有 GPU），但下次首次访问会自动用 `last_user_seed_`。
4. 不影响用户**显式**构造的 Generator。

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
    {t->GetDevice().Type(), "UniformRandom"}, t, low, high, impl.get());
```

> 注意：kernel 收到的是 `core::GeneratorImpl*` 裸指针，仅在 kernel 调用栈内有效；shared_ptr 由调用层（init.cc / dropout.cc 等）持有，保证整个调用周期内 impl 存活。

## 5. 算子改造

### 5.1 公共签名

```cpp
// init.h / tensor.h / functional.h
void Normal(Tensor&, float mean, float std,
            std::optional<Generator> = std::nullopt);
void Uniform(Tensor&, float low, float high,
             std::optional<Generator> = std::nullopt);
void KaimingUniform(Tensor&, float a,
                    std::optional<Generator> = std::nullopt);
Tensor Rand(const Shape&, DataType, Device,
            std::optional<Generator> = std::nullopt);
Tensor Randn(const Shape&, DataType, Device,
             std::optional<Generator> = std::nullopt);
Tensor Dropout(const Tensor&, float p, bool training,
               std::optional<Generator> = std::nullopt);
```

### 5.2 Dispatcher key


| Key                                                | CPU 实现                  | CUDA 实现                  |
| -------------------------------------------------- | ----------------------- | ------------------------ |
| `(*, "UniformRandom")`                             | `cpu/random/uniform.cc` | `cuda/random/uniform.cu` |
| `(*, "NormalRandom")`                              | `cpu/random/normal.cc`  | `cuda/random/normal.cu`  |
| `(*, "DropoutForward")` / `(*, "DropoutBackward")` | `cpu/random/dropout.cc` | `cuda/random/dropout.cu` |


`Rand`/`Randn` 复用 Uniform/Normal kernel，仅在 wrapper 层先 `Empty(shape)` 再调对应 kernel。

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
    auto* cuda_impl = static_cast<core::CUDAGeneratorImpl*>(impl);
    std::lock_guard<std::mutex> lock(cuda_impl->mutex());
    auto ph = cuda_impl->NextPhiloxState(t->NumElements());  // 推进 offset
    DropoutKernel<<<...>>>(..., ph.seed, ph.offset);
}
```

**Offset 推进规则**：每次调用 host 端把 offset 推进 `ceil(num_elements / 4)`（Philox 一次产 4 个 32-bit），下一次调用用新 offset，序列不重叠。

### 5.5 Dropout 完整链路

新增三层：

1. **kernel**（`{cpu,cuda}/random/dropout.{cc,cu}`）：forward 生成 mask + scale；backward 用 mask 反传梯度。
2. **autograd Function**（`autograd/dropout.cc`）：Forward 调 `DropoutForward` 并把 mask 存入 `saved_tensors`；Backward 调 `DropoutBackward(grad, mask, p)`。
3. **module**（`nn::Dropout(p)`）：训练态走 autograd，推断态直接返回输入。

## 6. 与 PyTorch 对齐


| 行为                          | PyTorch             | InfiniTrain (本设计)         |
| --------------------------- | ------------------- | ------------------------- |
| `torch.manual_seed(s)` 影响范围 | CPU + 所有 CUDA 默认    | CPU + 已/未初始化 CUDA 默认      |
| 显式 `Generator()`            | 不受 `manual_seed` 影响 | 一致                        |
| 默认 Generator 多次获取           | 同一份                 | 一致（shared_ptr）            |
| 不同 CUDA 设备默认                | 各自独立（共用同一 seed）     | 一致                        |
| 未 `manual_seed` 时           | `random_device`     | 一致                        |
| state 格式校验                  | tensor 类型 + size 检查 | magic + version + size 检查 |
| 同 seed 数值一致                 | bit 一致              | **不要求**（声明）               |
| CPU vs CUDA 同 seed 一致       | **不要求**             | **不要求**（与 PyTorch 一致）     |


## 7. 测试矩阵

所有测试用 `InfiniTrainTest : TestWithParam<DeviceType>`，自动覆盖 CPU + CUDA。


| 测试文件                                        | 内容                                                                                                 |
| ------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `tests/generator/test_generator_basic.cc`   | 句柄构造、`device()` 查询、显式 vs 默认获取、`default_generator()` 多次返回共享状态                                       |
| `tests/generator/test_seed.cc`              | `ManualSeed(123)` → 取 N → `ManualSeed(123)` → 序列一致；不同 seed 不同；**显式包含状态推进 trap test**（连续两次取 N 必须不同） |
| `tests/generator/test_state.cc`             | `s = GetState(); 取 N; SetState(s); 再取 N` 与第一次相同                                                    |
| `tests/generator/test_state_validation.cc`  | CPU state 喂给 CUDA 应抛异常（magic 不匹配）；version 不匹配抛异常；payload size 不一致抛异常                               |
| `tests/generator/test_default.cc`           | `manual_seed(s)` 后 CPU/CUDA 默认都用 `s`；不同 CUDA 设备默认互相独立；未初始化 CUDA 在 `manual_seed` 后首次访问也用 `s`        |
| `tests/generator/test_dispatch.cc`          | 不传 Generator 时 CPU tensor 走 CPU 默认、CUDA tensor 走 CUDA 默认；显式传入时不误用默认                                |
| `tests/generator/test_ops_uniform.cc`       | 同 seed 同形状 → 结果完全一致；不同 seed → 不同                                                                   |
| `tests/generator/test_ops_normal.cc`        | 同上，正态分布                                                                                            |
| `tests/generator/test_ops_dropout.cc`       | 同 seed → mask 一致；mask 大致符合 1-p 概率；scale 正确；backward 梯度正确                                           |
| `tests/generator/test_ops_kaiming.cc`       | 同 seed → 权重一致；fan_in 计算正确                                                                          |
| `tests/generator/test_pytorch_alignment.cc` | 文档型：复现"先 manual_seed → 算子 A → 算子 B"序列，验证语义对齐（不要求 bit 一致）                                           |


## 8. 报告内容

`docs/generator-design.md`（区别于本 spec，作为提交报告）：

1. **设计参考**：PyTorch `c10::GeneratorImpl` / `CPUGeneratorImpl` / `CUDAGeneratorImpl` / Philox4_32 / `at::detail::DefaultGenerator` 对应关系。
2. **接口对照表**：本实现 vs PyTorch 接口逐一列出（一致 / 缺失 / 命名差异）。
3. **行为对齐情况**：详尽列出每条对齐行为及验证方法。
4. **未对齐项及原因**：底层算法不同 → 同 seed 数值不一致；CPU 与 CUDA 同 seed 数值不一致（符合选题要求，故意为之）。
5. **可扩展方向**：分布式 RNG、graph capture 友好的 Philox 状态、其他后端接入路径、OMP 并行 RNG。

## 9. 验证清单（提供给 reviewer）

```bash
# 构建
mkdir -p build && cmake -B build -DUSE_CUDA=ON -DUSE_NCCL=ON && cmake --build build -j

# 跑测试
cd build && ctest -V -L generator
```

可复现性脚本：`scripts/check_reproducibility.sh` 跑两次 + diff，预期 0 差异。

## 10. 不在范围

- OMP 并行 RNG 优化（写入报告"后续可扩展方向"）。
- Bernoulli / randint 等其他随机算子（非选题强制）。
- ZeRO / 分布式 RNG（与本任务正交）。
- BF16 等非 FP32 dtype 下的 random（先做 FP32，dtype 扩展作为后续）。

## 11. 风险与缓解


| 风险                                 | 缓解                                                                                                                                        |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Philox 自实现引入 bug                   | 直接复制 PyTorch `at::Philox4_32` 的 device-side 引擎到 `philox_engine.cuh`（公开 source 可直接对照）；测试覆盖：同 (seed, offset, tid) 序列必须可复现，offset 推进后序列必须不重叠 |
| OMP 串行化导致 Uniform 大 tensor init 变慢 | 性能不在本次验收，但报告内说明并给基准数据                                                                                                                     |
| 全局 `manual_seed` 与多线程并发的竞态         | Registry 内部 `mutex_` 保护，状态修改原子化                                                                                                           |
| `std::shared_ptr` 引入开销             | Generator 句柄拷贝廉价，与 PyTorch `intrusive_ptr` 等价                                                                                             |
| 默认 Generator 在静态析构时的释放顺序           | Registry 自身是 Meyers singleton，析构先于全局变量；测试覆盖                                                                                               |


