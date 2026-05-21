# Generator 抽象 — Phase 1 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 落地 Generator 抽象 Phase 1 — 引入 `core::GeneratorImpl` + `GeneratorImplRegistry` + `CPUGeneratorImpl`，提供公共 `Generator` 句柄、`default_generator()`、`manual_seed()`、`manual_seed_cuda()` 入口，新增 CPU `UniformRandom`/`NormalRandom` kernel 通过 Dispatcher 暴露，改造 `nn::init::Uniform/Normal/KaimingUniform` 与 `Tensor::Uniform` 走新链路，并补足基础测试。

**Architecture:** PImpl 句柄 + 后端工厂注册表（镜像 `DeviceGuardImplRegistry`）；CPU 后端为 `mt19937_64` + 锁；状态序列化使用 `magic|version|payload_size|payload` 头部；算子通过 `Dispatcher::Call` 调用 kernel，`ResolveGenerator` 处理显式/默认 Generator 的回落。

**Tech Stack:** C++20，CMake，GoogleTest，glog/gflags。**仅 CPU**——CUDA 实现属于 Phase 2，不在本计划。所有命令以 `BUILD_TEST=ON` `USE_CUDA=OFF` 验证（Phase 2 再开 `USE_CUDA=ON`）。

**Spec 依据:** [`docs/superpowers/specs/2026-05-19-generator-abstraction-design.md`](../specs/2026-05-19-generator-abstraction-design.md)（frozen 2026-05-21）。

---

## 文件结构（Phase 1 范围）

### 新建

| 文件 | 责任 |
|------|------|
| `infini_train/include/generator.h` | 公共 `Generator` 句柄 + `default_generator()` / `manual_seed()` / `manual_seed_cuda()` 自由函数 |
| `infini_train/include/core/generator/generator_impl.h` | `GeneratorImpl` 抽象基类 + `GeneratorImplRegistry` + `INFINI_TRAIN_REGISTER_GENERATOR_IMPL` 宏 |
| `infini_train/include/core/generator/cpu_generator_impl.h` | `CPUGeneratorImpl` 声明（kernel 端 include 用） |
| `infini_train/src/core/generator/generator_impl.cc` | Registry 实现 + 默认池 + `manual_seed` + `Generator` 句柄方法 + `ResolveGenerator` |
| `infini_train/src/core/generator/cpu/cpu_generator_impl.cc` | CPU 实现 + `INFINI_TRAIN_REGISTER_GENERATOR_IMPL(kCPU, ...)` |
| `infini_train/src/kernels/cpu/uniform_random.cc` | `UniformRandom` kernel（FP32） |
| `infini_train/src/kernels/cpu/normal_random.cc` | `NormalRandom` kernel（FP32） |
| `tests/generator/CMakeLists.txt` | 测试套 CMake |
| `tests/generator/test_generator_basic.cc` | 句柄构造、`device()`、显式/默认共享语义 |
| `tests/generator/test_seed.cc` | seed 复现 + `Seed()` vs `InitialSeed()` + 状态推进 trap |
| `tests/generator/test_state.cc` | `GetState`/`SetState` 往返 |
| `tests/generator/test_state_validation.cc` | CPU SetState 格式校验（CPU-only target） |
| `tests/generator/test_default.cc` | `default_generator()` + `manual_seed()` 行为（CPU 部分） |
| `tests/generator/test_dispatch.cc` | 不传 Generator 时 CPU tensor 走 CPU 默认 |
| `tests/generator/test_ops_uniform.cc` | UniformRandom 同 seed 一致性 |
| `tests/generator/test_ops_normal.cc` | NormalRandom 同 seed 一致性 |
| `tests/generator/test_ops_kaiming.cc` | KaimingUniform 同 seed + fan_in 计算 |

### 修改

| 文件 | 改动 |
|------|------|
| `infini_train/include/tensor.h` | `Tensor::Uniform` 签名改 `std::optional<Generator>`；删 `<random>`；加 `#include "infini_train/include/generator.h"` |
| `infini_train/src/tensor.cc` | `Tensor::Uniform` 转发签名同步 |
| `infini_train/include/nn/init.h` | 三个 init 函数签名改 `std::optional<Generator>`；删 `<random>` |
| `infini_train/src/nn/init.cc` | 删 `kRandomSeed`、文件作用域 `gen`、双 OMP bug 路径；改走 dispatcher |
| `example/gpt2/checkpoint_loader.cc` | 删 `kRandomSeed`、`static std::mt19937 gen`、TODO 注释、`<random>` include |
| `example/llama3/checkpoint_loader.cc` | 同上 |
| `tests/CMakeLists.txt` | 增加 `add_subdirectory(generator)` |

### 不在本计划范围（Phase 2+）

- `infini_train/include/core/generator/cuda_generator_impl.h` 及 `.cu` 实现
- `philox_engine.cuh`
- `cuda/uniform_random.cu`、`cuda/normal_random.cu`
- Dropout 完整链路（kernel + autograd + module）
- `nn::function::Rand`/`Randn`
- `Module::Train`/`Eval`/`IsTraining`
- `docs/generator-design.md` 报告
- `scripts/check_reproducibility.sh`

---

## 验证命令（贯穿整份计划）

Phase 1 仅 CPU，构建配置：

```bash
cd /Users/guozhihao/work/mlsys/InfiniTrain
cmake -S . -B build -DBUILD_TEST=ON -DUSE_CUDA=OFF -DUSE_NCCL=OFF
cmake --build build -j
ctest --test-dir build -V -R '^test_generator_'
```

**约定**：每个 task 完成后必须能成功 `cmake --build build -j` 并跑通该 task 引入的测试；触及全仓签名的 task（10、11）还要确保已有套件不退化。

---

## Task 1: 测试脚手架与空 generator.h

**Files:**
- Create: `tests/generator/CMakeLists.txt`
- Create: `tests/generator/test_generator_basic.cc`
- Modify: `tests/CMakeLists.txt`

**目的：** 把 `tests/generator` 接入构建系统并跑通一个最小 placeholder gtest，证明测试 target 已就绪；`generator.h` 暂不创建（下一 task 引入）。

- [ ] **Step 1: 写 `tests/generator/CMakeLists.txt`**

```cmake
# ==========================================================================
# Generator tests
# ==========================================================================
file(GLOB GENERATOR_SHARED_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cc)
infini_train_add_test_suite(test_generator SOURCES ${GENERATOR_SHARED_SOURCES})

file(GLOB GENERATOR_CPU_ONLY_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cpu_only/test_*.cc)
if(GENERATOR_CPU_ONLY_SOURCES)
  infini_train_add_test(test_generator_cpu_only
    SOURCES ${GENERATOR_CPU_ONLY_SOURCES}
    LABELS cpu
  )
endif()
```

- [ ] **Step 2: 写最小可运行 placeholder 测试 `tests/generator/test_generator_basic.cc`**

```cpp
#include "gtest/gtest.h"

#include "infini_train/include/device.h"
#include "tests/common/test_utils.h"

using namespace infini_train;

class GeneratorBasicTest : public infini_train::test::InfiniTrainTest {};

TEST_P(GeneratorBasicTest, Placeholder) {
    // Will be replaced in Task 4 with real coverage.
    EXPECT_TRUE(true);
}

INFINI_TRAIN_REGISTER_TEST(GeneratorBasicTest);
```

- [ ] **Step 3: 在 `tests/CMakeLists.txt` 末尾追加注册**

把现有文件末尾的 `add_subdirectory(transformer)` 后追加：

```cmake
# Generator tests
add_subdirectory(generator)
```

- [ ] **Step 4: 构建并跑测试**

```bash
cmake -S . -B build -DBUILD_TEST=ON -DUSE_CUDA=OFF -DUSE_NCCL=OFF
cmake --build build -j
ctest --test-dir build -V -R '^test_generator_'
```

预期：`test_generator_cpu` 成功，单条 `GeneratorBasicTest.Placeholder/CPU` 通过。

- [ ] **Step 5: Commit**

```bash
git add tests/generator/CMakeLists.txt tests/generator/test_generator_basic.cc tests/CMakeLists.txt
git commit -m "test(generator): add tests/generator scaffold with placeholder gtest"
```

---

## Task 2: `GeneratorImpl` 抽象基类 + `GeneratorImplRegistry` 接口

**Files:**
- Create: `infini_train/include/core/generator/generator_impl.h`
- Create: `infini_train/src/core/generator/generator_impl.cc`（仅基类默认 `Seed()` + Registry 单例骨架，**不**含 `Default`/`ResetAllSeeds` 主体——Task 5 再补完）

**目的：** 落地纯抽象接口与注册表骨架；让任意后端可以注册到 `factories_`，但 `Default()`/`ResetAllSeeds()` 还是占位（在 Task 5 完成）。

- [ ] **Step 1: 创建 `infini_train/include/core/generator/generator_impl.h`**

```cpp
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

#include "infini_train/include/device.h"

namespace infini_train::core {

class GeneratorImpl {
public:
    explicit GeneratorImpl(Device device) : device_(device) {}
    virtual ~GeneratorImpl() = default;

    GeneratorImpl(const GeneratorImpl &) = delete;
    GeneratorImpl &operator=(const GeneratorImpl &) = delete;

    virtual void SetCurrentSeed(uint64_t seed) = 0;
    virtual uint64_t CurrentSeed() const = 0;
    virtual uint64_t InitialSeed() const = 0;
    virtual std::vector<uint8_t> GetState() const = 0;
    virtual void SetState(const std::vector<uint8_t> &state) = 0;
    virtual uint32_t StateMagic() const = 0;

    // Default impl: s = RandomSeed64(); Reseed(s); return CurrentSeed();
    // 不修改 InitialSeed()。
    virtual uint64_t Seed();

    Device device() const { return device_; }
    std::mutex &mutex() { return mutex_; }

protected:
    virtual void Reseed(uint64_t seed) = 0;

    Device device_;
    mutable std::mutex mutex_;
};

using GeneratorImplFactory = std::function<std::shared_ptr<GeneratorImpl>(int8_t /*device_index*/)>;

// kMaxGpus 与 cuda_guard_impl.cc:14 保持一致。
inline constexpr int kMaxGpus = 8;

class GeneratorImplRegistry {
public:
    static GeneratorImplRegistry &Instance();

    void Register(Device::DeviceType type, GeneratorImplFactory factory);

    // CPU: device.index() 必须为 0；其它 index 触发 CHECK 失败。
    // CUDA: 使用 device.index()。Phase 1 未注册 CUDA 工厂，调用即抛 std::runtime_error。
    std::shared_ptr<GeneratorImpl> Create(Device device) const;

    // 懒初始化默认池；若 manual_seed 已写过 last_user_seed_，用它做种子。
    std::shared_ptr<GeneratorImpl> Default(Device device);

    // manual_seed() 实现体（Task 5 落地）。
    void ResetAllSeeds(uint64_t seed);

    // manual_seed_cuda() 实现体（Phase 1 内 CUDA 工厂未注册即视为 no-op）。
    void ResetCudaSeeds(uint64_t seed);

private:
    GeneratorImplRegistry() = default;

    uint64_t LastUserSeedOrRandom();

    std::unordered_map<Device::DeviceType, GeneratorImplFactory> factories_;
    std::shared_ptr<GeneratorImpl> default_cpu_;
    std::array<std::shared_ptr<GeneratorImpl>, kMaxGpus> default_cuda_{};
    std::once_flag default_cpu_once_;
    std::array<std::once_flag, kMaxGpus> default_cuda_once_{};
    std::atomic<bool> cpu_initialized_{false};
    std::array<std::atomic<bool>, kMaxGpus> cuda_initialized_{};
    std::optional<uint64_t> last_user_seed_;
    std::mutex registry_mutex_;
};

}  // namespace infini_train::core

#define INFINI_TRAIN_REGISTER_GENERATOR_IMPL(device_type, FactoryFn)                                                   \
    namespace {                                                                                                        \
    struct GenReg_##device_type {                                                                                      \
        GenReg_##device_type() {                                                                                       \
            ::infini_train::core::GeneratorImplRegistry::Instance().Register(                                          \
                ::infini_train::Device::DeviceType::device_type, FactoryFn);                                           \
        }                                                                                                              \
    };                                                                                                                 \
    static GenReg_##device_type _gen_reg_##device_type;                                                                \
    }  // namespace
```

- [ ] **Step 2: 创建 `infini_train/src/core/generator/generator_impl.cc`（骨架）**

```cpp
#include "infini_train/include/core/generator/generator_impl.h"

#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>

#include "glog/logging.h"

namespace infini_train::core {

namespace {

uint64_t RandomDeviceSeed64() {
    std::random_device rd;
    const uint64_t hi = static_cast<uint64_t>(rd());
    const uint64_t lo = static_cast<uint64_t>(rd());
    return (hi << 32) ^ lo;
}

}  // namespace

uint64_t GeneratorImpl::Seed() {
    std::lock_guard<std::mutex> lk(mutex_);
    const uint64_t s = RandomDeviceSeed64();
    Reseed(s);
    return CurrentSeed();
}

GeneratorImplRegistry &GeneratorImplRegistry::Instance() {
    static GeneratorImplRegistry instance;
    return instance;
}

void GeneratorImplRegistry::Register(Device::DeviceType type, GeneratorImplFactory factory) {
    std::lock_guard<std::mutex> lk(registry_mutex_);
    CHECK(factories_.emplace(type, std::move(factory)).second)
        << "GeneratorImpl already registered for device type " << static_cast<int>(type);
}

std::shared_ptr<GeneratorImpl> GeneratorImplRegistry::Create(Device device) const {
    auto it = factories_.find(device.type());
    if (it == factories_.end()) {
        throw std::runtime_error("No GeneratorImpl factory registered for device type "
                                 + std::to_string(static_cast<int>(device.type()))
                                 + " (build with USE_CUDA=ON for CUDA support).");
    }
    if (device.IsCPU()) {
        CHECK_EQ(device.index(), 0) << "CPU Generator must use device index 0";
        return it->second(0);
    }
    const int8_t idx = device.index();
    CHECK(0 <= idx && idx < kMaxGpus) << "CUDA device index out of range: " << static_cast<int>(idx);
    return it->second(idx);
}

uint64_t GeneratorImplRegistry::LastUserSeedOrRandom() {
    {
        std::lock_guard<std::mutex> lk(registry_mutex_);
        if (last_user_seed_) {
            return *last_user_seed_;
        }
    }
    return RandomDeviceSeed64();
}

// NOTE: Default()/ResetAllSeeds()/ResetCudaSeeds() 在 Task 5 实现。
std::shared_ptr<GeneratorImpl> GeneratorImplRegistry::Default(Device /*device*/) {
    LOG(FATAL) << "GeneratorImplRegistry::Default() not implemented yet (Task 5)";
    return nullptr;
}

void GeneratorImplRegistry::ResetAllSeeds(uint64_t /*seed*/) {
    LOG(FATAL) << "GeneratorImplRegistry::ResetAllSeeds() not implemented yet (Task 5)";
}

void GeneratorImplRegistry::ResetCudaSeeds(uint64_t /*seed*/) {
    LOG(FATAL) << "GeneratorImplRegistry::ResetCudaSeeds() not implemented yet (Task 5)";
}

}  // namespace infini_train::core
```

- [ ] **Step 3: 构建确认无错（暂无新测试）**

```bash
cmake --build build -j
```

预期：编译通过。Registry 还没人 use，但骨架存在。

- [ ] **Step 4: Commit**

```bash
git add infini_train/include/core/generator/generator_impl.h infini_train/src/core/generator/generator_impl.cc
git commit -m "feat(generator): add GeneratorImpl base + Registry skeleton"
```

---

## Task 3: `CPUGeneratorImpl` 基本能力（seed/Reseed/CurrentSeed/InitialSeed）

**Files:**
- Create: `infini_train/include/core/generator/cpu_generator_impl.h`
- Create: `infini_train/src/core/generator/cpu/cpu_generator_impl.cc`
- Test: `tests/generator/cpu_only/test_cpu_generator_impl.cc`

**目的：** TDD 落地 `CPUGeneratorImpl` 的 seed 接口与 `mt19937_64` 引擎；`GetState/SetState` 占位实现，留到 Task 6 完整化。

- [ ] **Step 1: 写测试 `tests/generator/cpu_only/test_cpu_generator_impl.cc`**

```cpp
#include <memory>

#include "gtest/gtest.h"

#include "infini_train/include/core/generator/cpu_generator_impl.h"

using infini_train::core::CPUGeneratorImpl;

TEST(CPUGeneratorImplTest, ManualSeedSetsCurrentAndInitialSeed) {
    CPUGeneratorImpl impl(0);
    impl.SetCurrentSeed(42);
    EXPECT_EQ(impl.CurrentSeed(), 42u);
    EXPECT_EQ(impl.InitialSeed(), 42u);
}

TEST(CPUGeneratorImplTest, EngineAdvancesAcrossCalls) {
    CPUGeneratorImpl impl(0);
    impl.SetCurrentSeed(123);
    auto &eng = impl.engine();
    const uint64_t a = eng();
    const uint64_t b = eng();
    EXPECT_NE(a, b);
}

TEST(CPUGeneratorImplTest, SameManualSeedReproduces) {
    CPUGeneratorImpl a(0), b(0);
    a.SetCurrentSeed(7);
    b.SetCurrentSeed(7);
    EXPECT_EQ(a.engine()(), b.engine()());
    EXPECT_EQ(a.engine()(), b.engine()());
}

TEST(CPUGeneratorImplTest, SeedRandomizesCurrentButPreservesInitial) {
    CPUGeneratorImpl impl(0);
    impl.SetCurrentSeed(99);
    const uint64_t initial_before = impl.InitialSeed();
    const uint64_t new_seed = impl.Seed();
    EXPECT_EQ(impl.CurrentSeed(), new_seed);
    EXPECT_EQ(impl.InitialSeed(), initial_before);
}

TEST(CPUGeneratorImplTest, IndexNonZeroAborts) {
    EXPECT_DEATH(CPUGeneratorImpl(1), "");
}
```

> 注：`tests/generator/cpu_only/` 目录由 `tests/generator/CMakeLists.txt` Step 1 已经 `file(GLOB ...)`，无需再改 CMake。

- [ ] **Step 2: 创建 `infini_train/include/core/generator/cpu_generator_impl.h`**

```cpp
#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "infini_train/include/core/generator/generator_impl.h"
#include "infini_train/include/device.h"

namespace infini_train::core {

class CPUGeneratorImpl : public GeneratorImpl {
public:
    explicit CPUGeneratorImpl(int8_t device_index);

    void SetCurrentSeed(uint64_t seed) override;
    uint64_t CurrentSeed() const override { return seed_; }
    uint64_t InitialSeed() const override { return initial_seed_; }

    std::vector<uint8_t> GetState() const override;
    void SetState(const std::vector<uint8_t> &state) override;
    uint32_t StateMagic() const override { return kMagic; }

    // 调用前持有 mutex()。
    std::mt19937_64 &engine() { return engine_; }

protected:
    void Reseed(uint64_t seed) override;

private:
    static constexpr uint32_t kMagic = 0x47555043;   // 'CPUG'
    static constexpr uint32_t kVersion = 1;

    uint64_t initial_seed_{0};
    uint64_t seed_{0};
    std::mt19937_64 engine_{0};
};

}  // namespace infini_train::core
```

- [ ] **Step 3: 创建 `infini_train/src/core/generator/cpu/cpu_generator_impl.cc`（不含完整 GetState/SetState）**

```cpp
#include "infini_train/include/core/generator/cpu_generator_impl.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/core/generator/generator_impl.h"
#include "infini_train/include/device.h"

namespace infini_train::core {

CPUGeneratorImpl::CPUGeneratorImpl(int8_t device_index)
    : GeneratorImpl(Device(Device::DeviceType::kCPU, 0)) {
    CHECK_EQ(device_index, 0) << "CPU Generator only supports device index 0";
}

void CPUGeneratorImpl::SetCurrentSeed(uint64_t seed) {
    initial_seed_ = seed;
    seed_ = seed;
    engine_.seed(seed);
}

void CPUGeneratorImpl::Reseed(uint64_t seed) {
    seed_ = seed;
    engine_.seed(seed);
}

// Task 6 中替换为带 magic/version 头部的实现。
std::vector<uint8_t> CPUGeneratorImpl::GetState() const {
    LOG(FATAL) << "CPUGeneratorImpl::GetState() not implemented yet (Task 6)";
    return {};
}

void CPUGeneratorImpl::SetState(const std::vector<uint8_t> & /*state*/) {
    LOG(FATAL) << "CPUGeneratorImpl::SetState() not implemented yet (Task 6)";
}

namespace {

std::shared_ptr<GeneratorImpl> CpuFactory(int8_t device_index) {
    return std::make_shared<CPUGeneratorImpl>(device_index);
}

}  // namespace

}  // namespace infini_train::core

INFINI_TRAIN_REGISTER_GENERATOR_IMPL(kCPU, ::infini_train::core::CpuFactory)
```

> **重要**：`INFINI_TRAIN_REGISTER_GENERATOR_IMPL` 必须放到 `.cc` 文件最外层（不在 `infini_train::core` 命名空间内），否则匿名命名空间嵌套会引起重定义。宏内部 `::infini_train::...` 已限定。

- [ ] **Step 4: 跑测试**

```bash
cmake --build build -j
ctest --test-dir build -V -R '^test_generator_cpu_only$'
```

预期：5 条 CPUGeneratorImpl test 全部通过。`SeedRandomizesCurrentButPreservesInitial` 依赖基类默认 `Seed()` 实现（Task 2 已写）。

- [ ] **Step 5: Commit**

```bash
git add infini_train/include/core/generator/cpu_generator_impl.h \
        infini_train/src/core/generator/cpu/cpu_generator_impl.cc \
        tests/generator/cpu_only/test_cpu_generator_impl.cc
git commit -m "feat(generator): add CPUGeneratorImpl with mt19937_64 engine + tests"
```

---

## Task 4: `Generator` 用户句柄 + `default_generator()` 占位 + Registry::Create 测试

**Files:**
- Create: `infini_train/include/generator.h`
- Modify: `infini_train/src/core/generator/generator_impl.cc`（追加 `Generator` 类方法实现）
- Modify: `tests/generator/test_generator_basic.cc`（替换 placeholder）

**目的：** 引入 `Generator` PImpl 句柄，绑定 `Registry::Create()`；`default_generator()` 与 `manual_seed()` 暂保持占位（Task 5 完整实现）。

- [ ] **Step 1: 写完整测试替换 placeholder（`tests/generator/test_generator_basic.cc`）**

```cpp
#include <memory>

#include "gtest/gtest.h"

#include "infini_train/include/device.h"
#include "infini_train/include/generator.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class GeneratorBasicTest : public infini_train::test::InfiniTrainTest {};

TEST_P(GeneratorBasicTest, ConstructionAttachesDevice) {
    Generator gen(GetDevice());
    EXPECT_EQ(gen.device(), GetDevice());
    EXPECT_NE(gen.impl(), nullptr);
}

TEST_P(GeneratorBasicTest, ManualSeedRoundtripsCurrentAndInitial) {
    Generator gen(GetDevice());
    gen.ManualSeed(2026);
    EXPECT_EQ(gen.InitialSeed(), 2026u);
}

TEST_P(GeneratorBasicTest, CopyShareImpl) {
    Generator a(GetDevice());
    a.ManualSeed(1);
    Generator b = a;
    EXPECT_EQ(a.impl().get(), b.impl().get());
    b.ManualSeed(2);
    EXPECT_EQ(a.InitialSeed(), 2u);  // 浅共享
}

INFINI_TRAIN_REGISTER_TEST(GeneratorBasicTest);
```

- [ ] **Step 2: 创建 `infini_train/include/generator.h`**

```cpp
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "infini_train/include/device.h"

namespace infini_train {
namespace core {
class GeneratorImpl;
class GeneratorImplRegistry;
}  // namespace core

class Generator {
public:
    explicit Generator(Device device = Device(Device::DeviceType::kCPU, 0));

    Generator(const Generator &) = default;
    Generator &operator=(const Generator &) = default;
    Generator(Generator &&) = default;
    Generator &operator=(Generator &&) = default;

    void ManualSeed(uint64_t seed);
    uint64_t Seed();
    uint64_t InitialSeed() const;

    std::vector<uint8_t> GetState() const;
    void SetState(const std::vector<uint8_t> &state);

    Device device() const;

    const std::shared_ptr<core::GeneratorImpl> &impl() const { return impl_; }

private:
    explicit Generator(std::shared_ptr<core::GeneratorImpl> impl);
    static Generator FromImpl(std::shared_ptr<core::GeneratorImpl> impl);

    friend class core::GeneratorImplRegistry;
    friend Generator &default_generator(Device device);

    std::shared_ptr<core::GeneratorImpl> impl_;
};

Generator &default_generator(Device device = Device(Device::DeviceType::kCPU, 0));

void manual_seed(uint64_t seed);
void manual_seed_cuda(uint64_t seed);

}  // namespace infini_train
```

- [ ] **Step 3: 在 `infini_train/src/core/generator/generator_impl.cc` 末尾追加 `Generator` 句柄实现**

在文件末尾、`}  // namespace infini_train::core` 之外、添加新顶层段：

```cpp
// ---------------------------------------------------------------------------
// Generator handle (PImpl) — bound to GeneratorImplRegistry::Create().
// ---------------------------------------------------------------------------

#include "infini_train/include/generator.h"

namespace infini_train {

Generator::Generator(Device device)
    : impl_(core::GeneratorImplRegistry::Instance().Create(device)) {}

Generator::Generator(std::shared_ptr<core::GeneratorImpl> impl) : impl_(std::move(impl)) {}

Generator Generator::FromImpl(std::shared_ptr<core::GeneratorImpl> impl) {
    return Generator(std::move(impl));
}

void Generator::ManualSeed(uint64_t seed) {
    std::lock_guard<std::mutex> lk(impl_->mutex());
    impl_->SetCurrentSeed(seed);
}

uint64_t Generator::Seed() { return impl_->Seed(); }

uint64_t Generator::InitialSeed() const {
    std::lock_guard<std::mutex> lk(impl_->mutex());
    return impl_->InitialSeed();
}

std::vector<uint8_t> Generator::GetState() const {
    std::lock_guard<std::mutex> lk(impl_->mutex());
    return impl_->GetState();
}

void Generator::SetState(const std::vector<uint8_t> &state) {
    std::lock_guard<std::mutex> lk(impl_->mutex());
    impl_->SetState(state);
}

Device Generator::device() const { return impl_->device(); }

// 占位（Task 5 完整实现）。
Generator &default_generator(Device /*device*/) {
    LOG(FATAL) << "default_generator() not implemented yet (Task 5)";
    static Generator dummy{Device(Device::DeviceType::kCPU, 0)};
    return dummy;
}

void manual_seed(uint64_t /*seed*/) {
    LOG(FATAL) << "manual_seed() not implemented yet (Task 5)";
}

void manual_seed_cuda(uint64_t /*seed*/) {
    LOG(FATAL) << "manual_seed_cuda() not implemented yet (Task 5)";
}

}  // namespace infini_train
```

> 把 `#include "infini_train/include/generator.h"` 放在末尾段是 OK 的，但更整洁的是把它提到文件顶部 include 区。**实际写代码时把 include 放到文件顶部**，这里只是为了示意位置。`<mutex>` 已通过 `generator_impl.h` 间接获得。

- [ ] **Step 4: 跑测试**

```bash
cmake --build build -j
ctest --test-dir build -V -R '^test_generator_(cpu|cuda)$'
```

预期：3 条 `GeneratorBasicTest`（CPU 实例）全部通过。`USE_CUDA=OFF` 时不会有 `CUDA/` 实例。

- [ ] **Step 5: Commit**

```bash
git add infini_train/include/generator.h infini_train/src/core/generator/generator_impl.cc \
        tests/generator/test_generator_basic.cc
git commit -m "feat(generator): add Generator PImpl handle bound to Registry::Create()"
```

---

## Task 5: 默认池 `Default()` + `manual_seed()` + `default_generator()`

**Files:**
- Modify: `infini_train/src/core/generator/generator_impl.cc`
- Create: `tests/generator/test_default.cc`
- Create: `tests/generator/test_seed.cc`
- Create: `tests/generator/test_initial_seed.cc`

**目的：** 落地懒初始化默认池、`manual_seed`/`manual_seed_cuda` 入口与 `default_generator()` 句柄缓存（spec §4.6 锁域设计）。

- [ ] **Step 1: 写测试 `tests/generator/test_default.cc`**

```cpp
#include <thread>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/device.h"
#include "infini_train/include/generator.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class GeneratorDefaultTest : public infini_train::test::InfiniTrainTest {};

TEST_P(GeneratorDefaultTest, DefaultGeneratorIsStable) {
    auto &a = default_generator(GetDevice());
    auto &b = default_generator(GetDevice());
    EXPECT_EQ(&a, &b);
    EXPECT_EQ(a.impl().get(), b.impl().get());
}

TEST_P(GeneratorDefaultTest, ManualSeedTouchesDefault) {
    manual_seed(31415);
    auto &gen = default_generator(GetDevice());
    EXPECT_EQ(gen.InitialSeed(), 31415u);
}

TEST_P(GeneratorDefaultTest, ManualSeedBeforeFirstAccessRemembersSeed) {
    // 后访问的设备也应取到 last_user_seed_。CPU/CUDA 都成立。
    manual_seed(271828);
    auto &gen = default_generator(GetDevice());
    EXPECT_EQ(gen.InitialSeed(), 271828u);
}

INFINI_TRAIN_REGISTER_TEST(GeneratorDefaultTest);
```

> **测试隔离注意**：`default_generator()` 是进程级单例，跨 `TEST_P` 状态会泄漏。以上用例都先 `manual_seed(...)` 重置默认池后再断言，故无依赖。后续 `test_dispatch.cc` 与 `test_ops_*.cc` 也遵守"先 seed 再断言"的纪律。

- [ ] **Step 2: 写测试 `tests/generator/test_seed.cc`**

```cpp
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/device.h"
#include "infini_train/include/generator.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class GeneratorSeedTest : public infini_train::test::InfiniTrainTest {};

namespace {
// 直接驱动 CPUGeneratorImpl 的 engine 不暴露给用户层；这里改用 GetState 比较序列。
std::vector<uint8_t> SeedAndAdvance(Generator &gen, uint64_t seed, int n) {
    gen.ManualSeed(seed);
    auto state = gen.GetState();
    // 通过 SetState 等价于 reset；测试侧只对比 state 推进是否一致。
    return state;
}
}  // namespace

TEST_P(GeneratorSeedTest, ManualSeedReseedsState) {
    Generator gen(GetDevice());
    gen.ManualSeed(123);
    auto s1 = gen.GetState();
    gen.ManualSeed(123);
    auto s2 = gen.GetState();
    EXPECT_EQ(s1, s2);
}

TEST_P(GeneratorSeedTest, DifferentSeedsDifferState) {
    Generator a(GetDevice()), b(GetDevice());
    a.ManualSeed(1);
    b.ManualSeed(2);
    EXPECT_NE(a.GetState(), b.GetState());
}

INFINI_TRAIN_REGISTER_TEST(GeneratorSeedTest);
```

> 注：序列推进的"trap"测试（连续两次取 N 应不同）需要算子（uniform/normal kernel）才方便观察。Task 8 的 `test_ops_uniform.cc` 会覆盖这一点。

- [ ] **Step 3: 写测试 `tests/generator/test_initial_seed.cc`**

```cpp
#include "gtest/gtest.h"

#include "infini_train/include/device.h"
#include "infini_train/include/generator.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class GeneratorInitialSeedTest : public infini_train::test::InfiniTrainTest {};

TEST_P(GeneratorInitialSeedTest, SeedDoesNotChangeInitialSeed) {
    Generator gen(GetDevice());
    gen.ManualSeed(1);
    EXPECT_EQ(gen.InitialSeed(), 1u);
    const uint64_t new_seed = gen.Seed();
    EXPECT_NE(new_seed, 1u);   // 新种子来自 random_device，碰撞概率 ~0
    EXPECT_EQ(gen.InitialSeed(), 1u);
}

INFINI_TRAIN_REGISTER_TEST(GeneratorInitialSeedTest);
```

- [ ] **Step 4: 在 `infini_train/src/core/generator/generator_impl.cc` 实现 `Default()`、`ResetAllSeeds()`、`ResetCudaSeeds()`、`default_generator()`、`manual_seed()`、`manual_seed_cuda()`**

替换 Task 2/4 留下的 `LOG(FATAL)` 占位：

```cpp
std::shared_ptr<GeneratorImpl> GeneratorImplRegistry::Default(Device device) {
    if (device.IsCPU()) {
        std::call_once(default_cpu_once_, [&] {
            auto it = factories_.find(Device::DeviceType::kCPU);
            CHECK(it != factories_.end()) << "No CPU GeneratorImpl factory registered";
            default_cpu_ = it->second(0);
            default_cpu_->SetCurrentSeed(LastUserSeedOrRandom());
            cpu_initialized_.store(true, std::memory_order_release);
        });
        return default_cpu_;
    }
    const int8_t idx = device.index();
    CHECK(0 <= idx && idx < kMaxGpus) << "CUDA device index out of range: " << static_cast<int>(idx);
    std::call_once(default_cuda_once_[idx], [&] {
        auto it = factories_.find(Device::DeviceType::kCUDA);
        if (it == factories_.end()) {
            throw std::runtime_error("No CUDA GeneratorImpl factory registered "
                                     "(build with USE_CUDA=ON).");
        }
        default_cuda_[idx] = it->second(idx);
        default_cuda_[idx]->SetCurrentSeed(LastUserSeedOrRandom());
        cuda_initialized_[idx].store(true, std::memory_order_release);
    });
    return default_cuda_[idx];
}

void GeneratorImplRegistry::ResetAllSeeds(uint64_t seed) {
    {
        std::lock_guard<std::mutex> lk(registry_mutex_);
        last_user_seed_ = seed;
    }
    auto cpu_impl = Default(Device(Device::DeviceType::kCPU, 0));
    {
        std::lock_guard<std::mutex> lk(cpu_impl->mutex());
        cpu_impl->SetCurrentSeed(seed);
    }
    for (int i = 0; i < kMaxGpus; ++i) {
        if (!cuda_initialized_[i].load(std::memory_order_acquire)) {
            continue;
        }
        auto &impl = default_cuda_[i];
        std::lock_guard<std::mutex> lk(impl->mutex());
        impl->SetCurrentSeed(seed);
    }
}

void GeneratorImplRegistry::ResetCudaSeeds(uint64_t seed) {
    for (int i = 0; i < kMaxGpus; ++i) {
        if (!cuda_initialized_[i].load(std::memory_order_acquire)) {
            continue;
        }
        auto &impl = default_cuda_[i];
        std::lock_guard<std::mutex> lk(impl->mutex());
        impl->SetCurrentSeed(seed);
    }
}
```

并在 `Generator` 段替换 `default_generator()` / `manual_seed()` / `manual_seed_cuda()`：

```cpp
namespace {

// 缓存 default_generator() 返回的句柄；与 Registry::Default(...) 的 shared_ptr 同步。
struct DefaultHandles {
    Generator cpu{Device(Device::DeviceType::kCPU, 0)};   // ctor 会 Create 一个新的——下面 once_flag 内重绑
    std::array<Generator, core::kMaxGpus> cuda{};
    std::once_flag cpu_once;
    std::array<std::once_flag, core::kMaxGpus> cuda_once;
};

DefaultHandles &handles() {
    static DefaultHandles h;
    return h;
}

}  // namespace

Generator &default_generator(Device device) {
    auto &h = handles();
    auto impl = core::GeneratorImplRegistry::Instance().Default(device);
    if (device.IsCPU()) {
        std::call_once(h.cpu_once, [&] { h.cpu = Generator::FromImpl(impl); });
        DCHECK_EQ(h.cpu.impl().get(), impl.get());
        return h.cpu;
    }
    const int8_t idx = device.index();
    CHECK(0 <= idx && idx < core::kMaxGpus);
    std::call_once(h.cuda_once[idx], [&] { h.cuda[idx] = Generator::FromImpl(impl); });
    DCHECK_EQ(h.cuda[idx].impl().get(), impl.get());
    return h.cuda[idx];
}

void manual_seed(uint64_t seed) { core::GeneratorImplRegistry::Instance().ResetAllSeeds(seed); }

void manual_seed_cuda(uint64_t seed) { core::GeneratorImplRegistry::Instance().ResetCudaSeeds(seed); }
```

> **关于 `DefaultHandles::cpu` 的初始构造**：默认值 `Generator{Device(...)}` 会调用 `Registry::Create()`——这会构造**一个独立 impl**，与默认池不一致。`std::call_once` 内 `h.cpu = Generator::FromImpl(impl)` 会立刻覆盖；但如果 CUDA 未注册时 `Registry::Create(Device(kCUDA, ...))` 会抛——目前 `DefaultHandles::cuda{}` 用的是默认构造（即 `Generator{Device(Device::DeviceType::kCPU, 0)}`，因为 `Generator()` 默认构造已是 CPU），所以 array 元素初始化时不会触发 CUDA Create。**核查**：`Generator` 默认构造是 `Device(Device::DeviceType::kCPU, 0)`（generator.h Step 2），所以 `std::array<Generator, 8> cuda{}` 会用默认值聚合初始化为 8 个 CPU 句柄——OK，不抛。一旦 `default_generator(Device(kCUDA, idx))` 被调用，`call_once` 内将其覆盖为 CUDA 句柄。

- [ ] **Step 5: 跑所有 generator 测试**

```bash
cmake --build build -j
ctest --test-dir build -V -R '^test_generator_(cpu|cuda)$'
```

预期：基础 + default + seed + initial_seed 测试全部通过。

- [ ] **Step 6: Commit**

```bash
git add infini_train/src/core/generator/generator_impl.cc \
        tests/generator/test_default.cc tests/generator/test_seed.cc tests/generator/test_initial_seed.cc
git commit -m "feat(generator): implement default pool, manual_seed, default_generator()"
```

---

## Task 6: CPU `GetState`/`SetState` 序列化 + 格式校验

**Files:**
- Modify: `infini_train/src/core/generator/cpu/cpu_generator_impl.cc`
- Create: `tests/generator/test_state.cc`
- Create: `tests/generator/cpu_only/test_state_validation.cc`

**目的：** 实现统一头部 `magic|version|payload_size|payload` 的状态序列化；CPU payload 为 `mt19937_64` 文本流 + `seed_` + `initial_seed_`。

- [ ] **Step 1: 写 `tests/generator/test_state.cc`（roundtrip 行为）**

```cpp
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/device.h"
#include "infini_train/include/generator.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class GeneratorStateTest : public infini_train::test::InfiniTrainTest {};

TEST_P(GeneratorStateTest, GetSetStateRoundtrip) {
    Generator gen(GetDevice());
    gen.ManualSeed(2026);

    const auto baseline = gen.GetState();
    // 推进引擎：再 ManualSeed 一次，验证 SetState 能恢复。
    gen.ManualSeed(99);
    EXPECT_NE(gen.GetState(), baseline);
    gen.SetState(baseline);
    EXPECT_EQ(gen.GetState(), baseline);
    EXPECT_EQ(gen.InitialSeed(), 2026u);
}

INFINI_TRAIN_REGISTER_TEST(GeneratorStateTest);
```

- [ ] **Step 2: 写 `tests/generator/cpu_only/test_state_validation.cc`（CPU 直驱 + 格式校验）**

```cpp
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/core/generator/cpu_generator_impl.h"

using infini_train::core::CPUGeneratorImpl;

namespace {

constexpr uint32_t kCpuMagic = 0x47555043;   // 'CPUG'
constexpr uint32_t kCpuVersion = 1;

std::vector<uint8_t> MakeHeader(uint32_t magic, uint32_t version, uint64_t payload_size) {
    std::vector<uint8_t> buf(16);
    std::memcpy(buf.data() + 0, &magic, 4);
    std::memcpy(buf.data() + 4, &version, 4);
    std::memcpy(buf.data() + 8, &payload_size, 8);
    return buf;
}

}  // namespace

TEST(CpuStateValidation, TruncatedHeader) {
    CPUGeneratorImpl impl(0);
    impl.SetCurrentSeed(1);
    std::vector<uint8_t> too_short(15, 0);
    EXPECT_THROW(impl.SetState(too_short), std::runtime_error);
}

TEST(CpuStateValidation, BadMagic) {
    CPUGeneratorImpl impl(0);
    impl.SetCurrentSeed(1);
    auto buf = MakeHeader(0xDEADBEEF, kCpuVersion, 0);
    EXPECT_THROW(impl.SetState(buf), std::runtime_error);
}

TEST(CpuStateValidation, BadVersion) {
    CPUGeneratorImpl impl(0);
    impl.SetCurrentSeed(1);
    auto buf = MakeHeader(kCpuMagic, kCpuVersion + 1, 0);
    EXPECT_THROW(impl.SetState(buf), std::runtime_error);
}

TEST(CpuStateValidation, PayloadSizeMismatch) {
    CPUGeneratorImpl impl(0);
    impl.SetCurrentSeed(1);
    // 声明 payload_size = 100 但只附 4 字节
    auto buf = MakeHeader(kCpuMagic, kCpuVersion, 100);
    buf.insert(buf.end(), {0xAA, 0xBB, 0xCC, 0xDD});
    EXPECT_THROW(impl.SetState(buf), std::runtime_error);
}
```

- [ ] **Step 3: 在 `cpu_generator_impl.cc` 实现完整的 `GetState`/`SetState`（替换 LOG(FATAL)）**

```cpp
#include <cstring>
#include <sstream>
#include <stdexcept>

// （上方已经 #include 过 vector / cstdint / glog 等）

namespace {

inline void WriteBytes(std::vector<uint8_t> &dst, const void *src, size_t n) {
    const auto *p = static_cast<const uint8_t *>(src);
    dst.insert(dst.end(), p, p + n);
}

inline size_t ReadBytes(const std::vector<uint8_t> &src, size_t offset, void *dst, size_t n) {
    if (offset + n > src.size()) {
        throw std::runtime_error("CPUGeneratorImpl::SetState: buffer underflow");
    }
    std::memcpy(dst, src.data() + offset, n);
    return offset + n;
}

}  // namespace

std::vector<uint8_t> CPUGeneratorImpl::GetState() const {
    std::ostringstream oss;
    oss << engine_;
    const std::string engine_blob = oss.str();
    const uint64_t engine_size = engine_blob.size();

    std::vector<uint8_t> payload;
    payload.reserve(8 + engine_size + 8 + 8);
    WriteBytes(payload, &engine_size, 8);
    WriteBytes(payload, engine_blob.data(), engine_size);
    WriteBytes(payload, &seed_, 8);
    WriteBytes(payload, &initial_seed_, 8);

    const uint64_t payload_size = payload.size();
    std::vector<uint8_t> out;
    out.reserve(16 + payload_size);
    const uint32_t magic = kMagic;
    const uint32_t version = kVersion;
    WriteBytes(out, &magic, 4);
    WriteBytes(out, &version, 4);
    WriteBytes(out, &payload_size, 8);
    out.insert(out.end(), payload.begin(), payload.end());
    return out;
}

void CPUGeneratorImpl::SetState(const std::vector<uint8_t> &state) {
    if (state.size() < 16) {
        throw std::runtime_error("CPUGeneratorImpl::SetState: header truncated");
    }
    uint32_t magic = 0, version = 0;
    uint64_t payload_size = 0;
    size_t off = 0;
    off = ReadBytes(state, off, &magic, 4);
    off = ReadBytes(state, off, &version, 4);
    off = ReadBytes(state, off, &payload_size, 8);
    if (magic != kMagic) {
        throw std::runtime_error("CPUGeneratorImpl::SetState: magic mismatch");
    }
    if (version != kVersion) {
        throw std::runtime_error("CPUGeneratorImpl::SetState: version mismatch");
    }
    if (off + payload_size != state.size()) {
        throw std::runtime_error("CPUGeneratorImpl::SetState: payload size mismatch");
    }

    uint64_t engine_size = 0;
    off = ReadBytes(state, off, &engine_size, 8);
    if (off + engine_size > state.size()) {
        throw std::runtime_error("CPUGeneratorImpl::SetState: engine blob truncated");
    }
    std::string engine_blob(reinterpret_cast<const char *>(state.data() + off), engine_size);
    off += engine_size;
    std::istringstream iss(engine_blob);
    iss >> engine_;
    if (!iss) {
        throw std::runtime_error("CPUGeneratorImpl::SetState: engine deserialization failed");
    }

    uint64_t new_seed = 0, new_initial = 0;
    off = ReadBytes(state, off, &new_seed, 8);
    off = ReadBytes(state, off, &new_initial, 8);
    seed_ = new_seed;
    initial_seed_ = new_initial;
}
```

- [ ] **Step 4: 跑测试**

```bash
cmake --build build -j
ctest --test-dir build -V -R '^test_generator_(cpu|cuda|cpu_only)$'
```

预期：state roundtrip + 4 条 validation 全部通过。

- [ ] **Step 5: Commit**

```bash
git add infini_train/src/core/generator/cpu/cpu_generator_impl.cc \
        tests/generator/test_state.cc tests/generator/cpu_only/test_state_validation.cc
git commit -m "feat(generator): add CPU state serialization with magic/version header"
```

---

## Task 7: `ResolveGenerator` helper

**Files:**
- Modify: `infini_train/include/generator.h`（追加 `ResolveGenerator` 自由函数声明）
- Modify: `infini_train/src/core/generator/generator_impl.cc`（实现）

**目的：** 暴露算子层使用的桥接函数：`optional<Generator>` + `Device` → `shared_ptr<GeneratorImpl>`。无显式 generator 时回落到 `default_generator(device).impl()`。

- [ ] **Step 1: 在 `infini_train/include/generator.h` 末尾追加（`namespace infini_train` 内）**

```cpp
// 算子内部使用：显式 → 直接拿其 impl_；nullopt → 设备默认 Generator 的 impl_。
// 返回 shared_ptr 而非引用，避免悬空。
std::shared_ptr<core::GeneratorImpl> ResolveGenerator(const std::optional<Generator> &gen, Device device);
```

记得在 `generator.h` 顶部追加 `#include <optional>`。

- [ ] **Step 2: 在 `infini_train/src/core/generator/generator_impl.cc` 的 `infini_train` 命名空间内追加**

```cpp
std::shared_ptr<core::GeneratorImpl> ResolveGenerator(const std::optional<Generator> &gen, Device device) {
    if (gen.has_value()) {
        return gen->impl();
    }
    return default_generator(device).impl();
}
```

- [ ] **Step 3: 编译确认**

```bash
cmake --build build -j
```

无新测试（Task 8 通过算子使用 ResolveGenerator 间接覆盖）。

- [ ] **Step 4: Commit**

```bash
git add infini_train/include/generator.h infini_train/src/core/generator/generator_impl.cc
git commit -m "feat(generator): add ResolveGenerator helper for op layer"
```

---

## Task 8: CPU `UniformRandom` kernel + dispatcher

**Files:**
- Create: `infini_train/src/kernels/cpu/uniform_random.cc`
- Create: `tests/generator/test_ops_uniform.cc`
- Create: `tests/generator/test_dispatch.cc`

**目的：** 在 Dispatcher 注册 `(kCPU, "UniformRandom")` kernel，签名 `void(std::shared_ptr<Tensor>, float, float, core::GeneratorImpl*)`，单线程 + 锁 fill `[lo, hi)` FP32。

- [ ] **Step 1: 写测试 `tests/generator/test_ops_uniform.cc`**

```cpp
#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/generator.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class GeneratorUniformOpTest : public infini_train::test::InfiniTrainTest {};

namespace {
std::vector<float> ReadCpuFloats(const std::shared_ptr<Tensor> &t) {
    auto cpu = t->To(Device(Device::DeviceType::kCPU, 0));
    std::vector<float> out(cpu->NumElements());
    std::memcpy(out.data(), cpu->DataPtr(), out.size() * sizeof(float));
    return out;
}
}  // namespace

TEST_P(GeneratorUniformOpTest, SameSeedSameOutput) {
    if (GetParam() != Device::DeviceType::kCPU) {
        GTEST_SKIP() << "CUDA UniformRandom is Phase 2";
    }
    auto t1 = std::make_shared<Tensor>(std::vector<int64_t>{8}, DataType::kFLOAT32, GetDevice());
    auto t2 = std::make_shared<Tensor>(std::vector<int64_t>{8}, DataType::kFLOAT32, GetDevice());
    Generator g(GetDevice());

    g.ManualSeed(42);
    Dispatcher::Instance().Call<void>({GetDevice().type(), "UniformRandom"}, t1, 0.0f, 1.0f, g.impl().get());
    g.ManualSeed(42);
    Dispatcher::Instance().Call<void>({GetDevice().type(), "UniformRandom"}, t2, 0.0f, 1.0f, g.impl().get());

    EXPECT_EQ(ReadCpuFloats(t1), ReadCpuFloats(t2));
}

TEST_P(GeneratorUniformOpTest, ConsecutiveCallsAdvanceState) {
    if (GetParam() != Device::DeviceType::kCPU) {
        GTEST_SKIP() << "CUDA UniformRandom is Phase 2";
    }
    auto t1 = std::make_shared<Tensor>(std::vector<int64_t>{8}, DataType::kFLOAT32, GetDevice());
    auto t2 = std::make_shared<Tensor>(std::vector<int64_t>{8}, DataType::kFLOAT32, GetDevice());
    Generator g(GetDevice());
    g.ManualSeed(7);
    Dispatcher::Instance().Call<void>({GetDevice().type(), "UniformRandom"}, t1, 0.0f, 1.0f, g.impl().get());
    Dispatcher::Instance().Call<void>({GetDevice().type(), "UniformRandom"}, t2, 0.0f, 1.0f, g.impl().get());
    EXPECT_NE(ReadCpuFloats(t1), ReadCpuFloats(t2));
}

TEST_P(GeneratorUniformOpTest, OutputsWithinRange) {
    if (GetParam() != Device::DeviceType::kCPU) {
        GTEST_SKIP() << "CUDA UniformRandom is Phase 2";
    }
    auto t = std::make_shared<Tensor>(std::vector<int64_t>{1024}, DataType::kFLOAT32, GetDevice());
    Generator g(GetDevice());
    g.ManualSeed(2026);
    Dispatcher::Instance().Call<void>({GetDevice().type(), "UniformRandom"}, t, -2.0f, 5.0f, g.impl().get());
    for (float v : ReadCpuFloats(t)) {
        EXPECT_GE(v, -2.0f);
        EXPECT_LT(v, 5.0f);
    }
}

INFINI_TRAIN_REGISTER_TEST(GeneratorUniformOpTest);
```

- [ ] **Step 2: 写 `tests/generator/test_dispatch.cc`**

```cpp
#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/generator.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class GeneratorDispatchTest : public infini_train::test::InfiniTrainTest {};

TEST_P(GeneratorDispatchTest, NullGeneratorFallsBackToDefault) {
    if (GetParam() != Device::DeviceType::kCPU) {
        GTEST_SKIP() << "CUDA random kernels are Phase 2";
    }
    manual_seed(13);
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{8}, DataType::kFLOAT32, GetDevice());
    auto b = std::make_shared<Tensor>(std::vector<int64_t>{8}, DataType::kFLOAT32, GetDevice());

    auto resolved = ResolveGenerator(std::nullopt, GetDevice());
    Dispatcher::Instance().Call<void>({GetDevice().type(), "UniformRandom"}, a, 0.0f, 1.0f, resolved.get());

    manual_seed(13);
    auto resolved2 = ResolveGenerator(std::nullopt, GetDevice());
    Dispatcher::Instance().Call<void>({GetDevice().type(), "UniformRandom"}, b, 0.0f, 1.0f, resolved2.get());

    auto a_cpu = a->To(Device(Device::DeviceType::kCPU, 0));
    auto b_cpu = b->To(Device(Device::DeviceType::kCPU, 0));
    EXPECT_EQ(0, std::memcmp(a_cpu->DataPtr(), b_cpu->DataPtr(), 8 * sizeof(float)));
}

INFINI_TRAIN_REGISTER_TEST(GeneratorDispatchTest);
```

- [ ] **Step 3: 实现 `infini_train/src/kernels/cpu/uniform_random.cc`**

```cpp
#include <cstddef>
#include <memory>
#include <mutex>
#include <random>

#include "glog/logging.h"

#include "infini_train/include/core/generator/cpu_generator_impl.h"
#include "infini_train/include/core/generator/generator_impl.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {

void UniformRandom(std::shared_ptr<Tensor> tensor, float lo, float hi, core::GeneratorImpl *impl) {
    CHECK(impl != nullptr) << "UniformRandom: GeneratorImpl is null";
    CHECK_EQ(static_cast<int>(tensor->Dtype()), static_cast<int>(DataType::kFLOAT32))
        << "UniformRandom currently only supports FP32";

    auto *cpu_impl = static_cast<core::CPUGeneratorImpl *>(impl);
    std::lock_guard<std::mutex> lk(cpu_impl->mutex());
    auto &eng = cpu_impl->engine();
    std::uniform_real_distribution<float> dist(lo, hi);

    auto *data = static_cast<float *>(tensor->DataPtr());
    const size_t n = static_cast<size_t>(tensor->NumElements());
    for (size_t i = 0; i < n; ++i) {
        data[i] = dist(eng);
    }
}

}  // namespace infini_train::kernels::cpu

REGISTER_KERNEL(infini_train::Device::DeviceType::kCPU, UniformRandom,
                infini_train::kernels::cpu::UniformRandom)
```

- [ ] **Step 4: 跑测试**

```bash
cmake --build build -j
ctest --test-dir build -V -R '^test_generator_'
```

预期：3 条 UniformOp + DispatchTest 全部通过。

- [ ] **Step 5: Commit**

```bash
git add infini_train/src/kernels/cpu/uniform_random.cc \
        tests/generator/test_ops_uniform.cc tests/generator/test_dispatch.cc
git commit -m "feat(generator): add CPU UniformRandom kernel + dispatcher integration"
```

---

## Task 9: CPU `NormalRandom` kernel

**Files:**
- Create: `infini_train/src/kernels/cpu/normal_random.cc`
- Create: `tests/generator/test_ops_normal.cc`

**目的：** 镜像 Task 8，提供 `(kCPU, "NormalRandom")`：`void(Tensor, float mean, float std, core::GeneratorImpl*)`，FP32。

- [ ] **Step 1: 写测试 `tests/generator/test_ops_normal.cc`**

```cpp
#include <cmath>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/generator.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class GeneratorNormalOpTest : public infini_train::test::InfiniTrainTest {};

namespace {
std::vector<float> ReadCpuFloats(const std::shared_ptr<Tensor> &t) {
    auto cpu = t->To(Device(Device::DeviceType::kCPU, 0));
    std::vector<float> out(cpu->NumElements());
    std::memcpy(out.data(), cpu->DataPtr(), out.size() * sizeof(float));
    return out;
}
}  // namespace

TEST_P(GeneratorNormalOpTest, SameSeedSameOutput) {
    if (GetParam() != Device::DeviceType::kCPU) {
        GTEST_SKIP() << "CUDA NormalRandom is Phase 2";
    }
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{16}, DataType::kFLOAT32, GetDevice());
    auto b = std::make_shared<Tensor>(std::vector<int64_t>{16}, DataType::kFLOAT32, GetDevice());
    Generator g(GetDevice());
    g.ManualSeed(11);
    Dispatcher::Instance().Call<void>({GetDevice().type(), "NormalRandom"}, a, 0.0f, 1.0f, g.impl().get());
    g.ManualSeed(11);
    Dispatcher::Instance().Call<void>({GetDevice().type(), "NormalRandom"}, b, 0.0f, 1.0f, g.impl().get());
    EXPECT_EQ(ReadCpuFloats(a), ReadCpuFloats(b));
}

TEST_P(GeneratorNormalOpTest, MeanCloseToTarget) {
    if (GetParam() != Device::DeviceType::kCPU) {
        GTEST_SKIP() << "CUDA NormalRandom is Phase 2";
    }
    constexpr int kN = 16384;
    auto t = std::make_shared<Tensor>(std::vector<int64_t>{kN}, DataType::kFLOAT32, GetDevice());
    Generator g(GetDevice());
    g.ManualSeed(31);
    Dispatcher::Instance().Call<void>({GetDevice().type(), "NormalRandom"}, t, 5.0f, 2.0f, g.impl().get());
    auto v = ReadCpuFloats(t);
    double sum = 0;
    for (float x : v) {
        sum += x;
    }
    const double mean = sum / kN;
    EXPECT_NEAR(mean, 5.0, 0.1);
}

INFINI_TRAIN_REGISTER_TEST(GeneratorNormalOpTest);
```

- [ ] **Step 2: 实现 `infini_train/src/kernels/cpu/normal_random.cc`**

```cpp
#include <cstddef>
#include <memory>
#include <mutex>
#include <random>

#include "glog/logging.h"

#include "infini_train/include/core/generator/cpu_generator_impl.h"
#include "infini_train/include/core/generator/generator_impl.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {

void NormalRandom(std::shared_ptr<Tensor> tensor, float mean, float std, core::GeneratorImpl *impl) {
    CHECK(impl != nullptr) << "NormalRandom: GeneratorImpl is null";
    CHECK_EQ(static_cast<int>(tensor->Dtype()), static_cast<int>(DataType::kFLOAT32))
        << "NormalRandom currently only supports FP32";

    auto *cpu_impl = static_cast<core::CPUGeneratorImpl *>(impl);
    std::lock_guard<std::mutex> lk(cpu_impl->mutex());
    auto &eng = cpu_impl->engine();
    std::normal_distribution<float> dist(mean, std);

    auto *data = static_cast<float *>(tensor->DataPtr());
    const size_t n = static_cast<size_t>(tensor->NumElements());
    for (size_t i = 0; i < n; ++i) {
        data[i] = dist(eng);
    }
}

}  // namespace infini_train::kernels::cpu

REGISTER_KERNEL(infini_train::Device::DeviceType::kCPU, NormalRandom,
                infini_train::kernels::cpu::NormalRandom)
```

- [ ] **Step 3: 跑测试**

```bash
cmake --build build -j
ctest --test-dir build -V -R '^test_generator_'
```

预期：NormalOp 两条用例通过。

- [ ] **Step 4: Commit**

```bash
git add infini_train/src/kernels/cpu/normal_random.cc tests/generator/test_ops_normal.cc
git commit -m "feat(generator): add CPU NormalRandom kernel"
```

---

## Task 10: 改造 `nn::init` + `Tensor::Uniform` 走 dispatcher

**Files:**
- Modify: `infini_train/include/nn/init.h`
- Modify: `infini_train/src/nn/init.cc`
- Modify: `infini_train/include/tensor.h`
- Modify: `infini_train/src/tensor.cc`
- Create: `tests/generator/test_ops_kaiming.cc`

**目的：** 把 `Normal/Uniform/KaimingUniform` 三个 init 函数与 `Tensor::Uniform` 的 `std::optional<std::mt19937>` 全部换成 `std::optional<Generator>`，并通过 dispatcher 调 CPU kernel。删除 `init.cc` 内部 OMP 双 bug 路径与 `kRandomSeed`/文件作用域 `gen`。下游 11 处调用因默认参数兼容**无需改动**。

- [ ] **Step 1: 写测试 `tests/generator/test_ops_kaiming.cc`**

```cpp
#include <cmath>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/device.h"
#include "infini_train/include/generator.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class GeneratorKaimingTest : public infini_train::test::InfiniTrainTest {};

namespace {
std::vector<float> ReadCpuFloats(const std::shared_ptr<Tensor> &t) {
    auto cpu = t->To(Device(Device::DeviceType::kCPU, 0));
    std::vector<float> out(cpu->NumElements());
    std::memcpy(out.data(), cpu->DataPtr(), out.size() * sizeof(float));
    return out;
}
}  // namespace

TEST_P(GeneratorKaimingTest, SameSeedSameWeights) {
    if (GetParam() != Device::DeviceType::kCPU) {
        GTEST_SKIP() << "CUDA random kernels are Phase 2";
    }
    auto t1 = std::make_shared<Tensor>(std::vector<int64_t>{16, 16}, DataType::kFLOAT32, GetDevice());
    auto t2 = std::make_shared<Tensor>(std::vector<int64_t>{16, 16}, DataType::kFLOAT32, GetDevice());

    Generator g1(GetDevice());
    Generator g2(GetDevice());
    g1.ManualSeed(2026);
    g2.ManualSeed(2026);
    nn::init::KaimingUniform(t1, /*a=*/0.0f, nn::init::KaimingMode::kFanIn,
                             nn::init::NonLinearityType::kLeakyReLU, g1);
    nn::init::KaimingUniform(t2, /*a=*/0.0f, nn::init::KaimingMode::kFanIn,
                             nn::init::NonLinearityType::kLeakyReLU, g2);
    EXPECT_EQ(ReadCpuFloats(t1), ReadCpuFloats(t2));
}

TEST_P(GeneratorKaimingTest, FanInBoundsRespected) {
    if (GetParam() != Device::DeviceType::kCPU) {
        GTEST_SKIP() << "CUDA random kernels are Phase 2";
    }
    // weight: [out=8, in=32] → fan_in = 32, gain(LeakyReLU,a=0) = sqrt(2/(1+0.01^2)) ≈ sqrt(2)
    // std = gain / sqrt(fan_in); bound = sqrt(3) * std
    const int64_t out_dim = 8, in_dim = 32;
    auto t = std::make_shared<Tensor>(std::vector<int64_t>{out_dim, in_dim}, DataType::kFLOAT32, GetDevice());
    Generator g(GetDevice());
    g.ManualSeed(123);
    nn::init::KaimingUniform(t, /*a=*/0.0f, nn::init::KaimingMode::kFanIn,
                             nn::init::NonLinearityType::kLeakyReLU, g);
    const float gain = std::sqrt(2.0f / (1.0f + 0.01f * 0.01f));
    const float bound = std::sqrt(3.0f) * gain / std::sqrt(static_cast<float>(in_dim));
    for (float v : ReadCpuFloats(t)) {
        EXPECT_GE(v, -bound);
        EXPECT_LT(v, bound);
    }
}

INFINI_TRAIN_REGISTER_TEST(GeneratorKaimingTest);
```

- [ ] **Step 2: 修改 `infini_train/include/nn/init.h`**

替换三个签名（删 `<random>`，改用 Generator）：

```cpp
#pragma once

#include <memory>
#include <optional>
#include <utility>

#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/generator.h"

namespace infini_train {
class Tensor;
class Device;
}  // namespace infini_train

namespace infini_train::nn::init {
std::shared_ptr<Tensor> Normal(const std::shared_ptr<Tensor> &tensor, float mean = 0.0, float std = 1.0,
                               std::optional<Generator> generator = std::nullopt);

std::pair<int64_t, int64_t> CalculateFanInAndFanOut(const std::shared_ptr<Tensor> &tensor);

enum class KaimingMode : int8_t {
    kFanIn,
    kFanOut,
};

enum class NonLinearityType : int8_t {
    kLinear,           kConv1D,           kConv2D,
    kConv3D,           kConvTransposed1d, kConvTransposed2d,
    kConvTransposed3d, kSigmoid,          kTanh,
    kReLU,             kLeakyReLU,        kSELU,
};

std::shared_ptr<Tensor> KaimingUniform(const std::shared_ptr<Tensor> &tensor, float a = 0.0f,
                                       KaimingMode mode = KaimingMode::kFanIn,
                                       NonLinearityType non_linearity = NonLinearityType::kLeakyReLU,
                                       std::optional<Generator> generator = std::nullopt);

std::shared_ptr<Tensor> Uniform(const std::shared_ptr<Tensor> &tensor, float a = 0.0f, float b = 1.0f,
                                std::optional<Generator> generator = std::nullopt);

std::shared_ptr<Tensor> Ones(const std::shared_ptr<Tensor> &tensor);
std::shared_ptr<Tensor> Zeros(const std::shared_ptr<Tensor> &tensor);
std::shared_ptr<Tensor> Arange(int64_t start, int64_t end, DataType dtype, Device device = Device());
}  // namespace infini_train::nn::init
```

- [ ] **Step 3: 修改 `infini_train/include/tensor.h`**

定位到 `Uniform` 声明（line 150 附近），将 `std::optional<std::mt19937>` 改为 `std::optional<Generator>`；并在文件顶部把 `#include <random>` 删除（如果只在该签名中使用），加 `#include "infini_train/include/generator.h"`。

```cpp
// 顶部 includes（替换 <random>）
#include "infini_train/include/generator.h"

// ... 类内：
std::shared_ptr<Tensor> Uniform(float from = 0.0f, float to = 1.0f,
                                std::optional<Generator> generator = std::nullopt);
```

> 编译时若 `tensor.h` 其它地方还在用 `<random>` 中的类型（grep 一次确认），保留 `<random>`；当前实测仅 `Uniform` 使用，可删。

- [ ] **Step 4: 修改 `infini_train/src/tensor.cc:471` 的实现签名**

```cpp
std::shared_ptr<Tensor> Tensor::Uniform(float from, float to, std::optional<Generator> generator) {
    return nn::init::Uniform(shared_from_this(), from, to, generator);
}
```

- [ ] **Step 5: 重写 `infini_train/src/nn/init.cc`**

完整替换文件内容（删除 `kRandomSeed`、`gen`、双 OMP bug 路径；改走 dispatcher）：

```cpp
#include "infini_train/include/nn/init.h"

#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <unordered_set>

#include "glog/logging.h"

#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/generator.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::init {

std::shared_ptr<Tensor> Normal(const std::shared_ptr<Tensor> &tensor, float mean, float std,
                               std::optional<Generator> generator) {
    auto device = tensor->GetDevice();
    core::DeviceGuard guard(device);
    auto impl = ResolveGenerator(generator, device);
    Dispatcher::Instance().Call<void>({device.type(), "NormalRandom"}, tensor, mean, std, impl.get());
    return tensor;
}

std::pair<int64_t, int64_t> CalculateFanInAndFanOut(const std::shared_ptr<Tensor> &tensor) {
    if (tensor->Dims().size() < 2) {
        LOG(FATAL) << "Fan in and fan out can not be computed for tensor with less than 2 dimensions";
    }
    const auto num_input_fmaps = tensor->Dims()[1];
    const auto num_output_fmaps = tensor->Dims()[0];
    int64_t receptive_field_size = 1;
    if (tensor->Dims().size() > 2) {
        receptive_field_size
            *= std::accumulate(tensor->Dims().begin() + 2, tensor->Dims().end(), int64_t{1}, std::multiplies<int64_t>());
    }
    const auto fan_in = num_input_fmaps * receptive_field_size;
    const auto fan_out = num_output_fmaps * receptive_field_size;
    return {fan_in, fan_out};
}

namespace {

int64_t CalculateCorrectFan(const std::shared_ptr<Tensor> &tensor, KaimingMode mode) {
    const auto [fan_in, fan_out] = CalculateFanInAndFanOut(tensor);
    return mode == KaimingMode::kFanIn ? fan_in : fan_out;
}

float CalculateGain(NonLinearityType nonlinearity, std::optional<float> param = std::nullopt) {
    static std::unordered_set<NonLinearityType> kLinearFns = {
        NonLinearityType::kLinear,           NonLinearityType::kConv1D,           NonLinearityType::kConv2D,
        NonLinearityType::kConv3D,           NonLinearityType::kConvTransposed1d, NonLinearityType::kConvTransposed2d,
        NonLinearityType::kConvTransposed3d,
    };
    if (kLinearFns.contains(nonlinearity) || nonlinearity == NonLinearityType::kSigmoid) {
        return 1.0f;
    } else if (nonlinearity == NonLinearityType::kTanh) {
        return 5.0f / 3;
    } else if (nonlinearity == NonLinearityType::kReLU) {
        return std::sqrt(2.0f);
    } else if (nonlinearity == NonLinearityType::kLeakyReLU) {
        const float negative_slope = param ? *param : 0.01f;
        return std::sqrt(2.0f / (1.0f + negative_slope * negative_slope));
    } else if (nonlinearity == NonLinearityType::kSELU) {
        return 3.0f / 4;
    } else {
        LOG(FATAL) << "Unsupported non-linearity type: " << static_cast<int>(nonlinearity);
    }
    return -1.0f;
}

}  // namespace

std::shared_ptr<Tensor> KaimingUniform(const std::shared_ptr<Tensor> &tensor, float a, KaimingMode mode,
                                       NonLinearityType nonlinearity, std::optional<Generator> generator) {
    for (const auto dim : tensor->Dims()) {
        if (dim == 0) {
            LOG(WARNING) << "Initializing zero-element tensors is a no-op";
            return tensor;
        }
    }
    const auto fan = CalculateCorrectFan(tensor, mode);
    const auto gain = CalculateGain(nonlinearity, a);
    const float std = gain / std::sqrt(static_cast<float>(fan));
    const float bound = std::sqrt(3.0f) * std;
    return tensor->Uniform(-bound, bound, generator);
}

std::shared_ptr<Tensor> Uniform(const std::shared_ptr<Tensor> &tensor, float a, float b,
                                std::optional<Generator> generator) {
    auto device = tensor->GetDevice();
    core::DeviceGuard guard(device);
    auto impl = ResolveGenerator(generator, device);
    Dispatcher::Instance().Call<void>({device.type(), "UniformRandom"}, tensor, a, b, impl.get());
    return tensor;
}

std::shared_ptr<Tensor> Ones(const std::shared_ptr<Tensor> &tensor) {
    CHECK_EQ(static_cast<int>(tensor->Dtype()), static_cast<int>(DataType::kFLOAT32));
    const int64_t num_elements = tensor->NumElements();
    std::vector<float> buffer(num_elements, 1.0f);

    auto device = tensor->GetDevice();
    core::DeviceGuard guard(device);
    auto impl = core::GetDeviceGuardImpl(device.type());

    impl->MemcpyAsync(tensor->DataPtr(), buffer.data(), num_elements * sizeof(float),
                      device.IsCPU() ? core::MemcpyKind::kD2D : core::MemcpyKind::kH2D, impl->GetStream(device));
    return tensor;
}

std::shared_ptr<Tensor> Zeros(const std::shared_ptr<Tensor> &tensor) {
    CHECK_EQ(static_cast<int>(tensor->Dtype()), static_cast<int>(DataType::kFLOAT32));
    const int64_t num_elements = tensor->NumElements();
    std::vector<float> buffer(num_elements, 0.0f);

    auto device = tensor->GetDevice();
    core::DeviceGuard guard(device);
    auto impl = core::GetDeviceGuardImpl(device.type());

    impl->MemcpyAsync(tensor->DataPtr(), buffer.data(), num_elements * sizeof(float),
                      device.IsCPU() ? core::MemcpyKind::kD2D : core::MemcpyKind::kH2D, impl->GetStream(device));
    return tensor;
}

#define ARANGE_CASE(DATA_TYPE, TYPE)                                                                                   \
    case DATA_TYPE: {                                                                                                  \
        std::vector<TYPE> buffer(num_elements);                                                                        \
        std::iota(buffer.begin(), buffer.end(), static_cast<TYPE>(start));                                             \
        impl->MemcpyAsync(tensor->DataPtr(), buffer.data(), num_elements * sizeof(TYPE), kind, stream);                \
        break;                                                                                                         \
    }

std::shared_ptr<Tensor> Arange(int64_t start, int64_t end, DataType dtype, Device device) {
    const int64_t num_elements = end - start;
    auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{num_elements}, dtype, device);

    core::DeviceGuard guard(device);
    auto *impl = core::GetDeviceGuardImpl(device.type());

    const core::MemcpyKind kind = device.IsCPU() ? core::MemcpyKind::kD2D : core::MemcpyKind::kH2D;
    core::Stream *stream = impl->GetStream(device);

    switch (dtype) {
        ARANGE_CASE(DataType::kUINT8, uint8_t)
        ARANGE_CASE(DataType::kINT8, int8_t)
        ARANGE_CASE(DataType::kUINT16, uint16_t)
        ARANGE_CASE(DataType::kINT16, int16_t)
        ARANGE_CASE(DataType::kUINT32, uint32_t)
        ARANGE_CASE(DataType::kINT32, int32_t)
        ARANGE_CASE(DataType::kUINT64, uint64_t)
        ARANGE_CASE(DataType::kINT64, int64_t)
        ARANGE_CASE(DataType::kBFLOAT16, BF16)
        ARANGE_CASE(DataType::kFLOAT16, FP16)
        ARANGE_CASE(DataType::kFLOAT32, float)
        ARANGE_CASE(DataType::kFLOAT64, double)

    default:
        LOG(FATAL) << "Unsupported data type: " << static_cast<int>(dtype);
        break;
    }

    return tensor;
}

#undef ARANGE_CASE

}  // namespace infini_train::nn::init
```

> **行为变化提醒**（spec §2.3）：
> - 引擎从 `mt19937` 切到 `mt19937_64`，同 seed 数值不再与旧版本一致。
> - 删除了"OMP 默认分支每次重构 mt19937"与"OMP 显式分支多线程共享 generator"两个 bug 分支，串行 + 锁取而代之，性能下降但语义正确。
> - **CUDA 路径在 Phase 1 还未注册** `UniformRandom`/`NormalRandom` kernel；任何在 CUDA tensor 上调用 `nn::init::Uniform/Normal/KaimingUniform` 都会触发 `Dispatcher::GetKernel` CHECK 失败。
>   - 这是 Phase 1 的**已知回退**——例子（gpt2/llama3）若在 CUDA 上初始化随机权重将无法运行；推荐的过渡方案是用 `USE_CUDA=OFF` 跑测试，或在 Phase 2 落地 CUDA kernel 之前**保留 ssh 通道**：先临时给 CUDA 路径回退到"CPU 生成 + H2D"。
>   - **决定**：本任务**不**给 CUDA 添加临时回退（避免重复实现），但需要在 PR description 与 commit message 中显式说明这一窗口期；如果赛题需要在 Phase 1 就能跑 CUDA examples，开 Phase 2 PR 接续。
> - **测试覆盖**：Phase 1 测试套件全部 `if (GetParam() != kCPU) GTEST_SKIP()` CUDA 实例，因此即使 `BUILD_TEST=ON USE_CUDA=ON`，`test_generator_cuda` 二进制也会成功（仅 SKIP 实例不算失败）。

- [ ] **Step 6: 跑全量套件**

```bash
cmake --build build -j
ctest --test-dir build -V
```

预期：
- generator 套件全绿；
- 其他已有套件（tensor/lora/optimizer/...）不退化。
- 若 lora 等套件依赖随机数值断言（grep 一下 `lora` 测试是否对 init 后权重做精确数值断言），需要重新生成 baseline——**前置 grep**：

```bash
grep -rn "Normal\|Uniform\|KaimingUniform" /Users/guozhihao/work/mlsys/InfiniTrain/tests/ | grep -v "test_generator"
```

如有此类硬编码 baseline，那一刻补一份"重新捕获 baseline"的子任务并以新 mt19937_64 序列写回（不在本计划自动展开——遇到再处理）。

- [ ] **Step 7: Commit**

```bash
git add infini_train/include/nn/init.h infini_train/src/nn/init.cc \
        infini_train/include/tensor.h infini_train/src/tensor.cc \
        tests/generator/test_ops_kaiming.cc
git commit -m "refactor(nn/init): route Uniform/Normal/KaimingUniform through Generator + dispatcher"
```

---

## Task 11: 删除 example 中的死代码

**Files:**
- Modify: `example/gpt2/checkpoint_loader.cc`
- Modify: `example/llama3/checkpoint_loader.cc`

**目的：** 删除 `kRandomSeed`、`static std::mt19937 gen`、相关 TODO、`<random>` include（全文件无引用）。

- [ ] **Step 1: 二次确认 `gen` 全文件未被引用**

```bash
grep -n "gen\b\|kRandomSeed" /Users/guozhihao/work/mlsys/InfiniTrain/example/gpt2/checkpoint_loader.cc \
                            /Users/guozhihao/work/mlsys/InfiniTrain/example/llama3/checkpoint_loader.cc
```

预期：仅命中 spec §1.1 列出的两处声明（gpt2:31-34，llama3:29-32），无其它引用。

- [ ] **Step 2: 编辑 `example/gpt2/checkpoint_loader.cc`**

删除：
- line 8 的 `#include <random>`（若该文件其它地方未用 `<random>`，grep 一次确认）；
- 第 30-35 行整个匿名命名空间块（`constexpr int kRandomSeed = 42;`、TODO 注释、`static std::mt19937 gen{kRandomSeed};`）。

注意：第二个匿名命名空间 `kHeaderMagic` 等保留。

- [ ] **Step 3: 同样编辑 `example/llama3/checkpoint_loader.cc`**

删除 line 8 `<random>` include（grep 确认）+ 第 28-33 行匿名命名空间块。

- [ ] **Step 4: 构建确认**

```bash
cmake --build build -j
```

预期：gpt2 / llama3 example targets 都能编译。

- [ ] **Step 5: Commit**

```bash
git add example/gpt2/checkpoint_loader.cc example/llama3/checkpoint_loader.cc
git commit -m "refactor(example): drop dead kRandomSeed and unused std::mt19937 in checkpoint loaders"
```

---

## Task 12: 全量验证

**目的：** 整体回归 + 确认 Phase 1 验收。

- [ ] **Step 1: 全量 ctest（Phase 1 配置）**

```bash
cd /Users/guozhihao/work/mlsys/InfiniTrain
cmake -S . -B build -DBUILD_TEST=ON -DUSE_CUDA=OFF -DUSE_NCCL=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure
```

预期：所有 `test_*_cpu` 套件成功。

- [ ] **Step 2: 仅跑 generator 套件并附详细输出**

```bash
ctest --test-dir build -V -R '^test_generator_'
```

把输出贴到 PR description（便于 reviewer 检查覆盖）。

- [ ] **Step 3: 检查 spec §1.1 列出的 mt19937 残留是否清零**

```bash
grep -rn "kRandomSeed\|std::mt19937\b" \
  /Users/guozhihao/work/mlsys/InfiniTrain/example/ \
  /Users/guozhihao/work/mlsys/InfiniTrain/infini_train/ \
  | grep -v "third_party\|build/"
```

预期：仅命中本任务**新增**的 `mt19937_64`（CPUGeneratorImpl）+ 测试文件里的本地变量。`std::mt19937 ` 不应在公开头中再现。

- [ ] **Step 4: 形成 PR（暂不 push，等用户决定）**

```bash
git log --oneline master..HEAD
```

确认 Phase 1 共有 ~10 条 commit，叙事连贯（基础设施 → CPU impl → 序列化 → 默认池 → kernel → 算子改造 → 死代码清理）。

---

## 自检对照（spec §2.1 必达目标）

| 必达 | 落地 task |
|------|-----------|
| 1. 统一 Generator 抽象 | Task 2/4 |
| 2. CPU 后端 + CUDA 占位（Phase 2） | Task 3/6 |
| 3. 默认 Generator 池（CPU） | Task 5 |
| 4. 全局 manual_seed 入口 | Task 5 |
| 5. 现有算子改造 | Task 8/9/10 |
| 6. seed/state/格式校验/默认/显式/Dropout 行为测试 | Task 5/6/7/8/9/10（Dropout 留 Phase 3） |
| 7. 对齐报告 | Phase 4 |

## 已知 Phase 1 边界与回退

1. **CUDA 路径不可用**：`Generator(Device::kCUDA)` 抛 `runtime_error`；调用方依赖 `nn::init::Uniform` 在 CUDA tensor 上工作的 example 在 Phase 2 落地前不可用。
2. **`UniformRandom`/`NormalRandom` 仅 FP32**：BF16/FP16 需 dtype dispatch（spec §10 列为后续）。
3. **OMP 并行优化未做**：单线程 + 锁，性能下降；spec §2.2 已声明非目标。
4. **跨设备 RNG 一致性不保证**：spec §2.2、§6 已声明。
