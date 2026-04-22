# 测试体系设计

## 1. 架构

```
tests/
├── CMakeLists.txt              # 顶层：include 宏 + add_subdirectory
├── common/
│   ├── CMakeLists.txt          # header-only interface library
│   ├── test_macros.cmake       # CMake 宏：infini_train_add_test / infini_train_add_test_suite
│   └── test_utils.h            # C++ 基类、skip 宏、填充工具函数
├── tensor/                     # Tensor 创建 / 拷贝 / 销毁 / 算子
├── optimizer/                  # Optimizer 创建 / step
├── autograd/                   # 各 autograd op 的 forward / backward
├── hook/                       # Module hook + precision check
├── lora/                       # LoRA 相关
├── dtype/                      # Scalar / dtype dispatch + 编译期负面测试
└── transformer/                # Transformer 架构测试
```

### 核心设计：设备参数化

测试不区分 CPU / CUDA 平台。一个测试定义通过 GTest 参数化自动在所有可用设备上运行：

- `INFINI_TRAIN_REGISTER_TEST(TestName)` — 注册 CPU + CUDA 两个实例

无 GPU 时 CUDA 实例在注册阶段直接跳过（不会出现在测试列表里），并打印一条 `LOG(INFO)` 提示。

### 基类层次

| 基类 | 用途 | 提供的能力 |
|------|------|-----------|
| `InfiniTrainTest` | 通用参数化测试 | `GetDevice()`, `createTensor(shape, dtype, requires_grad)` |
| `AutogradTestBase` | Autograd 测试 | `createTensor(shape, value)` 自动 `requires_grad=true` + 顺序填充 |

### 跳过特定平台

在个别测试内部按需跳过：

```cpp
// 跳过 CPU 实例（用于硬编码加速器设备的测试，未来新平台仍会运行）
SKIP_CPU();

// 只在 CPU 实例运行（用于硬编码 CPU 设备的测试）
ONLY_CPU();

// 只在 CUDA 实例运行（用于硬编码 CUDA 设备的测试）
ONLY_CUDA();

// 需要 ≥n 个加速器设备
REQUIRE_MIN_DEVICES(n);
```

### CMake 宏

`test_macros.cmake` 提供两个宏减少 CMakeLists 样板：

- `infini_train_add_test(name SOURCES ... LABELS ...)` — 创建可执行文件、链接 GTest + 框架库、用 `gtest_discover_tests` 自动发现用例
- `infini_train_add_test_suite(name SOURCES ... LABELS ...)` — 按 label（cpu/cuda）拆分为多个 CTest target，通过 `TEST_FILTER` 路由到对应的参数化前缀（`CPU/*`, `CUDA/*`）

## 2. 构建与运行

```bash
# 构建（从 build 目录）
cmake -DBUILD_TEST=ON -DUSE_CUDA=ON ..
make -j$(nproc)

# 运行全部测试
ctest --output-on-failure

# 只运行 CPU 测试
ctest -L cpu --output-on-failure

# 只运行 CUDA 测试（分布式测试也在此标签下，不满足条件时自动 skip）
ctest -L cuda --output-on-failure

# 运行单个测试二进制（看完整 GTest 输出）
./test_tensor_cpu
./test_autograd_cuda

# GTest filter 过滤特定用例
./test_tensor_cpu --gtest_filter="CPU/TensorCreateTest.*"
```

无 GPU 机器上 `cmake -DBUILD_TEST=ON -DUSE_CUDA=OFF ..` 即可，CUDA 测试实例不会注册。

## 3. 新增测试

### 3.1 新增 GTest 参数化测试（推荐）

以新增 `tests/foo/` 为例，完整流程：

**Step 1: 创建目录和测试文件**

```bash
mkdir tests/foo
```

```cpp
// tests/foo/test_foo_basic.cc
#include <gtest/gtest.h>
#include "infini_train/include/tensor.h"
#include "test_utils.h"

using namespace infini_train;

class FooBasicTest : public infini_train::test::InfiniTrainTest {};

TEST_P(FooBasicTest, CreateTensor) {
    auto tensor = createTensor({2, 3});
    EXPECT_NE(tensor, nullptr);
}

TEST_P(FooBasicTest, CUDAOnlyFeature) {
    SKIP_CPU();
    // CUDA-specific logic ...
}

INFINI_TRAIN_REGISTER_TEST(FooBasicTest);
```

基类选择：

| 场景 | 基类 |
|------|------|
| 通用测试 | `InfiniTrainTest`（提供 `createTensor(shape, dtype, requires_grad)`） |
| 需要 autograd | `AutogradTestBase`（提供 `createTensor(shape, value)`，自动 `requires_grad=true` + 顺序填充） |

**Step 2: 写 CMakeLists.txt**

```cmake
# tests/foo/CMakeLists.txt
file(GLOB FOO_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cc)

infini_train_add_test_suite(test_foo
  SOURCES ${FOO_SOURCES}
)
```

`file(GLOB test_*.cc)` 会自动拾取目录下所有测试文件。以后新增 `test_foo_advanced.cc` 只需放入目录，无需改 CMakeLists.txt（重新 cmake 即可）。

**Step 3: 注册到顶层**

在 `tests/CMakeLists.txt` 中添加一行：

```cmake
add_subdirectory(foo)
```

**生成的 CTest target：** `test_foo_cpu`、`test_foo_cuda`，可通过 `ctest -L cpu` 等按标签筛选。

### 3.2 新增编译期负面测试

用于验证某段代码必须编译失败（如缺少 dtype 注册时 `static_assert` 应触发）。这类测试不走 GTest，而是用 CMake 的 `try_compile` 在 configure 阶段验证。

现有示例：`tests/dtype/test_dtype_dispatch_compile_fail.cc` — 验证未注册的 dtype 在 dispatch 时编译失败。

CMakeLists.txt 写法（参考 `tests/dtype/CMakeLists.txt`）：

```cmake
set(COMPILE_FAIL_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/test_xxx_compile_fail.cc)

try_compile(UNEXPECTEDLY_SUCCEEDED
  ${CMAKE_BINARY_DIR}/CMakeFiles/try_compile_xxx
  SOURCES ${COMPILE_FAIL_SOURCE}
  CMAKE_FLAGS
    "-DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}"
    "-DCMAKE_CXX_STANDARD_REQUIRED=ON"
    "-DCMAKE_CXX_EXTENSIONS=OFF"
    "-DCMAKE_CXX_FLAGS=-I${PROJECT_SOURCE_DIR}"
  OUTPUT_VARIABLE TRY_COMPILE_OUTPUT
)

if(UNEXPECTEDLY_SUCCEEDED)
  message(FATAL_ERROR
    "compile-fail test unexpectedly succeeded.\n"
    "Source: ${COMPILE_FAIL_SOURCE}\n"
    "Output:\n${TRY_COMPILE_OUTPUT}")
endif()

add_custom_target(test_xxx_compile_fail
  COMMAND ${CMAKE_COMMAND} -E echo "compile-fail check passed."
  VERBATIM
)
```

编译期负面测试的源文件用手写 `main()`，不依赖 GTest。它在 `cmake` 阶段就完成验证，不会生成可执行文件。

### 3.3 在已有目录新增测试文件

所有使用 `file(GLOB ...)` 的目录（autograd、tensor、optimizer、hook、lora）：

1. 把新文件放入对应目录，命名为 `test_*.cc`
2. 重新 `cmake ..`（glob 在 configure 时求值）
3. 完成

无需修改任何 CMakeLists.txt。

### 3.4 工具函数速查

`test_utils.h` 提供的常用工具：

| 函数 / 宏 | 用途 |
|-----------|------|
| `GetDevice()` | 返回当前参数化的 `Device`（基类方法） |
| `createTensor(shape, dtype, requires_grad)` | 在当前设备创建 tensor（`InfiniTrainTest` 基类方法） |
| `FillSequentialTensor(tensor, start)` | 填充递增值，自动处理 CUDA tensor（先填 CPU 再 copy） |
| `SKIP_CPU()` | 跳过 CPU 实例 |
| `ONLY_CPU()` | 只在 CPU 实例运行 |
| `ONLY_CUDA()` | 只在 CUDA 实例运行 |
| `REQUIRE_MIN_DEVICES(n)` | 加速器设备不足时 skip |

## 4. 扩展新设备平台（如沐曦 MUSA）

当前测试体系围绕 CPU / CUDA 两种设备参数化。如果需要支持新平台（以沐曦 MUSA 为例），需要改动以下几处：

### 4.1 框架层：注册新设备类型

在 `infini_train/include/device.h` 的 `DeviceType` 枚举中新增：

```cpp
enum class DeviceType : int8_t {
    kCPU = 0,
    kCUDA = 1,
    kMUSA = 2,  // 新增
};
```

### 4.2 测试工具层：`test_utils.h`

1. 新增运行时检测函数和 `CudaDeviceTypes` 的对称版本：

```cpp
#ifdef USE_MUSA
inline int GetMusaDeviceCount() { /* musaGetDeviceCount ... */ }
#else
inline int GetMusaDeviceCount() { return 0; }
#endif
inline bool HasMusaRuntime() { return GetMusaDeviceCount() > 0; }

inline std::vector<Device::DeviceType> MusaDeviceTypes() {
    if (HasMusaRuntime()) {
        return {Device::DeviceType::kMUSA};
    }
    LOG(INFO) << "No MUSA runtime found, skipping MUSA tests.";
    return {};
}
```

2. 新增 `ONLY_MUSA()` 宏：

```cpp
#define ONLY_MUSA() \
    do { if (GetParam() != infini_train::Device::DeviceType::kMUSA) { GTEST_SKIP() << "MUSA-only test"; } } while (0)
```

3. `FillSequentialTensor` 中新增 MUSA 设备的 host→device copy 路径。

### 4.3 注册宏：新增 MUSA 实例

```cpp
#define INFINI_TRAIN_REGISTER_TEST(TestName)                                    \
    INSTANTIATE_TEST_SUITE_P(CPU, TestName,                                     \
        ::testing::Values(infini_train::Device::DeviceType::kCPU));             \
    INSTANTIATE_TEST_SUITE_P(CUDA, TestName,                                    \
        ::testing::ValuesIn(infini_train::test::CudaDeviceTypes()));            \
    INSTANTIATE_TEST_SUITE_P(MUSA, TestName,                                    \
        ::testing::ValuesIn(infini_train::test::MusaDeviceTypes()))
```

### 4.4 CMake 层：`test_macros.cmake`

在 `infini_train_add_test_suite` 宏中新增 `musa` label 支持：

```cmake
elseif(_label STREQUAL "musa")
    set(_filter "MUSA/*")
```

并将默认 label 列表从 `cpu cuda` 扩展为 `cpu cuda musa`。

### 4.5 检查清单

| 步骤 | 文件 | 改动 |
|------|------|------|
| 1 | `device.h` | `DeviceType` 枚举新增 `kMUSA` |
| 2 | `test_utils.h` | 新增 `GetMusaDeviceCount()` / `HasMusaRuntime()` / `MusaDeviceTypes()` / `ONLY_MUSA()` |
| 3 | `test_utils.h` | `FillSequentialTensor` 新增 MUSA 路径 |
| 4 | `test_utils.h` | `INFINI_TRAIN_REGISTER_TEST` 新增 MUSA 实例 |
| 5 | `test_macros.cmake` | `infini_train_add_test_suite` 新增 `musa` label + filter |
| 6 | `CMakeLists.txt`（根） | 新增 `USE_MUSA` option + MUSA SDK 查找 + kernel 编译 |
