# 测试体系设计

**核心思路：** 测试屏蔽平台差异，底层实例化不同平台，特化需单独处理的测试。

## 1. 架构

```
tests/
├── CMakeLists.txt              # 顶层：include 宏 + add_subdirectory
├── common/
│   ├── CMakeLists.txt          # 公共 test_main target
│   └── test_utils.h            # C++ 基类、skip 宏、填充工具函数
├── tensor/                     # Tensor 创建 / 拷贝 / 销毁 / 算子
├── optimizer/                  # Optimizer 创建 / step
├── autograd/                   # 各 autograd op 的 forward / backward
├── hook/                       # Module hook + precision check
├── lora/                       # LoRA 相关
├── dtype/                      # Scalar / dtype dispatch + 编译期负面测试
├── transformer/                # Transformer 架构测试
└── checkpoint/                 # Checkpoint 序列化测试

cmake/
└── test_macros.cmake           # CMake 宏：infini_train_add_test / infini_train_add_test_suite
```

### 核心设计：设备参数化

测试不区分 CPU / CUDA 平台。一个测试定义通过 GTest 参数化自动在所有可用设备上运行：

- `INFINI_TRAIN_REGISTER_TEST(TestName)` — 注册 CPU + CUDA 两个实例

无 GPU 时 CUDA 实例在注册阶段直接跳过（不会出现在测试列表里），并打印一条 `LOG(INFO)` 提示。

### 基类层次

| 基类 | 用途 | 提供的能力 |
|------|------|-----------|
| `InfiniTrainTest` | 通用参数化测试 | `GetDevice()`（当前参数化的 `Device`） |

测试中的张量直接通过 `Tensor` 构造接口创建：

```cpp
auto t = std::make_shared<Tensor>(shape, DataType::kFLOAT32, GetDevice());
auto g = std::make_shared<Tensor>(shape, DataType::kFLOAT32, GetDevice(), /*requires_grad=*/true);
t->Fill(1.0f);                   // 常量填充（framework 内置 API）
```

### 跳过特定平台

这些宏函数涉及到了具体平台，用来针对性检验或跳过某些测试样例。

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

# 只运行 CUDA 测试
ctest -L cuda --output-on-failure

# 运行单个测试二进制（看完整 GTest 输出）
./tests/tensor/test_tensor_cpu
./tests/autograd/test_autograd_cuda

# GTest filter 过滤特定用例
./tests/tensor/test_tensor_cpu --gtest_filter="CPU/TensorCreateTest.*"
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
#include "gtest/gtest.h"

#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class FooBasicTest : public infini_train::test::InfiniTrainTest {};

TEST_P(FooBasicTest, CreateTensor) {
    auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice());
    EXPECT_NE(tensor, nullptr);
}

TEST_P(FooBasicTest, CUDAOnlyFeature) {
    SKIP_CPU();
    // CUDA-specific logic ...
}

INFINI_TRAIN_REGISTER_TEST(FooBasicTest);
```

**基类选择：**

所有测试类都继承 `InfiniTrainTest`。需要梯度时，给 `Tensor` 构造传 `requires_grad=true`；需要填充数据时用 `Tensor::Fill`。

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

### 3.2 在已有目录新增测试文件

所有使用 `file(GLOB ...)` 的目录（autograd、tensor、optimizer、hook、lora）：

1. 把新文件放入对应目录，命名为 `test_*.cc`
2. 重新 `cmake ..`（glob 在 configure 时求值）
3. 完成

无需修改任何 CMakeLists.txt。

### 3.3 工具函数速查

`test_utils.h` 提供的常用工具：

| 函数 / 宏 | 用途 |
|-----------|------|
| `GetDevice()` | 返回当前参数化的 `Device`（基类方法） |
| `SKIP_CPU()` | 跳过 CPU 实例 |
| `ONLY_CPU()` | 只在 CPU 实例运行 |
| `ONLY_CUDA()` | 只在 CUDA 实例运行 |
| `REQUIRE_MIN_DEVICES(n)` | 加速器设备不足时 skip |

## 4. 扩展新设备平台

第三方设备统一占用 `DeviceType::kPrivateUse1`，不再向框架枚举和根 CMake
增加厂商类型或 SDK 选项。厂商仓库负责注册 runtime、kernel 和可选 CCL，
并将 InfiniTrain 固定为同一链接图中的 submodule。

框架中的 `tests/backend/test_privateuse1_backend.cc` 使用不依赖硬件的 fake
backend 验证扩展契约。厂商硬件测试应放在厂商目录中，单独链接并显式初始化
provider；fake backend 与真实 backend 不能放进同一测试进程，因为一个进程
只能注册一个 `kPrivateUse1` provider。

厂商测试建议至少覆盖：

1. `DeviceGuardImpl` 的设备、stream、event、allocator 和 copy 行为；
2. `Cast`、`Fill`、`NoOpForward`、`NoOpBackward` 等基础 kernel；
3. CCL 初始化和 collective 行为；
4. provider 名称解析和重复初始化。
