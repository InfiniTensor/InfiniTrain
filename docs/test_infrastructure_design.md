# 测试体系设计

**核心思路：** 测试屏蔽平台差异，底层实例化不同平台，特化需单独处理的测试。

## 1. 架构

```
tests/
├── CMakeLists.txt              # 顶层：include 宏 + add_subdirectory
├── common/
│   ├── CMakeLists.txt          # 公共 test_main target
│   └── test_utils.h            # 设备测试基类与 skip 宏
├── backend/                    # PrivateUse1 扩展契约测试
├── tensor/                     # Tensor 创建 / 拷贝 / 销毁 / 算子
├── optimizer/                  # Optimizer 创建 / step
├── autograd/                   # 各 autograd op 的 forward / backward
├── hook/                       # Module hook + precision check
├── lora/                       # LoRA 相关
├── dtype/                      # Scalar / dtype dispatch + 编译期负面测试
├── transformer/                # Transformer 架构测试
└── checkpoint/                 # Checkpoint 序列化测试

cmake/
└── test_macros.cmake           # CMake 测试目标与设备注入接口
```

### 核心设计：设备参数化

测试不区分具体平台。一个测试定义通过 GTest 参数化，由 CMake 为每个已配置设备生成独立测试目标：

- `INFINI_TRAIN_REGISTER_TEST(TestName)` — 注册当前目标注入的设备实例
- 内建目标注入 CPU，以及 `USE_CUDA=ON` 时的 CUDA
- 外部仓库可注入 `kPrivateUse1` provider，无需修改测试源码

每个测试二进制只包含一种设备实例。`USE_CUDA=OFF` 时不会创建 CUDA 目标，避免空测试被误报为成功。

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

// 需要 ≥n 个加速器设备
REQUIRE_MIN_DEVICES(n);
```

### CMake 接口

`test_macros.cmake` 提供三类接口减少 CMakeLists 样板：

- `infini_train_add_test(name SOURCES ... LABELS ...)` — 创建可执行文件、链接 GTest + 框架库、用 `gtest_discover_tests` 自动发现用例
- `infini_train_add_test_suite(name SOURCES ... LABELS ...)` — 只登记公共 suite 及其 source/timeout 元数据
- `infini_train_add_privateuse1_test_suites(...)` — 以统一的 PrivateUse1 accelerator 身份复用所有公共 suite

所有 suite 登记完成后，InfiniTrain 统一实例化 CPU，以及 `USE_CUDA=ON` 时的
CUDA。嵌入方可在此基础上追加 PrivateUse1 provider 目标。由于 PrivateUse1
要求 `USE_CUDA=OFF`，Backends 测试构建包含 CPU 和 provider 变体，不包含 CUDA。

不应由 PrivateUse1 复用的框架内部 suite，可在
`infini_train_add_test_suite(... EXCLUDE_PRIVATEUSE1)` 中显式排除。

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

无 GPU 机器上 `cmake -DBUILD_TEST=ON -DUSE_CUDA=OFF ..` 即可，CUDA 测试目标不会生成。

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

TEST_P(FooBasicTest, AcceleratorOnlyFeature) {
    SKIP_CPU();
    // Accelerator-specific logic ...
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

InfiniTrain 会生成 `test_foo_cpu`；启用 CUDA 时还生成 `test_foo_cuda`。
PrivateUse1 provider 再使用其 `BACKEND_NAME` 追加对应目标，例如下游传入
`maca` 时生成 `test_foo_maca`，并可通过 `maca` 标签筛选。

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
| `REQUIRE_MIN_DEVICES(n)` | 加速器设备不足时 skip |

## 4. 扩展新设备平台

第三方设备统一占用 `DeviceType::kPrivateUse1`，不再向框架枚举和根 CMake
增加厂商类型或 SDK 选项。厂商仓库负责注册 runtime、kernel 和可选 CCL，
并通过 `infini_train_add_privateuse1_test_suites()` 追加公共 suite。该接口要求
`USE_CUDA=OFF`，保留上游 CPU 测试，并使用 `BACKEND_NAME` 作为 target 后缀和
CTest 标签；`DEVICE_INDEX` 默认为 `0`。调用方式见
[测试使用指南](test_usage_guide.md#外部-privateuse1-后端复用测试)。

框架中的 `tests/backend/test_privateuse1_backend.cc` 使用不依赖硬件的 fake
backend 验证扩展契约。它与真实 provider 使用不同测试进程，避免重复注册
`kPrivateUse1` provider。
