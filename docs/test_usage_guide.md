# 测试使用指南

本指南介绍如何在 InfiniTrain 中构建、运行和编写测试。底层设计详见 [test_infrastructure_design.md](test_infrastructure_design.md) 和 [test_lifecycle.md](test_lifecycle.md)。

---

## 构建测试

```bash
mkdir build && cd build
cmake -DBUILD_TEST=ON -DUSE_CUDA=ON -DUSE_NCCL=ON ..
make -j$(nproc)
```

仅构建 CPU（不含 CUDA）：

```bash
cmake -DBUILD_TEST=ON -DUSE_CUDA=OFF ..
```

---

## 运行测试

```bash
# 运行所有测试
ctest --output-on-failure

# 仅运行 CPU 测试
ctest -L cpu --output-on-failure

# 仅运行 CUDA 测试
ctest -L cuda --output-on-failure

# 运行指定测试套件
ctest -R tensor --output-on-failure

# 直接运行测试二进制，使用 GTest 过滤器
./tests/tensor/test_tensor_cpu --gtest_filter="CPU/TensorCreateTest.*"
./tests/tensor/test_tensor_cuda --gtest_filter="CUDA/TensorCreateTest.*"
```

---

## 编写新测试

### 1. 创建测试文件

在 `tests/` 下对应子目录中新建文件，例如 `tests/tensor/test_tensor_copy.cc`：

```cpp
#include "tests/common/test_utils.h"

class TensorCopyTest : public infini_train::test::InfiniTrainTest {};

TEST_P(TensorCopyTest, CopiesDataCorrectly) {
    auto src = std::make_shared<Tensor>(std::vector<int64_t>{4}, DataType::kFLOAT32, GetDevice());
    src->Fill(1.0f);

    auto dst = std::make_shared<Tensor>(std::vector<int64_t>{4}, DataType::kFLOAT32, GetDevice());
    // ... 执行拷贝并断言 ...
    EXPECT_EQ(dst->Dims(), src->Dims());
}

INFINI_TRAIN_REGISTER_TEST(TensorCopyTest);
```

注意事项：
- 继承 `InfiniTrainTest`。
- 使用 `TEST_P`，设备参数由测试目标注入。
- 文件末尾调用 `INFINI_TRAIN_REGISTER_TEST`；CMake 为每个已配置设备生成独立变体。

### 2. 在 CMakeLists.txt 中注册

在子目录的 `CMakeLists.txt`（例如 `tests/tensor/CMakeLists.txt`）中添加：

```cmake
infini_train_add_test_suite(test_tensor_copy
  SOURCES test_tensor_copy.cc
)
```

这一步只登记 suite。InfiniTrain 在所有登记完成后生成
`test_tensor_copy_cpu`；`USE_CUDA=ON` 时还生成 `test_tensor_copy_cuda`。外部
PrivateUse1 provider 可通过通用接口追加自己的目标。

若某个 suite 只验证框架内部实现、不适用于 PrivateUse1，在注册时添加
`EXCLUDE_PRIVATEUSE1`；否则默认进入 PrivateUse1 公共测试集合。

---

## 基类辅助方法

以下方法在 `TEST_P` 体内可直接调用（通过 `this->` 隐式访问）：

| 方法 | 说明 |
|---|---|
| `GetDevice()` | 返回当前测试目标注入的设备 |
| `DeviceCount()` | 通过 `DeviceGuardImpl` 查询当前后端设备数 |
| `IsAccelerator()` | 当前设备是否为非 CPU 加速器 |
| `tensor->Fill(value)` | 用常量填充张量所有元素（`Tensor` 内置方法） |

张量创建直接使用 `std::make_shared<Tensor>(shape, dtype, GetDevice(), requires_grad)`，`requires_grad` 参数默认 `false`，需要梯度的测试传 `true` 即可。

---

## 条件执行宏

在 `TEST_P` 体内使用，跳过不适用的场景：

```cpp
TEST_P(MyTest, 仅加速器) {
    SKIP_CPU();    // 在 CPU 上跳过
    // ...
}

TEST_P(MyTest, 仅CPU) {
    ONLY_CPU();    // 在所有加速器上跳过
    // ...
}

TEST_P(MyTest, 需要多卡) {
    REQUIRE_MIN_DEVICES(2);  // 当前后端设备数量不足时跳过
    // ...
}
```

---

## 添加新测试模块

1. 在 `tests/` 下新建子目录，例如 `tests/mymodule/`。
2. 在其中创建 `CMakeLists.txt`，对每个测试文件调用 `infini_train_add_test_suite()`。
3. 在 `tests/CMakeLists.txt` 中添加 `add_subdirectory(mymodule)`。

## 外部 PrivateUse1 后端复用测试

外部仓库以 `BUILD_TEST=ON`、`USE_CUDA=OFF` 添加 InfiniTrain，并在 provider
链接 target 创建后调用：

```cmake
infini_train_add_privateuse1_test_suites(
  BACKEND_NAME ${INFINITRAIN_BACKEND}
  DEVICE_INDEX 0
  LINK_LIBRARIES PrivateUse1Backend::Executable
  BACKEND_HEADER "privateuse1_backend/backend.h"
  BACKEND_REGISTRAR privateuse1_backend::RegisterBackend
  RUN_SERIAL
)
```

该调用为公共 suite 追加 `test_*_<BACKEND_NAME>` 目标；`DEVICE_INDEX` 默认为
`0`。例如 `BACKEND_NAME=maca` 使用 `maca;accelerator;hardware` CTest 标签：

```bash
ctest -L maca --output-on-failure  # MACA
ctest -L cpu --output-on-failure   # 上游 CPU
ctest --output-on-failure          # 全部
```

---

## Autograd 测试

创建启用自动微分的张量时，给 `Tensor` 构造的第四个参数传 `true`：

```cpp
#include "tests/common/test_utils.h"

class MyOpTest : public infini_train::test::InfiniTrainTest {};

TEST_P(MyOpTest, 前向传播) {
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    input->Fill(1.0f);
    auto weight = std::make_shared<Tensor>(std::vector<int64_t>{4, 3}, DataType::kFLOAT32, GetDevice(), true);
    weight->Fill(0.5f);
    auto output = MyOp(input, weight);
    EXPECT_NE(output, nullptr);
}

INFINI_TRAIN_REGISTER_TEST(MyOpTest);
```
