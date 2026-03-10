# InfiniTrain 算子开发指南

## 整体架构层次

```
用户代码 (example/gpt2)
    ↓
NN Modules (include/nn/modules/, src/nn/modules/)
    ↓
Autograd Functions (include/autograd/, src/autograd/)
    ↓
Dispatcher (include/dispatcher.h)  ← 统一分发中心
    ↓
Kernels: CPU (src/kernels/cpu/) / CUDA (src/kernels/cuda/)
```

## 调用链示例（Linear）

```
nn::Linear::Forward()
  └── autograd::Linear()->Apply({input, weight, bias})
        └── Function::Apply()
              ├── Forward() → Dispatcher::Call({kCUDA, "LinearForward"}, ...)
              │     └── kernels::cuda::LinearForward()  ← 实际计算
              ├── SetupContext()  ← 保存 input/weight 供 backward 用
              └── 输出 Tensor 的 grad_fn = this (Linear Function)

loss->Backward()
  └── autograd::Linear::Backward()
        └── Dispatcher::Call({kCUDA, "LinearBackward"}, ...)
              └── kernels::cuda::LinearBackward()
```

## 添加一个新算子的步骤

### 1. 实现 CUDA Kernel (`src/kernels/cuda/foo.cu`)

```cpp
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {

// Forward 实现
std::shared_ptr<Tensor> FooForward(const std::shared_ptr<Tensor> &input, ...) {
    auto dtype = input->Dtype();
    auto output = std::make_shared<Tensor>(output_dims, dtype, input->GetDevice());

    // 用 DispatchFunc 做类型分发
    DispatchFunc<INFINI_ALL_FLOATING_TYPES>(dtype, [=]<typename T>() {
        auto *cuda_stream = static_cast<cudaStream_t>(
            input->GetDevice().stream());
        FooKernel<T><<<grid, block, 0, cuda_stream>>>(...);
    }, "CUDA FooForward");

    return output;
}

// Backward 实现
std::shared_ptr<Tensor> FooBackward(const std::shared_ptr<Tensor> &grad_output, ...) { ... }

// 注册到 Dispatcher（静态初始化，程序启动自动注册）
REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, FooForward,
                infini_train::kernels::cuda::FooForward)
REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, FooBackward,
                infini_train::kernels::cuda::FooBackward)

} // namespace infini_train::kernels::cuda
```

### 2. 实现 CPU Kernel (`src/kernels/cpu/foo.cc`)

```cpp
#include "infini_train/include/dispatcher.h"

namespace infini_train::kernels::cpu {

std::shared_ptr<Tensor> FooForward(const std::shared_ptr<Tensor> &input, ...) {
    auto output = std::make_shared<Tensor>(dims, DataType::kFLOAT32);
    output->EigenMatrix() = input->EigenMatrix()...;  // Eigen 计算
    return output;
}

REGISTER_KERNEL(infini_train::Device::DeviceType::kCPU, FooForward,
                infini_train::kernels::cpu::FooForward)
REGISTER_KERNEL(infini_train::Device::DeviceType::kCPU, FooBackward,
                infini_train::kernels::cpu::FooBackward)

} // namespace infini_train::kernels::cpu
```

### 3. 实现 Autograd Function

**头文件** (`include/autograd/foo.h`):
```cpp
#pragma once
#include "infini_train/include/autograd/function.h"

namespace infini_train::autograd {
class Foo : public Function {
public:
    static constexpr char kType[] = "FooFunction";
    explicit Foo(float param) : Function(kType), param_(param) {}

    std::vector<std::shared_ptr<Tensor>> Forward(
        const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    void SetupContext(
        const std::vector<std::shared_ptr<Tensor>> &input_tensors,
        const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
    std::vector<std::shared_ptr<Tensor>> Backward(
        const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    float param_;
    // saved_tensors_ 继承自 Function，用于存 backward 需要的中间值
};
} // namespace infini_train::autograd
```

**实现** (`src/autograd/foo.cc`):
```cpp
#include "infini_train/include/autograd/foo.h"
#include "infini_train/include/dispatcher.h"

namespace infini_train::autograd {

std::vector<std::shared_ptr<Tensor>> Foo::Forward(
    const std::vector<std::shared_ptr<Tensor>> &inputs) {
    auto &input = inputs[0];
    auto device = input->GetDevice().type();
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
        {device, "FooForward"}, input, param_)};
}

void Foo::SetupContext(
    const std::vector<std::shared_ptr<Tensor>> &inputs,
    const std::vector<std::shared_ptr<Tensor>> &outputs) {
    saved_tensors_ = {inputs[0]};  // 保存 backward 需要的张量
}

std::vector<std::shared_ptr<Tensor>> Foo::Backward(
    const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    auto &input = saved_tensors_[0];
    auto device = input->GetDevice().type();
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
        {device, "FooBackward"}, grad_outputs[0], input, param_)};
}

} // namespace infini_train::autograd
```

### 4. 添加 NN Module（可选，若有权重）

**头文件** (`include/nn/modules/foo.h`):
```cpp
#pragma once
#include "infini_train/include/nn/modules/module.h"

namespace infini_train::nn {
class Foo : public CloneableModule<Foo> {
public:
    explicit Foo(int in_features, float param);
    std::vector<std::shared_ptr<Tensor>> Forward(
        const std::vector<std::shared_ptr<Tensor>> &inputs) override;
private:
    float param_;
    // 权重用 RegisterParameter() 注册
};
} // namespace infini_train::nn
```

**实现** (`src/nn/modules/foo.cc`):
```cpp
std::vector<std::shared_ptr<Tensor>> Foo::Forward(
    const std::vector<std::shared_ptr<Tensor>> &inputs) {
    return std::make_shared<autograd::Foo>(param_)->Apply({inputs[0]});
}
```

### 5. 添加 Functional API（可选）

`include/nn/functional.h`:
```cpp
std::shared_ptr<Tensor> Foo(const std::shared_ptr<Tensor> &input, float param);
```

`src/nn/functional.cc`:
```cpp
std::shared_ptr<Tensor> Foo(const std::shared_ptr<Tensor> &input, float param) {
    return std::make_shared<autograd::Foo>(param)->Apply({input});
}
```

### 6. 构建系统

**无需修改 CMakeLists.txt** —— 构建系统通过 GLOB 自动收集 `src/kernels/cpu/*.cc`、`src/kernels/cuda/*.cu`、`src/autograd/*.cc` 等目录下的文件。新增文件自动被编译。

`REGISTER_KERNEL` 宏利用静态变量初始化，在程序启动时自动将 kernel 注册到 `Dispatcher::Instance()` 的 map 中（`key = {DeviceType, "KernelName"}`）。需要确保所在 .o 文件被链接（框架用 `--whole-archive` 保证这一点）。

## 关键类说明

| 类 | 文件 | 职责 |
|----|------|------|
| `Tensor` | `include/tensor.h` | 数据存储 + autograd 元信息（grad_fn, requires_grad） |
| `Function` | `include/autograd/function.h` | 计算图节点基类，Apply/Forward/Backward |
| `Dispatcher` | `include/dispatcher.h` | kernel 注册表，按 (device, name) 分发 |
| `Module` | `include/nn/modules/module.h` | 神经网络层基类，管理参数/子模块 |
| `CloneableModule<T>` | 同上 | 支持 DDP 复制的 Module 模板 |

## 类型分发工具

```cpp
// 单类型分发
DispatchFunc<INFINI_ALL_FLOATING_TYPES>(dtype, [=]<typename T>() {
    MyKernel<T><<<...>>>(static_cast<const T *>(ptr), ...);
}, "context description");

// 双类型分发（如 mixed precision）
DispatchFunc<DataTypeList<kFLOAT16, kBFLOAT16>, DataTypeList<kFLOAT32>>(
    {input_dtype, output_dtype},
    [=]<typename InT, typename OutT>() { ... },
    "context");
```

## 常见注意事项

1. **Backward 中通过 `saved_tensors_` 获取 forward 保存的张量**，不要直接捕获外部变量（生命周期问题）。
2. **CUDA kernel 中获取 stream**：`static_cast<cudaStream_t>(tensor->GetDevice().stream())`。
3. **输出 Tensor 必须用 input 的 device 创建**：`std::make_shared<Tensor>(dims, dtype, input->GetDevice())`。
4. **`REGISTER_KERNEL` 宏的第二个参数是字符串名（不加引号），与 Dispatcher::Call 中的字符串要严格一致**。
5. **不需要 grad 的算子**（如 in-place 更新）可以不实现 Backward，直接在 `NoGradGuard` 中调用 Dispatcher。
