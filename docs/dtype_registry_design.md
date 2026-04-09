# Low-Precision DType Abstraction & Backend Registration Design
统一低精度类型抽象与后端显式注册 pr：https://github.com/InfiniTensor/InfiniTrain/pull/114

## 1. 背景与动机

InfiniTrain 在引入 BF16 / FP16 之前，框架层并没有低精度类型的统一抽象，所有关于 16-bit 浮点的语义都直接绑定在 CUDA 原生类型 `__half` / `__nv_bfloat16` 上。这
导致几个问题：

1. **框架代码被 `#ifdef USE_CUDA` 污染。**
   `infini_train/include/datatype.h`、`infini_train/src/nn/init.cc` 等通用模块都需要
   写出 `#ifdef USE_CUDA … #else …` 来在「有 CUDA」和「没有 CUDA」两个版本之间
   切换 16-bit 类型映射；非 CUDA 路径只能退化成 `uint16_t`，而 `uint16_t` 又会与
   `kUINT16` 的反向映射产生歧义。
2. **`TypeMap<DType>` 是「全后端共享」的单点表。**
   旧 `TypeMap` 把所有标量类型直接映射到 C++ 类型。CPU 与 CUDA 共享同一个表，
   意味着不可能在不同后端把 `kFLOAT16` 映射到不同的本地标量；要扩展新硬件必须改框架头文件。
3. **类型提升耦合具体后端类型。**
   旧的 `WidestType_t<T1, T2>` 在 C++ 模板层面做提升，需要每个调用点先 dispatch 出
   一对具体的标量类型（例如 `nv_bfloat16` + `float`），再交给元函数做选择。这把
   「类型提升」这一纯 dtype 级别的逻辑跟「后端具体标量」捆死了。
4. **静默 fallback 容易掩盖错误。**
   一旦某个后端忘记注册 BF16/FP16，旧实现会沉默地走到 `uint16_t` 路径，得到一个
   语义错误的内核，而不是显式报错。

本工作的目标是：

> **把 FP16/BF16 抽象成框架级类型**，让框架代码不再直接接触任何后端原生
> 16-bit 类型；同时把后端 dtype → 本地标量的映射改成**显式注册**机制，未注册的类型在编译期就被拦截。

## 2. Design In One Diagram

```
framework code ──► FP16 / BF16 (datatype.h, 纯软件实现，提供基本转换操作)
                   PromoteDataTypes(DataType, DataType)

kernel code    ──► DispatchCpuFunc / DispatchCudaFunc / DispatchXxxFunc
                         │
                         ▼
                   BackendTypeMap<Dev, DType>       (主模板只声明不定义)
                         │
                         ├─ kFLOAT16 / kBFLOAT16  → 后端在 *_dispatch.h 显式特化后注册
                         │      └── CUDA: __half / __nv_bfloat16
                         │      └── CPU : FP16 / BF16
                         └─ 其它 10 个标量 dtype 使用默认注册 → INFINI_REGISTER_STANDARD_BACKEND_TYPES(DEV)
```

要点：

- 框架层不提供任何「DataType → C++ 类型」映射路径；所有具体类型绑定均在后端通过 `BackendTypeMap<Dev, DType>` 完成。
- `BackendTypeMap<Dev, DType>` 主模板**只声明不定义**，只有后端显式特化并完成注册的 dtype 组合才允许参与 kernel dispatch；未注册组合会在模板实例化阶段被 `static_assert` 于编译期拦截。

## 3. Core API

| API | 位置 | 说明 |
| --- | --- | --- |
| `struct FP16 / BF16` | [datatype.h](../infini_train/include/datatype.h) | 16-bit 软件包装（IEEE-754 half / truncated bf16），承担框架身份、存储布局、fallback 转换；不承担后端高性能算术语义。 |
| `PromoteDataTypes(DataType, DataType)` | [datatype.h](../infini_train/include/datatype.h) | 纯枚举到枚举的类型提升。规则：FP16+BF16→FP32；浮点优先于整数；同类按字节宽取大。 |
| `BackendTypeMap<Dev, DType>` | [core/backend_type_map.h](../infini_train/include/core/backend_type_map.h) | 主模板**只声明不定义**；后端通过显式特化提供 `::type`。 |
| `INFINI_REGISTER_STANDARD_BACKEND_TYPES(DEV)` | [core/backend_type_map.h](../infini_train/include/core/backend_type_map.h) | 一次性注册 10 个非低精度 dtype（`kUINT8…kFLOAT64`）到对应 C++ 标量。 |
| `DispatchCpuFunc / DispatchCudaFunc<AllowedDTypes...>` | `src/core/runtime/{cpu,cuda}/{cpu,cuda}_dispatch.h` | 后端 dispatch 入口，底层转发到 `DispatchByTypeMap<TypeMap, AllowedDTypes...>`。 |

## 4. Scalar：框架层标量载体

`BackendTypeMap` 解决「DataType → 后端 C++ 类型」，但框架 API 还需要一种
**DataType 无关** 的方式接收标量参数：目标 tensor 的 DataType 运行期才确定，API 不可能
为每种数值类型都写重载，更不能把后端原生类型暴露给调用方。

为此引入 `Scalar`（[scalar.h](../infini_train/include/scalar.h)）：

- 固定存储：`double / int64_t / uint64_t` + `Kind` tag（`kBool / kDouble / kInt64 / kUInt64`）。
- 隐式构造覆盖所有框架标量：整数按符号分入 `kInt64 / kUInt64`，全部浮点（含 `FP16 / BF16`）归一到 `kDouble`，`bool` 独立。
- 唯一出口 `Scalar::to<T>()`，通过 `common::cpu::Cast<T>` 把存储值转换到 dispatch 选出的后端标量类型。

与其它抽象的边界：`BackendTypeMap` 管「DataType → 后端 C++ 类型」，`PromoteDataTypes` 管
「DataType → DataType」，`Scalar` 管「数值 → 后端 C++ 类型」，三者正交；`Scalar` 本身不参与类型提升决策。

### 4.1 使用模式

`Tensor::Fill(Scalar)` 是这套抽象的第一个落地点。kernel 侧使用模式如下：

```cpp
// kernels/cpu/fill.cc
void Fill(std::shared_ptr<Tensor> tensor, Scalar scalar) {
    core::cpu::DispatchCpuFunc<INFINI_ALL_TYPES>(
        tensor->Dtype(),
        [=]<typename T>() {
            auto data = reinterpret_cast<T *>(tensor->DataPtr());
            const T v = scalar.to<T>();   // Scalar 在此完成「数值 → 后端 C++ 类型」映射
            std::fill(data, data + tensor->NumElements(), v);
        },
        "CPU Fill");
}
```

`DispatchCpuFunc` 经 `BackendTypeMap` 把 `DataType` 解析为 `T`；`Scalar::to<T>()`
把用户传入值转换到该 `T`。

## 5. How To Add A New Backend

按以下清单操作，**不需要**修改 `infini_train/include/` 下的任何框架头文件，也不需要 `#ifdef`：

1. 在后端的 `*_dispatch.h` 里 include `core/backend_type_map.h` 与 `dtype_dispatch.h`。
2. 调用 `INFINI_REGISTER_STANDARD_BACKEND_TYPES(Device::DeviceType::kXxx)` 注册 10 个标准 dtype。
3. 若硬件支持低精度，显式特化 `BackendTypeMap<kXxx, kFLOAT16>` / `BackendTypeMap<kXxx, kBFLOAT16>`
   指向后端本地 16-bit 标量类型；不支持则直接跳过，调用方一旦 dispatch 到未注册的 dtype 会在
   编译期触发 `static_assert`。
4. 定义 `XxxTypeMap<DType>` 转发/继承到 `BackendTypeMap<kXxx, DType>`。
5. 提供 `DispatchXxxFunc` 入口，转发到 `DispatchByTypeMap<XxxTypeMap, AllowedDTypes...>`。

### 最小示例

```cpp
// xxx_dispatch.h
#include "infini_train/include/core/backend_type_map.h"
#include "infini_train/include/dtype_dispatch.h"

namespace infini_train::core {
// 若硬件支持低精度，显式特化 FP16/BF16
template <> struct BackendTypeMap<Device::DeviceType::kXxx, DataType::kFLOAT16>  { using type = xxx_half;   };
template <> struct BackendTypeMap<Device::DeviceType::kXxx, DataType::kBFLOAT16> { using type = xxx_bfloat; };
} // namespace infini_train::core

INFINI_REGISTER_STANDARD_BACKEND_TYPES(infini_train::Device::DeviceType::kXxx)

namespace infini_train::core::xxx {
template <DataType DType>
struct XxxTypeMap : BackendTypeMap<Device::DeviceType::kXxx, DType> {};

template <DataType... AllowedDTypes, typename Functor, typename... Args>
auto DispatchXxxFunc(DataType dtype, Functor &&f, std::string_view ctx = "", Args &&...a) {
    return DispatchByTypeMap<XxxTypeMap, AllowedDTypes...>(
        dtype, std::forward<Functor>(f), ctx, std::forward<Args>(a)...);
}
} // namespace infini_train::core::xxx
```

## 6. Failure Modes

| 情形 | 表现 |
| --- | --- |
| 后端未注册某个 dtype（`BackendTypeMap<Dev, DType>` 无特化），但被 dispatch 命中 | 编译期 `static_assert` 触发，错误信息指向 `BackendTypeMap` 的显式注册要求。 |
| dispatch 的 dtype 不在调用点 `AllowedDTypes...` 白名单内 | 运行期 `LOG_UNSUPPORTED_DTYPE` 报错。 |
