# Low-Precision DType Abstraction & Backend Registration Design
低精度 dtype 抽象是 InfiniTrain 面向多后端的统一类型语义与显式注册基础设施。

## 1. Design In One Diagram

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

- 框架层不提供任何「DataType → 后端 C++ 类型」映射路径；所有具体类型绑定均在后端通过 `BackendTypeMap<Dev, DType>` 完成。
- `BackendTypeMap<Dev, DType>` 主模板**只声明不定义**，只有后端显式特化并完成注册的组合才允许参与 kernel dispatch；未注册组合会在模板实例化阶段被 `static_assert` 于编译期拦截。

## 2. Core API

| API | 位置 | 说明 |
| --- | --- | --- |
| `struct FP16 / BF16` | [datatype.h](../infini_train/include/datatype.h) | 16-bit 软件包装（IEEE-754 half / truncated bf16），承担框架身份、存储布局、fallback 转换；不承担后端高性能算术语义。 |
| `PromoteDataTypes(DataType, DataType)` | [datatype.h](../infini_train/include/datatype.h) | 纯枚举到枚举的类型提升。规则：FP16+BF16→FP32；浮点优先于整数；同类按字节宽取大。 |
| `BackendTypeMap<Dev, DType>` | [core/backend_type_map.h](../infini_train/include/core/backend_type_map.h) | 主模板**只声明不定义**；后端通过显式特化提供 `::type`。 |
| `INFINI_REGISTER_STANDARD_BACKEND_TYPES(DEV)` | [core/backend_type_map.h](../infini_train/include/core/backend_type_map.h) | 一次性注册 10 个非低精度 dtype（`kUINT8…kFLOAT64`）到对应 C++ 标量。 |
| `DispatchCpuFunc / DispatchCudaFunc<AllowedDTypes...>` | `src/core/runtime/{cpu,cuda}/{cpu,cuda}_dispatch.h` | 后端 dispatch 入口，底层转发到 `DispatchByTypeMap<TypeMap, AllowedDTypes...>`。 |

## 3. How To Add A New Backend

按以下清单操作，**不需要**修改 `infini_train/include/` 下的任何框架头文件，也不需要 `#ifdef`：

1. 在后端的 `*_dispatch.h` 里 include `core/backend_type_map.h` 与 `dtype_dispatch.h`。
2. 调用 `INFINI_REGISTER_STANDARD_BACKEND_TYPES(Device::DeviceType::kXxx)` 注册 10 个标准 dtype。
3. 若硬件支持低精度，显式特化 `BackendTypeMap<kXxx, kFLOAT16>` / `BackendTypeMap<kXxx, kBFLOAT16>` 指向后端本地 16-bit 标量类型；不支持则直接跳过，调用方一旦 dispatch 到未注册的 dtype 会在编译期触发 `static_assert`。
4. 定义 `XxxTypeMap<DType>` 转发/继承到 `BackendTypeMap<kXxx, DType>`。
5. 提供 `DispatchXxxFunc` 入口，转发到 `DispatchByTypeMap<XxxTypeMap, AllowedDTypes...>`。

### Example

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

## 4. Failure Modes

| 情形 | 表现 |
| --- | --- |
| 后端未注册某个 dtype（`BackendTypeMap<Dev, DType>` 无特化），但被 dispatch 命中 | 编译期 `static_assert` 触发，错误信息指向 `BackendTypeMap` 的显式注册要求。 |
| dispatch 的 dtype 不在调用点 `AllowedDTypes...` 白名单内 | 运行期 `LOG_UNSUPPORTED_DTYPE` 报错。 |
