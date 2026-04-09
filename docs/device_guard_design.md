# Device Guard Design
device 注册初版基建 pr：https://github.com/InfiniTensor/InfiniTrain/pull/103

## 1. 设计背景与目标

### 1.1 背景

InfiniTrain 需要长期支持：

- 多种设备类型（CPU/CUDA/国产芯片）
- 多种运行时能力（stream、memory、blas、通信等）
- 在不侵入上层逻辑的前提下进行后端扩展与替换

在实际工程中，如果设备相关逻辑散落在框架各个模块，会导致：

- `#ifdef USE_CUDA/USE_MUSA/...` 泛滥
- 新硬件接入需要修改大量框架核心代码
- 设备切换与资源管理缺乏统一语义

### 1.2 设计目标

InfiniTrain 的 device 注册机制设计目标是：

1. 统一抽象：将所有与设备相关的运行时行为抽象到一个统一接口中。
2. 后端可插拔：新设备后端可通过注册机制接入，无需修改框架核心逻辑。
3. RAII 语义清晰：设备切换、资源恢复具备严格的作用域。
4. 最小上层侵入：上层模块（Tensor/Autograd/Module）只感知 DeviceGuard/DeviceGuardImpl，不感知具体后端实现。

## 2. 核心组件

InfiniTrain 的 device 机制由三类核心组件构成：

```C++
+-------------------+
|   DeviceGuard     |  ← 对外 RAII 接口（public）
+-------------------+
          |
          v
+-------------------+
| DeviceGuardImpl   |  ← 后端抽象接口（virtual）
+-------------------+
          ^
          |
+-------------------+
| DeviceGuardImpl   |
|   Registry        |  ← 全局注册表（singleton）
+-------------------+
```

其中 DeviceGuard 与 DeviceGuardImpl 的关系是：

| 组件            | 职责                                                         |
| --------------- | ------------------------------------------------------------ |
| DeviceGuard     | 管理 “当前在哪个 device 上” 的上下文语义（RAII），语义与 device index 绑定；负责 device 的保存/切换/恢复，并将具体 runtime 操作转发给对应的 DeviceGuardImpl。 |
| DeviceGuardImpl | 管理 “在该类 device 上如何执行 runtime 操作”，语义与 device type 绑定；对外提供 设备管理查询、stream、blas、同步、内存 等运行时能力接口。 |

### 2.1 DeviceGuardImpl：运行时能力抽象（对外暴露）

DeviceGuardImpl 是 InfiniTrain 中 device runtime 能力的统一抽象接口，并且是框架内部对外暴露的能力接口，封装了所有与 device 相关的行为（待补充 event 相关接口）：

```C++
// ----------------------------------------------------------------------
// Device management
// ----------------------------------------------------------------------

virtual Device GetDevice() const = 0;

virtual void SetDevice(Device device) const;

virtual int8_t DeviceCount() const;

virtual Device::DeviceType Type() const = 0;

// ----------------------------------------------------------------------
// Stream management
// ----------------------------------------------------------------------

virtual Stream *GetStream(Device) const;

// ----------------------------------------------------------------------
// Synchronization
// ----------------------------------------------------------------------

virtual void SynchronizeDevice(Device) const;

virtual void SynchronizeStream(Stream *) const;

// ----------------------------------------------------------------------
// BLAS handle
// ----------------------------------------------------------------------

virtual BlasHandle *GetBlasHandle(Device) const;

// ----------------------------------------------------------------------
// Memory operations
// ----------------------------------------------------------------------

virtual void Malloc(void **dev_ptr, size_t size) = 0;

virtual void MallocAsync(void **dev_ptr, size_t size, Stream *stream);

virtual void Free(void *dev_ptr) = 0;

virtual void FreeAsync(void *dev_ptr, Stream *stream);

virtual void Memcpy(void *dst, const void *src, size_t count, MemcpyKind kind) = 0;

virtual void MemcpyAsync(void *dst, const void *src, size_t count, MemcpyKind kind, Stream *stream);

virtual void ResetMemPoolHighWatermarks(Device device) const;

virtual std::pair<size_t, size_t> GetMemPoolPeakMB(Device device) const;
```

### 2.2 DeviceGuard：RAII 前端接口

DeviceGuard 是设备上下文的 RAII 管理器，其职责严格限定为：

- 保存当前 device
- 切换到目标 device
- 在作用域结束时恢复原 device

DeviceGuard 不直接提供任何运行时能力接口。

使用示例：

```C++
{
    DeviceGuard guard(Device(DeviceType::kCUDA, 1));
    // 当前线程的 device 上下文被切换到 CUDA:1
    // 所有 runtime 操作将发生在 CUDA:1
}
// 离开作用域后，自动恢复进入前的 device
```

### 2.3 DeviceGuardImplRegistry：全局注册表

`DeviceGuardImplRegistry`是 InfiniTrain 中用于管理 device runtime 后端实现的全局注册表，采用 singleton 模式，生命周期覆盖整个进程。

其核心职责是维护`DeviceType -> DeviceGuardImpl`的一对一映射关系：

```C++
std::unordered_map<Device::DeviceType, std::unique_ptr<DeviceGuardImpl>> impls_;
```

## 3. Runtime Capability 获取与使用范式

### 3.1 获取入口

```C++
DeviceGuardImpl* GetDeviceGuardImpl(Device::DeviceType type);
```

- 返回指定`DeviceType`的 DeviceGuardImpl
- 若未注册对应 backend，直接报错

### 3.2 推荐使用模式（标准范式）

```C++
auto device = tensor->GetDevice();
const int64_t num_elements = tensor->NumElements();
std::vector<float> buffer(num_elements);

{
    // 1. 切换 device 上下文（RAII scope）
    core::DeviceGuard guard(device);

    // 2. 获取 runtime capability
    auto* impl = core::GetDeviceGuardImpl(device.type());

    // 3. 执行 runtime 操作
    const core::MemcpyKind kind =
        device.type() == Device::DeviceType::kCPU
            ? core::MemcpyKind::kD2D   // CPU: host-host memcpy
            : core::MemcpyKind::kH2D;  // Device: host-device copy

    impl->MemcpyAsync(
        tensor->DataPtr(),               // dst
        buffer.data(),                   // src
        num_elements * sizeof(float),    // count
        kind,                            // kind（说明：在 CPU backend 中，kD2D 对应普通 memcpy）
        impl->GetStream(device)          // stream
    );
}  // <-- DeviceGuard 在此处析构，device 上下文被恢复
```

## 4. Backend 注册机制（静态注册）

### 4.1 注册宏

```C++
#define INFINI_TRAIN_REGISTER_DEVICE_GUARD_IMPL(device_type, class_impl)                                               \
    static const bool __infini_train_device_guard_registered##__COUNTER__ = []() {                                     \
        infini_train::core::DeviceGuardImplRegistry::Instance().Register(device_type, std::make_unique<class_impl>()); \
        return true;                                                                                                   \
    }();
```

采用静态变量 + lambda 在程序启动阶段完成注册。

### 4.2 使用示例（CUDA Backend）

```C++
class CudaGuardImpl : public DeviceGuardImpl {
    ...
};

INFINI_TRAIN_REGISTER_DEVICE_GUARD_IMPL(Device::DeviceType::kCUDA, CudaGuardImpl)
```

