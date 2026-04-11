# Device Guard Design
Device 注册机制是 InfiniTrain 面向多硬件后端的统一运行时抽象与插件化接入基础设施。

## 1. 核心组件

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

### 1.1 DeviceGuardImpl：运行时能力抽象（对外暴露）

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

### 1.2 DeviceGuard：RAII 前端接口

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

### 1.3 DeviceGuardImplRegistry：全局注册表

`DeviceGuardImplRegistry`是 InfiniTrain 中用于管理 device runtime 后端实现的全局注册表，采用 singleton 模式，生命周期覆盖整个进程。

其核心职责是维护`DeviceType -> DeviceGuardImpl`的一对一映射关系：

```C++
std::unordered_map<Device::DeviceType, std::unique_ptr<DeviceGuardImpl>> impls_;
```

## 2. Runtime Capability 获取与使用范式

### 2.1 获取入口

```C++
DeviceGuardImpl* GetDeviceGuardImpl(Device::DeviceType type);
```

- 返回指定`DeviceType`的 DeviceGuardImpl
- 若未注册对应 backend，直接报错

### 2.2 推荐使用模式（标准范式）

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

## 3. Backend 注册机制（静态注册）

### 3.1 注册宏

```C++
#define INFINI_TRAIN_REGISTER_DEVICE_GUARD_IMPL(device_type, class_impl)                                               \
    static const bool __infini_train_device_guard_registered##__COUNTER__ = []() {                                     \
        infini_train::core::DeviceGuardImplRegistry::Instance().Register(device_type, std::make_unique<class_impl>()); \
        return true;                                                                                                   \
    }();
```

采用静态变量 + lambda 在程序启动阶段完成注册。

### 3.2 使用示例（CUDA Backend）

```C++
class CudaGuardImpl : public DeviceGuardImpl {
    ...
};

INFINI_TRAIN_REGISTER_DEVICE_GUARD_IMPL(Device::DeviceType::kCUDA, CudaGuardImpl)
```

