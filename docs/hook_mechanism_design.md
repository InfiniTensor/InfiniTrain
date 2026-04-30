# Hook Mechanism Design

仿照 PyTorch 设计的 Hook 机制，支持 Module 和 Function 级别的 hook。

## 1. Module Hooks

### 1.1 Forward Pre-Hook
在 forward 执行前调用。

```cpp
auto handle = module->RegisterForwardPreHook(
    [](Module* mod, const std::vector<std::shared_ptr<Tensor>>& inputs) {
        // 在 forward 前执行的逻辑
    }
);
```

**调用栈**:
```
Module::operator()(inputs)
  └─> for (hook : forward_pre_hooks_) { hook(this, inputs); }
```

### 1.2 Forward Post-Hook
在 forward 执行后调用。

```cpp
auto handle = module->RegisterForwardPostHook(
    [](Module* mod,
       const std::vector<std::shared_ptr<Tensor>>& inputs,
       const std::vector<std::shared_ptr<Tensor>>& outputs) {
        // 在 forward 后执行的逻辑
    }
);
```

**调用栈**:
```
Module::operator()(inputs)
  ├─> outputs = Forward(inputs)
  └─> for (hook : forward_post_hooks_) { hook(this, inputs, outputs); }
```

### 1.3 Backward Pre-Hook
在 backward 执行前调用。

```cpp
auto handle = module->RegisterBackwardPreHook(
    [](Module* mod, const std::vector<std::shared_ptr<Tensor>>& grad_outputs) {
        // 在 backward 前执行的逻辑
    }
);
```

**调用栈**:
```
Module::operator()(inputs)
  ├─> outputs = Forward(inputs)
  └─> for (output : outputs) {
        output->grad_fn()->RegisterBackwardPreHook([module_hooks] {
          for (hook : module_hooks) { hook(module, grad_outputs); }
        });
      }

反向传播时:
Function::BackwardPartial()
  └─> for (hook : backward_pre_hooks_) { hook(this, grad_outputs); }
```

### 1.4 Backward Post-Hook
在 backward 执行后调用。

```cpp
auto handle = module->RegisterBackwardPostHook(
    [](Module* mod,
       const std::vector<std::shared_ptr<Tensor>>& grad_inputs,
       const std::vector<std::shared_ptr<Tensor>>& grad_outputs) {
        // 在 backward 后执行的逻辑
    }
);
```

**调用栈**:
```
Module::operator()(inputs)
  ├─> outputs = Forward(inputs)
  └─> for (output : outputs) {
        output->grad_fn()->RegisterBackwardPostHook([module_hooks] {
          for (hook : module_hooks) { hook(module, grad_inputs, grad_outputs); }
        });
      }

反向传播时:
Function::BackwardPartial()
  ├─> grad_inputs = Backward(grad_outputs)
  └─> for (hook : backward_post_hooks_) { hook(this, grad_inputs, grad_outputs); }
```

### 使用场景
- 特征提取和可视化
- 激活值监控
- 梯度流分析
- 性能分析和 profiling

### 实现位置
- `infini_train/include/nn/modules/module.h`
- `infini_train/include/nn/module_hook.h`
- Module forward hooks 在 `Module::operator()` 中被调用
- Module backward hooks 在 `Module::operator()` 中注册到输出 tensor 的 `grad_fn`，在反向传播时由 `Function::BackwardPartial()` 调用
- 子类只需重写 `Forward()` 方法，hooks 会自动执行

### 使用说明
- **调用方式**: 使用 `(*module)(inputs)` 而不是 `module->Forward(inputs)`
- **子类实现**: 只需重写 `Forward()` 方法，不需要手动调用 hooks
- **Hook 自动执行**: `operator()` 会自动调用 forward pre-hooks、Forward、forward post-hooks
- **Backward Hook 执行**: Module 的 backward hooks 会在 `operator()` 中注册到输出 tensor 的 `grad_fn` 上，在反向传播时自动执行

## 2. Function Hooks

Function hooks 使用统一的类型定义：
- `FunctionPreHook`: 用于 Forward Pre-Hook 和 Backward Pre-Hook
- `FunctionPostHook`: 用于 Forward Post-Hook 和 Backward Post-Hook

### 2.1 Function Forward Pre-Hook
在 Function 的 forward 执行前调用。

```cpp
auto handle = function->RegisterForwardPreHook(
    [](autograd::Function* func, const std::vector<std::shared_ptr<Tensor>>& inputs) {
        // 在 forward 前执行的逻辑
    }
);
```

**调用栈**:
```
Function::Apply(inputs)
  └─> for (hook : forward_pre_hooks_) { hook(this, inputs); }
```

### 2.2 Function Forward Post-Hook
在 Function 的 forward 执行后调用。

```cpp
auto handle = function->RegisterForwardPostHook(
    [](autograd::Function* func,
       const std::vector<std::shared_ptr<Tensor>>& inputs,
       const std::vector<std::shared_ptr<Tensor>>& outputs) {
        // 在 forward 后执行的逻辑
    }
);
```

**调用栈**:
```
Function::Apply(inputs)
  ├─> outputs = Forward(inputs)
  └─> for (hook : forward_post_hooks_) { hook(this, inputs, outputs); }
```

### 2.3 Function Backward Pre-Hook
在 Function 的 backward 执行前调用。

```cpp
auto handle = function->RegisterBackwardPreHook(
    [](autograd::Function* func, const std::vector<std::shared_ptr<Tensor>>& grad_outputs) {
        // 在 backward 前执行的逻辑
    }
);
```

**调用栈**:
```
Function::BackwardPartial(grad_output, idx)
  ├─> 累积 grad_outputs
  └─> 当所有依赖满足时:
        for (hook : backward_pre_hooks_) { hook(this, grad_outputs); }
```

### 2.4 Function Backward Post-Hook
在 Function 的 backward 执行后调用。

```cpp
auto handle = function->RegisterBackwardPostHook(
    [](autograd::Function* func,
       const std::vector<std::shared_ptr<Tensor>>& grad_inputs,
       const std::vector<std::shared_ptr<Tensor>>& grad_outputs) {
        // 在 backward 后执行的逻辑
    }
);
```

**调用栈**:
```
Function::BackwardPartial(grad_output, idx)
  ├─> 累积 grad_outputs
  └─> 当所有依赖满足时:
        ├─> grad_inputs = Backward(grad_outputs)
        └─> for (hook : backward_post_hooks_) { hook(this, grad_inputs, grad_outputs); }
```

### 使用场景
- 算子级别的性能分析
- 中间结果监控
- 自动微分图调试
- 梯度流分析

### 实现位置
- `infini_train/include/autograd/function_hook.h`
- `infini_train/include/autograd/function.h`
- Function forward hooks 在 `Function::Apply()` 中被调用
- Function backward hooks 在 `Function::BackwardPartial()` 中被调用

## 3. Hook 基础设施统一

为了减少代码重复，Function 和 Module 的 hook 基础设施已统一到 `infini_train/include/common/hook.h`：

```cpp
// 统一的 HookHandle 基类
class HookHandle {
public:
    virtual ~HookHandle() = default;
    virtual void Remove() = 0;
};

// 统一的 HookHandleImpl 模板
template <typename HookType>
class HookHandleImpl : public HookHandle {
    // 实现细节...
};
```

Function 和 Module 使用各自的 hook 类型定义：

```cpp
// Function hooks (在 function.h 中定义)
using FunctionPreHook = std::function<void(Function*, const std::vector<std::shared_ptr<Tensor>>&)>;
using FunctionPostHook = std::function<void(Function*, const std::vector<std::shared_ptr<Tensor>>&,
                                             const std::vector<std::shared_ptr<Tensor>>&)>;

// Module hooks (在 module_hook.h 中定义)
using ModulePreHook = std::function<void(Module*, const std::vector<std::shared_ptr<Tensor>>&)>;
using ModulePostHook = std::function<void(Module*, const std::vector<std::shared_ptr<Tensor>>&,
                                          const std::vector<std::shared_ptr<Tensor>>&)>;
```

- `FunctionPreHook` / `ModulePreHook` 用于 Forward Pre-Hook 和 Backward Pre-Hook（签名相同）
- `FunctionPostHook` / `ModulePostHook` 用于 Forward Post-Hook 和 Backward Post-Hook（签名相同）

## 4. Hook Handle 和移除机制

所有 hook 注册函数都返回 `std::shared_ptr<HookHandle>`，可用于移除 hook：

```cpp
auto handle = module->RegisterForwardPreHook(...);

// 移除 hook
handle->Remove();
```

移除后的 hook 会被设置为 `nullptr`，在执行时会被跳过，不会影响其他 hook 的执行。

## 5. 调用流程

### Forward Pass
```
Module::operator()
  ├─> Forward Pre-Hooks
  ├─> Forward()
  │     └─> Function::Apply()
  │           ├─> Function Forward Pre-Hooks
  │           ├─> Forward()
  │           └─> Function Forward Post-Hooks
  └─> Forward Post-Hooks
```

### Backward Pass
```
Tensor::Backward()
  └─> Function::BackwardPartial()
        ├─> 累积 grad_outputs (等待所有依赖)
        └─> 当所有依赖满足时:
              ├─> Backward Pre-Hooks (包括 Module backward pre-hooks)
              ├─> Backward()
              ├─> Backward Post-Hooks (包括 Module backward post-hooks)
              └─> 传播梯度到下一层
```

注：Module backward hooks 在 forward 时注册到输出 tensor 的 `grad_fn`，在反向传播时由 Function 层执行。

## 6. 示例代码

参见：
- `test/hook/test_hook.cc` - 完整的 hook 使用示例
- `infini_train/include/autograd/function_hook.h` - Hook API 定义

## 7. 注意事项

1. Hook 按注册顺序执行
2. 移除的 hook 会被设置为 nullptr，执行时会被跳过
3. **Module 调用**: 使用 `(*module)(inputs)` 而不是 `module->Forward(inputs)` 才能触发 hooks
4. **Module 子类**: 只需重写 `Forward()` 方法，hooks 会自动执行
5. **Module backward hooks**: 在 forward 时注册到输出 tensor 的 `grad_fn`，在反向传播时自动执行
6. Function hooks 在 `Function::Apply()` 和 `Function::BackwardPartial()` 中自动调用
