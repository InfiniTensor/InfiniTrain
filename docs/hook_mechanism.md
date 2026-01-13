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

### 1.3 Backward Pre-Hook
在 backward 执行前调用。

```cpp
auto handle = module->RegisterBackwardPreHook(
    [](Module* mod, const std::vector<std::shared_ptr<Tensor>>& grad_outputs) {
        // 在 backward 前执行的逻辑
    }
);
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

### 使用场景
- 特征提取和可视化
- 激活值监控
- 梯度流分析
- 性能分析和 profiling

### 实现位置
- `infini_train/include/nn/module_hook.h`
- Module hooks 在 `Module::operator()` 中被调用（forward_pre_hooks_ 和 forward_post_hooks_）
- 子类只需重写 `Forward()` 方法，hooks 会自动执行

### 使用说明
- **调用方式**: 使用 `(*module)(inputs)` 而不是 `module->Forward(inputs)`
- **子类实现**: 只需重写 `Forward()` 方法，不需要手动调用 hooks
- **Hook 自动执行**: `operator()` 会自动调用 pre-hooks、Forward、post-hooks

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

### 2.3 Function Backward Pre-Hook
在 Function 的 backward 执行前调用。

```cpp
auto handle = function->RegisterBackwardPreHook(
    [](autograd::Function* func, const std::vector<std::shared_ptr<Tensor>>& grad_outputs) {
        // 在 backward 前执行的逻辑
    }
);
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

## 3. Hook 类型简化

为了减少冗余，Function hooks 使用了统一的类型定义：

```cpp
// 在 function.h 中定义
using FunctionPreHook = std::function<void(Function*, const std::vector<std::shared_ptr<Tensor>>&)>;
using FunctionPostHook = std::function<void(Function*, const std::vector<std::shared_ptr<Tensor>>&,
                                             const std::vector<std::shared_ptr<Tensor>>&)>;
```

- `FunctionPreHook` 用于 Forward Pre-Hook 和 Backward Pre-Hook（签名相同）
- `FunctionPostHook` 用于 Forward Post-Hook 和 Backward Post-Hook（签名相同）

## 4. Hook Handle 和移除机制

所有 hook 注册函数都返回 `std::shared_ptr<HookHandle>`，可用于移除 hook：

```cpp
auto handle = function->RegisterForwardPreHook(...);

// 移除 hook
handle->Remove();
```

移除后的 hook 会被设置为 `nullptr`，不会影响其他 hook 的执行。

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
Function::BackwardPartial()
  ├─> Backward Pre-Hooks
  ├─> Backward()
  └─> Backward Post-Hooks
```

## 6. 示例代码

参见：
- `test/hook/test_hook.cc` - 完整的 hook 使用示例
- `infini_train/include/autograd/function_hook.h` - Hook API 定义

## 7. 注意事项

1. Hook 按注册顺序执行
2. 移除的 hook 会被设置为 nullptr，不会影响其他 hook
3. **Module 调用**: 使用 `(*module)(inputs)` 而不是 `module->Forward(inputs)`
4. **Module 子类**: 只需重写 `Forward()` 方法，hooks 会自动执行
5. Function hooks 在 Function::Apply() 和 Function::BackwardPartial() 中自动调用
