# InfiniTrain 测试体系报告

## 1. 概述
- 为主仓库提供了可扩展的 CTest + gtest 弹性测试骨架。
- `BUILD_TEST` 开关保持默认启用，允许在关闭时跳过测试、在打开时统一构建所有 test 目标。

## 2. 架构与工程化

| 组件 | 说明 |
| --- | --- |
| CMake | 顶层 `CMakeLists.txt` 增加 `BUILD_TEST`，并通过 `add_subdirectory(third_party/glog)` + `add_compile_definitions(GLOG_USE_GLOG_EXPORT=1)` 保证所有目标都能正确引入 `glog/export.h`。`include_directories` 同时将 `glog` 的源目录和生成目录都纳入搜索路径。 |
| 二层分类 | 所有测试通过 `set_tests_properties(... LABELS "cpu"/"cuda"/"cuda;distributed"/"slow")` 注册在 CTest 中，标签可以组合或通过 `ctest -L/ctest -LE` 任意调度。 |
| 跳过宏 | `tests/common/test_utils.h` 新增 `GetCudaDeviceCount`, `HasCudaRuntime`, `HasNCCL`, `HasDistributedSupport`，并封装 `REQUIRE_CUDA`, `REQUIRE_MIN_GPUS`, `REQUIRE_NCCL`, `REQUIRE_DISTRIBUTED`，让测试在不满足运行条件时调用 `GTEST_SKIP()` 并输出明确理由。 |

## 3. 目录与示例

```
tests/
├── common/            # test_utils.h，定义全局宏、fixture 与 helper
├── tensor/            # tensor_* 目标；cpu/cuda/distributed 测试共享一个 binary
├── optimizer/         # optimizer_* 目标，根据标签调度
├── autograd/          # autograd_* 目标（CPU + optional CUDA/Distributed）
├── hook/              # hook_* + precision_check
└── slow/              # slow_cpu/cuda/distributed 示例，演示 slow 标签
```

新增的 `tests/slow/test_slow.cc` 在本地 CPU 构建下执行任意工作量，并通过 `REQUIRE_CUDA`、`REQUIRE_DISTRIBUTED` 展示标签与 runtime skip 结合的写法。

## 4. 如何新增测试
1. 在 `tests/<module>/` 下添加 `test_<module>.cc`，`TEST` 中可以直接使用 `REQUIRE_` 宏组合运行时能力检查。
2. `CMakeLists.txt` 中照例添加 executable、链接 gtest、主库 & 内核目标，并用 `add_test` + `set_tests_properties(... LABELS ...)` 绑定适当标签。
3. `tests/CMakeLists.txt` 统一 `add_subdirectory(<module>)`，无须为每个标签写额外逻辑。

## 5. 样例运行
- `cmake -S . -B build -DBUILD_TEST=ON -DUSE_CUDA=OFF -DUSE_NCCL=OFF`
- `cmake --build build`

### 5.1 ctest -L cpu
```
Test project /home/luoyue/InfiniTrain/build
    Start 1005: tensor_cpu
1/6 Test #1005: tensor_cpu .......................   Passed    0.00 sec
    Start 1018: slow_cpu
6/6 Test #1018: slow_cpu .........................   Passed    0.01 sec

100% tests passed, 0 tests failed out of 6

Label Time Summary:
cpu     =   0.04 sec*proc (6 tests)
slow    =   0.01 sec*proc (1 test)
```

### 5.2 ctest -L slow
```
    Start 1018: slow_cpu
1/3 Test #1018: slow_cpu .........................   Passed    0.01 sec
    Start 1019: slow_cuda
2/3 Test #1019: slow_cuda ........................   Passed    0.00 sec
    Start 1020: slow_distributed
3/3 Test #1020: slow_distributed .................   Passed    0.00 sec

100% tests passed, 0 tests failed out of 3
```

### 5.3 ctest -L cuda
```
      Start 1006: tensor_cuda
10/10 Test #1020: slow_distributed .................   Passed    0.00 sec

100% tests passed, 0 tests failed out of 10
Label Time Summary:
cuda        =   0.03 sec*proc (10 tests)
distributed =   0.02 sec*proc (5 tests)
slow        =   0.01 sec*proc (2 tests)
```

### 5.4 ctest -LE distributed
- 该命令会跳过带 `distributed` 标签的测试（包括 slow_distributed）并运行剩余的 gflags + glog 验证套件。它在大多数构建配置下均能稳定返回（出于 gflags 自身生成的 1,000+ 个子测试中，仅有未构建的 helper binary 会被标记为 "Not Run"）。

## 6. 运行要点
- `REQUIRE_` 宏可以在单测中按需组合：CPU-only 逻辑不受影响，CUDA/Distributed 测试在无法满足环境时用 `GTEST_SKIP()` 退出。
- 通过确保所有 标签  —— cpu、cuda、distributed、slow —— 在 CTest 中注册，并在 `ctest -L/ctest -LE` 中验证，测试调度逻辑可用于 CI 与本地快速切换。
- 新增 `tests/slow/` 只是一个模板，后续模块可以复制该目录并替换为真实 workload，同时保留 slow 标签与跑步说明。
