# Clang-Tidy

InfiniTrain 使用 clang-tidy 对 C++ 代码进行静态检查，并通过自定义 clang-tidy 插件实现项目特定的代码规范检查。

clang-tidy 的配置位于仓库根目录的 `.clang-tidy` 文件中。项目自定义检查规则位于 `tools/clang_tidy/`，`scripts/clang_tidy.py` 则作为统一入口，用于本地开发和 CI 环境中的 clang-tidy 执行。

当前 clang-tidy 及自定义插件统一使用 LLVM/Clang 18。由于 clang-tidy 插件依赖 LLVM/Clang ABI，构建插件时应使用与运行 clang-tidy 相同主版本的 LLVM/Clang。

## 检查规则

当前仓库启用了以下规则：

```text
infinitrain-no-member-definitions-in-headers
readability-identifier-naming
```

### 头文件成员函数实现

该规则用于检查项目头文件中的普通 C++ 成员函数实现。

对于不需要在头文件中提供实现的普通成员函数，应仅在头文件中保留声明，并将函数实现放到对应的源文件中，以减少头文件暴露的实现细节，并降低不必要的编译依赖。

以下情况不会触发该规则：

* 函数模板；
* 类模板中的成员函数；
* `constexpr` 或 `consteval` 函数；
* `= default` 函数；
* `= delete` 函数；
* 编译器隐式生成的成员函数；
* Lambda；
* 由宏展开产生的函数定义。

通过以下配置可以忽略空函数体：

```yaml
CheckOptions:
  infinitrain-no-member-definitions-in-headers.IgnoreEmptyBodies: true
```

系统头文件、第三方代码以及 InfiniTrain 项目目录之外的头文件不会参与该规则检查。

### 标识符命名

`readability-identifier-naming` 是 clang-tidy 内置规则，本项目首期使用它演示如何优先复用现成规则，避免为已有能力编写
自定义检查。当前配置为：

```yaml
CheckOptions:
  readability-identifier-naming.FunctionCase: CamelCase
  readability-identifier-naming.MethodCase: CamelCase
  readability-identifier-naming.VariableCase: lower_case
```

自由函数和成员函数使用 `CamelCase`，变量使用 `lower_case`。命名规则只报告问题，不自动修改代码，避免跨声明和调用位置的
批量重命名影响代码语义或对外接口。

## CI 检查

clang-tidy CI 会在 Pull Request 以及向 `master` 分支推送代码时运行。

在安装 clang-tidy 和构建自定义插件之前，CI 会先判断当前提交是否包含与静态检查相关的修改，包括：

* C/C++ 源文件或项目头文件；
* CMake 配置文件；
* `.gitmodules` 或 submodule commit 引用；
* `.clang-tidy`；
* `scripts/clang_tidy.py`；
* `tools/clang_tidy/` 下的自定义插件代码；
* clang-tidy CI workflow。

如果不存在相关修改，则直接跳过 submodule 拉取、工具安装、插件构建和 clang-tidy 检查。

如果修改了 `.gitmodules` 或 submodule commit 引用，也会触发后续检查。拉取 submodule 时，CI 仅对本次命令将
`git@github.com:` 地址转换为 `https://github.com/`，因此公开 GitHub submodule 不依赖 SSH 密钥；如果引用的 commit
尚未推送、URL 无效或私有仓库缺少访问凭据，CI 仍会失败。

如果存在相关修改，CI 会生成 CPU-only 的 CMake compilation database，并对过滤后的所有 C/C++ translation unit 执行 clang-tidy。

需要注意：

**变更检测仅用于决定是否执行 clang-tidy，并不会将检查范围限制在本次修改的代码行。**

CI 的全量扫描范围包括：

```text
infini_train/
example/
tests/
tools/
```

以下目录或文件不会参与检查：

```text
third_party/
build/
tests/**/*_compile_fail.cc
```

当前 clang-tidy 使用 CPU-only compilation database，不包含 CUDA translation unit，因此纯 `.cu` 修改不会被判定为
clang-tidy relevant change，也不会执行后续扫描。CUDA clang-tidy 将在后续具备相应 toolchain 和 compilation database 后单独接入。

对于被 C++ translation unit 引用的 `.cuh` 等头文件，如果进入 clang-tidy 的 AST 分析范围，仍可能产生相关诊断。

项目头文件的 include/exclude 范围由 `.clang-tidy` 中的 `HeaderFilterRegex` 统一维护。自定义 check 只判断定义是否来自
被 translation unit 包含的头文件以及 AST 语义，不再维护目录或扩展名列表。`scripts/clang_tidy.py` 中的目录集合仅用于筛选
需要执行的 translation unit。

workflow 保持对所有 Pull Request 启动，以便无相关改动时也能返回明确的成功状态，避免 required check 因 workflow 被路径过滤
而保持 pending。Python 变更检测作为统一过滤入口；只有发现相关改动后才递归拉取 submodule 并执行后续检查。

## 检查报告

同一个头文件可能被多个 translation unit 引用，因此 clang-tidy 原始输出中，同一个头文件诊断可能重复出现多次。

`scripts/clang_tidy.py` 会同时生成完整报告和去重后的报告：

```text
build/lint/clang-tidy-report/clang-tidy-raw.txt
build/lint/clang-tidy-report/clang-tidy-unique.txt
```

其中：

* `clang-tidy-raw.txt` 保存 clang-tidy 的完整原始输出；
* `clang-tidy-unique.txt` 按诊断位置和规则进行去重，便于查看实际问题。

报告去重仅影响输出展示，不会改变 clang-tidy 的退出状态。

## 本地运行

以下命令可以在 Ubuntu 环境中复现 CI 使用的 clang-tidy 配置。

首先安装 LLVM/Clang 18 和 Ninja：

```bash
sudo apt-get install -y \
  clang-18 \
  clang-tidy-18 \
  libclang-18-dev \
  llvm-18-dev \
  ninja-build
```

### 构建自定义 clang-tidy 插件

```bash
cmake -S tools/clang_tidy -B build/clang-tidy-plugin -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang-18 \
  -DCMAKE_CXX_COMPILER=clang++-18 \
  -DLLVM_DIR=/usr/lib/llvm-18/lib/cmake/llvm \
  -DClang_DIR=/usr/lib/llvm-18/lib/cmake/clang

cmake --build build/clang-tidy-plugin
```

构建完成后会生成 clang-tidy 插件：

```text
build/clang-tidy-plugin/InfiniTrainTidy.so
```

### 生成 compilation database

```bash
cmake -S . -B build/lint -G Ninja \
  -DCMAKE_CXX_COMPILER=clang++-18 \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_CXX_SCAN_FOR_MODULES=OFF \
  -DUSE_CUDA=OFF \
  -DUSE_NCCL=OFF \
  -DUSE_OMP=OFF \
  -DBUILD_TEST=ON
```

生成的 compilation database 位于：

```text
build/lint/compile_commands.json
```

### 执行完整检查

执行与 CI 相同的全量扫描：

```bash
python3 scripts/clang_tidy.py \
  --all \
  --build-dir build/lint \
  --load-plugin build/clang-tidy-plugin/InfiniTrainTidy.so
```

该模式会对过滤后的所有 C/C++ translation unit 运行 clang-tidy。

### 检查当前分支修改

本地开发过程中，也可以仅关注当前分支相对于指定 Git revision 修改代码行产生的诊断：

```bash
python3 scripts/clang_tidy.py \
  --ref origin/master \
  --build-dir build/lint \
  --load-plugin build/clang-tidy-plugin/InfiniTrainTidy.so
```

该模式通过 `clang-tidy-diff` 对最终诊断结果进行行级过滤。

需要注意，clang-tidy 仍然会解析相关 translation unit 的完整 AST，只是最终仅报告修改行对应的诊断。

## 添加自定义检查规则

新的 InfiniTrain clang-tidy 检查规则应放置在：

```text
tools/clang_tidy/
```

并在：

```text
tools/clang_tidy/InfiniTrainTidyModule.cpp
```

中完成注册。

新增或修改自定义检查规则时，应先通过测试用例验证规则行为是否符合预期，包括应命中的问题、不应命中的合法代码以及需要豁免的场景。验证通过后，再执行全仓扫描，评估存量代码的命中情况及处理方式，最后在 .clang-tidy 中正式启用该规则。
