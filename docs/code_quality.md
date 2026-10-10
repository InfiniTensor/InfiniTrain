# IWYU 检查

InfiniTrain 使用 [Include What You Use（IWYU）](https://include-what-you-use.org/) 检查 C/C++ 代码中的头文件依赖。

IWYU 基于实际编译参数分析 translation unit，主要用于发现：

* 使用符号但未显式包含对应头文件；
* 未使用或可以移除的头文件；
* 可以使用前置声明替代完整 include 的场景。

IWYU 只生成修改建议，不会自动修改源码。工具建议可能受到第三方库、宏和编译配置影响，修改 include 后应重新编译并执行相关测试。

## 检查范围

IWYU 基于 CMake 生成的 `compile_commands.json` 运行，因此实际检查范围由当前 CMake 配置决定。

当前检查使用 CPU 构建配置：

```bash
cmake -S . -B build/lint -G Ninja \
  -DCMAKE_C_COMPILER=clang-17 \
  -DCMAKE_CXX_COMPILER=clang++-17 \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_CXX_SCAN_FOR_MODULES=OFF \
  -DUSE_CUDA=OFF \
  -DUSE_NCCL=OFF \
  -DUSE_OMP=OFF \
  -DBUILD_TEST=ON
```

`scripts/iwyu.py` 从 compilation database 中选择 `infini_train`、`example`、`tests` 和 `tools` 下的 C/C++ translation units，并排除第三方代码、构建目录以及预期编译失败的测试。

CUDA translation units 当前不包含在 IWYU 检查范围内。

头文件通常通过包含它们的 translation unit 接受分析，而不是作为独立 translation unit 单独运行 IWYU。

## 环境准备

Ubuntu 24.04 可以使用以下命令安装本地检查所需工具：

```bash
sudo apt-get update
sudo apt-get install -y clang-17 iwyu ninja-build

git submodule update --init --recursive
```

然后按照上一节的 CMake 命令生成：

```text
build/lint/compile_commands.json
```

IWYU 与 Clang 版本应保持匹配。升级 Clang 或 IWYU 时，需要同步确认工具版本兼容性，并重新执行全量检查。

## 本地运行

### 检查全部 translation units

```bash
python3 scripts/iwyu.py \
  --all \
  --build-dir build/lint
```

修改 IWYU 配置、mapping 或构建配置后，建议执行全量检查。

### 检查当前分支的相关修改

```bash
python3 scripts/iwyu.py \
  --ref origin/master \
  --build-dir build/lint
```

该模式比较 `origin/master` 与当前 `HEAD` 的共同祖先到 `HEAD` 之间的修改，并选择对应 translation units 执行检查。

### 检查暂存区修改

不指定 `--all` 或 `--ref` 时，脚本检查 Git 暂存区中的相关修改：

```bash
git add <files>

python3 scripts/iwyu.py \
  --build-dir build/lint
```

增量模式用于缩短本地检查时间。头文件可能被多个 translation units 间接使用，因此需要完整验证时应使用 `--all`。

### 并行度

默认最多同时运行 4 个 IWYU 任务，可以通过 `--jobs` 调整：

```bash
python3 scripts/iwyu.py \
  --all \
  --jobs 8 \
  --build-dir build/lint
```

如系统中的 IWYU driver 不在默认搜索路径，可以使用 `--iwyu-tool` 指定 `iwyu_tool.py` 或 `iwyu_tool`：

```bash
python3 scripts/iwyu.py \
  --all \
  --iwyu-tool /path/to/iwyu_tool.py \
  --build-dir build/lint
```

## 头文件映射

`scripts/iwyu.imp` 用于补充 IWYU 的头文件映射规则。

第三方库或标准库中的符号实际定义位置可能位于内部实现头文件中。例如某个符号虽然定义在 `bits/...` 或 `Eigen/src/...` 中，但这些文件不是项目代码应该直接依赖的公共接口。

mapping 用于告诉 IWYU 应使用对应的公共头文件，而不是直接建议 include 内部实现头文件。

新增 mapping 时，应确认：

1. 被映射的源头文件确实属于内部实现接口；
2. 目标头文件是对应符号的稳定公共接口；
3. mapping 不会掩盖真实的缺失 include；
4. 修改后重新运行全量 IWYU 检查。

不应仅为了消除 IWYU 输出而增加 mapping。

## 检查结果

IWYU 会输出建议增加或删除的 include、前置声明，以及建议保留的完整 include 列表。
脚本会将 IWYU 的完整结果直接输出到控制台，便于在 CI 日志中逐项查看具体建议。为避免命令本身
占用大量日志，启动信息只显示 translation unit 数量，不展开全部源码路径。

`scripts/iwyu.py` 向 IWYU 传入：

```text
--error=1
```

因此发现 IWYU violation 时命令返回非零状态，CI 检查也会失败。

例如：

```text
foo.cc should add these lines:
#include <memory>

foo.cc should remove these lines:
- #include <vector>

The full include-list for foo.cc:
#include <memory>
```

应根据实际依赖逐项检查 IWYU 建议，而不是直接批量应用。特别是第三方库、宏展开和条件编译相关代码，需要确认建议在项目构建配置下仍然正确。

修改完成后重新运行 IWYU，并执行相应的编译和测试。

## 自动修复

自动修复使用 IWYU 官方提供的 `fix_includes.py`。Ubuntu 24.04 的 `iwyu` 软件包将该命令安装为
`fix_include`，脚本会自动查找这两个名称；也可以通过 `--fix-tool` 指定路径。
项目不自行解析和改写 IWYU 建议，以免重复维护官方修复逻辑。

先预览全仓修复结果，不修改文件：

```bash
python3 scripts/iwyu.py \
  --all \
  --fix-dry-run \
  --build-dir build/lint
```

一键应用修复：

```bash
python3 scripts/iwyu.py \
  --all \
  --fix \
  --build-dir build/lint
```

修复器只允许修改 `infini_train`、`example`、`tests` 和 `tools` 下的文件，不修改 `third_party`。
默认使用 `--safe_headers`，因此可以向头文件添加缺失依赖，但不会自动从头文件删除 include 或前置声明；
源文件仍可正常增加和删除 include。默认也不添加 IWYU 的 “why” 注释、不重新排列现有 include，
并使用官方修复器的 `--blank_lines` 保留基本 include 分组。修复后仍需运行项目指定版本的 clang-format，
以应用项目完整的 include 分类规则。

确认建议可靠后，可以显式允许删除头文件中的 include：

```bash
python3 scripts/iwyu.py \
  --all \
  --fix \
  --fix-header-removals \
  --build-dir build/lint
```

增量模式也支持 `--fix` 和 `--fix-dry-run`，选择编译单元的方式与普通增量检查相同。
`--fix-dry-run` 的退出状态由系统安装的官方修复器决定；判断具体改动应以命令输出的 diff 为准。
部分版本的官方修复器在 dry-run 末尾仍使用 “edited files” 表述，脚本会额外明确提示没有修改文件。

自动修复不等于建议一定正确。IWYU 官方将工具标记为实验性质，宏、条件编译、第三方类型和间接依赖仍需要人工复核。
应用修复后，应运行 clang-format、重新生成或复用当前编译数据库再次执行 IWYU，并完成相关编译和测试。
CI 只检查，不会调用自动修复模式。

## CI

workflow 对所有 Pull Request 和 `master` 分支 push 启动。checkout 后先运行：

```bash
python3 scripts/iwyu.py \
  --has-relevant-changes \
  --ref <base-commit>
```

源文件、头文件、CMake 配置、IWYU 脚本或 mapping、workflow、`.gitmodules` 或 `third_party` 子模块发生变化时，
脚本返回 0，CI 才初始化子模块、安装 IWYU、生成 compilation database 并执行全量检查。
只有文档等无关文件发生变化时，脚本返回 1，后续耗时步骤跳过，IWYU job 仍以成功状态结束。
判断过程异常时返回 2，CI 失败。

CI 使用与本地相同的 compilation database 配置，并通过：

```bash
python3 scripts/iwyu.py \
  --all \
  --build-dir build/lint
```

执行全量检查。

CI 使用全量模式，避免仅根据文件变更关系筛选 translation unit 时遗漏头文件的间接使用场景。
CI 不上传报告 artifact，完整检查结果保留在 CI 日志中。全量检查的超时时间为 15 分钟。
