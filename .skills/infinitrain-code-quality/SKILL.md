---
name: infinitrain-code-quality
description: 按当前项目 docs/code_quality 的要求执行格式、clang-tidy 和 IWYU 检查，修复问题并复查；用于代码修改完成后的质量验收。也用于复现这些 CI 检查或处理其失败；纯说明、代码阅读和纯文档修改无需运行代码扫描。
---

# InfiniTrain 代码质量收尾

在代码修改完成、向用户报告完成之前执行。目标是让最终工作区通过当前仓库要求，保留可核实的命令、退出状态和覆盖范围。复用仓库脚本，不维护另一套检查规则。

## 读取当前要求

从当前任务所在目录确定 Git 根目录，所有命令在该根目录执行。始终使用当前任务对应的 clone/worktree，不依赖固定的本地目录。先记录 `git status --short`，区分本次修改与用户原有改动，包括未跟踪文件。

读取当前 checkout 的以下文件，路径均相对仓库根目录：

- `docs/code_quality/clang_format.md`、`clang_tidy.md`、`iwyu.md`。
- `.github/workflows/format-check.yaml`、`clang-tidy.yaml`、`iwyu.yaml`。
- 相应配置和 `scripts/clang_format_diff.py`、`clang_tidy.py`、`iwyu.py` 的参数与选取范围；需要时用 `--help`。

文档是要求入口，脚本与 workflow 决定实际执行方式；发现不一致时说明差异，不静默忽略失败。下面是创建本 skill 时的命令示例；仓库更新后以当前实现为准。命令直接调用项目脚本和标准工具，不依赖个人环境中的命令包装器。

## 决定检查范围

- 修改 C/C++ 源码、项目头文件、CMake、子模块或相关 lint 配置/脚本时，执行适用的全量检查。公共头文件、mapping 或构建配置变化不能只用增量检查验收。
- 修改 Python 时运行格式 workflow 的 Black 检查；格式项还包括全量 clang-format。仅 Python 改动通常不触发 clang-tidy/IWYU，但修改这两个检查入口脚本时应按当前 workflow 的触发规则判断。
- 纯 `.cu` 修改需要 clang-format 和任务相关的 CUDA 编译/测试；当前 CPU-only clang-tidy/IWYU 不覆盖 CUDA 编译单元。`.cuh` 可通过 C++ 编译单元被分析，不能当作 `.cu` 一律跳过。
- 纯文档修改、仅解释或阅读代码，不启动编译工具链。用户明确要求完整质量检查时，执行全部三项。

**覆盖未提交修改：** 默认增量模式读取暂存区，`--ref` 读取共同祖先到 HEAD 的已提交修改，均不能证明 agent 刚写的未暂存代码通过。`--has-relevant-changes` 也有同样限制，不能单凭它跳过检查。综合任务修改、工作区、暂存区、未跟踪文件以及任务分支的提交判断适用性，最终验收使用 `--all`。

**覆盖新文件：** clang-format 的 `--all` 只选择 Git 已跟踪文件。对本次新增但未跟踪的 C/C++/CUDA 文件，额外用同版本 `clang-format-16 --dry-run --Werror -style=file PATH` 检查；无需为了 lint 执行 `git add`。新增源文件应按项目构建方式接入 CMake，重新生成数据库并确认实际被选中。头文件通过适当编译单元分析；无法覆盖的文件须明确列出。不能把零个编译单元或文件未入选当作验收成功。

## 准备并复用环境

优先使用任务已经采用的开发环境或容器。确认容器挂载的是当前 worktree；编译数据库里的源码、工作目录和 include 路径必须在执行环境中有效，不能拿其他 worktree 或宿主机的绝对路径数据库直接使用。

检查现有工具和缓存后，仅补齐缺失或过期部分。依赖安装、子模块初始化、插件构建和数据库生成，采用上述文档/workflow 的命令；无需让用户手工逐条准备。需要安装或网络访问时遵守执行环境的权限机制。

当前版本约束：clang-format **16.0.6**；clang-tidy 与项目插件均为 LLVM/Clang **18**；IWYU 使用仓库 CI 的 Clang **17** 配套环境。Black 也是格式验收的一部分，按当前 workflow 安装/运行。

推荐持久复用以下目录，不与训练构建缓存混用：

| 用途 | 构建目录 |
| --- | --- |
| clang-tidy CPU 编译数据库 | `build/lint-tidy` |
| IWYU CPU 编译数据库 | `build/lint-iwyu` |
| clang-tidy 自定义插件 | `build/clang-tidy-plugin` |

用文档中的 CMake 参数生成数据库，仅将 `-B build/lint` 分别替换为上表目录。当前共同参数是 `CMAKE_EXPORT_COMPILE_COMMANDS=ON`、`CMAKE_CXX_SCAN_FOR_MODULES=OFF`、`USE_CUDA=OFF`、`USE_NCCL=OFF`、`USE_OMP=OFF`、`BUILD_TEST=ON`，编译器分别遵循两份文档。不同编译器不得复用同一个 CMake 缓存。

已有缓存只有在源码根路径、编译器和配置匹配时才能复用。修改构建配置、子模块或增删源码后重新配置；插件代码变化后增量重建插件。不要仅因 `.so` 或 JSON 文件存在就认定有效。

## 检查与修复

先运行适用项的只读检查，收集独立失败，不因第一项失败而遗漏其余适用项。默认并行度沿用脚本设置，按开发环境的可用资源调整。

```bash
python3 scripts/clang_format_diff.py --all --check --clang-format clang-format-16
python3 -m black --check scripts infini_train example

python3 scripts/clang_tidy.py --all \
  --build-dir build/lint-tidy \
  --load-plugin build/clang-tidy-plugin/InfiniTrainTidy.so

python3 scripts/iwyu.py --all --build-dir build/lint-iwyu
```

保留各命令的实际退出码。需要保存完整日志时使用不会丢失子命令状态的方式（例如 Bash `pipefail` 配合 `tee`）；不要以截断输出或最后一个成功命令替代前面失败。clang-tidy 的原始和去重报告默认在 `build/lint-tidy/clang-tidy-report/`。

修复本次任务引入的问题，并保护用户已有改动：

- 格式问题：用指定版本的 `clang-format-16 -i -style=file PATH` 或 `python3 -m black PATH` 修复本次文件，检查 diff。不要默认全仓自动格式化。使用 `clang_format_diff.py` 作为检查入口；旧 `scripts/format.py` 的错误处理不能替代严格验收。
- clang-tidy：按照当前规则改源码，当前主要检查头文件中的普通成员函数实现以及枚举项 `kCamelCase` 命名。对规则豁免以文档和实现为准。脚本没有 `--fix`，不要传入不存在的参数或通过关闭规则消除诊断。
- IWYU：先审查建议，特别是宏、条件编译和第三方依赖。需要预览时用 `scripts/iwyu.py --all --fix-dry-run --build-dir build/lint-iwyu`；只应用与任务有关且已核实的修改。`--all --fix` 可能修改无关文件，只有修复范围确实属于任务时才使用，否则按建议定点编辑。
- IWYU 默认保护头文件删除；只有核实依赖后才使用 `--fix-header-removals`。不要为了消除输出随意添加 mapping、豁免或降低检查标准。自动修复退出成功不等于检查通过。

IWYU 或 clang-tidy 修复后再格式化，重新运行受影响的检查。最终代码必须有适用的全量检查成功结果；之后有新修改时，仅重跑被该修改影响的检查。涉及 include 或实现移动的修复，还要完成任务所需的编译和相关测试；CPU lint 不能替代 CUDA 构建验证。

遇到已有且与任务无关的违规，保留位置和诊断，不扩大成全仓重构，也不声称全仓已通过。工具或权限阻塞时先完成可独立执行的检查；若同一失败没有新证据或可行修复，不反复重试，明确剩余阻塞。若需证明问题早已存在，用只读历史对比或隔离 checkout，不重置用户工作区。

## 完成标准

最终简要报告三项的状态：格式（包含适用的 Black）、clang-tidy、IWYU；每项为已通过、不适用（附原因）或未通过/受阻（附原因）。给出实际执行范围、关键命令或报告路径及仍未覆盖的文件。纯文档任务可以合并说明三项不适用。

只有实际命令成功且应覆盖的文件已被检查，才能报告通过；未运行、零选中、工具缺失、已有违规和 dry-run 修复预览都不是通过。不要为检查自动暂存、提交或推送代码。
