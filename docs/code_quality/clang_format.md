# clang-format 检查

InfiniTrain 使用 clang-format 检查 C/C++/CUDA 代码的缩进、换行和 include 排序。项目固定使用 **clang-format 16.0.6**，配置文件为仓库根目录的 `.clang-format`。

检查模式只报告格式问题，不修改源码；自动格式化需要显式省略 `--check`。

## 检查范围

`scripts/clang_format_diff.py --all` 检查全仓受 Git 管理的 C/C++/CUDA 文件，包括头文件，并排除路径中的 `third_party` 和 `build` 目录。未跟踪文件不参与检查。

clang-format 不需要编译数据库，也不负责类型检查、头文件依赖分析或 Python 格式化。`Format Check` 工作流另外使用 Black 检查 `scripts`、`infini_train`、`example` 中的 Python 文件。

## 环境准备

Ubuntu 24.04 可以使用以下命令安装工具：

```bash
sudo apt-get update
sudo apt-get install -y clang-format-16
python3 -m pip install black
clang-format-16 --version
```

确认 clang-format 输出版本为 `16.0.6`。脚本会校验版本和配置；工具缺失、版本不符或配置无效时检查失败。

## 本地运行

以下命令在仓库根目录执行。

### 检查全部文件

```bash
python3 scripts/clang_format_diff.py --all --check --clang-format clang-format-16
```

修改 `.clang-format` 或升级工具后，应执行全量检查。

### 检查当前分支的相关修改

```bash
python3 scripts/clang_format_diff.py --check --ref origin/master --clang-format clang-format-16
```

该模式比较指定 ref 与当前 `HEAD` 的共同祖先到 `HEAD` 之间的修改，不包含未提交修改。

### 检查暂存区修改

不指定 `--all` 或 `--ref` 时，脚本从暂存区 diff 获取修改行范围，再检查工作区对应文件：

```bash
git add <files>
python3 scripts/clang_format_diff.py --check --clang-format clang-format-16
```

增量模式用于本地开发。clang-format 可能调整多行语句或 include 块中的相邻行；完整验证应使用 `--all`。

## 格式规则

`.clang-format` 基于 Google 风格，并按以下顺序对 include 分组：

1. 配套头文件；
2. C 和 POSIX 系统头文件；
3. C++ 标准库头文件；
4. 显式匹配的 CUDA、cuBLAS、CUB、NCCL 和 OpenMP 头文件（兼容尖括号和双引号）；
5. third-party 头文件；
6. InfiniTrain 公共接口头文件；
7. InfiniTrain 内部头文件，包括 `infini_train/src`、`example`、`tests` 和 `tools`。

新增依赖时应核对 include 分类，并按需更新显式匹配规则。

## 检查结果

格式不符合规则时，脚本输出文件位置和 clang-format 诊断，并返回非零状态，CI 随之失败。版本或配置错误也会导致检查失败。

检查通过只表示符合格式规则，不能替代编译和功能测试。

## 自动格式化

格式化全部受管理的项目文件：

```bash
python3 scripts/clang_format_diff.py --all --clang-format clang-format-16
```

只格式化暂存区 diff 对应的工作区代码行：

```bash
python3 scripts/clang_format_diff.py --clang-format clang-format-16
```

格式化后检查 diff、重新执行全量检查，并重新 `git add`。大批量纯格式修改应单独提交。

### 保留 blame 历史

全仓格式化提交已记录在根目录的 `.git-blame-ignore-revs` 中。每个克隆执行一次以下配置：

```bash
git config --local blame.ignoreRevsFile .git-blame-ignore-revs
```

该设置只影响本地 `git blame`，不会随 push 同步。忽略文件会随代码提交；新增批量格式化时，先创建纯格式提交，再用后续提交记录其完整哈希。若 rebase 或 squash 改写该哈希，需要同步更新记录。

## CI

`.github/workflows/format-check.yaml` 对 Pull Request 和 `master` 分支 push 启动 `Format Check`，运行：

```bash
python3 scripts/clang_format_diff.py --all --check --clang-format clang-format-16
python3 -m black --check scripts infini_train example
```

CI 执行全量检查，不自动修改文件。Black 独立报告 Python 格式问题，不能通过 clang-format 修复。
