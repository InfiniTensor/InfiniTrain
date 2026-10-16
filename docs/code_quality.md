# InfiniTrain C/C++ 代码质量检查

## clang-format

项目固定使用 **clang-format 16.0.6**。

`.clang-format` 基于 Google 风格，并按以下顺序对 include 分组：

1. 配套头文件；
2. C 和 POSIX 系统头文件；
3. C++ 标准库头文件；
4. 显式匹配的 CUDA、cuBLAS、CUB、NCCL 和 OpenMP 头文件（兼容尖括号和双引号）；
5. third-party 头文件；
6. InfiniTrain 公共接口头文件；
7. InfiniTrain 内部实现头文件。

### 检查范围

`Format Check` 工作流检查全仓受 Git 管理的 C/C++/CUDA 文件，并排除：

- `third_party/`
- `build/`

计划等当前 open PR 处理完后，再进行全仓格式化。在此之前，全仓检查可能因已有代码的格式问题失败。

工作流同时保留 Black，检查 `scripts`、`infini_train`、`example` 中的 Python 文件。

### 本地使用

运行与 CI 相同的检查（不修改文件）：

```bash
python3 scripts/clang_format_diff.py --all --check --clang-format clang-format-16
python3 -m black --check scripts infini_train example
```

后续统一格式化时可去掉 `--check`。以下增量命令仍可用于本地开发，CI 使用全仓检查。

检查暂存区中的改动：

```bash
git add <files>

python3 scripts/clang_format_diff.py \
  --check \
  --clang-format clang-format-16
```

根据暂存区中的改动范围格式化工作区文件：

```bash
python3 scripts/clang_format_diff.py \
  --clang-format clang-format-16
```

格式化完成后，需要重新执行 `git add` 将修改加入暂存区。

检查当前分支相对主分支的改动：

```bash
python3 scripts/clang_format_diff.py \
  --check \
  --ref origin/master \
  --clang-format clang-format-16
```

传入 `--ref` 时，脚本会使用指定 ref 与当前 `HEAD` 的 merge base 作为 diff 起点。

统一格式化前，尽量避免在功能 PR 中混入大面积格式修改。
