# Repository Guidelines

## Skills

项目工作流程存放在 `.skills/` 中。如果任务属于某个 skill 的适用范围，
必须先阅读对应的 `SKILL.md`，再制定方案或执行工作。

按以下顺序开展任务：

1. 查看任务要求及相关文件、改动或诊断信息。
2. 判断并阅读适用的 skill，无需等待用户指定名称。
3. 按 skill 的说明完成实现与验证。

## Contributing

### Code Quality

- 修改代码时，遵循[代码质量 skill](.skills/infinitrain-code-quality/SKILL.md)
  和当前 `docs/code_quality/` 中的要求。报告任务完成前，执行适用的格式检查
  （clang-format 和 Black）、clang-tidy 和 IWYU 检查，修复本次引入的问题并复查。
- 验证范围应包含未暂存修改和新增文件。仅检查暂存区或使用 `--ref`，
  不能覆盖最终工作区。
- 复用与当前 worktree 匹配的工具和编译数据库，不要仅为执行检查而暂存或提交文件。
- 若 skill 不可用，直接遵循代码质量文档及对应的 CI workflow 完成检查。
- 如实报告各项检查结果，以及阻塞原因或未覆盖的范围。
  对于跳过、受阻或仍然失败的检查，不得声称通过。
- 纯代码阅读、解释和纯文档修改无需运行代码扫描。
