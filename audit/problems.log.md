# Problems Log

| Date | Problem Name | Description | Repro Steps | Expected Result | Actual Result | Solution | Solved? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-03-10 | Functional 接口编译错误 | 在 functional.h 中使用了 std::optional 但未包含头文件 | 运行 `make -j infini_train` | 编译通过 | 报错 `error: ‘std::optional’ has not been declared` | 在 functional.h 中添加 `#include <optional>` | Yes |
