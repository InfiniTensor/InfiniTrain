# Problems Log

| Date | Problem Name | Description | Repro Steps | Expected Result | Actual Result | Solution | Solved? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-03-10 | Functional 接口编译错误 | 在 functional.h 中使用了 std::optional 但未包含头文件 | 运行 `make -j infini_train` | 编译通过 | 报错 `error: ‘std::optional’ has not been declared` | 在 functional.h 中添加 `#include <optional>` | Yes |
| 2026-03-10 | gpt2 链接失败 | 增加 Autograd 新类后静态库链接未更新到最新产物 | 运行 `make -j gpt2` | 生成 gpt2 可执行文件 | 报错 `undefined reference to vtable for infini_train::autograd::ScaledDotProductAttention` | 清理构建目录并重新执行 cmake 与 make 全量构建 | Yes |
| 2026-03-10 | flash_attention.cu 编译失败 | CUDA 实现误用了不存在的 Tensor API 名称 | 运行 `make -j gpt2` | 编译通过 | 报错 `Tensor has no member GetDType/data/mutable_data` | 改为项目实际 API：`Dtype()` 与 `DataPtr()` | Yes |
