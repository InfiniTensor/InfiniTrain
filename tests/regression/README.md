# 训练回归测试使用指南

默认流程是：选择套件 → 增量构建 → CTest 和模型训练 → 与上次成功运行的原始日志比较 → 保存本次记录。普通使用不需要导出完整张量或安装 PyTorch；参考数据直接来自历史运行目录。

## 一键运行

在仓库根目录执行：

```bash
./scripts/run_all_tests.sh                         # 默认 PR 套件
./scripts/run_all_tests.sh --suite smoke            # 四个小模型冒烟
./scripts/run_all_models.sh                        # nightly：经典模型和组合覆盖
./scripts/run_all_tests.sh --suite perf             # 基础/DDP 算子 profiling，PROFILE_MODE=ON
./scripts/run_all_tests.sh --suite release          # 普通构建：nightly + 长程训练
```

| 套件 | 默认日志模式的内容 | CUDA 训练任务数 |
|---|---|---:|
| smoke | 四个小模型，4 步训练 | 4 |
| pr | 小模型的梯度累积、并行、ZeRO、LoRA、学习率、checkpoint 及重计算配置 | 87 |
| nightly | PR 配置和 212 个经典任务（含重计算） | 299 |
| perf | 基础/DDP × FP32/BF16 的 kernel profiling | 14 |
| release | nightly 及四模型 200 步训练 | 303 |

数量包含 checkpoint 必需的保存运行，实际值以 `--list` 为准。PR 最多需要四张 GPU，nightly/release 最多八张，perf 最多两张。CPU 支持合法 FP32 子集，perf 的 CPU 子集是四个单设备测例；不支持的 BF16、分布式组合会明确列在排除项中。原有 CTest 继续运行，`perf` 不运行 CTest。

默认不会生成单卡等价性参考、运行 PyTorch oracle 或保存梯度/参数快照。历史场景名字中的 `tiny_oracle` 在默认模式下也是普通训练任务。

## 日志怎么对齐

第一次没有可用历史时，训练成功的用例标为 `RECORDED`，保存日志，普通运行返回 0；这表示“已记录”，不表示已经完成回归比较。

后续自动选择同 suite/backend 的最近一次完整成功运行。筛选运行可使用同筛选范围或全套历史；全套运行不会拿一次局部运行充当完整参考。失败或未结束的运行不会成为参考。选择结果在本次运行开始时固定，报告会注明具体参考目录。

按 case ID 匹配后检查训练参数、数据/权重哈希和步数。checkpoint 的来源及恢复步数属于配置语义；临时输出目录变化不影响匹配。

| 指标 | 默认判定规则 |
|---|---|
| FP32 loss | 逐步绝对误差不超过 `1e-5` |
| BF16 loss | 逐步绝对误差不超过 `1e-2` |
| 吞吐 | 每次运行去掉首步，再取日志中 `tok/s` 的算术平均；降速超过 20% 失败 |
| 完整性 | 缺步骤、空日志、NaN/Inf、异常退出均失败 |

性能比较要求硬件、后端、构建及软件环境信息兼容。速度提升不算失败，loss 错误也不能被速度提升抵消。不同配置或不可比环境显示 `NOT_COMPARABLE`，而非 PASS。新增 case 显示 `RECORDED`。

```bash
# 明确选择某次运行作为参考（两个参数名等价）
./scripts/run_all_tests.sh --reference-run artifacts/test-runs/OLD_RUN_ID
./scripts/run_all_tests.sh --compare-to artifacts/test-runs/OLD_RUN_ID

# 使用独立的历史目录
./scripts/run_all_tests.sh --history-root /test-history

# 调整原有比较阈值
./scripts/run_all_tests.sh --loss-atol-fp32 1e-5 --loss-atol-bf16 1e-2 \
  --performance-threshold 0.20

# 有意改变配置或输入后，显式重新记录历史
./scripts/run_all_tests.sh --record-history
```

`--strict` 要求完整套件、包含单元测试，且每个用例都实际完成对齐；首次记录和新增 case 不能满足该门禁。`--record-history` 不能与 `--strict` 或显式参考目录组合。

## 每次保存什么

默认目录是 `artifacts/test-runs/<UTC时间>-<唯一标识>/`，可用 `--output` 指定。已存在的运行目录不会被覆盖。

```text
<run-id>/
  run.json                  # 时间、commit、环境及逐用例命令索引
  metadata.json             # 源码/构建来源、环境、参考运行
  resolved_plan.json        # 本次完整配置和参数模板
  fixtures.json             # 实际输入文件及 SHA256
  summary.md                # 首先查看这里：结果和失败原因
  summary.json              # 机器可读结果
  junit.xml
  coverage.json
  cases/<case-id>/run-1/
    train.log               # 原始 stdout/stderr，历史比较的数据来源
    metrics.normalized.json # 从日志解析的指标，便于查看
  cases/<case-id>/result.json
```

每个 `train.log` 都记录：

- UTC 开始/结束时间、退出码；
- run ID、case ID；
- 当前 checkout 的完整 commit、dirty 标记和源码差异哈希；
- 可读的实际命令、无损的 JSON 参数数组及工作目录。

若存在构建清单，汇总还保存二进制对应的构建来源；尤其使用 `--no-build` 时，应区分当前源码和已编译二进制。每个用例的 `reproduce` 是从仓库根目录执行的复现命令，会保留 backend、实际输入、参数覆盖和参考设置。

`RECORDED` 与 `PASS` 在结果及覆盖报告中分开统计。原始日志和清单必须一起保留；可以整体移动运行目录，比较器会在选定目录内重新定位日志。

## 筛选和重跑

```bash
./scripts/run_all_tests.sh --list
./scripts/run_all_tests.sh --dry-run --explain
./scripts/run_all_tests.sh --model llama3 --tag zero2
./scripts/run_all_tests.sh --case 'tiny_checkpoint/gpt2/*'
./scripts/run_all_tests.sh --no-build
./scripts/run_all_tests.sh --backend cpu
./scripts/run_all_tests.sh --rerun-failed artifacts/test-runs/RUN_ID/summary.json
./scripts/run_all_tests.sh --changed origin/main --dry-run --explain
```

重复 `--model`/`--case` 按并集选择，重复 `--tag` 要求同时满足。checkpoint 的生产者自动加入依赖；取消不必要的数值参考不会破坏保存/加载顺序。

`--changed` 保留固定锚点及受影响用例，公共或未知路径回退完整 PR。筛选、参数覆盖和 `--skip-unit` 属于部分验证，不能与严格全套门禁混用。

`--set learning_rate=0.0001` 使用 JSON 语法覆盖普通训练参数。输入文件改用 `--fixture ID=PATH`，以便校验实际内容。路径、batch 和拓扑的结构性修改应通过场景配置完成。

## 本机配置

小模型资产由标准库确定性生成，并校验 SHA256。经典模型需要本地权重和数据；根目录布局见 [经典资产清单](fixtures/classic.json)。

仓库根目录可放置不提交 Git 的 `tests.local.json`：

```json
{
  "backend": "cuda",
  "docker_container": "infinitrain-dev",
  "data_root": "/datasets/llmc",
  "checkpoint_root": "/fast-scratch/infinitrain-regression",
  "history_root": "/test-history",
  "gpus": "0,1,2,3,4,5,6,7"
}
```

不用容器时省略 `docker_container`。还支持 `build_dir`、`fixtures` 和 `reference_run`。显式命令行路径优先；经典数据也可通过 `INFINITRAIN_TEST_DATA_ROOT` 指定。跨宿主机/容器的路径应在执行环境中可访问。

普通构建默认位于 `.cache/regression/build-<backend>`，perf 使用独立的 `build-<backend>-profile`。本机配置中的 build_dir 也会为 perf 自动追加 `-profile`，显式 `--build-dir` 则使用指定目录。`--no-build` 不会更新旧二进制；perf 要求构建清单和缓存都证明 PROFILE_MODE=ON，且二进制哈希匹配。普通套件拒绝 PROFILE_MODE=ON 的构建。`--jobs` 控制并发，GPU 按用例整体预留，perf 在本 runner 内串行并独占预留设备。

## perf：kernel profiling

perf 用于评估框架 dispatcher 中的算子调用性能。默认使用经典模型权重，运行一次、10 步，micro batch=1、sequence length=64、梯度累积=1。

| 模型 | 单卡基础版本 | DDP 版本 | 精度 |
|---|---|---|---|
| GPT-2、Llama3、Qwen3 | 各 2 项 | 各 2 项，DP=2、线程启动 | FP32 / BF16 |
| Mixtral | 2 项 | 本套件暂未纳入；分布式入口已由 master 支持 | FP32 / BF16 |

DDP 保持每张卡的 micro batch 和算子 shape，与基础版本相同；global batch 随 DP 数增加。TP/PP/VPP 均为1，SP、ZeRO、LoRA、重计算关闭。Qwen3 不再用 TP8 替代基础版本，单卡和 DDP 都需要每卡容纳完整模型及训练状态，显存不足时不会偷偷换成 TP 或小模型。

```bash
./scripts/run_all_tests.sh --suite perf --list
./scripts/run_all_tests.sh --suite perf --model llama3
./scripts/run_all_tests.sh --suite perf --tag basic
./scripts/run_all_tests.sh --suite perf --tag ddp
```

每个测例的 `cases/<case-id>/run-1/` 保存：

```text
train.log
<模型>.report.rankN        # 按 step/tag、算子汇总次数、Host/Device 耗时及占比
<模型>.records.log.rankN   # 每次算子调用记录
profiles.json             # 文件索引、SHA256、各设备/算子的次数和总/平均耗时
```

原始 profiler 文件保留所有 tag 和步骤；`profiles.json` 的算子汇总排除首个训练步及初始化记录，其余9步按算子、设备分别汇总。缺少任意设备的 report/records、缺训练步或无有效调用会使测例失败，不能仅因训练退出码为0就通过。

现有 profiler 的 rank 表示设备索引；当前 DDP 测例使用同进程多线程，不把它冒充跨节点全局 rank。算子记录包含同步开销，Host 时间可能包含设备等待；一个 dispatcher 算子也可能包含多个底层 CUDA kernel。记录中的 Peak Mem 来自预留内存池峰值，不代表单个算子的独占显存。

perf 仍保留同配置历史日志的 loss/tok/s 比较，PROFILE_MODE 属于环境兼容检查；不能把普通构建吞吐与 profiling 吞吐直接比较。当前未设置单算子跨版本耗时门禁，算子报告用于分析和后续数据汇总。首次无历史仍为 RECORDED，算子产物完整性始终检查。

release 的303项使用普通构建；完整发布检查可分别运行 release 和 perf。未来接入 FlashAttention 后，在 kernel 场景中增加有代表性的开关配置即可，无需展开所有并行组合。后续同步导出仍是独立工作，本次没有实现飞书写入。

大型 checkpoint 应使用容量充足的本地临时盘。空间预检考虑 DP 副本、优化器状态和保留数量；每次运行使用带归属标记的独立子目录。成功训练的 checkpoint 默认清理，失败现场保留；`--keep-checkpoints` 可保留成功产物。日志和报告始终保留。

## 配置如何扩展

- `models/`：模型入口、默认参数、能力和资产；
- `scenarios/`：测试参数、有限矩阵、标签及所属套件；
- `suites/`：套件策略；
- `fixtures/`：数据与权重清单；
- `tests.local.json`：本机路径和设备。

普通新增组合只增加场景 JSON。同接口新模型只增加模型清单。合法两两组合、高风险锚点、batch 单位和完整示例见 [配置规范](schema/README.md)。

当前 ZeRO-1/2 要求 DP > 1，ZeRO-3 未实现；CPU 训练只支持 FP32。非法配置在启动前拒绝，不会隐式改成另一种测试。

## 重计算测例

重计算使用 `activation_recompute`、`recompute_granularity`、`recompute_method` 和
`recompute_num_layers` 四个参数；当前覆盖 `full` 的 `uniform`/`block`，不支持 `selective`。

| 场景 | 覆盖 | 不开重计算的对应配置 |
|---|---|---|
| `classic_recompute` | master 的 GPT-2/Llama3 各 14 个任务 | `classic_basic` 的 2/3/6/7/8 及 `classic_zero` 的 3/8 distopt，同精度 |
| `classic_recompute_qwen3` | Qwen3 现有 7 个经典任务各增加 uniform2/block2，共 14 个 | `classic_qwen3` 中去掉 `_recompute_uniform2`/`_recompute_block2` 后缀的同名任务 |
| `tiny_recompute` | GPT-2/Llama3/Qwen3 × FP32/BF16 × uniform2/block2，共 12 个 | `tiny_accumulation` 的同模型、同精度任务 |

Qwen3 的经典配置包含 TP8、TP8+SP、PP8、TP8+LoRA、DP2+TP2+PP2+VPP2+SP。
重计算与对应关闭版本的数据、权重、batch、步数、精度、学习率及并行配置一致；
配置自测会比较全部有效训练参数，确保差异只限于四个重计算选项。
新增经典测例采用显式模板保存与现有命令的一对一关系，未来基准配置变化会由自测提醒同步。

```bash
# 日常开发：12 个小模型重计算任务
./scripts/run_all_tests.sh --suite pr --tag recompute
# 查看全部 42 个经典重计算任务
./scripts/run_all_tests.sh --suite nightly --tag classic --tag recompute --list
# 只跑 Qwen3 的 14 个经典重计算任务
./scripts/run_all_tests.sh --suite nightly --tag classic --tag recompute --model qwen3
# 旧入口也支持新增组，包括 Qwen3
./scripts/run_models_and_profile.bash --only-run recompute
# 小模型开启/关闭重计算的同次直接数值对齐（额外运行关闭版本）
./scripts/run_all_tests.sh --suite pr --tag recompute --alignment deep
```

默认模式仍把开启重计算的测例与上次相同配置的历史日志比较，不把关闭重计算的吞吐作为性能门槛。
`deep` 的小模型参考只关闭重计算，保持 batch 和拓扑不变，比较 loss、梯度和参数。
新的测例首次没有历史会显示 `RECORDED`；这不表示完成了历史回归比较。

## 保留的专项能力

张量等价性、独立 PyTorch oracle、学习率公式等检查可通过显式模式启用：

```bash
./scripts/run_all_tests.sh --alignment deep --suite pr
```

该模式仍与历史运行日志比较 loss 和吞吐，同时执行场景声明的深度检查。它会增加本次运行中的参考训练和快照开销，包含 oracle 的环境需要 PyTorch。历史参考仍使用 `--reference-run` 或自动选择的运行目录。详细数值契约见 [测试资产说明](fixtures/README.md) 和 [容差策略](TOLERANCES.md)。

## 旧脚本和 CI

旧入口 `scripts/run_models_and_profile.bash` 运行 212 个经典任务：原有 155 个、master 新增的 28 个 GPT-2/Llama3 重计算任务、14 个 Qwen3 重计算任务，以及新迁入的7个 Qwen3 多进程和8个 Mixtral EP/EDP 任务。`--only-run basic,zero` 仍是分组并集，`--only-run recompute` 包含三个模型的经典重计算任务。`scripts/test_config.json` 保留为 master 迁移对照，Qwen3 重计算等新增用例使用新场景目录。旧入口不再自动清空 build 或重复 profiling。

本次变更提供本地运行入口，CI 接入将另行提供。计划在 CI 中运行 Python 自测、纯 CPU C++ 构建、CTest 和四模型 smoke。没有稳定历史的托管 CPU 环境可使用 `--record-history`，只记录本次结果，不宣称完成精度或性能对齐。

后续 GPU CI 计划使用自托管 runner，并配置模型数据路径和独立于 checkout 的持久历史目录。接入时可通过 `--history-root` 指定历史目录，首次使用 `--record-history` 建立记录，后续使用 `--strict` 检查完整套件的历史对齐结果。

本地执行测试框架自测：

```bash
python3 -m unittest discover -s tests/regression -p 'test_*.py' -v
```

这些自测验证配置、日志解析、历史选择、失败传播和执行流程；真实训练验收应查看对应运行的 `summary.json`。
