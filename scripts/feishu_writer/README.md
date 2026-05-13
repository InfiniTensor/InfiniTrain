# 飞书 Writer 使用指南

本文说明 `scripts/` 下飞书写入相关脚本的配置格式、参数和典型运行方式。

## TL;DR

### 1. 手动运行脚本

核心流程是两步：先 provisioning 出可用的 token JSON，再用这个 token JSON 写入远端表格。

默认你已经拿到可用的 `APP_ID` 和 `APP_SECRET`，并已按本文末尾“附录：Lark-cli 配置”完成本地授权。

最少保证 `scripts/feishu_writer/token.json` 中有：

```json
{
    "APP_ID": "...",
    "APP_SECRET": "..."
}
```

- **新增机器**：

```bash
cd ~/Github/InfiniTrain

python3 scripts/feishu_writer/provision_feishu_sheets.py \
  --new-machine \
  --machine-folder-name "202605 node1" \
  --yes
```

输出默认写到：

```text
scripts/feishu_writer/new_token.json
```

如果中途因为权限或网络失败，补完权限后用 `--grant-existing` 恢复：

```bash
python3 scripts/feishu_writer/provision_feishu_sheets.py \
  --new-machine \
  --machine-folder-name "202605 node1" \
  --yes \
  --grant-existing
```

写入结果：

```bash
python3 scripts/feishu_writer/write_to_feishu_sheet.py scripts/feishu_writer/new_token.json
```

- **已有机器新增 tag**：

```bash
python3 scripts/feishu_writer/provision_feishu_sheets.py \
  --tags new_tag \
  --yes
```

如果 `token.json` 里没有机器目录 token：

```bash
python3 scripts/feishu_writer/provision_feishu_sheets.py \
  --tags new_tag \
  --machine-folder-token <existing_machine_folder_token> \
  --yes
```

provisioning 成功后，spreadsheet token 会自动写入 `TAG_SPREADSHEET_CONFIGS`，后续不需要手工粘贴。

### 2. 利用 AI Agent

直接让 AI Agent 使用 `.agents/skills/infinitrain-feishu-writer/SKILL.md`。告诉它场景是“新增机器”还是“已有机器新增 tag”，以及机器目录名、目标 tag/model；`APP_SECRET` 通过本地 `token.json` 或隐藏输入提供，不要让 Agent 在回复中打印密钥。

## 文件说明

- `scripts/feishu_writer/provision_feishu_sheets.py`：创建或复用飞书云盘目录、复制模型模板表、授权文档应用，并生成可直接给 writer 使用的 token JSON。
- `scripts/feishu_writer/write_to_feishu_sheet.py`：读取本地 logs/profile reports，把 benchmark 结果写入已配置好的飞书表格。
- `scripts/feishu_writer/token.json`：本机真实运行配置，包含密钥，已被 Git 忽略。
- `scripts/feishu_writer/new_token.json`：新增机器时默认生成的新配置文件，包含密钥，也已被 Git 忽略。
- `scripts/feishu_writer/token.example.json`：安全的配置示例，不包含真实密钥。

## token.json 格式

`write_to_feishu_sheet.py` 最少需要这些字段（与原先的使用方式一致）：

```json
{
    "APP_ID": "your_feishu_app_id",
    "APP_SECRET": "your_feishu_app_secret",
    "TAG_SPREADSHEET_CONFIGS": [
        {
            "tag": "basic",
            "MODEL_SPREADSHEET_TOKEN": {
                "GPT2": "spreadsheet_token_for_gpt2_basic",
                "LLAMA3": "spreadsheet_token_for_llama3_basic"
            }
        }
    ]
}
```

`provision_feishu_sheets.py` 会额外使用并维护 `FEISHU_PROVISION`，故完整的 token.json 可能包括：
```json
{
    "APP_ID": "cli_or_writer_app_id",
    "APP_SECRET": "do_not_commit_real_secret",
    "FEISHU_PROVISION": {
        "root_folder_token": "MW8Nfsd3ulIRmpdYSo1c7EBsn1G",
        "machine_folder_token": "",
        "machine_folder_name": "202605 machine-name",
        "model_templates": {
            "GPT2": {
                "template_token": "X5mJskjzSh2mo3tzuERccAYxnib",
                "doc_type": "sheet"
            },
            "LLAMA3": {
                "template_token": "NtT3syaRThyGXDtQyzdcpiyfnXd",
                "doc_type": "sheet"
            }
        },
        "permission": {
            "member_type": "appid",
            "member_id": "",
            "perm": "edit"
        },
        "tag_folders": {}
    },
    "TAG_SPREADSHEET_CONFIGS": [...]
}
```

注意：

- 不要提交或打印 `APP_SECRET`。
- `FEISHU_PROVISION` 是 provisioning 的状态记录，writer 不依赖它。
- `TAG_SPREADSHEET_CONFIGS` 是 writer 真正消费的映射。provisioning 成功后，不需要再人工复制 spreadsheet token。
- 当 `permission.member_type` 是 `appid` 且 `permission.member_id` 为空时，provisioning 会自动使用顶层 `APP_ID` 授权。
- 模型名会归一化，`GPT2`、`gpt2`、`gpt-2` 都会映射为 `GPT2`。

## provision_feishu_sheets.py 参数

常用参数：

- `--test-config PATH`：测试配置 JSON，默认 `scripts/test_config.json`。
- `--token-file PATH`：输入的 seed token 配置，默认 `scripts/feishu_writer/token.json`。
- `--log-dir PATH`：运行日志目录。新增机器模式默认从该目录下的 tag 子目录发现已有模型日志，默认 `scripts/logs`。
- `--output-token-file PATH`：输出的 token 配置路径。
- `--new-machine`：新增机器模式。忽略输入 token 中已有的机器目录、tag 目录和 spreadsheet token，生成一份新的机器配置。未指定 `--output-token-file` 时默认写入 `scripts/feishu_writer/new_token.json`。
- `--tags tag1,tag2`：只处理指定 tag。显式传入时会检查 `scripts/logs/<tag>` 是否有当前模型的 `.log` 文件；没有会打印 warning，但仍继续创建。
- `--models GPT2,LLAMA3`：只处理指定模型。默认处理所有已配置模板。
- `--machine-folder-name "yyyymm name"`：新建机器目录名。
- `--machine-folder-token TOKEN`：复用已有机器目录，不再新建机器目录。
- `--template-token MODEL=TOKEN`：新增或覆盖某个模型的模板表 token。
- `--template-title MODEL=TITLE`：覆盖复制出来的新表标题。默认使用模型名，例如 `GPT2`、`LLAMA3`。
- `--dry-run`：只打印计划，不创建飞书资源，也不写 token JSON。
- `--yes`：允许创建飞书资源、授权、写 token JSON。

权限参数：

- `--skip-permission`：不授权文档应用。
- `--grant-existing`：对 token JSON 中已存在的 spreadsheet 也重新授权/检查。
- `--permission-member-type TYPE`：默认取配置里的值，否则为 `appid`。
- `--permission-member-id ID`：默认取配置里的值；当 member type 是 `appid` 时默认取 `APP_ID`。
- `--permission-perm view|edit|full_access`：默认取配置里的值，否则为 `edit`。

校验和调试参数：

- `--skip-template-check`：跳过“新表中是否存在 `模板` sheet”的检查。
- `--no-update-token`：创建/检查飞书资源，但不写 token JSON。
- `--as user|bot`：飞书 CLI 身份，默认 `user`。
- `--verbose`：打印实际执行的 `lark-cli` 命令。

## 场景一：新增机器

目标：在固定根目录下创建新机器目录，按 `scripts/logs/<tag>/<model>_*.log` 中已经出现过的 tag 创建子目录和模型表格副本，给文档应用授权，并输出可直接使用的 `scripts/feishu_writer/new_token.json`。

预览：

```bash
python3 scripts/feishu_writer/provision_feishu_sheets.py \
  --new-machine \
  --machine-folder-name "202605 new-machine" \
  --dry-run
```

执行：

```bash
python3 scripts/feishu_writer/provision_feishu_sheets.py \
  --new-machine \
  --machine-folder-name "202605 new-machine" \
  --yes
```

输出结果：

- 飞书云盘目录：固定根目录 -> `202605 new-machine`
- 机器目录下：`scripts/logs` 中每个有当前模型日志的 tag 一个子目录
- 每个 tag 子目录下：每个模型一个从模板复制出来的 spreadsheet
- 本地配置：`scripts/feishu_writer/new_token.json`

后续直接写入，不需要人工粘贴 token：

```bash
python3 scripts/feishu_writer/write_to_feishu_sheet.py scripts/feishu_writer/new_token.json
```

如果执行过程中因为 scope 不足或网络中断停在半截，下一次重跑建议加 `--grant-existing`，这样已复制出来但尚未授权/检查的 spreadsheet 也会被补处理：

```bash
python3 scripts/feishu_writer/provision_feishu_sheets.py \
  --new-machine \
  --machine-folder-name "202605 new-machine" \
  --yes \
  --grant-existing
```

如果希望新机器直接使用 `scripts/feishu_writer/token.json` 作为正式配置，可以指定输出路径：

```bash
python3 scripts/feishu_writer/provision_feishu_sheets.py \
  --new-machine \
  --machine-folder-name "202605 new-machine" \
  --output-token-file scripts/feishu_writer/token.json \
  --yes
```

## 场景二：已有机器新增 tag

目标：在当前机器目录下新增一个 tag 子目录，复制每个模型的模板表，授权，并把新 token 信息追加到现有 `scripts/feishu_writer/token.json`。

先确保 `scripts/test_config.json` 已经包含新的 `test_group.tag`，然后执行：

```bash
python3 scripts/feishu_writer/provision_feishu_sheets.py --tags new_tag --dry-run
python3 scripts/feishu_writer/provision_feishu_sheets.py --tags new_tag --yes
```

脚本会检查 `scripts/logs/new_tag` 下是否已有当前模型的 `.log` 文件。如果没有，会提示 warning，但不会阻止创建远端目录和表格。

如果当前 `scripts/feishu_writer/token.json` 里还没有 `FEISHU_PROVISION.machine_folder_token`，第一次需要显式传入已有机器目录 token：

```bash
python3 scripts/feishu_writer/provision_feishu_sheets.py \
  --tags new_tag \
  --machine-folder-token <existing_machine_folder_token> \
  --yes
```

这个流程会原地更新 `scripts/feishu_writer/token.json`。后续直接运行：

```bash
python3 scripts/feishu_writer/write_to_feishu_sheet.py scripts/feishu_writer/token.json
```

## 场景三：新增模型模板

如果未来新增模型类型，先准备该模型的模板 spreadsheet，再运行：

```bash
python3 scripts/feishu_writer/provision_feishu_sheets.py \
  --models GPT2,LLAMA3,NEWMODEL \
  --template-token NEWMODEL=<template_spreadsheet_token> \
  --template-title NEWMODEL="NewModel" \
  --dry-run
```

确认无误后把 `--dry-run` 换成 `--yes`。

## write_to_feishu_sheet.py 用法

```bash
python3 scripts/feishu_writer/write_to_feishu_sheet.py <token-config-json>
```

示例：

```bash
python3 scripts/feishu_writer/write_to_feishu_sheet.py scripts/feishu_writer/token.json
python3 scripts/feishu_writer/write_to_feishu_sheet.py scripts/feishu_writer/new_token.json
```

它会从这些目录发现本地数据：

- `scripts/logs/<tag>/<model>_<testcase>.log`
- `scripts/profile_logs/<tag>/<model>_<testcase>_profile_<model>.report.rank0`

对每个已配置的 tag/model spreadsheet，它会：

1. 发现本地 testcase。
2. 查询远端 spreadsheet 里已有的 sheet。
3. 如果 testcase sheet 不存在，从远端 `模板` sheet 复制一个。
4. 解析 benchmark/profile 数据并 prepend 到对应 sheet。
5. 设置样式并合并元信息列。

## 安全检查

- 先跑 `provision_feishu_sheets.py --dry-run`，确认计划后再跑 `--yes`。
- 不要提交 `scripts/feishu_writer/token.json` 或 `scripts/feishu_writer/new_token*.json`。
- 新增机器一定使用 `--new-machine`，否则会复用输入 token 中已有的 spreadsheet token。
- 已有机器新增 tag 使用 `--tags <tag>`，不要加 `--new-machine`。
- `write_to_feishu_sheet.py` 仍然依赖 `APP_ID` 和 `APP_SECRET` 直接鉴权，因此输出 token JSON 必须保留这两个字段。

## 附录：Lark-cli 配置

首次使用前，需要把 `lark-cli` 配置到对应飞书应用：

```bash
cd ~/Github/InfiniTrain

lark-cli config init \
  --app-id <APP_ID> \
  --app-secret-stdin \
  --brand feishu \
  --force-init
```

然后从 stdin 粘贴 `APP_SECRET`。不要把 `APP_SECRET` 写进命令行参数。

实测新增机器 provisioning 需要 user 身份至少授权这些 scope：

- `space:document:retrieve`：读取根目录和已有目录，避免重复创建。
- `space:folder:create`：创建机器目录和 tag 目录。
- `docs:document:copy`：从 GPT2/LLAMA3 模板复制 spreadsheet。
- `docs:permission.member:create`：把 `APP_ID` 加为新 spreadsheet 的协作者。
- `sheets:spreadsheet.meta:read`：校验新 spreadsheet 中存在 `模板` sheet。

推荐一次性授权：

```bash
lark-cli auth login --scope "space:document:retrieve space:folder:create docs:document:copy docs:permission.member:create sheets:spreadsheet.meta:read"
```

如果一次性授权不稳定，就按报错提示缺哪个补哪个：

```bash
lark-cli auth login --scope "space:document:retrieve"
lark-cli auth login --scope "space:folder:create"
lark-cli auth login --scope "docs:document:copy"
lark-cli auth login --scope "docs:permission.member:create"
lark-cli auth login --scope "sheets:spreadsheet.meta:read"
```

GPT2/LLAMA3 默认会使用模型名作为复制出来的新表标题，因此常规流程不需要传 `--template-title`。只有新增模型或需要自定义表标题时才需要显式传 `--template-title MODEL=TITLE`。
