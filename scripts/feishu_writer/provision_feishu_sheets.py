#!/usr/bin/env python3
"""Provision Feishu spreadsheets for InfiniTrain benchmark tags.

This script creates/reuses:
  root machine folder -> tag folder -> model spreadsheet copies

It then writes the resulting spreadsheet tokens into token.json so the existing
write_to_feishu_sheet.py script can run unchanged.
"""

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from typing import Any


DEFAULT_ROOT_FOLDER_TOKEN = "MW8Nfsd3ulIRmpdYSo1c7EBsn1G"
DEFAULT_ROOT_FOLDER_URL = (
    "https://gxtctab8no8.feishu.cn/drive/folder/"
    f"{DEFAULT_ROOT_FOLDER_TOKEN}"
)
DEFAULT_LOG_GLOB = "*.log"

DEFAULT_MODEL_TEMPLATES = {
    "GPT2": {
        "template_token": "X5mJskjzSh2mo3tzuERccAYxnib",
        "doc_type": "sheet",
        "title": "GPT2",
    },
    "LLAMA3": {
        "template_token": "NtT3syaRThyGXDtQyzdcpiyfnXd",
        "doc_type": "sheet",
        "title": "LLAMA3",
    },
}

PROVISION_KEY = "FEISHU_PROVISION"
TAG_CONFIGS_KEY = "TAG_SPREADSHEET_CONFIGS"
MODEL_TOKENS_KEY = "MODEL_SPREADSHEET_TOKEN"


class CLIError(RuntimeError):
    def __init__(self, cmd: list[str], returncode: int, stdout: str, stderr: str):
        self.cmd = cmd
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        super().__init__(self._format())

    def _format(self) -> str:
        return (
            f"command failed with exit code {self.returncode}: "
            f"{' '.join(self.cmd)}\n{self.stderr or self.stdout}"
        )


class LarkCLI:
    def __init__(self, identity: str, dry_run: bool, yes: bool, verbose: bool):
        self.identity = identity
        self.dry_run = dry_run
        self.yes = yes
        self.verbose = verbose

    def run_json(
        self,
        args: list[str],
        *,
        write: bool = False,
        high_risk: bool = False,
        allow_existing_error: bool = False,
    ) -> Any | None:
        cmd = ["lark-cli", *args]
        if self.identity and "--as" not in cmd:
            cmd.extend(["--as", self.identity])
        if high_risk and self.yes and "--yes" not in cmd:
            cmd.append("--yes")

        if write and self.dry_run:
            print(f"[dry-run] {' '.join(cmd)}")
            return None

        if write and not self.yes:
            raise SystemExit(
                "Refusing to perform Feishu write operations without --yes. "
                "Run once with --dry-run to inspect the plan, then rerun with --yes."
            )

        if self.verbose:
            print(f"[cmd] {' '.join(cmd)}")

        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode != 0:
            if allow_existing_error and _looks_like_existing_error(proc.stderr):
                print("[ok] permission already exists")
                return {"already_exists": True}
            raise CLIError(cmd, proc.returncode, proc.stdout, proc.stderr)

        stdout = proc.stdout.strip()
        if not stdout:
            return {}

        try:
            return json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"lark-cli did not return JSON for {' '.join(cmd)}") from exc


def _looks_like_existing_error(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "already",
            "exist",
            "duplicate",
            "duplicated",
            "repeated",
            "重复",
            "已存在",
        )
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(path.parent),
        delete=False,
    ) as tmp:
        json.dump(data, tmp, ensure_ascii=False, indent=4)
        tmp.write("\n")
        tmp_name = tmp.name
    os.replace(tmp_name, path)


def parse_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    items = [item.strip() for item in value.split(",")]
    return [item for item in items if item]


def parse_key_value(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in values:
        if "=" not in raw:
            raise SystemExit(f"Expected KEY=VALUE, got: {raw}")
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise SystemExit(f"Expected non-empty KEY=VALUE, got: {raw}")
        result[normalize_model_key(key)] = value
    return result


def normalize_model_key(value: str) -> str:
    compact = value.strip().replace("-", "").replace("_", "")
    return compact.upper()


def extract_folder_token(value: str | None) -> str | None:
    if not value:
        return None
    marker = "/drive/folder/"
    if marker in value:
        return value.split(marker, 1)[1].split("?", 1)[0].split("/", 1)[0]
    return value.strip()


def discover_tags(test_config: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    for group in test_config.get("test_groups", []):
        tag = group.get("tag")
        if tag and tag not in tags:
            tags.append(tag)
    return tags


def model_log_prefix(model: str) -> str:
    return f"{normalize_model_key(model).lower()}_"


def has_model_logs(tag_dir: Path, model_prefixes: list[str]) -> bool:
    if not tag_dir.exists() or not tag_dir.is_dir():
        return False
    for log_file in tag_dir.glob(DEFAULT_LOG_GLOB):
        name = log_file.name.lower()
        if any(name.startswith(prefix) for prefix in model_prefixes):
            return True
    return False


def discover_log_tags(log_dir: Path, model_prefixes: list[str]) -> list[str]:
    """Discover benchmark tags from scripts/logs/<tag>/<model>_*.log."""
    if not log_dir.exists():
        return []
    tags: list[str] = []
    for child in sorted(log_dir.iterdir()):
        if not child.is_dir():
            continue
        if has_model_logs(child, model_prefixes):
            tags.append(child.name)
    return tags


def warn_missing_log_tags(tags: list[str], log_dir: Path, model_prefixes: list[str]) -> None:
    for tag in tags:
        tag_dir = log_dir / tag
        if not has_model_logs(tag_dir, model_prefixes):
            print(
                f"[warn] no local model logs found under {tag_dir}; "
                "continuing provisioning anyway"
            )


def normalize_templates(token_config: dict[str, Any]) -> dict[str, dict[str, str]]:
    provision = token_config.get(PROVISION_KEY, {})
    raw_templates = (
        provision.get("model_templates")
        or token_config.get("MODEL_TEMPLATES")
        or {}
    )
    templates: dict[str, dict[str, str]] = {
        model: dict(values) for model, values in DEFAULT_MODEL_TEMPLATES.items()
    }

    for raw_model, raw_value in raw_templates.items():
        model = normalize_model_key(raw_model)
        base = dict(templates.get(model, {}))
        if isinstance(raw_value, str):
            templates[model] = {
                "template_token": raw_value,
                "doc_type": base.get("doc_type", "sheet"),
                "title": base.get("title", model),
            }
        elif isinstance(raw_value, dict):
            template_token = (
                raw_value.get("template_token")
                or raw_value.get("token")
                or raw_value.get("spreadsheet_token")
            )
            if not template_token:
                raise SystemExit(f"Missing template token for model {raw_model}")
            templates[model] = {
                "template_token": template_token,
                "doc_type": raw_value.get("doc_type", "sheet"),
                "title": raw_value.get("title") or base.get("title") or model,
            }
        else:
            raise SystemExit(f"Unsupported template config for model {raw_model}")

    return templates


def apply_template_overrides(
    templates: dict[str, dict[str, str]],
    token_overrides: dict[str, str],
    title_overrides: dict[str, str],
) -> dict[str, dict[str, str]]:
    merged = {model: dict(values) for model, values in templates.items()}
    for model, token in token_overrides.items():
        merged.setdefault(model, {"doc_type": "sheet"})
        merged[model]["template_token"] = token
        merged[model].setdefault("title", model)
    for model, title in title_overrides.items():
        merged.setdefault(model, {"doc_type": "sheet"})
        merged[model]["title"] = title
    return merged


def collect_file_items(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key in ("files", "items"):
                maybe_items = node.get(key)
                if isinstance(maybe_items, list):
                    for item in maybe_items:
                        if (
                            isinstance(item, dict)
                            and "name" in item
                            and "token" in item
                            and "type" in item
                        ):
                            found.append(item)
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(value)

    deduped: dict[str, dict[str, Any]] = {}
    for item in found:
        deduped.setdefault(item["token"], item)
    return list(deduped.values())


def extract_created_file(value: Any) -> dict[str, Any]:
    candidates: list[Any] = []
    if isinstance(value, dict):
        candidates.extend([value.get("file"), value.get("folder")])
        data = value.get("data")
        if isinstance(data, dict):
            candidates.extend([data.get("file"), data.get("folder")])
    for item in candidates:
        if isinstance(item, dict) and item.get("token"):
            return item
    folder_token = find_key(value, "folder_token")
    if folder_token:
        return {"token": folder_token}
    token = find_key(value, "token")
    if token:
        return {"token": token}
    raise RuntimeError(f"Unable to find created file token in response: {value}")


def find_key(value: Any, key: str) -> Any | None:
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for child in value.values():
            result = find_key(child, key)
            if result is not None:
                return result
    elif isinstance(value, list):
        for child in value:
            result = find_key(child, key)
            if result is not None:
                return result
    return None


def find_title(value: Any, title: str) -> bool:
    if isinstance(value, dict):
        if value.get("title") == title:
            return True
        return any(find_title(child, title) for child in value.values())
    if isinstance(value, list):
        return any(find_title(child, title) for child in value)
    return False


class Provisioner:
    def __init__(self, cli: LarkCLI):
        self.cli = cli

    def list_folder(self, folder_token: str) -> list[dict[str, Any]]:
        response = self.cli.run_json(
            [
                "drive",
                "files",
                "list",
                "--params",
                json.dumps({"folder_token": folder_token, "page_size": 200}),
                "--page-all",
                "--page-limit",
                "0",
            ]
        )
        return collect_file_items(response)

    def find_child(
        self,
        folder_token: str,
        name: str,
        allowed_types: set[str],
    ) -> dict[str, Any] | None:
        matches = [
            item
            for item in self.list_folder(folder_token)
            if item.get("name") == name
            and str(item.get("type", "")).lower() in allowed_types
        ]
        if len(matches) > 1:
            print(
                f"[warn] found {len(matches)} existing entries named {name!r}; "
                f"using token={matches[0].get('token')}"
            )
        return matches[0] if matches else None

    def ensure_folder(self, parent_token: str, name: str) -> str:
        if parent_token.startswith("DRYRUN_"):
            existing = None
        else:
            existing = self.find_child(parent_token, name, {"folder"})
            if existing:
                print(f"[exists] folder {name}: {existing['token']}")
                return existing["token"]

        print(f"[create] folder {name}")
        response = self.cli.run_json(
            [
                "drive",
                "+create-folder",
                "--folder-token",
                parent_token,
                "--name",
                name,
            ],
            write=True,
        )
        if response is None:
            return f"DRYRUN_FOLDER_{safe_token_fragment(name)}"
        created = extract_created_file(response)
        return created["token"]

    def template_title(self, model: str, template: dict[str, str]) -> str:
        return template.get("title") or model

    def ensure_spreadsheet(
        self,
        tag_folder_token: str,
        model: str,
        template: dict[str, str],
        title: str,
    ) -> tuple[str, bool]:
        if tag_folder_token.startswith("DRYRUN_"):
            existing = None
        else:
            existing = self.find_child(tag_folder_token, title, {"sheet"})
            if existing:
                print(f"[exists] {model} spreadsheet {title}: {existing['token']}")
                return existing["token"], False

        print(f"[copy] {model} template -> {title}")
        response = self.cli.run_json(
            [
                "drive",
                "files",
                "copy",
                "--params",
                json.dumps({"file_token": template["template_token"]}),
                "--data",
                json.dumps(
                    {
                        "folder_token": tag_folder_token,
                        "name": title,
                        "type": template.get("doc_type", "sheet"),
                    }
                ),
            ],
            write=True,
            high_risk=True,
        )
        if response is None:
            return f"DRYRUN_SHEET_{safe_token_fragment(model + '_' + title)}", True
        created = extract_created_file(response)
        return created["token"], True

    def grant_permission(
        self,
        spreadsheet_token: str,
        member_type: str,
        member_id: str,
        perm: str,
        collaborator_type: str | None,
    ) -> None:
        data: dict[str, Any] = {
            "member_type": member_type,
            "member_id": member_id,
            "perm": perm,
        }
        if collaborator_type:
            data["type"] = collaborator_type

        print(f"[permission] grant {perm} to {member_type}:{member_id}")
        self.cli.run_json(
            [
                "drive",
                "permission.members",
                "create",
                "--params",
                json.dumps(
                    {
                        "token": spreadsheet_token,
                        "type": "sheet",
                        "need_notification": False,
                    }
                ),
                "--data",
                json.dumps(data),
            ],
            write=True,
            high_risk=True,
            allow_existing_error=True,
        )

    def check_template_sheet(self, spreadsheet_token: str) -> None:
        response = self.cli.run_json(
            [
                "sheets",
                "+info",
                "--spreadsheet-token",
                spreadsheet_token,
            ]
        )
        if not find_title(response, "模板"):
            raise RuntimeError(
                f"spreadsheet {spreadsheet_token} does not contain a sheet titled 模板"
            )
        print(f"[check] spreadsheet {spreadsheet_token} contains 模板")


def safe_token_fragment(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value)[:48]


def tag_config_map(token_config: dict[str, Any]) -> dict[str, dict[str, str]]:
    configs = token_config.get(TAG_CONFIGS_KEY, [])
    result: dict[str, dict[str, str]] = {}
    for item in configs:
        tag = item.get("tag")
        model_tokens = item.get(MODEL_TOKENS_KEY, {})
        if tag and isinstance(model_tokens, dict):
            result[tag] = {
                normalize_model_key(model): token
                for model, token in model_tokens.items()
            }
    legacy = token_config.get(MODEL_TOKENS_KEY)
    if legacy and "basic" not in result:
        result["basic"] = {
            normalize_model_key(model): token
            for model, token in legacy.items()
        }
    return result


def write_tag_configs(
    token_config: dict[str, Any],
    mapping: dict[str, dict[str, str]],
    ordered_tags: list[str],
) -> None:
    configs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for tag in ordered_tags:
        if tag in mapping:
            configs.append({"tag": tag, MODEL_TOKENS_KEY: mapping[tag]})
            seen.add(tag)
    for tag, model_tokens in mapping.items():
        if tag not in seen:
            configs.append({"tag": tag, MODEL_TOKENS_KEY: model_tokens})
    token_config[TAG_CONFIGS_KEY] = configs
    token_config.pop(MODEL_TOKENS_KEY, None)


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_log_dir = script_dir.parent / "logs"
    parser = argparse.ArgumentParser(
        description="Create/reuse Feishu folders and spreadsheet copies for InfiniTrain tags."
    )
    parser.add_argument("--test-config", default=str(script_dir.parent / "test_config.json"))
    parser.add_argument("--token-file", default=str(script_dir / "token.json"))
    parser.add_argument(
        "--log-dir",
        default=str(default_log_dir),
        help=(
            "Directory containing run logs. New-machine tag discovery uses "
            "subdirectories under this path. Default: scripts/logs."
        ),
    )
    parser.add_argument(
        "--output-token-file",
        help=(
            "Where to write the provisioned token JSON. Defaults to --token-file, "
            "or scripts/feishu_writer/new_token.json when --new-machine is set."
        ),
    )
    parser.add_argument(
        "--new-machine",
        action="store_true",
        help=(
            "Create a fresh per-machine token file from the seed token config. "
            "Existing tag spreadsheet tokens and machine folder state are ignored."
        ),
    )
    parser.add_argument(
        "--tags",
        help=(
            "Comma-separated tag list. Default: tags with model logs under "
            "scripts/logs for --new-machine, otherwise all test_config tags."
        ),
    )
    parser.add_argument("--models", help="Comma-separated model list. Default: all configured templates.")
    parser.add_argument("--root-folder-token")
    parser.add_argument("--root-folder-url", default=DEFAULT_ROOT_FOLDER_URL)
    parser.add_argument("--machine-folder-token")
    parser.add_argument("--machine-folder-name")
    parser.add_argument("--template-token", action="append", default=[], metavar="MODEL=TOKEN")
    parser.add_argument("--template-title", action="append", default=[], metavar="MODEL=TITLE")
    parser.add_argument("--as", dest="identity", default="user", choices=["user", "bot"])
    parser.add_argument("--permission-member-type")
    parser.add_argument("--permission-member-id")
    parser.add_argument("--permission-perm", choices=["view", "edit", "full_access"])
    parser.add_argument("--permission-collaborator-type")
    parser.add_argument("--skip-permission", action="store_true")
    parser.add_argument("--grant-existing", action="store_true")
    parser.add_argument("--skip-template-check", action="store_true")
    parser.add_argument("--no-update-token", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Allow Feishu writes and token.json updates.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    test_config_path = Path(args.test_config)
    token_file_path = Path(args.token_file)
    log_dir = Path(args.log_dir)
    output_token_file_path = Path(
        args.output_token_file
        or (script_dir / "new_token.json" if args.new_machine else token_file_path)
    )
    test_config = load_json(test_config_path)
    token_config = load_json(token_file_path)
    original_token_config = json.dumps(
        token_config,
        ensure_ascii=False,
        sort_keys=True,
    )

    if args.new_machine:
        print(
            "[new-machine] ignoring existing machine folder, tag folders, "
            "and TAG_SPREADSHEET_CONFIGS from the seed token config"
        )
        token_config[TAG_CONFIGS_KEY] = []
        token_config.pop(MODEL_TOKENS_KEY, None)
        provision = token_config.setdefault(PROVISION_KEY, {})
        provision.pop("machine_folder_token", None)
        provision.pop("machine_folder_name", None)
        provision["tag_folders"] = {}

    templates = apply_template_overrides(
        normalize_templates(token_config),
        parse_key_value(args.template_token),
        parse_key_value(args.template_title),
    )
    selected_models = [normalize_model_key(m) for m in (parse_csv(args.models) or list(templates))]
    missing_models = [model for model in selected_models if model not in templates]
    if missing_models:
        raise SystemExit(
            "Missing model template config for: "
            + ", ".join(missing_models)
            + ". Use --template-token MODEL=TOKEN."
        )
    model_prefixes = [model_log_prefix(model) for model in selected_models]

    requested_tags = parse_csv(args.tags)
    if requested_tags:
        selected_tags = requested_tags
        warn_missing_log_tags(selected_tags, log_dir, model_prefixes)
    elif args.new_machine:
        selected_tags = discover_log_tags(log_dir, model_prefixes)
        if not selected_tags:
            raise SystemExit(
                f"No test group tags with model logs found under {log_dir}. "
                "Run benchmarks first or pass --tags explicitly."
            )
        print(f"[new-machine] discovered tags from {log_dir}: {selected_tags}")
    else:
        selected_tags = discover_tags(test_config)
    if not selected_tags:
        raise SystemExit("No test_group tags found.")

    provision = token_config.setdefault(PROVISION_KEY, {})
    root_folder_token = (
        args.root_folder_token
        or provision.get("root_folder_token")
        or extract_folder_token(args.root_folder_url)
        or DEFAULT_ROOT_FOLDER_TOKEN
    )
    machine_folder_token = args.machine_folder_token or provision.get("machine_folder_token")
    machine_folder_name = (
        args.machine_folder_name
        or provision.get("machine_folder_name")
        or f"{dt.datetime.now():%Y%m} {socket.gethostname()}"
    )

    permission_cfg = provision.get("permission", {})
    permission_member_type = (
        args.permission_member_type
        or permission_cfg.get("member_type")
        or "appid"
    )
    permission_member_id = args.permission_member_id
    if not permission_member_id:
        permission_member_id = permission_cfg.get("member_id")
    if not permission_member_id and permission_member_type == "appid":
        permission_member_id = token_config.get("APP_ID")
    if not args.skip_permission and not permission_member_id:
        raise SystemExit(
            "Cannot determine permission member id. Provide --permission-member-id "
            "or APP_ID in token.json."
        )

    cli = LarkCLI(args.identity, args.dry_run, args.yes, args.verbose)
    provisioner = Provisioner(cli)

    if not machine_folder_token:
        machine_folder_token = provisioner.ensure_folder(root_folder_token, machine_folder_name)
    else:
        print(f"[exists] machine folder token: {machine_folder_token}")

    provision["root_folder_token"] = root_folder_token
    provision["machine_folder_token"] = machine_folder_token
    provision["machine_folder_name"] = machine_folder_name
    provision["model_templates"] = templates
    provision["permission"] = {
        "member_type": permission_member_type,
        "member_id": permission_member_id,
        "perm": args.permission_perm or permission_cfg.get("perm") or "edit",
    }
    if args.permission_collaborator_type:
        provision["permission"]["type"] = args.permission_collaborator_type
    tag_folders = provision.setdefault("tag_folders", {})

    model_titles = {
        model: provisioner.template_title(model, templates[model])
        for model in selected_models
    }
    for model, title in model_titles.items():
        provision["model_templates"][model]["title"] = title

    tokens_by_tag = tag_config_map(token_config)
    changed = False

    for tag in selected_tags:
        print(f"\n=== tag: {tag} ===")
        tag_folder_token = tag_folders.get(tag)
        if tag_folder_token:
            print(f"[exists] tag folder {tag}: {tag_folder_token}")
        else:
            tag_folder_token = provisioner.ensure_folder(machine_folder_token, tag)
            tag_folders[tag] = tag_folder_token
            changed = True

        model_tokens = tokens_by_tag.setdefault(tag, {})
        for model in selected_models:
            existing_token = model_tokens.get(model)
            if existing_token:
                print(f"[exists] token.json {tag}/{model}: {existing_token}")
                if args.grant_existing and not args.skip_permission:
                    provisioner.grant_permission(
                        existing_token,
                        permission_member_type,
                        permission_member_id,
                        args.permission_perm or permission_cfg.get("perm") or "edit",
                        args.permission_collaborator_type,
                    )
                if not args.skip_template_check and not existing_token.startswith("DRYRUN_"):
                    provisioner.check_template_sheet(existing_token)
                continue

            spreadsheet_token, copied = provisioner.ensure_spreadsheet(
                tag_folder_token,
                model,
                templates[model],
                model_titles[model],
            )
            model_tokens[model] = spreadsheet_token
            changed = True

            if not args.skip_permission and (copied or args.grant_existing):
                provisioner.grant_permission(
                    spreadsheet_token,
                    permission_member_type,
                    permission_member_id,
                    args.permission_perm or permission_cfg.get("perm") or "edit",
                    args.permission_collaborator_type,
                )
            if not args.skip_template_check and not spreadsheet_token.startswith("DRYRUN_"):
                provisioner.check_template_sheet(spreadsheet_token)

    write_tag_configs(token_config, tokens_by_tag, selected_tags)

    if args.no_update_token:
        print("\n[skip] token.json update disabled")
    elif args.dry_run:
        print(f"\n[dry-run] {output_token_file_path} would be written")
    elif changed or original_token_config != json.dumps(
        token_config,
        ensure_ascii=False,
        sort_keys=True,
    ) or output_token_file_path != token_file_path:
        write_json_atomic(output_token_file_path, token_config)
        print(f"\n[write] wrote {output_token_file_path}")
    else:
        print("\n[ok] token.json already up to date")

    print("\nProvisioning complete. You can run write_to_feishu_sheet.py next.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CLIError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(exc.returncode)
