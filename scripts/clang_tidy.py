#!/usr/bin/env python3
"""Run the repository clang-tidy configuration on changed code or all TUs."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx"}
RELEVANT_SUFFIXES = SOURCE_SUFFIXES | {
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".cuh",
    ".cmake",
}
TRANSLATION_UNIT_ROOTS = {"infini_train", "example", "tests", "tools"}
RELEVANT_FILES = {
    ".gitmodules",
    ".clang-tidy",
    ".github/workflows/clang-tidy.yaml",
    "scripts/clang_tidy.py",
}
DIAGNOSTIC_RE = re.compile(
    r"^(.*):(\d+):(\d+):\s+(warning|error|fatal error):\s+"
    r"(.*?)(?:\s+\[([^\]]+)\])?\s*$"
)


def capture(command: list[str]) -> str:
    process = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=False
    )
    if process.returncode != 0:
        raise RuntimeError(process.stderr.strip() or "command failed")
    return process.stdout


def execute(command: list[str], *, input_text: str | None = None) -> int:
    print("+", " ".join(str(part) for part in command), flush=True)
    return subprocess.run(
        command, cwd=ROOT, input=input_text, text=True, check=False
    ).returncode


def execute_captured(command: list[str]) -> tuple[int, str]:
    print("+", " ".join(str(part) for part in command), flush=True)
    process = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return process.returncode, process.stdout


def find_tool(explicit: str | None, candidates: list[str]) -> str | None:
    if explicit:
        return explicit
    return next((path for name in candidates if (path := shutil.which(name))), None)


def merge_base(ref: str) -> str:
    return capture(["git", "merge-base", ref, "HEAD"]).strip()


def diff_text(ref: str | None) -> str:
    scope = [
        "--",
        ".",
        ":(exclude)**/third_party/**",
        ":(exclude)third_party/**",
        ":(exclude)build/**",
        ":(exclude)**/*_compile_fail.cc",
    ]
    if ref:
        return capture(
            ["git", "diff", "--unified=0", f"{merge_base(ref)}..HEAD", *scope]
        )
    return capture(["git", "diff", "--cached", "--unified=0", *scope])


def changed_paths(ref: str | None) -> list[Path]:
    scope = [
        "--",
        ".",
        ":(exclude)**/third_party/**",
        ":(exclude)third_party/**",
        ":(exclude)build/**",
    ]
    command = ["git", "diff", "--name-only", "--diff-filter=ACMR"]
    if ref:
        command.append(f"{merge_base(ref)}..HEAD")
    else:
        command.append("--cached")
    return [Path(line) for line in capture([*command, *scope]).splitlines()]


def changed_gitlinks(ref: str | None) -> list[Path]:
    command = ["git", "diff", "--raw", "--no-abbrev", "--diff-filter=ACDMR"]
    if ref:
        command.append(f"{merge_base(ref)}..HEAD")
    else:
        command.append("--cached")
    command.extend(["--", "."])

    paths = []
    for line in capture(command).splitlines():
        fields = line.split("\t")
        metadata = fields[0].split()
        if len(fields) < 2 or len(metadata) < 2:
            continue
        old_mode = metadata[0].removeprefix(":")
        new_mode = metadata[1]
        if "160000" in {old_mode, new_mode}:
            paths.extend(Path(path) for path in fields[1:])
    return paths


def relevant_change(path: Path) -> bool:
    normalized = path.as_posix()
    return (
        normalized in RELEVANT_FILES
        or path.name == "CMakeLists.txt"
        or path.suffix.lower() in RELEVANT_SUFFIXES
    )


def has_relevant_changes(ref: str | None) -> bool:
    relevant = [path for path in changed_paths(ref) if relevant_change(path)]
    relevant.extend(changed_gitlinks(ref))
    relevant = list(dict.fromkeys(relevant))
    if not relevant:
        print("No clang-tidy-relevant changes found.")
        return False
    print("clang-tidy-relevant changes:")
    for path in relevant:
        print(f"  {path.as_posix()}")
    return True


def translation_unit_entry(entry: dict[str, object]) -> bool:
    source = Path(str(entry["file"]))
    if not source.is_absolute():
        source = Path(str(entry.get("directory", ROOT))) / source
    source = source.resolve()
    try:
        relative = source.relative_to(ROOT)
    except ValueError:
        return False
    return (
        bool(relative.parts)
        and relative.parts[0] in TRANSLATION_UNIT_ROOTS
        and "third_party" not in relative.parts
        and "build" not in relative.parts
        and source.suffix.lower() in SOURCE_SUFFIXES
        and not source.name.endswith("_compile_fail.cc")
    )


def filtered_database(build_dir: Path) -> Path:
    source_path = build_dir / "compile_commands.json"
    database = json.loads(source_path.read_text(encoding="utf-8"))
    filtered = [entry for entry in database if translation_unit_entry(entry)]
    output_dir = build_dir / "clang-tidy-full"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "compile_commands.json").write_text(
        json.dumps(filtered, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Full scan database: {len(filtered)} compile commands in {output_dir}",
        flush=True,
    )
    return output_dir


def plugin_arguments(plugin: Path) -> list[str]:
    return ["-load", str(plugin)]


def display_path(path_text: str) -> str:
    path = Path(path_text)
    try:
        if path.is_absolute():
            path = path.resolve().relative_to(ROOT)
    except (OSError, ValueError):
        pass
    return path.as_posix()


def unique_diagnostics(output: str) -> tuple[list[str], int]:
    unique: dict[tuple[str, str, str, str, str, str], int] = {}
    for line in output.splitlines():
        match = DIAGNOSTIC_RE.match(line)
        if match is None:
            continue
        path, line_number, column, severity, message, check_name = match.groups()
        key = (
            display_path(path),
            line_number,
            column,
            severity,
            message,
            check_name or "",
        )
        unique[key] = unique.get(key, 0) + 1

    lines = []
    ordered = sorted(
        unique.items(),
        key=lambda item: (
            item[0][0],
            int(item[0][1]),
            int(item[0][2]),
            item[0][3:],
        ),
    )
    for key, count in ordered:
        path, line_number, column, severity, message, check_name = key
        suffix = f" [{check_name}]" if check_name else ""
        repeated = f" (reported {count} times)" if count > 1 else ""
        lines.append(
            f"{path}:{line_number}:{column}: {severity}: {message}{suffix}{repeated}"
        )
    return lines, sum(unique.values())


def write_full_scan_reports(output: str, report_dir: Path) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    raw_path = report_dir / "clang-tidy-raw.txt"
    unique_path = report_dir / "clang-tidy-unique.txt"
    raw_path.write_text(output, encoding="utf-8")

    diagnostics, total = unique_diagnostics(output)
    duplicate_count = total - len(diagnostics)
    summary = (
        f"Unique diagnostics: {len(diagnostics)}; raw diagnostics: {total}; "
        f"duplicates removed: {duplicate_count}."
    )
    unique_path.write_text(
        "\n".join([summary, *diagnostics, ""]), encoding="utf-8"
    )
    print(summary)
    for diagnostic in diagnostics:
        print(diagnostic)
    print(f"Raw log: {raw_path}")
    print(f"Deduplicated report: {unique_path}")
    return raw_path, unique_path


def run_changed(args: argparse.Namespace, clang_tidy: str) -> int:
    tidy_diff = find_tool(
        args.clang_tidy_diff,
        [
            "clang-tidy-diff-18.py",
            "clang-tidy-diff.py",
            "clang-tidy-diff-18",
            "clang-tidy-diff",
        ],
    )
    if tidy_diff is None:
        print("error: clang-tidy-diff is required", file=sys.stderr)
        return 2
    command = []
    if Path(tidy_diff).suffix.lower() == ".py":
        command = [sys.executable, "-W", "ignore::SyntaxWarning"]
    command += [
        tidy_diff,
        "-p1",
        "-path",
        str(args.build_dir),
        "-clang-tidy-binary",
        clang_tidy,
        "-config-file",
        str(ROOT / ".clang-tidy"),
        "-j",
        str(args.jobs),
        "-quiet",
    ]
    command += plugin_arguments(args.load_plugin)
    return execute(command, input_text=diff_text(args.ref))


def run_all(args: argparse.Namespace) -> int:
    runner = find_tool(
        args.run_clang_tidy, ["run-clang-tidy-18", "run-clang-tidy.py", "run-clang-tidy"]
    )
    if runner is None:
        print("error: run-clang-tidy is required for --all", file=sys.stderr)
        return 2
    command = []
    if Path(runner).suffix.lower() == ".py":
        command = [sys.executable, "-W", "ignore::SyntaxWarning"]
    command += [
        runner,
        "-p",
        str(filtered_database(args.build_dir)),
        "-config-file",
        str(ROOT / ".clang-tidy"),
        "-j",
        str(args.jobs),
        "-quiet",
    ]
    command += plugin_arguments(args.load_plugin)
    return_code, output = execute_captured(command)
    diagnostics, _ = unique_diagnostics(output)
    write_full_scan_reports(output, args.report_dir)
    if return_code != 0 and not diagnostics:
        print(output, file=sys.stderr, end="" if output.endswith("\n") else "\n")
    return return_code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all",
        action="store_true",
        help="scan every project TU in the filtered compile database",
    )
    parser.add_argument(
        "--has-relevant-changes",
        action="store_true",
        help="report whether the diff contains changes that require clang-tidy",
    )
    parser.add_argument(
        "--ref",
        help="comparison ref for changed-line mode or --has-relevant-changes",
    )
    parser.add_argument("--build-dir", type=Path, default=Path("build/lint"))
    parser.add_argument(
        "--report-dir",
        type=Path,
        help="full-scan report directory (default: BUILD_DIR/clang-tidy-report)",
    )
    parser.add_argument(
        "--jobs", type=int, default=min(4, max(1, os.cpu_count() or 1))
    )
    parser.add_argument("--clang-tidy")
    parser.add_argument("--clang-tidy-diff")
    parser.add_argument("--run-clang-tidy")
    parser.add_argument(
        "--load-plugin",
        type=Path,
        help="path to the InfiniTrain clang-tidy plugin",
    )
    args = parser.parse_args()

    if args.all and args.ref:
        parser.error("--ref cannot be used with --all; --all scans every project TU")

    try:
        if args.has_relevant_changes:
            return 0 if has_relevant_changes(args.ref) else 1
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if args.load_plugin is None:
        print("error: --load-plugin is required when running clang-tidy", file=sys.stderr)
        return 2

    args.build_dir = args.build_dir.resolve()
    args.load_plugin = args.load_plugin.resolve()
    args.report_dir = (
        args.report_dir.resolve()
        if args.report_dir is not None
        else args.build_dir / "clang-tidy-report"
    )

    if not (args.build_dir / "compile_commands.json").exists():
        print(
            f"error: {args.build_dir / 'compile_commands.json'} does not exist",
            file=sys.stderr,
        )
        return 2
    if not args.load_plugin.is_file():
        print(f"error: {args.load_plugin} does not exist", file=sys.stderr)
        return 2
    clang_tidy = find_tool(args.clang_tidy, ["clang-tidy-18", "clang-tidy"])
    if clang_tidy is None:
        print("error: clang-tidy is required", file=sys.stderr)
        return 2
    try:
        return run_all(args) if args.all else run_changed(args, clang_tidy)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
