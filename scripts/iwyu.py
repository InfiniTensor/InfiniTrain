#!/usr/bin/env python3
"""Run IWYU on changed translation units or the complete project database."""

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
CPP_SOURCES = {".c", ".cc", ".cpp", ".cxx"}
CPP_HEADERS = {".h", ".hh", ".hpp", ".hxx", ".cuh"}
PROJECT_ROOTS = {"infini_train", "example", "tests", "tools"}
RELEVANT_FILES = {
    Path(".github/workflows/iwyu.yaml"),
    Path(".gitmodules"),
    Path("CMakePresets.json"),
    Path("scripts/iwyu.imp"),
    Path("scripts/iwyu.py"),
}
INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.MULTILINE)


def capture(command: list[str]) -> str:
    process = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=False
    )
    if process.returncode != 0:
        raise RuntimeError(process.stderr.strip() or "command failed")
    return process.stdout


def merge_base(ref: str) -> str:
    return capture(["git", "merge-base", ref, "HEAD"]).strip()


def changed_files(ref: str | None) -> list[Path]:
    if ref:
        arguments = ["git", "diff", f"{merge_base(ref)}..HEAD"]
    else:
        arguments = ["git", "diff", "--cached"]
    output = capture(
        [*arguments, "--no-renames", "--diff-filter=ACDM", "--name-only"]
    )
    return [Path(line) for line in output.splitlines() if line.strip()]


def is_relevant_change(path: Path) -> bool:
    if not path.parts or "build" in path.parts:
        return False
    return (
        path in RELEVANT_FILES
        or path.parts[0] == "third_party"
        or path.name == "CMakeLists.txt"
        or path.suffix.lower() == ".cmake"
        or path.suffix.lower() in CPP_SOURCES | CPP_HEADERS
    )


def project_sources(build_dir: Path) -> list[Path]:
    database = json.loads(
        (build_dir / "compile_commands.json").read_text(encoding="utf-8")
    )
    sources = set()
    for entry in database:
        source = Path(entry["file"])
        if not source.is_absolute():
            source = Path(entry.get("directory", ROOT)) / source
        source = source.resolve()
        try:
            relative = source.relative_to(ROOT)
        except ValueError:
            continue
        if (
            not relative.parts
            or relative.parts[0] not in PROJECT_ROOTS
            or "third_party" in relative.parts
            or "build" in relative.parts
            or source.suffix.lower() not in CPP_SOURCES
            or source.name.endswith("_compile_fail.cc")
        ):
            continue
        sources.add(source)
    return sorted(sources)


def changed_sources(build_dir: Path, changes: list[Path]) -> list[Path]:
    available = project_sources(build_dir)
    available_set = set(available)
    selected = {
        (ROOT / path).resolve()
        for path in changes
        if path.suffix.lower() in CPP_SOURCES
    }.intersection(available_set)
    changed_headers = {
        (ROOT / path).resolve()
        for path in changes
        if path.suffix.lower() in CPP_HEADERS
    }

    for header in sorted(changed_headers):
        try:
            header_relative = header.relative_to(ROOT).as_posix()
        except ValueError:
            continue
        direct_users = []
        for source in available:
            try:
                includes = INCLUDE_RE.findall(source.read_text(encoding="utf-8"))
            except (OSError, UnicodeError):
                continue
            if header_relative in includes or any(
                (source.parent / include).resolve() == header for include in includes
            ):
                direct_users.append(source)
        if direct_users:
            selected.add(direct_users[0])
            continue
        same_stem = [source for source in available if source.stem == header.stem]
        if same_stem:
            selected.add(same_stem[0])
    return sorted(selected)


def project_file_pattern() -> str:
    roots = "|".join(re.escape(root) for root in sorted(PROJECT_ROOTS))
    repository = re.escape(ROOT.as_posix())
    return rf"^(?:{repository}/|\./)?(?:{roots})/"


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--all", action="store_true", help="scan every project TU")
    mode.add_argument(
        "--has-relevant-changes",
        action="store_true",
        help="return 0 when the selected diff should trigger a full IWYU scan",
    )
    parser.add_argument("--ref", help="compare with the merge-base of REF")
    parser.add_argument("--build-dir", type=Path, default=Path("build/lint"))
    parser.add_argument(
        "--jobs", type=int, default=min(4, max(1, os.cpu_count() or 1))
    )
    parser.add_argument("--iwyu-tool")
    fix_mode = parser.add_mutually_exclusive_group()
    fix_mode.add_argument(
        "--fix", action="store_true", help="apply IWYU output with fix_includes.py"
    )
    fix_mode.add_argument(
        "--fix-dry-run",
        action="store_true",
        help="print the fix_includes.py diff without modifying files",
    )
    parser.add_argument(
        "--fix-header-removals",
        action="store_true",
        help="allow fix_includes.py to remove includes from header files",
    )
    parser.add_argument("--fix-tool")
    args = parser.parse_args()
    args.build_dir = args.build_dir.resolve()

    if args.has_relevant_changes and (args.fix or args.fix_dry_run):
        parser.error("--has-relevant-changes cannot be combined with a fix mode")
    if args.fix_header_removals and not (args.fix or args.fix_dry_run):
        parser.error("--fix-header-removals requires --fix or --fix-dry-run")
    if args.fix_tool and not (args.fix or args.fix_dry_run):
        parser.error("--fix-tool requires --fix or --fix-dry-run")

    if args.has_relevant_changes:
        try:
            relevant = [
                path for path in changed_files(args.ref) if is_relevant_change(path)
            ]
        except RuntimeError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        if relevant:
            print("Relevant IWYU changes:")
            for path in relevant:
                print(f"  {path}")
            return 0
        print("No relevant changes require IWYU.")
        return 1

    tool = args.iwyu_tool or shutil.which("iwyu_tool.py") or shutil.which("iwyu_tool")
    if tool is None:
        print("error: iwyu_tool is required", file=sys.stderr)
        return 2
    fixer = None
    if args.fix or args.fix_dry_run:
        fixer = (
            args.fix_tool
            or shutil.which("fix_includes.py")
            or shutil.which("fix_include")
        )
        if fixer is None:
            print(
                "error: fix_includes.py or fix_include is required for fix mode",
                file=sys.stderr,
            )
            return 2
    try:
        sources = (
            project_sources(args.build_dir)
            if args.all
            else changed_sources(args.build_dir, changed_files(args.ref))
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if not sources:
        print("No translation units require IWYU.")
        return 0

    command = [
        tool,
        "-j",
        str(args.jobs),
        "-p",
        str(args.build_dir),
        *[str(source) for source in sources],
        "--",
        "-Xiwyu",
        "--cxx17ns",
        "-Xiwyu",
        "--mapping_file=" + str(ROOT / "scripts/iwyu.imp"),
    ]
    if not (args.fix or args.fix_dry_run):
        command.extend(["-Xiwyu", "--error=1"])
    display_command = [
        tool,
        "-j",
        str(args.jobs),
        "-p",
        str(args.build_dir),
        f"<{len(sources)} translation units>",
        "--",
        *command[command.index("--") + 1 :],
    ]
    print("+", " ".join(display_command), flush=True)
    if not (args.fix or args.fix_dry_run):
        return subprocess.run(command, cwd=ROOT, check=False).returncode

    analysis = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    print(analysis.stdout, end="")
    if analysis.returncode != 0:
        print(
            "error: IWYU analysis failed; no automatic fixes were applied",
            file=sys.stderr,
        )
        return analysis.returncode

    fix_command = [
        fixer,
        "--basedir",
        str(ROOT),
        "--only_re",
        project_file_pattern(),
        "--nocomments",
        "--blank_lines",
        "--noreorder",
        "--nosafe_headers" if args.fix_header_removals else "--safe_headers",
    ]
    if args.fix_dry_run:
        fix_command.append("--dry_run")
    print("+", " ".join(fix_command), flush=True)
    fixed = subprocess.run(
        fix_command,
        cwd=ROOT,
        input=analysis.stdout,
        text=True,
        check=False,
    )
    if args.fix_dry_run:
        print("IWYU fix preview completed; no files were modified.")
    elif fixed.returncode == 0:
        print("IWYU fixes applied. Run formatting, build, and tests before committing.")
    return fixed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
