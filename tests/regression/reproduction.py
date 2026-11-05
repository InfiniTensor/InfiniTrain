"""Portable, shell-quoted commands for reproducing one resolved training case.

Commands are intended to be run from the repository root. Project-owned paths
are relative so the same command works in the host checkout and its container
mount; external data paths retain their actual absolute location.
"""

from pathlib import Path
import shlex


def command_for_case(args, case, assets, repo, gpu_ids):
    """Return one concrete training command with effective configuration intact.

    Selection filters/output directories are deliberately replaced: an exact
    case is selected, its dependencies are planned normally, and a fresh run
    directory is allocated. No shell variables or shell-evaluated fragments
    are interpolated into the command.
    """
    repo = Path(repo).expanduser().resolve()

    def portable_path(value):
        path = Path(value).expanduser()
        # Effective arguments are normally absolute. Relative arguments retain
        # their documented interpretation relative to the repository root.
        if not path.is_absolute():
            path = repo / path
        path = path.resolve()
        try:
            relative = str(path.relative_to(repo)) or "."
            return "./" + relative if relative.startswith("-") else relative
        except ValueError:
            return str(path)

    backend = getattr(args, "backend", None)
    if not backend:
        raise ValueError("Reproduction requires the effective backend")
    if not case.get("id"):
        raise ValueError("Reproduction requires an exact case ID")
    argv = [
        "./scripts/run_all_tests.sh",
        "--suite",
        getattr(args, "suite", "pr"),
        "--case=" + case["id"],
        "--backend",
        backend,
        "--build-dir",
        portable_path(
            getattr(args, "build_dir", None)
            or repo / ".cache" / "regression" / ("build-" + backend)
        ),
        "--config-root",
        portable_path(
            getattr(args, "config_root", None) or repo / "tests" / "regression"
        ),
        "--jobs",
        "1",
        "--no-build",
        "--skip-unit",
    ]
    if gpu_ids:
        argv += ["--gpus", ",".join(str(gpu) for gpu in gpu_ids)]
    paths = assets.paths
    for fixture in sorted(set(case.get("fixtures", []))):
        if fixture not in paths:
            raise ValueError("Reproduction has no resolved path for fixture " + fixture)
        argv += ["--fixture=" + fixture + "=" + portable_path(paths[fixture])]
    for override in getattr(args, "set", []) or []:
        argv += ["--set=" + override]
    for attribute, option in (
        ("reference_run", "--reference-run"),
        ("history_root", "--history-root"),
        ("checkpoint_root", "--checkpoint-root"),
    ):
        value = getattr(args, attribute, None)
        if value is not None:
            argv += [option, portable_path(value)]
    for attribute, option in (
        ("perf_environment", "--perf-environment"),
        ("docker_container", "--docker-container"),
        ("alignment", "--alignment"),
    ):
        value = getattr(args, attribute, None)
        if value:
            argv += [option + "=" + str(value)]
    if getattr(args, "record_history", False):
        argv += ["--record-history"]
    for attribute, option in (
        ("loss_atol_fp32", "--loss-atol-fp32"),
        ("loss_atol_bf16", "--loss-atol-bf16"),
        ("performance_threshold", "--performance-threshold"),
    ):
        value = getattr(args, attribute, None)
        if value is not None:
            argv += [option + "=" + str(value)]
    return " ".join(shlex.quote(str(value)) for value in argv)
