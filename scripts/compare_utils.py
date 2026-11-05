from pathlib import Path
import sys


def collect_log_files(base_dir: Path):
    """Collect comparable training logs keyed by basename."""
    files = {}
    duplicates = {}

    for path in base_dir.rglob("*.log"):
        if path.name == "run_metadata.log":
            continue
        if path.name.startswith(("build", "ctest_")) or path.name.endswith(
            "_profile.log"
        ):
            continue

        key = path.name
        if key in files:
            duplicates.setdefault(key, [files[key]]).append(path)
            continue
        files[key] = path

    return files, duplicates


def exit_if_duplicate_logs(base_dir: Path, duplicates):
    """Abort when duplicate basenames make comparison ambiguous."""
    if not duplicates:
        return

    print(
        f"Found duplicate log basenames in {base_dir.resolve()}, cannot compare safely:"
    )
    for name, paths in sorted(duplicates.items()):
        print(f"  {name}: {', '.join(str(p.relative_to(base_dir)) for p in paths)}")
    sys.exit(1)


def matching_logs(dir1, dir2):
    """Fail closed on missing directories, empty selection, duplicates or case sets."""
    for directory in (dir1, dir2):
        if not directory.is_dir():
            raise ValueError(f"Not a log directory: {directory}")
    left, duplicates_left = collect_log_files(dir1)
    right, duplicates_right = collect_log_files(dir2)
    exit_if_duplicate_logs(dir1, duplicates_left)
    exit_if_duplicate_logs(dir2, duplicates_right)
    if not left or not right:
        raise ValueError("No comparable training logs")
    if left.keys() != right.keys():
        raise ValueError(
            f"Case sets differ: missing in candidate={sorted(left.keys() - right.keys())}; "
            f"missing in baseline={sorted(right.keys() - left.keys())}"
        )
    return left, right


# Standalone historical CLI scripts use the same validated parser as the runner.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "regression"))
from checks import MetricError, load_metrics, compare_metrics  # noqa: E402


def load_legacy_metrics(path, *, dtype=None, required=("loss",)):
    """Recover explicit resume bounds for retained historical test_config cases.

    Unknown files remain subject to the strict 1..declared-total contract.
    Known checkpoint starts come from configuration, never a truncated log.
    """
    import json
    import re

    configured = Path(__file__).with_name("test_config.json")
    expected = None
    if configured.exists():
        config = json.loads(configured.read_text())
        for group in config.get("test_groups", []):
            for case in group.get("tests", []):
                if Path(path).stem in (
                    case["id"],
                    *(
                        model + "_" + case["id"]
                        for model in ("gpt2", "llama3", "qwen3", "mixtral")
                    ),
                ):
                    args = case.get("args", {})
                    checkpoint = re.search(
                        r"(?:checkpoint_step_|iter_)(\d+)", str(args.get("load", ""))
                    )
                    if checkpoint and "num_iteration" in args:
                        expected = range(
                            int(checkpoint[1]) + 1, int(args["num_iteration"]) + 1
                        )
    return load_metrics(path, expected_steps=expected, dtype=dtype, required=required)
