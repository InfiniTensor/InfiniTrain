from pathlib import Path
import sys


def collect_log_files(base_dir: Path):
    """Collect comparable training logs keyed by basename."""
    files = {}
    duplicates = {}

    for path in base_dir.rglob("*.log"):
        if path.name.startswith("build") or path.name.endswith("_profile.log"):
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

    print(f"Found duplicate log basenames in {base_dir.resolve()}, cannot compare safely:")
    for name, paths in sorted(duplicates.items()):
        print(f"  {name}: {', '.join(str(p.relative_to(base_dir)) for p in paths)}")
    sys.exit(1)
