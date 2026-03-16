#!/usr/bin/env python3
"""Summarize perf metrics from InfiniTrain training logs.

Parses lines like:
  step   2/10 | train loss 1.234567 | lr 1.00e-04 | (12.34 ms | 5678 tok/s | peak used:  1234 MB | peak reserved:  2048 MB, ...)

Outputs a Markdown table comparing two directories.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from argparse import ArgumentParser
from typing import Dict, List, Optional, Tuple

LINE_RE = re.compile(
    r"step\s+(?P<step>\d+)/(?P<total>\d+)\s+\|\s+train loss\s+(?P<loss>[\d.]+)"
    r".*?\(\s*(?P<ms>[\d.]+)\s+ms\s+\|\s+(?P<tps>[\d.]+)\s+tok/s\s+\|\s+"
    r"peak used:\s+(?P<used_mb>\d+)\s+MB\s+\|\s+peak reserved:\s+(?P<reserved_mb>\d+)\s+MB"
)


@dataclass(frozen=True)
class Summary:
    steps: int
    avg_ms: float
    avg_tps: float
    peak_used_mb: int
    peak_reserved_mb: int


def iter_logs(d: Path) -> Tuple[Dict[str, Path], Dict[str, List[Path]]]:
    files: Dict[str, Path] = {}
    duplicates: Dict[str, List[Path]] = {}

    for f in d.rglob("*.log"):
        if f.name.startswith("build") or f.name.endswith("_profile.log"):
            continue

        key = f.name
        if key in files:
            duplicates.setdefault(key, [files[key]]).append(f)
            continue
        files[key] = f

    return files, duplicates


def _exit_if_duplicate_logs(base_dir: Path, duplicates: Dict[str, List[Path]]) -> None:
    if not duplicates:
        return

    print(f"Found duplicate log basenames in {base_dir.resolve()}, cannot summarize safely:")
    for name, paths in sorted(duplicates.items()):
        rels = ", ".join(str(p.relative_to(base_dir)) for p in paths)
        print(f"  {name}: {rels}")
    raise SystemExit(1)


def parse_rows(p: Path) -> List[Tuple[int, float, float, int, int]]:
    rows: List[Tuple[int, float, float, int, int]] = []
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            rows.append(
                (
                    int(m.group("step")),
                    float(m.group("ms")),
                    float(m.group("tps")),
                    int(m.group("used_mb")),
                    int(m.group("reserved_mb")),
                )
            )
    return rows


def summarize(p: Path, exclude_step1: bool = True) -> Optional[Summary]:
    rows = parse_rows(p)
    if exclude_step1:
        rows = [r for r in rows if r[0] > 1]
    if not rows:
        return None
    ms = [r[1] for r in rows]
    tps = [r[2] for r in rows]
    used = [r[3] for r in rows]
    reserved = [r[4] for r in rows]
    return Summary(
        steps=len(rows),
        avg_ms=sum(ms) / len(ms),
        avg_tps=sum(tps) / len(tps),
        peak_used_mb=max(used),
        peak_reserved_mb=max(reserved),
    )


def pct(x: float) -> str:
    return f"{x*100:.2f}%"


def main() -> int:
    ap = ArgumentParser()
    ap.add_argument("baseline", type=Path)
    ap.add_argument("flash", type=Path)
    ap.add_argument("--include-step1", action="store_true")
    args = ap.parse_args()

    base = args.baseline
    fl = args.flash
    excl = not args.include_step1

    bfiles, bdups = iter_logs(base)
    ffiles, fdups = iter_logs(fl)

    _exit_if_duplicate_logs(base, bdups)
    _exit_if_duplicate_logs(fl, fdups)

    common = sorted(set(bfiles) & set(ffiles))
    if not common:
        print("No common log files to compare")
        return 1

    print("| case | avg ms (baseline) | avg ms (flash) | speedup (ms_b/ms_f) | avg tok/s (baseline) | avg tok/s (flash) | tps gain (f/b) | peak used MB (baseline) | peak used MB (flash) | used saved | peak reserved MB (baseline) | peak reserved MB (flash) | reserved saved |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for name in common:
        sb = summarize(bfiles[name], excl)
        sf = summarize(ffiles[name], excl)
        if sb is None or sf is None:
            print(f"| {name} | - | - | - | - | - | - | - | - | - | - | - | - |")
            continue
        speedup = (sb.avg_ms / sf.avg_ms) if sf.avg_ms else float('inf')
        tps_gain = (sf.avg_tps / sb.avg_tps) if sb.avg_tps else float('inf')
        used_saved = (sb.peak_used_mb - sf.peak_used_mb) / sb.peak_used_mb if sb.peak_used_mb else 0.0
        res_saved = (sb.peak_reserved_mb - sf.peak_reserved_mb) / sb.peak_reserved_mb if sb.peak_reserved_mb else 0.0
        print(
            f"| {name} | {sb.avg_ms:.2f} | {sf.avg_ms:.2f} | {speedup:.3f} | "
            f"{sb.avg_tps:.2f} | {sf.avg_tps:.2f} | {tps_gain:.3f} | "
            f"{sb.peak_used_mb} | {sf.peak_used_mb} | {pct(used_saved)} | "
            f"{sb.peak_reserved_mb} | {sf.peak_reserved_mb} | {pct(res_saved)} |"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
