#!/usr/bin/env python3
# Usage:
# python tools/compare_tps.py \
#   /path/to/logs/dir1 \
#   /path/to/logs/dir2 \
#   --threshold 0.20

import re
import sys
from pathlib import Path
from argparse import ArgumentParser

def parse_log(file_path):
    """Extract step -> tok/s mapping from log file."""
    pattern = re.compile(r'step\s+(\d+)/\d+.*?\|\s+(\d+)\s+tok/s')
    tps_values = {}
    with open(file_path) as f:
        for line in f:
            match = pattern.search(line)
            if match:
                tps_values[int(match.group(1))] = float(match.group(2))
    return tps_values

def compare_files(file1, file2, threshold):
    """Compare tok/s values from two log files, excluding first step."""
    tps1 = parse_log(file1)
    tps2 = parse_log(file2)

    # Remove step 1
    tps1 = {k: v for k, v in tps1.items() if k > 1}
    tps2 = {k: v for k, v in tps2.items() if k > 1}

    if not tps1 or not tps2:
        return 0, 1, ["  No valid steps found (after excluding step 1)"], 0, 0, 0, []

    # Calculate averages
    avg1 = sum(tps1.values()) / len(tps1)
    avg2 = sum(tps2.values()) / len(tps2)

    # Calculate relative error
    rel_error = abs(avg1 - avg2) / max(avg1, avg2) if max(avg1, avg2) > 0 else 0
    diff = avg2 - avg1
    diff_pct = (diff / avg1) * 100 if avg1 > 0 else 0

    # Find steps exceeding threshold (absolute difference > threshold)
    step_mismatches = []
    for step in sorted(set(tps1.keys()) & set(tps2.keys())):
        v1, v2 = tps1[step], tps2[step]
        step_diff = v2 - v1
        step_diff_pct = (step_diff / v1) * 100 if v1 > 0 else 0
        if abs(step_diff_pct) >= threshold * 100:
            step_mismatches.append({
                'step': step,
                'v1': v1,
                'v2': v2,
                'diff': step_diff,
                'diff_pct': step_diff_pct
            })

    mismatches = []
    # Only fail when dir2 is slower (negative diff exceeds threshold)
    is_slow = diff < 0 and abs(diff_pct) >= threshold * 100

    if is_slow:
        sign = '+' if diff >= 0 else ''
        mismatches.append(f"  Average tok/s: {avg1:.2f} vs {avg2:.2f} ✗ SLOW ({sign}{diff:.2f} tok/s, {sign}{diff_pct:.2f}%, threshold: {threshold*100:.2f}%)")
        mismatches.append(f"  Steps compared: {len(tps1)} vs {len(tps2)} (excluding step 1)")
    elif diff > 0:
        mismatches.append(f"  Average tok/s: {avg1:.2f} vs {avg2:.2f} ✓ FAST (+{diff:.2f} tok/s, +{diff_pct:.2f}%, threshold: {threshold*100:.2f}%)")
    else:
        mismatches.append(f"  Average tok/s: {avg1:.2f} vs {avg2:.2f} ✓ (diff: {diff_pct:.2f}%, threshold: {threshold*100:.2f}%)")

    # Include step-level details if any exceed threshold
    if step_mismatches:
        mismatches.append(f"  Steps exceeding threshold ({threshold*100:.0f}%):")
        for m in step_mismatches:
            sign = '+' if m['diff'] >= 0 else ''
            mismatches.append(f"    Step {m['step']:3d}: {m['v1']:7.2f} vs {m['v2']:7.2f}  ({sign}{m['diff']:8.2f} tok/s, {sign}{m['diff_pct']:6.2f}%)")

    # Return is_slow as failure indicator (1 if slow, 0 otherwise)
    return 1, (1 if is_slow else 0), mismatches, avg1, avg2, rel_error, step_mismatches

def main():
    parser = ArgumentParser(description='Compare tok/s between two log directories')
    parser.add_argument('dir1', type=Path, help='First log directory')
    parser.add_argument('dir2', type=Path, help='Second log directory')
    parser.add_argument('--threshold', type=float, default=0.20, help='Relative error threshold (default: 0.20 = 20%%)')
    parser.add_argument('--verbose', action='store_true', help='Print detailed output for all files, including passed ones')
    args = parser.parse_args()

    files1 = {f.name: f for f in args.dir1.glob('*.log') if not f.name.startswith('build')}
    files2 = {f.name: f for f in args.dir2.glob('*.log') if not f.name.startswith('build')}

    only_in_1 = set(files1.keys()) - set(files2.keys())
    only_in_2 = set(files2.keys()) - set(files1.keys())
    common = set(files1.keys()) & set(files2.keys())

    if only_in_1:
        print(f"Files only in {args.dir1.resolve()}: {', '.join(sorted(only_in_1))}")
    if only_in_2:
        print(f"Files only in {args.dir2.resolve()}: {', '.join(sorted(only_in_2))}")
    if only_in_1 or only_in_2:
        print()

    total_mismatches = 0
    total_files = 0
    passed_files = 0

    for name in sorted(common):
        total_files += 1
        total_comparisons, num_failures, mismatches, avg1, avg2, rel_error, step_mismatches = compare_files(files1[name], files2[name], args.threshold)

        # Print always (to show step details that exceed threshold)
        print(f"Comparing {name}:")
        for msg in mismatches:
            print(msg)

        if num_failures > 0:
            total_mismatches += num_failures
        else:
            passed_files += 1

        # Print separator when verbose mode or always (to show step details)
        print()

    print("=" * 50)
    print(f"Overall Summary:")
    print(f"  {passed_files}/{total_files} test cases passed (threshold: {args.threshold*100:.0f}%)")
    print("=" * 50)

    sys.exit(1 if total_mismatches > 0 else 0)

if __name__ == '__main__':
    main()
