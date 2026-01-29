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
        return 0, 1, ["  No valid steps found (after excluding step 1)"], 0, 0, 0

    # Calculate averages
    avg1 = sum(tps1.values()) / len(tps1)
    avg2 = sum(tps2.values()) / len(tps2)

    # Calculate relative error
    rel_error = abs(avg1 - avg2) / max(avg1, avg2) if max(avg1, avg2) > 0 else 0

    mismatches = []
    if rel_error > threshold:
        mismatches.append(f"  Average tok/s: {avg1:.2f} vs {avg2:.2f} ✗ (error: {rel_error*100:.2f}%, threshold: {threshold*100:.2f}%)")
        mismatches.append(f"  Steps compared: {len(tps1)} vs {len(tps2)} (excluding step 1)")

    return 1, len(mismatches), mismatches, avg1, avg2, rel_error

def main():
    parser = ArgumentParser(description='Compare tok/s between two log directories')
    parser.add_argument('dir1', type=Path, help='First log directory')
    parser.add_argument('dir2', type=Path, help='Second log directory')
    parser.add_argument('--threshold', type=float, default=0.20, help='Relative error threshold (default: 0.20 = 20%%)')
    parser.add_argument('--verbose', action='store_true', help='Print detailed output for all files, including passed ones')
    args = parser.parse_args()

    files1 = {f.name: f for f in args.dir1.glob('*.log')}
    files2 = {f.name: f for f in args.dir2.glob('*.log')}

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
        total_comparisons, num_mismatches, mismatches, avg1, avg2, rel_error = compare_files(files1[name], files2[name], args.threshold)

        if mismatches:
            print(f"Comparing {name}:")
            for msg in mismatches:
                print(msg)
            total_mismatches += num_mismatches
        else:
            passed_files += 1
            # Only print details when verbose mode is enabled
            if args.verbose:
                print(f"Comparing {name}:")
                print(f"  Average tok/s: {avg1:.2f} vs {avg2:.2f} ✓ (error: {rel_error*100:.2f}%, threshold: {args.threshold*100:.2f}%)")
                print(f"  Steps compared: {len([k for k in parse_log(files1[name]) if k > 1])} (excluding step 1)")

        # Print separator when there are mismatches or verbose mode
        if mismatches or args.verbose:
            print()

    print("=" * 50)
    print(f"Overall Summary:")
    print(f"  {passed_files}/{total_files} test cases passed (threshold: {args.threshold*100:.0f}%)")
    print("=" * 50)

    sys.exit(1 if total_mismatches > 0 else 0)

if __name__ == '__main__':
    main()
