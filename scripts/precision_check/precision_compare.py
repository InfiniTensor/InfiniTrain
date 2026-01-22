#!/usr/bin/env python3
"""
Precision comparison tool for InfiniTrain tensor outputs.

Usage:
    python precision_compare.py --dir1 ./run1 --dir2 ./run2 [--atol 1e-5] [--rtol 1e-3]

Compares .npy files between two directories and reports differences.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def find_npy_files(directory: str) -> dict[str, Path]:
    """Find all .npy files in directory (recursively)."""
    files = {}
    for path in Path(directory).rglob("*.npy"):
        rel_path = path.relative_to(directory)
        files[str(rel_path)] = path
    return files


def compare_tensors(file1: Path, file2: Path, atol: float, rtol: float) -> dict:
    """Compare two tensor files and return comparison results."""
    arr1 = np.load(file1)
    arr2 = np.load(file2)

    result = {
        "file": str(file1.name),
        "shape1": arr1.shape,
        "shape2": arr2.shape,
        "dtype1": str(arr1.dtype),
        "dtype2": str(arr2.dtype),
        "match": False,
        "error": None,
    }

    if arr1.shape != arr2.shape:
        result["error"] = f"Shape mismatch: {arr1.shape} vs {arr2.shape}"
        return result

    if arr1.dtype != arr2.dtype:
        result["error"] = f"Dtype mismatch: {arr1.dtype} vs {arr2.dtype}"
        return result

    arr1_flat = arr1.astype(np.float64).flatten()
    arr2_flat = arr2.astype(np.float64).flatten()

    abs_diff = np.abs(arr1_flat - arr2_flat)
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)

    with np.errstate(divide="ignore", invalid="ignore"):
        rel_diff = abs_diff / (np.abs(arr2_flat) + 1e-12)
        rel_diff = np.where(np.isfinite(rel_diff), rel_diff, 0)
    max_rel_diff = np.max(rel_diff)
    mean_rel_diff = np.mean(rel_diff)

    result["max_abs_diff"] = float(max_abs_diff)
    result["mean_abs_diff"] = float(mean_abs_diff)
    result["max_rel_diff"] = float(max_rel_diff)
    result["mean_rel_diff"] = float(mean_rel_diff)
    result["match"] = np.allclose(arr1, arr2, atol=atol, rtol=rtol)

    return result


def main():
    parser = argparse.ArgumentParser(description="Compare precision check outputs")
    parser.add_argument("--dir1", required=True, help="First directory")
    parser.add_argument("--dir2", required=True, help="Second directory")
    parser.add_argument("--atol", type=float, default=1e-5, help="Absolute tolerance")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if not os.path.isdir(args.dir1):
        print(f"Error: {args.dir1} is not a directory")
        sys.exit(1)
    if not os.path.isdir(args.dir2):
        print(f"Error: {args.dir2} is not a directory")
        sys.exit(1)

    files1 = find_npy_files(args.dir1)
    files2 = find_npy_files(args.dir2)

    print(f"Directory 1: {args.dir1} ({len(files1)} files)")
    print(f"Directory 2: {args.dir2} ({len(files2)} files)")
    print(f"Tolerance: atol={args.atol}, rtol={args.rtol}")
    print()

    only_in_1 = set(files1.keys()) - set(files2.keys())
    only_in_2 = set(files2.keys()) - set(files1.keys())
    common = set(files1.keys()) & set(files2.keys())

    if only_in_1:
        print(f"Files only in dir1 ({len(only_in_1)}):")
        for f in sorted(only_in_1):
            print(f"  {f}")
        print()

    if only_in_2:
        print(f"Files only in dir2 ({len(only_in_2)}):")
        for f in sorted(only_in_2):
            print(f"  {f}")
        print()

    if not common:
        print("No common files to compare")
        sys.exit(1)

    print(f"Comparing {len(common)} common files...")
    print()

    passed = 0
    failed = 0
    errors = 0

    for rel_path in sorted(common):
        result = compare_tensors(files1[rel_path], files2[rel_path], args.atol, args.rtol)

        if result["error"]:
            errors += 1
            print(f"ERROR: {rel_path}")
            print(f"  {result['error']}")
        elif result["match"]:
            passed += 1
            if args.verbose:
                print(f"PASS: {rel_path}")
                print(f"  max_abs={result['max_abs_diff']:.2e} max_rel={result['max_rel_diff']:.2e}")
        else:
            failed += 1
            print(f"FAIL: {rel_path}")
            print(f"  shape={result['shape1']} dtype={result['dtype1']}")
            print(f"  max_abs={result['max_abs_diff']:.2e} mean_abs={result['mean_abs_diff']:.2e}")
            print(f"  max_rel={result['max_rel_diff']:.2e} mean_rel={result['mean_rel_diff']:.2e}")

    print()
    print("=" * 50)
    print(f"Summary: {passed} passed, {failed} failed, {errors} errors")
    print(f"Missing: {len(only_in_1)} in dir1 only, {len(only_in_2)} in dir2 only")

    if failed > 0 or errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
