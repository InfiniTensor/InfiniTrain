#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
test_binary=${1:-"$repo_root/build-cpu/tests/tensor/test_tensor_cpu_only"}

if [[ ! -x "$test_binary" ]]; then
  echo "Generator test binary is not executable: $test_binary" >&2
  exit 1
fi

run_digest() {
  "$test_binary" --gtest_filter='CPUGeneratorTest.CrossProcessReproducibilityDigest' 2>&1 \
    | sed -n 's/^GENERATOR_REPRODUCIBILITY_DIGEST=//p'
}

first=$(run_digest)
second=$(run_digest)

if [[ -z "$first" || -z "$second" ]]; then
  echo "Failed to capture reproducibility digest" >&2
  exit 1
fi
if [[ "$first" != "$second" ]]; then
  echo "Cross-process reproducibility mismatch: $first != $second" >&2
  exit 1
fi

echo "Cross-process Generator reproducibility passed: $first"
