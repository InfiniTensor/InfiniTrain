#!/usr/bin/env bash
set -euo pipefail

binary="${1:-./build/benchmarks/generator_benchmark}"
device="${2:-cpu}"
device_index="${DEVICE_INDEX:-0}"
warmup="${WARMUP:-10}"
iterations="${ITERATIONS:-100}"
seed="${SEED:-42}"

if [[ ! -x "${binary}" ]]; then
  echo "benchmark executable not found or not executable: ${binary}" >&2
  exit 2
fi

echo "# timestamp=$(date --iso-8601=seconds)" >&2
echo "# host=$(hostname)" >&2
echo "# kernel=$(uname -srmo)" >&2
echo "# binary=${binary}" >&2
echo "device,device_index,operation,generator,elements,iterations,latency_us,gsamples_s,bandwidth_gbps"

for generator in explicit default; do
  for operation in uniform normal; do
    for elements in 1024 1048576 16777216; do
      "${binary}" \
        --device "${device}" \
        --device-index "${device_index}" \
        --op "${operation}" \
        --generator "${generator}" \
        --elements "${elements}" \
        --warmup "${warmup}" \
        --iterations "${iterations}" \
        --seed "${seed}" | awk 'NR > 1'
    done
  done
done

"${binary}" \
  --device "${device}" \
  --device-index "${device_index}" \
  --op state \
  --generator explicit \
  --elements 1 \
  --warmup "${warmup}" \
  --iterations "${iterations}" \
  --seed "${seed}" | awk 'NR > 1'
