#!/usr/bin/env bash
# Compare uniform and custom layouts at the same PP degree while collecting
# per-stage forward/backward timings from the real pipeline runtime.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <gpt2|llama3> <cpu|cuda>" >&2
  exit 2
fi

model="$1"
device="$2"
[[ "$model" == gpt2 || "$model" == llama3 ]] || { echo "unsupported model: $model" >&2; exit 2; }
[[ "$device" == cpu || "$device" == cuda ]] || { echo "unsupported device: $device" >&2; exit 2; }

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-$repo_dir/build}"
exe="${EXE:-$build_dir/$model}"
out_dir="${OUT_DIR:-$repo_dir/artifacts/pipeline_excellent/$model}"
csv="${PERF_CSV:-$repo_dir/targets/pipeline_stage_perf.csv}"
mkdir -p "$out_dir" "$(dirname "$csv")"

if [[ "$model" == gpt2 ]]; then
  default_llmc="$repo_dir/data/gpt2/gpt2_124M.bin"
  default_tokenizer="$repo_dir/data/gpt2/gpt2_tokenizer.bin"
  custom_partition="${CUSTOM_PARTITION:-4,8}"
else
  default_llmc="$repo_dir/data/llama3/llama3.2_1B_fp32.bin"
  default_tokenizer=""
  custom_partition="${CUSTOM_PARTITION:-8,8}"
fi

batch_size="${BATCH_SIZE:-1}"
sequence_length="${SEQUENCE_LENGTH:-32}"
total_batch_size="${TOTAL_BATCH_SIZE:-$((batch_size * sequence_length * 4))}"
pp_size="${PIPELINE_PARALLEL:-2}"
micro_batches=$((total_batch_size / (batch_size * sequence_length)))
if (( pp_size != 2 )); then
  echo "This acceptance script currently reports stage0/stage1 columns and requires PIPELINE_PARALLEL=2." >&2
  exit 2
fi
if (( micro_batches <= 0 || total_batch_size % (batch_size * sequence_length) != 0 )); then
  echo "TOTAL_BATCH_SIZE must be a positive multiple of BATCH_SIZE*SEQUENCE_LENGTH" >&2
  exit 2
fi

common=(
  "--device=$device"
  "--pipeline_parallel=$pp_size"
  "--input_bin=${INPUT_BIN:-$repo_dir/data/$model/tiny_shakespeare_train.bin}"
  "--llmc_filepath=${LLMC_FILE:-$default_llmc}"
  "--overfit_single_batch=true"
  "--num_iteration=1"
  "--dtype=${DTYPE:-float32}"
  "--batch_size=$batch_size"
  "--sequence_length=$sequence_length"
  "--total_batch_size=$total_batch_size"
)
[[ -e "${INPUT_VAL_BIN:-$repo_dir/data/$model/tiny_shakespeare_val.bin}" ]] &&
  common+=("--input_val_bin=${INPUT_VAL_BIN:-$repo_dir/data/$model/tiny_shakespeare_val.bin}")
[[ -n "$default_tokenizer" && -e "${TOKENIZER_BIN:-$default_tokenizer}" ]] &&
  common+=("--tokenizer_bin=${TOKENIZER_BIN:-$default_tokenizer}")

launcher_spec="${LAUNCHER:-$build_dir/infini_run}"
run_case() {
  local name="$1" partition="$2"
  local -a cmd
  if [[ "$launcher_spec" == direct ]]; then
    cmd=("$exe")
  else
    read -r -a launcher_command <<< "$launcher_spec"
    cmd=("${launcher_command[@]}" --nnodes=1 --nproc_per_node="$pp_size" "$exe")
  fi
  cmd+=("${common[@]}")
  [[ -n "$partition" ]] && cmd+=("--pipeline_layer_partition=$partition")
  echo "[$name] ${cmd[*]}"
  INFINI_PIPELINE_STAGE_TIMING=1 "${cmd[@]}" 2>&1 | tee "$out_dir/${name}.log"
}

for required in "$exe" "${common[2]}" "${common[3]}"; do
  if [[ "$required" == --* ]]; then
    path="${required#*=}"
    [[ -e "$path" ]] || { echo "missing required path: $path" >&2; exit 3; }
  else
    [[ -e "$required" ]] || { echo "missing required path: $required" >&2; exit 3; }
  fi
done

run_case uniform ""
run_case custom "$custom_partition"

MICRO_BATCHES="$micro_batches" python3 - "$out_dir" "$csv" <<'PY'
import csv
import os
import re
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
csv_path = Path(sys.argv[2])
timing = re.compile(
    r"pipeline_stage_timing direction=(forward|backward) "
    r"stage=(\d+) chunk=(\d+) microbatch=(\d+) elapsed_ms=([0-9.]+)"
)
step = re.compile(r"step\s+\d+/\d+\s+\|.*?\(([-+0-9.]+) ms \|\s*([-+0-9.]+) tok/s")
rows = []
for name in ("uniform", "custom"):
    text = (out_dir / f"{name}.log").read_text(errors="replace")
    stage_totals = {}
    for direction, stage, chunk, microbatch, elapsed in timing.findall(text):
        stage = int(stage)
        stage_totals[stage] = stage_totals.get(stage, 0.0) + float(elapsed)
    if not stage_totals:
        raise SystemExit(f"no pipeline_stage_timing records found in {name}.log")
    match = step.findall(text)
    if not match:
        raise SystemExit(f"no step timing found in {name}.log")
    elapsed_ms, tok_per_s = map(float, match[-1])
    ordered = [stage_totals.get(i, 0.0) for i in range(2)]
    max_stage = max(ordered)
    min_stage = min(ordered)
    rows.append({
        "layout": name,
        "pipeline_parallel": 2,
        "micro_batches": int(os.environ["MICRO_BATCHES"]),
        "elapsed_ms": elapsed_ms,
        "tok_per_s": tok_per_s,
        "bubble_ratio": 1.0 / (int(os.environ["MICRO_BATCHES"]) + 1),
        "stage0_compute_ms": ordered[0],
        "stage1_compute_ms": ordered[1],
        "max_stage_compute_ms": max_stage,
        "stage_imbalance_ratio": max_stage / min_stage if min_stage > 0 else float("inf"),
    })
with csv_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {csv_path}")
PY
cat "$csv"
