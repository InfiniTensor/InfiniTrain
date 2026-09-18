#!/usr/bin/env bash
# Compare a one-stage baseline with a two-stage custom pipeline layout.
#
# This script intentionally runs exactly one optimization iteration.  It is an
# end-to-end smoke test, not a replacement for a training run.  The caller must
# provide the model checkpoint and input binaries (or override their paths).

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <gpt2|llama3> <cpu|cuda>" >&2
  exit 2
fi

model="$1"
device="$2"
case "$model" in
  gpt2|llama3) ;;
  *) echo "unsupported model '$model' (expected gpt2 or llama3)" >&2; exit 2 ;;
esac
case "$device" in
  cpu|cuda) ;;
  *) echo "unsupported device '$device' (expected cpu or cuda)" >&2; exit 2 ;;
esac

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-$repo_dir/build}"
exe="${EXE:-$build_dir/$model}"
compare_script="${COMPARE_LOSS:-$repo_dir/scripts/compare_loss.py}"
input_bin="${INPUT_BIN:-$repo_dir/data/$model/tiny_shakespeare_train.bin}"
input_val_bin="${INPUT_VAL_BIN:-$repo_dir/data/$model/tiny_shakespeare_val.bin}"
tokenizer_bin="${TOKENIZER_BIN:-$repo_dir/data/$model/${model}_tokenizer.bin}"
if [[ "$model" == "gpt2" ]]; then
  default_llmc="$repo_dir/data/gpt2/gpt2_124M.bin"
else
  default_llmc="$repo_dir/data/llama3/llama3.2_1B_fp32.bin"
fi
llmc_file="${LLMC_FILE:-$default_llmc}"
out_dir="${OUT_DIR:-$repo_dir/artifacts/pipeline_custom_layout/$model}"
num_iteration="${NUM_ITERATION:-1}"
launcher="${LAUNCHER:-$build_dir/infini_run}"

if [[ "$num_iteration" != "1" ]]; then
  echo "NUM_ITERATION must be 1 for this validation script (got '$num_iteration')" >&2
  exit 2
fi

for required in "$exe" "$compare_script" "$input_bin" "$llmc_file"; do
  if [[ ! -e "$required" ]]; then
    echo "missing required path: $required" >&2
    echo "Set BUILD_DIR/INPUT_BIN/LLMC_FILE (and related variables) to continue." >&2
    exit 3
  fi
done
if [[ "$device" == "cuda" && ! -e "$input_val_bin" ]]; then
  echo "warning: validation input '$input_val_bin' is absent; continuing without validation data" >&2
fi
if [[ ! -x "$exe" ]]; then
  echo "model executable is not executable: $exe" >&2
  exit 3
fi

mkdir -p "$out_dir/baseline" "$out_dir/custom"

common_args=(
  "--device=$device"
  "--input_bin=$input_bin"
  "--llmc_filepath=$llmc_file"
  "--overfit_single_batch=true"
  "--num_iteration=1"
  "--dtype=${DTYPE:-float32}"
)
if [[ -e "$input_val_bin" ]]; then common_args+=("--input_val_bin=$input_val_bin"); fi
if [[ -e "$tokenizer_bin" ]]; then common_args+=("--tokenizer_bin=$tokenizer_bin"); fi

run_case() {
  local name="$1" pp="$2" partition="$3" output="$4"
  local -a command
  if [[ "$launcher" == "direct" ]]; then
    command=("$exe")
  else
    read -r -a launcher_command <<< "$launcher"
    command=("${launcher_command[@]}" --nnodes=1 --nproc_per_node="$pp" "$exe")
  fi
  command+=("${common_args[@]}" "--pipeline_parallel=$pp")
  if [[ -n "$partition" ]]; then command+=("--pipeline_layer_partition=$partition"); fi
  echo "[$name] ${command[*]}"
  "${command[@]}" 2>&1 | tee "$output/run.log"
}

run_case baseline 1 "" "$out_dir/baseline"
run_case custom 2 "6,6" "$out_dir/custom"

python3 "$compare_script" "$out_dir/baseline" "$out_dir/custom" \
  --threshold-fp32 "${THRESHOLD_FP32:-1e-5}" \
  --threshold-bf16 "${THRESHOLD_BF16:-1e-2}"

echo "Pipeline custom-layout smoke test passed; logs are in $out_dir"
