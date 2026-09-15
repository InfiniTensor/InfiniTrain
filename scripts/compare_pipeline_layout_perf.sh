#!/usr/bin/env bash
# Run one baseline and one custom-layout smoke test, then emit a compact CSV
# with throughput and the idealized pipeline bubble estimate.
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <gpt2|llama3> <cpu|cuda>" >&2; exit 2
fi
model="$1"; device="$2"
[[ "$model" == gpt2 || "$model" == llama3 ]] || { echo "unsupported model: $model" >&2; exit 2; }
[[ "$device" == cpu || "$device" == cuda ]] || { echo "unsupported device: $device" >&2; exit 2; }
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-$repo_dir/build}"
exe="${EXE:-$build_dir/$model}"
out_dir="${OUT_DIR:-$repo_dir/artifacts/pipeline_layout_perf/$model}"
csv="${PERF_CSV:-$repo_dir/targets/pipeline_layout_perf.csv}"
mkdir -p "$out_dir/baseline" "$out_dir/custom" "$(dirname "$csv")"

if [[ "$model" == gpt2 ]]; then default_sequence=32; else default_sequence=4; fi
batch_size="${BATCH_SIZE:-1}"
sequence_length="${SEQUENCE_LENGTH:-$default_sequence}"
total_batch_size="${TOTAL_BATCH_SIZE:-$((batch_size * sequence_length))}"
if (( total_batch_size % (batch_size * sequence_length) != 0 )); then
  echo "TOTAL_BATCH_SIZE must be divisible by BATCH_SIZE*SEQUENCE_LENGTH" >&2; exit 2
fi
micro_batches=$((total_batch_size / (batch_size * sequence_length)))

common=(--device="$device" --input_bin="${INPUT_BIN:-$repo_dir/data/$model/tiny_shakespeare_train.bin}" \
  --llmc_filepath="${LLMC_FILE:-$repo_dir/data/$model/$([[ $model == gpt2 ]] && echo gpt2_124M.bin || echo llama3.2_1B_fp32.bin)}" \
  --overfit_single_batch=true --num_iteration=1 --dtype="${DTYPE:-float32}" \
  --batch_size="$batch_size" --sequence_length="$sequence_length" --total_batch_size="$total_batch_size")
[[ -e "${INPUT_VAL_BIN:-$repo_dir/data/$model/tiny_shakespeare_val.bin}" ]] && common+=(--input_val_bin="${INPUT_VAL_BIN:-$repo_dir/data/$model/tiny_shakespeare_val.bin}")
[[ "$model" == gpt2 && -e "${TOKENIZER_BIN:-$repo_dir/data/gpt2/gpt2_tokenizer.bin}" ]] && common+=(--tokenizer_bin="${TOKENIZER_BIN:-$repo_dir/data/gpt2/gpt2_tokenizer.bin}")
launcher_spec="${LAUNCHER:-$build_dir/infini_run}"
run_case() {
  local name="$1" pp="$2" layout="$3" log="$4"; local -a cmd
  if [[ "$launcher_spec" == direct ]]; then
    cmd=("$exe")
  else
    read -r -a launcher_command <<< "$launcher_spec"
    cmd=("${launcher_command[@]}" --nnodes=1 --nproc_per_node="$pp" "$exe")
  fi
  cmd+=("${common[@]}" --pipeline_parallel="$pp")
  [[ -n "$layout" ]] && cmd+=(--pipeline_layout="$layout")
  echo "[$name] ${cmd[*]}"
  "${cmd[@]}" 2>&1 | tee "$log"
}
run_case baseline 1 "" "$out_dir/baseline/run.log"
if [[ "$model" == gpt2 ]]; then layout='Etttttt|ttttttFH'; else layout='Etttttttt|ttttttttFH'; fi
run_case custom 2 "$layout" "$out_dir/custom/run.log"

MICRO_BATCHES="$micro_batches" python3 - "$out_dir" "$csv" <<'PY'
import csv, re, sys
from pathlib import Path
out, csv_path = Path(sys.argv[1]), Path(sys.argv[2])
rows=[]
for name, pp in (("baseline",1),("custom",2)):
    text=(out/name/"run.log").read_text(errors="replace")
    m=re.findall(r"step\s+\d+/\d+\s+\|\s+train loss[^\n]*?\(([-+0-9.]+) ms \|\s*([-+0-9.]+) tok/s", text)
    if not m: raise SystemExit(f"could not parse timing from {name} log")
    ms,tps=map(float,m[-1]); microbatches=float(__import__('os').environ.get('MICRO_BATCHES','1'))
    if microbatches <= 0: raise SystemExit('MICRO_BATCHES must be positive')
    bubble=(pp-1)/(microbatches+pp-1)
    rows.append(dict(layout=name,pipeline_parallel=pp,elapsed_ms=ms,tok_per_s=tps,bubble_ratio=bubble))
with csv_path.open("w", newline="") as f:
    w=csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print(f"wrote {csv_path}")
PY
cat "$csv"
