#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: $0 BUILD_DIR INPUT_BIN GPT2_LLMC_CHECKPOINT [GPU_IDS]" >&2
    exit 2
fi

build_dir="$(realpath "$1")"
input_bin="$(realpath "$2")"
checkpoint="$(realpath "$3")"
gpu_ids="${4:-0,1}"
source_dir="$(realpath "$(dirname "$0")/../..")"
gpt2="$build_dir/gpt2"
infini_run="$build_dir/infini_run"

for path in "$gpt2" "$infini_run" "$input_bin" "$checkpoint"; do
    if [[ ! -e "$path" ]]; then
        echo "Required test input does not exist: $path" >&2
        exit 2
    fi
done
if [[ "$gpu_ids" != *,* ]]; then
    echo "GPU_IDS must contain two comma-separated device IDs" >&2
    exit 2
fi

test_dir="$(mktemp -d /tmp/infinitrain-pipeline-layout-e2e.XXXXXX)"
trap 'rm -rf -- "$test_dir"' EXIT
single_grad="$test_dir/single-grad"
pipeline_grad="$test_dir/pipeline-grad"
vpp_grad="$test_dir/vpp-grad"
single_log="$test_dir/single.log"
pipeline_log="$test_dir/pipeline.log"
vpp_log="$test_dir/vpp.log"
first_gpu="${gpu_ids%%,*}"

common_args=(
    --device=cuda
    --input_bin="$input_bin"
    --llmc_filepath="$checkpoint"
    --batch_size=4
    --sequence_length=64
    --total_batch_size=512
    --num_iteration=1
    --freq_generate_txt=1000
    --dtype=float32
)

echo "Running single-GPU reference..."
env GLOG_logtostderr=1 CUDA_VISIBLE_DEVICES="$first_gpu" \
    "$gpt2" "${common_args[@]}" --dump_gradients="$single_grad" 2>&1 | tee "$single_log"

echo "Running two-stage automatic pipeline layout..."
env GLOG_logtostderr=1 CUDA_VISIBLE_DEVICES="$gpu_ids" \
    "$infini_run" --nproc_per_node=2 "$gpt2" "${common_args[@]}" \
    --pipeline_parallel=2 \
    --pipeline_layer_costs=10,1,1,1,1,1,1,1,1,1,1,1 \
    --dump_gradients="$pipeline_grad" 2>&1 | tee "$pipeline_log"

grep -Fq "stage 0: embedding layers[0,1)" "$pipeline_log"
grep -Fq "stage 1: layers[1,12) final_norm lm_head" "$pipeline_log"

single_loss="$(sed -n 's/.*train loss \([^ |]*\).*/\1/p' "$single_log" | tail -n 1)"
pipeline_loss="$(sed -n 's/.*train loss \([^ |]*\).*/\1/p' "$pipeline_log" | tail -n 1)"
if [[ -z "$single_loss" || -z "$pipeline_loss" ]]; then
    echo "Failed to extract training loss from logs" >&2
    exit 1
fi
awk -v reference="$single_loss" -v actual="$pipeline_loss" 'BEGIN {
    difference = reference - actual;
    if (difference < 0) difference = -difference;
    if (difference > 1e-5) {
        printf "Loss mismatch: reference=%s pipeline=%s difference=%g\n", reference, actual, difference > "/dev/stderr";
        exit 1;
    }
}'

find "$single_grad" -type f -name '*.npy' -printf '%f\n' | sort >"$test_dir/single-files"
find "$pipeline_grad" -type f -name '*.npy' -printf '%f\n' | sort >"$test_dir/pipeline-files"
diff -u "$test_dir/single-files" "$test_dir/pipeline-files"

python3 "$source_dir/scripts/precision_check/precision_compare.py" \
    --dir1 "$single_grad" --dir2 "$pipeline_grad" --atol 1e-5 --rtol 0

echo "Running arbitrary virtual Chunk-to-Stage mapping..."
env GLOG_logtostderr=1 CUDA_VISIBLE_DEVICES="$gpu_ids" \
    "$infini_run" --nproc_per_node=2 "$gpt2" "${common_args[@]}" \
    --pipeline_parallel=2 --virtual_pipeline_parallel=2 \
    --pipeline_chunk_layout=0:3,1:3,1:3,0:3 \
    --dump_gradients="$vpp_grad" 2>&1 | tee "$vpp_log"

grep -Fq "stage 0: embedding layers[0,3) layers[9,12) final_norm lm_head" "$vpp_log"
vpp_loss="$(sed -n 's/.*train loss \([^ |]*\).*/\1/p' "$vpp_log" | tail -n 1)"
awk -v reference="$single_loss" -v actual="$vpp_loss" 'BEGIN {
    difference = reference - actual;
    if (difference < 0) difference = -difference;
    if (difference > 1e-5) exit 1;
}'
find "$vpp_grad" -type f -name '*.npy' -printf '%f\n' | sort >"$test_dir/vpp-files"
diff -u "$test_dir/single-files" "$test_dir/vpp-files"
python3 "$source_dir/scripts/precision_check/precision_compare.py" \
    --dir1 "$single_grad" --dir2 "$vpp_grad" --atol 1e-5 --rtol 0

echo "PASS: automatic PP and arbitrary vPP layouts match the single-GPU loss and gradients"
