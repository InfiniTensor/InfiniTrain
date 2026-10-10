#!/usr/bin/env bash
# Multi-GPU regression for custom pipeline layouts combined with DDP and TP.
#
# Runs a single-GPU reference and then every layout x parallelism combination the
# layout feature claims to support, asserting that each one reproduces the reference
# loss (and, where the parameters are not TP-sharded, every per-parameter gradient).
#
# Usage:
#   tests/distributed/test_pipeline_layout_parallel_matrix.sh BUILD_DIR INPUT_BIN CHECKPOINT [NUM_GPUS]
#
# NUM_GPUS defaults to 8. Combinations that need more GPUs than are available are
# skipped and reported as SKIP, so the script is still useful on a 2- or 4-GPU host.
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: $0 BUILD_DIR INPUT_BIN CHECKPOINT [NUM_GPUS]" >&2
    exit 2
fi

build_dir="$(realpath "$1")"
input_bin="$(realpath "$2")"
checkpoint="$(realpath "$3")"
num_gpus="${4:-8}"
source_dir="$(realpath "$(dirname "$0")/../..")"
gpt2="$build_dir/gpt2"
infini_run="$build_dir/infini_run"
compare="$source_dir/scripts/precision_check/precision_compare.py"

for path in "$gpt2" "$infini_run" "$input_bin" "$checkpoint" "$compare"; do
    if [[ ! -e "$path" ]]; then
        echo "Required test input does not exist: $path" >&2
        exit 2
    fi
done

# The layer count decides every partition below, so read it out of the LLMC header
# instead of hard-coding one model size.
layers="$(python3 - "$checkpoint" <<'PY'
import struct, sys
with open(sys.argv[1], "rb") as f:
    header = struct.unpack("<256i", f.read(1024))
magic, version, _, _, n_layer = header[0], header[1], header[2], header[3], header[4]
assert magic == 20240326 and version == 3, f"not an fp32 LLMC checkpoint: {magic}/{version}"
print(n_layer)
PY
)"
if (( layers < 4 )); then
    echo "This regression needs a checkpoint with at least 4 Transformer layers; got $layers" >&2
    exit 2
fi

test_dir="$(mktemp -d /tmp/infinitrain-pipeline-matrix.XXXXXX)"
trap 'rm -rf -- "$test_dir"' EXIT

# Non-uniform partitions derived from the real layer count.
pp2_head=$(( layers / 4 )); (( pp2_head > 0 )) || pp2_head=1
pp2_partition="${pp2_head},$(( layers - pp2_head ))"
pp4_partition="1,$(( layers - 3 )),1,1"
quarter=$(( layers / 4 ))
quarter_remainder=$(( layers - 3 * quarter ))
chunk_layout="0:${quarter},1:${quarter},1:${quarter},0:${quarter_remainder}"
# Equal layer costs plus an lm head worth two layers; the balanced split is the smallest
# stage 0 size that minimises max(k, layers - k + 2).
lm_head_costs="$(python3 -c "print(','.join(['1'] * $layers) + ',L:2')")"
lm_head_head="$(python3 -c "
layers = $layers
print(min(range(1, layers), key=lambda k: (max(k, layers - k + 2), k)))")"
megatron_layout="Et*${quarter}|t*${quarter}|t*${quarter}|t*${quarter_remainder}NL"

common_args=(
    --device=cuda
    --input_bin="$input_bin"
    --llmc_filepath="$checkpoint"
    --batch_size=4
    --sequence_length=64
    # Must stay divisible by batch_size * sequence_length * DP size for every DP size below.
    --total_batch_size=2048
    --num_iteration=1
    --freq_generate_txt=1000
    --dtype=float32
)

gpu_list() { seq -s, 0 $(( $1 - 1 )); }

extract_loss() { sed -n 's/.*train loss \([^ |]*\).*/\1/p' "$1" | tail -n 1; }

assert_close() {
    # assert_close NAME REFERENCE ACTUAL TOLERANCE
    awk -v name="$1" -v reference="$2" -v actual="$3" -v tolerance="$4" 'BEGIN {
        if (reference == "" || actual == "") {
            printf "%s: missing loss (reference=%s actual=%s)\n", name, reference, actual > "/dev/stderr";
            exit 1;
        }
        difference = reference - actual;
        if (difference < 0) difference = -difference;
        if (difference > tolerance) {
            printf "%s: loss mismatch reference=%s actual=%s difference=%g\n",
                   name, reference, actual, difference > "/dev/stderr";
            exit 1;
        }
    }'
}

failures=0
results=()

echo "=== reference: single GPU, ${layers} layers ==="
reference_grad="$test_dir/reference-grad"
reference_log="$test_dir/reference.log"
env GLOG_logtostderr=1 CUDA_VISIBLE_DEVICES=0 \
    "$gpt2" "${common_args[@]}" --dump_gradients="$reference_grad" >"$reference_log" 2>&1
reference_loss="$(extract_loss "$reference_log")"
if [[ -z "$reference_loss" ]]; then
    echo "Failed to extract the reference loss from $reference_log" >&2
    exit 1
fi
find "$reference_grad" -type f -name '*.npy' -printf '%f\n' | sort >"$test_dir/reference-files"
echo "reference loss: $reference_loss ($(wc -l <"$test_dir/reference-files") gradients)"

run_case() {
    # run_case NAME GPUS TOLERANCE COMPARE_GRADIENTS EXPECTED_LAYOUT_LINE -- <gpt2 flags...>
    local name="$1" gpus="$2" tolerance="$3" compare_gradients="$4" expected_layout="$5"
    shift 6 # drop the parsed fields and the "--" separator

    if (( gpus > num_gpus )); then
        echo "--- $name: SKIP (needs $gpus GPUs, have $num_gpus)"
        results+=("SKIP  $name (needs $gpus GPUs)")
        return 0
    fi

    local log="$test_dir/$name.log"
    local grad_dir="$test_dir/$name-grad"
    local extra=()
    if [[ "$compare_gradients" == "grad" ]]; then extra=(--dump_gradients="$grad_dir"); fi

    echo "--- $name: $gpus GPU(s)"
    if ! env GLOG_logtostderr=1 CUDA_VISIBLE_DEVICES="$(gpu_list "$gpus")" \
        "$infini_run" --nproc_per_node="$gpus" "$gpt2" "${common_args[@]}" "${extra[@]}" "$@" \
        >"$log" 2>&1; then
        echo "$name: training process failed, see $log" >&2
        tail -n 20 "$log" >&2
        results+=("FAIL  $name (training failed)")
        failures=$(( failures + 1 ))
        return 0
    fi

    local ok=1
    if [[ -n "$expected_layout" ]] && ! grep -Fq "$expected_layout" "$log"; then
        echo "$name: expected layout line not found: $expected_layout" >&2
        ok=0
    fi
    if ! assert_close "$name" "$reference_loss" "$(extract_loss "$log")" "$tolerance"; then
        ok=0
    fi
    if [[ "$compare_gradients" == "grad" ]]; then
        find "$grad_dir" -type f -name '*.npy' -printf '%f\n' | sort >"$test_dir/$name-files"
        if ! diff -u "$test_dir/reference-files" "$test_dir/$name-files" >"$test_dir/$name-files.diff"; then
            echo "$name: gradient file set differs from the reference" >&2
            head -n 20 "$test_dir/$name-files.diff" >&2
            ok=0
        elif ! python3 "$compare" --dir1 "$reference_grad" --dir2 "$grad_dir" --atol 1e-5 --rtol 0 \
            >"$test_dir/$name-grad.log" 2>&1; then
            echo "$name: per-parameter gradients differ from the reference" >&2
            tail -n 10 "$test_dir/$name-grad.log" >&2
            ok=0
        fi
    fi

    if (( ok )); then
        results+=("PASS  $name (loss $(extract_loss "$log"), $gpus GPU)")
    else
        results+=("FAIL  $name")
        failures=$(( failures + 1 ))
    fi
}

# Layout x parallelism matrix. TP shards parameters, so TP cases compare the loss only.
run_case pp2_partition        2 1e-5 grad "stage 0: embedding layers[0,${pp2_head})" -- \
    --pipeline_parallel=2 --pipeline_layer_partition="$pp2_partition"
run_case pp4_partition        4 1e-5 grad "stage 1: layers[1,$(( layers - 2 )))" -- \
    --pipeline_parallel=4 --pipeline_layer_partition="$pp4_partition"
run_case pp2_ddp2             4 1e-5 grad "stage 0: embedding layers[0,${pp2_head})" -- \
    --pipeline_parallel=2 --pipeline_layer_partition="$pp2_partition"
run_case pp2_ddp4             8 1e-5 grad "stage 0: embedding layers[0,${pp2_head})" -- \
    --pipeline_parallel=2 --pipeline_layer_partition="$pp2_partition"
run_case pp2_tp2              4 1e-5 loss "stage 0: embedding layers[0,${pp2_head})" -- \
    --pipeline_parallel=2 --tensor_parallel=2 --pipeline_layer_partition="$pp2_partition"
run_case pp2_tp2_ddp2         8 1e-5 loss "stage 0: embedding layers[0,${pp2_head})" -- \
    --pipeline_parallel=2 --tensor_parallel=2 --pipeline_layer_partition="$pp2_partition"
run_case pp4_tp2              8 1e-5 loss "stage 1: layers[1,$(( layers - 2 )))" -- \
    --pipeline_parallel=4 --tensor_parallel=2 --pipeline_layer_partition="$pp4_partition"
run_case pp2_costs            2 1e-5 grad "stage 0: embedding layers[0,1)" -- \
    --pipeline_parallel=2 --pipeline_layer_costs="$(python3 -c "print(','.join(['10'] + ['1'] * ($layers - 1)))")"
run_case pp2_costs_lm_head    2 1e-5 grad "stage 0: embedding layers[0,${lm_head_head})" -- \
    --pipeline_parallel=2 --pipeline_layer_costs="$lm_head_costs"
run_case pp2_costs_lm_head_ddp2 4 1e-5 grad "stage 0: embedding layers[0,${lm_head_head})" -- \
    --pipeline_parallel=2 --pipeline_layer_costs="$lm_head_costs"
run_case pp2_vpp2_chunk       2 1e-5 grad "" -- \
    --pipeline_parallel=2 --virtual_pipeline_parallel=2 --pipeline_chunk_layout="$chunk_layout"
run_case pp2_vpp2_chunk_ddp2  4 1e-5 grad "" -- \
    --pipeline_parallel=2 --virtual_pipeline_parallel=2 --pipeline_chunk_layout="$chunk_layout"
run_case pp4_megatron         4 1e-5 grad "" -- \
    --pipeline_parallel=4 --pipeline_model_parallel_layout="$megatron_layout"
run_case pp4_megatron_ddp2    8 1e-5 grad "" -- \
    --pipeline_parallel=4 --pipeline_model_parallel_layout="$megatron_layout"

echo
echo "=== summary (reference loss $reference_loss) ==="
printf '%s\n' "${results[@]}"

if (( failures )); then
    echo "FAILED: $failures case(s) did not match the single-GPU reference" >&2
    exit 1
fi
echo "PASS: every layout x parallelism combination matched the single-GPU reference"
