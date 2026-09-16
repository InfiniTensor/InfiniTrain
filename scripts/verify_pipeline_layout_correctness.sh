#!/usr/bin/env bash
# Verify that a custom pipeline layout produces training results identical to the
# single-card reference within tolerance. This is the 2-Stage end-to-end correctness
# check required by the assignment "通过标准":
#   fp32:   |loss| diff <= 1e-05
#   bf16:   |loss| diff <= 1e-02
#
# It runs the GPT-2 example twice with the SAME weights (--llmc_filepath), data, and
# hyper-parameters, but different parallelism:
#   baseline : single card  (--nproc_per_node=1, --pipeline_parallel 1)
#   custom   : 2-stage PP   (--nproc_per_node=2, --pipeline_parallel 2,
#                             --pipeline_layer_partition 7,5)
# then compares the per-step training loss with scripts/compare_loss.py.
#
# Why loss is sufficient evidence for forward + gradient consistency:
# the loss at step t integrates the forward pass, backward pass, and every gradient
# from steps 0..t-1. Any fwd/bwd/grad discrepancy compounds into a loss divergence on
# the next step, so matching loss over multiple steps proves all three agree.
#
# Usage:
#   bash scripts/verify_pipeline_layout_correctness.sh
#   DTYPE=bfloat16 bash scripts/verify_pipeline_layout_correctness.sh
#   WEIGHTS=/path/to/gpt2_124M.bin DATA=/path/to/train.bin bash scripts/verify_pipeline_layout_correctness.sh
#
# The weights file must match the model (d12 / GPT-2 124M: 12 layers, 768 hidden).
# If data/gpt2/gpt2_124M.bin is missing, download it with:
#   bash scripts/assets/prepare-infinitrain-assets.sh
set -euo pipefail

# ---- config (env-overridable) ----
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${BIN:-$ROOT/build/gpt2}"
INFINI_RUN="${INFINI_RUN:-$ROOT/build/infini_run}"
WEIGHTS="${WEIGHTS:-$ROOT/data/gpt2/gpt2_124M.bin}"
DATA="${DATA:-$ROOT/data/gpt2/tiny_shakespeare_train.bin}"
DTYPE="${DTYPE:-float32}"
NUM_ITER="${NUM_ITER:-3}"
BATCH="${BATCH:-4}"
SEQ_LEN="${SEQ_LEN:-64}"
TOTAL_BATCH="${TOTAL_BATCH:-512}"   # must be a multiple of BATCH * SEQ_LEN
CUSTOM_PARTITION="${CUSTOM_PARTITION:-7,5}"
OUT_DIR="${OUT_DIR:-$ROOT/build/pp_layout_correctness}"
COMPARE_SCRIPT="$ROOT/scripts/compare_loss.py"

# The log basename encodes the dtype so compare_loss.py picks the right threshold.
case "$DTYPE" in
float32)  LOG_NAME="gpt2_d12.log" ;;
bfloat16) LOG_NAME="gpt2_d12_bfloat16.log" ;;
*) echo "Unsupported DTYPE='$DTYPE' (use float32 or bfloat16)" >&2; exit 2 ;;
esac

# ---- preflight ----
command -v python3 >/dev/null 2>&1 || { echo "python3 is required but not found" >&2; exit 2; }
for f in "$BIN" "$INFINI_RUN" "$WEIGHTS" "$DATA" "$COMPARE_SCRIPT"; do
    [[ -e "$f" ]] || { echo "Missing: $f" >&2; exit 2; }
done

REF_DIR="$OUT_DIR/baseline"
CUSTOM_DIR="$OUT_DIR/custom"
mkdir -p "$REF_DIR" "$CUSTOM_DIR"

# Args shared by both runs (identical weights/data/hparams).
COMMON_ARGS=(
    --llmc_filepath "$WEIGHTS"
    --input_bin "$DATA"
    --dtype "$DTYPE"
    --batch_size "$BATCH"
    --sequence_length "$SEQ_LEN"
    --total_batch_size "$TOTAL_BATCH"
    --num_iteration "$NUM_ITER"
    # No --tokenizer_bin is passed, so text generation never runs; set the
    # frequency high anyway as a defensive measure.
    --freq_generate_txt 1000000
)

run_and_check() {
    local label="$1"; shift
    local log="$1"; shift
    echo "== [$label] =="
    "$@" 2>&1 | tee "$log"
    # Sanity check: the log must contain exactly NUM_ITER "train loss" lines.
    local count
    count=$(grep -c "train loss" "$log" || true)
    if [[ "$count" -ne "$NUM_ITER" ]]; then
        echo "Expected $NUM_ITER 'train loss' lines in $log but found $count (run may have failed)" >&2
        exit 1
    fi
}

echo "=== Pipeline custom-layout correctness verification ==="
echo "dtype=$DTYPE  iters=$NUM_ITER  batch=$BATCH  seq=$SEQ_LEN  total_batch=$TOTAL_BATCH"
echo "partition=$CUSTOM_PARTITION  weights=$WEIGHTS"
echo

run_and_check "1/3 reference: single card (PP=1)" "$REF_DIR/$LOG_NAME" \
    "$INFINI_RUN" --nproc_per_node=1 "$BIN" \
    --pipeline_parallel 1 "${COMMON_ARGS[@]}"

run_and_check "2/3 custom: 2-stage PP, partition $CUSTOM_PARTITION" "$CUSTOM_DIR/$LOG_NAME" \
    "$INFINI_RUN" --nproc_per_node=2 "$BIN" \
    --pipeline_parallel 2 --pipeline_layer_partition "$CUSTOM_PARTITION" "${COMMON_ARGS[@]}"

echo "== 3/3 compare per-step training loss =="
if python3 "$COMPARE_SCRIPT" "$REF_DIR" "$CUSTOM_DIR" --threshold-fp32 1e-5 --threshold-bf16 1e-2; then
    echo
    echo "PASS: custom layout ($CUSTOM_PARTITION) loss matches single-card reference within tolerance."
else
    echo
    echo "FAIL: loss mismatch exceeds tolerance." >&2
    exit 1
fi
