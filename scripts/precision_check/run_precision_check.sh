#!/bin/bash
# InfiniTrain Precision Checker
set -e
set -o pipefail

CONFIG_FILE="${1:-scripts/precision_check/precision_check_config.json}"

# Dependencies check
if ! command -v jq >/dev/null 2>&1; then
    echo "Error: jq is required. Install with: sudo apt-get install -y jq"
    exit 1
fi

# Read global variables from config file
read_var() {
    local key="$1"
    jq -r --arg k "$key" '.variables[$k] // empty' "$CONFIG_FILE"
}

# Configuration with defaults
BUILD_DIR="$(read_var BUILD_DIR)";                      : "${BUILD_DIR:=../build}"
BASE_OUTPUT_DIR="$(read_var PRECISION_CHECK_OUTPUT_DIR)"; : "${BASE_OUTPUT_DIR:=./log_precision_check}"
COMPARE_SCRIPT="$(read_var COMPARE_SCRIPT)";            : "${COMPARE_SCRIPT:=scripts/precision_check/precision_compare.py}"

if [ ! -f "$COMPARE_SCRIPT" ]; then
    echo "Error: $COMPARE_SCRIPT not found."
    exit 1
fi

# Run tests for a single model
run_model_tests() {
    local MODEL="$1"
    local INPUT_BIN="$2"
    local LLMC_FILEPATH="$3"
    local PYTORCH_DIR="$4"

    local BIN="${BUILD_DIR}/${MODEL}"
    local MODEL_ARGS="--device cuda --input_bin ${INPUT_BIN} --llmc_filepath ${LLMC_FILEPATH}"
    local OUTPUT_DIR="${BASE_OUTPUT_DIR}_${MODEL}"

    echo ""
    echo "============================================================"
    echo "=== InfiniTrain Precision Checker - ${MODEL} ==="
    echo "============================================================"
    echo "Binary: $BIN"
    echo "Output directory: $OUTPUT_DIR"

    if [ ! -f "$BIN" ]; then
        echo "Error: $BIN not found. Skipping ${MODEL}."
        return 1
    fi

    # Clean test directory
    rm -rf "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"

    # 1. Single-rank test - Simple format
    echo ""
    echo "=== 1. Single-rank test (Simple format) ==="
    CMD="$BIN $MODEL_ARGS --precision_check \"level=1,path=$OUTPUT_DIR/test1_simple,format=simple,save_tensors=true\" --num_iteration 1"
    echo "Running: $CMD"
    eval $CMD

    TIMESTAMP_DIR=$(ls -t "$OUTPUT_DIR/test1_simple" | head -1)
    echo "Timestamp directory: $TIMESTAMP_DIR"
    NPY_COUNT=$(ls "$OUTPUT_DIR/test1_simple/$TIMESTAMP_DIR/rank_0"/*.npy 2>/dev/null | wc -l)
    echo "Rank 0 NPY files: $NPY_COUNT"

    # 2. Single-rank test - MD5 format
    echo ""
    echo "=== 2. Single-rank test (MD5 format) ==="
    CMD="$BIN $MODEL_ARGS --precision_check \"level=1,path=$OUTPUT_DIR/test2_md5,format=md5\" --num_iteration 1"
    echo "Running: $CMD"
    eval $CMD

    # 3. Multi-iter overwrite test
    echo ""
    echo "=== 3. Multi-iter overwrite test ==="
    CMD="$BIN $MODEL_ARGS --precision_check \"level=1,path=$OUTPUT_DIR/test3_overwrite,save_tensors=true\" --num_iteration 3"
    echo "Running: $CMD"
    eval $CMD

    TIMESTAMP_DIR=$(ls -t "$OUTPUT_DIR/test3_overwrite" | head -1)
    FILE_COUNT=$(ls "$OUTPUT_DIR/test3_overwrite/$TIMESTAMP_DIR/rank_0"/*.npy 2>/dev/null | wc -l)
    echo "Files after 3 iters: $FILE_COUNT (should be same as 1 iter - files overwritten)"

    # 4. Multi-rank test
    echo ""
    echo "=== 4. Multi-rank test ==="
    CMD="$BIN $MODEL_ARGS --nthread_per_process 8 --tensor_parallel 4 --pipeline_parallel 2 --precision_check \"level=1,path=$OUTPUT_DIR/test4_multi,save_tensors=true\" --num_iteration 1"
    echo "Running: $CMD"
    eval $CMD

    # 5. Comparison test (same-framework)
    echo ""
    echo "=== 5. Comparison test (same-framework) ==="
    CMD="$BIN $MODEL_ARGS --nthread_per_process 8 --tensor_parallel 4 --pipeline_parallel 2 --precision_check \"level=1,path=$OUTPUT_DIR/test5_compare_run1,save_tensors=true\" --num_iteration 1"
    echo "Running: $CMD"
    eval $CMD
    sleep 2
    CMD="$BIN $MODEL_ARGS --nthread_per_process 8 --tensor_parallel 4 --pipeline_parallel 2 --precision_check \"level=1,path=$OUTPUT_DIR/test5_compare_run2,save_tensors=true\" --num_iteration 1"
    echo "Running: $CMD"
    eval $CMD

    RUN1_DIR="$OUTPUT_DIR/test5_compare_run1/$(ls -t "$OUTPUT_DIR/test5_compare_run1" | head -1)"
    RUN2_DIR="$OUTPUT_DIR/test5_compare_run2/$(ls -t "$OUTPUT_DIR/test5_compare_run2" | head -1)"

    echo "Comparing two runs (rank_0):"
    python "$COMPARE_SCRIPT" --dir1 "$RUN1_DIR/rank_0" --dir2 "$RUN2_DIR/rank_0" --atol 1e-5 --rtol 1e-3 || true

    if [ -d "$RUN1_DIR/rank_1" ] && [ -d "$RUN2_DIR/rank_1" ]; then
        echo ""
        echo "Comparing two runs (rank_1):"
        python "$COMPARE_SCRIPT" --dir1 "$RUN1_DIR/rank_1" --dir2 "$RUN2_DIR/rank_1" --atol 1e-5 --rtol 1e-3 || true
    fi

    # 6. Error detection test
    echo ""
    echo "=== 6. Error detection test ==="
    CORRUPTED_DIR="$OUTPUT_DIR/test6_error"
    mkdir -p "$CORRUPTED_DIR/rank_0"
    cp "$RUN1_DIR/rank_0"/*.npy "$CORRUPTED_DIR/rank_0/" 2>/dev/null || true

    python3 -c "
import numpy as np
import glob
files = glob.glob('$CORRUPTED_DIR/rank_0/*.npy')
if files:
    f = files[0]
    arr = np.load(f)
    arr = arr + np.random.randn(*arr.shape).astype(arr.dtype) * 0.1
    np.save(f, arr)
    print(f'Corrupted: {f}')
"

    if python "$COMPARE_SCRIPT" --dir1 "$RUN1_DIR" --dir2 "$CORRUPTED_DIR" --atol 1e-5 --rtol 1e-3; then
        echo "ERROR: compare.py failed to detect corrupted tensor!"
    else
        echo "OK: compare.py correctly detected corrupted tensor"
    fi

    # 7. Cross-framework comparison
    echo ""
    echo "=== 7. Cross-framework comparison (InfiniTrain vs PyTorch) ==="
    SINGLE_RANK_DIR="$OUTPUT_DIR/test1_simple/$(ls -t "$OUTPUT_DIR/test1_simple" | head -1)/rank_0"

    if [ -d "$PYTORCH_DIR" ] && [ "$(ls -A "$PYTORCH_DIR" 2>/dev/null)" ]; then
        echo "PyTorch tensors directory: $PYTORCH_DIR"
        python "$COMPARE_SCRIPT" --dir1 "$SINGLE_RANK_DIR" --dir2 "$PYTORCH_DIR" --atol 1e-3 --rtol 1e-1 || true
    else
        echo "Skipping: PyTorch tensors directory not found ($PYTORCH_DIR)"
    fi

    echo ""
    echo "=== ${MODEL} Verification Complete ==="
    echo "Output directory: $OUTPUT_DIR"
}

# Main: loop through all models in config
echo "=== InfiniTrain Precision Checker ==="
echo "Config file: $CONFIG_FILE"

num_models=$(jq '.models | length' "$CONFIG_FILE")

for ((i=0; i<num_models; i++)); do
    MODEL=$(jq -r ".models[$i].name" "$CONFIG_FILE")
    INPUT_BIN=$(jq -r ".models[$i].input_bin" "$CONFIG_FILE")
    LLMC_FILEPATH=$(jq -r ".models[$i].llmc_filepath" "$CONFIG_FILE")
    PYTORCH_DIR=$(jq -r ".models[$i].pytorch_tensors_dir // empty" "$CONFIG_FILE")
    : "${PYTORCH_DIR:=../pytorch_tensors}"

    run_model_tests "$MODEL" "$INPUT_BIN" "$LLMC_FILEPATH" "$PYTORCH_DIR" || true
done

echo ""
echo "============================================================"
echo "=== All Models Complete ==="
echo "============================================================"
