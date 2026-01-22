#!/bin/bash
# InfiniTrain Precision Checker - GPT2
set -e

# Configuration
BIN="./build/gpt2"
MODEL_ARGS="--device cuda --input_bin /data/shared/InfiniTrain-dev/data/llmc/gpt2/tinyshakespeare/tiny_shakespeare_train.bin --llmc_filepath /data/shared/InfiniTrain-dev/data/llmc/gpt2/gpt2_124M.bin"
OUTPUT_DIR="./log_precision_check_gpt2"
COMPARE_SCRIPT="scripts/precision_check/precision_compare.py"

echo "=== InfiniTrain Precision Checker - GPT2 ==="

if [ ! -f "$BIN" ]; then
    echo "Error: $BIN not found. Please build the project first."
    exit 1
fi

if [ ! -f "$COMPARE_SCRIPT" ]; then
    echo "Error: $COMPARE_SCRIPT not found."
    exit 1
fi

# Clean test directory
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# 1. Single-rank test - Simple format
echo ""
echo "=== 1. Single-rank test (Simple format) ==="
CMD="$BIN $MODEL_ARGS --precision_check \"level=1,path=$OUTPUT_DIR/test1,format=simple,save_tensors=true\" --num_iteration 1"
echo "Running: $CMD"
eval $CMD

TIMESTAMP_DIR=$(ls -t "$OUTPUT_DIR/test1" | head -1)
echo "Timestamp directory: $TIMESTAMP_DIR"
NPY_COUNT=$(ls "$OUTPUT_DIR/test1/$TIMESTAMP_DIR/rank_0"/*.npy 2>/dev/null | wc -l)
echo "Rank 0 NPY files: $NPY_COUNT"
LOG_FILE=$(ls "$OUTPUT_DIR/test1/$TIMESTAMP_DIR"/*.log 2>/dev/null | head -1)
echo "Log file: $LOG_FILE"

# 2. Single-rank test - MD5 format
echo ""
echo "=== 2. Single-rank test (MD5 format) ==="
CMD="$BIN $MODEL_ARGS --precision_check \"level=1,path=$OUTPUT_DIR/test2,format=md5\" --num_iteration 1"
echo "Running: $CMD"
eval $CMD

TIMESTAMP_DIR=$(ls -t "$OUTPUT_DIR/test2" | head -1)

# 3. Multi-iter overwrite test
echo ""
echo "=== 3. Multi-iter overwrite test ==="
CMD="$BIN $MODEL_ARGS --precision_check \"level=1,path=$OUTPUT_DIR/test3,save_tensors=true\" --num_iteration 3"
echo "Running: $CMD"
eval $CMD

TIMESTAMP_DIR=$(ls -t "$OUTPUT_DIR/test3" | head -1)
FILE_COUNT=$(ls "$OUTPUT_DIR/test3/$TIMESTAMP_DIR/rank_0"/*.npy 2>/dev/null | wc -l)
echo "Files after 3 iters: $FILE_COUNT (should be same as 1 iter - files overwritten)"

# 4. Comparison test
echo ""
echo "=== 4. Comparison test ==="
CMD="$BIN $MODEL_ARGS --precision_check \"level=1,path=$OUTPUT_DIR/run1,save_tensors=true\" --num_iteration 1"
echo "Running: $CMD"
eval $CMD
sleep 2
CMD="$BIN $MODEL_ARGS --precision_check \"level=1,path=$OUTPUT_DIR/run2,save_tensors=true\" --num_iteration 1"
echo "Running: $CMD"
eval $CMD

RUN1_DIR="$OUTPUT_DIR/run1/$(ls -t "$OUTPUT_DIR/run1" | head -1)"
RUN2_DIR="$OUTPUT_DIR/run2/$(ls -t "$OUTPUT_DIR/run2" | head -1)"

echo "Comparing directories:"
echo "  Run1: $RUN1_DIR"
echo "  Run2: $RUN2_DIR"

python "$COMPARE_SCRIPT" --dir1 "$RUN1_DIR" --dir2 "$RUN2_DIR" --atol 1e-5 --rtol 1e-3 || true

# 5. Multi-rank test (if available)
echo ""
echo "=== 5. Multi-rank test ==="
CMD="$BIN $MODEL_ARGS --nthread_per_process 8 --tensor_parallel 4 --pipeline_parallel 2 --precision_check \"level=1,path=$OUTPUT_DIR/test_multi,save_tensors=true\" --num_iteration 1"
echo "Running: $CMD"
eval $CMD

echo ""
echo "=== Verification Complete ==="
echo "Test output directory: $OUTPUT_DIR"
