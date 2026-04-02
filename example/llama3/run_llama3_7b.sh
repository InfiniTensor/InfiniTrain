#!/bin/bash
set -e

# Minimum reproducible script for LLaMA-3 7B (or scaled down for testing)
# Requires data paths to be set via env vars or arguments
# DATA_ROOT should point to where tinyshakespeare and tokenizer are

DATA_ROOT=${DATA_ROOT:-"/data/shared/InfiniTrain-dev/data/llmc/llama3"}
INPUT_BIN="${DATA_ROOT}/tinyshakespeare/tiny_shakespeare_train.bin"
TOKENIZER_BIN="${DATA_ROOT}/llama3_tokenizer.bin"
# Default to small model for CI/Testing if 7B not available
MODEL_BIN="${DATA_ROOT}/llama3.2_1B_fp32.bin"

# Probe environment first
python3 probe_env.py

# Check if model exists
if [ ! -f "$MODEL_BIN" ]; then
    echo "Model binary not found at $MODEL_BIN. Please set DATA_ROOT correctly."
    exit 1
fi

echo "Running LLaMA-3 with FlashAttention..."
# Use safe defaults: SeqLen=1024 (fit in 16GB), Batch=1
# Force CUDA visible devices if needed
# export CUDA_VISIBLE_DEVICES=0

./build/llama3 \
    -input_bin="$INPUT_BIN" \
    -tokenizer_bin="$TOKENIZER_BIN" \
    -llmc_filepath="$MODEL_BIN" \
    -flash=true \
    -sequence_length=1024 \
    -num_iteration=10 \
    -batch_size=1 \
    -total_batch_size=1024 \
    -overfit_single_batch=false \
    -freq_generate_txt=9999 \
    > llama3_flash.log 2>&1

if [ $? -eq 0 ]; then
    echo "LLaMA-3 FlashAttention Run Successful."
else
    echo "LLaMA-3 FlashAttention Run Failed. Check llama3_flash.log."
    # Fallback logic handled inside binary (CPU fallback) or here?
    # User requirement 3.3 says "automatically fallback to eager mode ... export FLASH_ATTENTION_FORCE_DISABLE_TRITON=1"
    # Our C++ binary implementation might need to read this env var or argument.
    # But since we already implemented CPU fallback in binary, maybe this script just reports failure.
    
    # Let's try eager mode explicitly if flash failed
    echo "Attempting Fallback to Eager Mode..."
    echo "Reason: Flash Run Failed" >> fallback.log
    
    ./build/llama3 \
        -input_bin="$INPUT_BIN" \
        -tokenizer_bin="$TOKENIZER_BIN" \
        -llmc_filepath="$MODEL_BIN" \
        -flash=false \
        -sequence_length=1024 \
        -num_iteration=10 \
        -batch_size=1 \
        -total_batch_size=1024 \
        > llama3_eager.log 2>&1
        
    if [ $? -eq 0 ]; then
        echo "LLaMA-3 Eager Mode Run Successful."
    else
        echo "LLaMA-3 Eager Mode also Failed."
        exit 1
    fi
fi
