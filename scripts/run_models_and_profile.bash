#!/bin/bash

set -e
set -o pipefail

# ======== Set paths below ========
BUILD_DIR="../build"  # <== Set build directory here
GPT2_INPUT_BIN="../../data/llmc/gpt2/tinyshakespeare/tiny_shakespeare_train.bin"
GPT2_LLMC_FILEPATH="../../data/llmc/gpt2/gpt2_124M.bin"

LLAMA3_INPUT_BIN="../../data/llmc/llama3/tinyshakespeare/tiny_shakespeare_train.bin"
LLAMA3_LLMC_FILEPATH="../../data/llmc/llama3/llama3.1_1B_fp32.bin"

PROFILE_LOG_DIR="./profile_logs"
mkdir -p "$PROFILE_LOG_DIR"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# Global variable to save the last cmake command
LAST_CMAKE_CMD=""

# Clean the build directory
clean_build_dir() {
    echo -e "\033[1;31m[CLEAN] Removing all contents in: ${BUILD_DIR}\033[0m"
    mkdir -p "$BUILD_DIR"
    rm -rf "${BUILD_DIR:?}/"*
}

# Run a command and log output
run_and_log() {
    local cmd="$1"
    local log_name="$2"
    local is_profile="$3"
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local log_path="$(realpath "${LOG_DIR}/${log_name}.log")"

    echo -e "\033[1;32m============================================================\033[0m"
    echo -e "\033[1;36m[$timestamp] [Running] ${log_name}\033[0m"
    
    # Print the command being executed
    echo -e "\033[1;33mCommand:\033[0m $cmd"

    # ✅ Print the most recent CMake command
    if [[ -n "$LAST_CMAKE_CMD" ]]; then
        echo -e "\033[1;34mLast CMake Command:\033[0m $LAST_CMAKE_CMD"
    fi

    echo -e "\033[1;33mLog file:\033[0m $log_path"

    # Notify if profiling mode is enabled
    if [[ "$is_profile" == "yes" ]]; then
        echo -e "\033[1;35m[PROFILE MODE ON] Profiling logs will be saved to: ${PROFILE_LOG_DIR}\033[0m"
    fi

    echo -e "\033[1;32m============================================================\033[0m"

    pushd "$BUILD_DIR" > /dev/null

    # Write the last cmake command into the log file if available
    if [[ -n "$LAST_CMAKE_CMD" ]]; then
        echo "[LAST_CMAKE] $LAST_CMAKE_CMD" > "$log_path"
    else
        # If no cmake command has been run yet, clear the log
        > "$log_path"
    fi

    # Write the current run command to the log
    echo "[COMMAND] $cmd" >> "$log_path"

    # Run the command and append both stdout and stderr to the log file
    eval "$cmd" >> "$log_path" 2>&1

    popd > /dev/null

    # If profiling is enabled, move profiling files to the target directory
    if [[ "$is_profile" == "yes" ]]; then
        move_profile_logs "$log_name"
    fi
}


# Move profiling output logs
move_profile_logs() {
    local prefix="$1"

    # Move *.report.rankN files
    for report_file in "${BUILD_DIR}"/*.report.rank*; do
        if [[ -f "$report_file" ]]; then
            local base_name
            base_name=$(basename "$report_file")
            mv "$report_file" "${PROFILE_LOG_DIR}/${prefix}_${base_name}"
            echo "Moved $base_name to ${PROFILE_LOG_DIR}/${prefix}_${base_name}"
        fi
    done

    # Move *.records.log.rankN files
    for record_file in "${BUILD_DIR}"/*.records.log.rank*; do
        if [[ -f "$record_file" ]]; then
            local base_name
            base_name=$(basename "$record_file")
            mv "$record_file" "${PROFILE_LOG_DIR}/${prefix}_${base_name}"
            echo "Moved $base_name to ${PROFILE_LOG_DIR}/${prefix}_${base_name}"
        fi
    done
}


# ========== Block 1 (default config and dp with NCCL) ==========
# Float32
clean_build_dir
LAST_CMAKE_CMD="cmake -DUSE_CUDA=ON -DUSE_NCCL=ON .. && make -j"
run_and_log "$LAST_CMAKE_CMD" "build_1" "no"

run_and_log "./gpt2 --input_bin ${GPT2_INPUT_BIN} --llmc_filepath ${GPT2_LLMC_FILEPATH} --device cuda --dtype float32" "gpt2_1" "no"

run_and_log "./gpt2 --input_bin ${GPT2_INPUT_BIN} --llmc_filepath ${GPT2_LLMC_FILEPATH} --device cuda --batch_size 80 --total_batch_size 5120 --num_iteration 10 --dtype float32" "gpt2_2" "no"

run_and_log "./gpt2 --input_bin ${GPT2_INPUT_BIN} --llmc_filepath ${GPT2_LLMC_FILEPATH} --device cuda --nthread_per_process 8 --batch_size 10 --total_batch_size 5120 --num_iteration 10 --dtype float32" "gpt2_3" "no"

run_and_log "./llama3 --input_bin ${LLAMA3_INPUT_BIN} --llmc_filepath ${LLAMA3_LLMC_FILEPATH} --device cuda --dtype float32" "llama3_1" "no"

run_and_log "./llama3 --input_bin ${LLAMA3_INPUT_BIN} --llmc_filepath ${LLAMA3_LLMC_FILEPATH} --device cuda --batch_size 80 --total_batch_size 5120 --num_iteration 10 --dtype float32" "llama3_2" "no"

run_and_log "./llama3 --input_bin ${LLAMA3_INPUT_BIN} --llmc_filepath ${LLAMA3_LLMC_FILEPATH} --device cuda --nthread_per_process 8 --batch_size 10 --total_batch_size 5120 --num_iteration 10 --dtype float32" "llama3_3" "no"

# BFloat16
clean_build_dir
LAST_CMAKE_CMD="cmake -DUSE_CUDA=ON -DUSE_NCCL=ON .. && make -j"
run_and_log "$LAST_CMAKE_CMD" "build_1_bfloat16" "no"

run_and_log "./gpt2 --input_bin ${GPT2_INPUT_BIN} --llmc_filepath ${GPT2_LLMC_FILEPATH} --device cuda --dtype bfloat16" "gpt2_1_bfloat16" "no"

run_and_log "./gpt2 --input_bin ${GPT2_INPUT_BIN} --llmc_filepath ${GPT2_LLMC_FILEPATH} --device cuda --batch_size 80 --total_batch_size 5120 --num_iteration 10 --dtype bfloat16" "gpt2_2_bfloat16" "no"

run_and_log "./gpt2 --input_bin ${GPT2_INPUT_BIN} --llmc_filepath ${GPT2_LLMC_FILEPATH} --device cuda --nthread_per_process 8 --batch_size 10 --total_batch_size 5120 --num_iteration 10 --dtype bfloat16" "gpt2_3_bfloat16" "no"

run_and_log "./llama3 --input_bin ${LLAMA3_INPUT_BIN} --llmc_filepath ${LLAMA3_LLMC_FILEPATH} --device cuda --dtype bfloat16" "llama3_1_bfloat16" "no"

run_and_log "./llama3 --input_bin ${LLAMA3_INPUT_BIN} --llmc_filepath ${LLAMA3_LLMC_FILEPATH} --device cuda --batch_size 80 --total_batch_size 5120 --num_iteration 10 --dtype bfloat16" "llama3_2_bfloat16" "no"

run_and_log "./llama3 --input_bin ${LLAMA3_INPUT_BIN} --llmc_filepath ${LLAMA3_LLMC_FILEPATH} --device cuda --nthread_per_process 8 --batch_size 10 --total_batch_size 5120 --num_iteration 10 --dtype bfloat16" "llama3_3_bfloat16" "no"

# ========== Block 2 (default config and dp with NCCL, Profiling Enabled) ==========
# Float32
clean_build_dir
LAST_CMAKE_CMD="cmake -DUSE_CUDA=ON -DUSE_NCCL=ON -DPROFILE_MODE=ON .. && make -j"
run_and_log "$LAST_CMAKE_CMD" "build_3" "no"

run_and_log "./gpt2 --input_bin ${GPT2_INPUT_BIN} --llmc_filepath ${GPT2_LLMC_FILEPATH} --device cuda --dtype float32" "gpt2_1_profile" "yes"

run_and_log "./gpt2 --input_bin ${GPT2_INPUT_BIN} --llmc_filepath ${GPT2_LLMC_FILEPATH} --device cuda --batch_size 80 --total_batch_size 5120 --num_iteration 10 --dtype float32" "gpt2_2_profile" "yes"

run_and_log "./gpt2 --input_bin ${GPT2_INPUT_BIN} --llmc_filepath ${GPT2_LLMC_FILEPATH} --device cuda --nthread_per_process 8 --batch_size 10 --total_batch_size 5120 --num_iteration 10 --dtype float32" "gpt2_3_profile" "yes"

run_and_log "./llama3 --input_bin ${LLAMA3_INPUT_BIN} --llmc_filepath ${LLAMA3_LLMC_FILEPATH} --device cuda --dtype float32" "llama3_1_profile" "yes"

run_and_log "./llama3 --input_bin ${LLAMA3_INPUT_BIN} --llmc_filepath ${LLAMA3_LLMC_FILEPATH} --device cuda --batch_size 80 --total_batch_size 5120 --num_iteration 10 --dtype float32" "llama3_2_profile" "yes"

run_and_log "./llama3 --input_bin ${LLAMA3_INPUT_BIN} --llmc_filepath ${LLAMA3_LLMC_FILEPATH} --device cuda --nthread_per_process 8 --batch_size 10 --total_batch_size 5120 --num_iteration 10 --dtype float32" "llama3_3_profile" "yes"

# BFloat16
clean_build_dir
LAST_CMAKE_CMD="cmake -DUSE_CUDA=ON -DUSE_NCCL=ON -DPROFILE_MODE=ON .. && make -j"
run_and_log "$LAST_CMAKE_CMD" "build_3_bfloat16" "no"

run_and_log "./gpt2 --input_bin ${GPT2_INPUT_BIN} --llmc_filepath ${GPT2_LLMC_FILEPATH} --device cuda --dtype bfloat16" "gpt2_1_bfloat16_profile" "yes"

run_and_log "./gpt2 --input_bin ${GPT2_INPUT_BIN} --llmc_filepath ${GPT2_LLMC_FILEPATH} --device cuda --batch_size 80 --total_batch_size 5120 --num_iteration 10 --dtype bfloat16" "gpt2_2_bfloat16_profile" "yes"

run_and_log "./gpt2 --input_bin ${GPT2_INPUT_BIN} --llmc_filepath ${GPT2_LLMC_FILEPATH} --device cuda --nthread_per_process 8 --batch_size 10 --total_batch_size 5120 --num_iteration 10 --dtype bfloat16" "gpt2_3_bfloat16_profile" "yes"

run_and_log "./llama3 --input_bin ${LLAMA3_INPUT_BIN} --llmc_filepath ${LLAMA3_LLMC_FILEPATH} --device cuda --dtype bfloat16" "llama3_1_bfloat16_profile" "yes"

run_and_log "./llama3 --input_bin ${LLAMA3_INPUT_BIN} --llmc_filepath ${LLAMA3_LLMC_FILEPATH} --device cuda --batch_size 80 --total_batch_size 5120 --num_iteration 10 --dtype bfloat16" "llama3_2_bfloat16_profile" "yes"

run_and_log "./llama3 --input_bin ${LLAMA3_INPUT_BIN} --llmc_filepath ${LLAMA3_LLMC_FILEPATH} --device cuda --nthread_per_process 8 --batch_size 10 --total_batch_size 5120 --num_iteration 10 --dtype bfloat16" "llama3_3_bfloat16_profile" "yes"
