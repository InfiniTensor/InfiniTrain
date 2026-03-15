#!/bin/bash

set -e
set -o pipefail

# Parse arguments
REBUILD=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --rebuild)
            REBUILD=true
            shift
            ;;
        *)
            CONFIG_FILE="$1"
            shift
            ;;
    esac
done

CONFIG_FILE="${CONFIG_FILE:-test_config.json}"
export INFINI_FLASH_BF16_USE_FP32=0

# Dependencies check
if ! command -v jq >/dev/null 2>&1; then
    echo "Error: jq is required. Install with: sudo apt-get install -y jq"
    exit 1
fi

# Read variables
read_var() {
    local key="$1"
    jq -r --arg k "$key" '.variables[$k] // empty' "$CONFIG_FILE"
}

BUILD_DIR="$(read_var BUILD_DIR)";              : "${BUILD_DIR:=../build}"
LOG_DIR="$(read_var LOG_DIR)";                  : "${LOG_DIR:=logs}"
PROFILE_LOG_DIR="$(read_var PROFILE_LOG_DIR)";  : "${PROFILE_LOG_DIR:=./profile_logs}"
COMPARE_LOG_DIR="$(read_var COMPARE_LOG_DIR)";  : "${COMPARE_LOG_DIR:=}"

mkdir -p "$BUILD_DIR" "$LOG_DIR" "$PROFILE_LOG_DIR"

# export custom PATHs
export BUILD_DIR LOG_DIR PROFILE_LOG_DIR
while IFS="=" read -r k v; do
    [[ -z "$k" || "$k" == "null" ]] && continue
    export "$k"="$v"
done < <(jq -r '.variables | to_entries[] | "\(.key)=\(.value)"' "$CONFIG_FILE")

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
    local log_dir="$4"
    local profile_log_dir="$5"
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local log_path="$(realpath "${log_dir}/${log_name}.log")"

    echo -e "\033[1;32m============================================================\033[0m"
    echo -e "\033[1;36m[$timestamp] [Running] ${log_name}\033[0m"
    
    # Print the command being executed
    echo -e "\033[1;33mCommand:\033[0m $cmd"

    # Print the most recent CMake command
    if [[ -n "$LAST_CMAKE_CMD" ]]; then
        echo -e "\033[1;34mLast CMake Command:\033[0m $LAST_CMAKE_CMD"
    fi

    echo -e "\033[1;33mLog file:\033[0m $log_path"

    # Notify if profiling mode is enabled
    if [[ "$is_profile" == "yes" ]]; then
        echo -e "\033[1;35m[PROFILE MODE ON] Profiling logs will be saved to: ${profile_log_dir}\033[0m"
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
    if ! eval "$cmd" >> "$log_path" 2>&1; then
        echo -e "\033[1;31m============================================================\033[0m"
        echo -e "\033[1;31m[ERROR] Command failed: ${cmd}\033[0m"
        echo -e "\033[1;31m[ERROR] See log file for details: ${log_path}\033[0m"
        echo -e "\033[1;31m============================================================\033[0m"
        echo ""
        echo "[ERROR] Last 20 lines of log:"
        tail -20 "$log_path"
        exit 1
    fi

    popd > /dev/null

    # If profiling is enabled, move profiling files to the target directory
    if [[ "$is_profile" == "yes" ]]; then
        move_profile_logs "$log_name" "$profile_log_dir"
    fi
}


# Move profiling output logs
move_profile_logs() {
    local prefix="$1"
    local target_profile_log_dir="$2"

    # Move *.report.rankN files
    for report_file in "${BUILD_DIR}"/*.report.rank*; do
        if [[ -f "$report_file" ]]; then
            local base_name
            base_name=$(basename "$report_file")
            mv "$report_file" "${target_profile_log_dir}/${prefix}_${base_name}"
            echo "Moved $base_name to ${target_profile_log_dir}/${prefix}_${base_name}"
        fi
    done

    # Move *.records.log.rankN files
    for record_file in "${BUILD_DIR}"/*.records.log.rank*; do
        if [[ -f "$record_file" ]]; then
            local base_name
            base_name=$(basename "$record_file")
            mv "$record_file" "${target_profile_log_dir}/${prefix}_${base_name}"
            echo "Moved $base_name to ${target_profile_log_dir}/${prefix}_${base_name}"
        fi
    done
}

# Build "--key value" arg string from tests[i].args (shell-escaped)
args_string_for_test() {
    local idx="$1"
    jq -r --argjson i "$idx" '
      .tests[$i].args
      | to_entries[]
      | if .value == true then "--\(.key)"
        elif .value == false then "--no\(.key)"
        else "--\(.key)=\(.value|tostring)"
        end
    ' "$CONFIG_FILE" | paste -sd' ' -
}

# Run tests
num_builds=$(jq '.builds | length' "$CONFIG_FILE")
num_tests=$(jq '.tests  | length' "$CONFIG_FILE")

for ((id=0; id<num_builds; ++id)); do
    build_id=$(jq -r ".builds[$id].id" "$CONFIG_FILE")
    build_profile=$(jq -r ".builds[$id].profile" "$CONFIG_FILE")
    build_cmake=$(jq -r ".builds[$id].cmd" "$CONFIG_FILE")

    LAST_CMAKE_CMD="$build_cmake"

    # Check if rebuild is needed
    if [[ "$REBUILD" == true ]]; then
        # Clean and rebuild
        clean_build_dir
        run_and_log "$LAST_CMAKE_CMD" "${build_id}" "no" "$LOG_DIR" "$PROFILE_LOG_DIR"
    else
        # Check if build directory exists and executables are present
        if [[ -d "$BUILD_DIR" ]]; then
            # Check if gpt2 and llama3 executables exist
            if [[ -f "${BUILD_DIR}/gpt2" ]] && [[ -f "${BUILD_DIR}/llama3" ]]; then
                echo -e "\033[1;33m[SKIP] Build directory already exists and executables are present. Skipping build.\033[0m"
                echo -e "\033[1;33m        Use --rebuild to force a clean rebuild.\033[0m"
            else
                # Build executables that are missing
                echo -e "\033[1;33m[BUILD] Some executables are missing. Building...\033[0m"
                run_and_log "$LAST_CMAKE_CMD" "${build_id}" "no" "$LOG_DIR" "$PROFILE_LOG_DIR"
            fi
        else
            # Build directory doesn't exist, build from scratch
            echo -e "\033[1;33m[BUILD] Build directory doesn't exist. Building...\033[0m"
            run_and_log "$LAST_CMAKE_CMD" "${build_id}" "no" "$LOG_DIR" "$PROFILE_LOG_DIR"
        fi
    fi

    # profile flag for runs
    profile_flag="no"
    log_suffix=""
    if [[ "$build_profile" == "true" ]]; then
        profile_flag="yes"
        log_suffix="_profile"
    fi

    for ((ti=0; ti<num_tests; ++ti)); do
        test_id=$(jq -r ".tests[$ti].id" "$CONFIG_FILE")
        base_arg_str="$(args_string_for_test "$ti")"

        # Add --noflash to the beginning of arg_str (will override any existing flash arg)
        arg_str="--noflash $base_arg_str"

        # Add --flash to the beginning of arg_str (will override any existing flash arg)
        arg_str1="--flash $base_arg_str"

        # Run gpt2 with flash=false
        LOG_DIR="$(read_var LOG_DIR)"
        : "${LOG_DIR:=./logs}"
        PROFILE_LOG_DIR="$(read_var PROFILE_LOG_DIR)"
        : "${PROFILE_LOG_DIR:=./profile_logs}"

        mkdir -p "$LOG_DIR" "$PROFILE_LOG_DIR"

        gpt2_cmd="./gpt2 --input_bin ${GPT2_INPUT_BIN} --llmc_filepath ${GPT2_LLMC_FILEPATH} --device cuda ${arg_str}"
        run_and_log "$gpt2_cmd" "gpt2_${test_id}${log_suffix}" "$profile_flag" "$LOG_DIR" "$PROFILE_LOG_DIR"

        # Run gpt2 with flash=true
        COMPARE_LOG_DIR="./compare_logs"
        COMPARE_PROFILE_LOG_DIR="./compare_profile_logs"

        mkdir -p "$COMPARE_LOG_DIR" "$COMPARE_PROFILE_LOG_DIR"

        gpt2_cmd="./gpt2 --input_bin ${GPT2_INPUT_BIN} --llmc_filepath ${GPT2_LLMC_FILEPATH} --device cuda ${arg_str1}"
        run_and_log "$gpt2_cmd" "gpt2_${test_id}${log_suffix}" "$profile_flag" "$COMPARE_LOG_DIR" "$COMPARE_PROFILE_LOG_DIR"

        # Run llama3 with flash=false
        LOG_DIR="$(read_var LOG_DIR)"
        : "${LOG_DIR:=./logs}"
        PROFILE_LOG_DIR="$(read_var PROFILE_LOG_DIR)"
        : "${PROFILE_LOG_DIR:=./profile_logs}"

        mkdir -p "$LOG_DIR" "$PROFILE_LOG_DIR"

        llama3_cmd="./llama3 --input_bin ${LLAMA3_INPUT_BIN} --llmc_filepath ${LLAMA3_LLMC_FILEPATH} --device cuda ${arg_str}"
        run_and_log "$llama3_cmd" "llama3_${test_id}${log_suffix}" "$profile_flag" "$LOG_DIR" "$PROFILE_LOG_DIR"

        # Run llama3 with flash=true
        COMPARE_LOG_DIR="./compare_logs"
        COMPARE_PROFILE_LOG_DIR="./compare_profile_logs"

        mkdir -p "$COMPARE_LOG_DIR" "$COMPARE_PROFILE_LOG_DIR"

        llama3_cmd="./llama3 --input_bin ${LLAMA3_INPUT_BIN} --llmc_filepath ${LLAMA3_LLMC_FILEPATH} --device cuda ${arg_str1}"
        run_and_log "$llama3_cmd" "llama3_${test_id}${log_suffix}" "$profile_flag" "$COMPARE_LOG_DIR" "$COMPARE_PROFILE_LOG_DIR"
    done
done

echo -e "\n\033[1;32mAll done.\033[0m"

# Run comparison scripts if COMPARE_LOG_DIR is set
if [[ -n "$COMPARE_LOG_DIR" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    echo -e "\n\033[1;36m============================================================\033[0m"
    echo -e "\033[1;36m[Comparison] Comparing logs with: ${COMPARE_LOG_DIR}\033[0m"
    echo -e "\033[1;36m============================================================\033[0m"

    # Run compare_loss.py
    echo -e "\n\033[1;33m[Running] compare_loss.py\033[0m"
    python3 "${SCRIPT_DIR}/compare_loss.py" "$COMPARE_LOG_DIR" "$LOG_DIR" --threshold-fp32 1e-1 --threshold-bf16 1e-1 > compare_logs/loss_comparison.log 2>&1 || true

    # Run compare_tps.py
    echo -e "\n\033[1;33m[Running] compare_tps.py\033[0m"
    python3 "${SCRIPT_DIR}/compare_tps.py" "$COMPARE_LOG_DIR" "$LOG_DIR" --threshold 0.20 > compare_logs/tps_comparison.log 2>&1 || true

    echo -e "\n\033[1;32mComparison completed.\033[0m"
else
    echo -e "\n\033[1;33m============================================================\033[0m"
    echo -e "\033[1;33m[WARNING] COMPARE_LOG_DIR is not set. Skipping comparison.\033[0m"
    echo -e "\033[1;33m         To enable comparison, set 'variables.COMPARE_LOG_DIR' in ${CONFIG_FILE}\033[0m"
    echo -e "\033[1;33m         or export COMPARE_LOG_DIR=/path/to/baseline_logs before running.\033[0m"
    echo -e "\033[1;33m============================================================\033[0m"
fi
