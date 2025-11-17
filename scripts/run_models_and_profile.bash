#!/bin/bash

set -e
set -o pipefail

CONFIG_FILE="${1:-test_config.json}"

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
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local log_path="$(realpath "${LOG_DIR}/${log_name}.log")"

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

# Build "--key value" arg string from tests[i].args (shell-escaped)
args_string_for_test() {
    local idx="$1"
    jq -r --argjson i "$idx" '
      .tests[$i].args
      | to_entries[]
      | "--\(.key) \(.value|tostring)"
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

    # always clean before another build
    clean_build_dir
    run_and_log "$LAST_CMAKE_CMD" "${build_id}" "no"

    # profile flag for runs
    profile_flag="no"
    log_suffix=""
    if [[ "$build_profile" == "true" ]]; then
        profile_flag="yes"
        log_suffix="_profile"
    fi

    for ((ti=0; ti<num_tests; ++ti)); do
        test_id=$(jq -r ".tests[$ti].id" "$CONFIG_FILE")
        arg_str="$(args_string_for_test "$ti")"

        # FIXME(zbl): Use GROUP launch mode to ensure grouped for TP
        if [[ "$arg_str" == *"tensor_parallel"* ]]; then
            prefix="NCCL_LAUNCH_MODE=GROUP "
        else
            prefix=""
        fi

        # gpt2
        gpt2_cmd="${prefix}./gpt2 --input_bin ${GPT2_INPUT_BIN} --llmc_filepath ${GPT2_LLMC_FILEPATH} --device cuda ${arg_str}"
        run_and_log "$gpt2_cmd" "gpt2_${test_id}${log_suffix}" "$profile_flag"

        # llama3
        llama3_cmd="${prefix}./llama3 --input_bin ${LLAMA3_INPUT_BIN} --llmc_filepath ${LLAMA3_LLMC_FILEPATH} --device cuda ${arg_str}"
        run_and_log "$llama3_cmd" "llama3_${test_id}${log_suffix}" "$profile_flag"
    done
done

echo -e "\n\033[1;32mAll done.\033[0m"
