#!/bin/bash

set -e
set -o pipefail

usage() {
    cat <<'EOF'
Usage: run_models_and_profile.bash [--test-config path] [--only-run tag1,tag2]

Options:
  --test-config PATH  Path to test config JSON. Default: test_config.json.
  --only-run TAGS   Only run the specified tag groups, separated by commas.
  -h, --help        Show this help message.
EOF
}

CONFIG_FILE="test_config.json"
ONLY_RUN_TAGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --test-config)
            [[ $# -lt 2 ]] && { echo "Error: --test-config requires a file path."; exit 1; }
            CONFIG_FILE="$2"
            shift 2
            ;;
        --test-config=*)
            CONFIG_FILE="${1#*=}"
            shift
            ;;
        --only-run)
            [[ $# -lt 2 ]] && { echo "Error: --only-run requires a comma-separated tag list."; exit 1; }
            ONLY_RUN_TAGS="$2"
            shift 2
            ;;
        --only-run=*)
            ONLY_RUN_TAGS="${1#*=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "Error: Unknown option: $1"
            usage
            exit 1
            ;;
        *)
            echo "Error: Unknown positional argument: $1"
            usage
            exit 1
            ;;
    esac
done

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
declare -A SELECTED_TAGS=()

normalize_tag() {
    local raw="$1"
    raw="${raw#"${raw%%[![:space:]]*}"}"
    raw="${raw%"${raw##*[![:space:]]}"}"
    printf '%s' "$raw"
}

if [[ -n "$ONLY_RUN_TAGS" ]]; then
    IFS=',' read -r -a requested_tags <<< "$ONLY_RUN_TAGS"
    for raw_tag in "${requested_tags[@]}"; do
        tag="$(normalize_tag "$raw_tag")"
        [[ -z "$tag" ]] && continue
        SELECTED_TAGS["$tag"]=1
    done

    if [[ ${#SELECTED_TAGS[@]} -eq 0 ]]; then
        echo "Error: --only-run did not contain any valid tags."
        exit 1
    fi
fi

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
    local tag="${4:-basic}"
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local tag_log_dir="${LOG_DIR}/${tag}"
    mkdir -p "$tag_log_dir"
    local log_path="$(realpath "${tag_log_dir}/${log_name}.log")"

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
        move_profile_logs "$log_name" "$tag"
    fi
}


# Move profiling output logs
move_profile_logs() {
    local prefix="$1"
    local tag="${2:-basic}"
    local tag_profile_dir="${PROFILE_LOG_DIR}/${tag}"
    mkdir -p "$tag_profile_dir"

    # Move *.report.rankN files
    for report_file in "${BUILD_DIR}"/*.report.rank*; do
        if [[ -f "$report_file" ]]; then
            local base_name
            base_name=$(basename "$report_file")
            mv "$report_file" "${tag_profile_dir}/${prefix}_${base_name}"
            echo "Moved $base_name to ${tag_profile_dir}/${prefix}_${base_name}"
        fi
    done

    # Move *.records.log.rankN files
    for record_file in "${BUILD_DIR}"/*.records.log.rank*; do
        if [[ -f "$record_file" ]]; then
            local base_name
            base_name=$(basename "$record_file")
            mv "$record_file" "${tag_profile_dir}/${prefix}_${base_name}"
            echo "Moved $base_name to ${tag_profile_dir}/${prefix}_${base_name}"
        fi
    done
}

# Build "--key value" arg string from test_groups[gi].tests[ti].args (shell-escaped)
args_string_for_test() {
    local group_idx="$1"
    local test_idx="$2"
    jq -r --argjson g "$group_idx" --argjson t "$test_idx" '
      .test_groups[$g].tests[$t].args
      | to_entries[]
      | "--\(.key)=\(.value|tostring)"
    ' "$CONFIG_FILE" | paste -sd' ' -
}

# Run tests
num_builds=$(jq '.builds | length' "$CONFIG_FILE")
num_groups=$(jq '.test_groups | length' "$CONFIG_FILE")

selected_group_count=0
for ((gi=0; gi<num_groups; ++gi)); do
    group_tag=$(jq -r ".test_groups[$gi].tag" "$CONFIG_FILE")
    if [[ ${#SELECTED_TAGS[@]} -eq 0 || -n "${SELECTED_TAGS[$group_tag]}" ]]; then
        ((selected_group_count += 1))
    fi
done

if [[ "$selected_group_count" -eq 0 ]]; then
    echo "Error: No matching test groups found for --only-run=${ONLY_RUN_TAGS}"
    exit 1
fi

for ((id=0; id<num_builds; ++id)); do
    build_id=$(jq -r ".builds[$id].id" "$CONFIG_FILE")
    build_profile=$(jq -r ".builds[$id].profile" "$CONFIG_FILE")
    build_cmake=$(jq -r ".builds[$id].cmd" "$CONFIG_FILE")

    LAST_CMAKE_CMD="$build_cmake"

    # always clean before another build
    clean_build_dir
    run_and_log "$LAST_CMAKE_CMD" "${build_id}" "no" "build"

    # profile flag for runs
    profile_flag="no"
    log_suffix=""
    if [[ "$build_profile" == "true" ]]; then
        profile_flag="yes"
        log_suffix="_profile"
    fi

    for ((gi=0; gi<num_groups; ++gi)); do
        group_tag=$(jq -r ".test_groups[$gi].tag" "$CONFIG_FILE")
        if [[ ${#SELECTED_TAGS[@]} -gt 0 && -z "${SELECTED_TAGS[$group_tag]}" ]]; then
            continue
        fi

        num_tests=$(jq ".test_groups[$gi].tests | length" "$CONFIG_FILE")
        echo -e "\033[1;36m[TEST GROUP] tag=${group_tag}, cases=${num_tests}\033[0m"

        for ((ti=0; ti<num_tests; ++ti)); do
            test_id=$(jq -r ".test_groups[$gi].tests[$ti].id" "$CONFIG_FILE")
            arg_str="$(args_string_for_test "$gi" "$ti")"

            # gpt2
            gpt2_cmd="${prefix}./gpt2 --input_bin ${GPT2_INPUT_BIN} --llmc_filepath ${GPT2_LLMC_FILEPATH} --device cuda ${arg_str}"
            run_and_log "$gpt2_cmd" "gpt2_${test_id}${log_suffix}" "$profile_flag" "$group_tag"

            # llama3
            llama3_cmd="${prefix}./llama3 --input_bin ${LLAMA3_INPUT_BIN} --llmc_filepath ${LLAMA3_LLMC_FILEPATH} --device cuda ${arg_str}"
            run_and_log "$llama3_cmd" "llama3_${test_id}${log_suffix}" "$profile_flag" "$group_tag"
        done
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
    python3 "${SCRIPT_DIR}/compare_loss.py" "$COMPARE_LOG_DIR" "$LOG_DIR" || true

    # Run compare_tps.py
    echo -e "\n\033[1;33m[Running] compare_tps.py\033[0m"
    python3 "${SCRIPT_DIR}/compare_tps.py" "$COMPARE_LOG_DIR" "$LOG_DIR" || true

    echo -e "\n\033[1;32mComparison completed.\033[0m"
else
    echo -e "\n\033[1;33m============================================================\033[0m"
    echo -e "\033[1;33m[WARNING] COMPARE_LOG_DIR is not set. Skipping comparison.\033[0m"
    echo -e "\033[1;33m         To enable comparison, set 'variables.COMPARE_LOG_DIR' in ${CONFIG_FILE}\033[0m"
    echo -e "\033[1;33m         or export COMPARE_LOG_DIR=/path/to/baseline_logs before running.\033[0m"
    echo -e "\033[1;33m============================================================\033[0m"
fi

echo -e "\n\033[1;36m[END OF TEST] Cleaning build directory after all tests\033[0m"
clean_build_dir
