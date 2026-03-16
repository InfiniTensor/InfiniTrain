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

BUILD_DIR="$(read_var BUILD_DIR)";                  : "${BUILD_DIR:=../build}"
LOG_DIR="$(read_var LOG_DIR)";                      : "${LOG_DIR:=logs}"
PROFILE_LOG_DIR="$(read_var PROFILE_LOG_DIR)";      : "${PROFILE_LOG_DIR:=./profile_logs}"
COMPARE_LOG_DIR="$(read_var COMPARE_LOG_DIR)";      : "${COMPARE_LOG_DIR:=}"
FLASH="$(read_var FLASH)";                          : "${FLASH:=}"

# --- 关键修改 1: 初始化大容量分区的绝对路径临时目录 ---
# 先确保 build 目录存在，以便获取其绝对路径
mkdir -p "$BUILD_DIR" "$LOG_DIR" "$PROFILE_LOG_DIR"
# 获取绝对路径，防止 CMake 切换目录后找不到相对路径
export CUSTOM_TMP="$(readlink -f "$BUILD_DIR")/tmp_cache"
mkdir -p "$CUSTOM_TMP"
export TMPDIR="$CUSTOM_TMP"

# export custom PATHs
export BUILD_DIR LOG_DIR PROFILE_LOG_DIR
while IFS="=" read -r k v; do
    [[ -z "$k" || "$k" == "null" ]] && continue
    export "$k"="$v"
done < <(jq -r '.variables | to_entries[] | "\(.key)=\(.value)"' "$CONFIG_FILE")

# Global variable to save the last cmake command
LAST_CMAKE_CMD=""

# --- 关键修改 2: 在清理函数中重新创建临时目录 ---
clean_build_dir() {
    echo -e "\033[1;31m[CLEAN] Removing all contents in: ${BUILD_DIR}\033[0m"
    # 删除 build 下所有内容（这会删掉旧的 tmp_cache）
    rm -rf "${BUILD_DIR:?}/"*
    # 重新创建 build 目录
    mkdir -p "$BUILD_DIR"
    # 核心：必须重新创建 TMPDIR 目录，否则编译器的路径会失效
    mkdir -p "$TMPDIR"
    echo -e "\033[1;34m[TMP] Re-created temp space at: $TMPDIR\033[0m"
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
    echo -e "\033[1;33mCommand:\033[0m $cmd"
    if [[ -n "$LAST_CMAKE_CMD" ]]; then
        echo -e "\033[1;34mLast CMake Command:\033[0m $LAST_CMAKE_CMD"
    fi
    echo -e "\033[1;33mLog file:\033[0m $log_path"
    if [[ "$is_profile" == "yes" ]]; then
        echo -e "\033[1;35m[PROFILE MODE ON] Profiling logs will be saved to: ${PROFILE_LOG_DIR}\033[0m"
    fi
    echo -e "\033[1;32m============================================================\033[0m"

    pushd "$BUILD_DIR" > /dev/null

    if [[ -n "$LAST_CMAKE_CMD" ]]; then
        echo "[LAST_CMAKE] $LAST_CMAKE_CMD" > "$log_path"
    else
        > "$log_path"
    fi

    echo "[COMMAND] $cmd" >> "$log_path"

    # 执行命令并重定向输出
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

    if [[ "$is_profile" == "yes" ]]; then
        move_profile_logs "$log_name"
    fi
}

# Move profiling output logs
move_profile_logs() {
    local prefix="$1"
    for report_file in "${BUILD_DIR}"/*.report.rank*; do
        if [[ -f "$report_file" ]]; then
            local base_name=$(basename "$report_file")
            mv "$report_file" "${PROFILE_LOG_DIR}/${prefix}_${base_name}"
        fi
    done
    for record_file in "${BUILD_DIR}"/*.records.log.rank*; do
        if [[ -f "$record_file" ]]; then
            local base_name=$(basename "$record_file")
            mv "$record_file" "${PROFILE_LOG_DIR}/${prefix}_${base_name}"
        fi
    done
}

# Build args string
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

    # 调用修改后的清理函数
    clean_build_dir
    run_and_log "$LAST_CMAKE_CMD" "${build_id}" "no"

    profile_flag="no"
    log_suffix=""
    if [[ "$build_profile" == "true" ]]; then
        profile_flag="yes"
        log_suffix="_profile"
    fi

    for ((ti=0; ti<num_tests; ++ti)); do
        test_id=$(jq -r ".tests[$ti].id" "$CONFIG_FILE")
        arg_str="$(args_string_for_test "$ti")"
        global_flash_arg=""
        if [[ -n "$FLASH" ]]; then
            global_flash_arg="--flash=${FLASH}"
        fi

        gpt2_cmd="./gpt2 --input_bin ${GPT2_INPUT_BIN} --llmc_filepath ${GPT2_LLMC_FILEPATH} --device cuda ${global_flash_arg} ${arg_str}"
        run_and_log "$gpt2_cmd" "gpt2_${test_id}${log_suffix}" "$profile_flag"

        llama3_cmd="./llama3 --input_bin ${LLAMA3_INPUT_BIN} --llmc_filepath ${LLAMA3_LLMC_FILEPATH} --device cuda ${global_flash_arg} ${arg_str}"
        run_and_log "$llama3_cmd" "llama3_${test_id}${log_suffix}" "$profile_flag"
    done
done

echo -e "\n\033[1;32mAll done.\033[0m"

# Comparison part
if [[ -n "$COMPARE_LOG_DIR" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    echo -e "\n\033[1;36m[Comparison] Comparing logs...\033[0m"
    python3 "${SCRIPT_DIR}/compare_loss.py" "$COMPARE_LOG_DIR" "$LOG_DIR" || true
    python3 "${SCRIPT_DIR}/compare_tps.py" "$COMPARE_LOG_DIR" "$LOG_DIR" || true
    echo -e "\n\033[1;32mComparison completed.\033[0m"
else
    echo -e "\n\033[1;33m[WARNING] COMPARE_LOG_DIR is not set. Skipping comparison.\033[0m"
fi