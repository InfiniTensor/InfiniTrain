#!/usr/bin/env bash
set -euo pipefail

#------modify-start------------------------------------------
# Benchmark baseline vs --flash for GPT-2 and LLaMA-3.
# Outputs logs + a parsed markdown report under a local artifact directory.
#
# Usage:
#   bash scripts/flash_sdpa_benchmark.bash --gpu 5 --iters 30 --seq_len 256
#
# Notes:
# - If /tmp is small/full, we redirect TMPDIR to ~/tmp.
# - Uses shared tinyshakespeare bins if present under /data/shared/InfiniTrain-dev.
#---------modify-end-----------------------------------------

GPU=0
ITERS=20
SEQ_LEN=256
OUT_DIR="tmp/flash_sdpa"

GPT2_BIN_DEFAULT="/data/shared/InfiniTrain-dev/data/llmc/gpt2/tinyshakespeare/tiny_shakespeare_train.bin"
LLAMA3_BIN_DEFAULT="/data/shared/InfiniTrain-dev/data/llmc/llama3/tinyshakespeare/tiny_shakespeare_train.bin"
GPT2_BIN="$GPT2_BIN_DEFAULT"
LLAMA3_BIN="$LLAMA3_BIN_DEFAULT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2;;
    --iters) ITERS="$2"; shift 2;;
    --seq_len) SEQ_LEN="$2"; shift 2;;
    --out_dir) OUT_DIR="$2"; shift 2;;
    --gpt2_bin) GPT2_BIN="$2"; shift 2;;
    --llama3_bin) LLAMA3_BIN="$2"; shift 2;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"

mkdir -p "$REPO_ROOT/$OUT_DIR/logs"
mkdir -p "$REPO_ROOT/$OUT_DIR/env"
mkdir -p "$REPO_ROOT/tmp"

export TMPDIR="$REPO_ROOT/tmp"
export TMP="$REPO_ROOT/tmp"
export TEMP="$REPO_ROOT/tmp"
export CUDA_VISIBLE_DEVICES="$GPU"

ts="$(date +%Y%m%d_%H%M%S)"

env_file="$REPO_ROOT/$OUT_DIR/env/env_$ts.txt"
{
  echo "# Environment"
  echo "date: $(date)"
  echo "hostname: $(hostname)"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo
  echo "## nvidia-smi"
  nvidia-smi || true
  echo
  echo "## nvcc --version"
  (command -v nvcc && nvcc --version) || (test -x /usr/local/cuda/bin/nvcc && /usr/local/cuda/bin/nvcc --version) || true
  echo
  echo "## cudnn version header"
  (ls /usr/include/cudnn_version_v9.h >/dev/null 2>&1 && grep -n "CUDNN_MAJOR\|CUDNN_MINOR\|CUDNN_PATCHLEVEL" /usr/include/cudnn_version_v9.h) || true
  echo
  echo "## gcc/g++/cmake"
  (command -v gcc && gcc --version | head -n 1) || true
  (command -v g++ && g++ --version | head -n 1) || true
  (command -v cmake && cmake --version | head -n 1) || true
} > "$env_file"

function run_one() {
  local name="$1"; shift
  local log="$REPO_ROOT/$OUT_DIR/logs/${name}_$ts.log"
  echo "[RUN] $name" | tee "$log"
  echo "CMD: $*" | tee -a "$log"
  "$@" 2>&1 | tee -a "$log"
  echo "[DONE] $name" | tee -a "$log"
}

echo "Using GPT2_BIN=$GPT2_BIN"
echo "Using LLAMA3_BIN=$LLAMA3_BIN"

cd "$BUILD_DIR"

# GPT-2 config
GPT2_COMMON=(
  "./gpt2"
  --model d12
  --input_bin "$GPT2_BIN"
  --dtype bfloat16
  --batch_size 1
  --sequence_length "$SEQ_LEN"
  --total_batch_size "$SEQ_LEN"
  --num_iteration "$ITERS"
  --freq_generate_txt 1000000
  --sample_every 0
  --val_loss_every 0
  --learning_rate 0
)

run_one "gpt2_baseline" "${GPT2_COMMON[@]}" --flash=false
run_one "gpt2_flash"    "${GPT2_COMMON[@]}" --flash=true

# LLaMA3 config
LLAMA3_COMMON=(
  "./llama3"
  --model llama3
  --input_bin "$LLAMA3_BIN"
  --dtype bfloat16
  --batch_size 1
  --sequence_length "$SEQ_LEN"
  --total_batch_size "$SEQ_LEN"
  --num_iteration "$ITERS"
  --freq_generate_txt 1000000
  --sample_every 0
  --val_loss_every 0
)

run_one "llama3_baseline" "${LLAMA3_COMMON[@]}" --flash=false
run_one "llama3_flash"    "${LLAMA3_COMMON[@]}" --flash=true

cd "$REPO_ROOT"
python3 scripts/flash_sdpa_parse.py \
  --out_dir "$OUT_DIR" \
  --timestamp "$ts" \
  --seq_len "$SEQ_LEN" \
  --iters "$ITERS" \
  --env_file "$env_file"

echo
echo "Report generated: $REPO_ROOT/$OUT_DIR/report_$ts.md"
