#!/usr/bin/env bash
set -euo pipefail

# Prepare datasets / weights that can be read directly by InfiniTrain examples.
#
# Usage:
#   ./scripts/assets/prepare-infinitrain-assets.sh gpt2
#   ./scripts/assets/prepare-infinitrain-assets.sh llama3
#   ./scripts/assets/prepare-infinitrain-assets.sh mnist
#   ./scripts/assets/prepare-infinitrain-assets.sh all
#
# Optional environment variables:
#   DATA_DIR=/path/to/data
#   PYTHON=python3
#   HF_TOKEN=hf_xxx
#   FORCE=1
#   SKIP_LLAMA3_WEIGHTS=1
#
# Output layout:
#   data/
#   ├── gpt2/
#   │   ├── gpt2_124M.bin
#   │   ├── gpt2_tokenizer.bin
#   │   ├── tiny_shakespeare_train.bin
#   │   └── tiny_shakespeare_val.bin
#   ├── llama3/
#   │   ├── llama3.2_1B_fp32.bin
#   │   ├── tiny_shakespeare_train.bin
#   │   └── tiny_shakespeare_val.bin
#   └── mnist/
#       ├── train-images-idx3-ubyte
#       ├── train-labels-idx1-ubyte
#       ├── t10k-images-idx3-ubyte
#       └── t10k-labels-idx1-ubyte

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
CACHE_DIR="${DATA_DIR}/.cache"
PYTHON="${PYTHON:-python3}"
FORCE="${FORCE:-0}"
SKIP_LLAMA3_WEIGHTS="${SKIP_LLAMA3_WEIGHTS:-0}"

GPT2_DIR="${DATA_DIR}/gpt2"
LLAMA3_DIR="${DATA_DIR}/llama3"
MNIST_DIR="${DATA_DIR}/mnist"

TARGET="${1:-all}"

case "${TARGET}" in
  gpt2|llama3|mnist|all) ;;
  *)
    echo "Usage: $0 {gpt2|llama3|mnist|all}"
    exit 2
    ;;
esac

mkdir -p "${CACHE_DIR}" "${GPT2_DIR}" "${LLAMA3_DIR}" "${MNIST_DIR}"

log() {
  printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

download_file() {
  local url="$1"
  local dst="$2"

  if [[ -s "${dst}" && "${FORCE}" != "1" ]]; then
    echo "skip existing: ${dst}"
    return 0
  fi

  mkdir -p "$(dirname "${dst}")"
  local tmp="${dst}.part"
  rm -f "${tmp}"

  echo "download: ${url}"
  curl -fL --retry 4 --retry-delay 2 --connect-timeout 20 \
    -o "${tmp}" "${url}"
  mv "${tmp}" "${dst}"
}

prepare_gpt2() {
  log "Preparing GPT-2 dataset / weights"

  # InfiniTrain's GPT-2 LLMC loader currently accepts the FP32 v3 file.
  # These artifacts are the same llm.c starter-pack files used by TinyInfiniTrain.
  local base="https://huggingface.co/datasets/karpathy/llmc-starter-pack/resolve/main"

  local files=(
    "gpt2_124M.bin"
    "gpt2_tokenizer.bin"
    "tiny_shakespeare_train.bin"
    "tiny_shakespeare_val.bin"
  )

  local f
  for f in "${files[@]}"; do
    download_file "${base}/${f}?download=true" "${GPT2_DIR}/${f}"
  done

  echo
  echo "GPT-2 ready:"
  echo "  weights:   ${GPT2_DIR}/gpt2_124M.bin"
  echo "  tokenizer: ${GPT2_DIR}/gpt2_tokenizer.bin"
  echo "  train:     ${GPT2_DIR}/tiny_shakespeare_train.bin"
  echo "  val:       ${GPT2_DIR}/tiny_shakespeare_val.bin"
}

ensure_llama_python() {
  need_cmd "${PYTHON}"

  local venv="${CACHE_DIR}/llama3-venv"
  LLAMA_PY="${venv}/bin/python"
  local py="${LLAMA_PY}"

  if [[ ! -x "${py}" ]]; then
    log "Creating local Python environment for LLaMA3 preparation"
    "${PYTHON}" -m venv "${venv}"
  fi

  if ! "${py}" - <<'PY' >/dev/null 2>&1
import numpy
import huggingface_hub
import socksio
import transformers
PY
  then
    log "Installing LLaMA3 preparation dependencies into ${venv}"
    "${py}" -m pip install --upgrade pip
    "${py}" -m pip install \
      "numpy>=1.24" \
      "huggingface_hub>=0.24" \
      "socksio>=1.0" \
      "transformers>=4.43"
  fi

}

prepare_llama3() {
  log "Preparing LLaMA 3.2 1B dataset / weights"

  local LLAMA_PY=""
  ensure_llama_python
  local py="${LLAMA_PY}"

  local tiny_txt="${CACHE_DIR}/tiny_shakespeare.txt"
  download_file \
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt" \
    "${tiny_txt}"

  # The model repository is gated. The Python helper accepts either HF_TOKEN or
  # the token saved by `hf auth login`.
  TINY_SHAKESPEARE_TXT="${tiny_txt}" \
  LLAMA3_OUTPUT_DIR="${LLAMA3_DIR}" \
  LLAMA3_CACHE_DIR="${CACHE_DIR}/llama3-hf" \
  SKIP_LLAMA3_WEIGHTS="${SKIP_LLAMA3_WEIGHTS}" \
  FORCE="${FORCE}" \
  "${py}" "${SCRIPT_DIR}/prepare_llama3_assets.py"

  echo
  echo "LLaMA3 ready:"
  if [[ "${SKIP_LLAMA3_WEIGHTS}" != "1" ]]; then
    echo "  weights: ${LLAMA3_DIR}/llama3.2_1B_fp32.bin"
  fi
  echo "  train:   ${LLAMA3_DIR}/tiny_shakespeare_train.bin"
  echo "  val:     ${LLAMA3_DIR}/tiny_shakespeare_val.bin"
}

prepare_mnist() {
  log "Preparing MNIST IDX dataset"

  # TorchVision's public MNIST mirror.
  local base="https://ossci-datasets.s3.amazonaws.com/mnist"
  local files=(
    "train-images-idx3-ubyte"
    "train-labels-idx1-ubyte"
    "t10k-images-idx3-ubyte"
    "t10k-labels-idx1-ubyte"
  )

  local f
  for f in "${files[@]}"; do
    local gz="${MNIST_DIR}/${f}.gz"
    local dst="${MNIST_DIR}/${f}"

    download_file "${base}/${f}.gz" "${gz}"

    if [[ ! -s "${dst}" || "${FORCE}" == "1" ]]; then
      echo "extract: ${gz}"
      gzip -dc "${gz}" > "${dst}.part"
      mv "${dst}.part" "${dst}"
    else
      echo "skip existing: ${dst}"
    fi
  done

  echo
  echo "MNIST ready:"
  echo "  dataset: ${MNIST_DIR}"
}

need_cmd curl
need_cmd gzip

case "${TARGET}" in
  gpt2)
    prepare_gpt2
    ;;
  llama3)
    prepare_llama3
    ;;
  mnist)
    prepare_mnist
    ;;
  all)
    prepare_gpt2
    prepare_llama3
    prepare_mnist
    ;;
esac

log "Done"
