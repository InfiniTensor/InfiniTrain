#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/run_pytorch_fm9gv.sh"
"${SCRIPT_DIR}/run_infinitrain_fm9gv.sh"
"${SCRIPT_DIR}/compare_fm9gv.sh"
