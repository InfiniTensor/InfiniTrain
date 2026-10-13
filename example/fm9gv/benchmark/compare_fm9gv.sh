#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${CONTAINER:-duan-fm9g4bv-align}"
OUT_DIR="${OUT_DIR:-/workspace/InfiniTrain/build-dcu-bw1000/fm9gv_benchmark_results}"

docker exec \
  -e OUT_DIR="${OUT_DIR}" \
  "${CONTAINER}" bash -lc '
    set -euo pipefail
    cd /workspace/InfiniTrain
    python3 example/fm9gv/scripts/compare_fm9gv_training.py \
      --infinitrain-json="${OUT_DIR}/infinitrain_metrics.json" \
      --pytorch-json="${OUT_DIR}/pytorch_metrics.json" \
      --infinitrain-initial="${OUT_DIR}/infinitrain_patch_initial.bin" \
      --pytorch-initial="${OUT_DIR}/pytorch_patch_initial.bin" \
      --infinitrain-final="${OUT_DIR}/infinitrain_patch_after_final.bin" \
      --pytorch-final="${OUT_DIR}/pytorch_patch_after_final.bin" \
      --infinitrain-grad="${OUT_DIR}/infinitrain_patch_grad_final.bin" \
      --pytorch-grad="${OUT_DIR}/pytorch_patch_grad_final.bin" \
      --output-json="${OUT_DIR}/comparison.json" \
      | tee "${OUT_DIR}/comparison.log"
  '

echo "Comparison result: ${OUT_DIR}/comparison.json"
