#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${CONTAINER:-duan-fm9g4bv-align}"
GPU_ID="${GPU_ID:-0}"
STEPS="${STEPS:-20}"
WARMUP_STEPS="${WARMUP_STEPS:-2}"
OUT_DIR="${OUT_DIR:-/workspace/InfiniTrain/build-dcu-bw1000/fm9gv_benchmark_results}"

docker inspect "${CONTAINER}" >/dev/null
if [ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER}")" != "true" ]; then
  docker start "${CONTAINER}" >/dev/null
fi

docker exec \
  -e HIP_VISIBLE_DEVICES="${GPU_ID}" \
  -e STEPS="${STEPS}" \
  -e WARMUP_STEPS="${WARMUP_STEPS}" \
  -e OUT_DIR="${OUT_DIR}" \
  "${CONTAINER}" bash -lc '
    set -euo pipefail
    cd /workspace/InfiniTrain
    mkdir -p "${OUT_DIR}"

    python3 example/fm9gv/scripts/prepare_fm9gv_multimodal_alignment.py \
      --model-dir /workspace/FM9G4B-V-official \
      --finetune-dir /workspace/CPM-9G-8B-FM9G-V/FM9G4B-V/finetune \
      --data-json /workspace/CPM-9G-8B-FM9G-V/FM9G4B-V/finetune/data.json \
      --output-bin "${OUT_DIR}/pytorch_sample.bin" \
      --vpm-input-bin "${OUT_DIR}/vpm_input_bf16.bin" \
      --reference-grad-bin "${OUT_DIR}/pytorch_vision_grad.bin" \
      --reference-json "${OUT_DIR}/pytorch_metrics.json" \
      --dtype bfloat16 \
      --attention eager \
      --max-slice-nums 1 \
      --iterations "${STEPS}" \
      --warmup-iterations "${WARMUP_STEPS}" \
      --learning-rate 1e-6 \
      --adam-beta2 0.95 \
      --weight-decay 0.1 \
      --max-grad-norm 1.0 \
      --fp32-master-weights \
      --tune-vision \
      --no-tune-llm \
      --seed 0 \
      --parameter-initial-output-name vpm.embeddings.patch_embedding.weight \
      --parameter-initial-output-bin "${OUT_DIR}/pytorch_patch_initial.bin" \
      --parameter-grad-name vpm.embeddings.patch_embedding.weight \
      --parameter-grad-bin "${OUT_DIR}/pytorch_patch_grad_final.bin" \
      --parameter-output-name vpm.embeddings.patch_embedding.weight \
      --parameter-output-bin "${OUT_DIR}/pytorch_patch_after_final.bin" \
      2>&1 | tee "${OUT_DIR}/pytorch.log"
  '

echo "PyTorch result: ${OUT_DIR}/pytorch_metrics.json"
