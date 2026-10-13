#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${CONTAINER:-duan-fm9g4bv-align}"
GPU_ID="${GPU_ID:-0}"
STEPS="${STEPS:-20}"
WARMUP_STEPS="${WARMUP_STEPS:-2}"
BUILD_JOBS="${BUILD_JOBS:-16}"
OUT_DIR="${OUT_DIR:-/workspace/InfiniTrain/build-dcu-bw1000/fm9gv_benchmark_results}"

docker inspect "${CONTAINER}" >/dev/null
if [ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER}")" != "true" ]; then
  docker start "${CONTAINER}" >/dev/null
fi

docker exec \
  -e HIP_VISIBLE_DEVICES="${GPU_ID}" \
  -e GLOG_logtostderr=1 \
  -e STEPS="${STEPS}" \
  -e WARMUP_STEPS="${WARMUP_STEPS}" \
  -e BUILD_JOBS="${BUILD_JOBS}" \
  -e OUT_DIR="${OUT_DIR}" \
  "${CONTAINER}" bash -lc '
    set -euo pipefail
    cd /workspace/InfiniTrain
    mkdir -p "${OUT_DIR}"

    test -f "${OUT_DIR}/pytorch_sample.bin" || {
      echo "Missing ${OUT_DIR}/pytorch_sample.bin; run run_pytorch_fm9gv.sh first." >&2
      exit 2
    }
    test -f "${OUT_DIR}/vpm_input_bf16.bin" || {
      echo "Missing ${OUT_DIR}/vpm_input_bf16.bin; run run_pytorch_fm9gv.sh first." >&2
      exit 2
    }

    cmake --build build-dcu-bw1000 \
      --target fm9gv_multimodal_alignment \
      -j "${BUILD_JOBS}"

    timeout 1200 ./build-dcu-bw1000/fm9gv_multimodal_alignment \
      --fm9gv_filepath=/workspace/FM9G4B-V-official/fm9gv_text_fp32.bin \
      --sample_bin="${OUT_DIR}/pytorch_sample.bin" \
      --vision_filepath=/workspace/FM9G4B-V-official/fm9gv_vision_fp32.bin \
      --vpm_input_bin="${OUT_DIR}/vpm_input_bf16.bin" \
      --resampler_filepath=/workspace/FM9G4B-V-official/fm9gv_resampler_fp32.bin \
      --device=dcu \
      --dtype=bfloat16 \
      --autocast=true \
      --backward=true \
      --num_iterations="${STEPS}" \
      --warmup_iterations="${WARMUP_STEPS}" \
      --learning_rate=1e-6 \
      --adam_beta2=0.95 \
      --weight_decay=0.1 \
      --max_grad_norm=1 \
      --fp32_master_weights=true \
      --tune_vision=true \
      --tune_llm=false \
      --metrics_json="${OUT_DIR}/infinitrain_metrics.json" \
      --parameter_initial_output_name=vpm.embeddings.patch_embedding.weight \
      --parameter_initial_output_bin="${OUT_DIR}/infinitrain_patch_initial.bin" \
      --parameter_grad_output_name=vpm.embeddings.patch_embedding.weight \
      --parameter_grad_output_bin="${OUT_DIR}/infinitrain_patch_grad_final.bin" \
      --parameter_output_name=vpm.embeddings.patch_embedding.weight \
      --parameter_output_bin="${OUT_DIR}/infinitrain_patch_after_final.bin" \
      2>&1 | tee "${OUT_DIR}/infinitrain.log"
  '

echo "InfiniTrain result: ${OUT_DIR}/infinitrain_metrics.json"
