#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Heterogeneous GPT-2 Training Launcher (MPI MPMD)
# NVIDIA + MACA
# ============================================================================

# ----------------------------
# Node IPs
# ----------------------------
NV_IP="192.168.163.40"
MACA_IP="192.168.162.49"

# ----------------------------
# Binaries
# ----------------------------
NV_BIN="/nfs/duanchenjie/InfiniTrain/nv_build/gpt2"
MACA_BIN="/nfs/duanchenjie/InfiniTrain/muxi_build/gpt2"

# ----------------------------
# Training args (shared)
# ----------------------------
TRAIN_ARGS="\
--input_bin /nfs/InfiniTrain-dev/data/llmc/gpt2/tinyshakespeare/tiny_shakespeare_train.bin \
--llmc_filepath /nfs/InfiniTrain-dev/data/llmc/gpt2/gpt2_124M.bin \
--dtype float32 \
--num_iteration 10 \
--batch_size 10 \
--total_batch_size 2560 \
--nthread_per_process 4 \
--heterogeneous \
--tokenizer_bin /nfs/InfiniTrain-dev/data/llmc/gpt2/gpt2_tokenizer.bin --text_length 64
"

# device 参数因平台不同
NV_ARGS="--device cuda"
MACA_ARGS="--device maca"

# ----------------------------
# MPI options
# ----------------------------
MPI_BASE_OPTS="\
--allow-run-as-root \
--mca pml ucx \
--mca btl ^openib \
-x UCX_TLS=rc,dc,rc_verbs \
-x UCX_NET_DEVICES=mlx5_0:1
"

# 每个节点 1 个 MPI rank（你现在是 thread 并行）
NP_PER_NODE=1

# ============================================================================
# Launch
# ============================================================================

echo "============================================================================"
echo "   Heterogeneous GPT-2 Training (MPI MPMD)"
echo "============================================================================"
echo "NVIDIA node : ${NV_IP}"
echo "MACA node   : ${MACA_IP}"
echo ""

# CMD="mpirun ${MPI_BASE_OPTS} \
#     -np ${NP_PER_NODE} -host ${NV_IP}  ${NV_BIN}   ${NV_ARGS}   ${TRAIN_ARGS} \
#     : \
#     -np ${NP_PER_NODE} -host ${MACA_IP} ${MACA_BIN} ${MACA_ARGS} ${TRAIN_ARGS}
# "

CMD="mpirun ${MPI_BASE_OPTS} \
    -np ${NP_PER_NODE} -host ${NV_IP} \
        env CUDA_VISIBLE_DEVICES=0,1,2,3 \
        ${NV_BIN} ${NV_ARGS} ${TRAIN_ARGS} \
    : \
    -np ${NP_PER_NODE} -host ${MACA_IP} \
        env MACA_VISIBLE_DEVICES=0,1,2,3 \
        ${MACA_BIN} ${MACA_ARGS} ${TRAIN_ARGS}
"


echo "[MPIRUN CMD]"
echo "${CMD}"
echo "----------------------------------------------------------------------------"

eval ${CMD}
