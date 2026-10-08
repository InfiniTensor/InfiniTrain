#pragma once

namespace infini_train::nn::parallel {

// Model-parallel topology and communication policy. Runtime-only process
// information, such as threads per process, remains owned by GlobalEnv.
struct ModelParallelConfig {
    int tensor_model_parallel_size = 1;
    int pipeline_model_parallel_size = 1;
    int virtual_pipeline_model_parallel_size = 1;
    bool sequence_parallel = false;

    bool tp_comm_overlap = false;
    bool tp_comm_bulk_wgrad = true;
    bool tp_comm_bulk_dgrad = true;
    // Pipeline AllGather with Column forward / Row backward GEMMs.
    bool tp_comm_overlap_ag = true;
    // Pipeline Row forward GEMM chunks with ReduceScatter.
    bool tp_comm_overlap_rs = true;
    // Pipeline Column dgrad GEMM chunks with ReduceScatter (takes precedence over bulk RS).
    bool tp_comm_overlap_rs_dgrad = false;
    // Disable forward AG and split dgrad RS for these projections; bulk options still apply.
    bool tp_comm_overlap_disable_qkv = false;
    bool tp_comm_overlap_disable_fc1 = false;
};

} // namespace infini_train::nn::parallel
