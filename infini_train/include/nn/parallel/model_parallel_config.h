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
    bool tp_comm_overlap_ag = true;
    bool tp_comm_overlap_rs = true;
    bool tp_comm_overlap_rs_dgrad = false;
    bool tp_comm_overlap_disable_qkv = false;
    bool tp_comm_overlap_disable_fc1 = false;
};

} // namespace infini_train::nn::parallel
