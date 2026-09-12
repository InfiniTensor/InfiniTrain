#pragma once

#include <optional>

namespace infini_train::nn::parallel {

// Model-parallel topology and communication policy. Runtime-only process
// information, such as threads per process, remains owned by GlobalEnv.
struct ModelParallelConfig {
  int tensor_model_parallel_size = 1;
  int pipeline_model_parallel_size = 1;
  int virtual_pipeline_model_parallel_size = 1;
  bool sequence_parallel = false;
  int expert_model_parallel_size = 1;
  std::optional<int> expert_tensor_parallel_size = std::nullopt;

  bool tp_comm_overlap = false;
  // FIXME(zbl): These flags explicitly enable/disable backward bulk overlap
  //             here. Megatron/TE uses them to select Userbuffers bulk
  //             optimizations; its ordinary SP backward can still overlap
  //             async AG/RS when those optimizations are off. Revisit this
  //             distinction when adding an optimized overlap backend.
  // Overlap dgrad ReduceScatter with the wgrad GEMM.
  bool tp_comm_bulk_wgrad = true;
  // Overlap activation AllGather with the dgrad GEMM.
  bool tp_comm_bulk_dgrad = true;
  // Pipeline AllGather with Column forward / Row backward GEMMs.
  bool tp_comm_overlap_ag = true;
  // Pipeline Row forward GEMM chunks with ReduceScatter.
  bool tp_comm_overlap_rs = true;
  // Pipeline Column dgrad GEMM chunks with ReduceScatter (takes precedence over
  // bulk RS).
  bool tp_comm_overlap_rs_dgrad = false;
  // Disable forward AG and split dgrad RS for these projections; bulk options
  // still apply.
  bool tp_comm_overlap_disable_qkv = false;
  bool tp_comm_overlap_disable_fc1 = false;
};

}  // namespace infini_train::nn::parallel
