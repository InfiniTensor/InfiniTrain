#include <cstdlib>
#include <optional>

#include "gflags/gflags.h"
#include "gtest/gtest.h"

#include "infini_train/include/nn/parallel/global.h"

DEFINE_int32(context_parallel, 4, "CP group size");
DEFINE_int32(tensor_parallel, 1, "Use TP to exercise non-contiguous CP ranks");
DEFINE_string(cp_comm_type, "p2p", "CP communication type");
DEFINE_string(hierarchical_context_parallel_sizes, "", "A2A,P2P group sizes");
DEFINE_string(attention_cp_layout, "zigzag", "CP sequence layout");

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  setenv("WORLD_SIZE", "1", 1);
  setenv("LOCAL_WORLD_SIZE", "1", 1);
  setenv("RANK", "0", 1);
  setenv("LOCAL_RANK", "0", 1);
  infini_train::nn::parallel::global::InitAllEnv(
      FLAGS_context_parallel * FLAGS_tensor_parallel, FLAGS_tensor_parallel,
      false, 1, 1, 1, 1, FLAGS_context_parallel, FLAGS_cp_comm_type,
      FLAGS_hierarchical_context_parallel_sizes, FLAGS_attention_cp_layout);
  return RUN_ALL_TESTS();
}
